"""Async OpenAI-compatible LLM client.

Single wrapper used by all agents.  Capability flags (constrained decoding,
system role, thinking kwargs) are read from the family profile declared in
hpema_config.yaml — not discovered at runtime.  This eliminates wasted API
calls caused by sending wrong kwargs to models that reject them.

See backend/services/llm/profiles.py for the family registry.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, AsyncGenerator, Type

from openai import AsyncOpenAI, BadRequestError, InternalServerError, RateLimitError
from pydantic import BaseModel, ValidationError

from backend.services.llm.model_registry import ModelRegistry, ResolvedModel
from backend.services.llm.profiles import ModelFamilyProfile, get_profile

logger = logging.getLogger(__name__)

# 400 errors that are transient platform failures, not capability gaps.
# Retried like 500s — do NOT treat as permanent profile mismatches.
_TRANSIENT_REQUEST_ERRORS = (
    "degraded",
    "function cannot be invoked",
    "service unavailable",
    "temporarily unavailable",
)


class LLMClient:
    """Async client that talks to any OpenAI-compatible endpoint.

    Capability flags (constrained decoding, system role, thinking kwargs) are
    read from the model's family profile — not discovered at runtime.  This
    eliminates wasted API calls caused by sending wrong kwargs to models that
    reject them.  Transient errors (500, 429, DEGRADED 400s) are still retried.
    """

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self._registry = registry or ModelRegistry()
        self._clients: dict[str, AsyncOpenAI] = {}

    def _get_client(self, model: ResolvedModel) -> AsyncOpenAI:
        """Lazily create one AsyncOpenAI client per unique endpoint."""
        if model.endpoint not in self._clients:
            self._clients[model.endpoint] = AsyncOpenAI(
                base_url=model.endpoint,
                api_key=model.api_key or "unused",
                # Explicit timeouts: fail fast on connection problems, allow
                # up to 3 min for large models to stream their first token.
                timeout=180.0,
            )
        return self._clients[model.endpoint]

    # ── Profile helpers ──────────────────────────────────────────────────────

    def _messages(
        self,
        profile: ModelFamilyProfile,
        system_prompt: str,
        user_prompt: str,
    ) -> list[dict]:
        """Build messages list according to the family profile.

        For families that don't support the system role (Gemma 3), system
        content is prepended to the first user message.
        """
        if not profile.supports_system_role:
            return [{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}]
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _thinking_kwargs(
        self,
        resolved: ResolvedModel,
        profile: ModelFamilyProfile,
    ) -> tuple[dict, dict]:
        """Return (extra_body, extra_params) for thinking based on profile + config.

        extra_body  — merged into the API call's extra_body field
        extra_params — merged into top-level API call kwargs (e.g. reasoning_effort)
        """
        mode = profile.thinking_mode
        want = resolved.enable_thinking

        if mode in ("none", "always"):
            # "none": no thinking capability
            # "always": thinks natively — no control kwargs needed
            extra_body: dict = {}
            extra_params: dict = {}
        elif mode == "toggle-off":
            # Off by default; activate only when explicitly requested
            if want:
                extra_body = dict(profile.thinking_on_extra_body)
                extra_params = dict(profile.thinking_on_extra_params)
            else:
                extra_body = {}
                extra_params = {}
        elif mode == "toggle-on":
            # On by default; must explicitly disable if not wanted
            if want:
                extra_body = dict(profile.thinking_on_extra_body)
                extra_params = dict(profile.thinking_on_extra_params)
            else:
                extra_body = dict(profile.thinking_off_extra_body)
                extra_params = dict(profile.thinking_off_extra_params)
        else:
            extra_body = {}
            extra_params = {}

        # Merge model-level extra_body overrides on top (never mutates profile)
        if resolved.extra_body:
            extra_body = {**extra_body, **resolved.extra_body}

        return extra_body, extra_params

    def _build_kwargs(
        self,
        resolved: ResolvedModel,
        profile: ModelFamilyProfile,
        system_prompt: str,
        user_prompt: str,
        response_schema: type[BaseModel] | None,
        temperature: float,
        max_tokens: int,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Assemble the full kwargs dict for one API call using the family profile."""
        use_constrained = (
            response_schema is not None and profile.supports_json_schema
        )

        eff_system = (
            system_prompt + _json_prompt_suffix(response_schema)
            if response_schema and not use_constrained
            else system_prompt
        )

        extra_body, extra_params = self._thinking_kwargs(resolved, profile)

        kwargs: dict[str, Any] = {
            "model": resolved.model,
            "messages": self._messages(profile, eff_system, user_prompt),
            "temperature": temperature,
            "max_tokens": max_tokens,
            **extra_params,
        }
        if stream:
            kwargs["stream"] = True
        if extra_body:
            kwargs["extra_body"] = extra_body
        if use_constrained:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_schema.__name__,
                    "schema": response_schema.model_json_schema(),
                },
            }

        logger.info(
            "LLM call | role=%s family=%s model=%s constrained=%s "
            "thinking_mode=%s enable_thinking=%s",
            resolved.model.split("/")[-1],  # short name
            resolved.family,
            resolved.model,
            use_constrained,
            profile.thinking_mode,
            resolved.enable_thinking,
        )
        return kwargs

    # ── Transient-error retry ────────────────────────────────────────────────

    async def _api_call(self, client: AsyncOpenAI, kwargs: dict[str, Any]) -> Any:
        """Execute one chat completion, retrying only on transient errors.

        Retries:
          - InternalServerError (500)
          - RateLimitError (429)
          - BadRequestError (400) matching _TRANSIENT_REQUEST_ERRORS
            (NVIDIA NIM "DEGRADED function", platform unavailability)

        All other 400s are re-raised immediately — they indicate a profile
        misconfiguration, not a transient failure.
        """
        delays = [5, 15]  # seconds between attempt 1→2 and 2→3
        for attempt, delay in enumerate([-1] + delays):
            if delay >= 0:
                logger.warning(
                    "Transient API error — retrying in %ds (attempt %d/3)...",
                    delay, attempt + 1,
                )
                await asyncio.sleep(delay)
            try:
                return await client.chat.completions.create(**kwargs)
            except (InternalServerError, RateLimitError) as exc:
                if attempt == len(delays):
                    raise
                logger.warning("API error (%s): %s", type(exc).__name__, exc)
            except BadRequestError as exc:
                if any(kw in str(exc).lower() for kw in _TRANSIENT_REQUEST_ERRORS):
                    if attempt == len(delays):
                        raise
                    logger.warning("Transient 400 (%s): %s", type(exc).__name__, exc)
                else:
                    raise

    # ── Public interface ─────────────────────────────────────────────────────

    async def aclose(self) -> None:
        """Close all httpx connection pools (call before event loop shuts down)."""
        for client in self._clients.values():
            await client.close()
        self._clients.clear()

    async def generate(
        self,
        role: str,
        system_prompt: str,
        user_prompt: str,
        response_schema: type[BaseModel] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        extra_params: dict[str, Any] | None = None,
    ) -> str:
        """Send a chat completion and return the raw response text.

        Kwargs are built from the model's family profile — no runtime discovery.
        """
        resolved = self._registry.get(role)
        profile = get_profile(resolved.family)
        client = self._get_client(resolved)

        kwargs = self._build_kwargs(
            resolved, profile, system_prompt, user_prompt,
            response_schema, temperature, max_tokens,
        )
        if extra_params:
            kwargs.update(extra_params)

        response = await self._api_call(client, kwargs)
        content = response.choices[0].message.content or ""

        # Some reasoning models (o-series, step) route output through
        # reasoning_content when constrained decoding is active.
        if not content and response.choices:
            msg = response.choices[0].message
            content = getattr(msg, "reasoning_content", None) or ""

        logger.debug("LLM response from %s: %d chars", role, len(content))
        return content

    def parse_structured(self, raw: str, response_model: Type[BaseModel], role: str = "?") -> BaseModel:
        """Parse raw LLM text into a Pydantic model.

        Extracted so both generate_structured() and streaming callers can reuse
        the same JSON-extraction / fallback logic without duplicating it.
        """
        raw = _fix_code_formatting(raw)

        # First: try direct parse (works when constrained decoding succeeded)
        try:
            return response_model.model_validate_json(raw)
        except (ValidationError, ValueError):
            pass

        # Fallback: extract JSON from prose / markdown fences
        extracted = _extract_json(raw)
        if extracted is not None:
            try:
                return response_model.model_validate_json(extracted)
            except (ValidationError, ValueError):
                pass

        # Last resort: attempt partial construction with defaults
        logger.warning(
            "Could not parse structured response from %s. Attempting partial parse.", role
        )
        data = _extract_json_dict(raw)

        # Detect schema echo — model returned the JSON Schema itself instead of
        # an instance (common with small models given a raw JSON Schema prompt).
        if _is_schema_echo(data):
            logger.error(
                "Model %s echoed the schema instead of producing an instance. "
                "Raw response snippet: %.200s", role, raw
            )
            raise ValueError(
                f"Model returned the JSON schema instead of a response instance. "
                f"Raw output (first 300 chars): {raw[:300]}"
            )

        try:
            return response_model.model_validate(data)
        except ValidationError as exc:
            raise ValueError(
                f"All structured-parse attempts failed for {role}. "
                f"Last error: {exc}. "
                f"Raw output (first 300 chars): {raw[:300]}"
            ) from exc

    async def generate_structured(
        self,
        role: str,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
        **kwargs: Any,
    ) -> BaseModel:
        """Generate and parse into a Pydantic model.

        Tries constrained decoding first. If the provider rejected it (and the
        client fell back to prompt-only mode), the raw output may contain
        markdown fences or prose — _extract_json handles that gracefully.
        """
        raw = await self.generate(
            role=role,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=response_model,
            **kwargs,
        )
        return self.parse_structured(raw, response_model, role)

    async def generate_stream(
        self,
        role: str,
        system_prompt: str,
        user_prompt: str,
        response_schema: type[BaseModel] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        extra_params: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream text chunks from the LLM endpoint.

        Yields raw string chunks as they arrive from the model. Callers should
        accumulate them and call ``parse_structured()`` on the joined text.

        Graceful fallback: if the streaming call fails for any reason, falls
        back to a single blocking ``generate()`` call and yields the full
        response as one chunk.  Models without streaming support behave
        identically from the caller's perspective.
        """
        resolved = self._registry.get(role)
        profile = get_profile(resolved.family)
        client = self._get_client(resolved)

        kwargs = self._build_kwargs(
            resolved, profile, system_prompt, user_prompt,
            response_schema, temperature, max_tokens, stream=True,
        )
        if extra_params:
            kwargs.update(extra_params)

        use_constrained = (
            response_schema is not None and profile.supports_json_schema
        )

        # Attempt streaming — fall back on any error (profile mis-config, network, etc.)
        stream_obj = None
        try:
            stream_obj = await client.chat.completions.create(**kwargs)
        except Exception as exc:
            logger.warning(
                "Streaming request failed for %s (%s: %s) — falling back to "
                "blocking generate(). Check family profile if this is a 400.",
                role, type(exc).__name__, exc,
            )

        if stream_obj is not None:
            detector = _ThinkLoopDetector(role)
            yielded_any = False
            try:
                async for chunk in stream_obj:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta

                    # Primary content field (all standard models)
                    token = delta.content or ""

                    # Reasoning models may route output through reasoning_content
                    # when response_format=json_schema is active — delta.content
                    # stays empty for the whole stream.  Accumulate reasoning tokens
                    # so the empty-stream fallback is not triggered, but only yield
                    # them if no regular content ever arrives (they are raw reasoning,
                    # not structured output).
                    if not token:
                        token = getattr(delta, "reasoning_content", None) or ""

                    if token:
                        detector.feed(token)  # raises ThinkLoopError if loop found
                        yielded_any = True
                        yield token
            except ThinkLoopError:
                logger.warning(
                    "Think loop detected for %s — aborting stream and re-raising "
                    "for retry.",
                    role,
                )
                raise  # propagate up through run_streaming → orchestrator retry
            except Exception as exc:
                logger.warning(
                    "Streaming interrupted for %s (%s: %s) — falling back.",
                    role, type(exc).__name__, exc,
                )
                # Fall through to non-streaming fallback below.
            else:
                if not yielded_any and use_constrained:
                    # Model returned HTTP 200 but emitted zero content tokens
                    # while constrained decoding was active — silent capability gap.
                    # Fall through to prompt-only non-streaming fallback.
                    logger.info(
                        "Empty stream with constrained decoding for %s — "
                        "falling back to prompt-only non-streaming call. "
                        "Consider setting supports_json_schema=False in the profile.",
                        role,
                    )
                else:
                    return  # streaming completed successfully

        # Non-streaming fallback: yields the full response as one chunk.
        logger.info("Using non-streaming fallback for %s", role)
        text = await self.generate(
            role=role,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=response_schema,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_params=extra_params,
        )
        yield text


class ThinkLoopError(RuntimeError):
    """Raised when a streaming model is detected to be stuck in a think loop.

    This is a soft, retryable failure — the orchestrator retry logic will
    start a fresh request which re-initialises the KV cache and typically
    breaks the loop.
    """


class _ThinkLoopDetector:
    """Monitors a raw token stream for runaway <think> loops.

    Two independent signals, either sufficient to abort:

    1. **Hard cap** — total chars accumulated inside ``<think>`` blocks
       exceeds ``THINK_CHAR_CAP``.  Catches slow but genuine loops that
       would otherwise run to the model's full ``max_tokens`` budget.

    2. **Repetition detector** — scans the most recent ``SCAN_WINDOW``
       chars of think content for any ``SNIPPET_LEN``-char substring that
       appears at least ``REPEAT_THRESHOLD`` times.  Catches tight
       "I should reconsider…I should reconsider…" loops early, usually
       within 1–2 seconds of the loop starting.

    Call ``feed(chunk)`` for each streaming token; it raises
    ``ThinkLoopError`` the moment a loop is confirmed.
    """

    THINK_CHAR_CAP    = 8_000   # total think chars before hard abort
    SCAN_WINDOW       = 2_000   # recent think chars to scan for repeats
    SNIPPET_LEN       = 60      # pattern length to test for repetition
    REPEAT_THRESHOLD  = 5       # how many times a snippet must repeat
    CHECK_INTERVAL    = 400     # check once per this many new raw chars

    def __init__(self, role: str) -> None:
        self._role        = role
        self._raw         = ""   # all streamed chars so far
        self._think_chars = 0    # total chars inside <think> blocks
        self._last_check  = 0    # len(_raw) at last check

    # ------------------------------------------------------------------

    def feed(self, chunk: str) -> None:
        """Accumulate *chunk* and raise ThinkLoopError if a loop is found."""
        self._raw += chunk
        if len(self._raw) - self._last_check < self.CHECK_INTERVAL:
            return
        self._last_check = len(self._raw)
        self._think_chars = _count_think_chars(self._raw)
        self._check()

    def _check(self) -> None:
        if self._think_chars > self.THINK_CHAR_CAP:
            raise ThinkLoopError(
                f"Agent [{self._role}] think block exceeded "
                f"{self.THINK_CHAR_CAP} chars ({self._think_chars} seen) — "
                "aborting stream to trigger retry."
            )

        think_content = _extract_think_content(self._raw)
        if len(think_content) < self.SNIPPET_LEN * self.REPEAT_THRESHOLD:
            return

        recent = think_content[-self.SCAN_WINDOW:]
        stride = self.SNIPPET_LEN // 2   # overlapping scan for better coverage
        for start in range(0, len(recent) - self.SNIPPET_LEN, stride):
            snippet = recent[start : start + self.SNIPPET_LEN]
            if recent.count(snippet) >= self.REPEAT_THRESHOLD:
                raise ThinkLoopError(
                    f"Agent [{self._role}] think loop detected: "
                    f"a {self.SNIPPET_LEN}-char pattern repeated "
                    f"{recent.count(snippet)}× in the last "
                    f"{len(recent)} think chars — aborting stream."
                )


def _extract_think_content(text: str) -> str:
    """Return the concatenated content of all <think> blocks (open or closed)."""
    # Closed blocks
    parts = re.findall(r"<think>(.*?)</think>", text, re.DOTALL)
    # Unclosed trailing block
    unclosed = re.search(r"<think>((?:(?!</think>).)*)\Z", text, re.DOTALL)
    if unclosed:
        parts.append(unclosed.group(1))
    return "".join(parts)


def _count_think_chars(text: str) -> int:
    return len(_extract_think_content(text))


def _json_prompt_suffix(schema: Type[BaseModel]) -> str:
    """Return a system prompt suffix that instructs the model to output JSON.

    Uses a concrete example instance rather than the raw JSON Schema, because
    small models frequently echo the schema back instead of producing an instance
    when given a schema document.
    """
    example = _example_from_model(schema)
    example_str = json.dumps(example, indent=2)
    required = [n for n, f in schema.model_fields.items() if f.is_required()]
    optional = [n for n, f in schema.model_fields.items() if not f.is_required()]

    lines = [
        "\n\n---",
        "IMPORTANT: Respond with a single valid JSON object ONLY.",
        "No markdown fences, no prose, no explanation — raw JSON only.",
        "If you use internal reasoning tags (<think>), place the JSON AFTER them.",
        "The JSON must appear as the final content in your response.",
    ]
    if required:
        lines.append(f"Required fields: {', '.join(required)}")
    if optional:
        lines.append(f"Optional fields (include if relevant): {', '.join(optional)}")
    lines.append("Example (replace placeholder values with your actual output):")
    lines.append(example_str)
    return "\n".join(lines)


def _example_from_model(model: Type[BaseModel]) -> dict:
    """Build a human-readable placeholder example from a Pydantic model's fields.

    Produces string placeholders for str fields, proper defaults for others.
    This is far more effective than a raw JSON Schema for small/instruction models.
    """
    import inspect
    from pydantic_core import PydanticUndefinedType

    def _default(fi) -> object:
        """Return field default, or None if it is PydanticUndefined (required field)."""
        d = fi.default
        return None if isinstance(d, PydanticUndefinedType) else d

    example: dict = {}
    for name, field_info in model.model_fields.items():
        ann = field_info.annotation
        desc = (field_info.description or "").strip()
        placeholder = f"<{desc}>" if desc else f"<{name}>"

        # Unwrap Optional[X] → X
        origin = getattr(ann, "__origin__", None)
        args = getattr(ann, "__args__", ())
        if origin is type(None):
            ann = str
        elif origin is not None and type(None) in args:
            ann = next((a for a in args if a is not type(None)), str)

        default = _default(field_info)

        if ann is str:
            example[name] = default if isinstance(default, str) and default else placeholder
        elif ann is int:
            example[name] = default if isinstance(default, int) else 0
        elif ann is bool:
            example[name] = default if isinstance(default, bool) else False
        elif ann is float:
            example[name] = default if isinstance(default, float) else 0.0
        elif origin is list or (inspect.isclass(ann) and issubclass(ann, list)):
            example[name] = []
        elif origin is dict or (inspect.isclass(ann) and issubclass(ann, dict)):
            example[name] = {}
        else:
            # Enum or nested model
            if default is not None:
                example[name] = default.value if hasattr(default, "value") else default
            else:
                example[name] = placeholder

    return example


def _is_schema_echo(data: dict) -> bool:
    """Return True if the dict looks like a JSON Schema rather than an instance.

    Small models sometimes return the schema document they were shown instead of
    an instance that conforms to it.
    """
    if not data:
        return False
    # JSON Schema top-level markers
    schema_keys = {"properties", "definitions", "$schema", "$defs", "allOf", "anyOf"}
    if schema_keys & data.keys():
        return True
    # A schema root with type:object and no meaningful instance fields
    if data.get("type") == "object" and "title" in data:
        return True
    return False


def _extract_json(text: str) -> str | None:
    """Extract a JSON object from text that may contain markdown fences or prose."""
    # Try ```json ... ``` or ``` ... ``` blocks
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1)

    # Try the first { ... } spanning the entire content
    brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
    if brace_match:
        candidate = brace_match.group(1)
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    return None


def _extract_json_dict(text: str) -> dict:
    """Best-effort extraction of a dict from malformed JSON text."""
    candidate = _extract_json(text)
    if candidate:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    return {}


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks produced by reasoning models.

    Handles three cases:
    - Standard closed blocks: ``<think>…</think>`` followed by the response.
    - Unclosed blocks: model hit the token limit while still reasoning, so
      the closing ``</think>`` is absent.  We strip from ``<think>`` to end-of-
      text so the caller sees an empty string (and can fall back to searching
      the original text for embedded JSON).
    - JSON embedded inside the think block: some models write their answer
      inside ``<think>`` rather than after it.  Stripping here returns empty;
      the caller is expected to retry on the original text.
    """
    # 1. Remove all fully-closed think blocks.
    stripped = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()

    # 2. Remove any remaining unclosed ``<think>`` prefix (model stopped mid-think).
    stripped = re.sub(r"<think>.*$", "", stripped, flags=re.DOTALL).strip()

    return stripped


def _fix_code_formatting(raw_json: str) -> str:
    """Post-process LLM JSON to fix code formatting issues.

    Handles two common problems with small models:
    1. Literal '\\n' text in code instead of actual newlines
    2. Code with no newlines at all (single-line output)
    """
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return raw_json

    changed = False
    for field in ("source_code", "dafny_spec"):
        code = data.get(field, "")
        if not code:
            continue

        # Fix 1: Replace literal \n text with actual newlines
        # The model outputs the two characters '\' 'n' instead of a newline
        if "\\n" in code:
            code = code.replace("\\n", "\n")
            data[field] = code
            changed = True

        # Fix 2: No newlines at all — insert at statement boundaries
        if "\n" not in code and (";" in code or "{" in code):
            code = re.sub(r";\s*", ";\n", code)
            code = re.sub(r"\{\s*", "{\n", code)
            code = re.sub(r"\}\s*", "}\n", code)
            data[field] = code
            changed = True

    return json.dumps(data) if changed else raw_json
