"""Async OpenAI-compatible LLM client.

Single wrapper used by all agents. Supports constrained decoding via
response_format (guided_json) when the server supports it (vLLM / OpenAI).
Falls back to prompt-only JSON enforcement for providers that don't support
the json_schema response_format (Groq, Anthropic, older endpoints, etc.).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, AsyncGenerator, Type
import httpx

from openai import (
    AsyncOpenAI, 
    BadRequestError, 
    InternalServerError, 
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
)
from pydantic import BaseModel, ValidationError

from backend.services.llm.model_registry import ModelRegistry, ResolvedModel

logger = logging.getLogger(__name__)

# Error substrings that indicate the provider rejected the response_format param.
_CONSTRAINED_DECODING_ERRORS = (
    "response_format",
    "json_schema",
    "guided_json",
    "unsupported",
    "not supported",
    "invalid_request_error",
)

# Error substrings that indicate the provider doesn't support the system role
# (e.g. Gemma family models via NVIDIA API, some Mistral endpoints).
_NO_SYSTEM_ROLE_ERRORS = (
    "system role not supported",
    "system role is not supported",
    "does not support system",
    "system messages are not supported",
)

# Error substrings that indicate the endpoint rejected thinking-control params
# (e.g. enable_thinking / thinking_budget in extra_body.chat_template_kwargs).
# Minimax via NVIDIA NIM is the primary example: it always thinks and does not
# expose a toggle.  When detected we strip these keys and retry — the model
# will still produce <think> blocks; the pipeline handles them transparently.
_NO_THINKING_KWARGS_ERRORS = (
    "enable_thinking",
    "thinking_budget",
    "chat_template_kwargs",
    "extra_body",
    "unexpected keyword",
    "unknown field",
    "invalid parameter",
)

# 400 errors that are transient platform failures, not capability gaps.
# These should be retried like a 500, not treated as permanent incompatibilities.
_TRANSIENT_REQUEST_ERRORS = (
    "degraded",
    "function cannot be invoked",
    "service unavailable",
    "temporarily unavailable",
)


def _is_transient_transport_error(exc: BaseException) -> bool:
    """Return True for low-level network/transport failures worth retrying."""
    if isinstance(
        exc,
        (
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.ProtocolError,
            OSError,
            ConnectionError,
        ),
    ):
        return True

    msg = str(exc).lower()
    transient_markers = (
        "connection reset",
        "connection reset by peer",
        "broken pipe",
        "server disconnected",
        "remote protocol error",
        "temporarily unavailable",
        "timed out",
        "timeout",
        "econnreset",
        "econnaborted",
        "connection aborted",
    )
    return any(marker in msg for marker in transient_markers)


class LLMClient:
    """Async client that talks to any OpenAI-compatible endpoint."""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self._registry = registry or ModelRegistry()
        self._clients: dict[str, AsyncOpenAI] = {}
        # Track capability gaps per endpoint so we skip the overhead of a
        # failing first attempt on subsequent calls to the same endpoint.
        self._no_constrained_decoding: set[str] = set()
        self._no_system_role: set[str] = set()
        # Endpoints that reject thinking-control keys in extra_body
        # (e.g. enable_thinking, thinking_budget).  We strip those keys and retry.
        self._no_thinking_kwargs: set[str] = set()

    def _get_client(self, model: ResolvedModel) -> AsyncOpenAI:
        """Lazily create one AsyncOpenAI client per unique endpoint."""
        if model.endpoint not in self._clients:
            self._clients[model.endpoint] = AsyncOpenAI(
                base_url=model.endpoint,
                api_key=model.api_key or "unused",
                # Explicit timeouts: fail fast on connection problems, allow
                # up to 3 min for large models to stream their first token.
                # Without this httpx defaults to 600 s, making hangs invisible.
                timeout=180.0,
            )
        return self._clients[model.endpoint]

    def _messages(self, endpoint: str, system_prompt: str, user_prompt: str) -> list[dict]:
        """Build the messages list for a chat request.

        For endpoints that don't support the system role (e.g. Gemma via
        NVIDIA API), the system content is prepended to the first user
        message so the model still receives its full instructions.
        """
        if endpoint in self._no_system_role:
            return [{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}]
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _sanitized_extra_body(self, endpoint: str, extra_body: dict) -> dict:
        """Return extra_body with thinking-control keys removed for known endpoints.

        Called before each API request so that endpoints that previously
        rejected ``enable_thinking`` / ``thinking_budget`` don't receive them
        again.  The original ResolvedModel dict is never mutated.
        """
        if endpoint not in self._no_thinking_kwargs:
            return extra_body
        thinking_keys = {"enable_thinking", "thinking_budget"}
        result = dict(extra_body)
        if "chat_template_kwargs" in result:
            ctk = {k: v for k, v in result["chat_template_kwargs"].items()
                   if k not in thinking_keys}
            if ctk:
                result["chat_template_kwargs"] = ctk
            else:
                del result["chat_template_kwargs"]
        return result

    async def aclose(self) -> None:
        """Close all httpx connection pools while the event loop is still open.

        Must be called before asyncio.run() returns. Without this, Python 3.10+
        prints 'Event loop is closed' noise when the GC finalizes the
        AsyncOpenAI clients after the loop has already been shut down.
        """
        for client in self._clients.values():
            await client.close()
        self._clients.clear()

    async def _api_call(self, client: AsyncOpenAI, kwargs: dict[str, Any]) -> Any:
        """Execute one chat completion, retrying on transient API/transport errors.

        Retries up to 3 total attempts with backoff for:
          - 5xx / InternalServerError
          - 429 / RateLimitError
          - SDK timeout/connection failures
          - raw httpx/network transport failures (connection reset, broken pipe, etc.)
          - transient provider-side 400s such as degraded/unavailable platform errors

        Permanent 400 capability errors are re-raised immediately so the outer
        fallback logic in generate() can handle them.
        """
        backoff_delays = [1, 2]  # 3 total attempts: initial + 2 retries

        for attempt in range(3):
            try:
                return await client.chat.completions.create(**kwargs)

            except (
                InternalServerError,
                RateLimitError,
                APITimeoutError,
                APIConnectionError,
                TimeoutError,
            ) as exc:
                if attempt == len(backoff_delays):
                    raise
                delay = backoff_delays[attempt]
                logger.warning(
                    "Transient API error (%s): %s — retrying in %ss (attempt %d/3).",
                    type(exc).__name__,
                    exc,
                    delay,
                    attempt + 2,
                )
                await asyncio.sleep(delay)

            except BadRequestError as exc:
                # Some providers (NVIDIA NIM) return 400 for transient platform
                # failures ("DEGRADED function cannot be invoked"). These are not
                # capability gaps — retry them like a 500. Genuine 400s (wrong
                # response_format, unsupported system role) do NOT match these
                # patterns and are re-raised immediately for the outer loop.
                if any(kw in str(exc).lower() for kw in _TRANSIENT_REQUEST_ERRORS):
                    if attempt == len(backoff_delays):
                        raise
                    delay = backoff_delays[attempt]
                    logger.warning(
                        "Transient 400 from API (%s): %s — retrying in %ss (attempt %d/3).",
                        type(exc).__name__,
                        exc,
                        delay,
                        attempt + 2,
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

            except Exception as exc:
                if not _is_transient_transport_error(exc):
                    raise
                if attempt == len(backoff_delays):
                    raise
                delay = backoff_delays[attempt]
                logger.warning(
                    "Transient transport error (%s): %s — retrying in %ss (attempt %d/3).",
                    type(exc).__name__,
                    exc,
                    delay,
                    attempt + 2,
                )
                await asyncio.sleep(delay)

    async def generate(
        self,
        role: str,
        system_prompt: str,
        user_prompt: str,
        response_schema: Type[BaseModel] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        extra_params: dict[str, Any] | None = None,
    ) -> str:
        """Send a chat completion request and return the raw response text.

        Parameters
        ----------
        role : str
            Agent role name (actor, checker, policy) — resolved via registry.
        system_prompt : str
            System message for the agent.
        user_prompt : str
            User/task message.
        response_schema : Type[BaseModel] | None
            If provided, requests constrained decoding so the model's output
            matches the schema. Falls back to prompt-only JSON if the provider
            doesn't support response_format=json_schema.
        temperature : float
            Sampling temperature.
        max_tokens : int
            Max tokens to generate.
        extra_params : dict | None
            Additional params passed to the API (e.g., vLLM-specific flags).
        """
        resolved = self._registry.get(role)
        client = self._get_client(resolved)
        endpoint = resolved.endpoint

        kwargs_base: dict[str, Any] = {
            "model": resolved.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if extra_params:
            kwargs_base.update(extra_params)

        # Retry loop — each pass may discover one new capability gap and adapt.
        # At most 4 attempts: (1) preferred mode, (2-4) each fallback.
        for _attempt in range(4):
            use_constrained = (
                response_schema is not None
                and endpoint not in self._no_constrained_decoding
            )

            # In prompt-only mode inject the schema into the system prompt so
            # the model knows what structure to produce without response_format.
            eff_system = (
                system_prompt + _json_prompt_suffix(response_schema)
                if response_schema and not use_constrained
                else system_prompt
            )

            # Apply per-endpoint extra_body sanitization (strips thinking-control
            # keys for endpoints that have previously rejected them).
            eff_extra_body = (
                self._sanitized_extra_body(endpoint, resolved.extra_body)
                if resolved.extra_body else {}
            )

            kwargs: dict[str, Any] = {
                **kwargs_base,
                "messages": self._messages(endpoint, eff_system, user_prompt),
            }
            if eff_extra_body:
                kwargs["extra_body"] = eff_extra_body

            if use_constrained:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_schema.__name__,
                        "schema": response_schema.model_json_schema(),
                    },
                }

            logger.info(
                "LLM request to %s [%s] model=%s constrained=%s no_system_role=%s no_thinking_kwargs=%s",
                role, endpoint, resolved.model, use_constrained,
                endpoint in self._no_system_role,
                endpoint in self._no_thinking_kwargs,
            )

            try:
                response = await self._api_call(client, kwargs)
                content = response.choices[0].message.content or ""
                # NOTE: do NOT strip think tags here — generate() is used by the
                # non-streaming path; stripping happens in parse_structured() and
                # in run_streaming() after accumulation so we never lose raw content.
                logger.debug("LLM response from %s: %s chars", role, len(content))
                return content
            except BadRequestError as exc:
                err_lower = str(exc).lower()
                if any(kw in err_lower for kw in _NO_SYSTEM_ROLE_ERRORS):
                    logger.info(
                        "Endpoint %s does not support system role — "
                        "retrying with system content merged into user message.",
                        endpoint,
                    )
                    self._no_system_role.add(endpoint)
                elif any(kw in err_lower for kw in _NO_THINKING_KWARGS_ERRORS) and resolved.extra_body:
                    logger.info(
                        "Endpoint %s rejected thinking-control extra_body params — "
                        "retrying without enable_thinking / thinking_budget.",
                        endpoint,
                    )
                    self._no_thinking_kwargs.add(endpoint)
                elif response_schema is not None and any(
                    kw in err_lower for kw in _CONSTRAINED_DECODING_ERRORS
                ):
                    logger.info(
                        "Endpoint %s does not support constrained decoding — "
                        "retrying with prompt-only JSON enforcement.",
                        endpoint,
                    )
                    self._no_constrained_decoding.add(endpoint)
                else:
                    raise

        raise RuntimeError(
            f"Request to {endpoint} failed after 4 fallback attempts. "
            "Model may not be compatible with this pipeline."
        )

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
        response_schema: Type[BaseModel] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        extra_params: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream text chunks from the LLM endpoint.

        Yields raw string chunks as they arrive from the model. Callers should
        accumulate them and call ``parse_structured()`` on the joined text.

        Graceful fallback: if the endpoint does not support streaming (or the
        streaming call fails for any reason), this falls back to a single
        blocking ``generate()`` call and yields the full response as one chunk.
        Models without streaming support therefore behave identically from the
        caller's perspective — they just don't produce intermediate tokens.
        """
        resolved = self._registry.get(role)
        client = self._get_client(resolved)
        endpoint = resolved.endpoint

        kwargs_base: dict[str, Any] = {
            "model": resolved.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if extra_params:
            kwargs_base.update(extra_params)

        # Apply per-endpoint sanitization (strips enable_thinking / thinking_budget
        # for endpoints that have previously rejected those params).
        eff_extra_body = (
            self._sanitized_extra_body(endpoint, resolved.extra_body)
            if resolved.extra_body else {}
        )
        if eff_extra_body:
            kwargs_base["extra_body"] = eff_extra_body

        use_constrained = (
            response_schema is not None
            and endpoint not in self._no_constrained_decoding
        )
        eff_system = (
            system_prompt + _json_prompt_suffix(response_schema)
            if response_schema and not use_constrained
            else system_prompt
        )
        kwargs: dict[str, Any] = {
            **kwargs_base,
            "messages": self._messages(endpoint, eff_system, user_prompt),
        }
        if use_constrained:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_schema.__name__,
                    "schema": response_schema.model_json_schema(),
                },
            }

        # Attempt streaming — detect capability errors inline, fall back on anything else.
        stream_obj = None
        try:
            stream_obj = await client.chat.completions.create(**kwargs)
        except BadRequestError as exc:
            err_lower = str(exc).lower()
            if any(kw in err_lower for kw in _NO_THINKING_KWARGS_ERRORS) and resolved.extra_body:
                logger.info(
                    "Streaming: endpoint %s rejected thinking-control extra_body — "
                    "flagging; fallback will retry without those params.",
                    endpoint,
                )
                self._no_thinking_kwargs.add(endpoint)
            else:
                logger.warning(
                    "Streaming request failed for %s (%s: %s) — falling back.",
                    role, type(exc).__name__, exc,
                )
        except Exception as exc:
            logger.warning(
                "Streaming request failed for %s (%s: %s) — falling back.",
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

                    # Reasoning models (e.g. step-3.5-flash, QwQ) may route their
                    # output through reasoning_content when response_format=json_schema
                    # is active — delta.content stays empty the entire stream.
                    # Fall back to reasoning_content so we at least have raw text to
                    # attempt parsing.  We do NOT yield reasoning tokens for display
                    # (they aren't part of the structured output) but we do accumulate
                    # them so the empty-stream fallback below is not triggered.
                    if not token:
                        token = getattr(delta, "reasoning_content", None) or ""

                    if token:
                        detector.feed(token)     # raises ThinkLoopError if loop found
                        yielded_any = True
                        yield token
            except ThinkLoopError:
                logger.warning(
                    "Think loop detected for %s — aborting stream and re-raising "
                    "for retry.",
                    role,
                )
                raise   # propagate up through run_streaming → orchestrator retry
            except Exception as exc:
                logger.warning(
                    "Streaming interrupted for %s (%s: %s) — falling back.",
                    role, type(exc).__name__, exc,
                )
                # Fall through to non-streaming fallback below.
            else:
                # Stream completed without exception.
                if not yielded_any and use_constrained:
                    # Model returned HTTP 200 but emitted zero content tokens while
                    # constrained decoding (response_format=json_schema) was active.
                    # This is a silent capability gap — mark the endpoint and fall
                    # through to the prompt-only non-streaming fallback.
                    logger.info(
                        "Endpoint %s returned empty stream with constrained decoding — "
                        "flagging as no-constrained-decoding; falling back to "
                        "prompt-only non-streaming call.",
                        endpoint,
                    )
                    self._no_constrained_decoding.add(endpoint)
                else:
                    return  # streaming complete with content

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
