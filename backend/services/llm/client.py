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

from openai import AsyncOpenAI, BadRequestError, InternalServerError, RateLimitError
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

# 400 errors that are transient platform failures, not capability gaps.
# These should be retried like a 500, not treated as permanent incompatibilities.
_TRANSIENT_REQUEST_ERRORS = (
    "degraded",
    "function cannot be invoked",
    "service unavailable",
    "temporarily unavailable",
)


class LLMClient:
    """Async client that talks to any OpenAI-compatible endpoint."""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self._registry = registry or ModelRegistry()
        self._clients: dict[str, AsyncOpenAI] = {}
        # Track capability gaps per endpoint so we skip the overhead of a
        # failing first attempt on subsequent calls to the same endpoint.
        self._no_constrained_decoding: set[str] = set()
        self._no_system_role: set[str] = set()

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
        """Execute one chat completion, retrying on transient server errors.

        Retries on:
          - InternalServerError (500): EngineCore crash, vLLM restart
          - RateLimitError (429): upstream throttling
          - BadRequestError (400) matching _TRANSIENT_REQUEST_ERRORS: NVIDIA NIM
            "DEGRADED function" and similar platform-unavailability messages

        Genuine 400s (wrong response_format, unsupported system role) do NOT
        match _TRANSIENT_REQUEST_ERRORS and are re-raised immediately so the
        outer capability-detection loop in generate() can handle them.
        """
        delays = [5, 15]  # seconds between attempts 1→2 and 2→3
        for attempt, delay in enumerate([-1] + delays):  # attempt 0 has no pre-sleep
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
                # Some providers (NVIDIA NIM) return 400 for transient platform
                # failures ("DEGRADED function cannot be invoked"). These are not
                # capability gaps — retry them like a 500. Genuine 400s (wrong
                # response_format, unsupported system role) do NOT match these
                # patterns and are re-raised immediately for the outer loop.
                if any(kw in str(exc).lower() for kw in _TRANSIENT_REQUEST_ERRORS):
                    if attempt == len(delays):
                        raise
                    logger.warning("Transient 400 from API (%s): %s", type(exc).__name__, exc)
                else:
                    raise

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
        # Model-level extra_body (e.g. chat_template_kwargs for NVIDIA NIM) is
        # passed via the openai SDK's extra_body so it lands in the JSON payload
        # without the SDK stripping unknown fields.
        if resolved.extra_body:
            kwargs_base["extra_body"] = resolved.extra_body

        # Retry loop — each pass may discover one new capability gap and adapt.
        # At most 3 attempts: (1) preferred mode, (2) one fallback, (3) both fallbacks.
        for _attempt in range(3):
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

            logger.info(
                "LLM request to %s [%s] model=%s constrained=%s no_system_role=%s",
                role, endpoint, resolved.model, use_constrained,
                endpoint in self._no_system_role,
            )

            try:
                response = await self._api_call(client, kwargs)
                content = response.choices[0].message.content or ""
                content = _strip_think_tags(content)
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
            f"Request to {endpoint} failed after 3 fallback attempts. "
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
        if resolved.extra_body:
            kwargs_base["extra_body"] = resolved.extra_body

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

        # Attempt streaming — fall back to non-streaming on any failure.
        stream_obj = None
        try:
            stream_obj = await client.chat.completions.create(**kwargs)
        except Exception as exc:
            logger.warning(
                "Streaming request failed for %s (%s: %s) — falling back.",
                role, type(exc).__name__, exc,
            )

        if stream_obj is not None:
            try:
                async for chunk in stream_obj:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                return  # streaming complete
            except Exception as exc:
                logger.warning(
                    "Streaming interrupted for %s (%s: %s) — falling back.",
                    role, type(exc).__name__, exc,
                )
                # Fall through to non-streaming fallback below.

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

    Models like Gemma 4 (enable_thinking=True), DeepSeek-R1, and QwQ emit
    chain-of-thought inside these tags before the actual response. We want
    only the final answer — the think block content is discarded.
    """
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


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
