"""Async OpenAI-compatible LLM client.

Single wrapper used by all agents. Supports constrained decoding via
response_format (guided_json) when the server supports it (vLLM / OpenAI).
Falls back to prompt-only JSON enforcement for providers that don't support
the json_schema response_format (Groq, Anthropic, older endpoints, etc.).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Type

from openai import AsyncOpenAI, BadRequestError
from pydantic import BaseModel, ValidationError

from backend.services.llm.model_registry import ModelRegistry, ResolvedModel

logger = logging.getLogger(__name__)

# Error substrings that indicate the provider rejected the response_format param.
# This list covers common provider error messages.
_CONSTRAINED_DECODING_ERRORS = (
    "response_format",
    "json_schema",
    "guided_json",
    "unsupported",
    "not supported",
    "invalid_request_error",
)


class LLMClient:
    """Async client that talks to any OpenAI-compatible endpoint."""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self._registry = registry or ModelRegistry()
        self._clients: dict[str, AsyncOpenAI] = {}
        # Track which endpoints don't support constrained decoding so we skip
        # the first attempt on subsequent calls.
        self._no_constrained_decoding: set[str] = set()

    def _get_client(self, model: ResolvedModel) -> AsyncOpenAI:
        """Lazily create one AsyncOpenAI client per unique endpoint."""
        if model.endpoint not in self._clients:
            self._clients[model.endpoint] = AsyncOpenAI(
                base_url=model.endpoint,
                api_key=model.api_key or "unused",
            )
        return self._clients[model.endpoint]

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

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        kwargs: dict[str, Any] = {
            "model": resolved.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if extra_params:
            kwargs.update(extra_params)

        use_constrained = (
            response_schema is not None
            and resolved.endpoint not in self._no_constrained_decoding
        )

        if use_constrained:
            # Constrained decoding: json_schema response_format enforces the
            # schema at the token level. Supported by vLLM and OpenAI.
            schema_name = response_schema.__name__
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": response_schema.model_json_schema(),
                },
            }

        logger.info(
            "LLM request to %s [%s] model=%s constrained=%s",
            role, resolved.endpoint, resolved.model, use_constrained,
        )

        try:
            response = await client.chat.completions.create(**kwargs)
        except BadRequestError as exc:
            # Provider rejected response_format — fall back to prompt-only mode.
            err_lower = str(exc).lower()
            if response_schema is not None and any(
                kw in err_lower for kw in _CONSTRAINED_DECODING_ERRORS
            ):
                logger.warning(
                    "Endpoint %s rejected constrained decoding (%s). "
                    "Falling back to prompt-only JSON enforcement.",
                    resolved.endpoint, exc,
                )
                self._no_constrained_decoding.add(resolved.endpoint)
                # Retry without response_format, schema injected into system prompt
                kwargs.pop("response_format", None)
                kwargs["messages"] = [
                    {
                        "role": "system",
                        "content": system_prompt + _json_prompt_suffix(response_schema),
                    },
                    {"role": "user", "content": user_prompt},
                ]
                response = await client.chat.completions.create(**kwargs)
            else:
                raise

        content = response.choices[0].message.content or ""
        logger.debug("LLM response from %s: %s chars", role, len(content))
        return content

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
        # If this endpoint is already known to not support constrained decoding,
        # inject the schema into the system prompt up front.
        resolved = self._registry.get(role)
        if resolved.endpoint in self._no_constrained_decoding:
            system_prompt = system_prompt + _json_prompt_suffix(response_model)

        raw = await self.generate(
            role=role,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=response_model,
            **kwargs,
        )
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
            # Optional[X] — use the non-None arg
            ann = next((a for a in args if a is not type(None)), str)

        if ann is str:
            # Use the default if it's non-empty, else a placeholder
            default = field_info.default
            if default and isinstance(default, str):
                example[name] = default
            else:
                example[name] = placeholder
        elif ann is int:
            example[name] = field_info.default if field_info.default is not None else 0
        elif ann is bool:
            example[name] = field_info.default if field_info.default is not None else False
        elif ann is float:
            example[name] = field_info.default if field_info.default is not None else 0.0
        elif inspect.isclass(ann) and issubclass(ann, list):
            example[name] = []
        elif inspect.isclass(ann) and issubclass(ann, dict):
            example[name] = {}
        elif origin is list:
            example[name] = []
        elif origin is dict:
            example[name] = {}
        else:
            # Enum or nested model — use default or a string placeholder
            default = field_info.default
            if default is not None and default is not ...:
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
