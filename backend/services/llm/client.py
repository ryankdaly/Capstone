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
        return response_model.model_validate(data)


def _json_prompt_suffix(schema: Type[BaseModel]) -> str:
    """Return a system prompt suffix that instructs the model to output JSON.

    Used when the provider doesn't support response_format=json_schema.
    """
    schema_str = json.dumps(schema.model_json_schema(), indent=2)
    return (
        "\n\n---\nIMPORTANT: You MUST respond with a single valid JSON object and "
        "nothing else. No markdown fences, no prose, no explanation — only raw JSON.\n"
        f"The JSON must conform to this schema:\n{schema_str}"
    )


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
