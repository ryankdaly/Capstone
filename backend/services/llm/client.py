"""Async OpenAI-compatible LLM client.

Single wrapper used by all agents. Supports constrained decoding via
response_format (guided_json) when the server supports it (vLLM).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Type

from openai import AsyncOpenAI
from pydantic import BaseModel

from backend.services.llm.model_registry import ModelRegistry, ResolvedModel

logger = logging.getLogger(__name__)


class LLMClient:
    """Async client that talks to any OpenAI-compatible endpoint."""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self._registry = registry or ModelRegistry()
        self._clients: dict[str, AsyncOpenAI] = {}

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
            If provided, enables constrained decoding (guided_json) so the
            model's output is guaranteed to be valid JSON matching the schema.
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

        # Constrained decoding: use OpenAI-standard response_format with
        # json_schema type. vLLM and OpenAI both support this — it enforces
        # the schema at the token level during generation.
        if response_schema is not None:
            schema_name = response_schema.__name__
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": response_schema.model_json_schema(),
                },
            }

        if extra_params:
            kwargs.update(extra_params)

        logger.info("LLM request to %s [%s] model=%s", role, resolved.endpoint, resolved.model)

        response = await client.chat.completions.create(**kwargs)
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

        Uses constrained decoding to guarantee schema compliance.
        """
        raw = await self.generate(
            role=role,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=response_model,
            **kwargs,
        )
        raw = _fix_code_formatting(raw)
        return response_model.model_validate_json(raw)


def _fix_code_formatting(raw_json: str) -> str:
    """Post-process LLM JSON to fix single-line code output.

    Small models sometimes emit code without newlines in JSON strings.
    This detects that case and inserts newlines at statement boundaries.
    """
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return raw_json

    changed = False
    for field in ("source_code", "dafny_spec"):
        code = data.get(field, "")
        if not code or "\n" in code:
            continue
        # Code has no newlines but has statement-ending characters — reformat
        if ";" in code or "{" in code:
            # Insert newlines after ; { } (basic C/Dafny formatting)
            code = re.sub(r";\s*", ";\n", code)
            code = re.sub(r"\{\s*", "{\n", code)
            code = re.sub(r"\}\s*", "}\n", code)
            data[field] = code
            changed = True

    return json.dumps(data) if changed else raw_json
