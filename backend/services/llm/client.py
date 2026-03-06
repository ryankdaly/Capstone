"""Async OpenAI-compatible LLM client.

Single wrapper used by all agents. Supports constrained decoding via
response_format (guided_json) when the server supports it (vLLM).
"""

from __future__ import annotations

import json
import logging
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

        # Constrained decoding: use response_format for vLLM guided_json,
        # fall back to json_object mode for other providers.
        if response_schema is not None:
            kwargs["extra_body"] = {
                "guided_json": response_schema.model_json_schema(),
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
        return response_model.model_validate_json(raw)
