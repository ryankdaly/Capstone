"""Base agent abstract class.

All HPEMA agents (Actor, Checker, Policy) inherit from this. It handles
prompt loading, LLM invocation, and structured output parsing.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, AsyncGenerator, Type

from pydantic import BaseModel

from backend.services.llm.client import LLMClient, _strip_think_tags

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "llm" / "prompts"


class BaseAgent(ABC):
    """Abstract base for all pipeline agents."""

    role: str  # "actor", "checker", "policy" — matches model registry keys
    prompt_file: str  # filename in prompts/ directory
    max_tokens: int = 2000  # must leave room for input tokens within vLLM's max-model-len

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client
        self._system_prompt = self._load_prompt()

    def _load_prompt(self) -> str:
        path = PROMPTS_DIR / self.prompt_file
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return path.read_text().strip()

    @abstractmethod
    def _build_user_prompt(self, **kwargs: Any) -> str:
        """Build the user message from the agent's specific inputs."""
        ...

    @abstractmethod
    def _output_schema(self) -> Type[BaseModel]:
        """Return the Pydantic model class for this agent's output."""
        ...

    async def run(self, **kwargs: Any) -> BaseModel:
        """Execute the agent: build prompt → call LLM → parse structured output."""
        user_prompt = self._build_user_prompt(**kwargs)

        logger.info("Agent [%s] starting", self.role)

        result = await self._llm.generate_structured(
            role=self.role,
            system_prompt=self._system_prompt,
            user_prompt=user_prompt,
            response_model=self._output_schema(),
            max_tokens=self.max_tokens,
        )

        logger.info("Agent [%s] completed", self.role)
        return result

    async def run_streaming(self, **kwargs: Any) -> AsyncGenerator:
        """Streaming variant: yields raw str tokens while the LLM generates,
        then yields the final parsed BaseModel as the last item.

        Usage in orchestrator::

            async for item in agent.run_streaming(...):
                if isinstance(item, str):
                    # forward as AGENT_TOKEN event
                    ...
                else:
                    result = item   # final parsed output

        Falls back to a single-token yield (non-streaming) when the endpoint
        does not support streaming — callers are unaffected.
        """
        user_prompt = self._build_user_prompt(**kwargs)
        logger.info("Agent [%s] starting (streaming)", self.role)

        chunks: list[str] = []
        async for chunk in self._llm.generate_stream(
            role=self.role,
            system_prompt=self._system_prompt,
            user_prompt=user_prompt,
            response_schema=self._output_schema(),
            max_tokens=self.max_tokens,
        ):
            chunks.append(chunk)
            yield chunk

        full_text = _strip_think_tags("".join(chunks))
        result = self._llm.parse_structured(full_text, self._output_schema(), role=self.role)
        logger.info("Agent [%s] completed", self.role)
        yield result
