"""Base agent abstract class.

All HPEMA agents (Actor, Checker, Policy) inherit from this. It handles
prompt loading, LLM invocation, and structured output parsing.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Type

from pydantic import BaseModel

from backend.services.llm.client import LLMClient, ThinkLoopError, _strip_think_tags

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "llm" / "prompts"
_AGENT_LOG = Path(__file__).resolve().parent.parent.parent.parent / "logs" / "agent.log"
_agent_log_lock = threading.Lock()

# --------------------------------------------------------------------------- #
# Debug log helper                                                             #
# --------------------------------------------------------------------------- #

_DIVIDER = "=" * 80
_SEP     = "-" * 80

def _write_agent_log(
    role: str,
    user_prompt: str,
    full_raw: str,
    full_stripped: str,
    parse_ok: bool,
    parse_error: str | None = None,
) -> None:
    """Append one agent response record to agent.log (thread-safe, never raises)."""
    try:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        status = "OK" if parse_ok else f"PARSE_FAIL: {parse_error}"
        lines = [
            _DIVIDER,
            f"AGENT : {role}",
            f"TIME  : {ts}",
            f"STATUS: {status}",
            _SEP,
            "USER PROMPT:",
            user_prompt,
            _SEP,
            "RAW RESPONSE:",
            full_raw,
            _SEP,
            "STRIPPED RESPONSE:",
            full_stripped if full_stripped != full_raw else "(same as raw)",
            _DIVIDER,
            "",
        ]
        with _agent_log_lock:
            with _AGENT_LOG.open("a", encoding="utf-8") as fh:
                fh.write("\n".join(lines))
    except Exception:
        pass  # debug log must never crash the pipeline


class BaseAgent(ABC):
    """Abstract base for all pipeline agents."""

    role: str  # "actor", "checker", "policy" — matches model registry keys
    prompt_file: str  # filename in prompts/ directory
    max_tokens: int = 8192  # thinking models spend 1 000–4 000 tokens in <think> before JSON

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

        try:
            result = await self._llm.generate_structured(
                role=self.role,
                system_prompt=self._system_prompt,
                user_prompt=user_prompt,
                response_model=self._output_schema(),
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            _write_agent_log(self.role, user_prompt, "(non-streaming — raw not captured)", "", False, str(exc))
            raise

        _write_agent_log(self.role, user_prompt, "(non-streaming — raw not captured)", "", True)
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
        try:
            async for chunk in self._llm.generate_stream(
                role=self.role,
                system_prompt=self._system_prompt,
                user_prompt=user_prompt,
                response_schema=self._output_schema(),
                max_tokens=self.max_tokens,
            ):
                chunks.append(chunk)
                yield chunk
        except ThinkLoopError as exc:
            partial = "".join(chunks)
            _write_agent_log(
                self.role, user_prompt,
                partial + f"\n[ABORTED — ThinkLoopError: {exc}]",
                "", False, str(exc),
            )
            raise

        full_raw = "".join(chunks)
        full_stripped = _strip_think_tags(full_raw)

        # Try stripped text first; fall back to original (handles JSON-inside-think models).
        candidates = list(dict.fromkeys([full_stripped, full_raw]))  # deduplicated, ordered
        parse_error: Exception | None = None
        for candidate in candidates:
            try:
                result = self._llm.parse_structured(
                    candidate, self._output_schema(), role=self.role
                )
                _write_agent_log(self.role, user_prompt, full_raw, full_stripped, True)
                logger.info("Agent [%s] completed", self.role)
                yield result
                return
            except Exception as exc:
                parse_error = exc
                label = "stripped" if candidate is full_stripped else "original"
                logger.debug(
                    "Agent [%s] parse attempt failed on %s text: %s",
                    self.role, label, exc,
                )

        _write_agent_log(self.role, user_prompt, full_raw, full_stripped, False, str(parse_error))
        raise ValueError(
            f"Agent [{self.role}] failed to parse LLM output. "
            f"Last error: {parse_error}. "
            f"Raw output (first 300 chars): {full_raw[:300]}"
        ) from parse_error
