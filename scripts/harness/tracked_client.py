"""TrackedLLMClient — token-counting wrapper for batch eval harness runs.

Why force non-streaming?
  The live HPEMA CLI uses generate_stream() so users see tokens appear in real
  time. In batch eval runs we have no user watching — we care about token counts
  and final output, not latency UX. Forcing everything through the blocking
  generate() path gives us a single clean interception point (_api_call) where
  the full response object — including response.usage — is always present.

What is tracked:
  Per-role accumulation: input_tokens, output_tokens, cached_input_tokens.
  Roles: "actor", "checker", "policy", "dafny_architect".
  A summary dict is available via get_usage_summary() after the pipeline run.

Usage in run_hpema.py:
    client = TrackedLLMClient(registry=registry)
    orchestrator = PipelineOrchestrator(llm_client=client)
    async for _ in orchestrator.run(request):
        pass
    usage = client.get_usage_summary()
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, AsyncGenerator, Type

from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.llm.client import LLMClient
from backend.services.llm.model_registry import ModelRegistry


class TrackedLLMClient(LLMClient):
    """LLMClient subclass that accumulates token usage across all agent calls.

    Forces non-streaming mode so every call passes through _api_call(), which
    returns the full response object with .usage attached. This is safe for
    batch harness runs where live token display is not needed.
    """

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        super().__init__(registry=registry)
        # role → {"input": int, "output": int, "cached": int}
        self._usage: dict[str, dict[str, int]] = defaultdict(
            lambda: {"input": 0, "output": 0, "cached": 0}
        )
        self._current_role: str = "unknown"

    # ── Usage capture hook ────────────────────────────────────────────────────

    def _accum(self, role: str, response: Any) -> None:
        """Pull token counts from the API response object and accumulate."""
        usage = getattr(response, "usage", None)
        if not usage:
            return
        self._usage[role]["input"] += getattr(usage, "prompt_tokens", 0) or 0
        self._usage[role]["output"] += getattr(usage, "completion_tokens", 0) or 0
        # Cached input tokens (OpenAI prompt_tokens_details; may be absent)
        details = getattr(usage, "prompt_tokens_details", None)
        self._usage[role]["cached"] += getattr(details, "cached_tokens", 0) or 0

    async def _api_call(self, client: Any, kwargs: dict[str, Any]) -> Any:
        """Override: intercept every blocking completion to capture usage."""
        response = await super()._api_call(client, kwargs)
        self._accum(self._current_role, response)
        return response

    # ── Role tagging ─────────────────────────────────────────────────────────

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
        """Tag current role, then delegate to parent generate()."""
        self._current_role = role
        return await super().generate(
            role=role,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=response_schema,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_params=extra_params,
        )

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
        """Override: force non-streaming for batch harness runs.

        The parent's generate_stream() is a live-display feature. In eval batch
        mode we have no viewer — skip it and route through the blocking generate()
        path so _api_call() captures usage. Output is yielded as a single chunk.
        """
        self._current_role = role
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

    # ── Summary accessors ─────────────────────────────────────────────────────

    def get_usage_summary(self) -> dict:
        """Return accumulated token counts, both per-role and totals."""
        by_role = {role: dict(counts) for role, counts in self._usage.items()}
        return {
            "by_role": by_role,
            "total_input_tokens": sum(v["input"] for v in self._usage.values()),
            "total_output_tokens": sum(v["output"] for v in self._usage.values()),
            "total_cached_input_tokens": sum(v["cached"] for v in self._usage.values()),
        }

    def actor_tokens(self) -> dict[str, int]:
        """Convenience: return only actor role token counts."""
        return dict(self._usage.get("actor", {"input": 0, "output": 0, "cached": 0}))

    def reset(self) -> None:
        """Clear accumulated counts — call between pipeline runs if reusing client."""
        self._usage.clear()
        self._current_role = "unknown"
