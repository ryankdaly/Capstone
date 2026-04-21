"""Unit tests for DafnyArchitectAgent prompt building and run() behavior.

Verifies:
  - _build_user_prompt() includes requirement and source code always
  - verification_feedback injection: only failed verifications trigger the error section
  - solver_output is truncated to 20 lines
  - max_tokens is 1000 (compact spec requirement)
  - run() returns DafnySpec
  - run_streaming() yields tokens then DafnySpec
"""

from __future__ import annotations

from typing import Any, AsyncGenerator
from unittest.mock import MagicMock

import pytest

from backend.api.schemas.agents import DafnySpec, VerificationResult
from backend.services.agents.dafny_architect import DafnyArchitectAgent
from backend.services.llm.client import LLMClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_llm(chunks: list[str]) -> LLMClient:
    async def _fake_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
        for chunk in chunks:
            yield chunk

    llm = MagicMock(spec=LLMClient)
    llm.generate_stream = _fake_stream
    llm.parse_structured = LLMClient.parse_structured.__get__(
        MagicMock(spec=LLMClient), LLMClient
    )
    return llm


def _make_dafny_architect(chunks: list[str] | None = None) -> DafnyArchitectAgent:
    agent = DafnyArchitectAgent.__new__(DafnyArchitectAgent)
    agent._llm = _mock_llm(chunks or [])
    agent._system_prompt = "You are a Dafny architect."
    return agent


def _spec_json() -> str:
    return DafnySpec(dafny_source="method F(x: int) returns (r: int)\n  ensures r >= 0\n{}").model_dump_json()


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

class TestDafnyArchitectPromptBuilding:
    def _agent(self) -> DafnyArchitectAgent:
        return _make_dafny_architect()

    def test_requirement_always_in_prompt(self):
        agent = self._agent()
        prompt = agent._build_user_prompt(
            source_code="int abs(int x) { return x < 0 ? -x : x; }",
            requirement="Implement absolute value",
        )
        assert "## Requirement" in prompt
        assert "Implement absolute value" in prompt

    def test_source_code_always_in_prompt(self):
        agent = self._agent()
        prompt = agent._build_user_prompt(
            source_code="int abs(int x) { return x < 0 ? -x : x; }",
            requirement="Implement absolute value",
        )
        assert "## Source Code" in prompt
        assert "int abs(int x)" in prompt

    def test_language_defaults_to_c(self):
        agent = self._agent()
        prompt = agent._build_user_prompt(
            source_code="int f() { return 0; }",
            requirement="Return 0",
        )
        assert "(C)" in prompt

    def test_language_included_when_specified(self):
        agent = self._agent()
        prompt = agent._build_user_prompt(
            source_code="def f(): pass",
            requirement="Return nothing",
            language="Python",
        )
        assert "(Python)" in prompt

    def test_no_error_section_when_no_verification_feedback(self):
        agent = self._agent()
        prompt = agent._build_user_prompt(
            source_code="int f() { return 0; }",
            requirement="Return 0",
            verification_feedback=None,
        )
        assert "Previous Dafny Attempt" not in prompt
        assert "Errors from Previous" not in prompt

    def test_no_error_section_when_verification_passed(self):
        agent = self._agent()
        vr = VerificationResult(verified=True)
        prompt = agent._build_user_prompt(
            source_code="int f() { return 0; }",
            requirement="Return 0",
            verification_feedback=vr,
        )
        assert "Errors from Previous" not in prompt

    def test_error_section_injected_when_verification_failed(self):
        agent = self._agent()
        vr = VerificationResult(
            verified=False,
            failing_assertions=["postcondition might not hold"],
        )
        prompt = agent._build_user_prompt(
            source_code="int f() { return 0; }",
            requirement="Return 0",
            verification_feedback=vr,
        )
        assert "## Errors from Previous Dafny Attempt" in prompt
        assert "postcondition might not hold" in prompt

    def test_solver_output_truncated_to_20_lines(self):
        agent = self._agent()
        lines = [f"solver line {i}" for i in range(50)]
        vr = VerificationResult(
            verified=False,
            failing_assertions=["err"],
            solver_output="\n".join(lines),
        )
        prompt = agent._build_user_prompt(
            source_code="int f() { return 0; }",
            requirement="Return 0",
            verification_feedback=vr,
        )
        assert "solver line 19" in prompt
        assert "solver line 20" not in prompt

    def test_solver_output_omitted_when_empty(self):
        agent = self._agent()
        vr = VerificationResult(
            verified=False,
            failing_assertions=["err"],
            solver_output="",
        )
        prompt = agent._build_user_prompt(
            source_code="int f() { return 0; }",
            requirement="Return 0",
            verification_feedback=vr,
        )
        assert "Solver output" not in prompt

    def test_max_tokens_is_1000(self):
        assert DafnyArchitectAgent.max_tokens == 1000


# ---------------------------------------------------------------------------
# run() and run_streaming()
# ---------------------------------------------------------------------------

class TestDafnyArchitectAgentRun:
    @pytest.mark.asyncio
    async def test_run_returns_dafny_spec(self):
        from unittest.mock import AsyncMock
        agent = _make_dafny_architect()
        agent._llm.generate_structured = AsyncMock(
            return_value=DafnySpec(dafny_source="method F() {}")
        )
        result = await agent.run(
            source_code="int f() { return 0; }",
            requirement="Return 0",
        )
        assert isinstance(result, DafnySpec)

    @pytest.mark.asyncio
    async def test_run_dafny_source_populated(self):
        from unittest.mock import AsyncMock
        agent = _make_dafny_architect()
        agent._llm.generate_structured = AsyncMock(
            return_value=DafnySpec(dafny_source="method F() { }")
        )
        result = await agent.run(
            source_code="int f() { return 0; }",
            requirement="Return 0",
        )
        assert result.dafny_source == "method F() { }"

    @pytest.mark.asyncio
    async def test_run_streaming_yields_tokens_then_spec(self):
        spec_json = _spec_json()
        agent = _make_dafny_architect(chunks=["tok1 ", "tok2 ", spec_json])

        tokens: list[str] = []
        final = None
        async for item in agent.run_streaming(
            source_code="int f() { return 0; }",
            requirement="Return 0",
        ):
            if isinstance(item, str):
                tokens.append(item)
            else:
                final = item

        assert len(tokens) == 3
        assert isinstance(final, DafnySpec)

    @pytest.mark.asyncio
    async def test_verification_feedback_propagates_to_prompt(self):
        """Failed verification errors must appear in the user prompt."""
        captured_prompts: list[str] = []

        async def _capture_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
            captured_prompts.append(kwargs.get("user_prompt", ""))
            yield _spec_json()

        agent = _make_dafny_architect()
        agent._llm.generate_stream = _capture_stream

        vr = VerificationResult(verified=False, failing_assertions=["invariant failed"])
        tokens = []
        async for item in agent.run_streaming(
            source_code="int f() { return 0; }",
            requirement="Return 0",
            verification_feedback=vr,
        ):
            tokens.append(item)

        assert captured_prompts, "generate_stream not called"
        assert "invariant failed" in captured_prompts[0]
