"""Unit tests for PolicyAgent prompt building and run() behavior.

Verifies:
  - _build_user_prompt() includes/excludes sections based on inputs
  - run() returns PolicyVerdict from mocked LLM
  - run_streaming() yields str tokens then PolicyVerdict
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator
from unittest.mock import MagicMock

import pytest

from backend.api.schemas.agents import (
    CheckerReport,
    CheckerVerdict,
    Issue,
    PolicyViolation,
    PolicyVerdict,
    RiskLevel,
    Severity,
    VerificationResult,
)
from backend.services.agents.policy import PolicyAgent
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


def _make_policy(chunks: list[str] | None = None) -> PolicyAgent:
    agent = PolicyAgent.__new__(PolicyAgent)
    agent._llm = _mock_llm(chunks or [])
    agent._system_prompt = "You are a policy auditor."
    return agent


def _passing_verdict_json() -> str:
    return PolicyVerdict(compliant=True, risk_level=RiskLevel.LOW).model_dump_json()


def _failing_verdict_json() -> str:
    return PolicyVerdict(
        compliant=False,
        risk_level=RiskLevel.HIGH,
        violations=[PolicyViolation(rule_id="DO-178C-§6.3.1", description="missing null check")],
    ).model_dump_json()


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

class TestPolicyPromptBuilding:
    def _agent(self) -> PolicyAgent:
        return _make_policy()

    def test_source_code_always_in_prompt(self):
        agent = self._agent()
        prompt = agent._build_user_prompt(source_code="def f(): pass", standard="DO_178C")
        assert "## Source Code" in prompt
        assert "def f(): pass" in prompt

    def test_task_section_always_present(self):
        agent = self._agent()
        prompt = agent._build_user_prompt(source_code="x = 1", standard="DO_178C")
        assert "## Task" in prompt

    def test_standard_always_in_prompt(self):
        agent = self._agent()
        prompt = agent._build_user_prompt(source_code="x = 1", standard="DO_178C")
        assert "DO_178C" in prompt

    def test_dafny_spec_included_when_provided(self):
        agent = self._agent()
        prompt = agent._build_user_prompt(
            source_code="x = 1", standard="DO_178C", dafny_spec="method F(){}"
        )
        assert "## Dafny Specification" in prompt
        assert "method F(){}" in prompt

    def test_dafny_spec_omitted_when_empty(self):
        agent = self._agent()
        prompt = agent._build_user_prompt(source_code="x = 1", standard="DO_178C", dafny_spec="")
        assert "## Dafny Specification" not in prompt

    def test_checker_verdict_in_prompt(self):
        agent = self._agent()
        report = CheckerReport(verdict=CheckerVerdict.FAIL)
        prompt = agent._build_user_prompt(
            source_code="x = 1", standard="DO_178C", checker_report=report
        )
        assert "Checker Verdict" in prompt
        assert "fail" in prompt

    def test_checker_issues_listed(self):
        agent = self._agent()
        issue = Issue(description="null pointer dereference", severity=Severity.CRITICAL)
        report = CheckerReport(verdict=CheckerVerdict.FAIL, issues=[issue])
        prompt = agent._build_user_prompt(
            source_code="x = 1", standard="DO_178C", checker_report=report
        )
        assert "null pointer dereference" in prompt
        assert "critical" in prompt

    def test_verification_passed_in_prompt(self):
        agent = self._agent()
        vr = VerificationResult(verified=True)
        prompt = agent._build_user_prompt(
            source_code="x = 1", standard="DO_178C", verification_result=vr
        )
        assert "PASSED" in prompt

    def test_verification_failed_in_prompt(self):
        agent = self._agent()
        vr = VerificationResult(verified=False, failing_assertions=["postcondition might not hold"])
        prompt = agent._build_user_prompt(
            source_code="x = 1", standard="DO_178C", verification_result=vr
        )
        assert "FAILED" in prompt
        assert "postcondition might not hold" in prompt

    def test_policy_context_included_when_nonempty(self):
        agent = self._agent()
        prompt = agent._build_user_prompt(
            source_code="x = 1", standard="DO_178C", policy_context="§6.3.1 requirements"
        )
        assert "## Retrieved Policy Context" in prompt
        assert "§6.3.1 requirements" in prompt

    def test_policy_context_omitted_when_empty(self):
        agent = self._agent()
        prompt = agent._build_user_prompt(
            source_code="x = 1", standard="DO_178C", policy_context=""
        )
        assert "## Retrieved Policy Context" not in prompt

    def test_solver_output_not_in_policy_prompt(self):
        """solver_output belongs to DafnyArchitect, not Policy."""
        agent = self._agent()
        vr = VerificationResult(
            verified=False,
            failing_assertions=["precondition failed"],
            solver_output="Z3 solver output line 1\nline 2",
        )
        prompt = agent._build_user_prompt(
            source_code="x = 1", standard="DO_178C", verification_result=vr
        )
        assert "Z3 solver output" not in prompt

    def test_failing_assertions_in_prompt(self):
        agent = self._agent()
        vr = VerificationResult(
            verified=False,
            failing_assertions=["postcondition a", "postcondition b"],
        )
        prompt = agent._build_user_prompt(
            source_code="x = 1", standard="DO_178C", verification_result=vr
        )
        assert "postcondition a" in prompt
        assert "postcondition b" in prompt


# ---------------------------------------------------------------------------
# run() and run_streaming()
# ---------------------------------------------------------------------------

class TestPolicyAgentRun:
    @pytest.mark.asyncio
    async def test_run_returns_policy_verdict(self):
        from unittest.mock import AsyncMock
        agent = _make_policy()
        agent._llm.generate_structured = AsyncMock(
            return_value=PolicyVerdict(compliant=True, risk_level=RiskLevel.LOW)
        )
        result = await agent.run(source_code="def f(): pass", standard="DO_178C")
        assert isinstance(result, PolicyVerdict)

    @pytest.mark.asyncio
    async def test_run_compliant_true(self):
        from unittest.mock import AsyncMock
        agent = _make_policy()
        agent._llm.generate_structured = AsyncMock(
            return_value=PolicyVerdict(compliant=True, risk_level=RiskLevel.LOW)
        )
        result = await agent.run(source_code="def f(): pass", standard="DO_178C")
        assert result.compliant is True

    @pytest.mark.asyncio
    async def test_run_compliant_false_with_violations(self):
        from unittest.mock import AsyncMock
        agent = _make_policy()
        agent._llm.generate_structured = AsyncMock(
            return_value=PolicyVerdict(
                compliant=False,
                risk_level=RiskLevel.HIGH,
                violations=[PolicyViolation(rule_id="DO-178C-§6.3.1", description="missing null check")],
            )
        )
        result = await agent.run(source_code="def f(): pass", standard="DO_178C")
        assert result.compliant is False
        assert len(result.violations) > 0

    @pytest.mark.asyncio
    async def test_run_streaming_yields_tokens_then_verdict(self):
        verdict_json = _passing_verdict_json()
        agent = _make_policy(chunks=["chunk1 ", "chunk2 ", verdict_json])

        tokens: list[str] = []
        final = None
        async for item in agent.run_streaming(source_code="def f(): pass", standard="DO_178C"):
            if isinstance(item, str):
                tokens.append(item)
            else:
                final = item

        assert len(tokens) == 3
        assert isinstance(final, PolicyVerdict)
        assert final.compliant is True
