"""Unit tests for orchestrator feedback loop, convergence, and regression detection.

Verifies:
  - Single passing iteration terminates immediately (awaiting_approval)
  - Fail-then-pass loop: Actor receives FeedbackMessage on 2nd call
  - max_iterations exhausted → status=failed
  - TestRunner gating: run_tests flag, language, test_cases presence
  - Regression detection: REGRESSION: prefix when score drops
  - Policy stage convergence: checker+policy both must pass
"""

from __future__ import annotations

from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.api.schemas.agents import (
    CheckerReport,
    CheckerVerdict,
    CodeCandidate,
    DafnySpec,
    FeedbackMessage,
    Issue,
    PolicyVerdict,
    PolicyViolation,
    RiskLevel,
    Severity,
    TestRunResult,
    VerificationResult,
)
from backend.api.schemas.pipeline import (
    PipelineRequest,
    PipelineStage,
    PipelineStatus,
    StreamEventType,
)
from backend.services.orchestrator import PipelineOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _SequencedMock:
    """Agent mock that returns a different result on each call."""

    def __init__(self, outcomes: list[Any]) -> None:
        self.calls: int = 0
        self.outcomes = outcomes
        self._call_kwargs: list[dict] = []

    async def run_streaming(self, **kwargs: Any) -> AsyncGenerator:
        result = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self._call_kwargs.append(kwargs)
        self.calls += 1
        yield result


def _make_orchestrator(
    actor_outcomes: list[Any] | None = None,
    checker_outcomes: list[Any] | None = None,
    policy_outcomes: list[Any] | None = None,
    stage: PipelineStage = PipelineStage.CHECKER,
) -> PipelineOrchestrator:
    orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
    orch._audit = MagicMock()
    orch._retriever = MagicMock()
    orch._retriever.retrieve.return_value = "context"
    orch.last_state = None

    default_code = CodeCandidate(source_code="def f(): pass", language="Python")
    actor_outs = actor_outcomes or [default_code]
    orch._actor = _SequencedMock(actor_outs)

    default_arch = DafnySpec(dafny_source="method F() {}")

    async def _simple_arch(**kwargs: Any) -> AsyncGenerator:
        yield default_arch

    arch_mock = MagicMock()
    arch_mock.run_streaming = _simple_arch
    orch._dafny_architect = arch_mock

    default_checker = CheckerReport(verdict=CheckerVerdict.PASS)
    checker_outs = checker_outcomes or [default_checker]
    orch._checker = _SequencedMock(checker_outs)

    default_policy = PolicyVerdict(compliant=True, risk_level=RiskLevel.LOW)
    policy_outs = policy_outcomes or [default_policy]
    orch._policy = _SequencedMock(policy_outs)

    dafny_runner = MagicMock()
    dafny_runner.verify = AsyncMock(return_value=VerificationResult(verified=True))
    orch._dafny = dafny_runner

    test_runner = MagicMock()
    test_runner.run = AsyncMock(return_value=TestRunResult(executed=True, total=1, passed=1, failed=0))
    orch._test_runner = test_runner

    orch._llm = MagicMock()
    return orch


def _req(
    stage: PipelineStage = PipelineStage.CHECKER,
    max_iterations: int = 3,
    run_tests: bool = False,
    language: str = "Python",
) -> PipelineRequest:
    from backend.api.schemas.generation import TargetLanguage
    lang_map = {"Python": TargetLanguage.PYTHON, "C": TargetLanguage.C}
    return PipelineRequest(
        requirement_text="test requirement",
        stage=stage,
        max_iterations=max_iterations,
        run_tests=run_tests,
        target_language=lang_map[language],
    )


async def _collect(orch: PipelineOrchestrator, req: PipelineRequest) -> list:
    events = []
    async for e in orch.run(req):
        events.append(e)
    return events


def _pipeline_status(events: list) -> str:
    complete = [e for e in events if e.event_type == StreamEventType.PIPELINE_COMPLETE]
    assert complete, "no PIPELINE_COMPLETE event"
    return complete[0].data["status"]


def _iteration_count(events: list) -> int:
    return len([e for e in events if e.event_type == StreamEventType.ITERATION_COMPLETE])


# ---------------------------------------------------------------------------
# Checker convergence
# ---------------------------------------------------------------------------

class TestCheckerConvergence:
    @pytest.mark.asyncio
    async def test_single_pass_terminates_in_one_iteration(self):
        orch = _make_orchestrator(checker_outcomes=[CheckerReport(verdict=CheckerVerdict.PASS)])
        events = await _collect(orch, _req())
        assert _iteration_count(events) == 1
        assert _pipeline_status(events) == PipelineStatus.AWAITING_APPROVAL.value

    @pytest.mark.asyncio
    async def test_fail_then_pass_takes_two_iterations(self):
        orch = _make_orchestrator(checker_outcomes=[
            CheckerReport(verdict=CheckerVerdict.FAIL),
            CheckerReport(verdict=CheckerVerdict.PASS),
        ])
        events = await _collect(orch, _req(max_iterations=3))
        assert _iteration_count(events) == 2
        assert _pipeline_status(events) == PipelineStatus.AWAITING_APPROVAL.value

    @pytest.mark.asyncio
    async def test_actor_receives_feedback_on_second_iteration(self):
        orch = _make_orchestrator(checker_outcomes=[
            CheckerReport(verdict=CheckerVerdict.FAIL),
            CheckerReport(verdict=CheckerVerdict.PASS),
        ])
        await _collect(orch, _req(max_iterations=3))
        assert orch._actor.calls == 2
        second_kwargs = orch._actor._call_kwargs[1]
        assert "feedback" in second_kwargs
        assert second_kwargs["feedback"] is not None
        assert isinstance(second_kwargs["feedback"], FeedbackMessage)

    @pytest.mark.asyncio
    async def test_max_iterations_exhausted_gives_failed_status(self):
        always_fail = [CheckerReport(verdict=CheckerVerdict.FAIL)] * 3
        orch = _make_orchestrator(checker_outcomes=always_fail)
        events = await _collect(orch, _req(max_iterations=2))
        assert _pipeline_status(events) == PipelineStatus.FAILED.value

    @pytest.mark.asyncio
    async def test_test_runner_called_when_enabled_python_with_test_cases(self):
        checker_with_tests = CheckerReport(
            verdict=CheckerVerdict.PASS,
            test_cases=["assert f(1) == 1"],
        )
        orch = _make_orchestrator(checker_outcomes=[checker_with_tests])
        await _collect(orch, _req(run_tests=True, language="Python"))
        orch._test_runner.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_test_runner_not_called_when_run_tests_false(self):
        checker_with_tests = CheckerReport(
            verdict=CheckerVerdict.PASS,
            test_cases=["assert f(1) == 1"],
        )
        orch = _make_orchestrator(checker_outcomes=[checker_with_tests])
        await _collect(orch, _req(run_tests=False, language="Python"))
        orch._test_runner.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_test_runner_not_called_for_c_language(self):
        checker_with_tests = CheckerReport(
            verdict=CheckerVerdict.PASS,
            test_cases=["assert f(1) == 1"],
        )
        orch = _make_orchestrator(checker_outcomes=[checker_with_tests])
        await _collect(orch, _req(run_tests=True, language="C"))
        orch._test_runner.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_failing_tests_block_convergence(self):
        orch = _make_orchestrator(checker_outcomes=[CheckerReport(verdict=CheckerVerdict.PASS)])
        orch._test_runner.run = AsyncMock(
            return_value=TestRunResult(executed=True, total=1, passed=0, failed=1)
        )
        checker_with_tests = CheckerReport(
            verdict=CheckerVerdict.PASS,
            test_cases=["assert f(1) == 2"],
        )
        orch._checker = _SequencedMock([checker_with_tests, checker_with_tests])
        events = await _collect(orch, _req(run_tests=True, language="Python", max_iterations=2))
        assert _iteration_count(events) == 2
        assert _pipeline_status(events) == PipelineStatus.FAILED.value


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------

class TestRegressionDetection:
    @pytest.mark.asyncio
    async def test_regression_prefix_added_when_score_drops(self):
        """Both iterations fail, but iteration 1 has dafny PASS (higher score).
        Iteration 2 has dafny FAIL (lower score) → triggers REGRESSION: prefix."""
        from unittest.mock import AsyncMock as _AsyncMock
        orch = _make_orchestrator(
            stage=PipelineStage.CHECKER,
            checker_outcomes=[
                CheckerReport(verdict=CheckerVerdict.FAIL),
                CheckerReport(verdict=CheckerVerdict.FAIL),
                CheckerReport(verdict=CheckerVerdict.FAIL),
            ],
        )
        # Iteration 1: dafny passes (score = 0 + 1 + 0.5 + 0.25 = 1.75)
        # Iteration 2: dafny fails (score = 0 + 1 + 0 + 0.25 = 1.25) → regression
        orch._dafny.verify = _AsyncMock(side_effect=[
            VerificationResult(verified=True),
            VerificationResult(verified=False, failing_assertions=["postcondition failed"]),
            VerificationResult(verified=False),
        ])
        await _collect(orch, _req(max_iterations=3))
        state = orch.last_state
        assert state is not None
        assert len(state.iterations) >= 2
        feedback = state.iterations[1].feedback
        assert feedback is not None
        assert feedback.priority_summary.startswith("REGRESSION:")

    @pytest.mark.asyncio
    async def test_regression_prefix_not_added_on_improvement(self):
        orch = _make_orchestrator(
            stage=PipelineStage.CHECKER,
            checker_outcomes=[
                CheckerReport(verdict=CheckerVerdict.FAIL),
                CheckerReport(verdict=CheckerVerdict.PASS),
            ],
        )
        await _collect(orch, _req(max_iterations=3))
        state = orch.last_state
        # Iteration 0 (index 0) had a fail, iteration 1 passed — no regression
        assert len(state.iterations) == 2
        feedback_first = state.iterations[0].feedback
        assert feedback_first is None or not feedback_first.priority_summary.startswith("REGRESSION:")

    @pytest.mark.asyncio
    async def test_no_regression_on_single_iteration(self):
        orch = _make_orchestrator(
            stage=PipelineStage.CHECKER,
            checker_outcomes=[CheckerReport(verdict=CheckerVerdict.PASS)],
        )
        await _collect(orch, _req(max_iterations=3))
        state = orch.last_state
        assert len(state.iterations) == 1
        # Only one iteration: regression guard (i > 1) never fires
        feedback = state.iterations[0].feedback
        assert feedback is None or not feedback.priority_summary.startswith("REGRESSION:")


# ---------------------------------------------------------------------------
# Policy convergence
# ---------------------------------------------------------------------------

class TestPolicyConvergence:
    @pytest.mark.asyncio
    async def test_checker_pass_policy_fail_does_not_converge(self):
        orch = _make_orchestrator(
            stage=PipelineStage.POLICY,
            checker_outcomes=[CheckerReport(verdict=CheckerVerdict.PASS)] * 3,
            policy_outcomes=[
                PolicyVerdict(compliant=False, risk_level=RiskLevel.HIGH),
                PolicyVerdict(compliant=True, risk_level=RiskLevel.LOW),
            ],
        )
        events = await _collect(orch, _req(stage=PipelineStage.POLICY, max_iterations=3))
        assert _iteration_count(events) == 2
        assert _pipeline_status(events) == PipelineStatus.AWAITING_APPROVAL.value

    @pytest.mark.asyncio
    async def test_checker_pass_policy_pass_converges(self):
        orch = _make_orchestrator(
            stage=PipelineStage.POLICY,
            checker_outcomes=[CheckerReport(verdict=CheckerVerdict.PASS)],
            policy_outcomes=[PolicyVerdict(compliant=True, risk_level=RiskLevel.LOW)],
        )
        events = await _collect(orch, _req(stage=PipelineStage.POLICY, max_iterations=3))
        assert _iteration_count(events) == 1
        assert _pipeline_status(events) == PipelineStatus.AWAITING_APPROVAL.value

    @pytest.mark.asyncio
    async def test_policy_feedback_forwarded_to_actor(self):
        violation = PolicyViolation(rule_id="DO-178C-§6.3.1", description="missing null check")
        orch = _make_orchestrator(
            stage=PipelineStage.POLICY,
            checker_outcomes=[CheckerReport(verdict=CheckerVerdict.PASS)] * 2,
            policy_outcomes=[
                PolicyVerdict(compliant=False, risk_level=RiskLevel.HIGH, violations=[violation]),
                PolicyVerdict(compliant=True, risk_level=RiskLevel.LOW),
            ],
        )
        await _collect(orch, _req(stage=PipelineStage.POLICY, max_iterations=3))
        assert orch._actor.calls == 2
        second_kwargs = orch._actor._call_kwargs[1]
        feedback: FeedbackMessage = second_kwargs["feedback"]
        assert feedback is not None
        assert feedback.policy_feedback is not None
        assert not feedback.policy_feedback.compliant
