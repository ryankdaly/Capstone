from __future__ import annotations

from pathlib import Path

import pytest

from backend.api.schemas.agents import (
    CheckerReport,
    CheckerVerdict,
    CodeCandidate,
    DafnySpec,
    PolicyVerdict,
    RiskLevel,
    TestRunResult,
    TestCaseResult,
    VerificationResult,
)
from backend.api.schemas.generation import SafetyStandard, TargetLanguage
from backend.api.schemas.pipeline import (
    PipelineRequest,
    PipelineStage,
    PipelineStatus,
    StreamEventType,
)
from backend.services.audit.logger import AuditLogger
from backend.services.orchestrator import PipelineOrchestrator


class FakeActorAgent:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def run_streaming(
        self,
        *,
        requirement: str,
        language: str,
        standard: str,
        feedback=None,
        retry_hint=None,
    ):
        self.calls.append(
            {
                "requirement": requirement,
                "language": language,
                "standard": standard,
                "feedback": feedback,
                "retry_hint": retry_hint,
            }
        )
        yield "actor-token"
        yield CodeCandidate(
            source_code="def clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n",
            dafny_spec="",
            reasoning_trace="Generated a simple clamp implementation.",
            language="Python",
        )


class FakeDafnyArchitectAgent:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def run_streaming(
        self,
        *,
        source_code: str,
        requirement: str,
        language: str,
        verification_feedback=None,
        retry_hint=None,
    ):
        self.calls.append(
            {
                "source_code": source_code,
                "requirement": requirement,
                "language": language,
                "verification_feedback": verification_feedback,
                "retry_hint": retry_hint,
            }
        )
        yield "dafny-token"
        yield DafnySpec(
            dafny_source=(
                "method Clamp(x: int, lo: int, hi: int) returns (y: int)\n"
                "  requires lo <= hi\n"
                "  ensures lo <= y <= hi\n"
                "{\n"
                "  if x < lo { y := lo; }\n"
                "  else if x > hi { y := hi; }\n"
                "  else { y := x; }\n"
                "}\n"
            ),
            reasoning_trace="Added a bounded postcondition.",
        )


class FakeCheckerAgent:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def run_streaming(
        self,
        *,
        source_code: str,
        language: str,
        standard: str,
        retry_hint=None,
        dafny_contracts=None,
        test_fix_hint=None,
    ):
        self.calls.append(
            {
                "source_code": source_code,
                "language": language,
                "standard": standard,
                "retry_hint": retry_hint,
                "dafny_contracts": dafny_contracts,
                "test_fix_hint": test_fix_hint,
            }
        )
        yield "checker-token"
        yield CheckerReport(
            verdict=CheckerVerdict.PASS,
            issues=[],
            test_cases=[
                "def test_clamp_in_range():\n    assert clamp(5, 0, 10) == 5"
            ],
            reasoning_trace="No issues found.",
        )


class FakePolicyAgent:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def run_streaming(
        self,
        *,
        source_code: str,
        dafny_spec: str,
        standard: str,
        checker_report,
        verification_result,
        policy_context: str,
    ):
        self.calls.append(
            {
                "source_code": source_code,
                "dafny_spec": dafny_spec,
                "standard": standard,
                "checker_report": checker_report,
                "verification_result": verification_result,
                "policy_context": policy_context,
            }
        )
        yield "policy-token"
        yield PolicyVerdict(
            compliant=True,
            risk_level=RiskLevel.LOW,
            violations=[],
            recommendations=["Maintain traceability."],
            reasoning_trace="Complies with requested standard.",
        )


class FakeDafnyRunner:
    def __init__(self) -> None:
        self.specs: list[str] = []

    async def verify(self, dafny_spec: str) -> VerificationResult:
        self.specs.append(dafny_spec)
        return VerificationResult(
            verified=True,
            prover="dafny",
            solver_output="verified successfully",
            failing_assertions=[],
            execution_time_seconds=0.01,
        )


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def retrieve(self, query: str, standard: str | None = None, n_results=None) -> str:
        self.calls.append((query, standard or ""))
        return "DO-178C objective excerpt"


class FakeTestRunner:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def run(self, *, source_code: str, test_cases: list[str]) -> TestRunResult:
        self.calls.append(
            {
                "source_code": source_code,
                "test_cases": test_cases,
            }
        )
        return TestRunResult(
            executed=True,
            total=1,
            passed=1,
            failed=0,
            errors=0,
            test_results=[
                TestCaseResult(name="test_clamp_in_range", passed=True)
            ],
            pytest_output="1 passed",
            execution_time_seconds=0.02,
        )


@pytest.mark.asyncio
async def test_full_pipeline_with_mocked_components_runs_all_stages(tmp_path: Path):
    fake_dafny = FakeDafnyRunner()
    fake_retriever = FakeRetriever()
    fake_test_runner = FakeTestRunner()
    audit = AuditLogger(
        log_dir=str(tmp_path / "audit"),
        db_path=str(tmp_path / "audit.db"),
    )

    orchestrator = PipelineOrchestrator(
        llm_client=object(),
        dafny_runner=fake_dafny,
        retriever=fake_retriever,
        audit_logger=audit,
        test_runner=fake_test_runner,
    )

    orchestrator._actor = FakeActorAgent()
    orchestrator._dafny_architect = FakeDafnyArchitectAgent()
    orchestrator._checker = FakeCheckerAgent()
    orchestrator._policy = FakePolicyAgent()

    request = PipelineRequest(
        requirement_text="Write a clamp function for flight control bounds.",
        safety_standard=SafetyStandard.DO_178C,
        target_language=TargetLanguage.PYTHON,
        max_iterations=3,
        stage=PipelineStage.POLICY,
        run_tests=True,
    )

    events = [event async for event in orchestrator.run(request)]

    assert len(orchestrator._actor.calls) == 1
    assert len(orchestrator._dafny_architect.calls) == 1
    assert len(orchestrator._checker.calls) == 1
    assert len(orchestrator._policy.calls) == 1

    assert fake_dafny.specs
    assert fake_retriever.calls == [
        (
            "Write a clamp function for flight control bounds. DO_178C",
            "DO_178C",
        )
    ]
    assert len(fake_test_runner.calls) == 1

    assert [event.event_type for event in events] == [
        StreamEventType.AGENT_START,
        StreamEventType.AGENT_TOKEN,
        StreamEventType.AGENT_OUTPUT,
        StreamEventType.AGENT_START,
        StreamEventType.AGENT_TOKEN,
        StreamEventType.AGENT_OUTPUT,
        StreamEventType.AGENT_START,
        StreamEventType.AGENT_OUTPUT,
        StreamEventType.AGENT_START,
        StreamEventType.AGENT_TOKEN,
        StreamEventType.AGENT_OUTPUT,
        StreamEventType.AGENT_START,
        StreamEventType.TEST_RUN,
        StreamEventType.AGENT_START,
        StreamEventType.AGENT_TOKEN,
        StreamEventType.AGENT_OUTPUT,
        StreamEventType.ITERATION_COMPLETE,
        StreamEventType.PIPELINE_COMPLETE,
    ]

    assert [event.agent for event in events] == [
        "actor",
        "actor",
        "actor",
        "dafny_architect",
        "dafny_architect",
        "dafny_architect",
        "dafny_verifier",
        "dafny_verifier",
        "checker",
        "checker",
        "checker",
        "test_runner",
        "test_runner",
        "policy",
        "policy",
        "policy",
        None,
        None,
    ]

    final_state = orchestrator.last_state
    assert final_state is not None
    assert final_state.status == PipelineStatus.AWAITING_APPROVAL
    assert final_state.final_code == "def clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n"
    assert "method Clamp" in (final_state.final_proof or "")
    assert len(final_state.iterations) == 1

    iteration = final_state.iterations[0]
    assert iteration.code_candidate is not None
    assert iteration.checker_report is not None
    assert iteration.verification_result is not None
    assert iteration.test_result is not None
    assert iteration.policy_verdict is not None
    assert iteration.feedback is not None
    assert iteration.feedback.priority_summary == "All checks passed."

    run_log = audit.get_run_log(final_state.run_id)
    event_types = [entry.event_type for entry in run_log]
    assert "pipeline_start" in event_types
    assert "verification_result" in event_types
    assert "test_run" in event_types
    assert "pipeline_complete" in event_types