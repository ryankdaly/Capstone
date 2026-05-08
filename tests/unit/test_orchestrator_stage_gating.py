"""Unit tests for orchestrator stage gating.

Verifies that request.stage correctly gates which agents and services execute:
  - ACTOR: only Actor runs; no checker/dafny/policy
  - CHECKER: Actor + DafnyArchitect + Checker + DafnyRunner; no policy
  - POLICY: full pipeline including policy agent and RAG retriever
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
    PolicyVerdict,
    RiskLevel,
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

def _make_streaming_agent_mock(tokens: list[str], result: Any) -> MagicMock:
    """Return an agent mock whose run_streaming() yields tokens then result."""
    calls: list[dict] = []

    async def _run_streaming(**kwargs: Any) -> AsyncGenerator:
        calls.append(kwargs)
        for tok in tokens:
            yield tok
        yield result

    mock = MagicMock()
    mock.run_streaming = _run_streaming
    mock._calls = calls
    return mock


def _make_orchestrator(
    stage: PipelineStage = PipelineStage.POLICY,
    checker_verdict: CheckerVerdict = CheckerVerdict.PASS,
    policy_compliant: bool = True,
) -> PipelineOrchestrator:
    """Build a fully mocked orchestrator with configurable verdicts."""
    orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
    orch._audit = MagicMock()
    orch._retriever = MagicMock()
    orch._retriever.retrieve.return_value = "retrieved policy context"
    orch.last_state = None

    actor_result = CodeCandidate(source_code="def f(): pass", language="Python")
    orch._actor = _make_streaming_agent_mock(["tok1"], actor_result)

    dafny_arch_result = DafnySpec(dafny_source="method F() {}")
    orch._dafny_architect = _make_streaming_agent_mock(["d1"], dafny_arch_result)

    checker_result = CheckerReport(verdict=checker_verdict)
    orch._checker = _make_streaming_agent_mock(["c1"], checker_result)

    policy_result = PolicyVerdict(compliant=policy_compliant, risk_level=RiskLevel.LOW)
    orch._policy = _make_streaming_agent_mock(["p1"], policy_result)

    dafny_runner = MagicMock()
    dafny_runner.verify = AsyncMock(return_value=VerificationResult(verified=True))
    orch._dafny = dafny_runner

    test_runner = MagicMock()
    test_runner.run = AsyncMock(return_value=TestRunResult(executed=False, total=0))
    orch._test_runner = test_runner

    orch._llm = MagicMock()
    orch._llm.aclose = AsyncMock()

    return orch


def _req(stage: PipelineStage, **kwargs: Any) -> PipelineRequest:
    return PipelineRequest(
        requirement_text="test requirement",
        stage=stage,
        run_tests=False,
        **kwargs,
    )


async def _collect_events(orch: PipelineOrchestrator, request: PipelineRequest) -> list:
    events = []
    async for event in orch.run(request):
        events.append(event)
    return events


def _events_of_type(events: list, event_type: StreamEventType) -> list:
    return [e for e in events if e.event_type == event_type]


def _output_events_for(events: list, agent: str) -> list:
    return [
        e for e in events
        if e.event_type == StreamEventType.AGENT_OUTPUT and e.agent == agent
    ]


# ---------------------------------------------------------------------------
# ACTOR stage
# ---------------------------------------------------------------------------

class TestStageActorOnly:
    @pytest.mark.asyncio
    async def test_actor_run_streaming_called(self):
        orch = _make_orchestrator(stage=PipelineStage.ACTOR)
        await _collect_events(orch, _req(PipelineStage.ACTOR))
        assert len(orch._actor._calls) == 1

    @pytest.mark.asyncio
    async def test_checker_not_called(self):
        orch = _make_orchestrator(stage=PipelineStage.ACTOR)
        await _collect_events(orch, _req(PipelineStage.ACTOR))
        assert len(orch._checker._calls) == 0

    @pytest.mark.asyncio
    async def test_dafny_architect_not_called(self):
        orch = _make_orchestrator(stage=PipelineStage.ACTOR)
        await _collect_events(orch, _req(PipelineStage.ACTOR))
        assert len(orch._dafny_architect._calls) == 0

    @pytest.mark.asyncio
    async def test_policy_not_called(self):
        orch = _make_orchestrator(stage=PipelineStage.ACTOR)
        await _collect_events(orch, _req(PipelineStage.ACTOR))
        assert len(orch._policy._calls) == 0

    @pytest.mark.asyncio
    async def test_dafny_verify_not_called(self):
        orch = _make_orchestrator(stage=PipelineStage.ACTOR)
        await _collect_events(orch, _req(PipelineStage.ACTOR))
        orch._dafny.verify.assert_not_called()

    @pytest.mark.asyncio
    async def test_max_iterations_forced_to_one(self):
        orch = _make_orchestrator(stage=PipelineStage.ACTOR)
        events = await _collect_events(orch, _req(PipelineStage.ACTOR, max_iterations=5))
        iteration_events = _events_of_type(events, StreamEventType.ITERATION_COMPLETE)
        assert len(iteration_events) == 1

    @pytest.mark.asyncio
    async def test_pipeline_complete_status_awaiting_approval(self):
        orch = _make_orchestrator(stage=PipelineStage.ACTOR)
        events = await _collect_events(orch, _req(PipelineStage.ACTOR))
        complete = _events_of_type(events, StreamEventType.PIPELINE_COMPLETE)
        assert len(complete) == 1
        assert complete[0].data["status"] == PipelineStatus.AWAITING_APPROVAL.value

    @pytest.mark.asyncio
    async def test_no_checker_output_events(self):
        orch = _make_orchestrator(stage=PipelineStage.ACTOR)
        events = await _collect_events(orch, _req(PipelineStage.ACTOR))
        assert _output_events_for(events, "checker") == []

    @pytest.mark.asyncio
    async def test_no_policy_output_events(self):
        orch = _make_orchestrator(stage=PipelineStage.ACTOR)
        events = await _collect_events(orch, _req(PipelineStage.ACTOR))
        assert _output_events_for(events, "policy") == []


# ---------------------------------------------------------------------------
# CHECKER stage
# ---------------------------------------------------------------------------

class TestStageCheckerOnly:
    @pytest.mark.asyncio
    async def test_actor_and_checker_both_run(self):
        orch = _make_orchestrator(stage=PipelineStage.CHECKER)
        await _collect_events(orch, _req(PipelineStage.CHECKER))
        assert len(orch._actor._calls) == 1
        assert len(orch._checker._calls) == 1

    @pytest.mark.asyncio
    async def test_dafny_architect_runs(self):
        orch = _make_orchestrator(stage=PipelineStage.CHECKER)
        await _collect_events(orch, _req(PipelineStage.CHECKER))
        assert len(orch._dafny_architect._calls) == 1

    @pytest.mark.asyncio
    async def test_dafny_verify_called(self):
        orch = _make_orchestrator(stage=PipelineStage.CHECKER)
        await _collect_events(orch, _req(PipelineStage.CHECKER))
        orch._dafny.verify.assert_called_once()

    @pytest.mark.asyncio
    async def test_policy_not_called(self):
        orch = _make_orchestrator(stage=PipelineStage.CHECKER)
        await _collect_events(orch, _req(PipelineStage.CHECKER))
        assert len(orch._policy._calls) == 0

    @pytest.mark.asyncio
    async def test_retriever_not_called(self):
        orch = _make_orchestrator(stage=PipelineStage.CHECKER)
        await _collect_events(orch, _req(PipelineStage.CHECKER))
        orch._retriever.retrieve.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_policy_output_event(self):
        orch = _make_orchestrator(stage=PipelineStage.CHECKER)
        events = await _collect_events(orch, _req(PipelineStage.CHECKER))
        assert _output_events_for(events, "policy") == []

    @pytest.mark.asyncio
    async def test_checker_output_event_present(self):
        orch = _make_orchestrator(stage=PipelineStage.CHECKER)
        events = await _collect_events(orch, _req(PipelineStage.CHECKER))
        assert len(_output_events_for(events, "checker")) == 1

    @pytest.mark.asyncio
    async def test_dafny_verifier_output_event_present(self):
        orch = _make_orchestrator(stage=PipelineStage.CHECKER)
        events = await _collect_events(orch, _req(PipelineStage.CHECKER))
        assert len(_output_events_for(events, "dafny_verifier")) == 1


# ---------------------------------------------------------------------------
# POLICY stage (full pipeline)
# ---------------------------------------------------------------------------

class TestStagePolicyFull:
    @pytest.mark.asyncio
    async def test_all_four_agents_run(self):
        orch = _make_orchestrator(stage=PipelineStage.POLICY)
        await _collect_events(orch, _req(PipelineStage.POLICY))
        assert len(orch._actor._calls) == 1
        assert len(orch._dafny_architect._calls) == 1
        assert len(orch._checker._calls) == 1
        assert len(orch._policy._calls) == 1

    @pytest.mark.asyncio
    async def test_retriever_called(self):
        orch = _make_orchestrator(stage=PipelineStage.POLICY)
        await _collect_events(orch, _req(PipelineStage.POLICY))
        orch._retriever.retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_policy_receives_checker_report(self):
        orch = _make_orchestrator(stage=PipelineStage.POLICY)
        await _collect_events(orch, _req(PipelineStage.POLICY))
        assert len(orch._policy._calls) == 1
        kwargs = orch._policy._calls[0]
        assert "checker_report" in kwargs
        assert kwargs["checker_report"] is not None

    @pytest.mark.asyncio
    async def test_all_output_events_present(self):
        orch = _make_orchestrator(stage=PipelineStage.POLICY)
        events = await _collect_events(orch, _req(PipelineStage.POLICY))
        for agent in ("actor", "dafny_architect", "checker", "dafny_verifier", "policy"):
            assert len(_output_events_for(events, agent)) >= 1, f"missing AGENT_OUTPUT for {agent}"


# ---------------------------------------------------------------------------
# Parametrized sweep
# ---------------------------------------------------------------------------

class TestStageGatingMatrix:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("stage,expect_checker,expect_policy", [
        (PipelineStage.ACTOR,   False, False),
        (PipelineStage.CHECKER, True,  False),
        (PipelineStage.POLICY,  True,  True),
    ])
    async def test_stage_gating_matrix(self, stage, expect_checker, expect_policy):
        orch = _make_orchestrator(stage=stage)
        await _collect_events(orch, _req(stage))
        assert (len(orch._checker._calls) > 0) == expect_checker, (
            f"checker called={len(orch._checker._calls)>0}, expected={expect_checker} for stage={stage}"
        )
        assert (len(orch._policy._calls) > 0) == expect_policy, (
            f"policy called={len(orch._policy._calls)>0}, expected={expect_policy} for stage={stage}"
        )
