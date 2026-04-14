"""Unit tests for streaming pipeline event flow.

Verifies:
  - AGENT_TOKEN exists in StreamEventType and serializes correctly
  - The orchestrator emits AGENT_TOKEN events for every LLM agent
  - AGENT_OUTPUT fires after (not before) all tokens for each agent
  - Dafny verifier output event fires even when checker streams
  - Token→Output ordering is preserved per agent
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from backend.api.schemas.agents import (
    CheckerReport,
    CheckerVerdict,
    CodeCandidate,
    DafnySpec,
    PolicyVerdict,
    RiskLevel,
    VerificationResult,
)
from backend.api.schemas.pipeline import (
    PipelineRequest,
    PipelineStage,
    StreamEvent,
    StreamEventType,
)
from backend.services.orchestrator import PipelineOrchestrator


# ---------------------------------------------------------------------------
# Schema: AGENT_TOKEN exists and round-trips
# ---------------------------------------------------------------------------

class TestAgentTokenSchema:
    def test_agent_token_in_enum(self):
        assert hasattr(StreamEventType, "AGENT_TOKEN")
        assert StreamEventType.AGENT_TOKEN == "agent_token"

    def test_stream_event_with_agent_token_serializes(self):
        event = StreamEvent(
            event_type=StreamEventType.AGENT_TOKEN,
            run_id=uuid4(),
            agent="actor",
            data={"token": "hello"},
        )
        raw = event.model_dump_json()
        restored = StreamEvent.model_validate_json(raw)

        assert restored.event_type == StreamEventType.AGENT_TOKEN
        assert restored.agent == "actor"
        assert restored.data["token"] == "hello"

    def test_stream_event_agent_token_round_trips_unicode(self):
        """Tokens may contain unicode (Chinese, emoji, etc.)."""
        event = StreamEvent(
            event_type=StreamEventType.AGENT_TOKEN,
            run_id=uuid4(),
            agent="actor",
            data={"token": "中文 🚀 αβγ"},
        )
        restored = StreamEvent.model_validate_json(event.model_dump_json())
        assert restored.data["token"] == "中文 🚀 αβγ"

    def test_agent_token_is_distinct_from_agent_output(self):
        assert StreamEventType.AGENT_TOKEN != StreamEventType.AGENT_OUTPUT

    def test_all_expected_event_types_present(self):
        """Regression guard: confirm all required event types coexist."""
        required = {
            "AGENT_START", "AGENT_TOKEN", "AGENT_OUTPUT",
            "AGENT_ERROR", "TEST_RUN", "ITERATION_COMPLETE",
            "PIPELINE_COMPLETE",
        }
        present = {e.name for e in StreamEventType}
        assert required <= present


# ---------------------------------------------------------------------------
# Orchestrator helpers
# ---------------------------------------------------------------------------

def _make_streaming_agent_mock(tokens: list[str], result: Any):
    """Return an agent mock whose run_streaming() yields tokens then result."""

    async def _run_streaming(**kwargs: Any) -> AsyncGenerator:
        for tok in tokens:
            yield tok
        yield result

    mock = MagicMock()
    mock.run_streaming = _run_streaming
    return mock


def _make_orchestrator_with_mocks(
    actor_tokens=("a1", "a2"),
    dafny_arch_tokens=("d1",),
    checker_tokens=("c1", "c2"),
    policy_tokens=("p1",),
) -> PipelineOrchestrator:
    """Build an orchestrator with all LLM agents mocked."""

    orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
    orch._audit = MagicMock()
    orch._retriever = MagicMock()
    orch._retriever.retrieve.return_value = "policy context"
    orch.last_state = None

    # Actor
    actor_result = CodeCandidate(source_code="def f(): pass", language="Python")
    orch._actor = _make_streaming_agent_mock(list(actor_tokens), actor_result)

    # DafnyArchitect
    dafny_arch_result = DafnySpec(dafny_source="method F() {}")
    orch._dafny_architect = _make_streaming_agent_mock(
        list(dafny_arch_tokens), dafny_arch_result
    )

    # Checker
    checker_result = CheckerReport(verdict=CheckerVerdict.PASS)
    orch._checker = _make_streaming_agent_mock(list(checker_tokens), checker_result)

    # Policy
    policy_result = PolicyVerdict(compliant=True, risk_level=RiskLevel.LOW)
    orch._policy = _make_streaming_agent_mock(list(policy_tokens), policy_result)

    # Dafny runner (subprocess, not streaming)
    dafny_runner = MagicMock()
    dafny_runner.verify = AsyncMock(
        return_value=VerificationResult(verified=True)
    )
    orch._dafny = dafny_runner

    # Test runner
    from backend.api.schemas.agents import TestRunResult
    test_runner = MagicMock()
    test_runner.run = AsyncMock(
        return_value=TestRunResult(executed=False, total=0)
    )
    orch._test_runner = test_runner

    # LLM client (only used for aclose)
    orch._llm = MagicMock()
    orch._llm.aclose = AsyncMock()

    return orch


async def _collect_events(orch: PipelineOrchestrator, request: PipelineRequest):
    """Drain all events from one orchestrator run."""
    events: list[StreamEvent] = []
    async for event in orch.run(request):
        events.append(event)
    return events


def _events_of_type(events, event_type: StreamEventType) -> list[StreamEvent]:
    return [e for e in events if e.event_type == event_type]


def _token_events_for(events, agent: str) -> list[StreamEvent]:
    return [
        e for e in events
        if e.event_type == StreamEventType.AGENT_TOKEN and e.agent == agent
    ]


def _output_index(events, agent: str) -> int:
    for i, e in enumerate(events):
        if e.event_type == StreamEventType.AGENT_OUTPUT and e.agent == agent:
            return i
    return -1


def _token_indices(events, agent: str) -> list[int]:
    return [
        i for i, e in enumerate(events)
        if e.event_type == StreamEventType.AGENT_TOKEN and e.agent == agent
    ]


# ---------------------------------------------------------------------------
# Orchestrator emits AGENT_TOKEN events
# ---------------------------------------------------------------------------

class TestOrchestratorTokenEvents:
    def _request(self, stage=PipelineStage.POLICY) -> PipelineRequest:
        return PipelineRequest(
            requirement_text="test requirement",
            safety_standard="DO_178C",
            target_language="Python",
            max_iterations=1,
            stage=stage,
        )

    @pytest.mark.asyncio
    async def test_actor_tokens_emitted(self):
        orch = _make_orchestrator_with_mocks(actor_tokens=["tok_a", "tok_b"])
        events = await _collect_events(orch, self._request())

        actor_tokens = _token_events_for(events, "actor")
        assert len(actor_tokens) == 2
        assert actor_tokens[0].data["token"] == "tok_a"
        assert actor_tokens[1].data["token"] == "tok_b"

    @pytest.mark.asyncio
    async def test_dafny_architect_tokens_emitted(self):
        orch = _make_orchestrator_with_mocks(dafny_arch_tokens=["da1", "da2", "da3"])
        events = await _collect_events(orch, self._request())

        da_tokens = _token_events_for(events, "dafny_architect")
        assert len(da_tokens) == 3
        texts = [e.data["token"] for e in da_tokens]
        assert texts == ["da1", "da2", "da3"]

    @pytest.mark.asyncio
    async def test_checker_tokens_emitted(self):
        orch = _make_orchestrator_with_mocks(checker_tokens=["ck1", "ck2"])
        events = await _collect_events(orch, self._request())

        ck_tokens = _token_events_for(events, "checker")
        assert len(ck_tokens) == 2

    @pytest.mark.asyncio
    async def test_policy_tokens_emitted(self):
        orch = _make_orchestrator_with_mocks(policy_tokens=["pol"])
        events = await _collect_events(orch, self._request())

        pol_tokens = _token_events_for(events, "policy")
        assert len(pol_tokens) == 1
        assert pol_tokens[0].data["token"] == "pol"

    @pytest.mark.asyncio
    async def test_no_tokens_when_actor_only_stage(self):
        """Actor-only stage still emits actor tokens; no checker/policy tokens."""
        orch = _make_orchestrator_with_mocks(
            actor_tokens=["x"], checker_tokens=["c"], policy_tokens=["p"]
        )
        events = await _collect_events(orch, self._request(stage=PipelineStage.ACTOR))

        actor_toks = _token_events_for(events, "actor")
        checker_toks = _token_events_for(events, "checker")
        policy_toks = _token_events_for(events, "policy")

        assert len(actor_toks) == 1
        assert len(checker_toks) == 0
        assert len(policy_toks) == 0


# ---------------------------------------------------------------------------
# Token → Output ordering per agent
# ---------------------------------------------------------------------------

class TestTokenOutputOrdering:
    def _request(self) -> PipelineRequest:
        return PipelineRequest(
            requirement_text="req",
            safety_standard="DO_178C",
            target_language="Python",
            max_iterations=1,
            stage=PipelineStage.POLICY,
        )

    @pytest.mark.asyncio
    async def test_actor_tokens_before_actor_output(self):
        orch = _make_orchestrator_with_mocks(actor_tokens=["t1", "t2"])
        events = await _collect_events(orch, self._request())

        tok_idxs = _token_indices(events, "actor")
        out_idx = _output_index(events, "actor")

        assert tok_idxs, "no actor tokens found"
        assert out_idx >= 0, "no actor output event found"
        assert all(i < out_idx for i in tok_idxs), (
            "some actor tokens appeared AFTER actor output"
        )

    @pytest.mark.asyncio
    async def test_dafny_arch_tokens_before_output(self):
        orch = _make_orchestrator_with_mocks(dafny_arch_tokens=["da"])
        events = await _collect_events(orch, self._request())

        tok_idxs = _token_indices(events, "dafny_architect")
        out_idx = _output_index(events, "dafny_architect")

        assert tok_idxs
        assert out_idx >= 0
        assert all(i < out_idx for i in tok_idxs)

    @pytest.mark.asyncio
    async def test_checker_tokens_before_checker_output(self):
        orch = _make_orchestrator_with_mocks(checker_tokens=["ck"])
        events = await _collect_events(orch, self._request())

        tok_idxs = _token_indices(events, "checker")
        out_idx = _output_index(events, "checker")

        assert tok_idxs
        assert out_idx >= 0
        assert all(i < out_idx for i in tok_idxs)

    @pytest.mark.asyncio
    async def test_policy_tokens_before_policy_output(self):
        orch = _make_orchestrator_with_mocks(policy_tokens=["po"])
        events = await _collect_events(orch, self._request())

        tok_idxs = _token_indices(events, "policy")
        out_idx = _output_index(events, "policy")

        assert tok_idxs
        assert out_idx >= 0
        assert all(i < out_idx for i in tok_idxs)


# ---------------------------------------------------------------------------
# Dafny verifier still fires its OUTPUT event when checker streams
# ---------------------------------------------------------------------------

class TestDafnyVerifierWithStreamingChecker:
    def _request(self) -> PipelineRequest:
        return PipelineRequest(
            requirement_text="req",
            safety_standard="DO_178C",
            target_language="Python",
            max_iterations=1,
            stage=PipelineStage.POLICY,
        )

    @pytest.mark.asyncio
    async def test_dafny_verifier_output_event_fires(self):
        """dafny_verifier AGENT_OUTPUT must arrive even when checker is streaming."""
        orch = _make_orchestrator_with_mocks(checker_tokens=["c1", "c2", "c3"])
        events = await _collect_events(orch, self._request())

        dafny_outputs = _events_of_type(events, StreamEventType.AGENT_OUTPUT)
        dafny_out = [e for e in dafny_outputs if e.agent == "dafny_verifier"]
        assert len(dafny_out) == 1, (
            "dafny_verifier AGENT_OUTPUT event missing"
        )

    @pytest.mark.asyncio
    async def test_dafny_verify_called_once(self):
        """DafnyRunner.verify() must be called exactly once per iteration."""
        orch = _make_orchestrator_with_mocks()
        await _collect_events(orch, self._request())

        orch._dafny.verify.assert_called_once()

    @pytest.mark.asyncio
    async def test_checker_tokens_and_dafny_output_both_present(self):
        """checker tokens and dafny_verifier output coexist in the same event list."""
        orch = _make_orchestrator_with_mocks(checker_tokens=["tok"])
        events = await _collect_events(orch, self._request())

        checker_toks = _token_events_for(events, "checker")
        dafny_outs = [
            e for e in events
            if e.event_type == StreamEventType.AGENT_OUTPUT and e.agent == "dafny_verifier"
        ]

        assert checker_toks, "checker token events missing"
        assert dafny_outs, "dafny_verifier output event missing"


# ---------------------------------------------------------------------------
# AGENT_TOKEN data payload integrity
# ---------------------------------------------------------------------------

class TestAgentTokenPayload:
    def _request(self) -> PipelineRequest:
        return PipelineRequest(
            requirement_text="r",
            safety_standard="DO_178C",
            target_language="Python",
            max_iterations=1,
            stage=PipelineStage.ACTOR,
        )

    @pytest.mark.asyncio
    async def test_token_event_has_token_key_in_data(self):
        orch = _make_orchestrator_with_mocks(actor_tokens=["hello"])
        events = await _collect_events(orch, self._request())

        tok_events = _token_events_for(events, "actor")
        assert tok_events
        for e in tok_events:
            assert "token" in e.data, f"token key missing in data: {e.data}"

    @pytest.mark.asyncio
    async def test_token_event_has_correct_agent_field(self):
        orch = _make_orchestrator_with_mocks(actor_tokens=["x"])
        events = await _collect_events(orch, self._request())

        tok_events = _token_events_for(events, "actor")
        for e in tok_events:
            assert e.agent == "actor"

    @pytest.mark.asyncio
    async def test_token_text_preserved_exactly(self):
        """Token strings must be forwarded verbatim — no truncation or encoding."""
        special_tokens = ["  indent", "\nnewline", "tab\there", "unicode→αβ"]
        orch = _make_orchestrator_with_mocks(actor_tokens=special_tokens)
        events = await _collect_events(orch, self._request())

        received = [e.data["token"] for e in _token_events_for(events, "actor")]
        assert received == special_tokens
