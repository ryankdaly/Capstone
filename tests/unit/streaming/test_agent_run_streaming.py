"""Unit tests for BaseAgent.run_streaming().

Verifies that the streaming variant of each agent:
  - yields str tokens while the LLM is generating
  - yields the final parsed model as the very last item
  - strips <think> blocks before parsing
  - degrades gracefully when generate_stream() yields a single full-text chunk
    (non-streaming endpoint fallback)
  - propagates LLM / parse errors without swallowing them
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from backend.api.schemas.agents import (
    CheckerReport,
    CheckerVerdict,
    CodeCandidate,
    DafnySpec,
    PolicyVerdict,
    RiskLevel,
)
from backend.services.agents.actor import ActorAgent
from backend.services.agents.checker import CheckerAgent
from backend.services.agents.dafny_architect import DafnyArchitectAgent
from backend.services.agents.policy import PolicyAgent
from backend.services.llm.client import LLMClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _collect(agent_gen: AsyncGenerator) -> tuple[list[str], Any]:
    """Drain an agent streaming generator.

    Returns (tokens, final_model) — tokens are all str items yielded before
    the terminal BaseModel item.
    """
    tokens: list[str] = []
    result = None
    async for item in agent_gen:
        if isinstance(item, str):
            tokens.append(item)
        else:
            result = item
    return tokens, result


def _mock_llm(chunks: list[str]) -> LLMClient:
    """Return a LLMClient whose generate_stream() yields the given string chunks."""

    async def _fake_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
        for chunk in chunks:
            yield chunk

    llm = MagicMock(spec=LLMClient)
    llm.generate_stream = _fake_stream
    llm.parse_structured = LLMClient.parse_structured.__get__(  # bind unbound method
        MagicMock(spec=LLMClient), LLMClient
    )
    return llm


def _make_actor(chunks: list[str]) -> ActorAgent:
    agent = ActorAgent.__new__(ActorAgent)
    agent._llm = _mock_llm(chunks)
    agent._system_prompt = "system"
    return agent


def _make_checker(chunks: list[str]) -> CheckerAgent:
    agent = CheckerAgent.__new__(CheckerAgent)
    agent._llm = _mock_llm(chunks)
    agent._system_prompt = "system"
    return agent


# ---------------------------------------------------------------------------
# Token yield ordering
# ---------------------------------------------------------------------------

class TestRunStreamingTokenOrdering:
    @pytest.mark.asyncio
    async def test_tokens_yielded_before_model(self):
        """All str tokens appear before the final BaseModel item."""
        payload = json.dumps({"source_code": "def f(): pass", "language": "Python"})
        chunks = ["def ", "f():", " pass"]
        full = "".join(chunks)  # not valid JSON — we patch parse_structured

        agent = _make_actor(chunks)

        # Patch parse_structured to return a valid CodeCandidate
        expected_model = CodeCandidate(source_code="def f(): pass", language="Python")
        agent._llm.parse_structured = MagicMock(return_value=expected_model)

        tokens, result = await _collect(agent.run_streaming(
            requirement="req", language="Python", standard="DO_178C"
        ))

        assert tokens == ["def ", "f():", " pass"]
        assert isinstance(result, CodeCandidate)

    @pytest.mark.asyncio
    async def test_final_item_is_model_not_str(self):
        """The last item from run_streaming() must be a BaseModel, not a str."""
        payload = json.dumps({"source_code": "x = 1", "language": "Python"})
        agent = _make_actor([payload])

        items = []
        async for item in agent.run_streaming(
            requirement="req", language="Python", standard="DO_178C"
        ):
            items.append(item)

        assert items, "generator yielded nothing"
        assert not isinstance(items[-1], str), "last item must be a model"
        assert isinstance(items[-1], BaseModel)

    @pytest.mark.asyncio
    async def test_single_chunk_fallback_still_yields_model(self):
        """When generate_stream() yields one big chunk (non-streaming fallback),
        run_streaming() still emits a valid final model."""
        payload = json.dumps({"source_code": "return 0", "language": "C"})
        agent = _make_actor([payload])  # one chunk = fallback behaviour

        tokens, result = await _collect(agent.run_streaming(
            requirement="r", language="C", standard="DO_178C"
        ))

        assert len(tokens) == 1  # single fallback chunk
        assert isinstance(result, CodeCandidate)

    @pytest.mark.asyncio
    async def test_many_chunks_all_forwarded(self):
        """Every chunk from generate_stream() is forwarded as a str token."""
        n = 20
        chunks = [f"tok{i}" for i in range(n)]
        full_json = json.dumps({"source_code": "pass", "language": "Python"})

        agent = _make_actor(chunks)
        agent._llm.parse_structured = MagicMock(
            return_value=CodeCandidate(source_code="pass", language="Python")
        )

        tokens, _ = await _collect(agent.run_streaming(
            requirement="r", language="Python", standard="DO_178C"
        ))

        assert tokens == chunks
        assert len(tokens) == n


# ---------------------------------------------------------------------------
# Think-tag stripping
# ---------------------------------------------------------------------------

class TestThinkTagStripping:
    @pytest.mark.asyncio
    async def test_think_tags_stripped_before_parsing(self):
        """<think>…</think> blocks are removed before parse_structured() is called."""
        think_block = "<think>I should think about this carefully.</think>"
        json_body = json.dumps({"source_code": "def f(): pass", "language": "Python"})
        # model outputs think block then JSON
        full_output = think_block + json_body

        # Split into chunks that each straddle the think boundary
        chunks = [full_output[:20], full_output[20:]]

        agent = _make_actor(chunks)
        parse_calls: list[str] = []

        real_parse = LLMClient.parse_structured

        def _capturing_parse(self_unused, raw: str, model, role="?"):
            parse_calls.append(raw)
            return real_parse(MagicMock(spec=LLMClient), raw, model, role)

        agent._llm.parse_structured = _capturing_parse.__get__(
            MagicMock(spec=LLMClient), type(MagicMock(spec=LLMClient))
        )

        # Simpler: just mock parse and verify what it receives
        expected = CodeCandidate(source_code="def f(): pass", language="Python")
        captured_raws: list[str] = []

        def _capture(raw: str, model: Any, role: str = "?") -> BaseModel:
            captured_raws.append(raw)
            return expected

        agent._llm.parse_structured = _capture

        await _collect(agent.run_streaming(
            requirement="r", language="Python", standard="DO_178C"
        ))

        assert captured_raws, "parse_structured was never called"
        raw_seen = captured_raws[0]
        assert "<think>" not in raw_seen
        assert "</think>" not in raw_seen
        assert "I should think" not in raw_seen

    @pytest.mark.asyncio
    async def test_output_after_think_tag_is_preserved(self):
        """Content outside <think> blocks passes through to the parser unchanged."""
        think = "<think>reasoning</think>"
        code_json = json.dumps({"source_code": "z = 99", "language": "Python"})
        full = think + code_json

        agent = _make_actor([full])

        received_raws: list[str] = []

        def _capture(raw: str, model: Any, role: str = "?") -> BaseModel:
            received_raws.append(raw)
            return CodeCandidate(source_code="z = 99", language="Python")

        agent._llm.parse_structured = _capture

        await _collect(agent.run_streaming(
            requirement="r", language="Python", standard="DO_178C"
        ))

        assert received_raws
        assert "z = 99" in received_raws[0] or code_json in received_raws[0]


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------

class TestRunStreamingErrors:
    @pytest.mark.asyncio
    async def test_parse_error_propagates(self):
        """If parse_structured() raises, run_streaming() propagates the error."""
        agent = _make_actor(["not json at all"])
        agent._llm.parse_structured = MagicMock(
            side_effect=ValueError("parse failed")
        )

        with pytest.raises(ValueError, match="parse failed"):
            await _collect(agent.run_streaming(
                requirement="r", language="Python", standard="DO_178C"
            ))

    @pytest.mark.asyncio
    async def test_stream_error_propagates(self):
        """If generate_stream() itself raises mid-iteration, error propagates."""

        async def _broken_stream(*a: Any, **kw: Any) -> AsyncGenerator[str, None]:
            yield "first"
            raise RuntimeError("network error")

        agent = ActorAgent.__new__(ActorAgent)
        agent._llm = MagicMock(spec=LLMClient)
        agent._llm.generate_stream = _broken_stream
        agent._system_prompt = "sys"

        with pytest.raises(RuntimeError, match="network error"):
            await _collect(agent.run_streaming(
                requirement="r", language="Python", standard="DO_178C"
            ))


# ---------------------------------------------------------------------------
# All four LLM agents expose run_streaming()
# ---------------------------------------------------------------------------

class TestAllAgentsHaveStreaming:
    """Smoke-check: every LLM agent class inherits run_streaming()."""

    @pytest.mark.parametrize("agent_cls", [
        ActorAgent, CheckerAgent, DafnyArchitectAgent, PolicyAgent,
    ])
    def test_run_streaming_exists(self, agent_cls):
        assert hasattr(agent_cls, "run_streaming"), (
            f"{agent_cls.__name__} is missing run_streaming()"
        )

    @pytest.mark.parametrize("agent_cls", [
        ActorAgent, CheckerAgent, DafnyArchitectAgent, PolicyAgent,
    ])
    def test_run_streaming_is_async_generator(self, agent_cls):
        import inspect
        assert inspect.isasyncgenfunction(agent_cls.run_streaming), (
            f"{agent_cls.__name__}.run_streaming must be an async generator function"
        )
