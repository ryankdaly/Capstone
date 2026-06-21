"""Unit tests for selective CLI runners (cli/runner.py).

Focuses on error-path behavior: verifying that LLMClient.aclose() is always
called and that AGENT_ERROR events are emitted on failure.
"""

from __future__ import annotations

from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from backend.api.schemas.pipeline import StreamEventType


# ---------------------------------------------------------------------------
# Helpers — minimal DisplayManager stub
# ---------------------------------------------------------------------------

class _FakeDisplay:
    def __init__(self):
        self.events: list = []

    def handle_event(self, event):
        self.events.append(event)

    def event_types(self):
        return [e.event_type for e in self.events]

    def has_error(self, agent: str | None = None) -> bool:
        return any(
            e.event_type == StreamEventType.AGENT_ERROR
            and (agent is None or e.agent == agent)
            for e in self.events
        )


def _make_llm_client_mock(aclose_mock: AsyncMock | None = None) -> MagicMock:
    client = MagicMock()
    client.aclose = aclose_mock or AsyncMock()
    return client


async def _streaming_raises(error: Exception):
    async def _gen(**kwargs: Any) -> AsyncGenerator:
        raise error
        yield  # make it an async generator  # noqa: unreachable

    return _gen


# ---------------------------------------------------------------------------
# run_checker_only — error paths
# ---------------------------------------------------------------------------

class TestRunCheckerOnlyErrors:
    @pytest.mark.asyncio
    async def test_checker_exception_emits_agent_error_and_closes_client(self):
        """When CheckerAgent.run_streaming raises, AGENT_ERROR is emitted and aclose called."""
        from cli.runner import _run_checker_only_async

        aclose_mock = AsyncMock()
        client = _make_llm_client_mock(aclose_mock)
        display = _FakeDisplay()

        async def _bad_streaming(**kwargs: Any) -> AsyncGenerator:
            raise RuntimeError("LLM timeout")
            yield  # noqa: unreachable

        with (
            patch("cli.runner._build_llm_client", return_value=client),
            patch("backend.services.agents.checker.CheckerAgent") as MockChecker,
        ):
            MockChecker.return_value.run_streaming = _bad_streaming
            await _run_checker_only_async(
                code="def f(): pass",
                standard="DO_178C",
                language="Python",
                display=display,
            )

        assert display.has_error("checker")
        aclose_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_checker_success_closes_client(self):
        """After a successful checker run, aclose() must still be called."""
        from cli.runner import _run_checker_only_async
        from backend.api.schemas.agents import CheckerReport, CheckerVerdict

        aclose_mock = AsyncMock()
        client = _make_llm_client_mock(aclose_mock)
        display = _FakeDisplay()

        async def _good_streaming(**kwargs: Any) -> AsyncGenerator:
            yield "tok"
            yield CheckerReport(verdict=CheckerVerdict.PASS)

        with (
            patch("cli.runner._build_llm_client", return_value=client),
            patch("backend.services.agents.checker.CheckerAgent") as MockChecker,
        ):
            MockChecker.return_value.run_streaming = _good_streaming
            await _run_checker_only_async(
                code="def f(): pass",
                standard="DO_178C",
                language="Python",
                display=display,
                run_tests=False,
            )

        assert not display.has_error("checker")
        aclose_mock.assert_awaited_once()


# ---------------------------------------------------------------------------
# run_dafny_only — error paths
# ---------------------------------------------------------------------------

class TestRunDafnyOnlyErrors:
    @pytest.mark.asyncio
    async def test_architect_exception_emits_agent_error_and_closes_client(self):
        from cli.runner import _run_dafny_only_async

        aclose_mock = AsyncMock()
        client = _make_llm_client_mock(aclose_mock)
        display = _FakeDisplay()

        async def _bad_streaming(**kwargs: Any) -> AsyncGenerator:
            raise ConnectionError("endpoint unreachable")
            yield  # noqa: unreachable

        with (
            patch("cli.runner._build_llm_client", return_value=client),
            patch("backend.services.agents.dafny_architect.DafnyArchitectAgent") as MockArch,
        ):
            MockArch.return_value.run_streaming = _bad_streaming
            await _run_dafny_only_async(
                code="def f(): pass",
                language="Python",
                display=display,
            )

        assert display.has_error("dafny_architect")
        aclose_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_dafny_success_closes_client(self):
        from cli.runner import _run_dafny_only_async
        from backend.api.schemas.agents import DafnySpec

        aclose_mock = AsyncMock()
        client = _make_llm_client_mock(aclose_mock)
        display = _FakeDisplay()

        async def _good_streaming(**kwargs: Any) -> AsyncGenerator:
            yield "tok"
            yield DafnySpec(dafny_source="method F() {}")

        with (
            patch("cli.runner._build_llm_client", return_value=client),
            patch("backend.services.agents.dafny_architect.DafnyArchitectAgent") as MockArch,
            patch("backend.services.verification.dafny_runner.DafnyRunner") as MockRunner,
        ):
            MockArch.return_value.run_streaming = _good_streaming
            MockRunner.return_value.verify = AsyncMock(
                return_value=MagicMock(verified=True, solver_output="ok",
                                      failing_assertions=[], execution_time_seconds=0.1)
            )
            await _run_dafny_only_async(
                code="def f(): pass",
                language="Python",
                display=display,
            )

        assert not display.has_error("dafny_architect")
        aclose_mock.assert_awaited_once()


# ---------------------------------------------------------------------------
# run_policy_only — error paths
# ---------------------------------------------------------------------------

class TestRunPolicyOnlyErrors:
    @pytest.mark.asyncio
    async def test_policy_exception_emits_agent_error_and_closes_client(self):
        from cli.runner import _run_policy_only_async

        aclose_mock = AsyncMock()
        client = _make_llm_client_mock(aclose_mock)
        display = _FakeDisplay()

        async def _bad_streaming(**kwargs: Any) -> AsyncGenerator:
            raise TimeoutError("policy LLM timed out")
            yield  # noqa: unreachable

        with (
            patch("cli.runner._build_llm_client", return_value=client),
            patch("backend.services.agents.policy.PolicyAgent") as MockPolicy,
            patch("cli.runner.StandardsRetriever") as MockRetriever,
        ):
            MockRetriever.return_value.retrieve = AsyncMock(return_value="ctx")
            MockPolicy.return_value.run_streaming = _bad_streaming
            await _run_policy_only_async(
                code="def f(): pass",
                standard="DO_178C",
                display=display,
            )

        assert display.has_error("policy")
        aclose_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retriever_failure_falls_back_to_empty_context(self):
        """A retriever error should not abort the policy run — falls back to empty string."""
        from cli.runner import _run_policy_only_async
        from backend.api.schemas.agents import PolicyVerdict, RiskLevel

        aclose_mock = AsyncMock()
        client = _make_llm_client_mock(aclose_mock)
        display = _FakeDisplay()

        async def _good_streaming(**kwargs: Any) -> AsyncGenerator:
            assert kwargs.get("policy_context") == ""  # empty fallback
            yield "tok"
            yield PolicyVerdict(compliant=True, risk_level=RiskLevel.LOW)

        with (
            patch("cli.runner._build_llm_client", return_value=client),
            patch("backend.services.agents.policy.PolicyAgent") as MockPolicy,
            patch("cli.runner.StandardsRetriever") as MockRetriever,
        ):
            MockRetriever.return_value.retrieve = AsyncMock(
                side_effect=RuntimeError("chroma unavailable")
            )
            MockPolicy.return_value.run_streaming = _good_streaming
            await _run_policy_only_async(
                code="def f(): pass",
                standard="DO_178C",
                display=display,
            )

        assert not display.has_error("policy")
        aclose_mock.assert_awaited_once()


# ---------------------------------------------------------------------------
# chat_stream — exception masking fix (try/finally)
# ---------------------------------------------------------------------------

class TestChatStream:
    @pytest.mark.asyncio
    async def test_aclose_called_even_when_streaming_raises(self):
        """aclose() must be called via finally, not only in except."""
        from cli.runner import chat_stream

        aclose_mock = AsyncMock()
        client = _make_llm_client_mock(aclose_mock)

        async def _bad_stream(**kwargs: Any) -> AsyncGenerator:
            raise RuntimeError("stream broke")
            yield  # noqa: unreachable

        client.generate_stream = _bad_stream

        # LLMClient is imported at module level in cli.runner
        with patch("cli.runner.LLMClient", return_value=client):
            with pytest.raises(RuntimeError, match="stream broke"):
                await chat_stream("hi", on_token=lambda t: None)

        aclose_mock.assert_awaited_once()
