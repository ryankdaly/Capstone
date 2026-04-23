"""Unit tests for LLMClient.generate_stream() and LLMClient.parse_structured().

These tests never touch a real LLM endpoint — all HTTP calls are intercepted
by mocking the AsyncOpenAI client that lives inside LLMClient.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from backend.api.schemas.agents import CodeCandidate, CheckerReport, CheckerVerdict
from backend.services.llm.client import LLMClient

from tests.unit.streaming.conftest import (
    FAKE_ENDPOINT,
    _AsyncIter,
    make_stream_chunk,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(mock_registry) -> tuple[LLMClient, MagicMock]:
    """Return (client, mock_openai_client) with HTTP replaced."""
    client = LLMClient(mock_registry)
    mock_openai = MagicMock()
    mock_openai.chat = MagicMock()
    mock_openai.chat.completions = MagicMock()
    client._clients[FAKE_ENDPOINT] = mock_openai
    return client, mock_openai


# ---------------------------------------------------------------------------
# generate_stream — happy path
# ---------------------------------------------------------------------------

class TestGenerateStreamHappyPath:
    @pytest.mark.asyncio
    async def test_yields_all_chunk_content(self, mock_registry):
        """Chunks with non-None content are forwarded in order."""
        client, openai = _make_client(mock_registry)
        chunks = [
            make_stream_chunk("Hello"),
            make_stream_chunk(", "),
            make_stream_chunk("world"),
        ]
        openai.chat.completions.create = AsyncMock(
            return_value=_AsyncIter(chunks)
        )

        received = []
        async for token in client.generate_stream(
            role="actor", system_prompt="sys", user_prompt="usr"
        ):
            received.append(token)

        assert received == ["Hello", ", ", "world"]

    @pytest.mark.asyncio
    async def test_skips_none_delta_content(self, mock_registry):
        """Chunks whose delta.content is None must be silently skipped."""
        client, openai = _make_client(mock_registry)
        chunks = [
            make_stream_chunk(None),   # role chunk — no content
            make_stream_chunk("abc"),
            make_stream_chunk(None),   # finish-reason chunk
        ]
        openai.chat.completions.create = AsyncMock(
            return_value=_AsyncIter(chunks)
        )

        received = []
        async for token in client.generate_stream(
            role="actor", system_prompt="sys", user_prompt="usr"
        ):
            received.append(token)

        assert received == ["abc"]

    @pytest.mark.asyncio
    async def test_empty_stream_yields_nothing(self, mock_registry):
        """An empty stream (no chunks at all) produces zero tokens."""
        client, openai = _make_client(mock_registry)
        openai.chat.completions.create = AsyncMock(
            return_value=_AsyncIter([])
        )

        received = []
        async for token in client.generate_stream(
            role="actor", system_prompt="sys", user_prompt="usr"
        ):
            received.append(token)

        assert received == []

    @pytest.mark.asyncio
    async def test_stream_includes_response_schema_param(self, mock_registry):
        """When response_schema is given the API call includes response_format."""
        client, openai = _make_client(mock_registry)
        openai.chat.completions.create = AsyncMock(
            return_value=_AsyncIter([make_stream_chunk('{"source_code":"x","language":"C"}')])
        )

        async for _ in client.generate_stream(
            role="actor",
            system_prompt="sys",
            user_prompt="usr",
            response_schema=CodeCandidate,
        ):
            pass

        call_kwargs = openai.chat.completions.create.call_args.kwargs
        # Constrained decoding: response_format present and stream=True
        assert call_kwargs.get("stream") is True
        assert "response_format" in call_kwargs

    @pytest.mark.asyncio
    async def test_stream_flag_always_true(self, mock_registry):
        """stream=True is always passed regardless of other parameters."""
        client, openai = _make_client(mock_registry)
        openai.chat.completions.create = AsyncMock(
            return_value=_AsyncIter([])
        )

        async for _ in client.generate_stream(
            role="actor", system_prompt="sys", user_prompt="usr"
        ):
            pass

        call_kwargs = openai.chat.completions.create.call_args.kwargs
        assert call_kwargs["stream"] is True


# ---------------------------------------------------------------------------
# generate_stream — fallback behaviour
# ---------------------------------------------------------------------------

class TestGenerateStreamFallback:
    @pytest.mark.asyncio
    async def test_fallback_on_create_exception(self, mock_registry):
        """If create() itself raises, fall back to non-streaming generate()."""
        client, openai = _make_client(mock_registry)
        openai.chat.completions.create = AsyncMock(
            side_effect=Exception("streaming not supported")
        )

        # Patch the fallback generate() so we don't need a real endpoint
        with patch.object(
            client, "generate", new=AsyncMock(return_value="fallback text")
        ):
            received = []
            async for token in client.generate_stream(
                role="actor", system_prompt="sys", user_prompt="usr"
            ):
                received.append(token)

        assert received == ["fallback text"]

    @pytest.mark.asyncio
    async def test_fallback_on_mid_stream_exception(self, mock_registry):
        """If the stream raises after yielding some chunks, the partial chunks
        that arrived before the failure are still yielded, followed by the
        complete fallback response.

        This is intentional: the display already received partial content; the
        fallback then yields the authoritative full response so the caller can
        parse the whole text.
        """

        class _BrokenIter:
            async def __aiter__(self):
                yield make_stream_chunk("partial")
                raise ConnectionError("dropped")

        client, openai = _make_client(mock_registry)
        openai.chat.completions.create = AsyncMock(
            return_value=_BrokenIter()
        )

        with patch.object(
            client, "generate", new=AsyncMock(return_value="full fallback")
        ):
            received = []
            async for token in client.generate_stream(
                role="actor", system_prompt="sys", user_prompt="usr"
            ):
                received.append(token)

        # Partial chunk arrives before the break, then fallback text follows.
        assert received == ["partial", "full fallback"]

    @pytest.mark.asyncio
    async def test_fallback_passes_same_kwargs_to_generate(self, mock_registry):
        """Fallback must forward the same role/system/user/schema kwargs."""
        client, openai = _make_client(mock_registry)
        openai.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("no stream")
        )

        generate_mock = AsyncMock(return_value="ok")
        with patch.object(client, "generate", new=generate_mock):
            async for _ in client.generate_stream(
                role="checker",
                system_prompt="check sys",
                user_prompt="check usr",
                response_schema=CheckerReport,
                temperature=0.1,
                max_tokens=512,
            ):
                pass

        generate_mock.assert_called_once()
        call_kwargs = generate_mock.call_args.kwargs
        assert call_kwargs["role"] == "checker"
        assert call_kwargs["system_prompt"] == "check sys"
        assert call_kwargs["user_prompt"] == "check usr"
        assert call_kwargs["response_schema"] is CheckerReport
        assert call_kwargs["temperature"] == 0.1
        assert call_kwargs["max_tokens"] == 512

    @pytest.mark.asyncio
    async def test_fallback_yields_exactly_one_chunk(self, mock_registry):
        """Non-streaming fallback must yield the whole text as a single token."""
        client, openai = _make_client(mock_registry)
        openai.chat.completions.create = AsyncMock(
            side_effect=Exception("unsupported")
        )

        with patch.object(
            client, "generate", new=AsyncMock(return_value="the whole answer")
        ):
            received = []
            async for token in client.generate_stream(
                role="actor", system_prompt="s", user_prompt="u"
            ):
                received.append(token)

        assert len(received) == 1
        assert received[0] == "the whole answer"

    @pytest.mark.asyncio
    async def test_no_constrained_decoding_flag_skips_response_format(
        self, mock_registry, resolved_model
    ):
        """If endpoint is in _no_constrained_decoding, response_format is omitted."""
        client, openai = _make_client(mock_registry)
        client._no_constrained_decoding.add(FAKE_ENDPOINT)
        openai.chat.completions.create = AsyncMock(
            return_value=_AsyncIter([make_stream_chunk("x")])
        )

        async for _ in client.generate_stream(
            role="actor",
            system_prompt="sys",
            user_prompt="usr",
            response_schema=CodeCandidate,
        ):
            pass

        call_kwargs = openai.chat.completions.create.call_args.kwargs
        assert "response_format" not in call_kwargs

    @pytest.mark.asyncio
    async def test_no_system_role_flag_merges_messages(
        self, mock_registry
    ):
        """If endpoint is in _no_system_role, system content is merged into user msg."""
        client, openai = _make_client(mock_registry)
        client._no_system_role.add(FAKE_ENDPOINT)
        openai.chat.completions.create = AsyncMock(
            return_value=_AsyncIter([make_stream_chunk("ok")])
        )

        async for _ in client.generate_stream(
            role="actor", system_prompt="SYS", user_prompt="USR"
        ):
            pass

        call_kwargs = openai.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        # Only one message when system role is disabled
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "SYS" in messages[0]["content"]
        assert "USR" in messages[0]["content"]


# ---------------------------------------------------------------------------
# parse_structured — all parse paths
# ---------------------------------------------------------------------------

class _SimpleModel(BaseModel):
    value: str
    count: int = 0


class TestParseStructured:
    def _client(self, mock_registry) -> LLMClient:
        return LLMClient(mock_registry)

    def test_direct_json_parse(self, mock_registry):
        """Clean JSON string parses directly."""
        raw = json.dumps({"value": "hello", "count": 3})
        result = self._client(mock_registry).parse_structured(raw, _SimpleModel)
        assert result.value == "hello"
        assert result.count == 3

    def test_json_in_markdown_fence(self, mock_registry):
        """JSON inside ```json ... ``` fences is extracted and parsed."""
        raw = '```json\n{"value": "fence", "count": 1}\n```'
        result = self._client(mock_registry).parse_structured(raw, _SimpleModel)
        assert result.value == "fence"

    def test_json_in_plain_fence(self, mock_registry):
        """JSON inside plain ``` ... ``` (no language tag) is extracted."""
        raw = '```\n{"value": "plain", "count": 0}\n```'
        result = self._client(mock_registry).parse_structured(raw, _SimpleModel)
        assert result.value == "plain"

    def test_json_embedded_in_prose(self, mock_registry):
        """JSON object buried in prose text is extracted."""
        raw = 'Sure, here is the result: {"value": "prose"} Hope that helps!'
        result = self._client(mock_registry).parse_structured(raw, _SimpleModel)
        assert result.value == "prose"

    def test_partial_construction_with_defaults(self, mock_registry):
        """If required fields are missing but have defaults, partial parse succeeds."""
        # _SimpleModel only requires 'value'; 'count' has a default of 0
        raw = '{"value": "partial"}'
        result = self._client(mock_registry).parse_structured(raw, _SimpleModel)
        assert result.value == "partial"
        assert result.count == 0

    def test_schema_echo_raises(self, mock_registry):
        """When the model echoes back the JSON Schema, ValueError is raised."""
        # A schema echo has 'properties' at the top level
        raw = json.dumps({
            "title": "_SimpleModel",
            "type": "object",
            "properties": {"value": {"type": "string"}},
        })
        with pytest.raises(ValueError, match="schema"):
            self._client(mock_registry).parse_structured(raw, _SimpleModel)

    def test_completely_malformed_raises(self, mock_registry):
        """If all parse paths fail, ValueError is raised."""
        raw = "This is not JSON at all, just prose with no braces."
        with pytest.raises(ValueError):
            self._client(mock_registry).parse_structured(raw, _SimpleModel)

    def test_role_in_error_message(self, mock_registry):
        """The role string appears in the ValueError for easier debugging."""
        raw = "no json here"
        with pytest.raises(ValueError, match="my_agent"):
            self._client(mock_registry).parse_structured(
                raw, _SimpleModel, role="my_agent"
            )

    def test_code_formatting_is_applied(self, mock_registry):
        r"""Literal \n sequences in source_code are converted to real newlines."""
        # CodeCandidate has a source_code field that _fix_code_formatting handles
        raw = json.dumps({
            "source_code": "def f():\\n    return 1",
            "language": "Python",
        })
        result = self._client(mock_registry).parse_structured(raw, CodeCandidate)
        assert "\n" in result.source_code
        assert "\\n" not in result.source_code
