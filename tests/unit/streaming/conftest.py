"""Shared fixtures for streaming unit tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from backend.services.llm.model_registry import ResolvedModel


# ---------------------------------------------------------------------------
# Helpers for faking OpenAI streaming responses
# ---------------------------------------------------------------------------

@dataclass
class _FakeDelta:
    content: str | None


@dataclass
class _FakeChoice:
    delta: _FakeDelta


@dataclass
class _FakeChunk:
    choices: list[_FakeChoice]


def make_stream_chunk(content: str | None) -> _FakeChunk:
    """Build a minimal fake SSE chunk with the given delta content."""
    return _FakeChunk(choices=[_FakeChoice(delta=_FakeDelta(content=content))])


class _AsyncIter:
    """Async iterable over a fixed list of chunks."""

    def __init__(self, chunks: list[_FakeChunk]) -> None:
        self._chunks = chunks
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


# ---------------------------------------------------------------------------
# Resolved model fixture (no real API key needed)
# ---------------------------------------------------------------------------

FAKE_ENDPOINT = "https://fake.endpoint/v1"
FAKE_MODEL = "fake-model"


@pytest.fixture
def resolved_model() -> ResolvedModel:
    return ResolvedModel(
        endpoint=FAKE_ENDPOINT,
        model=FAKE_MODEL,
        api_key="sk-fake",
        extra_body={},
    )


@pytest.fixture
def mock_registry(resolved_model):
    """ModelRegistry that always returns the fake resolved model."""
    reg = MagicMock()
    reg.get.return_value = resolved_model
    return reg
