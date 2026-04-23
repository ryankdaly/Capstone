"""Unit tests for LLM client retry/timeout behavior."""

from types import SimpleNamespace

import pytest

import backend.services.llm.client as llm_client_module
from backend.services.llm.client import LLMClient
from backend.services.llm.model_registry import ResolvedModel


class _DummyResponse:
    def __init__(self, content: str = "ok") -> None:
        self.choices = [
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ]

class _FakeCompletions:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

class _FakeSDKClient:
    def __init__(self, outcomes):
        self.chat = SimpleNamespace(
            completions=_FakeCompletions(outcomes),
        )

class FakeInternalServerError(Exception):
    pass
class FakeRateLimitError(Exception):
    pass
class FakeBadRequestError(Exception):
    pass
class FakeAPITimeoutError(Exception):
    pass
class FakeAPIConnectionError(Exception):
    pass

class TestLLMClientRetries:
    @pytest.mark.asyncio
    async def test_api_call_retries_timeout_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(llm_client_module, "APITimeoutError", FakeAPITimeoutError)
        monkeypatch.setattr(llm_client_module, "APIConnectionError", FakeAPIConnectionError)
        monkeypatch.setattr(llm_client_module, "InternalServerError", FakeInternalServerError)
        monkeypatch.setattr(llm_client_module, "RateLimitError", FakeRateLimitError)
        monkeypatch.setattr(llm_client_module, "BadRequestError", FakeBadRequestError)

        client = LLMClient()
        sdk_client = _FakeSDKClient(
            [FakeAPITimeoutError("temporary timeout"), _DummyResponse("success")]
        )

        sleep_calls = []

        async def fake_sleep(delay):
            sleep_calls.append(delay)

        monkeypatch.setattr(llm_client_module.asyncio, "sleep", fake_sleep)

        response = await client._api_call(sdk_client, {"model": "dummy"})

        assert response.choices[0].message.content == "success"
        assert sdk_client.chat.completions.calls == 2
        assert sleep_calls == [5]

    @pytest.mark.asyncio
    async def test_api_call_retries_connection_error_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(llm_client_module, "APITimeoutError", FakeAPITimeoutError)
        monkeypatch.setattr(llm_client_module, "APIConnectionError", FakeAPIConnectionError)
        monkeypatch.setattr(llm_client_module, "InternalServerError", FakeInternalServerError)
        monkeypatch.setattr(llm_client_module, "RateLimitError", FakeRateLimitError)
        monkeypatch.setattr(llm_client_module, "BadRequestError", FakeBadRequestError)

        client = LLMClient()
        sdk_client = _FakeSDKClient(
            [FakeAPIConnectionError("temporary connection issue"), _DummyResponse("success")]
        )

        sleep_calls = []

        async def fake_sleep(delay):
            sleep_calls.append(delay)

        monkeypatch.setattr(llm_client_module.asyncio, "sleep", fake_sleep)

        response = await client._api_call(sdk_client, {"model": "dummy"})

        assert response.choices[0].message.content == "success"
        assert sdk_client.chat.completions.calls == 2
        assert sleep_calls == [5]

    @pytest.mark.asyncio
    async def test_api_call_raises_after_max_retries(self, monkeypatch):
        monkeypatch.setattr(llm_client_module, "APITimeoutError", FakeAPITimeoutError)
        monkeypatch.setattr(llm_client_module, "APIConnectionError", FakeAPIConnectionError)
        monkeypatch.setattr(llm_client_module, "InternalServerError", FakeInternalServerError)
        monkeypatch.setattr(llm_client_module, "RateLimitError", FakeRateLimitError)
        monkeypatch.setattr(llm_client_module, "BadRequestError", FakeBadRequestError)

        client = LLMClient()
        sdk_client = _FakeSDKClient(
            [
                FakeAPITimeoutError("still timing out"),
                FakeAPITimeoutError("still timing out"),
                FakeAPITimeoutError("still timing out"),
            ]
        )

        sleep_calls = []

        async def fake_sleep(delay):
            sleep_calls.append(delay)

        monkeypatch.setattr(llm_client_module.asyncio, "sleep", fake_sleep)

        with pytest.raises(FakeAPITimeoutError, match="still timing out"):
            await client._api_call(sdk_client, {"model": "dummy"})

        assert sdk_client.chat.completions.calls == 3
        assert sleep_calls == [5, 15]

    @pytest.mark.asyncio
    async def test_api_call_retries_transient_bad_request_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(llm_client_module, "APITimeoutError", FakeAPITimeoutError)
        monkeypatch.setattr(llm_client_module, "APIConnectionError", FakeAPIConnectionError)
        monkeypatch.setattr(llm_client_module, "InternalServerError", FakeInternalServerError)
        monkeypatch.setattr(llm_client_module, "RateLimitError", FakeRateLimitError)
        monkeypatch.setattr(llm_client_module, "BadRequestError", FakeBadRequestError)

        client = LLMClient()
        sdk_client = _FakeSDKClient(
            [
                FakeBadRequestError("DEGRADED function cannot be invoked"),
                _DummyResponse("success"),
            ]
        )

        sleep_calls = []

        async def fake_sleep(delay):
            sleep_calls.append(delay)

        monkeypatch.setattr(llm_client_module.asyncio, "sleep", fake_sleep)

        response = await client._api_call(sdk_client, {"model": "dummy"})

        assert response.choices[0].message.content == "success"
        assert sdk_client.chat.completions.calls == 2
        assert sleep_calls == [5]

    @pytest.mark.asyncio
    async def test_api_call_does_not_retry_non_transient_bad_request(self, monkeypatch):
        monkeypatch.setattr(llm_client_module, "APITimeoutError", FakeAPITimeoutError)
        monkeypatch.setattr(llm_client_module, "APIConnectionError", FakeAPIConnectionError)
        monkeypatch.setattr(llm_client_module, "InternalServerError", FakeInternalServerError)
        monkeypatch.setattr(llm_client_module, "RateLimitError", FakeRateLimitError)
        monkeypatch.setattr(llm_client_module, "BadRequestError", FakeBadRequestError)

        client = LLMClient()
        sdk_client = _FakeSDKClient([FakeBadRequestError("bad request")])

        sleep_calls = []

        async def fake_sleep(delay):
            sleep_calls.append(delay)

        monkeypatch.setattr(llm_client_module.asyncio, "sleep", fake_sleep)

        with pytest.raises(FakeBadRequestError, match="bad request"):
            await client._api_call(sdk_client, {"model": "dummy"})

        assert sdk_client.chat.completions.calls == 1
        assert sleep_calls == []

    @pytest.mark.asyncio
    async def test_generate_routes_request_through_api_call(self, monkeypatch):
        registry = SimpleNamespace(
            get=lambda role: ResolvedModel(
                endpoint="http://fake.test/v1",
                model="dummy-model",
                api_key="test-key",
            )
        )
        client = LLMClient(registry=registry)

        fake_sdk_client = object()
        monkeypatch.setattr(client, "_get_client", lambda resolved: fake_sdk_client)

        captured = {}

        async def fake_api_call(sdk_client, kwargs):
            captured["sdk_client"] = sdk_client
            captured["kwargs"] = kwargs
            return _DummyResponse("hello world")

        monkeypatch.setattr(client, "_api_call", fake_api_call)

        result = await client.generate(
            role="actor",
            system_prompt="system text",
            user_prompt="user text",
            temperature=0.7,
            max_tokens=123,
        )

        assert result == "hello world"
        assert captured["sdk_client"] is fake_sdk_client
        assert captured["kwargs"]["model"] == "dummy-model"
        assert captured["kwargs"]["temperature"] == 0.7
        assert captured["kwargs"]["max_tokens"] == 123
        assert captured["kwargs"]["messages"] == [
            {"role": "system", "content": "system text"},
            {"role": "user", "content": "user text"},
        ]


    @pytest.mark.asyncio
    async def test_api_call_retries_connection_reset_oserror_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(llm_client_module, "APITimeoutError", FakeAPITimeoutError)
        monkeypatch.setattr(llm_client_module, "APIConnectionError", FakeAPIConnectionError)
        monkeypatch.setattr(llm_client_module, "InternalServerError", FakeInternalServerError)
        monkeypatch.setattr(llm_client_module, "RateLimitError", FakeRateLimitError)
        monkeypatch.setattr(llm_client_module, "BadRequestError", FakeBadRequestError)

        client = LLMClient()
        sdk_client = _FakeSDKClient(
            [OSError("Connection reset by peer"), _DummyResponse("success")]
        )

        sleep_calls = []

        async def fake_sleep(delay):
            sleep_calls.append(delay)

        monkeypatch.setattr(llm_client_module.asyncio, "sleep", fake_sleep)

        response = await client._api_call(sdk_client, {"model": "dummy"})

        assert response.choices[0].message.content == "success"
        assert sdk_client.chat.completions.calls == 2
        assert sleep_calls == [1]

    @pytest.mark.asyncio
    async def test_api_call_raises_after_max_retries_on_connection_reset(self, monkeypatch):
        monkeypatch.setattr(llm_client_module, "APITimeoutError", FakeAPITimeoutError)
        monkeypatch.setattr(llm_client_module, "APIConnectionError", FakeAPIConnectionError)
        monkeypatch.setattr(llm_client_module, "InternalServerError", FakeInternalServerError)
        monkeypatch.setattr(llm_client_module, "RateLimitError", FakeRateLimitError)
        monkeypatch.setattr(llm_client_module, "BadRequestError", FakeBadRequestError)

        client = LLMClient()
        sdk_client = _FakeSDKClient(
            [
                OSError("Connection reset by peer"),
                OSError("Connection reset by peer"),
                OSError("Connection reset by peer"),
            ]
        )

        sleep_calls = []

        async def fake_sleep(delay):
            sleep_calls.append(delay)

        monkeypatch.setattr(llm_client_module.asyncio, "sleep", fake_sleep)

        with pytest.raises(OSError, match="Connection reset by peer"):
            await client._api_call(sdk_client, {"model": "dummy"})

        assert sdk_client.chat.completions.calls == 3
        assert sleep_calls == [1, 2]