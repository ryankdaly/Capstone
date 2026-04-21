"""Unit tests for FastAPI routers.

Tests the /api/v1/generation/generate, /api/v1/pipeline/run (SSE), and
/api/v1/audit/* endpoints without a real LLM or Dafny binary.

Dependency injection via app.dependency_overrides replaces the orchestrator
and audit logger with lightweight mocks.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4, UUID

import pytest

from backend.api.schemas.agents import (
    CodeCandidate,
    PolicyVerdict,
    RiskLevel,
)
from backend.api.schemas.pipeline import (
    PipelineRequest,
    PipelineStage,
    PipelineStatus,
    StreamEvent,
    StreamEventType,
)
from backend.services.orchestrator import PipelineOrchestrator


# ---------------------------------------------------------------------------
# Shared mock factory
# ---------------------------------------------------------------------------

def _make_mock_orchestrator(status: str = PipelineStatus.AWAITING_APPROVAL.value) -> PipelineOrchestrator:
    """Return a mock orchestrator that yields a single PIPELINE_COMPLETE event."""
    from uuid import uuid4

    run_id = uuid4()

    async def _run(req: PipelineRequest):
        yield StreamEvent(
            event_type=StreamEventType.PIPELINE_COMPLETE,
            run_id=run_id,
            data={"status": status, "iterations": 1, "run_id": str(run_id), "stage": "policy"},
        )

    mock = MagicMock(spec=PipelineOrchestrator)
    mock.run = _run
    mock.last_state = None
    return mock


# ---------------------------------------------------------------------------
# /api/v1/generation/generate
# ---------------------------------------------------------------------------

class TestGenerateEndpoint:
    @pytest.fixture(autouse=True)
    def override_orchestrator(self):
        from backend.main import app
        from backend.api.dependencies import get_orchestrator
        mock = _make_mock_orchestrator()
        app.dependency_overrides[get_orchestrator] = lambda: mock
        yield
        app.dependency_overrides.clear()

    def test_post_generate_returns_200(self):
        from starlette.testclient import TestClient
        from backend.main import app
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post("/api/v1/generation/generate", json={
            "requirement_text": "Implement abs()",
            "safety_standard": "DO_178C",
            "target_language": "Python",
        })
        assert resp.status_code == 200

    def test_post_generate_response_schema(self):
        from starlette.testclient import TestClient
        from backend.main import app
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post("/api/v1/generation/generate", json={
            "requirement_text": "Implement abs()",
            "safety_standard": "DO_178C",
            "target_language": "Python",
        })
        body = resp.json()
        assert "generated_code" in body
        assert "formal_proof" in body
        assert "compliance_status" in body

    def test_compliance_true_when_awaiting_approval(self):
        from starlette.testclient import TestClient
        from backend.main import app
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post("/api/v1/generation/generate", json={
            "requirement_text": "Implement abs()",
            "safety_standard": "DO_178C",
            "target_language": "Python",
        })
        assert resp.json()["compliance_status"] is True

    def test_compliance_false_when_pipeline_failed(self):
        from starlette.testclient import TestClient
        from backend.main import app
        from backend.api.dependencies import get_orchestrator
        app.dependency_overrides[get_orchestrator] = lambda: _make_mock_orchestrator(
            status=PipelineStatus.FAILED.value
        )
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post("/api/v1/generation/generate", json={
            "requirement_text": "Implement abs()",
            "safety_standard": "DO_178C",
            "target_language": "Python",
        })
        assert resp.json()["compliance_status"] is False

    def test_missing_required_fields_returns_422(self):
        from starlette.testclient import TestClient
        from backend.main import app
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/v1/generation/generate", json={})
        assert resp.status_code == 422

    def test_invalid_safety_standard_returns_422(self):
        from starlette.testclient import TestClient
        from backend.main import app
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/v1/generation/generate", json={
            "requirement_text": "Implement abs()",
            "safety_standard": "NOT_A_STANDARD",
            "target_language": "Python",
        })
        assert resp.status_code == 422

    def test_invalid_target_language_returns_422(self):
        from starlette.testclient import TestClient
        from backend.main import app
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/v1/generation/generate", json={
            "requirement_text": "Implement abs()",
            "safety_standard": "DO_178C",
            "target_language": "COBOL",
        })
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /api/v1/pipeline/run (SSE)
# ---------------------------------------------------------------------------

class TestPipelineStreamEndpoint:
    @pytest.fixture(autouse=True)
    def override_orchestrator(self):
        from backend.main import app
        from backend.api.dependencies import get_orchestrator
        mock = _make_mock_orchestrator()
        app.dependency_overrides[get_orchestrator] = lambda: mock
        yield
        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_sse_response_content_type(self):
        import httpx
        from backend.main import app
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/pipeline/run", json={
                "requirement_text": "Implement abs()",
            })
        assert "text/event-stream" in resp.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_sse_response_has_cache_control(self):
        import httpx
        from backend.main import app
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/pipeline/run", json={
                "requirement_text": "Implement abs()",
            })
        assert resp.headers.get("cache-control") == "no-cache"

    @pytest.mark.asyncio
    async def test_sse_events_are_parseable(self):
        import httpx
        from backend.main import app
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/pipeline/run", json={
                "requirement_text": "Implement abs()",
            })
        raw = resp.text
        blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
        assert blocks, "no SSE blocks received"
        for block in blocks:
            lines = block.splitlines()
            data_lines = [l[len("data: "):] for l in lines if l.startswith("data: ")]
            for data in data_lines:
                event = StreamEvent.model_validate_json(data)
                assert event.run_id is not None

    @pytest.mark.asyncio
    async def test_sse_contains_pipeline_complete(self):
        import httpx
        from backend.main import app
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/pipeline/run", json={
                "requirement_text": "Implement abs()",
            })
        assert "pipeline_complete" in resp.text

    @pytest.mark.asyncio
    async def test_sse_format_correct(self):
        import httpx
        from backend.main import app
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/pipeline/run", json={
                "requirement_text": "Implement abs()",
            })
        raw = resp.text
        blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
        for block in blocks:
            lines = block.splitlines()
            event_lines = [l for l in lines if l.startswith("event: ")]
            data_lines = [l for l in lines if l.startswith("data: ")]
            assert event_lines, f"SSE block missing 'event:' line: {block!r}"
            assert data_lines, f"SSE block missing 'data:' line: {block!r}"


# ---------------------------------------------------------------------------
# /api/v1/audit/* endpoints
# ---------------------------------------------------------------------------

class TestAuditEndpoints:
    @pytest.fixture
    def client_with_real_audit(self, tmp_path):
        from starlette.testclient import TestClient
        from backend.main import app
        from backend.api.dependencies import get_audit_logger
        from backend.services.audit.logger import AuditLogger

        audit = AuditLogger(
            log_dir=str(tmp_path / "logs"),
            db_path=str(tmp_path / "audit.db"),
        )
        app.dependency_overrides[get_audit_logger] = lambda: audit
        yield TestClient(app, raise_server_exceptions=True), audit
        app.dependency_overrides.pop(get_audit_logger, None)

    def test_query_audit_returns_200_empty(self, client_with_real_audit):
        client, _ = client_with_real_audit
        resp = client.post("/api/v1/audit/query", json={})
        assert resp.status_code == 200
        body = resp.json()
        assert body["entries"] == []
        assert body["total"] == 0

    def test_query_audit_with_run_id(self, client_with_real_audit):
        client, audit = client_with_real_audit
        run_id = uuid4()
        audit.log(run_id, "agent_output", agent="actor", data={"key": "val"})
        resp = client.post("/api/v1/audit/query", json={"run_id": str(run_id)})
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["entries"][0]["event_type"] == "agent_output"

    def test_query_audit_with_agent_filter(self, client_with_real_audit):
        client, audit = client_with_real_audit
        run_id = uuid4()
        audit.log(run_id, "agent_output", agent="actor", data={})
        audit.log(run_id, "agent_output", agent="checker", data={})
        resp = client.post("/api/v1/audit/query", json={"agent": "actor"})
        body = resp.json()
        assert body["total"] == 1
        assert body["entries"][0]["agent"] == "actor"

    def test_get_run_audit_returns_all_entries(self, client_with_real_audit):
        client, audit = client_with_real_audit
        run_id = uuid4()
        audit.log(run_id, "pipeline_start", data={})
        audit.log(run_id, "agent_output", agent="actor", data={})
        resp = client.get(f"/api/v1/audit/run/{run_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2

    def test_get_run_audit_unknown_run_id_returns_empty(self, client_with_real_audit):
        client, _ = client_with_real_audit
        unknown = uuid4()
        resp = client.get(f"/api/v1/audit/run/{unknown}")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0
