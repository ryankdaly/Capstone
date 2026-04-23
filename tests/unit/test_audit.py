"""Unit tests for the audit logger."""

import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from backend.services.audit.logger import AuditLogger


@pytest.fixture
def audit_logger(tmp_path):
    return AuditLogger(
        log_dir=str(tmp_path / "logs"),
        db_path=str(tmp_path / "test_audit.db"),
    )


class TestAuditLogger:
    def test_log_creates_jsonl(self, audit_logger, tmp_path):
        run_id = uuid4()
        audit_logger.log(run_id, "test_event", agent="actor", data={"key": "value"})

        jsonl_path = tmp_path / "logs" / f"{run_id}.jsonl"
        assert jsonl_path.exists()
        content = jsonl_path.read_text()
        assert "test_event" in content

    def test_get_run_log(self, audit_logger):
        run_id = uuid4()
        audit_logger.log(run_id, "event_1")
        audit_logger.log(run_id, "event_2", agent="checker")

        entries = audit_logger.get_run_log(run_id)
        assert len(entries) == 2
        assert entries[0].event_type == "event_1"
        assert entries[1].agent == "checker"

    def test_query_by_run_id(self, audit_logger):
        run_1 = uuid4()
        run_2 = uuid4()
        audit_logger.log(run_1, "event_a")
        audit_logger.log(run_2, "event_b")

        entries = audit_logger.query(run_id=run_1)
        assert len(entries) == 1
        assert entries[0].event_type == "event_a"

    def test_query_by_agent(self, audit_logger):
        run_id = uuid4()
        audit_logger.log(run_id, "start", agent="actor")
        audit_logger.log(run_id, "start", agent="checker")

        entries = audit_logger.query(agent="actor")
        assert len(entries) == 1

    def test_empty_run_log(self, audit_logger):
        entries = audit_logger.get_run_log(uuid4())
        assert entries == []
