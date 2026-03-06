"""Append-only audit logger — JSONL files + SQLite.

Every agent call, verification result, and policy decision is recorded.
This is the foundation for DO-178C traceability evidence.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from backend.api.schemas.audit import AuditEntry
from backend.config import settings

logger = logging.getLogger(__name__)


class AuditLogger:
    """Dual-write audit logger: JSONL (per-run) + SQLite (cross-run queries)."""

    def __init__(
        self,
        log_dir: str | None = None,
        db_path: str = "data/audit.db",
    ) -> None:
        self._log_dir = Path(log_dir or settings.pipeline.audit_log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Create the audit table if it doesn't exist."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    agent TEXT,
                    data TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_run_id ON audit_log(run_id)
            """)

    def log(
        self,
        run_id: UUID,
        event_type: str,
        agent: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Record an audit event to both JSONL and SQLite."""
        entry = AuditEntry(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            agent=agent,
            data=data or {},
        )

        # 1. Append to per-run JSONL file
        jsonl_path = self._log_dir / f"{run_id}.jsonl"
        with open(jsonl_path, "a") as f:
            f.write(entry.model_dump_json() + "\n")

        # 2. Insert into SQLite
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "INSERT INTO audit_log (run_id, timestamp, event_type, agent, data) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    str(entry.run_id),
                    entry.timestamp.isoformat(),
                    entry.event_type,
                    entry.agent,
                    json.dumps(entry.data, default=str),
                ),
            )

        return entry

    def get_run_log(self, run_id: UUID) -> list[AuditEntry]:
        """Read all audit entries for a given run from JSONL."""
        jsonl_path = self._log_dir / f"{run_id}.jsonl"
        if not jsonl_path.exists():
            return []

        entries: list[AuditEntry] = []
        for line in jsonl_path.read_text().splitlines():
            if line.strip():
                entries.append(AuditEntry.model_validate_json(line))
        return entries

    def query(
        self,
        run_id: UUID | None = None,
        agent: str | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Query audit entries from SQLite."""
        conditions: list[str] = []
        params: list[Any] = []

        if run_id:
            conditions.append("run_id = ?")
            params.append(str(run_id))
        if agent:
            conditions.append("agent = ?")
            params.append(agent)
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT run_id, timestamp, event_type, agent, data FROM audit_log {where} ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            AuditEntry(
                run_id=UUID(row[0]),
                timestamp=datetime.fromisoformat(row[1]),
                event_type=row[2],
                agent=row[3],
                data=json.loads(row[4]),
            )
            for row in rows
        ]
