"""Schemas for audit trail queries."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class AuditEntry(BaseModel):
    """Single entry in the append-only audit log."""

    run_id: UUID
    timestamp: datetime
    event_type: str
    agent: Optional[str] = None
    data: dict[str, Any] = Field(default_factory=dict)


class AuditQueryRequest(BaseModel):
    run_id: Optional[UUID] = None
    agent: Optional[str] = None
    event_type: Optional[str] = None
    limit: int = Field(default=100, ge=1, le=1000)


class AuditQueryResponse(BaseModel):
    entries: list[AuditEntry] = Field(default_factory=list)
    total: int = 0


class TraceabilityRow(BaseModel):
    """One row of the requirement → code → test → proof → policy matrix."""

    requirement: str
    code_artifact: str = ""
    test_cases: list[str] = Field(default_factory=list)
    formal_proof_status: str = ""
    policy_verdict: str = ""
    standard_references: list[str] = Field(default_factory=list)


class TraceabilityMatrix(BaseModel):
    run_id: UUID
    rows: list[TraceabilityRow] = Field(default_factory=list)
