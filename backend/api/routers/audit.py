"""Audit router — query audit trails and traceability matrices."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends

from backend.api.dependencies import get_audit_logger
from backend.api.schemas.audit import (
    AuditQueryRequest,
    AuditQueryResponse,
    TraceabilityMatrix,
)
from backend.services.audit.logger import AuditLogger

router = APIRouter()


@router.post(
    "/query",
    response_model=AuditQueryResponse,
    summary="Query audit trail entries",
    tags=["audit"],
)
async def query_audit(
    request: AuditQueryRequest,
    audit_logger: AuditLogger = Depends(get_audit_logger),
) -> AuditQueryResponse:
    entries = audit_logger.query(
        run_id=request.run_id,
        agent=request.agent,
        event_type=request.event_type,
        limit=request.limit,
    )
    return AuditQueryResponse(entries=entries, total=len(entries))


@router.get(
    "/run/{run_id}",
    summary="Get full audit log for a pipeline run",
    tags=["audit"],
)
async def get_run_audit(
    run_id: UUID,
    audit_logger: AuditLogger = Depends(get_audit_logger),
) -> AuditQueryResponse:
    entries = audit_logger.get_run_log(run_id)
    return AuditQueryResponse(entries=entries, total=len(entries))
