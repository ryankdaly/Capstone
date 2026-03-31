"""Pipeline-level schemas: run state, iteration records, SSE stream events."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from backend.api.schemas.agents import (
    CheckerReport,
    CodeCandidate,
    FeedbackMessage,
    PolicyVerdict,
    VerificationResult,
)
from backend.api.schemas.generation import SafetyStandard, TargetLanguage


# ---------------------------------------------------------------------------
# Pipeline request (superset of GenerationRequest for the full pipeline)
# ---------------------------------------------------------------------------

class PipelineRequest(BaseModel):
    requirement_text: str
    safety_standard: SafetyStandard = SafetyStandard.DO_178C
    target_language: TargetLanguage = TargetLanguage.C
    max_iterations: int = Field(default=3, ge=1, le=10)
    stage: PipelineStage = PipelineStage.POLICY  # default: full pipeline


# ---------------------------------------------------------------------------
# Iteration record — one pass through the Actor-Checker loop
# ---------------------------------------------------------------------------

class IterationRecord(BaseModel):
    iteration: int
    code_candidate: Optional[CodeCandidate] = None
    checker_report: Optional[CheckerReport] = None
    verification_result: Optional[VerificationResult] = None
    policy_verdict: Optional[PolicyVerdict] = None
    feedback: Optional[FeedbackMessage] = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Pipeline status
# ---------------------------------------------------------------------------

class PipelineStage(str, Enum):
    """Ordered pipeline stages — orchestrator runs up to (and including) this stage.

    Add new stages between existing ones to incrementally enable pipeline sections.
    The integer values define execution order.
    """
    ACTOR = "actor"             # Code generation only
    CHECKER = "checker"         # + Checker review + Dafny verification (parallel)
    POLICY = "policy"           # + Policy compliance audit (full pipeline)


# Execution order for stage comparison
_STAGE_ORDER = {
    PipelineStage.ACTOR: 1,
    PipelineStage.CHECKER: 2,
    PipelineStage.POLICY: 3,
}


def stage_enabled(current: PipelineStage, max_stage: PipelineStage) -> bool:
    """Return True if `current` stage should run given the configured `max_stage`."""
    return _STAGE_ORDER[current] <= _STAGE_ORDER[max_stage]


class PipelineStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    FAILED = "failed"
    COMPLETED = "completed"


# ---------------------------------------------------------------------------
# Full pipeline state
# ---------------------------------------------------------------------------

class PipelineState(BaseModel):
    """Complete state of a single pipeline run — the core audit artifact."""

    run_id: UUID = Field(default_factory=uuid4)
    request: Optional[PipelineRequest] = None
    status: PipelineStatus = PipelineStatus.PENDING
    iterations: list[IterationRecord] = Field(default_factory=list)
    current_iteration: int = 0
    final_code: Optional[str] = None
    final_proof: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# SSE stream event (sent to CLI / dashboard in real-time)
# ---------------------------------------------------------------------------

class StreamEventType(str, Enum):
    AGENT_START = "agent_start"
    AGENT_OUTPUT = "agent_output"
    AGENT_ERROR = "agent_error"
    ITERATION_COMPLETE = "iteration_complete"
    PIPELINE_COMPLETE = "pipeline_complete"
    HUMAN_APPROVAL_REQUIRED = "human_approval_required"


class StreamEvent(BaseModel):
    event_type: StreamEventType
    run_id: UUID
    agent: Optional[str] = None
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
