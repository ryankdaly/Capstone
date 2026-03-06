"""Pydantic schemas for inter-agent communication.

Every agent produces one of these typed models. Constrained decoding (vLLM
guided_json) enforces these schemas at the token level during generation —
the model literally cannot produce malformed output.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Actor → CodeCandidate
# ---------------------------------------------------------------------------

class CodeCandidate(BaseModel):
    """Output of the Actor agent: generated source code + Dafny specification."""

    source_code: str = Field(..., description="Generated source code")
    dafny_spec: str = Field(
        default="",
        description="Dafny annotations (requires/ensures/invariants) for formal verification",
    )
    reasoning_trace: str = Field(
        default="",
        description="Actor's chain-of-thought explaining design decisions",
    )
    language: str = Field(default="C", description="Target language of generated code")
    annotations: dict[str, str] = Field(
        default_factory=dict,
        description="Metadata annotations (e.g., traceability tags)",
    )


# ---------------------------------------------------------------------------
# Checker → CheckerReport
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "info"


class Issue(BaseModel):
    description: str
    severity: Severity = Severity.MAJOR
    line_reference: Optional[str] = None
    suggested_fix: str = ""


class CheckerVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"


class CheckerReport(BaseModel):
    """Output of the Checker agent: test cases, issues, and overall verdict."""

    verdict: CheckerVerdict
    issues: list[Issue] = Field(default_factory=list)
    test_cases: list[str] = Field(
        default_factory=list,
        description="Generated test case snippets",
    )
    reasoning_trace: str = ""


# ---------------------------------------------------------------------------
# Dafny/Lean → VerificationResult
# ---------------------------------------------------------------------------

class VerificationResult(BaseModel):
    """Output of the formal verifier subprocess (Dafny or Lean 4)."""

    verified: bool
    prover: str = "dafny"
    solver_output: str = ""
    failing_assertions: list[str] = Field(default_factory=list)
    execution_time_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Policy → PolicyVerdict
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PolicyViolation(BaseModel):
    rule_id: str = Field(..., description="Standard clause reference (e.g., DO-178C §6.3.1)")
    description: str
    severity: Severity = Severity.MAJOR
    standard: str = ""


class PolicyVerdict(BaseModel):
    """Output of the Policy agent: compliance verdict with cited standard references."""

    compliant: bool
    risk_level: RiskLevel = RiskLevel.MEDIUM
    violations: list[PolicyViolation] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    reasoning_trace: str = ""


# ---------------------------------------------------------------------------
# Feedback (orchestrator → actor on loop iteration)
# ---------------------------------------------------------------------------

class FeedbackMessage(BaseModel):
    """Structured feedback composed by the orchestrator from all checker outputs."""

    iteration: int
    checker_feedback: Optional[CheckerReport] = None
    verification_feedback: Optional[VerificationResult] = None
    policy_feedback: Optional[PolicyVerdict] = None
    priority_summary: str = Field(
        default="",
        description="Orchestrator-composed summary prioritizing critical failures",
    )
