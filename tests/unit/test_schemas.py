"""Unit tests for Pydantic schemas — verify serialization and constrained decoding schemas."""

import json

import pytest

from backend.api.schemas.agents import (
    CheckerReport,
    CheckerVerdict,
    CodeCandidate,
    FeedbackMessage,
    Issue,
    PolicyVerdict,
    PolicyViolation,
    RiskLevel,
    Severity,
    VerificationResult,
)
from backend.api.schemas.pipeline import (
    PipelineRequest,
    PipelineState,
    PipelineStatus,
    StreamEvent,
    StreamEventType,
)


class TestCodeCandidate:
    def test_minimal(self):
        cc = CodeCandidate(source_code="int main() { return 0; }")
        assert cc.source_code == "int main() { return 0; }"
        assert cc.dafny_spec == ""
        assert cc.language == "C"

    def test_full(self):
        cc = CodeCandidate(
            source_code="int add(int a, int b) { return a + b; }",
            dafny_spec="method Add(a: int, b: int) returns (r: int) ensures r == a + b",
            reasoning_trace="Simple addition function",
            language="C",
            annotations={"REQ": "REQ-001"},
        )
        assert cc.annotations["REQ"] == "REQ-001"

    def test_json_schema_exportable(self):
        """Verify the schema can be exported for constrained decoding."""
        schema = CodeCandidate.model_json_schema()
        assert "properties" in schema
        assert "source_code" in schema["properties"]


class TestCheckerReport:
    def test_pass_verdict(self):
        cr = CheckerReport(verdict=CheckerVerdict.PASS)
        assert cr.verdict == CheckerVerdict.PASS
        assert cr.issues == []

    def test_fail_with_issues(self):
        cr = CheckerReport(
            verdict=CheckerVerdict.FAIL,
            issues=[
                Issue(description="Buffer overflow", severity=Severity.CRITICAL),
                Issue(description="Missing null check", severity=Severity.MAJOR),
            ],
        )
        assert len(cr.issues) == 2
        assert cr.issues[0].severity == Severity.CRITICAL


class TestVerificationResult:
    def test_passed(self):
        vr = VerificationResult(verified=True, prover="dafny")
        assert vr.verified is True
        assert vr.failing_assertions == []

    def test_failed(self):
        vr = VerificationResult(
            verified=False,
            prover="dafny",
            failing_assertions=["postcondition might not hold"],
        )
        assert vr.verified is False
        assert len(vr.failing_assertions) == 1


class TestPolicyVerdict:
    def test_compliant(self):
        pv = PolicyVerdict(compliant=True, risk_level=RiskLevel.LOW)
        assert pv.compliant is True

    def test_non_compliant(self):
        pv = PolicyVerdict(
            compliant=False,
            risk_level=RiskLevel.HIGH,
            violations=[
                PolicyViolation(
                    rule_id="DO-178C-CR-001",
                    description="Dynamic memory allocation detected",
                    severity=Severity.CRITICAL,
                    standard="DO_178C",
                )
            ],
        )
        assert len(pv.violations) == 1
        assert pv.violations[0].rule_id == "DO-178C-CR-001"


class TestPipelineState:
    def test_initial_state(self):
        state = PipelineState()
        assert state.status == PipelineStatus.PENDING
        assert state.iterations == []
        assert state.run_id is not None

    def test_roundtrip_json(self):
        state = PipelineState(
            request=PipelineRequest(requirement_text="test"),
            status=PipelineStatus.RUNNING,
        )
        json_str = state.model_dump_json()
        restored = PipelineState.model_validate_json(json_str)
        assert restored.run_id == state.run_id
        assert restored.request.requirement_text == "test"


class TestFeedbackMessage:
    def test_empty_feedback(self):
        fb = FeedbackMessage(iteration=1)
        assert fb.priority_summary == ""

    def test_with_all_feedback(self):
        fb = FeedbackMessage(
            iteration=2,
            checker_feedback=CheckerReport(verdict=CheckerVerdict.FAIL),
            verification_feedback=VerificationResult(verified=False),
            policy_feedback=PolicyVerdict(compliant=False, risk_level=RiskLevel.HIGH),
            priority_summary="CRITICAL: multiple failures",
        )
        assert fb.iteration == 2
        assert "CRITICAL" in fb.priority_summary
