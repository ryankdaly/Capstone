"""Unit tests for feedback composition."""

from backend.api.schemas.agents import (
    CheckerReport,
    CheckerVerdict,
    Issue,
    PolicyVerdict,
    PolicyViolation,
    RiskLevel,
    Severity,
    VerificationResult,
)
from backend.services.feedback import compose_feedback


class TestComposeFeedback:
    def test_all_pass(self):
        fb = compose_feedback(
            iteration=1,
            checker_report=CheckerReport(verdict=CheckerVerdict.PASS),
            verification_result=VerificationResult(verified=True),
            policy_verdict=PolicyVerdict(compliant=True, risk_level=RiskLevel.LOW),
        )
        assert "All checks passed" in fb.priority_summary

    def test_verification_failure_prioritized(self):
        fb = compose_feedback(
            iteration=1,
            checker_report=CheckerReport(verdict=CheckerVerdict.PASS),
            verification_result=VerificationResult(
                verified=False,
                failing_assertions=["postcondition might not hold"],
            ),
            policy_verdict=PolicyVerdict(compliant=True, risk_level=RiskLevel.LOW),
        )
        assert "FORMAL VERIFICATION FAILED" in fb.priority_summary

    def test_critical_issues_included(self):
        fb = compose_feedback(
            iteration=1,
            checker_report=CheckerReport(
                verdict=CheckerVerdict.FAIL,
                issues=[
                    Issue(description="Buffer overflow", severity=Severity.CRITICAL),
                ],
            ),
            verification_result=VerificationResult(verified=True),
            policy_verdict=PolicyVerdict(compliant=True, risk_level=RiskLevel.LOW),
        )
        assert "CRITICAL ISSUES" in fb.priority_summary

    def test_policy_violations_included(self):
        fb = compose_feedback(
            iteration=1,
            checker_report=CheckerReport(verdict=CheckerVerdict.PASS),
            verification_result=VerificationResult(verified=True),
            policy_verdict=PolicyVerdict(
                compliant=False,
                risk_level=RiskLevel.HIGH,
                violations=[
                    PolicyViolation(
                        rule_id="DO-178C-CR-001",
                        description="Dynamic memory allocation",
                    ),
                ],
            ),
        )
        assert "POLICY VIOLATIONS" in fb.priority_summary
        assert "DO-178C-CR-001" in fb.priority_summary

    def test_none_inputs(self):
        fb = compose_feedback(iteration=1)
        assert fb.priority_summary == "All checks passed."
