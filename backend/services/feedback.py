"""Feedback composition — builds structured feedback from checker outputs.

The orchestrator uses this to compose a prioritized FeedbackMessage
that tells the Actor exactly what to fix in the next iteration.
"""

from __future__ import annotations

from backend.api.schemas.agents import (
    CheckerReport,
    CheckerVerdict,
    FeedbackMessage,
    PolicyVerdict,
    Severity,
    VerificationResult,
)


def compose_feedback(
    iteration: int,
    checker_report: CheckerReport | None = None,
    verification_result: VerificationResult | None = None,
    policy_verdict: PolicyVerdict | None = None,
) -> FeedbackMessage:
    """Build a prioritized feedback message from all checker outputs.

    Priority order:
    1. Formal verification failures (mathematical proof failed)
    2. Critical checker issues (safety hazards)
    3. Policy violations (compliance failures)
    4. Major/minor checker issues
    """
    priority_parts: list[str] = []

    # Priority 1: Formal verification failures
    if verification_result and not verification_result.verified:
        priority_parts.append("FORMAL VERIFICATION FAILED.")
        if verification_result.failing_assertions:
            priority_parts.append(
                "Failing assertions: "
                + "; ".join(verification_result.failing_assertions[:5])
            )

    # Priority 2: Critical checker issues
    if checker_report and checker_report.verdict == CheckerVerdict.FAIL:
        critical = [
            i for i in checker_report.issues if i.severity == Severity.CRITICAL
        ]
        if critical:
            priority_parts.append(
                f"CRITICAL ISSUES ({len(critical)}): "
                + "; ".join(i.description for i in critical[:3])
            )

    # Priority 3: Policy violations
    if policy_verdict and not policy_verdict.compliant:
        violations_summary = "; ".join(
            f"[{v.rule_id}] {v.description}" for v in policy_verdict.violations[:5]
        )
        priority_parts.append(f"POLICY VIOLATIONS: {violations_summary}")

    # Priority 4: Non-critical checker issues
    if checker_report:
        non_critical = [
            i
            for i in checker_report.issues
            if i.severity in (Severity.MAJOR, Severity.MINOR)
        ]
        if non_critical:
            priority_parts.append(
                f"Additional issues ({len(non_critical)}): "
                + "; ".join(i.description for i in non_critical[:3])
            )

    return FeedbackMessage(
        iteration=iteration,
        checker_feedback=checker_report,
        verification_feedback=verification_result,
        policy_feedback=policy_verdict,
        priority_summary=" | ".join(priority_parts) if priority_parts else "All checks passed.",
    )
