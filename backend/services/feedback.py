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
    TestRunResult,
    VerificationResult,
)


def compose_feedback(
    iteration: int,
    checker_report: CheckerReport | None = None,
    verification_result: VerificationResult | None = None,
    policy_verdict: PolicyVerdict | None = None,
    test_result: TestRunResult | None = None,
) -> FeedbackMessage:
    """Build a prioritized feedback message from all checker outputs.

    Priority order:
    1. Test execution failures (ground truth — code actually crashed)
    2. Formal verification failures (mathematical proof failed)
    3. Critical checker issues (safety hazards)
    4. Policy violations (compliance failures)
    5. Major/minor checker issues
    """
    priority_parts: list[str] = []

    # Compute test pass ratio to gate low-priority feedback.
    # When most tests pass, the actor should focus on failing tests only —
    # not rewrite working code to address cosmetic or style issues.
    test_pass_ratio = 1.0
    if test_result and test_result.executed and test_result.total > 0:
        test_pass_ratio = test_result.passed / test_result.total

    mostly_passing = test_pass_ratio >= 0.7

    # When mostly passing, prepend a directive so the actor doesn't nuke
    # working logic to chase low-priority feedback.
    if mostly_passing and test_result and test_result.executed and test_result.failed > 0:
        priority_parts.append(
            "MOSTLY PASSING — fix ONLY the failing test(s) below. "
            "Do NOT rewrite working logic or address style issues."
        )

    # Priority 0: Actual test execution failures (highest priority — real evidence)
    if test_result and test_result.executed and test_result.failed > 0:
        failing_names = [
            t.name for t in test_result.test_results if not t.passed
        ]
        priority_parts.append(
            f"PYTEST FAILURES ({test_result.failed}/{test_result.total}): "
            f"Tests that failed: {', '.join(failing_names[:5])}."
        )
        # Include truncated pytest output for context
        if test_result.pytest_output:
            # Get the FAILURES section if present
            output = test_result.pytest_output
            if "FAILED" in output:
                # Extract just failure lines (not the full verbose output)
                fail_lines = [
                    line for line in output.splitlines()
                    if "FAILED" in line or "AssertionError" in line or "Error" in line
                ]
                if fail_lines:
                    priority_parts.append(
                        "Pytest output: " + "; ".join(fail_lines[:5])
                    )

    # Priority 1: Formal verification failures
    if verification_result and not verification_result.verified:
        priority_parts.append("FORMAL VERIFICATION FAILED.")
        if verification_result.failing_assertions:
            priority_parts.append(
                "Failing assertions: "
                + "; ".join(verification_result.failing_assertions[:5])
            )

    # Priority 2: Critical checker issues (always included — safety hazards)
    if checker_report and checker_report.verdict == CheckerVerdict.FAIL:
        critical = [
            i for i in checker_report.issues if i.severity == Severity.CRITICAL
        ]
        if critical:
            priority_parts.append(
                f"CRITICAL ISSUES ({len(critical)}): "
                + "; ".join(i.description for i in critical[:3])
            )

    # Priority 3: Policy violations — always included (gates convergence)
    if policy_verdict and not policy_verdict.compliant:
        violations_summary = "; ".join(
            f"[{v.rule_id}] {v.description}" for v in policy_verdict.violations[:5]
        )
        priority_parts.append(f"POLICY VIOLATIONS: {violations_summary}")

    # Priority 4: Non-critical checker issues — suppress when mostly passing
    if not mostly_passing and checker_report:
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
        test_feedback=test_result,
        priority_summary=" | ".join(priority_parts) if priority_parts else "All checks passed.",
    )
