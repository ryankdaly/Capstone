"""Traceability matrix generator.

Produces the requirement → code → test → proof → policy mapping
that DO-178C auditors need as certification evidence.
"""

from __future__ import annotations

from uuid import UUID

from backend.api.schemas.agents import CheckerVerdict
from backend.api.schemas.audit import TraceabilityMatrix, TraceabilityRow
from backend.api.schemas.pipeline import PipelineState


def build_traceability_matrix(state: PipelineState) -> TraceabilityMatrix:
    """Build a traceability matrix from a completed pipeline run."""
    rows: list[TraceabilityRow] = []

    requirement = state.request.requirement_text if state.request else "Unknown"

    # Use the final iteration's data (or the last available)
    if state.iterations:
        final = state.iterations[-1]

        code_artifact = ""
        if final.code_candidate:
            # Truncate for the matrix view — full code is in the audit log
            lines = final.code_candidate.source_code.splitlines()
            code_artifact = f"{len(lines)} lines of {final.code_candidate.language}"

        test_cases: list[str] = []
        if final.checker_report:
            test_cases = final.checker_report.test_cases

        proof_status = "Not attempted"
        if final.verification_result:
            proof_status = "VERIFIED" if final.verification_result.verified else "FAILED"

        policy_verdict = "Not evaluated"
        standard_refs: list[str] = []
        if final.policy_verdict:
            policy_verdict = "COMPLIANT" if final.policy_verdict.compliant else "NON-COMPLIANT"
            standard_refs = [v.rule_id for v in final.policy_verdict.violations]

        rows.append(
            TraceabilityRow(
                requirement=requirement,
                code_artifact=code_artifact,
                test_cases=test_cases,
                formal_proof_status=proof_status,
                policy_verdict=policy_verdict,
                standard_references=standard_refs,
            )
        )

    return TraceabilityMatrix(run_id=state.run_id, rows=rows)
