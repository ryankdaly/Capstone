"""Actor agent — generates safety-critical source code.

Dafny formal specification is handled exclusively by the DafnyArchitect
agent which runs after this one. The Actor focuses solely on producing
correct, defensively-written source code.
"""

from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel

from backend.api.schemas.agents import CodeCandidate, FeedbackMessage
from backend.services.agents.base import BaseAgent


class ActorAgent(BaseAgent):
    role = "actor"
    prompt_file = "actor.txt"

    def _output_schema(self) -> Type[BaseModel]:
        return CodeCandidate

    def _build_user_prompt(self, **kwargs: Any) -> str:
        requirement: str = kwargs["requirement"]
        language: str = kwargs.get("language", "Python")
        standard: str = kwargs.get("standard", "DO_178C")
        feedback: FeedbackMessage | None = kwargs.get("feedback")
        retry_hint: str | None = kwargs.get("retry_hint")

        parts = [
            f"## Requirement\n{requirement}",
            f"## Target Language\n{language}",
            f"## Safety Standard\n{standard}",
        ]

        # Within-attempt retry hint (parse errors, think loops, etc.)
        if retry_hint:
            parts.append(
                f"## Retry Notice\n"
                f"Your previous attempt failed: {retry_hint[:400]}\n"
                f"Output ONLY the JSON object — no prose, no markdown, no <think> blocks."
            )

        if feedback:
            parts.append(
                f"## Feedback from Iteration {feedback.iteration} — "
                f"address EVERY item below before writing new code"
            )

            if feedback.priority_summary and feedback.priority_summary != "All checks passed.":
                parts.append(f"**Priority:** {feedback.priority_summary}")

            # Checker issues — full detail with suggested fixes
            if feedback.checker_feedback:
                cf = feedback.checker_feedback
                parts.append(f"### Checker verdict: {cf.verdict.value.upper()}")
                for issue in cf.issues:
                    line = f"  [{issue.severity.value}] {issue.description}"
                    if issue.line_reference:
                        line += f" (near line {issue.line_reference})"
                    parts.append(line)
                    if issue.suggested_fix:
                        parts.append(f"    Fix: {issue.suggested_fix}")

            # Formal verification errors — surface so actor can simplify constructs
            # that are structurally hard to verify (e.g., avoid floating-point math,
            # reduce branch complexity, add explicit bounds checks).
            if feedback.verification_feedback and not feedback.verification_feedback.verified:
                vf = feedback.verification_feedback
                parts.append("### Dafny verification: FAILED — simplify logic to aid verification")
                if vf.failing_assertions:
                    parts.append("Failing assertions:")
                    for a in vf.failing_assertions[:5]:
                        parts.append(f"  - {a}")
                if vf.solver_output:
                    err_lines = [
                        ln for ln in vf.solver_output.splitlines()
                        if any(k in ln for k in ("Error", "error", "Warning", "failed"))
                    ][:8]
                    if err_lines:
                        parts.append("Solver errors:\n" + "\n".join(f"  {l}" for l in err_lines))

            # Test execution results — highest-signal feedback (real crashes)
            if feedback.test_feedback and feedback.test_feedback.executed:
                tf = feedback.test_feedback
                if tf.failed > 0:
                    parts.append(f"### Tests: {tf.failed}/{tf.total} FAILED")
                    for tr in tf.test_results:
                        if not tr.passed:
                            parts.append(f"  FAIL  {tr.name}")
                            if tr.error_message:
                                parts.append(f"        {tr.error_message[:300]}")
                    if tf.pytest_output:
                        fail_lines = [
                            ln for ln in tf.pytest_output.splitlines()
                            if any(k in ln for k in
                                   ("FAILED", "AssertionError", "Error", "assert", "TypeError"))
                        ][:10]
                        if fail_lines:
                            parts.append("Pytest output:\n" + "\n".join(f"  {l}" for l in fail_lines))

            # Policy violations
            if feedback.policy_feedback and not feedback.policy_feedback.compliant:
                pf = feedback.policy_feedback
                parts.append("### Policy: NON-COMPLIANT")
                for v in pf.violations:
                    parts.append(f"  [{v.rule_id}] {v.description}")

        return "\n\n".join(parts)

    async def run(self, **kwargs: Any) -> CodeCandidate:  # type: ignore[override]
        return await super().run(**kwargs)  # type: ignore[return-value]
