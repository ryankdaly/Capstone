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

        parts = [
            f"## Requirement\n{requirement}",
            f"## Target Language\n{language}",
            f"## Safety Standard\n{standard}",
        ]

        if feedback:
            parts.append("## Feedback from Previous Iteration")
            parts.append(f"Iteration: {feedback.iteration}")
            if feedback.priority_summary:
                parts.append(f"Priority Summary: {feedback.priority_summary}")
            if feedback.checker_feedback:
                parts.append(
                    f"Checker Verdict: {feedback.checker_feedback.verdict.value}\n"
                    f"Issues: {len(feedback.checker_feedback.issues)}"
                )
                for issue in feedback.checker_feedback.issues:
                    parts.append(f"  - [{issue.severity.value}] {issue.description}")
                    if issue.suggested_fix:
                        parts.append(f"    Fix: {issue.suggested_fix}")
            if feedback.policy_feedback:
                pf = feedback.policy_feedback
                parts.append(f"Policy Compliance: {'COMPLIANT' if pf.compliant else 'NON-COMPLIANT'}")
                for v in pf.violations:
                    parts.append(f"  - [{v.rule_id}] {v.description}")

        return "\n\n".join(parts)

    async def run(self, **kwargs: Any) -> CodeCandidate:  # type: ignore[override]
        return await super().run(**kwargs)  # type: ignore[return-value]
