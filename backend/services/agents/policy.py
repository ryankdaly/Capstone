"""Policy agent — enforces safety standard compliance using RAG context."""

from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel

from backend.api.schemas.agents import (
    CheckerReport,
    PolicyVerdict,
    VerificationResult,
)
from backend.services.agents.base import BaseAgent


class PolicyAgent(BaseAgent):
    role = "policy"
    prompt_file = "policy.txt"

    def _output_schema(self) -> Type[BaseModel]:
        return PolicyVerdict

    def _build_user_prompt(self, **kwargs: Any) -> str:
        source_code: str = kwargs["source_code"]
        dafny_spec: str = kwargs.get("dafny_spec", "")
        standard: str = kwargs.get("standard", "DO_178C")
        checker_report: CheckerReport | None = kwargs.get("checker_report")
        verification_result: VerificationResult | None = kwargs.get("verification_result")
        policy_context: str = kwargs.get("policy_context", "")

        parts = [
            f"## Source Code\n```\n{source_code}\n```",
        ]

        if dafny_spec:
            parts.append(f"## Dafny Specification\n```dafny\n{dafny_spec}\n```")

        if checker_report:
            parts.append(
                f"## Checker Verdict: {checker_report.verdict.value}\n"
                f"Issues found: {len(checker_report.issues)}"
            )
            for issue in checker_report.issues:
                parts.append(f"  - [{issue.severity.value}] {issue.description}")

        if verification_result:
            status = "PASSED" if verification_result.verified else "FAILED"
            parts.append(f"## Formal Verification: {status}")
            if verification_result.failing_assertions:
                for fa in verification_result.failing_assertions:
                    parts.append(f"  - {fa}")

        parts.append(f"## Applicable Standard\n{standard}")

        if policy_context:
            parts.append(
                f"## Retrieved Policy Context (from knowledge base)\n{policy_context}"
            )

        parts.append(
            "## Task\nDetermine compliance. Cite specific standard clauses for any violations."
        )

        return "\n\n".join(parts)

    async def run(self, **kwargs: Any) -> PolicyVerdict:  # type: ignore[override]
        return await super().run(**kwargs)  # type: ignore[return-value]
