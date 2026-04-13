"""DafnyArchitect agent — generates standalone Dafny formal specifications.

Runs after the Actor produces source code. Takes the implementation and
requirement as inputs, produces a verifiable Dafny method as output.
The Actor no longer handles Dafny; this agent owns it exclusively.
"""

from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel

from backend.api.schemas.agents import DafnySpec, VerificationResult
from backend.services.agents.base import BaseAgent


class DafnyArchitectAgent(BaseAgent):
    role = "dafny_architect"
    prompt_file = "dafny_architect.txt"
    max_tokens = 1000  # specs are compact; stay within vLLM context limits

    def _output_schema(self) -> Type[BaseModel]:
        return DafnySpec

    def _build_user_prompt(self, **kwargs: Any) -> str:
        source_code: str = kwargs["source_code"]
        requirement: str = kwargs["requirement"]
        language: str = kwargs.get("language", "C")
        prior_verification: VerificationResult | None = kwargs.get("verification_feedback")

        parts = [
            f"## Requirement\n{requirement}",
            f"## Source Code ({language})\n{source_code}",
        ]

        if prior_verification and not prior_verification.verified:
            parts.append("## Errors from Previous Dafny Attempt — fix ALL of these")
            for err in prior_verification.failing_assertions:
                parts.append(f"  - {err}")
            if prior_verification.solver_output:
                # Surface only the first 20 lines of raw solver output to avoid
                # flooding the context window with Z3 noise.
                raw_lines = prior_verification.solver_output.splitlines()[:20]
                parts.append("Solver output (first 20 lines):\n" + "\n".join(raw_lines))

        return "\n\n".join(parts)

    async def run(self, **kwargs: Any) -> DafnySpec:  # type: ignore[override]
        return await super().run(**kwargs)  # type: ignore[return-value]
