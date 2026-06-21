"""Checker agent — reviews code for correctness and generates test cases."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Type

from pydantic import BaseModel

from backend.api.schemas.agents import CheckerReport
from backend.services.agents.base import BaseAgent

if TYPE_CHECKING:
    from backend.services.verification.contract_extractor import DafnyContracts


class CheckerAgent(BaseAgent):
    role = "checker"
    prompt_file = "checker.txt"

    def _output_schema(self) -> Type[BaseModel]:
        return CheckerReport

    def _build_user_prompt(self, **kwargs: Any) -> str:
        source_code: str = kwargs["source_code"]
        language: str = kwargs.get("language", "Python")
        standard: str = kwargs.get("standard", "DO_178C")
        retry_hint: str | None = kwargs.get("retry_hint")
        dafny_contracts: "DafnyContracts | None" = kwargs.get("dafny_contracts")
        dafny_unverified_spec: str | None = kwargs.get("dafny_unverified_spec")
        dafny_solver_output: str | None = kwargs.get("dafny_solver_output")
        test_fix_hint: str | None = kwargs.get("test_fix_hint")

        parts = [
            f"## Source Code ({language})\n```\n{source_code}\n```",
            f"## Safety Standard\n{standard}",
            "## Task\nReview this code for correctness, safety issues, and undefined behavior. "
            "Generate test cases. Do NOT comment on Dafny specifications — focus only on the source code.",
        ]

        if dafny_unverified_spec and dafny_unverified_spec.strip():
            hint_lines = [
                "## Dafny Specification ⚠ (did not verify — use for inspiration only)",
                "A Dafny specification was generated for this code but **failed to verify**. "
                "The spec may contain logical errors or overly strong postconditions. "
                "Do NOT assert its ensures clauses as facts. Instead, absorb the intended "
                "contract — the preconditions and postconditions capture what the author "
                "meant — and use them to write more targeted, boundary-aware test cases.",
            ]
            if dafny_solver_output and dafny_solver_output.strip():
                truncated = dafny_solver_output[:800] + ("…" if len(dafny_solver_output) > 800 else "")
                hint_lines.append(
                    f"### Verification failure (Dafny output)\n```\n{truncated}\n```"
                )
            hint_lines.append(f"### Unverified spec\n```dafny\n{dafny_unverified_spec}\n```")
            parts.append("\n".join(hint_lines))

        if dafny_contracts and dafny_contracts.has_contracts:
            contract_lines = [
                f"## Formal Contracts (from verified Dafny spec for `{dafny_contracts.method_name}`)",
                "These postconditions were formally verified by Dafny. "
                "Generate one named contract test per ensures clause "
                "(test_contract_P1, test_contract_P2, …).",
            ]
            if dafny_contracts.requires:
                contract_lines.append("### Preconditions (restrict test inputs to this domain)")
                for clause in dafny_contracts.requires:
                    contract_lines.append(f"  requires {clause}")
            if dafny_contracts.ensures:
                contract_lines.append("### Postconditions (each must be asserted in a test)")
                for clause in dafny_contracts.ensures:
                    contract_lines.append(f"  ensures {clause}")
            parts.append("\n".join(contract_lines))

        if test_fix_hint:
            parts.append(
                f"## Test Syntax Fix Required\n"
                f"Your previous test cases failed to parse — pytest could not collect them:\n\n"
                f"```\n{test_fix_hint[:600]}\n```\n\n"
                f"Rewrite ALL test cases from scratch. Common causes:\n"
                f"- Empty `try:` block with no body (add `pass` or a real statement)\n"
                f"- Missing indentation after `def`, `if`, `for`, `try`, `with`\n"
                f"- Incomplete function body (every `def test_X():` needs ≥1 statement)\n"
                f"Output ONLY the JSON object — no prose, no markdown, no <think> blocks."
            )

        if retry_hint:
            parts.append(
                f"## Retry Notice\n"
                f"Your previous attempt failed: {retry_hint[:400]}\n"
                f"Output ONLY the JSON object — no prose, no markdown, no <think> blocks."
            )

        return "\n\n".join(p for p in parts if p)

    async def run(self, **kwargs: Any) -> CheckerReport:  # type: ignore[override]
        return await super().run(**kwargs)  # type: ignore[return-value]
