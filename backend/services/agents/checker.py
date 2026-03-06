"""Checker agent — reviews code for correctness and generates test cases."""

from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel

from backend.api.schemas.agents import CheckerReport
from backend.services.agents.base import BaseAgent


class CheckerAgent(BaseAgent):
    role = "checker"
    prompt_file = "checker.txt"

    def _output_schema(self) -> Type[BaseModel]:
        return CheckerReport

    def _build_user_prompt(self, **kwargs: Any) -> str:
        source_code: str = kwargs["source_code"]
        dafny_spec: str = kwargs.get("dafny_spec", "")
        language: str = kwargs.get("language", "C")
        standard: str = kwargs.get("standard", "DO_178C")

        parts = [
            f"## Source Code ({language})\n```\n{source_code}\n```",
            f"## Dafny Specification\n```dafny\n{dafny_spec}\n```" if dafny_spec else "",
            f"## Safety Standard\n{standard}",
            "## Task\nReview this code. Find bugs, safety issues, and spec gaps. Generate test cases.",
        ]

        return "\n\n".join(p for p in parts if p)

    async def run(self, **kwargs: Any) -> CheckerReport:  # type: ignore[override]
        return await super().run(**kwargs)  # type: ignore[return-value]
