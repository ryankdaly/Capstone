"""Dafny formal verification runner.

Writes Dafny specs to temp files, runs `dafny verify` as an async subprocess,
parses the output for pass/fail and failing assertions.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
import time
from pathlib import Path

from backend.api.schemas.agents import VerificationResult
from backend.config import settings

logger = logging.getLogger(__name__)


def _resolve_binary(raw: str) -> str:
    """Resolve a binary path that may be a literal path, $VAR, or bare VAR name.

    YAML doesn't expand env vars, so config values like 'LOCAL_DAFNY_INSTALL'
    or '$LOCAL_DAFNY_INSTALL' need to be resolved here.
    """
    # 1. Expand $VAR / ${VAR} syntax
    expanded = os.path.expandvars(os.path.expanduser(raw))
    if expanded != raw:
        return expanded
    # 2. Bare env var name (no path separators, all caps) — look it up directly
    if os.sep not in raw and os.environ.get(raw):
        return os.path.expanduser(os.environ[raw])
    return expanded


class DafnyRunner:
    """Async wrapper around the Dafny CLI."""

    def __init__(
        self,
        binary_path: str | None = None,
        timeout: int | None = None,
        solver_path: str | None = None,
    ) -> None:
        self._binary = _resolve_binary(binary_path or settings.verification.binary_path)
        self._timeout = timeout or settings.verification.timeout_seconds
        self._solver_path = solver_path or settings.verification.solver_path

    async def verify(self, dafny_source: str) -> VerificationResult:
        """Write the Dafny source to a temp file and run verification."""
        if not dafny_source.strip():
            return VerificationResult(
                verified=False,
                prover="dafny",
                solver_output="No Dafny specification provided.",
            )

        precheck_errors = self._syntax_precheck(dafny_source)
        if precheck_errors:
            logger.warning("Dafny pre-check failed (%d errors) — skipping subprocess", len(precheck_errors))
            return VerificationResult(
                verified=False,
                prover="dafny",
                solver_output="Pre-verification structural check failed. Dafny subprocess not invoked.",
                failing_assertions=precheck_errors,
            )

        start = time.monotonic()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".dfy", delete=False
        ) as f:
            f.write(dafny_source)
            spec_path = Path(f.name)

        try:
            result = await self._run_dafny(spec_path)
        finally:
            spec_path.unlink(missing_ok=True)

        result.execution_time_seconds = round(time.monotonic() - start, 2)
        return result

    async def _run_dafny(self, spec_path: Path) -> VerificationResult:
        cmd = [self._binary, "verify", str(spec_path)]
        if self._solver_path:
            cmd += ["--solver-path", self._solver_path]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=self._timeout
            )
        except FileNotFoundError:
            logger.warning("Dafny binary not found at %s", self._binary)
            logger.warning("Spec Path is %s", spec_path)
            return VerificationResult(
                verified=False,
                prover="dafny",
                solver_output=f"Dafny binary not found: {self._binary}. Install Dafny to enable formal verification.",
            )
        except asyncio.TimeoutError:
            proc.kill()
            return VerificationResult(
                verified=False,
                prover="dafny",
                solver_output=f"Verification timed out after {self._timeout}s",
            )

        stdout = stdout_bytes.decode(errors="replace")
        stderr = stderr_bytes.decode(errors="replace")
        combined = f"{stdout}\n{stderr}".strip()

        verified = proc.returncode == 0
        failing = self._parse_failing_assertions(combined)

        return VerificationResult(
            verified=verified,
            prover="dafny",
            solver_output=combined,
            failing_assertions=failing,
        )

    @staticmethod
    def _syntax_precheck(source: str) -> list[str]:
        """Catch structural LLM mistakes before invoking the Dafny subprocess.

        Returns a list of human-readable error strings. When non-empty the
        subprocess is skipped entirely — the errors feed directly into the
        DafnyArchitect retry via failing_assertions.
        """
        errors: list[str] = []

        # Must have at least one method or function declaration
        if not re.search(r"\b(method|function)\b", source):
            errors.append(
                "Spec must contain at least one 'method' or 'function' declaration"
            )

        # Must have at least one postcondition — otherwise nothing is verified
        if not re.search(r"\bensures\b", source):
            errors.append(
                "Spec must have at least one 'ensures' postcondition — "
                "without it Dafny cannot prove anything about the code"
            )

        # External library references — Dafny has no stdlib
        extern_calls = re.findall(r"\b(?:math|Math|std|System|os|numpy|scipy)\.\w+", source)
        if extern_calls:
            errors.append(
                f"Dafny has no external libraries. Found: {extern_calls}. "
                "Express all logic inline — there is no math.exp, math.sqrt, etc."
            )

        # {:extern} — spec must be self-contained
        if re.search(r"\{:extern\}", source):
            errors.append("Spec must be self-contained — remove {:extern} attributes")

        # C / Python types that are not valid Dafny
        bad_types = re.findall(
            r"\b(int32_t|uint8_t|uint32_t|int64_t|size_t|float|double)\b", source
        )
        if bad_types:
            errors.append(
                f"Found non-Dafny types: {sorted(set(bad_types))}. "
                "Use Dafny types: int, bool, real, array<int>, seq<T>"
            )

        return errors

    @staticmethod
    def _parse_failing_assertions(output: str) -> list[str]:
        """Extract failing assertion messages from Dafny output.

        Catches Dafny 3.x ("might not hold") and 4.x ("could not be proved")
        phrasing, plus general Error/Warning lines from the verifier.
        """
        failures: list[str] = []
        seen: set[str] = set()
        for line in output.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            # Match any Dafny error/warning line (file.dfy(line,col): Error: ...)
            if re.search(r"Error:", stripped, re.IGNORECASE):
                if stripped not in seen:
                    failures.append(stripped)
                    seen.add(stripped)
            elif re.search(r"Warning:", stripped, re.IGNORECASE) and "deprecated" not in stripped.lower():
                if stripped not in seen:
                    failures.append(stripped)
                    seen.add(stripped)
        return failures
