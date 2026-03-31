"""Dafny formal verification runner.

Writes Dafny specs to temp files, runs `dafny verify` as an async subprocess,
parses the output for pass/fail and failing assertions.
"""

from __future__ import annotations

import asyncio
import logging
import re
import tempfile
import time
from pathlib import Path

from backend.api.schemas.agents import VerificationResult
from backend.config import settings

logger = logging.getLogger(__name__)


class DafnyRunner:
    """Async wrapper around the Dafny CLI."""

    def __init__(
        self,
        binary_path: str | None = None,
        timeout: int | None = None,
        solver_path: str | None = None,
    ) -> None:
        self._binary = binary_path or settings.verification.binary_path
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
    def _parse_failing_assertions(output: str) -> list[str]:
        """Extract failing assertion messages from Dafny output."""
        # Generalized patterns to catch both Dafny 3.x and 4.x phrasing.
        # This list is non-exhaustive and should be expanded as Dafny updates, or custom error returns are written.
        patterns = [
            r"Error:.*assertion.*",
            r"Error:.*postcondition.*",
            r"Error:.*precondition.*",
            r"Error:.*invariant.*",
            r"Error:.*decreases.*",
            r"Error:.*termination.*",
            r"Error:.*could not be proved.*",
            r"Error:.*might not hold.*",
        ]
        failures: list[str] = []
        for line in output.splitlines():
            for pattern in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    failures.append(line.strip())
                    break
        return failures
