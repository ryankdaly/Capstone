"""Execution sandboxing for generated code and verification tools.

Provides a safe execution context for running Dafny/Lean and generated test code.
Uses temp directories and process isolation.
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


@contextmanager
def sandbox_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for sandboxed execution.

    All files created within are cleaned up on exit.
    """
    with tempfile.TemporaryDirectory(prefix="hpema_sandbox_") as tmpdir:
        yield Path(tmpdir)
