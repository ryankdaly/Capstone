"""SQLite database connection management.

Used by the audit logger for cross-run queries. Lightweight — no ORM needed
for the current scope. If we need more complex queries later, we can add
SQLAlchemy.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

DEFAULT_DB_PATH = Path("data/audit.db")


@contextmanager
def get_connection(
    db_path: Path = DEFAULT_DB_PATH,
) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for SQLite connections."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
