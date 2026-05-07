#!/usr/bin/env python3
"""Count DO-178C rules loaded in ChromaDB.

The PA formula uses R = total rules as the denominator:
    PA = max(0, 1 - sum(weights) / (2 * R))

This script queries ChromaDB to measure R. Fix the constant in
evaluate.py and aggregate.py before running data collection.

Usage:
    python scripts/harness/count_rules.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import resolve_data_path, settings


def main() -> None:
    try:
        import chromadb
    except ImportError:
        print("ERROR: chromadb not installed. Run: pip install chromadb")
        sys.exit(1)

    # Try project-local data/chromadb first (editable install with data in repo)
    project_chromadb = PROJECT_ROOT / "data" / "chromadb"
    if project_chromadb.exists():
        chromadb_dir = str(project_chromadb)
    else:
        chromadb_dir = str(resolve_data_path(settings.policies.chromadb_dir, "chromadb"))

    standards_dir = str(resolve_data_path(settings.policies.standards_dir, "standards"))

    print(f"ChromaDB dir:  {chromadb_dir}")
    print(f"Standards dir: {standards_dir}")

    client = chromadb.PersistentClient(path=chromadb_dir)
    collections = client.list_collections()

    if not collections:
        print("\nNo collections found. Standards may not be ingested yet.")
        print("Start HPEMA and run a prompt to trigger auto-ingest, then re-run this script.")
        return

    total_chunks = 0
    for col in collections:
        collection = client.get_collection(col.name)
        n = collection.count()
        print(f"  collection={col.name!r:40s} chunks={n}")
        total_chunks += n

    print(f"\nTotal chunks (R): {total_chunks}")
    print()
    print("Update R_RULES in:")
    print("  scripts/harness/evaluate.py   (line ~25: R_RULES = ...)")
    print("  scripts/harness/aggregate.py  (line ~30: R_RULES = ...)")


if __name__ == "__main__":
    main()
