"""Knowledge graph / standards ingestion utilities.

Handles loading safety standard documents into the RAG knowledge base.
For the capstone, we use synthetic documents derived from public standards.
Boeing replaces these with their actual SDP documents.
"""

from __future__ import annotations

import logging
from pathlib import Path

from backend.config import settings
from backend.services.rag.retriever import StandardsRetriever

logger = logging.getLogger(__name__)


def ensure_standards_dir() -> Path:
    """Create the standards directory structure if it doesn't exist."""
    base = Path(settings.policies.standards_dir)
    for subdir in ["DO_178C", "MISRA_C", "NASA", "Boeing_SDP"]:
        (base / subdir).mkdir(parents=True, exist_ok=True)
    return base


def ingest_standards(retriever: StandardsRetriever | None = None) -> int:
    """Ingest all standards documents into the vector store.

    Call this at startup or when standards are updated.
    Returns the number of documents ingested.
    """
    ensure_standards_dir()
    retriever = retriever or StandardsRetriever()
    count = retriever.ingest_directory()
    logger.info("Standards ingestion complete: %d documents", count)
    return count
