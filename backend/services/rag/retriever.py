"""RAG retriever — vector search over safety standards using ChromaDB.

Boeing swaps standards by dropping documents into the standards_dir
configured in hpema_config.yaml. No code changes needed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from backend.config import settings

logger = logging.getLogger(__name__)

# ChromaDB + sentence-transformers are optional dependencies.
# Import lazily so the rest of the system works without them installed.
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


COLLECTION_NAME = "safety_standards"


class StandardsRetriever:
    """Retrieves relevant safety standard clauses for the Policy Agent."""

    def __init__(
        self,
        persist_dir: str = "data/chromadb",
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._collection = None

        if CHROMADB_AVAILABLE:
            self._client = chromadb.Client(
                ChromaSettings(
                    persist_directory=persist_dir,
                    anonymized_telemetry=False,
                )
            )
        else:
            self._client = None
            logger.warning(
                "ChromaDB not installed. RAG retrieval disabled. "
                "Install with: pip install chromadb sentence-transformers"
            )

    def _get_collection(self):
        """Get or create the collection (lazy init)."""
        if self._collection is None and self._client is not None:
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"description": "Safety standards knowledge base"},
            )
        return self._collection

    def ingest_directory(self, standards_dir: str | None = None) -> int:
        """Ingest all .txt and .md files from the standards directory.

        Returns the number of documents ingested.
        """
        collection = self._get_collection()
        if collection is None:
            logger.warning("ChromaDB not available — skipping ingestion")
            return 0

        dir_path = Path(standards_dir or settings.policies.standards_dir)
        if not dir_path.exists():
            logger.warning("Standards directory not found: %s", dir_path)
            return 0

        count = 0
        for file_path in sorted(dir_path.glob("**/*")):
            if file_path.suffix not in (".txt", ".md"):
                continue

            text = file_path.read_text(errors="replace").strip()
            if not text:
                continue

            # Use relative path as document ID for deduplication
            doc_id = str(file_path.relative_to(dir_path))

            # Extract standard name from parent directory or filename
            standard = file_path.parent.name if file_path.parent != dir_path else "general"

            collection.upsert(
                ids=[doc_id],
                documents=[text],
                metadatas=[{
                    "source": str(file_path),
                    "standard": standard,
                    "filename": file_path.name,
                }],
            )
            count += 1

        logger.info("Ingested %d documents from %s", count, dir_path)
        return count

    def retrieve(
        self,
        query: str,
        standard: Optional[str] = None,
        n_results: int = 5,
    ) -> str:
        """Retrieve relevant standard clauses for a query.

        Returns concatenated text of the top-N most relevant documents.
        """
        collection = self._get_collection()
        if collection is None or collection.count() == 0:
            return ""

        where_filter = {"standard": standard} if standard else None

        try:
            results = collection.query(
                query_texts=[query],
                n_results=min(n_results, collection.count()),
                where=where_filter,
            )
        except Exception as e:
            logger.warning("RAG retrieval failed: %s", e)
            return ""

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        if not documents:
            return ""

        # Format retrieved context with source attribution
        sections: list[str] = []
        for doc, meta in zip(documents, metadatas):
            source = meta.get("filename", "unknown")
            sections.append(f"[Source: {source}]\n{doc}")

        return "\n\n---\n\n".join(sections)
