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
        chunk_size: int = 1200,
        chunk_overlap: int = 200
    ) -> None:
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._collection = None
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

        if CHROMADB_AVAILABLE:
            Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=self._persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            try:
                self._client.heartbeat()
                logger.info("ChromaDB initialized at %s", self._persist_dir)
            except Exception as e:
                logger.warning("ChromaDB heartbeat failed: %s", e)
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
    
    def _chunk_text(self, text: str) -> list[str]:
        """Simple overlapping character chunker"""
        text = text.strip()
        if not text:
            return []

        chunks: list[str] = []
        start = 0
        n = len(text)

        while start < n:
            end = min(start + self._chunk_size, n)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= n:
                break

            start = max(end - self._chunk_overlap, start + 1)

        return chunks



    def ingest_directory(self, standards_dir: str | None = None) -> int:
        """Ingest all .txt and .md files from the standards directory.

        Returns the number of chunks from documents ingested.
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

            # Extract standard name from parent directory or filename
            standard = file_path.parent.name if file_path.parent != dir_path else "general"
            rel_path = file_path.relative_to(dir_path)
            chunks = self._chunk_text(text)
            for chunk_index, chunk in enumerate(chunks): 
                chunk_id = f"{rel_path.as_posix()}::chunk_{chunk_index}"
                collection.upsert(
                    ids=[chunk_id],
                    documents=[chunk],
                    metadatas=[{
                        "source": str(file_path),
                        "standard": standard,
                        "filename": file_path.name,
                        "chunk_index": chunk_index,
                        "chunk_count": len(chunks),
                    }],
                )
                count += 1

        logger.info("Ingested %d chunks from %s", count, dir_path)
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
            chunk_index = meta.get("chunk_index", "?")
            sections.append(f"[Source: {source}][Chunk: {chunk_index}]\n{doc}")

        return "\n\n---\n\n".join(sections)
