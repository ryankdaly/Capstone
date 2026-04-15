"""Public interface for RAG retrieval — StandardsRetriever.

Architecture:
- ChromaDBStore handles embeddings internally via DefaultEmbeddingFunction
  (all-MiniLM-L6-v2 via ONNX, bundled with chromadb) — no sentence-transformers
  dependency required.
- retrieve() is async via asyncio.to_thread so it never blocks the event loop.
- If auto_ingest_path is provided and the collection is empty on first use,
  documents are ingested automatically before the first query.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import List, Tuple

from backend.services.rag.config import RAGConfig
from backend.services.rag.chunking.chunker import Chunker
from backend.services.rag.loaders.document_loader import DocumentLoader
from backend.services.rag.storage.chromadb_store import ChromaDBStore

logger = logging.getLogger(__name__)


class StandardsRetriever:
    """Orchestrator for document ingestion and standards retrieval.

    Parameters
    ----------
    persist_dir:
        Directory for persisting ChromaDB data across restarts.
        Pass None for ephemeral in-memory storage (tests).
    config:
        RAGConfig (chunk size, overlap, n_results). Defaults if None.
    auto_ingest_path:
        If set, the retriever will check whether the collection is empty on
        first use and — if so — ingest all .md files from this directory.
        Idempotent: skipped if the collection already contains documents.
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        config: RAGConfig | None = None,
        auto_ingest_path: str | None = None,
    ) -> None:
        self.config = config or RAGConfig()
        self._store = ChromaDBStore(persist_dir=persist_dir)
        self._loader = DocumentLoader()
        self._chunker = Chunker(self.config)
        self._auto_ingest_path = auto_ingest_path
        self._ingestion_checked = False  # guard: check at most once per process

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_directory(self, path: str) -> int:
        """Load, chunk, and store all .md files from a directory.

        Returns the number of chunks upserted.
        """
        documents = self._loader.load_from_directory(path)
        if not documents:
            logger.warning("No documents found in %s", path)
            return 0

        chunks = self._chunker.chunk(documents)
        added = self._store.add_chunks(chunks)
        logger.info("Ingested %d chunks from %s", added, path)
        return added

    def _maybe_auto_ingest(self) -> None:
        """Ingest on first call if collection is empty and path is configured."""
        if self._ingestion_checked:
            return
        self._ingestion_checked = True  # mark before work to avoid re-entry

        if not self._auto_ingest_path:
            return

        path = Path(self._auto_ingest_path)
        if not path.exists():
            logger.warning("auto_ingest_path does not exist: %s", path)
            return

        try:
            count = self._store.count()
            if count > 0:
                logger.info(
                    "Collection already has %d chunks — skipping auto-ingest", count
                )
                return
            logger.info("Empty collection — auto-ingesting from %s", path)
            ingested = self.ingest_directory(str(path))
            logger.info("Auto-ingest complete: %d chunks", ingested)
        except Exception as exc:
            logger.warning("Auto-ingest failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _sync_retrieve(
        self, query: str, standard: str | None, n_results: int
    ) -> str:
        """Synchronous retrieve — called from a thread pool by retrieve()."""
        self._maybe_auto_ingest()

        results = self._store.query(
            query_text=query,
            standard=standard,
            n_results=n_results,
        )

        if not results:
            return "No relevant standards found for your query."
        return self._format_results(results)

    async def retrieve(
        self,
        query: str,
        standard: str | None = None,
        n_results: int | None = None,
    ) -> str:
        """Query standards and return formatted results (async, non-blocking).

        The underlying ChromaDB query and embedding operations run in a thread
        pool so they do not block the asyncio event loop.

        Returns a human-readable string of retrieved policy context, or an
        empty string if the query is blank.
        """
        if not query or not query.strip():
            return ""
        n = n_results or self.config.default_n_results
        return await asyncio.to_thread(self._sync_retrieve, query, standard, n)

    def retrieve_raw(
        self,
        query: str,
        standard: str | None = None,
        n_results: int | None = None,
    ) -> list[tuple[str, dict, float]]:
        """Synchronous raw retrieval for tooling / tests.

        Returns list of (chunk_text, metadata, cosine_distance) tuples.
        """
        self._maybe_auto_ingest()
        return self._store.query(
            query_text=query,
            standard=standard,
            n_results=n_results or self.config.default_n_results,
        )

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_results(results: list[tuple[str, dict, float]]) -> str:
        lines = [f"Found {len(results)} relevant standards:\n"]
        for idx, (text, meta, dist) in enumerate(results, 1):
            # Cosine distance in [0, 2]; 0 = identical. Convert to 0–100% relevance.
            relevance = max(0.0, 1.0 - dist / 2.0)
            lines += [
                f"--- Result {idx} ---",
                f"Standard: {meta.get('standard', 'Unknown')}",
                f"File: {meta.get('file', 'unknown.md')}",
                f"Relevance: {relevance:.0%}",
                f"Content:\n{text}",
                "",
            ]
        return "\n".join(lines)
