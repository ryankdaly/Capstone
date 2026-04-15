"""ChromaDB-based vector storage for RAG.

Uses ChromaDB 0.4+ API (PersistentClient / EphemeralClient) and the built-in
DefaultEmbeddingFunction (all-MiniLM-L6-v2 via ONNX) so that no external
sentence-transformers installation is required.
"""

from __future__ import annotations

import hashlib
import logging
from typing import List

from backend.services.rag.chunking.chunker import Chunk

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "policy_standards"


class ChromaDBStore:
    """Manages a ChromaDB collection for standards storage and retrieval."""

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str = _COLLECTION_NAME,
    ) -> None:
        self.persist_dir = persist_dir
        self._collection_name = collection_name
        self._client = None
        self._collection = None

    def _lazy_init(self) -> None:
        """Initialize client and collection on first use."""
        if self._client is not None:
            return
        try:
            import chromadb
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

            if self.persist_dir:
                logger.info("ChromaDB PersistentClient: %s", self.persist_dir)
                self._client = chromadb.PersistentClient(path=self.persist_dir)
            else:
                logger.info("ChromaDB EphemeralClient (in-memory)")
                self._client = chromadb.EphemeralClient()

            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                embedding_function=DefaultEmbeddingFunction(),
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "ChromaDB collection '%s' ready (%d docs)",
                _COLLECTION_NAME,
                self._collection.count(),
            )
        except ImportError:
            raise ImportError(
                "chromadb not installed. Install with: pip install chromadb"
            )

    def count(self) -> int:
        """Return number of documents in the collection."""
        self._lazy_init()
        return self._collection.count()

    def add_chunks(self, chunks: List[Chunk]) -> int:
        """Upsert chunks into the collection using deterministic IDs.

        Uses SHA-256 of chunk text for deduplication — re-ingesting the same
        document is idempotent.

        Returns number of chunks upserted.
        """
        if not chunks:
            logger.warning("No chunks to add")
            return 0

        self._lazy_init()

        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict] = []
        seen: set[str] = set()

        for chunk in chunks:
            chunk_id = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()[:32]
            # Handle (unlikely) hash collisions within this batch
            if chunk_id in seen:
                chunk_id = f"{chunk_id}_{len(seen)}"
            seen.add(chunk_id)
            ids.append(chunk_id)
            texts.append(chunk.text)
            # ChromaDB requires metadata values to be str/int/float/bool
            safe_meta = {
                k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                for k, v in chunk.metadata.items()
            }
            metadatas.append(safe_meta)

        # Upsert: safe to call multiple times for same content
        self._collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info("Upserted %d chunks into '%s'", len(chunks), _COLLECTION_NAME)
        return len(chunks)

    def query(
        self,
        query_text: str,
        standard: str | None = None,
        n_results: int = 5,
    ) -> list[tuple[str, dict, float]]:
        """Query the collection using ChromaDB's built-in embedding function.

        Returns list of (chunk_text, metadata, cosine_distance) tuples,
        ordered by ascending distance (most relevant first).
        """
        self._lazy_init()

        collection_count = self._collection.count()
        if collection_count == 0:
            logger.warning("Collection is empty — no results possible")
            return []

        n_results = min(n_results, collection_count)
        where = {"standard": {"$eq": standard}} if standard else None

        try:
            results = self._collection.query(
                query_texts=[query_text],
                where=where,
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.error("ChromaDB query failed: %s", exc)
            raise

        output: list[tuple[str, dict, float]] = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                output.append((doc, meta, dist))

        return output

    def clear(self) -> None:
        """Delete the collection and reset to a fresh empty state.

        Resets the client reference so _lazy_init creates a completely fresh
        instance — this guarantees clean state even for EphemeralClient.
        """
        if self._client is not None:
            try:
                self._client.delete_collection(name=self._collection_name)
                logger.info("Deleted collection '%s'", self._collection_name)
            except Exception as exc:
                logger.warning("delete_collection failed (non-fatal): %s", exc)
        self._client = None
        self._collection = None
        self._lazy_init()
        logger.info("Recreated empty collection '%s'", self._collection_name)
