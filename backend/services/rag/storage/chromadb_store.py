"""ChromaDB-based vector storage for RAG."""

from __future__ import annotations

import logging
from typing import List, Optional

from backend.services.rag.chunking.chunker import Chunk

logger = logging.getLogger(__name__)


class ChromaDBStore:
    """Manages ChromaDB collections for standards storage and retrieval."""

    def __init__(self, persist_dir: str | None = None) -> None:
        """Initialize ChromaDB store with optional persistence.
        
        Args:
            persist_dir: Directory for persisting ChromaDB data.
                        If None, uses ephemeral in-memory storage.
        """
        self.persist_dir = persist_dir
        self._client = None
        self._collection = None
        self._lazy_init()

    def _lazy_init(self) -> None:
        """Lazily initialize ChromaDB client and collection."""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings
                
                if self.persist_dir:
                    logger.info(f"Initializing ChromaDB with persistence: {self.persist_dir}")
                    settings = Settings(
                        chroma_db_impl="duckdb+parquet",
                        persist_directory=self.persist_dir,
                        anonymized_telemetry=False,
                    )
                    self._client = chromadb.Client(settings)
                else:
                    logger.info("Initializing ChromaDB with ephemeral in-memory storage")
                    self._client = chromadb.Client()
                
                # Get or create a collection for policy standards
                self._collection = self._client.get_or_create_collection(
                    name="policy_standards",
                    metadata={"hnsw:space": "cosine"},  # Use cosine similarity
                )
                
            except ImportError:
                raise ImportError(
                    "chromadb not installed. "
                    "Install with: pip install chromadb"
                )

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> int:
        """Add chunks with their embeddings to the collection.
        
        Args:
            chunks: List of Chunk objects with text and metadata
            embeddings: List of embedding vectors (from Embedder)
            
        Returns:
            Number of chunks successfully added
            
        Raises:
            ValueError: If chunk and embedding counts don't match
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk count ({len(chunks)}) must equal "
                f"embedding count ({len(embeddings)})"
            )
        
        if not chunks:
            logger.warning("No chunks to add")
            return 0
        
        self._lazy_init()
        
        # Prepare data for ChromaDB
        ids = []
        texts = []
        metadatas = []
        embeddings_list = []
        
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Generate unique ID
            chunk_id = f"chunk_{id(chunk)}_{idx}"
            ids.append(chunk_id)
            texts.append(chunk.text)
            metadatas.append(chunk.metadata)
            embeddings_list.append(embedding)
        
        try:
            self._collection.add(
                ids=ids,
                embeddings=embeddings_list,
                documents=texts,
                metadatas=metadatas,
            )
            logger.info(f"Added {len(chunks)} chunks to ChromaDB collection")
            return len(chunks)
        except Exception as e:
            logger.error(f"Failed to add chunks: {e}")
            raise

    def query(
        self,
        query_embedding: List[float],
        standard: str | None = None,
        n_results: int = 5,
    ) -> List[tuple]:
        """Query the collection for similar chunks.
        
        Args:
            query_embedding: Embedding vector of the query
            standard: Optional filter by standard name (DO_178C, etc.)
            n_results: Number of results to return
            
        Returns:
            List of tuples: (chunk_text, metadata, distance)
        """
        self._lazy_init()
        
        # Build where filter if standard is specified
        where_condition = None
        if standard:
            where_condition = {"standard": {"$eq": standard}}
        
        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                where=where_condition,
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
            
            # Flatten results (query returns lists because it supports batch queries)
            output = []
            if results["documents"] and results["documents"][0]:
                for doc, metadata, distance in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    output.append((doc, metadata, distance))
            
            return output
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise

    def clear(self) -> None:
        """Clear all documents from the collection."""
        self._lazy_init()
        try:
            # Delete the collection and recreate it
            self._client.delete_collection(name="policy_standards")
            self._collection = self._client.get_or_create_collection(
                name="policy_standards",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("Cleared ChromaDB collection")
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise

    def persist(self) -> None:
        """Persist the collection to disk (only if initialized with persist_dir)."""
        if self.persist_dir:
            self._lazy_init()
            try:
                self._client.persist()
                logger.info("Persisted ChromaDB collection to disk")
            except Exception as e:
                logger.error(f"Failed to persist collection: {e}")
                raise
