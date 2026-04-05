"""Public interface for RAG retrieval — StandardsRetriever.

This is the main API that agents and the pipeline use to query standards.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from backend.services.rag.config import RAGConfig
from backend.services.rag.chunking.chunker import Chunker
from backend.services.rag.embeddings.embedder import Embedder
from backend.services.rag.loaders.document_loader import DocumentLoader
from backend.services.rag.storage.chromadb_store import ChromaDBStore

logger = logging.getLogger(__name__)


class StandardsRetriever:
    """Orchestrator for document loading, chunking, embedding, and retrieval.
    
    Public interface for retrieving relevant standards based on queries.
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        config: RAGConfig | None = None,
    ) -> None:
        """Initialize the retriever.
        
        Args:
            persist_dir: Optional directory for persisting ChromaDB data.
                        If None, uses ephemeral storage.
            config: Optional RAGConfig. If None, uses defaults.
        """
        self.persist_dir = persist_dir
        self.config = config or RAGConfig()
        
        # Lazy-initialized components
        self._loader: DocumentLoader | None = None
        self._chunker: Chunker | None = None
        self._embedder: Embedder | None = None
        self._store: ChromaDBStore | None = None

    def _ensure_components(self) -> None:
        """Lazily initialize all components on first use."""
        if self._loader is None:
            self._loader = DocumentLoader()
        if self._chunker is None:
            self._chunker = Chunker(self.config)
        if self._embedder is None:
            self._embedder = Embedder(self.config)
        if self._store is None:
            self._store = ChromaDBStore(persist_dir=self.persist_dir)

    def ingest_directory(self, path: str) -> int:
        """Load, chunk, embed, and store documents from a directory.
        
        Workflow:
        1. Load all .md files recursively from path
        2. Chunk documents by markdown sections
        3. Generate embeddings for each chunk
        4. Store in ChromaDB with metadata
        
        Args:
            path: Root directory containing standards (e.g., data/standards/)
            
        Returns:
            Total number of chunks ingested
        """
        self._ensure_components()
        
        try:
            logger.info(f"Starting ingestion from: {path}")
            
            # Phase 1: Load documents
            documents = self._loader.load_from_directory(path)
            if not documents:
                logger.warning(f"No documents found in {path}")
                return 0
            
            logger.info(f"Loaded {len(documents)} documents")
            
            # Phase 2: Chunk
            chunks = self._chunker.chunk(documents)
            logger.info(f"Created {len(chunks)} chunks")
            
            # Phase 3: Embed
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = self._embedder.embed(chunk_texts)
            # Convert numpy arrays to lists for ChromaDB
            embeddings_list = [emb.tolist() for emb in embeddings]
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            # Phase 4: Store
            added_count = self._store.add_chunks(chunks, embeddings_list)
            
            # Persist if configured
            if self.persist_dir:
                self._store.persist()
            
            logger.info(f"Successfully ingested {added_count} chunks")
            return added_count
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            raise

    def retrieve(
        self,
        query: str,
        standard: str | None = None,
        n_results: int | None = None,
    ) -> str:
        """Query standards and return formatted results.
        
        Workflow:
        1. Embed the query
        2. Search ChromaDB (optionally filtered by standard)
        3. Format and return results as a string
        
        Args:
            query: Natural language query (e.g., "memory allocation")
            standard: Optional standard name to filter (e.g., "DO_178C")
            n_results: Number of results to return. Defaults to config.default_n_results.
            
        Returns:
            Formatted string of results with source metadata
        """
        self._ensure_components()
        
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        n_results = n_results or self.config.default_n_results
        
        try:
            logger.info(f"Retrieving with query: '{query}' (standard={standard}, n_results={n_results})")
            
            # Phase 1: Embed query
            query_embedding = self._embedder.embed_single(query)
            query_embedding_list = query_embedding.tolist()
            logger.debug(f"Query embedding dimension: {len(query_embedding_list)}")
            
            # Phase 2: Query ChromaDB
            results = self._store.query(
                query_embedding=query_embedding_list,
                standard=standard,
                n_results=n_results,
            )
            
            logger.info(f"Retrieved {len(results)} results")
            
            # Phase 3: Format results
            if not results:
                return self._format_empty_results()
            
            return self._format_results(results)
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise

    def retrieve_raw(
        self,
        query: str,
        standard: str | None = None,
        n_results: int | None = None,
    ) -> List[Tuple[str, dict, float]]:
        """Query standards and return raw results (text, metadata, distance).
        
        Useful when the caller wants to format results themselves.
        
        Args:
            query: Natural language query
            standard: Optional filter by standard
            n_results: Number of results to return
            
        Returns:
            List of (chunk_text, metadata, distance) tuples
        """
        self._ensure_components()
        
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        n_results = n_results or self.config.default_n_results
        
        # Embed and query
        query_embedding = self._embedder.embed_single(query)
        results = self._store.query(
            query_embedding=query_embedding.tolist(),
            standard=standard,
            n_results=n_results,
        )
        
        return results

    def _format_empty_results(self) -> str:
        """Format empty results response."""
        return "No relevant standards found for your query."

    def _format_results(self, results: List[Tuple[str, dict, float]]) -> str:
        """Format retrieval results as a readable string.
        
        Args:
            results: List of (chunk_text, metadata, distance) tuples
            
        Returns:
            Formatted string with clear headers and structure
        """
        output_lines = [f"Found {len(results)} relevant standards:\n"]
        
        for idx, (chunk_text, metadata, distance) in enumerate(results, 1):
            standard = metadata.get("standard", "Unknown")
            file = metadata.get("file", "unknown.md")
            
            # Distance is similarity distance; lower = more similar
            confidence = 1 - (distance / 2)  # Rough conversion to 0-1
            
            output_lines.append(f"--- Result {idx} ---")
            output_lines.append(f"Standard: {standard}")
            output_lines.append(f"File: {file}")
            output_lines.append(f"Relevance: {confidence:.1%}")
            output_lines.append(f"Content:\n{chunk_text}")
            output_lines.append("")
        
        return "\n".join(output_lines)
