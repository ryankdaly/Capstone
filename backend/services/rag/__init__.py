"""RAG (Retrieval-Augmented Generation) service for policy standards retrieval.

Main components:
- DocumentLoader: Load markdown documents from data/standards/
- Chunker: Split documents into semantic chunks
- Embedder: Generate embeddings using sentence-transformers
- ChromaDBStore: Persist and query vectors
- StandardsRetriever: Public orchestrator interface
"""

from backend.services.rag.config import RAGConfig
from backend.services.rag.retriever import StandardsRetriever

__all__ = ["StandardsRetriever", "RAGConfig"]
