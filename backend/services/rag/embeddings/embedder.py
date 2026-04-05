"""Embedding generation using sentence-transformers."""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from backend.services.rag.config import RAGConfig

logger = logging.getLogger(__name__)


class Embedder:
    """Generate embeddings using sentence-transformers."""

    def __init__(self, config: RAGConfig) -> None:
        """Initialize embedder with model.
        
        Args:
            config: RAG configuration with embedding_model
        """
        self.config = config
        self._model = None
        self._lazy_init()

    def _lazy_init(self) -> None:
        """Lazily load the embedding model on first use."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.debug(f"Loading embedding model: {self.config.embedding_model}")
                self._model = SentenceTransformer(self.config.embedding_model)
                logger.info(f"Loaded embedding model: {self.config.embedding_model}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )

    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])
        
        self._lazy_init()
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        return self.embed([text])[0]
