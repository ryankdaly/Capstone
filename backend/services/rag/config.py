"""RAG service configuration."""

from dataclasses import dataclass


@dataclass
class RAGConfig:
    """Configuration for RAG service components.
    
    Attributes:
        embedding_model: Sentence-transformers model identifier
        chunk_size: Token-level chunk size (approximate)
        chunk_overlap: Overlap tokens between chunks
        default_n_results: Default number of results for retrieve()
    """

    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 64
    default_n_results: int = 5
