"""Test suite for RAG (Retrieval-Augmented Generation) functionality.

Tests core functionality:
- Standards ingestion from data/standards/ directory
- Basic query retrieval
- Filtering by specific standards
"""

import shutil
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import patch

import pytest

from backend.services.rag.retriever import StandardsRetriever


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_chromadb():
    """Reset ChromaDB state between tests to avoid ephemeral client conflicts."""
    try:
        import chromadb
        # Reset chromadb's shared system client state
        chromadb.api.shared_system_client.SharedSystemClient._identifier_to_system.clear()
    except (ImportError, AttributeError):
        pass
    yield
    try:
        import chromadb
        chromadb.api.shared_system_client.SharedSystemClient._identifier_to_system.clear()
    except (ImportError, AttributeError):
        pass


@pytest.fixture
def temp_chromadb_dir() -> Generator[str, None, None]:
    """Create a temporary directory for ChromaDB persistence."""
    temp_dir = tempfile.mkdtemp(prefix="test_chromadb_")
    yield temp_dir
    if Path(temp_dir).exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def temp_standards_dir() -> Generator[Path, None, None]:
    """Create a temporary standards directory with test content."""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_standards_"))
    
    for standard in ["DO_178C", "MISRA_C", "Boeing_SDP", "NASA"]:
        (temp_dir / standard).mkdir(exist_ok=True)
    
    project_root = Path(__file__).resolve().parent.parent.parent
    source_standards_dir = project_root / "data" / "standards"
    
    for standard in ["DO_178C", "MISRA_C", "Boeing_SDP", "NASA"]:
        source_std_dir = source_standards_dir / standard
        if source_std_dir.exists():
            for file_path in source_std_dir.glob("*"):
                if file_path.suffix in (".md", ".txt"):
                    dest_path = temp_dir / standard / file_path.name
                    dest_path.write_text(file_path.read_text())
    
    yield temp_dir
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def retriever_with_ingested_standards(
    temp_chromadb_dir: str,
    temp_standards_dir: Path,
) -> StandardsRetriever:
    """Create a retriever with ingested test standards."""
    retriever = StandardsRetriever(persist_dir=temp_chromadb_dir)
    retriever.ingest_directory(str(temp_standards_dir))
    return retriever


# =============================================================================
# Tests
# =============================================================================


def test_ingest_standards_returns_count(
    temp_chromadb_dir: str,
    temp_standards_dir: Path,
) -> None:
    """Test that ingest_standards returns a non-zero count."""
    retriever = StandardsRetriever(persist_dir=temp_chromadb_dir)
    count = retriever.ingest_directory(str(temp_standards_dir))
    assert count > 0, "Should ingest at least one document"


def test_retrieve_returns_results(
    retriever_with_ingested_standards: StandardsRetriever,
) -> None:
    """Test that retrieve returns results for a valid query."""
    results = retriever_with_ingested_standards.retrieve("memory allocation", n_results=5)
    assert isinstance(results, str), "Should return string results"


def test_retrieve_with_standard_filter(
    retriever_with_ingested_standards: StandardsRetriever,
) -> None:
    """Test that retrieve works with standard filtering."""
    results = retriever_with_ingested_standards.retrieve(
        "memory allocation",
        standard="DO_178C",
        n_results=3,
    )
    assert isinstance(results, str), "Should return string results with filter"


def test_end_to_end_workflow(
    temp_chromadb_dir: str,
    temp_standards_dir: Path,
) -> None:
    """Test complete workflow: create, ingest, and retrieve."""
    retriever = StandardsRetriever(persist_dir=temp_chromadb_dir)
    count = retriever.ingest_directory(str(temp_standards_dir))
    assert count > 0, "Should ingest documents"
    
    results = retriever.retrieve("memory allocation", standard="DO_178C", n_results=3)
    assert isinstance(results, str), "Should return string results"