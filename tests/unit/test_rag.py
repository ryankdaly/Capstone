"""Unit tests for RAG components: DocumentLoader, Chunker, StandardsRetriever."""

from __future__ import annotations

import textwrap
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.services.rag.chunking.chunker import Chunk, Chunker
from backend.services.rag.config import RAGConfig
from backend.services.rag.loaders.document_loader import Document, DocumentLoader
from backend.services.rag.retriever import StandardsRetriever
from backend.services.rag.storage.chromadb_store import ChromaDBStore


# ---------------------------------------------------------------------------
# DocumentLoader
# ---------------------------------------------------------------------------


class TestDocumentLoader:
    def test_load_from_real_standards_dir(self):
        root = Path(__file__).resolve().parents[2] / "data" / "standards"
        if not root.exists():
            pytest.skip("data/standards directory not present")
        loader = DocumentLoader()
        docs = loader.load_from_directory(str(root))
        assert len(docs) > 0, "Expected at least one document"
        for doc in docs:
            assert doc.content.strip(), "Document content should not be empty"
            assert "standard" in doc.metadata
            assert "file" in doc.metadata

    def test_metadata_includes_standard_name(self, tmp_path):
        do178_dir = tmp_path / "DO_178C"
        do178_dir.mkdir()
        (do178_dir / "rules.md").write_text("# Rule 1\nNo recursion.", encoding="utf-8")

        loader = DocumentLoader()
        docs = loader.load_from_directory(str(tmp_path))
        assert len(docs) == 1
        assert docs[0].metadata["standard"] == "DO_178C"
        assert docs[0].metadata["file"] == "rules.md"
        assert "No recursion" in docs[0].content

    def test_missing_directory_raises(self):
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_from_directory("/nonexistent/path/xyz")

    def test_ignores_non_md_files(self, tmp_path):
        (tmp_path / "Standard").mkdir()
        (tmp_path / "Standard" / "file.md").write_text("content", encoding="utf-8")
        (tmp_path / "Standard" / "readme.txt").write_text("ignored", encoding="utf-8")
        (tmp_path / "Standard" / "data.csv").write_text("a,b,c", encoding="utf-8")
        loader = DocumentLoader()
        docs = loader.load_from_directory(str(tmp_path))
        assert len(docs) == 1

    def test_empty_directory_returns_empty_list(self, tmp_path):
        loader = DocumentLoader()
        docs = loader.load_from_directory(str(tmp_path))
        assert docs == []


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------


class TestChunker:
    def _make_doc(self, content: str, standard: str = "TEST") -> Document:
        return Document(content=content, metadata={"standard": standard, "file": "test.md"})

    def test_short_doc_produces_single_chunk(self):
        config = RAGConfig(chunk_size=512)
        chunker = Chunker(config)
        doc = self._make_doc("Short content that fits in one chunk.")
        chunks = chunker.chunk([doc])
        assert len(chunks) == 1
        assert "Short content" in chunks[0].text

    def test_metadata_preserved(self):
        config = RAGConfig(chunk_size=512)
        chunker = Chunker(config)
        doc = self._make_doc("## Section\nContent here.", standard="DO_178C")
        chunks = chunker.chunk([doc])
        for chunk in chunks:
            assert chunk.metadata.get("standard") == "DO_178C"
            assert chunk.metadata.get("file") == "test.md"

    def test_long_doc_produces_multiple_chunks(self):
        config = RAGConfig(chunk_size=50, chunk_overlap=5)
        chunker = Chunker(config)
        long_text = " ".join([f"Sentence number {i}." for i in range(200)])
        doc = self._make_doc(long_text)
        chunks = chunker.chunk([doc])
        assert len(chunks) > 1

    def test_markdown_sections_split_correctly(self):
        config = RAGConfig(chunk_size=512)
        chunker = Chunker(config)
        content = textwrap.dedent("""\
            ## Section 1
            Content about safety rules.

            ## Section 2
            Content about verification.
        """)
        doc = self._make_doc(content)
        chunks = chunker.chunk([doc])
        texts = " ".join(c.text for c in chunks)
        assert "Section 1" in texts
        assert "Section 2" in texts

    def test_empty_document_list(self):
        config = RAGConfig()
        chunker = Chunker(config)
        chunks = chunker.chunk([])
        assert chunks == []

    def test_chunk_text_not_empty(self):
        config = RAGConfig(chunk_size=512)
        chunker = Chunker(config)
        doc = self._make_doc("## Rule 1\nContent.\n\n## Rule 2\nMore content.")
        chunks = chunker.chunk([doc])
        for chunk in chunks:
            assert chunk.text.strip(), "Chunk text should not be empty"


# ---------------------------------------------------------------------------
# ChromaDBStore (ephemeral)
# ---------------------------------------------------------------------------


def _fresh_store() -> ChromaDBStore:
    """Create an isolated ephemeral store with a unique collection name."""
    return ChromaDBStore(persist_dir=None, collection_name=f"test_{uuid.uuid4().hex[:12]}")


class TestChromaDBStore:
    def test_ephemeral_store_add_and_query(self):
        store = _fresh_store()
        chunks = [
            Chunk(text="No dynamic memory allocation is allowed.", metadata={"standard": "DO_178C", "file": "rules.md"}),
            Chunk(text="All loops must have bounded iterations.", metadata={"standard": "DO_178C", "file": "rules.md"}),
            Chunk(text="Recursion is prohibited in safety-critical code.", metadata={"standard": "DO_178C", "file": "rules.md"}),
        ]
        added = store.add_chunks(chunks)
        assert added == 3
        assert store.count() == 3

        results = store.query("memory allocation", n_results=2)
        assert len(results) <= 2
        texts = " ".join(r[0] for r in results)
        assert "memory" in texts.lower()

    def test_upsert_is_idempotent(self):
        store = _fresh_store()
        chunk = Chunk(text="Same content ingested twice.", metadata={"standard": "X", "file": "f.md"})
        store.add_chunks([chunk])
        store.add_chunks([chunk])
        assert store.count() == 1  # upsert, not append

    def test_empty_collection_returns_no_results(self):
        store = _fresh_store()
        results = store.query("anything")
        assert results == []

    def test_standard_filter(self):
        store = _fresh_store()
        chunks = [
            Chunk(text="DO-178C specific rule about coverage.", metadata={"standard": "DO_178C", "file": "a.md"}),
            Chunk(text="MISRA rule about undefined behavior.", metadata={"standard": "MISRA_C", "file": "b.md"}),
        ]
        store.add_chunks(chunks)
        results = store.query("rule", standard="DO_178C", n_results=5)
        assert all(r[1]["standard"] == "DO_178C" for r in results)

    def test_clear_empties_collection(self):
        store = _fresh_store()
        store.add_chunks([Chunk(text="content", metadata={"standard": "X", "file": "f.md"})])
        assert store.count() == 1
        store.clear()
        assert store.count() == 0


# ---------------------------------------------------------------------------
# StandardsRetriever (mocked ChromaDB)
# ---------------------------------------------------------------------------


class TestStandardsRetriever:
    def _make_retriever_with_mock_store(self, mock_results=None):
        """Create a StandardsRetriever with a mocked ChromaDBStore."""
        if mock_results is None:
            mock_results = [
                ("No dynamic memory allocation.", {"standard": "DO_178C", "file": "rules.md"}, 0.1),
            ]
        retriever = StandardsRetriever(persist_dir=None)
        retriever._ingestion_checked = True  # skip auto-ingest
        mock_store = MagicMock(spec=ChromaDBStore)
        mock_store.query.return_value = mock_results
        mock_store.count.return_value = len(mock_results)
        retriever._store = mock_store
        return retriever, mock_store

    @pytest.mark.asyncio
    async def test_retrieve_returns_formatted_string(self):
        retriever, _ = self._make_retriever_with_mock_store()
        result = await retriever.retrieve("memory allocation", standard="DO_178C")
        assert "DO_178C" in result
        assert "No dynamic memory allocation" in result

    @pytest.mark.asyncio
    async def test_retrieve_empty_query_returns_empty_string(self):
        retriever, mock_store = self._make_retriever_with_mock_store()
        result = await retriever.retrieve("")
        assert result == ""
        mock_store.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_retrieve_no_results_returns_no_found_message(self):
        retriever, _ = self._make_retriever_with_mock_store(mock_results=[])
        result = await retriever.retrieve("something")
        assert "No relevant" in result

    def test_ingest_directory_loads_and_stores(self, tmp_path):
        std_dir = tmp_path / "DO_178C"
        std_dir.mkdir()
        (std_dir / "rules.md").write_text("## Rule 1\nNo malloc.", encoding="utf-8")

        retriever = StandardsRetriever(persist_dir=None)
        retriever._ingestion_checked = True  # prevent auto-ingest from firing

        mock_store = MagicMock(spec=ChromaDBStore)
        mock_store.add_chunks.return_value = 1
        retriever._store = mock_store

        count = retriever.ingest_directory(str(tmp_path))
        assert count == 1
        mock_store.add_chunks.assert_called_once()

    def test_auto_ingest_skipped_when_collection_not_empty(self, tmp_path):
        (tmp_path / "DO_178C").mkdir()
        (tmp_path / "DO_178C" / "f.md").write_text("content", encoding="utf-8")

        retriever = StandardsRetriever(persist_dir=None, auto_ingest_path=str(tmp_path))

        mock_store = MagicMock(spec=ChromaDBStore)
        mock_store.count.return_value = 42  # not empty
        retriever._store = mock_store

        retriever._maybe_auto_ingest()
        mock_store.add_chunks.assert_not_called()

    def test_auto_ingest_fires_when_collection_empty(self, tmp_path):
        std_dir = tmp_path / "TEST"
        std_dir.mkdir()
        (std_dir / "f.md").write_text("## Rule\nContent.", encoding="utf-8")

        retriever = StandardsRetriever(persist_dir=None, auto_ingest_path=str(tmp_path))

        mock_store = MagicMock(spec=ChromaDBStore)
        mock_store.count.return_value = 0  # empty
        mock_store.add_chunks.return_value = 2
        retriever._store = mock_store

        retriever._maybe_auto_ingest()
        mock_store.add_chunks.assert_called_once()

    def test_auto_ingest_called_only_once(self, tmp_path):
        std_dir = tmp_path / "X"
        std_dir.mkdir()
        (std_dir / "f.md").write_text("content", encoding="utf-8")

        retriever = StandardsRetriever(persist_dir=None, auto_ingest_path=str(tmp_path))
        mock_store = MagicMock(spec=ChromaDBStore)
        mock_store.count.return_value = 0
        mock_store.add_chunks.return_value = 1
        retriever._store = mock_store

        retriever._maybe_auto_ingest()
        retriever._maybe_auto_ingest()
        retriever._maybe_auto_ingest()
        assert mock_store.add_chunks.call_count == 1  # guarded by _ingestion_checked
