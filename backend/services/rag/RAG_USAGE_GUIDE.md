# RAG System Usage Guide

## Quick Start

```python
from backend.services.rag import StandardsRetriever

retriever = StandardsRetriever()
retriever.ingest_directory("data/standards")
results = retriever.retrieve("memory allocation safety")
print(results)
```

## Basic Usage

### Ingest Documents
```python
num_chunks = retriever.ingest_directory("data/standards")
print(f"Ingested {num_chunks} chunks")
```

### Retrieve with Formatted Output
```python
results = retriever.retrieve("software verification")
print(results)
```

### Retrieve Raw Results (for agents)
```python
raw_results = retriever.retrieve_raw("safety requirements", n_results=5)
for text, metadata, distance in raw_results:
    print(f"{metadata['standard']}: {text}")
```

### Filter by Standard
```python
results = retriever.retrieve("memory safety", standard="DO_178C")
```

## API Reference

### StandardsRetriever

| Method | Description |
|--------|-------------|
| `ingest_directory(path: str)` | Load, chunk, and embed documents. Returns chunk count. |
| `retrieve(query: str, standard: str \| None, n_results: int \| None)` | Query with formatted string output. |
| `retrieve_raw(query: str, standard: str \| None, n_results: int \| None)` | Query with raw tuple output: `(text, metadata, distance)` |

## Configuration

```python
from backend.services.rag import RAGConfig, StandardsRetriever

config = RAGConfig(
    chunk_size=512,
    chunk_overlap=50,
    embedding_model="all-MiniLM-L6-v2",
    n_results=5
)

retriever = StandardsRetriever(config=config)
```

## Dependencies

```bash
pip install chromadb sentence-transformers numpy
```