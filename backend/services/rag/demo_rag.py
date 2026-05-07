#!/usr/bin/env python
"""Full production-grade RAG pipeline demo script.

This script demonstrates the complete end-to-end RAG workflow:
1. Load policy documents from data/standards/
2. Chunk them semantically  
3. Generate real embeddings with SentenceTransformer
4. Store in ChromaDB with persistence
5. Retrieve with vector similarity search

Run with: python demo_rag.py
"""

import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.services.rag.config import RAGConfig
from backend.services.rag.loaders.document_loader import DocumentLoader
from backend.services.rag.chunking.chunker import Chunker
from backend.services.rag.embeddings.embedder import Embedder
from backend.services.rag.storage.chromadb_store import ChromaDBStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(title):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def main():
    """Run the complete RAG pipeline demo."""
    
    # Setup
    print_header("🚀 PRODUCTION RAG PIPELINE END-TO-END DEMO")
    
    standards_path = project_root / "data" / "standards"
    if not standards_path.exists():
        print(f"❌ Error: {standards_path} not found")
        print(f"   Please run this script from the project root directory.")
        return 1
    
    # Configure RAG
    config = RAGConfig()
    print(f"Configuration:")
    print(f"  • Embedding Model: {config.embedding_model}")
    print(f"  • Chunk Size: {config.chunk_size} tokens")
    print(f"  • Chunk Overlap: {config.chunk_overlap} tokens")
    print(f"  • Default Results: {config.default_n_results}\n")
    
    # Check dependencies
    try:
        import chromadb
        import sentence_transformers
        print("✅ Dependencies installed: chromadb, sentence-transformers\n")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nInstall with:")
        print("  pip install chromadb sentence-transformers\n")
        return 1
    
    # Phase 1: Document Loading
    print_header("Phase 1: Document Loading")
    
    try:
        loader = DocumentLoader()
        documents = loader.load_from_directory(str(standards_path))
        
        if not documents:
            print("❌ No documents found")
            return 1
        
        print(f"✅ Loaded {len(documents)} documents\n")
        
        for i, doc in enumerate(documents, 1):
            standard = doc.metadata["standard"]
            filename = doc.metadata["file"]
            size = len(doc.content)
            print(f"  [{i}] {standard}/{filename}")
            print(f"       Size: {size:,} characters")
            print(f"       Content:")
            print(f"       {doc.content}")
    
    except Exception as e:
        print(f"❌ Document loading failed: {e}")
        logger.exception("Document loading error")
        return 1
    
    # Phase 2: Semantic Chunking
    print_header("Phase 2: Semantic Chunking")
    
    try:
        chunker = Chunker(config)
        chunks = chunker.chunk(documents)
        
        if not chunks:
            print("❌ No chunks created")
            return 1
        
        print(f"✅ Created {len(chunks)} chunks from {len(documents)} documents\n")
        
        # Show chunk distribution
        standards_count = {}
        for chunk in chunks:
            standard = chunk.metadata["standard"]
            standards_count[standard] = standards_count.get(standard, 0) + 1
        
        for standard, count in sorted(standards_count.items()):
            print(f"  • {standard}: {count} chunks")
    
    except Exception as e:
        print(f"❌ Chunking failed: {e}")
        logger.exception("Chunking error")
        return 1
    
    # Phase 3: Real Embedding Generation
    print_header("Phase 3: Embedding Generation (SentenceTransformer)")
    
    try:
        print(f"  Initializing embedder with model: {config.embedding_model}")
        embedder = Embedder(config)
        
        # Extract text from chunks
        chunk_texts = [chunk.text for chunk in chunks]
        
        # Generate real embeddings (batch operation)
        print(f"  Embedding {len(chunk_texts)} chunks...\n")
        embeddings = embedder.embed(chunk_texts)
        
        print(f"✅ Generated {len(embeddings)} embeddings")
        print(f"   Embedding dimension: {embeddings.shape[1]}")
        print(f"   Vector shape: {embeddings.shape}\n")
        
        # Show embedding stats
        chunk_sizes = [len(text.split()) for text in chunk_texts]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        max_size = max(chunk_sizes)
        min_size = min(chunk_sizes)
        
        print(f"  Chunk statistics (words):")
        print(f"  • Average: {avg_size:.0f} words")
        print(f"  • Maximum: {max_size} words")
        print(f"  • Minimum: {min_size} words\n")
        
        print(f"  Embedding vector statistics (first chunk):")
        print(f"  • Min value: {embeddings[0].min():.6f}")
        print(f"  • Max value: {embeddings[0].max():.6f}")
        print(f"  • Mean value: {embeddings[0].mean():.6f}")
        print(f"  • Norm (L2): {(embeddings[0] ** 2).sum() ** 0.5:.6f}")
    
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        logger.exception("Embedding error")
        return 1
    
    # Phase 4: ChromaDB Setup & Storage
    print_header("Phase 4: ChromaDB Storage & Indexing")
    
    try:
        # Initialize ChromaDB with optional persistence
        persist_dir = None  # Set to a path for persistent storage
        # persist_dir = "./chroma_data"  # Uncomment for persistent storage
        
        print(f"  Initializing ChromaDB...")
        if persist_dir:
            print(f"  Persistence directory: {persist_dir}")
        else:
            print(f"  Storage: Ephemeral (in-memory)")
        
        chroma_store = ChromaDBStore(persist_dir=persist_dir)
        print(f"  ✓ ChromaDB client initialized")
        print(f"  ✓ Collection 'policy_standards' created\n")
        
        # Clear collection if it already exists (to avoid duplicates on re-run)
        try:
            chroma_store.clear()
            print(f"  Cleared existing collection\n")
        except Exception:
            pass  # First run, collection doesn't exist yet
        
        # Convert embeddings to list format for ChromaDB
        embeddings_list = [emb.tolist() for emb in embeddings]
        
        print(f"  Ingesting {len(chunks)} chunks with embeddings...")
        num_added = chroma_store.add_chunks(chunks, embeddings_list)
        
        print(f"✅ Successfully stored {num_added} chunks in ChromaDB")
        print(f"   Total vectors indexed: {num_added}")
        print(f"   Similarity metric: Cosine distance")
        
        # Persist if configured
        if persist_dir:
            chroma_store.persist()
            print(f"   Data persisted to: {persist_dir}")
    
    except ImportError as e:
        print(f"❌ ChromaDB dependency error: {e}")
        print("\nInstall with: pip install chromadb")
        logger.exception("ChromaDB import error")
        return 1
    
    except Exception as e:
        print(f"❌ ChromaDB storage failed: {e}")
        logger.exception("ChromaDB error")
        return 1
    
    # Phase 5: Vector Similarity Retrieval
    print_header("Phase 5: Semantic Retrieval (Vector Similarity)")
    
    demo_queries = [
        "software safety requirements",
        "memory allocation bounds checking",
        "code verification and testing",
        "failure analysis procedures",
        "design assurance standards",
    ]
    
    print("Semantic (vector-based) retrieval:\n")
    
    try:
        for query in demo_queries:
            print(f"  Query: '{query}'")
            
            # Embed the query with same model
            query_embedding = embedder.embed_single(query)
            
            # Retrieve from ChromaDB (vector similarity search)
            results = chroma_store.query(
                query_embedding.tolist(),
                n_results=2
            )
            
            if results:
                for idx, (text, metadata, distance) in enumerate(results, 1):
                    standard = metadata["standard"]
                    file = metadata["file"]
                    # Convert distance to relevance (cosine distance: lower = more similar)
                    relevance = 1 - distance
                    
                    print(f"    [{idx}] {standard}/{file}")
                    print(f"        Relevance: {relevance:.1%} (distance: {distance:.4f})")
                    print(f"        Content: {text}")
            else:
                print(f"    (No results found)")
            
            print()
    
    except Exception as e:
        print(f"❌ Retrieval failed: {e}")
        logger.exception("Retrieval error")
        return 1
    
    # Phase 6: Filtered Retrieval by Standard
    print_header("Phase 6: Filtered Retrieval by Standard")
    
    test_query = "safety verification"
    test_standard = "DO_178C"
    
    print(f"  Query: '{test_query}'")
    print(f"  Filter: Standard = '{test_standard}'\n")
    
    try:
        query_embedding = embedder.embed_single(test_query)
        filtered_results = chroma_store.query(
            query_embedding.tolist(),
            standard=test_standard,
            n_results=3
        )
        
        if filtered_results:
            print(f"  ✅ Found {len(filtered_results)} results in {test_standard}:\n")
            for idx, (text, metadata, distance) in enumerate(filtered_results, 1):
                relevance = 1 - distance
                print(f"    [{idx}] Relevance: {relevance:.1%}")
                print(f"        {text}\n")
        else:
            print(f"  ℹ️  No results found for this query in {test_standard}")
    
    except Exception as e:
        print(f"❌ Filtered retrieval failed: {e}")
        logger.exception("Filtered retrieval error")
        return 1
    
    
    # Summary
    print_header("Summary")
    
    print(f"""
The production RAG pipeline successfully completed:

  ✅ Phase 1: Load      {len(documents)} policy documents
  ✅ Phase 2: Chunk     {len(chunks)} semantic chunks
  ✅ Phase 3: Embed     Real SentenceTransformer embeddings ({embeddings.shape[1]}D vector space)
  ✅ Phase 4: Store     ChromaDB ingestion with {num_added} vectors indexed
  ✅ Phase 5: Retrieve  Vector similarity search (cosine distance metric)
  ✅ Phase 6: Filter    Standard-based filtering working

Key Technical Details:
  • Embedding Model: {config.embedding_model} (frozen pre-trained)
  • Vector Dimension: {embeddings.shape[1]} (semantic space)
  • Similarity Metric: Cosine distance (0.0 = identical, 2.0 = opposite)
  • Storage: ChromaDB with HNSW indexing
  • Persistence: {'Enabled' if persist_dir else 'Ephemeral (in-memory)'}
  • Collection: policy_standards ({len(chunks)} documents)

Production Usage:
  from backend.services.rag import StandardsRetriever
  
  retriever = StandardsRetriever()
  retriever.ingest_directory("data/standards")
  results = retriever.retrieve("your query", n_results=5)
  print(results)

For more information, see RAG_USAGE_GUIDE.md
""")
    
    print("="*70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
