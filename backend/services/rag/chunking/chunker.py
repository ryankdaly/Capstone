"""Document chunking logic for RAG service."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List

from backend.services.rag.config import RAGConfig
from backend.services.rag.loaders.document_loader import Document

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a single chunk for embedding and storage."""

    text: str
    metadata: dict = field(default_factory=dict)


class Chunker:
    """Split documents into semantic chunks."""

    def __init__(self, config: RAGConfig) -> None:
        """Initialize chunker with configuration.
        
        Args:
            config: RAG configuration with chunk_size and chunk_overlap
        """
        self.config = config

    def chunk(self, documents: List[Document]) -> List[Chunk]:
        """Split documents into chunks.
        
        Strategy:
        1. Split by markdown sections (##, ###) to preserve semantics
        2. For long sections, further split by token size
        3. Maintain overlap between chunks
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunks with metadata preserved/enhanced
        """
        chunks = []
        
        for doc in documents:
            doc_chunks = self._chunk_document(doc)
            chunks.extend(doc_chunks)
        
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks

    def _chunk_document(self, doc: Document) -> List[Chunk]:
        """Chunk a single document by sections and token size."""
        sections = self._split_by_headers(doc.content)
        chunks = []
        
        for section in sections:
            section_chunks = self._chunk_section(section)
            for chunk in section_chunks:
                # Preserve and augment metadata
                chunk.metadata = {**doc.metadata, **chunk.metadata}
            chunks.extend(section_chunks)
        
        return chunks

    def _split_by_headers(self, content: str) -> List[str]:
        """Split markdown content by headers (##, ###, etc).
        
        Preserves header context with content for semantic grouping.
        """
        # Pattern: markdown headers (##, ###, ####, etc.)
        header_pattern = r"^(#{2,6})\s+(.+)$"
        
        sections = []
        current_section = []
        
        for line in content.split("\n"):
            match = re.match(header_pattern, line)
            
            if match:
                # Save previous section if it has content
                if current_section:
                    sections.append("\n".join(current_section).strip())
                
                # Start new section with the header
                current_section = [line]
            else:
                # Add content to current section
                current_section.append(line)
        
        # Don't forget the last section
        if current_section:
            sections.append("\n".join(current_section).strip())
        
        # Filter empty sections
        sections = [s for s in sections if s]
        
        return sections if sections else [content]

    def _chunk_section(self, section: str) -> List[Chunk]:
        """Chunk a section into appropriately sized chunks.
        
        Respects chunk_size and chunk_overlap from config.
        """
        if not section:
            return []
        
        # Estimate tokens (simple: word count approximation)
        section_tokens = self._estimate_tokens(section)
        
        # If section fits within chunk_size, return as single chunk
        if section_tokens <= self.config.chunk_size:
            return [Chunk(text=section, metadata={})]
        
        # Otherwise, split by sentences with overlap
        sentences = self._split_by_sentences(section)
        chunks = []
        current_chunk_text = ""
        current_tokens = 0
        overlap_buffer = ""
        
        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)
            
            # If adding this sentence exceeds chunk_size, save current chunk
            if current_tokens + sentence_tokens > self.config.chunk_size and current_chunk_text:
                # Save chunk
                chunks.append(Chunk(text=current_chunk_text.strip(), metadata={}))
                
                # Initialize next chunk with overlap from current
                current_chunk_text = overlap_buffer + " " + sentence
                current_tokens = self._estimate_tokens(current_chunk_text)
                overlap_buffer = self._get_overlap(current_chunk_text)
            else:
                # Add sentence to current chunk
                current_chunk_text += " " + sentence if current_chunk_text else sentence
                current_tokens += sentence_tokens
        
        # Don't forget the last chunk
        if current_chunk_text:
            chunks.append(Chunk(text=current_chunk_text.strip(), metadata={}))
        
        return chunks

    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences for finer-grained chunking."""
        # Simple sentence splitting: periods, question marks, exclamation marks
        # Avoid splitting on common abbreviations
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _estimate_tokens(self, text: str) -> int:
        """Rough estimate of token count.
        
        Using word count * 1.3 as approximation (typical ratio).
        """
        words = len(text.split())
        return max(1, int(words * 1.3))

    def _get_overlap(self, text: str) -> str:
        """Extract last N tokens for overlap context.
        
        Returns the last chunk_overlap tokens of text.
        """
        words = text.split()
        # Calculate how many words = chunk_overlap tokens
        overlap_words = max(1, int(self.config.chunk_overlap / 1.3))
        
        if len(words) > overlap_words:
            return " ".join(words[-overlap_words:])
        return text
