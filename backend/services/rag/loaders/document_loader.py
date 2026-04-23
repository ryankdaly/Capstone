"""Document loader for markdown files in data/standards/."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a single document chunk for ingestion."""

    content: str
    metadata: dict


class DocumentLoader:
    """Loads markdown documents from a directory tree."""

    def load_from_directory(self, directory_path: str) -> List[Document]:
        """Load all markdown files from a directory and its subdirectories.
        
        Expects structure:
            directory_path/
            ├── Standard1/
            │   ├── file1.md
            │   └── file2.md
            ├── Standard2/
            │   └── file3.md
        
        Each file is returned as a Document with metadata:
        - standard: Name of the standard directory
        - file: Filename (e.g., "design_practices.md")
        - path: Relative path from directory_path root
        
        Args:
            directory_path: Root directory to scan for .md files
            
        Returns:
            List of Document objects with content and metadata
            
        Raises:
            FileNotFoundError: If directory_path does not exist
        """
        root_path = Path(directory_path)
        
        if not root_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if not root_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")
        
        documents = []
        
        # Recursively find all .md files
        for md_file in sorted(root_path.rglob("*.md")):
            try:
                content = md_file.read_text(encoding="utf-8")
                
                # Extract standard name from immediate parent directory
                standard_name = md_file.parent.name
                
                # Build relative path
                relative_path = md_file.relative_to(root_path).as_posix()
                
                metadata = {
                    "standard": standard_name,
                    "file": md_file.name,
                    "path": relative_path,
                }
                
                doc = Document(content=content, metadata=metadata)
                documents.append(doc)
                logger.debug(f"Loaded document: {standard_name}/{md_file.name}")
                
            except Exception as e:
                logger.error(f"Failed to load {md_file}: {e}")
                continue
        
        logger.info(f"Loaded {len(documents)} documents from {directory_path}")
        return documents
