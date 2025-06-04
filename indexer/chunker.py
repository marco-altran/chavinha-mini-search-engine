#!/usr/bin/env python3
"""Document chunking algorithm for semantic search optimization"""

import re
import hashlib
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class DocumentChunker:
    """
    Chunking algorithm that splits documents into overlapping segments
    optimized for semantic search while preserving context.
    """
    
    def __init__(self, chunk_size: int = 800, overlap: int = 150, min_chunk_size: int = 100):
        """
        Initialize the chunker with configuration.
        
        Args:
            chunk_size: Target size for each chunk in characters
            overlap: Number of characters to overlap between chunks
            min_chunk_size: Minimum size for a chunk to be valid
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        
        # Sentence splitting patterns (ordered by priority)
        self.sentence_patterns = [
            r'(?<=[.!?])\s+(?=[A-Z])',  # Period/exclamation/question + space + capital letter
            r'(?<=\.)\s+(?=\w)',        # Period + space + word
            r'(?<=\n\n)',               # Double newline (paragraph break)
            r'(?<=\n)',                 # Single newline
        ]
        
        # Word boundary pattern for emergency splitting
        self.word_boundary = r'\s+'
    
    def chunk_document(self, doc: Dict) -> List[Dict]:
        """
        Split a document into overlapping chunks.
        
        Args:
            doc: Document dictionary with content
            
        Returns:
            List of chunk dictionaries
        """
        content = doc.get('content', '')
        title = doc.get('title', '')
        
        if not content.strip():
            logger.warning(f"Empty content for document {doc.get('id', 'unknown')}")
            return []
        
        # Add title to content for better context
        full_content = f"{title}\n\n{content}" if title else content
        
        # Clean content
        cleaned_content = self._clean_content(full_content)
        
        if len(cleaned_content) <= self.chunk_size:
            # Document is small enough to be a single chunk
            return [self._create_chunk(doc, cleaned_content, 0, 0, len(cleaned_content))]
        
        # Split into chunks
        chunks = self._split_into_chunks(cleaned_content)
        
        # Create chunk documents
        chunk_docs = []
        for i, (chunk_text, start_pos, end_pos) in enumerate(chunks):
            if len(chunk_text.strip()) >= self.min_chunk_size:
                chunk_doc = self._create_chunk(doc, chunk_text, i, start_pos, end_pos)
                chunk_docs.append(chunk_doc)
        
        # Removed verbose logging - chunk count is tracked in indexer stats
        return chunk_docs
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content for chunking."""
        # Remove excessive whitespace while preserving structure
        content = re.sub(r'\n{3,}', '\n\n', content)  # Max 2 consecutive newlines
        content = re.sub(r'[ \t]{2,}', ' ', content)   # Multiple spaces/tabs to single space
        content = content.strip()
        return content
    
    def _split_into_chunks(self, content: str) -> List[Tuple[str, int, int]]:
        """
        Split content into chunks using intelligent boundary detection.
        
        Returns:
            List of (chunk_text, start_position, end_position) tuples
        """
        chunks = []
        start_pos = 0
        
        while start_pos < len(content):
            # Calculate end position for this chunk
            end_pos = min(start_pos + self.chunk_size, len(content))
            
            # If this is the last chunk, take everything
            if end_pos >= len(content):
                chunk_text = content[start_pos:]
                chunks.append((chunk_text, start_pos, len(content)))
                break
            
            # Find the best boundary near the target end position
            boundary_pos = self._find_best_boundary(content, start_pos, end_pos)
            
            # Extract chunk
            chunk_text = content[start_pos:boundary_pos]
            chunks.append((chunk_text, start_pos, boundary_pos))
            
            # Calculate next start position with overlap
            next_start = max(start_pos + self.min_chunk_size, boundary_pos - self.overlap)
            start_pos = next_start
        
        return chunks
    
    def _find_best_boundary(self, content: str, start: int, target_end: int) -> int:
        """
        Find the best boundary position near the target end.
        
        Args:
            content: Full content string
            start: Start position of the chunk
            target_end: Target end position
            
        Returns:
            Optimal boundary position
        """
        search_start = max(start + self.min_chunk_size, target_end - 200)
        search_end = min(len(content), target_end + 200)
        
        # Search for sentence boundaries in order of preference
        for pattern in self.sentence_patterns:
            matches = list(re.finditer(pattern, content[search_start:search_end]))
            if matches:
                # Find the match closest to target_end
                best_match = min(matches, key=lambda m: abs((search_start + m.end()) - target_end))
                boundary = search_start + best_match.end()
                
                # Ensure boundary doesn't create too small chunks
                if boundary - start >= self.min_chunk_size:
                    return boundary
        
        # Fallback to word boundary
        word_matches = list(re.finditer(self.word_boundary, content[search_start:search_end]))
        if word_matches:
            best_match = min(word_matches, key=lambda m: abs((search_start + m.start()) - target_end))
            boundary = search_start + best_match.start()
            
            if boundary - start >= self.min_chunk_size:
                return boundary
        
        # Last resort: use target position
        return target_end
    
    def _create_chunk(self, parent_doc: Dict, chunk_content: str, index: int, 
                     start_pos: int, end_pos: int) -> Dict:
        """
        Create a chunk document from parent document and chunk content.
        
        Args:
            parent_doc: Original document
            chunk_content: Content for this chunk
            index: Chunk index (0-based)
            start_pos: Start position in original content
            end_pos: End position in original content
            
        Returns:
            Chunk document dictionary
        """
        # Generate chunk ID
        chunk_id = f"{parent_doc['id']}_chunk_{index:03d}"
        
        # Create chunk document
        chunk_doc = {
            "id": chunk_id,
            "doc_type": "chunk",
            "parent_id": parent_doc['id'],
            "url": parent_doc['url'],
            "domain": parent_doc['domain'],
            "title": parent_doc['title'],
            "chunk_content": chunk_content.strip(),
            "chunk_index": index,
            "chunk_start": start_pos,
            "chunk_end": end_pos,
            "chunk_size": len(chunk_content.strip()),
            "crawled_at": parent_doc['crawled_at'],
            "parent_doc_type": parent_doc.get('doc_type', 'general')
        }
        
        return chunk_doc
    
    def get_chunk_stats(self, chunks: List[Dict]) -> Dict:
        """Get statistics about the chunks."""
        if not chunks:
            return {"count": 0}
        
        sizes = [chunk["chunk_size"] for chunk in chunks]
        
        return {
            "count": len(chunks),
            "total_size": sum(sizes),
            "avg_size": sum(sizes) / len(sizes),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "size_std": (sum((s - sum(sizes)/len(sizes))**2 for s in sizes) / len(sizes))**0.5
        }

def test_chunker():
    """Test the chunker with sample content."""
    # Sample document
    sample_doc = {
        "id": "test_doc_001",
        "url": "https://example.com/test",
        "domain": "example.com",
        "title": "Sample Document for Testing",
        "content": """
        This is a sample document that we'll use to test the chunking algorithm.
        
        It contains multiple paragraphs and sentences. The algorithm should split this content
        into meaningful chunks while preserving context through overlapping segments.
        
        Here's another paragraph with some technical content. When we implement semantic search,
        these chunks will have their own embeddings, allowing for more precise matching of
        user queries to specific sections of documents.
        
        The chunking algorithm tries to split on sentence boundaries when possible. This helps
        maintain the semantic coherence of each chunk while ensuring that related information
        stays together.
        
        Finally, this last paragraph demonstrates how the overlap mechanism works. Some content
        from the previous chunk will be included in the next chunk to maintain context continuity.
        This is especially important for technical documentation where context matters significantly.
        """,
        "crawled_at": "2025-05-28T16:00:00Z",
        "doc_type": "general"
    }
    
    # Test chunker
    chunker = DocumentChunker(chunk_size=300, overlap=50)
    chunks = chunker.chunk_document(sample_doc)
    
    print(f"Original document size: {len(sample_doc['content'])} characters")
    print(f"Generated {len(chunks)} chunks:")
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} (ID: {chunk['id']}):")
        print(f"  Size: {chunk['chunk_size']} characters")
        print(f"  Position: {chunk['chunk_start']}-{chunk['chunk_end']}")
        print(f"  Content preview: {chunk['chunk_content'][:100]}...")
    
    # Get statistics
    stats = chunker.get_chunk_stats(chunks)
    print(f"\nChunk Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.1f}")
        else:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    test_chunker()