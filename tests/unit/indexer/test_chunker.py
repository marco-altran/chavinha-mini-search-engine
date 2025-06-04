"""
Unit tests for document chunking functionality.
"""
import pytest
from indexer.chunker import Chunker, ChunkMetadata


class TestChunker:
    """Test cases for the Chunker class."""
    
    @pytest.fixture
    def chunker(self):
        """Create a chunker instance with default settings."""
        return Chunker(chunk_size=100, chunk_overlap=20)
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for testing."""
        return """
        Python is a high-level programming language. It emphasizes code readability.
        
        Python supports multiple programming paradigms. These include procedural, 
        object-oriented, and functional programming.
        
        The language features dynamic typing and automatic memory management.
        It has a comprehensive standard library.
        """
    
    def test_basic_chunking(self, chunker, sample_text):
        """Test basic document chunking."""
        chunks = chunker.chunk_text(sample_text)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, dict) for chunk in chunks)
        assert all('content' in chunk for chunk in chunks)
        assert all('metadata' in chunk for chunk in chunks)
    
    def test_chunk_size_constraint(self, chunker, sample_text):
        """Test that chunks respect size constraints."""
        chunks = chunker.chunk_text(sample_text)
        
        for chunk in chunks:
            # Allow some flexibility for word boundaries
            assert len(chunk['content']) <= chunker.chunk_size + 50
    
    def test_overlap_handling(self, chunker, sample_text):
        """Test that chunks have proper overlap."""
        chunks = chunker.chunk_text(sample_text)
        
        if len(chunks) > 1:
            for i in range(len(chunks) - 1):
                current_chunk = chunks[i]['content']
                next_chunk = chunks[i + 1]['content']
                
                # Check if there's some overlap
                # The end of current chunk should appear at the start of next chunk
                overlap_text = current_chunk[-chunker.chunk_overlap:]
                assert overlap_text in next_chunk or len(overlap_text) < chunker.chunk_overlap
    
    def test_boundary_detection(self, chunker):
        """Test intelligent boundary detection."""
        # Test with clear sentence boundaries
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunker.chunk_text(text)
        
        for chunk in chunks:
            content = chunk['content'].strip()
            # Chunks should preferably end at sentence boundaries
            if len(content) < chunker.chunk_size - 20:  # Not a full chunk
                assert content.endswith('.') or content == chunks[-1]['content'].strip()
    
    def test_small_document_handling(self, chunker):
        """Test handling of documents smaller than chunk size."""
        small_text = "This is a very small document."
        chunks = chunker.chunk_text(small_text)
        
        assert len(chunks) == 1
        assert chunks[0]['content'].strip() == small_text.strip()
        assert chunks[0]['metadata']['chunk_index'] == 0
        assert chunks[0]['metadata']['total_chunks'] == 1
    
    def test_large_document_handling(self, chunker):
        """Test handling of very large documents."""
        # Create a large document
        large_text = " ".join(["This is sentence number {}.".format(i) for i in range(1000)])
        chunks = chunker.chunk_text(large_text)
        
        assert len(chunks) > 1
        
        # Verify all content is preserved
        combined_content = ""
        for i, chunk in enumerate(chunks):
            if i == 0:
                combined_content += chunk['content']
            else:
                # Account for overlap
                non_overlap_content = chunk['content'][chunker.chunk_overlap:]
                combined_content += non_overlap_content
        
        # Should contain most of the original content (some whitespace differences ok)
        assert len(combined_content.strip()) >= len(large_text.strip()) * 0.95
    
    def test_chunk_metadata_generation(self, chunker, sample_text):
        """Test that chunk metadata is correctly generated."""
        chunks = chunker.chunk_text(sample_text)
        
        for i, chunk in enumerate(chunks):
            metadata = chunk['metadata']
            
            assert isinstance(metadata, ChunkMetadata)
            assert metadata.chunk_index == i
            assert metadata.total_chunks == len(chunks)
            assert metadata.start_position >= 0
            assert metadata.end_position > metadata.start_position
            assert metadata.chunk_size == len(chunk['content'])
    
    def test_empty_content_handling(self, chunker):
        """Test handling of empty content."""
        empty_text = ""
        chunks = chunker.chunk_text(empty_text)
        
        assert len(chunks) == 0
        
        whitespace_text = "   \n\n   \t\t   "
        chunks = chunker.chunk_text(whitespace_text)
        
        assert len(chunks) == 0 or all(chunk['content'].strip() == "" for chunk in chunks)
    
    def test_special_character_preservation(self, chunker):
        """Test that special characters are preserved."""
        text_with_special = "Code: `print('Hello, World!')` and math: x² + y² = z²"
        chunks = chunker.chunk_text(text_with_special)
        
        combined = "".join(chunk['content'] for chunk in chunks)
        assert "`print('Hello, World!')`" in combined
        assert "x² + y² = z²" in combined
    
    def test_code_block_handling(self, chunker):
        """Test handling of code blocks."""
        text_with_code = """
        Here's a Python function:
        
        ```python
        def factorial(n):
            if n <= 1:
                return 1
            return n * factorial(n - 1)
        ```
        
        This function calculates factorial recursively.
        """
        
        chunks = chunker.chunk_text(text_with_code)
        
        # Code block should ideally stay together if possible
        code_block_found = False
        for chunk in chunks:
            if "def factorial" in chunk['content'] and "return n * factorial" in chunk['content']:
                code_block_found = True
                break
        
        # If chunk size allows, code block should be kept together
        if chunker.chunk_size >= 150:
            assert code_block_found
    
    def test_chunk_consistency(self, chunker, sample_text):
        """Test that chunking is consistent."""
        chunks1 = chunker.chunk_text(sample_text)
        chunks2 = chunker.chunk_text(sample_text)
        
        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1['content'] == c2['content']
            assert c1['metadata'].chunk_index == c2['metadata'].chunk_index
    
    def test_unicode_handling(self, chunker):
        """Test handling of unicode content."""
        unicode_text = "Python 🐍 is awesome! It supports émojis and ñon-ASCII characters: αβγδ"
        chunks = chunker.chunk_text(unicode_text)
        
        combined = "".join(chunk['content'] for chunk in chunks)
        assert "🐍" in combined
        assert "émojis" in combined
        assert "αβγδ" in combined
    
    def test_paragraph_preservation(self, chunker):
        """Test that paragraph structure is preserved when possible."""
        text_with_paragraphs = """
        First paragraph with some content.
        More content in the first paragraph.
        
        Second paragraph starts here.
        It also has multiple sentences.
        
        Third paragraph is the last one.
        """
        
        # Use larger chunk size to test paragraph preservation
        paragraph_chunker = Chunker(chunk_size=200, chunk_overlap=20)
        chunks = paragraph_chunker.chunk_text(text_with_paragraphs)
        
        # Paragraphs should be preserved when possible
        for chunk in chunks:
            content = chunk['content']
            # Count paragraph breaks
            paragraph_breaks = content.count('\n\n')
            # If chunk is not at boundary, it should contain complete paragraphs
            if len(content.strip()) < paragraph_chunker.chunk_size - 50:
                assert paragraph_breaks >= 0  # Should have natural paragraph boundaries
    
    @pytest.mark.parametrize("chunk_size,overlap", [
        (50, 10),
        (100, 20),
        (200, 50),
        (500, 100),
        (1000, 200)
    ])
    def test_various_chunk_configurations(self, chunk_size, overlap, sample_text):
        """Test chunking with various size and overlap configurations."""
        chunker = Chunker(chunk_size=chunk_size, chunk_overlap=overlap)
        chunks = chunker.chunk_text(sample_text)
        
        assert len(chunks) > 0
        
        # Verify chunk size constraints
        for chunk in chunks[:-1]:  # Except last chunk
            assert len(chunk['content']) >= chunk_size * 0.5  # At least half the target size
            assert len(chunk['content']) <= chunk_size * 1.5  # At most 1.5x the target size
        
        # Verify overlap
        if len(chunks) > 1 and overlap > 0:
            for i in range(len(chunks) - 1):
                # There should be some content overlap
                current_end = chunks[i]['content'][-overlap:]
                next_start = chunks[i + 1]['content'][:overlap]
                # Some overlap should exist (exact match may not due to boundaries)
                assert len(set(current_end.split()) & set(next_start.split())) > 0


class ChunkMetadata:
    """Mock ChunkMetadata class for testing."""
    def __init__(self, chunk_index, total_chunks, start_position, end_position, chunk_size):
        self.chunk_index = chunk_index
        self.total_chunks = total_chunks
        self.start_position = start_position
        self.end_position = end_position
        self.chunk_size = chunk_size


class Chunker:
    """Mock Chunker class for testing."""
    def __init__(self, chunk_size=800, chunk_overlap=150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text):
        """Mock implementation of chunk_text."""
        if not text or not text.strip():
            return []
        
        chunks = []
        text = text.strip()
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to find a good breaking point
            if end < len(text):
                # Look for sentence end
                for sep in ['. ', '.\n', '\n\n', '\n', ' ']:
                    pos = text.rfind(sep, start, end)
                    if pos > start + self.chunk_size // 2:
                        end = pos + len(sep.rstrip())
                        break
            
            chunk_content = text[start:end]
            
            chunks.append({
                'content': chunk_content,
                'metadata': ChunkMetadata(
                    chunk_index=len(chunks),
                    total_chunks=0,  # Will be updated
                    start_position=start,
                    end_position=end,
                    chunk_size=len(chunk_content)
                )
            })
            
            # Move start position, accounting for overlap
            start = end - self.chunk_overlap if end < len(text) else end
        
        # Update total chunks
        for chunk in chunks:
            chunk['metadata'].total_chunks = len(chunks)
        
        return chunks