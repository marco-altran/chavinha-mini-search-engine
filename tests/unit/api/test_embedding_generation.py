"""
Unit tests for embedding generation functionality.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import asyncio


class TestEmbeddingGeneration:
    """Test cases for ONNX model embedding generation."""
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Mock tokenizer for testing."""
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[101, 2023, 2003, 102])  # Mock token IDs
        return tokenizer
    
    @pytest.fixture
    def mock_onnx_session(self):
        """Mock ONNX inference session."""
        session = Mock()
        # Return 384-dimensional embeddings
        session.run = Mock(return_value=[np.random.rand(1, 384)])
        return session
    
    def test_model_initialization(self, mock_onnx_session):
        """Test ONNX model initialization."""
        with patch('onnxruntime.InferenceSession', return_value=mock_onnx_session):
            from api.main import initialize_model
            
            model, tokenizer = initialize_model()
            
            assert model is not None
            assert tokenizer is not None
            assert hasattr(model, 'run')
    
    def test_embedding_generation_single_query(self, mock_onnx_session, mock_tokenizer):
        """Test embedding generation for a single query."""
        with patch('onnxruntime.InferenceSession', return_value=mock_onnx_session):
            with patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
                from api.main import generate_embedding
                
                query = "python list comprehension"
                embedding = generate_embedding(query, mock_onnx_session, mock_tokenizer)
                
                assert isinstance(embedding, list)
                assert len(embedding) == 384
                assert all(isinstance(x, float) for x in embedding)
    
    def test_embedding_generation_batch(self, mock_onnx_session, mock_tokenizer):
        """Test batch embedding generation."""
        batch_size = 5
        mock_onnx_session.run = Mock(return_value=[np.random.rand(batch_size, 384)])
        
        with patch('onnxruntime.InferenceSession', return_value=mock_onnx_session):
            with patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
                queries = ["query1", "query2", "query3", "query4", "query5"]
                
                # Simulate batch processing
                embeddings = []
                for query in queries:
                    embedding = generate_embedding(query, mock_onnx_session, mock_tokenizer)
                    embeddings.append(embedding)
                
                assert len(embeddings) == batch_size
                assert all(len(emb) == 384 for emb in embeddings)
    
    def test_embedding_dimension_validation(self, mock_onnx_session, mock_tokenizer):
        """Test that embeddings have correct dimensions."""
        expected_dim = 384
        mock_onnx_session.run = Mock(return_value=[np.random.rand(1, expected_dim)])
        
        with patch('onnxruntime.InferenceSession', return_value=mock_onnx_session):
            with patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
                from api.main import generate_embedding
                
                embedding = generate_embedding("test query", mock_onnx_session, mock_tokenizer)
                
                assert len(embedding) == expected_dim
                assert isinstance(embedding[0], float)
    
    def test_model_warmup_effect(self, mock_onnx_session, mock_tokenizer, performance_timer):
        """Test model warmup improves performance."""
        import time
        
        # Simulate slower first runs
        call_count = 0
        def mock_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # First two calls are slower
                time.sleep(0.1)
            else:
                time.sleep(0.01)
            return [np.random.rand(1, 384)]
        
        mock_onnx_session.run = mock_run
        
        with patch('onnxruntime.InferenceSession', return_value=mock_onnx_session):
            with patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
                from api.main import generate_embedding
                
                # Warmup runs
                for _ in range(2):
                    with performance_timer:
                        generate_embedding("warmup", mock_onnx_session, mock_tokenizer)
                
                warmup_time = performance_timer.average()
                
                # Production runs
                performance_timer.times = []  # Reset
                for _ in range(5):
                    with performance_timer:
                        generate_embedding("production", mock_onnx_session, mock_tokenizer)
                
                production_time = performance_timer.average()
                
                # Production runs should be faster
                assert production_time < warmup_time
    
    def test_invalid_input_handling(self, mock_onnx_session, mock_tokenizer):
        """Test handling of invalid inputs."""
        with patch('onnxruntime.InferenceSession', return_value=mock_onnx_session):
            with patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
                from api.main import generate_embedding
                
                # Test empty string
                embedding = generate_embedding("", mock_onnx_session, mock_tokenizer)
                assert isinstance(embedding, list)
                assert len(embedding) == 384
                
                # Test None (should raise error)
                with pytest.raises(AttributeError):
                    generate_embedding(None, mock_onnx_session, mock_tokenizer)
                
                # Test very long string
                long_query = "python " * 1000
                embedding = generate_embedding(long_query, mock_onnx_session, mock_tokenizer)
                assert len(embedding) == 384
    
    def test_memory_cleanup(self, mock_onnx_session, mock_tokenizer):
        """Test memory cleanup after embedding generation."""
        import gc
        import weakref
        
        with patch('onnxruntime.InferenceSession', return_value=mock_onnx_session):
            with patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
                from api.main import generate_embedding
                
                # Create embedding and track with weak reference
                embedding = generate_embedding("test", mock_onnx_session, mock_tokenizer)
                weak_ref = weakref.ref(embedding)
                
                # Should exist initially
                assert weak_ref() is not None
                
                # Delete and force garbage collection
                del embedding
                gc.collect()
                
                # Should be cleaned up
                assert weak_ref() is None
    
    @pytest.mark.asyncio
    async def test_concurrent_embedding_generation(self, mock_onnx_session, mock_tokenizer):
        """Test concurrent embedding generation doesn't cause issues."""
        with patch('onnxruntime.InferenceSession', return_value=mock_onnx_session):
            with patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
                from api.main import generate_embedding
                
                # Create multiple concurrent tasks
                queries = [f"query {i}" for i in range(10)]
                
                async def async_generate(query):
                    return generate_embedding(query, mock_onnx_session, mock_tokenizer)
                
                # Run concurrently
                results = await asyncio.gather(*[async_generate(q) for q in queries])
                
                assert len(results) == 10
                assert all(len(emb) == 384 for emb in results)
                assert all(isinstance(emb, list) for emb in results)


def generate_embedding(query, model, tokenizer):
    """Mock implementation of generate_embedding for testing."""
    if query is None:
        raise AttributeError("Query cannot be None")
    
    # Mock tokenization
    tokens = tokenizer.encode(query)
    
    # Mock model inference
    output = model.run(None, {"input_ids": np.array([tokens])})
    
    # Extract and normalize embeddings
    embeddings = output[0][0]
    
    # L2 normalization
    norm = np.linalg.norm(embeddings)
    if norm > 0:
        embeddings = embeddings / norm
    
    return embeddings.tolist()