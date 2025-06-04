"""
Integration tests for end-to-end search workflows.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import json


class TestSearchWorkflows:
    """Test complete search workflows from API to Vespa."""
    
    @pytest.fixture
    async def api_with_mocked_vespa(self, mock_vespa_client, mock_onnx_model):
        """Set up API with mocked Vespa client."""
        with patch('api.main.vespa_client', mock_vespa_client):
            with patch('api.main.model', mock_onnx_model):
                from api.main import app
                yield app
    
    @pytest.fixture
    def search_responses(self):
        """Mock search responses for different search types."""
        return {
            "bm25": {
                "root": {
                    "fields": {"totalCount": 3},
                    "children": [
                        {
                            "id": "doc1",
                            "relevance": 4.5,
                            "fields": {
                                "title": "Python List Methods",
                                "content": "Python lists have several built-in methods...",
                                "url": "https://docs.python.org/3/tutorial/lists.html",
                                "snippet": "Lists have methods like append(), extend(), remove()...",
                                "code_snippets": ["list.append(x)", "list.extend(iterable)"]
                            }
                        },
                        {
                            "id": "doc2",
                            "relevance": 3.2,
                            "fields": {
                                "title": "Working with Lists",
                                "content": "Lists are one of the most versatile data structures...",
                                "url": "https://realpython.com/python-lists/",
                                "snippet": "Python lists can contain different data types...",
                                "code_snippets": ["my_list = [1, 'hello', 3.14]"]
                            }
                        }
                    ]
                }
            },
            "semantic": {
                "root": {
                    "fields": {"totalCount": 2},
                    "children": [
                        {
                            "id": "doc3",
                            "relevance": 0.47,
                            "fields": {
                                "title": "Array Manipulation in Python",
                                "content": "Arrays and lists are fundamental data structures...",
                                "url": "https://numpy.org/doc/stable/user/basics.html",
                                "snippet": "NumPy arrays provide efficient operations...",
                                "code_snippets": ["import numpy as np", "arr = np.array([1, 2, 3])"]
                            }
                        }
                    ]
                }
            },
            "hybrid": {
                "root": {
                    "fields": {"totalCount": 5},
                    "children": [
                        {
                            "id": "doc1",
                            "relevance": 3.8,
                            "fields": {
                                "title": "Python List Methods",
                                "content": "Python lists have several built-in methods...",
                                "url": "https://docs.python.org/3/tutorial/lists.html",
                                "snippet": "Lists have methods like append(), extend(), remove()...",
                                "code_snippets": ["list.append(x)", "list.extend(iterable)"]
                            }
                        },
                        {
                            "id": "doc3",
                            "relevance": 2.5,
                            "fields": {
                                "title": "Array Manipulation in Python",
                                "content": "Arrays and lists are fundamental data structures...",
                                "url": "https://numpy.org/doc/stable/user/basics.html",
                                "snippet": "NumPy arrays provide efficient operations...",
                                "code_snippets": ["import numpy as np", "arr = np.array([1, 2, 3])"]
                            }
                        }
                    ]
                }
            }
        }
    
    @pytest.mark.asyncio
    async def test_bm25_search_workflow(self, async_api_client, api_with_mocked_vespa, search_responses):
        """Test complete BM25 search workflow."""
        # Configure mock response
        mock_vespa = api_with_mocked_vespa.state.vespa_client
        mock_vespa.session.post.return_value.json.return_value = search_responses["bm25"]
        
        # Perform search
        response = await async_api_client.get(
            "/api/search",
            params={"q": "python list methods", "search_type": "bm25", "limit": 10}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "results" in data
        assert "query" in data
        assert "search_type" in data
        assert "processing_time_ms" in data
        assert "total_results" in data
        
        # Verify results
        assert len(data["results"]) == 2
        assert data["total_results"] == 3
        assert data["search_type"] == "bm25"
        assert data["query"] == "python list methods"
        
        # Verify first result
        first_result = data["results"][0]
        assert first_result["title"] == "Python List Methods"
        assert first_result["url"] == "https://docs.python.org/3/tutorial/lists.html"
        assert "relevance_score" in first_result
        assert first_result["relevance_score"] > 0
        assert len(first_result["code_snippets"]) == 2
    
    @pytest.mark.asyncio
    async def test_semantic_search_workflow(self, async_api_client, api_with_mocked_vespa, search_responses):
        """Test complete semantic search workflow."""
        # Configure mock response
        mock_vespa = api_with_mocked_vespa.state.vespa_client
        mock_vespa.session.post.return_value.json.return_value = search_responses["semantic"]
        
        # Mock embedding generation
        with patch('api.main.generate_embedding', return_value=[0.1] * 384):
            response = await async_api_client.get(
                "/api/search",
                params={"q": "how to work with arrays", "search_type": "semantic", "limit": 10}
            )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify semantic search was performed
        assert data["search_type"] == "semantic"
        assert len(data["results"]) == 1
        assert "Array Manipulation" in data["results"][0]["title"]
        
        # Verify embedding was generated
        assert mock_vespa.session.post.called
        call_args = mock_vespa.session.post.call_args
        assert "yql" in call_args[1]["json"]
        assert "ranking.features.query(embedding)" in call_args[1]["json"]
    
    @pytest.mark.asyncio
    async def test_hybrid_search_workflow(self, async_api_client, api_with_mocked_vespa, search_responses):
        """Test complete hybrid search workflow."""
        # Configure mock response
        mock_vespa = api_with_mocked_vespa.state.vespa_client
        mock_vespa.session.post.return_value.json.return_value = search_responses["hybrid"]
        
        # Mock embedding generation
        with patch('api.main.generate_embedding', return_value=[0.15] * 384):
            response = await async_api_client.get(
                "/api/search",
                params={"q": "python lists and arrays", "search_type": "hybrid", "limit": 10}
            )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify hybrid search
        assert data["search_type"] == "hybrid"
        assert len(data["results"]) == 2
        
        # Verify ranking (hybrid should blend scores)
        assert data["results"][0]["relevance_score"] > data["results"][1]["relevance_score"]
        
        # Verify both BM25 and semantic results are included
        titles = [r["title"] for r in data["results"]]
        assert "Python List Methods" in titles
        assert "Array Manipulation in Python" in titles
    
    @pytest.mark.asyncio
    async def test_empty_results_handling(self, async_api_client, api_with_mocked_vespa):
        """Test handling of empty search results."""
        # Configure empty response
        empty_response = {"root": {"fields": {"totalCount": 0}, "children": []}}
        mock_vespa = api_with_mocked_vespa.state.vespa_client
        mock_vespa.session.post.return_value.json.return_value = empty_response
        
        response = await async_api_client.get(
            "/api/search",
            params={"q": "nonexistent query xyz123", "search_type": "bm25"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["results"] == []
        assert data["total_results"] == 0
        assert "processing_time_ms" in data
    
    @pytest.mark.asyncio
    async def test_error_response_handling(self, async_api_client, api_with_mocked_vespa):
        """Test handling of Vespa errors."""
        # Configure error response
        mock_vespa = api_with_mocked_vespa.state.vespa_client
        mock_vespa.session.post.side_effect = Exception("Vespa connection failed")
        
        response = await async_api_client.get(
            "/api/search",
            params={"q": "test query", "search_type": "bm25"}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Search failed" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_query_validation(self, async_api_client, api_with_mocked_vespa):
        """Test query parameter validation."""
        # Test empty query
        response = await async_api_client.get(
            "/api/search",
            params={"q": "", "search_type": "bm25"}
        )
        assert response.status_code == 422
        
        # Test invalid search type
        response = await async_api_client.get(
            "/api/search",
            params={"q": "test", "search_type": "invalid"}
        )
        assert response.status_code == 422
        
        # Test invalid limit
        response = await async_api_client.get(
            "/api/search",
            params={"q": "test", "search_type": "bm25", "limit": -1}
        )
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_performance_mode(self, async_api_client, api_with_mocked_vespa, search_responses):
        """Test performance mode search."""
        mock_vespa = api_with_mocked_vespa.state.vespa_client
        mock_vespa.session.post.return_value.json.return_value = search_responses["bm25"]
        
        # Test with performance mode
        response = await async_api_client.get(
            "/api/search",
            params={"q": "test", "search_type": "bm25", "performance_mode": True}
        )
        
        assert response.status_code == 200
        
        # Verify performance optimizations were applied
        call_args = mock_vespa.session.post.call_args
        yql = call_args[1]["json"]["yql"]
        assert "weakAnd" in yql or "timeout" in str(call_args)
    
    @pytest.mark.asyncio
    async def test_deduplication(self, async_api_client, api_with_mocked_vespa):
        """Test result deduplication."""
        # Create response with duplicate URLs
        duplicate_response = {
            "root": {
                "fields": {"totalCount": 3},
                "children": [
                    {
                        "id": "doc1",
                        "relevance": 4.5,
                        "fields": {
                            "title": "Title 1",
                            "url": "https://example.com/doc",
                            "content": "Content 1",
                            "snippet": "Snippet 1"
                        }
                    },
                    {
                        "id": "doc2",
                        "relevance": 3.2,
                        "fields": {
                            "title": "Title 2",
                            "url": "https://example.com/doc",  # Same URL
                            "content": "Content 2",
                            "snippet": "Snippet 2"
                        }
                    }
                ]
            }
        }
        
        mock_vespa = api_with_mocked_vespa.state.vespa_client
        mock_vespa.session.post.return_value.json.return_value = duplicate_response
        
        response = await async_api_client.get(
            "/api/search",
            params={"q": "test", "search_type": "bm25"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have deduplicated results
        assert len(data["results"]) == 1
        assert data["results"][0]["title"] == "Title 1"  # Higher relevance kept
    
    @pytest.mark.asyncio
    async def test_concurrent_searches(self, async_api_client, api_with_mocked_vespa, search_responses):
        """Test concurrent search requests."""
        mock_vespa = api_with_mocked_vespa.state.vespa_client
        mock_vespa.session.post.return_value.json.return_value = search_responses["bm25"]
        
        # Launch multiple concurrent searches
        queries = ["query1", "query2", "query3", "query4", "query5"]
        
        async def search(query):
            return await async_api_client.get(
                "/api/search",
                params={"q": query, "search_type": "bm25"}
            )
        
        # Execute concurrently
        responses = await asyncio.gather(*[search(q) for q in queries])
        
        # All should succeed
        assert all(r.status_code == 200 for r in responses)
        assert all(r.json()["query"] == q for r, q in zip(responses, queries))
    
    @pytest.mark.asyncio
    async def test_search_latency(self, async_api_client, api_with_mocked_vespa, search_responses, performance_timer):
        """Test search latency meets requirements."""
        mock_vespa = api_with_mocked_vespa.state.vespa_client
        mock_vespa.session.post.return_value.json.return_value = search_responses["bm25"]
        
        # Warm up
        await async_api_client.get("/api/search", params={"q": "warmup", "search_type": "bm25"})
        
        # Measure latency
        for _ in range(10):
            with performance_timer:
                response = await async_api_client.get(
                    "/api/search",
                    params={"q": "test query", "search_type": "bm25"}
                )
                assert response.status_code == 200
        
        # Check median latency is under 50ms
        median_latency = performance_timer.median() * 1000  # Convert to ms
        assert median_latency < 50, f"Median latency {median_latency}ms exceeds 50ms target"