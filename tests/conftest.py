"""
Pytest configuration and shared fixtures for the test suite.
"""
import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock

import pytest
import aiohttp
from fastapi.testclient import TestClient


# Configure event loop for async tests
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# API Fixtures
@pytest.fixture
def api_client():
    """Create a test client for the FastAPI app."""
    from api.main import app
    return TestClient(app)


@pytest.fixture
async def async_api_client():
    """Create an async test client for the FastAPI app."""
    from api.main import app
    from httpx import AsyncClient
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


# Mock Fixtures
@pytest.fixture
def mock_vespa_client():
    """Mock Vespa client for testing without real Vespa instance."""
    mock_client = Mock()
    mock_session = AsyncMock()
    
    # Mock successful search response
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={
        "root": {
            "fields": {
                "totalCount": 10
            },
            "children": [
                {
                    "id": "doc1",
                    "relevance": 0.9,
                    "fields": {
                        "title": "Test Document",
                        "content": "This is test content",
                        "url": "https://example.com/doc1",
                        "snippet": "Test snippet",
                        "code_snippets": ["print('hello')"]
                    }
                }
            ]
        }
    })
    
    mock_session.post = AsyncMock(return_value=mock_response)
    mock_session.get = AsyncMock(return_value=mock_response)
    mock_session.close = AsyncMock()
    
    mock_client.session = mock_session
    return mock_client


@pytest.fixture
def mock_onnx_model():
    """Mock ONNX model for testing."""
    mock_model = Mock()
    mock_session = Mock()
    
    # Mock embedding generation
    def mock_run(output_names, input_dict):
        batch_size = input_dict['input_ids'].shape[0]
        # Return 384-dimensional embeddings
        return [[[0.1] * 384] * batch_size]
    
    mock_session.run = mock_run
    mock_model.create_inference_session = Mock(return_value=mock_session)
    
    return mock_model


# Test Data Fixtures
@pytest.fixture
def sample_documents():
    """Load sample documents for testing."""
    return [
        {
            "title": "Python Lists Documentation",
            "url": "https://docs.python.org/3/tutorial/lists.html",
            "content": "Lists are mutable sequences, typically used to store collections of homogeneous items.",
            "code_snippets": ["fruits = ['apple', 'banana', 'cherry']"],
            "domain": "docs.python.org"
        },
        {
            "title": "JavaScript Arrays",
            "url": "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array",
            "content": "The Array object, as with arrays in other programming languages, enables storing a collection of multiple items.",
            "code_snippets": ["const fruits = ['Apple', 'Banana'];"],
            "domain": "developer.mozilla.org"
        }
    ]


@pytest.fixture
def test_queries():
    """Sample search queries for testing."""
    return {
        "programming": [
            "python list comprehension",
            "javascript async await",
            "how to sort array",
            "exception handling",
            "recursion examples"
        ],
        "edge_cases": [
            "",  # empty query
            "a",  # single character
            "SELECT * FROM users WHERE id=1; DROP TABLE users;--",  # SQL injection attempt
            "🐍 python unicode",  # unicode
            "x" * 1000  # very long query
        ]
    }


@pytest.fixture
def test_embeddings():
    """Pre-generated embeddings for testing."""
    return {
        "python list comprehension": [0.1] * 384,
        "javascript async await": [0.2] * 384,
        "how to sort array": [0.15] * 384
    }


@pytest.fixture
def crawler_state():
    """Sample crawler state for testing."""
    return {
        "visited": [
            "https://docs.python.org/3/",
            "https://docs.python.org/3/tutorial/",
            "https://developer.mozilla.org/"
        ],
        "queue": [
            "https://docs.python.org/3/library/",
            "https://developer.mozilla.org/en-US/docs/Web/"
        ],
        "domain_counts": {
            "docs.python.org": 2,
            "developer.mozilla.org": 1
        },
        "total_pages": 3
    }


# File System Fixtures
@pytest.fixture
def temp_crawl_dir():
    """Create a temporary directory for crawled data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        crawl_dir = Path(tmpdir) / "crawled_data"
        crawl_dir.mkdir(parents=True)
        yield crawl_dir


@pytest.fixture
def sample_crawl_data(temp_crawl_dir):
    """Create sample crawl data in temporary directory."""
    crawl_subdir = temp_crawl_dir / "crawl_2024-01-01T12-00-00"
    crawl_subdir.mkdir(parents=True)
    
    # Create sample JSON files
    for i in range(3):
        doc_data = {
            "title": f"Test Document {i}",
            "url": f"https://example.com/doc{i}",
            "content": f"This is test content for document {i}",
            "code_snippets": [f"print('Document {i}')"],
            "domain": "example.com"
        }
        
        with open(crawl_subdir / f"doc_{i}.json", "w") as f:
            json.dump(doc_data, f)
    
    return crawl_subdir


# Mock External Services
@pytest.fixture
def mock_language_detector():
    """Mock language detection service."""
    def detect_language(text):
        # Simple mock: return 'en' for English-like text
        if any(word in text.lower() for word in ['the', 'is', 'are', 'python', 'javascript']):
            return 'en'
        return 'unknown'
    
    return detect_language


@pytest.fixture
async def mock_http_session():
    """Mock aiohttp session for testing."""
    session = AsyncMock(spec=aiohttp.ClientSession)
    response = AsyncMock()
    response.status = 200
    response.text = AsyncMock(return_value="<html><body>Test content</body></html>")
    response.json = AsyncMock(return_value={"status": "ok"})
    
    session.get = AsyncMock(return_value=response)
    session.post = AsyncMock(return_value=response)
    session.close = AsyncMock()
    
    return session


# Performance Testing Fixtures
@pytest.fixture
def performance_timer():
    """Timer for performance testing."""
    import time
    
    class Timer:
        def __init__(self):
            self.times = []
        
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        
        def __exit__(self, *args):
            self.end = time.perf_counter()
            self.elapsed = self.end - self.start
            self.times.append(self.elapsed)
        
        def average(self):
            return sum(self.times) / len(self.times) if self.times else 0
        
        def median(self):
            sorted_times = sorted(self.times)
            n = len(sorted_times)
            if n == 0:
                return 0
            if n % 2 == 0:
                return (sorted_times[n//2-1] + sorted_times[n//2]) / 2
            return sorted_times[n//2]
    
    return Timer()


# Environment Setup
@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("VESPA_URL", "http://test-vespa:8080")
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")


# Cleanup Fixtures
@pytest.fixture(autouse=True)
async def cleanup_async_tasks():
    """Ensure all async tasks are cleaned up after each test."""
    yield
    # Cancel any remaining tasks
    tasks = [t for t in asyncio.all_tasks() if not t.done()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)