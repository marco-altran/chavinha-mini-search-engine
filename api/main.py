from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict
from contextlib import asynccontextmanager
import aiohttp
import asyncio
from datetime import datetime
import json
from pathlib import Path
import logging
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import time
import ssl

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up API server...")
    setup_ssl_context()
    setup_http_session()
    setup_onnx_model()
    warmup_model()
    # Warm up Vespa connection
    await warmup_vespa_connection()
    logger.info("API server ready")
    yield
    # Shutdown
    logger.info("Shutting down API server...")
    await cleanup_http_session()

app = FastAPI(title="Mini Search Engine API", version="1.0.0", lifespan=lifespan)

# Mount static files (if directory exists)
import os
static_dir = "api/static"
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Global variables
tokenizer = None
ort_session = None
vespa_url = os.getenv("VESPA_URL", "http://localhost:4080")
vespa_token = os.getenv("VESPA_CLOUD_SECRET_TOKEN")

# Global HTTP session for connection pooling
global_session = None
ssl_context = None

def setup_ssl_context():
    """Setup SSL context for mTLS"""
    global ssl_context
    
    if vespa_url.startswith("https://"):
        ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        cert_path = "/app/certs/data-plane-public-cert.pem"
        key_path = "/app/certs/data-plane-private-key.pem"
        
        if os.path.exists(cert_path) and os.path.exists(key_path):
            try:
                ssl_context.load_cert_chain(cert_path, key_path)
                logger.info("Loaded mTLS certificates for Vespa Cloud")
                logger.info(f"Certificate path: {cert_path}")
                logger.info(f"Key path: {key_path}")
            except Exception as e:
                logger.error(f"Failed to load certificates: {e}")
                ssl_context = None
        else:
            logger.warning(f"mTLS certificates not found at {cert_path} or {key_path}")
            logger.warning("Available files in /app/certs/:")
            if os.path.exists("/app/certs"):
                for f in os.listdir("/app/certs"):
                    logger.warning(f"  - {f}")
            ssl_context = None

def setup_http_session():
    """Setup global HTTP session with optimized connection pooling"""
    global global_session, ssl_context
    
    # Detect deployment environment
    is_cloud_deployment = vespa_url.startswith("https://") or os.getenv("CLOUD_RUN_SERVICE") is not None
    
    if is_cloud_deployment:
        # Cloud deployment optimizations (Google Cloud Run to Vespa Cloud)
        connector = aiohttp.TCPConnector(
            ssl=ssl_context,
            limit=50,
            limit_per_host=30,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=120,
            enable_cleanup_closed=True,
            force_close=False,  # Enable connection reuse
            resolver=aiohttp.AsyncResolver(),
            # Cloud networking optimizations
            happy_eyeballs_delay=0.25,  # Enable IPv4/IPv6 dual stack
        )
        
        timeout = aiohttp.ClientTimeout(
            total=30,
            connect=5,
            sock_read=15,
            sock_connect=3
        )
        
        keepalive_header = "timeout=120, max=200"
        log_msg = "HTTP session with cloud deployment optimization initialized"
        
    else:
        # Local development optimizations (localhost to Docker)
        connector = aiohttp.TCPConnector(
            ssl=ssl_context,
            limit=20,  # Reduced total pool size for localhost
            limit_per_host=10,  # Fewer connections for localhost
            ttl_dns_cache=600,  # Longer DNS cache for localhost (10 minutes)
            use_dns_cache=True,
            keepalive_timeout=300,  # Longer keepalive for persistent local connections
            enable_cleanup_closed=True,
            force_close=False,  # Critical: Enable connection reuse
            resolver=aiohttp.AsyncResolver(),
            # Local networking optimizations
            local_addr=None,  # Let system choose optimal local address
            happy_eyeballs_delay=None,  # Disable for localhost
        )
        
        # Localhost-optimized timeout settings
        timeout = aiohttp.ClientTimeout(
            total=10,  # Reduced total timeout for localhost
            connect=2,  # Fast connection timeout for localhost
            sock_read=5,  # Reduced socket read timeout
            sock_connect=1  # Very fast socket connect for localhost
        )
        
        keepalive_header = "timeout=300, max=100"
        log_msg = "HTTP session with localhost optimization initialized"
    
    global_session = aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        # Headers optimized for persistent connections
        headers={
            "Connection": "keep-alive",
            "Keep-Alive": keepalive_header
        }
    )
    
    logger.info(log_msg)

async def cleanup_http_session():
    """Cleanup global HTTP session"""
    global global_session
    if global_session:
        await global_session.close()
        logger.info("HTTP session closed")


def clean_snippet(snippet: str) -> str:
    """Remove highlighting and separator tags from snippet"""
    import re
    # Remove <hi>, </hi>, and <sep /> tags
    cleaned = re.sub(r'</?hi>', '', snippet)
    cleaned = re.sub(r'<sep\s*/>', ' ... ', cleaned)
    return cleaned


class SearchResult(BaseModel):
    id: str
    url: str
    title: str
    snippet: str
    domain: str
    relevance: float
    doc_type: str


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_hits: int
    search_time_ms: float
    search_type: str
    embedding_time_ms: Optional[float] = None
    vespa_querytime_ms: Optional[float] = None
    vespa_summaryfetchtime_ms: Optional[float] = None
    vespa_searchtime_ms: Optional[float] = None


class DomainStats(BaseModel):
    domain: str
    document_count: int


class StatsResponse(BaseModel):
    total_documents: int
    domains: List[DomainStats]
    last_crawl: Optional[str]
    last_index: Optional[str]


def setup_onnx_model():
    """Setup ONNX model with optimizations"""
    global tokenizer, ort_session
    
    model_path = "api/models/all-MiniLM-L6-v2-onnx"
    
    # Check if ONNX model exists, if not, export it
    if not os.path.exists(model_path):
        logger.info("ONNX model not found, exporting from sentence-transformers...")
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        
        model_id = "sentence-transformers/all-MiniLM-L6-v2"
        model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
        model.save_pretrained(model_path)
        
        tokenizer_temp = AutoTokenizer.from_pretrained(model_id)
        tokenizer_temp.save_pretrained(model_path)
        logger.info(f"Model exported to {model_path}")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Create ONNX session with optimization
    logger.info("Creating ONNX inference session...")
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.intra_op_num_threads = 1  # Single thread for consistent latency
    
    ort_session = ort.InferenceSession(
        f"{model_path}/model.onnx",
        session_options,
        providers=['CPUExecutionProvider']
    )
    
    logger.info("ONNX model loaded successfully")


def warmup_model():
    """Warmup the model with sample queries"""
    logger.info("Warming up ONNX model...")
    warmup_queries = [
        "test query",
        "angular framework",
        "python programming",
        "javascript tutorial",
        "docker container"
    ]
    
    for query in warmup_queries:
        _ = encode_query(query)
    
    logger.info("Model warmup complete")

async def warmup_vespa_connection():
    """Warm up Vespa connection to reduce cold start latency"""
    logger.info("Warming up Vespa connection...")
    try:
        # Fire multiple warm-up queries to establish persistent connection
        headers = {}
        if vespa_token:
            headers["Authorization"] = f"Bearer {vespa_token}"
        
        warmup_params = {
            "yql": "select * from sources * where doc_type contains 'full_doc' limit 1",
            "hits": 1
        }
        
        # Perform multiple warmup requests to establish connection pool
        for i in range(3):
            try:
                async with global_session.get(
                    f"{vespa_url}/search/",
                    params=warmup_params,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        if i == 0:  # Only log first success
                            logger.info("Vespa connection warmed up successfully")
                    else:
                        logger.warning(f"Vespa warmup attempt {i+1} returned status {response.status}")
            except Exception as e:
                logger.warning(f"Vespa warmup attempt {i+1} failed: {e}")
        
        # Also fire a simple health check to establish TCP connection
        try:
            async with global_session.get(f"{vespa_url}/state/v1/health", headers=headers) as response:
                if response.status == 200:
                    logger.info("Vespa health check passed and TCP connection established")
        except Exception as e:
            logger.debug(f"Health check preconnect failed: {e}")
            
    except Exception as e:
        logger.warning(f"Vespa warmup failed: {e}")


def encode_query(text: str) -> np.ndarray:
    """Encode query text to embedding using ONNX"""
    # Tokenize
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="np", max_length=512)
    
    # Run inference
    outputs = ort_session.run(None, dict(inputs))
    
    # Mean pooling
    embeddings = outputs[0]
    attention_mask = inputs['attention_mask']
    mask_expanded = np.expand_dims(attention_mask, -1)
    
    sum_embeddings = np.sum(embeddings * mask_expanded, 1)
    sum_mask = np.clip(mask_expanded.sum(1), a_min=1e-9, a_max=None)
    mean_pooled = sum_embeddings / sum_mask
    
    # Normalize
    norm = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
    normalized = mean_pooled / norm
    
    return normalized[0]




async def generate_query_embedding(query: str) -> tuple[List[float], float]:
    """Generate embedding for search query and return timing"""
    if ort_session is None:
        logger.error("ONNX model not loaded!")
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        start_time = time.time()
        embedding = encode_query(query)
        end_time = time.time()
        embedding_time_ms = (end_time - start_time) * 1000
        logger.info(f"Embedding generation took {embedding_time_ms:.2f}ms for query: '{query}'")
        return embedding.tolist(), embedding_time_ms
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise HTTPException(status_code=500, detail="Embedding generation failed")


async def search_vespa(query: str, search_type: str = "hybrid", limit: int = 10, performance_mode: str = "ultra") -> Dict:
    """Perform search on Vespa"""
    start_time = datetime.now()
    
    # Determine ranking profile based on performance mode
    ranking_suffix = "_ultra" if performance_mode == "ultra" else ""
    
    # Prepare search request based on search type
    if search_type == "semantic":
        # Search chunks only for semantic search
        search_params = {
            "yql": f"select * from sources * where {{targetHits: {limit}}}nearestNeighbor(chunk_embedding, query_embedding) and doc_type contains 'chunk'",
            "ranking": f"semantic{'_chunks' if not ranking_suffix else '_ultra'}",
            "hits": limit,
            "summary": "chunk_summary",
            "presentation.timing": "true"
        }
    elif search_type == "hybrid":
        # For hybrid, use BM25 on full docs with embedding boost
        search_params = {
            "yql": f"select * from sources * where userQuery() limit {limit}",
            "query": query,
            "ranking": f"hybrid{ranking_suffix}",
            "hits": limit,
            "summary": "dynamic_snippet",
            "presentation.timing": "true"
        }
    else:  # bm25
        # Search full documents only for BM25
        search_params = {
            "yql": f"select * from sources * where doc_type contains 'full_doc' and userQuery() limit {limit}",
            "query": query,
            "ranking": f"bm25{'_full' if not ranking_suffix else '_ultra'}",
            "hits": limit,
            "summary": "dynamic_snippet",
            "presentation.timing": "true"
        }
    
    # Add query embedding for semantic and hybrid search
    embedding_time_ms = None
    if search_type in ["semantic", "hybrid"]:
        embedding, embedding_time_ms = await generate_query_embedding(query)
        search_params["input.query(query_embedding)"] = json.dumps(embedding)
    
    # Prepare headers for authentication and compression
    headers = {
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json"
    }
    if vespa_token:
        headers["Authorization"] = f"Bearer {vespa_token}"
    
    # Perform search using global session
    try:
        vespa_start_time = datetime.now()
        async with global_session.get(
            f"{vespa_url}/search/",
            params=search_params,
            headers=headers
        ) as response:
                if response.status == 200:
                    response_received_time = datetime.now()
                    data = await response.json()
                    json_parsed_time = datetime.now()
                    vespa_end_time = datetime.now()
                    
                    # Detailed timing breakdown
                    vespa_search_time = (vespa_end_time - vespa_start_time).total_seconds() * 1000
                    network_time = (response_received_time - vespa_start_time).total_seconds() * 1000
                    json_parse_time = (json_parsed_time - response_received_time).total_seconds() * 1000
                    total_search_time = (vespa_end_time - start_time).total_seconds() * 1000
                    
                    # Extract Vespa native timing (convert from seconds to ms)
                    vespa_timing = data.get("timing", {})
                    vespa_querytime_ms = vespa_timing.get("querytime", 0) * 1000
                    vespa_summaryfetchtime_ms = vespa_timing.get("summaryfetchtime", 0) * 1000
                    vespa_searchtime_ms = vespa_timing.get("searchtime", 0) * 1000
                    
                    # Calculate overhead
                    network_overhead = network_time - vespa_searchtime_ms
                    
                    # Log normal timing
                    logger.info(f"Vespa search took {vespa_search_time:.2f}ms for query: '{query}' (type: {search_type})")
                    logger.info(f"Detailed timing - network: {network_time:.2f}ms, json_parse: {json_parse_time:.2f}ms")
                    logger.info(f"Vespa native timing - query: {vespa_querytime_ms:.2f}ms, summary: {vespa_summaryfetchtime_ms:.2f}ms, total: {vespa_searchtime_ms:.2f}ms")
                    
                    # Detect and log timing anomalies
                    if network_overhead > 50:  # More than 50ms overhead is unusual
                        logger.warning(f"HIGH NETWORK OVERHEAD DETECTED: {network_overhead:.2f}ms for query '{query}'")
                        logger.warning(f"Network breakdown - total_network: {network_time:.2f}ms, vespa_native: {vespa_searchtime_ms:.2f}ms")
                        # Log connection pool stats if available
                        if hasattr(global_session.connector, '_conns'):
                            active_conns = len(global_session.connector._conns)
                            logger.warning(f"Connection pool stats - active_connections: {active_conns}")
                    elif network_overhead > 20:  # More than 20ms is notable
                        logger.info(f"Moderate network overhead: {network_overhead:.2f}ms for query '{query}'")
                    
                    return {
                        "results": data,
                        "search_time_ms": total_search_time,
                        "vespa_search_time_ms": vespa_search_time,
                        "vespa_querytime_ms": vespa_querytime_ms,
                        "vespa_summaryfetchtime_ms": vespa_summaryfetchtime_ms,
                        "vespa_searchtime_ms": vespa_searchtime_ms,
                        "embedding_time_ms": embedding_time_ms
                    }
                else:
                    logger.error(f"Vespa search failed: {response.status}")
                    text = await response.text()
                    logger.error(f"Response: {text}")
                    raise HTTPException(status_code=500, detail="Search service error")
    except aiohttp.ClientError as e:
        logger.error(f"Failed to connect to Vespa: {str(e)}")
        raise HTTPException(status_code=503, detail="Search service unavailable")


def format_search_results(vespa_response: Dict, query: str, search_type: str, performance_mode: str = "ultra") -> SearchResponse:
    """Format Vespa response into SearchResponse"""
    hits = vespa_response["results"].get("root", {}).get("children", [])
    total_hits = vespa_response["results"].get("root", {}).get("totalCount", 0)
    
    results = []
    seen_docs = set()  # Prevent duplicate documents in hybrid search
    
    for hit in hits:
        fields = hit.get("fields", {})
        doc_type = fields.get("doc_type", "full_doc")
        
        # Determine document ID and handle deduplication
        if doc_type == "chunk":
            # Use parent document ID for chunks to avoid duplicates
            document_id = fields.get("parent_id", fields.get("id", ""))
            
            # Skip if we already have this document
            if document_id in seen_docs:
                continue
            seen_docs.add(document_id)
            
            # Use chunk content as snippet for semantic search
            snippet = fields.get("chunk_content", "")
            result_id = document_id  # Use parent ID for result
            
        else:  # full_doc
            document_id = fields.get("id", "")
            
            # Skip if we already have this document
            if document_id in seen_docs:
                continue
            seen_docs.add(document_id)
            
            # Use dynamic snippet for BM25/hybrid search
            snippet = fields.get("dynamic_content", "")
            if not snippet:
                # Fallback to regular content if dynamic snippet not available
                content = fields.get("content", "")
                snippet = content[:250] + "..." if len(content) > 250 else content
            
                
            result_id = document_id
        
        # Create search result
        result = SearchResult(
            id=result_id,
            url=fields.get("url", ""),
            title=fields.get("title", "Untitled"),
            snippet=clean_snippet(snippet),
            domain=fields.get("domain", ""),
            relevance=hit.get("relevance", 0.0),
            doc_type=fields.get("parent_doc_type", fields.get("doc_type", "general"))
        )
        results.append(result)
    
    return SearchResponse(
        query=query,
        results=results,
        total_hits=len(results),  # Use actual deduplicated result count
        search_time_ms=vespa_response["search_time_ms"],
        search_type=search_type,
        embedding_time_ms=vespa_response.get("embedding_time_ms"),
        vespa_querytime_ms=vespa_response.get("vespa_querytime_ms"),
        vespa_summaryfetchtime_ms=vespa_response.get("vespa_summaryfetchtime_ms"),
        vespa_searchtime_ms=vespa_response.get("vespa_searchtime_ms")
    )


@app.get("/")
async def root():
    """Serve the main search interface"""
    with open("api/templates/index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/stats")
async def stats_page():
    """Serve the statistics page"""
    with open("api/templates/stats.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., description="Search query"),
    search_type: str = Query("hybrid", description="Search type: bm25, semantic, or hybrid"),
    limit: int = Query(10, ge=1, le=50, description="Number of results"),
    performance_mode: str = Query("ultra", description="Performance mode: normal or ultra")
):
    """
    Search for documents using different search types:
    - bm25: Traditional keyword-based search
    - semantic: Vector similarity search
    - hybrid: Combination of BM25 and semantic search
    """
    if search_type not in ["bm25", "semantic", "hybrid"]:
        raise HTTPException(status_code=400, detail="Invalid search type")
    
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Perform search
    vespa_response = await search_vespa(q, search_type, limit, performance_mode)
    
    # Format results
    return format_search_results(vespa_response, q, search_type, performance_mode)


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get statistics about indexed documents"""
    # Prepare headers for authentication and compression
    headers = {
        "Accept-Encoding": "gzip, deflate", 
        "Accept": "application/json"
    }
    if vespa_token:
        headers["Authorization"] = f"Bearer {vespa_token}"
    
    # Get stats from Vespa using global session
    try:
        # First, get total document count (only count full documents, not chunks)
        count_params = {
            "yql": 'select * from sources * where doc_type contains "full_doc" limit 0',
            "hits": 0
        }
        
        async with global_session.get(
            f"{vespa_url}/search/",
            params=count_params,
            headers=headers
        ) as response:
            if response.status != 200:
                logger.error(f"Failed to get document count: {response.status}")
                text = await response.text()
                logger.error(f"Response: {text}")
                raise HTTPException(status_code=500, detail="Failed to get statistics")
            
            count_data = await response.json()
            logger.info(f"Count query response: {count_data}")
            total_docs = count_data.get("root", {}).get("fields", {}).get("totalCount", 0)
            logger.info(f"Total documents: {total_docs}")
        
        # Get documents grouped by domain (only count full documents, not chunks)
        domain_params = {
            "yql": 'select * from sources * where doc_type contains "full_doc" | all(group(domain) max(1000) each(output(count())))',
            "hits": 0,
            "presentation.format": "json"
        }
        
        async with global_session.get(
            f"{vespa_url}/search/",
            params=domain_params,
            headers=headers
        ) as response:
            if response.status != 200:
                logger.error(f"Failed to get domain stats: {response.status}")
                text = await response.text()
                logger.error(f"Response: {text}")
                raise HTTPException(status_code=500, detail="Failed to get statistics")
            
            data = await response.json()
            
            # Parse group results
            domain_stats = []
            group_list = data.get("root", {}).get("children", [])
            
            if group_list and len(group_list) > 0:
                # The first child contains the group root
                group_root = group_list[0]
                if "children" in group_root:
                    # Look for the grouplist:domain
                    for child in group_root["children"]:
                        if child.get("id", "").startswith("grouplist:domain"):
                            # This contains the actual domain groups
                            for domain_group in child.get("children", []):
                                domain = domain_group.get("value")
                                count = domain_group.get("fields", {}).get("count()", 0)
                                if domain and count > 0:
                                    domain_stats.append(DomainStats(domain=domain, document_count=count))
            
            # Get crawl metadata
            crawl_info = await get_latest_crawl_info()
            
            # If total_docs is 0, calculate from domain stats
            if total_docs == 0 and domain_stats:
                total_docs = sum(d.document_count for d in domain_stats)
            
            return StatsResponse(
                total_documents=total_docs,
                domains=sorted(domain_stats, key=lambda x: x.document_count, reverse=True),
                last_crawl=crawl_info.get("last_crawl"),
                last_index=crawl_info.get("last_index")
            )
                
    except aiohttp.ClientError as e:
        logger.error(f"Failed to connect to Vespa: {str(e)}")
        raise HTTPException(status_code=503, detail="Statistics service unavailable")


async def get_latest_crawl_info() -> Dict:
    """Get information about the latest crawl"""
    # Try relative path from root first, then fallback to absolute path
    crawled_data_dir = Path("crawled_data")
    if not crawled_data_dir.exists():
        return {}
    
    # Find all metadata files
    metadata_files = []
    for crawl_dir in crawled_data_dir.iterdir():
        if crawl_dir.is_dir() and crawl_dir.name.startswith("crawl_"):
            metadata_file = crawl_dir / "metadata.json"
            if metadata_file.exists():
                metadata_files.append(metadata_file)
    
    if not metadata_files:
        return {}
    
    # Sort by modification time to get the latest
    latest_metadata = max(metadata_files, key=lambda f: f.stat().st_mtime)
    
    with open(latest_metadata, 'r') as f:
        metadata = json.load(f)
        return {
            "last_crawl": metadata.get("end_time"),
            "last_index": metadata.get("indexed_at")
        }
    
    return {}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Prepare headers for authentication and compression
    headers = {
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json" 
    }
    if vespa_token:
        headers["Authorization"] = f"Bearer {vespa_token}"
    
    # Check Vespa health using global session
    try:
        logger.info(f"Testing Vespa health at: {vespa_url}/state/v1/health")
        logger.info(f"Using SSL context: {ssl_context is not None}")
        async with global_session.get(f"{vespa_url}/state/v1/health", headers=headers) as response:
                logger.info(f"Vespa health response: {response.status}")
                vespa_healthy = response.status == 200
                if not vespa_healthy:
                    error_text = await response.text()
                    logger.error(f"Vespa health check failed: {error_text}")
    except Exception as e:
        logger.error(f"Vespa health check exception: {e}")
        vespa_healthy = False
    
    return {
        "status": "healthy" if vespa_healthy else "degraded",
        "vespa": "up" if vespa_healthy else "down",
        "api": "up"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)