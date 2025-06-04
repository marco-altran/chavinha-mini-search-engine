#!/usr/bin/env python3
"""
Performance testing script for search queries
"""
import asyncio
import aiohttp
import time
import statistics
import json
import ssl
import os
import argparse
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PerformanceConfig:
    """Performance testing configuration"""
    test_result_limit: int = 10
    base_url: str = "http://localhost:8000"
    test_repetitions: int = 3
    sleep_between_requests: float = 0.1


@dataclass
class Config:
    """Main configuration class"""
    performance: PerformanceConfig


def get_config() -> Config:
    """Get the configuration object"""
    return Config(performance=PerformanceConfig())


# Load configuration
config = get_config()

# Test queries
TEST_QUERIES = [
    "angular framework",
    "python async programming",
    "javascript promises",
    "docker container",
    "kubernetes deployment",
    "react hooks",
    "vue composition api",
    "typescript generics",
    "golang channels",
    "rust ownership"
]


def setup_ssl_context(cert_path: Optional[str] = None, key_path: Optional[str] = None) -> Optional[ssl.SSLContext]:
    """Setup SSL context for mTLS if certificates are provided"""
    if not cert_path or not key_path:
        return None
        
    if os.path.exists(cert_path) and os.path.exists(key_path):
        try:
            ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            ssl_context.load_cert_chain(cert_path, key_path)
            print(f"Loaded mTLS certificates from {cert_path} and {key_path}")
            return ssl_context
        except Exception as e:
            print(f"Failed to load certificates: {e}")
            return None
    else:
        print(f"Certificate files not found at {cert_path} or {key_path}")
        return None


async def test_search_performance(session: aiohttp.ClientSession, 
                                base_url: str,
                                query: str,
                                search_type: str,
                                headers: Optional[Dict] = None) -> Dict:
    """Test a single search query and return timing info"""
    start_time = time.time()
    
    params = {
        "q": query,
        "search_type": search_type,
        "limit": config.performance.test_result_limit
    }
    
    try:
        async with session.get(f"{base_url}/api/search", params=params, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                end_time = time.time()
                total_time_ms = (end_time - start_time) * 1000
                
                return {
                    "query": query,
                    "search_type": search_type,
                    "status": "success",
                    "total_time_ms": total_time_ms,
                    "search_time_ms": data.get("search_time_ms", 0),
                    "embedding_time_ms": data.get("embedding_time_ms", 0),
                    "result_count": len(data.get("results", []))
                }
            else:
                error_text = await response.text()
                print(f"HTTP {response.status} error for '{query}' ({search_type}): {error_text}")
                return {
                    "query": query,
                    "search_type": search_type,
                    "status": "error",
                    "error": f"HTTP {response.status}: {error_text}"
                }
    except Exception as e:
        print(f"Error during search for '{query}' ({search_type}): {str(e)}")
        return {
            "query": query,
            "search_type": search_type,
            "status": "error",
            "error": str(e)
        }


async def warmup_searches(session: aiohttp.ClientSession, base_url: str, headers: Optional[Dict] = None):
    """Perform warmup searches to ensure caches are populated"""
    print("Warming up...")
    warmup_queries = ["test", "search", "warm"]
    
    for query in warmup_queries:
        for search_type in ["bm25", "semantic", "hybrid"]:
            await test_search_performance(session, base_url, query, search_type, headers)
    
    print("Warmup complete\n")


async def run_performance_tests(base_url: str = None, 
                              ssl_context: Optional[ssl.SSLContext] = None,
                              auth_token: Optional[str] = None):
    """Run performance tests for all search types"""
    # Use config default if no base_url provided
    if base_url is None:
        base_url = config.performance.base_url
    
    # Prepare headers
    headers = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    
    # Configure connection with SSL context if provided
    connector = aiohttp.TCPConnector(
        ssl=ssl_context,
        limit=100,
        limit_per_host=20,
        ttl_dns_cache=300,
        use_dns_cache=True,
        keepalive_timeout=120,
        enable_cleanup_closed=True,
        force_close=False
    )
    
    timeout = aiohttp.ClientTimeout(
        total=30,
        connect=5,
        sock_read=10,
        sock_connect=5
    )
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Warmup
        await warmup_searches(session, base_url, headers)
        
        results = {
            "bm25": [],
            "semantic": [],
            "hybrid": []
        }
        
        # Test each search type
        for search_type in ["bm25", "semantic", "hybrid"]:
            print(f"\nTesting {search_type.upper()} search:")
            print("-" * 50)
            
            for query in TEST_QUERIES:
                # Run each query 3 times and take the median
                query_results = []
                
                for _ in range(config.performance.test_repetitions):
                    result = await test_search_performance(session, base_url, query, search_type, headers)
                    if result["status"] == "success":
                        query_results.append(result)
                    await asyncio.sleep(config.performance.sleep_between_requests)  # Configurable delay between requests
                
                if query_results:
                    # Take median of search times
                    search_times = [r["search_time_ms"] for r in query_results]
                    median_result = query_results[len(search_times)//2]
                    median_result["search_time_ms"] = statistics.median(search_times)
                    
                    results[search_type].append(median_result)
                    
                    print(f"Query: '{query}'")
                    print(f"  Search time: {median_result['search_time_ms']:.2f}ms")
                    if median_result.get("embedding_time_ms"):
                        print(f"  Embedding time: {median_result['embedding_time_ms']:.2f}ms")
                    print(f"  Results: {median_result['result_count']}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        
        for search_type, type_results in results.items():
            if type_results:
                search_times = [r["search_time_ms"] for r in type_results]
                embedding_times = [r.get("embedding_time_ms", 0) for r in type_results if r.get("embedding_time_ms")]
                
                print(f"\n{search_type.upper()} Search:")
                print(f"  Average search time: {statistics.mean(search_times):.2f}ms")
                print(f"  Median search time: {statistics.median(search_times):.2f}ms")
                print(f"  Min search time: {min(search_times):.2f}ms")
                print(f"  Max search time: {max(search_times):.2f}ms")
                print(f"  95th percentile: {sorted(search_times)[int(len(search_times) * 0.95)]:.2f}ms")
                
                if embedding_times:
                    print(f"  Average embedding time: {statistics.mean(embedding_times):.2f}ms")
                
                # Check how many queries are under 50ms
                under_50ms = sum(1 for t in search_times if t < 50)
                print(f"  Queries under 50ms: {under_50ms}/{len(search_times)} ({under_50ms/len(search_times)*100:.1f}%)")
        
        # Save detailed results
        with open("performance_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nDetailed results saved to performance_results.json")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="Test search API performance")
    parser.add_argument("--url", type=str, help="API base URL (e.g., https://api.example.com)")
    parser.add_argument("--cert", type=str, help="Path to client certificate file (default: certs/data-plane-public-cert.pem)")
    parser.add_argument("--key", type=str, help="Path to client key file (default: certs/data-plane-private-key.pem)")
    parser.add_argument("--token", type=str, help="Bearer token for authentication (for non-Vespa APIs)")
    
    args = parser.parse_args()
    
    # Setup SSL context if certificates are provided or if URL is HTTPS
    ssl_context = None
    auth_token = args.token
    
    if args.url and args.url.startswith("https://"):
        # For HTTPS URLs, check for certificates
        cert_path = args.cert or "certs/data-plane-public-cert.pem"
        key_path = args.key or "certs/data-plane-private-key.pem"
        
        # Try to load certificates for HTTPS
        ssl_context = setup_ssl_context(cert_path, key_path)
        if ssl_context:
            print(f"Using mTLS authentication with certificates")
            # Don't use token auth when using certificate auth
            auth_token = None
        elif args.token:
            print(f"Using bearer token authentication")
        else:
            print("Warning: HTTPS URL provided but no valid certificates or token found")
    elif args.cert and args.key:
        # Explicit certificate paths provided
        ssl_context = setup_ssl_context(args.cert, args.key)
    elif (args.cert and not args.key) or (args.key and not args.cert):
        print("Error: Both --cert and --key must be provided together")
        return
    
    # Run performance tests
    asyncio.run(run_performance_tests(
        base_url=args.url,
        ssl_context=ssl_context,
        auth_token=auth_token
    ))


if __name__ == "__main__":
    main()