#!/usr/bin/env python3
"""
Test script for cloud API performance benchmarking
"""
import requests
import time
import json
import statistics
from datetime import datetime
from typing import List, Dict, Any
import argparse

# Default API endpoint
DEFAULT_API_URL = "https://mini-search-engine-api-543728486451.us-central1.run.app"

# Test queries - varied to avoid cache hits
TEST_QUERIES = [
    "react useState hook tutorial",
    "nodejs express middleware",
    "python flask routing",
    "java spring boot configuration",
    "postgresql query optimization",
    "mongodb aggregation pipeline",
    "redis cache implementation",
    "graphql schema design",
    "webpack module bundling",
    "nginx reverse proxy setup",
    "terraform infrastructure code",
    "ansible playbook examples",
    "git rebase workflow",
    "docker multi-stage builds",
    "kubernetes service mesh",
    "elasticsearch text search",
    "apache kafka streaming",
    "jenkins pipeline syntax",
    "prometheus metrics collection",
    "grafana dashboard creation"
]

# Number of runs per query for averaging
RUNS_PER_QUERY = 1


def test_search(api_url: str, query: str, search_type: str = "hybrid") -> Dict[str, Any]:
    """Test a single search query"""
    url = f"{api_url}/api/search"
    params = {
        "q": query,
        "search_type": search_type
    }
    
    try:
        start_time = time.time()
        response = requests.get(url, params=params, timeout=30)
        total_time = (time.time() - start_time) * 1000  # Convert to ms
        
        if response.status_code == 200:
            data = response.json()
            return {
                "status": "success",
                "query": query,
                "search_type": search_type,
                "total_time_ms": round(total_time, 2),
                "api_reported_time_ms": data.get("search_time_ms", 0),
                "vespa_querytime_ms": data.get("vespa_querytime_ms", 0),
                "vespa_searchtime_ms": data.get("vespa_searchtime_ms", 0),
                "embedding_time_ms": data.get("embedding_time_ms", 0),
                "result_count": data.get("total_hits", 0),
                "status_code": response.status_code
            }
        else:
            return {
                "status": "error",
                "query": query,
                "search_type": search_type,
                "error": f"HTTP {response.status_code}",
                "response_text": response.text[:200]
            }
    except Exception as e:
        return {
            "status": "error",
            "query": query,
            "search_type": search_type,
            "error": str(e)
        }


def run_performance_tests(api_url: str) -> Dict[str, Any]:
    """Run comprehensive performance tests"""
    print(f"Testing Cloud API at: {api_url}")
    print("=" * 60)
    
    results = {
        "api_url": api_url,
        "test_timestamp": datetime.now().isoformat(),
        "test_queries": TEST_QUERIES,
        "runs_per_query": RUNS_PER_QUERY,
        "search_types": {},
        "summary": {}
    }
    
    # Warm up the API with a few requests
    print("\nWarming up API...")
    for _ in range(3):
        test_search(api_url, "test query", "hybrid")
        time.sleep(0.5)
    
    # Test only hybrid search
    for search_type in ["hybrid"]:
        print(f"\nTesting {search_type.upper()} search:")
        print("-" * 50)
        
        search_results = []
        
        for query in TEST_QUERIES:
            query_times = []
            
            # Run query once
            result = test_search(api_url, query, search_type)
            if result["status"] == "success":
                query_time = result["api_reported_time_ms"]
                print(f"  Query: '{query}' - {query_time:.2f}ms")
                search_results.append({
                    "query": query,
                    "time_ms": round(query_time, 2),
                    "vespa_time_ms": result.get("vespa_searchtime_ms", 0),
                    "embedding_time_ms": result.get("embedding_time_ms", 0),
                    "result_count": result.get("result_count", 0)
                })
            else:
                print(f"  Query: '{query}' - ERROR: {result.get('error', 'Unknown error')}")
            
            time.sleep(0.5)  # Small delay between requests
        
        # Calculate statistics for this search type
        if search_results:
            all_times = [r["time_ms"] for r in search_results]
            results["search_types"][search_type] = {
                "queries": search_results,
                "statistics": {
                    "average_ms": round(statistics.mean(all_times), 2),
                    "median_ms": round(statistics.median(all_times), 2),
                    "min_ms": round(min(all_times), 2),
                    "max_ms": round(max(all_times), 2),
                    "stdev_ms": round(statistics.stdev(all_times), 2) if len(all_times) > 1 else 0,
                    "p95_ms": round(sorted(all_times)[int(len(all_times) * 0.95)], 2) if all_times else 0,
                    "under_50ms_count": sum(1 for t in all_times if t < 50),
                    "under_50ms_percentage": round(sum(1 for t in all_times if t < 50) / len(all_times) * 100, 1)
                }
            }
    
    # Test cold start performance
    print("\nTesting cold start performance...")
    print("-" * 50)
    
    # Wait a bit to let the instance potentially scale down
    print("Waiting 60 seconds for potential instance cooldown...")
    time.sleep(60)
    
    cold_start_results = []
    for i in range(3):
        print(f"Cold start test {i+1}/3...")
        result = test_search(api_url, "cold start test", "hybrid")
        if result["status"] == "success":
            cold_start_results.append(result["api_reported_time_ms"])
            print(f"  Response time: {result['api_reported_time_ms']:.2f}ms")
        time.sleep(30)  # Wait between cold start tests
    
    if cold_start_results:
        results["cold_start_tests"] = {
            "times_ms": cold_start_results,
            "average_ms": round(statistics.mean(cold_start_results), 2),
            "max_ms": round(max(cold_start_results), 2)
        }
    
    # Generate overall summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    for search_type, data in results["search_types"].items():
        stats = data["statistics"]
        print(f"\n{search_type.upper()} Search:")
        print(f"  Average: {stats['average_ms']}ms")
        print(f"  Median: {stats['median_ms']}ms")
        print(f"  Range: {stats['min_ms']}ms - {stats['max_ms']}ms")
        print(f"  Under 50ms: {stats['under_50ms_count']}/{len(data['queries'])} ({stats['under_50ms_percentage']}%)")
    
    if "cold_start_tests" in results:
        cold_data = results["cold_start_tests"]
        print(f"\nCold Start Performance:")
        print(f"  Average: {cold_data['average_ms']}ms")
        print(f"  Maximum: {cold_data['max_ms']}ms")
    
    # Overall summary
    all_times = []
    for search_type_data in results["search_types"].values():
        all_times.extend([q["time_ms"] for q in search_type_data["queries"]])
    
    if all_times:
        results["summary"] = {
            "total_queries_tested": len(all_times),
            "overall_average_ms": round(statistics.mean(all_times), 2),
            "overall_median_ms": round(statistics.median(all_times), 2),
            "overall_min_ms": round(min(all_times), 2),
            "overall_max_ms": round(max(all_times), 2),
            "overall_under_50ms_percentage": round(sum(1 for t in all_times if t < 50) / len(all_times) * 100, 1)
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test cloud API performance")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="API URL to test")
    parser.add_argument("--output", default="cloud_performance_results.json", help="Output JSON file")
    parser.add_argument("--skip-cold-start", action="store_true", help="Skip cold start tests")
    
    args = parser.parse_args()
    
    # Run tests
    results = run_performance_tests(args.api_url)
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")
    
    # Display key metrics
    if "summary" in results:
        summary = results["summary"]
        print(f"\nKey Metrics:")
        print(f"  Overall Average: {summary['overall_average_ms']}ms")
        print(f"  SLA Compliance (under 50ms): {summary['overall_under_50ms_percentage']}%")


if __name__ == "__main__":
    main()