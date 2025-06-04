# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a hybrid search engine for programming documentation that combines BM25 keyword search with semantic vector search using Vespa. The system uses a custom high-performance multi-threaded Python crawler, generates embeddings, and provides sub-50ms search responses through a FastAPI backend.

## Essential Commands

### Setup and Development
```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# RECOMMENDED: Quick start everything (handles all timing and setup)
./run_vespa.sh

# Manual setup (if needed):
# Start Vespa (required for all operations)
docker-compose down -v && ./run_vespa.sh 
```

### Clean State Rebuild
```bash
# Stop all services and remove containers/volumes
docker-compose down -v

# Remove all crawled data (optional)
rm -rf crawled_data/

# RECOMMENDED: Use quickstart for clean rebuild
./run_vespa.sh

# OR Manual rebuild:
# docker-compose up -d
# sleep 30
# ./deploy_vespa.sh

# Re-crawl documents
cd crawler && poetry run python doc_scraper.py

# Re-index with embeddings
poetry run python indexer/indexer.py

# Start API server
poetry run python api/main.py
```

### Crawling and Indexing
```bash
# Run crawler (from crawler directory)
cd crawler && poetry run python doc_scraper.py

# Run with custom domains file (default: test_domains.txt)
poetry run python doc_scraper.py --domains all_domains.txt

# Run with custom page limits
poetry run python doc_scraper.py --global-max-pages 10000 --max-pages-per-domain 100

# Run with web monitor dashboard
poetry run python doc_scraper.py --web-monitor

# Index documents to Vespa
poetry run python indexer/indexer.py

# Index with custom settings
poetry run python indexer/indexer.py --max-per-domain 5000 --batch-size 200 --embedding-batch-size 64
```

### Running the API
```bash
# Start API server (from project root)
poetry run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Or simply:
poetry run python api/main.py
```

### Testing
```bash
# Test performance
poetry run python test_performance.py

# Run all tests
poetry run pytest tests/
```

## Architecture

The system follows this pipeline:
1. **Crawler** (`crawler/doc_scraper.py`) - Custom multi-threaded Python crawler with up to 3000 workers
2. **Storage** (`crawled_data/{domain}/`) - JSON files organized by domain
3. **Indexer** (`indexer/indexer.py`) - Generates embeddings and indexes to Vespa
4. **Vespa** (`config/vespa/`) - Handles BM25 + vector search with HNSW index
5. **API** (`api/main.py`) - FastAPI endpoints for search and stats
6. **UI** (`api/templates/`) - Web interface for search

## Key Technical Details

- **Embeddings**: 384-dimensional vectors using sentence-transformers/all-MiniLM-L6-v2
- **Search Types**: BM25 (keyword), semantic (vector), hybrid (70% BM25 + 30% semantic)
- **Vespa Schema**: Located at `config/vespa/schemas/doc.sd` - defines fields and ranking profiles
- **Latency Target**: Sub-50ms for all search operations
- **Crawler Performance**: Up to 3000 concurrent workers with domain-aware rate limiting
- **Crawler Concurrency**: Default 50 concurrent requests per domain
- **State Management**: Automatic save/resume with `crawler_state.json`
- **Language Detection**: Filters for English content using langdetect

## Common Development Tasks

When modifying search behavior:
- Update Vespa schema in `config/vespa/schemas/doc.sd`
- Redeploy with `./deploy_vespa.sh`
- Test with `poetry run python test_performance.py`

When adding new documentation sources:
- Add domains to `crawler/test_domains.txt` or `crawler/all_domains.txt` (one per line)
- The crawler will automatically handle them with fair scheduling

When optimizing performance:
- Adjust crawler settings: `MAX_WORKERS`, `MAX_CONCURRENT_REQUESTS_PER_DOMAIN`, `DELAY_BETWEEN_REQUESTS`
- Tune indexer batching: `--batch-size`, `--embedding-batch-size`, `--max-concurrent-embeddings`
- Monitor latency in API responses

## Crawler Configuration

Edit `crawler/doc_scraper.py` or use command line arguments:
- `MAX_WORKERS`: Total concurrent workers (default: 3000)
- `MAX_CONCURRENT_REQUESTS_PER_DOMAIN`: Parallel requests per domain (default: 50)
- `DELAY_BETWEEN_REQUESTS`: Delay between requests to same domain (default: 0.1s)
- `REQUEST_TIMEOUT`: Request timeout in seconds (default: 8)
- `--global-max-pages`: Maximum total pages to crawl (default: 2,000,000)
- `--max-pages-per-domain`: Maximum pages per domain (default: 100)
- `--domains`: Path to domains file (default: test_domains.txt)
- `--web-monitor`: Enable web-based monitoring dashboard

## Troubleshooting Vespa Startup

**RECOMMENDED SOLUTION**: Use `./run_vespa.sh` which handles all timing automatically.

If you encounter Vespa startup issues with manual commands:
1. **Wait for initialization**: Vespa needs 20-30 seconds to start config server
2. **Deploy application**: Container needs an app package to run properly
3. **Check timing**: `docker-compose up -d && sleep 30 && ./deploy_vespa.sh`
4. **Verify health**: Config server (19071) should be up before deploying
5. **Check logs**: `docker logs vespa --tail 50` for detailed error info

## Testing Approach

Always test changes with:
1. Unit tests for specific components
2. End-to-end search quality with `test_performance.py`
3. Manual testing via web UI at http://localhost:8000
4. Crawler testing with limited page count: `poetry run python doc_scraper.py --max-pages-per-domain 10`

## Search Types Explained

1. **BM25 Search**: Traditional keyword-based search
   - Best for: Exact term matching, specific function/class names
   - Relevance scores: 2.5-5.0+ for strong keyword matches

2. **Semantic Search**: Vector similarity search using embeddings
   - Best for: Conceptual queries, finding similar content
   - Uses HNSW index for fast nearest neighbor search
   - Relevance scores: 0.45-0.47 (closeness function results)

3. **Hybrid Search**: Weighted combination (70% BM25, 30% semantic)
   - Best for: General purpose searching
   - Combines keyword precision with semantic understanding
   - Relevance scores: 1.8-4.5+ (highest overall scores)
   - **Recommended default option**

## Recent Improvements

### Indexer Optimization
- **Batched Embedding Generation**: Processes up to 32 texts simultaneously
- **Async Pipeline**: Background embedding processor with queue-based architecture
- **Per-Domain Limits**: `--max-per-domain` flag prevents index imbalance
- **GPU Utilization**: Improved from 60% to 80-90% through optimized batching

### Crawler Features
- **Performance**: Multi-threaded with up to 3000 concurrent workers
- **Rate Limiting**: Configurable per-domain concurrent requests and delays
- **State Management**: Automatic save/resume with crawler state
- **Language Detection**: Filters for English content
- **Rich Dashboard**: Real-time progress monitoring with domain statistics
- **Web Monitor**: Optional web-based dashboard for remote monitoring