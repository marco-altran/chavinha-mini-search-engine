# Chavinha - Mini Search Engine

This project was created as a [test project to these requirements](ORIGINAL_REQUIREMENTS.txt). Chavinha is a Brazilian word that means "little key" and takes inspiration from [Kagi](https://kagi.com/) that means "key" in Japanese.

Chavinha is a specialized search engine for programming documentation that uses hybrid search combining BM25 (keyword-based) and semantic (vector-based) search capabilities.

## Live Demo

~~**Working Demo**: [https://mini-search-engine-api-543728486451.us-central1.run.app/](https://mini-search-engine-api-543728486451.us-central1.run.app/)~~

*No longer available since June 14th, 2025*

### Cost-Optimized Cloud Architecture
- **API**: Google Cloud Run (auto-scaling, cold start optimization)
- **Search Engine**: Vespa Cloud Dev instance (2 vCPUs, 8GB RAM)
- **Total Cost**: ~$0/month (using free tiers and credits)
- **Latency**: 35-67ms typical (47.79ms average, 75% under 50ms)

## Requirements Fulfillment

This project was built to fulfill [these specific requirements](ORIGINAL_REQUIREMENTS.txt):

**Search Engine**: Vespa selected over Tantivy for native hybrid search capabilities  
**Latency SLA**: Sub-50ms achieved locally (9-26ms typical), cloud deployment optimized for cost  
**Crawler Selection**: Custom Python implementation chosen over Scrapy for performance and observability  
**Interface**: Modern web UI with search box, results, and statistics  
**Documentation**: Comprehensive installation, challenges, and solutions  
**Live Deployment**: Production-ready cloud deployment available

## Features

- **Hybrid Search**: Combines traditional BM25 keyword search with modern semantic search
- **Multi-domain Crawling**: Supports crawling from 160+ programming documentation sites
- **Fast Search**: Sub-50ms search latency with optimized Vespa backend
- **Simple Interface**: Clean web UI with search options and statistics
- **Scalable Architecture**: Built with custom Python crawler, FastAPI, and Vespa

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Python     │────▶│  JSON Files  │────▶│   Indexer   │
│  Crawler    │     │  (Storage)   │     │  (Python)   │
└─────────────┘     └──────────────┘     └─────────────┘
                                                 │
                                                 ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Web UI    │◀────│   FastAPI    │◀────│    Vespa    │
│   (HTML)    │     │    (API)     │     │  (Search)   │
└─────────────┘     └──────────────┘     └─────────────┘
```

1. **Custom Crawler**: High-performance multi-threaded crawler
   - Optimized for concurrent crawling with up to 3000 workers
   - Domain round-robin scheduling for fairness
   - Built-in state persistence for pause/resume support
   - Automatic language detection and filtering
2. **Vespa**: Selected for native hybrid search support and low latency
3. **FastAPI**: Modern async framework with automatic API documentation
4. **Sentence Transformers**: Efficient embeddings with good programming content understanding

### Why Vespa, not Tantivy?

Both technologies are capable of serving sub-50ms search results at scale, but given this is a search engine for programming documentation, there is a lot gain in having a semantic search component, which only Vespa can offer out-of-the-box. Programming documentation is uniquely dual-natured: it combines precise technical syntax with conceptual explanations. Neither search method alone can handle both effectively. Hybrid search leverages lexical precision for code and API references while using semantic understanding for concepts, problems, and solutions - delivering what developers actually need.

Vespa's plugin architecture lets you add complex features through configuration - implementing relevant snippets, code syntax highlighting, faceted search, or ML ranking requires just schema updates and ranking profiles, while Tantivy demands custom Rust code for each feature. For instance, adding multilingual search with dynamic snippets in Vespa takes a few configuration lines, but in Tantivy you'd build entire pipelines for tokenization, snippet extraction, and highlighting from scratch. 

### Production Architecture Plan

The current monolithic architecture was optimized for rapid development and demonstration. For production deployment at scale, the following architectural changes are proposed:

#### Distributed Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Crawler Workers │────▶│  Message Queue   │────▶│ Indexing Workers│
│  (Python/K8s)   │     │    (Kafka)       │     │  (Rust/Python)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                       │                         │
         ▼                       ▼                         ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Crawler State   │     │  Object Storage  │     │  Vespa Cluster  │
│  (PostgreSQL)   │     │   (GCS/S3)       │     │  (Multi-node)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
┌─────────────────┐                           ┌───────────▼─────┐
│   Web UI        │──────────────────────────▶│  Search API     │
│  (React/CDN)    │                           │  (Rust/Axum)    │
└─────────────────┘                           └─────────────────┘
```

#### 1. Distributed Crawling System

**Current Limitation**: Single-process crawler with in-memory state management

**Production Solution**:
- **Orchestration**: Kubernetes-based crawler workers with horizontal scaling
- **State Management**: PostgreSQL for distributed crawler state
- **Work Distribution**: Redis-based distributed queue with domain-aware sharding
- **Monitoring**: Grafana metrics + custom crawler dashboard

#### 2. Message Queue Architecture

**Current Limitation**: Direct file-based communication between crawler and indexer

**Production Solution**:
- **Apache Kafka**: For high-throughput document streaming
- **Schema Registry**: Protocol Buffers for efficient serialization
- **Partitioning**: By domain for ordered processing within domains
- **Retention**: 7-day retention for replay capability
- **Cloud Storage Backup**: All crawled documents are also stored in GCS/S3 buckets as compressed JSON for long-term archival and reprocessing capabilities

#### 3. Scalable Indexing Pipeline

**Current Limitation**: Single python script for batch indexing

**Production Solution**:
- **Rust-based Workers**: For high-performance document processing
- **Embedding Service**: Dedicated GPU nodes for embedding generation
- **Batching Strategy**: Dynamic batching based on queue depth
- **Incremental Updates**: Support for document updates without full reindex

#### 4. API Layer Optimization

**Current Limitation**: Python FastAPI with synchronous Vespa calls

**Production Solution**:
- **Rust API Service**: Using Axum for <10ms overhead and high concurrency
- **Connection Pooling**: Persistent HTTP/2 connections to Vespa
- **Caching Layer**: Redis for common queries (no caching mechanism was implemented so we can easily verify the SLA)

#### 5. Proxy Usage

To implement proxy support in your crawler, you would maintain a pool of proxy servers (datacenter IPs) that act as intermediaries between your crawler and the target websites. When making HTTP requests, instead of connecting directly to the documentation sites, your crawler would route requests through these proxies by configuring the proxies parameter in your requests library (e.g., requests.get(url, proxies={'http': 'http://proxy-server:port', 'https': 'https://proxy-server:port'})). A proxy manager would handle rotation logic - selecting different proxies for each request or domain to distribute the load and avoid detection, tracking success/failure rates per proxy, blacklisting failing proxies temporarily, and maintaining domain-specific proxy mappings for optimal performance. This approach masks your crawler's real IP address, prevents rate limiting by appearing as multiple different users, and enables geographic distribution of requests, though it adds complexity in handling proxy failures, authentication, and the additional latency from the proxy hop.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker and Docker Compose
- 4GB+ RAM (for Vespa and embedding models)
- Poetry (for dependency management)

### Installation & Setup

1. **Install Poetry** (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. **Install dependencies**:
```bash
poetry install
```

3. **Start everything** (recommended - handles all timing and setup):
```bash
./run_vespa.sh
```

4. **Access the search interface**:
   - Search UI: http://localhost:8000
   - Statistics: http://localhost:8000/stats
   - API Docs: http://localhost:8000/docs

### Manual Setup (if needed)

1. **Start Vespa**:
```bash
docker-compose up -d
sleep 30  # Wait for Vespa to initialize
./deploy_vespa.sh
```

2. **Crawl documents**:
```bash
cd crawler && poetry run python doc_scraper.py
```

3. **Index with embeddings**:
```bash
poetry run python indexer/indexer.py
```

4. **Start API server**:
```bash
poetry run python api/main.py
```

## Usage

### 1. Crawling Documents

Run the crawler to fetch documents:

```bash
# Run crawler (from crawler directory)
cd crawler && poetry run python doc_scraper.py
```

**Optional parameters:**
```bash
# Run with custom domains file (default: test_domains.txt)
poetry run python doc_scraper.py --domains all_domains.txt

# Run with custom page limits
poetry run python doc_scraper.py --global-max-pages 10000 --max-pages-per-domain 100

# Run with web monitor dashboard
poetry run python doc_scraper.py --web-monitor
```

The crawler will:
- Respect rate limits with configurable delays
- Save documents to `crawled_data/{domain}/` directories
- Maintain crawl state for pause/resume support
- Use multi-threaded concurrent crawling (up to 3000 workers)
- Track progress per domain with status updates
- Filter for English-language content by default

### 2. Indexing to Vespa

Index the crawled documents with embeddings:

```bash
poetry run python indexer/indexer.py
```

The indexer will:
- Generate 384-dimensional embeddings using all-MiniLM-L6-v2
- Split page contents into multiple chunks with overlap between them for more relevant results
- Index documents to Vespa using the correct JSON format
- Create HNSW index for fast semantic search
- Apply per-domain document limits to prevent index imbalance

Optional parameters:
- `--crawl-dir`: Specific crawl directory (default: latest)
- `--vespa-url`: Vespa URL (default: http://localhost:4080)
- `--batch-size`: Indexing batch size (default: 100)
- `--max-per-domain`: Maximum documents per domain (default: 10000)
- `--embedding-batch-size`: Batch size for embedding generation (default: 32)
- `--max-concurrent-embeddings`: Maximum concurrent embeddings in queue (default: 200)

Example:
```bash
# Index with default 10k limit per domain
poetry run python indexer/indexer.py

# Index with custom domain limit
poetry run python indexer/indexer.py --max-per-domain 5000
```

### 3. Starting the API Server

Start the FastAPI server:

```bash
# Using default configuration
poetry run python api/main.py

# Or with uvicorn directly
poetry run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Accessing the Search Interface

Open your browser and navigate to:
- Search UI: http://localhost:8000
- Statistics: http://localhost:8000/stats
- API Docs: http://localhost:8000/docs

## Clean State Rebuild

To completely tear down all services and rebuild from scratch:

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

## Search Types Explained

1. **BM25 Search**: Traditional keyword-based search using Vespa's built-in BM25 ranking
   - **Target documents**: Full documents only (`doc_type contains 'full_doc'`)
   - **Best for**: Exact term matching, specific function/class names
   - **Example**: "python tutorial" → finds docs with exact keyword matches
   - **Query**: Uses `userQuery()` in YQL with BM25 ranking profile
   - **Ranking formula**: `bm25(title) + 0.8*bm25(content) + 0.5*bm25(description)`
   - **Relevance scores**: 2.5-5.0+ for strong keyword matches
   - **Summary**: Dynamic snippets with highlighting

2. **Semantic Search**: Vector similarity search using HNSW index on document chunks
   - **Target documents**: Document chunks only (`doc_type contains 'chunk'`)
   - **Best for**: Conceptual queries, finding similar content across chunk boundaries
   - **Example**: "tutorial programming" → finds conceptually related content
   - **Query**: Uses `nearestNeighbor(chunk_embedding, query_embedding)` in YQL
   - **Model**: all-MiniLM-L6-v2 (384-dimensional embeddings)
   - **Distance metric**: Angular (cosine similarity)
   - **Relevance scores**: 0.45-0.47 (closeness function results)
   - **Summary**: Chunk content with parent document metadata

3. **Hybrid Search**: Intelligent combination of BM25 and semantic search
   - **Target documents**: All documents (both full docs and chunks)
   - **Best for**: General purpose searching combining keyword precision with semantic understanding
   - **Example**: "error handling techniques" → leverages both exact matches and conceptual similarity
   - **Weighting**: 50% normalized BM25 + 50% semantic similarity
   - **BM25 normalization**: Uses sigmoid function `1/(1+exp(-bm25_score/5))` to map to 0-1 range
   - **Behavior**: Full documents use hybrid scoring; chunks use pure semantic scoring
   - **Relevance scores**: 0.5-1.0+ (balanced scoring across both approaches)
   - **Summary**: Dynamic snippets for full docs, chunk content for chunks
   - **Recommended default option**

## Performance Results

### Local Search Latency Benchmarks
- **BM25 Search**: 7-15ms average (9.92ms typical)
- **Semantic Search**: 22-28ms average (25.61ms typical, includes 3.7ms embedding generation)
- **Hybrid Search**: 9-17ms average (12.55ms typical, includes 4.45ms embedding generation)
- **SLA Achievement**: 100% of queries under 50ms locally

### Cloud Deployment Performance
Based on comprehensive testing with 20 diverse queries ([full results](cloud_performance_results.json)):

- **Hybrid Search**: 35-67ms range (47.79ms average, 46.65ms median)
- **SLA Compliance**: 75% of queries under 50ms
- **Cold Start Performance**: 35-41ms (36.98ms average)
- **Vespa Internal Latency**: 6-10ms (from API logs)
- **Embedding Generation**: 4-5ms (ONNX optimized)

The cloud deployment achieves sub-50ms latency for most queries while running on free tier infrastructure. Performance testing script available at [test_cloud_api.py](test_cloud_api.py).

### Crawler Performance
- **Pages Scraped**: ~300,000 across 160+ domains
- **Concurrent Workers**: Up to 3,000 with domain-aware rate limiting
- **Throughput**: Variable per domain (Apple: 10 req/s, others: up to 50 req/s)
- **Success Rate**: >95% with automatic retry logic

### Indexing Performance
- **Embedding Generation**: Batched processing (32 texts/batch)
- **GPU Utilization**: 80-90% (improved from 60% baseline)
- **ONNX Optimization**: <5ms embedding generation in production
- **Document Processing**: 800-character chunks with 150-character overlap

## Advanced Technical Features

### Document Chunking Strategy
- **Chunk Size**: 800 characters with 150-character overlap
- **Purpose**: Enables semantic search across content boundaries
- **Benefit**: Improves relevance for conceptual queries while maintaining context
- **Implementation**: Dual storage approach (full documents + chunks)

### Embedding Optimization
- **Model**: Initially implemented CodeBERT but switched to all-MiniLM-L6-v2 for faster inference and optimized for natural language in programming docs
- **ONNX Runtime**: <5ms embedding generation in production API
- **Batching Strategy**: 32 texts per batch with async queue processing
- **Performance Gain**: GPU utilization improved from 60% to 80-90%

### Hybrid Search Implementation
- **Weighting**: 50% normalized BM25 + 50% semantic similarity
- **BM25 Normalization**: Sigmoid function `1/(1+exp(-bm25_score/5))` maps scores to 0-1 range
- **Scoring Logic**: Full documents use hybrid scoring; chunks use pure semantic scoring
- **Result Quality**: Combines exact term matching with conceptual understanding

## Main Challenges

### Scraping the 160+ Domains

A large portion of the time devoted to this project was to create a high-performance and fault-tolerant crawler. I found particularly hard to balance between high throughput vs safe/slow scraping, and getting the right kind of observability to understand where's the bottlekneck and troubleshoot why a domain was stuck and no longer scraping. I initially used Scrapy to take advantage of the their various built-in features, but I had multiple issues with their persistence and their observability features, and had better results building my own crawler on top of beautifulsoup.

These challenges were largely overcome with:

- O(1) data structures that uses sets for visited URLs and queue membership checks instead of lists for
  constant-time lookups
- Queue management and round-robin scheduling
- Domain aware and time based rate limiting
- Persistent state
- Terminal and web based monitoring
- Agressive restriction of link following that is not within the domain/subdomain. The only exception was the first redirect for the provided domain, otherwise that domain would be completely empty.

Nonetheless, I still didn't manage to scrape the entirety of the domains. In total, I scraped almost 300k pages.

### High Latency

I had to work on several optimizations to get most of the response times under 50ms. This SLA was particularly challenging for the cloud deployment on a tight budget. The stack chosen for the cloud deployment was Google Cloud Platform (GCP) Cloud Run + Vespa Cloud running on the same zone:

- GCP was chosen given it has the best reliability and lowest latency from my previous experience working with different cloud providers
- Cloud Run was chosen for the tight budget, where I preferrably want to run the API at $0 cost. It comes with drawbacks like longer cold start that are minimized by warm-up queries.
- Vespa Cloud was chosen for its easy to setup and the $300 free credit so I can run the demo for free on a dev container.

The optimizations include:

For the API:

- ONNX model optimization to get embedding generation under 5ms
- HTTP connection pooling to reduce connection and TCP handshake overhead
- Model and Vespa warm up
- Request compression using gzip
- Cloud: deploy the cloud run in the same cloud provider and region where Vespa Cloud was hosted
- Profiling to understand the bottlekneck locally and on the cloud

For the Vespa configuration:

- Finetuned configuration with optimal values for the Vespa Cloud dev environment (2vCPU, 8GB RAM)
- Single search thread `num-threads-per-search: 1` for reduced latency
- Minimize GC-induce latency spikes by using `-XX:MaxGCPauseMillis=10`
- HTTP Server Optimization with `tcpNoDelay=true` and `reuseAddress=true`
- HNSW Vector Index Tuning with `max-links-per-node: 16` and `neighbors-to-explore-at-insert: 200` that balances speed and recall.

### Search Relevancy

This was the feature that I least worked given the other two challenges took considerate time. The main features that improved search relevancy were:

- Hybrid search with embedding chunks for the sementatic search component. Initially, CodeBERT was used given that we are indexing programming documentation, but I noticed that most of the relevant documentation is not code. `all-MiniLM-L6-v2` was then chosen for its speed and accuracy in natural language domains.
- Normalization for the hybrid search to combine semantic + lexical search with a fair score.
- Implementation of Vespa's built-in feature "relevant snippets" to retrieve meaningful passages of each search result.

## Cloud Deployment

### Production Cloud Stack

**Current Live Demo Architecture:**
- **Search Engine**: Vespa Cloud Dev (2 vCPUs, 8GB RAM) in `gcp-us-central1-f`
- **API Service**: Google Cloud Run (2 vCPUs, 4GB RAM) in `us-central1`
- **Authentication**: mTLS certificate-based security for Vespa data plane
- **Performance**: 35-67ms typical end-to-end (47.79ms average)
- **Cost**: $0/month using free tiers and credits

### Vespa Cloud Deployment

Deploy your Vespa application to Vespa Cloud for production-grade search infrastructure:

#### Prerequisites

1. **Create Vespa Cloud account**: Sign up at https://cloud.vespa.ai
2. **Install Vespa CLI**:
```bash
# macOS
brew install vespa-cli

# Linux/Windows - download from https://github.com/vespa-engine/vespa/releases
```

3. **Authenticate with Vespa Cloud**:
```bash
vespa auth login
```

#### Deploy to Vespa Cloud

1. **Configure your application**:
```bash
# Set your tenant, application, and instance names
vespa config set target cloud
vespa config set application {tenant}.{application}.{instance}

# Set deployment zone to dev environment in GCP US Central
vespa config set zone dev.gcp-us-central1-f
```

2. **Deploy the Vespa application**:
```bash
# From project root directory
vespa deploy config/vespa --wait 300 --zone dev.gcp-us-central1-f
```

3. **Generate data plane certificates** for API access:
```bash
vespa auth cert
```

4. **Test your deployment**:
```bash
# Health check
vespa status

# Test search functionality
vespa query "select * from sources * limit 5"
```

#### Indexing to Vespa Cloud

Once deployed, index your documents to Vespa Cloud:

```bash
# Get your Vespa Cloud endpoint URL
VESPA_ENDPOINT=$(vespa status --wait 300 | grep "Discovery URL" | awk '{print $3}')

# Index documents using the cloud endpoint
poetry run python indexer/indexer.py --vespa-url $VESPA_ENDPOINT
```

#### Environment Configuration

For production usage, configure the API to use your Vespa Cloud endpoint:

```bash
# Set environment variable for API server
export VESPA_URL="https://your-app.vespa-app.cloud"

# Start API server
poetry run python api/main.py
```

### Google Cloud Run Deployment

Deploy the API to Google Cloud Run with automatic scaling:

#### Prerequisites

**Note**: Vespa Cloud uses mTLS (mutual TLS) certificate authentication for data plane access.

1. **Install Vespa CLI**:
```bash
# macOS
brew install vespa-cli

# Or download from https://github.com/vespa-engine/vespa/releases
```

2. **Generate Vespa Cloud certificates**:
```bash
# Configure Vespa CLI for your application
vespa config set target cloud
vespa config set application {tenant}.{application}.{instance}

# Generate data plane certificates
vespa auth cert
```

3. **Copy certificates to your project**:
```bash
# Create certs directory and copy certificates
mkdir -p certs
cp ~/.vespa/data-plane-public-cert.pem certs/
cp ~/.vespa/data-plane-private-key.pem certs/
```

#### Deployment

```bash
# Deploy to Google Cloud Run with Vespa endpoint
VESPA_URL="https://your-vespa-endpoint.z.vespa-app.cloud" ./deploy_cloud.sh
```

#### Local Testing

Before deploying, test the setup locally:

```bash
# Build Docker image
docker build -t mini-search-engine-api:latest .

# Run locally
docker run -p 8000:8000 \
    -e VESPA_URL=https://your-vespa-endpoint.z.vespa-app.cloud \
    mini-search-engine-api:latest

# Test health endpoint (in another terminal)
curl http://localhost:8000/health
```

The deployment script will:
- Enable required Google Cloud services
- Create an Artifact Registry repository
- Build and push a Docker image with embedded mTLS certificates
- Deploy to Cloud Run with optimized settings (2 CPU, 4GB RAM)
- Configure auto-scaling and performance features
- Set environment variables for Vespa Cloud endpoint

### Deployment Configuration

The deployment uses:
- **CPU**: 2 vCPUs with CPU boost enabled
- **Memory**: 4GB RAM
- **Scaling**: 0-1 instances (configurable)
- **Concurrency**: 20 requests per instance
- **Environment**: Generation 2 execution environment
- **Authentication**: mTLS certificate-based authentication

#### Required Environment Variables:
- `VESPA_URL`: Your Vespa Cloud application endpoint

## Troubleshooting

### Vespa won't start or exits immediately
- **Most common issue**: Vespa needs an application deployed to start properly
- Run deployment after starting: `docker-compose up -d && sleep 20 && ./deploy_vespa.sh`
- Check Docker logs: `docker logs vespa --tail 50`
- Ensure sufficient memory (4GB+ recommended)
- Verify ports 4080 and 19071 are available
- If container keeps exiting, wait 30+ seconds for initialization before deploying

### Crawler issues
- Check logs in `crawled_data/scraper.log`
- Verify domains file exists: `crawler/test_domains.txt` or `crawler/all_domains.txt`
- Check crawler state: `crawled_data/crawler_state.json`
- For rate limiting issues, adjust `DELAY_BETWEEN_REQUESTS` or domain-specific limits

### Search returns no results
- Verify documents are indexed: check `/stats` endpoint
- Ensure Vespa is running and healthy
- Check query syntax and search type
