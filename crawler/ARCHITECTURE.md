# Documentation Web Scraper Architecture

## Overview

The Documentation Web Scraper is a Python-based concurrent web crawler designed to systematically download technical documentation from specified domains. It features state persistence, language filtering, and real-time monitoring.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Main Orchestrator                         │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │State Manager│  │Thread Pool   │  │  Dashboard Generator   │ │
│  │(JSON File)  │  │(5 Workers)   │  │  (Rich Console)        │ │
│  └─────────────┘  └──────────────┘  └────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Domain Processors                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │URL Fetcher   │  │HTML Parser   │  │  Content Extractor   │ │
│  │(Requests)    │  │(BeautifulSoup)│ │  (Language Filter)   │ │
│  └──────────────┘  └──────────────┘  └──────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Data Storage                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    crawled_data/                          │  │
│  │  ├── crawler_state.json    (Persistent state)            │  │
│  │  ├── angular.dev/          (Domain folder)               │  │
│  │  │   ├── <md5_hash1>.json  (Page data)                  │  │
│  │  │   └── <md5_hash2>.json                               │  │
│  │  └── docs.python.org/                                    │  │
│  │      └── ...                                             │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. State Management
- **Purpose**: Maintains crawler state for resumability and progress tracking
- **Implementation**: JSON file (`crawler_state.json`) with atomic updates
- **Key Data Structures**:
  ```python
  crawler_state = {
      "resolved_domains_map": {},      # initial -> resolved domain mapping
      "effective_allowed_domains": [], # actual domains to crawl
      "domains_status": {},           # per-domain crawl status
      "global_pages_fetched": 0,      # total pages across all domains
      "start_time": "ISO-8601",       # crawler start time
      "last_save_time": "ISO-8601"    # last state persistence
  }
  ```

### 2. Domain Resolution
- **Purpose**: Handles domain redirects and normalizes URLs
- **Process**:
  1. Attempts HTTPS then HTTP HEAD requests
  2. Follows redirects to final destination
  3. Maps initial domains to resolved domains
  4. Updates allowed domains list with resolved values

### 3. Concurrent Processing
- **Thread Pool**: 5 workers for parallel URL processing
- **Task Distribution**: Round-robin across active domains
- **Queue Management**: Per-domain URL queues with visited tracking
- **Collision Prevention**: Active URLs tracked to prevent duplicate processing

### 4. Content Processing Pipeline

#### Fetch Stage
- HTTP GET with custom User-Agent
- Timeout handling (10 seconds)
- Content-type validation (HTML only)
- Error categorization and logging

#### Parse Stage
- HTML parsing with BeautifulSoup/lxml
- Language detection (English only)
- Main content extraction heuristics
- Code snippet extraction from `<pre>` and `<code>` tags

#### Link Extraction
- Absolute URL resolution
- Domain filtering (allowed domains only)
- Fragment removal and normalization
- Subdomain handling

### 5. Data Storage
- **Structure**: Domain-based folder organization
- **Format**: JSON files named by URL MD5 hash
- **Schema**:
  ```json
  {
    "id": "md5_hash",
    "url": "https://example.com/page",
    "domain": "example.com",
    "title": "Page Title",
    "content": "Extracted text content",
    "code_snippets": ["snippet1", "snippet2"],
    "crawled_at": "ISO-8601 timestamp"
  }
  ```

### 6. Monitoring Dashboard
- **Technology**: Rich console library for terminal UI
- **Updates**: Real-time refresh every 2 seconds
- **Displays**:
  - Overview statistics
  - Per-domain status and progress
  - Error tracking
  - Timing information

## Domain Status Lifecycle

```
TO_BE_STARTED → ACTIVE → REACHED_MAX
                  ↓
               GAVE_UP
```

- **TO_BE_STARTED**: Initial state, waiting to begin
- **ACTIVE**: Currently crawling
- **REACHED_MAX**: Hit page limit (10 pages in test mode)
- **GAVE_UP**: Too many errors or empty queue

## Key Design Decisions

### 1. Concurrent vs Parallel
- Chose ThreadPoolExecutor over multiprocessing for shared state simplicity
- I/O bound workload suits threading model

### 2. State Persistence
- JSON format for human readability and debugging
- Periodic saves (every 30 seconds) + on exit
- Atomic write pattern to prevent corruption

### 3. URL Management
- Per-domain queues prevent one domain from dominating
- FIFO queue processing for breadth-first crawling
- Visited set prevents re-crawling

### 4. Error Handling
- Consecutive error tracking per domain
- Automatic give-up after threshold
- Detailed error logging for debugging

### 5. Content Extraction
- Heuristic-based main content detection
- Limited content/snippet sizes for storage efficiency
- Language filtering at parse stage (not fetch) for accuracy

## Performance Characteristics

- **Concurrency**: 5 simultaneous downloads
- **Memory Usage**: O(n) where n = total URLs seen
- **Disk Usage**: ~5-10KB per page (JSON format)
- **Network**: Respects timeouts, no retry logic
- **CPU**: Light usage, mostly I/O waiting

## Limitations

1. **No JavaScript Support**: Static HTML only
2. **No Robots.txt**: Doesn't check crawl permissions
3. **Basic Duplicate Detection**: URL-based only
4. **Simple Queue**: No priority or depth limits
5. **Memory Bound**: All state in memory

## Extension Points

1. **Storage Backend**: Replace JSON with database
2. **Queue System**: Add Redis for distributed crawling
3. **Content Extraction**: Plug in advanced extractors
4. **Monitoring**: Export metrics to Prometheus
5. **Scheduling**: Add cron-like recrawl support