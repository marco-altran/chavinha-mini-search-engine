# Documentation Web Scraper

A high-performance concurrent web scraper designed to crawl and download technical documentation from specified domains with language filtering, state persistence, and real-time monitoring.

## Features

- 🚀 **High-Performance Crawling**: Multi-threaded architecture with up to 3000 concurrent workers
- 🔄 **Resumable**: Full state persistence for stopping/resuming crawls
- 🌐 **Smart Domain Handling**: Automatic redirect resolution and subdomain support
- 🗣️ **Language Filtering**: English-only content filtering using language detection
- 📊 **Real-time Monitoring**: Rich console dashboard showing live progress
- 🌐 **Web Monitor**: Optional web-based dashboard for remote monitoring
- 💾 **Organized Storage**: JSON files organized by domain folders
- 🎯 **Configurable Limits**: Per-domain and global page limits
- ⚡ **Rate Limiting**: Configurable per-domain concurrent requests and delays

## Quick Start

### Basic Usage

```bash
# Run the scraper with test domains (default)
poetry run python doc_scraper.py
```

The scraper will:
1. Start crawling 4 test documentation websites (from test_domains.txt)
2. Fetch up to 100 pages per domain
3. Save results in `../crawled_data/` directory
4. Display real-time progress in the terminal

### Usage Examples

```bash
# Test with 4 domains (default)
poetry run python doc_scraper.py

# Use all domains (160+ domains)
poetry run python doc_scraper.py --domains all_domains.txt

# Custom limits
poetry run python doc_scraper.py --domains all_domains.txt --global-max-pages 50000 --max-pages-per-domain 200

# With web monitor (access at http://localhost:5001)
poetry run python doc_scraper.py --web-monitor --domains test_domains.txt

# Short form arguments
poetry run python doc_scraper.py -d all_domains.txt -g 10000 -m 50
```

### Command Line Options

- `--domains` / `-d` - Specify domains file (default: `test_domains.txt`)
- `--global-max-pages` / `-g` - Set global page limit (default: 2,000,000)
- `--max-pages-per-domain` / `-m` - Set per-domain page limit (default: 100)
- `--web-monitor` - Enable web monitor mode (disables console dashboard)

### Resuming a Crawl

If the crawler is interrupted (Ctrl+C), simply run it again with the same arguments:

```bash
poetry run python doc_scraper.py --domains all_domains.txt
```

It will automatically resume from where it left off using the saved state in `crawler_state.json`.

## Configuration

### Domain Configuration

Domains are configured via text files:

- `test_domains.txt` - 4 test domains for development
- `all_domains.txt` - 160+ production domains

Create custom domain files as needed:
```bash
# Create a custom domains file
echo "docs.python.org" > my_domains.txt
echo "developer.mozilla.org" >> my_domains.txt
echo "reactjs.org" >> my_domains.txt

# Use it
poetry run python doc_scraper.py --domains my_domains.txt
```

### Runtime Configuration

All key settings can be configured via command line arguments (see Usage Examples above).

### Advanced Configuration

For other settings, edit `doc_scraper.py`:

```python
# Performance settings
MAX_WORKERS = 3000              # Concurrent downloads (optimized for 12-core CPU)
REQUEST_TIMEOUT = 8             # HTTP timeout in seconds
REFRESH_INTERVAL = 5            # Dashboard update frequency

# Rate limiting
MAX_CONCURRENT_REQUESTS_PER_DOMAIN = 50  # Max concurrent requests per domain
DELAY_BETWEEN_REQUESTS = 0.1             # Seconds between requests to same domain

# Error thresholds
MAX_ERRORS_PER_DOMAIN_BEFORE_GIVE_UP = 100
MAX_CONSECUTIVE_EMPTY_QUEUE_FETCHES = 1000

# Domain-specific limits (in DOMAIN_SPECIFIC_LIMITS dict)
"developer.apple.com": {"max_concurrent": 10, "delay": 1.0}
"docs.microsoft.com": {"max_concurrent": 30, "delay": 0.2}
```

## Output Structure

```
../crawled_data/
├── crawler_state.json         # Crawler state for resumability
├── scraper.log               # Detailed logs
├── angular.dev/              # Domain folder
│   ├── 4a3f2b1c...json      # Page data (MD5 hash as filename)
│   └── 7d8e9f0a...json
├── docs.python.org/
│   └── ...
└── ...
```

### Page Data Format

Each JSON file contains:

```json
{
  "id": "4a3f2b1c7d8e9f0a...",
  "url": "https://docs.python.org/3/tutorial/",
  "domain": "docs.python.org",
  "title": "The Python Tutorial",
  "content": "This tutorial introduces the reader informally...",
  "code_snippets": [
    "import sys",
    "print('Hello, World!')"
  ],
  "crawled_at": "2025-05-31T10:30:45.123456+00:00"
}
```

## Monitoring Dashboard

The real-time dashboard shows:

```
DOCUMENTATION SCRAPER 
┌─────────────────────────────────────┐
│ CRAWLER OVERVIEW                    │
│ Total Domains: 166                  │
│ TO_BE_STARTED: 5                   │
│ ACTIVE: 50                          │
│ REACHED_MAX: 100                    │
│ GAVE_UP: 5                          │
│ Total Pages: 85000 / 2000000        │
└─────────────────────────────────────┘

ALL DOMAINS STATUS
┌──────────────────────┬────────────┬────────┬────────┬────────┬────────┬────────────┐
│ Domain               │ Status     │  Pages │  Queue │ Active │ Errors │ Last Update│
├──────────────────────┼────────────┼────────┼────────┼────────┼────────┼────────────┤
│ angular.io→angular.dev│ ACTIVE     │    700 │   2300 │     25 │      0 │   14:23:42 │
│ docs.python.org      │ ACTIVE     │    900 │   1500 │     30 │      1 │   14:23:44 │
│ developer.mozilla.org│ REACHED_MAX│  10000 │      0 │      0 │      0 │   14:22:15 │
└──────────────────────┴────────────┴────────┴────────┴────────┴────────┴────────────┘

PROGRESS STATISTICS
Pages/Second: 125.50
Pages/Hour: 451800
ETA: 4.2 hours
```

## Domain Status

- **TO_BE_STARTED**: Waiting to begin crawling
- **ACTIVE**: Currently crawling
- **REACHED_MAX**: Hit the page limit
- **GAVE_UP**: Too many errors or no more URLs
- **COMPLETED_GLOBAL**: Global limit reached

## Advanced Usage

### Web Monitor Dashboard

Run with web monitor for remote monitoring:

```bash
poetry run python doc_scraper.py --web-monitor --domains all_domains.txt
```

Access the dashboard at http://localhost:5001/

### Re-seeding Domains

The crawler automatically re-seeds domains that have given up when restarted:
- Clears visited set if all seed URLs were previously visited
- Resets error counts for fresh retry

### State Management

The crawler saves state every 30 seconds and on exit:
- `crawler_state.json` contains all crawl progress
- Visited URLs tracked with O(1) lookups using sets
- Queue management with FIFO ordering
- Domain status and statistics

### Performance Optimization

For maximum performance:
- Increase `MAX_WORKERS` (respect target sites)
- Adjust `MAX_CONCURRENT_REQUESTS_PER_DOMAIN` carefully
- Use connection pooling (already configured)
- Monitor with `--web-monitor` for bottlenecks

## Ethical Considerations

1. **Rate limiting**: Built-in configurable rate limiting per domain
2. **User-Agent**: Identifies itself as a documentation scraper
3. **Concurrent limits**: Respects server resources with per-domain limits
4. **Purpose**: Designed for educational and research purposes

## Full Production Deployment

For production use with all 160+ domains:

```bash
# Production crawl with all domains
poetry run python doc_scraper.py --domains all_domains.txt --max-pages-per-domain 10000 --global-max-pages 2000000

# Production with web monitoring
poetry run python doc_scraper.py --domains all_domains.txt --max-pages-per-domain 10000 --web-monitor

# Conservative production settings (slower but safer)
poetry run python doc_scraper.py --domains all_domains.txt --max-pages-per-domain 5000 --global-max-pages 500000
```

Monitor progress via:
- Console dashboard (default)
- Web dashboard at http://localhost:5001/ (with --web-monitor)
- Log file at `../crawled_data/scraper.log`

## Architecture Details

### Threading Model
- Main thread: Dashboard updates and coordination
- Worker threads: Concurrent page fetching and processing
- Thread-safe state management with locks

### Performance Features
- O(1) URL lookup using sets for visited tracking
- FIFO queue with deque for fair URL processing
- Connection pooling with 500 connections, 3000 max pool size
- Domain-aware round-robin scheduling

### State Persistence
- Automatic state saves every 30 seconds
- Graceful shutdown on Ctrl+C with state save
- Full recovery from any interruption point

## Dependencies

- `requests`: HTTP library with connection pooling
- `beautifulsoup4`: HTML parsing
- `langdetect`: Language detection
- `rich`: Terminal UI dashboard
- `flask`: Web monitor interface (optional)

## License

This scraper is provided for educational purposes. Please respect website terms of service and rate limits when crawling.