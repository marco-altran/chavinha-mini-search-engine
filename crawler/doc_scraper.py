#!/usr/bin/env python3
"""
Documentation Web Scraper - Production Configuration (FINAL VERSION)

All performance issues fixed:
1. Uses sets for visited URLs (O(1) lookups) ✓
2. Uses sets for queue membership checks (O(1) lookups) ✓  
3. Maintains full worker pool utilization (2000 workers) ✓
4. Thread-safe operations ✓
"""

import os
import json
import time
import hashlib
import logging
import argparse
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin
import re
import requests
from bs4 import BeautifulSoup
from langdetect import detect, LangDetectException
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.text import Text
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import deque

# Configuration - will be loaded from domains file
ALLOWED_DOMAINS_INITIAL = []

# Default settings - can be overridden by command line arguments
DATA_DIR = "../crawled_data"
STATE_FILE = os.path.join(DATA_DIR, "crawler_state.json")
GLOBAL_MAX_PAGES = 10000 * 200  # Default - can be overridden
MAX_PAGES_PER_DOMAIN = 100  # Default - can be overridden

# Performance settings optimized for production
REQUEST_TIMEOUT = 8  # Optimized timeout for reliable crawling
USER_AGENT = "Mozilla/5.0 (compatible; DocScraper/1.0; +http://example.com/bot)"
MAX_WORKERS = 3000  # Optimized for 12-core CPU with improved macOS host limits
REFRESH_INTERVAL = 5  # Less frequent updates to reduce overhead
MAX_ERRORS_PER_DOMAIN_BEFORE_GIVE_UP = 100
MAX_CONSECUTIVE_EMPTY_QUEUE_FETCHES = 1000  # Much higher tolerance for high-concurrency environment

# Rate limiting settings
MAX_CONCURRENT_REQUESTS_PER_DOMAIN = 50  # Max concurrent requests per domain
DELAY_BETWEEN_REQUESTS = 0.1  # Seconds between requests to same domain
DOMAIN_SPECIFIC_LIMITS = {
    # Add domain-specific limits here if needed
    "developer.apple.com": {"max_concurrent": 10, "delay": 1.0},  # Apple is strict
    "docs.microsoft.com": {"max_concurrent": 30, "delay": 0.2},
    "dev.mysql.com": {"max_concurrent": 20, "delay": 0.5},
}

# --- Logging ---
os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(DATA_DIR, 'scraper.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Global State ---
crawler_state = {
    "resolved_domains_map": {},
    "effective_allowed_domains": [],
    "domains_status": {},
    "global_pages_fetched": 0,
    "start_time": None,
    "last_save_time": None
}

# PERFORMANCE FIXES:
visited_sets = {}      # domain -> set of visited URLs (O(1) lookups)
queue_sets = {}        # domain -> set of URLs in queue (O(1) membership checks)
queue_deques = {}      # domain -> deque of URLs (maintain order for FIFO)
state_lock = threading.Lock()  # Thread safety for state updates

# Rate limiting structures
domain_active_requests = {}  # domain -> current active request count
domain_last_request_time = {}  # domain -> last request timestamp
domain_request_locks = {}  # domain -> lock for request counting

console = Console()

# Language detection cache
lang_detection_cache = {}

# Session for connection pooling
session = requests.Session()
session.headers.update({'User-Agent': USER_AGENT})
# Enable connection pooling - optimized for high-performance setup
adapter = requests.adapters.HTTPAdapter(
    pool_connections=500,
    pool_maxsize=3000,
    max_retries=1
)
session.mount('http://', adapter)
session.mount('https://', adapter)

# --- Argument Parsing and Configuration ---
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Documentation Web Scraper')
    parser.add_argument('--domains', '-d', 
                       default='test_domains.txt',
                       help='Path to domains file (default: test_domains.txt)')
    parser.add_argument('--global-max-pages', '-g',
                       type=int,
                       default=10000 * 200,
                       help='Global maximum pages to crawl (default: 2000000)')
    parser.add_argument('--max-pages-per-domain', '-m',
                       type=int, 
                       default=100,
                       help='Maximum pages per domain (default: 100)')
    parser.add_argument('--web-monitor', 
                       action='store_true',
                       help='Run with web monitor (disables console dashboard)')
    return parser.parse_args()

def load_domains_from_file(domains_file):
    """Load domains from a text file."""
    global ALLOWED_DOMAINS_INITIAL
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    domains_path = os.path.join(script_dir, domains_file)
    
    if not os.path.exists(domains_path):
        logger.error(f"Domains file not found: {domains_path}")
        raise FileNotFoundError(f"Domains file not found: {domains_path}")
    
    domains = []
    with open(domains_path, 'r') as f:
        for line in f:
            domain = line.strip()
            if domain and not domain.startswith('#'):
                domains.append(domain)
    
    ALLOWED_DOMAINS_INITIAL = sorted(list(set(domains)))
    logger.info(f"Loaded {len(ALLOWED_DOMAINS_INITIAL)} domains from {domains_file}")
    return ALLOWED_DOMAINS_INITIAL

# --- Utility Functions ---
def get_domain_key(url_or_domain_name):
    """Gets the initial domain key, handling subdomains."""
    parsed_url = urlparse(url_or_domain_name if "://" in url_or_domain_name else "http://" + url_or_domain_name)
    netloc = parsed_url.netloc
    
    for initial, resolved in crawler_state["resolved_domains_map"].items():
        if netloc == resolved or netloc.endswith(f".{resolved}"):
            return initial
    
    if netloc in ALLOWED_DOMAINS_INITIAL:
        return netloc
    
    return netloc

def get_domain_limits(domain_key):
    """Get rate limiting parameters for a domain."""
    if domain_key in DOMAIN_SPECIFIC_LIMITS:
        return (
            DOMAIN_SPECIFIC_LIMITS[domain_key].get("max_concurrent", MAX_CONCURRENT_REQUESTS_PER_DOMAIN),
            DOMAIN_SPECIFIC_LIMITS[domain_key].get("delay", DELAY_BETWEEN_REQUESTS)
        )
    return MAX_CONCURRENT_REQUESTS_PER_DOMAIN, DELAY_BETWEEN_REQUESTS

def can_make_request(domain_key):
    """Check if we can make a request to this domain (rate limiting)."""
    max_concurrent, min_delay = get_domain_limits(domain_key)
    
    # Initialize if needed
    if domain_key not in domain_request_locks:
        domain_request_locks[domain_key] = threading.Lock()
        domain_active_requests[domain_key] = 0
        domain_last_request_time[domain_key] = 0
    
    with domain_request_locks[domain_key]:
        # Check concurrent request limit
        if domain_active_requests[domain_key] >= max_concurrent:
            return False
        
        # Check time-based rate limit
        time_since_last = time.time() - domain_last_request_time[domain_key]
        if time_since_last < min_delay:
            return False
        
        return True

def acquire_request_slot(domain_key):
    """Acquire a request slot for the domain."""
    with domain_request_locks[domain_key]:
        domain_active_requests[domain_key] += 1
        domain_last_request_time[domain_key] = time.time()
        
        # Log if we're approaching limits
        max_concurrent, _ = get_domain_limits(domain_key)
        if domain_active_requests[domain_key] > max_concurrent * 0.8:
            logger.debug(f"Domain {domain_key}: {domain_active_requests[domain_key]}/{max_concurrent} concurrent requests")

def release_request_slot(domain_key):
    """Release a request slot for the domain."""
    with domain_request_locks[domain_key]:
        domain_active_requests[domain_key] = max(0, domain_active_requests[domain_key] - 1)

def sanitize_filename(name):
    """Sanitizes a string to be a valid filename."""
    return re.sub(r'[^\w\.-]', '_', name)

def extract_code_snippets(soup):
    """Extract code snippets from HTML."""
    snippets = []
    
    for pre_tag in soup.find_all('pre'):
        code_tag = pre_tag.find('code')
        snippet_text = (code_tag.get_text(strip=True) if code_tag else pre_tag.get_text(strip=True))
        if snippet_text and len(snippet_text) > 10:
            snippets.append(snippet_text[:1000])  # Larger limit for production
    
    for code_tag in soup.find_all('code'):
        if code_tag.parent.name != 'pre':
            snippet_text = code_tag.get_text(strip=True)
            if snippet_text and 5 < len(snippet_text) < 500:
                snippets.append(snippet_text)
    
    return snippets[:20]  # Reduced from 50 to 20 for better performance

# --- State Management ---
def load_state():
    global crawler_state, visited_sets, queue_sets, queue_deques
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                loaded = json.load(f)
                crawler_state.update(loaded)
                
                # PERFORMANCE FIX: Convert visited lists to sets
                visited_sets = {}
                queue_sets = {}
                queue_deques = {}
                
                for domain, data in crawler_state["domains_status"].items():
                    # Convert visited list to set
                    visited_list = data.get("visited", [])
                    visited_sets[domain] = set(visited_list)
                    
                    # Convert queue list to set and deque
                    queue_list = data.get("queue", [])
                    queue_sets[domain] = set(queue_list)
                    queue_deques[domain] = deque(queue_list)
                    
                    if len(visited_list) > 100 or len(queue_list) > 100:
                        logger.info(f"{domain}: {len(visited_list)} visited, {len(queue_list)} queued")
                
                logger.info(f"Crawler state loaded. Progress: {crawler_state['global_pages_fetched']}/{GLOBAL_MAX_PAGES}")
                logger.info("PERFORMANCE FIXES APPLIED:")
                logger.info("- Visited URLs: lists → sets (O(1) lookups)")
                logger.info("- Queue URLs: lists → sets+deques (O(1) membership + FIFO order)")
        except Exception as e:
            logger.error(f"Failed to load state: {e}. Starting fresh.")
            initialize_state()
    else:
        logger.info("No state file found. Starting fresh.")
        initialize_state()

def save_state():
    global crawler_state, visited_sets, queue_deques
    with state_lock:
        crawler_state["last_save_time"] = datetime.now(timezone.utc).isoformat()
        
        # Convert sets/deques back to lists for JSON serialization
        for domain in crawler_state["domains_status"]:
            if domain in visited_sets:
                crawler_state["domains_status"][domain]["visited"] = list(visited_sets[domain])
            if domain in queue_deques:
                crawler_state["domains_status"][domain]["queue"] = list(queue_deques[domain])
        
        os.makedirs(DATA_DIR, exist_ok=True)
        try:
            with open(STATE_FILE, 'w') as f:
                json.dump(crawler_state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving state: {e}")

def initialize_state():
    global crawler_state, visited_sets, queue_sets, queue_deques
    crawler_state["resolved_domains_map"] = {}
    crawler_state["effective_allowed_domains"] = []
    crawler_state["domains_status"] = {}
    crawler_state["global_pages_fetched"] = 0
    crawler_state["start_time"] = datetime.now(timezone.utc).isoformat()
    visited_sets = {}
    queue_sets = {}
    queue_deques = {}
    logger.info("State initialized for production crawl.")

# --- Redirect Resolution ---
def resolve_domain(domain):
    """Resolves a single domain, trying https and http."""
    urls_to_try = [f"https://{domain}", f"http://{domain}"]
    for url_to_try in urls_to_try:
        try:
            response = session.head(
                url_to_try, 
                timeout=REQUEST_TIMEOUT, 
                allow_redirects=True
            )
            final_netloc = urlparse(response.url).netloc
            if final_netloc:
                logger.info(f"Resolved {domain} -> {final_netloc}")
                return final_netloc
        except requests.RequestException as e:
            logger.debug(f"Could not resolve {url_to_try}: {e}")
    
    logger.warning(f"Failed to resolve {domain}, using original")
    return domain

def resolve_initial_domains():
    """Resolves all initial domains and populates state."""
    global crawler_state, visited_sets, queue_sets, queue_deques
    logger.info(f"Resolving {len(ALLOWED_DOMAINS_INITIAL)} initial domains...")
    resolved_map = {}
    effective_domains_set = set()

    with ThreadPoolExecutor(max_workers=20) as executor:  # More workers for resolution
        future_to_domain = {executor.submit(resolve_domain, domain): domain for domain in ALLOWED_DOMAINS_INITIAL}
        for i, future in enumerate(as_completed(future_to_domain)):
            initial_domain = future_to_domain[future]
            try:
                resolved = future.result()
                resolved_map[initial_domain] = resolved
                effective_domains_set.add(resolved)
                if (i + 1) % 10 == 0:
                    logger.info(f"Resolved {i + 1}/{len(ALLOWED_DOMAINS_INITIAL)} domains...")
            except Exception as exc:
                logger.error(f"{initial_domain} resolution exception: {exc}")
                resolved_map[initial_domain] = initial_domain
                effective_domains_set.add(initial_domain)

    crawler_state["resolved_domains_map"] = resolved_map
    crawler_state["effective_allowed_domains"] = sorted(list(effective_domains_set))

    # Initialize domain status
    for initial_domain, resolved_domain in crawler_state["resolved_domains_map"].items():
        if initial_domain not in crawler_state["domains_status"]:
            base_url = f"https://{resolved_domain}/"
            crawler_state["domains_status"][initial_domain] = {
                "resolved_domain": resolved_domain,
                "queue": [base_url],
                "visited": [],  # Still stored as list in state
                "pages_fetched": 0,
                "errors": [],
                "last_updated": None,
                "status": "TO_BE_STARTED",
                "consecutive_empty_queue_fetches": 0,
                "consecutive_errors": 0
            }
            # Initialize empty sets/deques
            visited_sets[initial_domain] = set()
            queue_sets[initial_domain] = {base_url}
            queue_deques[initial_domain] = deque([base_url])
        else:
            # For existing domains that gave up, re-seed with some starting URLs
            domain_status = crawler_state["domains_status"][initial_domain]
            if len(domain_status["queue"]) == 0 and domain_status["status"] == "GAVE_UP":
                # Re-seed with various documentation paths
                seed_urls = [
                    f"https://{resolved_domain}",
                ]
                
                # Add URLs that haven't been visited yet
                new_queue = []
                for url in seed_urls:
                    if url not in visited_sets.get(initial_domain, set()):
                        new_queue.append(url)
                
                if new_queue:
                    domain_status["queue"] = new_queue
                    queue_sets[initial_domain] = set(new_queue)
                    queue_deques[initial_domain] = deque(new_queue)
                    domain_status["status"] = "TO_BE_STARTED"
                    domain_status["consecutive_empty_queue_fetches"] = 0
                    domain_status["consecutive_errors"] = 0  # Reset error count too
                    logger.info(f"Re-seeded {initial_domain} with {len(new_queue)} new URLs")
                else:
                    # If all seed URLs were visited, clear visited set to allow retry
                    logger.info(f"All seed URLs visited for {initial_domain}, clearing visited set for full retry")
                    visited_sets[initial_domain] = set()
                    domain_status["visited"] = []
                    domain_status["queue"] = [f"https://{resolved_domain}/"]
                    queue_sets[initial_domain] = {f"https://{resolved_domain}/"}
                    queue_deques[initial_domain] = deque([f"https://{resolved_domain}/"])
                    domain_status["status"] = "TO_BE_STARTED"
                    domain_status["consecutive_empty_queue_fetches"] = 0
                    domain_status["consecutive_errors"] = 0
                    domain_status["pages_fetched"] = 0
    
    logger.info(f"Resolution complete. {len(crawler_state['effective_allowed_domains'])} unique domains ready.")
    save_state()

# --- Crawler Logic ---
def fetch_page(url):
    """Fetches a single page."""
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type:
            return None, f"Non-HTML content-type: {content_type}"

        # Check language detection cache first
        domain = urlparse(url).netloc
        if domain in lang_detection_cache:
            if not lang_detection_cache[domain]:
                return None, f"Domain {domain} cached as non-English"
        else:
            try:
                soup_for_lang = BeautifulSoup(response.content, 'html.parser', from_encoding=response.encoding)
                text_sample = soup_for_lang.get_text(separator=' ', strip=True)[:10000]  # Larger sample as requested
                if not text_sample.strip():
                    return None, "No text content found"
                lang = detect(text_sample)
                is_english = lang == 'en'
                lang_detection_cache[domain] = is_english
                if not is_english:
                    return None, f"Non-English content (lang={lang})"
            except LangDetectException:
                lang_detection_cache[domain] = True  # Assume English if detection fails
            except Exception as e:
                lang_detection_cache[domain] = True  # Assume English on error

        return response.content, response.url
    except requests.HTTPError as e:
        return None, f"HTTP error: {e.response.status_code}"
    except requests.RequestException as e:
        return None, f"Request failed: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error: {e}"

def parse_page(html_content, page_url, current_resolved_domain):
    """Parses HTML, extracts title, content, code snippets, and links."""
    soup = BeautifulSoup(html_content, 'html.parser')  # Faster parser

    title_tag = soup.find('title')
    title = title_tag.string.strip() if title_tag and title_tag.string else urlparse(page_url).path

    # Content extraction - optimized
    main_selectors = ['main', 'article', '[role="main"]', '.main-content', '#content', '#main', '.content', '.documentation']
    main_content = None
    for selector in main_selectors:
        main_content = soup.select_one(selector)
        if main_content:
            break
    
    content_soup = main_content if main_content else soup
    
    # Remove unwanted tags
    for tag in content_soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
        tag.decompose()
    
    # Fast text extraction
    content = content_soup.get_text(separator=' ', strip=True)
    content = re.sub(r'\s+', ' ', content).strip()[:10000]  # Keep larger content limit as requested

    code_snippets = extract_code_snippets(soup)

    # Extract links - optimized
    links = set()
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        try:
            abs_url = urljoin(page_url, href)
            parsed_abs_url = urlparse(abs_url)
            
            if parsed_abs_url.scheme in ['http', 'https']:
                normalized_url = parsed_abs_url._replace(fragment="").geturl()
                link_netloc = parsed_abs_url.netloc
                
                is_current_domain = (link_netloc == current_resolved_domain or 
                                   link_netloc.endswith(f".{current_resolved_domain}"))
                is_allowed_domain = link_netloc in crawler_state["effective_allowed_domains"]
                
                if is_current_domain or is_allowed_domain:
                    links.add(normalized_url)
        except Exception:
            pass
    
    return title, content, code_snippets, list(links)

def process_url_task(initial_domain_key, url_to_fetch):
    """Task function to fetch and process a single URL."""
    domain_s = crawler_state["domains_status"][initial_domain_key]
    current_resolved_domain = domain_s["resolved_domain"]
    
    parsed_url = urlparse(url_to_fetch)
    if not (parsed_url.netloc == current_resolved_domain or 
            parsed_url.netloc.endswith(f".{current_resolved_domain}")):
        return initial_domain_key, url_to_fetch, None, "URL domain mismatch", []

    html_content, final_url_or_error = fetch_page(url_to_fetch)
    
    if html_content is None:
        return initial_domain_key, url_to_fetch, None, final_url_or_error, []

    final_parsed_url = urlparse(final_url_or_error)
    final_netloc = final_parsed_url.netloc
    
    if not (final_netloc == current_resolved_domain or 
            final_netloc.endswith(f".{current_resolved_domain}") or
            final_netloc in crawler_state["effective_allowed_domains"]):
        return initial_domain_key, url_to_fetch, None, f"Redirected to disallowed domain: {final_netloc}", []

    storage_domain_key = get_domain_key(final_netloc)
    storage_resolved_domain = crawler_state["resolved_domains_map"].get(storage_domain_key, final_netloc)
    
    title, content, code_snippets, new_links = parse_page(html_content, final_url_or_error, storage_resolved_domain)

    page_data = {
        "id": hashlib.md5(final_url_or_error.encode()).hexdigest(),
        "url": final_url_or_error,
        "domain": storage_resolved_domain,
        "title": title,
        "content": content,
        "code_snippets": code_snippets,
        "crawled_at": datetime.now(timezone.utc).isoformat()
    }

    try:
        domain_folder_name = sanitize_filename(storage_resolved_domain)
        domain_path = os.path.join(DATA_DIR, domain_folder_name)
        os.makedirs(domain_path, exist_ok=True)
        
        file_path = os.path.join(domain_path, f"{page_data['id']}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(page_data, f, indent=2, ensure_ascii=False)
        
        return initial_domain_key, url_to_fetch, final_url_or_error, page_data, new_links
    except Exception as e:
        return initial_domain_key, url_to_fetch, None, f"Error saving file: {e}", []

# --- Observability ---
def generate_dashboard():
    """Generates the Rich table for the dashboard."""
    global crawler_state
    
    overview_table = Table(title="CRAWLER OVERVIEW", show_header=False, box=None, padding=(0, 1))
    overview_table.add_row("Total Domains:", str(len(crawler_state["domains_status"])))
    
    status_counts = {"TO_BE_STARTED": 0, "ACTIVE": 0, "REACHED_MAX": 0, "GAVE_UP": 0, "COMPLETED_GLOBAL": 0}
    for data in crawler_state["domains_status"].values():
        status_counts[data["status"]] = status_counts.get(data["status"], 0) + 1

    overview_table.add_row("TO_BE_STARTED:", str(status_counts["TO_BE_STARTED"]))
    overview_table.add_row("ACTIVE:", str(status_counts["ACTIVE"]))
    overview_table.add_row("REACHED_MAX:", str(status_counts["REACHED_MAX"]))
    overview_table.add_row("GAVE_UP:", str(status_counts["GAVE_UP"]))
    overview_table.add_row("COMPLETED_GLOBAL:", str(status_counts["COMPLETED_GLOBAL"]))
    overview_table.add_row("Total Pages:", f"{crawler_state['global_pages_fetched']} / {GLOBAL_MAX_PAGES}")

    # All domains table
    all_domains_table = Table(title="ALL DOMAINS STATUS", title_style="bold yellow")
    all_domains_table.add_column("Domain", style="cyan", max_width=40, overflow="fold")
    all_domains_table.add_column("Status", style="magenta", width=12)
    all_domains_table.add_column("Pages", style="blue", justify="right", width=8)
    all_domains_table.add_column("Queue", style="green", justify="right", width=8)
    all_domains_table.add_column("Active", style="yellow", justify="right", width=8)
    all_domains_table.add_column("Errors", style="red", justify="right", width=8)
    all_domains_table.add_column("Updated", style="dim", width=12)

    # Sort domains by status priority, then by name
    sorted_domains = sorted(
        crawler_state["domains_status"].items(),
        key=lambda x: (
            0 if x[1]["status"] == "ACTIVE" else
            1 if x[1]["status"] == "TO_BE_STARTED" else
            2 if x[1]["status"] == "REACHED_MAX" else
            3 if x[1]["status"] == "COMPLETED_GLOBAL" else
            4,  # GAVE_UP
            x[0]  # domain name
        )
    )

    # Show ALL domains
    for domain_key, data in sorted_domains:
        last_update = datetime.fromisoformat(data["last_updated"]).strftime('%H:%M:%S') if data["last_updated"] else "Never"
        resolved = data['resolved_domain']
        display_name = f"{domain_key}" if domain_key == resolved else f"{domain_key}→{resolved}"
        
        # Get active request count
        active_requests = domain_active_requests.get(domain_key, 0)
        
        all_domains_table.add_row(
            display_name,
            data["status"],
            str(data["pages_fetched"]),
            str(len(data["queue"])),
            str(active_requests),
            str(len(data["errors"])),
            last_update
        )

    # Progress stats
    stats_table = Table(title="PROGRESS STATISTICS", show_header=False, box=None, padding=(0, 1))
    
    if crawler_state.get("start_time"):
        start_dt = datetime.fromisoformat(crawler_state["start_time"])
        elapsed = datetime.now(timezone.utc) - start_dt
        elapsed_seconds = elapsed.total_seconds()
        
        if elapsed_seconds > 0:
            pages_per_second = crawler_state['global_pages_fetched'] / elapsed_seconds
            pages_per_hour = pages_per_second * 3600
            
            if pages_per_second > 0:
                eta_seconds = (GLOBAL_MAX_PAGES - crawler_state['global_pages_fetched']) / pages_per_second
                eta_hours = eta_seconds / 3600
                
                stats_table.add_row("Pages/Second:", f"{pages_per_second:.2f}")
                stats_table.add_row("Pages/Hour:", f"{pages_per_hour:.0f}")
                stats_table.add_row("ETA:", f"{eta_hours:.1f} hours")

    # Layout
    layout = Table.grid(expand=True)
    layout.add_column()
    layout.add_row(Text(f"DOCUMENTATION SCRAPER (PRODUCTION) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                       justify="center", style="bold white on blue"))
    layout.add_row(overview_table)
    layout.add_row(all_domains_table)
    layout.add_row(stats_table)
    
    return layout

def get_next_url_for_domain(domain_key):
    """Get the next URL to crawl for a domain (thread-safe)."""
    visited_set = visited_sets.get(domain_key, set())
    queue_set = queue_sets.get(domain_key, set())
    queue_deque = queue_deques.get(domain_key, deque())
    
    while queue_deque:
        url = queue_deque.popleft()
        queue_set.discard(url)  # Remove from set too
        if url not in visited_set:
            return url
    
    return None

def add_url_to_queue(domain_key, url):
    """Add a URL to domain's queue (thread-safe)."""
    if domain_key not in queue_sets:
        queue_sets[domain_key] = set()
        queue_deques[domain_key] = deque()
    
    if url not in queue_sets[domain_key] and url not in visited_sets.get(domain_key, set()):
        queue_sets[domain_key].add(url)
        queue_deques[domain_key].append(url)
        return True
    return False

# --- Main Orchestration ---
def main():
    global crawler_state, GLOBAL_MAX_PAGES, MAX_PAGES_PER_DOMAIN
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Load domains from file
    load_domains_from_file(args.domains)
    
    # Update global configuration with command line arguments
    GLOBAL_MAX_PAGES = args.global_max_pages
    MAX_PAGES_PER_DOMAIN = args.max_pages_per_domain
    
    logger.info(f"Starting crawler for {len(ALLOWED_DOMAINS_INITIAL)} domains...")
    logger.info(f"Configuration:")
    logger.info(f"- Domains file: {args.domains}")
    logger.info(f"- Global max pages: {GLOBAL_MAX_PAGES:,}")
    logger.info(f"- Max pages per domain: {MAX_PAGES_PER_DOMAIN}")
    load_state()

    if not crawler_state["resolved_domains_map"]:
        logger.info("Initial domain resolution required.")
        resolve_initial_domains()
    else:
        logger.info("Using existing resolved domains. Checking for domains that need re-seeding...")
        resolve_initial_domains()  # This will now also handle re-seeding of GAVE_UP domains

    # Check if running with web monitor (disable rich console)
    use_web_monitor = args.web_monitor or os.environ.get('WEB_MONITOR') == '1'
    
    if use_web_monitor:
        logger.info("Running in web monitor mode - console dashboard disabled")
        logger.info("Access web dashboard at http://localhost:5001")
        # Run without Rich Live display
        run_crawler_loop()
    else:
        # Run with Rich Live display
        with Live(generate_dashboard(), refresh_per_second=1.0/REFRESH_INTERVAL, console=console) as live:
            run_crawler_loop(live)

def run_crawler_loop(live=None):
    global crawler_state, visited_sets, queue_sets, queue_deques
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        last_save_time = time.monotonic()
        active_futures = {}  # future -> (domain_key, url)
        active_urls = set()
        
        # Round-robin domain index
        domain_keys = list(crawler_state["domains_status"].keys())
        domain_index = 0
        
        logger.info(f"Starting main loop with {MAX_WORKERS} workers")
        
        while crawler_state["global_pages_fetched"] < GLOBAL_MAX_PAGES:
            # Periodic state save
            if time.monotonic() - last_save_time > 30:
                save_state()
                last_save_time = time.monotonic()
                logger.info(f"State saved. Active workers: {len(active_futures)}/{MAX_WORKERS}")

            # Check if any domains are still active
            active_domains_exist = any(
                ds["status"] in ["ACTIVE", "TO_BE_STARTED"] 
                for ds in crawler_state["domains_status"].values()
            )
            
            if not active_domains_exist and crawler_state["global_pages_fetched"] > 0:
                logger.info("No more active domains. Stopping.")
                break

            # OPTIMIZATION: Keep worker pool full
            while len(active_futures) < MAX_WORKERS:
                # Find a domain with work
                attempts = 0
                url_submitted = False
                
                while attempts < len(domain_keys) and not url_submitted:
                    domain_key = domain_keys[domain_index % len(domain_keys)]
                    domain_index += 1
                    attempts += 1
                    
                    with state_lock:
                        domain_s = crawler_state["domains_status"][domain_key]
                        
                        if domain_s["status"] in ["REACHED_MAX", "GAVE_UP", "COMPLETED_GLOBAL"]:
                            continue
                        
                        # Check domain page limit
                        if domain_s["pages_fetched"] >= MAX_PAGES_PER_DOMAIN:
                            domain_s["status"] = "REACHED_MAX"
                            logger.info(f"Domain {domain_key} reached max pages limit")
                            continue
                        
                        # Activate domain if needed
                        if domain_s["status"] == "TO_BE_STARTED":
                            domain_s["status"] = "ACTIVE"
                            active_count = len([d for d in crawler_state['domains_status'].values() if d['status'] == 'ACTIVE'])
                            logger.info(f"Domain {domain_key} now ACTIVE ({active_count} total active)")
                        
                        if domain_s["status"] == "ACTIVE":
                            # Check rate limiting before fetching URL
                            if can_make_request(domain_key):
                                url_to_fetch = get_next_url_for_domain(domain_key)
                                
                                if url_to_fetch and url_to_fetch not in active_urls:
                                    # Acquire request slot
                                    acquire_request_slot(domain_key)
                                    # Submit the task
                                    future = executor.submit(process_url_task, domain_key, url_to_fetch)
                                    active_futures[future] = (domain_key, url_to_fetch)
                                    active_urls.add(url_to_fetch)
                                    url_submitted = True
                            elif not queue_deques.get(domain_key):
                                domain_s["consecutive_empty_queue_fetches"] += 1
                                if domain_s["consecutive_empty_queue_fetches"] >= MAX_CONSECUTIVE_EMPTY_QUEUE_FETCHES:
                                    domain_s["status"] = "GAVE_UP"
                                    logger.info(f"Domain {domain_key} GAVE_UP (empty queue)")
                            else:
                                domain_s["consecutive_empty_queue_fetches"] = 0
                
                if not url_submitted:
                    # No work available, break inner loop
                    break
            
            # Process completed futures
            completed_futures = []
            for future in active_futures:
                if future.done():
                    completed_futures.append(future)
            
            for future in completed_futures:
                domain_key, original_url = active_futures.pop(future)
                active_urls.discard(original_url)
                
                # Release the request slot for this domain
                release_request_slot(domain_key)
                
                try:
                    domain_key, original_url, final_url, result, new_links = future.result()
                    
                    with state_lock:
                        domain_s = crawler_state["domains_status"][domain_key]
                        domain_s["last_updated"] = datetime.now(timezone.utc).isoformat()

                        if isinstance(result, dict):  # Success
                            # Update visited set
                            visited_sets[domain_key].add(final_url)
                            domain_s["pages_fetched"] += 1
                            crawler_state["global_pages_fetched"] += 1
                            domain_s["consecutive_errors"] = 0

                            # Add new links to appropriate queues
                            for link in new_links:
                                parsed_link = urlparse(link)
                                link_netloc = parsed_link.netloc
                                
                                # Find target domain
                                target_domain_key = None
                                for init_dk, res_dn in crawler_state["resolved_domains_map"].items():
                                    if link_netloc == res_dn or link_netloc.endswith(f".{res_dn}"):
                                        target_domain_key = init_dk
                                        break
                                
                                if target_domain_key:
                                    target_domain_s = crawler_state["domains_status"][target_domain_key]
                                    if target_domain_s["pages_fetched"] < MAX_PAGES_PER_DOMAIN:
                                        add_url_to_queue(target_domain_key, link)

                            if crawler_state["global_pages_fetched"] >= GLOBAL_MAX_PAGES:
                                logger.info("Global page limit reached!")
                                for d_key, d_status in crawler_state["domains_status"].items():
                                    if d_status["status"] == "ACTIVE":
                                        d_status["status"] = "COMPLETED_GLOBAL"
                                break

                        else:  # Error
                            error_msg = result
                            domain_s["errors"].append(error_msg)
                            domain_s["consecutive_errors"] += 1
                            
                            if domain_s["consecutive_errors"] >= MAX_ERRORS_PER_DOMAIN_BEFORE_GIVE_UP:
                                domain_s["status"] = "GAVE_UP"
                                logger.error(f"Domain {domain_key} GAVE_UP due to errors")
                        
                        # Update visited set
                        if original_url not in visited_sets.get(domain_key, set()):
                            visited_sets[domain_key].add(original_url)

                except Exception as e:
                    logger.error(f"Error processing future: {e}", exc_info=True)

            # Update display if Live console is active
            if live:
                live.update(generate_dashboard())
            
            # Log progress every 100 pages
            if crawler_state["global_pages_fetched"] % 100 == 0 and crawler_state["global_pages_fetched"] > 0:
                logger.info(f"Progress: {crawler_state['global_pages_fetched']}/{GLOBAL_MAX_PAGES} pages crawled. Active workers: {len(active_futures)}")
            
            # Brief sleep to prevent CPU spinning
            time.sleep(0.01)

    logger.info("Crawling completed!")
    save_state()
    
    # Only print dashboard if not using web monitor
    if live:
        console.print(generate_dashboard())
    
    logger.info(f"Total pages fetched: {crawler_state['global_pages_fetched']}")
    
    # Summary statistics
    successful_domains = sum(1 for d in crawler_state["domains_status"].values() if d["pages_fetched"] > 0)
    logger.info(f"Successfully crawled {successful_domains} out of {len(ALLOWED_DOMAINS_INITIAL)} domains")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Saving state...")
        save_state()
        logger.info("State saved. Exiting.")
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        save_state()