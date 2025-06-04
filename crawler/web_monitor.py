#!/usr/bin/env python3
"""
Web-based monitoring dashboard for the documentation scraper.
Provides real-time updates, filtering, and better visibility for all domains.
"""

import os
import json
import time
from datetime import datetime, timezone
from flask import Flask, render_template, jsonify
import threading
import logging

# Configuration
DATA_DIR = "crawled_data"
STATE_FILE = os.path.join(DATA_DIR, "crawler_state.json")
SUMMARY_FILE = os.path.join(DATA_DIR, "crawler_state_summary.json")
GLOBAL_MAX_PAGES = 2000000  # Match production settings

app = Flask(__name__)
logging.getLogger('werkzeug').setLevel(logging.WARNING)

def load_crawler_summary():
    """Load the current crawler state summary."""
    try:
        # First try the summary file (small and fast)
        if os.path.exists(SUMMARY_FILE):
            with open(SUMMARY_FILE, 'r') as f:
                summary = json.load(f)
                print(f"Loaded summary with {len(summary.get('domains', []))} domains")
                return summary
    except Exception as e:
        print(f"Error loading summary: {e}")
    
    # Fallback: return empty state
    return {
        "overview": {
            "total_domains": 0,
            "global_pages_fetched": 0,
            "global_max_pages": GLOBAL_MAX_PAGES,
            "start_time": None,
            "last_save_time": None
        },
        "domains": []
    }

def load_crawler_state():
    """Load the current crawler state from file."""
    for attempt in range(3):
        try:
            if os.path.exists(STATE_FILE):
                # Get file size
                file_size = os.path.getsize(STATE_FILE)
                if file_size > 100 * 1024 * 1024:  # 100MB
                    print(f"Warning: State file is very large ({file_size / 1024 / 1024:.1f}MB)")
                
                # Try to read with timeout
                with open(STATE_FILE, 'r') as f:
                    content = f.read()
                    
                if not content or content.strip() == "":
                    print(f"Attempt {attempt + 1}: Empty state file")
                    time.sleep(0.5)
                    continue
                    
                data = json.loads(content)
                
                # Validate data
                if not isinstance(data, dict):
                    print(f"Attempt {attempt + 1}: Invalid data format")
                    time.sleep(0.5)
                    continue
                
                # Log some stats
                if data.get("domains_status"):
                    print(f"Loaded state with {len(data['domains_status'])} domains, "
                          f"{data.get('global_pages_fetched', 0)} pages fetched")
                return data
                
        except json.JSONDecodeError as e:
            print(f"Attempt {attempt + 1}: Error decoding JSON: {e}")
            time.sleep(0.5)
        except MemoryError as e:
            print(f"Memory error loading state file: {e}")
            break
        except Exception as e:
            print(f"Attempt {attempt + 1}: Error loading state: {e}")
            time.sleep(0.5)
    
    print("Failed to load state after 3 attempts, returning empty state")
    return {
        "domains_status": {},
        "global_pages_fetched": 0,
        "start_time": None,
        "last_save_time": None
    }

def calculate_stats(crawler_state):
    """Calculate statistics from crawler state."""
    status_counts = {"TO_BE_STARTED": 0, "ACTIVE": 0, "REACHED_MAX": 0, "GAVE_UP": 0, "COMPLETED_GLOBAL": 0}
    
    for data in crawler_state.get("domains_status", {}).values():
        status = data.get("status", "UNKNOWN")
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Calculate performance metrics
    performance = {}
    if crawler_state.get("start_time"):
        try:
            start_dt = datetime.fromisoformat(crawler_state["start_time"])
            elapsed = datetime.now(timezone.utc) - start_dt
            elapsed_seconds = elapsed.total_seconds()
            
            if elapsed_seconds > 0:
                pages_per_second = crawler_state.get("global_pages_fetched", 0) / elapsed_seconds
                pages_per_hour = pages_per_second * 3600
                
                if pages_per_second > 0:
                    eta_seconds = (GLOBAL_MAX_PAGES - crawler_state.get("global_pages_fetched", 0)) / pages_per_second
                    eta_hours = eta_seconds / 3600
                    
                    performance = {
                        "pages_per_second": round(pages_per_second, 2),
                        "pages_per_hour": round(pages_per_hour, 0),
                        "eta_hours": round(eta_hours, 1),
                        "elapsed_seconds": elapsed_seconds
                    }
        except Exception as e:
            print(f"Error calculating performance: {e}")
    
    return status_counts, performance

@app.route('/')
def dashboard():
    """Main dashboard page."""
    return render_template('dashboard.html')

@app.route('/api/status')
def api_status():
    """API endpoint for current crawler status using summary."""
    summary = load_crawler_summary()
    
    # Calculate status counts from domains
    status_counts = {"TO_BE_STARTED": 0, "ACTIVE": 0, "REACHED_MAX": 0, "GAVE_UP": 0, "COMPLETED_GLOBAL": 0}
    for domain in summary.get("domains", []):
        status = domain.get("status", "UNKNOWN")
        if status in status_counts:
            status_counts[status] += 1
    
    # Calculate performance metrics
    performance = {}
    overview = summary.get("overview", {})
    if overview.get("start_time"):
        try:
            start_dt = datetime.fromisoformat(overview["start_time"])
            elapsed = datetime.now(timezone.utc) - start_dt
            elapsed_seconds = elapsed.total_seconds()
            
            if elapsed_seconds > 0:
                pages_fetched = overview.get("global_pages_fetched", 0)
                pages_per_second = pages_fetched / elapsed_seconds
                pages_per_hour = pages_per_second * 3600
                
                if pages_per_second > 0:
                    eta_seconds = (GLOBAL_MAX_PAGES - pages_fetched) / pages_per_second
                    eta_hours = eta_seconds / 3600
                    
                    performance = {
                        "pages_per_second": round(pages_per_second, 2),
                        "pages_per_hour": round(pages_per_hour, 0),
                        "eta_hours": round(eta_hours, 1),
                        "elapsed_seconds": elapsed_seconds
                    }
        except Exception as e:
            print(f"Error calculating performance: {e}")
    
    # Sort domains by pages fetched (descending order)
    domains = summary.get("domains", [])
    domains_sorted = sorted(domains, key=lambda x: x.get("pages_fetched", 0), reverse=True)
    
    # Return the summary structure with sorted domains
    return jsonify({
        "overview": overview,
        "status_counts": status_counts,
        "performance": performance,
        "domains": domains_sorted
    })

@app.route('/api/domain/<domain_name>')
def api_domain_detail(domain_name):
    """API endpoint for detailed domain information."""
    crawler_state = load_crawler_state()
    domain_data = crawler_state.get("domains_status", {}).get(domain_name)
    
    if not domain_data:
        return jsonify({"error": "Domain not found"}), 404
    
    # Count actual files for this domain
    resolved_domain = domain_data.get('resolved_domain', domain_name)
    domain_folder = os.path.join(DATA_DIR, resolved_domain.replace('/', '_'))
    actual_files = 0
    if os.path.exists(domain_folder):
        actual_files = len([f for f in os.listdir(domain_folder) if f.endswith('.json')])
    
    return jsonify({
        "domain": domain_name,
        "resolved_domain": resolved_domain,
        "status": domain_data.get("status"),
        "pages_fetched": domain_data.get("pages_fetched", 0),
        "actual_files": actual_files,
        "queue_size": len(domain_data.get("queue", [])),
        "visited_count": len(domain_data.get("visited", [])),
        "errors": domain_data.get("errors", []),
        "last_updated": domain_data.get("last_updated"),
        "consecutive_errors": domain_data.get("consecutive_errors", 0),
        "consecutive_empty_queue_fetches": domain_data.get("consecutive_empty_queue_fetches", 0),
        "sample_queue": domain_data.get("queue", [])[:10]  # First 10 URLs in queue
    })

if __name__ == '__main__':
    print("Starting web monitoring dashboard...")
    print("Access at: http://localhost:5001")
    print("Press Ctrl+C to stop")
    
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)