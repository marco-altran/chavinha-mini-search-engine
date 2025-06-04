#!/usr/bin/env python3
"""
Create a smaller summary of the crawler state for the web monitor.
This avoids loading the massive 70MB+ state file.
"""
import json
import os
import time

STATE_FILE = "crawled_data_production/crawler_state.json"
SUMMARY_FILE = "crawled_data_production/crawler_state_summary.json"

def create_summary():
    """Create a summary of the crawler state."""
    try:
        if not os.path.exists(STATE_FILE):
            print("State file not found")
            return
            
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
        
        # Create summary with essential info
        summary = {
            "overview": {
                "total_domains": len(state.get("domains_status", {})),
                "global_pages_fetched": state.get("global_pages_fetched", 0),
                "global_max_pages": 2000000,
                "start_time": state.get("start_time"),
                "last_save_time": state.get("last_save_time")
            },
            "domains": []
        }
        
        # Add domain summaries
        for domain, data in state.get("domains_status", {}).items():
            summary["domains"].append({
                "name": domain,
                "resolved": data.get("resolved_domain", domain),
                "status": data.get("status", "UNKNOWN"),
                "pages_fetched": data.get("pages_fetched", 0),
                "queue_size": len(data.get("queue", [])),
                "errors": len(data.get("errors", [])),
                "last_error": data.get("errors", [])[-1] if data.get("errors") else None,
                "last_updated": data.get("last_updated")
            })
        
        # Save summary
        with open(SUMMARY_FILE, 'w') as f:
            json.dump(summary, f)
            
        print(f"Created summary with {len(summary['domains'])} domains")
        
    except Exception as e:
        print(f"Error creating summary: {e}")

if __name__ == "__main__":
    while True:
        create_summary()
        time.sleep(10)  # Update every 10 seconds