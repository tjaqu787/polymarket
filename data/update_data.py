#!/usr/bin/env python3
"""
Production data pipeline for Polymarket data.
Runs incremental updates for events, markets, tokens, and price history.

Usage:
    python3 update_data.py              # Run full pipeline
    python3 update_data.py --test       # Run in test mode
    python3 update_data.py --bulk       # Run in bulk mode (fetch all data)
"""

import subprocess
import sys
import os
from datetime import datetime

# Config
DB_PATH = "polymarket.db"
FLATTEN_MARKETS_SQL = "queries/01_flatten_markets.sql"

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {description}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            text=True,
            capture_output=False
        )
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with exit code {e.returncode}")
        return False

def main():
    """Run the full data pipeline."""
    start_time = datetime.now()

    # Parse arguments
    test_mode = "--test" in sys.argv
    bulk_mode = "--bulk" in sys.argv

    if test_mode:
        print("\n⚠️  Running in TEST MODE\n")
    elif bulk_mode:
        print("\n⚠️  Running in BULK MODE - This will fetch all data\n")
    else:
        print("\n✓ Running in INCREMENTAL MODE\n")

    # Check if database exists
    if not os.path.exists(DB_PATH):
        print(f"⚠️  Database {DB_PATH} not found. Running initial setup...")

    # Step 1: Fetch new events from API
    success = run_command(
        "python3 downloaders/get_all_events.py",
        "Step 1/4: Fetching new events from Polymarket API"
    )
    if not success:
        print("\n❌ Pipeline failed at step 1")
        return 1

    # Step 2: Flatten markets from events JSON
    success = run_command(
        f"sqlite3 {DB_PATH} < {FLATTEN_MARKETS_SQL}",
        "Step 2/4: Flattening markets from events"
    )
    if not success:
        print("\n❌ Pipeline failed at step 2")
        return 1

    # Step 3: Populate token IDs for new markets
    success = run_command(
        "python3 downloaders/populate_token_ids.py",
        "Step 3/4: Fetching token IDs for new markets"
    )
    if not success:
        print("\n❌ Pipeline failed at step 3")
        return 1

    # Step 4: Fetch price history
    success = run_command(
        "python3 downloaders/get_pricing.py",
        "Step 4/4: Fetching price history"
    )
    if not success:
        print("\n❌ Pipeline failed at step 4")
        return 1

    # Success summary
    elapsed = datetime.now() - start_time
    print(f"\n{'='*60}")
    print(f"✓ Pipeline completed successfully in {elapsed}")
    print(f"  Database: {DB_PATH}")
    print(f"{'='*60}\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())
