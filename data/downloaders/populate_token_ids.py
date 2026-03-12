"""
Fetch clobTokenIds from Polymarket Gamma API and store them in the database.
"""

import sqlite3
import requests
import time
import json

# Config
DB_PATH = "polymarket.db"
GAMMA_API_URL = "https://gamma-api.polymarket.com/markets"
RATE_LIMIT = 0.2  # seconds between requests
TEST_MODE = False  # Set to True to test with just 5 markets
INCREMENTAL_MODE = True  # Set to False to reprocess all markets


def create_tokens_table(conn: sqlite3.Connection):
    """Create a table to store token IDs for each market."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS market_tokens (
            market_id   TEXT    NOT NULL,
            token_id    TEXT    NOT NULL,
            outcome     TEXT    NOT NULL,
            token_index INTEGER NOT NULL,
            PRIMARY KEY (market_id, token_index)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_token_id ON market_tokens(token_id)
    """)
    conn.commit()


def get_markets(conn: sqlite3.Connection):
    """Get market IDs from the view. In incremental mode, only get markets without tokens."""
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    if INCREMENTAL_MODE:
        # Only get markets that don't have tokens yet
        cur.execute("""
            SELECT DISTINCT market_id
            FROM bets_for_timing_view
            WHERE market_id IS NOT NULL
              AND market_id NOT IN (SELECT DISTINCT market_id FROM market_tokens)
            ORDER BY created_at DESC
        """)
        mode = "INCREMENTAL - only new markets"
    else:
        # Get all markets
        cur.execute("""
            SELECT DISTINCT market_id
            FROM bets_for_timing_view
            WHERE market_id IS NOT NULL
            ORDER BY created_at DESC
        """)
        mode = "BULK - all markets"

    markets = [row["market_id"] for row in cur.fetchall()]
    print(f"[mode] {mode}")
    return markets


def fetch_token_ids(market_id: str) -> list[str]:
    """Fetch clobTokenIds from Gamma API."""
    try:
        url = f"{GAMMA_API_URL}/{market_id}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        # Parse clobTokenIds (it's a JSON string)
        token_ids_str = data.get("clobTokenIds", "[]")
        outcomes_str = data.get("outcomes", "[]")

        token_ids = json.loads(token_ids_str)
        outcomes = json.loads(outcomes_str)

        return token_ids, outcomes
    except Exception as e:
        print(f"  [error] {e}")
        return [], []


def save_tokens(conn: sqlite3.Connection, market_id: str, token_ids: list[str], outcomes: list[str]):
    """Save token IDs to database."""
    rows = []
    for i, (token_id, outcome) in enumerate(zip(token_ids, outcomes)):
        rows.append((market_id, token_id, outcome, i))

    conn.executemany("""
        INSERT OR REPLACE INTO market_tokens
            (market_id, token_id, outcome, token_index)
        VALUES (?, ?, ?, ?)
    """, rows)
    conn.commit()


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)

    # Create tokens table
    create_tokens_table(conn)

    # Get all markets
    markets = get_markets(conn)
    print(f"[db] Found {len(markets)} markets to process")

    if TEST_MODE:
        markets = markets[:5]
        print(f"[TEST MODE] Processing only {len(markets)} markets\n")

    # Fetch and save token IDs
    success = 0
    errors = 0

    for i, market_id in enumerate(markets, 1):
        print(f"[{i:>4}/{len(markets)}] market={market_id}", end="  ")

        token_ids, outcomes = fetch_token_ids(market_id)

        if token_ids:
            save_tokens(conn, market_id, token_ids, outcomes)
            print(f"→ {len(token_ids)} tokens saved ({', '.join(outcomes)})")
            success += 1
        else:
            print("→ failed")
            errors += 1

        time.sleep(RATE_LIMIT)

    conn.close()

    print(f"\n{'─'*50}")
    print(f"Done.  Success: {success}")
    print(f"       Errors:  {errors}")
    print(f"       Database: {DB_PATH}")
