"""
Fetch daily price history from Polymarket CLOB API for all markets in the local DB.
Saves results to: data/polymarket_price_history.db (SQLite).
"""

import sqlite3
import requests
import time
import os
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH          = "polymarket.db"
GAMMA_API_URL    = "https://gamma-api.polymarket.com/markets"
CLOB_URL         = "https://clob.polymarket.com/prices-history"
INTERVAL         = "1d"       # daily aggregation
FIDELITY         = 1440       # 1440 min = 1 day, matches daily interval
RATE_LIMIT       = 0.15       # seconds between requests (4 req/s is safe)
TEST_MODE        = True       # Set to True to test with just 2 markets
INCREMENTAL_MODE = True       # Set to False to fetch all, True to skip recently updated tokens
SKIP_THRESHOLD   = 86400      # Skip tokens updated within this many seconds (86400 = 24 hours)
# ─────────────────────────────────────────────────────────────────────────────


def get_market_ids(db_path: str) -> list[dict]:
    """Pull distinct market_id + event_id from the view, ordered by creation date DESC."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT market_id, event_id
        FROM bets_for_timing_view
        WHERE market_id IS NOT NULL
        ORDER BY created_at DESC
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    print(f"[db] Found {len(rows)} distinct markets across "
          f"{len(set(r['event_id'] for r in rows))} events")
    return rows


def get_token_ids_from_db(conn: sqlite3.Connection, market_id: str) -> list[tuple[str, str]]:
    """Fetch token IDs and outcomes from the database."""
    cur = conn.cursor()
    cur.execute("""
        SELECT token_id, outcome
        FROM bets_for_timing_view
        WHERE market_id = ?
        ORDER BY token_index
    """, (market_id,))
    return cur.fetchall()


def get_last_update_time(conn: sqlite3.Connection, token_id: str) -> int:
    """Get the most recent timestamp for a token in price_history. Returns 0 if not found."""
    cur = conn.cursor()
    cur.execute("""
        SELECT MAX(ts) as last_ts
        FROM price_history
        WHERE token_id = ?
    """, (token_id,))
    result = cur.fetchone()
    return result[0] if result and result[0] else 0


def should_skip_token(conn: sqlite3.Connection, token_id: str) -> bool:
    """Check if token was updated recently and should be skipped in incremental mode."""
    if not INCREMENTAL_MODE:
        return False

    last_ts = get_last_update_time(conn, token_id)
    if last_ts == 0:
        return False  # Never fetched, don't skip

    current_ts = int(time.time())
    age = current_ts - last_ts

    return age < SKIP_THRESHOLD


def fetch_price_history(token_id: str, interval: str = INTERVAL,
                        fidelity: int = FIDELITY) -> list[dict]:
    """Call Polymarket /prices-history for one token_id."""
    params = {
        "market":   token_id,
        "interval": interval,
        "fidelity": fidelity,
    }
    try:
        resp = requests.get(CLOB_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        history = data.get("history", [])
        return history
    except requests.exceptions.HTTPError as e:
        print(f"  [warn] HTTP {resp.status_code} for token {token_id[:20]}...")
        print(f"  [debug] Response: {resp.text}")
        return []
    except Exception as e:
        print(f"  [error] token {token_id[:20]}...: {e}")
        return []


def init_price_history_table(conn: sqlite3.Connection):
    """Ensure the price_history table exists in the database."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            market_id   TEXT    NOT NULL,
            event_id    TEXT,
            token_id    TEXT    NOT NULL,
            outcome     TEXT,              -- Yes/No
            ts          INTEGER NOT NULL,  -- unix timestamp
            date        TEXT    NOT NULL,  -- YYYY-MM-DD for readability
            price       REAL    NOT NULL,
            PRIMARY KEY (token_id, ts)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_event ON price_history(event_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_market ON price_history(market_id)
    """)
    conn.commit()


def save_rows(conn: sqlite3.Connection, market_id: str, event_id: str,
              token_id: str, outcome: str, history: list[dict]) -> int:
    """Upsert price rows into output DB. Returns count inserted."""
    rows = []
    for point in history:
        ts = point.get("t")
        price = point.get("p")
        if ts is None or price is None:
            continue
        date_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        rows.append((market_id, event_id, token_id, outcome, ts, date_str, price))

    conn.executemany("""
        INSERT OR REPLACE INTO price_history
            (market_id, event_id, token_id, outcome, ts, date, price)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()
    return len(rows)



if __name__ == "__main__":
    # 1. Load market IDs from database
    markets = get_market_ids(DB_PATH)
    if not markets:
        print("[error] No markets found. Check DB path and view name.")
        exit(1)

    # Test mode: only process first 2 markets
    if TEST_MODE:
        markets = markets[:2]
        print(f"[TEST MODE] Processing only {len(markets)} markets\n")

    # Show mode
    mode_str = "INCREMENTAL" if INCREMENTAL_MODE else "BULK"
    print(f"[mode] {mode_str} - {'skipping recently updated tokens' if INCREMENTAL_MODE else 'fetching all tokens'}\n")

    # 2. Open database connection
    conn = sqlite3.connect(DB_PATH)

    # 3. Ensure price_history table exists
    init_price_history_table(conn)

    # 4. Fetch & store price history for each market
    total_rows = 0
    errors     = 0
    skipped_markets = 0
    skipped_tokens  = 0

    for i, row in enumerate(markets, 1):
        mid = row["market_id"]
        eid = row["event_id"]
        print(f"[{i:>4}/{len(markets)}] market={mid}  event={eid}")

        # Get token IDs from database
        tokens = get_token_ids_from_db(conn, mid)
        if not tokens:
            print(f"  → No tokens found in DB, skipping")
            skipped_markets += 1
            continue

        # Fetch price history for each outcome token
        for token_id, outcome in tokens:
            # Check if we should skip this token
            if should_skip_token(conn, token_id):
                last_ts = get_last_update_time(conn, token_id)
                age_hours = (int(time.time()) - last_ts) / 3600
                print(f"  [{outcome}] token={token_id[:20]}...  → skipped (updated {age_hours:.1f}h ago)")
                skipped_tokens += 1
                continue

            print(f"  [{outcome}] token={token_id[:20]}...", end="  ")
            history = fetch_price_history(token_id, INTERVAL, FIDELITY)

            if history:
                n = save_rows(conn, mid, eid, token_id, outcome, history)
                total_rows += n
                print(f"→ {n} points")
            else:
                errors += 1
                print("→ no data")

            time.sleep(RATE_LIMIT)

    conn.close()

    # 5. Summary
    print(f"\n{'─'*50}")
    print(f"Done.  Markets processed : {len(markets)}")
    print(f"       Markets skipped   : {skipped_markets}")
    print(f"       Tokens skipped    : {skipped_tokens}")
    print(f"       Total rows saved  : {total_rows}")
    print(f"       Errors            : {errors}")
    print(f"       Database          : {DB_PATH}")