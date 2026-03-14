"""
Fetch daily price history from Polymarket CLOB API for all markets in the local DB.
Saves results to: data/polymarket_price_history.db (SQLite).
"""

import sqlite3
import asyncio
import aiohttp
import time
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH          = "../polymarket.db"  # Database is in data/ directory
CLOB_URL         = "https://clob.polymarket.com/prices-history"
INTERVAL         = "max"      # max = all available data (works for new and old markets)
FIDELITY         = None       # Not used with interval=max
CONCURRENT_LIMIT = 20         # Max concurrent requests (increased from 4 req/s)
TEST_MODE        = False      # Set to True to test with just 2 markets
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


def get_last_fetch_time(conn: sqlite3.Connection, market_id: str) -> int:
    """Get the last price fetch time for a market. Returns 0 if not found."""
    cur = conn.cursor()
    cur.execute("""
        SELECT last_price_fetch
        FROM markets
        WHERE market_id = ?
    """, (market_id,))
    result = cur.fetchone()
    return result[0] if result and result[0] else 0


def should_skip_market(conn: sqlite3.Connection, market_id: str, skip_threshold: int = SKIP_THRESHOLD) -> bool:
    """Check if market was fetched recently and should be skipped in incremental mode.

    Args:
        conn: Database connection
        market_id: Market ID to check
        skip_threshold: Skip markets updated within this many seconds (default: 86400 = 24 hours)
    """
    if not INCREMENTAL_MODE:
        return False

    last_ts = get_last_fetch_time(conn, market_id)
    if last_ts == 0:
        return False  # Never fetched, don't skip

    current_ts = int(time.time())
    age = current_ts - last_ts

    return age < skip_threshold


def update_last_fetch_time(conn: sqlite3.Connection, market_id: str):
    """Update the last_price_fetch timestamp for a market."""
    current_ts = int(time.time())
    cur = conn.cursor()
    cur.execute("""
        UPDATE markets
        SET last_price_fetch = ?
        WHERE market_id = ?
    """, (current_ts, market_id))
    conn.commit()


async def fetch_price_history(session: aiohttp.ClientSession, token_id: str,
                               interval: str = INTERVAL, fidelity: int = FIDELITY) -> list[dict]:
    """Call Polymarket /prices-history for one token_id asynchronously."""
    params = {
        "market": token_id,
        "interval": interval,
    }
    if fidelity is not None:
        params["fidelity"] = fidelity

    try:
        async with session.get(CLOB_URL, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            resp.raise_for_status()
            data = await resp.json()
            history = data.get("history", [])
            return history
    except aiohttp.ClientResponseError as e:
        print(f"  [warn] HTTP {e.status} for token {token_id[:20]}...")
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


async def process_token(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore,
                        market_id: str, event_id: str, token_id: str, outcome: str) -> dict:
    """Process a single token with concurrency control."""
    async with semaphore:
        history = await fetch_price_history(session, token_id, INTERVAL, FIDELITY)
        return {
            'market_id': market_id,
            'event_id': event_id,
            'token_id': token_id,
            'outcome': outcome,
            'history': history
        }


async def process_market_tokens(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore,
                                 conn: sqlite3.Connection, market_id: str, event_id: str,
                                 tokens: list[tuple[str, str]]) -> dict:
    """Process all tokens for a market concurrently."""
    tasks = []

    for token_id, outcome in tokens:
        task = process_token(session, semaphore, market_id, event_id, token_id, outcome)
        tasks.append(task)

    if not tasks:
        return {'total_rows': 0, 'errors': 0, 'skipped': 0}

    results = await asyncio.gather(*tasks)

    total_rows = 0
    errors = 0

    for result in results:
        token_id = result['token_id']
        outcome = result['outcome']
        history = result['history']

        if history:
            n = save_rows(conn, result['market_id'], result['event_id'],
                         token_id, outcome, history)
            total_rows += n
            print(f"  [{outcome}] token={token_id[:20]}...  → {n} points")
        else:
            errors += 1
            print(f"  [{outcome}] token={token_id[:20]}...  → no data")

    return {
        'total_rows': total_rows,
        'errors': errors,
        'skipped': 0
    }


async def main(skip_threshold: int = SKIP_THRESHOLD, db_path: str = DB_PATH,
               test_mode: bool = TEST_MODE, incremental_mode: bool = INCREMENTAL_MODE):
    """
    Main function to fetch price history for all markets.

    Args:
        skip_threshold: Skip markets updated within this many seconds (default: 86400 = 24 hours)
        db_path: Path to database (default: ../polymarket.db)
        test_mode: Process only 2 markets for testing (default: False)
        incremental_mode: Skip recently updated markets (default: True)
    """
    # 1. Load market IDs from database
    markets = get_market_ids(db_path)
    if not markets:
        print("[error] No markets found. Check DB path and view name.")
        exit(1)

    # Test mode: only process first 2 markets
    if test_mode:
        markets = markets[:2]
        print(f"[TEST MODE] Processing only {len(markets)} markets\n")

    # Show mode and skip threshold
    mode_str = "INCREMENTAL" if incremental_mode else "BULK"
    skip_hours = skip_threshold / 3600
    print(f"[mode] {mode_str} - {'skipping markets fetched within %.1fh' % skip_hours if incremental_mode else 'fetching all markets'}\n")

    # 2. Open database connection
    conn = sqlite3.connect(db_path)

    # 3. Ensure price_history table exists
    init_price_history_table(conn)

    # 4. Fetch & store price history for each market
    total_rows = 0
    errors = 0
    skipped_markets = 0

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)

    async with aiohttp.ClientSession() as session:
        for i, row in enumerate(markets, 1):
            mid = row["market_id"]
            eid = row["event_id"]
            print(f"[{i:>4}/{len(markets)}] market={mid}  event={eid}")

            # Check if market should be skipped (based on last fetch time)
            if should_skip_market(conn, mid, skip_threshold):
                last_ts = get_last_fetch_time(conn, mid)
                age_hours = (int(time.time()) - last_ts) / 3600
                print(f"  → skipped (fetched {age_hours:.1f}h ago)")
                skipped_markets += 1
                continue

            # Get token IDs from database
            tokens = get_token_ids_from_db(conn, mid)
            if not tokens:
                print(f"  → No tokens found in DB, skipping")
                skipped_markets += 1
                continue

            # Process all tokens for this market concurrently
            result = await process_market_tokens(session, semaphore, conn, mid, eid, tokens)
            total_rows += result['total_rows']
            errors += result['errors']

            # Update last fetch time for this market
            if result['total_rows'] > 0:
                update_last_fetch_time(conn, mid)

    conn.close()

    # 5. Summary
    print(f"\n{'─'*50}")
    print(f"Done.  Markets processed : {len(markets) - skipped_markets}")
    print(f"       Markets skipped   : {skipped_markets}")
    print(f"       Total rows saved  : {total_rows}")
    print(f"       Errors            : {errors}")
    print(f"       Database          : {db_path}")


if __name__ == "__main__":
    asyncio.run(main())
