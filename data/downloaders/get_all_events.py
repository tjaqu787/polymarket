import requests
import sqlite3
import json
from datetime import datetime
import os

# Config
INCREMENTAL_MODE = True  # Set to False for bulk download of all events

conn = sqlite3.connect("../polymarket.db")
cur = conn.cursor()

# Create events table
cur.execute("""
    CREATE TABLE IF NOT EXISTS events (
        id TEXT PRIMARY KEY,
        title TEXT,
        slug TEXT,
        series_slug TEXT,
        end_date TEXT,
        active INTEGER,
        closed INTEGER,
        liquidity REAL,
        volume REAL,
        markets TEXT
    )
""")

# Check if markets table exists
cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='markets'")
markets_exists = cur.fetchone() is not None

# If markets table doesn't exist, create it using SQL query file
if not markets_exists:
    print("[setup] markets table doesn't exist, creating from SQL file...")
    sql_file = "../queries/01_flatten_markets.sql"
    if os.path.exists(sql_file):
        with open(sql_file, 'r') as f:
            sql_script = f.read()
        cur.executescript(sql_script)
        print("[setup] markets table created successfully")
    else:
        print(f"[error] SQL file not found: {sql_file}")

# Check if market_tokens view exists
cur.execute("SELECT name FROM sqlite_master WHERE type='view' AND name='market_tokens'")
market_tokens_exists = cur.fetchone() is not None

# If market_tokens view doesn't exist, create it using SQL query file
if not market_tokens_exists:
    print("[setup] market_tokens view doesn't exist, creating from SQL file...")
    sql_file = "../queries/02_market_tokens_view.sql"
    if os.path.exists(sql_file):
        with open(sql_file, 'r') as f:
            sql_script = f.read()
        cur.executescript(sql_script)
        print("[setup] market_tokens view created successfully")
    else:
        print(f"[error] SQL file not found: {sql_file}")


conn.commit()

def get_last_end_date():
    """Get the most recent end_date from the database for incremental updates."""
    cur.execute("SELECT MAX(end_date) FROM events")
    result = cur.fetchone()[0]
    if result:
        print(f"[db] Last end_date in database: {result}")
        return result
    return None


def flatten_and_insert_markets(event_id, markets_list):
    """
    Flatten markets from event JSON and insert into markets table.
    market_tokens view will automatically reflect the changes.
    """
    if not markets_list:
        return 0

    markets_inserted = 0

    for market in markets_list:
        # Extract all market fields
        market_id = market.get("id")
        if not market_id:
            continue

        # Prepare market data tuple (must match table column order)
        market_data = (
            market_id,
            event_id,
            market.get("question"),
            market.get("slug"),
            market.get("description"),
            market.get("category"),
            market.get("marketType"),
            market.get("createdAt"),
            market.get("updatedAt"),
            market.get("startDate"),
            market.get("endDate"),
            market.get("closedTime"),
            market.get("endDateIso"),
            market.get("startDateIso"),
            market.get("active"),
            market.get("closed"),
            market.get("archived"),
            market.get("restricted"),
            market.get("wideFormat"),
            market.get("new"),
            market.get("sentDiscord"),
            market.get("featured"),
            market.get("approved"),
            market.get("ready"),
            market.get("funded"),
            market.get("cyom"),
            market.get("fpmmLive"),
            market.get("clearBookOnStart"),
            market.get("manualActivation"),
            market.get("negRiskOther"),
            market.get("pendingDeployment"),
            market.get("deploying"),
            market.get("hasReviewedDates"),
            market.get("readyForCron"),
            market.get("volumeNum"),
            market.get("liquidityNum"),
            market.get("volume"),
            market.get("liquidity"),
            market.get("bestBid"),
            market.get("bestAsk"),
            market.get("spread"),
            market.get("lastTradePrice"),
            market.get("volume24hr"),
            market.get("volume1wk"),
            market.get("volume1mo"),
            market.get("volume1yr"),
            market.get("volume1wkAmm"),
            market.get("volume1moAmm"),
            market.get("volume1yrAmm"),
            market.get("volume1wkClob"),
            market.get("volume1moClob"),
            market.get("volume1yrClob"),
            market.get("oneDayPriceChange"),
            market.get("oneHourPriceChange"),
            market.get("oneWeekPriceChange"),
            market.get("oneMonthPriceChange"),
            market.get("oneYearPriceChange"),
            json.dumps(market.get("outcomes")) if market.get("outcomes") else None,
            json.dumps(market.get("outcomePrices")) if market.get("outcomePrices") else None,
            market.get("umaResolutionStatus"),
            json.dumps(market.get("umaResolutionStatuses")) if market.get("umaResolutionStatuses") else None,
            market.get("resolutionSource"),
            market.get("resolvedBy"),
            market.get("conditionId"),
            market.get("marketMakerAddress"),
            json.dumps(market.get("clobTokenIds")) if market.get("clobTokenIds") else None,
            market.get("fee"),
            market.get("rewardsMinSize"),
            market.get("rewardsMaxSpread"),
            market.get("competitive"),
            market.get("pagerDutyNotificationEnabled"),
            market.get("rfqEnabled"),
            market.get("holdingRewardsEnabled"),
            market.get("feesEnabled"),
            market.get("requiresTranslation"),
            market.get("image"),
            market.get("icon"),
            market.get("twitterCardLocation"),
            market.get("twitterCardLastRefreshed"),
            market.get("submitted_by"),
            market.get("creator"),
            market.get("updatedBy"),
            market.get("feeType"),
        )

        # Insert market
        cur.execute("""
            INSERT OR REPLACE INTO markets VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
                ?,?,?,?,?,?,?,?,?,?,?
            )
        """, market_data)
        markets_inserted += 1

    return markets_inserted

def fetch_all_events():
    url = "https://gamma-api.polymarket.com/events"
    offset = 0
    limit = 500
    total_events = 0
    total_markets = 0

    # Get the last end_date for incremental mode
    params = {"limit": limit, "offset": offset}
    if INCREMENTAL_MODE:
        last_end_date = get_last_end_date()
        if last_end_date:
            params["end_date_min"] = last_end_date
            print(f"[mode] INCREMENTAL - fetching events with end_date >= {last_end_date}")
        else:
            print("[mode] INCREMENTAL - no existing data, fetching all events")
    else:
        print("[mode] BULK - fetching all events")

    while True:
        params["offset"] = offset
        r = requests.get(url, params=params).json()

        if not r:
            break

        for e in r:
            event_id = e.get("id")
            markets_list = e.get("markets", [])

            # Insert event with markets JSON (kept for backup/reference)
            cur.execute("""
                INSERT OR REPLACE INTO events
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event_id,
                e.get("title"),
                e.get("slug"),
                e.get("seriesSlug"),
                e.get("endDate"),
                e.get("active"),
                e.get("closed"),
                e.get("liquidity"),
                e.get("volume"),
                json.dumps(markets_list)
            ))
            total_events += 1

            # Flatten and insert markets (market_tokens view auto-updates)
            if markets_list:
                num_markets = flatten_and_insert_markets(event_id, markets_list)
                total_markets += num_markets

        conn.commit()
        offset += limit
        print(f"Processed {offset} events | Events: {total_events} | Markets: {total_markets}")

        if len(r) < limit:
            break

    print(f"\n{'─'*50}")
    print(f"Done!")
    print(f"  Events inserted/updated:  {total_events}")
    print(f"  Markets inserted/updated: {total_markets}")
    print(f"  Tokens auto-populated via market_tokens view")
    print(f"  Database: polymarket.db")
    return total_events, total_markets

if __name__ == "__main__":
    fetch_all_events()
    conn.close()