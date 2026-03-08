import requests
import sqlite3
import json
from datetime import datetime

# Config
INCREMENTAL_MODE = True  # Set to False for bulk download of all events

conn = sqlite3.connect("polymarket.db")
cur = conn.cursor()
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
conn.commit()

def get_last_end_date():
    """Get the most recent end_date from the database for incremental updates."""
    cur.execute("SELECT MAX(end_date) FROM events")
    result = cur.fetchone()[0]
    if result:
        print(f"[db] Last end_date in database: {result}")
        return result
    return None

def fetch_all_events():
    url = "https://gamma-api.polymarket.com/events"
    offset = 0
    limit = 500
    total = 0

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
            cur.execute("""
                INSERT OR REPLACE INTO events 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                e.get("id"),
                e.get("title"),
                e.get("slug"),
                e.get("seriesSlug"),
                e.get("endDate"),
                e.get("active"),
                e.get("closed"),
                e.get("liquidity"),
                e.get("volume"),
                json.dumps(e.get("markets", []))
            ))
            total += 1
        
        conn.commit()
        offset += limit
        print(f"Processed {offset} events, total inserted/updated: {total}...")

        if len(r) < limit:
            break

    print(f"\n{'─'*50}")
    print(f"Done. Total events inserted/updated: {total}")
    print(f"      Database: polymarket.db")
    return total

if __name__ == "__main__":
    fetch_all_events()
    conn.close()