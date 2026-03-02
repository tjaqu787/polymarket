import requests
import sqlite3
import json

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

def fetch_all_events():
    url = "https://gamma-api.polymarket.com/events"
    offset = 0
    limit = 500
    total = 0
    
    while True:
        params = {"limit": limit, "offset": offset}
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
        print(f"Fetched {offset} events, total so far: {total}...")
        
        if len(r) < limit:
            break
    
    print(f"Done. Total events: {total}")

fetch_all_events()
conn.close()