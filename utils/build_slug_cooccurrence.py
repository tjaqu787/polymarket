#!/usr/bin/env python3
import argparse
import re
import sqlite3
from collections import Counter
from itertools import combinations

STOPWORDS = {
    # Common words
    "a","an","and","are","as","at","be","by","for","from","has","have","how",
    "if","in","is","it","its","of","on","or","s","she","he","they","the","to",
    "was","were","will","with","what","when","where","who","whom","why","would",
    "vs","versus","before","until","yes","no",
    # Months (full names)
    "january","february","march","april","may","june","july","august",
    "september","october","november","december",
    # Months (short forms)
    "jan","feb","mar","apr","jun","jul","aug","sep","sept","oct","nov","dec",
    # Years
    "2020","2021","2022","2023","2024","2025","2026","2027","2028","2029","2030"
}

TOKEN_RE = re.compile(r"[a-z0-9]+")

def normalize_text(x: str) -> str:
    if x is None:
        return ""
    x = str(x).lower()
    x = x.replace("-", " ").replace("_", " ")
    return x

def tokenize(text: str):
    text = normalize_text(text)
    toks = TOKEN_RE.findall(text)
    toks = [t for t in toks if t not in STOPWORDS and len(t) >= 2]
    toks = [t for t in toks if not (t.isdigit() and len(t) <= 2)]
    # De-duplicate within a market so cooccurrence means "appears together in a market"
    return sorted(set(toks))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="polymarket.db")
    ap.add_argument("--source", required=True, help="Timing base view/table name (e.g., timing_markets_base)")
    ap.add_argument("--id-col", default="market_id")
    ap.add_argument("--text-col", default="text_for_tokens", help="Column containing text to tokenize")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA temp_store=MEMORY;")

    limit_sql = f" LIMIT {args.limit}" if args.limit else ""
    q = f"""
        SELECT
            {args.id_col} AS market_id,
            {args.text_col} AS text
        FROM {args.source}
        {limit_sql}
    """

    rows = con.execute(q).fetchall()
    if not rows:
        raise RuntimeError(f"No rows returned from {args.source}. Check view/table name and column names.")

    con.executescript("""
        DROP TABLE IF EXISTS timing_text_features;
        DROP TABLE IF EXISTS timing_text_tokens;
        DROP TABLE IF EXISTS timing_token_stats;
        DROP TABLE IF EXISTS timing_token_cooccurrence;

        CREATE TABLE timing_text_features (
            market_id INTEGER PRIMARY KEY,
            raw_text TEXT,
            token_count INTEGER
        );

        CREATE TABLE timing_text_tokens (
            market_id INTEGER,
            token TEXT
        );

        CREATE TABLE timing_token_stats (
            token TEXT PRIMARY KEY,
            df INTEGER
        );

        CREATE TABLE timing_token_cooccurrence (
            token1 TEXT,
            token2 TEXT,
            co_df INTEGER,
            PRIMARY KEY (token1, token2)
        );
    """)

    token_df = Counter()
    pair_df = Counter()

    for market_id, text in rows:
        text = "" if text is None else str(text)
        tokens = tokenize(text)

        token_df.update(tokens)
        for a, b in combinations(tokens, 2):
            pair_df[(a, b)] += 1

        con.execute(
            "INSERT INTO timing_text_features (market_id, raw_text, token_count) VALUES (?,?,?)",
            (market_id, text, len(tokens))
        )

        con.executemany(
            "INSERT INTO timing_text_tokens (market_id, token) VALUES (?,?)",
            [(market_id, t) for t in tokens]
        )

    con.executemany(
        "INSERT INTO timing_token_stats (token, df) VALUES (?,?)",
        list(token_df.items())
    )

    con.executemany(
        "INSERT INTO timing_token_cooccurrence (token1, token2, co_df) VALUES (?,?,?)",
        [(a, b, c) for (a, b), c in pair_df.items()]
    )

    con.executescript("""
        CREATE INDEX IF NOT EXISTS idx_timing_text_tokens_market ON timing_text_tokens(market_id);
        CREATE INDEX IF NOT EXISTS idx_timing_text_tokens_token ON timing_text_tokens(token);
        CREATE INDEX IF NOT EXISTS idx_timing_token_cooccur_token1 ON timing_token_cooccurrence(token1);
        CREATE INDEX IF NOT EXISTS idx_timing_token_cooccur_token2 ON timing_token_cooccurrence(token2);
    """)

    con.commit()

    n_markets = con.execute("SELECT COUNT(*) FROM timing_text_features").fetchone()[0]
    n_tokens = con.execute("SELECT COUNT(*) FROM timing_token_stats").fetchone()[0]
    n_pairs = con.execute("SELECT COUNT(*) FROM timing_token_cooccurrence").fetchone()[0]
    print(f"[OK] built text features for {n_markets} markets")
    print(f"[OK] unique tokens: {n_tokens}")
    print(f"[OK] unique cooccurring pairs: {n_pairs}")

    print("\nTop tokens by df:")
    for token, df in con.execute("SELECT token, df FROM timing_token_stats ORDER BY df DESC LIMIT 20"):
        print(f"  {token:20s} {df}")

    print("\nTop token pairs by co_df:")
    for t1, t2, co_df in con.execute("SELECT token1, token2, co_df FROM timing_token_cooccurrence ORDER BY co_df DESC LIMIT 20"):
        print(f"  ({t1}, {t2}) {co_df}")

    con.close()

if __name__ == "__main__":
    main()