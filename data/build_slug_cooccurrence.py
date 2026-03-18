#!/usr/bin/env python3
"""
Build slug cooccurrence tables for feature engineering.

This script tokenizes text from a source table and builds:
- timing_text_features: Per-market token counts
- timing_text_tokens: Market-to-token mapping
- timing_token_stats: Token document frequencies
- timing_token_cooccurrence: Token pair cooccurrence counts

Production usage: Modify CONFIG below and run directly.
"""
import re
import sqlite3
from collections import Counter
from itertools import combinations

# ============================================================================
# CONFIGURATION - Modify these parameters for production runs
# ============================================================================

CONFIG = {
    # Database configuration
    'db_path': 'polymarket.db',

    # Source table/view configuration
    'source_table': 'bets_for_timing_view',  # Change this to your source table
    'id_column': 'market_id',
    'text_column': 'market_slug',  # Column containing text to tokenize

    # Output table names (can be customized per run)
    'output_table_prefix': 'timing',  # Will create timing_text_features, timing_text_tokens, etc.

    # Processing limits
    'limit': None,  # Set to integer to limit number of rows processed (None = all rows)

    # Performance tuning
    'batch_size': 1000,  # Batch size for inserts
}

# ============================================================================

STOPWORDS = {
    # Common words
    "a","an","and","are","as","at","be","by","for","from","has","have","how",
    "if","in","is","it","its","of","on","or","s","she","he","they","the","to",
    "was","were","there","will","with","what","when","where","who","whom","why","would",
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

def main(config: dict = None):
    """
    Build slug cooccurrence tables.

    Args:
        config: Optional configuration dict. If None, uses global CONFIG.
    """
    if config is None:
        config = CONFIG

    # Validate configuration
    required_keys = ['db_path', 'source_table', 'id_column', 'text_column', 'output_table_prefix']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    # Extract config values
    db_path = config['db_path']
    source_table = config['source_table']
    id_column = config['id_column']
    text_column = config['text_column']
    output_prefix = config['output_table_prefix']
    limit = config.get('limit')

    # Define output table names based on prefix
    table_features = f"{output_prefix}_text_features"
    table_tokens = f"{output_prefix}_text_tokens"
    table_stats = f"{output_prefix}_token_stats"
    table_cooccurrence = f"{output_prefix}_token_cooccurrence"

    print(f"[INFO] Configuration:")
    print(f"  Database: {db_path}")
    print(f"  Source table: {source_table}")
    print(f"  ID column: {id_column}")
    print(f"  Text column: {text_column}")
    print(f"  Output prefix: {output_prefix}")
    print(f"  Limit: {limit if limit else 'None (all rows)'}")
    print()

    # Connect to database with production-friendly settings
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA temp_store=MEMORY;")
    con.execute("PRAGMA cache_size=-64000;")  # 64MB cache

    # Build query
    limit_sql = f" LIMIT {limit}" if limit else ""
    q = f"""
        SELECT
            {id_column} AS market_id,
            {text_column} AS text
        FROM {source_table}
        {limit_sql}
    """

    print(f"[INFO] Fetching data from {source_table}...")
    rows = con.execute(q).fetchall()
    if not rows:
        raise RuntimeError(f"No rows returned from {source_table}. Check table name and column names.")
    print(f"[INFO] Fetched {len(rows)} rows")

    # Drop and recreate tables with configurable names
    print(f"[INFO] Creating output tables with prefix '{output_prefix}'...")
    con.executescript(f"""
        DROP TABLE IF EXISTS {table_features};
        DROP TABLE IF EXISTS {table_tokens};
        DROP TABLE IF EXISTS {table_stats};
        DROP TABLE IF EXISTS {table_cooccurrence};

        CREATE TABLE {table_features} (
            market_id TEXT PRIMARY KEY,
            raw_text TEXT,
            token_count INTEGER
        );

        CREATE TABLE {table_tokens} (
            market_id TEXT,
            token TEXT
        );

        CREATE TABLE {table_stats} (
            token TEXT PRIMARY KEY,
            df INTEGER
        );

        CREATE TABLE {table_cooccurrence} (
            token1 TEXT,
            token2 TEXT,
            co_df INTEGER,
            PRIMARY KEY (token1, token2)
        );
    """)

    # Process rows and build counters
    print(f"[INFO] Tokenizing and building cooccurrence matrix...")
    token_df = Counter()
    pair_df = Counter()

    batch_size = config.get('batch_size', 1000)
    feature_batch = []
    token_batch = []

    for idx, (market_id, text) in enumerate(rows, 1):
        if idx % 10000 == 0:
            print(f"[INFO] Processed {idx}/{len(rows)} rows...")

        text = "" if text is None else str(text)
        tokens = tokenize(text)

        token_df.update(tokens)
        for a, b in combinations(tokens, 2):
            pair_df[(a, b)] += 1

        feature_batch.append((market_id, text, len(tokens)))
        token_batch.extend([(market_id, t) for t in tokens])

        # Batch insert for performance
        if len(feature_batch) >= batch_size:
            con.executemany(
                f"INSERT INTO {table_features} (market_id, raw_text, token_count) VALUES (?,?,?)",
                feature_batch
            )
            con.executemany(
                f"INSERT INTO {table_tokens} (market_id, token) VALUES (?,?)",
                token_batch
            )
            con.commit()
            feature_batch = []
            token_batch = []

    # Insert remaining batches
    if feature_batch:
        con.executemany(
            f"INSERT INTO {table_features} (market_id, raw_text, token_count) VALUES (?,?,?)",
            feature_batch
        )
    if token_batch:
        con.executemany(
            f"INSERT INTO {table_tokens} (market_id, token) VALUES (?,?)",
            token_batch
        )

    print(f"[INFO] Inserting token statistics...")
    con.executemany(
        f"INSERT INTO {table_stats} (token, df) VALUES (?,?)",
        list(token_df.items())
    )

    print(f"[INFO] Inserting cooccurrence pairs...")
    con.executemany(
        f"INSERT INTO {table_cooccurrence} (token1, token2, co_df) VALUES (?,?,?)",
        [(a, b, c) for (a, b), c in pair_df.items()]
    )

    print(f"[INFO] Creating indexes...")
    con.executescript(f"""
        CREATE INDEX IF NOT EXISTS idx_{output_prefix}_text_tokens_market ON {table_tokens}(market_id);
        CREATE INDEX IF NOT EXISTS idx_{output_prefix}_text_tokens_token ON {table_tokens}(token);
        CREATE INDEX IF NOT EXISTS idx_{output_prefix}_token_cooccur_token1 ON {table_cooccurrence}(token1);
        CREATE INDEX IF NOT EXISTS idx_{output_prefix}_token_cooccur_token2 ON {table_cooccurrence}(token2);
    """)

    con.commit()

    # Summary statistics
    n_markets = con.execute(f"SELECT COUNT(*) FROM {table_features}").fetchone()[0]
    n_tokens = con.execute(f"SELECT COUNT(*) FROM {table_stats}").fetchone()[0]
    n_pairs = con.execute(f"SELECT COUNT(*) FROM {table_cooccurrence}").fetchone()[0]

    print()
    print("="*70)
    print("[SUCCESS] Slug cooccurrence tables built successfully")
    print("="*70)
    print(f"  Markets processed: {n_markets}")
    print(f"  Unique tokens: {n_tokens}")
    print(f"  Cooccurring pairs: {n_pairs}")
    print()
    print(f"  Output tables created:")
    print(f"    - {table_features}")
    print(f"    - {table_tokens}")
    print(f"    - {table_stats}")
    print(f"    - {table_cooccurrence}")
    print()

    print("Top 20 tokens by document frequency:")
    for token, df in con.execute(f"SELECT token, df FROM {table_stats} ORDER BY df DESC LIMIT 20"):
        print(f"  {token:20s} {df:6d}")

    print()
    print("Top 20 token pairs by cooccurrence:")
    for t1, t2, co_df in con.execute(f"SELECT token1, token2, co_df FROM {table_cooccurrence} ORDER BY co_df DESC LIMIT 20"):
        print(f"  ({t1:15s}, {t2:15s}) {co_df:6d}")

    con.close()
    print()
    print("[DONE]")

def run_with_custom_config(**kwargs):
    """
    Run with custom configuration parameters.

    Allows programmatic usage with overrides:
        run_with_custom_config(
            source_table='my_custom_table',
            output_table_prefix='custom_prefix',
            limit=1000
        )

    Args:
        **kwargs: Configuration parameters to override from CONFIG
    """
    custom_config = CONFIG.copy()
    custom_config.update(kwargs)
    main(custom_config)


if __name__ == "__main__":
    # Production usage: modify CONFIG at the top of this file and run directly
    # python data/build_slug_cooccurrence.py

    # Or import and call programmatically:
    # from data.build_slug_cooccurrence import run_with_custom_config
    # run_with_custom_config(source_table='my_table', output_table_prefix='my_prefix')

    main()