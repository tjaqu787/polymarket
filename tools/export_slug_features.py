import os
import sqlite3
import csv

DB = r"C:\Users\jamie\Documents\polymarket.db"
OUTDIR = os.path.join(os.getcwd(), "submissions", "slug_cooccurrence")
os.makedirs(OUTDIR, exist_ok=True)

con = sqlite3.connect(DB)
cur = con.cursor()

def export_query(filename, query, params=()):
    path = os.path.join(OUTDIR, filename)
    rows = cur.execute(query, params).fetchall()
    cols = [d[0] for d in cur.description]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)
    return path, len(rows)

# 1) Basic counts (for the summary)
n_timing = cur.execute("SELECT COUNT(*) FROM timing_markets_base").fetchone()[0]
n_markets_feat = cur.execute("SELECT COUNT(*) FROM timing_text_features").fetchone()[0]
n_tokens = cur.execute("SELECT COUNT(*) FROM timing_token_stats").fetchone()[0]
n_pairs = cur.execute("SELECT COUNT(*) FROM timing_token_cooccurrence").fetchone()[0]

# 2) Export token stats (top 500 tokens)
p1, k1 = export_query(
    "token_stats_top500.csv",
    "SELECT token, df FROM timing_token_stats ORDER BY df DESC LIMIT 500"
)

# 3) Export cooccurrence (top 500 pairs)
p2, k2 = export_query(
    "token_cooccurrence_top500.csv",
    "SELECT token1, token2, co_df FROM timing_token_cooccurrence ORDER BY co_df DESC LIMIT 500"
)

# 4) Export a small sample of tokens-per-market (first 200 markets)
p3, k3 = export_query(
    "tokens_per_market_sample.csv",
    """
    SELECT t.market_id, group_concat(t.token, ' ') AS tokens
    FROM timing_text_tokens t
    GROUP BY t.market_id
    LIMIT 200
    """
)

# 5) Export the timing base set (so Tyrell can subset prices)
p4, k4 = export_query(
    "timing_markets_base_ids.csv",
    "SELECT market_id, event_id, end_date, question FROM timing_markets_base"
)

# 6) Write a summary file
summary_path = os.path.join(OUTDIR, "SUMMARY_slug_cooccurrence.txt")
top_tokens = cur.execute(
    "SELECT token, df FROM timing_token_stats ORDER BY df DESC LIMIT 20"
).fetchall()
top_pairs = cur.execute(
    "SELECT token1, token2, co_df FROM timing_token_cooccurrence ORDER BY co_df DESC LIMIT 20"
).fetchall()

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("Slug/Text tokenization + cooccurrence deliverable\n")
    f.write("===============================================\n\n")
    f.write(f"DB: {DB}\n")
    f.write(f"timing_markets_base count: {n_timing}\n")
    f.write(f"timing_text_features count: {n_markets_feat}\n")
    f.write(f"unique tokens: {n_tokens}\n")
    f.write(f"unique token pairs: {n_pairs}\n\n")

    f.write("Top 20 tokens (by df):\n")
    for tok, df in top_tokens:
        f.write(f"  {tok}\t{df}\n")
    f.write("\nTop 20 token pairs (by co_df):\n")
    for a, b, c in top_pairs:
        f.write(f"  ({a}, {b})\t{c}\n")

    f.write("\nExports created:\n")
    f.write(f"  {p1}\n  {p2}\n  {p3}\n  {p4}\n\n")
    f.write("Reproduce:\n")
    f.write("  python scripts/build_slug_cooccurrence.py --db C:\\Users\\jamie\\Documents\\polymarket.db --source timing_markets_base --id-col market_id --text-col text_for_tokens\n")

con.close()

print("[OK] Wrote exports to:", OUTDIR)
print("[OK] Summary:", summary_path)