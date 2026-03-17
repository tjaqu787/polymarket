import sqlite3

db = r"C:\Users\jamie\Documents\polymarket.db"
con = sqlite3.connect(db)
cur = con.cursor()

for name in ["timing_text_features","timing_text_tokens","timing_token_stats","timing_token_cooccurrence"]:
    c = cur.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
    print(name, c)

print("\nTop 15 token pairs:")
for row in cur.execute("SELECT token1, token2, co_df FROM timing_token_cooccurrence ORDER BY co_df DESC LIMIT 15"):
    print(row)

con.close()