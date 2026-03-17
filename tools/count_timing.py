import sqlite3

db = r"C:\Users\jamie\Documents\polymarket.db"
con = sqlite3.connect(db)
cur = con.cursor()

n = cur.execute("SELECT COUNT(*) FROM timing_markets_base").fetchone()[0]
print("timing_markets_base:", n)

rows = cur.execute("""
SELECT market_id, end_date, substr(question,1,120)
FROM timing_markets_base
LIMIT 5
""").fetchall()

print("\nExamples:")
for r in rows:
    print(r)

con.close()