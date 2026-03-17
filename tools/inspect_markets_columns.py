import sqlite3

db = r"C:\Users\jamie\Documents\polymarket.db"
con = sqlite3.connect(db)
cur = con.cursor()

cols = cur.execute("PRAGMA table_info(markets)").fetchall()
print("markets columns:")
for c in cols:
    # c = (cid, name, type, notnull, dflt_value, pk)
    print(f"  {c[1]} ({c[2]})")

con.close()