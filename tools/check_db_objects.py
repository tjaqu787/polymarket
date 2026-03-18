import sqlite3
from pathlib import Path

db_path = Path(r"C:\Users\jamie\Documents\polymarket\polymarket.db")

print("exists:", db_path.exists())
if db_path.exists():
    print("size:", db_path.stat().st_size)

con = sqlite3.connect(str(db_path))
rows = con.execute("""
SELECT name, type
FROM sqlite_master
ORDER BY type, name
""").fetchall()

print("objects:", len(rows))
print(rows[:30])

con.close()