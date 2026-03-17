import argparse
import sqlite3
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    con = sqlite3.connect(str(db_path))
    cur = con.cursor()

    print(f"DB: {db_path}  ({db_path.stat().st_size/1e6:.2f} MB)\n")

    tables = cur.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table'
        ORDER BY name
    """).fetchall()

    print("Tables (first 60):")
    for (name,) in tables[:60]:
        print(" ", name)
    print(f"... total tables: {len(tables)}\n")

    def count_if_exists(name):
        exists = cur.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (name,)
        ).fetchone()
        if not exists:
            print(f"{name}: (missing)")
            return
        c = cur.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        print(f"{name}: {c}")

    count_if_exists("events")
    count_if_exists("markets")

    timing_objs = cur.execute("""
        SELECT type, name FROM sqlite_master
        WHERE type IN ('table','view') AND name LIKE '%timing%'
        ORDER BY type, name
    """).fetchall()

    print("\nObjects with 'timing' in name:")
    for t, n in timing_objs:
        print(f"  {t}: {n}")

    con.close()

if __name__ == "__main__":
    main()