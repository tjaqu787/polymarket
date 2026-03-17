import argparse
import sqlite3
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to sqlite database file")
    ap.add_argument("--sql", required=True, help="Path to .sql file to execute")
    args = ap.parse_args()

    db_path = Path(args.db)
    sql_path = Path(args.sql)

    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")
    if not sql_path.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")

    print(f"DB:  {db_path} ({db_path.stat().st_size/1e6:.2f} MB)")
    print(f"SQL: {sql_path}")

    sql_text = sql_path.read_text(encoding="utf-8")

    con = sqlite3.connect(str(db_path))
    try:
        con.executescript(sql_text)
        con.commit()
    finally:
        con.close()

    print("[OK] SQL executed successfully.")

if __name__ == "__main__":
    main()