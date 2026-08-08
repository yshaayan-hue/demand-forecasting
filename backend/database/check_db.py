from pathlib import Path
import sqlite3

DB_PATH = Path(__file__).resolve().parent / "demand.db"

with sqlite3.connect(DB_PATH) as connection:
    row_count = connection.execute("SELECT COUNT(*) FROM demand_data").fetchone()[0]

print(f"demand_data contains {row_count} rows.")
