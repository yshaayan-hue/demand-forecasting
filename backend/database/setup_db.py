from pathlib import Path
import sqlite3
import csv

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "demand.db"
SCHEMA_PATH = BASE_DIR / "schema.sql"
CSV_PATH = BASE_DIR.parent / "data" / "sales.csv"

# Create database and table
connection = sqlite3.connect(DB_PATH)

with open(SCHEMA_PATH, "r", encoding="utf-8") as file:
    schema = file.read()

connection.executescript(schema)

# Read CSV
with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)

    data = []

    for row in reader:
        data.append((
            row["Date"],
            row["Store ID"],
            row["Product ID"],
            row["Category"],
            row["Region"],
            int(row["Inventory Level"]),
            int(row["Units Sold"]),
            int(row["Units Ordered"]),
            float(row["Demand Forecast"]),
            float(row["Price"]),
            int(row["Discount"]),
            row["Weather Condition"],
            int(row["Holiday/Promotion"]),
            float(row["Competitor Pricing"]),
            row["Seasonality"]
        ))

# Insert data
connection.executemany("""
    INSERT INTO demand_data (
        date,
        store_id,
        product_id,
        category,
        region,
        inventory_level,
        units_sold,
        units_ordered,
        predicted_demand,
        price,
        discount,
        weather,
        holiday_promotion,
        competitor_pricing,
        seasonality
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", data)

connection.commit()

print("Database setup successful!")
print("Rows imported:", len(data))
print("Database:", DB_PATH)

connection.close()