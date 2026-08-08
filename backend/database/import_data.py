import sqlite3
import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "demand.db"
CSV_PATH = BASE_DIR.parent / "data" / "sales.csv"

# Load CSV
df = pd.read_csv(CSV_PATH)

# Rename CSV columns to database columns
df = df.rename(columns={
    "Date": "date",
    "Store ID": "store_id",
    "Product ID": "product_id",
    "Category": "category",
    "Region": "region",
    "Inventory Level": "inventory_level",
    "Units Sold": "units_sold",
    "Units Ordered": "units_ordered",
    "Demand Forecast": "predicted_demand",
    "Price": "price",
    "Discount": "discount",
    "Weather Condition": "weather",
    "Holiday/Promotion": "holiday_promotion",
    "Competitor Pricing": "competitor_pricing",
    "Seasonality": "seasonality"
})

# Connect to SQLite
connection = sqlite3.connect(DB_PATH)

# Insert data
df.to_sql(
    "demand_data",
    connection,
    if_exists="append",
    index=False
)

connection.close()

print("Data imported successfully!")
print(f"Rows imported: {len(df)}")