DROP TABLE IF EXISTS demand_data;

CREATE TABLE demand_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    store_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    category TEXT NOT NULL,
    region TEXT NOT NULL,
    inventory_level INTEGER NOT NULL,
    units_sold INTEGER NOT NULL,
    units_ordered INTEGER NOT NULL,
    predicted_demand REAL NOT NULL,
    price REAL NOT NULL,
    discount INTEGER NOT NULL,
    weather TEXT NOT NULL,
    holiday_promotion INTEGER NOT NULL,
    competitor_pricing REAL NOT NULL,
    seasonality TEXT NOT NULL
);