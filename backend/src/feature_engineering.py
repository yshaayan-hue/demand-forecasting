import pandas as pd

def create_features(df, is_train=True):
    # -------------------------
    # Fix column names
    # -------------------------
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

    # -------------------------
    # Time-based features
    # -------------------------
    df["day"] = df["Date"].dt.day
    df["month"] = df["Date"].dt.month
    df["year"] = df["Date"].dt.year
    df["dayofweek"] = df["Date"].dt.dayofweek

    # -------------------------
    # Encode categorical variables
    # -------------------------
    df["Store ID"] = df["Store ID"].astype("category").cat.codes
    df["Product ID"] = df["Product ID"].astype("category").cat.codes
    df["Category"] = df["Category"].astype("category").cat.codes
    df["Region"] = df["Region"].astype("category").cat.codes
    df["Weather Condition"] = df["Weather Condition"].astype("category").cat.codes
    df["Holiday/Promotion"] = df["Holiday/Promotion"].astype("category").cat.codes
    df["Seasonality"] = df["Seasonality"].astype("category").cat.codes

    # -------------------------
    # Feature set
    # -------------------------
    feature_cols = [
        "Store ID",
        "Product ID",
        "Category",
        "Region",
        "Inventory Level",
        "Units Ordered",
        "Price",
        "Discount",
        "Competitor Pricing",
        "Weather Condition",
        "Holiday/Promotion",
        "Seasonality",
        "day",
        "month",
        "year",
        "dayofweek"
    ]

    X = df[feature_cols]

    # -------------------------
    # Target
    # -------------------------
    if is_train:
        y = df["Units Sold"]
        return X, y
    else:
        return X
