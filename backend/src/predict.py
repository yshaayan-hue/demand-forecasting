import os
import pandas as pd
import joblib

from feature_engineering import create_features

# -------------------------
# Paths
# -------------------------
MODEL_PATH = os.path.join("backend", "model", "forecast_model.pkl")
DATA_PATH = os.path.join("backend", "data", "sales.csv")

# -------------------------
# Load model
# -------------------------
model = joblib.load(MODEL_PATH)
print("✅ Model loaded")

# -------------------------
# Load data
# -------------------------
df = pd.read_csv(DATA_PATH)

# -------------------------
# Create features
# -------------------------
X = create_features(df, is_train=False)

# -------------------------
# Predict
# -------------------------
predictions = model.predict(X)

df["Predicted Demand"] = predictions

# -------------------------
# Show sample output
# -------------------------
print(df[["Date", "Store ID", "Product ID", "Units Sold", "Predicted Demand"]].head())

# -------------------------
# Save predictions
# -------------------------
OUTPUT_PATH = os.path.join("backend", "data", "predictions.csv")
df.to_csv(OUTPUT_PATH, index=False)

print(f"📊 Predictions saved to {OUTPUT_PATH}")
