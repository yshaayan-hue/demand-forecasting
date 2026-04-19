import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from feature_engineering import create_features

# -------------------------
# Paths (ABSOLUTE-SAFE)
# -------------------------
DATA_PATH = os.path.join("backend", "data", "sales.csv")
MODEL_DIR = os.path.join("backend", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "forecast_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------
# Load data
# -------------------------
print("📥 Loading dataset...")
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# -------------------------
# Feature engineering
# -------------------------
print("⚙️ Creating features...")
X, y = create_features(df, is_train=True)

# HARD STOP if data is bad
assert len(X) > 0, "❌ X is empty"
assert len(y) > 0, "❌ y is empty"

print("X shape:", X.shape)
print("y shape:", y.shape)

# -------------------------
# Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Train model
# -------------------------
print("🧠 Training model...")
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)
print("✅ Model training complete")

# -------------------------
# Save model (ONLY after fit)
# -------------------------
joblib.dump(model, MODEL_PATH)

# VERIFY file is not empty
assert os.path.getsize(MODEL_PATH) > 0, "❌ Model file is EMPTY"

print(f"💾 Model saved successfully at: {MODEL_PATH}")

# -----------------------------
# Feature engineering
# -----------------------------
X, y = create_features(df)

# -----------------------------
# Train-test split (time-aware)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

# -----------------------------
# Train model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -----------------------------
# Save model
# -----------------------------
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("✅ Training complete")
print("✅ Model saved at:", MODEL_PATH)
