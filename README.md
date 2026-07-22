# Demand Forecasting

A machine learning project that forecasts retail product demand (units sold) using historical sales data — store, product, pricing, weather, and seasonality features feed a Random Forest regression model.

## What it does

Given a sales dataset with per-store, per-product records (inventory levels, pricing, promotions, weather, seasonality, etc.), this project:

1. Engineers time-based and categorical features from the raw data.
2. Trains a `RandomForestRegressor` to predict **Units Sold / Demand**.
3. Evaluates the model with MAE and RMSE.
4. Generates demand predictions on new data and saves them to CSV.

Exploratory data analysis and the initial modeling walkthrough live in the Jupyter notebook; the same logic is organized into reusable scripts under `backend/src`.

## Tech stack

- Python
- pandas, NumPy
- scikit-learn (RandomForestRegressor)
- joblib (model persistence)
- Flask / flask-cors (listed in requirements, for a future API layer)
- Jupyter Notebook (EDA)

## Project structure

```
demand-forecasting/
└── backend/
    ├── requirements.txt
    ├── notebook/
    │   └── EDA_and_Modeling.ipynb   # Exploratory data analysis + initial modeling
    └── src/
        ├── feature_engineering.py    # create_features(): time + categorical features
        ├── train.py                  # Loads data, trains RandomForestRegressor, saves model
        ├── predict.py                 # Loads saved model, scores new data, saves predictions
        └── evaluate.py                # Computes MAE/RMSE on a held-out split
```

Expected data/model locations (created automatically or expected to exist):
```
backend/data/sales.csv              # input training data
backend/data/predictions.csv        # output predictions (created by predict.py)
backend/model/forecast_model.pkl    # saved trained model
```

## Features used

- `Store ID`, `Product ID`, `Category`, `Region` (encoded)
- `Inventory Level`, `Units Ordered`, `Price`, `Discount`, `Competitor Pricing`
- `Weather Condition`, `Holiday/Promotion`, `Seasonality` (encoded)
- `day`, `month`, `year`, `dayofweek` (derived from `Date`)

Target: `Units Sold`

## Getting started

### Prerequisites
- Python 3.9+

### Installation

```bash
git clone https://github.com/yshaayan-hue/demand-forecasting.git
cd demand-forecasting/backend
pip install -r requirements.txt
```

### Add your data
Place a `sales.csv` file (with the columns listed above) at `backend/data/sales.csv`.

### Train the model

```bash
cd backend/src
python train.py
```

This saves a trained model to `backend/model/forecast_model.pkl`.

### Generate predictions

```bash
python predict.py
```

This scores `backend/data/sales.csv` and writes results to `backend/data/predictions.csv`.

### Evaluate

```bash
python evaluate.py
```

Prints MAE and RMSE on a held-out test split. Note: `evaluate.py` currently reads from `../../dataset/sales.csv` and `../model/forecast_model.pkl` — adjust these paths to match where your data and trained model actually live if you hit a file-not-found error.

## License

MIT — see [LICENSE](LICENSE).
