from flask import Flask, render_template
from pathlib import Path
import sqlite3

app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static"
)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "database" / "demand.db"


def get_db_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


@app.route("/")
def dashboard():

    connection = get_db_connection()

    actual_demand = connection.execute("""
        SELECT SUM(units_sold)
        FROM demand_data
    """).fetchone()[0]

    forecast_demand = connection.execute("""
        SELECT SUM(predicted_demand)
        FROM demand_data
    """).fetchone()[0]

    forecast_difference = connection.execute("""
        SELECT ROUND(
            SUM(predicted_demand) - SUM(units_sold),
            2
        )
        FROM demand_data
    """).fetchone()[0]

    connection.close()

    return render_template(
        "dashboard.html",
        actual_demand=actual_demand,
        forecast_demand=forecast_demand,
        forecast_difference=forecast_difference
    )


if __name__ == "__main__":
    app.run(debug=True)