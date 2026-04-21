"""
db_setup.py
-----------
One-time setup script that loads the Olist Brazilian E-Commerce CSV files
into a local SQLite database (ecommerce.db) and exposes a small reusable
`query()` helper for use by the rest of the analysis project.

Run directly to (re)build the database:
    python src/db_setup.py
"""

from pathlib import Path
import sqlite3
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Project root is the parent of the `src/` folder that holds this file.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = PROJECT_ROOT / "ecommerce.db"


# ---------------------------------------------------------------------------
# Table definitions
# ---------------------------------------------------------------------------
# Maps the target SQLite table name -> source CSV filename.
TABLES = {
    "orders":              "olist_orders_dataset.csv",
    "customers":           "olist_customers_dataset.csv",
    "order_items":         "olist_order_items_dataset.csv",
    "order_payments":      "olist_order_payments_dataset.csv",
    "order_reviews":       "olist_order_reviews_dataset.csv",
    "products":            "olist_products_dataset.csv",
    "sellers":             "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
    "geolocation":         "olist_geolocation_dataset.csv",  # used for Q1 geographic residual analysis
}

# Date columns that should be parsed as datetimes on load.
# Any column name listed here is converted with pd.to_datetime if it
# exists in the source CSV.
DATE_COLUMNS = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "order_reviews": [
        "review_creation_date",
        "review_answer_timestamp",
    ],
}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def _get_connection() -> sqlite3.Connection:
    """Return a SQLite connection to the project database."""
    return sqlite3.connect(DB_PATH)


def query(sql: str, params: tuple | list | dict | None = None) -> pd.DataFrame:
    """
    Run any SQL statement against ecommerce.db and return a DataFrame.

    Example:
        df = query("SELECT * FROM orders LIMIT 5")
    """
    with _get_connection() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def _load_csv(table: str, filename: str) -> pd.DataFrame:
    """Read a single CSV, parsing any known date columns for this table."""
    csv_path = DATA_DIR / filename
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV for '{table}': {csv_path}")

    # Only ask pandas to parse date columns that actually exist in the CSV,
    # so a schema change upstream won't break the load.
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    parse_dates = [c for c in DATE_COLUMNS.get(table, []) if c in header]

    return pd.read_csv(csv_path, parse_dates=parse_dates)


def build_database() -> None:
    """Load every configured CSV into ecommerce.db, replacing existing tables."""
    print(f"Building database at: {DB_PATH}")
    print(f"Reading CSVs from:    {DATA_DIR}\n")

    with _get_connection() as conn:
        for table, filename in TABLES.items():
            df = _load_csv(table, filename)
            df.to_sql(table, conn, if_exists="replace", index=False)
            print(f"  loaded {table:<22} {len(df):>8,} rows")

    print("\nDone.")


def print_row_counts() -> None:
    """Sanity check: print SELECT COUNT(*) for every loaded table."""
    print("\nRow counts (from SQLite):")
    for table in TABLES:
        n = query(f"SELECT COUNT(*) AS n FROM {table}").iloc[0, 0]
        print(f"  {table:<22} {n:>8,}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    build_database()
    print_row_counts()
