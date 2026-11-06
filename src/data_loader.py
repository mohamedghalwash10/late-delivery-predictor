"""Load the nine Olist CSV files into pandas DataFrames."""
import pandas as pd

from src.config import DATA_RAW, OLIST_FILES

# Columns that hold dates, parsed as datetime on load
DATE_COLUMNS = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "items": ["shipping_limit_date"],
    "reviews": ["review_creation_date", "review_answer_timestamp"],
}


def load_table(name: str) -> pd.DataFrame:
    """Load one table by its key in OLIST_FILES (e.g. 'orders')."""
    path = DATA_RAW / OLIST_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, parse_dates=DATE_COLUMNS.get(name, []))


def load_all() -> dict[str, pd.DataFrame]:
    """Load every table into a dict keyed by table name."""
    return {name: load_table(name) for name in OLIST_FILES}