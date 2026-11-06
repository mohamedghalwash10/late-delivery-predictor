"""Build the order-level table: one row per labelled (delivered) order.

Columns are split into groups so later stages cannot leak the future:
    TARGET            what we predict (1 = late, 0 = on time)
    FEATURE_COLUMNS   known at purchase time -> allowed as model inputs
    ANALYSIS_ONLY     known only after purchase -> EDA / error analysis, NEVER features
"""
import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED
from src.data_loader import load_all

BASE_PATH = DATA_PROCESSED / "orders_base.csv"

TARGET = "is_late"
ID_COLUMNS = ["order_id", "customer_id", "seller_id"]

NUMERIC_FEATURES = [
    "estimated_delivery_days",  # promised window shown to the customer at purchase
    "distance_km",              # customer zip -> first seller zip
    "n_items", "n_products", "n_sellers",
    "total_price", "total_freight", "freight_ratio",
    "total_weight_g", "total_volume_cm3",
    "max_installments", "n_payments",
    "shipping_limit_days",      # seller's hand-over deadline, set at purchase
    "purchase_month", "purchase_dayofweek", "purchase_hour",
    "same_state",
]
CATEGORICAL_FEATURES = [
    "customer_state", "seller_state", "product_category", "payment_type",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

DATE_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]
ANALYSIS_ONLY = DATE_COLUMNS + [
    "actual_delivery_days",   # uses the delivery date -> leakage if used as a feature
    "days_vs_estimate",       # same
    "review_score",           # written after delivery
    "purchase_year_month",
    "customer_city",
    "customer_zip_code_prefix",
    "seller_zip_code_prefix",
]


def haversine_km(lat1, lng1, lat2, lng2):
    """Great-circle distance in km between two lat/lng points."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lng2 - lng1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def build_order_table() -> pd.DataFrame:
    t = load_all()

    # ---- 1. Labelled orders and target ---------------------------------
    df = t["orders"]
    df = df[df["order_status"] == "delivered"]
    df = df.dropna(subset=["order_delivered_customer_date",
                           "order_estimated_delivery_date"]).copy()

    purchase_day = df["order_purchase_timestamp"].dt.normalize()
    delivered_day = df["order_delivered_customer_date"].dt.normalize()
    estimated_day = df["order_estimated_delivery_date"].dt.normalize()

    df[TARGET] = (delivered_day > estimated_day).astype(int)
    df["days_vs_estimate"] = (delivered_day - estimated_day).dt.days
    df["actual_delivery_days"] = (delivered_day - purchase_day).dt.days
    df["estimated_delivery_days"] = (estimated_day - purchase_day).dt.days
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
    df["purchase_year_month"] = df["order_purchase_timestamp"].dt.strftime("%Y-%m")
    n_labelled = len(df)
    print(f"Labelled delivered orders: {n_labelled:,}")

    # ---- 2. Customer ----------------------------------------------------
    cust = t["customers"][["customer_id", "customer_state", "customer_city",
                           "customer_zip_code_prefix"]]
    df = df.merge(cust, on="customer_id", how="left")

    # ---- 3. Items -> products -> sellers, aggregated to one row per order
    products = t["products"].merge(
        t["category_translation"], on="product_category_name", how="left")
    products["product_category"] = (
        products["product_category_name_english"]
        .fillna(products["product_category_name"])
        .fillna("unknown"))
    products["volume_cm3"] = (products["product_length_cm"]
                              * products["product_height_cm"]
                              * products["product_width_cm"])

    items = (
        t["items"]
        .merge(products[["product_id", "product_category",
                         "product_weight_g", "volume_cm3"]],
               on="product_id", how="left")
        .merge(t["sellers"], on="seller_id", how="left")
        .sort_values(["order_id", "order_item_id"])
    )
    g = items.groupby("order_id")
    item_agg = g.agg(
        n_items=("order_item_id", "size"),
        n_products=("product_id", "nunique"),
        n_sellers=("seller_id", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        latest_shipping_limit=("shipping_limit_date", "max"),
        seller_id=("seller_id", "first"),            # seller of the first item
        seller_state=("seller_state", "first"),
        seller_zip_code_prefix=("seller_zip_code_prefix", "first"),
        product_category=("product_category", "first"),
    )
    item_agg["total_weight_g"] = g["product_weight_g"].sum(min_count=1)
    item_agg["total_volume_cm3"] = g["volume_cm3"].sum(min_count=1)
    item_agg = item_agg.reset_index()
    df = df.merge(item_agg, on="order_id", how="left")

    # ---- 4. Payments (chosen at checkout, so known at purchase time) ----
    pay = t["payments"].sort_values(["order_id", "payment_sequential"])
    pay_agg = pay.groupby("order_id").agg(
        payment_type=("payment_type", "first"),
        max_installments=("payment_installments", "max"),
        n_payments=("payment_sequential", "size"),
    ).reset_index()
    df = df.merge(pay_agg, on="order_id", how="left")

    # ---- 5. Reviews (analysis only: written after delivery) -------------
    review = (t["reviews"].groupby("order_id")["review_score"].mean()
              .rename("review_score").reset_index())
    df = df.merge(review, on="order_id", how="left")

    # ---- 6. Geolocation: ONE coordinate per zip prefix (median) ---------
    geo = (t["geolocation"].groupby("geolocation_zip_code_prefix")
           .agg(lat=("geolocation_lat", "median"),
                lng=("geolocation_lng", "median"))
           .reset_index())
    df = df.merge(
        geo.rename(columns={"geolocation_zip_code_prefix": "customer_zip_code_prefix",
                            "lat": "customer_lat", "lng": "customer_lng"}),
        on="customer_zip_code_prefix", how="left")
    df = df.merge(
        geo.rename(columns={"geolocation_zip_code_prefix": "seller_zip_code_prefix",
                            "lat": "seller_lat", "lng": "seller_lng"}),
        on="seller_zip_code_prefix", how="left")

    assert len(df) == n_labelled, "A join duplicated order rows!"

    # ---- 7. Orders without item rows cannot be described -> drop --------
    no_items = int(df["n_items"].isna().sum())
    print(f"Labelled orders with no item rows (dropped): {no_items:,}")
    df = df[df["n_items"].notna()].copy()
    for col in ["n_items", "n_products", "n_sellers"]:
        df[col] = df[col].astype(int)

    # ---- 8. Derived purchase-time features ------------------------------
    df["distance_km"] = haversine_km(df["customer_lat"], df["customer_lng"],
                                     df["seller_lat"], df["seller_lng"])
    df["same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)
    df["freight_ratio"] = df["total_freight"] / df["total_price"].replace(0, np.nan)
    df["shipping_limit_days"] = (
        df["latest_shipping_limit"].dt.normalize()
        - df["order_purchase_timestamp"].dt.normalize()).dt.days

    out = df[ID_COLUMNS + [TARGET] + FEATURE_COLUMNS + ANALYSIS_ONLY].copy()
    assert out["order_id"].is_unique, "order_id must be unique"
    return out


def load_base_table() -> pd.DataFrame:
    """Load the saved order-level table (run stage3_build_dataset.py first)."""
    if not BASE_PATH.exists():
        raise FileNotFoundError("Run stage3_build_dataset.py first.")
    return pd.read_csv(BASE_PATH, parse_dates=DATE_COLUMNS)