"""Shared helpers for the Streamlit app: paths, cached loading, scoring and explanations."""
import json
import sys
from pathlib import Path

# Make the project root importable: the saved pipeline refers to classes in `src/`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.build_dataset import CATEGORICAL_FEATURES
from src.config import MODEL_PATH, MODELS_DIR
from src.preprocessing import MODEL_FEATURES, NUMERIC_MODEL

APP_DATA = ROOT / "app" / "data"
MAX_BATCH_ROWS = 50_000

TIER_COLORS = {"Low": "#2E8B57", "Medium": "#D98E04", "High": "#C0392B"}
TIER_ACTIONS = {
    "High": ("Give this order proactive attention: contact the customer early, ask the seller to "
             "confirm dispatch before the hand-over deadline, and consider a faster carrier or "
             "resetting the promised date."),
    "Medium": ("Put the order on a watch list: check its dispatch status at the seller's "
               "hand-over deadline and escalate only if it slips."),
    "Low": "No extra action: follow the standard process.",
}
FEATURE_LABELS = {
    "estimated_delivery_days": "Promised delivery time",
    "distance_km": "Seller-customer distance",
    "n_items": "Number of items",
    "n_products": "Number of distinct products",
    "n_sellers": "Number of sellers",
    "total_price": "Order price",
    "total_freight": "Freight cost",
    "freight_ratio": "Freight / price ratio",
    "total_weight_g": "Order weight",
    "total_volume_cm3": "Order volume",
    "max_installments": "Payment installments",
    "n_payments": "Number of payments",
    "shipping_limit_days": "Seller hand-over deadline",
    "purchase_dayofweek": "Weekday of purchase",
    "purchase_hour": "Hour of purchase",
    "same_state": "Seller and customer in the same state",
    "customer_state": "Customer state",
    "seller_state": "Seller state",
    "product_category": "Product category",
    "payment_type": "Payment type",
}


# ------------------------------------------------------------------ loading
@st.cache_resource(show_spinner="Loading the trained model...")
def load_model():
    """Load the saved preprocessing + model pipeline (never retrained by the app)."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_json(name: str) -> dict:
    return json.loads((APP_DATA / name).read_text(encoding="utf-8"))


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(APP_DATA / name)


@st.cache_data
def load_metadata() -> dict:
    return json.loads((MODELS_DIR / "model_metadata.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ scoring
def predict_scores(df: pd.DataFrame) -> np.ndarray:
    """Late-delivery probability for each row (the DataFrame needs the model features)."""
    return load_model().predict_proba(df[MODEL_FEATURES])[:, 1]


def tier_of(scores, meta: dict) -> np.ndarray:
    cuts = meta["risk_tiers"]
    scores = np.asarray(scores, dtype=float)
    return np.where(scores >= cuts["high_cutoff"], "High",
                    np.where(scores >= cuts["medium_cutoff"], "Medium", "Low"))


def tier_badge(tier: str) -> str:
    return (f"<span style='background:{TIER_COLORS[tier]};color:white;padding:6px 16px;"
            f"border-radius:16px;font-weight:600;font-size:1.15rem'>{tier.upper()} RISK</span>")


def tier_history(tier: str, meta: dict) -> str:
    """What this tier really looked like in the training data and in the test period."""
    rt = meta["risk_tiers"]
    train = rt.get("train_late_rate_pct", {}).get(tier)
    test = rt.get("test_late_rate_pct", {}).get(tier)
    if train is None or test is None:
        return ""
    return (f"Orders in the **{tier}** tier were late **{train:.1f}%** of the time on the training "
            f"data and **{test:.1f}%** in the held-out test period (June to August 2018).")


def explain(row: pd.DataFrame):
    """Contribution of each original feature to this order's log-odds (Logistic Regression).

    Returns a Series sorted from most risk-lowering to most risk-raising, or None when the
    saved model has no coefficients.
    """
    model = load_model()
    prep, clf = model.named_steps["prep"], model.named_steps["model"]
    if not hasattr(clf, "coef_"):
        return None
    x = prep.transform(row[MODEL_FEATURES])[0]
    names = prep.get_feature_names_out()
    contrib = clf.coef_[0] * x
    grouped: dict[str, float] = {}
    for name, value in zip(names, contrib):
        owner = next((f for f in CATEGORICAL_FEATURES if name.startswith(f + "_")), name)
        grouped[owner] = grouped.get(owner, 0.0) + float(value)
    return pd.Series(grouped).sort_values()


# ------------------------------------------------------------------ batch input
RAW_OLIST_HINTS = {"geolocation_lat", "geolocation_zip_code_prefix", "order_status", "review_score",
                   "customer_unique_id", "seller_zip_code_prefix", "product_name_lenght",
                   "payment_sequential", "order_item_id", "review_comment_message"}


@st.cache_data
def load_examples():
    """Real orders from the held-out test period (made by scripts/make_example_orders.py)."""
    path = APP_DATA / "example_orders.csv"
    return pd.read_csv(path) if path.exists() else None


def synthetic_orders(n: int, seed: int) -> pd.DataFrame:
    """Random made-up orders inside the ranges seen in training (fallback when no example file)."""
    d = load_json("form_defaults.json")
    rng = np.random.default_rng(seed)
    out = pd.DataFrame({c: rng.uniform(d["numeric_p01"][c], d["numeric_p99"][c], n) for c in NUMERIC_MODEL})
    for c in ("estimated_delivery_days", "n_items", "n_products", "n_sellers", "max_installments",
              "n_payments", "shipping_limit_days", "purchase_dayofweek", "purchase_hour"):
        out[c] = out[c].round().astype(int)
    out["n_products"] = np.minimum(out["n_products"], out["n_items"])
    for c in CATEGORICAL_FEATURES:
        out[c] = rng.choice(d["categorical"][c], n)
    out["same_state"] = (out["customer_state"] == out["seller_state"]).astype(int)
    out["freight_ratio"] = out["total_freight"] / out["total_price"].replace(0, np.nan)
    out = out[MODEL_FEATURES]
    out.insert(0, "order_id", [f"random_{i + 1}" for i in range(n)])
    return out


def prepare_batch(df: pd.DataFrame):
    """Validate an uploaded table. Returns (clean_frame_or_None, errors, warnings, notes)."""
    errors, warns, notes = [], [], []
    work = df.copy()

    # Columns that can be worked out from other columns are added automatically
    cols = set(work.columns)
    if "same_state" not in cols and {"customer_state", "seller_state"} <= cols:
        work["same_state"] = (work["customer_state"].astype(str).str.strip().str.upper()
                              == work["seller_state"].astype(str).str.strip().str.upper()).astype(int)
        notes.append("'same_state' was calculated from customer_state and seller_state.")
    if "freight_ratio" not in cols and {"total_freight", "total_price"} <= cols:
        price = pd.to_numeric(work["total_price"], errors="coerce").replace(0, np.nan)
        work["freight_ratio"] = pd.to_numeric(work["total_freight"], errors="coerce") / price
        notes.append("'freight_ratio' was calculated as total_freight / total_price.")
    if "order_purchase_timestamp" in cols and not {"purchase_dayofweek", "purchase_hour"} <= cols:
        stamp = pd.to_datetime(work["order_purchase_timestamp"], errors="coerce")
        if "purchase_dayofweek" not in cols:
            work["purchase_dayofweek"] = stamp.dt.dayofweek
        if "purchase_hour" not in cols:
            work["purchase_hour"] = stamp.dt.hour
        notes.append("Weekday and hour were taken from 'order_purchase_timestamp'.")

    missing = [c for c in MODEL_FEATURES if c not in work.columns]
    if missing:
        if RAW_OLIST_HINTS & cols:
            errors.append(
                "This looks like one of the raw Olist tables (for example orders, items or geolocation). "
                "The app needs **one prepared table with one row per order** and the model's columns. "
                "Use the example file below, or the 'Try random example orders' option.")
        errors.append("Missing required columns: " + ", ".join(missing))
    if len(work) == 0:
        errors.append("The file has no rows.")
    if len(work) > MAX_BATCH_ROWS:
        errors.append(f"The file has more than {MAX_BATCH_ROWS:,} rows, which is the limit for this app.")
    if errors:
        return None, errors, warns, notes

    clean = work
    for col in NUMERIC_MODEL:
        before = clean[col].isna().sum()
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
        bad = int(clean[col].isna().sum() - before)
        if bad:
            warns.append(f"{bad} non-numeric value(s) in '{col}' were treated as missing.")
    for col in CATEGORICAL_FEATURES:
        clean[col] = clean[col].map(lambda v: np.nan if pd.isna(v) else str(v).strip()).astype(object)
    for col in ("customer_state", "seller_state"):
        clean[col] = clean[col].map(lambda v: v.upper() if isinstance(v, str) else v).astype(object)
    n_missing = int(clean[NUMERIC_MODEL].isna().sum().sum() + clean[CATEGORICAL_FEATURES].isna().sum().sum())
    if n_missing:
        warns.append(f"{n_missing} missing value(s) were filled the same way as in training "
                     "(median for numbers, 'missing' for text).")
    return clean, errors, warns, notes


def batch_template(n: int = 3) -> pd.DataFrame:
    """A tiny example table built from training medians (used only when there is no example file)."""
    return synthetic_orders(n, seed=0)
