"""Batch prediction: random example orders or an uploaded CSV."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils import (MAX_BATCH_ROWS, TIER_COLORS, load_examples, load_metadata, predict_scores,
                   prepare_batch, synthetic_orders, tier_of)
from src.preprocessing import MODEL_FEATURES

st.set_page_config(page_title="Batch Prediction", page_icon="📦", layout="wide")
st.title("Batch prediction")
st.write("Score **many orders at once**: every order gets a late probability and a Low, Medium or High risk tier, "
         "sorted with the riskiest first. Try random example orders, or upload your own file.")

meta = load_metadata()
examples = load_examples()


def show_results(raw: pd.DataFrame, clean: pd.DataFrame, key: str) -> None:
    scores = predict_scores(clean)
    result = raw.copy().reset_index(drop=True)
    result["late_probability"] = scores.round(4)
    result["risk_tier"] = tier_of(scores, meta)
    lead = [c for c in ("order_id", "late_probability", "risk_tier", "actual_is_late") if c in result.columns]
    result = result[lead + [c for c in result.columns if c not in lead]]
    result = result.sort_values("late_probability", ascending=False).reset_index(drop=True)

    st.subheader(f"Results for {len(result):,} orders")
    counts = result["risk_tier"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Orders scored", f"{len(result):,}")
    for col, tier in zip((m2, m3, m4), ("High", "Medium", "Low")):
        col.metric(f"{tier} risk", f"{counts[tier]:,}", f"{counts[tier] / len(result) * 100:.1f}% of orders",
                   delta_color="off")
    fig = px.bar(counts.reset_index(), x="risk_tier", y="count", color="risk_tier", color_discrete_map=TIER_COLORS,
                 labels={"risk_tier": "Risk tier", "count": "Orders"})
    fig.update_layout(height=260, showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(result.head(2000), use_container_width=True, hide_index=True)
    if len(result) > 2000:
        st.caption("Showing the 2,000 riskiest orders. The download contains all rows.")
    st.download_button("Download all predictions as CSV", result.to_csv(index=False).encode("utf-8"),
                       file_name="late_delivery_predictions.csv", mime="text/csv", type="primary",
                       key=f"download_{key}")

    if "actual_is_late" in result.columns:
        truth = pd.to_numeric(result["actual_is_late"], errors="coerce")
        if truth.notna().all() and set(truth.unique()) <= {0, 1}:
            st.subheader("How did the model do on these orders?")
            check = (result.assign(late=truth).groupby("risk_tier")
                     .agg(orders=("late", "size"), really_late=("late", "sum")).reindex(["High", "Medium", "Low"])
                     .fillna(0).astype(int))
            check["late_rate_pct"] = (check["really_late"] / check["orders"].replace(0, np.nan) * 100).round(1)
            st.dataframe(check.reset_index(), hide_index=True, use_container_width=True)
            st.caption(f"{truth.mean() * 100:.1f}% of these orders were really late. A small sample is noisy: "
                       "the tier differences become reliable only with thousands of orders.")
    st.caption("`late_probability` is relative risk: valid for the training period and about 2.5 times too high "
               "in the 2018 test period.")


mode = st.radio("How do you want to provide orders?", ["Try random example orders", "Upload my own CSV"],
                horizontal=True)

# =============================================================== random examples
if mode == "Try random example orders":
    if examples is not None:
        st.write(f"A pool of **{len(examples):,} real orders** from the held-out test period (June to August 2018) "
                 "is used. The model never saw them during training, and each order's real outcome is included "
                 "(`actual_is_late`) so you can check the predictions.")
    else:
        st.info("The real example file was not found, so random **made-up** orders are generated instead.")
    c1, c2 = st.columns([3, 1])
    n = c1.slider("Number of random orders", 10, 500, 100, step=10)
    if "draw" not in st.session_state:
        st.session_state["draw"] = 0
    if c2.button("Draw a new random sample", use_container_width=True):
        st.session_state["draw"] += 1
    seed = st.session_state["draw"]

    if examples is not None:
        raw = examples.sample(n=min(n, len(examples)), random_state=seed)
    else:
        raw = synthetic_orders(n, seed)
    clean, errors, warns, notes = prepare_batch(raw)
    for message in errors:
        st.error(message)
    if clean is not None:
        st.download_button("Download this sample as a CSV (to try the upload option)",
                           raw.to_csv(index=False).encode("utf-8"), file_name="random_example_orders.csv",
                           mime="text/csv", key="download_sample")
        show_results(raw, clean, "random")

# =============================================================== upload
else:
    with st.expander("What file do I need?", expanded=True):
        st.markdown("One CSV with **one row per order** and these **20 columns** (extra columns such as `order_id` are kept):")
        st.code(", ".join(MODEL_FEATURES), language="text")
        st.markdown(
            "- Prices, freight, weight and volume are totals per order.\n"
            "- `same_state`, `freight_ratio`, `purchase_dayofweek` and `purchase_hour` are added automatically if "
            "you give `customer_state` and `seller_state`, `total_price` and `total_freight`, or `order_purchase_timestamp`.\n"
            f"- Maximum {MAX_BATCH_ROWS:,} rows. The raw Olist files (orders, items, geolocation...) are **not** "
            "valid: they are separate tables, not one row per order.")
        if examples is not None:
            st.download_button("Download an example file (500 real orders)",
                               examples.head(500).to_csv(index=False).encode("utf-8"),
                               file_name="example_orders.csv", mime="text/csv", key="download_examples")
        else:
            st.download_button("Download a small example file", synthetic_orders(20, 0).to_csv(index=False).encode("utf-8"),
                               file_name="example_orders.csv", mime="text/csv", key="download_examples")

    upload = st.file_uploader("Upload a CSV file", type=["csv"])
    if upload is None:
        st.info("Waiting for a file. Download the example above if you are not sure about the format.")
        st.stop()
    try:
        raw = pd.read_csv(upload, nrows=MAX_BATCH_ROWS + 1)   # never load a huge file completely
    except Exception as exc:
        st.error(f"The file could not be read as a CSV: {exc}")
        st.stop()

    clean, errors, warns, notes = prepare_batch(raw)
    for message in errors:
        st.error(message)
    if clean is None:
        st.stop()
    for message in notes:
        st.info(message)
    for message in warns:
        st.warning(message)
    show_results(raw, clean, "upload")
