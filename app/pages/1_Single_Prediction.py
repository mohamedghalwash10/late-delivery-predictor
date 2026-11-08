"""Single order prediction with explanation and suggested action."""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils import (FEATURE_LABELS, TIER_ACTIONS, TIER_COLORS, explain, load_json, load_metadata,
                   predict_scores, tier_badge, tier_history, tier_of)
from src.preprocessing import MODEL_FEATURES

st.set_page_config(page_title="Single Prediction", page_icon="📦", layout="wide")
st.title("Single order prediction")
st.write("Describe an order as it looks **at purchase time**. Defaults are typical training values.")

defaults = load_json("form_defaults.json")
cats, med = defaults["categorical"], defaults["numeric_median"]
lo, hi = defaults["numeric_p01"], defaults["numeric_p99"]
meta = load_metadata()
metrics = load_json("metrics.json")


def clamp(value, low, high):
    return int(min(max(round(value), low), high))


def default_index(options, preferred):
    return options.index(preferred) if preferred in options else 0


side = round(float(med["total_volume_cm3"]) ** (1 / 3))
now = datetime.now()

with st.form("order_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Order")
        purchase_date = st.date_input("Purchase date", value=now.date())
        purchase_time = st.time_input("Purchase time", value=now.time().replace(second=0, microsecond=0))
        promised = st.number_input("Promised delivery time (days)", 1, 160,
                                   clamp(med["estimated_delivery_days"], 1, 160),
                                   help="Days between purchase and the estimated delivery date shown to the customer.")
        ship_limit = st.number_input("Seller hand-over deadline (days after purchase)", 0, 60,
                                     clamp(med["shipping_limit_days"], 0, 60))
        payment_type = st.selectbox("Payment type", cats["payment_type"],
                                    index=default_index(cats["payment_type"], "credit_card"))
        installments = st.number_input("Installments", 1, 24, clamp(med["max_installments"], 1, 24))
        n_payments = st.number_input("Number of payments", 1, 10, clamp(med["n_payments"], 1, 10))
    with c2:
        st.subheader("Customer and seller")
        customer_state = st.selectbox("Customer state", cats["customer_state"],
                                      index=default_index(cats["customer_state"], "SP"))
        seller_state = st.selectbox("Seller state", cats["seller_state"],
                                    index=default_index(cats["seller_state"], "SP"))
        distance = st.number_input("Distance seller to customer (km)", 0.0, 9000.0,
                                   float(round(med["distance_km"], 1)), step=10.0,
                                   help="Straight-line distance between the two zip codes.")
        n_sellers = st.number_input("Number of sellers in the order", 1, 10, clamp(med["n_sellers"], 1, 10))
    with c3:
        st.subheader("Items")
        category = st.selectbox("Product category", cats["product_category"],
                                index=default_index(cats["product_category"], "bed_bath_table"))
        n_items = st.number_input("Number of items", 1, 30, clamp(med["n_items"], 1, 30))
        n_products = st.number_input("Number of distinct products", 1, 30, clamp(med["n_products"], 1, 30))
        price = st.number_input("Total price (BRL)", 0.0, 20000.0, float(round(med["total_price"], 2)), step=5.0)
        freight = st.number_input("Total freight (BRL)", 0.0, 2000.0, float(round(med["total_freight"], 2)), step=1.0)
        weight = st.number_input("Total weight (g)", 0.0, 200000.0, float(round(med["total_weight_g"])), step=50.0)
        d1, d2, d3 = st.columns(3)
        length = d1.number_input("Length (cm)", 1.0, 200.0, float(side))
        width = d2.number_input("Width (cm)", 1.0, 200.0, float(side))
        height = d3.number_input("Height (cm)", 1.0, 200.0, float(side))
    submitted = st.form_submit_button("Predict delivery risk", type="primary")

if submitted:
    if n_products > n_items:
        st.error("The number of distinct products cannot be larger than the number of items.")
        st.stop()
    row = {
        "estimated_delivery_days": promised, "distance_km": distance, "n_items": n_items,
        "n_products": n_products, "n_sellers": n_sellers, "total_price": price,
        "total_freight": freight, "freight_ratio": freight / price if price > 0 else np.nan,
        "total_weight_g": weight, "total_volume_cm3": length * width * height,
        "max_installments": installments, "n_payments": n_payments,
        "shipping_limit_days": ship_limit, "purchase_dayofweek": purchase_date.weekday(),
        "purchase_hour": purchase_time.hour, "same_state": int(customer_state == seller_state),
        "customer_state": customer_state, "seller_state": seller_state,
        "product_category": category, "payment_type": payment_type,
    }
    frame = pd.DataFrame([row])[MODEL_FEATURES]
    score = float(predict_scores(frame)[0])
    tier = str(tier_of([score], meta)[0])

    st.divider()
    left, right = st.columns([1, 2])
    with left:
        st.metric("Estimated late probability", f"{score * 100:.1f}%")
        st.markdown(tier_badge(tier), unsafe_allow_html=True)
        st.write("")
        st.markdown(tier_history(tier, meta))
        check = metrics.get("probability_check")
        if check:
            st.caption(
                f"Read the probability as relative risk: in the June to August 2018 test the model "
                f"averaged {check['mean_predicted_test_pct']}% while {check['actual_test_pct']}% of orders were "
                "really late.")
    with right:
        st.subheader("Why this score?")
        contrib = explain(frame)
        if contrib is None:
            st.info("This model type has no per-feature explanation.")
        else:
            top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)
            plot = pd.DataFrame({"factor": [FEATURE_LABELS.get(k, k) for k in top.index],
                                 "effect": top.values})
            plot["direction"] = np.where(plot["effect"] > 0, "raises risk", "lowers risk")
            fig = px.bar(plot.iloc[::-1], x="effect", y="factor", color="direction", orientation="h",
                         color_discrete_map={"raises risk": TIER_COLORS["High"], "lowers risk": TIER_COLORS["Low"]},
                         labels={"effect": "Effect on the risk score (right raises it, left lowers it)", "factor": ""})
            fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
            up = [FEATURE_LABELS.get(k, k) for k in top[top > 0].index[:3]]
            down = [FEATURE_LABELS.get(k, k) for k in top[top < 0].index[:3]]
            if up:
                st.write("**Raising the risk:** " + ", ".join(up) + ".")
            if down:
                st.write("**Lowering the risk:** " + ", ".join(down) + ".")
            st.caption("These are patterns the model learned from past orders, not proven causes.")

    st.subheader("Suggested business action")
    if tier == "Low":
        st.success(TIER_ACTIONS[tier])
    elif tier == "Medium":
        st.warning(TIER_ACTIONS[tier])
    else:
        st.error(TIER_ACTIONS[tier])

    outside = [FEATURE_LABELS[c] for c in ("estimated_delivery_days", "distance_km", "total_price", "total_freight",
                                          "total_weight_g", "total_volume_cm3")
               if row[c] is not None and not (lo[c] <= row[c] <= hi[c])]
    if outside:
        st.caption("Outside the range common in the training data (the model caps such values): "
                   + ", ".join(outside) + ".")
