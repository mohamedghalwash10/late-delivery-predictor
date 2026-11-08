"""Interactive analytics dashboard."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import FEATURE_LABELS, TIER_COLORS, load_csv, load_json

st.set_page_config(page_title="Analytics Dashboard", page_icon="📦", layout="wide")
st.title("Analytics dashboard")

month = load_csv("stats_by_month.csv")
state = load_csv("stats_by_customer_state.csv")
category = load_csv("stats_by_category.csv")
seller = load_csv("stats_by_seller.csv")
test = load_csv("dashboard_test_predictions.csv")
importance = load_csv("feature_importance.csv")
metrics = load_json("metrics.json")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Delivery statistics", "Risk distribution", "States and categories",
                                         "Sellers", "Feature importance"])

# ---------------------------------------------------------------- 1. delivery statistics
with tab1:
    total = int(month["orders"].sum())
    late_rate = (month["orders"] * month["late_rate_pct"]).sum() / total
    wavg = lambda col: (state["orders"] * state[col]).sum() / state["orders"].sum()
    a, b, c, d = st.columns(4)
    a.metric("Delivered orders analysed", f"{total:,}")
    b.metric("Late delivery rate", f"{late_rate:.2f}%")
    c.metric("Average promised delivery time", f"{wavg('avg_promised_days'):.1f} days")
    d.metric("Average actual delivery time", f"{wavg('avg_actual_days'):.1f} days")
    st.caption("Whole dataset, 2016 to 2018 purchases. Averages are weighted by orders.")

    min_orders_month = st.slider("Hide months with fewer orders than", 0, 2000, 200, step=50)
    shown = month[month["orders"] >= min_orders_month]
    fig = go.Figure()
    fig.add_bar(x=shown["purchase_year_month"], y=shown["orders"], name="Orders", yaxis="y2",
                marker_color="#C9D6EA")
    fig.add_scatter(x=shown["purchase_year_month"], y=shown["late_rate_pct"], name="Late rate (%)",
                    mode="lines+markers", marker_color="#C0392B")
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=30, b=0), title="Late rate and order volume by purchase month",
                      yaxis=dict(title="Late rate (%)"),
                      yaxis2=dict(title="Orders", overlaying="y", side="right", showgrid=False),
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)
    st.write("Delays arrive in **waves** (for example November 2017 and February to March 2018), not in a stable seasonal pattern.")

# ---------------------------------------------------------------- 2. risk distribution
with tab2:
    st.write(f"Model scores on the **held-out test set** ({len(test):,} orders from June to August 2018), "
             "never used for training.")
    f1, f2 = st.columns(2)
    states = ["All"] + sorted(test["customer_state"].dropna().unique())
    pick_state = f1.selectbox("Customer state", states)
    pick_tiers = f2.multiselect("Risk tiers", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    view = test if pick_state == "All" else test[test["customer_state"] == pick_state]
    view = view[view["risk_tier"].isin(pick_tiers)]
    if view.empty:
        st.warning("No orders match these filters.")
    else:
        tiers = (view.groupby("risk_tier").agg(orders=("is_late", "size"), late_orders=("is_late", "sum"),
                                                late_rate_pct=("is_late", "mean")).reindex(["High", "Medium", "Low"])
                 .dropna().reset_index())
        tiers["late_rate_pct"] = (tiers["late_rate_pct"] * 100).round(2)
        g1, g2 = st.columns(2)
        fig = px.bar(tiers, x="risk_tier", y="orders", color="risk_tier", color_discrete_map=TIER_COLORS,
                     title="Orders per risk tier", labels={"risk_tier": "Risk tier"})
        fig.update_layout(showlegend=False, height=340)
        g1.plotly_chart(fig, use_container_width=True)
        fig = px.bar(tiers, x="risk_tier", y="late_rate_pct", color="risk_tier", color_discrete_map=TIER_COLORS,
                     title="Observed late rate per tier (%)", labels={"risk_tier": "Risk tier", "late_rate_pct": "Late rate (%)"})
        fig.add_hline(y=view["is_late"].mean() * 100, line_dash="dash", annotation_text="average of selection")
        fig.update_layout(showlegend=False, height=340)
        g2.plotly_chart(fig, use_container_width=True)
        st.dataframe(tiers, hide_index=True, use_container_width=True)

        hist = view.assign(outcome=np.where(view["is_late"] == 1, "Late", "On time"))
        fig = px.histogram(hist, x="score", color="outcome", nbins=50, barmode="overlay", opacity=0.65,
                           histnorm="percent", color_discrete_map={"Late": "#C0392B", "On time": "#7FA6D6"},
                           title="Distribution of model scores, by actual outcome",
                           labels={"score": "Model score (late probability)"})
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("The two distributions overlap heavily: that overlap is why precision is low.")

# ---------------------------------------------------------------- 3. states and categories
with tab3:
    min_state = st.slider("Minimum orders for a group to be shown", 50, 1000, 200, step=50)
    s1, s2 = st.columns(2)
    st_view = state[state["orders"] >= min_state].sort_values("late_rate_pct", ascending=False)
    fig = px.bar(st_view, x="late_rate_pct", y="customer_state", orientation="h", hover_data=["orders"],
                 title="Late rate by customer state (%)", labels={"late_rate_pct": "Late rate (%)", "customer_state": ""})
    fig.update_layout(height=max(360, 22 * len(st_view)), yaxis=dict(autorange="reversed"))
    s1.plotly_chart(fig, use_container_width=True)
    cat_view = category[category["orders"] >= max(min_state, 300)].sort_values("late_rate_pct", ascending=False).head(15)
    fig = px.bar(cat_view, x="late_rate_pct", y="product_category", orientation="h", hover_data=["orders"],
                 title="Late rate by product category, top 15 (%)", labels={"late_rate_pct": "Late rate (%)", "product_category": ""})
    fig.update_layout(height=max(360, 22 * len(cat_view)), yaxis=dict(autorange="reversed"))
    s2.plotly_chart(fig, use_container_width=True)
    st.caption("2016 to 2018 as a whole. On later data the ranking was only partly stable: several states with very high "
               "early late rates improved a lot in the June to August 2018 test period, so treat these as descriptive, not as fixed rules.")

# ---------------------------------------------------------------- 4. sellers
with tab4:
    min_seller = st.slider("Minimum orders per seller", 30, 300, 30, step=10)
    sv = seller[seller["orders"] >= min_seller]
    st.write(f"{len(sv):,} sellers with at least {min_seller} orders.")
    fig = px.histogram(sv, x="late_rate_pct", nbins=30, title="Distribution of seller late rates (%)",
                       labels={"late_rate_pct": "Late rate (%)"})
    fig.update_layout(height=340)
    st.plotly_chart(fig, use_container_width=True)
    top = sv.sort_values("late_rate_pct", ascending=False).head(15).copy()
    top["seller"] = top["seller_id"].str[:8] + "..."
    st.dataframe(top[["seller", "seller_state", "orders", "late_orders", "late_rate_pct"]],
                 hide_index=True, use_container_width=True)
    st.warning("**Treat this as a watch list, not a verdict.** When sellers were re-checked on later data, their individual "
               "late rates barely persisted (rank correlation 0.045), and most of the worst sellers had no late order afterwards.")

# ---------------------------------------------------------------- 5. feature importance
with tab5:
    imp = importance.copy()
    imp["feature"] = imp["feature"].map(lambda k: FEATURE_LABELS.get(k, k))
    imp = imp.sort_values("importance", ascending=False)
    n_show = st.slider("Features to show", 5, len(imp), min(15, len(imp)))
    fig = px.bar(imp.head(n_show).iloc[::-1], x="importance", y="feature", orientation="h", error_x="std",
                 title="Permutation importance on the test set", labels={"importance": "Drop in PR-AUC when the feature is shuffled", "feature": ""})
    fig.update_layout(height=max(360, 26 * n_show))
    st.plotly_chart(fig, use_container_width=True)
    st.write("The model relies mostly on the **promised delivery time**. A value near zero means the model barely uses the feature; "
             "a negative value (such as *same state*) means shuffling it slightly *improved* test performance, so the pattern it learned no longer held.")
