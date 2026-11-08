"""Model documentation: algorithms, evaluation and limitations."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from utils import load_json, load_metadata

st.set_page_config(page_title="About the Model", page_icon="📦", layout="wide")
metrics = load_json("metrics.json")
meta = load_metadata()

st.title("About the model")

st.header("Algorithms compared")
st.markdown(
    "| Model | What it is |\n|---|---|\n"
    "| Dummy (prior) | Always outputs the training late rate. The floor every model must beat. |\n"
    "| Logistic Regression | Linear model of the log-odds of being late. Simple and explainable. |\n"
    "| Random Forest | 300 averaged decision trees; finds non-linear effects and interactions. |\n"
    "| XGBoost | Small trees built one after another, each correcting the previous ones. |"
)
st.write(f"**Selected model: {meta['model_name']}** with parameters `{meta['best_params']}`. The three real models were within "
         "noise of each other; the choice followed a rule fixed before seeing the tuned results (PR-AUC within 0.01 of the best, "
         "then the highest recall in the riskiest 10%).")
cmp = pd.DataFrame(metrics["model_comparison_cv"]).rename(columns={
    "model": "Model", "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC", "recall_top10": "Recall in riskiest 10%"})
st.caption("Default settings, mean of 3 time-ordered validation folds (training data only):")
st.dataframe(cmp, hide_index=True, use_container_width=True)

st.header("How it was built")
st.markdown(
    f"- **Data:** {meta['train']['orders']:,} training orders ({meta['train']['first_purchase']} to {meta['train']['last_purchase']}, "
    f"{meta['train']['late_rate_pct']}% late). The test set holds the {metrics['test_orders']:,} orders purchased from {meta['test_start']} onward.\n"
    "- **Time-based split** with purging, because the deployed model must predict future orders.\n"
    "- **Only purchase-time information** is used. Delivery dates, order status and reviews are never model inputs.\n"
    "- **Pipeline:** outlier clipping, median imputation, log transform and scaling for numbers, one-hot encoding for categories, all fitted on training data only.\n"
    "- **Tuning:** grid and randomised search with time-ordered cross-validation on training data only."
)

st.header("Evaluation on the untouched test set")
a, b, c, d = st.columns(4)
a.metric("PR-AUC", f"{metrics['pr_auc']:.3f}", f"random guessing: {metrics['pr_auc_dummy']:.3f}", delta_color="off")
b.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}", "random guessing: 0.500", delta_color="off")
c.metric("Test orders", f"{metrics['test_orders']:,}")
d.metric("Late in test", f"{metrics['test_late']:,}", f"{metrics['test_late_rate_pct']}% of orders", delta_color="off")

st.subheader("What acting on the riskiest orders buys")
budget = pd.DataFrame(metrics["budget_table"]).rename(columns={
    "orders_flagged_pct": "Orders flagged (%)", "precision": "Precision", "recall": "Recall (late orders caught)",
    "lift": "Lift vs. chance", "false_alarms_per_catch": "False alarms per late order caught"})
st.dataframe(budget.round(3), hide_index=True, use_container_width=True)

st.subheader("Risk tiers")
ops = pd.DataFrame(metrics["operating_points_v2"]).rename(columns={"operating point": "Operating point"})
st.dataframe(ops.round(3), hide_index=True, use_container_width=True)
tiers = pd.DataFrame(metrics["tiers_test_v2"]).rename(columns={
    "tier": "Tier", "orders": "Test orders", "share_of_orders_pct": "Share of orders (%)", "late_orders": "Late orders",
    "late_rate_pct": "Late rate (%)", "share_of_late_orders_pct": "Share of late orders (%)"})
st.dataframe(tiers, hide_index=True, use_container_width=True)
st.caption("The High tier is the top 10% of scores on the training data, the Medium tier the next 20%. "
           "In the calmer test period the High tier covered 15.9% of orders.")

st.header("Metrics explained")
st.markdown(
    "- **PR-AUC** (main metric): how well late orders are ranked above on-time ones when late orders are rare. Random guessing equals the late rate.\n"
    "- **ROC-AUC:** overall ranking quality; 0.5 is chance.\n"
    "- **Precision:** of the orders flagged, the share that really were late.\n"
    "- **Recall:** of all late orders, the share that was flagged.\n"
    "- **F1-score:** the balance of precision and recall at a chosen cut-off.\n"
    "- **Accuracy is not used** as a decision metric: always predicting \"on time\" would already be right about 93% of the time."
)

st.header("Limitations")
st.warning(
    "- **Modest signal.** The model ranks risk better than chance, but precision is low: most flagged orders still arrive on time.\n"
    "- **Probabilities are relative risk.** They matched the training period but ran about 2.5 times too high in the 2018 test period "
    f"({metrics['probability_check']['mean_predicted_test_pct']}% predicted against {metrics['probability_check']['actual_test_pct']}% real).\n"
    "- **Time shift.** The late rate, promised delivery windows and even the effect of distance changed between periods, so the model needs monitoring and regular retraining.\n"
    "- **Blind spot for local deliveries.** Orders under 100 km are 22% of test orders but 43% of the late ones, and the model catches very few of them.\n"
    "- **Missing drivers.** No weather, carrier, strike, holiday, warehouse-capacity or seller-stock data.\n"
    "- **Small test sample.** Only 672 late orders, so results carry noise.\n"
    "- **Label selection.** Only delivered orders with a delivery date can be labelled. Multi-seller orders (1.3%) are attributed to the first seller.\n"
    "- **One marketplace, 2016 to 2018.** Not guaranteed to transfer to another company or period."
)

st.header("Data and versions")
st.markdown("Dataset: [Olist Brazilian E-Commerce Public Dataset (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).")
st.caption("Library versions used for training: " + ", ".join(f"{k} {v}" for k, v in meta["versions"].items()))
