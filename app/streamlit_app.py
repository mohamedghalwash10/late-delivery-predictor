"""Home page of the Late-Delivery Predictor app."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from utils import load_json, load_metadata

st.set_page_config(page_title="Late-Delivery Predictor", page_icon="📦", layout="wide")

metrics = load_json("metrics.json")
meta = load_metadata()

st.title("📦 Late-Delivery Predictor")
st.markdown("**A machine learning system that ranks e-commerce orders by their risk of arriving "
            "after the promised delivery date.** Built on the Olist Brazilian E-Commerce dataset "
            "(about 96,000 delivered orders, 2016 to 2018).")

st.header("The business problem")
st.markdown(
    "A marketplace promises every customer an estimated delivery date. When an order arrives late, "
    "reviews drop (late orders averaged 2.3 stars against 4.3 for on-time ones in this data), "
    "support tickets rise and repeat purchases suffer. Normally the marketplace finds out only "
    "*after* the delay has happened.\n\n"
    "This model uses only information available **at purchase time** to rank orders by delivery "
    "risk, so operations teams can spend limited attention where it matters most: contacting the "
    "customer early, pushing a seller to ship, or resetting the promise."
)

st.header("Key model performance")
st.caption(f"Measured once on an untouched test set: {metrics['test_orders']:,} orders purchased "
           f"from {meta['test_start']} onward, of which {metrics['test_late_rate_pct']}% were late.")
budget10 = next(r for r in metrics["budget_table"] if r["orders_flagged_pct"] == 10)
ratio = metrics["pr_auc"] / metrics["pr_auc_dummy"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("PR-AUC", f"{metrics['pr_auc']:.3f}", f"{ratio:.1f}x random guessing ({metrics['pr_auc_dummy']:.3f})",
          delta_color="off")
c2.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}", "0.500 = random", delta_color="off")
c3.metric("Late orders caught in the riskiest 10%", f"{budget10['recall'] * 100:.0f}%",
          "10% would be pure chance", delta_color="off")
c4.metric("Model", metrics["model"], "chosen on time-ordered validation", delta_color="off")

st.info(
    "**Read this honestly.** The model ranks orders better than chance but is *not* an accurate "
    "predictor: most orders it flags as risky still arrive on time. Its probabilities are valid for "
    "the training period; on the test period they ran about 2.5 times too high. Use the output as "
    "**relative risk** to prioritise orders. See *About the Model* for the full evaluation and limitations."
)

st.header("How to use the app")
st.markdown(
    "- **Single Prediction**: describe one order and get its risk tier, a probability, the main drivers and a suggested action.\n"
    "- **Batch Prediction**: upload a CSV of orders, score them all and download the results.\n"
    "- **Analytics Dashboard**: delivery statistics, risk distribution, states, categories, sellers and feature importance.\n"
    "- **About the Model**: algorithms compared, evaluation metrics, results and limitations."
)
st.caption("Use the sidebar to move between pages. The app loads a saved model; nothing is retrained.")
