"""Create app/data/example_orders.csv: 1,000 real orders from the held-out test period.

Run from the project root (after the notebook has created data/processed/orders_base.csv):
    python scripts/make_example_orders.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.build_dataset import BASE_PATH
from src.preprocessing import MODEL_FEATURES, TEST_START

N_EXAMPLES = 1000

if not BASE_PATH.exists():
    raise SystemExit(f"Missing {BASE_PATH}. Run the notebook first (section 2 creates it).")

df = pd.read_csv(BASE_PATH, parse_dates=["order_purchase_timestamp"])
test = df[df["order_purchase_timestamp"] >= TEST_START]
sample = test.sample(n=min(N_EXAMPLES, len(test)), random_state=42)

out = sample[["order_id"] + MODEL_FEATURES + ["is_late"]].rename(columns={"is_late": "actual_is_late"})
float_cols = out.select_dtypes("float").columns
out[float_cols] = out[float_cols].round(4)

target = ROOT / "app" / "data" / "example_orders.csv"
target.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(target, index=False)
size_kb = target.stat().st_size / 1024
print(f"Saved {len(out):,} real test-period orders to {target}  ({size_kb:.0f} KB)")
print(f"Share really late in this sample: {out['actual_is_late'].mean() * 100:.1f}%")
