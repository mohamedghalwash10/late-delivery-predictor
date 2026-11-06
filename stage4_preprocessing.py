"""Stage 4: time-based split, leakage checks and a demonstration of the preprocessing."""
import numpy as np
import pandas as pd

from src.build_dataset import (
    ANALYSIS_ONLY, CATEGORICAL_FEATURES, ID_COLUMNS, TARGET, load_base_table,
)
from src.config import PROJECT_ROOT
from src.preprocessing import (
    LOG_FEATURES, MODEL_FEATURES, NUMERIC_MODEL, PLAIN_NUMERIC, TEST_START,
    build_preprocessor, split_xy, time_based_split,
)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

REPORT_PATH = PROJECT_ROOT / "reports" / "stage4_preprocessing_report.txt"
_lines: list[str] = []


def log(text="") -> None:
    text = str(text)
    print(text)
    _lines.append(text)


def section(title: str) -> None:
    log("\n" + "=" * 70)
    log(title)
    log("=" * 70)


def main() -> None:
    df = load_base_table()

    section("1. DUPLICATES")
    log(f"Duplicated order_id: {int(df['order_id'].duplicated().sum())}")
    log(f"Fully duplicated rows: {int(df.duplicated().sum())}")

    section("2. TIME-BASED SPLIT")
    train, test, purged = time_based_split(df)
    log(f"Test period starts: {TEST_START.date()}")
    log(f"Purged from training (bought before, delivered after the cut-off): {purged:,}")
    rows = []
    for name, part in (("train", train), ("test", test)):
        rows.append({
            "set": name,
            "orders": len(part),
            "late": int(part[TARGET].sum()),
            "late_rate_pct": round(part[TARGET].mean() * 100, 2),
            "first_purchase": part["order_purchase_timestamp"].min().date(),
            "last_purchase": part["order_purchase_timestamp"].max().date(),
        })
    log(pd.DataFrame(rows).to_string(index=False))

    section("3. LEAKAGE CHECKS")
    checks = {
        "no analysis-only column is a model feature":
            not set(MODEL_FEATURES) & set(ANALYSIS_ONLY),
        "no id column is a model feature":
            not set(MODEL_FEATURES) & set(ID_COLUMNS),
        "target is not a model feature":
            TARGET not in MODEL_FEATURES,
        "every training purchase is before the test period":
            train["order_purchase_timestamp"].max() < test["order_purchase_timestamp"].min(),
        "every training delivery happened before the test period":
            train["order_delivered_customer_date"].max() < TEST_START,
        "no order appears in both sets":
            not set(train["order_id"]) & set(test["order_id"]),
    }
    for name, ok in checks.items():
        log(f"[{'OK' if ok else 'FAIL'}] {name}")
    assert all(checks.values()), "A leakage check failed"
    log(f"\nModel features ({len(MODEL_FEATURES)}): {MODEL_FEATURES}")

    X_train, y_train = split_xy(train)
    X_test, y_test = split_xy(test)

    section("4. MISSING VALUES IN TRAINING FEATURES (imputed inside the pipeline)")
    miss = X_train.isna().sum()
    log(miss[miss > 0].to_string() if (miss > 0).any() else "None")

    section("5. PREPROCESSOR FITTED ON TRAINING DATA ONLY")
    pre = build_preprocessor()
    Xt_train = pre.fit_transform(X_train)   # fit happens on training data only
    Xt_test = pre.transform(X_test)         # test is only transformed
    names = pre.get_feature_names_out()

    log("Outlier clipping bounds learned from training data (1st / 99th percentile):")
    bounds = []
    for key, cols in (("log_num", LOG_FEATURES), ("num", PLAIN_NUMERIC)):
        clipper = pre.named_transformers_[key].named_steps["clip"]
        for col, lo, hi in zip(cols, clipper.lower_bounds_, clipper.upper_bounds_):
            bounds.append({"feature": col, "lower_p1": lo, "upper_p99": hi,
                           "train_max": X_train[col].max()})
    log(pd.DataFrame(bounds).round(2).to_string(index=False))

    log("\nCategorical encoding:")
    ohe = pre.named_transformers_["cat"].named_steps["onehot"]
    for col, cats, rare in zip(CATEGORICAL_FEATURES, ohe.categories_,
                               ohe.infrequent_categories_):
        n_rare = 0 if rare is None else len(rare)
        log(f"  {col:<18} {len(cats):>3} categories seen in training, "
            f"{n_rare} grouped as infrequent")

    section("6. TRANSFORMED MATRICES")
    log(f"Input features: {len(MODEL_FEATURES)} "
        f"({len(NUMERIC_MODEL)} numeric, {len(CATEGORICAL_FEATURES)} categorical)")
    log(f"Train: {Xt_train.shape}   Test: {Xt_test.shape}")
    log(f"NaN left in train: {int(np.isnan(Xt_train).sum())}   "
        f"in test: {int(np.isnan(Xt_test).sum())}")

    n = len(NUMERIC_MODEL)
    tr = pd.DataFrame(Xt_train[:, :n], columns=names[:n])
    te = pd.DataFrame(Xt_test[:, :n], columns=names[:n])
    stats = pd.DataFrame({
        "train_mean": tr.mean(), "train_std": tr.std(),
        "test_mean": te.mean(), "test_std": te.std(),
    }).round(3)
    log("\nScaled numeric columns. Train should be about 0 / 1. Test is NOT exactly 0 / 1,")
    log("because it is scaled with training statistics (that is correct, no leakage):")
    log(stats.to_string())

    REPORT_PATH.write_text("\n".join(_lines), encoding="utf-8")
    log(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()