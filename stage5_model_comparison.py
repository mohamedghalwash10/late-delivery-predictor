"""Stage 5: compare four models with time-ordered cross-validation on the TRAINING set.

The test set is NOT used in this stage.
"""
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.build_dataset import load_base_table
from src.config import FIGURES_DIR, PROJECT_ROOT
from src.models import MODEL_NAMES, build_pipeline
from src.preprocessing import split_xy, time_based_split
from src.validation import evaluate_scores, make_time_folds

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)

REPORT_PATH = PROJECT_ROOT / "reports" / "stage5_model_comparison.txt"
RESULTS_PATH = PROJECT_ROOT / "reports" / "stage5_cv_results.csv"
BLUE = "#4C72B0"
_lines: list[str] = []

SUMMARY_COLS = ["pr_auc", "pr_auc_x_baseline", "roc_auc", "recall_top10",
                "precision_top10", "lift_top10", "f1_05", "fit_seconds"]


def log(text="") -> None:
    text = str(text)
    print(text, flush=True)
    _lines.append(text)


def section(title: str) -> None:
    log("\n" + "=" * 70)
    log(title)
    log("=" * 70)


def main() -> None:
    df = load_base_table()
    train, _test, _ = time_based_split(df)   # the test set is never used in this script
    X_train, y_train = split_xy(train)
    folds = make_time_folds(train)
    bought = train["order_purchase_timestamp"]

    section("1. TIME-ORDERED VALIDATION FOLDS (inside the training set)")
    rows = []
    for i, (fit_idx, val_idx) in enumerate(folds, 1):
        rows.append({
            "fold": i,
            "fit_orders": len(fit_idx),
            "fit_late_pct": round(y_train.iloc[fit_idx].mean() * 100, 2),
            "val_orders": len(val_idx),
            "val_late_pct": round(y_train.iloc[val_idx].mean() * 100, 2),
            "val_from": bought.iloc[val_idx].min().date(),
            "val_to": bought.iloc[val_idx].max().date(),
        })
    log(pd.DataFrame(rows).to_string(index=False))
    log("Each fold is fitted only on orders bought AND delivered before its validation window.")
    log("Test set used in this stage: NO")

    section("2. TRAINING AND VALIDATING (progress)")
    records = []
    for name in MODEL_NAMES:
        for i, (fit_idx, val_idx) in enumerate(folds, 1):
            pipe = build_pipeline(name, y_train.iloc[fit_idx])
            t0 = time.time()
            pipe.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
            fit_seconds = time.time() - t0
            proba = pipe.predict_proba(X_train.iloc[val_idx])[:, 1]
            rec = {"model": name, "fold": i, "fit_seconds": fit_seconds}
            rec.update(evaluate_scores(y_train.iloc[val_idx], proba))
            records.append(rec)
            log(f"  {name:<20} fold {i}: PR-AUC {rec['pr_auc']:.3f}  "
                f"ROC-AUC {rec['roc_auc']:.3f}  recall@top10% {rec['recall_top10']:.3f}  "
                f"({fit_seconds:.0f}s)")

    results = pd.DataFrame(records)
    results["pr_auc_x_baseline"] = results["pr_auc"] / results["prevalence"]
    results.to_csv(RESULTS_PATH, index=False)

    section("3. PER-FOLD RESULTS")
    log(results[["model", "fold", "prevalence", "pr_auc", "pr_auc_x_baseline", "roc_auc",
                 "recall_top10", "lift_top10", "f1_05"]].round(3).to_string(index=False))

    section("4. SUMMARY (mean +/- standard deviation across the folds)")
    order = results.groupby("model")["pr_auc"].mean().sort_values(ascending=False).index
    summary_rows = []
    for name in order:
        r = results[results["model"] == name]
        row = {"model": name}
        for col in SUMMARY_COLS:
            row[col] = f"{r[col].mean():.3f} +/- {r[col].std():.3f}"
        summary_rows.append(row)
    log(pd.DataFrame(summary_rows).to_string(index=False))
    log("\nHow to read the columns:")
    log("  pr_auc             area under the precision-recall curve (main metric, rare positive class)")
    log("  pr_auc_x_baseline  pr_auc divided by the fold's late rate (1.0 = no better than guessing)")
    log("  recall_top10       share of all late orders found inside the 10% of orders ranked riskiest")
    log("  precision_top10    share of those flagged orders that really were late")
    log("  lift_top10         precision_top10 divided by the fold's late rate")
    log("  f1_05              F1 at the default 0.5 cut-off (the real cut-off is chosen in Stage 6)")

    section("5. CHART")
    means = results.groupby("model")[["pr_auc", "recall_top10"]].agg(["mean", "std"]).loc[MODEL_NAMES]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (col, title) in zip(axes, [("pr_auc", "PR-AUC (mean across folds)"),
                                       ("recall_top10", "Recall within the riskiest 10% of orders")]):
        ax.bar(MODEL_NAMES, means[(col, "mean")], yerr=means[(col, "std")],
               capsize=5, color=BLUE)
        ax.set_title(title)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "26_model_comparison.png", dpi=130)
    plt.close(fig)
    log("  [figure] 26_model_comparison.png")

    REPORT_PATH.write_text("\n".join(_lines), encoding="utf-8")
    log(f"\nReport saved to:  {REPORT_PATH}")
    log(f"Fold results CSV: {RESULTS_PATH}")


if __name__ == "__main__":
    main()