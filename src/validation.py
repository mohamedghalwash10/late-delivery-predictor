"""Time-ordered validation folds and the evaluation metrics used to compare models."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score, roc_auc_score,
)

# Validation windows start on these dates (expanding-window folds).
FOLD_CUTOFFS = [pd.Timestamp("2017-09-01"),
                pd.Timestamp("2017-12-01"),
                pd.Timestamp("2018-03-01")]


def make_time_folds(train: pd.DataFrame, cutoffs=FOLD_CUTOFFS):
    """Expanding-window folds over the TRAINING set (positional indices).

    For a cut-off date c:
      fit on        orders purchased AND delivered before c   (labels known on c)
      validate on   orders purchased from c until the next cut-off
    This is the same purge rule as the final train/test split, so a fold never
    trains on a label that would not have been known yet.
    `train` must have a clean 0..n-1 index (time_based_split guarantees this).
    """
    bought = train["order_purchase_timestamp"]
    delivered = train["order_delivered_customer_date"]
    ends = list(cutoffs[1:]) + [bought.max() + pd.Timedelta(days=1)]
    folds = []
    for start, end in zip(cutoffs, ends):
        fit_idx = np.flatnonzero((bought < start) & (delivered < start))
        val_idx = np.flatnonzero((bought >= start) & (bought < end))
        folds.append((fit_idx, val_idx))
    return folds


def top_k_metrics(y_true, scores, frac=0.10, seed=42):
    """Precision, recall and lift among the `frac` of orders the model ranks riskiest.

    Business reading: if the marketplace can only review/act on the riskiest 10% of
    orders, how many of those are truly late, and what share of ALL late orders
    does that catch?  Ties are broken randomly, so a model with constant scores
    (the dummy) gets chance-level results.
    """
    y = np.asarray(y_true)
    scores = np.asarray(scores)
    n = len(y)
    k = max(1, int(round(n * frac)))
    order = np.lexsort((np.random.default_rng(seed).random(n), -scores))
    caught = y[order[:k]].sum()
    precision = caught / k
    recall = caught / y.sum() if y.sum() > 0 else np.nan
    lift = precision / y.mean() if y.mean() > 0 else np.nan
    return precision, recall, lift


def evaluate_scores(y_true, proba, threshold=0.5, top_frac=0.10) -> dict:
    """All comparison metrics for one set of predicted late-probabilities."""
    y = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    p_k, r_k, lift = top_k_metrics(y, proba, top_frac)
    return {
        "prevalence": y.mean(),
        "pr_auc": average_precision_score(y, proba),
        "roc_auc": roc_auc_score(y, proba),
        "precision_05": precision_score(y, pred, zero_division=0),
        "recall_05": recall_score(y, pred, zero_division=0),
        "f1_05": f1_score(y, pred, zero_division=0),
        "precision_top10": p_k,
        "recall_top10": r_k,
        "lift_top10": lift,
    }