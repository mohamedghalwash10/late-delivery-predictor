"""The four models, each wrapped in a Pipeline with the shared preprocessor."""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import RANDOM_STATE
from src.preprocessing import build_preprocessor

MODEL_NAMES = ["Dummy (prior)", "Logistic Regression", "Random Forest", "XGBoost"]


def build_pipeline(name: str, y_fit=None) -> Pipeline:
    """Return an UNFITTED pipeline: preprocessor -> model.

    y_fit is the training labels of the data the pipeline is about to be fitted on.
    It is only used to compute XGBoost's class weight, so no validation or test
    labels ever influence a model.
    """
    if name == "Dummy (prior)":
        clf = DummyClassifier(strategy="prior")

    elif name == "Logistic Regression":
        clf = LogisticRegression(
            C=1.0,
            class_weight="balanced",   # re-weights the rare late class
            max_iter=2000,
            random_state=RANDOM_STATE,
        )

    elif name == "Random Forest":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,              # limits overfitting and keeps the saved file small
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    elif name == "XGBoost":
        if y_fit is None:
            raise ValueError("XGBoost needs y_fit to compute scale_pos_weight")
        n_pos = float((y_fit == 1).sum())
        n_neg = float((y_fit == 0).sum())
        clf = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=n_neg / n_pos,   # imbalance ratio from training labels only
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unknown model: {name}")

    return Pipeline([("prep", build_preprocessor()), ("model", clf)])