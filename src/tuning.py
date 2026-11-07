"""Base pipelines and search spaces for hyperparameter tuning (training data only)."""
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import RANDOM_STATE
from src.preprocessing import build_preprocessor
from src.validation import top_k_metrics

TUNED_MODELS = ["Logistic Regression", "Random Forest", "XGBoost"]


def recall_top10_scorer(estimator, X, y):
    """Share of all late orders caught inside the riskiest 10% (higher is better)."""
    return top_k_metrics(y, estimator.predict_proba(X)[:, 1], 0.10)[1]


SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "recall_top10": recall_top10_scorer,
}


def base_pipeline(name: str) -> Pipeline:
    """Unfitted preprocessor + model. Hyperparameters are set by the search."""
    if name == "Logistic Regression":
        clf = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    elif name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                     random_state=RANDOM_STATE)
    elif name == "XGBoost":
        clf = XGBClassifier(eval_metric="aucpr", tree_method="hist", n_jobs=-1,
                            random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("prep", build_preprocessor()), ("model", clf)])


# name -> (search type, parameter space, number of random candidates)
SEARCH_SPACES = {
    "Logistic Regression": (
        "grid",
        {
            "model__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
            "model__class_weight": ["balanced", None],
        },
        None,
    ),
    "Random Forest": (
        "random",
        {
            "model__max_depth": [6, 8, 10, 12, 16],
            "model__min_samples_leaf": [10, 20, 40, 80],
            "model__max_features": ["sqrt", 0.2],
            "model__class_weight": ["balanced_subsample", None],
        },
        12,
    ),
    "XGBoost": (
        "random",
        {
            "model__n_estimators": randint(100, 500),
            "model__learning_rate": loguniform(0.01, 0.2),
            "model__max_depth": randint(2, 7),
            "model__min_child_weight": randint(1, 30),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.5, 0.5),
            "model__reg_lambda": loguniform(0.5, 20),
            "model__scale_pos_weight": [1, 3, 6, 12, 25],
        },
        30,
    ),
}
