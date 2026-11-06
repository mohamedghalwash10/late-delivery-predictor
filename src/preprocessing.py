"""Time-based train/test split and the scikit-learn preprocessing pipeline."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.build_dataset import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET

# Orders purchased on or after this date form the untouched test set.
TEST_START = pd.Timestamp("2018-06-01")

# purchase_month is dropped: each calendar month is seen only once or twice, and the same
# month behaves very differently across years (EDA: March 2017 4.6% late, March 2018 19.0%).
# The model would memorise one-off events instead of learning a repeatable pattern.
DROPPED_FEATURES = ["purchase_month"]

# Heavily right-skewed, non-negative features: clip -> impute -> log1p -> scale
LOG_FEATURES = [
    "total_price", "total_freight", "freight_ratio",
    "total_weight_g", "total_volume_cm3",
]

NUMERIC_MODEL = [c for c in NUMERIC_FEATURES if c not in DROPPED_FEATURES]
PLAIN_NUMERIC = [c for c in NUMERIC_MODEL if c not in LOG_FEATURES]
MODEL_FEATURES = NUMERIC_MODEL + CATEGORICAL_FEATURES


class QuantileClipper(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Clip each column to the [lower, upper] quantiles learned during fit().

    Because the bounds are learned from the training data only, extreme values in
    new data are capped using training statistics (no information leaks back).
    """

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(X, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(X, self.upper, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)


def time_based_split(df: pd.DataFrame, test_start: pd.Timestamp = TEST_START):
    """Train = purchased AND delivered before test_start. Test = purchased on/after it.

    Returns (train, test, n_purged). Purged orders were bought before the cut-off but
    delivered after it, so their label would not be known yet on the cut-off date.
    """
    bought = df["order_purchase_timestamp"]
    test = df[bought >= test_start]
    before = df[bought < test_start]
    train = before[before["order_delivered_customer_date"] < test_start]

    def tidy(part):
        return part.sort_values("order_purchase_timestamp").reset_index(drop=True)

    return tidy(train), tidy(test), len(before) - len(train)


def split_xy(df: pd.DataFrame):
    """Return the model inputs and the target. Only MODEL_FEATURES are ever used."""
    return df[MODEL_FEATURES], df[TARGET]


def build_preprocessor() -> ColumnTransformer:
    """Unfitted preprocessing. Fit it on training data only (inside a Pipeline)."""
    skewed = Pipeline([
        ("clip", QuantileClipper(0.01, 0.99)),
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    plain = Pipeline([
        ("clip", QuantileClipper(0.01, 0.99)),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist",  # unseen category -> 'infrequent' bucket
            min_frequency=100,                     # rare categories share one bucket
            sparse_output=False,
        )),
    ])
    return ColumnTransformer(
        [
            ("log_num", skewed, LOG_FEATURES),
            ("num", plain, PLAIN_NUMERIC),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )