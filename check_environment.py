"""Stage 1 verification: Python version, libraries, folders and dataset files."""
import sys
from importlib import import_module
from importlib.metadata import version, PackageNotFoundError

from src.config import (
    DATA_RAW, DATA_PROCESSED, MODELS_DIR, FIGURES_DIR, OLIST_FILES,
)

REQUIRED = {  # import name -> pip package name
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "plotly": "plotly",
    "streamlit": "streamlit",
    "joblib": "joblib",
    "shap": "shap",
    "ipykernel": "ipykernel",
}


def check_python() -> bool:
    v = sys.version_info
    ok = (3, 10) <= (v.major, v.minor) <= (3, 12)
    print(f"[{'OK' if ok else 'WARN'}] Python {v.major}.{v.minor}.{v.micro}")
    print(f"      Interpreter: {sys.executable}")
    if ".venv" not in sys.executable:
        print("[WARN] You are NOT using the project's .venv interpreter.")
        ok = False
    return ok


def check_libraries() -> bool:
    all_ok = True
    for module, package in REQUIRED.items():
        try:
            import_module(module)
            print(f"[OK]   {package:<14} {version(package)}")
        except (ImportError, PackageNotFoundError) as exc:
            print(f"[FAIL] {package:<14} {exc}")
            all_ok = False
    return all_ok


def check_folders() -> bool:
    all_ok = True
    for folder in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, FIGURES_DIR):
        exists = folder.exists()
        print(f"[{'OK' if exists else 'FAIL'}] folder: {folder.name}")
        all_ok &= exists
    return all_ok


def check_dataset() -> bool:
    all_ok = True
    for key, filename in OLIST_FILES.items():
        path = DATA_RAW / filename
        if path.exists():
            size_mb = path.stat().st_size / 1_048_576
            print(f"[OK]   {filename:<42} {size_mb:7.1f} MB")
        else:
            print(f"[FAIL] {filename:<42} MISSING")
            all_ok = False
    return all_ok


def smoke_test() -> bool:
    """Train a tiny XGBoost model inside a sklearn Pipeline, save and reload it."""
    import joblib
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    import tempfile
    from pathlib import Path

    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", XGBClassifier(n_estimators=10, eval_metric="logloss")),
    ]).fit(X, y)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.joblib"
        joblib.dump(pipe, path)
        loaded = joblib.load(path)
        same = np.allclose(pipe.predict_proba(X), loaded.predict_proba(X))
    print(f"[{'OK' if same else 'FAIL'}] sklearn + XGBoost pipeline save/load")
    return same


if __name__ == "__main__":
    print("=" * 60, "\nPYTHON\n" + "=" * 60)
    r1 = check_python()
    print("\n" + "=" * 60, "\nLIBRARIES\n" + "=" * 60)
    r2 = check_libraries()
    print("\n" + "=" * 60, "\nFOLDERS\n" + "=" * 60)
    r3 = check_folders()
    print("\n" + "=" * 60, "\nDATASET FILES\n" + "=" * 60)
    r4 = check_dataset()
    print("\n" + "=" * 60, "\nSMOKE TEST\n" + "=" * 60)
    r5 = smoke_test() if r2 else False

    print("\n" + "=" * 60)
    if all([r1, r2, r3, r4, r5]):
        print("ALL CHECKS PASSED. Ready for Stage 2.")
    else:
        print("Some checks failed. Fix the [FAIL]/[WARN] lines above.")
