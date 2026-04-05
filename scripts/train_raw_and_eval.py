"""
train_raw_and_eval.py
---------------------
Train LightGBM and XGBoost on the pre-built raw feature store and report Gini.

Loads from data/processed/X_raw_features.parquet (already built by eval_raw_features.py).
Skips the slow CSV rebuild step — run this after X_raw_features.parquet exists.
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import src  # noqa: E402
sys.modules["credit_engine"] = src

from credit_engine.model import train_lightgbm_optuna, train_xgboost_optuna  # noqa: E402
from credit_engine.utils import evaluate_model  # noqa: E402

_RAW_FEATURES_PATH = project_root / "data" / "processed" / "X_raw_features.parquet"
_Y_PATH = project_root / "data" / "processed" / "y_train.parquet"
_REPORTS_DIR = project_root / "reports"
_MODELS_DIR = project_root / "models"

_LGB_BASELINE_GINI = 0.452
_XGB_BASELINE_GINI = 0.547
_N_TRIALS = 50


def main() -> None:
    print("=" * 70)
    print("Train on Raw Features + Gini Evaluation")
    print("=" * 70)

    # Load pre-built feature store
    print(f"\nLoading raw features from {_RAW_FEATURES_PATH.name}...")
    X_raw = pd.read_parquet(_RAW_FEATURES_PATH)
    y = pd.read_parquet(_Y_PATH).squeeze()
    print(f"  X_raw: {X_raw.shape}  |  y default rate: {y.mean():.2%}")

    # Align indices in case of any row-order mismatch
    common_idx = X_raw.index.intersection(y.index)
    X_raw = X_raw.loc[common_idx]
    y = y.loc[common_idx]
    print(f"  Aligned shape: {X_raw.shape}")

    # Train LightGBM
    print(f"\nTraining LightGBM ({_N_TRIALS} Optuna trials, 10-fold CV)...")
    lgb_model, lgb_metrics, X_lgb_test, y_lgb_test, lgb_params = (
        train_lightgbm_optuna(X_raw, y, n_trials=_N_TRIALS)
    )
    lgb_eval = evaluate_model(lgb_model, X_lgb_test, y_lgb_test, "LightGBM_raw")
    lgb_gini = lgb_eval["Gini"]
    print(f"  LightGBM Gini: {lgb_gini:.4f}  (baseline {_LGB_BASELINE_GINI:.3f}, "
          f"delta {lgb_gini - _LGB_BASELINE_GINI:+.4f})")

    joblib.dump(lgb_model, _MODELS_DIR / "lightgbm_raw.pkl")
    print("  Saved: models/lightgbm_raw.pkl")

    # Train XGBoost
    print(f"\nTraining XGBoost ({_N_TRIALS} Optuna trials, 10-fold CV)...")
    xgb_model, xgb_metrics, X_xgb_test, y_xgb_test, xgb_params = (
        train_xgboost_optuna(X_raw, y, n_trials=_N_TRIALS)
    )
    xgb_eval = evaluate_model(xgb_model, X_xgb_test, y_xgb_test, "XGBoost_raw")
    xgb_gini = xgb_eval["Gini"]
    print(f"  XGBoost Gini: {xgb_gini:.4f}  (baseline {_XGB_BASELINE_GINI:.3f}, "
          f"delta {xgb_gini - _XGB_BASELINE_GINI:+.4f})")

    joblib.dump(xgb_model, _MODELS_DIR / "xgboost_raw.pkl")
    print("  Saved: models/xgboost_raw.pkl")

    # Save results
    results = {
        "lgb_raw_gini": round(lgb_gini, 6),
        "lgb_raw_auc": round(lgb_eval["AUC-ROC"], 6),
        "xgb_raw_gini": round(xgb_gini, 6),
        "xgb_raw_auc": round(xgb_eval["AUC-ROC"], 6),
        "lgb_baseline_gini": _LGB_BASELINE_GINI,
        "xgb_baseline_gini": _XGB_BASELINE_GINI,
        "lgb_delta": round(lgb_gini - _LGB_BASELINE_GINI, 6),
        "xgb_delta": round(xgb_gini - _XGB_BASELINE_GINI, 6),
        "n_raw_features": X_raw.shape[1],
        "n_trials": _N_TRIALS,
    }
    out_path = _REPORTS_DIR / "eval_raw_features.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved: {out_path.name}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  LightGBM  raw Gini = {lgb_gini:.4f}  ({lgb_gini - _LGB_BASELINE_GINI:+.4f} vs baseline)")
    print(f"  XGBoost   raw Gini = {xgb_gini:.4f}  ({xgb_gini - _XGB_BASELINE_GINI:+.4f} vs baseline)")
    print(f"  Raw features used:   {X_raw.shape[1]}")
    print("=" * 70)


if __name__ == "__main__":
    main()
