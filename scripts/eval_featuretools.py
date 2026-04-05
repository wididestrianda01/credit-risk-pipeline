"""
eval_featuretools.py
--------------------
Evaluate Gini contribution of featuretools DFS features.

Runs two XGBoost evaluations:
  1. Featuretools features only  (measures isolated DFS contribution)
  2. Raw features + featuretools  (measures additive gain over raw baseline)

Requires:
  - data/processed/X_featuretools.parquet  (from build_featuretools_store.py)
  - data/processed/y_train.parquet
  - data/processed/X_raw_features.parquet  (optional, for combined eval)

Usage
-----
    python -u scripts/eval_featuretools.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import src  # noqa: E402
sys.modules["credit_engine"] = src

from credit_engine.model import train_xgboost_optuna  # noqa: E402
from credit_engine.utils import evaluate_model  # noqa: E402

_Y_PATH = project_root / "data" / "processed" / "y_train.parquet"
_FT_PATH = project_root / "data" / "processed" / "X_featuretools.parquet"
_RAW_PATH = project_root / "data" / "processed" / "X_raw_features.parquet"
_OUTPUT_PATH = project_root / "reports" / "featuretools_eval_results.json"
_N_TRIALS = 30
_TEST_SIZE = 0.2
_RANDOM_STATE = 42


def _train_and_eval(X: pd.DataFrame, y: pd.Series, tag: str) -> dict:
    """Train XGB with Optuna HPO and return metrics dict."""
    print(f"\n[{tag}] Feature matrix: {X.shape}")
    print(f"[{tag}] Starting XGBoost Optuna ({_N_TRIALS} trials)...")

    model, metrics, X_test, y_test, best_params = train_xgboost_optuna(
        X, y, n_trials=_N_TRIALS
    )
    gini = metrics["Gini"]
    print(f"[{tag}] Gini = {gini:.4f}  |  AUC = {metrics['AUC-ROC']:.4f}  |  KS = {metrics['KS']:.4f}")
    return {"tag": tag, "n_features": X.shape[1], **metrics}


def main() -> None:
    print("=" * 70)
    print("Featuretools DFS Feature Evaluation")
    print("=" * 70)

    # Load targets
    y = pd.read_parquet(_Y_PATH).squeeze()
    print(f"y_train: {len(y):,} rows, default rate {y.mean():.2%}")

    # Load featuretools features
    if not _FT_PATH.exists():
        print(f"\nERROR: {_FT_PATH} not found. Run build_featuretools_store.py first.")
        sys.exit(1)

    X_ft = pd.read_parquet(_FT_PATH)
    print(f"Featuretools matrix: {X_ft.shape}")

    results = []

    # --- Evaluation 1: Featuretools only ---
    result_ft = _train_and_eval(X_ft, y, tag="featuretools_only")
    results.append(result_ft)

    # --- Evaluation 2: Raw + Featuretools (if raw available) ---
    if _RAW_PATH.exists():
        X_raw = pd.read_parquet(_RAW_PATH)
        print(f"\nRaw features matrix: {X_raw.shape}")

        # Align indices
        common_idx = X_raw.index.intersection(X_ft.index)
        X_raw = X_raw.loc[common_idx]
        X_ft_aligned = X_ft.loc[common_idx]
        y_aligned = y.loc[common_idx]

        # Merge, avoiding column name collisions (suffix featuretools cols)
        ft_new_cols = [c for c in X_ft_aligned.columns if c not in X_raw.columns]
        X_combined = pd.concat([X_raw, X_ft_aligned[ft_new_cols]], axis=1)
        print(f"Overlapping cols dropped: {len(X_ft_aligned.columns) - len(ft_new_cols)}")

        result_combined = _train_and_eval(X_combined, y_aligned, tag="raw_plus_featuretools")
        results.append(result_combined)

        # Delta vs featuretools only (same baseline: raw features)
        raw_only_gini = result_ft["Gini"]  # conservative: compare combined vs ft-only
        delta = result_combined["Gini"] - raw_only_gini
        print(f"\nAdditive gain (raw+ft vs ft_only): Δ Gini = {delta:+.4f}")
    else:
        print(f"\n(Skipping combined eval — {_RAW_PATH.name} not found)")

    # Save results
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_PATH.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for r in results:
        print(f"  [{r['tag']}]  n_features={r['n_features']}  Gini={r['Gini']:.4f}")
    print(f"\nResults saved to {_OUTPUT_PATH.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
