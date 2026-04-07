#!/usr/bin/env python3
"""
build_comparison_table.py
--------------------------
Plan 04-03: 4-model comparison table (LR, LGB, XGB, CatBoost).

Evaluates all four models on the identical 80/20 held-out test split
(random_state=42, stratify=y) and produces reports/model_comparison_final.json.

Usage
-----
python scripts/build_comparison_table.py

Notes
-----
- LR baseline uses 68 WoE-encoded features (D-03 legacy exception).
  All other models use 63 raw continuous features (X_combined).
- Same 80/20 split seed is used for LR evaluation so sample sizes
  match, but feature spaces differ per D-03.
- XGB and CatBoost models are Platt-calibrated (FrozenEstimator).
- LGB model is not calibrated (Gini rank-based metric is unaffected).
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import src  # noqa: E402
sys.modules["credit_engine"] = src

from credit_engine.utils import evaluate_model  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_COMBINED_FEATURES_PATH = "data/processed/X_combined_features.parquet"
_WOE_FEATURES_PATH = "data/processed/X_features.parquet"
_Y_TRAIN_PATH = "data/processed/y_train.parquet"

_LR_MODEL_PATH = "models/logistic_baseline.pkl"
_LGB_MODEL_PATH = "models/lightgbm_combined.pkl"
_XGB_MODEL_PATH = "models/xgboost_combined_calibrated.pkl"
_CAT_MODEL_PATH = "models/catboost_combined_calibrated.pkl"

_OUTPUT_PATH = "reports/model_comparison_final.json"

_TEST_SIZE = 0.2
_RANDOM_STATE = 42
_EXPECTED_TEST_ROWS = 61503


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_and_split(
    features_path: str,
    y_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load feature store and reproduce the standard 80/20 test split."""
    X = pd.read_parquet(features_path)
    y = pd.read_parquet(y_path).squeeze()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=_TEST_SIZE, stratify=y, random_state=_RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("\n" + "=" * 70)
    print("PLAN 04-03: 4-Model Comparison Table")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load combined feature store and reproduce split for LGB/XGB/CatBoost
    # ------------------------------------------------------------------
    print("\nLoading X_combined (63 raw features) ...")
    _, X_test_combined, _, y_test_combined = _load_and_split(
        _COMBINED_FEATURES_PATH, _Y_TRAIN_PATH
    )
    assert X_test_combined.shape[0] == _EXPECTED_TEST_ROWS, (
        f"Expected {_EXPECTED_TEST_ROWS} test rows, got {X_test_combined.shape[0]}"
    )
    print(f"  X_test_combined: {X_test_combined.shape}, y_test: {y_test_combined.shape}")

    # ------------------------------------------------------------------
    # Load WoE feature store for LR baseline (D-03 legacy exception)
    # ------------------------------------------------------------------
    print("\nLoading X_woe (WoE-encoded features) for LR ...")
    _, X_test_woe, _, y_test_woe = _load_and_split(
        _WOE_FEATURES_PATH, _Y_TRAIN_PATH
    )
    print(f"  X_test_woe: {X_test_woe.shape}, y_test: {y_test_woe.shape}")

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    print("\nLoading models ...")
    lr_model = joblib.load(_LR_MODEL_PATH)
    lgb_model = joblib.load(_LGB_MODEL_PATH)
    xgb_model = joblib.load(_XGB_MODEL_PATH)
    cat_model = joblib.load(_CAT_MODEL_PATH)
    print("  All 4 models loaded.")

    # ------------------------------------------------------------------
    # Evaluate each model
    # ------------------------------------------------------------------
    print("\nEvaluating models ...")

    # LR: WoE features (D-03 legacy exception)
    print("  Logistic Regression (WoE features) ...")
    lr_metrics = evaluate_model(lr_model, X_test_woe, y_test_woe, "Logistic Regression")

    # LGB: raw combined features (uncalibrated — Gini is rank-based)
    print("  LightGBM (raw combined features) ...")
    lgb_metrics = evaluate_model(lgb_model, X_test_combined, y_test_combined, "LightGBM")

    # XGB: raw combined features + Platt calibration
    print("  XGBoost calibrated (raw combined features) ...")
    xgb_metrics = evaluate_model(xgb_model, X_test_combined, y_test_combined, "XGBoost")

    # CatBoost: raw combined features + native categoricals + Platt calibration
    print("  CatBoost calibrated (raw combined features) ...")
    cat_metrics = evaluate_model(cat_model, X_test_combined, y_test_combined, "CatBoost")

    results = {
        "LR": lr_metrics,
        "LGB": lgb_metrics,
        "XGB": xgb_metrics,
        "CatBoost": cat_metrics,
    }

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("4-Model Comparison Table")
    print("=" * 70)
    print(f"\n{'Model':<15} {'Gini':>8} {'KS':>8} {'BrierSkill':>12} {'AvgPrecision':>14} {'AUC-ROC':>9}")
    print("-" * 70)
    for model_name, metrics in results.items():
        print(
            f"{model_name:<15}"
            f" {metrics['Gini']:>8.4f}"
            f" {metrics['KS']:>8.4f}"
            f" {metrics['BrierSkill']:>12.4f}"
            f" {metrics['AvgPrecision']:>14.4f}"
            f" {metrics['AUC-ROC']:>9.4f}"
        )

    best_model = max(results, key=lambda m: results[m]["Gini"])
    best_gini = results[best_model]["Gini"]
    gini_target = 0.60
    gap_to_target = gini_target - best_gini
    print(f"\nBest model: {best_model} (Gini={best_gini:.4f})")
    print(f"Gap to Gini≥0.60 target: {gap_to_target:+.4f}")

    # ------------------------------------------------------------------
    # Assemble and save comparison JSON
    # ------------------------------------------------------------------
    comparison = {
        "evaluation_date": date.today().isoformat(),
        "test_split": {
            "combined_train_size": 307511 - _EXPECTED_TEST_ROWS,
            "combined_test_size": X_test_combined.shape[0],
            "woe_test_size": X_test_woe.shape[0],
            "random_state": _RANDOM_STATE,
            "stratified": True,
            "test_size_fraction": _TEST_SIZE,
        },
        "models": {
            "LR": {
                "path": _LR_MODEL_PATH,
                "feature_set": f"{X_test_woe.shape[1]} WoE-encoded (legacy benchmark — D-03 exception)",
                "calibrated": False,
                "n_features": X_test_woe.shape[1],
            },
            "LGB": {
                "path": _LGB_MODEL_PATH,
                "feature_set": "63 raw continuous",
                "calibrated": False,
                "n_features": X_test_combined.shape[1],
            },
            "XGB": {
                "path": _XGB_MODEL_PATH,
                "feature_set": "63 raw continuous",
                "calibrated": True,
                "calibration_method": "Platt (sigmoid, FrozenEstimator)",
                "n_features": X_test_combined.shape[1],
            },
            "CatBoost": {
                "path": _CAT_MODEL_PATH,
                "feature_set": "63 raw continuous (4 categorical native: CODE_GENDER, NAME_EDUCATION_TYPE, NAME_INCOME_TYPE, ORGANIZATION_TYPE)",
                "calibrated": True,
                "calibration_method": "Platt (sigmoid, FrozenEstimator)",
                "n_features": X_test_combined.shape[1],
            },
        },
        "metrics_per_model": {k: {mk: float(mv) if isinstance(mv, float) else mv for mk, mv in v.items()} for k, v in results.items()},
        "best_model": best_model,
        "best_gini": float(best_gini),
        "gini_target": gini_target,
        "gap_to_target": float(gap_to_target),
        "note_lr_legacy": (
            f"LR trained on {X_test_woe.shape[1]} WoE-encoded features (legacy benchmark). "
            "All other models trained on 63 raw continuous features (X_combined). "
            "LR not retrained per D-03 planning decision; enters comparison table "
            "as an interpretable IRB-compatible benchmark using a different test split. "
            "Gini values are not directly comparable to LGB/XGB/CatBoost for D-01 models."
        ),
    }

    Path(_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(_OUTPUT_PATH, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\n✓ Comparison table saved: {_OUTPUT_PATH}")

    # ------------------------------------------------------------------
    # Verify output
    # ------------------------------------------------------------------
    with open(_OUTPUT_PATH) as f:
        loaded = json.load(f)
    assert "metrics_per_model" in loaded
    assert all(m in loaded["metrics_per_model"] for m in ["LR", "LGB", "XGB", "CatBoost"])
    assert "Gini" in loaded["metrics_per_model"]["XGB"]
    print("✓ Verification passed.")

    print("\n" + "=" * 70)
    print("DONE: Plan 04-03 complete")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
