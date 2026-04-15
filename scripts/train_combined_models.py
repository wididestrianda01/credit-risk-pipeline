#!/usr/bin/env python3
"""
train_combined_models.py
------------------------
Train XGBoost and CatBoost on the 63-feature combined store with Optuna HPO.
Calibrate both models via Platt scaling (FrozenEstimator pattern).
Persist all four artifacts: xgboost_combined.pkl, xgboost_combined_calibrated.pkl,
catboost_combined.pkl, catboost_combined_calibrated.pkl.

Usage
-----
python scripts/train_combined_models.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Add src to path and set up credit_engine alias (matching conftest.py)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import src  # noqa: E402
sys.modules["credit_engine"] = src

from credit_engine.data_loader import load_data
from credit_engine.model import (
    calibrate_model,
    train_catboost_optuna,
    train_xgboost_optuna,
    prepare_catboost_features,
)
from credit_engine.utils import evaluate_model, gini_coefficient

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

import os

_DATA_DIR = "data/"
_MODELS_DIR = "models/"
_REPORTS_DIR = "reports/"

# HPO trial counts (can be overridden via environment variables for testing)
_XGB_N_TRIALS = int(os.environ.get("XGB_N_TRIALS", "50"))
_CAT_N_TRIALS = int(os.environ.get("CAT_N_TRIALS", "100"))

# CV folds can be reduced for faster testing via FAST_MODE=1
_FAST_MODE = os.environ.get("FAST_MODE", "0") == "1"

_XGB_COMBINED_MODEL_PATH = f"{_MODELS_DIR}xgboost_combined.pkl"
_XGB_COMBINED_CAL_MODEL_PATH = f"{_MODELS_DIR}xgboost_combined_calibrated.pkl"
_CAT_COMBINED_MODEL_PATH = f"{_MODELS_DIR}catboost_combined.pkl"
_CAT_COMBINED_CAL_MODEL_PATH = f"{_MODELS_DIR}catboost_combined_calibrated.pkl"

_XGB_COMBINED_RESULTS_PATH = f"{_REPORTS_DIR}xgboost_combined_results.json"
_CAT_COMBINED_RESULTS_PATH = f"{_REPORTS_DIR}catboost_combined_results.json"


# ---------------------------------------------------------------------------
# Task 1: Load combined feature store
# ---------------------------------------------------------------------------


def load_combined_store() -> tuple[pd.DataFrame, pd.Series]:
    """Load combined 63-feature store and y_train."""
    print("\n" + "=" * 70)
    print("TASK 1: Load combined feature store")
    print("=" * 70)

    X_combined = pd.read_parquet("data/processed/X_combined_features.parquet")
    y_train = pd.read_parquet("data/processed/y_train.parquet").squeeze()

    assert X_combined.shape == (307511, 63), f"Expected (307511, 63), got {X_combined.shape}"
    assert len(y_train) == 307511, f"Expected 307511 samples, got {len(y_train)}"

    print(f"✓ X_combined shape: {X_combined.shape}")
    print(f"✓ y_train shape: {y_train.shape}")
    print(f"✓ Default rate: {y_train.mean():.4f}")

    return X_combined, y_train


# ---------------------------------------------------------------------------
# Task 2: Train XGBoost on combined store with Optuna HPO
# ---------------------------------------------------------------------------


def train_xgboost_combined(X_combined: pd.DataFrame, y_train: pd.Series) -> tuple:
    """
    Train XGBoost on combined 63-feature store with Optuna HPO (50 trials).

    Returns
    -------
    xgb_model, xgb_metrics, X_test, y_test, xgb_best_params
    """
    print("\n" + "=" * 70)
    print(f"TASK 2: Train XGBoost on combined store (Optuna HPO, {_XGB_N_TRIALS} trials)")
    if _FAST_MODE:
        print("  [FAST_MODE enabled: using 2-fold CV for quick iteration]")
    print("=" * 70)

    # Monkey-patch CV folds for fast mode testing
    if _FAST_MODE:
        import src.model as model_module
        original_cv_n_splits = model_module._XGB_CV_N_SPLITS
        model_module._XGB_CV_N_SPLITS = 2
        try:
            xgb_model, xgb_metrics, X_test, y_test, xgb_best_params = train_xgboost_optuna(
                X_combined, y_train, n_trials=_XGB_N_TRIALS
            )
        finally:
            model_module._XGB_CV_N_SPLITS = original_cv_n_splits
    else:
        xgb_model, xgb_metrics, X_test, y_test, xgb_best_params = train_xgboost_optuna(
            X_combined, y_train, n_trials=_XGB_N_TRIALS
        )

    # Evaluate uncalibrated
    uncal_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost (combined, uncalibrated)")
    xgb_uncal_gini = uncal_metrics["Gini"]

    print(f"\nXGBoost uncalibrated metrics:")
    print(f"  Gini:        {xgb_uncal_gini:.4f}")
    print(f"  AUC-ROC:     {uncal_metrics['AUC-ROC']:.4f}")
    print(f"  KS:          {uncal_metrics['KS']:.4f}")
    print(f"  BrierSkill:  {uncal_metrics['BrierSkill']:.4f}")

    # Save uncalibrated model
    joblib.dump(xgb_model, _XGB_COMBINED_MODEL_PATH)
    print(f"\n✓ Saved uncalibrated model: {_XGB_COMBINED_MODEL_PATH}")

    return xgb_model, X_test, y_test, uncal_metrics, xgb_best_params


# ---------------------------------------------------------------------------
# Task 3: Calibrate XGBoost via Platt scaling
# ---------------------------------------------------------------------------


def calibrate_xgboost_combined(
    xgb_model: object,
    X_combined: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    uncal_metrics: dict,
) -> tuple:
    """
    Apply Platt scaling calibration to XGBoost model.

    Returns
    -------
    xgb_calibrated, cal_metrics, brier_uncal, brier_cal
    """
    print("\n" + "=" * 70)
    print("TASK 3: Calibrate XGBoost via Platt scaling (FrozenEstimator)")
    print("=" * 70)

    xgb_calibrated, brier_uncal, brier_cal = calibrate_model(
        xgb_model, X_combined, y_train, X_test, y_test, method="sigmoid"
    )

    # Verify calibration gate (BrierSkill > 0)
    prevalence = y_test.mean()
    brier_skill = 1.0 - (brier_cal / (prevalence * (1 - prevalence)))

    print(f"\nCalibration results:")
    print(f"  Brier (uncalibrated): {brier_uncal:.4f}")
    print(f"  Brier (calibrated):   {brier_cal:.4f}")
    print(f"  BrierSkill:           {brier_skill:.4f} (gate: > 0) {'✓' if brier_skill > 0 else '✗'}")

    if brier_skill <= 0:
        raise AssertionError(f"BrierSkill {brier_skill:.4f} <= 0; gate failed")

    # Evaluate calibrated model
    cal_metrics = evaluate_model(xgb_calibrated, X_test, y_test, "XGBoost (combined, calibrated)")
    xgb_cal_gini = cal_metrics["Gini"]

    gini_delta = uncal_metrics["Gini"] - xgb_cal_gini
    print(f"\nXGBoost calibrated metrics:")
    print(f"  Gini:        {xgb_cal_gini:.4f} (delta: {gini_delta:.6f}, expect < 0.001)")
    print(f"  AUC-ROC:     {cal_metrics['AUC-ROC']:.4f}")
    print(f"  KS:          {cal_metrics['KS']:.4f}")
    print(f"  BrierSkill:  {cal_metrics['BrierSkill']:.4f}")

    # Save calibrated model
    joblib.dump(xgb_calibrated, _XGB_COMBINED_CAL_MODEL_PATH)
    print(f"\n✓ Saved calibrated model: {_XGB_COMBINED_CAL_MODEL_PATH}")

    return xgb_calibrated, cal_metrics, brier_uncal, brier_cal


# ---------------------------------------------------------------------------
# Task 4: Train CatBoost on combined store with native categoricals
# ---------------------------------------------------------------------------


def train_catboost_combined(
    X_combined: pd.DataFrame, y_train: pd.Series
) -> tuple:
    """
    Train CatBoost on combined 63-feature store with native categorical features.
    Uses prepare_catboost_features() to swap WoE cols back to raw strings.

    Returns
    -------
    cat_model, cat_metrics, X_test, y_test, cat_best_params
    """
    print("\n" + "=" * 70)
    print("TASK 4: Train CatBoost on combined store (100 trials, native categoricals)")
    print("=" * 70)

    # Load raw DataFrame for categorical column lookup
    print("\nLoading raw DataFrame for categorical preparation...")
    df_raw = load_data(_DATA_DIR)
    df_raw = df_raw.loc[X_combined.index]
    print(f"  Raw DataFrame shape: {df_raw.shape}")

    # Prepare CatBoost features (swap WoE cols back to raw strings)
    print("\nPreparing CatBoost features (swap WoE → raw strings)...")
    X_combined_cat, cat_cols = prepare_catboost_features(X_woe=X_combined, df_raw=df_raw)

    print(f"  Categorical columns: {cat_cols}")
    print(f"  X_combined_cat shape: {X_combined_cat.shape}")

    # Verify categorical dtypes
    for col in cat_cols:
        dtype_str = str(X_combined_cat[col].dtype)
        is_category = dtype_str.startswith("category")
        print(f"    {col}: {dtype_str} {'✓' if is_category else '✗'}")

    # Run Optuna HPO with 100 trials
    print(f"\nRunning CatBoost Optuna HPO ({_CAT_N_TRIALS} trials)...")
    if _FAST_MODE:
        print("  [FAST_MODE enabled: using 2-fold CV for quick iteration]")
        import src.model as model_module
        original_cv_n_splits = model_module._CAT_CV_N_SPLITS
        model_module._CAT_CV_N_SPLITS = 2
        try:
            cat_model, cat_metrics, X_test, y_test, cat_best_params = train_catboost_optuna(
                X_combined_cat, y_train, n_trials=_CAT_N_TRIALS
            )
        finally:
            model_module._CAT_CV_N_SPLITS = original_cv_n_splits
    else:
        cat_model, cat_metrics, X_test, y_test, cat_best_params = train_catboost_optuna(
            X_combined_cat, y_train, n_trials=_CAT_N_TRIALS
        )

    # Evaluate uncalibrated
    uncal_metrics = evaluate_model(cat_model, X_test, y_test, "CatBoost (combined, uncalibrated)")
    cat_uncal_gini = uncal_metrics["Gini"]

    print(f"\nCatBoost uncalibrated metrics:")
    print(f"  Gini:        {cat_uncal_gini:.4f}")
    print(f"  AUC-ROC:     {uncal_metrics['AUC-ROC']:.4f}")
    print(f"  KS:          {uncal_metrics['KS']:.4f}")
    print(f"  BrierSkill:  {uncal_metrics['BrierSkill']:.4f}")

    # Save uncalibrated model
    joblib.dump(cat_model, _CAT_COMBINED_MODEL_PATH)
    print(f"\n✓ Saved uncalibrated model: {_CAT_COMBINED_MODEL_PATH}")

    return cat_model, X_test, y_test, uncal_metrics, cat_best_params


# ---------------------------------------------------------------------------
# Task 5: Calibrate CatBoost via Platt scaling
# ---------------------------------------------------------------------------


def calibrate_catboost_combined(
    cat_model: object,
    X_combined: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    uncal_metrics: dict,
) -> tuple:
    """
    Apply Platt scaling calibration to CatBoost model.

    Returns
    -------
    cat_calibrated, cal_metrics, brier_uncal, brier_cal
    """
    print("\n" + "=" * 70)
    print("TASK 5: Calibrate CatBoost via Platt scaling (FrozenEstimator)")
    print("=" * 70)

    cat_calibrated, brier_uncal, brier_cal = calibrate_model(
        cat_model, X_combined, y_train, X_test, y_test, method="sigmoid"
    )

    # Verify calibration gate (BrierSkill > 0)
    prevalence = y_test.mean()
    brier_skill = 1.0 - (brier_cal / (prevalence * (1 - prevalence)))

    print(f"\nCalibration results:")
    print(f"  Brier (uncalibrated): {brier_uncal:.4f}")
    print(f"  Brier (calibrated):   {brier_cal:.4f}")
    print(f"  BrierSkill:           {brier_skill:.4f} (gate: > 0) {'✓' if brier_skill > 0 else '✗'}")

    if brier_skill <= 0:
        raise AssertionError(f"BrierSkill {brier_skill:.4f} <= 0; gate failed")

    # Evaluate calibrated model
    cal_metrics = evaluate_model(cat_model, X_test, y_test, "CatBoost (combined, calibrated)")
    cat_cal_gini = cal_metrics["Gini"]

    gini_delta = uncal_metrics["Gini"] - cat_cal_gini
    print(f"\nCatBoost calibrated metrics:")
    print(f"  Gini:        {cat_cal_gini:.4f} (delta: {gini_delta:.6f}, expect < 0.001)")
    print(f"  AUC-ROC:     {cal_metrics['AUC-ROC']:.4f}")
    print(f"  KS:          {cal_metrics['KS']:.4f}")
    print(f"  BrierSkill:  {cal_metrics['BrierSkill']:.4f}")

    # Save calibrated model
    joblib.dump(cat_calibrated, _CAT_COMBINED_CAL_MODEL_PATH)
    print(f"\n✓ Saved calibrated model: {_CAT_COMBINED_CAL_MODEL_PATH}")

    return cat_calibrated, cal_metrics, brier_uncal, brier_cal


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Execute all training and calibration tasks."""
    print("\n" + "=" * 70)
    print("PHASE 4 PLAN 02: Combined Model Training & Calibration")
    print("=" * 70)

    # Task 1: Load data
    X_combined, y_train = load_combined_store()

    # Task 2: Train XGBoost
    xgb_model, X_test_xgb, y_test_xgb, xgb_uncal_metrics, xgb_best_params = (
        train_xgboost_combined(X_combined, y_train)
    )

    # Task 3: Calibrate XGBoost
    xgb_calibrated, xgb_cal_metrics, xgb_brier_uncal, xgb_brier_cal = (
        calibrate_xgboost_combined(xgb_model, X_combined, y_train, X_test_xgb, y_test_xgb, xgb_uncal_metrics)
    )

    # Task 4: Train CatBoost
    cat_model, X_test_cat, y_test_cat, cat_uncal_metrics, cat_best_params = (
        train_catboost_combined(X_combined, y_train)
    )

    # Task 5: Calibrate CatBoost
    cat_calibrated, cat_cal_metrics, cat_brier_uncal, cat_brier_cal = (
        calibrate_catboost_combined(cat_model, X_combined, y_train, X_test_cat, y_test_cat, cat_uncal_metrics)
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: All Models Trained & Calibrated")
    print("=" * 70)

    print(f"\nXGBoost (combined):")
    print(f"  Uncalibrated Gini: {xgb_uncal_metrics['Gini']:.4f}")
    print(f"  Calibrated Gini:   {xgb_cal_metrics['Gini']:.4f}")
    print(f"  BrierSkill:        {xgb_cal_metrics['BrierSkill']:.4f} (gate: > 0) ✓")
    print(f"  Model path:        {_XGB_COMBINED_CAL_MODEL_PATH}")

    print(f"\nCatBoost (combined):")
    print(f"  Uncalibrated Gini: {cat_uncal_metrics['Gini']:.4f}")
    print(f"  Calibrated Gini:   {cat_cal_metrics['Gini']:.4f}")
    print(f"  BrierSkill:        {cat_cal_metrics['BrierSkill']:.4f} (gate: > 0) ✓")
    print(f"  Model path:        {_CAT_COMBINED_CAL_MODEL_PATH}")

    # Save results for Plan 02 Task 1 (comparison table)
    results = {
        "xgboost": {
            "uncalibrated_gini": float(xgb_uncal_metrics["Gini"]),
            "calibrated_gini": float(xgb_cal_metrics["Gini"]),
            "brier_skill": float(xgb_cal_metrics["BrierSkill"]),
            "ks": float(xgb_cal_metrics["KS"]),
            "auc_roc": float(xgb_cal_metrics["AUC-ROC"]),
        },
        "catboost": {
            "uncalibrated_gini": float(cat_uncal_metrics["Gini"]),
            "calibrated_gini": float(cat_cal_metrics["Gini"]),
            "brier_skill": float(cat_cal_metrics["BrierSkill"]),
            "ks": float(cat_cal_metrics["KS"]),
            "auc_roc": float(cat_cal_metrics["AUC-ROC"]),
        },
    }

    Path(_REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    with open(_XGB_COMBINED_RESULTS_PATH, "w") as f:
        json.dump(results["xgboost"], f, indent=2)
    with open(_CAT_COMBINED_RESULTS_PATH, "w") as f:
        json.dump(results["catboost"], f, indent=2)

    print(f"\n✓ Results saved to {_XGB_COMBINED_RESULTS_PATH}")
    print(f"✓ Results saved to {_CAT_COMBINED_RESULTS_PATH}")

    print("\n" + "=" * 70)
    print("DONE: Phase 4 Plan 02 execution complete")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
