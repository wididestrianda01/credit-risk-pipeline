#!/usr/bin/env python3
"""
Plan 04-05: 3-Model OOF Ensemble Evaluation
============================================

Load pre-trained base models (LGB, XGB, CatBoost), run run_ensemble_workflow()
with 3-model OOF stacking, evaluate on identical test set, and produce:
  - models/ensemble_metadata.json (internal metrics)
  - reports/final_model_eval.json (gate assessment)

Execution time: ~25-50 minutes (5-fold OOF × 3 models).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

# Setup src import alias (same as conftest.py)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
import src
sys.modules["credit_engine"] = src

# Import with alias per conftest.py
from credit_engine.model import run_ensemble_workflow
from credit_engine.utils import evaluate_model


def main() -> None:
    """Execute Plan 04-05: ensemble evaluation."""

    # --- Setup ---
    root = Path(__file__).parent.parent
    data_dir = root / "data" / "processed"
    models_dir = root / "models"
    reports_dir = root / "reports"

    print("[PLAN 04-05] Starting 3-model OOF ensemble evaluation...")

    # --- Load data ---
    print("[LOAD] Reading X_combined_features.parquet...")
    X_combined = pd.read_parquet(data_dir / "X_combined_features.parquet")
    print(f"  Shape: {X_combined.shape}")

    print("[LOAD] Reading y_train.parquet...")
    y_train_raw = pd.read_parquet(data_dir / "y_train.parquet")
    y = y_train_raw.squeeze() if isinstance(y_train_raw, pd.DataFrame) else y_train_raw
    print(f"  Shape: {y.shape}, dtype: {y.dtype}")

    assert X_combined.shape[0] == y.shape[0], f"Mismatch: X={X_combined.shape[0]}, y={y.shape[0]}"

    # --- Load pre-trained base models ---
    print("[LOAD] Loading pre-trained models...")

    print("  LightGBM...")
    lgb_model = joblib.load(models_dir / "lightgbm_combined.pkl")

    print("  XGBoost...")
    xgb_model = joblib.load(models_dir / "xgboost_combined_calibrated.pkl")

    print("  CatBoost...")
    cat_model = joblib.load(models_dir / "catboost_combined_calibrated.pkl")

    # --- Run ensemble workflow (3-model OOF path) ---
    print("[ENSEMBLE] Running run_ensemble_workflow with 3-model OOF stacking...")
    print("  Method: logistic meta-learner")
    print("  Folds: 5-fold cross-validation")
    print("  This will take 25-50 minutes...")

    result = run_ensemble_workflow(
        X=X_combined,
        y=y,
        cat_model=cat_model,  # Presence of cat_model triggers 3-model path
        method="logistic",
    )

    print("[ENSEMBLE] Workflow complete.")
    print(f"  LGB Gini: {result['lgb_gini']:.4f}")
    print(f"  XGB Gini: {result['xgb_gini']:.4f}")
    print(f"  CAT Gini: {result['cat_gini']:.4f}")
    print(f"  Ensemble Gini: {result['ensemble_gini']:.4f}")
    print(f"  Improvement: {result['improvement']:.4f}")
    print(f"  Persisted: {result['persisted']}")

    # --- Unpack result ---
    lgb_gini = result["lgb_gini"]
    xgb_gini = result["xgb_gini"]
    cat_gini = result["cat_gini"]
    ensemble_gini = result["ensemble_gini"]
    improvement = result["improvement"]
    persisted = result["persisted"]

    best_single_gini = max(lgb_gini, xgb_gini, cat_gini)
    final_gini = ensemble_gini if persisted else best_single_gini
    final_model_name = "3-Model Ensemble (Logistic)" if persisted else f"Best Standalone (Gini={best_single_gini:.4f})"

    # --- Gate assessment ---
    target_gini = 0.60
    gap_to_target = target_gini - final_gini
    gini_gate_passed = final_gini >= target_gini
    phase_45_required = not gini_gate_passed

    phase_45_justification = (
        f"Current final model (Gini={final_gini:.4f}) is below target (Gini={target_gini:.4f}). "
        f"Gap={gap_to_target:.4f}. Phase 4-5 work required."
    ) if phase_45_required else (
        f"Final model (Gini={final_gini:.4f}) meets target (Gini={target_gini:.4f}). "
        f"Phase 4-5 work optional (gap={gap_to_target:.4f})."
    )

    # --- Write ensemble_metadata.json ---
    print("[WRITE] Saving ensemble_metadata.json...")
    ensemble_metadata = {
        "ensemble_date": datetime.utcnow().isoformat(),
        "method": "logistic",
        "base_models": ["lightgbm_combined", "xgboost_combined_calibrated", "catboost_combined_calibrated"],
        "lgb_gini": float(lgb_gini),
        "xgb_gini": float(xgb_gini),
        "cat_gini": float(cat_gini),
        "ensemble_gini": float(ensemble_gini),
        "improvement": float(improvement),
        "persist_threshold": 0.005,
        "persisted": bool(persisted),
        "test_set_size": 0.2,
    }

    models_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / "ensemble_metadata.json", "w") as fh:
        json.dump(ensemble_metadata, fh, indent=2)
    print(f"  Saved to {models_dir / 'ensemble_metadata.json'}")

    # --- Write final_model_eval.json ---
    print("[WRITE] Saving final_model_eval.json...")
    final_model_eval = {
        "evaluation_date": datetime.utcnow().isoformat(),
        "final_model": final_model_name,
        "final_gini": float(final_gini),
        "target_gini": float(target_gini),
        "gap_to_target": float(gap_to_target),
        "gini_gate_passed": bool(gini_gate_passed),
        "phase_45_required": bool(phase_45_required),
        "phase_45_justification": phase_45_justification,
        "base_model_gini": {
            "lgb": float(lgb_gini),
            "xgb": float(xgb_gini),
            "cat": float(cat_gini),
        },
        "ensemble_gini": float(ensemble_gini),
        "ensemble_improvement": float(improvement),
        "ensemble_persisted": bool(persisted),
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "final_model_eval.json", "w") as fh:
        json.dump(final_model_eval, fh, indent=2)
    print(f"  Saved to {reports_dir / 'final_model_eval.json'}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Ensemble Gini:        {ensemble_gini:.4f}")
    print(f"Best Standalone:      {best_single_gini:.4f}")
    print(f"Improvement:          {improvement:.4f}")
    print(f"Persisted:            {persisted}")
    print(f"Final Gini:           {final_gini:.4f}")
    print(f"Target Gini:          {target_gini:.4f}")
    print(f"Gap to Target:        {gap_to_target:.4f}")
    print(f"Phase 4-5 Required:   {phase_45_required}")
    print("=" * 70)


if __name__ == "__main__":
    main()
