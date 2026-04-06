#!/usr/bin/env python3
"""
Calibrate XGBoost raw model via Platt scaling and save with explicit paths.

Applies post-hoc probability calibration to the pre-fitted XGBoost model.
Verifies BrierSkill > 0 and Gini preservation (monotone transform, ±0.001 tolerance).
Saves calibrated model and reliability diagram.
"""

import sys
from pathlib import Path

# Add project root and set up credit_engine alias
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
import src
sys.modules["credit_engine"] = src

import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from credit_engine.model import calibrate_model
from credit_engine.utils import gini_coefficient


def calibrate_xgboost_raw():
    """Load XGBoost model, create test split, calibrate, and evaluate."""
    # ========================================================================
    # Load model and data
    # ========================================================================
    print("Loading pre-trained XGBoost raw model...")
    model_xgb = joblib.load("models/xgboost_raw.pkl")

    print("Loading feature matrix and target...")
    X = pd.read_parquet("data/processed/X_raw_features.parquet")
    y = pd.read_parquet("data/processed/y_train.parquet").squeeze()
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Default rate: {y.mean():.4f}")

    # ========================================================================
    # Create deterministic 80/20 stratified split (must match adversarial validation)
    # ========================================================================
    print("\nCreating deterministic 80/20 stratified split (random_state=42)...")
    X_train_main, X_test, y_train_main, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  X_train: {X_train_main.shape}")
    print(f"  X_test: {X_test.shape}")

    # ========================================================================
    # Compute uncalibrated Gini baseline
    # ========================================================================
    print("\nComputing uncalibrated baseline metrics...")
    y_pred_uncal = model_xgb.predict_proba(X_test)[:, 1]
    gini_uncal = gini_coefficient(y_test, y_pred_uncal)
    print(f"  Uncalibrated Gini: {gini_uncal:.4f}")

    # ========================================================================
    # Calibrate model via Platt scaling with explicit output paths
    # ========================================================================
    print("\nCalibratingXGBoost raw model via Platt scaling...")
    print("  (Using 30% of training data for calibration layer)")

    calibrated_model, brier_uncal, brier_cal = calibrate_model(
        model_xgb,
        X_train_main, y_train_main,
        X_test, y_test,
        method="sigmoid",
        output_model_path="models/xgboost_raw_calibrated.pkl",
        output_figure_path="reports/figures/calibration_reliability.png"
    )

    # ========================================================================
    # Evaluate calibrated model
    # ========================================================================
    print("\nEvaluating calibrated model...")
    y_pred_cal = calibrated_model.predict_proba(X_test)[:, 1]
    gini_cal = gini_coefficient(y_test, y_pred_cal)

    # Verify Gini preservation (monotone transform)
    gini_delta = abs(gini_cal - gini_uncal)
    print(f"  Uncalibrated Gini: {gini_uncal:.4f}")
    print(f"  Calibrated Gini: {gini_cal:.4f}")
    print(f"  Delta: {gini_delta:.4f} (tolerance ±0.001)")
    assert gini_delta <= 0.001, f"Gini delta {gini_delta:.4f} exceeds ±0.001"
    print(f"  ✓ Gini preservation verified (monotone transform property)")

    # Compute BrierSkill
    prevalence = y_test.mean()
    brier_skill = 1.0 - brier_cal / (prevalence * (1.0 - prevalence))
    print(f"\n  Brier Score (uncalibrated): {brier_uncal:.4f}")
    print(f"  Brier Score (calibrated): {brier_cal:.4f}")
    print(f"  BrierSkill: {brier_skill:.4f} (must be > 0)")
    assert brier_skill > 0, f"BrierSkill {brier_skill:.4f} must be > 0"
    print(f"  ✓ BrierSkill gate passed")

    # ========================================================================
    # Verify artifacts
    # ========================================================================
    print("\nVerifying artifacts...")
    assert Path("models/xgboost_raw_calibrated.pkl").exists(), \
        "models/xgboost_raw_calibrated.pkl not found"
    print(f"  ✓ models/xgboost_raw_calibrated.pkl exists")

    assert Path("reports/figures/calibration_reliability.png").exists(), \
        "reports/figures/calibration_reliability.png not found"
    print(f"  ✓ reports/figures/calibration_reliability.png exists")

    # ========================================================================
    # Save summary JSON
    # ========================================================================
    summary = {
        "model": "xgboost_raw_calibrated",
        "brier_uncalibrated": float(brier_uncal),
        "brier_calibrated": float(brier_cal),
        "brier_skill": float(brier_skill),
        "gini_uncalibrated": float(gini_uncal),
        "gini_calibrated": float(gini_cal),
        "gini_delta": float(gini_delta),
        "model_path": "models/xgboost_raw_calibrated.pkl",
        "figure_path": "reports/figures/calibration_reliability.png",
    }

    summary_path = Path("reports/figures/calibration_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ reports/figures/calibration_summary.json created")

    # ========================================================================
    # Print final summary
    # ========================================================================
    print(f"\n{'='*70}")
    print("XGBoost CALIBRATION COMPLETE")
    print(f"{'='*70}")
    print(f"Model: XGBoost raw (63 features)")
    print(f"Method: Platt scaling (sigmoid)")
    print(f"Test set: 80/20 stratified split (random_state=42)")
    print(f"")
    print(f"Brier Score Improvement:")
    print(f"  Uncalibrated: {brier_uncal:.4f}")
    print(f"  Calibrated: {brier_cal:.4f}")
    print(f"  Improvement: {brier_uncal - brier_cal:.4f}")
    print(f"")
    print(f"BrierSkill (Skill vs Prevalence Baseline):")
    print(f"  BrierSkill: {brier_skill:.4f}")
    print(f"  Gate: BrierSkill > 0 ✓ PASSED")
    print(f"")
    print(f"Gini Coefficient (Preserved via Monotone Transform):")
    print(f"  Uncalibrated: {gini_uncal:.4f}")
    print(f"  Calibrated: {gini_cal:.4f}")
    print(f"  Delta: {gini_delta:.4f} (tolerance ±0.001)")
    print(f"  Gate: Delta ≤ 0.001 ✓ PASSED")
    print(f"")
    print(f"Artifacts:")
    print(f"  Model: models/xgboost_raw_calibrated.pkl")
    print(f"  Diagram: reports/figures/calibration_reliability.png")
    print(f"  Summary: reports/figures/calibration_summary.json")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    try:
        calibrate_xgboost_raw()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
