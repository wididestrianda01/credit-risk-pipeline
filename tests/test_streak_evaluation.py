"""Phase 1 Plan 02 (Wave 2): Streak feature evaluation test runner."""
import json
import shutil
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from credit_engine.data_loader import load_data
from credit_engine.features import engineer_instalment_streaks
from credit_engine.model import train_lightgbm_optuna
from credit_engine.utils import gini_coefficient


# Suppress optuna verbosity
warnings.filterwarnings("ignore", category=UserWarning)

# Execution constants
_COMMIT_THRESHOLD: float = 0.01
_RANDOM_STATE: int = 42
_TEST_SIZE: float = 0.2
_N_TRIALS: int = 50


@pytest.mark.slow
def test_streak_feature_evaluation():
    """Execute Phase 1 Plan 02 (Wave 2) streak feature evaluation."""
    _MIN_PRODUCTION_ROWS = 100_000
    raw_path = Path("data/processed/X_raw_features.parquet")
    if not raw_path.exists():
        pytest.skip("data/processed/X_raw_features.parquet not found — production data required")
    row_count = pd.read_parquet(raw_path, columns=[pd.read_parquet(raw_path).columns[0]]).shape[0]
    if row_count < _MIN_PRODUCTION_ROWS:
        pytest.skip(
            f"X_raw_features.parquet has {row_count} rows — production data required "
            f"(>= {_MIN_PRODUCTION_ROWS}). Feature store may have been overwritten by mock test data."
        )

    print("\n" + "=" * 80)
    print("Phase 1 Plan 02 (Wave 2): Evaluate Instalment Streak Features")
    print("=" * 80)

    # ========================================================================
    # TASK 1: Create streak-free baseline feature set and train baseline XGBoost
    # ========================================================================
    print("\n" + "=" * 80)
    print("TASK 1: Streak-free baseline XGBoost training")
    print("=" * 80)

    # Step 1: Load and inspect current feature store
    print("\nStep 1: Load feature store")
    X_raw = pd.read_parquet("data/processed/X_raw_features.parquet")
    y_train = pd.read_parquet("data/processed/y_train.parquet").squeeze()

    print(f"✓ Feature store shape: {X_raw.shape}")
    print(f"  Target shape: {y_train.shape}")
    print(f"  Default rate: {y_train.mean():.2%}")

    # Step 2: Create streak-free feature set (per D-05)
    print("\nStep 2: Drop ALL streak columns for clean baseline")
    streak_cols = [c for c in X_raw.columns if "streak" in c.lower()]
    print(f"  Streak columns found: {streak_cols}")
    X_no_streaks = X_raw.drop(columns=streak_cols).copy()
    print(f"✓ Streak-free shape: {X_no_streaks.shape}")
    print(f"  Columns dropped: {len(streak_cols)}")
    assert (
        X_no_streaks.shape[1] == X_raw.shape[1] - len(streak_cols)
    ), "Drop failed"

    # Step 3: Create deterministic 80/20 split (MUST match calibration)
    print("\nStep 3: Create 80/20 train/test split (random_state=42)")
    X_train, X_test, y_train_split, y_test = train_test_split(
        X_no_streaks, y_train, test_size=_TEST_SIZE, random_state=_RANDOM_STATE,
        stratify=y_train
    )
    print(f"✓ Train shape: {X_train.shape}")
    print(f"  Test shape: {X_test.shape}")
    print(f"  Train default rate: {y_train_split.mean():.2%}")
    print(f"  Test default rate: {y_test.mean():.2%}")

    # Step 4: Train LightGBM baseline WITHOUT any streak features
    # Note: train_xgboost_optuna was refactored to a file-path API; use LGB (DataFrame API) for AB eval
    print(f"\nStep 4: Train baseline LightGBM (Optuna n_trials={_N_TRIALS})")
    baseline_model, metrics_baseline, X_test_ret, y_test_ret, best_params_baseline = (
        train_lightgbm_optuna(X_train, y_train_split, n_trials=_N_TRIALS)
    )
    print(f"✓ Baseline model trained")
    print(f"  Best params keys: {list(best_params_baseline.keys())}")

    # Measure baseline Gini
    print("\nStep 5: Measure baseline Gini")
    y_pred_baseline = baseline_model.predict_proba(X_test)[:, 1]
    gini_baseline = gini_coefficient(y_test, y_pred_baseline)
    print(f"✓ Baseline Gini (no streaks): {gini_baseline:.4f}")

    # Step 6: Save baseline model and results
    print("\nStep 6: Save baseline model and results")
    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(baseline_model, "models/xgboost_raw_baseline_no_streaks.pkl")
    baseline_results = {
        "gini": float(gini_baseline),
        "best_params": best_params_baseline,
        "n_features": X_no_streaks.shape[1],
        "feature_names": X_no_streaks.columns.tolist(),
    }
    Path("reports").mkdir(parents=True, exist_ok=True)
    with open("reports/streak_evaluation_baseline.json", "w") as f:
        json.dump(baseline_results, f, indent=2)
    print(f"✓ Baseline model saved to models/xgboost_raw_baseline_no_streaks.pkl")
    print(f"✓ Baseline results saved to reports/streak_evaluation_baseline.json")
    print(f"✓ Baseline Gini: {gini_baseline:.4f}")

    # ========================================================================
    # TASK 2: Compute 5 OLS-based instalment streak features
    # ========================================================================
    print("\n" + "=" * 80)
    print("TASK 2: Compute 5 OLS-based instalment streak features")
    print("=" * 80)

    # Step 7: Load raw installments data and compute OLS-based streak features
    print("\nStep 7: Load raw installments data")
    data_dict = load_data("data/", mode="train")
    df_inst = data_dict["installments_payments"]
    print(f"✓ Installments table shape: {df_inst.shape}")
    print(f"  Columns: {df_inst.columns.tolist()}")

    print("\nStep 8: Compute 5 OLS-based streak features")
    streak_features = engineer_instalment_streaks(df_inst)
    print(f"✓ Streak features shape: {streak_features.shape}")
    print(f"  Streak columns: {streak_features.columns.tolist()}")
    assert set(streak_features.columns) == {
        "inst_longest_dpd_streak",
        "inst_months_since_last_dpd",
        "inst_payment_amt_slope",
        "inst_payment_ratio_trend",
        "inst_recent_vs_historical_dpd",
    }, "Unexpected streak feature columns"
    print(f"✓ All 5 expected streak features present")

    # Step 9: Augment feature store with streak features
    print("\nStep 9: Augment feature store with streak features")
    X_raw_original = pd.read_parquet("data/processed/X_raw_features.parquet")
    X_augmented = X_raw_original.join(streak_features, how="left")
    print(f"✓ Augmented shape: {X_augmented.shape}")
    print(f"  Columns increased: {X_raw_original.shape[1]} → {X_augmented.shape[1]}")

    # Verify all 5 streak columns present
    new_streak_cols = [c for c in X_augmented.columns if c in streak_features.columns]
    assert len(new_streak_cols) == 5, (
        f"Expected 5 new streak columns, got {len(new_streak_cols)}"
    )
    print(f"✓ All 5 new streak columns present")

    # Verify no NaN introduced by join
    n_nan_new = X_augmented[new_streak_cols].isna().sum().sum()
    print(f"  NaN in new streak columns: {n_nan_new}")

    # Save augmented store to temporary path
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    X_augmented.to_parquet("data/processed/X_raw_features_with_streaks.parquet")
    print(f"✓ Augmented feature store saved (temporary): {X_augmented.shape}")

    # ========================================================================
    # TASK 3: Train XGBoost with full feature set and measure Gini delta
    # ========================================================================
    print("\n" + "=" * 80)
    print("TASK 3: Train XGBoost with full feature set and measure delta")
    print("=" * 80)

    # Step 10: Train XGBoost on augmented feature set WITH 5 new streak features
    print("\nStep 10: Create 80/20 split on augmented feature set (random_state=42)")
    X_raw_with_streaks = pd.read_parquet(
        "data/processed/X_raw_features_with_streaks.parquet"
    )
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X_raw_with_streaks, y_train, test_size=_TEST_SIZE, random_state=_RANDOM_STATE,
        stratify=y_train
    )
    print(f"✓ Full-feature train shape: {X_train_full.shape}")
    print(f"  Full-feature test shape: {X_test_full.shape}")

    print(f"\nStep 11: Train full LightGBM (Optuna n_trials={_N_TRIALS})")
    full_model, metrics_full, _, _, best_params_full = train_lightgbm_optuna(
        X_train_full, y_train_full, n_trials=_N_TRIALS
    )
    print(f"✓ Full model trained")

    # Measure Gini with streaks
    print("\nStep 12: Measure full Gini")
    y_pred_full = full_model.predict_proba(X_test_full)[:, 1]
    gini_full = gini_coefficient(y_test_full, y_pred_full)
    print(f"✓ Full Gini (with 5 new streaks): {gini_full:.4f}")

    # Step 13: Calculate Gini delta and make commit/revert decision (per D-06)
    print("\nStep 13: Calculate Gini delta and commit decision")
    gini_delta = gini_full - gini_baseline
    print(f"  Baseline Gini: {gini_baseline:.4f}")
    print(f"  Full Gini:     {gini_full:.4f}")
    print(f"  Gini delta:    {gini_delta:+.4f}")

    if gini_delta >= _COMMIT_THRESHOLD:
        decision = "commit"
        print(f"✓ Delta {gini_delta:+.4f} >= {_COMMIT_THRESHOLD:.4f} → COMMIT streak features")
    else:
        decision = "revert"
        print(f"✗ Delta {gini_delta:+.4f} < {_COMMIT_THRESHOLD:.4f} → REVERT streak features")

    # Step 14: Save full model and results
    print("\nStep 14: Save full model and evaluation results")
    joblib.dump(full_model, "models/xgboost_raw_with_streaks.pkl")
    results = {
        "baseline_gini": float(gini_baseline),
        "baseline_n_features": X_no_streaks.shape[1],
        "full_gini": float(gini_full),
        "full_n_features": X_raw_with_streaks.shape[1],
        "gini_delta": float(gini_delta),
        "commit_threshold": float(_COMMIT_THRESHOLD),
        "decision": decision,
        "new_streak_features": [
            "inst_longest_dpd_streak",
            "inst_months_since_last_dpd",
            "inst_payment_amt_slope",
            "inst_payment_ratio_trend",
            "inst_recent_vs_historical_dpd",
        ],
    }

    with open("reports/streak_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Full model saved to models/xgboost_raw_with_streaks.pkl")
    print(f"✓ Results saved to reports/streak_evaluation_results.json")

    # Step 15: Execute commit or revert (per D-06 + immutability)
    print("\nStep 15: Execute commit or revert decision")
    if decision == "commit":
        shutil.move(
            "data/processed/X_raw_features_with_streaks.parquet",
            "data/processed/X_raw_features.parquet",
        )
        print(
            f"✓ Feature store updated: {X_no_streaks.shape[1]} + 5 = "
            f"{X_augmented.shape[1]} columns"
        )
    else:
        Path("data/processed/X_raw_features_with_streaks.parquet").unlink()
        print(
            f"✗ Feature store unchanged: streak features do not meet "
            f"+{_COMMIT_THRESHOLD:.2f} Gini threshold"
        )

    # ========================================================================
    # Final Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 1 PLAN 02 (WAVE 2) SUMMARY")
    print("=" * 80)
    print(f"Baseline Gini (no streaks):  {gini_baseline:.4f}")
    print(f"Full Gini (with 5 streaks):  {gini_full:.4f}")
    print(f"Gini delta:                  {gini_delta:+.4f}")
    print(f"Commit threshold:            {_COMMIT_THRESHOLD:.4f}")
    print(f"Decision:                    {decision.upper()}")
    print(f"\nFinal parquet state:         {X_augmented.shape if decision == 'commit' else X_raw_original.shape}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    # Run the test when executed directly
    test_streak_feature_evaluation()
