#!/usr/bin/env python3
"""
Wave 2 Tasks 2-3: Augment feature store with OLS streak features and evaluate.

Task 1 (baseline Gini = 0.5548, 62 features) already complete.
This script picks up at Task 2:
  - Load installments_payments.csv directly (not via load_data — avoids loading all 7 tables)
  - Call engineer_instalment_streaks()
  - Join onto X_raw_features.parquet (drop existing streak cols first)
  - Train XGBoost on augmented set (n_trials=50)
  - Measure delta vs baseline; commit if >= +0.01, else revert

Imports use sys.path + src.* (not credit_engine alias — that's pytest-only via conftest.py)
"""

import json
import os
import shutil
import sys
import warnings
from pathlib import Path

sys.path.insert(0, ".")

# Replicate conftest.py alias: src/ module is imported as both src and credit_engine
import src  # noqa: E402
sys.modules["credit_engine"] = src

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.features import engineer_instalment_streaks
from src.model import train_xgboost_optuna
from src.utils import gini_coefficient

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_COMMIT_THRESHOLD: float = 0.01
_RANDOM_STATE: int = 42
_TEST_SIZE: float = 0.2
_N_TRIALS: int = 50

RAW_FEATURES_PATH = "data/processed/X_raw_features.parquet"
Y_TRAIN_PATH = "data/processed/y_train.parquet"
INST_CSV = "data/installments_payments.csv"
AUGMENTED_PATH = "data/processed/X_raw_features_with_streaks.parquet"
BASELINE_JSON = "reports/streak_evaluation_baseline.json"
RESULTS_JSON = "reports/streak_evaluation_results.json"

# Baseline dropped cols matching "streak" in name (63 - 1 = 62 features)
# Must match baseline drop logic exactly for a fair comparison
EXISTING_STREAK_COLS = ["inst_max_consec_late_streak"]

NEW_STREAK_COLS = [
    "inst_longest_dpd_streak",
    "inst_months_since_last_dpd",
    "inst_payment_amt_slope",
    "inst_payment_ratio_trend",
    "inst_recent_vs_historical_dpd",
]

FILL_MAP = {
    "inst_longest_dpd_streak": 0.0,
    "inst_months_since_last_dpd": 999.0,
    "inst_payment_amt_slope": 0.0,
    "inst_payment_ratio_trend": 0.0,
    "inst_recent_vs_historical_dpd": 0.0,
}


def load_baseline_gini() -> float:
    with open(BASELINE_JSON) as f:
        data = json.load(f)
    gini = data["gini"]
    n_feat = data["n_features"]
    print(f"  Baseline Gini = {gini:.6f} ({n_feat} features)")
    return gini


def build_augmented_parquet() -> pd.DataFrame:
    """Drop existing streak cols, add 5 new OLS streak features, save temp parquet."""
    print("Loading X_raw_features.parquet...")
    X = pd.read_parquet(RAW_FEATURES_PATH)
    assert X.shape[0] >= 100_000, f"Corrupt feature store: {X.shape[0]} rows"
    print(f"  Loaded {X.shape}")

    # Drop existing streak cols (same isolation as baseline)
    cols_to_drop = [c for c in EXISTING_STREAK_COLS if c in X.columns]
    X = X.drop(columns=cols_to_drop)
    print(f"  After dropping existing streak cols: {X.shape} (dropped: {cols_to_drop})")

    print("Loading installments_payments.csv...")
    df_inst = pd.read_csv(INST_CSV)
    print(f"  Loaded {df_inst.shape}")

    print("Computing 5 OLS streak features...")
    streaks = engineer_instalment_streaks(df_inst)
    print(f"  Streak features: {list(streaks.columns)} — shape {streaks.shape}")

    # Left join: preserves all 307511 loans (missing -> fill)
    X = X.join(streaks, how="left")
    for col, fill_val in FILL_MAP.items():
        if col in X.columns:
            X[col] = X[col].fillna(fill_val)

    present = [c for c in NEW_STREAK_COLS if c in X.columns]
    print(f"  Augmented shape: {X.shape} ({len(present)}/5 new streak cols)")
    assert len(present) == 5, f"Expected 5 streak cols, got {len(present)}: {present}"

    X.to_parquet(AUGMENTED_PATH, index=True)
    print(f"  Saved temporary augmented parquet: {AUGMENTED_PATH}")
    return X


def train_and_evaluate(X: pd.DataFrame) -> float:
    print("Loading y_train...")
    y = pd.read_parquet(Y_TRAIN_PATH).squeeze()

    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    print(f"  Aligned: X={X.shape}, y={y.shape}, default_rate={y.mean():.3%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=_TEST_SIZE, random_state=_RANDOM_STATE, stratify=y
    )
    print(f"  Train {X_train.shape} | Test {X_test.shape}")

    print(f"Training XGBoost with Optuna (n_trials={_N_TRIALS})...")
    model, metrics, _, _, best_params = train_xgboost_optuna(
        X_train, y_train, n_trials=_N_TRIALS
    )
    joblib.dump(model, "models/xgboost_raw_with_streaks.pkl")
    print(f"  Model saved: models/xgboost_raw_with_streaks.pkl")
    print(f"  Best params: {best_params}")

    y_prob = model.predict_proba(X_test)[:, 1]
    gini = gini_coefficient(y_test.values, y_prob)
    print(f"  Augmented Gini: {gini:.6f}")
    return gini


def execute_decision(gini_full: float, baseline_gini: float, X_aug: pd.DataFrame) -> dict:
    delta = gini_full - baseline_gini
    decision = "commit" if delta >= _COMMIT_THRESHOLD else "revert"

    print("\n" + "=" * 50)
    print(f"  Baseline Gini : {baseline_gini:.6f}")
    print(f"  Full Gini     : {gini_full:.6f}")
    print(f"  Delta         : {delta:+.6f}")
    print(f"  Threshold     : +{_COMMIT_THRESHOLD:.2f}")
    print(f"  Decision      : {decision.upper()}")
    print("=" * 50)

    results = {
        "baseline_gini": float(baseline_gini),
        "full_gini": float(gini_full),
        "delta": float(delta),
        "threshold": float(_COMMIT_THRESHOLD),
        "decision": decision,
        "new_streak_cols": NEW_STREAK_COLS,
        "augmented_shape": list(X_aug.shape),
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {RESULTS_JSON}")

    if decision == "commit":
        shutil.move(AUGMENTED_PATH, RAW_FEATURES_PATH)
        print(f"  COMMITTED: augmented parquet -> {RAW_FEATURES_PATH}")
        print(f"  Final shape: {X_aug.shape}")
    else:
        if os.path.exists(AUGMENTED_PATH):
            os.remove(AUGMENTED_PATH)
        print(f"  REVERTED: original parquet unchanged (63 cols)")

    return results


def main():
    print("=" * 60)
    print("Wave 2 Tasks 2-3: Streak Feature Augment & Evaluate")
    print("=" * 60)

    print("\n[Task 2] Loading baseline...")
    baseline_gini = load_baseline_gini()

    print("\n[Task 2] Building augmented feature store...")
    X_aug = build_augmented_parquet()

    print("\n[Task 3] Training and evaluating...")
    gini_full = train_and_evaluate(X_aug)

    print("\n[Task 3] Decision...")
    results = execute_decision(gini_full, baseline_gini, X_aug)

    print("\n[DONE] Streak evaluation complete.")
    return results


if __name__ == "__main__":
    main()
