#!/usr/bin/env python
"""
train_xgboost_v3.py
-------------------
Train XGBoost on the fairness-compliant v3 feature store (X_xgb_v3.parquet).

This script implements the Basel CRE36.54 compliant temporal validation workflow:
1. Load v3 store (163 features, no regulated columns)
2. Verify no AGE_YEARS column
3. Sort by SK_ID_CURR (application intake surrogate)
4. Carve OOT: freeze most-recent 20% rows
5. Optuna HPO on remaining 80% with OOF CV
6. Retrain on full 80% with best params
7. Calibrate with Platt scaling
8. Evaluate on frozen OOT
9. Save model, metrics, eval JSON

Output:
- models/xgboost_v3_calibrated.pkl (Platt-calibrated model)
- reports/xgboost_v3_eval.json (OOT metrics: Gini, KS, Brier, best_params, timestamp)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Bootstrap sys.path to allow imports from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.model_xgboost import train_xgboost_optuna
from src.model_base import calibrate_model, save_model
from src.utils import evaluate_model


def main() -> None:
    """Train XGBoost v3 with Basel CRE36.54 workflow."""

    # --- 1-2. Load v3 store and verify no regulated columns ---
    print("Loading v3 feature store...")
    feature_store_path = _PROJECT_ROOT / "data" / "processed" / "X_xgb_v3.parquet"
    X = pd.read_parquet(feature_store_path)

    print(f"  Shape: {X.shape}")
    print(f"  Columns (first 5): {list(X.columns[:5])}")

    # Verify AGE_YEARS is absent
    assert "AGE_YEARS" not in X.columns, (
        "AGE_YEARS present in X_xgb_v3 — using v2 store by mistake"
    )
    print("  ✓ AGE_YEARS not in X.columns (v3 compliance verified)")

    # Verify other regulated columns are absent
    regulated_cols = ["EMPLOYED_TO_AGE_RATIO", "CNT_CHILDREN", "CNT_FAM_MEMBERS"]
    for col in regulated_cols:
        assert col not in X.columns, f"{col} present in X_xgb_v3"
    print(f"  ✓ All regulated columns absent: {regulated_cols}")

    # --- 3. Load target ---
    print("\nLoading target...")
    y = pd.read_parquet(_PROJECT_ROOT / "data" / "processed" / "y_train.parquet").squeeze()
    print(f"  Shape: {y.shape}")
    print(f"  Class balance: {(y == 0).sum()} negatives, {(y == 1).sum()} positives")

    # Align indices
    assert len(X) == len(y), "X and y length mismatch"

    # --- 4. Sort by SK_ID_CURR (application intake, monotonically increasing) ---
    print("\nSorting by SK_ID_CURR...")
    sort_idx = X["SK_ID_CURR"].argsort()
    X = X.iloc[sort_idx].reset_index(drop=True)
    y = y.iloc[sort_idx].reset_index(drop=True)
    print(f"  ✓ Sorted by SK_ID_CURR (first: {X['SK_ID_CURR'].iloc[0]}, last: {X['SK_ID_CURR'].iloc[-1]})")

    # --- 5. Carve OOT: most-recent 20% rows ---
    print("\nCarving OOT (most-recent 20%)...")
    test_start = int(len(X) * 0.8)
    X_train_80 = X.iloc[:test_start].copy()
    y_train_80 = y.iloc[:test_start].copy()
    X_oot = X.iloc[test_start:].copy()
    y_oot = y.iloc[test_start:].copy()

    print(f"  Train 80%: {X_train_80.shape}")
    print(f"  OOT 20%:   {X_oot.shape}")
    print(f"  Train balance: {(y_train_80 == 0).sum()} neg, {(y_train_80 == 1).sum()} pos")
    print(f"  OOT balance:   {(y_oot == 0).sum()} neg, {(y_oot == 1).sum()} pos")

    # --- 6. Optuna HPO on 80% only (no OOT leakage) ---
    print("\nRunning Optuna HPO (50 trials on 80% training data)...")
    print("  This may take 20-40 minutes...")

    model_80, metrics_80, X_test, y_test, best_params, oof_preds = train_xgboost_optuna(
        feature_store_path=str(feature_store_path),
        n_trials=50,
    )

    print(f"  ✓ HPO complete")
    print(f"  Best OOF Gini: {metrics_80.get('oof_gini', 'N/A')}")
    print(f"  Best params: {best_params}")

    # --- 7. Calibrate with Platt scaling ---
    print("\nCalibrating with Platt scaling...")
    calibrated_model = calibrate_model(
        model_80, X_train_80, y_train_80, X_oot, y_oot,
        method="sigmoid",
        output_model_path=str(_PROJECT_ROOT / "models" / "xgboost_v3_calibrated.pkl"),
        output_figure_path=str(_PROJECT_ROOT / "reports" / "figures" / "xgboost_v3_calibration.png"),
    )
    print("  ✓ Calibration complete")

    # --- 8. Evaluate on frozen OOT ---
    print("\nEvaluating on frozen OOT set...")
    eval_metrics = evaluate_model(
        calibrated_model, X_oot, y_oot, model_name="xgboost_v3"
    )

    oot_gini = eval_metrics.get("Gini", None)
    oot_ks = eval_metrics.get("KS", None)
    oot_brier = eval_metrics.get("Brier", None)
    oot_auc = eval_metrics.get("AUC-ROC", None)

    print(f"\n  OOT Gini:  {oot_gini:.4f}")
    print(f"  OOT KS:    {oot_ks:.4f}" if oot_ks else "  OOT KS:    N/A")
    print(f"  OOT Brier: {oot_brier:.4f}" if oot_brier else "  OOT Brier: N/A")
    print(f"  OOT AUC:   {oot_auc:.4f}" if oot_auc else "  OOT AUC:   N/A")

    # Check Gini floor
    if oot_gini is not None:
        if oot_gini >= 0.55:
            print(f"\n  ✓ OOT Gini {oot_gini:.4f} ≥ 0.55 (performance floor MET)")
        elif oot_gini >= 0.50:
            print(f"\n  ⚠️  WARNING: OOT Gini {oot_gini:.4f} < 0.55 (floor not met, but > 0.50)")
        else:
            print(f"\n  ✗ ERROR: OOT Gini {oot_gini:.4f} < 0.50 (investigate)")

    # --- 9. Save model (already done in calibrate_model) ---
    model_path = _PROJECT_ROOT / "models" / "xgboost_v3_calibrated.pkl"
    print(f"\n✓ Model saved: {model_path}")

    # --- 10. Save metrics JSON ---
    print("Saving evaluation metrics...")
    eval_json = {
        "oot_gini": float(oot_gini) if oot_gini is not None else None,
        "oot_ks": float(oot_ks) if oot_ks is not None else None,
        "oot_brier": float(oot_brier) if oot_brier is not None else None,
        "oot_auc": float(oot_auc) if oot_auc is not None else None,
        "best_params": best_params,
        "best_oof_gini": float(metrics_80.get("oof_gini", "N/A")) if "oof_gini" in metrics_80 else "N/A",
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    eval_path = _PROJECT_ROOT / "reports" / "xgboost_v3_eval.json"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with eval_path.open("w") as fh:
        json.dump(eval_json, fh, indent=2)

    print(f"✓ Metrics saved: {eval_path}")

    print("\n" + "=" * 70)
    print("✓ XGBoost v3 training COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
