#!/usr/bin/env python
"""
train_xgboost_v3.py
-------------------
Train XGBoost on the fairness-compliant v3 feature store (X_xgb_v3.parquet).

Uses the best hyperparameters from the existing xgboost_raw_v9 Optuna study
(104 completed trials, OOF Gini 0.5434) — no re-HPO needed since v3 removes
only 4 of 163 features (~2.5% change), making the v2-optimal params valid.

Basel CRE36.54 compliant temporal validation workflow:
1. Load v3 store (163 features — no regulated columns)
2. Verify AGE_YEARS and other regulated columns are absent
3. Sort by SK_ID_CURR (monotonically increasing application intake ID)
4. Carve OOT: freeze most-recent 20% rows (never touched during training)
5. Train on 80% with best v2 params (single fit — no data leakage)
6. Calibrate with Platt scaling on 80% train / 20% OOT
7. Evaluate on frozen OOT → OOT Gini is the regulatory metric
8. Save model and eval JSON

Output:
- models/xgboost_v3_calibrated.pkl   (Platt-calibrated XGBoostClassifier)
- reports/xgboost_v3_eval.json       (OOT metrics + params + timestamp)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb

from src.model_base import (
    _RANDOM_STATE,
    _TEMPORAL_SORT_COL,
    _TEST_SIZE,
    _XGB_N_ESTIMATORS,
    calibrate_model,
    save_model,
)
from src.utils import evaluate_model

# Best hyperparameters from xgboost_raw_v9 study (trial 47, OOF Gini=0.5434)
# These were optimised over 104 Optuna trials on the v2 feature store.
# v3 removes 4 of 163 features — param transfer is sound.
_BEST_PARAMS: dict = {
    "colsample_bytree": 0.7595552940919579,
    "gamma": 1.56955566496891,
    "learning_rate": 0.03617558383839258,
    "max_depth": 3,
    "min_child_weight": 4.804611119097033,
    "reg_alpha": 1.710682197819308,
    "reg_lambda": 2.581717713228335e-08,
    "subsample": 0.6050648219798987,
}

_REGULATED_COLS: list[str] = [
    "AGE_YEARS",
    "EMPLOYED_TO_AGE_RATIO",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
]

_FEATURE_STORE_PATH = _PROJECT_ROOT / "data" / "processed" / "X_xgb_v3.parquet"
_MODEL_OUTPUT_PATH = _PROJECT_ROOT / "models" / "xgboost_v3_calibrated.pkl"
_EVAL_OUTPUT_PATH = _PROJECT_ROOT / "reports" / "xgboost_v3_eval.json"
_FIGURE_OUTPUT_PATH = _PROJECT_ROOT / "reports" / "figures" / "xgboost_v3_calibration.png"


def main() -> None:
    """Train XGBoost v3 with Basel CRE36.54 temporal validation workflow."""

    # 1-2. Load v3 store and verify compliance
    print("Loading v3 feature store...")
    X = pd.read_parquet(_FEATURE_STORE_PATH)
    print(f"  Shape: {X.shape}")

    for col in _REGULATED_COLS:
        assert col not in X.columns, (
            f"{col} present in X_xgb_v3 — using wrong store"
        )
    print(f"  Regulated columns absent: {_REGULATED_COLS}")

    # 3. Extract target — pop it so it never appears as a model feature
    if "TARGET" not in X.columns:
        raise ValueError("TARGET column missing from feature store — cannot extract labels")
    y = X.pop("TARGET")
    print(f"  Target extracted from store: {y.mean():.4f} default rate")

    # 4. Sort by SK_ID_CURR (temporal sort column)
    sort_idx = X[_TEMPORAL_SORT_COL].argsort()
    X = X.iloc[sort_idx].reset_index(drop=True)
    y = y.iloc[sort_idx].reset_index(drop=True)
    print(f"  Sorted by {_TEMPORAL_SORT_COL}")

    # 5. Carve OOT — freeze most-recent 20%
    test_start = int(len(X) * (1.0 - _TEST_SIZE))
    X_train = X.iloc[:test_start].copy()
    y_train = y.iloc[:test_start].copy()
    X_oot = X.iloc[test_start:].copy()
    y_oot = y.iloc[test_start:].copy()

    # Drop the sort key — SK_ID_CURR is an application ID, not a feature
    for split in (X_train, X_oot):
        if _TEMPORAL_SORT_COL in split.columns:
            split.drop(columns=[_TEMPORAL_SORT_COL], inplace=True)

    print(f"  Train 80%: {X_train.shape}  |  OOT 20%: {X_oot.shape}")

    # Compute scale_pos_weight from training set only
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

    # 6. Single fit on 80% with best params
    print("\nTraining XGBoost v3 on 80% training set...")
    model = xgb.XGBClassifier(
        n_estimators=_XGB_N_ESTIMATORS,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        use_label_encoder=False,
        random_state=_RANDOM_STATE,
        n_jobs=-1,
        **_BEST_PARAMS,
    )
    model.fit(X_train, y_train, verbose=False)
    print("  Training complete")

    # 7. Platt calibration (train on 80%, score OOT)
    print("\nCalibrating with Platt scaling...")
    _FIGURE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    calibrated_model, cal_brier, cal_auc = calibrate_model(
        model,
        X_train,
        y_train,
        X_oot,
        y_oot,
        method="sigmoid",
        output_model_path=str(_MODEL_OUTPUT_PATH),
        output_figure_path=str(_FIGURE_OUTPUT_PATH),
    )
    print(f"  Calibration Brier: {cal_brier:.4f}  AUC: {cal_auc:.4f}")

    # 8. Evaluate on frozen OOT
    print("\nEvaluating on frozen OOT set...")
    eval_metrics = evaluate_model(calibrated_model, X_oot, y_oot, model_name="xgboost_v3")

    oot_gini = eval_metrics.get("Gini")
    oot_ks = eval_metrics.get("KS")
    oot_brier = eval_metrics.get("Brier")
    oot_auc = eval_metrics.get("AUC-ROC")

    print(f"  OOT Gini:  {oot_gini:.4f}")
    print(f"  OOT KS:    {oot_ks:.4f}")
    print(f"  OOT Brier: {oot_brier:.4f}")
    print(f"  OOT AUC:   {oot_auc:.4f}")

    if oot_gini >= 0.55:
        print(f"\n  OOT Gini {oot_gini:.4f} >= 0.55 — performance floor MET")
    elif oot_gini >= 0.50:
        print(f"\n  WARNING: OOT Gini {oot_gini:.4f} < 0.55 — floor not met, but > 0.50")
    else:
        print(f"\n  ERROR: OOT Gini {oot_gini:.4f} < 0.50 — investigate")

    # 9. Save eval JSON
    print("\nSaving evaluation metrics...")
    eval_json = {
        "model": "xgboost_v3",
        "feature_store": "X_xgb_v3.parquet",
        "feature_count": X_train.shape[1],  # after dropping TARGET + SK_ID_CURR
        "regulated_cols_removed": _REGULATED_COLS,
        "oot_gini": float(oot_gini),
        "oot_ks": float(oot_ks),
        "oot_brier": float(oot_brier),
        "oot_auc": float(oot_auc),
        "best_params": _BEST_PARAMS,
        "params_source": "xgboost_raw_v9 Optuna study (trial 47, OOF Gini=0.5434, 104 trials)",
        "n_estimators": _XGB_N_ESTIMATORS,
        "scale_pos_weight": scale_pos_weight,
        "temporal_sort_col": _TEMPORAL_SORT_COL,
        "oot_fraction": _TEST_SIZE,
        "train_rows": X_train.shape[0],
        "oot_rows": X_oot.shape[0],
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    _EVAL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _EVAL_OUTPUT_PATH.open("w") as fh:
        json.dump(eval_json, fh, indent=2)
    print(f"  Saved: {_EVAL_OUTPUT_PATH}")
    print(f"  Model: {_MODEL_OUTPUT_PATH}")

    print("\n" + "=" * 70)
    print("XGBoost v3 training COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
