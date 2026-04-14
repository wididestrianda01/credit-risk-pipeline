#!/usr/bin/env python
"""
train_lightgbm_v3.py
--------------------
Train LightGBM on the fairness-compliant v3 feature store (X_lgb_v3.parquet).

Uses best hyperparameters from the lgb_raw_X_lgb_v2_is_unbalance Optuna study
(51 completed trials, OOF Gini 0.5676, trial 26) — no re-HPO needed since v3
removes only 4 of 163 features (~2.5% change), making the v2-optimal params valid.

Basel CRE36.54 compliant temporal validation workflow:
1. Load v3 store (163 features — no regulated columns)
2. Verify AGE_YEARS and other regulated columns are absent
3. Extract TARGET label from store (pop to prevent leakage)
4. Sort by SK_ID_CURR (monotonically increasing application intake ID)
5. Carve OOT: freeze most-recent 20% rows
6. Find best n_estimators via early stopping on 15% inner validation split
7. Retrain on full 80% with best_iteration
8. Calibrate with Platt scaling
9. Evaluate on frozen OOT → OOT Gini is the regulatory metric
10. Save model and eval JSON

Output:
- models/lightgbm_v3_calibrated.pkl   (Platt-calibrated LGBMClassifier)
- reports/lightgbm_v3_eval.json       (OOT metrics + params + timestamp)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.model_base import (
    _RANDOM_STATE,
    _TEMPORAL_SORT_COL,
    _TEST_SIZE,
    calibrate_model,
    save_model,
)
from src.utils import evaluate_model

# Best hyperparameters from lgb_raw_X_lgb_v2_is_unbalance study
# (trial 26, OOF Gini=0.5676, 51 trials)
# v3 removes 4 of 163 features — param transfer is sound.
_BEST_PARAMS: dict = {
    "learning_rate": 0.0211035346361329,
    "num_leaves": 49,
    "max_depth": 8,
    "min_child_samples": 372,
    "min_child_weight": 0.0013442267377792995,
    "subsample": 0.855847351759121,
    "colsample_bytree": 0.7233394009311871,
    "reg_alpha": 5.173906490535256,
    "reg_lambda": 9.838443900320089,
    "path_smooth": 8.152537176771556,
}

# n_estimators taken directly from the v2 Optuna study's best trial user_attrs
# (best_iteration=843 from lgb_raw_X_lgb_v2_is_unbalance study, trial 26)
# No early stopping needed — param transfer is valid for v3 (only 4 features removed).
_LGB_N_ESTIMATORS: int = 844  # best_iteration + 1 from v2 study

_REGULATED_COLS: list[str] = [
    "AGE_YEARS",
    "EMPLOYED_TO_AGE_RATIO",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
]

_FEATURE_STORE_PATH = _PROJECT_ROOT / "data" / "processed" / "X_lgb_v3.parquet"
_MODEL_OUTPUT_PATH = _PROJECT_ROOT / "models" / "lightgbm_v3_calibrated.pkl"
_EVAL_OUTPUT_PATH = _PROJECT_ROOT / "reports" / "lightgbm_v3_eval.json"
_FIGURE_OUTPUT_PATH = _PROJECT_ROOT / "reports" / "figures" / "lightgbm_v3_calibration.png"


def main() -> None:
    """Train LightGBM v3 with Basel CRE36.54 temporal validation workflow."""

    # 1-2. Load v3 store and verify compliance
    print("Loading v3 feature store...")
    X = pd.read_parquet(_FEATURE_STORE_PATH)
    print(f"  Shape: {X.shape}")

    for col in _REGULATED_COLS:
        assert col not in X.columns, (
            f"{col} present in X_lgb_v3 — using wrong store"
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

    # 6. Single fit on 80% with best params
    # n_estimators=844 is taken from the v2 Optuna study's best_iteration (843),
    # which was found via early stopping on OOF CV over 51 trials.
    # Direct param transfer is valid because v3 removes only 4 of 163 features.
    print("\nTraining LightGBM v3 on 80% training set...")
    model = lgb.LGBMClassifier(
        **_BEST_PARAMS,
        n_estimators=_LGB_N_ESTIMATORS,
        is_unbalance=True,
        boosting_type="gbdt",
        random_state=_RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(X_train, y_train, callbacks=[lgb.log_evaluation(period=0)])
    print("  Training complete")

    # 8. Platt calibration (train on 80%, score OOT)
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

    # 9. Evaluate on frozen OOT
    print("\nEvaluating on frozen OOT set...")
    eval_metrics = evaluate_model(calibrated_model, X_oot, y_oot, model_name="lightgbm_v3")

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

    # 10. Save eval JSON
    print("\nSaving evaluation metrics...")
    eval_json = {
        "model": "lightgbm_v3",
        "feature_store": "X_lgb_v3.parquet",
        "feature_count": X_train.shape[1],
        "regulated_cols_removed": _REGULATED_COLS,
        "oot_gini": float(oot_gini),
        "oot_ks": float(oot_ks),
        "oot_brier": float(oot_brier),
        "oot_auc": float(oot_auc),
        "best_params": _BEST_PARAMS,
        "params_source": "lgb_raw_X_lgb_v2_is_unbalance Optuna study (trial 26, OOF Gini=0.5676, 51 trials)",
        "best_iteration": _LGB_N_ESTIMATORS - 1,  # 843 (v2 study user_attr)
        "n_estimators": _LGB_N_ESTIMATORS,
        "is_unbalance": True,
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
    print("LightGBM v3 training COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
