#!/usr/bin/env python
"""
train_catboost_v3.py
--------------------
Train CatBoost on the fairness-compliant v3 feature store (X_cat_v3.parquet).

Uses best hyperparameters from the catboost_raw_v3 Optuna study
(51 completed trials, OOF Gini 0.5532, trial 36) — no re-HPO needed since v3
removes only 4 of 167 features (~2.4% change), making the v2-optimal params valid.

Basel CRE36.54 compliant temporal validation workflow:
1. Load v3 store (167 features — no regulated columns)
2. Verify AGE_YEARS and other regulated columns are absent
3. Extract TARGET label from store (pop to prevent leakage)
4. Sort by SK_ID_CURR (monotonically increasing application intake ID)
5. Carve OOT: freeze most-recent 20% rows
6. Single fit on 80% with best params + auto_class_weights=Balanced
7. Calibrate with Platt scaling
8. Evaluate on frozen OOT → OOT Gini is the regulatory metric
9. Save model and eval JSON

Output:
- models/catboost_v3_calibrated.pkl   (Platt-calibrated CatBoostClassifier)
- reports/catboost_v3_eval.json       (OOT metrics + params + timestamp)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
from catboost import CatBoostClassifier

from src.model_base import (
    _CAT_ITERATIONS,
    _RANDOM_STATE,
    _TEMPORAL_SORT_COL,
    _TEST_SIZE,
    calibrate_model,
    save_model,
)
from src.utils import evaluate_model

# Best hyperparameters from catboost_raw_v3 Optuna study
# (trial 36, OOF Gini=0.5532, 51 trials)
# v3 removes 4 of 167 features — param transfer is sound.
# bootstrap_type="Bayesian" is required for bagging_temperature to be valid.
_BEST_PARAMS: dict = {
    "depth": 10,
    "learning_rate": 0.00831889320253293,
    "l2_leaf_reg": 22.046459293778387,
    "min_data_in_leaf": 5,
    "bagging_temperature": 1.0436052172524026,
    "random_strength": 0.00026211230471279157,
    "grow_policy": "Lossguide",
}

_REGULATED_COLS: list[str] = [
    "AGE_YEARS",
    "EMPLOYED_TO_AGE_RATIO",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
]

_FEATURE_STORE_PATH = _PROJECT_ROOT / "data" / "processed" / "X_cat_v3.parquet"
_MODEL_OUTPUT_PATH = _PROJECT_ROOT / "models" / "catboost_v3_calibrated.pkl"
_EVAL_OUTPUT_PATH = _PROJECT_ROOT / "reports" / "catboost_v3_eval.json"
_FIGURE_OUTPUT_PATH = _PROJECT_ROOT / "reports" / "figures" / "catboost_v3_calibration.png"


def main() -> None:
    """Train CatBoost v3 with Basel CRE36.54 temporal validation workflow."""

    # 1-2. Load v3 store and verify compliance
    print("Loading v3 feature store...")
    X = pd.read_parquet(_FEATURE_STORE_PATH)
    print(f"  Shape: {X.shape}")

    for col in _REGULATED_COLS:
        assert col not in X.columns, (
            f"{col} present in X_cat_v3 — using wrong store"
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
    # Identify categorical (string/object) feature indices for CatBoost
    cat_feature_names = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_feature_indices = [X_train.columns.get_loc(c) for c in cat_feature_names]
    print(f"  Categorical features: {cat_feature_names}")

    print("\nTraining CatBoost v3 on 80% training set...")
    model = CatBoostClassifier(
        **_BEST_PARAMS,
        iterations=_CAT_ITERATIONS,
        bootstrap_type="Bayesian",  # required for bagging_temperature
        auto_class_weights="Balanced",
        eval_metric="AUC",
        random_seed=_RANDOM_STATE,
        thread_count=-1,
        verbose=False,
    )
    model.fit(X_train, y_train, cat_features=cat_feature_indices)
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
    eval_metrics = evaluate_model(calibrated_model, X_oot, y_oot, model_name="catboost_v3")

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
        "model": "catboost_v3",
        "feature_store": "X_cat_v3.parquet",
        "feature_count": X_train.shape[1],
        "regulated_cols_removed": _REGULATED_COLS,
        "oot_gini": float(oot_gini),
        "oot_ks": float(oot_ks),
        "oot_brier": float(oot_brier),
        "oot_auc": float(oot_auc),
        "best_params": _BEST_PARAMS,
        "params_source": "catboost_raw_v3 Optuna study (trial 36, OOF Gini=0.5532, 51 trials)",
        "iterations": _CAT_ITERATIONS,
        "auto_class_weights": "Balanced",
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
    print("CatBoost v3 training COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
