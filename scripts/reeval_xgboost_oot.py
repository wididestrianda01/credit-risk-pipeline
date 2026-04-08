#!/usr/bin/env python3
"""
Re-evaluate XGBoost on the corrected OOT split (NaN-fill fix).

Skips HPO entirely — loads best params from the existing Optuna study,
applies the fixed temporal split (NaN → nanmin-1 sentinel), re-runs the
5-fold OOF accumulation loop, refits a fresh model on the corrected training
pool, and scores the true temporal holdout.

Usage:
    python scripts/reeval_xgboost_oot.py

Writes:
    reports/xgboost_raw_eval.json  (updated oof_gini, oot_gini, gap)
    models/xgboost_raw_best.pkl    (refitted on corrected training pool)
    models/xgboost_raw_calibrated.pkl
"""
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
import src as _src  # noqa: E402
if "credit_engine" not in sys.modules:
    sys.modules["credit_engine"] = _src

from credit_engine.utils import gini_coefficient, evaluate_model  # noqa: E402
from credit_engine.model import (  # noqa: E402
    calibrate_model,
    _TEMPORAL_SORT_COL,
    _TEST_SIZE,
    _RANDOM_STATE,
    _XGB_CV_N_SPLITS,
    _XGB_RAW_STUDY_NAME,
    _XGB_RAW_N_ESTIMATORS_MAX,
    _make_cv,
)

FEATURE_STORE = _ROOT / "data" / "processed" / "X_tree_dfs.parquet"
OPTUNA_DB = f"sqlite:///{_ROOT}/models/optuna_studies.db"
PARAMS_JSON = _ROOT / "models" / "xgboost_raw_params.json"
EVAL_JSON = _ROOT / "reports" / "xgboost_raw_eval.json"


def corrected_oot_split(X: pd.DataFrame, y: pd.Series):
    """Apply the NaN-aware temporal OOT split."""
    temporal_sort_values = X[_TEMPORAL_SORT_COL].values
    # NaN = no previous applications — assign to training (oldest position)
    _nan_fill = float(np.nanmin(temporal_sort_values)) - 1.0
    temporal_sort_values_filled = np.where(
        np.isnan(temporal_sort_values), _nan_fill, temporal_sort_values
    )
    temporal_indices = np.argsort(temporal_sort_values_filled)
    oot_threshold_idx = int(len(X) * (1 - _TEST_SIZE))

    oot_idx = temporal_indices[oot_threshold_idx:]
    train_idx = temporal_indices[:oot_threshold_idx]

    X_oot = X.iloc[oot_idx].copy()
    y_oot = y.iloc[oot_idx].copy()
    X_remaining = X.iloc[train_idx].copy()
    y_remaining = y.iloc[train_idx].copy()
    return X_oot, y_oot, X_remaining, y_remaining


def main():
    print(f"Loading feature store: {FEATURE_STORE}")
    df = pd.read_parquet(FEATURE_STORE)
    y = df["TARGET"].copy()
    X = df.drop(columns=["TARGET"])
    print(f"  Shape: {X.shape}, positives: {int(y.sum())}")

    nan_count = int(X[_TEMPORAL_SORT_COL].isna().sum())
    print(f"  NaN in '{_TEMPORAL_SORT_COL}': {nan_count:,} ({nan_count/len(X)*100:.1f}%)")

    # --- Corrected OOT split ---
    X_oot, y_oot, X_remaining, y_remaining = corrected_oot_split(X, y)
    print(f"  OOT set:     {len(X_oot):,} rows ({len(X_oot)/len(X)*100:.1f}%)")
    print(f"  Train pool:  {len(X_remaining):,} rows")

    # --- Train/test stratified split from remaining 80% ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_remaining, y_remaining,
        test_size=_TEST_SIZE, stratify=y_remaining, random_state=_RANDOM_STATE,
    )
    scale_pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())

    # --- Load best params from Optuna study ---
    print(f"\nLoading Optuna study '{_XGB_RAW_STUDY_NAME}'...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=_XGB_RAW_STUDY_NAME, storage=OPTUNA_DB)
    best_params = study.best_params
    best_iteration = study.best_trial.user_attrs.get(
        "best_iteration", _XGB_RAW_N_ESTIMATORS_MAX - 1
    )
    print(f"  Best trial: #{study.best_trial.number}  best_iteration={best_iteration}")
    print(f"  Best params: {best_params}")

    # --- CV for OOF accumulation ---
    # Fill NaN in temporal groups with (nanmin-1) so first-time-applicant rows
    # sort to "oldest" in _TemporalCV and stay in early training folds, not val.
    if _TEMPORAL_SORT_COL in X_train.columns:
        g_arr = X_train[_TEMPORAL_SORT_COL].to_numpy()
        _g_fill = float(np.nanmin(g_arr)) - 1.0 if not np.all(np.isnan(g_arr)) else 0.0
        groups_train = np.where(np.isnan(g_arr), _g_fill, g_arr)
    else:
        groups_train = None
    cv = _make_cv(groups_train, n_splits=_XGB_CV_N_SPLITS)

    fold_params = {
        **best_params,
        "n_estimators": best_iteration + 1,
        "tree_method": "hist",
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "auc",
        "use_label_encoder": False,
        "verbosity": 0,
        "random_state": _RANDOM_STATE,
    }

    # --- 5-fold OOF accumulation ---
    print(f"\nRunning {_XGB_CV_N_SPLITS}-fold OOF accumulation (n_estimators={best_iteration+1})...")
    oof_predictions = np.zeros(len(X_train))
    for fold_i, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        fold_model = xgb.XGBClassifier(**fold_params)
        fold_model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx], verbose=False)
        oof_predictions[val_idx] = fold_model.predict_proba(X_train.iloc[val_idx])[:, 1]
        print(f"  Fold {fold_i}/{_XGB_CV_N_SPLITS} done")

    oof_gini = gini_coefficient(y_train.to_numpy(), oof_predictions)
    print(f"\nOOF Gini (corrected split): {oof_gini:.4f}")

    # --- Refit on full training pool ---
    print("Refitting on full training pool...")
    final_params = {**fold_params}
    model_best = xgb.XGBClassifier(**final_params)
    model_best.fit(X_train, y_train, verbose=False)

    # --- OOT Gini ---
    y_prob_oot = model_best.predict_proba(X_oot)[:, 1]
    oot_gini = gini_coefficient(y_oot.to_numpy(), y_prob_oot)
    print(f"OOT Gini (corrected split): {oot_gini:.4f}")

    gap = abs(oof_gini - oot_gini)
    print(f"Gap |OOF-OOT|:              {gap:.4f}")

    # --- Holdout evaluation ---
    metrics = evaluate_model(model_best, X_test, y_test, "XGBoost (Raw, Calibrated)")
    metrics["oof_gini"] = oof_gini
    metrics["oot_gini"] = oot_gini

    # --- Calibrate ---
    print("Calibrating with Platt scaling...")
    model_calibrated, _, _ = calibrate_model(
        model_best, X_train, y_train, X_test, y_test, method="sigmoid",
        output_model_path="models/xgboost_raw_calibrated.pkl",
        output_figure_path="reports/figures/xgboost_raw_calibration.png",
    )

    # --- Gate check ---
    print("\n" + "="*60)
    print("GATE CHECK")
    print("="*60)
    d12 = oot_gini > 0.60
    d16 = gap <= 0.10
    print(f"D-12  oot_gini > 0.60:        {oot_gini:.4f}  {'PASS' if d12 else 'FAIL'}")
    print(f"D-16  |oof-oot| <= 0.10:      {gap:.4f}   {'PASS' if d16 else 'FAIL'}")
    print(f"Holdout Gini:                 {metrics['Gini']:.4f}")
    print("="*60)

    if d12 and d16:
        disposition = "ACCEPT"
    elif d12 and not d16:
        if gap <= 0.15:
            disposition = "CONDITIONAL_ACCEPT"
        else:
            disposition = "REJECT"
    else:
        disposition = "REJECT"

    # --- Write eval JSON ---
    eval_results = {
        "Model": "XGBoost (Raw, Calibrated)",
        "AUC-ROC": metrics.get("AUC-ROC"),
        "Gini": metrics.get("Gini"),
        "KS": metrics.get("KS"),
        "Brier": metrics.get("Brier"),
        "BrierSkill": metrics.get("BrierSkill"),
        "AvgPrecision": metrics.get("AvgPrecision"),
        "oof_gini": oof_gini,
        "oot_gini": oot_gini,
        "gap": gap,
        "disposition": disposition,
        "nan_oot_fix": True,
        "nan_count_temporal_col": nan_count,
    }
    EVAL_JSON.parent.mkdir(exist_ok=True)
    with open(EVAL_JSON, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nResults written to {EVAL_JSON}")
    print(f"Disposition: {disposition}")

    return 0 if disposition in ("ACCEPT", "CONDITIONAL_ACCEPT") else 1


if __name__ == "__main__":
    sys.exit(main())
