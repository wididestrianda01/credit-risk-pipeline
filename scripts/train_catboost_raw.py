#!/usr/bin/env python3
"""
Train CatBoost with Optuna HPO on raw + eng feature store (X_tree_raw.parquet).

Strategy
--------
- Stratified 5-fold CV for HPO trial selection (fast, representative)
- Temporal OOT holdout (most-recent 20%) for final gate evaluation
- Optimises OOF AUC → Gini directly per fold
- scale_pos_weight = n_neg/n_pos: adjusts gradient weights for 8% imbalance
- 2-stage refit: Stage 1 (80/20 holdout, early stopping) → best_iteration_;
  Stage 2 (full X_train, fixed iterations, no early stopping)
- Platt calibration: FrozenEstimator + CalibratedClassifierCV(cv="prefit")

Produces
--------
  models/catboost_raw_best.pkl         (uncalibrated final model)
  models/catboost_raw_calibrated.pkl   (Platt-calibrated)
  models/catboost_raw_params.json
  reports/catboost_raw_eval.json

Usage
-----
  python scripts/train_catboost_raw.py [--feature-store PATH] [--n-trials N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
import src as _src  # noqa: E402
if "credit_engine" not in sys.modules:
    sys.modules["credit_engine"] = _src

from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
import optuna
import joblib

# ── Constants ────────────────────────────────────────────────────────────────
_RANDOM_STATE: int = 42
_TEST_SIZE: float = 0.2
_N_CV_SPLITS: int = 5
_N_STARTUP_TRIALS: int = 15
_TEMPORAL_SORT_COL: str = "prev_days_decision_mean"
_TARGET_COL: str = "TARGET"
_ITERATIONS_MAX: int = 4000
_OBJ_EARLY_STOPPING_ROUNDS: int = 50   # inside Optuna objective (fast triage)
_FINAL_EARLY_STOPPING_ROUNDS: int = 100  # Stage 1 refit (more patience)
_OPTUNA_DB: str = "models/optuna_studies.db"
_STUDY_NAME: str = "catboost_raw_v3"
_MODEL_PATH: str = "models/catboost_raw_best.pkl"
_CALIBRATED_MODEL_PATH: str = "models/catboost_raw_calibrated.pkl"
_PARAMS_PATH: str = "models/catboost_raw_params.json"
_EVAL_PATH: str = "reports/catboost_raw_eval.json"

# HPO search bounds (aligned with Phase 04.2.5 constants)
_DEPTH_MIN: int = 6
_DEPTH_MAX: int = 14
_LR_MIN: float = 0.005
_LR_MAX: float = 0.15
_L2_MIN: float = 0.01
_L2_MAX: float = 50.0
_MIN_DATA_MIN: int = 5
_MIN_DATA_MAX: int = 50

# OOT gate thresholds
_OOT_MIN: float = 0.60
_GAP_OPTIMAL: float = 0.05
_GAP_WARN: float = 0.10


# ── Helpers ──────────────────────────────────────────────────────────────────

def _oot_split(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Temporal OOT split: most-recent 20% → OOT, rest → train pool."""
    temporal_values = X[_TEMPORAL_SORT_COL].values
    nan_mask = np.isnan(temporal_values)
    known_pos = np.where(~nan_mask)[0]
    unknown_pos = np.where(nan_mask)[0]

    known_sorted = known_pos[np.argsort(temporal_values[known_pos])]
    cut = int(len(known_sorted) * (1 - _TEST_SIZE))
    oot_known = known_sorted[cut:]
    train_known = known_sorted[:cut]

    rng = np.random.default_rng(_RANDOM_STATE)
    perm = rng.permutation(len(unknown_pos))
    unk_cut = int(len(unknown_pos) * (1 - _TEST_SIZE))
    oot_unknown = unknown_pos[perm[unk_cut:]]
    train_unknown = unknown_pos[perm[:unk_cut]]

    oot_idx = np.concatenate([oot_known, oot_unknown])
    train_idx = np.concatenate([train_known, train_unknown])

    return (
        X.iloc[train_idx].copy(),
        y.iloc[train_idx].copy(),
        X.iloc[oot_idx].copy(),
        y.iloc[oot_idx].copy(),
    )


def _build_params(trial: optuna.Trial) -> dict:
    return {
        "iterations": _ITERATIONS_MAX,
        "depth": trial.suggest_int("depth", _DEPTH_MIN, _DEPTH_MAX),
        "learning_rate": trial.suggest_float("learning_rate", _LR_MIN, _LR_MAX, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", _L2_MIN, _L2_MAX, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", _MIN_DATA_MIN, _MIN_DATA_MAX),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.5),
        "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
        "bootstrap_type": "Bayesian",
        "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Lossguide"]),
        "auto_class_weights": "Balanced",
        "random_seed": _RANDOM_STATE,
        "verbose": 0,
        "allow_writing_files": False,
        "early_stopping_rounds": _OBJ_EARLY_STOPPING_ROUNDS,
    }


def _run_cv(params: dict, X_tr: pd.DataFrame, y_tr: pd.Series) -> float:
    """Return mean OOF Gini over stratified 5-fold CV."""
    cv = StratifiedKFold(n_splits=_N_CV_SPLITS, shuffle=True, random_state=_RANDOM_STATE)
    ginis: list[float] = []
    for _, (ti, vi) in enumerate(cv.split(X_tr, y_tr)):
        X_f, X_v = X_tr.iloc[ti].to_numpy(), X_tr.iloc[vi].to_numpy()
        y_f, y_v = y_tr.iloc[ti].to_numpy(), y_tr.iloc[vi].to_numpy()
        m = CatBoostClassifier(**params)
        m.fit(
            X_f, y_f,
            eval_set=(X_v, y_v),
            verbose=False,
        )
        preds = m.predict_proba(X_v)[:, 1]
        gini = 2 * roc_auc_score(y_v, preds) - 1
        ginis.append(gini)
    return float(np.mean(ginis))


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-store", default="data/processed/X_tree_raw.parquet")
    parser.add_argument("--n-trials", type=int, default=50)
    args = parser.parse_args()

    fs_path = Path(args.feature_store)
    if not fs_path.exists():
        print(f"ERROR: feature store not found: {fs_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading feature store: {fs_path}  ({fs_path.stat().st_size // 1024 // 1024} MB)")
    df = pd.read_parquet(fs_path)
    y = df.pop(_TARGET_COL).astype(int)
    X = df
    print(f"Shape after popping TARGET: {X.shape}")

    # Replace -999 sentinel with NaN (CatBoost handles NaN natively)
    X = X.replace(-999.0, np.nan)

    # Temporal OOT split
    if _TEMPORAL_SORT_COL not in X.columns:
        print(f"WARNING: temporal column '{_TEMPORAL_SORT_COL}' not found — using random OOT split")
        X_train, X_oot, y_train, y_oot = train_test_split(
            X, y, test_size=_TEST_SIZE, stratify=y, random_state=_RANDOM_STATE
        )
    else:
        X_train, y_train, X_oot, y_oot = _oot_split(X, y)

    if _TEMPORAL_SORT_COL in X_train.columns:
        X_train = X_train.drop(columns=[_TEMPORAL_SORT_COL])
        X_oot = X_oot.drop(columns=[_TEMPORAL_SORT_COL])

    print(f"Train: {X_train.shape}, OOT: {X_oot.shape}")
    print(f"Positive rate — train: {y_train.mean():.3%}, OOT: {y_oot.mean():.3%}")
    print(f"Running {args.n_trials} trials …")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=_STUDY_NAME,
        storage=f"sqlite:///{_ROOT / _OPTUNA_DB}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=_RANDOM_STATE, n_startup_trials=_N_STARTUP_TRIALS),
        pruner=optuna.pruners.HyperbandPruner(min_resource=10, max_resource=50, reduction_factor=3),
        load_if_exists=True,
    )

    best_oof: float = 0.0

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_oof
        params = _build_params(trial)
        try:
            oof_gini = _run_cv(params, X_train, y_train)
        except Exception:
            raise

        if oof_gini > best_oof:
            best_oof = oof_gini

        ts = __import__("datetime").datetime.now().strftime("%H:%M:%S")
        print(
            f"[{ts}] Trial {trial.number:3d} | OOF={oof_gini:.4f} | BestOOF={best_oof:.4f}",
            flush=True,
        )
        return oof_gini

    study.optimize(objective, n_trials=args.n_trials)

    best_params = study.best_params.copy()
    best_oof_final = study.best_value
    print(f"\nHPO complete. Best OOF Gini: {best_oof_final:.4f}")

    # ── 2-stage refit ────────────────────────────────────────────────────────
    print("Stage 1: Refit on 80% of X_train with early stopping to capture best_iteration_ …")
    X_tr, X_val_es, y_tr, y_val_es = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=_RANDOM_STATE
    )
    stage1_params = {
        **best_params,
        "bootstrap_type": "Bayesian",
        "auto_class_weights": "Balanced",
        "random_seed": _RANDOM_STATE,
        "verbose": 0,
        "allow_writing_files": False,
        "iterations": _ITERATIONS_MAX,
        "early_stopping_rounds": _FINAL_EARLY_STOPPING_ROUNDS,
    }
    stage1_model = CatBoostClassifier(**stage1_params)
    stage1_model.fit(
        X_tr.to_numpy(), y_tr.to_numpy(),
        eval_set=(X_val_es.to_numpy(), y_val_es.to_numpy()),
        verbose=False,
    )
    best_iterations = stage1_model.best_iteration_ or _ITERATIONS_MAX
    print(f"Stage 1 best_iteration_: {best_iterations}")

    print("Stage 2: Refit on full X_train with fixed iterations, no early stopping …")
    stage2_params = {
        **best_params,
        "bootstrap_type": "Bayesian",
        "auto_class_weights": "Balanced",
        "random_seed": _RANDOM_STATE,
        "verbose": 0,
        "allow_writing_files": False,
        "iterations": best_iterations,
    }
    final_model = CatBoostClassifier(**stage2_params)
    final_model.fit(X_train.to_numpy(), y_train.to_numpy(), verbose=False)

    # ── OOT evaluation ───────────────────────────────────────────────────────
    oot_preds = final_model.predict_proba(X_oot.to_numpy())[:, 1]
    oot_auc = roc_auc_score(y_oot, oot_preds)
    oot_gini = 2 * oot_auc - 1

    gap = best_oof_final - oot_gini
    if oot_gini < _OOT_MIN:
        disposition = "REJECT"
        reason = f"OOT Gini {oot_gini:.4f} below minimum {_OOT_MIN}"
    elif gap <= _GAP_OPTIMAL:
        disposition = "ACCEPT"
        reason = f"Gap {gap:.4f} within optimal range (≤{_GAP_OPTIMAL})"
    elif gap <= _GAP_WARN:
        disposition = "CONDITIONAL_ACCEPT"
        reason = f"Gap {gap:.4f} in acceptable range ({_GAP_OPTIMAL}–{_GAP_WARN})"
    else:
        disposition = "REJECT"
        reason = f"Gap {gap:.4f} exceeds acceptable range (>{_GAP_WARN})"

    print(f"\nOOF Gini : {best_oof_final:.4f}")
    print(f"OOT Gini : {oot_gini:.4f}")
    print(f"Gap      : {gap:.4f}")
    print(f"Decision : {disposition} — {reason}")

    # ── Platt calibration ────────────────────────────────────────────────────
    print("\nApplying Platt calibration …")
    calibrated_model = CalibratedClassifierCV(
        FrozenEstimator(final_model), cv="prefit", method="sigmoid"
    )
    calibrated_model.fit(X_train.to_numpy(), y_train.to_numpy())

    # ── Persist ──────────────────────────────────────────────────────────────
    model_path = _ROOT / _MODEL_PATH
    calibrated_model_path = _ROOT / _CALIBRATED_MODEL_PATH
    params_path = _ROOT / _PARAMS_PATH
    eval_path = _ROOT / _EVAL_PATH

    joblib.dump(final_model, model_path)
    joblib.dump(calibrated_model, calibrated_model_path)
    with open(params_path, "w") as fh:
        json.dump(best_params, fh, indent=2)

    eval_results = {
        "Model": "CatBoost (Raw, auto_class_weights=Balanced)",
        "oof_gini": best_oof_final,
        "oot_gini": oot_gini,
        "oot_auc": oot_auc,
        "gap": gap,
        "disposition": disposition,
        "reason": reason,
        "best_iterations": best_iterations,
        "n_trials": args.n_trials,
        "n_features": X_train.shape[1],
        "best_params": best_params,
    }
    with open(eval_path, "w") as fh:
        json.dump(eval_results, fh, indent=2)

    print(f"\nArtifacts saved:")
    print(f"  {model_path}")
    print(f"  {calibrated_model_path}")
    print(f"  {params_path}")
    print(f"  {eval_path}")

    sys.exit(0 if disposition != "REJECT" else 1)


if __name__ == "__main__":
    main()
