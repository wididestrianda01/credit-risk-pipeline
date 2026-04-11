#!/usr/bin/env python3
"""
Train LightGBM with Optuna HPO on raw + DFS feature store (X_tree_dfs.parquet).

Strategy
--------
- Stratified 5-fold CV for HPO trial selection (fast, representative)
- Temporal OOT holdout (most-recent 20%) for final gate evaluation
- Optimises OOF Gini directly (2 × AUC − 1) per fold
- is_unbalance=True: adjusts both gradient weights AND leaf output values
  (preferred over scale_pos_weight for rank-based Gini metrics)

Produces
--------
  models/lgb_raw_best.pkl
  models/lgb_raw_params.json
  reports/lgb_raw_eval.json
  reports/hpo_progress.jsonl  (per-trial, same format as XGBoost script)

Usage
-----
  python scripts/train_lgb_raw.py [--feature-store PATH] [--n-trials N]
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

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from lightgbm import LGBMClassifier
import optuna
import joblib

# ── Constants ────────────────────────────────────────────────────────────────
_RANDOM_STATE: int = 42
_TEST_SIZE: float = 0.2          # OOT holdout fraction
_N_CV_SPLITS: int = 5
_N_STARTUP_TRIALS: int = 20
_TEMPORAL_SORT_COL: str = "prev_days_decision_mean"
_TARGET_COL: str = "TARGET"
_EARLY_STOPPING_ROUNDS: int = 100
_N_ESTIMATORS_MAX: int = 3000    # upper bound; early stopping controls actual count
_OPTUNA_DB: str = "models/lgb_raw_optuna.db"
_STUDY_NAME: str = "lgb_raw_v1"
_PROGRESS_LOG: str = "reports/hpo_progress.jsonl"
_MODEL_PATH: str = "models/lgb_raw_best.pkl"
_PARAMS_PATH: str = "models/lgb_raw_params.json"
_EVAL_PATH: str = "reports/lgb_raw_eval.json"

# OOT gate thresholds (D-12 tiered policy)
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
        "n_estimators": _N_ESTIMATORS_MAX,          # controlled by early stopping
        "num_leaves": trial.suggest_int("num_leaves", 31, 511),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 300),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 5),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "is_unbalance": True,
        "verbosity": -1,
        "n_jobs": -1,
        "random_state": _RANDOM_STATE,
    }


def _run_cv(params: dict, X_tr: pd.DataFrame, y_tr: pd.Series) -> float:
    """Return mean OOF Gini over stratified 5-fold CV."""
    cv = StratifiedKFold(n_splits=_N_CV_SPLITS, shuffle=True, random_state=_RANDOM_STATE)
    ginis: list[float] = []
    for fold_idx, (ti, vi) in enumerate(cv.split(X_tr, y_tr)):
        X_f, X_v = X_tr.iloc[ti], X_tr.iloc[vi]
        y_f, y_v = y_tr.iloc[ti], y_tr.iloc[vi]
        m = LGBMClassifier(
            **params,
            callbacks=[
                lgb.early_stopping(_EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        m.fit(X_f, y_f, eval_set=[(X_v, y_v)], eval_metric="auc")
        preds = m.predict_proba(X_v)[:, 1]
        gini = 2 * roc_auc_score(y_v, preds) - 1
        ginis.append(gini)
    return float(np.mean(ginis))


def _write_progress(
    trial_number: int,
    trial_id: int,
    oof_gini: float,
    best_oof: float,
    status: str,
    progress_path: Path,
) -> None:
    from datetime import datetime
    record = {
        "trial_number": trial_number,
        "trial_id": trial_id,
        "oof_gini": oof_gini,
        "oot_gini": None,
        "status": status,
        "best_oof_gini_so_far": best_oof,
        "timestamp": datetime.now().isoformat(),
    }
    with open(progress_path, "a") as fh:
        fh.write(json.dumps(record) + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-store", default="data/processed/X_tree_dfs.parquet")
    parser.add_argument("--n-trials", type=int, default=100)
    args = parser.parse_args()

    fs_path = Path(args.feature_store)
    if not fs_path.exists():
        print(f"ERROR: feature store not found: {fs_path}", file=sys.stderr)
        sys.exit(1)

    progress_path = _ROOT / _PROGRESS_LOG
    progress_path.parent.mkdir(exist_ok=True)
    progress_path.unlink(missing_ok=True)   # fresh run

    print(f"Loading feature store: {fs_path}  ({fs_path.stat().st_size // 1024 // 1024} MB)")
    X = pd.read_parquet(fs_path)
    y = X.pop(_TARGET_COL)
    print(f"Shape after popping TARGET: {X.shape}")

    # Replace -999 sentinel with NaN (XGBoost/LGB handle NaN natively)
    X = X.replace(-999.0, np.nan)

    # OOT temporal split
    if _TEMPORAL_SORT_COL not in X.columns:
        print(f"WARNING: temporal column '{_TEMPORAL_SORT_COL}' not found — using random OOT split")
        X_train, X_oot, y_train, y_oot = train_test_split(
            X, y, test_size=_TEST_SIZE, stratify=y, random_state=_RANDOM_STATE
        )
    else:
        X_train, y_train, X_oot, y_oot = _oot_split(X, y)

    # Drop temporal sort column from features (it's a CV grouping aid, not a predictor)
    if _TEMPORAL_SORT_COL in X_train.columns:
        X_train = X_train.drop(columns=[_TEMPORAL_SORT_COL])
        X_oot = X_oot.drop(columns=[_TEMPORAL_SORT_COL])

    print(f"Train: {X_train.shape}, OOT: {X_oot.shape}")
    print(f"Positive rate — train: {y_train.mean():.3%}, OOT: {y_oot.mean():.3%}")
    print(f"Running {args.n_trials} trials. Progress: tail -f {_PROGRESS_LOG}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=_STUDY_NAME,
        storage=f"sqlite:///{_ROOT / _OPTUNA_DB}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=_RANDOM_STATE, n_startup_trials=_N_STARTUP_TRIALS),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=2),
        load_if_exists=True,    # resume if study already exists
    )

    best_oof: float = 0.0

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_oof
        params = _build_params(trial)
        try:
            oof_gini = _run_cv(params, X_train, y_train)
        except Exception:
            _write_progress(trial.number, trial._trial_id, 0.0, best_oof, "FAILED", progress_path)
            raise

        if oof_gini > best_oof:
            best_oof = oof_gini

        _write_progress(trial.number, trial._trial_id, oof_gini, best_oof, "COMPLETE", progress_path)
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

    # ── Refit final model on full train set ──────────────────────────────────
    print("Refitting final model on full train set …")
    final_params = {
        **best_params,
        "n_estimators": int(_N_ESTIMATORS_MAX * 1.1),  # slight boost vs CV-stopped count
        "is_unbalance": True,
        "verbosity": -1,
        "n_jobs": -1,
        "random_state": _RANDOM_STATE,
    }
    final_model = LGBMClassifier(**final_params)
    final_model.fit(
        X_train, y_train,
        callbacks=[lgb.log_evaluation(period=0)],
    )

    # ── OOT evaluation ───────────────────────────────────────────────────────
    oot_preds = final_model.predict_proba(X_oot)[:, 1]
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

    # ── Persist ──────────────────────────────────────────────────────────────
    model_path = _ROOT / _MODEL_PATH
    params_path = _ROOT / _PARAMS_PATH
    eval_path = _ROOT / _EVAL_PATH

    joblib.dump(final_model, model_path)
    with open(params_path, "w") as fh:
        json.dump(best_params, fh, indent=2)

    eval_results = {
        "Model": "LightGBM (Raw, is_unbalance)",
        "oof_gini": best_oof_final,
        "oot_gini": oot_gini,
        "oot_auc": oot_auc,
        "gap": gap,
        "disposition": disposition,
        "reason": reason,
        "n_trials": args.n_trials,
        "n_features": X_train.shape[1],
    }
    with open(eval_path, "w") as fh:
        json.dump(eval_results, fh, indent=2)

    print(f"\nArtifacts saved:")
    print(f"  {model_path}")
    print(f"  {params_path}")
    print(f"  {eval_path}")

    sys.exit(0 if disposition != "REJECT" else 1)


if __name__ == "__main__":
    main()
