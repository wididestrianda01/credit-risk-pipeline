#!/usr/bin/env python3
"""
LightGBM feature-store ablation: compare OOT Gini across the three feature stores.

Feature stores evaluated
------------------------
  raw          data/processed/X_train.parquet       (~195 cols, no TARGET)
  raw+eng      data/processed/X_tree_raw.parquet    (~155+ cols, no TARGET)
  raw+eng+dfs  data/processed/X_tree_dfs.parquet    (~291 cols, with TARGET)

Strategy
--------
- 30 Optuna trials per store (TPE + MedianPruner)
- Stratified 5-fold OOF CV as objective
- Temporal OOT holdout (most-recent 20%) for gate evaluation
- Results saved to reports/lgb_feature_store_selection.json

Usage
-----
  python scripts/lgb_feature_store_ablation.py [--n-trials N]
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

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from lightgbm import LGBMClassifier
import optuna

# ── Constants ────────────────────────────────────────────────────────────────
_RANDOM_STATE: int = 42
_TEST_SIZE: float = 0.20
_N_CV_SPLITS: int = 5
_N_STARTUP_TRIALS: int = 10
_TEMPORAL_SORT_COL: str = "prev_days_decision_mean"
_TARGET_COL: str = "TARGET"
_EARLY_STOPPING_ROUNDS: int = 100
_N_ESTIMATORS_MAX: int = 3000
_OOT_MIN: float = 0.60
_OUTPUT_PATH: str = "reports/lgb_feature_store_selection.json"

_STORES: list[dict] = [
    {
        "name": "raw+eng",
        "path": "data/processed/X_tree_raw.parquet",
        "y_path": "data/processed/y_train.parquet",
        "study": "lgb_ablation_raweng_v1",
        "db": "models/lgb_ablation_raweng.db",
    },
    {
        "name": "raw+eng+dfs",
        "path": "data/processed/X_tree_dfs.parquet",
        "y_path": None,    # TARGET embedded in parquet
        "study": "lgb_ablation_dfs_v1",
        "db": "models/lgb_ablation_dfs.db",
    },
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _oot_split(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Temporal OOT split: most-recent 20% → OOT, rest → train pool."""
    if _TEMPORAL_SORT_COL not in X.columns:
        X_tr, X_ot, y_tr, y_ot = train_test_split(
            X, y, test_size=_TEST_SIZE, stratify=y, random_state=_RANDOM_STATE
        )
        return X_tr, y_tr, X_ot, y_ot

    temporal = X[_TEMPORAL_SORT_COL].values
    nan_mask = np.isnan(temporal)
    known = np.where(~nan_mask)[0]
    unknown = np.where(nan_mask)[0]

    known_sorted = known[np.argsort(temporal[known])]
    cut = int(len(known_sorted) * (1 - _TEST_SIZE))

    rng = np.random.default_rng(_RANDOM_STATE)
    perm = rng.permutation(len(unknown))
    unk_cut = int(len(unknown) * (1 - _TEST_SIZE))

    oot_idx = np.concatenate([known_sorted[cut:], unknown[perm[unk_cut:]]])
    train_idx = np.concatenate([known_sorted[:cut], unknown[perm[:unk_cut]]])

    return (
        X.iloc[train_idx].copy(),
        y.iloc[train_idx].copy(),
        X.iloc[oot_idx].copy(),
        y.iloc[oot_idx].copy(),
    )


def _build_params(trial: optuna.Trial) -> dict:
    params = {
        "n_estimators": _N_ESTIMATORS_MAX,
        "num_leaves": trial.suggest_int("num_leaves", 31, 511),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 300),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "is_unbalance": True,
        "verbosity": -1,
        "n_jobs": -1,
        "random_state": _RANDOM_STATE,
    }
    subsample = params["subsample"]
    if subsample < 1.0:
        params["subsample_freq"] = trial.suggest_int("subsample_freq", 1, 5)
        params["bagging_freq"] = 1
    return params


def _run_cv(params: dict, X_tr: pd.DataFrame, y_tr: pd.Series) -> float:
    cv = StratifiedKFold(n_splits=_N_CV_SPLITS, shuffle=True, random_state=_RANDOM_STATE)
    ginis: list[float] = []
    for ti, vi in cv.split(X_tr, y_tr):
        m = LGBMClassifier(
            **params,
            callbacks=[
                lgb.early_stopping(_EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        m.fit(X_tr.iloc[ti], y_tr.iloc[ti], eval_set=[(X_tr.iloc[vi], y_tr.iloc[vi])], eval_metric="auc")
        preds = m.predict_proba(X_tr.iloc[vi])[:, 1]
        ginis.append(2 * roc_auc_score(y_tr.iloc[vi], preds) - 1)
    return float(np.mean(ginis))


def _run_store(store: dict, n_trials: int) -> dict:
    """Run HPO for one feature store. Returns result dict."""
    fs_path = _ROOT / store["path"]
    if not fs_path.exists():
        return {"name": store["name"], "skipped": True, "reason": f"file not found: {fs_path}"}

    print(f"\n{'='*60}")
    print(f"Store: {store['name']}  ({fs_path.stat().st_size // 1024 // 1024} MB)")

    _ID_COLS = {"SK_ID_CURR"}

    X = pd.read_parquet(fs_path)
    X = X.replace(-999.0, np.nan)

    # Drop loan ID — not a predictive feature
    id_cols_present = _ID_COLS & set(X.columns)
    if id_cols_present:
        X = X.drop(columns=list(id_cols_present))

    # Load target
    if _TARGET_COL in X.columns:
        y = X.pop(_TARGET_COL)
    else:
        y_path = _ROOT / store["y_path"]
        y = pd.read_parquet(y_path).squeeze()

    # Drop temporal column after split
    X_train, y_train, X_oot, y_oot = _oot_split(X, y)
    for col in [_TEMPORAL_SORT_COL]:
        if col in X_train.columns:
            X_train = X_train.drop(columns=[col])
            X_oot = X_oot.drop(columns=[col])

    print(f"Train: {X_train.shape}  OOT: {X_oot.shape}")
    print(f"Positive rate — train: {y_train.mean():.3%}  OOT: {y_oot.mean():.3%}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=store["study"],
        storage=f"sqlite:///{_ROOT / store['db']}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=_RANDOM_STATE, n_startup_trials=_N_STARTUP_TRIALS),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=2),
        load_if_exists=True,
    )

    best_oof: float = 0.0

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_oof
        params = _build_params(trial)
        oof_gini = _run_cv(params, X_train, y_train)
        if oof_gini > best_oof:
            best_oof = oof_gini
        ts = __import__("datetime").datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] Trial {trial.number:3d} | OOF={oof_gini:.4f} | BestOOF={best_oof:.4f}", flush=True)
        return oof_gini

    study.optimize(objective, n_trials=n_trials)

    best_oof_final = study.best_value
    best_params = study.best_params.copy()

    # Refit on full train
    final_params = {
        **best_params,
        "n_estimators": int(_N_ESTIMATORS_MAX * 1.1),
        "is_unbalance": True,
        "verbosity": -1,
        "n_jobs": -1,
        "random_state": _RANDOM_STATE,
    }
    final_model = LGBMClassifier(**final_params)
    final_model.fit(X_train, y_train, callbacks=[lgb.log_evaluation(period=0)])

    oot_preds = final_model.predict_proba(X_oot)[:, 1]
    oot_auc = roc_auc_score(y_oot, oot_preds)
    oot_gini = 2 * oot_auc - 1
    gap = best_oof_final - oot_gini

    if oot_gini < _OOT_MIN:
        disposition = "REJECT"
    elif gap <= 0.05:
        disposition = "ACCEPT"
    elif gap <= 0.10:
        disposition = "CONDITIONAL_ACCEPT"
    else:
        disposition = "REJECT"

    print(f"\n  OOF Gini : {best_oof_final:.4f}")
    print(f"  OOT Gini : {oot_gini:.4f}")
    print(f"  Gap      : {gap:.4f}")
    print(f"  Decision : {disposition}")

    return {
        "name": store["name"],
        "feature_store": store["path"],
        "n_features": X_train.shape[1],
        "n_trials": n_trials,
        "oof_gini": best_oof_final,
        "oot_gini": oot_gini,
        "oot_auc": oot_auc,
        "gap": gap,
        "disposition": disposition,
        "skipped": False,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument(
        "--stores",
        nargs="+",
        choices=["raw+eng", "raw+eng+dfs", "all"],
        default=["all"],
        help="Which stores to run (default: all)",
    )
    args = parser.parse_args()

    run_names = set(_STORES[i]["name"] for i in range(len(_STORES))) if "all" in args.stores else set(args.stores)

    results = []
    for store in _STORES:
        if store["name"] not in run_names:
            print(f"Skipping {store['name']} (not in --stores)")
            continue
        result = _run_store(store, args.n_trials)
        results.append(result)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FEATURE STORE COMPARISON — LightGBM")
    print(f"{'='*60}")
    print(f"{'Store':<16} {'n_feat':>7} {'OOF Gini':>10} {'OOT Gini':>10} {'Gap':>8} {'Decision'}")
    print("-" * 65)
    for r in results:
        if r.get("skipped"):
            print(f"{r['name']:<16} {'SKIPPED':>37}  ({r.get('reason', '')})")
        else:
            print(
                f"{r['name']:<16} {r['n_features']:>7} {r['oof_gini']:>10.4f} "
                f"{r['oot_gini']:>10.4f} {r['gap']:>8.4f}  {r['disposition']}"
            )

    valid = [r for r in results if not r.get("skipped")]
    if valid:
        winner = max(valid, key=lambda r: r["oot_gini"])
        print(f"\nWinner: {winner['name']} — OOT Gini {winner['oot_gini']:.4f}")

    output_path = _ROOT / _OUTPUT_PATH
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump({"results": results, "winner": winner["name"] if valid else None}, fh, indent=2)
    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()
