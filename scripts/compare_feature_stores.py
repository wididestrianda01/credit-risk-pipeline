"""
Compare X_tree_raw.parquet vs X_tree_dfs.parquet using fixed XGBoost hyperparameters.

Isolates the effect of DFS augmentation from HPO variance by holding hyperparameters
constant across both feature stores. Uses the same temporal OOT split as
train_xgboost_optuna (prev_days_decision_mean ordering, 20% OOT holdout).

Usage:
    python scripts/compare_feature_stores.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import xgboost as xgb

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TEMPORAL_SORT_COL = "prev_days_decision_mean"
_OOT_FRAC = 0.20
_RANDOM_STATE = 42
_N_FOLDS = 5

# Fixed params from xgboost_raw_best.pkl — best model trained on X_tree_raw
_FIXED_PARAMS: dict = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.011202764693866375,
    "max_depth": 4,
    "colsample_bytree": 0.7494805465555474,
    "subsample": 0.8238384351992375,
    "gamma": 1.338944462454472,
    "min_child_weight": 3.7259921349675644,
    "reg_alpha": 4.936553831366166,
    "reg_lambda": 1.043378679353705e-08,
    "tree_method": "hist",
    "verbosity": 0,
    "random_state": _RANDOM_STATE,
    # n_estimators & scale_pos_weight set per-run
    "n_estimators": 400,
    "early_stopping_rounds": 30,
}


def gini(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return 2 * roc_auc_score(y_true, y_prob) - 1


def temporal_oot_split(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Replicate the OOT split from train_xgboost_optuna."""
    temporal_vals = X[_TEMPORAL_SORT_COL].values
    nan_mask = np.isnan(temporal_vals)
    known_pos = np.where(~nan_mask)[0]
    unknown_pos = np.where(nan_mask)[0]

    known_sorted = known_pos[np.argsort(temporal_vals[known_pos])]
    oot_known_cut = int(len(known_sorted) * (1 - _OOT_FRAC))
    oot_known = known_sorted[oot_known_cut:]
    train_known = known_sorted[:oot_known_cut]

    rng = np.random.default_rng(_RANDOM_STATE)
    unknown_perm = rng.permutation(len(unknown_pos))
    oot_unknown_cut = int(len(unknown_pos) * (1 - _OOT_FRAC))
    oot_unknown = unknown_pos[unknown_perm[oot_unknown_cut:]]
    train_unknown = unknown_pos[unknown_perm[:oot_unknown_cut]]

    oot_indices = np.concatenate([oot_known, oot_unknown])
    train_indices = np.concatenate([train_known, train_unknown])

    return (
        X.iloc[train_indices].copy(),
        y.iloc[train_indices].copy(),
        X.iloc[oot_indices].copy(),
        y.iloc[oot_indices].copy(),
    )


def evaluate_store(label: str, X: pd.DataFrame, y: pd.Series) -> dict:
    """Run temporal split → 5-fold OOF → OOT evaluation with fixed params."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"  Feature shape: {X.shape}")

    # Drop non-feature columns
    drop_cols = [c for c in ["SK_ID_CURR", "TARGET"] if c in X.columns]
    X = X.drop(columns=drop_cols)

    # Replace -999 sentinel with NaN (XGBoost handles natively)
    X = X.replace(-999.0, np.nan)

    t0 = time.time()
    X_train, y_train, X_oot, y_oot = temporal_oot_split(X, y)
    print(f"  Train rows: {len(X_train)}, OOT rows: {len(X_oot)}")

    scale_pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())
    params = {**_FIXED_PARAMS, "scale_pos_weight": scale_pos_weight}
    early_stop = params.pop("early_stopping_rounds")
    n_est = params.pop("n_estimators")

    # --- 5-fold OOF ---
    skf = StratifiedKFold(n_splits=_N_FOLDS, shuffle=True, random_state=_RANDOM_STATE)
    oof_probs = np.zeros(len(X_train))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        ytr, yval = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        model = xgb.XGBClassifier(
            **params,
            n_estimators=n_est,
            early_stopping_rounds=early_stop,
        )
        model.fit(
            Xtr, ytr,
            eval_set=[(Xval, yval)],
            verbose=False,
        )
        oof_probs[val_idx] = model.predict_proba(Xval)[:, 1]
        print(f"  Fold {fold}: best_iteration={model.best_iteration}")

    oof_gini = gini(y_train.values, oof_probs)

    # --- Final model trained on full training set for OOT eval ---
    # Use best_iteration from last fold as proxy for full-data n_estimators
    n_final = int(model.best_iteration * 1.1) if model.best_iteration else n_est
    model_final = xgb.XGBClassifier(**params, n_estimators=n_final)
    model_final.fit(X_train, y_train, verbose=False)
    oot_probs = model_final.predict_proba(X_oot)[:, 1]
    oot_gini = gini(y_oot.values, oot_probs)

    elapsed = time.time() - t0

    result = {
        "label": label,
        "n_features": X.shape[1],
        "n_train": len(X_train),
        "n_oot": len(X_oot),
        "oof_gini": round(oof_gini, 4),
        "oot_gini": round(oot_gini, 4),
        "elapsed_sec": round(elapsed, 1),
    }

    print(f"\n  {'OOF Gini':>12}: {oof_gini:.4f}")
    print(f"  {'OOT Gini':>12}: {oot_gini:.4f}")
    print(f"  {'Elapsed':>12}: {elapsed:.0f}s")

    return result


def main() -> None:
    raw_path = _PROJECT_ROOT / "data" / "processed" / "X_tree_raw.parquet"
    dfs_path = _PROJECT_ROOT / "data" / "processed" / "X_tree_dfs.parquet"

    print("Loading feature stores...")
    X_dfs_full = pd.read_parquet(dfs_path)
    y_all = X_dfs_full["TARGET"].copy()

    # Raw store: add TARGET (same row order, verified identical integer index)
    X_raw_full = pd.read_parquet(raw_path)
    if "TARGET" not in X_raw_full.columns:
        X_raw_full["TARGET"] = y_all.values

    print(f"X_tree_raw shape:  {X_raw_full.shape}")
    print(f"X_tree_dfs shape:  {X_dfs_full.shape}")
    print(f"Target prevalence: {y_all.mean():.4f}")

    results = []
    results.append(evaluate_store("X_tree_raw (Plans 01-04, no DFS)", X_raw_full, y_all))
    results.append(evaluate_store("X_tree_dfs (Plans 01-04 + DFS)", X_dfs_full, y_all))

    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    for r in results:
        print(
            f"  {r['label']:<40} | "
            f"features={r['n_features']:>3} | "
            f"OOF Gini={r['oof_gini']:.4f} | "
            f"OOT Gini={r['oot_gini']:.4f}"
        )

    delta_oof = results[1]["oof_gini"] - results[0]["oof_gini"]
    delta_oot = results[1]["oot_gini"] - results[0]["oot_gini"]
    print(f"\n  DFS delta  OOF: {delta_oof:+.4f} | OOT: {delta_oot:+.4f}")
    if delta_oot >= 0.005:
        print("  VERDICT: DFS adds measurable OOT lift — use X_tree_dfs.parquet for HPO")
    elif delta_oot <= -0.005:
        print("  VERDICT: DFS hurts OOT Gini — use X_tree_raw.parquet for HPO")
    else:
        print("  VERDICT: DFS effect is within noise — either store is acceptable")

    out_path = _PROJECT_ROOT / "reports" / "feature_store_comparison.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
