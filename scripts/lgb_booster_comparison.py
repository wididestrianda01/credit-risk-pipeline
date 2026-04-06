"""
lgb_booster_comparison.py — Step A6
------------------------------------
Benchmark GBDT / DART / GOSS on raw continuous features with fixed
hyperparameters from the ablation baseline. Each booster is evaluated
under 5-fold temporal CV. Results identify which algorithm best exploits
the continuous feature space and unblocks the monotone constraint test (A7).

Output: reports/lgb_booster_comparison.json

Usage
-----
    .venv/bin/python scripts/lgb_booster_comparison.py
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_X_PATH = "data/processed/X_raw_features.parquet"
_Y_PATH = "data/processed/y_train.parquet"
_OUTPUT_PATH = "reports/lgb_booster_comparison.json"

_MIN_ROWS = 100_000

# Fixed hyperparameters from ablation baseline (best Optuna config, ablation b)
_FIXED_HP: dict = {
    "num_leaves": 125,
    "max_depth": 4,
    "min_child_samples": 90,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 4.25,
    "reg_lambda": 9.54,
    "is_unbalance": True,
    "verbosity": -1,
    "random_state": 42,
    "metric": "auc",
}

_CV_N_SPLITS = 5
_TEMPORAL_SORT_COL = "prev_days_decision_mean"
_CV_EMBARGO_FRAC = 0.02

_BOOSTERS: list[str] = ["gbdt", "dart", "goss"]

# DART-specific extra params (conservative defaults — not searched here)
_DART_EXTRA: dict = {"drop_rate": 0.1}

# GOSS-specific extra params
_GOSS_EXTRA: dict = {"top_rate": 0.1, "other_rate": 0.05}


# ---------------------------------------------------------------------------
# Temporal CV with embargo
# ---------------------------------------------------------------------------

class _TemporalCV:
    """Temporal k-fold CV with embargo strip at each train/val boundary."""

    def __init__(self, n_splits: int, embargo_frac: float, random_state: int = 42) -> None:
        self.n_splits = n_splits
        self.embargo_frac = embargo_frac
        self.random_state = random_state
        self._skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def split(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None):
        n = len(y)
        embargo_n = max(1, int(n * self.embargo_frac))
        for train_idx, val_idx in self._skf.split(X, y):
            # Sort training indices by group (temporal order)
            if groups is not None:
                order = np.argsort(groups[train_idx])
                train_idx = train_idx[order]
            # Remove the last embargo_n rows of the sorted training set
            train_idx = train_idx[: max(1, len(train_idx) - embargo_n)]
            yield train_idx, val_idx


def _make_cv(groups: np.ndarray | None, n_splits: int) -> _TemporalCV | StratifiedKFold:
    if groups is not None:
        return _TemporalCV(n_splits=n_splits, embargo_frac=_CV_EMBARGO_FRAC)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _run_booster(
    booster: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: _TemporalCV | StratifiedKFold,
    groups: np.ndarray | None,
) -> dict:
    """Train one booster under CV and return per-fold Gini stats."""
    from sklearn.metrics import roc_auc_score

    params = {**_FIXED_HP, "boosting_type": booster}
    if booster == "dart":
        params.update(_DART_EXTRA)
    elif booster == "goss":
        params.update(_GOSS_EXTRA)

    # DART does not support early stopping in LGB 4.x
    use_es = booster != "dart"
    callbacks = [lgb.log_evaluation(period=0)]
    if use_es:
        callbacks.insert(0, lgb.early_stopping(stopping_rounds=20, verbose=False))

    fold_ginis: list[float] = []
    t0 = time.perf_counter()
    for fold_i, (train_idx, val_idx) in enumerate(cv.split(X_train.values, y_train.values, groups)):
        X_fold_tr = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_tr = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        model = lgb.LGBMClassifier(**params)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Early stopping is not available in dart mode")
            model.fit(
                X_fold_tr, y_fold_tr,
                eval_set=[(X_fold_val, y_fold_val)],
                callbacks=callbacks,
            )
        y_prob = model.predict_proba(X_fold_val)[:, 1]
        gini = 2.0 * float(roc_auc_score(y_fold_val, y_prob)) - 1.0
        fold_ginis.append(gini)
        print(f"  {booster.upper()} fold {fold_i + 1}/{_CV_N_SPLITS}: Gini={gini:.4f}")

    elapsed = time.perf_counter() - t0
    return {
        "booster": booster,
        "mean_gini": float(np.mean(fold_ginis)),
        "std_gini": float(np.std(fold_ginis)),
        "min_gini": float(np.min(fold_ginis)),
        "max_gini": float(np.max(fold_ginis)),
        "fold_ginis": fold_ginis,
        "elapsed_sec": round(elapsed, 1),
    }


def run_booster_comparison() -> list[dict]:
    """
    Run booster comparison on X_raw_features.parquet.

    Returns
    -------
    list[dict]
        One record per booster with mean/std/min/max Gini and fold details.
    """
    print("Loading data...")
    X = pd.read_parquet(_X_PATH)
    y = pd.read_parquet(_Y_PATH).squeeze()

    assert X.shape[0] >= _MIN_ROWS, (
        f"Pre-flight check failed: X has {X.shape[0]} rows, expected >= {_MIN_ROWS}. "
        "Run scripts/rebuild_feature_store.py to regenerate."
    )
    assert X.isnull().sum().sum() == 0, "NaN values found in X_raw_features."
    print(f"Data loaded: {X.shape[0]:,} rows × {X.shape[1]} cols")

    # Temporal CV setup
    groups: np.ndarray | None = None
    if _TEMPORAL_SORT_COL in X.columns:
        groups = X[_TEMPORAL_SORT_COL].to_numpy()
        print(f"Temporal CV: using '{_TEMPORAL_SORT_COL}' for group ordering")
    else:
        print(f"Warning: '{_TEMPORAL_SORT_COL}' not found — falling back to StratifiedKFold")

    cv = _make_cv(groups, _CV_N_SPLITS)

    results: list[dict] = []
    for booster in _BOOSTERS:
        print(f"\nRunning {booster.upper()} booster...")
        record = _run_booster(booster, X, y, cv, groups)
        results.append(record)
        print(f"  → mean Gini={record['mean_gini']:.4f} ± {record['std_gini']:.4f}")

    # Sort by mean_gini descending
    results.sort(key=lambda r: r["mean_gini"], reverse=True)

    best = results[0]
    print(f"\nBest booster: {best['booster'].upper()} (mean Gini={best['mean_gini']:.4f})")

    output = {
        "results": results,
        "best_booster": best["booster"],
        "best_mean_gini": best["mean_gini"],
        "baseline_mean_gini": next(r["mean_gini"] for r in results if r["booster"] == "gbdt"),
        "metadata": {
            "X_path": _X_PATH,
            "n_rows": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "cv_folds": _CV_N_SPLITS,
            "embargo_frac": _CV_EMBARGO_FRAC,
            "fixed_hp": _FIXED_HP,
        },
    }

    out_path = Path(_OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nResults saved to {_OUTPUT_PATH}")

    return results


if __name__ == "__main__":
    run_booster_comparison()
