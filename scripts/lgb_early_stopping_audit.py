"""
lgb_early_stopping_audit.py — Step A9
---------------------------------------
Compare early-stopping patience configs {20, 50, 100, None} under
5-fold temporal CV. Measures:
  - Mean Gini per patience value
  - Per-fold n_estimators_when_stopped (variance = instability signal)
  - High variance across folds → patience too low

The current two-tier config (20 in Optuna objective, 50 in final refit)
was tuned for mock-data test speed. On 307K rows it may differ.

Output: reports/lgb_early_stopping_audit.json

Usage
-----
    .venv/bin/python scripts/lgb_early_stopping_audit.py [--booster BOOSTER]
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_X_PATH = "data/processed/X_raw_features.parquet"
_Y_PATH = "data/processed/y_train.parquet"
_A6_RESULTS_PATH = "reports/lgb_booster_comparison.json"
_OUTPUT_PATH = "reports/lgb_early_stopping_audit.json"

_MIN_ROWS = 100_000
_RANDOM_STATE = 42
_CV_N_SPLITS = 5
_TEMPORAL_SORT_COL = "prev_days_decision_mean"
_CV_EMBARGO_FRAC = 0.02

# Fixed HP (from ablation baseline) — we vary only patience
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
    "random_state": _RANDOM_STATE,
    "metric": "auc",
}

_DART_EXTRA: dict = {"drop_rate": 0.1}
_GOSS_EXTRA: dict = {"top_rate": 0.1, "other_rate": 0.05}

_PATIENCE_CONFIGS: list[int | None] = [20, 50, 100, None]


# ---------------------------------------------------------------------------
# CV helpers
# ---------------------------------------------------------------------------

class _TemporalCV:
    def __init__(self, n_splits: int, embargo_frac: float) -> None:
        self.n_splits = n_splits
        self.embargo_frac = embargo_frac
        self._skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=_RANDOM_STATE)

    def split(self, X, y, groups=None):
        n = len(y)
        embargo_n = max(1, int(n * self.embargo_frac))
        for train_idx, val_idx in self._skf.split(X, y):
            if groups is not None:
                order = np.argsort(groups[train_idx])
                train_idx = train_idx[order]
            train_idx = train_idx[: max(1, len(train_idx) - embargo_n)]
            yield train_idx, val_idx


def _make_cv(groups, n_splits):
    if groups is not None:
        return _TemporalCV(n_splits=n_splits, embargo_frac=_CV_EMBARGO_FRAC)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=_RANDOM_STATE)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_early_stopping_audit(booster: str = "gbdt") -> list[dict]:
    """
    Benchmark early stopping patience configs under temporal CV.

    Parameters
    ----------
    booster : str
        Booster algorithm (use DART-safe logic — no early stopping for DART).

    Returns
    -------
    list[dict]
        One record per patience config with Gini and n_estimators stats.
    """
    print("Loading data...")
    X = pd.read_parquet(_X_PATH)
    y = pd.read_parquet(_Y_PATH).squeeze()

    assert X.shape[0] >= _MIN_ROWS, f"Pre-flight: {X.shape[0]} rows < {_MIN_ROWS}"
    assert X.isnull().sum().sum() == 0, "NaN values in X."
    print(f"Data loaded: {X.shape[0]:,} rows × {X.shape[1]} cols, booster={booster.upper()}")

    if booster == "dart":
        print("Note: DART does not support early stopping — patience audit will show constant n_estimators.")

    groups: np.ndarray | None = None
    if _TEMPORAL_SORT_COL in X.columns:
        groups = X[_TEMPORAL_SORT_COL].to_numpy()
    cv = _make_cv(groups, _CV_N_SPLITS)

    base_params = {**_FIXED_HP, "boosting_type": booster}
    if booster == "dart":
        base_params.update(_DART_EXTRA)
    elif booster == "goss":
        base_params.update(_GOSS_EXTRA)

    results: list[dict] = []

    for patience in _PATIENCE_CONFIGS:
        label = f"patience={patience}" if patience is not None else "patience=None (no ES)"
        print(f"\nRunning {label}...")

        fold_ginis: list[float] = []
        fold_n_estimators: list[int] = []
        t0 = time.perf_counter()

        for fold_i, (train_idx, val_idx) in enumerate(
            cv.split(X.values, y.values, groups)
        ):
            X_fold_tr = X.iloc[train_idx]
            X_fold_val = X.iloc[val_idx]
            y_fold_tr = y.iloc[train_idx]
            y_fold_val = y.iloc[val_idx]

            callbacks = [lgb.log_evaluation(period=0)]
            if patience is not None and booster != "dart":
                callbacks.insert(0, lgb.early_stopping(stopping_rounds=patience, verbose=False))

            model = lgb.LGBMClassifier(**base_params)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Early stopping is not available in dart mode")
                model.fit(
                    X_fold_tr, y_fold_tr,
                    eval_set=[(X_fold_val, y_fold_val)],
                    callbacks=callbacks,
                )

            n_trees = getattr(model, "best_iteration_", -1)
            if n_trees <= 0:
                n_trees = base_params["n_estimators"]
            fold_n_estimators.append(int(n_trees))

            y_prob = model.predict_proba(X_fold_val)[:, 1]
            gini = 2.0 * float(roc_auc_score(y_fold_val, y_prob)) - 1.0
            fold_ginis.append(gini)
            print(f"  Fold {fold_i + 1}/{_CV_N_SPLITS}: Gini={gini:.4f}, n_estimators={n_trees}")

        elapsed = round(time.perf_counter() - t0, 1)
        mean_gini = float(np.mean(fold_ginis))
        std_gini = float(np.std(fold_ginis))
        mean_n_est = float(np.mean(fold_n_estimators))
        std_n_est = float(np.std(fold_n_estimators))
        cv_n_est = std_n_est / mean_n_est if mean_n_est > 0 else 0.0

        print(f"  → Gini={mean_gini:.4f} ± {std_gini:.4f}")
        print(f"  → n_estimators: mean={mean_n_est:.0f}, std={std_n_est:.0f}, CV={cv_n_est:.2%}")
        if cv_n_est > 0.3 and patience is not None:
            print(f"  ⚠ High n_estimators variance (CV={cv_n_est:.1%}) — patience={patience} may be too low")

        results.append({
            "patience": patience,
            "mean_gini": mean_gini,
            "std_gini": std_gini,
            "fold_ginis": fold_ginis,
            "mean_n_estimators": round(mean_n_est, 1),
            "std_n_estimators": round(std_n_est, 1),
            "cv_n_estimators": round(cv_n_est, 4),
            "fold_n_estimators": fold_n_estimators,
            "elapsed_sec": elapsed,
        })

    # Recommendation
    best = max(results, key=lambda r: r["mean_gini"])
    # Prefer stable configs (low cv_n_estimators) when Gini differences are tiny
    stable_results = [r for r in results if r["cv_n_estimators"] < 0.3]
    recommended = (
        max(stable_results, key=lambda r: r["mean_gini"])
        if stable_results else best
    )
    print(f"\nRecommended patience: {recommended['patience']} "
          f"(Gini={recommended['mean_gini']:.4f}, "
          f"n_est CV={recommended['cv_n_estimators']:.1%})")

    output = {
        "booster": booster,
        "results": results,
        "recommended_patience": recommended["patience"],
        "metadata": {
            "X_path": _X_PATH,
            "n_rows": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "cv_folds": _CV_N_SPLITS,
        },
    }

    out_path = Path(_OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(output, fh, indent=2)
    print(f"Results saved to {_OUTPUT_PATH}")

    return results


def _resolve_booster(override: str | None) -> str:
    if override:
        return override
    a6_path = Path(_A6_RESULTS_PATH)
    if a6_path.exists():
        with a6_path.open() as fh:
            data = json.load(fh)
        best = data.get("best_booster", "gbdt")
        print(f"Using best booster from A6: {best.upper()}")
        return best
    return "gbdt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Early stopping patience audit")
    parser.add_argument("--booster", type=str, default=None)
    args = parser.parse_args()
    booster = _resolve_booster(args.booster)
    run_early_stopping_audit(booster=booster)
