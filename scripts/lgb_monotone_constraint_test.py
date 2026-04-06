"""
lgb_monotone_constraint_test.py — Step A7
------------------------------------------
Apply domain-knowledge monotone constraints to LightGBM on raw features,
using the best booster identified in Step A6.

Constraints encode known credit-risk directions:
  - AGE_YEARS          +1  (older applicants → lower default risk)
  - YEARS_EMPLOYED     +1  (longer employment → lower risk)
  - CREDIT_INCOME_RATIO -1 (higher debt load → higher risk)
  - EXT_SOURCE_1       +1  (higher external score → lower risk)
  - EXT_SOURCE_2       +1
  - EXT_SOURCE_3       +1
  - inst_days_past_due_mean  -1  (more past-due days → higher risk)

Compares unconstrained baseline (best booster, same fixed HP) against
constrained model. Constraints are additive — dropping them if they hurt
Gini costs nothing.

Output: reports/lgb_monotone_constraint_test.json

Usage
-----
    .venv/bin/python scripts/lgb_monotone_constraint_test.py [--booster BOOSTER]

    BOOSTER defaults to best_booster from lgb_booster_comparison.json, or
    falls back to 'gbdt' if that file does not exist.
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
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_X_PATH = "data/processed/X_raw_features.parquet"
_Y_PATH = "data/processed/y_train.parquet"
_A6_RESULTS_PATH = "reports/lgb_booster_comparison.json"
_OUTPUT_PATH = "reports/lgb_monotone_constraint_test.json"

_MIN_ROWS = 100_000

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

_DART_EXTRA: dict = {"drop_rate": 0.1}
_GOSS_EXTRA: dict = {"top_rate": 0.1, "other_rate": 0.05}

_CV_N_SPLITS = 5
_TEMPORAL_SORT_COL = "prev_days_decision_mean"
_CV_EMBARGO_FRAC = 0.02

# All 7 directional features from domain knowledge
MONOTONE_CONSTRAINTS: dict[str, int] = {
    "AGE_YEARS": 1,
    "YEARS_EMPLOYED": 1,
    "CREDIT_INCOME_RATIO": -1,
    "EXT_SOURCE_1": 1,
    "EXT_SOURCE_2": 1,
    "EXT_SOURCE_3": 1,
    "inst_days_past_due_mean": -1,
}


# ---------------------------------------------------------------------------
# CV helpers (duplicated from lgb_booster_comparison for script independence)
# ---------------------------------------------------------------------------

class _TemporalCV:
    def __init__(self, n_splits: int, embargo_frac: float, random_state: int = 42) -> None:
        self.n_splits = n_splits
        self.embargo_frac = embargo_frac
        self._skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

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
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------

def _cv_gini(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    cv,
    groups: np.ndarray | None,
    label: str,
) -> tuple[float, float]:
    """Run CV and return (mean_gini, std_gini)."""
    booster = params.get("boosting_type", "gbdt")
    use_es = booster != "dart"
    callbacks = [lgb.log_evaluation(period=0)]
    if use_es:
        callbacks.insert(0, lgb.early_stopping(stopping_rounds=20, verbose=False))

    fold_ginis: list[float] = []
    for fold_i, (train_idx, val_idx) in enumerate(cv.split(X.values, y.values, groups)):
        model = lgb.LGBMClassifier(**params)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Early stopping is not available in dart mode")
            model.fit(
                X.iloc[train_idx], y.iloc[train_idx],
                eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                callbacks=callbacks,
            )
        y_prob = model.predict_proba(X.iloc[val_idx])[:, 1]
        gini = 2.0 * float(roc_auc_score(y.iloc[val_idx], y_prob)) - 1.0
        fold_ginis.append(gini)
        print(f"  {label} fold {fold_i + 1}/{_CV_N_SPLITS}: Gini={gini:.4f}")

    return float(np.mean(fold_ginis)), float(np.std(fold_ginis))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_monotone_constraint_test(booster: str = "gbdt") -> dict:
    """
    Compare unconstrained vs. constrained LightGBM under temporal CV.

    Parameters
    ----------
    booster : str
        Booster algorithm to use. Should be the best from A6.

    Returns
    -------
    dict
        Results with unconstrained/constrained mean Gini and delta.
    """
    print("Loading data...")
    X = pd.read_parquet(_X_PATH)
    y = pd.read_parquet(_Y_PATH).squeeze()

    assert X.shape[0] >= _MIN_ROWS, (
        f"Pre-flight check failed: {X.shape[0]} rows, expected >= {_MIN_ROWS}"
    )
    assert X.isnull().sum().sum() == 0, "NaN values found in X_raw_features."
    print(f"Data loaded: {X.shape[0]:,} rows × {X.shape[1]} cols, booster={booster.upper()}")

    # Validate constraint keys against actual feature columns
    missing_cols = [c for c in MONOTONE_CONSTRAINTS if c not in X.columns]
    if missing_cols:
        print(f"  Warning: {len(missing_cols)} constraint features not in X: {missing_cols}")
        print("  These constraints will be skipped.")
    active_constraints = {k: v for k, v in MONOTONE_CONSTRAINTS.items() if k in X.columns}
    print(f"  Active monotone constraints: {len(active_constraints)}/{len(MONOTONE_CONSTRAINTS)}")
    for col, direction in active_constraints.items():
        print(f"    {col}: {'+1 (increasing)' if direction == 1 else '-1 (decreasing)'}")

    # Build constraint list in column order
    cols = X.columns.tolist()
    constraint_list = [active_constraints.get(c, 0) for c in cols]

    # Temporal CV setup
    groups: np.ndarray | None = None
    if _TEMPORAL_SORT_COL in X.columns:
        groups = X[_TEMPORAL_SORT_COL].to_numpy()
    cv = _make_cv(groups, _CV_N_SPLITS)

    # Base params with booster
    base_params = {**_FIXED_HP, "boosting_type": booster}
    if booster == "dart":
        base_params.update(_DART_EXTRA)
    elif booster == "goss":
        base_params.update(_GOSS_EXTRA)

    # --- Unconstrained baseline ---
    print(f"\nRunning UNCONSTRAINED baseline ({booster.upper()})...")
    t0 = time.perf_counter()
    gini_unconstrained, std_unconstrained = _cv_gini(X, y, base_params, cv, groups, "UNCONSTRAINED")
    elapsed_unconstrained = round(time.perf_counter() - t0, 1)
    print(f"  → mean Gini={gini_unconstrained:.4f} ± {std_unconstrained:.4f}")

    # --- Constrained model ---
    constrained_params = {**base_params, "monotone_constraints": constraint_list}
    print(f"\nRunning CONSTRAINED model ({booster.upper()} + {len(active_constraints)} constraints)...")
    t0 = time.perf_counter()
    gini_constrained, std_constrained = _cv_gini(X, y, constrained_params, cv, groups, "CONSTRAINED")
    elapsed_constrained = round(time.perf_counter() - t0, 1)
    print(f"  → mean Gini={gini_constrained:.4f} ± {std_constrained:.4f}")

    delta = gini_constrained - gini_unconstrained
    verdict = "improvement" if delta >= 0.005 else ("negligible" if delta >= 0 else "degraded")
    print(f"\nDelta: {delta:+.4f} → {verdict}")

    result = {
        "booster": booster,
        "unconstrained": {"mean_gini": gini_unconstrained, "std_gini": std_unconstrained, "elapsed_sec": elapsed_unconstrained},
        "constrained": {"mean_gini": gini_constrained, "std_gini": std_constrained, "elapsed_sec": elapsed_constrained},
        "delta_gini": round(delta, 6),
        "verdict": verdict,
        "active_constraints": active_constraints,
        "missing_constraint_features": missing_cols,
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
        json.dump(result, fh, indent=2)
    print(f"Results saved to {_OUTPUT_PATH}")

    return result


def _resolve_booster(override: str | None) -> str:
    if override:
        return override
    a6_path = Path(_A6_RESULTS_PATH)
    if a6_path.exists():
        with a6_path.open() as fh:
            data = json.load(fh)
        best = data.get("best_booster", "gbdt")
        print(f"Using best booster from A6 results: {best.upper()}")
        return best
    print(f"A6 results not found at {_A6_RESULTS_PATH} — defaulting to 'gbdt'")
    return "gbdt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monotone constraint test for LightGBM")
    parser.add_argument("--booster", type=str, default=None, help="Booster type override")
    args = parser.parse_args()
    booster = _resolve_booster(args.booster)
    run_monotone_constraint_test(booster=booster)
