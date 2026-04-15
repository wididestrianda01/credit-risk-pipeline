"""
Ensemble combination ablation study.

Tests all 2- and 3-model subsets of (LGB, XGB, CatBoost) using OOF
predictions already stored in ensemble_best.pkl.  No re-training needed.

For each combination, two meta-strategies are evaluated:
  - logistic  : L2-regularised logistic meta-learner (C=1.0) trained on OOF stack
  - avg       : simple unweighted average of base-model OOT predictions

Results are sorted by OOT Gini and printed as a ranked table.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import load_model
from src.utils import gini_coefficient, ks_statistic

# ---------------------------------------------------------------------------
# Constants (must match run_ensemble.py)
# ---------------------------------------------------------------------------
_TEMPORAL_SORT_COL = "prev_days_decision_mean"
_TEST_SIZE = 0.20
_RANDOM_SEED = 42
_DATA_DIR = ROOT / "data" / "processed"
_MODELS_DIR = ROOT / "models"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _temporal_split(parquet_path: Path) -> tuple:
    """Reproduce the exact 80/20 temporal split used in run_ensemble.py."""
    df = pd.read_parquet(parquet_path)
    X = df.drop(columns=["TARGET"])
    y = df["TARGET"]

    sort_col = X[_TEMPORAL_SORT_COL].copy()
    nan_mask = sort_col.isna()
    np.random.seed(_RANDOM_SEED)
    nan_indices = np.where(nan_mask)[0]
    max_val = sort_col[~nan_mask].max() if (~nan_mask).any() else 0
    sort_key = sort_col.copy()
    sort_key[nan_mask] = max_val + np.arange(len(nan_indices))

    sorted_idx = sort_key.argsort().values
    X_s = X.iloc[sorted_idx].reset_index(drop=True)
    y_s = y.iloc[sorted_idx].reset_index(drop=True)

    n_oot = int(np.ceil(len(X_s) * _TEST_SIZE))
    n_train = len(X_s) - n_oot

    X_train = X_s.iloc[:n_train].copy()
    y_train = y_s.iloc[:n_train].copy()
    X_oot   = X_s.iloc[n_train:].copy()
    y_oot   = y_s.iloc[n_train:].copy()

    return X_train, y_train, X_oot, y_oot


def _base_oot_preds(model, X_oot: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return per-model OOT probability predictions."""
    return {
        "lgb": model.lgb_model.predict_proba(X_oot)[:, 1],
        "xgb": model.xgb_model.predict_proba(X_oot)[:, 1],
        "cat": model.cat_model.predict_proba(X_oot.to_numpy())[:, 1],
    }


def _fit_meta(oof_stack: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=_RANDOM_SEED)
    lr.fit(oof_stack, y_train)
    return lr


def _evaluate(y_true: pd.Series, y_prob: np.ndarray) -> dict:
    gini = gini_coefficient(y_true.to_numpy(), y_prob)
    ks, _ = ks_statistic(y_true.to_numpy(), y_prob)
    return {"gini": round(gini, 4), "ks": round(ks, 4)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("Ensemble Combination Ablation Study")
    print("=" * 70)

    # 1. Load stored OOF arrays and base models
    print("\n[1/3] Loading ensemble_best.pkl ...")
    ensemble_model = load_model(str(_MODELS_DIR / "ensemble_best.pkl"))
    oof = {
        "lgb": ensemble_model.oof_lgb,
        "xgb": ensemble_model.oof_xgb,
        "cat": ensemble_model.oof_cat,
    }
    print(f"  OOF arrays loaded — {len(oof['lgb'])} training rows each")

    # 2. Reconstruct OOT split
    print("\n[2/3] Reconstructing temporal split from X_tree_raw.parquet ...")
    X_train, y_train, X_oot, y_oot = _temporal_split(
        _DATA_DIR / "X_tree_raw.parquet"
    )
    print(f"  Train: {len(X_train)} | OOT: {len(X_oot)} | OOT positives: {y_oot.mean():.2%}")

    # 3. Compute per-model OOT predictions
    print("\n[3/3] Generating base-model OOT predictions ...")
    oot_preds = _base_oot_preds(ensemble_model, X_oot)
    print("  Base-model OOT Gini:")
    for name, preds in oot_preds.items():
        metrics = _evaluate(y_oot, preds)
        print(f"    {name.upper():8s}  Gini={metrics['gini']:.4f}  KS={metrics['ks']:.4f}")

    # ---------------------------------------------------------------------------
    # Ablation: all subsets of size ≥ 2
    # ---------------------------------------------------------------------------
    models = ["lgb", "xgb", "cat"]
    results = []

    # Single-model baselines
    for name in models:
        m = _evaluate(y_oot, oot_preds[name])
        results.append({
            "combo": name.upper(),
            "strategy": "—",
            "n_models": 1,
            "gini": m["gini"],
            "ks": m["ks"],
            "coefs": "—",
        })

    # Pairs and triples
    for r in (2, 3):
        for subset in combinations(models, r):
            label = "+".join(m.upper() for m in subset)

            # ---- logistic meta-learner ----
            oof_stack = np.column_stack([oof[m] for m in subset])
            oot_stack = np.column_stack([oot_preds[m] for m in subset])

            meta = _fit_meta(oof_stack, y_train.to_numpy())
            oot_meta_prob = meta.predict_proba(oot_stack)[:, 1]
            m_logistic = _evaluate(y_oot, oot_meta_prob)

            coef_str = "  ".join(
                f"{name.upper()}={c:.3f}"
                for name, c in zip(subset, meta.coef_[0])
            )
            results.append({
                "combo": label,
                "strategy": "logistic",
                "n_models": r,
                "gini": m_logistic["gini"],
                "ks": m_logistic["ks"],
                "coefs": coef_str,
            })

            # ---- simple average ----
            oot_avg_prob = np.mean(oot_stack, axis=1)
            m_avg = _evaluate(y_oot, oot_avg_prob)
            results.append({
                "combo": label,
                "strategy": "avg",
                "n_models": r,
                "gini": m_avg["gini"],
                "ks": m_avg["ks"],
                "coefs": "—",
            })

    # ---------------------------------------------------------------------------
    # Print ranked table
    # ---------------------------------------------------------------------------
    df = pd.DataFrame(results).sort_values("gini", ascending=False).reset_index(drop=True)
    df.index += 1  # 1-based rank

    print("\n" + "=" * 70)
    print("RESULTS — ranked by OOT Gini")
    print("=" * 70)
    print(f"{'Rank':<5} {'Combination':<18} {'Strategy':<10} {'Gini':>7} {'KS':>7}  Meta-learner coefficients")
    print("-" * 70)
    for rank, row in df.iterrows():
        marker = " ◀ best" if rank == 1 else ""
        print(
            f"{rank:<5} {row['combo']:<18} {row['strategy']:<10} "
            f"{row['gini']:>7.4f} {row['ks']:>7.4f}  {row['coefs']}{marker}"
        )
    print("=" * 70)

    # Save
    out_path = ROOT / "reports" / "ensemble_ablation.csv"
    df.to_csv(out_path, index_label="rank")
    print(f"\nSaved to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
