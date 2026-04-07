#!/usr/bin/env python3
"""
Learning Curve Analysis — Phase 04.2 Plan 05.

Measures OOF Gini at 50%, 75%, and 100% training fractions using the best
XGBoost HPO params.  Saturation check: if delta_75_100 < 0.005, the model
has plateaued and additional data will not close the Gini gap.

Output:
    reports/learning_curve_results.json
    reports/figures/learning_curve.png

Usage:
    python scripts/learning_curve_analysis.py
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

if "credit_engine" not in sys.modules:
    import src  # noqa: F401
    sys.modules["credit_engine"] = sys.modules["src"]

from credit_engine.model import _make_cv  # noqa: E402
from credit_engine.utils import gini_coefficient  # noqa: E402

warnings.filterwarnings("ignore")

_RANDOM_STATE = 42
_N_FOLDS = 5
_TEMPORAL_SORT_COL = "prev_days_decision_mean"
_SATURATION_THRESHOLD = 0.005
_LEARNING_FRACTIONS = [0.50, 0.75, 1.00]


def _oof_gini_at_fraction(
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    params: dict,
    fraction: float,
    scale_pos_weight: float,
) -> float:
    """
    Generate OOF Gini on the first `fraction` of the training data.

    Subsampling is done before CV to measure whether Gini is still
    improving as data volume increases.

    Parameters
    ----------
    X : pd.DataFrame
        Full training feature matrix (reset index expected).
    y : pd.Series
        Binary target aligned with X.
    cv : CV splitter
        Temporal or stratified CV yielding integer indices into X.
    params : dict
        XGBoost best hyperparameters.
    fraction : float
        Fraction in (0, 1] to subsample from X.
    scale_pos_weight : float
        n_neg / n_pos imbalance ratio.

    Returns
    -------
    float
        OOF Gini coefficient.
    """
    n = int(len(X) * fraction)
    X_sub = X.iloc[:n].reset_index(drop=True)
    y_sub = y.iloc[:n].reset_index(drop=True)

    # Rebuild CV on the subsampled groups so folds are consistent
    if _TEMPORAL_SORT_COL in X_sub.columns:
        groups_sub = X_sub[_TEMPORAL_SORT_COL].to_numpy()
        cv_sub = _make_cv(groups_sub, n_splits=_N_FOLDS)
    else:
        from sklearn.model_selection import StratifiedKFold
        cv_sub = StratifiedKFold(n_splits=_N_FOLDS, shuffle=True, random_state=_RANDOM_STATE)

    oof = np.zeros(len(y_sub))

    for fold_idx, (train_idx, val_idx) in enumerate(cv_sub.split(X_sub)):
        X_tr, y_tr = X_sub.iloc[train_idx], y_sub.iloc[train_idx]
        X_val, y_val = X_sub.iloc[val_idx], y_sub.iloc[val_idx]

        spw_fold = float((y_tr == 0).sum()) / float(max((y_tr == 1).sum(), 1))

        model = xgb.XGBClassifier(
            **params,
            scale_pos_weight=spw_fold,
            eval_metric="auc",
            verbosity=0,
            random_state=_RANDOM_STATE,
            use_label_encoder=False,
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr)
        oof[val_idx] = model.predict_proba(X_val)[:, 1]
        print(f"    fold {fold_idx + 1}/{_N_FOLDS} done", flush=True)

    return float(2 * roc_auc_score(y_sub, oof) - 1)


def main() -> None:
    print("[LearningCurve] Loading data...", flush=True)
    X_raw = pd.read_parquet("data/processed/X_raw_features.parquet")
    y = pd.read_parquet("data/processed/y_train.parquet").squeeze()
    print(f"  X_raw shape: {X_raw.shape}", flush=True)

    X_train, _, y_train, _ = train_test_split(
        X_raw, y, test_size=0.2, stratify=y, random_state=_RANDOM_STATE
    )
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    print(f"  X_train: {X_train.shape}", flush=True)

    # Sort by temporal column so fraction slices are time-ordered
    if _TEMPORAL_SORT_COL in X_train.columns:
        sort_order = X_train[_TEMPORAL_SORT_COL].argsort()
        X_train = X_train.iloc[sort_order].reset_index(drop=True)
        y_train = y_train.iloc[sort_order].reset_index(drop=True)
        print(f"  Sorted by {_TEMPORAL_SORT_COL} for temporal integrity", flush=True)

    # Load best XGB params from HPO
    hpo_results = json.load(open("reports/xgb_hpo_results.json"))
    best_params = hpo_results["best_params"]
    print(f"  Loaded best params: n_estimators={best_params.get('n_estimators')}, "
          f"learning_rate={best_params.get('learning_rate'):.4f}", flush=True)

    scale_pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())
    print(f"  scale_pos_weight: {scale_pos_weight:.4f}", flush=True)

    groups_train = (
        X_train[_TEMPORAL_SORT_COL].to_numpy()
        if _TEMPORAL_SORT_COL in X_train.columns
        else None
    )
    cv = _make_cv(groups_train, n_splits=_N_FOLDS)

    # ── Evaluate Gini at each fraction ──────────────────────────────────────
    results_list = []
    for frac in _LEARNING_FRACTIONS:
        n_samples = int(len(X_train) * frac)
        print(f"\n[LearningCurve] Fraction {frac:.0%} (n={n_samples:,})", flush=True)
        gini = _oof_gini_at_fraction(
            X_train, y_train, cv, best_params, frac, scale_pos_weight
        )
        print(f"  => OOF Gini: {gini:.4f}", flush=True)
        results_list.append({
            "fraction": frac,
            "n_samples": n_samples,
            "oof_gini": round(gini, 6),
        })

    # ── Saturation analysis ──────────────────────────────────────────────────
    g50, g75, g100 = [r["oof_gini"] for r in results_list]
    delta_50_75 = round(g75 - g50, 6)
    delta_75_100 = round(g100 - g75, 6)
    is_saturated = delta_75_100 < _SATURATION_THRESHOLD

    saturation = {
        "gini_50pct": g50,
        "gini_75pct": g75,
        "gini_100pct": g100,
        "delta_50_75": delta_50_75,
        "delta_75_100": delta_75_100,
        "saturation_threshold": _SATURATION_THRESHOLD,
        "is_saturated": is_saturated,
        "interpretation": (
            "Model has plateaued — more data will not close the gap"
            if is_saturated
            else "Model still improving with data — ceiling not yet reached"
        ),
    }

    output = {
        "phase": "04.2",
        "plan": "05",
        "model": "xgboost_hpo_best",
        "learning_curve": results_list,
        "saturation_check": saturation,
    }

    Path("reports").mkdir(exist_ok=True)
    with open("reports/learning_curve_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n[LearningCurve] Saved reports/learning_curve_results.json", flush=True)

    # ── Plot ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        fracs = [r["fraction"] for r in results_list]
        ginis = [r["oof_gini"] for r in results_list]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(fracs, ginis, "o-", linewidth=2, markersize=8, label="XGB HPO (OOF Gini)")
        ax.axhline(0.57, color="r", linestyle="--", label="Exit gate target (0.57)")
        ax.axhline(0.5519, color="g", linestyle="--", label="Phase 04.1 best (0.5519)")

        for r in results_list:
            ax.annotate(
                f"{r['oof_gini']:.4f}",
                xy=(r["fraction"], r["oof_gini"]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )

        ax.set_xlabel("Training Fraction", fontsize=12)
        ax.set_ylabel("OOF Gini Coefficient", fontsize=12)
        ax.set_title("Learning Curve: XGBoost (Best HPO Params)", fontsize=14)
        ax.set_xlim([0.40, 1.10])
        ax.set_ylim([max(0.50, min(ginis) - 0.02), min(0.60, max(ginis) + 0.025)])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        fig.tight_layout()

        Path("reports/figures").mkdir(parents=True, exist_ok=True)
        fig.savefig("reports/figures/learning_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("[LearningCurve] Saved reports/figures/learning_curve.png", flush=True)
    except Exception as e:
        print(f"[LearningCurve] Plot skipped: {e}", flush=True)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LEARNING CURVE SUMMARY")
    print("=" * 60)
    print(f"  50% data  → Gini {g50:.4f}")
    print(f"  75% data  → Gini {g75:.4f}  (delta: {delta_50_75:+.4f})")
    print(f" 100% data  → Gini {g100:.4f}  (delta: {delta_75_100:+.4f})")
    print(f"  Saturated: {is_saturated}  (threshold: {_SATURATION_THRESHOLD})")
    print(f"  {saturation['interpretation']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
