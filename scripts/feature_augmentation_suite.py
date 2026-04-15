#!/usr/bin/env python3
"""
Feature Augmentation Comparison Suite — Phase 04.2 Plan 03.

Evaluates 4 augmentation strategies × 3 models = 12 (model, augmentation)
comparison cells.  Each augmentation is applied per-fold inside CV to prevent
data leakage.  Uses fixed HPO-tuned best params for each model (not re-HPO)
so the Gini delta isolates the augmentation effect.

Augmentation strategies:
    rank_norm        — per-fold min-max rank normalization [0, 1]
    poly_interactions — top-15 SHAP degree-2 interactions, IV gate >= 0.02
    pseudo_labeling  — high-confidence holdout pseudo-labels in training fold
    target_encoding  — fold-safe target encoding of 4 categorical columns

Data loading:
    Loads X_raw_features.parquet (full 307K rows, intact index) and
    reconstructs the same 80/20 split used by xgb_optuna_hpo.py.
    This avoids the index=False misalignment bug in X_xgb_features.parquet.

Command:
    python scripts/feature_augmentation_suite.py

Output:
    reports/feature_augmentation_results.json
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

if "credit_engine" not in sys.modules:
    import src
    sys.modules["credit_engine"] = src

from credit_engine.features import (
    apply_polynomial_interactions,
    polynomial_interactions_from_shap,
    pseudo_label_from_predictions,
    rank_normalize_fold_safe,
)
from credit_engine.model import _make_cv, apply_target_encoding_fold_safe

warnings.filterwarnings("ignore")

# ============================================================================
# Constants
# ============================================================================

RANDOM_STATE = 42
N_FOLDS = 5
_TEMPORAL_SORT_COL = "prev_days_decision_mean"

# Categorical columns for target encoding
_CAT_COLS = ["CODE_GENDER", "NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE", "ORGANIZATION_TYPE"]

# Confidence thresholds for pseudo-labeling
_PSEUDO_LOW = 0.05
_PSEUDO_HIGH = 0.95
_PSEUDO_MAX_ROWS = 5000  # Cap synthetic rows to avoid memory/time blowout

# Polynomial interactions: top features and IV gate
_POLY_TOP_N = 15
_POLY_IV_GATE = 0.02

# XGB best params from Plan 02 HPO (reports/xgb_hpo_results.json)
_XGB_BEST_PARAMS = {
    "n_estimators": 972,
    "max_depth": 3,
    "learning_rate": 0.030561582197053343,
    "subsample": 0.6984393769376698,
    "colsample_bytree": 0.744757389328118,
    "min_child_weight": 4,
    "gamma": 1.0675942522425461,
    "max_delta_step": 0,
    "reg_alpha": 3.540144922650167,
    "reg_lambda": 6.802800085821854,
}

# LGB best params from models/lightgbm_params.json
_LGB_BEST_PARAMS = {
    "num_leaves": 146,
    "max_depth": 12,
    "learning_rate": 0.13335802858293552,
    "n_estimators": 232,
    "min_child_samples": 63,
    "subsample": 0.888107774227634,
    "colsample_bytree": 0.9103111800824533,
    "reg_alpha": 2.4664443604295787,
    "reg_lambda": 13.445831341258797,
}

# CatBoost best params from models/catboost_params.json
_CAT_BEST_PARAMS = {
    "depth": 5,
    "learning_rate": 0.1785436060870726,
    "l2_leaf_reg": 14.907884894416696,
    "bagging_temperature": 0.5986584841970366,
    "random_strength": 0.15601864044243652,
    "iterations": 500,
    "bootstrap_type": "Bayesian",
}

# ============================================================================
# OOF Evaluation Core
# ============================================================================


def _oof_gini(
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    model_class,
    model_params: dict,
    augmentation: str,
    X_pseudo_source: pd.DataFrame | None = None,
    scale_pos_weight: float | None = None,
) -> float:
    """
    Run OOF cross-validation with a given augmentation and return OOF Gini.

    Parameters
    ----------
    X : pd.DataFrame
        Training feature matrix (full 80% split).
    y : pd.Series
        Binary target aligned with X.
    cv : _TemporalCV or StratifiedKFold
        Cross-validation splitter.
    model_class : type
        XGBClassifier, LGBMClassifier, or CatBoostClassifier.
    model_params : dict
        Fixed best hyperparameters (no HPO).
    augmentation : str
        One of: 'none', 'rank_norm', 'poly', 'pseudo', 'target_enc'.
    X_pseudo_source : pd.DataFrame, optional
        Holdout feature matrix used as pseudo-label source.
        Required when augmentation == 'pseudo'.
    scale_pos_weight : float, optional
        Class imbalance weight for XGB/LGB.

    Returns
    -------
    float
        OOF Gini = 2 * AUC - 1.
    """
    oof_preds = np.zeros(len(y))
    X_arr = X.reset_index(drop=True)
    y_arr = y.reset_index(drop=True)

    # Pre-compute pseudo-label predictions on holdout using a quick preliminary
    # model trained on full X (for pseudo_labeling augmentation).
    prelim_pseudo_preds = None
    if augmentation == "pseudo" and X_pseudo_source is not None:
        print("    [pseudo] Training preliminary model for holdout predictions...")
        prelim_params = {**model_params}
        # Reduce estimators for speed
        for k in ("n_estimators", "iterations", "num_leaves"):
            if k in prelim_params:
                prelim_params[k] = min(prelim_params[k], 100)
        prelim_params.pop("bootstrap_type", None)  # only for CatBoost Bayesian
        try:
            prelim_params["bootstrap_type"] = "Bayesian"
        except Exception:
            pass

        prelim_model = model_class(**_build_model_kwargs(model_class, prelim_params, scale_pos_weight))
        prelim_model.fit(X_arr, y_arr)
        prelim_pseudo_preds = prelim_model.predict_proba(X_pseudo_source)[:, 1]

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_arr)):
        X_tr = X_arr.iloc[train_idx].copy()
        y_tr = y_arr.iloc[train_idx].copy()
        X_val = X_arr.iloc[val_idx].copy()
        y_val = y_arr.iloc[val_idx].copy()

        # ── Apply augmentation per-fold ────────────────────────────────────
        if augmentation == "rank_norm":
            excl = [_TEMPORAL_SORT_COL] if _TEMPORAL_SORT_COL in X_tr.columns else []
            X_tr, X_val = rank_normalize_fold_safe(X_tr, X_val, exclude_cols=excl)

        elif augmentation == "poly":
            # Fit a fast preliminary model on training fold for SHAP importance
            fast_params = {**model_params}
            for k in ("n_estimators", "iterations"):
                if k in fast_params:
                    fast_params[k] = min(fast_params[k], 50)
            fast_params_clean = {k: v for k, v in fast_params.items()
                                 if k not in ("bootstrap_type",)}
            try:
                fast_model = model_class(
                    **_build_model_kwargs(model_class, fast_params, scale_pos_weight)
                )
                fast_model.fit(X_tr, y_tr)
                X_tr, pairs = polynomial_interactions_from_shap(
                    X_tr, y_tr, fast_model,
                    top_n=_POLY_TOP_N, iv_gate=_POLY_IV_GATE,
                )
                X_val = apply_polynomial_interactions(X_val, pairs)
            except Exception as exc:
                print(f"    [poly] Fold {fold_idx}: interaction step failed ({exc}); skipping")

        elif augmentation == "pseudo" and prelim_pseudo_preds is not None:
            X_pseudo_fold, y_pseudo_fold = pseudo_label_from_predictions(
                prelim_pseudo_preds,
                X_pseudo_source,
                confidence_threshold=(_PSEUDO_LOW, _PSEUDO_HIGH),
                max_synthetic_rows=_PSEUDO_MAX_ROWS,
                random_state=RANDOM_STATE + fold_idx,
            )
            # Ensure column alignment (pseudo source might have different index)
            common_cols = [c for c in X_tr.columns if c in X_pseudo_fold.columns]
            X_pseudo_aligned = X_pseudo_fold[common_cols].copy()
            # Pad missing columns with -999
            for col in X_tr.columns:
                if col not in X_pseudo_aligned.columns:
                    X_pseudo_aligned[col] = -999.0
            X_pseudo_aligned = X_pseudo_aligned[X_tr.columns]
            X_tr = pd.concat(
                [X_tr, X_pseudo_aligned], axis=0, ignore_index=True
            )
            y_tr = pd.concat([y_tr, y_pseudo_fold], axis=0, ignore_index=True)

        elif augmentation == "target_enc":
            cat_present = [c for c in _CAT_COLS if c in X_tr.columns]
            if cat_present:
                X_tr, X_val = apply_target_encoding_fold_safe(
                    X_tr, y_tr, X_val, cat_cols=cat_present
                )

        # ── Train and predict ─────────────────────────────────────────────
        kwargs = _build_model_kwargs(model_class, model_params, scale_pos_weight)
        model = model_class(**kwargs)

        # Handle CatBoost fit kwargs
        fit_kwargs: dict = {}
        if model_class.__name__ == "CatBoostClassifier":
            fit_kwargs["verbose"] = 0
            cat_present = [c for c in _CAT_COLS if c in X_tr.columns]
            if cat_present and augmentation != "target_enc":
                fit_kwargs["cat_features"] = cat_present

        model.fit(X_tr, y_tr, **fit_kwargs)

        val_idx_arr = np.arange(len(y_arr))[val_idx]
        oof_preds[val_idx_arr] = model.predict_proba(X_val)[:, 1]

    return float(2 * roc_auc_score(y_arr, oof_preds) - 1)


def _build_model_kwargs(model_class, params: dict, scale_pos_weight: float | None) -> dict:
    """Return model constructor kwargs appropriate for the model class."""
    name = model_class.__name__
    kwargs = {k: v for k, v in params.items()}

    if name == "XGBClassifier":
        kwargs["eval_metric"] = "auc"
        kwargs["verbosity"] = 0
        kwargs["random_state"] = RANDOM_STATE
        kwargs["use_label_encoder"] = False
        if scale_pos_weight is not None:
            kwargs["scale_pos_weight"] = scale_pos_weight

    elif name == "LGBMClassifier":
        kwargs["verbose"] = -1
        kwargs["random_state"] = RANDOM_STATE
        kwargs["is_unbalance"] = True

    elif name == "CatBoostClassifier":
        kwargs["random_seed"] = RANDOM_STATE
        kwargs["allow_writing_files"] = False
        kwargs["verbose"] = 0
        if scale_pos_weight is not None:
            kwargs["scale_pos_weight"] = scale_pos_weight

    return kwargs


# ============================================================================
# Main
# ============================================================================


def main() -> dict:
    """Run the 12-cell feature augmentation comparison suite."""

    print("[AugSuite] Loading feature data...")
    X_raw = pd.read_parquet("data/processed/X_raw_features.parquet")
    y = pd.read_parquet("data/processed/y_train.parquet").squeeze()

    print(f"  X_raw shape : {X_raw.shape}")
    print(f"  y shape     : {y.shape}")
    print(f"  Default rate: {y.mean():.4f}")

    # Reconstruct same split as xgb_optuna_hpo.py — avoids index=False bug
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    # Temporal CV groups (same column used in HPO)
    groups_train = (
        X_train[_TEMPORAL_SORT_COL].to_numpy()
        if _TEMPORAL_SORT_COL in X_train.columns
        else None
    )
    cv = _make_cv(groups_train, n_splits=N_FOLDS)

    scale_pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())
    print(f"  scale_pos_weight: {scale_pos_weight:.4f}")

    # ------------------------------------------------------------------
    # Import model classes (lazy to avoid slow imports at module level)
    # ------------------------------------------------------------------
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier

    model_registry = {
        "xgboost": (xgb.XGBClassifier, _XGB_BEST_PARAMS, scale_pos_weight),
        "lightgbm": (lgb.LGBMClassifier, _LGB_BEST_PARAMS, None),  # is_unbalance handles it
        "catboost": (CatBoostClassifier, _CAT_BEST_PARAMS, scale_pos_weight),
    }

    augmentations = ["none", "rank_norm", "poly", "pseudo", "target_enc"]

    results = {
        "phase": "04.2",
        "plan": "03",
        "n_folds": N_FOLDS,
        "comparison_table": [],
        "baseline_ginis": {},
        "augmentation_details": {},
        "summary": {},
    }

    # ------------------------------------------------------------------
    # Evaluate each (model, augmentation) cell
    # ------------------------------------------------------------------
    for model_name, (model_class, best_params, spw) in model_registry.items():
        print(f"\n{'='*60}")
        print(f"[AugSuite] Model: {model_name.upper()}")
        print(f"{'='*60}")

        baseline_gini: float | None = None

        for aug_name in augmentations:
            print(f"\n  Augmentation: {aug_name}")

            try:
                gini = _oof_gini(
                    X=X_train,
                    y=y_train,
                    cv=cv,
                    model_class=model_class,
                    model_params=best_params,
                    augmentation=aug_name,
                    X_pseudo_source=X_test if aug_name == "pseudo" else None,
                    scale_pos_weight=spw,
                )
            except Exception as exc:
                print(f"  [ERROR] {model_name}/{aug_name} failed: {exc}")
                gini = float("nan")

            if aug_name == "none":
                baseline_gini = gini
                results["baseline_ginis"][model_name] = gini
                print(f"  Baseline OOF Gini: {gini:.4f}")
            else:
                delta = (gini - baseline_gini) if baseline_gini is not None else float("nan")
                improvement = bool(delta > 0.001)
                print(f"  OOF Gini: {gini:.4f} (Δ {delta:+.4f}, {'IMPROVE' if improvement else 'no change'})")

                results["comparison_table"].append(
                    {
                        "model": model_name,
                        "augmentation": aug_name,
                        "baseline_gini": round(baseline_gini or 0.0, 6),
                        "augmented_gini": round(gini, 6),
                        "delta_gini": round(delta, 6),
                        "improvement": improvement,
                    }
                )

        # Best augmentation per model
        model_rows = [r for r in results["comparison_table"] if r["model"] == model_name]
        if model_rows:
            best_row = max(model_rows, key=lambda r: r["delta_gini"])
            results["augmentation_details"][model_name] = {
                "baseline_gini": results["baseline_ginis"].get(model_name),
                "best_augmentation": best_row["augmentation"],
                "best_augmented_gini": best_row["augmented_gini"],
                "best_delta": best_row["delta_gini"],
            }

    # ------------------------------------------------------------------
    # Summary: overall best augmentation
    # ------------------------------------------------------------------
    if results["comparison_table"]:
        best_overall = max(results["comparison_table"], key=lambda r: r["delta_gini"])
        all_positive = [r for r in results["comparison_table"] if r["improvement"]]
        results["summary"] = {
            "best_overall_model": best_overall["model"],
            "best_overall_augmentation": best_overall["augmentation"],
            "best_overall_delta": best_overall["delta_gini"],
            "n_improving_cells": len(all_positive),
            "n_total_cells": len(results["comparison_table"]),
        }

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output_path = Path("reports/feature_augmentation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[AugSuite] Saved results to {output_path}")

    print(f"\n{'='*60}")
    print("[AugSuite] COMPLETE — Comparison Table (12 cells)")
    print(f"{'='*60}")
    print(f"{'Model':<12} {'Augmentation':<18} {'Baseline':>10} {'Augmented':>10} {'Delta':>8} {'Improve':>8}")
    print("-" * 70)
    for row in results["comparison_table"]:
        print(
            f"{row['model']:<12} {row['augmentation']:<18} "
            f"{row['baseline_gini']:>10.4f} {row['augmented_gini']:>10.4f} "
            f"{row['delta_gini']:>+8.4f} {'YES' if row['improvement'] else 'no':>8}"
        )

    if results.get("summary"):
        s = results["summary"]
        print(f"\nBest: {s['best_overall_model']} + {s['best_overall_augmentation']} "
              f"(Δ {s['best_overall_delta']:+.4f})")
        print(f"Improving cells: {s['n_improving_cells']}/{s['n_total_cells']}")

    return results


if __name__ == "__main__":
    main()
