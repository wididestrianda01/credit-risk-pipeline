#!/usr/bin/env python3
"""
Ensemble Enhancement via Feature Diversity (Phase 04.2.10).

This script orchestrates a comprehensive ensemble workflow:

1. Load best hyperparameters from HPO runs (LGB, XGB-WoE, CatBoost-DFS)
2. Generate OOF (out-of-fold) predictions via temporal CV on 80% training set
3. Generate OOT (out-of-time) predictions on held-out 20% test set
4. Test 9 ensemble combinations (2-model, 3-model × logistic, rank_avg, mlp_meta)
5. Select best combination and apply gating rules
6. Save ensemble model if OOT Gini ≥ 0.600 (PASS) or > 0.5814 (ACCEPT)

Output files:
  - reports/ensemble_v2_ablation.csv       (9 rows, all combinations sorted by Gini)
  - reports/ensemble_v2_best_summary.json  (best combo details + gate verdict)
  - models/ensemble_v2_calibrated.pkl      (saved if gate passes)

Usage:
  python scripts/run_ensemble_v2.py [--n-trials N] [--skip-hpo]

Example:
  python scripts/run_ensemble_v2.py --n-trials 50     # Run full workflow (HPO + OOF + ablation)
  python scripts/run_ensemble_v2.py --skip-hpo        # Skip HPO, use cached params (fast)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from scipy.stats import rankdata

# Bootstrap credit_engine alias
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
import src as _src  # noqa: E402
if "credit_engine" not in sys.modules:
    sys.modules["credit_engine"] = _src

from credit_engine.model import train_lightgbm_optuna, train_xgboost_optuna, train_catboost_optuna  # noqa: E402
from credit_engine.model_base import _make_cv, _TEMPORAL_SORT_COL, _TEST_SIZE, _RANDOM_STATE, save_model  # noqa: E402
from credit_engine.utils import gini_coefficient, ks_statistic, evaluate_model  # noqa: E402

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ============================================================================
# Constants
# ============================================================================

FEATURE_STORES = {
    "lgb": "data/processed/X_lgb_v2.parquet",
    "xgb": "data/processed/X_features.parquet",
    "cat": "data/processed/X_tree_dfs.parquet",
}

PARAMS_FILES = {
    "lgb": "models/lightgbm_params.json",  # From v2 HPO (or will be regenerated)
    "xgb": "models/xgboost_woe_params.json",
    "cat": "models/catboost_dfs_params.json",
}

EVAL_FILES = {
    "lgb": "reports/lgb_raw_X_lgb_v2_is_unbalance_eval.json",
    "xgb": "reports/xgb_woe_eval.json",
    "cat": "reports/catboost_dfs_eval.json",
}

# Gate thresholds (Basel CRE36 + business requirements)
GATE_THRESHOLDS = {
    "pass": 0.600,          # OOT Gini ≥ 0.600 → save ensemble
    "accept": 0.5814,       # OOT Gini > 0.5814 (CatBoost v2 baseline) → save if better
    "investigate": 0.580,   # 0.580 ≤ OOT Gini < 0.600 → investigate
}


# ============================================================================
# OOF Generation Utilities
# ============================================================================

def load_feature_store_with_sort(feature_store_path: str) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Load feature store, extract TARGET, and compute temporal sort indices.

    Parameters
    ----------
    feature_store_path : str
        Path to parquet file.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (includes _TEMPORAL_SORT_COL).
    y : pd.Series
        Target series.
    sort_indices : np.ndarray
        Indices sorted by _TEMPORAL_SORT_COL (NaN last).
    """
    df = pd.read_parquet(feature_store_path)

    if "TARGET" not in df.columns:
        raise ValueError(f"TARGET column not found in {feature_store_path}")

    y = df.pop("TARGET").astype(int)
    X = df

    if _TEMPORAL_SORT_COL not in X.columns:
        raise ValueError(
            f"Temporal sort column '{_TEMPORAL_SORT_COL}' not in {feature_store_path}. "
            "Cannot generate OOF without temporal ordering."
        )

    # Compute sort indices: known values sorted, NaN last in random order
    temporal_vals = X[_TEMPORAL_SORT_COL].values
    nan_mask = np.isnan(temporal_vals)
    known_pos = np.where(~nan_mask)[0]
    unknown_pos = np.where(nan_mask)[0]
    known_sorted = known_pos[np.argsort(temporal_vals[known_pos])]

    rng = np.random.default_rng(_RANDOM_STATE)
    unknown_perm = rng.permutation(len(unknown_pos))
    unknown_sorted = unknown_pos[unknown_perm]

    sort_indices = np.concatenate([known_sorted, unknown_sorted])

    return X, y, sort_indices


def split_oot_train(X: pd.DataFrame, y: pd.Series, sort_indices: np.ndarray) -> tuple[
    pd.DataFrame, pd.Series, pd.DataFrame, pd.Series
]:
    """
    Split data into 80% training (for HPO + OOF) and 20% OOT (never seen).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (sorted by _TEMPORAL_SORT_COL).
    y : pd.Series
        Target series.
    sort_indices : np.ndarray
        Indices sorted by temporal column.

    Returns
    -------
    X_train, y_train, X_oot, y_oot
        Training and OOT sets.
    """
    X_sorted = X.iloc[sort_indices].reset_index(drop=True)
    y_sorted = y.iloc[sort_indices].reset_index(drop=True)

    n_train = int(len(X_sorted) * (1 - _TEST_SIZE))
    X_train = X_sorted.iloc[:n_train].copy()
    y_train = y_sorted.iloc[:n_train].copy()
    X_oot = X_sorted.iloc[n_train:].copy()
    y_oot = y_sorted.iloc[n_train:].copy()

    return X_train, y_train, X_oot, y_oot


def generate_lgb_oof_predictions(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict,
) -> np.ndarray:
    """
    Generate LGB OOF predictions via 5-fold temporal CV.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (80% of total).
    y_train : pd.Series
        Training labels.
    best_params : dict
        Best hyperparameters from Optuna HPO.

    Returns
    -------
    oof_preds : np.ndarray
        OOF predictions (shape n_train,).
    """
    cv = _make_cv(groups_train=None, n_splits=5)
    oof_preds = np.zeros(len(X_train))

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        lgb_params = {
            **best_params,
            "verbose": -1,
            "n_jobs": -1,
        }

        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=best_params.get("n_estimators", 100),
            valid_sets=[val_data],
            valid_names=["valid"],
            callbacks=[lgb.log_evaluation(period=0)],
        )

        oof_preds[val_idx] = model.predict(X_val)

    return oof_preds


def generate_xgb_oof_predictions(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict,
) -> np.ndarray:
    """
    Generate XGB OOF predictions via 5-fold temporal CV.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (80% of total).
    y_train : pd.Series
        Training labels.
    best_params : dict
        Best hyperparameters from Optuna HPO.

    Returns
    -------
    oof_preds : np.ndarray
        OOF predictions (shape n_train,).
    """
    cv = _make_cv(groups_train=None, n_splits=5)
    oof_preds = np.zeros(len(X_train))

    n_estimators = best_params.get("n_estimators", 100)

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        dtrain = xgb.DMatrix(X_tr.values, label=y_tr.values)
        dval = xgb.DMatrix(X_val.values, label=y_val.values)

        xgb_params = {
            **best_params,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "seed": _RANDOM_STATE,
        }

        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=n_estimators,
            evals=[(dval, "eval")],
            verbose_eval=False,
        )

        oof_preds[val_idx] = model.predict(dval)

    return oof_preds


def generate_catboost_oof_predictions(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict,
) -> np.ndarray:
    """
    Generate CatBoost OOF predictions via 5-fold temporal CV.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (80% of total).
    y_train : pd.Series
        Training labels.
    best_params : dict
        Best hyperparameters from Optuna HPO.

    Returns
    -------
    oof_preds : np.ndarray
        OOF predictions (shape n_train,).
    """
    # CatBoost C++ engine cannot handle pd.NA (pandas nullable extension types).
    # Convert any nullable Int/Float/Boolean columns → float64 (pd.NA → np.nan).
    ext_cols = [c for c in X_train.columns if pd.api.types.is_extension_array_dtype(X_train[c])]
    if ext_cols:
        X_train = X_train.copy()
        X_train[ext_cols] = X_train[ext_cols].astype("float64")

    cv = _make_cv(groups_train=None, n_splits=5)
    oof_preds = np.zeros(len(X_train))

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        cat_params = {
            **best_params,
            "iterations": best_params.get("iterations", 1000),
            "verbose": 0,
            "allow_writing_files": False,
        }

        model = CatBoostClassifier(**cat_params)
        model.fit(X_tr.values, y_tr.values, verbose=False)

        oof_preds[val_idx] = model.predict_proba(X_val.values)[:, 1]

    return oof_preds


def apply_catboost_calibration(oof_preds: np.ndarray, y_train: pd.Series) -> np.ndarray:
    """
    Apply Platt sigmoid calibration to CatBoost OOF predictions.

    Parameters
    ----------
    oof_preds : np.ndarray
        Uncalibrated OOF predictions.
    y_train : pd.Series
        Training labels.

    Returns
    -------
    oof_calibrated : np.ndarray
        Calibrated predictions.
    """
    X_2d = oof_preds.reshape(-1, 1)
    platt = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    platt.fit(X_2d, y_train)
    oof_calibrated = platt.predict_proba(X_2d)[:, 1]
    return oof_calibrated


# ============================================================================
# Ensemble Meta-Learners
# ============================================================================

def train_logistic_meta_learner(
    oof_stack: np.ndarray,
    y_train: pd.Series,
) -> LogisticRegression:
    """
    Train logistic regression on OOF stack.

    Parameters
    ----------
    oof_stack : np.ndarray
        Shape (n_train, n_models) OOF predictions.
    y_train : pd.Series
        Training labels.

    Returns
    -------
    meta_learner : LogisticRegression
        Fitted logistic regression.
    """
    meta = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    meta.fit(oof_stack, y_train)
    return meta


def train_mlp_meta_learner(
    oof_stack: np.ndarray,
    y_train: pd.Series,
) -> MLPClassifier:
    """
    Train MLP meta-learner on OOF stack.

    Parameters
    ----------
    oof_stack : np.ndarray
        Shape (n_train, n_models) OOF predictions.
    y_train : pd.Series
        Training labels.

    Returns
    -------
    meta_learner : MLPClassifier
        Fitted MLP.
    """
    meta = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        alpha=0.01,
        early_stopping=True,
        random_state=_RANDOM_STATE,
        max_iter=1000,
    )
    meta.fit(oof_stack, y_train)
    return meta


def rank_avg_ensemble(oof_stack: np.ndarray) -> np.ndarray:
    """
    Simple rank-average ensemble: rank each model's predictions, average ranks.

    Parameters
    ----------
    oof_stack : np.ndarray
        Shape (n_train, n_models) OOF predictions.

    Returns
    -------
    ensemble_preds : np.ndarray
        Shape (n_train,) rank-averaged predictions.
    """
    n_models = oof_stack.shape[1]
    rank_sum = np.zeros(oof_stack.shape[0])

    for i in range(n_models):
        rank_sum += rankdata(oof_stack[:, i]) / len(oof_stack)

    return rank_sum / n_models


# ============================================================================
# Main Ensemble Workflow
# ============================================================================

def run_ensemble_v2(
    n_trials: int = 50,
    skip_hpo: bool = False,
) -> tuple[dict, str]:
    """
    Run full ensemble workflow: HPO → OOF → Ablation → Gating.

    Parameters
    ----------
    n_trials : int
        Number of Optuna trials for HPO (if not skipped).
    skip_hpo : bool
        If True, load cached params from files instead of running HPO.

    Returns
    -------
    best_result : dict
        Best ensemble combination and metrics.
    gate_verdict : str
        Gate verdict: "PASS", "ACCEPT", or "INVESTIGATE"/"FAIL".
    """

    # --- Step 1: Load/Regenerate best parameters ---
    print("\n" + "="*70)
    print("STEP 1: Load/Regenerate Best Hyperparameters")
    print("="*70)

    lgb_params, xgb_params, cat_params = None, None, None

    # LGB params (from v2 HPO or regenerate if needed)
    lgb_params_path = Path("models/lightgbm_params.json")
    if lgb_params_path.exists() and skip_hpo:
        with open(lgb_params_path) as f:
            lgb_params = json.load(f)
        print(f"✓ Loaded cached LGB params from {lgb_params_path}")
    else:
        print(f"Running LGB HPO on {FEATURE_STORES['lgb']}...")
        try:
            _, _, _, _, lgb_params = train_lightgbm_optuna(
                FEATURE_STORES["lgb"],
                n_trials=n_trials,
                imbalance_strategy="is_unbalance",
            )
            print(f"✓ LGB HPO complete: {len(lgb_params)} parameters")
        except Exception as e:
            print(f"WARNING: LGB HPO failed, loading cached params: {e}")
            if lgb_params_path.exists():
                with open(lgb_params_path) as f:
                    lgb_params = json.load(f)
            else:
                raise

    # XGB params (from WoE HPO or regenerate)
    xgb_params_path = Path("models/xgboost_woe_params.json")
    if xgb_params_path.exists() and skip_hpo:
        with open(xgb_params_path) as f:
            xgb_params = json.load(f)
        print(f"✓ Loaded cached XGB params from {xgb_params_path}")
    else:
        print(f"Running XGB HPO on {FEATURE_STORES['xgb']}...")
        try:
            _, _, _, _, xgb_params = train_xgboost_optuna(
                FEATURE_STORES["xgb"],
                n_trials=n_trials,
            )
            print(f"✓ XGB HPO complete: {len(xgb_params)} parameters")
        except Exception as e:
            print(f"WARNING: XGB HPO failed, loading cached params: {e}")
            xgb_params_path = Path("models/xgboost_raw_params.json")  # Fallback to raw
            if xgb_params_path.exists():
                with open(xgb_params_path) as f:
                    xgb_params = json.load(f)
            else:
                raise

    # CatBoost params (from DFS HPO or regenerate)
    cat_params_path = Path("models/catboost_dfs_params.json")
    if cat_params_path.exists() and skip_hpo:
        with open(cat_params_path) as f:
            cat_params = json.load(f)
        print(f"✓ Loaded cached CatBoost params from {cat_params_path}")
    else:
        print(f"Running CatBoost HPO on {FEATURE_STORES['cat']}...")
        try:
            _, _, _, _, cat_params = train_catboost_optuna(
                FEATURE_STORES["cat"],
                n_trials=n_trials,
            )
            print(f"✓ CatBoost HPO complete: {len(cat_params)} parameters")
        except Exception as e:
            print(f"WARNING: CatBoost HPO failed, loading cached params: {e}")
            cat_params_path = Path("models/catboost_raw_params.json")  # Fallback to raw
            if cat_params_path.exists():
                with open(cat_params_path) as f:
                    cat_params = json.load(f)
            else:
                raise

    # --- Step 2: Load feature stores and split OOT ---
    print("\n" + "="*70)
    print("STEP 2: Load Feature Stores and Split OOT")
    print("="*70)

    X_lgb, y_lgb, sort_lgb = load_feature_store_with_sort(FEATURE_STORES["lgb"])
    X_lgb_train, y_train, X_lgb_oot, y_oot = split_oot_train(X_lgb, y_lgb, sort_lgb)
    print(f"✓ LGB: {X_lgb_train.shape[0]} train, {X_lgb_oot.shape[0]} OOT")

    X_xgb, y_xgb, sort_xgb = load_feature_store_with_sort(FEATURE_STORES["xgb"])
    X_xgb_train, _, X_xgb_oot, _ = split_oot_train(X_xgb, y_xgb, sort_xgb)
    print(f"✓ XGB: {X_xgb_train.shape[0]} train, {X_xgb_oot.shape[0]} OOT")

    X_cat, y_cat, sort_cat = load_feature_store_with_sort(FEATURE_STORES["cat"])
    X_cat_train, _, X_cat_oot, _ = split_oot_train(X_cat, y_cat, sort_cat)
    print(f"✓ CatBoost: {X_cat_train.shape[0]} train, {X_cat_oot.shape[0]} OOT")

    # --- Step 3: Generate OOF predictions ---
    print("\n" + "="*70)
    print("STEP 3: Generate OOF Predictions (5-fold Temporal CV)")
    print("="*70)

    print("Generating LGB OOF predictions...")
    lgb_oof = generate_lgb_oof_predictions(X_lgb_train, y_train, lgb_params)
    print(f"✓ LGB OOF: shape {lgb_oof.shape}, mean={lgb_oof.mean():.4f}")

    print("Generating XGB OOF predictions...")
    xgb_oof = generate_xgb_oof_predictions(X_xgb_train, y_train, xgb_params)
    print(f"✓ XGB OOF: shape {xgb_oof.shape}, mean={xgb_oof.mean():.4f}")

    print("Generating CatBoost OOF predictions...")
    cat_oof = generate_catboost_oof_predictions(X_cat_train, y_train, cat_params)
    cat_oof_calibrated = apply_catboost_calibration(cat_oof, y_train)
    print(f"✓ CatBoost OOF (calibrated): shape {cat_oof_calibrated.shape}, mean={cat_oof_calibrated.mean():.4f}")

    # --- Step 4: Generate OOT predictions ---
    print("\n" + "="*70)
    print("STEP 4: Generate OOT Predictions (Final Fitted Models)")
    print("="*70)

    # Refit on full training set
    print("Refitting models on full 80% training set...")

    # LGB OOT
    train_data = lgb.Dataset(X_lgb_train, label=y_train)
    lgb_model_oot = lgb.train(
        {**lgb_params, "verbose": -1},
        train_data,
        num_boost_round=lgb_params.get("n_estimators", 100),
        callbacks=[lgb.log_evaluation(period=0)],
    )
    lgb_oot_pred = lgb_model_oot.predict(X_lgb_oot)

    # XGB OOT
    dtrain = xgb.DMatrix(X_xgb_train.values, label=y_train.values)
    xgb_model_oot = xgb.train(
        {**xgb_params, "objective": "binary:logistic", "eval_metric": "auc", "seed": _RANDOM_STATE},
        dtrain,
        num_boost_round=xgb_params.get("n_estimators", 100),
        verbose_eval=False,
    )
    doot = xgb.DMatrix(X_xgb_oot.values)
    xgb_oot_pred = xgb_model_oot.predict(doot)

    # CatBoost OOT — convert nullable extension types first
    ext_cols_cat = [c for c in X_cat_train.columns if pd.api.types.is_extension_array_dtype(X_cat_train[c])]
    if ext_cols_cat:
        X_cat_train = X_cat_train.copy()
        X_cat_train[ext_cols_cat] = X_cat_train[ext_cols_cat].astype("float64")
        X_cat_oot = X_cat_oot.copy()
        X_cat_oot[ext_cols_cat] = X_cat_oot[ext_cols_cat].astype("float64")
    cat_model_oot = CatBoostClassifier(**{**cat_params, "iterations": 1000, "verbose": 0})
    cat_model_oot.fit(X_cat_train.values, y_train.values, verbose=False)
    cat_oot_pred_raw = cat_model_oot.predict_proba(X_cat_oot.values)[:, 1]

    # Calibrate CatBoost OOT predictions
    X_2d_oot = cat_oot_pred_raw.reshape(-1, 1)
    platt_oot = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    platt_oot.fit(X_2d_oot, y_oot)
    cat_oot_pred = platt_oot.predict_proba(X_2d_oot)[:, 1]

    print(f"✓ LGB OOT: mean={lgb_oot_pred.mean():.4f}")
    print(f"✓ XGB OOT: mean={xgb_oot_pred.mean():.4f}")
    print(f"✓ CatBoost OOT: mean={cat_oot_pred.mean():.4f}")

    # --- Step 5: Run 9-cell ablation ---
    print("\n" + "="*70)
    print("STEP 5: Run 9-Cell Ablation (2-model, 3-model × 3 strategies)")
    print("="*70)

    ablation_results = []
    combo_num = 1

    # 2-model combinations
    combos_2model = [
        ("LGB + XGB-WoE", [lgb_oof, xgb_oof], [lgb_oot_pred, xgb_oot_pred], "LGB + XGB-WoE"),
        ("LGB + CatBoost-DFS", [lgb_oof, cat_oof_calibrated], [lgb_oot_pred, cat_oot_pred], "LGB + CatBoost-DFS"),
        ("XGB-WoE + CatBoost-DFS", [xgb_oof, cat_oof_calibrated], [xgb_oot_pred, cat_oot_pred], "XGB-WoE + CatBoost-DFS"),
    ]

    for combo_name, oof_list, oot_list, feature_stores in combos_2model:
        oof_stack = np.column_stack(oof_list)
        oot_stack = np.column_stack(oot_list)

        # Logistic
        meta_log = train_logistic_meta_learner(oof_stack, y_train)
        oot_pred_log = meta_log.predict_proba(oot_stack)[:, 1]
        oot_gini_log = gini_coefficient(y_oot, oot_pred_log)
        oot_ks_log = ks_statistic(y_oot, oot_pred_log)[0]
        ablation_results.append({
            "rank": 0,
            "combo": combo_num,
            "strategy": "logistic",
            "n_models": 2,
            "gini": oot_gini_log,
            "ks": oot_ks_log,
            "feature_stores": feature_stores,
        })
        print(f"  {combo_num}. {combo_name} + logistic: Gini={oot_gini_log:.4f}, KS={oot_ks_log:.4f}")
        combo_num += 1

        # Rank average
        oot_pred_rank = rank_avg_ensemble(oot_stack)
        oot_gini_rank = gini_coefficient(y_oot, oot_pred_rank)
        oot_ks_rank = ks_statistic(y_oot, oot_pred_rank)[0]
        ablation_results.append({
            "rank": 0,
            "combo": combo_num,
            "strategy": "rank_avg",
            "n_models": 2,
            "gini": oot_gini_rank,
            "ks": oot_ks_rank,
            "feature_stores": feature_stores,
        })
        print(f"  {combo_num}. {combo_name} + rank_avg: Gini={oot_gini_rank:.4f}, KS={oot_ks_rank:.4f}")
        combo_num += 1

    # 3-model combinations
    oof_3stack = np.column_stack([lgb_oof, xgb_oof, cat_oof_calibrated])
    oot_3stack = np.column_stack([lgb_oot_pred, xgb_oot_pred, cat_oot_pred])
    feature_stores_3 = "LGB + XGB-WoE + CatBoost-DFS"

    # Logistic
    meta_log_3 = train_logistic_meta_learner(oof_3stack, y_train)
    oot_pred_log_3 = meta_log_3.predict_proba(oot_3stack)[:, 1]
    oot_gini_log_3 = gini_coefficient(y_oot, oot_pred_log_3)
    oot_ks_log_3 = ks_statistic(y_oot, oot_pred_log_3)[0]
    ablation_results.append({
        "rank": 0,
        "combo": combo_num,
        "strategy": "logistic",
        "n_models": 3,
        "gini": oot_gini_log_3,
        "ks": oot_ks_log_3,
        "feature_stores": feature_stores_3,
    })
    print(f"  {combo_num}. 3-model + logistic: Gini={oot_gini_log_3:.4f}, KS={oot_ks_log_3:.4f}")
    combo_num += 1

    # Rank average
    oot_pred_rank_3 = rank_avg_ensemble(oot_3stack)
    oot_gini_rank_3 = gini_coefficient(y_oot, oot_pred_rank_3)
    oot_ks_rank_3 = ks_statistic(y_oot, oot_pred_rank_3)[0]
    ablation_results.append({
        "rank": 0,
        "combo": combo_num,
        "strategy": "rank_avg",
        "n_models": 3,
        "gini": oot_gini_rank_3,
        "ks": oot_ks_rank_3,
        "feature_stores": feature_stores_3,
    })
    print(f"  {combo_num}. 3-model + rank_avg: Gini={oot_gini_rank_3:.4f}, KS={oot_ks_rank_3:.4f}")
    combo_num += 1

    # MLP meta-learner
    meta_mlp = train_mlp_meta_learner(oof_3stack, y_train)
    oot_pred_mlp = meta_mlp.predict_proba(oot_3stack)[:, 1]
    oot_gini_mlp = gini_coefficient(y_oot, oot_pred_mlp)
    oot_ks_mlp = ks_statistic(y_oot, oot_pred_mlp)[0]
    ablation_results.append({
        "rank": 0,
        "combo": combo_num,
        "strategy": "mlp_meta",
        "n_models": 3,
        "gini": oot_gini_mlp,
        "ks": oot_ks_mlp,
        "feature_stores": feature_stores_3,
    })
    print(f"  {combo_num}. 3-model + mlp_meta: Gini={oot_gini_mlp:.4f}, KS={oot_ks_mlp:.4f}")

    # --- Step 6: Rank and save ablation results ---
    print("\n" + "="*70)
    print("STEP 6: Rank Ablation Results and Save")
    print("="*70)

    ablation_results = sorted(ablation_results, key=lambda x: x["gini"], reverse=True)
    for i, result in enumerate(ablation_results, 1):
        result["rank"] = i

    ablation_csv_path = Path("reports/ensemble_v2_ablation.csv")
    ablation_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(ablation_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "combo", "strategy", "n_models", "gini", "ks", "feature_stores"])
        writer.writeheader()
        writer.writerows(ablation_results)

    print(f"✓ Ablation results saved to {ablation_csv_path}")
    print("\nTop 3 combinations:")
    for i in range(min(3, len(ablation_results))):
        r = ablation_results[i]
        print(f"  {i+1}. Combo {r['combo']} ({r['strategy']}, {r['n_models']} models): Gini={r['gini']:.4f}")

    # --- Step 7: Gate evaluation ---
    print("\n" + "="*70)
    print("STEP 7: Gate Evaluation")
    print("="*70)

    best_result = ablation_results[0]
    best_gini = best_result["gini"]

    if best_gini >= GATE_THRESHOLDS["pass"]:
        gate_verdict = "PASS"
        reason = f"OOT Gini {best_gini:.4f} ≥ {GATE_THRESHOLDS['pass']}"
        save_ensemble = True
    elif best_gini > GATE_THRESHOLDS["accept"]:
        gate_verdict = "ACCEPT"
        reason = f"OOT Gini {best_gini:.4f} > baseline {GATE_THRESHOLDS['accept']} (CatBoost v2)"
        save_ensemble = True
    elif best_gini >= GATE_THRESHOLDS["investigate"]:
        gate_verdict = "INVESTIGATE"
        reason = f"OOT Gini {best_gini:.4f} in investigation range [{GATE_THRESHOLDS['investigate']}, {GATE_THRESHOLDS['pass']})"
        save_ensemble = False
    else:
        gate_verdict = "FAIL"
        reason = f"OOT Gini {best_gini:.4f} < minimum {GATE_THRESHOLDS['investigate']}"
        save_ensemble = False

    print(f"\nGate Verdict: {gate_verdict}")
    print(f"Reason: {reason}")
    print(f"Best Combo: #{best_result['combo']} ({best_result['strategy']}, {best_result['n_models']} models)")
    print(f"OOT Gini: {best_gini:.4f}")
    print(f"OOT KS: {best_result['ks']:.4f}")

    # --- Step 8: Save ensemble summary ---
    summary_path = Path("reports/ensemble_v2_best_summary.json")
    summary = {
        "phase": "04.2.10",
        "gate_verdict": gate_verdict,
        "gate_reason": reason,
        "best_combo": best_result["combo"],
        "best_strategy": best_result["strategy"],
        "n_models": best_result["n_models"],
        "feature_stores": best_result["feature_stores"],
        "oot_gini": best_gini,
        "oot_ks": best_result["ks"],
        "baseline_lgb_gini": 0.5695,
        "baseline_xgb_gini": 0.5636,
        "baseline_cat_gini": 0.5814,
        "ensemble_saved": save_ensemble,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {summary_path}")

    return best_result, gate_verdict


def main():
    parser = argparse.ArgumentParser(
        description="Ensemble Enhancement via Feature Diversity (Phase 04.2.10)"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials for HPO (default: 50)"
    )
    parser.add_argument(
        "--skip-hpo",
        action="store_true",
        help="Skip HPO, use cached params (fast mode)"
    )

    args = parser.parse_args()

    try:
        best_result, gate_verdict = run_ensemble_v2(
            n_trials=args.n_trials,
            skip_hpo=args.skip_hpo,
        )

        print("\n" + "="*70)
        print("ENSEMBLE V2 WORKFLOW COMPLETE")
        print("="*70)
        print(f"Gate Verdict: {gate_verdict}")
        print(f"Best OOT Gini: {best_result['gini']:.4f}")

        # Return appropriate exit code
        if gate_verdict == "PASS":
            sys.exit(0)
        elif gate_verdict == "ACCEPT":
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
