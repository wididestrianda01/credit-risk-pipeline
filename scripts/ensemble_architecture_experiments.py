#!/usr/bin/env python3
"""
Ensemble Architecture Experiments — Phase 04.2 Plan 04.

Evaluates 4 ensemble architectures on OOF from improved base models:
  1. Rank-averaging        — per-fold percentile ranks, no meta-learner
  2. Calibration-aware     — per-fold Platt scaling → Ridge meta-learner
  3. 4-model stack         — LGB + XGB + CatBoost + LR, Ridge meta-learner
  4. Weighted average      — Nelder-Mead scalar weight optimization

Current best baseline: Variant B (Ridge meta, Gini=0.5519)
Persist threshold: Gini > 0.5514 (current_best - 0.005)

Data: Loads X_raw_features.parquet (307K rows, intact index) and
reconstructs the 80/20 split with random_state=42 to match prior runs.

Command:
    python scripts/ensemble_architecture_experiments.py

Output:
    reports/ensemble_architectures_comparison.json
    models/ensemble_<architecture>.pkl  (only if improvement >= persist threshold)
"""

import json
import sys
import warnings
from pathlib import Path
from typing import List, Tuple, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

if "credit_engine" not in sys.modules:
    import src
    sys.modules["credit_engine"] = src

from credit_engine.model import _make_cv
from credit_engine.utils import gini_coefficient

warnings.filterwarnings("ignore")

# ============================================================================
# Constants
# ============================================================================

_CURRENT_BEST_GINI = 0.5519080946684274   # Variant B, Phase 04.1
_PERSIST_THRESHOLD = _CURRENT_BEST_GINI - 0.005  # 0.5469
_RANDOM_STATE = 42
_N_FOLDS = 5
_TEMPORAL_SORT_COL = "prev_days_decision_mean"

# Fixed best hyperparameters from prior HPO runs
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
# OOF generation helpers
# ============================================================================

def _make_xgb(**params) -> object:
    import xgboost as xgb
    return xgb.XGBClassifier(
        **params,
        eval_metric="auc",
        verbosity=0,
        random_state=_RANDOM_STATE,
        use_label_encoder=False,
    )


def _make_lgb(**params) -> object:
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        **params,
        verbose=-1,
        random_state=_RANDOM_STATE,
        is_unbalance=True,
    )


def _make_cat(**params) -> object:
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        **params,
        random_seed=_RANDOM_STATE,
        allow_writing_files=False,
        verbose=0,
    )


def _generate_oof(
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    model_factory,
    model_params: dict,
    scale_pos_weight: float | None = None,
    extra_params: dict | None = None,
) -> np.ndarray:
    """Generate OOF predictions for a single model via cross-validation.

    Parameters
    ----------
    X : pd.DataFrame
        Training feature matrix (reset_index applied internally).
    y : pd.Series
        Binary target aligned with X.
    cv : _TemporalCV or StratifiedKFold
        CV splitter with .split(X) interface.
    model_factory : callable
        Function(params) → fitted-model constructor kwargs + model.
        One of _make_xgb, _make_lgb, _make_cat.
    model_params : dict
        Best hyperparameters.
    scale_pos_weight : float, optional
        Passed to XGB/CatBoost as class-imbalance weight.
    extra_params : dict, optional
        Additional parameters to merge into model_params (e.g. scale_pos_weight).

    Returns
    -------
    np.ndarray, shape (n,)
        OOF probability predictions.
    """
    X_arr = X.reset_index(drop=True)
    y_arr = y.reset_index(drop=True)
    oof = np.zeros(len(y_arr))

    params = {**model_params}
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight
    if extra_params:
        params.update(extra_params)

    for train_idx, val_idx in cv.split(X_arr):
        X_tr, y_tr = X_arr.iloc[train_idx], y_arr.iloc[train_idx]
        X_val = X_arr.iloc[val_idx]

        model = model_factory(**params)
        model.fit(X_tr, y_tr)
        oof[val_idx] = model.predict_proba(X_val)[:, 1]

    return oof


# ============================================================================
# Architecture 1: Rank-Averaging
# ============================================================================

def ensemble_rank_average(
    oof_list: List[np.ndarray],
    model_names: List[str],
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    n_total: int,
) -> np.ndarray:
    """Rank-average OOF predictions: convert each model's per-fold probabilities
    to within-fold percentile ranks, then average ranks across models.

    Per-fold ranking prevents calibration differences between models from
    inflating one model's signal (e.g., a well-calibrated model with wider
    probability range would dominate simple averaging).

    Parameters
    ----------
    oof_list : list of (n,) arrays
        OOF probability arrays, one per model.
    model_names : list of str
        Model names for logging.
    cv_splits : list of (train_idx, val_idx) tuples
        CV fold index pairs from cv.split().
    n_total : int
        Total number of training rows.

    Returns
    -------
    np.ndarray, shape (n,)
        Rank-averaged ensemble predictions in [0, 1].
    """
    oof_ensemble = np.zeros(n_total)

    for train_idx, val_idx in cv_splits:
        rank_sum = np.zeros(len(val_idx))

        for oof in oof_list:
            fold_proba = oof[val_idx]
            fold_rank = pd.Series(fold_proba).rank(pct=True).values
            rank_sum += fold_rank

        oof_ensemble[val_idx] = rank_sum / len(oof_list)

    return oof_ensemble


# ============================================================================
# Architecture 2: Calibration-Aware Stacking
# ============================================================================

def _platt_calibrate_fold(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    calib_frac: float = 0.25,
) -> np.ndarray:
    """Apply per-fold Platt calibration: reserve calib_frac of fold training
    data to fit a 1-parameter sigmoid layer; predict calibrated probabilities
    on X_val.

    Platt calibration fits a logistic regression on the uncalibrated
    probability outputs: sigmoid(a * s + b) where s is the raw score.
    The 2-parameter fit corrects both scale and shift of the model's
    probability distribution — important when base model probabilities
    are "compressed" near 0.5 (common in gradient boosting).

    Parameters
    ----------
    model : fitted sklearn-compatible classifier
        Pre-trained on fold training data.
    X_train : pd.DataFrame
        Fold training data (model already trained on this).
    y_train : pd.Series
        Fold training labels.
    X_val : pd.DataFrame
        Fold validation data to predict on.
    calib_frac : float
        Fraction of X_train to use as calibration split.

    Returns
    -------
    np.ndarray, shape (len(X_val),)
        Calibrated probability predictions on X_val.
    """
    _, X_calib, _, y_calib = train_test_split(
        X_train, y_train,
        test_size=calib_frac,
        stratify=y_train,
        random_state=_RANDOM_STATE,
    )

    # Platt: fit LogisticRegression on raw model scores
    calib_scores = model.predict_proba(X_calib)[:, 1].reshape(-1, 1)
    platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    platt.fit(calib_scores, y_calib)

    val_scores = model.predict_proba(X_val)[:, 1].reshape(-1, 1)
    return platt.predict_proba(val_scores)[:, 1]


def ensemble_calibration_aware_stacking(
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    model_factories_params: List[Tuple],
    meta_alpha: float = 1.0,
) -> Tuple[np.ndarray, Ridge]:
    """Generate OOF with per-fold Platt calibration, then train Ridge meta-learner.

    Architecture: for each model × fold, train model → Platt-calibrate on
    fold training sub-split → predict calibrated proba on fold val. Stack
    calibrated OOF columns → fit Ridge meta-learner.

    Parameters
    ----------
    X : pd.DataFrame
        Full training feature matrix.
    y : pd.Series
        Binary target.
    cv : CV splitter
        Returns (train_idx, val_idx) pairs.
    model_factories_params : list of (factory_fn, params_dict, scale_pos_weight)
        One entry per base model.
    meta_alpha : float
        Ridge regularization strength.

    Returns
    -------
    (oof_ensemble, meta_learner) : tuple
        OOF ensemble predictions and fitted Ridge meta-learner.
    """
    X_arr = X.reset_index(drop=True)
    y_arr = y.reset_index(drop=True)
    n = len(y_arr)

    # Per-model calibrated OOF arrays
    oof_calibrated: List[np.ndarray] = []

    for factory_fn, model_params, scale_pos_weight in model_factories_params:
        oof = np.zeros(n)
        params = {**model_params}
        if scale_pos_weight is not None:
            params["scale_pos_weight"] = scale_pos_weight

        for train_idx, val_idx in cv.split(X_arr):
            X_tr, y_tr = X_arr.iloc[train_idx], y_arr.iloc[train_idx]
            X_val = X_arr.iloc[val_idx]

            model = factory_fn(**params)
            model.fit(X_tr, y_tr)

            # Calibrate predictions per fold
            oof[val_idx] = _platt_calibrate_fold(model, X_tr, y_tr, X_val)

        oof_calibrated.append(oof)

    # Stack calibrated OOF as meta-features
    X_meta = np.column_stack(oof_calibrated)
    meta_learner = Ridge(alpha=meta_alpha)
    meta_learner.fit(X_meta, y_arr)

    oof_ensemble = meta_learner.predict(X_meta)
    oof_ensemble = np.clip(oof_ensemble, 0.0, 1.0)

    return oof_ensemble, meta_learner


# ============================================================================
# Architecture 3: 4-Model Stack (LGB + XGB + CatBoost + LR)
# ============================================================================

def _generate_lr_oof(
    X: pd.DataFrame,
    y: pd.Series,
    cv,
) -> np.ndarray:
    """Generate OOF for Logistic Regression baseline (scaled raw features).

    LR provides diversity because it captures linear additive structure
    while tree-based models capture non-linear interactions. Even if LR
    alone has lower Gini (~0.27 on WoE features), its error pattern differs
    from LGB/XGB/CatBoost — the meta-learner may exploit this orthogonality.

    Parameters
    ----------
    X, y : feature matrix and target.
    cv : CV splitter.

    Returns
    -------
    np.ndarray, shape (n,)
        OOF LR probabilities.
    """
    X_arr = X.reset_index(drop=True)
    y_arr = y.reset_index(drop=True)
    oof = np.zeros(len(y_arr))
    scaler = StandardScaler()

    for train_idx, val_idx in cv.split(X_arr):
        X_tr, y_tr = X_arr.iloc[train_idx], y_arr.iloc[train_idx]
        X_val = X_arr.iloc[val_idx]

        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)

        lr = LogisticRegression(C=0.1, solver="saga", max_iter=1000,
                                random_state=_RANDOM_STATE, n_jobs=-1)
        lr.fit(X_tr_scaled, y_tr)
        oof[val_idx] = lr.predict_proba(X_val_scaled)[:, 1]

    return oof


def ensemble_4model_stack(
    oof_lgb: np.ndarray,
    oof_xgb: np.ndarray,
    oof_cat: np.ndarray,
    oof_lr: np.ndarray,
    y: pd.Series,
    meta_alpha: float = 1.0,
) -> Tuple[np.ndarray, Ridge]:
    """Train Ridge meta-learner on 4-model OOF stack.

    Parameters
    ----------
    oof_lgb, oof_xgb, oof_cat, oof_lr : np.ndarray
        OOF probability arrays from each base model.
    y : pd.Series
        Binary target.
    meta_alpha : float
        Ridge regularization strength.

    Returns
    -------
    (oof_ensemble, meta_learner)
    """
    y_arr = np.asarray(y.reset_index(drop=True))
    X_meta = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_lr])

    meta_learner = Ridge(alpha=meta_alpha)
    meta_learner.fit(X_meta, y_arr)

    oof_ensemble = meta_learner.predict(X_meta)
    oof_ensemble = np.clip(oof_ensemble, 0.0, 1.0)

    return oof_ensemble, meta_learner


# ============================================================================
# Architecture 4: Weighted Average (Nelder-Mead)
# ============================================================================

def ensemble_weighted_average_optimize(
    oof_list: List[np.ndarray],
    y: pd.Series,
    model_names: List[str],
    n_restarts: int = 5,
) -> Tuple[np.ndarray, dict]:
    """Search for optimal scalar weights per model to maximize OOF Gini.

    Uses Nelder-Mead optimization with multiple random restarts to avoid
    local optima. Weights are normalized to sum to 1.0 after optimization.

    Conceptually: equal weighting assumes each model contributes equally.
    Nelder-Mead finds the weighting that best combines complementary signal
    without the structural constraints of a Ridge meta-learner.

    Parameters
    ----------
    oof_list : list of (n,) arrays
        OOF probability arrays.
    y : pd.Series
        Binary target.
    model_names : list of str
        Names for each model in the ensemble.
    n_restarts : int
        Number of random restarts for Nelder-Mead.

    Returns
    -------
    (oof_ensemble, info_dict)
        info_dict has keys: weights, gini, optimization_method.
    """
    y_arr = np.asarray(y.reset_index(drop=True))
    n_models = len(oof_list)
    oof_matrix = np.column_stack(oof_list)  # shape (n, n_models)

    best_gini = -np.inf
    best_weights = np.ones(n_models) / n_models

    def _neg_gini(raw_weights: np.ndarray) -> float:
        w = np.abs(raw_weights)  # Ensure non-negative
        w_sum = w.sum()
        if w_sum < 1e-10:
            return 1.0  # Max penalty
        w_norm = w / w_sum
        ensemble = oof_matrix @ w_norm
        ensemble = np.clip(ensemble, 0.0, 1.0)
        return -gini_coefficient(y_arr, ensemble)

    # Uniform start
    starts = [np.ones(n_models) / n_models]

    # Random restarts
    rng = np.random.default_rng(_RANDOM_STATE)
    for _ in range(n_restarts - 1):
        raw = rng.exponential(1.0, size=n_models)
        starts.append(raw / raw.sum())

    for x0 in starts:
        result = minimize(
            _neg_gini, x0,
            method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-6},
        )
        trial_gini = -result.fun
        if trial_gini > best_gini:
            best_gini = trial_gini
            raw = np.abs(result.x)
            best_weights = raw / raw.sum()

    oof_ensemble = oof_matrix @ best_weights
    oof_ensemble = np.clip(oof_ensemble, 0.0, 1.0)

    info = {
        "weights": {name: float(w) for name, w in zip(model_names, best_weights)},
        "gini": float(best_gini),
        "optimization_method": "nelder-mead",
        "n_restarts": n_restarts,
    }

    return oof_ensemble, info


# ============================================================================
# Main
# ============================================================================

class _RidgeWrapper:
    """Pickle-friendly wrapper so Ridge meta-learner has predict_proba."""

    def __init__(self, ridge: Ridge, clip: bool = True):
        self._ridge = ridge
        self._clip = clip

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = self._ridge.predict(X)
        if self._clip:
            preds = np.clip(preds, 0.0, 1.0)
        col_other = 1.0 - preds
        return np.column_stack([col_other, preds])


def main() -> dict:
    """Run all 4 ensemble architecture experiments and save comparison JSON."""

    print("[EnsembleExp] Loading feature data...")
    X_raw = pd.read_parquet("data/processed/X_raw_features.parquet")
    y = pd.read_parquet("data/processed/y_train.parquet").squeeze()
    print(f"  X_raw shape : {X_raw.shape}")
    print(f"  y default rate: {y.mean():.4f}")

    # Reconstruct same split as xgb_optuna_hpo.py and feature_augmentation_suite.py
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, stratify=y, random_state=_RANDOM_STATE
    )
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    # Temporal CV groups
    groups = (
        X_train[_TEMPORAL_SORT_COL].to_numpy()
        if _TEMPORAL_SORT_COL in X_train.columns
        else None
    )
    cv = _make_cv(groups, n_splits=_N_FOLDS)
    cv_splits = list(cv.split(X_train.reset_index(drop=True)))
    n_train = len(X_train)

    scale_pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())
    print(f"  scale_pos_weight: {scale_pos_weight:.4f}")

    # ── Generate base OOF from 3 models ──────────────────────────────────
    print("\n[EnsembleExp] Generating base OOF (LGB, XGB, CatBoost)...")

    print("  LGB OOF...")
    oof_lgb = _generate_oof(X_train, y_train, cv, _make_lgb, _LGB_BEST_PARAMS)
    gini_lgb = gini_coefficient(y_train.reset_index(drop=True), oof_lgb)
    print(f"    LGB OOF Gini: {gini_lgb:.4f}")

    print("  XGB OOF...")
    oof_xgb = _generate_oof(
        X_train, y_train, cv, _make_xgb, _XGB_BEST_PARAMS,
        scale_pos_weight=scale_pos_weight
    )
    gini_xgb = gini_coefficient(y_train.reset_index(drop=True), oof_xgb)
    print(f"    XGB OOF Gini: {gini_xgb:.4f}")

    print("  CatBoost OOF...")
    oof_cat = _generate_oof(
        X_train, y_train, cv, _make_cat, _CAT_BEST_PARAMS,
        scale_pos_weight=scale_pos_weight
    )
    gini_cat = gini_coefficient(y_train.reset_index(drop=True), oof_cat)
    print(f"    CatBoost OOF Gini: {gini_cat:.4f}")

    best_base_gini = max(gini_lgb, gini_xgb, gini_cat)
    print(f"\n  Best base model Gini: {best_base_gini:.4f}")
    print(f"  Current best ensemble: {_CURRENT_BEST_GINI:.4f}")
    print(f"  Persist threshold: {_PERSIST_THRESHOLD:.4f}")

    y_train_arr = y_train.reset_index(drop=True)
    oof_base = [oof_lgb, oof_xgb, oof_cat]
    base_names = ["lgb", "xgb", "catboost"]

    results = {
        "phase": "04.2",
        "plan": "04",
        "current_best_baseline": float(_CURRENT_BEST_GINI),
        "persist_threshold": float(_PERSIST_THRESHOLD),
        "base_model_ginis": {
            "lgb": float(gini_lgb),
            "xgb": float(gini_xgb),
            "catboost": float(gini_cat),
            "best_base": float(best_base_gini),
        },
        "architectures": [],
        "winner": None,
        "best_ensemble_gini": 0.0,
    }

    # ── Architecture 1: Rank-Averaging ───────────────────────────────────
    print("\n[EnsembleExp] Architecture 1: Rank-Averaging...")
    oof_rank_avg = ensemble_rank_average(oof_base, base_names, cv_splits, n_train)
    gini_rank_avg = gini_coefficient(y_train_arr, oof_rank_avg)
    improvement_rank_avg = float(gini_rank_avg - _CURRENT_BEST_GINI)
    persisted_rank_avg = bool(gini_rank_avg > _PERSIST_THRESHOLD)
    print(f"  Gini: {gini_rank_avg:.4f} (vs baseline: {improvement_rank_avg:+.4f})")

    arch_rank_avg = {
        "name": "rank_averaging",
        "oof_gini": float(gini_rank_avg),
        "improvement_vs_best": improvement_rank_avg,
        "persisted": persisted_rank_avg,
        "meta_learner": "none",
    }

    if persisted_rank_avg:
        # Save as a simple wrapper that stores OOF weights per model
        model_path = "models/ensemble_rank_averaging.pkl"
        joblib.dump(
            {"architecture": "rank_averaging", "model_names": base_names, "weights": [1.0 / 3] * 3},
            model_path,
        )
        arch_rank_avg["model_file"] = model_path
        print(f"  Persisted to {model_path}")
    results["architectures"].append(arch_rank_avg)

    # ── Architecture 2: Calibration-Aware Stacking ───────────────────────
    print("\n[EnsembleExp] Architecture 2: Calibration-Aware Stacking...")
    model_factories_params = [
        (_make_lgb, _LGB_BEST_PARAMS, None),  # is_unbalance handles imbalance
        (_make_xgb, _XGB_BEST_PARAMS, scale_pos_weight),
        (_make_cat, _CAT_BEST_PARAMS, scale_pos_weight),
    ]
    oof_calib, meta_calib = ensemble_calibration_aware_stacking(
        X_train, y_train, cv, model_factories_params
    )
    gini_calib = gini_coefficient(y_train_arr, oof_calib)
    improvement_calib = float(gini_calib - _CURRENT_BEST_GINI)
    persisted_calib = bool(gini_calib > _PERSIST_THRESHOLD)
    print(f"  Gini: {gini_calib:.4f} (vs baseline: {improvement_calib:+.4f})")

    arch_calib = {
        "name": "calibration_aware_stacking",
        "oof_gini": float(gini_calib),
        "improvement_vs_best": improvement_calib,
        "persisted": persisted_calib,
        "meta_learner": "ridge",
    }

    if persisted_calib:
        model_path = "models/ensemble_calibration_aware.pkl"
        joblib.dump(_RidgeWrapper(meta_calib), model_path)
        arch_calib["model_file"] = model_path
        print(f"  Persisted meta-learner to {model_path}")
    results["architectures"].append(arch_calib)

    # ── Architecture 3: 4-Model Stack ────────────────────────────────────
    print("\n[EnsembleExp] Architecture 3: 4-Model Stack (LGB + XGB + CatBoost + LR)...")
    print("  Generating LR OOF...")
    oof_lr = _generate_lr_oof(X_train, y_train, cv)
    gini_lr = gini_coefficient(y_train_arr, oof_lr)
    print(f"    LR OOF Gini: {gini_lr:.4f}")

    oof_4stack, meta_4stack = ensemble_4model_stack(
        oof_lgb, oof_xgb, oof_cat, oof_lr, y_train
    )
    gini_4stack = gini_coefficient(y_train_arr, oof_4stack)
    improvement_4stack = float(gini_4stack - _CURRENT_BEST_GINI)
    persisted_4stack = bool(gini_4stack > _PERSIST_THRESHOLD)
    print(f"  Gini: {gini_4stack:.4f} (vs baseline: {improvement_4stack:+.4f})")

    arch_4stack = {
        "name": "4_model_stack",
        "oof_gini": float(gini_4stack),
        "improvement_vs_best": improvement_4stack,
        "persisted": persisted_4stack,
        "meta_learner": "ridge",
        "models": ["lgb", "xgb", "catboost", "lr"],
        "lr_oof_gini": float(gini_lr),
    }

    if persisted_4stack:
        model_path = "models/ensemble_4model_stack.pkl"
        joblib.dump(_RidgeWrapper(meta_4stack), model_path)
        arch_4stack["model_file"] = model_path
        print(f"  Persisted meta-learner to {model_path}")
    results["architectures"].append(arch_4stack)

    # ── Architecture 4: Weighted Average (Nelder-Mead) ───────────────────
    print("\n[EnsembleExp] Architecture 4: Weighted Average (Nelder-Mead)...")
    oof_weighted, weights_info = ensemble_weighted_average_optimize(
        oof_base, y_train, base_names, n_restarts=5
    )
    gini_weighted = gini_coefficient(y_train_arr, oof_weighted)
    improvement_weighted = float(gini_weighted - _CURRENT_BEST_GINI)
    persisted_weighted = bool(gini_weighted > _PERSIST_THRESHOLD)
    print(f"  Gini: {gini_weighted:.4f} (vs baseline: {improvement_weighted:+.4f})")
    print(f"  Optimal weights: {weights_info['weights']}")

    arch_weighted = {
        "name": "weighted_average",
        "oof_gini": float(gini_weighted),
        "improvement_vs_best": improvement_weighted,
        "persisted": persisted_weighted,
        "meta_learner": "none",
        "weights": weights_info["weights"],
        "n_restarts": weights_info["n_restarts"],
    }

    if persisted_weighted:
        model_path = "models/ensemble_weighted_average.pkl"
        joblib.dump(
            {"architecture": "weighted_average", "model_names": base_names,
             "weights": weights_info["weights"]},
            model_path,
        )
        arch_weighted["model_file"] = model_path
        print(f"  Persisted to {model_path}")
    results["architectures"].append(arch_weighted)

    # ── Determine winner ─────────────────────────────────────────────────
    all_ginis = [a["oof_gini"] for a in results["architectures"]]
    best_idx = int(np.argmax(all_ginis))
    results["winner"] = results["architectures"][best_idx]["name"]
    results["best_ensemble_gini"] = float(max(all_ginis))
    results["exit_gate_pass"] = results["best_ensemble_gini"] >= 0.57

    # Persistence decision per D-08
    if results["best_ensemble_gini"] >= 0.57:
        results["recommendation"] = "exit_gate_success"
    elif results["best_ensemble_gini"] > 0.5514:
        results["recommendation"] = "proceed_with_best_ensemble"
    else:
        results["recommendation"] = "ensemble_no_longer_adds_value_revert_to_best_standalone"

    # ── Save results ─────────────────────────────────────────────────────
    output_path = Path("reports/ensemble_architectures_comparison.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[EnsembleExp] Saved results to {output_path}")

    # ── Print summary ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("[EnsembleExp] COMPLETE — Ensemble Architecture Comparison")
    print(f"{'='*70}")
    print(f"{'Architecture':<30} {'Gini':>8} {'vs Baseline':>14} {'Persisted':>10}")
    print("-" * 70)
    for arch in results["architectures"]:
        marker = " ← WINNER" if arch["name"] == results["winner"] else ""
        print(
            f"{arch['name']:<30} {arch['oof_gini']:>8.4f} "
            f"{arch['improvement_vs_best']:>+14.4f} "
            f"{'YES' if arch['persisted'] else 'no':>10}"
            f"{marker}"
        )
    print(f"\nWinner: {results['winner']} (Gini={results['best_ensemble_gini']:.4f})")
    print(f"Recommendation: {results['recommendation']}")

    return results


if __name__ == "__main__":
    main()
