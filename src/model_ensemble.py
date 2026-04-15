"""
model_ensemble.py
-----------------
Ensemble workflow: training, stacking, meta-learner classes, and gating logic.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from src.features import _PROJECT_ROOT
from src.utils import gini_coefficient, ks_statistic, evaluate_model, plot_roc_and_pr
from src.model_base import (
    _make_cv, _TemporalCV, save_model, load_model, calibrate_model,
    _TEST_SIZE, _RANDOM_STATE, _CV_N_SPLITS, _CV_EMBARGO_FRAC, _TEMPORAL_SORT_COL,
    _ENSEMBLE_LGB_DEFAULTS, _ENSEMBLE_XGB_DEFAULTS, _ENSEMBLE_CAT_DEFAULTS,
    _ENSEMBLE_WORKFLOW_MODEL_PATH, _ENSEMBLE_WORKFLOW_WEIGHTS_PATH, _ENSEMBLE_3MODEL_WORKFLOW_MODEL_PATH, _ENSEMBLE_3MODEL_WORKFLOW_WEIGHTS_PATH,
    _ENSEMBLE_PERSIST_THRESHOLD, _CALIB_SPLIT,
)

# ---------------------------------------------------------------------------
# Ensemble meta-learners
# ---------------------------------------------------------------------------

class _AverageEnsemble:
    """
    Simple probability average of two fitted estimators.

    Used when method='average' in train_ensemble; takes the mean of
    LightGBM and XGBoost predicted probabilities for the positive class.

    Parameters
    ----------
    lgb_model : object
        Fitted LightGBM model with predict_proba method.
    xgb_model : object
        Fitted XGBoost model with predict_proba method.
    """

    def __init__(self, lgb_model: object, xgb_model: object):
        """Initialize with pre-fitted base models."""
        self.lgb_model = lgb_model
        self.xgb_model = xgb_model

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities via simple averaging.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        proba : np.ndarray
            Shape (n_samples, 2) with columns [P(y=0), P(y=1)].
            Each row sums to 1.0.
        """
        p_lgb = self.lgb_model.predict_proba(X)[:, 1]
        p_xgb = self.xgb_model.predict_proba(X)[:, 1]
        avg = (p_lgb + p_xgb) / 2.0
        return np.column_stack([1 - avg, avg])


# ---------------------------------------------------------------------------
# Temporal cross-validation (López de Prado, Ch. 7)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 2-model ensemble training
# ---------------------------------------------------------------------------

def train_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    lgb_params: dict | None = None,
    xgb_params: dict | None = None,
    n_splits: int = 5,
    method: Literal["average", "logistic"] = "average",
    seed: int = _RANDOM_STATE,
) -> tuple[object, dict]:
    """
    Blend LightGBM + XGBoost via out-of-fold (OOF) stacking.

    Trains two base models (LightGBM, XGBoost) on stratified k-fold CV folds.
    Out-of-fold predictions serve as features for a meta-learner (either
    simple averaging or logistic regression). No data leakage: each base
    model predicts only on rows it never trained on.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary target labels.
    lgb_params : dict, optional
        LightGBM hyperparameters. If None, uses _ENSEMBLE_LGB_DEFAULTS.
    xgb_params : dict, optional
        XGBoost hyperparameters. If None, uses _ENSEMBLE_XGB_DEFAULTS.
        Note: scale_pos_weight is computed and added automatically.
    n_splits : int, default=5
        Number of stratified k-folds for OOF generation.
    method : {'average', 'logistic'}, default='average'
        Meta-learner type:
        - 'average': simple probability average of the two base models
        - 'logistic': logistic regression trained on OOF features [oof_lgb, oof_xgb]
    seed : int, default=42
        Random state for reproducibility.

    Returns
    -------
    ensemble_model : object
        - For method='average': _AverageEnsemble with lgb_final, xgb_final
        - For method='logistic': _LogisticEnsemble wrapping lgb_final, xgb_final, meta_lr
    metrics : dict
        Evaluation dict from evaluate_model() on holdout (20% test set):
        Keys: {'Model', 'AUC-ROC', 'Gini', 'KS', 'Brier', 'BrierSkill', 'AvgPrecision'}

    Notes
    -----
    Algorithm:
    1. Split X/y into 80% train (OOF) / 20% holdout via train_test_split with stratify=y
    2. On the 80% train portion, run n_splits-fold CV:
       - Uses _TemporalCV (walk-forward with embargo) when _TEMPORAL_SORT_COL is present in X
       - Falls back to StratifiedKFold when temporal column is absent
       - Each fold: train LightGBM on fold training data, XGBoost on same training data
       - Collect OOF predictions: oof_lgb[val_idx], oof_xgb[val_idx]
    3a. If method='average': final models are lgb_final, xgb_final trained on full 80% train
    3b. If method='logistic': fit LogisticRegression on np.column_stack([oof_lgb, oof_xgb]) → y_train
    4. Evaluate on the 20% holdout set
    5. Return (ensemble_model, metrics)

    Data leakage prevention:
    - Base models fit only on their training folds (never on validation)
    - OOF predictions are withheld for fold validation folds
    - Holdout set (20%) never seen by any model during training or meta-learning
    - scale_pos_weight computed only on training data
    - Temporal CV prevents double leakage: base model OOF ordering must match meta-learner CV
    """
    import lightgbm as lgb
    import xgboost as xgb

    # --- Use defaults if params not provided ---
    if lgb_params is None:
        lgb_params = _ENSEMBLE_LGB_DEFAULTS.copy()
    if xgb_params is None:
        xgb_params = _ENSEMBLE_XGB_DEFAULTS.copy()

    # --- Train / holdout split (80 / 20) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=_TEST_SIZE,
        stratify=y,
        random_state=seed,
    )

    n_train = len(X_train)

    # --- Initialize OOF arrays (for training set only) ---
    oof_lgb = np.zeros(n_train)
    oof_xgb = np.zeros(n_train)

    # --- Auto-detect temporal groups (same pattern as train_lightgbm_optuna) ---
    # Using _TemporalCV prevents double temporal leakage: base model OOF predictions
    # are already temporally ordered; a StratifiedKFold meta-layer would amplify any
    # residual time-structure signal in the OOF stack (López de Prado, Ch. 7).
    if _TEMPORAL_SORT_COL in X_train.columns:
        groups_train = X_train[_TEMPORAL_SORT_COL].to_numpy()
    else:
        groups_train = None
    cv = _make_cv(groups_train, n_splits=n_splits)

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]

        # --- Compute imbalance ratio from fold training labels (per-fold for consistency) ---
        n_neg = (y_fold_train == 0).sum()
        n_pos = (y_fold_train == 1).sum()
        scale_pos_weight = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0

        # --- Fit LightGBM on fold training data ---
        lgb_params_fold = lgb_params.copy()
        lgb_params_fold["scale_pos_weight"] = scale_pos_weight
        lgb_fold = lgb.LGBMClassifier(**lgb_params_fold)
        lgb_fold.fit(X_fold_train, y_fold_train)
        oof_lgb[val_idx] = lgb_fold.predict_proba(X_fold_val)[:, 1]

        # --- Fit XGBoost on fold training data ---
        xgb_params_fold = xgb_params.copy()
        xgb_params_fold["scale_pos_weight"] = scale_pos_weight

        xgb_fold = xgb.XGBClassifier(**xgb_params_fold)
        xgb_fold.fit(X_fold_train, y_fold_train)
        oof_xgb[val_idx] = xgb_fold.predict_proba(X_fold_val)[:, 1]

    # --- Train final base models on full training set ---
    n_neg_train = (y_train == 0).sum()
    n_pos_train = (y_train == 1).sum()
    scale_pos_weight_train = float(n_neg_train) / float(n_pos_train) if n_pos_train > 0 else 1.0

    lgb_params_final = lgb_params.copy()
    lgb_params_final["scale_pos_weight"] = scale_pos_weight_train
    lgb_final = lgb.LGBMClassifier(**lgb_params_final)
    lgb_final.fit(X_train, y_train)

    xgb_params_final = xgb_params.copy()
    xgb_params_final["scale_pos_weight"] = scale_pos_weight_train
    xgb_final = xgb.XGBClassifier(**xgb_params_final)
    xgb_final.fit(X_train, y_train)

    # --- Create ensemble model ---
    if method == "average":
        ensemble_model = _AverageEnsemble(lgb_final, xgb_final)
    elif method == "logistic":
        # Meta-learner: logistic regression on OOF features
        X_meta = np.column_stack([oof_lgb, oof_xgb])
        meta_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        meta_lr.fit(X_meta, y_train)

        # Wrap all three components (base models + meta) in a wrapper object
        ensemble_model = _LogisticEnsemble(lgb_final, xgb_final, meta_lr)
    else:
        raise ValueError(f"method must be 'average' or 'logistic', got '{method}'")

    # --- Evaluate on holdout set ---
    metrics = evaluate_model(ensemble_model, X_test, y_test, f"Ensemble ({method})")

    return ensemble_model, metrics



# ---------------------------------------------------------------------------
# 3-model ensemble training
# ---------------------------------------------------------------------------

def train_ensemble_3model(
    X: pd.DataFrame,
    y: pd.Series,
    lgb_params: dict | None = None,
    xgb_params: dict | None = None,
    cat_params: dict | None = None,
    n_splits: int = 5,
    method: Literal["average", "logistic"] = "logistic",
    groups: np.ndarray | None = None,
    X_oot: pd.DataFrame | None = None,
    y_oot: pd.Series | None = None,
) -> tuple:
    """
    Train 3-model OOF ensemble (LGB + XGB + CatBoost) with logistic meta-learner.

    Parameters
    ----------
    X : pd.DataFrame
        Full training feature matrix. Will be split 80/20 into train/test.
    y : pd.Series
        Binary target labels aligned with X.
    lgb_params : dict, optional
        LightGBM hyperparameters. Defaults to _ENSEMBLE_LGB_DEFAULTS.
    xgb_params : dict, optional
        XGBoost hyperparameters. Defaults to _ENSEMBLE_XGB_DEFAULTS.
    cat_params : dict, optional
        CatBoost hyperparameters. Defaults to _ENSEMBLE_CAT_DEFAULTS.
    n_splits : int, default 5
        Number of CV folds for OOF generation.
    method : {"average", "logistic"}, default "logistic"
        Ensemble combination method.
        - "logistic": Logistic Regression meta-learner (L2, C=1.0)
        - "average": Simple mean of 3 OOF columns
    groups : np.ndarray, optional
        Sample group labels for grouped CV. If None and
        _TEMPORAL_SORT_COL is in X.columns, auto-detected.
    X_oot : pd.DataFrame, optional
        Frozen OOT feature matrix (carved before HPO by the caller).
        When provided, base models and the ensemble are evaluated here.
    y_oot : pd.Series, optional
        Frozen OOT labels aligned with X_oot.

    Returns
    -------
    tuple
        (ensemble_model, metrics_dict, X_test, y_test, base_gini_dict)
        X_test, y_test are the 20% holdout set from the 80/20 split.
        metrics_dict contains Gini/KS evaluated on X_oot if provided, else NaN.
        base_gini_dict = {"lgb": lgb_gini, "xgb": xgb_gini, "cat": cat_gini}
    """
    import lightgbm as lgb
    import xgboost as xgb

    if lgb_params is None:
        lgb_params = _ENSEMBLE_LGB_DEFAULTS.copy()
    if xgb_params is None:
        xgb_params = _ENSEMBLE_XGB_DEFAULTS.copy()
    if cat_params is None:
        cat_params = _ENSEMBLE_CAT_DEFAULTS.copy()

    # Split into 80% train and 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=_TEST_SIZE, stratify=y, random_state=_RANDOM_STATE
    )

    n_train = len(X_train)
    oof_lgb = np.zeros(n_train)
    oof_xgb = np.zeros(n_train)
    oof_cat = np.zeros(n_train)

    # --- Auto-detect temporal groups ---
    if groups is None and _TEMPORAL_SORT_COL in X_train.columns:
        groups_train = X_train[_TEMPORAL_SORT_COL].to_numpy()
    else:
        groups_train = groups
    cv = _make_cv(groups_train, n_splits=n_splits)

    for _fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]

        n_neg = (y_fold_train == 0).sum()
        n_pos = (y_fold_train == 1).sum()
        scale_pos_weight = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0

        # LightGBM — use is_unbalance=True to match standalone HPO strategy;
        # scale_pos_weight causes OOF rank reversal vs standalone LGB.
        # verbosity=-1 silences C++ engine noise that bypasses log_evaluation.
        lgb_params_fold = {k: v for k, v in lgb_params.items() if k != "scale_pos_weight"}
        lgb_params_fold["is_unbalance"] = True
        lgb_params_fold["verbosity"] = -1
        lgb_fold = lgb.LGBMClassifier(**lgb_params_fold)
        lgb_fold.fit(X_fold_train, y_fold_train)
        oof_lgb[val_idx] = lgb_fold.predict_proba(X_fold_val)[:, 1]

        # XGBoost
        xgb_params_fold = {**xgb_params, "scale_pos_weight": scale_pos_weight}
        xgb_fold = xgb.XGBClassifier(**xgb_params_fold)
        xgb_fold.fit(X_fold_train, y_fold_train)
        oof_xgb[val_idx] = xgb_fold.predict_proba(X_fold_val)[:, 1]

        # CatBoost
        cat_params_fold = {**cat_params, "scale_pos_weight": scale_pos_weight}
        cat_fold = CatBoostClassifier(**cat_params_fold)
        cat_fold.fit(X_fold_train.to_numpy(), y_fold_train.to_numpy(), verbose=False)
        oof_cat[val_idx] = cat_fold.predict_proba(X_fold_val.to_numpy())[:, 1]

    # --- Train final base models on full training set ---
    n_neg_full = (y_train == 0).sum()
    n_pos_full = (y_train == 1).sum()
    scale_pos_weight_full = float(n_neg_full) / float(n_pos_full) if n_pos_full > 0 else 1.0

    lgb_params_final = {k: v for k, v in lgb_params.items() if k != "scale_pos_weight"}
    lgb_params_final["is_unbalance"] = True
    lgb_params_final["verbosity"] = -1
    lgb_final = lgb.LGBMClassifier(**lgb_params_final)
    lgb_final.fit(X_train, y_train)

    xgb_params_final = {**xgb_params, "scale_pos_weight": scale_pos_weight_full}
    xgb_final = xgb.XGBClassifier(**xgb_params_final)
    xgb_final.fit(X_train, y_train)

    cat_params_final = {**cat_params, "scale_pos_weight": scale_pos_weight_full}
    cat_final = CatBoostClassifier(**cat_params_final)
    cat_final.fit(X_train.to_numpy(), y_train.to_numpy(), verbose=False)

    # --- Evaluate individual base models on OOT (if provided) ---
    if X_oot is not None and y_oot is not None:
        lgb_metrics = evaluate_model(lgb_final, X_oot, y_oot, "LightGBM")
        xgb_metrics = evaluate_model(xgb_final, X_oot, y_oot, "XGBoost")
        cat_metrics = evaluate_model(cat_final, X_oot, y_oot, "CatBoost")
        base_gini_dict = {
            "lgb": float(lgb_metrics["Gini"]),
            "xgb": float(xgb_metrics["Gini"]),
            "cat": float(cat_metrics["Gini"]),
        }
    else:
        base_gini_dict = {"lgb": float("nan"), "xgb": float("nan"), "cat": float("nan")}

    # --- Create ensemble model ---
    if method == "average":
        ensemble_model = _AverageEnsemble3(lgb_final, xgb_final, cat_final)
    elif method == "logistic":
        X_meta = np.column_stack([oof_lgb, oof_xgb, oof_cat])
        meta_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=_RANDOM_STATE)
        meta_lr.fit(X_meta, y_train)
        ensemble_model = _LogisticEnsemble3(
            lgb_final, xgb_final, cat_final, meta_lr,
            oof_lgb=oof_lgb, oof_xgb=oof_xgb, oof_cat=oof_cat,
        )
    else:
        raise ValueError(f"method must be 'average' or 'logistic', got '{method}'")

    # --- Evaluate ensemble on OOT (if provided) or internal test set ---
    if X_oot is not None and y_oot is not None:
        metrics_dict = evaluate_model(ensemble_model, X_oot, y_oot, f"Ensemble3 ({method})")
    else:
        metrics_dict = evaluate_model(ensemble_model, X_test, y_test, f"Ensemble3 ({method})")

    return ensemble_model, metrics_dict, X_test, y_test, base_gini_dict



# ---------------------------------------------------------------------------
# Ensemble gating logic
# ---------------------------------------------------------------------------

def _evaluate_ensemble_gate(oot_gini: float, best_single_gini: float) -> str:
    """
    Gate logic for ensemble Gini thresholds (D-12).

    Determines ensemble acceptance based on out-of-time Gini coefficient and
    improvement over the best single base model.

    Parameters
    ----------
    oot_gini : float
        Out-of-time Gini coefficient for ensemble model.
    best_single_gini : float
        Best single model Gini coefficient (max of LGB, XGB, CatBoost).

    Returns
    -------
    str
        Gate result indicating ensemble acceptance:
        - 'full_pass': oot_gini >= 0.65 (aspirational, strong ensemble)
        - 'accept_best_available': oot_gini >= 0.58 AND lift >= 0.005
          (acceptable ensemble with meaningful improvement)
        - 'investigate': oot_gini < 0.58 OR lift < 0.005
          (ensemble underperforms; further analysis needed)
    """
    if oot_gini >= 0.65:
        return "full_pass"
    elif oot_gini >= 0.58 and (oot_gini - best_single_gini) >= 0.005:
        return "accept_best_available"
    else:
        return "investigate"



# ---------------------------------------------------------------------------
# Ensemble workflow orchestration
# ---------------------------------------------------------------------------

def run_ensemble_workflow(
    X: pd.DataFrame,
    y: pd.Series,
    X_raw: pd.DataFrame | None = None,
    lgb_params: dict | None = None,
    xgb_params: dict | None = None,
    cat_model: "CatBoostClassifier | None" = None,
    cat_params: dict | None = None,
    method: Literal["average", "logistic"] = "logistic",
    X_oot: pd.DataFrame | None = None,
    y_oot: pd.Series | None = None,
) -> dict:
    """
    Train LGB + XGB base models and a stacked ensemble; persist if Gini improves.

    Uses pre-supplied hyperparameters (or _ENSEMBLE_LGB/XGB_DEFAULTS when None)
    so that callers can pass Optuna best_params without re-running HPO.  All three
    models are evaluated on the same held-out test set (identical train_test_split
    seed), making the Gini comparison statistically fair.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (typically WoE-encoded for logistic meta-learner).
    y : pd.Series
        Binary target labels.
    X_raw : pd.DataFrame, optional
        Raw (non-WoE) feature matrix for tree models (LightGBM, XGBoost).
        When provided, tree models receive X_raw instead of X, allowing them
        to find continuous splits. When None (default), tree models use X.
    lgb_params : dict, optional
        LightGBM hyperparameters. Defaults to _ENSEMBLE_LGB_DEFAULTS.
    xgb_params : dict, optional
        XGBoost hyperparameters. Defaults to _ENSEMBLE_XGB_DEFAULTS.
    cat_model : CatBoostClassifier, optional
        When provided, routes to the 3-model ensemble path (LGB + XGB + CatBoost).
        When None (default), uses the existing 2-model path (LGB + XGB).
    cat_params : dict, optional
        CatBoost hyperparameters. Defaults to _ENSEMBLE_CAT_DEFAULTS.
        Only used when cat_model is not None.
    method : {'average', 'logistic'}, default='logistic'
        Meta-learner type passed to train_ensemble() or train_ensemble_3model().

    Returns
    -------
    result : dict
        Keys: lgb_gini, xgb_gini, ensemble_gini, improvement, persisted.
        When cat_model is provided, also includes cat_gini.
        ``improvement`` = ensemble_gini − max(lgb_gini, xgb_gini[, cat_gini]).
        ``persisted`` is True when the ensemble was written to disk.
    """
    import json as _json
    import lightgbm as lgb
    import xgboost as xgb

    if lgb_params is None:
        lgb_params = _ENSEMBLE_LGB_DEFAULTS.copy()
    if xgb_params is None:
        xgb_params = _ENSEMBLE_XGB_DEFAULTS.copy()

    if cat_model is not None:
        # 3-model path: LGB + XGB + CatBoost OOF stacking
        # X_oot/y_oot are passed through so base models and the ensemble are
        # evaluated on the regulatory OOT set (Basel CRE36.54), not an
        # internal random holdout carved from the 80% training set.
        ensemble, metrics_dict, X_internal_test, y_internal_test, base_gini = train_ensemble_3model(
            X, y,
            lgb_params=lgb_params,
            xgb_params=xgb_params,
            cat_params=cat_params,
            method=method,
            X_oot=X_oot,
            y_oot=y_oot,
        )
        ensemble_gini = float(metrics_dict["Gini"])
        lgb_gini = base_gini["lgb"]
        xgb_gini = base_gini["xgb"]
        cat_gini = base_gini["cat"]
        best_single_gini = max(lgb_gini, xgb_gini, cat_gini)
        improvement = ensemble_gini - best_single_gini
        persisted = improvement >= _ENSEMBLE_PERSIST_THRESHOLD
        gate_result = _evaluate_ensemble_gate(ensemble_gini, best_single_gini)
        if persisted:
            save_model(ensemble, _ENSEMBLE_3MODEL_WORKFLOW_MODEL_PATH)
            weights_payload = {
                "lgb_gini": lgb_gini,
                "xgb_gini": xgb_gini,
                "cat_gini": cat_gini,
                "ensemble_gini": ensemble_gini,
                "improvement": improvement,
                "method": method,
                "gate_result": gate_result,
            }
            weights_path = Path(_ENSEMBLE_3MODEL_WORKFLOW_WEIGHTS_PATH)
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            with weights_path.open("w") as fh:
                _json.dump(weights_payload, fh, indent=2)
        return {
            "lgb_gini": lgb_gini,
            "xgb_gini": xgb_gini,
            "cat_gini": cat_gini,
            "ensemble_gini": ensemble_gini,
            "improvement": improvement,
            "persisted": persisted,
            "gate_result": gate_result,
            "ensemble_model": ensemble,
        }

    # 2-model path (backward compatible) — unchanged
    X_tree = X_raw if X_raw is not None else X

    X_tree_train, X_tree_test, y_train, y_test = train_test_split(
        X_tree, y, test_size=_TEST_SIZE, stratify=y, random_state=_RANDOM_STATE
    )

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0

    lgb_params_final = {**lgb_params, "scale_pos_weight": scale_pos_weight}
    lgb_model = lgb.LGBMClassifier(**lgb_params_final)
    lgb_model.fit(X_tree_train, y_train)
    lgb_metrics = evaluate_model(lgb_model, X_tree_test, y_test, "LightGBM")
    lgb_gini: float = float(lgb_metrics["Gini"])

    xgb_params_final = {**xgb_params, "scale_pos_weight": scale_pos_weight}
    xgb_model = xgb.XGBClassifier(**xgb_params_final)
    xgb_model.fit(X_tree_train, y_train)
    xgb_metrics = evaluate_model(xgb_model, X_tree_test, y_test, "XGBoost")
    xgb_gini: float = float(xgb_metrics["Gini"])

    ensemble_model, ensemble_metrics = train_ensemble(
        X, y, lgb_params=lgb_params, xgb_params=xgb_params, method=method
    )
    ensemble_gini: float = float(ensemble_metrics["Gini"])

    best_single_gini = max(lgb_gini, xgb_gini)
    improvement = ensemble_gini - best_single_gini
    persisted = improvement >= _ENSEMBLE_PERSIST_THRESHOLD

    if persisted:
        save_model(ensemble_model, _ENSEMBLE_WORKFLOW_MODEL_PATH)

        weights_payload = {
            "lgb_gini": lgb_gini,
            "xgb_gini": xgb_gini,
            "ensemble_gini": ensemble_gini,
            "improvement": improvement,
            "method": method,
        }
        weights_path = Path(_ENSEMBLE_WORKFLOW_WEIGHTS_PATH)
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        with weights_path.open("w") as fh:
            _json.dump(weights_payload, fh, indent=2)

    return {
        "lgb_gini": lgb_gini,
        "xgb_gini": xgb_gini,
        "ensemble_gini": ensemble_gini,
        "improvement": improvement,
        "persisted": persisted,
    }



# ---------------------------------------------------------------------------
# Logistic regression meta-learner (2-model ensemble)
# ---------------------------------------------------------------------------

class _LogisticEnsemble:
    """
    Stacked ensemble with logistic meta-learner.

    Base models (LightGBM, XGBoost) generate OOF features that feed
    a logistic regression meta-model. This allows the meta-model to
    learn optimal weighting of base predictions.

    Parameters
    ----------
    lgb_model : object
        Fitted LightGBM model.
    xgb_model : object
        Fitted XGBoost model.
    meta_lr : LogisticRegression
        Fitted logistic regression on OOF features [oof_lgb, oof_xgb].
    """

    def __init__(self, lgb_model: object, xgb_model: object, meta_lr: LogisticRegression):
        """Initialize with pre-fitted base models and meta-learner."""
        self.lgb_model = lgb_model
        self.xgb_model = xgb_model
        self.meta_lr = meta_lr

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities via stacked logistic meta-learner.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        proba : np.ndarray
            Shape (n_samples, 2) with columns [P(y=0), P(y=1)].
            Each row sums to 1.0.
        """
        p_lgb = self.lgb_model.predict_proba(X)[:, 1]
        p_xgb = self.xgb_model.predict_proba(X)[:, 1]
        X_meta = np.column_stack([p_lgb, p_xgb])
        return self.meta_lr.predict_proba(X_meta)



# ---------------------------------------------------------------------------
# Average ensemble (3-model)
# ---------------------------------------------------------------------------

class _AverageEnsemble3:
    """Simple probability average of three fitted estimators (LGB + XGB + CatBoost)."""

    def __init__(self, lgb_model: object, xgb_model: object, cat_model: object):
        """Initialize with pre-fitted base models."""
        self.lgb_model = lgb_model
        self.xgb_model = xgb_model
        self.cat_model = cat_model

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities via simple averaging of three base models.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        proba : np.ndarray
            Shape (n_samples, 2) with columns [P(y=0), P(y=1)].
        """
        p_lgb = self.lgb_model.predict_proba(X)[:, 1]
        p_xgb = self.xgb_model.predict_proba(X)[:, 1]
        X_np = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        p_cat = self.cat_model.predict_proba(X_np)[:, 1]
        avg = (p_lgb + p_xgb + p_cat) / 3.0
        return np.column_stack([1 - avg, avg])



# ---------------------------------------------------------------------------
# Logistic regression meta-learner (3-model ensemble)
# ---------------------------------------------------------------------------

class _LogisticEnsemble3:
    """
    Stacked ensemble with logistic meta-learner — 3 base models (LGB + XGB + CatBoost).

    Meta-learner trained on np.column_stack([oof_lgb, oof_xgb, oof_cat]) with L2
    regularisation (C=1.0).
    """

    # Required by sklearn's is_classifier() so CalibratedClassifierCV / FrozenEstimator
    # treats this as a classifier (enables predict_proba routing).
    _estimator_type = "classifier"

    def __init__(
        self,
        lgb_model: object,
        xgb_model: object,
        cat_model: object,
        meta_lr: LogisticRegression,
        oof_lgb: np.ndarray | None = None,
        oof_xgb: np.ndarray | None = None,
        oof_cat: np.ndarray | None = None,
    ):
        """
        Initialize with pre-fitted base models and meta-learner.

        Parameters
        ----------
        lgb_model : object
            Fitted LightGBM model.
        xgb_model : object
            Fitted XGBoost model.
        cat_model : object
            Fitted CatBoost model.
        meta_lr : LogisticRegression
            Fitted logistic regression meta-learner.
        oof_lgb : np.ndarray, optional
            Out-of-fold predictions from LGB during training. Used for
            regulatory documentation and model audit trails.
        oof_xgb : np.ndarray, optional
            Out-of-fold predictions from XGB during training.
        oof_cat : np.ndarray, optional
            Out-of-fold predictions from CatBoost during training.
        """
        self.lgb_model = lgb_model
        self.xgb_model = xgb_model
        self.cat_model = cat_model
        self.meta_lr = meta_lr
        self.oof_lgb = oof_lgb if oof_lgb is not None else np.array([])
        self.oof_xgb = oof_xgb if oof_xgb is not None else np.array([])
        self.oof_cat = oof_cat if oof_cat is not None else np.array([])
        # Required by sklearn's cross_val_predict/_fit_and_predict pipeline
        # when CalibratedClassifierCV uses cross-validation on a FrozenEstimator.
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities via stacked logistic meta-learner (3 inputs).

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        proba : np.ndarray
            Shape (n_samples, 2) with columns [P(y=0), P(y=1)].
        """
        p_lgb = self.lgb_model.predict_proba(X)[:, 1]
        p_xgb = self.xgb_model.predict_proba(X)[:, 1]
        X_np = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        p_cat = self.cat_model.predict_proba(X_np)[:, 1]
        X_meta = np.column_stack([p_lgb, p_xgb, p_cat])
        return self.meta_lr.predict_proba(X_meta)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict binary class labels (threshold at 0.5).

        Required to satisfy sklearn's estimator protocol so that
        FrozenEstimator wrapping this class passes cross_val_predict's
        parameter validation in CalibratedClassifierCV.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        labels : np.ndarray
            Shape (n_samples,) with integer class labels {0, 1}.
        """
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "_LogisticEnsemble3":
        """
        No-op fit — base models and meta-learner are already trained.

        Required so that FrozenEstimator.fit() can call check_is_fitted()
        without raising TypeError, and so that cross_val_predict's estimator
        validator (which checks for 'fit' on the wrapped estimator) passes.

        Parameters
        ----------
        X : pd.DataFrame
            Ignored.
        y : pd.Series or None
            Ignored.

        Returns
        -------
        self : _LogisticEnsemble3
        """
        return self

    def __getattr__(self, name: str) -> object:
        """
        Fallback for attributes missing from pickled instances.

        Pickled objects predate any attributes added to __init__ after pickling
        — the instance __dict__ is restored verbatim from the pickle, so new
        __init__ assignments are never re-executed. __getattr__ is only called
        when normal attribute lookup fails, making it safe as a fallback.

        Currently handles:
        - classes_: required by sklearn's cross_val_predict/_fit_and_predict
          after CalibratedClassifierCV calls predict_proba on each fold.
        """
        if name == "classes_":
            return np.array([0, 1])
        raise AttributeError(f"'_LogisticEnsemble3' object has no attribute '{name}'")

    def __sklearn_is_fitted__(self) -> bool:
        """
        Signal to sklearn's check_is_fitted that this estimator is always ready.

        CalibratedClassifierCV (FrozenEstimator path) calls
        check_is_fitted(inner_estimator) during fit. Without this hook,
        check_is_fitted would look for attributes ending with '_' and raise
        NotFittedError because _LogisticEnsemble3 stores models under plain
        names (lgb_model, xgb_model, etc.).
        """
        return True


# ---------------------------------------------------------------------------
# Priority 2.2 — CatBoost Optuna HPO + feature preparation helper
# ---------------------------------------------------------------------------


