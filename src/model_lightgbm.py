"""
model_lightgbm.py
-----------------
model_lightgbm training and optimization functions.
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
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import optuna
from optuna.integration import LightGBMPruningCallback
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_validate, StratifiedKFold

from src.features import _PROJECT_ROOT
from src.utils import gini_coefficient, ks_statistic, evaluate_model
from src.model_base import (
    _make_cv, _find_optimal_threshold_f1_macro, save_model, load_model, calibrate_model,
    _OOFGiniMonitorCallback, _TemporalCV,
    _TEST_SIZE, _RANDOM_STATE, _CV_N_SPLITS, _CV_EMBARGO_FRAC, _TEMPORAL_SORT_COL,
    _THRESHOLD_MIN, _THRESHOLD_MAX,
    # XGBoost constants
    _XGB_OPTUNA_N_TRIALS, _XGB_N_ESTIMATORS_MIN, _XGB_N_ESTIMATORS_MAX,
    _XGB_MAX_DEPTH_MIN, _XGB_MAX_DEPTH_MAX, _XGB_LEARNING_RATE_MIN, _XGB_LEARNING_RATE_MAX,
    _XGB_SUBSAMPLE_MIN, _XGB_SUBSAMPLE_MAX, _XGB_COLSAMPLE_BYTREE_MIN, _XGB_COLSAMPLE_BYTREE_MAX,
    _XGB_MIN_CHILD_WEIGHT_MIN, _XGB_MIN_CHILD_WEIGHT_MAX, _XGB_GAMMA_MIN, _XGB_GAMMA_MAX,
    _XGB_MAX_DELTA_STEP_MIN, _XGB_MAX_DELTA_STEP_MAX, _XGB_REG_ALPHA_MIN, _XGB_REG_ALPHA_MAX,
    _XGB_REG_LAMBDA_MIN, _XGB_REG_LAMBDA_MAX,
    _XGB_RAW_N_ESTIMATORS, _XGB_RAW_EARLY_STOPPING_ROUNDS, _XGB_RAW_MIN_CHILD_WEIGHT_MAX,
    _XGB_RAW_GAMMA_MAX, _XGB_RAW_REG_ALPHA_MIN, _XGB_RAW_REG_LAMBDA_MIN, _XGB_RAW_REG_MAX,
    _XGB_RAW_STUDY_NAME, _HPO_PROGRESS_LOG_PATH, _XGB_OPTUNA_MODEL_PATH, _XGB_OPTUNA_PARAMS_PATH,
    _XGB_OPTUNA_FIGURE_PATH, _XGB_EXTENDED_OPTUNA_N_TRIALS,
    _XGB_RAW_N_ESTIMATORS_MIN, _XGB_RAW_N_ESTIMATORS_MAX, _XGB_RAW_MAX_DEPTH_MIN, _XGB_RAW_MAX_DEPTH_MAX,
    _XGB_RAW_LEARNING_RATE_MIN, _XGB_RAW_LEARNING_RATE_MAX, _XGB_RAW_SUBSAMPLE_MIN, _XGB_RAW_SUBSAMPLE_MAX,
    _XGB_RAW_COLSAMPLE_BYTREE_MIN, _XGB_RAW_COLSAMPLE_BYTREE_MAX, _XGB_RAW_MIN_CHILD_WEIGHT_MIN, _XGB_RAW_GAMMA_MIN,
    # LightGBM constants
    _LGB_OPTUNA_N_TRIALS, _LGB_NUM_LEAVES_MIN, _LGB_NUM_LEAVES_MAX, _LGB_RAW_NUM_LEAVES_MAX,
    _LGB_MAX_DEPTH_MIN, _LGB_MAX_DEPTH_MAX, _LGB_LEARNING_RATE_MIN, _LGB_LEARNING_RATE_MAX,
    _LGB_N_ESTIMATORS_MIN, _LGB_N_ESTIMATORS_MAX, _LGB_MIN_CHILD_SAMPLES_MIN, _LGB_MIN_CHILD_SAMPLES_MAX,
    _LGB_SUBSAMPLE_MIN, _LGB_SUBSAMPLE_MAX, _LGB_COLSAMPLE_BYTREE_MIN, _LGB_COLSAMPLE_BYTREE_MAX,
    _LGB_REG_ALPHA_MIN, _LGB_REG_ALPHA_MAX, _LGB_REG_LAMBDA_MIN, _LGB_REG_LAMBDA_MAX, _LGB_MIN_CHILD_WEIGHT_MIN, _LGB_MIN_CHILD_WEIGHT_MAX,
    _LGB_OBJ_EARLY_STOPPING_ROUNDS, _LGB_EARLY_STOPPING_ROUNDS, _LGB_FINAL_VAL_SIZE, _LGB_PATH_SMOOTH_MIN, _LGB_PATH_SMOOTH_MAX,
    _LGB_DART_DROP_RATE_MIN, _LGB_DART_DROP_RATE_MAX, _LGB_GOSS_TOP_RATE_MIN, _LGB_GOSS_TOP_RATE_MAX,
    _LGB_GOSS_OTHER_RATE_MIN, _LGB_GOSS_OTHER_RATE_MAX, _LGB_OPTUNA_MODEL_PATH, _LGB_OPTUNA_PARAMS_PATH,
    _LGB_OPTUNA_FIGURE_PATH, _LGB_RAW_NUM_LEAVES_MIN, _LGB_RAW_MAX_DEPTH_MIN, _LGB_RAW_MAX_DEPTH_MAX,
    _LGB_RAW_LEARNING_RATE_MIN, _LGB_RAW_LEARNING_RATE_MAX, _LGB_RAW_N_ESTIMATORS_MIN, _LGB_RAW_N_ESTIMATORS_MAX,
    _LGB_RAW_MIN_CHILD_SAMPLES_MIN, _LGB_RAW_MIN_CHILD_SAMPLES_MAX, _LGB_RAW_SUBSAMPLE_MIN, _LGB_RAW_SUBSAMPLE_MAX,
    _LGB_RAW_COLSAMPLE_BYTREE_MIN, _LGB_RAW_COLSAMPLE_BYTREE_MAX, _LGB_RAW_REG_ALPHA_MIN, _LGB_RAW_REG_ALPHA_MAX,
    _LGB_RAW_REG_LAMBDA_MIN, _LGB_RAW_REG_LAMBDA_MAX, _LGB_EXTENDED_OPTUNA_N_TRIALS, _LGB_METRIC,
    _LGB_RAW_N_ESTIMATORS, _LGB_EXTENDED_OPTUNA_N_TRIALS, _LGB_LEARNING_RATE_MIN as _LGB_MIN_LR,
    # CatBoost constants
    _CAT_DEPTH_MIN, _CAT_DEPTH_MAX, _CAT_LEARNING_RATE_MIN, _CAT_LEARNING_RATE_MAX,
    _CAT_L2_LEAF_REG_MIN, _CAT_L2_LEAF_REG_MAX, _CAT_BAGGING_TEMP_MIN, _CAT_BAGGING_TEMP_MAX,
    _CAT_RANDOM_STRENGTH_MIN, _CAT_RANDOM_STRENGTH_MAX, _CAT_BOOTSTRAP_TYPE, _CAT_ITERATIONS,
    _CAT_OBJ_EARLY_STOPPING_ROUNDS, _CAT_EARLY_STOPPING_ROUNDS, _CAT_FINAL_VAL_SIZE, _CAT_OPTUNA_N_TRIALS,
    _CAT_MODEL_PATH, _CAT_PARAMS_PATH, _CAT_FIGURE_PATH, _CAT_RAW_MIN_DATA_IN_LEAF_MIN, _CAT_RAW_MIN_DATA_IN_LEAF_MAX,
    _CATBOOST_RAW_CATS, _CAT_EXTENDED_OPTUNA_N_TRIALS, _CAT_RAW_DEPTH_MIN, _CAT_RAW_DEPTH_MAX,
    _CAT_RAW_LEARNING_RATE_MIN, _CAT_RAW_LEARNING_RATE_MAX, _CAT_RAW_L2_LEAF_REG_MIN, _CAT_RAW_L2_LEAF_REG_MAX,
    _CAT_RAW_ITERATIONS_MIN, _CAT_RAW_ITERATIONS_MAX,
    # Other constants  
    _STRATEGY_SMOTE, _STRATEGY_COST_SENSITIVE, _ENSEMBLE_LGB_DEFAULTS, _ENSEMBLE_XGB_DEFAULTS,
    _ENSEMBLE_CAT_DEFAULTS, _ENSEMBLE_3MODEL_WORKFLOW_MODEL_PATH, _ENSEMBLE_3MODEL_WORKFLOW_WEIGHTS_PATH,
    _ENSEMBLE_PERSIST_THRESHOLD, _OPTUNA_DB_PATH, _BENCHMARK_REPORT_PATH, _IMBALANCE_STRATEGIES,
)

def _lightgbm_optuna_objective(
    trial: "optuna.Trial",
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: "StratifiedKFold | _TemporalCV",
    scale_pos_weight: float | None = None,
    num_leaves_max: int = _LGB_NUM_LEAVES_MAX,
    boosting_type: Literal["gbdt", "dart", "goss"] = "gbdt",
    monotone_constraints: dict[str, int] | None = None,
) -> float:
    """
    Optuna objective: 5-fold CV AUC-ROC for a suggested LightGBM configuration.

    Called once per trial by ``study.optimize()``. Samples 9 base hyperparameters
    plus booster-specific parameters, runs temporal CV on X_train with early
    stopping inside each fold (skipped for DART — not supported in LGB 4.x),
    and returns the mean out-of-fold AUC-ROC. X_test is never passed in.

    Parameters
    ----------
    trial : optuna.Trial
        Current Optuna trial handle for hyperparameter suggestions.
    X_train : pd.DataFrame
        Training features (80% split). Never the held-out test set.
    y_train : pd.Series
        Training labels.
    cv : StratifiedKFold | _TemporalCV
        CV splitter, seeded for reproducibility.
    scale_pos_weight : float | None
        If provided, used for cost-sensitive imbalance handling.
    num_leaves_max : int
        Upper bound for num_leaves search.
    boosting_type : {'gbdt', 'dart', 'goss'}, optional
        LightGBM booster algorithm. Default 'gbdt'. DART and GOSS add
        booster-specific hyperparameters to the Optuna search space.
        Note: DART does not support early stopping (LGB 4.x) — the objective
        trains to full n_estimators in that case.
    monotone_constraints : dict[str, int] | None, optional
        Map of feature name → direction (+1 = monotone increasing,
        -1 = monotone decreasing, 0 = unconstrained). Features not in
        the dict default to 0. Converted to the column-ordered list
        required by LightGBM internally.

    Returns
    -------
    float
        Mean out-of-fold AUC-ROC across all CV folds.
    """
    import lightgbm as lgb

    params: dict = {
        "num_leaves": trial.suggest_int(
            "num_leaves", _LGB_NUM_LEAVES_MIN, num_leaves_max
        ),
        "max_depth": trial.suggest_int(
            "max_depth", _LGB_MAX_DEPTH_MIN, _LGB_MAX_DEPTH_MAX
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", _LGB_LEARNING_RATE_MIN, _LGB_LEARNING_RATE_MAX, log=True
        ),
        "n_estimators": trial.suggest_int(
            "n_estimators", _LGB_N_ESTIMATORS_MIN, _LGB_N_ESTIMATORS_MAX
        ),
        "min_child_samples": trial.suggest_int(
            "min_child_samples", _LGB_MIN_CHILD_SAMPLES_MIN, _LGB_MIN_CHILD_SAMPLES_MAX
        ),
        "subsample": trial.suggest_float(
            "subsample", _LGB_SUBSAMPLE_MIN, _LGB_SUBSAMPLE_MAX
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", _LGB_COLSAMPLE_BYTREE_MIN, _LGB_COLSAMPLE_BYTREE_MAX
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", _LGB_REG_ALPHA_MIN, _LGB_REG_ALPHA_MAX
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", _LGB_REG_LAMBDA_MIN, _LGB_REG_LAMBDA_MAX
        ),
        "boosting_type": boosting_type,
        "metric": "auc",      # CRITICAL: binary_logloss early-stop fires at iter 1 with is_unbalance=True
        "verbosity": -1,
        "random_state": _RANDOM_STATE,
    }

    # DART: add dropout hyperparameters to the search space.
    # drop_rate controls the fraction of trees dropped per boosting round —
    # the primary regularisation knob in DART.
    if boosting_type == "dart":
        params["drop_rate"] = trial.suggest_float(
            "drop_rate", _LGB_DART_DROP_RATE_MIN, _LGB_DART_DROP_RATE_MAX
        )
    # GOSS: gradient-based one-side sampling — retain top-gradient instances
    # (top_rate) plus a random sample of the remainder (other_rate).
    elif boosting_type == "goss":
        params["top_rate"] = trial.suggest_float(
            "top_rate", _LGB_GOSS_TOP_RATE_MIN, _LGB_GOSS_TOP_RATE_MAX
        )
        params["other_rate"] = trial.suggest_float(
            "other_rate", _LGB_GOSS_OTHER_RATE_MIN, _LGB_GOSS_OTHER_RATE_MAX
        )

    # Imbalance handling: scale_pos_weight preserves natural probability range
    # (adjusts gradient weights only); is_unbalance compresses leaf outputs.
    # scale_pos_weight is preferred when rank-based Gini is the target metric.
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight
    else:
        params["is_unbalance"] = True

    # Monotone constraints: convert feature-name dict to column-ordered list.
    # LightGBM requires constraints as a list aligned to X_train.columns.
    if monotone_constraints is not None:
        cols = X_train.columns.tolist()
        params["monotone_constraints"] = [monotone_constraints.get(c, 0) for c in cols]

    # Early stopping callbacks — silenced for DART (not supported in LGB 4.x;
    # dropout is the regulariser and trains to full n_estimators by design).
    _use_early_stopping = boosting_type != "dart"
    callbacks = [lgb.log_evaluation(period=0)]
    if _use_early_stopping:
        callbacks.insert(
            0,
            lgb.early_stopping(stopping_rounds=_LGB_OBJ_EARLY_STOPPING_ROUNDS, verbose=False),
        )

    fold_aucs: list[float] = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        model = lgb.LGBMClassifier(**params)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Early stopping is not available in dart mode")
            model.fit(
                X_fold_train,
                y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                callbacks=callbacks,
            )
        y_prob_val = model.predict_proba(X_fold_val)[:, 1]
        fold_aucs.append(float(roc_auc_score(y_fold_val, y_prob_val)))

    return float(np.mean(fold_aucs))


def train_lightgbm_optuna(
    feature_store_path: str,
    n_trials: int = _LGB_OPTUNA_N_TRIALS,
    imbalance_strategy: str = "scale_pos_weight",
    groups: pd.Series | None = None,
    boosting_type: str = "gbdt",
    monotone_constraints: dict | None = None,
) -> tuple[object, dict, pd.DataFrame, pd.Series, dict]:
    """
    Train LightGBM with Optuna HPO on raw feature store.

    Loads feature store from disk, runs Bayesian HPO via Optuna (HyperbandPruner + TPESampler),
    applies Platt calibration, and saves artifacts. Supports 3 imbalance strategies:
    - "scale_pos_weight": LGB native cost-sensitive weighting (pos_weight = n_neg/n_pos)
    - "is_unbalance": LGB internal rebalancing (gradient + leaf value adjustment)
    - "smote": data-level rebalancing via imblearn.pipeline.Pipeline (inside CV folds only)

    **Basel CRE36.54 compliance:** An out-of-time (OOT) holdout of the most-recent
    ``_TEST_SIZE`` (20%) of rows — sorted by ``_TEMPORAL_SORT_COL`` — is carved out
    *before* HPO begins and is never seen during Optuna trials or CV folds. Final OOT
    Gini is reported after full refit on the remaining 80%, satisfying the regulatory
    requirement for temporal model validation in internal ratings-based (IRB) models.

    Parameters
    ----------
    feature_store_path : str
        Path to parquet file containing feature matrix + TARGET column.
    n_trials : int
        Number of Optuna trials (default _LGB_OPTUNA_N_TRIALS = 50).
    imbalance_strategy : str
        Strategy for class imbalance: "scale_pos_weight" | "is_unbalance" | "smote"
    groups : pd.Series, optional
        Temporal CV groups (e.g., application year); if None, temporal CV auto-detected from DATA_LOAD_YEAR.
    boosting_type : str, optional
        LGB boosting algorithm: "gbdt" (default) | "dart" | "goss".
        DART adds drop_rate to the Optuna search space; GOSS adds top_rate/other_rate.
        Early stopping is disabled for DART (unsupported by LGB).
    monotone_constraints : dict[str, int] | None, optional
        Column-name → direction (+1 increasing, -1 decreasing, 0 none).
        Keys must all exist in X columns. Overrides the EXT_SOURCE auto-detect.

    Returns
    -------
    tuple[object, dict, pd.DataFrame, pd.Series, dict]
        (calibrated_model, metrics_dict, X_test, y_test, best_params)
        - calibrated_model: LGBMClassifier with Platt calibration applied
        - metrics_dict: {Gini, AUC-ROC, KS, Brier, BrierSkill, oof_gini, oot_gini}
        - X_test: OOT held-out test set features (most-recent 20% by temporal order)
        - y_test: OOT held-out test set labels
        - best_params: best hyperparameters from Optuna study
    """
    import json as _json

    import lightgbm as lgb
    import matplotlib.pyplot as plt
    import optuna

    # Load feature store
    feature_path = Path(feature_store_path)
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature store not found: {feature_path}")

    df = pd.read_parquet(feature_path)

    # Extract TARGET
    if "TARGET" not in df.columns:
        raise ValueError(f"TARGET column not found in {feature_path}")

    y = df.pop("TARGET")
    X = df

    # Input guards
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}.")

    if imbalance_strategy not in _IMBALANCE_STRATEGIES:
        raise ValueError(
            f"imbalance_strategy must be one of {_IMBALANCE_STRATEGIES}, "
            f"got {imbalance_strategy!r}."
        )

    _VALID_BOOSTING_TYPES = {"gbdt", "dart", "goss"}
    if boosting_type not in _VALID_BOOSTING_TYPES:
        raise ValueError(
            f"boosting_type must be one of {sorted(_VALID_BOOSTING_TYPES)}, "
            f"got {boosting_type!r}."
        )

    if monotone_constraints is not None:
        unknown_keys = set(monotone_constraints.keys()) - set(X.columns)
        if unknown_keys:
            raise ValueError(
                f"monotone_constraints keys not found in X: {sorted(unknown_keys)}"
            )

    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        raise ValueError("y has no positive samples.")
    if n_neg == 0:
        raise ValueError("y has no negative samples.")

    # --- OOT temporal split — Basel CRE36.54 ---
    # Hold out most-recent 20% BEFORE HPO; X and y are reassigned to training 80%
    # so every downstream reference (CV objective, refit, calibration) is clean.
    if _TEMPORAL_SORT_COL not in X.columns:
        raise ValueError(
            f"Temporal sort column '{_TEMPORAL_SORT_COL}' not in X. "
            "OOT split is required for Basel CRE36 compliance. "
            "Rebuild feature store with build_tree_feature_store()."
        )
    _lgb_temporal_vals = X[_TEMPORAL_SORT_COL].values
    _lgb_nan_mask = np.isnan(_lgb_temporal_vals)
    _lgb_known_pos = np.where(~_lgb_nan_mask)[0]
    _lgb_unknown_pos = np.where(_lgb_nan_mask)[0]
    _lgb_known_sorted = _lgb_known_pos[np.argsort(_lgb_temporal_vals[_lgb_known_pos])]
    _lgb_oot_known_cut = int(len(_lgb_known_sorted) * (1 - _TEST_SIZE))
    _lgb_oot_known = _lgb_known_sorted[_lgb_oot_known_cut:]
    _lgb_train_known = _lgb_known_sorted[:_lgb_oot_known_cut]
    _lgb_rng = np.random.default_rng(_RANDOM_STATE)
    _lgb_unknown_perm = _lgb_rng.permutation(len(_lgb_unknown_pos))
    _lgb_oot_unknown_cut = int(len(_lgb_unknown_pos) * (1 - _TEST_SIZE))
    _lgb_oot_unknown = _lgb_unknown_pos[_lgb_unknown_perm[_lgb_oot_unknown_cut:]]
    _lgb_train_unknown = _lgb_unknown_pos[_lgb_unknown_perm[:_lgb_oot_unknown_cut]]
    _lgb_oot_indices = np.concatenate([_lgb_oot_known, _lgb_oot_unknown])
    _lgb_train_indices = np.concatenate([_lgb_train_known, _lgb_train_unknown])
    X_oot = X.iloc[_lgb_oot_indices].copy()
    y_oot = y.iloc[_lgb_oot_indices].copy()
    X = X.iloc[_lgb_train_indices].copy()
    y = y.iloc[_lgb_train_indices].copy()
    if groups is not None:
        groups = groups.iloc[_lgb_train_indices].reset_index(drop=True)

    # Temporal CV
    # Convert groups to numpy array if provided; _make_cv expects (groups_train, n_splits)
    groups_array = groups.values if groups is not None else None
    cv = _make_cv(groups_train=groups_array, n_splits=_CV_N_SPLITS)

    # Determine OOT index (most recent 20% by temporal order, if groups present)
    # Per Task 1: OOT is the most-recent 20% of samples, OOF is the remaining 80%
    if groups is not None:
        # Assume groups are time periods (e.g., application year)
        # OOT is the top 20% of time values
        unique_times = sorted(groups.unique())
        oot_threshold = unique_times[int(len(unique_times) * 0.8)]
        oot_mask = groups >= oot_threshold
        oot_indices = np.where(oot_mask)[0]
        oof_indices = np.where(~oot_mask)[0]
    else:
        # Fallback: OOT is bottom 20% of indices (if no temporal groups)
        oot_size = int(len(y) * 0.2)
        oot_indices = np.arange(len(y) - oot_size, len(y))
        oof_indices = np.arange(len(y) - oot_size)

    # Initialize predictions arrays with NaN (NaN marks unvalidated samples per XGB pattern)
    oof_preds = np.full(len(oof_indices), np.nan)
    oot_preds = np.full(len(oot_indices), np.nan)
    # Track best-trial OOF predictions for post-HPO OOF Gini computation.
    # Using a list container so the objective can update it without nonlocal reassignment.
    _best_oof_preds = [np.full(len(oof_indices), np.nan)]
    _best_oot_gini_seen = [-np.inf]

    # HPO objective with full parameter search space per D-10
    def objective(trial: optuna.Trial) -> float:
        """
        Optuna objective for LGB HPO. Maximizes OOT Gini via temporal CV.

        Imbalance strategy applied within CV folds (SMOTE) or in model params (scale_pos_weight, is_unbalance).
        Returns OOT Gini as the optimization metric.

        OOT (Out-of-Time): most-recent 20% of data by temporal order, used as optimization gate.
        OOF (Out-of-Fold): remaining 80%, accumulated across CV folds for development set discrimination.
        """
        # Suggest hyperparameters (all per CONTEXT.md D-10)
        params = {
            "objective": "binary",
            "metric": _LGB_METRIC,  # "auc" — CRITICAL per D-09
            "verbosity": -1,
            "boosting_type": boosting_type,
            "n_estimators": _LGB_RAW_N_ESTIMATORS,  # Fixed to 1000 per D-07
            "learning_rate": trial.suggest_float("learning_rate", _LGB_LEARNING_RATE_MIN, _LGB_LEARNING_RATE_MAX, log=True),
            "num_leaves": trial.suggest_int("num_leaves", _LGB_NUM_LEAVES_MIN, _LGB_NUM_LEAVES_MAX),
            "max_depth": trial.suggest_int("max_depth", _LGB_MAX_DEPTH_MIN, _LGB_MAX_DEPTH_MAX),
            "min_child_samples": trial.suggest_int("min_child_samples", _LGB_MIN_CHILD_SAMPLES_MIN, _LGB_MIN_CHILD_SAMPLES_MAX),
            "min_child_weight": trial.suggest_float("min_child_weight", _LGB_MIN_CHILD_WEIGHT_MIN, _LGB_MIN_CHILD_WEIGHT_MAX, log=True),
            "subsample": trial.suggest_float("subsample", _LGB_SUBSAMPLE_MIN, _LGB_SUBSAMPLE_MAX),
            "colsample_bytree": trial.suggest_float("colsample_bytree", _LGB_COLSAMPLE_BYTREE_MIN, _LGB_COLSAMPLE_BYTREE_MAX),
            "reg_alpha": trial.suggest_float("reg_alpha", _LGB_REG_ALPHA_MIN, _LGB_REG_ALPHA_MAX, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", _LGB_REG_LAMBDA_MIN, _LGB_REG_LAMBDA_MAX, log=True),
            "path_smooth": trial.suggest_float("path_smooth", _LGB_PATH_SMOOTH_MIN, _LGB_PATH_SMOOTH_MAX),
            "random_state": 42,
        }

        # Boosting-type-specific hyperparameters
        if boosting_type == "dart":
            params["drop_rate"] = trial.suggest_float("drop_rate", 0.05, 0.3)
        elif boosting_type == "goss":
            params["top_rate"] = trial.suggest_float("top_rate", 0.05, 0.5)
            params["other_rate"] = trial.suggest_float("other_rate", 0.05, 0.3)

        # Apply bagging_freq=1 when subsample < 1.0 (per D-03 from CONTEXT.md and lgb.md)
        # GOSS uses its own sampling and cannot use bagging_freq
        if params["subsample"] < 1.0 and boosting_type != "goss":
            params["bagging_freq"] = 1

        # Apply imbalance strategy
        if imbalance_strategy == "scale_pos_weight":
            pos_weight = (y == 0).sum() / (y == 1).sum()
            params["scale_pos_weight"] = pos_weight
        elif imbalance_strategy == "is_unbalance":
            params["is_unbalance"] = True
        elif imbalance_strategy == "smote":
            # SMOTE applied inside CV folds (Task 3), not in params
            pass
        else:
            raise ValueError(f"Unknown imbalance_strategy: {imbalance_strategy}")

        # Monotone constraints: explicit argument wins; fall back to EXT_SOURCE auto-detect
        if monotone_constraints is not None:
            params["monotone_constraints"] = [
                monotone_constraints.get(c, 0) for c in X.columns
            ]
        elif "EXT_SOURCE_1" in X.columns:
            # Auto-detect: external credit bureau scores → decreasing default probability
            monotone_map = {
                col: -1 for col in X.columns if col.startswith("EXT_SOURCE_")
            }
            if monotone_map:
                params["monotone_constraints"] = [
                    monotone_map.get(c, 0) for c in X.columns
                ]

        # Train with temporal CV (OOF + OOT Gini)
        # Use nonlocal to access outer scope oof_preds and oot_preds arrays
        nonlocal oof_preds, oot_preds
        oof_preds_trial = np.full(len(oof_indices), np.nan)
        oot_preds_trial = np.full(len(oot_indices), np.nan)

        best_iteration_list = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Apply SMOTE inside training fold only (per D-14 and CLAUDE.md rule)
            if imbalance_strategy == "smote":
                from imblearn.over_sampling import SMOTE

                smote = SMOTE(sampling_strategy=0.3, random_state=42)
                X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
                # Use resampled data for training
                train_data = lgb.Dataset(X_train_sm, label=y_train_sm, categorical_feature='auto')
            else:
                train_data = lgb.Dataset(X_train, label=y_train, categorical_feature='auto')

            # Train model
            model = lgb.LGBMClassifier(**params)
            # DART does not support early stopping in LightGBM
            fit_callbacks = [lgb.log_evaluation(period=0)]
            if boosting_type != "dart":
                fit_callbacks.append(
                    lgb.early_stopping(_LGB_EARLY_STOPPING_ROUNDS, verbose=False)
                )
            model.fit(
                X_train if imbalance_strategy != "smote" else X_train_sm,
                y_train if imbalance_strategy != "smote" else y_train_sm,
                eval_set=[(X_test, y_test)],
                callbacks=fit_callbacks,
            )

            # Capture best_iteration (per D-08)
            best_iteration_list.append(model.best_iteration_)

            # Accumulate predictions into OOF and OOT arrays
            # OOF: accumulate test_idx predictions in oof_preds_trial
            oof_test_idx = np.intersect1d(test_idx, oof_indices)
            if len(oof_test_idx) > 0:
                oof_positions = np.searchsorted(oof_indices, oof_test_idx)
                oof_preds_trial[oof_positions] = model.predict_proba(X_test.iloc[np.isin(test_idx, oof_test_idx)])[:, 1]

            # OOT: accumulate test_idx predictions in oot_preds_trial
            oot_test_idx = np.intersect1d(test_idx, oot_indices)
            if len(oot_test_idx) > 0:
                oot_positions = np.searchsorted(oot_indices, oot_test_idx)
                oot_preds_trial[oot_positions] = model.predict_proba(X_test.iloc[np.isin(test_idx, oot_test_idx)])[:, 1]

        # Update outer scope arrays
        oof_preds = oof_preds_trial
        oot_preds = oot_preds_trial

        # Compute OOT Gini (primary metric for HPO)
        # Remove NaN predictions (folds that didn't touch OOT)
        from sklearn.metrics import roc_auc_score
        oot_valid = ~np.isnan(oot_preds)
        if oot_valid.sum() > 0:
            oot_gini = 2 * roc_auc_score(y.iloc[oot_indices][oot_valid], oot_preds[oot_valid]) - 1
        else:
            # Fallback to OOF Gini if OOT is empty (rare edge case)
            oof_valid = ~np.isnan(oof_preds)
            oot_gini = 2 * roc_auc_score(y.iloc[oof_indices][oof_valid], oof_preds[oof_valid]) - 1

        # Set user_attr for best_iteration (per D-08)
        trial.set_user_attr("best_iteration", int(np.mean(best_iteration_list)))

        # Track best-trial OOF preds so post-HPO can compute true OOF Gini
        if oot_gini > _best_oot_gini_seen[0]:
            _best_oot_gini_seen[0] = oot_gini
            _best_oof_preds[0] = oof_preds_trial.copy()

        # Report trial value for pruning
        trial.report(oot_gini, step=0)

        return oot_gini

    # Create/load Optuna study
    # Study name: lgb_raw_{store_tag}_{strategy}
    store_name = Path(feature_store_path).stem  # e.g., "X_tree_dfs"
    store_tag_map = {
        "X_train": "Xtrain",
        "X_tree_raw": "Xtreeraw",
        "X_tree_dfs": "Xtreeds",
    }
    store_tag = store_tag_map.get(store_name, store_name)
    study_name = f"lgb_raw_{store_tag}_{imbalance_strategy}"

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=20)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=10, max_resource=50, reduction_factor=3
    )

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=f"sqlite:///{_OPTUNA_DB_PATH}",
        load_if_exists=True,
    )

    # Run HPO
    study.optimize(objective, n_trials=n_trials)

    # Task 2: Best-model selection and return tuple per XGBoost pattern
    best_trial = study.best_trial
    best_params = best_trial.params
    best_iteration = best_trial.user_attrs.get("best_iteration", _LGB_RAW_N_ESTIMATORS)

    # Refit best model on full training set with best parameters + best_iteration
    best_model = lgb.LGBMClassifier(
        **best_params,
        n_estimators=best_iteration + 1,  # +1 because best_iteration is 0-indexed
        boosting_type=boosting_type,
    )

    # Train on full X, y (no CV for final model)
    best_model.fit(X, y, callbacks=[lgb.log_evaluation(period=0)])

    # Split for calibration (70% train, 30% calibration)
    # Use temporal CV if groups present; else stratified
    if groups is not None:
        # Temporal split: train on older 70%, calibrate on newer 30%
        sorted_indices = np.argsort(groups.values)
        split_idx = int(len(sorted_indices) * 0.7)
        train_indices = sorted_indices[:split_idx]
        calib_indices = sorted_indices[split_idx:]
        X_train_cal, X_calib = X.iloc[train_indices], X.iloc[calib_indices]
        y_train_cal, y_calib = y.iloc[train_indices], y.iloc[calib_indices]
    else:
        # Stratified split
        from sklearn.model_selection import train_test_split
        X_train_cal, X_calib, y_train_cal, y_calib = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

    # Task 2: Apply Platt calibration with output paths for best model
    # Paths are determined based on store and strategy (saved to reports/ and models/)
    model_path = _PROJECT_ROOT / "models" / "lightgbm_raw_calibrated.pkl"
    figure_path = _PROJECT_ROOT / "reports" / "figures" / "lgb_raw_calibration_plot.png"

    calibrated_model, brier_uncal, brier_cal = calibrate_model(
        best_model, X_train_cal, y_train_cal, X_calib, y_calib,
        output_model_path=str(model_path),
        output_figure_path=str(figure_path)
    )

    # Evaluate on OOT holdout (carved out before HPO — Basel CRE36.54)
    metrics = evaluate_model(calibrated_model, X_oot, y_oot, model_name="LGB_RAW")

    # Compute true OOF Gini from best-trial OOF predictions (accumulated during HPO).
    # These are uncalibrated LGB scores from held-out CV folds — rank-preserving,
    # so Platt calibration does not affect Gini.
    _best_oof_valid = ~np.isnan(_best_oof_preds[0])
    if _best_oof_valid.sum() > 0:
        oof_gini = 2 * roc_auc_score(y.iloc[oof_indices][_best_oof_valid], _best_oof_preds[0][_best_oof_valid]) - 1
    else:
        oof_gini = float("nan")
    metrics["oof_gini"] = oof_gini

    # Compute OOT Gini on held-out 20%
    oot_gini = 2 * roc_auc_score(y_oot, calibrated_model.predict_proba(X_oot)[:, 1]) - 1
    metrics["oot_gini"] = oot_gini

    # Persist metrics JSON for orchestrator
    # Determine output filename based on store_path and imbalance_strategy
    store_name = Path(feature_store_path).stem  # e.g., "X_tree_dfs"
    store_tag_map = {
        "X_train": "Xtrain",
        "X_tree_raw": "Xtreeraw",
        "X_tree_dfs": "Xtreeds",
    }
    store_tag = store_tag_map.get(store_name, store_name)
    metrics_filename = f"lgb_raw_{store_tag}_{imbalance_strategy}_metrics.json"
    metrics_path = _PROJECT_ROOT / "reports" / metrics_filename

    # Ensure reports/ exists
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # Save metrics dict to JSON
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saved metrics to {metrics_path}")

    # Return tuple per XGBoost pattern
    return calibrated_model, metrics, X_oot, y_oot, best_params


def run_lightgbm_ablation_workflow(
    n_trials: int = _LGB_OPTUNA_N_TRIALS,
) -> dict:
    """
    Execute 3-store × 3-strategy LightGBM ablation and aggregate results.

    Runs train_lightgbm_optuna() 9 times with all combinations of:
    - Feature stores: X_train, X_tree_raw, X_tree_dfs
    - Imbalance strategies: scale_pos_weight, is_unbalance, smote

    Aggregates results, selects best model by OOT Gini, saves comparison table.

    Parameters
    ----------
    n_trials : int
        Number of Optuna trials per ablation cell (default _LGB_OPTUNA_N_TRIALS = 50).

    Returns
    -------
    dict[(store_name, strategy) -> (model, metrics, X_test, y_test, best_params)]
        Results dict with 9 entries, one per ablation cell.
        Keys are tuples like ("X_train", "scale_pos_weight").
    """
    results: dict = {}
    best_model_key = None
    best_oot_gini = -1.0

    stores = [
        ("X_train", _PROJECT_ROOT / "data" / "processed" / "X_train.parquet"),
        ("X_tree_raw", _PROJECT_ROOT / "data" / "processed" / "X_tree_raw.parquet"),
        ("X_tree_dfs", _PROJECT_ROOT / "data" / "processed" / "X_tree_dfs.parquet"),
    ]
    strategies = ["scale_pos_weight", "is_unbalance", "smote"]

    comparison_rows = []

    for store_name, store_path in stores:
        for strategy in strategies:
            key = (store_name, strategy)
            print(f"\n{'='*60}")
            print(f"Running: {store_name} + {strategy}")
            print(f"{'='*60}")

            try:
                model, metrics, X_test, y_test, best_params = train_lightgbm_optuna(
                    feature_store_path=str(store_path),
                    n_trials=n_trials,
                    imbalance_strategy=strategy,
                )

                results[key] = (model, metrics, X_test, y_test, best_params)

                # Track best model by OOT Gini
                oot_gini = metrics.get("oot_gini", metrics.get("Gini", 0.0))
                if oot_gini > best_oot_gini:
                    best_oot_gini = oot_gini
                    best_model_key = key

                # Log metrics
                print(f"Gini: {metrics.get('Gini', 'N/A'):.4f}")
                print(f"OOT Gini: {oot_gini:.4f}")
                print(f"KS: {metrics.get('KS', 'N/A'):.4f}")
                print(f"BrierSkill: {metrics.get('BrierSkill', 'N/A'):.4f}")

                # Append to comparison table
                comparison_rows.append({
                    "Store": store_name,
                    "Strategy": strategy,
                    "Gini": metrics.get("Gini", np.nan),
                    "OOT_Gini": oot_gini,
                    "OOF_Gini": metrics.get("oof_gini", np.nan),
                    "KS": metrics.get("KS", np.nan),
                    "Brier": metrics.get("Brier", np.nan),
                    "BrierSkill": metrics.get("BrierSkill", np.nan),
                    "AUC_ROC": metrics.get("AUC-ROC", np.nan),
                })

            except Exception as e:
                print(f"ERROR: {key} failed — {e}")
                continue

    # Save comparison table
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = _PROJECT_ROOT / "reports" / "lgb_raw_ablation_comparison.csv"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nSaved comparison table to {comparison_path}")

    # Report best model
    if best_model_key is not None:
        best_store, best_strategy = best_model_key
        best_metrics = results[best_model_key][1]
        print(f"\n{'='*60}")
        print(f"BEST MODEL: {best_store} + {best_strategy}")
        print(f"OOT Gini: {best_oot_gini:.4f}")
        print(f"BrierSkill: {best_metrics.get('BrierSkill', 'N/A'):.4f}")
        print(f"{'='*60}")

        # Verify gates
        if best_oot_gini > 0.60:
            print("✓ OOT Gini gate PASSED (> 0.60)")
        else:
            print(f"✗ OOT Gini gate FAILED (< 0.60, actual: {best_oot_gini:.4f})")

        if best_metrics.get("BrierSkill", 0) > 0:
            print("✓ BrierSkill gate PASSED (> 0)")
        else:
            print(f"✗ BrierSkill gate FAILED")
    else:
        print("ERROR: No valid model from ablation")

    return results


# ---------------------------------------------------------------------------
# Probability calibration
# ---------------------------------------------------------------------------

