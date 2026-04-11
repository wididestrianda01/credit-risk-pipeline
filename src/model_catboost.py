"""
model_catboost.py
-----------------
model_catboost training and optimization functions.
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
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split

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
    _LGB_REG_ALPHA_MIN, _LGB_REG_ALPHA_MAX, _LGB_REG_LAMBDA_MIN, _LGB_REG_LAMBDA_MAX,
    _LGB_OBJ_EARLY_STOPPING_ROUNDS, _LGB_EARLY_STOPPING_ROUNDS, _LGB_FINAL_VAL_SIZE,
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
    _ENSEMBLE_PERSIST_THRESHOLD, _OPTUNA_DB_PATH, _BENCHMARK_REPORT_PATH,
)

def prepare_catboost_features(
    X_woe: pd.DataFrame,
    df_raw: pd.DataFrame | None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Prepare feature matrix for CatBoost by optionally substituting raw
    categorical columns for their WoE-encoded counterparts.

    CatBoost can exploit raw categorical strings natively (ordered target
    encoding internally), but the WoE feature matrix encodes them as floats.
    When ``df_raw`` is supplied, each column in ``_CATBOOST_RAW_CATS`` that
    exists in both ``X_woe`` and ``df_raw`` is replaced with the raw string
    series cast to ``category`` dtype.

    Parameters
    ----------
    X_woe : pd.DataFrame
        WoE-encoded feature matrix (all-numeric, shape n × p).
    df_raw : pd.DataFrame or None
        Raw application DataFrame from data_loader.  If None, X_woe is
        returned unchanged with an empty cat_cols list.

    Returns
    -------
    X_out : pd.DataFrame
        Feature matrix with raw categoricals substituted (or X_woe unchanged).
    cat_cols : list[str]
        Column names that are categorical in X_out (for CatBoost's
        ``cat_features`` argument).
    """
    if df_raw is None:
        return X_woe, []

    X_out = X_woe.copy()
    cat_cols: list[str] = []
    for col in _CATBOOST_RAW_CATS:
        if col in X_out.columns and col in df_raw.columns:
            X_out[col] = df_raw[col].values.astype(str)
            X_out[col] = X_out[col].astype("category")
            cat_cols.append(col)

    return X_out, cat_cols


def train_catboost_optuna(
    feature_store_path: str,
    n_trials: int = _CAT_OPTUNA_N_TRIALS,
    groups: pd.Series | None = None,
) -> tuple[CatBoostClassifier, dict, pd.DataFrame, pd.Series, dict]:
    """
    Train CatBoost with Optuna HPO on raw feature store.

    Loads feature store from disk, runs Bayesian HPO via Optuna (HyperbandPruner + TPESampler),
    applies Platt calibration, and saves artifacts. Supports 2-stage refit with early stopping.

    Search space (per Phase 04.2.5 D-11):
    - ``depth``               : int   [5, 10]     symmetric-tree depth (expanded for raw features)
    - ``learning_rate``       : float [0.01, 0.2] log-uniform (floor expanded)
    - ``l2_leaf_reg``         : float [0.1, 30]   log-scale expansion for raw features
    - ``min_data_in_leaf``    : int   [5, 50]     NEW parameter to prevent singleton leaves
    - ``bagging_temperature`` : float [0, 1]      Bayesian bootstrap temperature
    - ``random_strength``     : float [0, 1]      feature-split randomisation

    Class imbalance handled via ``scale_pos_weight = n_neg / n_pos`` only — single strategy
    per phase scope (D-03: no SMOTE, no is_unbalance, CatBoost's ordered boosting).

    **Basel CRE36.54 compliance:** An out-of-time (OOT) holdout of the most-recent
    ``_TEST_SIZE`` (20%) of rows — sorted by ``_TEMPORAL_SORT_COL`` — is carved out
    *before* HPO begins and is never seen during Optuna trials or CV folds. Final OOT
    Gini is reported after full refit on the remaining 80%, satisfying the regulatory
    requirement for temporal model validation in internal ratings-based (IRB) models.

    Parameters
    ----------
    feature_store_path : str
        Path to parquet file containing X_tree_raw.parquet (129+ features + TARGET column).
    n_trials : int
        Number of Optuna trials (default _CAT_OPTUNA_N_TRIALS = 50).
    groups : pd.Series, optional
        Temporal CV groups (e.g., application year); if None, temporal CV auto-detected from _TEMPORAL_SORT_COL.

    Returns
    -------
    tuple[CatBoostClassifier, dict, pd.DataFrame, pd.Series, dict]
        (calibrated_model, metrics_dict, X_test, y_test, best_params)
        - calibrated_model: CatBoost with Platt calibration applied
        - metrics_dict: {Gini, AUC-ROC, KS, Brier, BrierSkill, oof_gini, oot_gini}
        - X_test: OOT held-out test set features (most-recent 20% by temporal order)
        - y_test: OOT held-out test set labels
        - best_params: best hyperparameters from Optuna study
    """
    import json as _json

    import optuna

    # Load feature store
    feature_path = Path(feature_store_path)
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature store not found: {feature_path}")

    df = pd.read_parquet(feature_path)

    # Extract TARGET
    if "TARGET" not in df.columns:
        raise ValueError(f"TARGET column not found in {feature_path}")

    y = df.pop("TARGET").astype(int)
    X = df

    # Input guards
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}.")

    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        raise ValueError("y has no positive samples.")
    if n_neg == 0:
        raise ValueError("y has no negative samples.")

    # --- OOT temporal split — Basel CRE36.54 ---
    # Hold out most-recent 20% BEFORE HPO; never seen during training or calibration.
    if _TEMPORAL_SORT_COL not in X.columns:
        raise ValueError(
            f"Temporal sort column '{_TEMPORAL_SORT_COL}' not in X. "
            "OOT split is required for Basel CRE36 compliance. "
            "Rebuild feature store with build_tree_feature_store()."
        )
    _cat_temporal_vals = X[_TEMPORAL_SORT_COL].values
    _cat_nan_mask = np.isnan(_cat_temporal_vals)
    _cat_known_pos = np.where(~_cat_nan_mask)[0]
    _cat_unknown_pos = np.where(_cat_nan_mask)[0]
    _cat_known_sorted = _cat_known_pos[np.argsort(_cat_temporal_vals[_cat_known_pos])]
    _cat_oot_known_cut = int(len(_cat_known_sorted) * (1 - _TEST_SIZE))
    _cat_oot_known = _cat_known_sorted[_cat_oot_known_cut:]
    _cat_train_known = _cat_known_sorted[:_cat_oot_known_cut]
    _cat_rng = np.random.default_rng(_RANDOM_STATE)
    _cat_unknown_perm = _cat_rng.permutation(len(_cat_unknown_pos))
    _cat_oot_unknown_cut = int(len(_cat_unknown_pos) * (1 - _TEST_SIZE))
    _cat_oot_unknown = _cat_unknown_pos[_cat_unknown_perm[_cat_oot_unknown_cut:]]
    _cat_train_unknown = _cat_unknown_pos[_cat_unknown_perm[:_cat_oot_unknown_cut]]
    _cat_oot_indices = np.concatenate([_cat_oot_known, _cat_oot_unknown])
    _cat_train_indices = np.concatenate([_cat_train_known, _cat_train_unknown])
    X_oot = X.iloc[_cat_oot_indices].copy()
    y_oot = y.iloc[_cat_oot_indices].copy()
    X_train = X.iloc[_cat_train_indices].copy()
    y_train = y.iloc[_cat_train_indices].copy()

    # Compute scale_pos_weight (D-03)
    scale_pos_weight = n_neg / max(n_pos, 1)

    # Auto-detect temporal CV (D-17): check if _TEMPORAL_SORT_COL is in X_train
    groups_array = groups.values if groups is not None else None
    cv = _make_cv(groups_train=groups_array, n_splits=_CV_N_SPLITS)

    # Optuna study (D-16–D-18)
    sampler = optuna.samplers.TPESampler(seed=_RANDOM_STATE, n_startup_trials=20)
    pruner = optuna.pruners.HyperbandPruner(min_resource=10, max_resource=50, reduction_factor=3)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="catboost_raw_scalepos",
        storage=f"sqlite:///{_PROJECT_ROOT / 'models' / 'optuna_studies.db'}",
        load_if_exists=True,
    )
    # Capture count of pre-existing (possibly contaminated) trials before this run
    _n_trials_before = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )

    # HPO objective — 7 hyperparameters (D-11)
    def _objective(trial: optuna.Trial) -> float:
        params = {
            "depth": trial.suggest_int("depth", _CAT_DEPTH_MIN, _CAT_DEPTH_MAX),
            "learning_rate": trial.suggest_float(
                "learning_rate", _CAT_LEARNING_RATE_MIN, _CAT_LEARNING_RATE_MAX, log=True
            ),
            "l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg", _CAT_L2_LEAF_REG_MIN, _CAT_L2_LEAF_REG_MAX, log=True
            ),
            "min_data_in_leaf": trial.suggest_int(
                "min_data_in_leaf", _CAT_RAW_MIN_DATA_IN_LEAF_MIN, _CAT_RAW_MIN_DATA_IN_LEAF_MAX
            ),
            "bagging_temperature": trial.suggest_float(
                "bagging_temperature", _CAT_BAGGING_TEMP_MIN, _CAT_BAGGING_TEMP_MAX
            ),
            "random_strength": trial.suggest_float(
                "random_strength", _CAT_RANDOM_STRENGTH_MIN, _CAT_RANDOM_STRENGTH_MAX
            ),
            "bootstrap_type": _CAT_BOOTSTRAP_TYPE,
            "scale_pos_weight": scale_pos_weight,
            "random_seed": _RANDOM_STATE,
            "verbose": 0,
            "allow_writing_files": False,
        }

        # CV loop over folds
        fold_aucs: list[float] = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model_fold = CatBoostClassifier(
                **params,
                iterations=_CAT_ITERATIONS,
                early_stopping_rounds=_CAT_OBJ_EARLY_STOPPING_ROUNDS,
            )
            model_fold.fit(
                X_tr.to_numpy(), y_tr.to_numpy(),
                eval_set=(X_val.to_numpy(), y_val.to_numpy()),
                verbose=False,
            )
            y_pred = model_fold.predict_proba(X_val.to_numpy())[:, 1]
            fold_aucs.append(roc_auc_score(y_val, y_pred))

        return float(np.mean(fold_aucs))

    # Run study
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)

    # Select best params from compliant (current-run) trials only.
    # Pre-existing contaminated trials (e.g. run on mock data) have number < _n_trials_before
    # and must not influence the final model.
    _compliant_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.number >= _n_trials_before
    ]
    if not _compliant_trials:
        raise RuntimeError(
            "No compliant trials completed — cannot select best_params safely."
        )
    _best_trial = max(_compliant_trials, key=lambda t: t.value)
    best_params = _best_trial.params

    # 2-stage refit (D-09, D-10)
    # Stage 1: 80/20 holdout on X_train with early stopping
    X_tr, X_val_es, y_tr, y_val_es = train_test_split(
        X_train, y_train,
        test_size=_CAT_FINAL_VAL_SIZE,
        stratify=y_train,
        random_state=_RANDOM_STATE,
    )
    stage1_params = {
        **best_params,
        "bootstrap_type": _CAT_BOOTSTRAP_TYPE,
        "scale_pos_weight": scale_pos_weight,
        "random_seed": _RANDOM_STATE,
        "verbose": 0,
        "allow_writing_files": False,
        "iterations": _CAT_ITERATIONS,
        "early_stopping_rounds": _CAT_EARLY_STOPPING_ROUNDS,
    }
    stage1_model = CatBoostClassifier(**stage1_params)
    stage1_model.fit(
        X_tr.to_numpy(), y_tr.to_numpy(),
        eval_set=(X_val_es.to_numpy(), y_val_es.to_numpy()),
        verbose=False,
    )
    best_iterations = stage1_model.best_iteration_ or _CAT_ITERATIONS

    # Stage 2: full X_train with best_iterations, no early stopping
    stage2_params = {
        **best_params,
        "bootstrap_type": _CAT_BOOTSTRAP_TYPE,
        "scale_pos_weight": scale_pos_weight,
        "random_seed": _RANDOM_STATE,
        "verbose": 0,
        "allow_writing_files": False,
        "iterations": best_iterations,
    }
    final_model = CatBoostClassifier(**stage2_params)
    final_model.fit(X_train.to_numpy(), y_train.to_numpy(), verbose=False)

    # Calibrate on 70/30 split within X_train (not OOT — calibrator must not see holdout)
    X_train_cal, X_calib, y_train_cal, y_calib = train_test_split(
        X_train, y_train, test_size=0.3, random_state=_RANDOM_STATE, stratify=y_train
    )
    calibrated_model, _, _ = calibrate_model(
        final_model,
        X_train_cal, y_train_cal,
        X_calib, y_calib,
        method="sigmoid",
        output_model_path=str(_CAT_MODEL_PATH),
        output_figure_path=str(_CAT_FIGURE_PATH),
    )

    # Evaluate on OOT holdout using the calibrated model (Brier, Gini, KS all on calibrated probs)
    metrics = evaluate_model(calibrated_model, X_oot, y_oot, "CatBoost (Raw, scale_pos_weight)")
    metrics["n_trials"] = n_trials
    metrics["n_features"] = X.shape[1]
    oot_gini = 2 * roc_auc_score(y_oot, calibrated_model.predict_proba(X_oot)[:, 1]) - 1
    metrics["oot_gini"] = oot_gini

    # Save params JSON
    params_path = Path(_CAT_PARAMS_PATH)
    params_path.parent.mkdir(parents=True, exist_ok=True)
    with params_path.open("w") as fh:
        _json.dump(best_params, fh, indent=2)

    # Save metrics JSON (D-21)
    metrics_out_path = _PROJECT_ROOT / "reports" / "catboost_raw_eval.json"
    metrics_out_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_out_path.open("w") as fh:
        _json.dump({"Model": "CatBoost (Raw, scale_pos_weight)", **metrics}, fh, indent=2)

    return calibrated_model, metrics, X_oot, y_oot, best_params


# ---------------------------------------------------------------------------
# EXT_SOURCE_3 Supervised Imputation (Phase 2.2)
# ---------------------------------------------------------------------------

