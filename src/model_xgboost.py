"""
model_xgboost.py
----------------
model_xgboost training and optimization functions.
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
from src.utils import gini_coefficient, ks_statistic, evaluate_model, plot_roc_and_pr
from src.model_base import (
    _make_cv, _find_optimal_threshold_f1_macro, save_model, load_model, calibrate_model,
    _OOFGiniMonitorCallback, _TemporalCV,
    _TEST_SIZE, _RANDOM_STATE, _CV_N_SPLITS, _CV_EMBARGO_FRAC, _TEMPORAL_SORT_COL, _XGB_CV_N_SPLITS,
    _THRESHOLD_MIN, _THRESHOLD_MAX,
    # XGBoost constants
    _XGB_OPTUNA_N_TRIALS, _XGB_N_ESTIMATORS, _XGB_N_ESTIMATORS_MIN, _XGB_N_ESTIMATORS_MAX,
    _XGB_MAX_DEPTH_MIN, _XGB_MAX_DEPTH_MAX, _XGB_LEARNING_RATE_MIN, _XGB_LEARNING_RATE_MAX,
    _XGB_SUBSAMPLE_MIN, _XGB_SUBSAMPLE_MAX, _XGB_COLSAMPLE_BYTREE_MIN, _XGB_COLSAMPLE_BYTREE_MAX,
    _XGB_MIN_CHILD_WEIGHT_MIN, _XGB_MIN_CHILD_WEIGHT_MAX, _XGB_GAMMA_MIN, _XGB_GAMMA_MAX,
    _XGB_MAX_DELTA_STEP_MIN, _XGB_MAX_DELTA_STEP_MAX, _XGB_REG_ALPHA_MIN, _XGB_REG_ALPHA_MAX,
    _XGB_REG_LAMBDA_MIN, _XGB_REG_LAMBDA_MAX,
    _XGB_RAW_N_ESTIMATORS, _XGB_RAW_EARLY_STOPPING_ROUNDS, _XGB_RAW_MIN_CHILD_WEIGHT_MAX,
    _XGB_RAW_GAMMA_MAX, _XGB_RAW_REG_ALPHA_MIN, _XGB_RAW_REG_LAMBDA_MIN, _XGB_RAW_REG_MAX,
    _XGB_RAW_STUDY_NAME, _HPO_PROGRESS_LOG_PATH, _XGB_OPTUNA_MODEL_PATH, _XGB_OPTUNA_PARAMS_PATH,
    _XGB_OPTUNA_FIGURE_PATH, _XGB_EXTENDED_OPTUNA_N_TRIALS,
    _XGB_RAW_MODEL_PATH, _XGB_RAW_PARAMS_PATH, _XGB_RAW_EVAL_PATH, _XGB_RAW_CAL_FIGURE_PATH,
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

def _xgboost_optuna_objective(
    trial: "optuna.Trial",
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float,
    cv: "StratifiedKFold | _TemporalCV",
) -> float:
    """
    Optuna objective: 5-fold CV AUC-ROC for a suggested XGBoost configuration (raw features).

    This function is called once per trial by ``study.optimize()``. It samples
    hyperparameters over an 8-dimensional raw-features search space, runs k-fold CV
    on X_train, and returns the mean out-of-fold AUC-ROC. Early stopping inside
    each fold prunes weak trials.

    Parameters
    ----------
    trial : optuna.Trial
        Current Optuna trial handle for hyperparameter suggestions.
    X_train : pd.DataFrame
        Training features (80% split). Never the held-out test set.
    y_train : pd.Series
        Training labels.
    scale_pos_weight : float
        Cost-sensitive weight = n_negatives / n_positives.
    cv : StratifiedKFold | _TemporalCV
        k-fold CV splitter, seeded for reproducibility.

    Returns
    -------
    float
        Mean out-of-fold AUC-ROC across all CV folds.
    """
    import xgboost as xgb

    # 8-dimensional Bayesian HPO search space for raw features
    # (Improvement 1, 2, 4, 5, 6 applied here)
    params = {
        # Fixed parameters (Improvement 7):
        "n_estimators": _XGB_RAW_N_ESTIMATORS,  # Fixed at 3000, not tuned
        "tree_method": "hist",  # Improvement 6: histogram-based splitting
        # Tuned parameters:
        "max_depth": trial.suggest_int(
            "max_depth", _XGB_MAX_DEPTH_MIN, _XGB_MAX_DEPTH_MAX
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", _XGB_LEARNING_RATE_MIN, _XGB_LEARNING_RATE_MAX, log=True
        ),
        "subsample": trial.suggest_float(
            "subsample", _XGB_SUBSAMPLE_MIN, _XGB_SUBSAMPLE_MAX
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", _XGB_COLSAMPLE_BYTREE_MIN, _XGB_COLSAMPLE_BYTREE_MAX
        ),
        # Improvement 1: min_child_weight as log-scale float [1, 30]
        "min_child_weight": trial.suggest_float(
            "min_child_weight", _XGB_MIN_CHILD_WEIGHT_MIN, _XGB_RAW_MIN_CHILD_WEIGHT_MAX, log=True
        ),
        # Improvement 2: gamma extended to [0, 5]
        "gamma": trial.suggest_float(
            "gamma", _XGB_GAMMA_MIN, _XGB_RAW_GAMMA_MAX
        ),
        # Improvement 4: reg_alpha log-scale [1e-8, 5]
        "reg_alpha": trial.suggest_float(
            "reg_alpha", _XGB_RAW_REG_ALPHA_MIN, _XGB_RAW_REG_MAX, log=True
        ),
        # Improvement 5: reg_lambda log-scale [1e-8, 5]
        "reg_lambda": trial.suggest_float(
            "reg_lambda", _XGB_RAW_REG_LAMBDA_MIN, _XGB_RAW_REG_MAX, log=True
        ),
        # Improvement 3: max_delta_step removed (dropped entirely)
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "auc",
        "use_label_encoder": False,
        "verbosity": 0,
        "random_state": _RANDOM_STATE,
        # XGBoost 3.x: early_stopping_rounds moved to constructor (not fit kwarg)
        "early_stopping_rounds": _XGB_RAW_EARLY_STOPPING_ROUNDS,
    }

    fold_aucs: list[float] = []
    # NaN-init: walk-forward CV never validates the oldest block (~1/(n_splits+1) of samples).
    # Zero-init contaminates OOF Gini — the dead zone gets score=0, dragging AUC toward 0.5.
    # NaN marks unvalidated samples; we compute OOF Gini only on actually-validated rows.
    oof_predictions = np.full(len(X_train), np.nan)
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_fold_train,
            y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            verbose=False,
        )
        best_iteration = getattr(model, "best_iteration", params.get("n_estimators", 3000) - 1)
        trial.set_user_attr("best_iteration", best_iteration)

        y_prob_val = model.predict_proba(X_fold_val)[:, 1]
        fold_aucs.append(float(roc_auc_score(y_fold_val, y_prob_val)))
        oof_predictions[val_idx] = y_prob_val

    # Compute OOF Gini only over validated rows (NaN-filtered).
    # Used as the Optuna objective (maximised) and logged for monitoring (D-17).
    validated_mask = ~np.isnan(oof_predictions)
    try:
        oof_gini = gini_coefficient(y_train.values[validated_mask], oof_predictions[validated_mask])
        trial.set_user_attr("oof_gini", float(oof_gini))
    except Exception:
        oof_gini = float(np.mean(fold_aucs)) - 1.0  # Fallback: penalise degenerate trials
        trial.set_user_attr("oof_gini", None)

    # Store mean fold AUC for reference only — NOT used as objective
    trial.set_user_attr("mean_fold_auc", float(np.mean(fold_aucs)))

    # Optimise directly on OOF Gini so study.best_params → highest-Gini trial
    return float(oof_gini)


def train_xgboost_optuna(
    feature_store_path: str,
    n_trials: int = _XGB_OPTUNA_N_TRIALS,
    groups: pd.Series | None = None,
    progress_log_path: str | None = None,
) -> tuple[object, dict, pd.DataFrame, pd.Series, dict, np.ndarray]:
    """
    Train XGBoost with Bayesian hyperparameter optimisation via Optuna on raw features.

    Loads X_tree_dfs.parquet from disk (raw+DFS features, ~323 columns), extracts
    TARGET column, and runs ``n_trials`` of TPE-based search over an 8-dimensional
    space, selecting the configuration that maximises mean out-of-fold AUC-ROC
    on the training split. The final model is retrained on the full training split
    with the best parameters, then evaluated on the held-out test split (never seen
    during optimisation).

    Imbalance handling uses ``scale_pos_weight = n_neg / n_pos`` (Cost-Sensitive
    strategy). Early stopping inside fold loop prunes weak trials. Platt calibration
    applied to best model before return.

    Parameters
    ----------
    feature_store_path : str
        Path to X_tree_dfs.parquet file (includes TARGET column as last column).
    n_trials : int, optional
        Number of Optuna trials. Default 100 for raw features.
    groups : pd.Series | None, optional
        Optional temporal groups for CV embargo. If None and _TEMPORAL_SORT_COL
        exists in X, groups are auto-detected.
    progress_log_path : str, optional
        Path for the per-trial OOF Gini JSONL log written by ``_OOFGiniMonitorCallback``.
        Override in tests to avoid polluting the production ``reports/hpo_progress.jsonl``.
        Default: ``"reports/hpo_progress.jsonl"``.

    Returns
    -------
    model_calibrated : CalibratedClassifierCV
        Fitted and Platt-calibrated XGBoost model (production-ready for EL = PD × LGD × EAD).
    metrics_dict : dict
        Evaluation metrics on X_test (calibrated model). Keys: Model, AUC-ROC, Gini, KS,
        Brier, BrierSkill, AvgPrecision, oof_gini, oot_gini.
    X_test : pd.DataFrame
        Held-out test features (20% stratified split, seed=42).
    y_test : pd.Series
        Held-out test labels.
    best_params : dict
        Optimised hyperparameters from best trial.
    oof_predictions : np.ndarray
        Uncalibrated out-of-fold predictions (shape (n_train_rows,)), accumulated across CV folds.

    Raises
    ------
    FileNotFoundError
        If feature_store_path does not exist.
    ValueError
        If TARGET column not found in parquet, or if y has no positive/negative samples.

    Notes
    -----
    Artefacts written to disk:
    - ``models/xgboost_raw_best.pkl`` — uncalibrated best model
    - ``models/xgboost_raw_calibrated.pkl`` — Platt-calibrated (production-ready)
    - ``models/xgboost_raw_params.json`` — best hyperparameters as JSON
    - ``reports/xgboost_raw_eval.json`` — evaluation metrics
    - ``reports/figures/xgboost_raw_roc_pr.png`` — ROC + PR curves
    - ``reports/figures/xgboost_raw_calibration.png`` — reliability diagram
    """
    import json as _json

    import matplotlib.pyplot as plt
    import optuna
    import xgboost as xgb

    # Resolve progress_log_path sentinel — allows tests to monkeypatch _HPO_PROGRESS_LOG_PATH
    # without passing the argument explicitly, preventing production JSONL contamination.
    if progress_log_path is None:
        progress_log_path = _HPO_PROGRESS_LOG_PATH

    # --- Load features + target from parquet ---
    X = pd.read_parquet(feature_store_path)
    if "TARGET" not in X.columns:
        raise ValueError(f"TARGET column not found in {feature_store_path}")
    y = X.pop("TARGET")  # Modifies X in-place; removes TARGET from features
    assert "TARGET" not in X.columns, "TARGET column should be removed after pop()"

    # --- Input guards ---
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}.")
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        raise ValueError("y has no positive samples — cannot compute scale_pos_weight.")
    if n_neg == 0:
        raise ValueError("y has no negative samples — cannot compute scale_pos_weight.")

    # --- OOT temporal split (before train/test, hold out most-recent 20% by temporal column) ---
    # Basel CRE36.54 requires temporal validation: hold out the most-recent 20% as separate OOT set
    # before stratified train/test split. The final model is trained on full training set (after OOT holdout),
    # then evaluated on OOT as a separate temporal validation metric.
    if _TEMPORAL_SORT_COL not in X.columns:
        raise ValueError(
            f"Temporal sort column '{_TEMPORAL_SORT_COL}' not in X. "
            "OOT split is required for Basel CRE36 compliance. "
            "Rebuild X_tree_dfs.parquet with X_tree_raw merge (see D-06-new)."
        )

    X_oot = None
    y_oot = None
    temporal_sort_values = X[_TEMPORAL_SORT_COL].values

    # Stratified OOT split: preserve the ~42% first-time / ~58% repeat applicant mix.
    # Prior approach (NaN-fill → nanmin-1) sent ALL first-time applicants to training,
    # making OOT a 100%-repeat-applicant cohort — unrepresentative of production scoring.
    #
    # Strategy:
    #   Known-timing rows (non-NaN): temporal holdout — most-recent 20% → OOT
    #   Unknown-timing rows (NaN):   random 20% → OOT (no temporal order available)
    # This preserves the observed NaN proportion (~41.8%) in both training and OOT.
    nan_mask = np.isnan(temporal_sort_values)
    known_pos = np.where(~nan_mask)[0]
    unknown_pos = np.where(nan_mask)[0]

    # Known-timing: sort ascending and take most-recent 20%
    known_sorted = known_pos[np.argsort(temporal_sort_values[known_pos])]
    oot_known_cut = int(len(known_sorted) * (1 - _TEST_SIZE))
    oot_known = known_sorted[oot_known_cut:]
    train_known = known_sorted[:oot_known_cut]

    # Unknown-timing: random 20% to OOT (seeded for reproducibility)
    rng = np.random.default_rng(_RANDOM_STATE)
    unknown_perm = rng.permutation(len(unknown_pos))
    oot_unknown_cut = int(len(unknown_pos) * (1 - _TEST_SIZE))
    oot_unknown = unknown_pos[unknown_perm[oot_unknown_cut:]]
    train_unknown = unknown_pos[unknown_perm[:oot_unknown_cut]]

    oot_indices = np.concatenate([oot_known, oot_unknown])
    temporal_indices = np.concatenate([train_known, train_unknown])  # remaining rows

    # OOT test set
    X_oot = X.iloc[oot_indices].copy()
    y_oot = y.iloc[oot_indices].copy()

    # Remaining data for stratified train/test split (excludes OOT)
    # temporal_indices already contains only non-OOT rows — no further slicing needed.
    X_remaining = X.iloc[temporal_indices].copy()
    y_remaining = y.iloc[temporal_indices].copy()

    # --- Train / test split (stratified, identical seed to LR baseline) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_remaining, y_remaining, test_size=_TEST_SIZE, stratify=y_remaining, random_state=_RANDOM_STATE
    )

    # Replace -999 sentinel with NaN so XGBoost uses its native split-direction learning.
    # The sentinel fill (from features.py / auto_features.py) treats missing as an extreme
    # continuous value, causing systematic score drift across temporal folds when NaN rates differ.
    # XGBoost handles NaN natively: it learns the optimal default branch per split node.
    X_train = X_train.replace(-999.0, np.nan)
    X_test = X_test.replace(-999.0, np.nan)
    if X_oot is not None:
        X_oot = X_oot.replace(-999.0, np.nan)

    # Cost-sensitive weight: Task 3.3 winner strategy
    scale_pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())

    # --- Optuna study ---
    # Suppress INFO/DEBUG trial logs — keeps library stdout clean.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Auto-detect temporal groups from _TEMPORAL_SORT_COL if not supplied.
    if groups is None and _TEMPORAL_SORT_COL in X.columns:
        groups = X[_TEMPORAL_SORT_COL]
    if groups is not None:
        groups_arr = groups.loc[X_train.index].to_numpy()
        # Fill NaN with (nanmin - 1) so unknown-timing rows sort "oldest" in
        # _TemporalCV, keeping them in early training folds rather than the
        # validation folds (same fix as the OOT split above).
        _g_nan_fill = float(np.nanmin(groups_arr)) - 1.0 if not np.all(np.isnan(groups_arr)) else 0.0
        groups_train = np.where(np.isnan(groups_arr), _g_nan_fill, groups_arr)
    else:
        groups_train = None
    cv = _make_cv(groups_train, n_splits=_XGB_CV_N_SPLITS)

    def objective(trial: optuna.Trial) -> float:
        return _xgboost_optuna_objective(
            trial, X_train, y_train, scale_pos_weight, cv
        )

    study = optuna.create_study(
        study_name=_XGB_RAW_STUDY_NAME,
        storage=f"sqlite:///{_OPTUNA_DB_PATH}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=20),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=75),
        load_if_exists=True,
    )

    # Instantiate monitoring callback for OOF Gini gating (D-17, D-18)
    callback = _OOFGiniMonitorCallback(
        progress_log_path=progress_log_path,
        oof_gini_threshold=0.85,
        consecutive_threshold=3
    )

    study.optimize(objective, n_trials=n_trials, callbacks=[callback])

    best_params: dict = study.best_params

    # --- OOF accumulation loop (D-07, D-08: uncalibrated fold predictions for OOF Gini) ---
    # NaN-init: _TemporalCV dead zone leaves oldest block unvalidated; NaN marks unscored rows
    # so OOF Gini is computed only on rows that were actually validated (validated_mask below).
    oof_predictions = np.full(len(X_train), np.nan)
    best_iteration = max(
        study.best_trial.user_attrs.get(
            "best_iteration", best_params.get("n_estimators", _XGB_RAW_N_ESTIMATORS_MAX) - 1
        ),
        _XGB_N_ESTIMATORS - 1,  # floor at 100 rounds to prevent underfit on easy/small data
    )

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]

        fold_params = {
            **best_params,
            "n_estimators": best_iteration + 1,
            "tree_method": "hist",
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "auc",
            "use_label_encoder": False,
            "verbosity": 0,
            "random_state": _RANDOM_STATE,
        }

        fold_model = xgb.XGBClassifier(**fold_params)
        fold_model.fit(X_fold_train, y_fold_train, verbose=False)

        # Accumulate raw OOF predictions (D-07).
        # Raw probabilities used directly; OOF Gini computed only on validated rows (validated_mask).
        y_prob_fold = fold_model.predict_proba(X_fold_val)[:, 1]
        oof_predictions[val_idx] = y_prob_fold

    # --- Final model: retrain on full X_train with best params ---
    # best_iteration already resolved above; fallback caps at Optuna search ceiling (1000)
    final_n_estimators = best_iteration + 1

    best_params_final = {
        **best_params,
        "n_estimators": final_n_estimators,
        "tree_method": "hist",
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "auc",
        "use_label_encoder": False,
        "verbosity": 0,
        "random_state": _RANDOM_STATE,
    }

    model_best = xgb.XGBClassifier(**best_params_final)
    model_best.fit(X_train, y_train, verbose=False)

    # --- Evaluate best model on held-out test set ---
    metrics_best = evaluate_model(model_best, X_test, y_test, "XGBoost (Raw)")

    # --- Compute OOF Gini (D-08: development set discrimination across all folds) ---
    # validated_mask excludes the _TemporalCV dead zone (oldest block, never in any val set).
    validated_mask = ~np.isnan(oof_predictions)
    oof_gini = gini_coefficient(y_train.to_numpy()[validated_mask], oof_predictions[validated_mask])
    metrics_best["oof_gini"] = oof_gini

    # --- Compute OOT Gini (D-11: temporal holdout validation, Basel CRE36) ---
    if X_oot is not None and len(X_oot) > 0:
        y_prob_oot = model_best.predict_proba(X_oot)[:, 1]
        oot_gini = gini_coefficient(y_oot.to_numpy(), y_prob_oot)
        metrics_best["oot_gini"] = oot_gini

    # --- ROC + PR figure for best model ---
    figure_path = _PROJECT_ROOT / "reports" / "figures" / "xgboost_raw_roc_pr.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_roc_and_pr(model_best, X_test, y_test, "XGBoost (Raw)", save_path=str(figure_path))
    plt.close(fig)

    # --- Platt calibration ---
    # Pass explicit paths to avoid overwriting LightGBM artifacts (default paths).
    model_calibrated, brier_uncal, brier_cal = calibrate_model(
        model_best, X_train, y_train, X_test, y_test, method="sigmoid",
        output_model_path=str(_XGB_RAW_MODEL_PATH),
        output_figure_path=str(_XGB_RAW_CAL_FIGURE_PATH),
    )

    # --- Evaluate calibrated model ---
    metrics_dict = evaluate_model(model_calibrated, X_test, y_test, "XGBoost (Raw, Calibrated)")
    # Preserve OOF and OOT metrics from uncalibrated model evaluation
    metrics_dict["oof_gini"] = oof_gini
    if X_oot is not None and len(X_oot) > 0:
        metrics_dict["oot_gini"] = oot_gini

    # --- Persist best (uncalibrated) model ---
    save_model(model_best, _PROJECT_ROOT / "models" / "xgboost_raw_best.pkl")

    # --- Persist params ---
    params_path = _XGB_RAW_PARAMS_PATH
    params_path.parent.mkdir(parents=True, exist_ok=True)
    with params_path.open("w") as fh:
        _json.dump(best_params, fh, indent=2)

    # --- Persist evaluation metrics ---
    eval_path = _XGB_RAW_EVAL_PATH
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with eval_path.open("w") as fh:
        _json.dump(metrics_dict, fh, indent=2)

    return model_calibrated, metrics_dict, X_test, y_test, best_params, oof_predictions


# ---------------------------------------------------------------------------
# LightGBM with Optuna hyperparame

def train_xgboost_extended_hpo(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = _XGB_EXTENDED_OPTUNA_N_TRIALS,
) -> object:
    """
    Extended HPO for XGBoost with aggressive HyperbandPruner (50 trials).

    Non-regression: ensure final Gini >= 0.5449 (Phase 4 XGBoost baseline).
    HyperbandPruner reduces 50 trials to ~10–15 effective trials via aggressive pruning.
    Optuna study persists in SQLite DB — resumable across runs.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (raw continuous features, 63 columns).
    y : pd.Series
        Binary target series.
    n_trials : int
        Number of Optuna trials (default 50, effective ~10–15 with pruning).

    Returns
    -------
    XGBClassifier
        Fitted XGBoost classifier (retrained on full X, y with best params).

    Raises
    ------
    RuntimeError
        If final best Gini < 0.5449 (Phase 4 XGBoost baseline).
    """
    import optuna
    import xgboost as xgb

    # Auto-detect temporal CV groups from _TEMPORAL_SORT_COL
    # If not present, use standard stratified CV (for unit tests with mock data)
    if _TEMPORAL_SORT_COL in X.columns:
        groups = X[_TEMPORAL_SORT_COL].values
    else:
        groups = None

    # Resume or create Optuna study
    study_name = "xgboost_extended_study"
    storage = f"sqlite:///{_OPTUNA_DB_PATH}"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage, load_if_exists=True)
        if len(study.trials) > 0:
            print(f"Loaded existing XGBoost study with {len(study.trials)} trials")
        else:
            # Study exists but is empty; warm-start with Phase 4 best params
            prior_best = {
                'n_estimators': 400,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 5,
                'gamma': 0.5,
                'reg_alpha': 0.5,
                'reg_lambda': 2.0
            }
            study.enqueue_trial(prior_best)
            print("Created new XGBoost study, warm-started with Phase 4 best params")
    except Exception:
        # Create new study with full initialization
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.HyperbandPruner(min_resource=1, reduction_factor=3),
            load_if_exists=True
        )
        # Warm-start with Phase 4 best params
        prior_best = {
            'n_estimators': 400,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'gamma': 0.5,
            'reg_alpha': 0.5,
            'reg_lambda': 2.0
        }
        if len(study.trials) == 0:
            study.enqueue_trial(prior_best)
        print("Created new XGBoost study, warm-started with Phase 4 best params")

    # Phase 4 baseline for non-regression check (STRICT)
    PHASE4_XGB_BASELINE = 0.5449

    # Define objective function
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', _XGB_N_ESTIMATORS_MIN, _XGB_N_ESTIMATORS_MAX),
            'max_depth': trial.suggest_int('max_depth', _XGB_MAX_DEPTH_MIN, _XGB_MAX_DEPTH_MAX),
            'learning_rate': trial.suggest_float('learning_rate', _XGB_LEARNING_RATE_MIN, _XGB_LEARNING_RATE_MAX, log=True),
            'subsample': trial.suggest_float('subsample', _XGB_SUBSAMPLE_MIN, _XGB_SUBSAMPLE_MAX),
            'colsample_bytree': trial.suggest_float('colsample_bytree', _XGB_COLSAMPLE_BYTREE_MIN, _XGB_COLSAMPLE_BYTREE_MAX),
            'min_child_weight': trial.suggest_int('min_child_weight', _XGB_MIN_CHILD_WEIGHT_MIN, _XGB_MIN_CHILD_WEIGHT_MAX),
            'gamma': trial.suggest_float('gamma', _XGB_GAMMA_MIN, _XGB_GAMMA_MAX),
            'reg_alpha': trial.suggest_float('reg_alpha', _XGB_REG_ALPHA_MIN, _XGB_REG_ALPHA_MAX),
            'reg_lambda': trial.suggest_float('reg_lambda', _XGB_REG_LAMBDA_MIN, _XGB_REG_LAMBDA_MAX),
            'verbosity': 0,
        }

        # Cross-validated AUC with temporal CV
        cv = _make_cv(groups_train=groups, n_splits=_CV_N_SPLITS)
        auc_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

            # Compute scale_pos_weight on training fold
            scale_pos = (y_tr == 0).sum() / (y_tr == 1).sum()

            # Train XGBoost with cost-sensitive learning
            xgb_model = xgb.XGBClassifier(**params, scale_pos_weight=scale_pos, random_state=_RANDOM_STATE)
            xgb_model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False
            )

            # Compute AUC on validation fold
            auc = roc_auc_score(y_va, xgb_model.predict_proba(X_va)[:, 1])
            auc_scores.append(auc)

            # Report intermediate value for pruning
            trial.report(np.mean(auc_scores), fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(auc_scores)

    # Optimize
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_auc = study.best_value
    best_gini = 2 * best_auc - 1
    print(f"XGBoost HPO complete. Best AUC: {best_auc:.4f}, Best Gini: {best_gini:.4f} ({len(study.trials)} trials, ~{int(len(study.trials)/5)} effective due to pruning)")

    # Non-regression check (STRICT)
    if best_gini < PHASE4_XGB_BASELINE:
        raise RuntimeError(
            f"Non-regression violated: XGBoost Gini {best_gini:.4f} < Phase 4 baseline {PHASE4_XGB_BASELINE:.4f}"
        )

    # Refit on full data with best params
    best_params = study.best_params.copy()
    best_params['verbosity'] = 0
    best_params['scale_pos_weight'] = (y == 0).sum() / (y == 1).sum()
    best_params['random_state'] = _RANDOM_STATE

    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X, y, verbose=False)

    # Save model
    joblib.dump(final_model, _PROJECT_ROOT / "models" / "xgboost_extended.pkl")
    print(f"Saved XGBoost model to models/xgboost_extended.pkl")

    return final_model


def filter_dfs_by_iv(
    X_dfs: pd.DataFrame,
    y: pd.Series,
    iv_threshold: float = 0.02,
) -> pd.DataFrame:
    """
    Filter DFS features by Information Value threshold.

    Computes IV for all numeric features in X_dfs using the existing
    select_features_by_iv utility, then returns only those features with
    IV >= iv_threshold.

    Parameters
    ----------
    X_dfs : pd.DataFrame
        Feature matrix from DFS (Featuretools auto-generated features).
    y : pd.Series
        Binary target series, aligned with X_dfs.
    iv_threshold : float, optional
        Minimum IV threshold for feature retention (default 0.02, i.e., weak).

    Returns
    -------
    pd.DataFrame
        Filtered feature matrix with only high-IV features (IV >= iv_threshold).
        Original row order and index preserved. Low-IV features removed.

    Notes
    -----
    **IV computation:** Uses existing select_features_by_iv() which internally
    calls compute_woe_iv() on each numeric feature with quantile binning (10 bins).

    **Feature count reduction:** Typical DFS output (80–150 features) is filtered
    to ~20–40 features at IV >= 0.02 threshold.

    **Security:** Any 'SK_ID' columns should be dropped by the caller before
    IV filtering (DFS entityset should drop_contains=['SK_ID']).
    """
    from src.features import select_features_by_iv

    # Compute IV for all numeric features
    iv_dict = select_features_by_iv(X_dfs, y, min_iv=iv_threshold, bins=10)

    # Extract feature names that passed the IV threshold
    keep_features = list(iv_dict.keys())

    print(f"DFS IV filter: {X_dfs.shape[1]} features -> {len(keep_features)} features (IV >= {iv_threshold})")

    # Return filtered DataFrame with only high-IV features
    return X_dfs[keep_features].copy()


# ---------------------------------------------------------------------------
# Ensemble Stacking (Wave 3: Ensemble Variants A & B)
# ---------------------------------------------------------------------------

_ENSEMBLE_PERSIST_THRESHOLD: float = 0.005


def train_ensemble_variant_a(
    X_lgb: pd.DataFrame,
    X_xgb: pd.DataFrame,
    X_cat: pd.DataFrame,
    X_lr: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = _RANDOM_STATE,
) -> tuple[object, dict]:
    """
    4-model ensemble: LGB + XGB + CatBoost + Logistic Regression with logistic meta-learner.

    Uses OOF stacking on identical temporal CV folds to avoid meta-learner overfitting.

    Parameters
    ----------
    X_lgb, X_xgb, X_cat, X_lr : pd.DataFrame
        Per-model feature matrices from Wave 2 preparation.
    y : pd.Series
        Binary target (0/1).
    test_size : float
        Test split fraction (default 0.2).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    (meta_model, metrics_dict) where metrics_dict contains:
        'lgb_gini', 'xgb_gini', 'cat_gini', 'lr_gini', 'best_base_gini', 'ensemble_gini',
        'improvement', 'persisted'
    """
    import lightgbm as lgb_lib
    import xgboost as xgb_lib

    # Split data (identical split for all models)
    X_lgb_tr, X_lgb_te, y_tr, y_te = train_test_split(
        X_lgb, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_xgb_tr, X_xgb_te, _, _ = train_test_split(
        X_xgb, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_cat_tr, X_cat_te, _, _ = train_test_split(
        X_cat, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_lr_tr, X_lr_te, _, _ = train_test_split(
        X_lr, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Detect categorical columns for CatBoost
    cat_cols = [c for c in X_cat_tr.columns if X_cat_tr[c].dtype == "category"]

    # Get temporal CV folds (identical across all models)
    groups = X_lgb_tr[_TEMPORAL_SORT_COL].values if _TEMPORAL_SORT_COL in X_lgb_tr.columns else None
    cv = _make_cv(groups_train=groups, n_splits=5)

    # Initialize OOF arrays
    oof_lgb = np.zeros(len(X_lgb_tr))
    oof_xgb = np.zeros(len(X_xgb_tr))
    oof_cat = np.zeros(len(X_cat_tr))
    oof_lr = np.zeros(len(X_lr_tr))

    # Generate OOF predictions
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_lgb_tr, y_tr, groups=groups)):
        print(f"Ensemble Variant A OOF Fold {fold_idx + 1}/5...")

        # LGB
        X_lgb_f_tr, X_lgb_f_va = X_lgb_tr.iloc[train_idx], X_lgb_tr.iloc[val_idx]
        y_f_tr, y_f_va = y_tr.iloc[train_idx], y_tr.iloc[val_idx]

        lgb_model = lgb_lib.LGBMClassifier(
            n_estimators=500, is_unbalance=True, verbose=-1, random_state=random_state
        )
        lgb_model.fit(X_lgb_f_tr, y_f_tr)
        oof_lgb[val_idx] = lgb_model.predict_proba(X_lgb_f_va)[:, 1]

        # XGB
        X_xgb_f_tr, X_xgb_f_va = X_xgb_tr.iloc[train_idx], X_xgb_tr.iloc[val_idx]
        xgb_model = xgb_lib.XGBClassifier(
            n_estimators=500, random_state=random_state, verbosity=0, early_stopping_rounds=20
        )
        xgb_model.fit(
            X_xgb_f_tr,
            y_f_tr,
            eval_set=[(X_xgb_f_va, y_f_va)],
            verbose=False,
        )
        oof_xgb[val_idx] = xgb_model.predict_proba(X_xgb_f_va)[:, 1]

        # CatBoost
        X_cat_f_tr, X_cat_f_va = X_cat_tr.iloc[train_idx], X_cat_tr.iloc[val_idx]
        cat_model = CatBoostClassifier(iterations=500, verbose=0, allow_writing_files=False)
        cat_model.fit(
            X_cat_f_tr,
            y_f_tr,
            cat_features=cat_cols,
            eval_set=[(X_cat_f_va, y_f_va)],
            early_stopping_rounds=20,
        )
        oof_cat[val_idx] = cat_model.predict_proba(X_cat_f_va)[:, 1]

        # LR (WoE-encoded features)
        X_lr_f_tr, X_lr_f_va = X_lr_tr.iloc[train_idx], X_lr_tr.iloc[val_idx]
        lr_model = LogisticRegression(
            C=0.1, solver="lbfgs", max_iter=1000, random_state=random_state
        )
        lr_model.fit(X_lr_f_tr, y_f_tr)
        oof_lr[val_idx] = lr_model.predict_proba(X_lr_f_va)[:, 1]

    # Evaluate base models on test set
    lgb_test = lgb_lib.LGBMClassifier(
        n_estimators=500, is_unbalance=True, verbose=-1, random_state=random_state
    )
    lgb_test.fit(X_lgb_tr, y_tr)
    lgb_gini = gini_coefficient(y_te, lgb_test.predict_proba(X_lgb_te)[:, 1])

    xgb_test = xgb_lib.XGBClassifier(n_estimators=500, random_state=random_state, verbosity=0, early_stopping_rounds=20)
    xgb_test.fit(X_xgb_tr, y_tr, eval_set=[(X_xgb_te, y_te)], verbose=False)
    xgb_gini = gini_coefficient(y_te, xgb_test.predict_proba(X_xgb_te)[:, 1])

    cat_test = CatBoostClassifier(iterations=500, verbose=0, allow_writing_files=False)
    cat_test.fit(X_cat_tr, y_tr, cat_features=cat_cols)
    cat_gini = gini_coefficient(y_te, cat_test.predict_proba(X_cat_te)[:, 1])

    lr_test = LogisticRegression(
        C=0.1, solver="lbfgs", max_iter=1000, random_state=random_state
    )
    lr_test.fit(X_lr_tr, y_tr)
    lr_gini = gini_coefficient(y_te, lr_test.predict_proba(X_lr_te)[:, 1])

    # Train meta-learner on OOF predictions
    meta_input = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_lr])
    meta_model = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=1000, random_state=random_state
    )
    meta_model.fit(meta_input, y_tr)

    # Evaluate ensemble on test set
    test_lgb_pred = lgb_test.predict_proba(X_lgb_te)[:, 1]
    test_xgb_pred = xgb_test.predict_proba(X_xgb_te)[:, 1]
    test_cat_pred = cat_test.predict_proba(X_cat_te)[:, 1]
    test_lr_pred = lr_test.predict_proba(X_lr_te)[:, 1]

    test_meta_input = np.column_stack([test_lgb_pred, test_xgb_pred, test_cat_pred, test_lr_pred])
    ensemble_pred = meta_model.predict_proba(test_meta_input)[:, 1]
    ensemble_gini = gini_coefficient(y_te, ensemble_pred)

    # Calculate improvement
    best_base_gini = max(lgb_gini, xgb_gini, cat_gini, lr_gini)
    improvement = ensemble_gini - best_base_gini
    persisted = improvement >= _ENSEMBLE_PERSIST_THRESHOLD

    # Save ensemble if improvement sufficient
    if persisted:
        joblib.dump(meta_model, _PROJECT_ROOT / "models" / "ensemble_variant_a.pkl")
        print(f"Ensemble Variant A persisted (improvement: {improvement:+.4f})")
    else:
        print(
            f"Ensemble Variant A NOT persisted (improvement: {improvement:+.4f} < {_ENSEMBLE_PERSIST_THRESHOLD})"
        )

    metrics = {
        "lgb_gini": float(lgb_gini),
        "xgb_gini": float(xgb_gini),
        "cat_gini": float(cat_gini),
        "lr_gini": float(lr_gini),
        "best_base_gini": float(best_base_gini),
        "ensemble_gini": float(ensemble_gini),
        "improvement": float(improvement),
        "persisted": persisted,
    }

    return meta_model, metrics


def train_ensemble_variant_b(
    X_lgb: pd.DataFrame,
    X_xgb: pd.DataFrame,
    X_cat: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = _RANDOM_STATE,
) -> tuple[object, dict]:
    """
    3-model ensemble: LGB + XGB + CatBoost with Ridge meta-learner.

    Ridge meta-learner preferred for tree-only stacking (simpler, less overfitting risk).

    Parameters
    ----------
    X_lgb, X_xgb, X_cat : pd.DataFrame
        Per-model feature matrices from Wave 2 preparation.
    y : pd.Series
        Binary target (0/1).
    test_size : float
        Test split fraction (default 0.2).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    (meta_model, metrics_dict) with keys: lgb_gini, xgb_gini, cat_gini, best_base_gini,
                                         ensemble_gini, improvement, persisted, meta_alpha
    """
    from sklearn.linear_model import RidgeCV

    import lightgbm as lgb_lib
    import xgboost as xgb_lib

    # Split data (identical to Variant A for fairness)
    X_lgb_tr, X_lgb_te, y_tr, y_te = train_test_split(
        X_lgb, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_xgb_tr, X_xgb_te, _, _ = train_test_split(
        X_xgb, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_cat_tr, X_cat_te, _, _ = train_test_split(
        X_cat, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Detect categorical columns for CatBoost
    cat_cols = [c for c in X_cat_tr.columns if X_cat_tr[c].dtype == "category"]

    # OOF generation (identical temporal CV)
    groups = X_lgb_tr[_TEMPORAL_SORT_COL].values if _TEMPORAL_SORT_COL in X_lgb_tr.columns else None
    cv = _make_cv(groups_train=groups, n_splits=5)

    oof_lgb = np.zeros(len(X_lgb_tr))
    oof_xgb = np.zeros(len(X_xgb_tr))
    oof_cat = np.zeros(len(X_cat_tr))

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_lgb_tr, y_tr, groups=groups)):
        print(f"Ensemble Variant B OOF Fold {fold_idx + 1}/5...")

        X_lgb_f_tr, X_lgb_f_va = X_lgb_tr.iloc[train_idx], X_lgb_tr.iloc[val_idx]
        X_xgb_f_tr, X_xgb_f_va = X_xgb_tr.iloc[train_idx], X_xgb_tr.iloc[val_idx]
        X_cat_f_tr, X_cat_f_va = X_cat_tr.iloc[train_idx], X_cat_tr.iloc[val_idx]
        y_f_tr = y_tr.iloc[train_idx]

        # LGB
        lgb_model = lgb_lib.LGBMClassifier(
            n_estimators=500, is_unbalance=True, verbose=-1, random_state=random_state
        )
        lgb_model.fit(X_lgb_f_tr, y_f_tr)
        oof_lgb[val_idx] = lgb_model.predict_proba(X_lgb_f_va)[:, 1]

        # XGB
        xgb_model = xgb_lib.XGBClassifier(
            n_estimators=500, random_state=random_state, verbosity=0, early_stopping_rounds=20
        )
        xgb_model.fit(
            X_xgb_f_tr,
            y_f_tr,
            eval_set=[(X_xgb_f_va, y_tr.iloc[val_idx])],
            verbose=False,
        )
        oof_xgb[val_idx] = xgb_model.predict_proba(X_xgb_f_va)[:, 1]

        # CatBoost
        cat_model = CatBoostClassifier(iterations=500, verbose=0, allow_writing_files=False)
        cat_model.fit(
            X_cat_f_tr,
            y_f_tr,
            cat_features=cat_cols,
            eval_set=[(X_cat_f_va, y_tr.iloc[val_idx])],
            early_stopping_rounds=20,
        )
        oof_cat[val_idx] = cat_model.predict_proba(X_cat_f_va)[:, 1]

    # Evaluate base models on test set
    lgb_test = lgb_lib.LGBMClassifier(
        n_estimators=500, is_unbalance=True, verbose=-1, random_state=random_state
    )
    lgb_test.fit(X_lgb_tr, y_tr)
    lgb_gini = gini_coefficient(y_te, lgb_test.predict_proba(X_lgb_te)[:, 1])

    xgb_test = xgb_lib.XGBClassifier(n_estimators=500, random_state=random_state, verbosity=0, early_stopping_rounds=20)
    xgb_test.fit(X_xgb_tr, y_tr, eval_set=[(X_xgb_te, y_te)], verbose=False)
    xgb_gini = gini_coefficient(y_te, xgb_test.predict_proba(X_xgb_te)[:, 1])

    cat_test = CatBoostClassifier(iterations=500, verbose=0, allow_writing_files=False)
    cat_test.fit(X_cat_tr, y_tr, cat_features=cat_cols)
    cat_gini = gini_coefficient(y_te, cat_test.predict_proba(X_cat_te)[:, 1])

    # Train meta-learner (RidgeCV with alpha search)
    meta_input = np.column_stack([oof_lgb, oof_xgb, oof_cat])
    meta_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], cv=5)
    meta_model.fit(meta_input, y_tr)

    # Evaluate ensemble on test set
    test_lgb_pred = lgb_test.predict_proba(X_lgb_te)[:, 1]
    test_xgb_pred = xgb_test.predict_proba(X_xgb_te)[:, 1]
    test_cat_pred = cat_test.predict_proba(X_cat_te)[:, 1]

    test_meta_input = np.column_stack([test_lgb_pred, test_xgb_pred, test_cat_pred])
    ensemble_pred = meta_model.predict(test_meta_input)
    ensemble_pred = np.clip(ensemble_pred, 0, 1)  # Ensure [0, 1] range
    ensemble_gini = gini_coefficient(y_te, ensemble_pred)

    # Improvement and persistence
    best_base_gini = max(lgb_gini, xgb_gini, cat_gini)
    improvement = ensemble_gini - best_base_gini
    persisted = improvement >= _ENSEMBLE_PERSIST_THRESHOLD

    if persisted:
        joblib.dump(meta_model, _PROJECT_ROOT / "models" / "ensemble_variant_b.pkl")
        print(f"Ensemble Variant B persisted (improvement: {improvement:+.4f})")
    else:
        print(
            f"Ensemble Variant B NOT persisted (improvement: {improvement:+.4f} < {_ENSEMBLE_PERSIST_THRESHOLD})"
        )

    metrics = {
        "lgb_gini": float(lgb_gini),
        "xgb_gini": float(xgb_gini),
        "cat_gini": float(cat_gini),
        "best_base_gini": float(best_base_gini),
        "ensemble_gini": float(ensemble_gini),
        "improvement": float(improvement),
        "persisted": persisted,
        "meta_alpha": float(meta_model.alpha_) if hasattr(meta_model, "alpha_") else None,
    }

    return meta_model, metrics


# ---------------------------------------------------------------------------
# Stubs (Phase 3.3+ — LightGBM and XGBoost, implemented in later tasks)
# ---------------------------------------------------------------------------

def train(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> object:
    """Train a LightGBM classifier with stratified k-fold CV."""
    raise NotImplementedError
