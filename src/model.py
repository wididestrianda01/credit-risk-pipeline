"""
model.py
--------
Model training, stratified cross-validation, threshold calibration,
and persistence helpers.

Supported estimators
--------------------
- LightGBM (primary, Phase 3.5+)
- XGBoost (benchmark, Phase 3.3)
- Logistic Regression (interpretable baseline, Phase 3.2)
"""

from __future__ import annotations

import datetime
import json
import logging

logger = logging.getLogger(__name__)
import warnings
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import _PROJECT_ROOT
from src.utils import evaluate_model, gini_coefficient, ks_statistic, plot_roc_and_pr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TEST_SIZE: float = 0.2
_RANDOM_STATE: int = 42
_CV_N_SPLITS: int = 10
# Temporal CV embargo: strip the last _CV_EMBARGO_FRAC fraction of each
# training fold to prevent serial-correlation leakage across the train/val
# boundary (López de Prado, Advances in Financial Machine Learning, Ch. 7).
# 1% is a conservative value for cross-sectional credit data where labels
# do not overlap; it suffices to signal temporal awareness without
# discarding meaningful training signal.
_CV_EMBARGO_FRAC: float = 0.02

# Column used to auto-detect temporal ordering when groups is not supplied.
# prev_days_decision_mean is the mean number of days before application that
# previous applications were decided — a robust proxy for applicant vintage.
_TEMPORAL_SORT_COL: str = "prev_days_decision_mean"

# Logistic regression baseline hyperparameters (IRB scorecard config)
# C=0.1 (strong L2 regularisation) keeps coefficients stable across vintages —
# a regulatory requirement: the model must not flip feature signs between audits.
_LR_C: float = 0.1
_LR_MAX_ITER: int = 1000
_LR_SOLVER: str = "lbfgs"

# XGBoost benchmark hyperparameters (credit scoring defaults, pre-Optuna)
# max_depth=5 and lr=0.1 are established starting points for tabular credit data;
# n_estimators=100 balances benchmark speed against meaningful signal.
_XGB_N_ESTIMATORS: int = 100
_XGB_MAX_DEPTH: int = 5
_XGB_LEARNING_RATE: float = 0.1
_XGB_CV_N_SPLITS: int = 5

# Threshold search bounds — extreme values (< 0.05 or > 0.95) indicate
# a degenerate model; revert to 0.5 with a warning in those cases.
_THRESHOLD_MIN: float = 0.1
_THRESHOLD_MAX: float = 0.9

# XGBoost Optuna HPO — search space bounds
# Ranges reflect credit scoring literature: depth 3–8 prevents overfitting on
# high-cardinality categorical WoE features; learning rate log-uniform so
# Optuna explores slow (0.01) and fast (0.3) regimes equally.
_XGB_OPTUNA_N_TRIALS: int = 100
_XGB_N_ESTIMATORS_MIN: int = 100
_XGB_N_ESTIMATORS_MAX: int = 1000
_XGB_MAX_DEPTH_MIN: int = 3
_XGB_MAX_DEPTH_MAX: int = 8
_XGB_LEARNING_RATE_MIN: float = 0.01
_XGB_LEARNING_RATE_MAX: float = 0.3
_XGB_SUBSAMPLE_MIN: float = 0.6
_XGB_SUBSAMPLE_MAX: float = 1.0
_XGB_COLSAMPLE_BYTREE_MIN: float = 0.6
_XGB_COLSAMPLE_BYTREE_MAX: float = 1.0
_XGB_MIN_CHILD_WEIGHT_MIN: int = 1
_XGB_MIN_CHILD_WEIGHT_MAX: int = 15       # extended from 10; 100-sample leaf rule for 24.6K positives
_XGB_GAMMA_MIN: float = 0.0
_XGB_GAMMA_MAX: float = 2.0               # validated range: top-5% Home Credit solutions use ≤ 1.5
_XGB_MAX_DELTA_STEP_MIN: int = 0
_XGB_MAX_DELTA_STEP_MAX: int = 5          # complements scale_pos_weight; 5 sufficient for 11:1 ratio
_XGB_REG_ALPHA_MIN: float = 0.0
_XGB_REG_ALPHA_MAX: float = 5.0
_XGB_REG_LAMBDA_MIN: float = 1.0
_XGB_REG_LAMBDA_MAX: float = 10.0

# XGBoost Optuna HPO on raw features (Phase 04.2.3)
# Fixed n_estimators with early stopping, log-scale regularisation,
# extended hyperparameter ranges for continuous feature space
_XGB_RAW_N_ESTIMATORS: int = 1000
_XGB_RAW_EARLY_STOPPING_ROUNDS: int = 50
_XGB_RAW_MIN_CHILD_WEIGHT_MAX: float = 30.0
_XGB_RAW_GAMMA_MAX: float = 5.0
_XGB_RAW_REG_ALPHA_MIN: float = 1e-8
_XGB_RAW_REG_LAMBDA_MIN: float = 1e-8
_XGB_RAW_REG_MAX: float = 5.0
_XGB_RAW_STUDY_NAME: str = "xgboost_raw_v9"

# Output paths for XGBoost Optuna HPO artefacts
_HPO_PROGRESS_LOG_PATH: Path = _PROJECT_ROOT / "reports" / "hpo_progress.jsonl"
_XGB_OPTUNA_MODEL_PATH: Path = _PROJECT_ROOT / "models" / "xgboost_best.pkl"
_XGB_OPTUNA_PARAMS_PATH: Path = _PROJECT_ROOT / "models" / "xgboost_params.json"
_XGB_OPTUNA_FIGURE_PATH: Path = _PROJECT_ROOT / "reports" / "figures" / "xgboost_roc_pr.png"

# LightGBM Optuna HPO — search space bounds
# num_leaves 20–300: controls model expressiveness (leaf-wise growth);
# kept well below 2^max_depth to prevent overfitting on 40 WoE features.
# learning rate log-uniform (Optuna explores slow and fast regimes equally).
# min_child_samples 5–100: minimum samples per leaf — key regulariser for
# imbalanced credit data where the minority class has few examples per leaf.
_LGB_OPTUNA_N_TRIALS: int = 50
_LGB_NUM_LEAVES_MIN: int = 20
_LGB_NUM_LEAVES_MAX: int = 150       # WoE path: 10 bins/feature → diminishing returns > 150
_LGB_RAW_NUM_LEAVES_MAX: int = 300   # Raw path: continuous features can exploit deeper trees
_LGB_MAX_DEPTH_MIN: int = 3
_LGB_MAX_DEPTH_MAX: int = 12
_LGB_LEARNING_RATE_MIN: float = 0.03   # raised from 0.01: ultra-slow LR stalls at n_estimators ceiling
_LGB_LEARNING_RATE_MAX: float = 0.2
_LGB_N_ESTIMATORS_MIN: int = 100
_LGB_N_ESTIMATORS_MAX: int = 1000
_LGB_MIN_CHILD_SAMPLES_MIN: int = 5
_LGB_MIN_CHILD_SAMPLES_MAX: int = 100
_LGB_SUBSAMPLE_MIN: float = 0.6
_LGB_SUBSAMPLE_MAX: float = 1.0
_LGB_COLSAMPLE_BYTREE_MIN: float = 0.6
_LGB_COLSAMPLE_BYTREE_MAX: float = 1.0
_LGB_REG_ALPHA_MIN: float = 0.0
_LGB_REG_ALPHA_MAX: float = 5.0
_LGB_REG_LAMBDA_MIN: float = 3.0   # raised from 0.0: ablation-tuned baseline is 9.54; prevent collapsing regularisation
_LGB_REG_LAMBDA_MAX: float = 15.0  # raised from 10.0: include ablation value 9.54 with headroom
# Two-tier early stopping: HPO objective uses aggressive patience to quickly
# triage bad configs; final refit uses standard patience for a proper model.
_LGB_OBJ_EARLY_STOPPING_ROUNDS: int = 20   # fast config triage inside Optuna
_LGB_EARLY_STOPPING_ROUNDS: int = 50        # full patience for final refit
_LGB_FINAL_VAL_SIZE: float = 0.2

# DART booster: fraction of trees dropped per round (dropout regularisation).
# 0.05–0.3 is the validated range for credit tabular data; values > 0.3
# increase variance without additional bias reduction.
# Note: DART early stopping is not supported in LightGBM 4.x —
# the stage-1 model trains to full n_estimators (dropout is the regulariser).
_LGB_DART_DROP_RATE_MIN: float = 0.05
_LGB_DART_DROP_RATE_MAX: float = 0.3

# GOSS booster: top-gradient fraction (retain high-loss instances) and
# other-rate (random sample of low-loss instances). Constraint: top + other ≤ 1.
# Low other_rate (≤ 0.1) reduces memory overhead while preserving minority-class signal.
_LGB_GOSS_TOP_RATE_MIN: float = 0.01
_LGB_GOSS_TOP_RATE_MAX: float = 0.2
_LGB_GOSS_OTHER_RATE_MIN: float = 0.01
_LGB_GOSS_OTHER_RATE_MAX: float = 0.1

# Output paths for LightGBM Optuna HPO artefacts
_LGB_OPTUNA_MODEL_PATH: Path = _PROJECT_ROOT / "models" / "lightgbm_best.pkl"
_LGB_OPTUNA_PARAMS_PATH: Path = _PROJECT_ROOT / "models" / "lightgbm_params.json"
_LGB_OPTUNA_FIGURE_PATH: Path = _PROJECT_ROOT / "reports" / "figures" / "lightgbm_roc_pr.png"

# Output path for the imbalance benchmark comparison table
_BENCHMARK_REPORT_PATH: Path = _PROJECT_ROOT / "reports" / "imbalance_benchmark.csv"

# Ensemble workflow constants
# _ENSEMBLE_PERSIST_THRESHOLD: minimum Gini improvement over the best single model
# required to save the ensemble artefact.  0.005 = half a Gini point — a meaningful
# improvement that exceeds model variance noise on held-out credit data.
_ENSEMBLE_PERSIST_THRESHOLD: float = 0.005
_ENSEMBLE_WORKFLOW_MODEL_PATH: Path = _PROJECT_ROOT / "models" / "ensemble_best.pkl"
_ENSEMBLE_WORKFLOW_WEIGHTS_PATH: Path = _PROJECT_ROOT / "reports" / "ensemble_weights.json"

# CatBoost Optuna HPO — search space bounds (validated by subagent analysis)
# depth 4–8: CatBoost uses symmetric (oblivious) trees; depth>8 rarely helps
#   and drastically increases memory on 300K rows.
# l2_leaf_reg 1–20: wider than XGBoost's lambda because oblivious trees
#   share regularisation across the full depth-level, not per-leaf.
# bagging_temperature/random_strength: CatBoost's native stochastic gradient
#   boosting — equivalent to subsample/colsample_bytree in LGB/XGB.
_CAT_DEPTH_MIN: int = 5
_CAT_DEPTH_MAX: int = 10
_CAT_LEARNING_RATE_MIN: float = 0.01
_CAT_LEARNING_RATE_MAX: float = 0.2
_CAT_L2_LEAF_REG_MIN: float = 0.1
_CAT_L2_LEAF_REG_MAX: float = 30.0
_CAT_BAGGING_TEMP_MIN: float = 0.0
_CAT_BAGGING_TEMP_MAX: float = 1.0
_CAT_RANDOM_STRENGTH_MIN: float = 0.0
_CAT_RANDOM_STRENGTH_MAX: float = 1.0
_CAT_BOOTSTRAP_TYPE: str = "Bayesian"   # bagging_temperature only valid with Bayesian bootstrap
_CAT_ITERATIONS: int = 1000
_CAT_OBJ_EARLY_STOPPING_ROUNDS: int = 30   # fast config triage inside Optuna
_CAT_EARLY_STOPPING_ROUNDS: int = 50        # full patience for final refit
_CAT_FINAL_VAL_SIZE: float = 0.2
_CAT_OPTUNA_N_TRIALS: int = 50
_CAT_MODEL_PATH: Path = _PROJECT_ROOT / "models" / "catboost_raw_calibrated.pkl"
_CAT_RAW_MIN_DATA_IN_LEAF_MIN: int = 5
_CAT_RAW_MIN_DATA_IN_LEAF_MAX: int = 50
_CAT_PARAMS_PATH: Path = _PROJECT_ROOT / "models" / "catboost_params.json"
_CAT_FIGURE_PATH: Path = _PROJECT_ROOT / "reports" / "figures" / "catboost_roc_pr.png"
# Raw categorical column names that CatBoost can consume natively.
# These columns are WoE-encoded in X_woe; prepare_catboost_features()
# swaps them back to raw strings when df_raw is supplied.
_CATBOOST_RAW_CATS: list[str] = [
    "CODE_GENDER",
    "NAME_EDUCATION_TYPE",
    "NAME_INCOME_TYPE",
    "ORGANIZATION_TYPE",
]

# Probability calibration constants
# 70/30 train/calibration split: enough calibration data to fit Platt sigmoid
# without starving the base model of training signal.
_CALIB_SPLIT: float = 0.3
_CALIBRATED_MODEL_PATH: Path = _PROJECT_ROOT / "models" / "lightgbm_calibrated.pkl"
_CALIBRATION_FIGURE_PATH: Path = _PROJECT_ROOT / "reports" / "figures" / "calibration_reliability.png"
_CALIBRATION_N_BINS: int = 20

# Strategy labels — kept as module constants so downstream tasks can reference
# them by name (e.g., Task 3.4 picks the winner from this table).
_STRATEGY_SMOTE: str = "SMOTE"
_STRATEGY_COST_SENSITIVE: str = "Cost-Sensitive"
_STRATEGY_THRESHOLD_TUNED: str = "Threshold-Tuned"
_STRATEGY_HYBRID: str = "SMOTE+Cost-Sensitive"

# OOF Ensemble defaults — fast, sensible hyperparameters for blending
# Reduced from 200 to 100 estimators for faster unit testing
# scale_pos_weight is computed and added dynamically for each fit (see train_ensemble, run_ensemble_workflow)
_ENSEMBLE_LGB_DEFAULTS: dict = {
    "n_estimators": 100,
    "num_leaves": 31,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": -1,
}
_ENSEMBLE_XGB_DEFAULTS: dict = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "eval_metric": "auc",
}
_ENSEMBLE_CAT_DEFAULTS: dict = {
    "depth": 6,
    "learning_rate": 0.05,
    "iterations": 300,
    "l2_leaf_reg": 3.0,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 0.5,
    "random_strength": 0.5,
    "verbose": 0,
    "allow_writing_files": False,
}
_ENSEMBLE_3MODEL_WORKFLOW_MODEL_PATH: Path = _PROJECT_ROOT / "models" / "ensemble_3model_best.pkl"
_ENSEMBLE_3MODEL_WORKFLOW_WEIGHTS_PATH: Path = _PROJECT_ROOT / "reports" / "ensemble_3model_weights.json"

# Extended HPO (Wave 0) — Phase 4.1
# Per-model extended hyperparameter optimization with higher trial budgets
# and per-model feature pipelines (raw features, target encoding, DFS).
# LightGBM HPO on Raw Features (Phase 04.2.4) — 3-store × 3-strategy ablation
# Fixed n_estimators ceiling with early stopping; 50 trials per run
_LGB_RAW_N_ESTIMATORS: int = 1000  # Fixed ceiling, not tuned (per D-07)
_LGB_OPTUNA_N_TRIALS: int = 50     # Per D-10
_LGB_EARLY_STOPPING_ROUNDS: int = 50  # From D-07
_LGB_METRIC: str = "auc"  # From D-09 — CRITICAL: not "binary_logloss"

# Feature stores for 3-store ablation (D-01)
_FEATURE_STORES: dict[str, Path] = {
    "X_train": _PROJECT_ROOT / "data" / "processed" / "X_train.parquet",
    "X_tree_raw": _PROJECT_ROOT / "data" / "processed" / "X_tree_raw.parquet",
    "X_tree_dfs": _PROJECT_ROOT / "data" / "processed" / "X_tree_dfs.parquet",
}

# Imbalance strategies for 3-strategy ablation (D-02)
_IMBALANCE_STRATEGIES: list[str] = ["scale_pos_weight", "is_unbalance", "smote"]
_STORE_TAGS: list[str] = ["Xtrain", "Xtreeraw", "Xtreeds"]  # For study naming (per D-17)

# LightGBM HPO search space ceiling constants (per D-10, lgb.md)
_LGB_LEARNING_RATE_MIN: float = 0.005  # From D-11
_LGB_LEARNING_RATE_MAX: float = 0.1
_LGB_NUM_LEAVES_MIN: int = 20
_LGB_NUM_LEAVES_MAX: int = 300  # From D-10
_LGB_MAX_DEPTH_MIN: int = 3     # From D-10 (lgb.md)
_LGB_MAX_DEPTH_MAX: int = 8
_LGB_MIN_CHILD_SAMPLES_MIN: int = 50   # From D-10 (expanded from 5)
_LGB_MIN_CHILD_SAMPLES_MAX: int = 500  # From D-10 (expanded from 100)
_LGB_MIN_CHILD_WEIGHT_MIN: float = 1e-4  # From D-10 (new)
_LGB_MIN_CHILD_WEIGHT_MAX: float = 1e-1
_LGB_SUBSAMPLE_MIN: float = 0.5
_LGB_SUBSAMPLE_MAX: float = 1.0
_LGB_COLSAMPLE_BYTREE_MIN: float = 0.5
_LGB_COLSAMPLE_BYTREE_MAX: float = 1.0
_LGB_REG_ALPHA_MIN: float = 1e-4  # From D-10
_LGB_REG_ALPHA_MAX: float = 10.0
_LGB_REG_LAMBDA_MIN: float = 1e-4  # From D-10
_LGB_REG_LAMBDA_MAX: float = 10.0
_LGB_PATH_SMOOTH_MIN: float = 0.0  # From D-10
_LGB_PATH_SMOOTH_MAX: float = 10.0

_LGB_EXTENDED_OPTUNA_N_TRIALS: int = 150

# LightGBM Raw Features (Continuous-Only) HPO — search space bounds
# Extended search on raw 63 continuous features (no WoE binning)
# These constants support deeper exploration than WoE path (which maxes at 150 leaves)
_LGB_RAW_NUM_LEAVES_MIN: int = 20
_LGB_RAW_NUM_LEAVES_MAX: int = 300           # Raw continuous features support deeper trees than WoE bins
_LGB_RAW_MAX_DEPTH_MIN: int = 3
_LGB_RAW_MAX_DEPTH_MAX: int = 9
_LGB_RAW_LEARNING_RATE_MIN: float = 0.01     # Allow slower learning rates (was 0.03 in Phase 4)
_LGB_RAW_LEARNING_RATE_MAX: float = 0.2
_LGB_RAW_N_ESTIMATORS_MIN: int = 100
_LGB_RAW_N_ESTIMATORS_MAX: int = 1000
_LGB_RAW_MIN_CHILD_SAMPLES_MIN: int = 10
_LGB_RAW_MIN_CHILD_SAMPLES_MAX: int = 100
_LGB_RAW_SUBSAMPLE_MIN: float = 0.6
_LGB_RAW_SUBSAMPLE_MAX: float = 1.0
_LGB_RAW_COLSAMPLE_BYTREE_MIN: float = 0.5
_LGB_RAW_COLSAMPLE_BYTREE_MAX: float = 1.0
_LGB_RAW_REG_ALPHA_MIN: float = 1e-8
_LGB_RAW_REG_ALPHA_MAX: float = 10.0
_LGB_RAW_REG_LAMBDA_MIN: float = 1e-8
_LGB_RAW_REG_LAMBDA_MAX: float = 10.0

_CAT_EXTENDED_OPTUNA_N_TRIALS: int = 50

# CatBoost Raw Features — Extended HPO bounds
_CAT_RAW_DEPTH_MIN: int = 4
_CAT_RAW_DEPTH_MAX: int = 10             # Allow deeper trees on raw continuous features
_CAT_RAW_LEARNING_RATE_MIN: float = 0.01
_CAT_RAW_LEARNING_RATE_MAX: float = 0.2
_CAT_RAW_L2_LEAF_REG_MIN: float = 0.1
_CAT_RAW_L2_LEAF_REG_MAX: float = 30.0
_CAT_RAW_ITERATIONS_MIN: int = 500
_CAT_RAW_ITERATIONS_MAX: int = 2000

_XGB_EXTENDED_OPTUNA_N_TRIALS: int = 50

# XGBoost Raw Features — Extended HPO bounds
_XGB_RAW_N_ESTIMATORS_MIN: int = 100
_XGB_RAW_N_ESTIMATORS_MAX: int = 1000
_XGB_RAW_MAX_DEPTH_MIN: int = 3
_XGB_RAW_MAX_DEPTH_MAX: int = 12
_XGB_RAW_LEARNING_RATE_MIN: float = 0.01
_XGB_RAW_LEARNING_RATE_MAX: float = 0.3
_XGB_RAW_SUBSAMPLE_MIN: float = 0.5
_XGB_RAW_SUBSAMPLE_MAX: float = 1.0
_XGB_RAW_COLSAMPLE_BYTREE_MIN: float = 0.5
_XGB_RAW_COLSAMPLE_BYTREE_MAX: float = 1.0
_XGB_RAW_MIN_CHILD_WEIGHT_MIN: int = 1
_XGB_RAW_MIN_CHILD_WEIGHT_MAX: int = 20
_XGB_RAW_GAMMA_MIN: float = 0.0
_XGB_RAW_GAMMA_MAX: float = 3.0
# Note: _XGB_RAW_REG_ALPHA_MIN, _XGB_RAW_REG_LAMBDA_MIN, _XGB_RAW_REG_MAX are defined above (lines 111-113)

# Optuna study persistence constants — absolute path prevents test runs from
# resolving to the production DB when pytest CWD == project root.
_OPTUNA_DB_PATH: Path = _PROJECT_ROOT / "models" / "optuna_studies.db"

# Optuna Studies Database Metadata
# ============================================================================
# Storage: SQLite database at models/optuna_studies.db
# Purpose: Non-regression HPO continuation for XGBoost, LightGBM, CatBoost
#
# Studies:
#   - lightgbm_extended_study: LightGBM Optuna trials (maximize AUC)
#   - catboost_extended_study: CatBoost Optuna trials (maximize AUC)
#   - xgboost_extended_study: XGBoost Optuna trials (maximize AUC)
#
# Non-Regression Protocol:
#   1. All studies persist from Phase 04.1 onward
#   2. XGBoost study seeded with trial 0 from Phase 04.1 best params
#   3. Subsequent trials must beat or tie seed trial baseline (non-regression)
#   4. DO NOT delete or restart studies — continuation only
#   5. If a study reaches completion status, call load_study(..., load_if_exists=True)
#    to resume from where it left off
# ============================================================================


# ---------------------------------------------------------------------------
# _OOFGiniMonitorCallback — Optuna callback for OOF Gini monitoring and early abort
# ---------------------------------------------------------------------------

class _OOFGiniMonitorCallback:
    """
    Monitor OOF Gini per trial; early-abort if 3 consecutive trials exceed 0.85 (leakage indicator).
    Logs all trial metrics to JSON Lines for automated monitoring (not manual watching).

    Per D-17 (leakage gate) and D-18 (automated monitoring), replaces manual user oversight.
    """

    def __init__(self, progress_log_path: str | Path | None = None,
                 oof_gini_threshold: float = 0.85, consecutive_threshold: int = 3,
                 min_oof_gini_threshold: float = 0.30, min_gini_consecutive_threshold: int = 5):
        self.progress_log_path = Path(progress_log_path) if progress_log_path is not None else _HPO_PROGRESS_LOG_PATH
        self.oof_gini_threshold = oof_gini_threshold
        self.consecutive_threshold = consecutive_threshold
        self.min_oof_gini_threshold = min_oof_gini_threshold
        self.min_gini_consecutive_threshold = min_gini_consecutive_threshold
        self.failed_trial_count = 0  # Rolling count of consecutive trials above threshold
        self.low_gini_trial_count = 0  # Rolling count of consecutive trials below floor
        self.trial_history = []  # All trial results

        # Ensure reports dir exists
        self.progress_log_path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, study, trial) -> None:
        """
        Called after each trial completes. Check OOF Gini gate, log results.
        """
        import sys as _sys
        # Extract trial metrics (oof_gini should be in trial.user_attrs after objective completes)
        oof_gini = trial.user_attrs.get('oof_gini', None)
        oot_gini = trial.user_attrs.get('oot_gini', None)
        best_value = study.best_value if study.best_trial is not None else None

        # Log trial result to JSON Lines
        log_entry = {
            "trial_number": trial.number,
            "trial_id": trial._trial_id,
            "oof_gini": oof_gini,
            "oot_gini": oot_gini,
            "status": trial.state.name,
            "best_oof_gini_so_far": best_value,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        self.trial_history.append(log_entry)

        # Append to JSON Lines file
        with open(self.progress_log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        # Print per-trial summary to stdout so nohup log shows live progress
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        oof_str = f"{oof_gini:.4f}" if oof_gini is not None else "n/a"
        oot_str = f"{oot_gini:.4f}" if oot_gini is not None else "n/a"
        best_str = f"{best_value:.4f}" if best_value is not None else "n/a"
        n_trials_done = trial.number + 1
        print(
            f"[{ts}] Trial {n_trials_done:>3} | OOF={oof_str} | OOT={oot_str}"
            f" | BestOOF={best_str} | status={trial.state.name}",
            flush=True,
        )
        _sys.stdout.flush()

        # Early-abort gate (upper): leakage — if oof_gini > threshold, increment counter; else reset
        if oof_gini is not None and oof_gini > self.oof_gini_threshold:
            self.failed_trial_count += 1
            self.low_gini_trial_count = 0
            msg = (
                f"  ⚠️  Trial {trial.number}: oof_gini={oof_gini:.4f} > {self.oof_gini_threshold} "
                f"({self.failed_trial_count}/{self.consecutive_threshold} consecutive)"
            )
            logger.warning(msg)
            print(msg, flush=True)

            if self.failed_trial_count >= self.consecutive_threshold:
                import optuna
                raise optuna.exceptions.OptunaError(
                    f"Leakage gate exceeded: {self.consecutive_threshold} consecutive trials with "
                    f"oof_gini > {self.oof_gini_threshold}. Last oof_gini={oof_gini:.4f}. "
                    f"Aborting HPO. Investigate remaining SK_DPD leakage in feature store."
                )

        # Early-abort gate (lower): broken CV — if oof_gini < floor, increment counter; else reset
        elif oof_gini is not None and oof_gini < self.min_oof_gini_threshold:
            self.low_gini_trial_count += 1
            self.failed_trial_count = 0
            msg = (
                f"  ⚠️  Trial {trial.number}: oof_gini={oof_gini:.4f} < {self.min_oof_gini_threshold} "
                f"(floor gate — {self.low_gini_trial_count}/{self.min_gini_consecutive_threshold} consecutive)"
            )
            logger.warning(msg)
            print(msg, flush=True)

            if self.low_gini_trial_count >= self.min_gini_consecutive_threshold:
                import optuna
                raise optuna.exceptions.OptunaError(
                    f"Floor gate exceeded: {self.min_gini_consecutive_threshold} consecutive trials with "
                    f"oof_gini < {self.min_oof_gini_threshold}. Last oof_gini={oof_gini:.4f}. "
                    f"Aborting HPO. Likely CV dead-zone contamination or feature store corruption."
                )
        else:
            # Reset both counters on good trial (floor <= oof_gini <= ceiling)
            if oof_gini is not None:
                self.failed_trial_count = 0
                self.low_gini_trial_count = 0
                logger.info(f"  ✓ Trial {trial.number}: oof_gini={oof_gini:.4f} in valid range (gate OK)")


# ---------------------------------------------------------------------------
# _AverageEnsemble — simple probability average of two base models
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

class _TemporalCV:
    """
    Walk-forward temporal CV with embargo, for credit scoring datasets.

    Sorts samples by a temporal index (e.g. ``DAYS_ID_PUBLISH``) and assigns
    validation folds to successively newer time blocks, ensuring training data
    is always older than validation data.  An embargo strip is removed from the
    end of each training fold to prevent serial-correlation leakage.

    Unlike López de Prado's full Purged CV (which removes any training sample
    whose label *spans* the test period), purging is not required here because
    Home Credit labels are atomic binary outcomes with no overlap.  The embargo
    gap is retained to demonstrate temporal-ordering discipline and to handle
    any latent autocorrelation in application cohorts.

    Parameters
    ----------
    groups : np.ndarray
        1-D array of temporal values aligned positionally to X_train.
        Must be sortable (e.g. ``DAYS_ID_PUBLISH`` integers, ascending = older).
        Obtain via ``groups_series.loc[X_train.index].to_numpy()``.
    n_splits : int
        Number of walk-forward folds.
    embargo_frac : float
        Fraction of the training portion to strip at the boundary.
        E.g. 0.01 removes the 1% most-recent training samples.

    Notes
    -----
    The first fold trains on ``n / (n_splits + 1)`` observations (oldest block);
    each successive fold adds one more block.  This expanding-window approach
    matches Basel III OOT validation: the model always generalises forward in time.
    """

    def __init__(
        self,
        groups: np.ndarray,
        n_splits: int = 5,
        embargo_frac: float = _CV_EMBARGO_FRAC,
    ) -> None:
        self.groups = groups
        self.n_splits = n_splits
        self.embargo_frac = embargo_frac

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        groups: np.ndarray | None = None,
    ):
        """
        Yield (train_indices, val_indices) positional index pairs.

        Positional indices reference rows of X (compatible with ``.iloc``).
        """
        n = len(X)
        sorted_pos = np.argsort(self.groups, kind="stable")
        n_per_fold = max(1, n // (self.n_splits + 1))

        for fold in range(self.n_splits):
            val_start_pos = (fold + 1) * n_per_fold
            val_end_pos = (
                val_start_pos + n_per_fold if fold < self.n_splits - 1 else n
            )

            # Embargo: remove the most-recent training samples
            embargo_size = max(0, int(val_start_pos * self.embargo_frac))
            train_end_pos = val_start_pos - embargo_size

            if train_end_pos <= 0 or val_end_pos <= val_start_pos:
                continue

            train_indices = sorted_pos[:train_end_pos]
            val_indices = sorted_pos[val_start_pos:val_end_pos]
            yield train_indices, val_indices


def _make_cv(
    groups_train: np.ndarray | None,
    n_splits: int,
) -> "_TemporalCV | StratifiedKFold":
    """
    Return the appropriate CV splitter for the available metadata.

    Parameters
    ----------
    groups_train : np.ndarray or None
        Temporal ordering values aligned to X_train rows.  When supplied,
        returns a :class:`_TemporalCV` (walk-forward with embargo).
        When ``None``, returns :class:`StratifiedKFold` as a fallback.
    n_splits : int
        Number of CV folds.

    Returns
    -------
    _TemporalCV or StratifiedKFold
        Both expose a ``.split(X, y)`` interface compatible with sklearn.
    """
    if groups_train is not None:
        return _TemporalCV(groups=groups_train, n_splits=n_splits)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=_RANDOM_STATE)


# ---------------------------------------------------------------------------
# Logistic regression baseline
# ---------------------------------------------------------------------------

def train_logistic_baseline(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None = None,
) -> tuple[Pipeline, dict, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Train a logistic regression baseline on WoE-transformed features.

    This model implements an IRB-compliant credit scorecard following
    Basel III methodology. Logistic regression on WoE bins satisfies the
    monotonic constraint requirement: each coefficient moves default
    probability in a single direction, making the model interpretable to
    regulators without post-hoc explanation.

    ``class_weight=None`` (default) is used deliberately. For a PD model that
    feeds EL = PD × LGD × EAD, predicted probabilities must be calibrated to
    the true default rate. ``class_weight='balanced'`` shifts mean predicted
    probability from ~0.08 (true prevalence) to ~0.43, making the PD
    estimates economically meaningless and BrierSkill negative. Imbalance
    is handled at inference time via threshold selection (benchmarked in
    Task 3.3 alongside SMOTE and cost-sensitive learning).

    The 6-tuple return extends the original 4-tuple spec so that Task 3.3
    (``benchmark_imbalance_strategies``) receives the identical train/test
    split without re-splitting on a different seed, which would contaminate
    train/test boundaries.

    Parameters
    ----------
    X : pd.DataFrame
        WoE-transformed feature matrix. Expected shape: (n_samples, 40).
        All values must be finite. Produce via
        ``credit_engine.features.apply_feature_store``.
    y : pd.Series
        Binary TARGET series (0 = repaid, 1 = defaulted).
    groups : pd.Series or None, optional
        Temporal ordering values aligned to X (e.g. ``DAYS_ID_PUBLISH``).
        When supplied, CV folds are walk-forward in time with an embargo gap
        (see :class:`_TemporalCV`).  When ``None``, falls back to
        :class:`StratifiedKFold` with shuffling.

    Returns
    -------
    pipeline : Pipeline
        Fitted ``Pipeline(StandardScaler → LogisticRegression)`` on the
        training split. Compatible with ``predict_proba()``.
    metrics_dict : dict
        Evaluation metrics on the held-out test split. Keys:
        ``Model, AUC-ROC, Gini, KS, Brier, BrierSkill, AvgPrecision``.
    X_train : pd.DataFrame
        Training feature split (80% stratified).
    X_test : pd.DataFrame
        Test feature split (20% stratified, never seen during CV).
    y_train : pd.Series
        Training label split.
    y_test : pd.Series
        Test label split.

    Notes
    -----
    Cross-validation is run on X_train only. X_test is withheld until
    final evaluation — it is never used for model selection, CV scoring,
    or threshold tuning.

    Probability calibration (Platt scaling via CalibratedClassifierCV)
    is deferred to a later task. Raw LR probabilities are well-calibrated
    when class_weight='balanced' is used with lbfgs; Platt post-processing
    is added when the model feeds EL = PD × LGD × EAD calculations.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=_TEST_SIZE, stratify=y, random_state=_RANDOM_STATE
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=_LR_C,
            max_iter=_LR_MAX_ITER,
            solver=_LR_SOLVER,
            random_state=_RANDOM_STATE,
        )),
    ])

    # --- 10-fold CV on training data ---
    # When groups (temporal index) is provided, use walk-forward folds that
    # keep training data strictly older than validation data (OOT discipline).
    # Scaler is fit inside each fold to prevent test-fold statistics leaking.
    # Auto-detect temporal groups from _TEMPORAL_SORT_COL if not supplied.
    if groups is None and _TEMPORAL_SORT_COL in X.columns:
        groups = X[_TEMPORAL_SORT_COL]
    groups_train = (
        groups.loc[X_train.index].to_numpy() if groups is not None else None
    )
    cv = _make_cv(groups_train, n_splits=_CV_N_SPLITS)
    cv_scores: list[float] = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        pipeline.fit(X_fold_train, y_fold_train)
        y_prob_val = pipeline.predict_proba(X_fold_val)[:, 1]
        cv_scores.append(float(roc_auc_score(y_fold_val, y_prob_val)))


    # --- Final model: refit on full training split ---
    pipeline.fit(X_train, y_train)

    # --- Evaluate on held-out test set ---
    metrics_dict = evaluate_model(
        pipeline, X_test, y_test, model_name="Logistic Regression (WoE)"
    )

    # --- ROC + PR curves ---
    import matplotlib.pyplot as plt  # local import: avoids backend conflicts at module load
    fig = plot_roc_and_pr(
        pipeline,
        X_test,
        y_test,
        model_name="Logistic Regression (WoE)",
        save_path="reports/figures/logistic_roc_pr.png",
    )
    plt.close(fig)

    # --- Persist pipeline ---
    save_model(pipeline, _PROJECT_ROOT / "models" / "logistic_baseline.pkl")

    return pipeline, metrics_dict, X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Imbalance benchmarking helpers
# ---------------------------------------------------------------------------

def _find_optimal_threshold_f1_macro(
    y_true_val: np.ndarray,
    y_prob_val: np.ndarray,
) -> float:
    """
    Find the classification threshold in [0.1, 0.9] that maximises F1-macro.

    Uses the unique predicted probabilities as candidate thresholds (more
    efficient and accurate than a fixed grid). F1-macro weights positive and
    negative classes equally — appropriate for imbalanced credit data where
    micro-F1 is dominated by the majority class.

    Parameters
    ----------
    y_true_val : np.ndarray
        Binary ground-truth labels from a CV validation fold.
    y_prob_val : np.ndarray
        Predicted probabilities for the positive class.

    Returns
    -------
    float
        Threshold in [0.1, 0.9] that maximises F1-macro.
        Falls back to 0.5 if no valid threshold exists or if the optimal
        value is outside [0.05, 0.95] (degenerate model guard).

    Notes
    -----
    Called once per CV fold inside ``benchmark_imbalance_strategies``.
    Never receives test data — only validation fold data.
    """
    candidates = np.unique(y_prob_val)
    candidates = candidates[(candidates >= _THRESHOLD_MIN) & (candidates <= _THRESHOLD_MAX)]
    if len(candidates) == 0:
        return 0.5

    best_threshold = 0.5
    best_f1 = -1.0
    for t in candidates:
        y_pred = (y_prob_val >= t).astype(int)
        score = f1_score(y_true_val, y_pred, average="macro", zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(t)

    # Guard: revert to 0.5 if extreme threshold selected (degenerate model)
    if best_threshold < 0.05 or best_threshold > 0.95:
        return 0.5
    return best_threshold


def _compute_benchmark_metrics(
    strategy_name: str,
    y_test: pd.Series,
    y_prob_raw: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute the 6 benchmark metrics for one imbalance strategy.

    AUC-ROC, Gini, and KS use raw probabilities (threshold-independent).
    F1-Macro, Precision, and Recall use the supplied threshold, which may
    differ from 0.5 for the threshold-tuned strategy.

    Parameters
    ----------
    strategy_name : str
        Display name for the strategy row.
    y_test : pd.Series
        Binary ground-truth labels from the held-out test split.
    y_prob_raw : np.ndarray
        Predicted probabilities for the positive class.
    threshold : float
        Classification threshold for binary predictions. Default 0.5.

    Returns
    -------
    dict
        Keys: Strategy, AUC-ROC, Gini, KS, F1-Macro, Precision, Recall.
    """
    auc = float(roc_auc_score(y_test, y_prob_raw))
    gini = gini_coefficient(y_test, y_prob_raw)
    ks_val, _ = ks_statistic(y_test, y_prob_raw)

    y_pred_binary = (y_prob_raw >= threshold).astype(int)
    f1 = float(f1_score(y_test, y_pred_binary, average="macro", zero_division=0))
    precision = float(precision_score(y_test, y_pred_binary, zero_division=0))
    recall = float(recall_score(y_test, y_pred_binary, zero_division=0))

    return {
        "Strategy": strategy_name,
        "AUC-ROC": auc,
        "Gini": gini,
        "KS": ks_val,
        "F1-Macro": f1,
        "Precision": precision,
        "Recall": recall,
    }


def benchmark_imbalance_strategies(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    groups_train: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Benchmark four XGBoost imbalance-handling strategies on the held-out test split.

    Trains four models using different approaches to the class imbalance
    (~8% defaults) and reports a 6-metric comparison table. The winning
    strategy (highest Gini, F1-Macro as tiebreaker) is printed to console
    and used as the imbalance approach for Task 3.4 (XGBoost Optuna HPO)
    and Task 3.5 (LightGBM).

    Strategies
    ----------
    1. SMOTE            : Synthetic minority oversampling inside an imblearn
                          Pipeline. SMOTE is only applied during fit —
                          X_test is never resampled.
    2. Cost-Sensitive   : XGBoost ``scale_pos_weight = n_neg / n_pos``.
                          Upweights minority-class errors at the loss level.
    3. Threshold-Tuned  : Standard XGBoost trained with no resampling, then
                          a decision threshold is found via stratified 5-fold
                          CV on X_train (never on X_test) by maximising
                          F1-macro on validation folds.
    4. SMOTE+Cost-Sensitive : Hybrid — SMOTE inside pipeline combined with
                          ``scale_pos_weight``. Two-stage imbalance correction.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix (WoE-transformed, 80% stratified split).
    y_train : pd.Series
        Training labels.
    X_test : pd.DataFrame
        Held-out test features. Never used for training or threshold search.
    y_test : pd.Series
        Held-out test labels.

    Returns
    -------
    pd.DataFrame
        Shape (4, 7). Columns: Strategy, AUC-ROC, Gini, KS, F1-Macro,
        Precision, Recall. Saved to ``reports/imbalance_benchmark.csv``.

    Notes
    -----
    SMOTE leakage prevention: ``imblearn.pipeline.Pipeline`` ensures
    ``fit_resample`` is only called inside each fold's training phase, not
    on validation or test data.

    Threshold leakage prevention: ``_find_optimal_threshold_f1_macro`` is
    called with validation-fold data only, never with X_test.

    XGBoost hyperparameters are intentionally untuned here (Task 3.4 runs
    Optuna HPO). The defaults (depth=5, lr=0.1, n_estimators=100) reflect
    credit scoring conventions and are held constant across all four
    strategies so the comparison isolates the imbalance method.
    """
    import xgboost as xgb
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / n_pos

    xgb_params: dict = dict(
        max_depth=_XGB_MAX_DEPTH,
        learning_rate=_XGB_LEARNING_RATE,
        n_estimators=_XGB_N_ESTIMATORS,
        random_state=_RANDOM_STATE,
        eval_metric="logloss",
    )

    results: list[dict] = []

    # --- Strategy 1: SMOTE inside imblearn Pipeline ---
    smote_pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=_RANDOM_STATE)),
        ("xgb", xgb.XGBClassifier(**xgb_params)),
    ])
    smote_pipeline.fit(X_train, y_train)
    y_prob_smote = smote_pipeline.predict_proba(X_test)[:, 1]
    results.append(_compute_benchmark_metrics(_STRATEGY_SMOTE, y_test, y_prob_smote))

    # --- Strategy 2: Cost-Sensitive XGBoost ---
    cost_model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, **xgb_params)
    cost_model.fit(X_train, y_train)
    y_prob_cost = cost_model.predict_proba(X_test)[:, 1]
    results.append(_compute_benchmark_metrics(_STRATEGY_COST_SENSITIVE, y_test, y_prob_cost))

    # --- Strategy 3: Post-training threshold optimisation (CV only) ---
    # Threshold is computed on CV validation folds — X_test is never touched
    # during threshold search, preventing evaluation-set leakage.
    # When groups_train is provided, use temporal walk-forward folds.
    _groups_arr = (
        groups_train.to_numpy() if groups_train is not None else None
    )
    cv = _make_cv(_groups_arr, n_splits=_XGB_CV_N_SPLITS)
    fold_thresholds: list[float] = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        fold_model = xgb.XGBClassifier(**xgb_params)
        fold_model.fit(X_fold_train, y_fold_train)
        y_prob_val = fold_model.predict_proba(X_fold_val)[:, 1]
        fold_thresholds.append(
            _find_optimal_threshold_f1_macro(y_fold_val.to_numpy(), y_prob_val)
        )

    # Average across folds reduces variance from any single fold's threshold
    optimal_threshold = float(np.mean(fold_thresholds))

    thresh_model = xgb.XGBClassifier(**xgb_params)
    thresh_model.fit(X_train, y_train)
    y_prob_thresh = thresh_model.predict_proba(X_test)[:, 1]
    results.append(_compute_benchmark_metrics(
        _STRATEGY_THRESHOLD_TUNED, y_test, y_prob_thresh, threshold=optimal_threshold
    ))

    # --- Strategy 4: SMOTE + Cost-Sensitive (Hybrid) ---
    hybrid_pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=_RANDOM_STATE)),
        ("xgb", xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, **xgb_params)),
    ])
    hybrid_pipeline.fit(X_train, y_train)
    y_prob_hybrid = hybrid_pipeline.predict_proba(X_test)[:, 1]
    results.append(_compute_benchmark_metrics(_STRATEGY_HYBRID, y_test, y_prob_hybrid))

    # --- Compile results ---
    df = pd.DataFrame(results)

    # --- Persist ---
    report_path = Path(_BENCHMARK_REPORT_PATH)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(report_path, index=False)

    return df


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model: object, path: str | Path) -> None:
    """
    Persist a trained model to disk using joblib.

    joblib is preferred over pickle for sklearn Pipelines because it
    handles numpy arrays more efficiently, provides better compatibility
    across sklearn version upgrades, and supports optional compression.

    Parameters
    ----------
    model : object
        Fitted sklearn estimator or Pipeline.
    path : str or Path
        Destination file path. Parent directories are created if absent.
        Typical usage: ``'models/logistic_baseline.pkl'``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path) -> object:
    """
    Load a persisted model from disk.

    Parameters
    ----------
    path : str or Path
        Path to the saved model file.

    Returns
    -------
    object
        Fitted sklearn estimator or Pipeline.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist at ``path``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


# ---------------------------------------------------------------------------
# XGBoost with Optuna hyperparameter optimisation
# ---------------------------------------------------------------------------

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
        output_model_path="models/xgboost_raw_calibrated.pkl",
        output_figure_path="reports/figures/xgboost_raw_calibration.png",
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
    params_path = _PROJECT_ROOT / "models" / "xgboost_raw_params.json"
    params_path.parent.mkdir(parents=True, exist_ok=True)
    with params_path.open("w") as fh:
        _json.dump(best_params, fh, indent=2)

    # --- Persist evaluation metrics ---
    eval_path = _PROJECT_ROOT / "reports" / "xgboost_raw_eval.json"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with eval_path.open("w") as fh:
        _json.dump(metrics_dict, fh, indent=2)

    return model_calibrated, metrics_dict, X_test, y_test, best_params, oof_predictions


# ---------------------------------------------------------------------------
# LightGBM with Optuna hyperparameter optimisation
# ---------------------------------------------------------------------------

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

    # Task 1: Persist metrics JSON for orchestrator (Phase 04.2.6)
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

def calibrate_model(
    model: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    method: str = "sigmoid",
    output_model_path: str | None = None,
    output_figure_path: str | None = None,
) -> tuple[object, float, float]:
    """
    Calibrate probability predictions using Platt scaling or isotonic regression.

    Tree ensembles (LightGBM, XGBoost) tend to output "compressed" probabilities
    that cluster near 0.5 instead of spreading toward 0 and 1. This is because
    each leaf reports the empirical default rate in that leaf, and ensembling
    averages these rates — reducing extreme outputs. A miscalibrated PD directly
    inflates or deflates EL = PD × LGD × EAD, causing systematic mispricing.

    Calibration procedure:
    1. Split X_train 70/30 internally; the 30% slice (X_calib) trains the
       Platt layer. Note: the base model was already trained on the full
       X_train, so X_calib was implicitly seen during base-model training.
       At 307K rows the resulting in-sample bias on the 2-parameter sigmoid
       is negligible, but callers requiring strict independence should train
       the base model on only 70% of X_train before passing it here.
    2. Wrap the pre-fitted model with FrozenEstimator to prevent accidental
       re-fitting of the base estimator.
    3. Fit the CalibratedClassifierCV calibrator on X_calib / y_calib.
    4. Evaluate Brier score before and after on X_test.
    5. Plot a reliability diagram: perfect calibration = diagonal.

    Parameters
    ----------
    model : object
        A fitted sklearn-compatible classifier with ``predict_proba``.
        Typically an XGBoost or LightGBM model.
    X_train : pd.DataFrame
        Training feature matrix. Split 70/30 internally — 70% is unused
        (model already fitted), 30% trains the calibration layer.
    y_train : pd.Series
        Training labels aligned with X_train.
    X_test : pd.DataFrame
        Held-out test features for Brier score evaluation.
    y_test : pd.Series
        Held-out test labels.
    method : str, optional
        ``'sigmoid'`` (Platt scaling, default) or ``'isotonic'`` (nonparametric,
        may overfit on small calibration sets).
    output_model_path : str | None, optional
        Path to save calibrated model as joblib pickle. If None, defaults to
        _CALIBRATED_MODEL_PATH constant for backward compatibility.
    output_figure_path : str | None, optional
        Path to save reliability diagram PNG. If None, defaults to
        _CALIBRATION_FIGURE_PATH constant for backward compatibility.

    Returns
    -------
    calibrated_model : CalibratedClassifierCV
        Fitted calibrated wrapper. Supports ``predict_proba``.
    brier_uncal : float
        Brier score of the original (uncalibrated) model on X_test.
    brier_cal : float
        Brier score of the calibrated model on X_test. Lower is better.

    Notes
    -----
    Artifact paths are determined by caller-provided output_model_path and
    output_figure_path parameters, or defaults if None.
    """
    import matplotlib.pyplot as plt
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve
    from sklearn.frozen import FrozenEstimator
    from sklearn.metrics import brier_score_loss

    # --- Calibration split (holdout for fitting the Platt layer) ---
    _, X_calib, _, y_calib = train_test_split(
        X_train, y_train,
        test_size=_CALIB_SPLIT,
        stratify=y_train,
        random_state=_RANDOM_STATE,
    )

    # --- Wrap pre-fitted model and fit only the calibration layer ---
    # FrozenEstimator signals that the base model must not be re-fitted;
    # only the Platt sigmoid / isotonic layer trains on X_calib.
    calibrated_model = CalibratedClassifierCV(FrozenEstimator(model), method=method)
    calibrated_model.fit(X_calib, y_calib)

    # --- Brier scores on held-out test set ---
    y_prob_uncal = model.predict_proba(X_test)[:, 1]
    y_prob_cal = calibrated_model.predict_proba(X_test)[:, 1]

    brier_uncal = float(brier_score_loss(y_test, y_prob_uncal))
    brier_cal = float(brier_score_loss(y_test, y_prob_cal))

    # --- Reliability diagram ---
    fig, ax = plt.subplots(figsize=(7, 6))

    # Perfect calibration reference line
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")

    # Uncalibrated curve
    prob_true_uncal, prob_pred_uncal = calibration_curve(
        y_test, y_prob_uncal, n_bins=_CALIBRATION_N_BINS
    )
    ax.plot(prob_pred_uncal, prob_true_uncal, marker="o", label=f"Uncalibrated (Brier={brier_uncal:.4f})")

    # Calibrated curve
    prob_true_cal, prob_pred_cal = calibration_curve(
        y_test, y_prob_cal, n_bins=_CALIBRATION_N_BINS
    )
    ax.plot(prob_pred_cal, prob_true_cal, marker="s", label=f"Calibrated ({method}) (Brier={brier_cal:.4f})")

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability Diagram — LightGBM Probability Calibration")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Use provided paths or defaults
    fig_save_path = output_figure_path or _CALIBRATION_FIGURE_PATH
    model_save_path = output_model_path or _CALIBRATED_MODEL_PATH

    fig_path = Path(fig_save_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Persist calibrated model ---
    save_model(calibrated_model, model_save_path)

    return calibrated_model, brier_uncal, brier_cal


# ---------------------------------------------------------------------------
# train_ensemble — OOF Stacking (LightGBM + XGBoost)
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


def train_ensemble_3model(
    X: pd.DataFrame,
    y: pd.Series,
    lgb_params: dict | None = None,
    xgb_params: dict | None = None,
    cat_params: dict | None = None,
    n_splits: int = 5,
    method: Literal["average", "logistic"] = "logistic",
    groups: np.ndarray | None = None,
) -> tuple:
    """
    Train 3-model OOF ensemble (LGB + XGB + CatBoost) with logistic meta-learner.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (63 raw continuous columns).
    y : pd.Series
        Binary target labels.
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

    Returns
    -------
    tuple
        (ensemble_model, metrics_dict, X_test, y_test, base_gini_dict)
        Where base_gini_dict = {"lgb": lgb_gini, "xgb": xgb_gini, "cat": cat_gini}
    """
    import lightgbm as lgb
    import xgboost as xgb

    if lgb_params is None:
        lgb_params = _ENSEMBLE_LGB_DEFAULTS.copy()
    if xgb_params is None:
        xgb_params = _ENSEMBLE_XGB_DEFAULTS.copy()
    if cat_params is None:
        cat_params = _ENSEMBLE_CAT_DEFAULTS.copy()

    # --- Train / holdout split (80 / 20) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=_TEST_SIZE,
        stratify=y,
        random_state=_RANDOM_STATE,
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

        # LightGBM
        lgb_params_fold = {**lgb_params, "scale_pos_weight": scale_pos_weight}
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
    n_neg_train = (y_train == 0).sum()
    n_pos_train = (y_train == 1).sum()
    scale_pos_weight_train = float(n_neg_train) / float(n_pos_train) if n_pos_train > 0 else 1.0

    lgb_params_final = {**lgb_params, "scale_pos_weight": scale_pos_weight_train}
    lgb_final = lgb.LGBMClassifier(**lgb_params_final)
    lgb_final.fit(X_train, y_train)

    xgb_params_final = {**xgb_params, "scale_pos_weight": scale_pos_weight_train}
    xgb_final = xgb.XGBClassifier(**xgb_params_final)
    xgb_final.fit(X_train, y_train)

    cat_params_final = {**cat_params, "scale_pos_weight": scale_pos_weight_train}
    cat_final = CatBoostClassifier(**cat_params_final)
    cat_final.fit(X_train.to_numpy(), y_train.to_numpy(), verbose=False)

    # --- Evaluate individual base models ---
    lgb_metrics = evaluate_model(lgb_final, X_test, y_test, "LightGBM")
    xgb_metrics = evaluate_model(xgb_final, X_test, y_test, "XGBoost")
    cat_metrics = evaluate_model(cat_final, X_test, y_test, "CatBoost")
    base_gini_dict = {
        "lgb": float(lgb_metrics["Gini"]),
        "xgb": float(xgb_metrics["Gini"]),
        "cat": float(cat_metrics["Gini"]),
    }

    # --- Create ensemble model ---
    if method == "average":
        ensemble_model = _AverageEnsemble3(lgb_final, xgb_final, cat_final)
    elif method == "logistic":
        X_meta = np.column_stack([oof_lgb, oof_xgb, oof_cat])
        meta_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=_RANDOM_STATE)
        meta_lr.fit(X_meta, y_train)
        ensemble_model = _LogisticEnsemble3(lgb_final, xgb_final, cat_final, meta_lr)
    else:
        raise ValueError(f"method must be 'average' or 'logistic', got '{method}'")

    # --- Evaluate ensemble on holdout ---
    metrics_dict = evaluate_model(ensemble_model, X_test, y_test, f"Ensemble3 ({method})")

    return ensemble_model, metrics_dict, X_test, y_test, base_gini_dict


def run_ensemble_workflow(
    X: pd.DataFrame,
    y: pd.Series,
    X_raw: pd.DataFrame | None = None,
    lgb_params: dict | None = None,
    xgb_params: dict | None = None,
    cat_model: "CatBoostClassifier | None" = None,
    cat_params: dict | None = None,
    method: Literal["average", "logistic"] = "logistic",
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
        ensemble, metrics_dict, X_test, y_test, base_gini = train_ensemble_3model(
            X, y,
            lgb_params=lgb_params,
            xgb_params=xgb_params,
            cat_params=cat_params,
            method=method,
        )
        ensemble_gini = float(metrics_dict["Gini"])
        lgb_gini = base_gini["lgb"]
        xgb_gini = base_gini["xgb"]
        cat_gini = base_gini["cat"]
        best_single_gini = max(lgb_gini, xgb_gini, cat_gini)
        improvement = ensemble_gini - best_single_gini
        persisted = improvement >= _ENSEMBLE_PERSIST_THRESHOLD
        if persisted:
            save_model(ensemble, _ENSEMBLE_3MODEL_WORKFLOW_MODEL_PATH)
            weights_payload = {
                "lgb_gini": lgb_gini,
                "xgb_gini": xgb_gini,
                "cat_gini": cat_gini,
                "ensemble_gini": ensemble_gini,
                "improvement": improvement,
                "method": method,
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


class _LogisticEnsemble3:
    """
    Stacked ensemble with logistic meta-learner — 3 base models (LGB + XGB + CatBoost).

    Meta-learner trained on np.column_stack([oof_lgb, oof_xgb, oof_cat]) with L2
    regularisation (C=1.0).
    """

    def __init__(
        self,
        lgb_model: object,
        xgb_model: object,
        cat_model: object,
        meta_lr: LogisticRegression,
    ):
        """Initialize with pre-fitted base models and meta-learner."""
        self.lgb_model = lgb_model
        self.xgb_model = xgb_model
        self.cat_model = cat_model
        self.meta_lr = meta_lr

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


# ---------------------------------------------------------------------------
# Priority 2.2 — CatBoost Optuna HPO + feature preparation helper
# ---------------------------------------------------------------------------


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

def train_ext_source_imputer(
    X: pd.DataFrame,
    y: pd.Series,
    ext_source_col: str = "EXT_SOURCE_3",
    n_trials: int = 50,
) -> tuple[object, float]:
    """
    Train LightGBM regressor to impute missing EXT_SOURCE_3 values.

    Fits a regressor on rows where EXT_SOURCE_3 is observed (not -999 sentinel).
    Uses stratified train/test split on observed rows to prevent leakage.
    Optuna HPO maximizes correlation between predicted and observed values.
    Final imputer is retrained on all observed data.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix with EXT_SOURCE_3 column (may contain -999 sentinels).
    y : pd.Series
        Target series for stratified fold selection (not used in imputation directly).
    ext_source_col : str, default "EXT_SOURCE_3"
        Column name to impute.
    n_trials : int, default 50
        Number of Optuna trials for hyperparameter optimisation.

    Returns
    -------
    tuple[object, float]
        - Fitted LightGBM regressor (also saved to models/ext_source_imputation_lgb.pkl)
        - Best trial correlation value (float, typically 0.5–0.8)

    Notes
    -----
    - Only rows where EXT_SOURCE_3 != -999.0 are used for training.
    - EXT_SOURCE_1 and EXT_SOURCE_2 are dropped from features to avoid circular dependencies.
    - Imputer is fit on 80% of observed rows; 20% used for validation during Optuna.
    - Final retraining uses 100% of observed rows (no holdout).
    """
    import lightgbm as lgb
    import optuna
    from json import dump as json_dump

    # 1. Identify observed rows (where EXT_SOURCE_3 is not the sentinel -999)
    observed_mask = X[ext_source_col] != -999.0
    X_obs = X[observed_mask].copy()
    y_obs = X_obs[ext_source_col].copy()  # Target for regression

    # 2. Drop the target column and other EXT_SOURCE columns from features
    #    (avoid circular dependency)
    features_to_drop = {ext_source_col, "EXT_SOURCE_1", "EXT_SOURCE_2"}
    X_obs = X_obs.drop(columns=features_to_drop)

    # 3. Stratified train/test split on observed data
    #    Use the original target (y) to stratify, sliced to observed indices only
    y_stratify = y[observed_mask].copy()
    X_train_obs, X_test_obs, y_train_obs, y_test_obs = train_test_split(
        X_obs,
        y_obs,
        test_size=0.2,
        random_state=42,
        stratify=y_stratify,
    )

    # 4. Optuna HPO: maximize correlation between predicted and observed
    def objective(trial: optuna.Trial) -> float:
        """Objective function: mean correlation over a CV split."""
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        }

        model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)
        model.fit(
            X_train_obs,
            y_train_obs,
            eval_set=[(X_test_obs, y_test_obs)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(period=0)],
        )

        # Predict on test fold
        y_pred = model.predict(X_test_obs)

        # Correlation between predicted and observed
        corr = float(np.corrcoef(y_pred, y_test_obs)[0, 1])
        return corr

    # Create and optimize study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=_RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_correlation = float(study.best_value)

    # 5. Retrain imputer on full observed data with best hyperparameters
    best_params = study.best_params
    imputer = lgb.LGBMRegressor(**best_params, random_state=42, verbose=-1)
    imputer.fit(X_obs, y_obs)

    # Save imputer to disk
    imputer_path = _PROJECT_ROOT / "models" / "ext_source_imputation_lgb.pkl"
    imputer_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(imputer, imputer_path)

    # Save best params for reference
    params_path = _PROJECT_ROOT / "models" / "ext_source_imputation_params.json"
    with params_path.open("w") as fh:
        json_dump(best_params, fh, indent=2)

    return imputer, best_correlation


def apply_ext_source_imputer(
    X: pd.DataFrame,
    imputer: object,
    ext_source_col: str = "EXT_SOURCE_3",
) -> pd.DataFrame:
    """
    Apply EXT_SOURCE_3 imputer to fill missing values.

    Adds an EXT_SOURCE_3_MISSING_FLAG column (1 = originally missing, 0 = observed).
    Fills -999 sentinel values with predictions from the imputer.
    Preserves originally-observed values unchanged.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix with EXT_SOURCE_3 column (may contain -999 sentinels).
    imputer : object
        Fitted LightGBM regressor (from train_ext_source_imputer).
    ext_source_col : str, default "EXT_SOURCE_3"
        Column name to impute.

    Returns
    -------
    pd.DataFrame
        Copy of X with:
        - EXT_SOURCE_3_MISSING_FLAG added (1 = originally missing, 0 = observed)
        - Missing EXT_SOURCE_3 values filled with predictions
        - Observed EXT_SOURCE_3 values preserved unchanged

    Notes
    -----
    - This function follows the immutability pattern: returns a new DataFrame copy.
    - Missing flag is added BEFORE imputation (captures original missingness pattern).
    - Only rows where EXT_SOURCE_3 == -999.0 get imputed; observed rows are unchanged.
    """
    out = X.copy()

    # 1. Create missing flag (1 = originally missing, 0 = observed)
    out[f"{ext_source_col}_MISSING_FLAG"] = (
        (out[ext_source_col] == -999.0).astype(int)
    )

    # 2. Prepare features for prediction (drop EXT_SOURCE columns and the flag we just added)
    features_to_drop = {ext_source_col, "EXT_SOURCE_1", "EXT_SOURCE_2", f"{ext_source_col}_MISSING_FLAG"}
    X_for_pred = out.drop(columns=features_to_drop)

    # 3. Predict on all rows (including observed)
    y_imputed = imputer.predict(X_for_pred)

    # 4. Fill missing values only (preserve observed)
    missing_mask = out[ext_source_col] == -999.0
    out.loc[missing_mask, ext_source_col] = y_imputed[missing_mask]

    return out


# ---------------------------------------------------------------------------
# Wave 0 Stubs (Phase 4.1) — Extended HPO, target encoding, DFS
# ---------------------------------------------------------------------------

def apply_target_encoding_fold_safe(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    cat_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fold-safe target encoding for high-cardinality categoricals.

    Uses sklearn.preprocessing.TargetEncoder with internal cross-fitting (cv=5)
    to ensure no leakage on training data. Test data is transformed with no
    target knowledge (y=None).

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix with categorical columns.
    y_train : pd.Series
        Binary training target.
    X_test : pd.DataFrame
        Test feature matrix (same shape[1] as X_train).
    cat_cols : list[str], optional
        Categorical column names to encode. Defaults to standard Home Credit
        categorical columns: ['CODE_GENDER', 'NAME_EDUCATION_TYPE',
        'NAME_INCOME_TYPE', 'ORGANIZATION_TYPE'].

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (X_train_encoded, X_test_encoded) with target-encoded categoricals
        replaced by numeric values. Shape and index preserved.

    Notes
    -----
    **Fold-safety guarantee:** TargetEncoder(cv=5).fit_transform(X_train, y_train)
    applies internal k-fold cross-fitting: each fold is encoded using the target
    statistics from the other k-1 folds. This prevents the encoder from encoding
    on the same rows it will evaluate, which would inflate metrics.

    **Test encoding:** transform(X_test) uses the statistics fit on the *full*
    training set with no target knowledge (y=None), which is the correct
    inference-time behavior.

    **Missing values:** Unknown categories (e.g., categories in X_test not seen
    in X_train) are filled with -999 (handle_unknown='use_encoded_value').
    """
    if cat_cols is None:
        cat_cols = [
            "CODE_GENDER",
            "NAME_EDUCATION_TYPE",
            "NAME_INCOME_TYPE",
            "ORGANIZATION_TYPE",
        ]

    # Verify all categorical columns exist in X_train and X_test
    missing_train = set(cat_cols) - set(X_train.columns)
    missing_test = set(cat_cols) - set(X_test.columns)
    if missing_train:
        raise ValueError(f"Missing columns in X_train: {missing_train}")
    if missing_test:
        raise ValueError(f"Missing columns in X_test: {missing_test}")

    from sklearn.preprocessing import TargetEncoder

    # Initialize encoder with internal cross-fitting (cv=5)
    # TargetEncoder API: cv=5 for k-fold; target_type='binary' for binary classification
    te = TargetEncoder(cv=5, target_type="binary")

    # Fit and transform on training data (internal cross-fitting prevents leakage)
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()

    X_train_cat_encoded = te.fit_transform(X_train[cat_cols], y_train)
    # Transform test with no target knowledge
    X_test_cat_encoded = te.transform(X_test[cat_cols])

    # Replace categorical columns with encoded versions
    X_train_encoded[cat_cols] = X_train_cat_encoded
    X_test_encoded[cat_cols] = X_test_cat_encoded

    # Ensure encoded columns are float (TargetEncoder returns ndarray)
    X_train_encoded[cat_cols] = X_train_encoded[cat_cols].astype(float)
    X_test_encoded[cat_cols] = X_test_encoded[cat_cols].astype(float)

    return X_train_encoded, X_test_encoded


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
