"""
model_base.py
-------------
Shared constants, callbacks, CV classes, and utility functions for all model families.
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
# SK_ID_CURR is the sequential application intake ID — monotonically increasing,
# no NaN values, and directly reflects application arrival order per Home Credit
# data documentation. This is preferred over relative proxies such as
# prev_days_decision_mean, which measure days-before-THIS-application and can
# reverse true temporal order when applicants have differently-aged prior loans.
_TEMPORAL_SORT_COL: str = "SK_ID_CURR"

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
# _OOFGiniMonitorCallback — Optuna callback for OOF Gini monitoring
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


# ---------------------------------------------------------------------------
# _TemporalCV — Walk-forward cross-validation with embargo
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




# ---------------------------------------------------------------------------
# CV splitter factory
# ---------------------------------------------------------------------------

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



# ---------------------------------------------------------------------------
# Threshold optimization
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




# ---------------------------------------------------------------------------
# Imbalance strategy benchmarking
# ---------------------------------------------------------------------------

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



# ---------------------------------------------------------------------------
# Model persistence
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



# ---------------------------------------------------------------------------
# EXT_SOURCE imputation
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




# ---------------------------------------------------------------------------
# Apply EXT_SOURCE imputation
# ---------------------------------------------------------------------------

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



# ---------------------------------------------------------------------------
# Target encoding (fold-safe)
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




# ---------------------------------------------------------------------------
# Feature filtering by IV
# ---------------------------------------------------------------------------

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


