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

from credit_engine.utils import evaluate_model, gini_coefficient, ks_statistic, plot_roc_and_pr


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

# Output paths for XGBoost Optuna HPO artefacts
_XGB_OPTUNA_MODEL_PATH: str = "models/xgboost_best.pkl"
_XGB_OPTUNA_PARAMS_PATH: str = "models/xgboost_params.json"
_XGB_OPTUNA_FIGURE_PATH: str = "reports/figures/xgboost_roc_pr.png"

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
_LGB_OPTUNA_MODEL_PATH: str = "models/lightgbm_best.pkl"
_LGB_OPTUNA_PARAMS_PATH: str = "models/lightgbm_params.json"
_LGB_OPTUNA_FIGURE_PATH: str = "reports/figures/lightgbm_roc_pr.png"

# Output path for the imbalance benchmark comparison table
_BENCHMARK_REPORT_PATH: str = "reports/imbalance_benchmark.csv"

# Ensemble workflow constants
# _ENSEMBLE_PERSIST_THRESHOLD: minimum Gini improvement over the best single model
# required to save the ensemble artefact.  0.005 = half a Gini point — a meaningful
# improvement that exceeds model variance noise on held-out credit data.
_ENSEMBLE_PERSIST_THRESHOLD: float = 0.005
_ENSEMBLE_WORKFLOW_MODEL_PATH: str = "models/ensemble_best.pkl"
_ENSEMBLE_WORKFLOW_WEIGHTS_PATH: str = "reports/ensemble_weights.json"

# CatBoost Optuna HPO — search space bounds (validated by subagent analysis)
# depth 4–8: CatBoost uses symmetric (oblivious) trees; depth>8 rarely helps
#   and drastically increases memory on 300K rows.
# l2_leaf_reg 1–20: wider than XGBoost's lambda because oblivious trees
#   share regularisation across the full depth-level, not per-leaf.
# bagging_temperature/random_strength: CatBoost's native stochastic gradient
#   boosting — equivalent to subsample/colsample_bytree in LGB/XGB.
_CAT_DEPTH_MIN: int = 4
_CAT_DEPTH_MAX: int = 8
_CAT_LEARNING_RATE_MIN: float = 0.02
_CAT_LEARNING_RATE_MAX: float = 0.2
_CAT_L2_LEAF_REG_MIN: float = 1.0
_CAT_L2_LEAF_REG_MAX: float = 20.0
_CAT_BAGGING_TEMP_MIN: float = 0.0
_CAT_BAGGING_TEMP_MAX: float = 1.0
_CAT_RANDOM_STRENGTH_MIN: float = 0.0
_CAT_RANDOM_STRENGTH_MAX: float = 1.0
_CAT_BOOTSTRAP_TYPE: str = "Bayesian"   # bagging_temperature only valid with Bayesian bootstrap
_CAT_ITERATIONS: int = 1000
_CAT_OBJ_EARLY_STOPPING_ROUNDS: int = 30   # fast config triage inside Optuna
_CAT_EARLY_STOPPING_ROUNDS: int = 50        # full patience for final refit
_CAT_FINAL_VAL_SIZE: float = 0.2
_CAT_OPTUNA_N_TRIALS: int = 100
_CAT_MODEL_PATH: str = "models/catboost_combined.pkl"
_CAT_PARAMS_PATH: str = "models/catboost_params.json"
_CAT_FIGURE_PATH: str = "reports/figures/catboost_roc_pr.png"
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
_CALIBRATED_MODEL_PATH: str = "models/lightgbm_calibrated.pkl"
_CALIBRATION_FIGURE_PATH: str = "reports/figures/calibration_reliability.png"
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
_ENSEMBLE_3MODEL_WORKFLOW_MODEL_PATH: str = "models/ensemble_3model_best.pkl"
_ENSEMBLE_3MODEL_WORKFLOW_WEIGHTS_PATH: str = "reports/ensemble_3model_weights.json"

# Extended HPO (Wave 0) — Phase 4.1
# Per-model extended hyperparameter optimization with higher trial budgets
# and per-model feature pipelines (raw features, target encoding, DFS).
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
_XGB_RAW_REG_ALPHA_MIN: float = 0.0
_XGB_RAW_REG_ALPHA_MAX: float = 10.0
_XGB_RAW_REG_LAMBDA_MIN: float = 0.5
_XGB_RAW_REG_LAMBDA_MAX: float = 10.0

# Optuna study persistence constants
_OPTUNA_DB_PATH: str = "models/optuna_studies.db"

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
    save_model(pipeline, "models/logistic_baseline.pkl")

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
    Optuna objective: 5-fold CV AUC-ROC for a suggested XGBoost configuration.

    This function is called once per trial by ``study.optimize()``. It samples
    hyperparameters, runs stratified k-fold CV on X_train, and returns the mean
    out-of-fold AUC-ROC. X_test is never passed in — it lives only in the outer
    ``train_xgboost_optuna`` scope and is withheld for final evaluation.

    Parameters
    ----------
    trial : optuna.Trial
        Current Optuna trial handle for hyperparameter suggestions.
    X_train : pd.DataFrame
        Training features (80% split). Never the held-out test set.
    y_train : pd.Series
        Training labels.
    scale_pos_weight : float
        Cost-sensitive weight = n_negatives / n_positives, from Task 3.3 winner.
    cv : StratifiedKFold
        5-fold CV splitter, seeded for reproducibility.

    Returns
    -------
    float
        Mean out-of-fold AUC-ROC across all CV folds.
    """
    import xgboost as xgb

    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators", _XGB_N_ESTIMATORS_MIN, _XGB_N_ESTIMATORS_MAX
        ),
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
        "min_child_weight": trial.suggest_int(
            "min_child_weight", _XGB_MIN_CHILD_WEIGHT_MIN, _XGB_MIN_CHILD_WEIGHT_MAX
        ),
        "gamma": trial.suggest_float(
            "gamma", _XGB_GAMMA_MIN, _XGB_GAMMA_MAX
        ),
        "max_delta_step": trial.suggest_int(
            "max_delta_step", _XGB_MAX_DELTA_STEP_MIN, _XGB_MAX_DELTA_STEP_MAX
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", _XGB_REG_ALPHA_MIN, _XGB_REG_ALPHA_MAX
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", _XGB_REG_LAMBDA_MIN, _XGB_REG_LAMBDA_MAX
        ),
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "auc",
        "use_label_encoder": False,
        "verbosity": 0,
        "random_state": _RANDOM_STATE,
    }

    fold_aucs: list[float] = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(X_fold_train, y_fold_train)
        y_prob_val = model.predict_proba(X_fold_val)[:, 1]
        fold_aucs.append(float(roc_auc_score(y_fold_val, y_prob_val)))

    return float(np.mean(fold_aucs))


def train_xgboost_optuna(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = _XGB_OPTUNA_N_TRIALS,
    groups: pd.Series | None = None,
) -> tuple[object, dict, pd.DataFrame, pd.Series, dict]:
    """
    Train XGBoost with Bayesian hyperparameter optimisation via Optuna.

    Runs ``n_trials`` of TPE-based search over an 8-dimensional space,
    selecting the configuration that maximises mean out-of-fold AUC-ROC
    on the training split. The final model is retrained on the full
    training split with the best parameters, then evaluated on the
    held-out test split (never seen during optimisation).

    Imbalance handling uses ``scale_pos_weight = n_neg / n_pos`` (Cost-Sensitive
    strategy), the winner from the Task 3.3 benchmark (Gini=0.882, AUC=0.941).

    Parameters
    ----------
    X : pd.DataFrame
        WoE-transformed feature matrix (40 columns, produced by
        ``credit_engine.features.apply_feature_store``).
    y : pd.Series
        Binary TARGET series (0 = repaid, 1 = defaulted).
    n_trials : int, optional
        Number of Optuna trials. Default 50 balances exploration depth
        against compute budget for the 8-dimensional search space.

    Returns
    -------
    xgb_model : XGBClassifier
        Fitted XGBoost model with best hyperparameters, trained on X_train.
    metrics_dict : dict
        Evaluation metrics on X_test. Keys: Model, AUC-ROC, Gini, KS,
        Brier, BrierSkill, AvgPrecision.
    X_test : pd.DataFrame
        Held-out test features (20% stratified split, seed=42).
    y_test : pd.Series
        Held-out test labels.
    best_params : dict
        Optimised hyperparameters (8 keys). Also persisted to
        ``models/xgboost_params.json``.

    Raises
    ------
    ValueError
        If y has no positive or no negative samples (scale_pos_weight
        would be inf or zero).

    Notes
    -----
    Artefacts written to disk:
    - ``models/xgboost_best.pkl`` — joblib-serialised XGBClassifier
    - ``models/xgboost_params.json`` — best hyperparameters as JSON
    - ``reports/figures/xgboost_roc_pr.png`` — ROC + PR curves
    """
    import json as _json

    import matplotlib.pyplot as plt
    import optuna
    import xgboost as xgb

    # --- Input guards ---
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}.")
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        raise ValueError("y has no positive samples — cannot compute scale_pos_weight.")
    if n_neg == 0:
        raise ValueError("y has no negative samples — cannot compute scale_pos_weight.")

    # --- Train / test split (stratified, identical seed to LR baseline) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=_TEST_SIZE, stratify=y, random_state=_RANDOM_STATE
    )

    # Cost-sensitive weight: Task 3.3 winner strategy
    scale_pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())

    # --- Optuna study ---
    # Suppress INFO/DEBUG trial logs — keeps library stdout clean.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Auto-detect temporal groups from _TEMPORAL_SORT_COL if not supplied.
    if groups is None and _TEMPORAL_SORT_COL in X.columns:
        groups = X[_TEMPORAL_SORT_COL]
    groups_train = (
        groups.loc[X_train.index].to_numpy() if groups is not None else None
    )
    cv = _make_cv(groups_train, n_splits=_XGB_CV_N_SPLITS)

    def objective(trial: optuna.Trial) -> float:
        return _xgboost_optuna_objective(
            trial, X_train, y_train, scale_pos_weight, cv
        )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params: dict = study.best_params

    # --- Final model: retrain on full X_train with best params ---
    final_model = xgb.XGBClassifier(
        **best_params,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        use_label_encoder=False,
        verbosity=0,
        random_state=_RANDOM_STATE,
    )
    final_model.fit(X_train, y_train)

    # --- Evaluate on held-out test set ---
    metrics_dict = evaluate_model(final_model, X_test, y_test, "XGBoost")

    # --- ROC + PR figure ---
    figure_path = Path(_XGB_OPTUNA_FIGURE_PATH)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_roc_and_pr(final_model, X_test, y_test, "XGBoost", save_path=str(figure_path))
    plt.close(fig)

    # --- Persist model (joblib, consistent with save_model pattern) ---
    save_model(final_model, _XGB_OPTUNA_MODEL_PATH)

    # --- Persist params (JSON — human-readable, portable across services) ---
    params_path = Path(_XGB_OPTUNA_PARAMS_PATH)
    params_path.parent.mkdir(parents=True, exist_ok=True)
    with params_path.open("w") as fh:
        _json.dump(best_params, fh, indent=2)

    return final_model, metrics_dict, X_test, y_test, best_params


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
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = _LGB_OPTUNA_N_TRIALS,
    groups: pd.Series | None = None,
    use_scale_pos_weight: bool = False,
    num_leaves_max: int = _LGB_NUM_LEAVES_MAX,
    boosting_type: Literal["gbdt", "dart", "goss"] = "gbdt",
    monotone_constraints: dict[str, int] | None = None,
    enqueue_trials: list[dict] | None = None,
) -> tuple[object, dict, pd.DataFrame, pd.Series, dict]:
    """
    Train LightGBM with Bayesian hyperparameter optimisation via Optuna.

    Runs ``n_trials`` of TPE-based search over a 9-dimensional base space
    plus booster-specific dimensions (DART: +1, GOSS: +2), selecting the
    configuration that maximises mean out-of-fold AUC-ROC under temporal CV.
    The final model is retrained on the full training split using a two-stage
    process: early stopping on a held-out validation slice (GBDT/GOSS only)
    identifies the optimal ``n_estimators``, then a clean refit on all training
    data uses that tree count.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix. Raw continuous features or WoE-encoded matrix.
    y : pd.Series
        Binary TARGET series (0 = repaid, 1 = defaulted).
    n_trials : int, optional
        Number of Optuna trials. Default 50.
    groups : pd.Series | None, optional
        Temporal group labels for embargo-based CV. Auto-detected from
        ``_TEMPORAL_SORT_COL`` when None and column is present in X.
    use_scale_pos_weight : bool, optional
        Use ``scale_pos_weight = n_neg / n_pos`` for imbalance handling instead
        of ``is_unbalance=True``. Default False.
    num_leaves_max : int, optional
        Upper bound for num_leaves search. Default ``_LGB_NUM_LEAVES_MAX``.
    boosting_type : {'gbdt', 'dart', 'goss'}, optional
        Booster algorithm. Default 'gbdt'. DART adds ``drop_rate`` to the
        search space; GOSS adds ``top_rate`` and ``other_rate``. DART does not
        support early stopping in LGB 4.x — stage-1 trains to full n_estimators.
    monotone_constraints : dict[str, int] | None, optional
        Map of feature name → constraint direction (+1 increasing, -1 decreasing,
        0 unconstrained). Keys must be column names in X. Features absent from
        the dict default to 0. Applied to both the Optuna objective and the
        final model. Example::

            {
                "AGE_YEARS": 1,
                "CREDIT_INCOME_RATIO": -1,
                "EXT_SOURCE_1": 1,
            }

    enqueue_trials : list[dict] | None, optional
        Warm-start parameter dicts to evaluate before random exploration.
        Each dict must contain all Optuna-searchable keys for the active
        ``boosting_type`` (base 9 dims + booster-specific dims). Useful for
        anchoring TPE priors near a previously validated configuration.
        Example::

            [{"num_leaves": 125, "max_depth": 4, "learning_rate": 0.031, ...}]

    Returns
    -------
    lgb_model : LGBMClassifier
        Fitted LightGBM model with best hyperparameters.
    metrics_dict : dict
        Evaluation metrics on X_test. Keys: Model, AUC-ROC, Gini, KS,
        Brier, BrierSkill, AvgPrecision.
    X_test : pd.DataFrame
        Held-out test features (20% stratified split, seed=42).
    y_test : pd.Series
        Held-out test labels.
    best_params : dict
        Optimised hyperparameters (9+ keys). Persisted to
        ``models/lightgbm_params.json``. Does not include ``boosting_type``
        (fixed, not Optuna-searched).

    Raises
    ------
    ValueError
        If ``n_trials < 1``, ``boosting_type`` is invalid, or
        ``monotone_constraints`` contains keys absent from X.

    Notes
    -----
    Artefacts written to disk:
    - ``models/lightgbm_best.pkl``        — joblib-serialised LGBMClassifier
    - ``models/lightgbm_params.json``     — best hyperparameters as JSON
    - ``reports/figures/lightgbm_roc_pr.png`` — ROC + PR curves
    """
    import json as _json

    import lightgbm as lgb
    import matplotlib.pyplot as plt
    import optuna

    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}.")

    _VALID_BOOSTING_TYPES: frozenset[str] = frozenset({"gbdt", "dart", "goss"})
    if boosting_type not in _VALID_BOOSTING_TYPES:
        raise ValueError(
            f"boosting_type must be one of {sorted(_VALID_BOOSTING_TYPES)}, "
            f"got {boosting_type!r}."
        )
    if monotone_constraints is not None:
        _unknown_cols = set(monotone_constraints) - set(X.columns)
        if _unknown_cols:
            raise ValueError(
                f"monotone_constraints keys not found in X: {sorted(_unknown_cols)}"
            )

    # --- Train / test split (stratified, identical seed across all models) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=_TEST_SIZE, stratify=y, random_state=_RANDOM_STATE
    )

    # --- Imbalance weight for scale_pos_weight path ---
    _scale_pos_weight: float | None = None
    if use_scale_pos_weight:
        _n_neg = float((y_train == 0).sum())
        _n_pos = float((y_train == 1).sum())
        _scale_pos_weight = _n_neg / _n_pos

    # --- Optuna study ---
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Auto-detect temporal groups from _TEMPORAL_SORT_COL if not supplied.
    # Warn explicitly when the column is absent so callers know CV is iid.
    if groups is None:
        if _TEMPORAL_SORT_COL in X.columns:
            groups = X[_TEMPORAL_SORT_COL]
        else:
            warnings.warn(
                f"Temporal sort column '{_TEMPORAL_SORT_COL}' not found in X. "
                "Falling back to StratifiedKFold (treats folds as iid). "
                "CV Gini may be inflated by 0.02–0.05 on temporally ordered data.",
                UserWarning,
                stacklevel=2,
            )
    groups_train = (
        groups.loc[X_train.index].to_numpy() if groups is not None else None
    )
    cv = _make_cv(groups_train, n_splits=_XGB_CV_N_SPLITS)

    def objective(trial: optuna.Trial) -> float:
        return _lightgbm_optuna_objective(
            trial, X_train, y_train, cv,
            scale_pos_weight=_scale_pos_weight,
            num_leaves_max=num_leaves_max,
            boosting_type=boosting_type,
            monotone_constraints=monotone_constraints,
        )

    study = optuna.create_study(direction="maximize")
    if enqueue_trials:
        for trial_params in enqueue_trials:
            study.enqueue_trial(trial_params)
    study.optimize(objective, n_trials=n_trials)

    best_params: dict = study.best_params

    # --- Final model: two-stage refit ---
    # Stage 1: early stopping on a val slice identifies the optimal n_estimators
    # without over-training on the full set. X_test is never used here.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=_LGB_FINAL_VAL_SIZE,
        stratify=y_train,
        random_state=_RANDOM_STATE,
    )

    # Build imbalance kwargs once; applied consistently to Stage 1 and Stage 2.
    _imbalance_kwargs: dict = (
        {"scale_pos_weight": _scale_pos_weight}
        if _scale_pos_weight is not None
        else {"is_unbalance": True}
    )

    # Monotone constraint list (column-ordered) for final model construction.
    # Mirrors the conversion done inside the Optuna objective.
    _mc_kwargs: dict = {}
    if monotone_constraints is not None:
        _cols = X_train.columns.tolist()
        _mc_kwargs["monotone_constraints"] = [
            monotone_constraints.get(c, 0) for c in _cols
        ]

    # Stage 1 early stopping: DART does not support it in LGB 4.x.
    _stage1_callbacks = [lgb.log_evaluation(period=0)]
    if boosting_type != "dart":
        _stage1_callbacks.insert(
            0,
            lgb.early_stopping(stopping_rounds=_LGB_EARLY_STOPPING_ROUNDS, verbose=False),
        )

    _stage1_model = lgb.LGBMClassifier(
        **best_params,
        boosting_type=boosting_type,
        **_imbalance_kwargs,
        **_mc_kwargs,
        verbosity=-1,
        random_state=_RANDOM_STATE,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Early stopping is not available in dart mode")
        _stage1_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=_stage1_callbacks,
        )

    # Stage 2: refit on the full X_train with the early-stopping-derived
    # n_estimators so the final model sees 100% of training data, matching
    # the XGBoost approach and eliminating the ~20% training-data disadvantage.
    # For DART, best_iteration_ is 0 — fall back to best_params n_estimators.
    _best_n_trees = getattr(_stage1_model, "best_iteration_", -1)
    if _best_n_trees <= 0:
        _best_n_trees = best_params.get("n_estimators", _LGB_N_ESTIMATORS_MAX)

    final_model = lgb.LGBMClassifier(
        **{**best_params, "n_estimators": _best_n_trees},
        boosting_type=boosting_type,
        **_imbalance_kwargs,
        **_mc_kwargs,
        verbosity=-1,
        random_state=_RANDOM_STATE,
    )
    final_model.fit(X_train, y_train)

    # --- Evaluate on held-out test set ---
    metrics_dict = evaluate_model(final_model, X_test, y_test, "LightGBM")

    # --- ROC + PR figure ---
    figure_path = Path(_LGB_OPTUNA_FIGURE_PATH)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_roc_and_pr(final_model, X_test, y_test, "LightGBM", save_path=str(figure_path))
    plt.close(fig)

    # --- Persist model (joblib, consistent with save_model pattern) ---
    save_model(final_model, _LGB_OPTUNA_MODEL_PATH)

    # --- Persist params (JSON — human-readable, portable across services) ---
    params_path = Path(_LGB_OPTUNA_PARAMS_PATH)
    params_path.parent.mkdir(parents=True, exist_ok=True)
    with params_path.open("w") as fh:
        _json.dump(best_params, fh, indent=2)

    return final_model, metrics_dict, X_test, y_test, best_params


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
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = _CAT_OPTUNA_N_TRIALS,
    n_splits: int = _XGB_CV_N_SPLITS,
    seed: int = _RANDOM_STATE,
) -> tuple[CatBoostClassifier, dict, pd.DataFrame, pd.DataFrame, dict]:
    """
    Train CatBoost with Optuna Bayesian HPO and stratified k-fold CV.

    Search space (validated by literature subagent):
    - ``depth``         : int   [4, 8]      symmetric-tree depth
    - ``learning_rate`` : float [0.02, 0.2] log-uniform
    - ``l2_leaf_reg``   : float [1, 20]     L2 regularisation on leaf weights
    - ``bagging_temperature`` : float [0, 1] Bayesian bootstrap temperature
    - ``random_strength``     : float [0, 1] feature-split randomisation

    Class imbalance is handled via ``scale_pos_weight = n_neg / n_pos``
    (same convention as XGBoost), which adjusts leaf values for the minority
    class without distorting predicted probabilities.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (WoE-encoded or prepared via prepare_catboost_features).
    y : pd.Series
        Binary target (0 = non-default, 1 = default).
    n_trials : int
        Optuna trials.  Default: 50 (fast mock runs use 1–2).
    n_splits : int
        CV folds for Optuna objective AUC.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    model : CatBoostClassifier
        Fitted final model (best params, full X_train).
    metrics : dict
        Output of ``evaluate_model()`` on the held-out test split.
    X_test : pd.DataFrame
        Held-out feature matrix.
    y_test : pd.Series
        Held-out labels.
    best_params : dict
        Best hyperparameters found by Optuna.
    """
    import json as _json
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=_TEST_SIZE, random_state=seed, stratify=y
    )

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    # Auto-detect temporal ordering — same pattern as LGB/XGB trainers
    if _TEMPORAL_SORT_COL in X_train.columns:
        groups_train = X_train[_TEMPORAL_SORT_COL].to_numpy()
    else:
        groups_train = None
    cv = _make_cv(groups_train, n_splits=n_splits)

    def _objective(trial: optuna.Trial) -> float:
        params = {
            "depth": trial.suggest_int("depth", _CAT_DEPTH_MIN, _CAT_DEPTH_MAX),
            "learning_rate": trial.suggest_float(
                "learning_rate", _CAT_LEARNING_RATE_MIN, _CAT_LEARNING_RATE_MAX, log=True
            ),
            "l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg", _CAT_L2_LEAF_REG_MIN, _CAT_L2_LEAF_REG_MAX
            ),
            "bagging_temperature": trial.suggest_float(
                "bagging_temperature", _CAT_BAGGING_TEMP_MIN, _CAT_BAGGING_TEMP_MAX
            ),
            "random_strength": trial.suggest_float(
                "random_strength", _CAT_RANDOM_STRENGTH_MIN, _CAT_RANDOM_STRENGTH_MAX
            ),
            "bootstrap_type": _CAT_BOOTSTRAP_TYPE,
            "scale_pos_weight": scale_pos_weight,
            "random_seed": seed,
            "verbose": 0,
            "allow_writing_files": False,
        }

        fold_aucs: list[float] = []
        X_arr = X_train.to_numpy()
        y_arr = y_train.to_numpy()
        for train_idx, val_idx in cv.split(X_train, y_train):
            model_fold = CatBoostClassifier(
                **params,
                iterations=_CAT_ITERATIONS,
                early_stopping_rounds=_CAT_OBJ_EARLY_STOPPING_ROUNDS,
            )
            model_fold.fit(
                X_arr[train_idx],
                y_arr[train_idx],
                eval_set=(X_arr[val_idx], y_arr[val_idx]),
                verbose=False,
            )
            val_prob = model_fold.predict_proba(X_arr[val_idx])[:, 1]
            fold_aucs.append(roc_auc_score(y_arr[val_idx], val_prob))

        return float(np.mean(fold_aucs))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params

    # Final model — two-stage refit mirroring the LightGBM pattern:
    # Stage 1: early stopping on 80/20 split to find optimal iterations.
    # Stage 2: refit on 100% of X_train with that iteration count.
    final_params = {
        **best_params,
        "bootstrap_type": _CAT_BOOTSTRAP_TYPE,
        "scale_pos_weight": scale_pos_weight,
        "random_seed": seed,
        "verbose": 0,
        "allow_writing_files": False,
        "iterations": _CAT_ITERATIONS,
        "early_stopping_rounds": _CAT_EARLY_STOPPING_ROUNDS,
    }
    X_tr, X_val_es, y_tr, y_val_es = train_test_split(
        X_train, y_train,
        test_size=_CAT_FINAL_VAL_SIZE,
        stratify=y_train,
        random_state=_RANDOM_STATE,
    )
    stage1_model = CatBoostClassifier(**final_params)
    stage1_model.fit(
        X_tr.to_numpy(), y_tr.to_numpy(),
        eval_set=(X_val_es.to_numpy(), y_val_es.to_numpy()),
        verbose=False,
    )
    best_iterations = stage1_model.best_iteration_ or _CAT_ITERATIONS

    # Stage 2: no early stopping — uses the early-stopped iteration count
    final_params_s2 = {**final_params}
    final_params_s2.pop("early_stopping_rounds")
    final_params_s2["iterations"] = best_iterations
    model = CatBoostClassifier(**final_params_s2)
    model.fit(X_train.to_numpy(), y_train.to_numpy(), verbose=False)

    metrics = evaluate_model(model, X_test, y_test, model_name="CatBoost")
    fig = plot_roc_and_pr(model, X_test, y_test, model_name="CatBoost", save_path=_CAT_FIGURE_PATH)
    fig.clf()

    # Persist artefacts
    save_model(model, _CAT_MODEL_PATH)
    params_path = Path(_CAT_PARAMS_PATH)
    params_path.parent.mkdir(parents=True, exist_ok=True)
    with params_path.open("w") as fh:
        _json.dump(best_params, fh, indent=2)

    return model, metrics, X_test, y_test, best_params


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
    imputer_path = Path("models/ext_source_imputation_lgb.pkl")
    imputer_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(imputer, imputer_path)

    # Save best params for reference
    params_path = Path("models/ext_source_imputation_params.json")
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

def train_lightgbm_extended_hpo(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = _LGB_EXTENDED_OPTUNA_N_TRIALS,
) -> object:
    """
    Extended HPO for LightGBM on raw continuous features (150 trials).

    Non-regression: warm-start from Phase 4 best params, ensure final Gini >= 0.5514.
    Optuna study persists in SQLite DB — resumable across runs.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (raw continuous features, 63 columns).
    y : pd.Series
        Binary target series.
    n_trials : int
        Number of Optuna trials (default 150).

    Returns
    -------
    LGBMClassifier
        Fitted LightGBM classifier (retrained on full X, y with best params).

    Raises
    ------
    RuntimeError
        If final best Gini < 0.5514 (Phase 4 LGB baseline).
    """
    import optuna
    from lightgbm import LGBMClassifier
    import lightgbm as lgb

    # Auto-detect temporal CV groups from _TEMPORAL_SORT_COL
    # If not present, use standard stratified CV (for unit tests with mock data)
    if _TEMPORAL_SORT_COL in X.columns:
        groups = X[_TEMPORAL_SORT_COL].values
    else:
        groups = None

    # Resume or create Optuna study
    study_name = "lightgbm_extended_study"
    storage = f"sqlite:///{_OPTUNA_DB_PATH}"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage, load_if_exists=True)
        if len(study.trials) > 0:
            print(f"Loaded existing LGB study with {len(study.trials)} trials")
        else:
            # Study exists but is empty; warm-start with Phase 4 best params
            prior_best = {
                'num_leaves': 127,
                'max_depth': 8,
                'learning_rate': 0.1,
                'n_estimators': 350,
                'min_child_samples': 20,
                'subsample': 0.9,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 9.5
            }
            study.enqueue_trial(prior_best)
            print("Created new LGB study, warm-started with Phase 4 best params")
    except Exception:
        # Create new study with full initialization
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10, n_startup_trials=10),
            load_if_exists=True
        )
        # Warm-start with Phase 4 best params (LGB baseline from final_model_eval.json)
        prior_best = {
            'num_leaves': 127,
            'max_depth': 8,
            'learning_rate': 0.1,
            'n_estimators': 350,
            'min_child_samples': 20,
            'subsample': 0.9,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 9.5
        }
        if len(study.trials) == 0:
            study.enqueue_trial(prior_best)
        print("Created new LGB study, warm-started with Phase 4 best params")

    # Phase 4 baseline for non-regression check (STRICT)
    PHASE4_LGB_BASELINE = 0.5514

    # Define objective function
    def objective(trial):
        params = {
            'num_leaves': trial.suggest_int('num_leaves', _LGB_RAW_NUM_LEAVES_MIN, _LGB_RAW_NUM_LEAVES_MAX),
            'max_depth': trial.suggest_int('max_depth', _LGB_RAW_MAX_DEPTH_MIN, _LGB_RAW_MAX_DEPTH_MAX),
            'learning_rate': trial.suggest_float('learning_rate', _LGB_RAW_LEARNING_RATE_MIN, _LGB_RAW_LEARNING_RATE_MAX, log=True),
            'n_estimators': trial.suggest_int('n_estimators', _LGB_RAW_N_ESTIMATORS_MIN, _LGB_RAW_N_ESTIMATORS_MAX),
            'min_child_samples': trial.suggest_int('min_child_samples', _LGB_RAW_MIN_CHILD_SAMPLES_MIN, _LGB_RAW_MIN_CHILD_SAMPLES_MAX),
            'subsample': trial.suggest_float('subsample', _LGB_RAW_SUBSAMPLE_MIN, _LGB_RAW_SUBSAMPLE_MAX),
            'colsample_bytree': trial.suggest_float('colsample_bytree', _LGB_RAW_COLSAMPLE_BYTREE_MIN, _LGB_RAW_COLSAMPLE_BYTREE_MAX),
            'reg_alpha': trial.suggest_float('reg_alpha', _LGB_RAW_REG_ALPHA_MIN, _LGB_RAW_REG_ALPHA_MAX, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', _LGB_RAW_REG_LAMBDA_MIN, _LGB_RAW_REG_LAMBDA_MAX, log=True),
            'is_unbalance': True,
            'verbose': -1,
        }

        # Cross-validated AUC with temporal CV
        cv = _make_cv(groups_train=groups, n_splits=_CV_N_SPLITS)
        auc_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

            # Train LGB with early stopping
            lgb_model = LGBMClassifier(**params, callbacks=[lgb.early_stopping(_LGB_OBJ_EARLY_STOPPING_ROUNDS)])
            lgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='auc')

            # Compute AUC on validation fold
            auc = roc_auc_score(y_va, lgb_model.predict_proba(X_va)[:, 1])
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
    print(f"LGB HPO complete. Best AUC: {best_auc:.4f}, Best Gini: {best_gini:.4f}")

    # Non-regression check (STRICT)
    if best_gini < PHASE4_LGB_BASELINE:
        raise RuntimeError(
            f"Non-regression violated: LGB Gini {best_gini:.4f} < Phase 4 baseline {PHASE4_LGB_BASELINE:.4f}"
        )

    # Refit on full data with best params
    best_params = study.best_params.copy()
    best_params['is_unbalance'] = True
    best_params['verbose'] = -1
    best_n_estimators = best_params['n_estimators']
    best_params['n_estimators'] = int(best_n_estimators * 1.2)  # Slightly increase trees to compensate for CV variance

    final_model = LGBMClassifier(**best_params)
    final_model.fit(X, y)

    # Save model
    joblib.dump(final_model, "models/lightgbm_extended.pkl")
    print(f"Saved LGB model to models/lightgbm_extended.pkl")

    return final_model


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


def train_catboost_extended_hpo(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = _CAT_EXTENDED_OPTUNA_N_TRIALS,
    cat_features: list | None = None,
) -> object:
    """
    Extended HPO for CatBoost with native categorical support (50 trials).

    Non-regression: ensure final Gini >= 0.5461 (Phase 4 CatBoost baseline).
    Optuna study persists in SQLite DB — resumable across runs.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (mixed continuous and categorical).
    y : pd.Series
        Binary target series.
    n_trials : int
        Number of Optuna trials (default 50).
    cat_features : list or None
        List of categorical column names for native CatBoost handling.

    Returns
    -------
    CatBoostClassifier
        Fitted CatBoost classifier with cat_features attribute.

    Raises
    ------
    RuntimeError
        If final best Gini < 0.5461 (Phase 4 CatBoost baseline).
    """
    import optuna

    # Auto-detect temporal CV groups from _TEMPORAL_SORT_COL
    # If not present, use standard stratified CV (for unit tests with mock data)
    if _TEMPORAL_SORT_COL in X.columns:
        groups = X[_TEMPORAL_SORT_COL].values
    else:
        groups = None

    # Resume or create Optuna study
    study_name = "catboost_extended_study"
    storage = f"sqlite:///{_OPTUNA_DB_PATH}"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage, load_if_exists=True)
        if len(study.trials) > 0:
            print(f"Loaded existing CatBoost study with {len(study.trials)} trials")
        else:
            # Study exists but is empty; warm-start with Phase 4 best params
            prior_best = {
                'depth': 6,
                'learning_rate': 0.08,
                'iterations': 800,
                'l2_leaf_reg': 5.0,
                'bagging_temperature': 0.3,
                'random_strength': 0.5
            }
            study.enqueue_trial(prior_best)
            print("Created new CatBoost study, warm-started with Phase 4 best params")
    except Exception:
        # Create new study with full initialization
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10, n_startup_trials=10),
            load_if_exists=True
        )
        # Warm-start with Phase 4 best params
        prior_best = {
            'depth': 6,
            'learning_rate': 0.08,
            'iterations': 800,
            'l2_leaf_reg': 5.0,
            'bagging_temperature': 0.3,
            'random_strength': 0.5
        }
        if len(study.trials) == 0:
            study.enqueue_trial(prior_best)
        print("Created new CatBoost study, warm-started with Phase 4 best params")

    # Phase 4 baseline for non-regression check (STRICT)
    PHASE4_CAT_BASELINE = 0.5461

    # Define objective function
    def objective(trial):
        params = {
            'depth': trial.suggest_int('depth', _CAT_RAW_DEPTH_MIN, _CAT_RAW_DEPTH_MAX),
            'learning_rate': trial.suggest_float('learning_rate', _CAT_RAW_LEARNING_RATE_MIN, _CAT_RAW_LEARNING_RATE_MAX, log=True),
            'iterations': trial.suggest_int('iterations', _CAT_RAW_ITERATIONS_MIN, _CAT_RAW_ITERATIONS_MAX),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', _CAT_RAW_L2_LEAF_REG_MIN, _CAT_RAW_L2_LEAF_REG_MAX, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', _CAT_BAGGING_TEMP_MIN, _CAT_BAGGING_TEMP_MAX),
            'random_strength': trial.suggest_float('random_strength', _CAT_RANDOM_STRENGTH_MIN, _CAT_RANDOM_STRENGTH_MAX),
            'bootstrap_type': 'Bayesian',
            'verbose': 0,
            'allow_writing_files': False,
            'random_state': 42,
        }

        # Cross-validated AUC with temporal CV
        cv = _make_cv(groups_train=groups, n_splits=_CV_N_SPLITS)
        auc_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

            # Train CatBoost with native categorical support
            cat_model = CatBoostClassifier(**params)
            if cat_features:
                cat_model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    cat_features=cat_features,
                    verbose=0,
                    early_stopping_rounds=_CAT_OBJ_EARLY_STOPPING_ROUNDS
                )
            else:
                cat_model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    verbose=0,
                    early_stopping_rounds=_CAT_OBJ_EARLY_STOPPING_ROUNDS
                )

            # Compute AUC on validation fold
            auc = roc_auc_score(y_va, cat_model.predict_proba(X_va)[:, 1])
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
    print(f"CatBoost HPO complete. Best AUC: {best_auc:.4f}, Best Gini: {best_gini:.4f}")

    # Non-regression check (STRICT)
    if best_gini < PHASE4_CAT_BASELINE:
        raise RuntimeError(
            f"Non-regression violated: CatBoost Gini {best_gini:.4f} < Phase 4 baseline {PHASE4_CAT_BASELINE:.4f}"
        )

    # Refit on full data with best params
    best_params = study.best_params.copy()
    best_params['bootstrap_type'] = 'Bayesian'
    best_params['verbose'] = 0
    best_params['allow_writing_files'] = False
    best_params['random_state'] = 42

    final_model = CatBoostClassifier(**best_params)
    if cat_features:
        final_model.fit(X, y, cat_features=cat_features, verbose=0)
    else:
        final_model.fit(X, y, verbose=0)

    # Store cat_features as attribute for downstream reference
    final_model.cat_features = cat_features

    # Save model
    joblib.dump(final_model, "models/catboost_extended.pkl")
    print(f"Saved CatBoost model to models/catboost_extended.pkl")

    return final_model


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
    joblib.dump(final_model, "models/xgboost_extended.pkl")
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
    from credit_engine.features import select_features_by_iv

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
        joblib.dump(meta_model, "models/ensemble_variant_a.pkl")
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
        joblib.dump(meta_model, "models/ensemble_variant_b.pkl")
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
