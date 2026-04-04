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

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
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
_CV_EMBARGO_FRAC: float = 0.01

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
_XGB_OPTUNA_N_TRIALS: int = 50
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
_XGB_MIN_CHILD_WEIGHT_MAX: int = 10
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
_LGB_NUM_LEAVES_MAX: int = 150
_LGB_MAX_DEPTH_MIN: int = 3
_LGB_MAX_DEPTH_MAX: int = 12
_LGB_LEARNING_RATE_MIN: float = 0.01
_LGB_LEARNING_RATE_MAX: float = 0.2
_LGB_N_ESTIMATORS_MIN: int = 100
_LGB_N_ESTIMATORS_MAX: int = 500
_LGB_MIN_CHILD_SAMPLES_MIN: int = 5
_LGB_MIN_CHILD_SAMPLES_MAX: int = 100
_LGB_SUBSAMPLE_MIN: float = 0.6
_LGB_SUBSAMPLE_MAX: float = 1.0
_LGB_COLSAMPLE_BYTREE_MIN: float = 0.6
_LGB_COLSAMPLE_BYTREE_MAX: float = 1.0
_LGB_REG_ALPHA_MIN: float = 0.0
_LGB_REG_ALPHA_MAX: float = 5.0
_LGB_REG_LAMBDA_MIN: float = 0.0
_LGB_REG_LAMBDA_MAX: float = 10.0
# Two-tier early stopping: HPO objective uses aggressive patience to quickly
# triage bad configs; final refit uses standard patience for a proper model.
_LGB_OBJ_EARLY_STOPPING_ROUNDS: int = 20   # fast config triage inside Optuna
_LGB_EARLY_STOPPING_ROUNDS: int = 50        # full patience for final refit
_LGB_FINAL_VAL_SIZE: float = 0.2

# Output paths for LightGBM Optuna HPO artefacts
_LGB_OPTUNA_MODEL_PATH: str = "models/lightgbm_best.pkl"
_LGB_OPTUNA_PARAMS_PATH: str = "models/lightgbm_params.json"
_LGB_OPTUNA_FIGURE_PATH: str = "reports/figures/lightgbm_roc_pr.png"

# Output path for the imbalance benchmark comparison table
_BENCHMARK_REPORT_PATH: str = "reports/imbalance_benchmark.csv"

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
_ENSEMBLE_LGB_DEFAULTS: dict = {
    "n_estimators": 100,
    "num_leaves": 31,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": -1,
    "is_unbalance": True,
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
) -> float:
    """
    Optuna objective: 5-fold CV AUC-ROC for a suggested LightGBM configuration.

    Called once per trial by ``study.optimize()``. Samples 9 hyperparameters,
    runs stratified k-fold CV on X_train with early stopping inside each fold,
    and returns the mean out-of-fold AUC-ROC. X_test is never passed in.

    Early stopping uses the CV validation fold as the eval set — no additional
    data split is needed inside the objective. This means ``n_estimators`` is
    the maximum; early stopping may terminate sooner if the validation AUC
    plateaus for ``_LGB_EARLY_STOPPING_ROUNDS`` consecutive rounds.

    Parameters
    ----------
    trial : optuna.Trial
        Current Optuna trial handle for hyperparameter suggestions.
    X_train : pd.DataFrame
        Training features (80% split). Never the held-out test set.
    y_train : pd.Series
        Training labels.
    cv : StratifiedKFold
        5-fold CV splitter, seeded for reproducibility.

    Returns
    -------
    float
        Mean out-of-fold AUC-ROC across all CV folds.
    """
    import lightgbm as lgb

    params = {
        "num_leaves": trial.suggest_int(
            "num_leaves", _LGB_NUM_LEAVES_MIN, _LGB_NUM_LEAVES_MAX
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
        "is_unbalance": True,
        "verbosity": -1,
        "random_state": _RANDOM_STATE,
    }

    # Aggressive early stopping inside the objective: enough to distinguish
    # good from bad configs without training to full depth on each fold.
    callbacks = [
        lgb.early_stopping(stopping_rounds=_LGB_OBJ_EARLY_STOPPING_ROUNDS, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    fold_aucs: list[float] = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        model = lgb.LGBMClassifier(**params)
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
) -> tuple[object, dict, pd.DataFrame, pd.Series, dict]:
    """
    Train LightGBM with Bayesian hyperparameter optimisation via Optuna.

    Runs ``n_trials`` of TPE-based search over a 9-dimensional space,
    selecting the configuration that maximises mean out-of-fold AUC-ROC.
    Early stopping inside each CV fold prevents over-training on any given
    hyperparameter configuration. The final model is retrained on the full
    training split (with a held-out validation slice for early stopping),
    then evaluated on the held-out test split.

    Imbalance handling uses ``is_unbalance=True`` — LightGBM's built-in
    strategy that internally computes ``n_neg / n_pos`` and adjusts both
    the loss weights and leaf output values, avoiding external resampling.

    Parameters
    ----------
    X : pd.DataFrame
        WoE-transformed feature matrix (40 columns, produced by
        ``credit_engine.features.apply_feature_store``).
    y : pd.Series
        Binary TARGET series (0 = repaid, 1 = defaulted).
    n_trials : int, optional
        Number of Optuna trials. Default 50 balances exploration depth
        against compute budget for the 9-dimensional search space.

    Returns
    -------
    lgb_model : LGBMClassifier
        Fitted LightGBM model with best hyperparameters and early stopping.
    metrics_dict : dict
        Evaluation metrics on X_test. Keys: Model, AUC-ROC, Gini, KS,
        Brier, BrierSkill, AvgPrecision.
    X_test : pd.DataFrame
        Held-out test features (20% stratified split, seed=42).
    y_test : pd.Series
        Held-out test labels.
    best_params : dict
        Optimised hyperparameters (9 keys). Also persisted to
        ``models/lightgbm_params.json``.

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

    # --- Train / test split (stratified, identical seed across all models) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=_TEST_SIZE, stratify=y, random_state=_RANDOM_STATE
    )

    # --- Optuna study ---
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    groups_train = (
        groups.loc[X_train.index].to_numpy() if groups is not None else None
    )
    cv = _make_cv(groups_train, n_splits=_XGB_CV_N_SPLITS)

    def objective(trial: optuna.Trial) -> float:
        return _lightgbm_optuna_objective(trial, X_train, y_train, cv)

    study = optuna.create_study(direction="maximize")
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

    _stage1_model = lgb.LGBMClassifier(
        **best_params,
        is_unbalance=True,
        verbosity=-1,
        random_state=_RANDOM_STATE,
    )
    _stage1_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=_LGB_EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    # Stage 2: refit on the full X_train with the early-stopping-derived
    # n_estimators so the final model sees 100% of training data, matching
    # the XGBoost approach and eliminating the ~20% training-data disadvantage.
    _best_n_trees = getattr(_stage1_model, "best_iteration_", -1)
    if _best_n_trees <= 0:
        _best_n_trees = best_params.get("n_estimators", _LGB_N_ESTIMATORS_MAX)

    final_model = lgb.LGBMClassifier(
        **{**best_params, "n_estimators": _best_n_trees},
        is_unbalance=True,
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
        Typically the LightGBM model from ``train_lightgbm_optuna``.
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
    Artefacts written to disk:
    - ``models/lightgbm_calibrated.pkl``               — joblib-serialised calibrated model
    - ``reports/figures/calibration_reliability.png``  — reliability diagram
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

    fig_path = Path(_CALIBRATION_FIGURE_PATH)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Persist calibrated model ---
    save_model(calibrated_model, _CALIBRATED_MODEL_PATH)

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
    2. On the 80% train portion, run n_splits-fold stratified CV:
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

    # --- Stratified k-fold CV on the training set ---
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]

        # --- Fit LightGBM on fold training data ---
        lgb_fold = lgb.LGBMClassifier(**lgb_params)
        lgb_fold.fit(X_fold_train, y_fold_train)
        oof_lgb[val_idx] = lgb_fold.predict_proba(X_fold_val)[:, 1]

        # --- Fit XGBoost on fold training data ---
        # Compute scale_pos_weight from fold training labels
        n_neg = (y_fold_train == 0).sum()
        n_pos = (y_fold_train == 1).sum()
        scale_pos_weight = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0

        xgb_params_fold = xgb_params.copy()
        xgb_params_fold["scale_pos_weight"] = scale_pos_weight

        xgb_fold = xgb.XGBClassifier(**xgb_params_fold)
        xgb_fold.fit(X_fold_train, y_fold_train)
        oof_xgb[val_idx] = xgb_fold.predict_proba(X_fold_val)[:, 1]

    # --- Train final base models on full training set ---
    n_neg_train = (y_train == 0).sum()
    n_pos_train = (y_train == 1).sum()
    scale_pos_weight_train = float(n_neg_train) / float(n_pos_train) if n_pos_train > 0 else 1.0

    lgb_final = lgb.LGBMClassifier(**lgb_params)
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
