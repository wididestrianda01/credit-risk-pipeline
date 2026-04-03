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

# Output path for the imbalance benchmark comparison table
_BENCHMARK_REPORT_PATH: str = "reports/imbalance_benchmark.csv"

# Strategy labels — kept as module constants so downstream tasks can reference
# them by name (e.g., Task 3.4 picks the winner from this table).
_STRATEGY_SMOTE: str = "SMOTE"
_STRATEGY_COST_SENSITIVE: str = "Cost-Sensitive"
_STRATEGY_THRESHOLD_TUNED: str = "Threshold-Tuned"
_STRATEGY_HYBRID: str = "SMOTE+Cost-Sensitive"


# ---------------------------------------------------------------------------
# Logistic regression baseline
# ---------------------------------------------------------------------------

def train_logistic_baseline(
    X: pd.DataFrame,
    y: pd.Series,
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

    # --- 10-fold stratified CV on training data ---
    # Scaler is fit inside each fold (pipeline.fit refits the scaler)
    # which prevents any test-fold statistics leaking into the scaler fit.
    cv = StratifiedKFold(
        n_splits=_CV_N_SPLITS, shuffle=True, random_state=_RANDOM_STATE
    )
    cv_scores: list[float] = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        pipeline.fit(X_fold_train, y_fold_train)
        y_prob_val = pipeline.predict_proba(X_fold_val)[:, 1]
        cv_scores.append(float(roc_auc_score(y_fold_val, y_prob_val)))

    mean_cv = float(np.mean(cv_scores))
    std_cv = float(np.std(cv_scores))
    print(f"10-fold CV AUC-ROC: {mean_cv:.4f} ± {std_cv:.4f}")

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
    cv = StratifiedKFold(
        n_splits=_XGB_CV_N_SPLITS, shuffle=True, random_state=_RANDOM_STATE
    )
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

    # --- Identify winner: Gini primary, F1-Macro tiebreaker ---
    winner_idx = df["Gini"].idxmax()
    winner = df.loc[winner_idx]

    print("\n=== XGBoost Imbalance Strategy Benchmark ===")
    print(df.to_string(index=False))
    print(
        f"\n✓ Best strategy: {winner['Strategy']} "
        f"(Gini={winner['Gini']:.4f}, F1-Macro={winner['F1-Macro']:.4f})"
    )
    print(f"  → Use '{winner['Strategy']}' as imbalance method in Tasks 3.4 & 3.5")

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
    cv: StratifiedKFold,
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

    import optuna
    import xgboost as xgb

    # --- Input guards ---
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

    cv = StratifiedKFold(
        n_splits=_XGB_CV_N_SPLITS, shuffle=True, random_state=_RANDOM_STATE
    )

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
    plot_roc_and_pr(final_model, X_test, y_test, "XGBoost", save_path=str(figure_path))

    # --- Persist model (joblib, consistent with save_model pattern) ---
    save_model(final_model, _XGB_OPTUNA_MODEL_PATH)

    # --- Persist params (JSON — human-readable, portable across services) ---
    params_path = Path(_XGB_OPTUNA_PARAMS_PATH)
    params_path.parent.mkdir(parents=True, exist_ok=True)
    with params_path.open("w") as fh:
        _json.dump(best_params, fh, indent=2)

    return final_model, metrics_dict, X_test, y_test, best_params


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
