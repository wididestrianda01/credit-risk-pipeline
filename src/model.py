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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_engine.utils import evaluate_model, plot_roc_and_pr


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
