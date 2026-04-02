"""
utils.py
--------
Shared evaluation metrics and plotting helpers for credit risk models.

Metrics
-------
- gini_coefficient   (= 2 * AUC - 1)
- ks_statistic       (max CDF separation between default / non-default)
- evaluate_model     (full metric suite including Brier Skill Score)

Plots
-----
- plot_roc_and_pr    (ROC + Precision-Recall two-panel figure)
- roc_curve_plot     (stub — reserved for future task)
- calibration_plot   (stub — reserved for future task)

Industry benchmarks (Basel III IRB credit scoring)
---------------------------------------------------
KS > 0.30  : good separation
KS > 0.40  : strong separation
Gini > 0.60: good discriminatory power
Gini > 0.75: target for this pipeline
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_KS_GOOD: float = 0.30    # KS ≥ this: good separation (credit scoring standard)
_KS_STRONG: float = 0.40  # KS ≥ this: strong separation (Basel III benchmark)
_PLOT_FIGSIZE: tuple[int, int] = (12, 5)
_PLOT_DPI: int = 300
_PREVALENCE_LABEL: str = "Baseline (prevalence)"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def gini_coefficient(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Gini coefficient for binary classification.

    Defined as Gini = 2 × AUC − 1. Ranges from −1 (perfectly inverted
    predictions) through 0 (no discrimination) to 1 (perfect separation).
    The primary regulatory metric in Basel III IRB credit models.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth labels (0 = non-default, 1 = default).
    y_prob : np.ndarray
        Predicted default probabilities in [0, 1].

    Returns
    -------
    float
        Gini coefficient in [−1, 1].

    Raises
    ------
    ValueError
        If ``y_true`` contains only one class (sklearn propagates this).

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_prob = np.array([0.1, 0.2, 0.7, 0.8])
    >>> gini_coefficient(y_true, y_prob)
    1.0
    """
    n_classes = len(np.unique(y_true))
    if n_classes < 2:
        raise ValueError(
            f"gini_coefficient requires at least 2 classes in y_true, found {n_classes}."
        )
    auc = roc_auc_score(y_true, y_prob)
    return float(2 * auc - 1)


def ks_statistic(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[float, float]:
    """
    Kolmogorov-Smirnov statistic for credit scoring.

    Measures the maximum vertical distance between the empirical CDF of
    default scores and the empirical CDF of non-default scores. A key
    deliverable in Basel III model validation and scorecard development.

    The KS statistic is computed via ``scipy.stats.ks_2samp`` (handles ties
    correctly). The threshold at which this maximum gap occurs is recovered
    by scanning unique observed probabilities.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth labels (0 = non-default, 1 = default).
    y_prob : np.ndarray
        Predicted default probabilities in [0, 1].

    Returns
    -------
    ks_value : float
        KS statistic in [0, 1]. Values > 0.30 are considered good;
        values > 0.40 are strong in credit scoring contexts.
    threshold_at_ks : float
        The probability cut-off where CDF separation is maximised.
        Useful as a natural risk-tier boundary for scorecard cut-off policy.

    Raises
    ------
    ValueError
        If ``y_true`` contains no positive samples or no negative samples.

    Notes
    -----
    The threshold scan uses the empirical CDF definition:
        CDF_pos(t) = P(score ≤ t | default)
        CDF_neg(t) = P(score ≤ t | non-default)
        threshold_at_ks = argmax_t |CDF_pos(t) − CDF_neg(t)|

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_prob = np.array([0.1, 0.2, 0.7, 0.8])
    >>> ks, thresh = ks_statistic(y_true, y_prob)
    >>> ks
    1.0
    """
    probs_pos = y_prob[y_true == 1]
    probs_neg = y_prob[y_true == 0]

    if len(probs_pos) == 0:
        raise ValueError(
            f"ks_statistic: no positive samples in y_true "
            f"(n_neg={len(probs_neg)}, n_pos=0). Both classes required."
        )
    if len(probs_neg) == 0:
        raise ValueError(
            f"ks_statistic: no negative samples in y_true "
            f"(n_pos={len(probs_pos)}, n_neg=0). Both classes required."
        )

    ks_val, _ = ks_2samp(probs_pos, probs_neg)

    # Recover the threshold: scan all unique observed probabilities
    thresholds = np.unique(y_prob)
    cdf_pos = (probs_pos[:, None] <= thresholds).mean(axis=0)
    cdf_neg = (probs_neg[:, None] <= thresholds).mean(axis=0)
    gaps = np.abs(cdf_pos - cdf_neg)
    threshold_at_ks = float(thresholds[np.argmax(gaps)])

    return float(ks_val), threshold_at_ks


def evaluate_model(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "model",
) -> dict:
    """
    Compute the full evaluation metric suite for a fitted classifier.

    Metrics returned
    ----------------
    - AUC-ROC      : Area under the ROC curve (standard discrimination metric)
    - Gini         : 2 × AUC − 1 (primary regulatory metric)
    - KS           : KS statistic (max CDF separation)
    - Brier        : Raw Brier score (mean squared probability error)
    - BrierSkill   : Brier Skill Score = 1 − Brier / Brier_baseline,
                     where Brier_baseline = prevalence × (1 − prevalence).
                     Preferred over raw Brier for imbalanced datasets because
                     the naive "predict always the mean" baseline is small (~0.07)
                     at 8% prevalence, making raw Brier misleadingly optimistic.
    - AvgPrecision : Area under the Precision-Recall curve (most informative
                     metric under severe class imbalance).

    Parameters
    ----------
    model : object
        Fitted sklearn-compatible estimator with a ``predict_proba`` method.
    X_test : np.ndarray
        Feature matrix for the test split.
    y_test : np.ndarray
        Binary ground-truth labels for the test split.
    model_name : str, optional
        Label stored in the ``Model`` field of the returned dict.

    Returns
    -------
    dict
        Keys: ``Model``, ``AUC-ROC``, ``Gini``, ``KS``, ``Brier``,
        ``BrierSkill``, ``AvgPrecision``.

    Notes
    -----
    Metric priority for credit risk model selection (8% default prevalence):
    1. Gini      — primary regulatory capital adequacy metric
    2. KS        — standard scorecard deliverable, Basel III model validation
    3. AvgPrecision — most informative for imbalanced data, drives threshold choice
    4. BrierSkill   — calibration quality (normalised by prevalence baseline)
    5. AUC-ROC   — reported for completeness
    """
    y_prob = model.predict_proba(X_test)[:, 1]

    auc_roc = float(roc_auc_score(y_test, y_prob))
    gini = gini_coefficient(y_test, y_prob)
    ks, _ = ks_statistic(y_test, y_prob)
    brier = float(brier_score_loss(y_test, y_prob))
    avg_prec = float(average_precision_score(y_test, y_prob))

    prevalence = float(np.mean(y_test))
    brier_baseline = prevalence * (1.0 - prevalence)
    brier_skill = float(1.0 - brier / brier_baseline) if brier_baseline > 0 else float("nan")

    result = {
        "Model": model_name,
        "AUC-ROC": auc_roc,
        "Gini": gini,
        "KS": ks,
        "Brier": brier,
        "BrierSkill": brier_skill,
        "AvgPrecision": avg_prec,
    }

    import pandas as pd  # local import — pandas is available but not a module-level dep
    print(pd.Series({k: v for k, v in result.items() if k != "Model"}, name=model_name).to_string())

    return result


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_roc_and_pr(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "model",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Two-panel ROC + Precision-Recall figure for a fitted classifier.

    Left panel: ROC curve with Gini annotation and diagonal reference.
    Right panel: PR curve with Average Precision annotation and a dashed
    horizontal baseline at the dataset prevalence (fraction of positives).
    The prevalence baseline is the PR curve of a "predict always positive"
    model — any curve above it indicates positive skill.

    Parameters
    ----------
    model : object
        Fitted sklearn-compatible estimator with a ``predict_proba`` method.
    X_test : np.ndarray
        Feature matrix for the test split.
    y_test : np.ndarray
        Binary ground-truth labels for the test split.
    model_name : str, optional
        Used in the figure suptitle.
    save_path : str or Path or None, optional
        If provided, saves the figure to this path at 300 DPI.
        The directory must already exist.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object. Call ``plt.close(fig)`` when done to free memory.

    Notes
    -----
    ``plt.show()`` is never called — the figure is returned for the caller
    to display or save as needed. This keeps the function safe in headless
    (CI / notebook-execution) environments.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    prevalence = float(np.mean(y_test))

    gini = gini_coefficient(y_test, y_prob)
    avg_prec = float(average_precision_score(y_test, y_prob))

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=_PLOT_FIGSIZE)
    fig.suptitle(model_name, fontsize=13, fontweight="bold")

    # --- ROC curve ---
    ax_roc.plot(fpr, tpr, lw=2, label=f"Gini = {gini:.4f}")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="grey", lw=1, label="Random")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend(loc="lower right")
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])

    # --- PR curve ---
    ax_pr.plot(recall, precision, lw=2, label=f"AP = {avg_prec:.4f}")
    ax_pr.axhline(
        y=prevalence,
        linestyle="--",
        color="grey",
        lw=1,
        label=f"{_PREVALENCE_LABEL} ({prevalence:.3f})",
    )
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve")
    ax_pr.legend(loc="upper right")
    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.05])

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=_PLOT_DPI, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Stubs (reserved for future tasks)
# ---------------------------------------------------------------------------

def roc_curve_plot(y_true: np.ndarray, y_score: np.ndarray, ax=None) -> plt.Axes:
    """Plot ROC curve with Gini annotation. TODO: implement in future task."""
    raise NotImplementedError


def calibration_plot(
    y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10, ax=None
) -> plt.Axes:
    """Reliability diagram. TODO: implement in future task."""
    raise NotImplementedError
