"""
utils.py
--------
Shared evaluation metrics and plotting helpers.

Metrics
-------
- gini_coefficient   (= 2 * AUC - 1)
- log_loss
- brier_score
- ks_statistic

Plots
-----
- roc_curve_plot
- calibration_plot
- ks_plot
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def gini_coefficient(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Gini = 2 * AUC - 1."""
    auc = metrics.roc_auc_score(y_true, y_score)
    return 2 * auc - 1


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic between default and non-default score distributions."""
    # TODO: implement
    raise NotImplementedError


def roc_curve_plot(y_true: np.ndarray, y_score: np.ndarray, ax=None) -> plt.Axes:
    """Plot ROC curve with Gini annotation."""
    # TODO: implement
    raise NotImplementedError


def calibration_plot(
    y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10, ax=None
) -> plt.Axes:
    """Reliability diagram."""
    # TODO: implement
    raise NotImplementedError
