"""
explain.py
----------
SHAP-based explainability and fairness analysis.

Key outputs
-----------
- Global feature importance (beeswarm / bar)
- Local explanations (waterfall / force plots)
- Fairness metrics by sensitive attribute (demographic parity, equalised odds)
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
import pandas as pd


class AdverseActionFactor(TypedDict):
    """Adverse action factor for regulatory compliance (GDPR Art. 22)."""

    feature_name: str  # internal column name (e.g. "EXT_SOURCE_2")
    human_label: str  # mapped from FEATURE_LABELS dict
    shap_value: float  # signed SHAP value for this applicant
    direction: str  # "increases_risk" or "decreases_risk"
    rank: int  # 1 = most influential factor


# Module-level constant: maps internal column names to human-readable descriptions
# CRITICAL: must cover all columns in data/processed/X_cat_v2.parquet for GDPR Art. 22
# See D-06, D-09
FEATURE_LABELS: dict[str, str] = {}


def compute_shap_values(model: object, X: pd.DataFrame) -> Any:
    """Return a SHAP Explanation object for the given model and data."""
    # TODO: implement
    raise NotImplementedError


def plot_shap_summary(
    shap_explanation: Any, X: pd.DataFrame, plot_type: str = "dot", save_path: str | None = None
) -> None:
    """Plot global SHAP summary (beeswarm or bar)."""
    # TODO: implement
    raise NotImplementedError


def plot_shap_local(
    shap_explanation: Any,
    idx: int,
    X: pd.DataFrame,
    plot_type: str = "waterfall",
    save_path: str | None = None,
) -> None:
    """Plot local SHAP explanation (waterfall or force)."""
    # TODO: implement
    raise NotImplementedError


def compute_fairness_metrics(
    model: object, X: pd.DataFrame, y: pd.Series, sensitive_cols: list[str]
) -> pd.DataFrame:
    """Compute group-level fairness metrics by sensitive attribute."""
    # TODO: implement
    raise NotImplementedError


def get_adverse_action_factors(
    shap_explanation: Any, idx: int, feature_labels: dict[str, str], top_n: int = 5
) -> list[AdverseActionFactor]:
    """Get top-N risk-increasing factors for a single applicant (GDPR Art. 22)."""
    # TODO: implement
    raise NotImplementedError


def compute_shap_stability(shap_train: np.ndarray, shap_oot: np.ndarray) -> float:
    """
    Spearman correlation of mean(|SHAP|) feature rankings between train and OOT.

    Returns correlation coefficient in [-1, 1]. Value >= 0.90 is considered stable.
    """
    # TODO: implement
    raise NotImplementedError


def fairness_report(
    y_true: pd.Series,
    y_pred: pd.Series,
    sensitive_col: pd.Series,
) -> pd.DataFrame:
    """Compute group-level fairness metrics."""
    # TODO: implement
    raise NotImplementedError
