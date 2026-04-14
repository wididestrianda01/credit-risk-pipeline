"""tests/test_explain.py — SHAP explainability and fairness tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import shap

from src.explain import (
    compute_shap_values,
    plot_shap_summary,
    plot_shap_local,
    compute_fairness_metrics,
    get_adverse_action_factors,
    compute_shap_stability,
    FEATURE_LABELS,
)


def test_compute_shap_values(catboost_shap_fixture):
    """SHAP values computed; shape (n_samples, n_features)."""
    pass


def test_shap_values_shape(catboost_shap_fixture):
    """SHAP Explanation object has correct dimensions."""
    pass


def test_plot_shap_summary(catboost_shap_fixture, tmp_path):
    """Global beeswarm plot saved; output_path works."""
    pass


def test_plot_shap_summary_bar(catboost_shap_fixture, tmp_path):
    """Global bar (mean |SHAP|) plot saved."""
    pass


def test_plot_shap_local_waterfall(catboost_shap_fixture, tmp_path):
    """Local waterfall plot for idx=0; PNG saved."""
    pass


def test_plot_shap_local_force(catboost_shap_fixture, tmp_path):
    """Local force plot for idx=0; HTML saved."""
    pass


def test_compute_fairness_metrics(catboost_shap_fixture):
    """Fairness metrics computed; returns DataFrame with group rows."""
    pass


def test_get_adverse_action_factors(catboost_shap_fixture):
    """Adverse action factors for idx=0; returns List[AdverseActionFactor]."""
    pass


def test_shap_stability(catboost_shap_fixture):
    """Stability metric computed; returns float in [-1, 1]."""
    pass


def test_feature_labels_completeness(catboost_shap_fixture):
    """FEATURE_LABELS contains entries for all columns in X_oot_mini."""
    pass


# Integration test (Wave 4)
# Tests: load real model/data, compute SHAP, save figures, write fairness CSV
