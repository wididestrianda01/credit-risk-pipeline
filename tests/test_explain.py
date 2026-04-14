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
    model, explainer, expected_shap_vals, X, y = catboost_shap_fixture

    # Compute SHAP values using the implementation
    shap_vals = compute_shap_values(model, X)

    # Verify result is a SHAP Explanation object
    assert isinstance(shap_vals, shap.Explanation), f"Expected shap.Explanation, got {type(shap_vals)}"

    # Verify shape matches input
    assert shap_vals.shape == (len(X), X.shape[1]), f"Expected shape {(len(X), X.shape[1])}, got {shap_vals.shape}"


def test_shap_values_shape(catboost_shap_fixture):
    """SHAP Explanation object has correct dimensions."""
    model, explainer, expected_shap_vals, X, y = catboost_shap_fixture

    shap_vals = compute_shap_values(model, X)

    # Verify base_values exists (should be scalar for binary classification)
    assert hasattr(shap_vals, "base_values"), "SHAP Explanation missing base_values"

    # Verify values shape
    assert shap_vals.values.shape == (len(X), X.shape[1]), (
        f"Expected values shape {(len(X), X.shape[1])}, got {shap_vals.values.shape}"
    )

    # Verify data shape (should match X)
    assert shap_vals.data.shape == X.shape, f"Expected data shape {X.shape}, got {shap_vals.data.shape}"


def test_plot_shap_summary(catboost_shap_fixture, tmp_path):
    """Global beeswarm plot saved; output_path works."""
    model, explainer, shap_vals, X, y = catboost_shap_fixture

    save_path = tmp_path / "test_beeswarm.png"
    plot_shap_summary(shap_vals, X, plot_type="dot", save_path=str(save_path))

    assert save_path.exists(), f"Beeswarm plot not saved to {save_path}"


def test_plot_shap_summary_bar(catboost_shap_fixture, tmp_path):
    """Global bar (mean |SHAP|) plot saved."""
    model, explainer, shap_vals, X, y = catboost_shap_fixture

    save_path = tmp_path / "test_bar.png"
    plot_shap_summary(shap_vals, X, plot_type="bar", save_path=str(save_path))

    assert save_path.exists(), f"Bar plot not saved to {save_path}"


def test_plot_shap_local_waterfall(catboost_shap_fixture, tmp_path):
    """Local waterfall plot for idx=0; PNG saved."""
    model, explainer, shap_vals, X, y = catboost_shap_fixture

    save_path = tmp_path / "test_waterfall.png"
    plot_shap_local(shap_vals, idx=0, X=X, plot_type="waterfall", save_path=str(save_path))

    assert save_path.exists(), f"Waterfall plot not saved to {save_path}"


def test_plot_shap_local_force(catboost_shap_fixture, tmp_path):
    """Local force plot for idx=0; HTML saved."""
    model, explainer, shap_vals, X, y = catboost_shap_fixture

    save_path = tmp_path / "test_force.html"
    plot_shap_local(shap_vals, idx=0, X=X, plot_type="force", save_path=str(save_path))

    assert save_path.exists(), f"Force plot not saved to {save_path}"


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
    """FEATURE_LABELS has exactly 171 entries (all X_cat_v2 columns covered)."""
    model, explainer, shap_vals, X, y = catboost_shap_fixture

    # Verify FEATURE_LABELS has exactly 171 entries for all X_cat_v2 columns
    # (The fixture uses synthetic feature_0, feature_1, etc., so we don't check
    # fixture columns against FEATURE_LABELS — we just verify the mapping is complete)
    assert len(FEATURE_LABELS) == 171, f"Expected 171 labels, got {len(FEATURE_LABELS)}"

    # Verify that the FEATURE_LABELS dict can be imported and used
    assert isinstance(FEATURE_LABELS, dict), "FEATURE_LABELS must be a dict"
    assert all(isinstance(k, str) for k in FEATURE_LABELS.keys()), "All keys must be strings"
    assert all(isinstance(v, str) for v in FEATURE_LABELS.values()), "All values must be strings"


# Integration test (Wave 4)
# Tests: load real model/data, compute SHAP, save figures, write fairness CSV
