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
    model, _, _, X, y = catboost_shap_fixture

    # Add CODE_GENDER and AGE_YEARS columns for fairness analysis
    # (model.predict_proba will use only the features it was trained on)
    X_fair = X.copy()
    X_fair["CODE_GENDER"] = np.random.choice(["M", "F"], size=len(X), p=[0.5, 0.5])
    X_fair["AGE_YEARS"] = np.random.uniform(20, 70, size=len(X))

    # Compute fairness metrics (function extracts protected cols, passes original features to predict_proba)
    df_fair = compute_fairness_metrics(model, X_fair, y, ["CODE_GENDER", "AGE_YEARS"])

    assert isinstance(df_fair, pd.DataFrame)
    assert "group_name" in df_fair.columns
    assert "demographic_parity" in df_fair.columns
    assert "tpr" in df_fair.columns
    assert "fpr" in df_fair.columns
    assert len(df_fair) > 0


def test_get_adverse_action_factors(catboost_shap_fixture):
    """Adverse action factors for idx=0; returns List[AdverseActionFactor]."""
    model, _, shap_vals, X, y = catboost_shap_fixture

    factors = get_adverse_action_factors(shap_vals, idx=0, feature_labels=FEATURE_LABELS, top_n=5)

    assert isinstance(factors, list)
    assert len(factors) <= 5
    assert all(isinstance(f, dict) for f in factors)
    assert all("feature_name" in f and "human_label" in f for f in factors)
    assert all(f["direction"] in ("increases_risk", "decreases_risk") for f in factors)
    assert all(1 <= f["rank"] <= 5 for f in factors)


def test_shap_stability(catboost_shap_fixture):
    """Stability metric computed; returns float in [-1, 1]."""
    model, _, shap_vals, X, y = catboost_shap_fixture

    # Split data: first 100 as "train", last 100 as "OOT"
    shap_train = shap_vals[:100]
    shap_oot = shap_vals[100:]

    stability = compute_shap_stability(shap_train, shap_oot)

    assert isinstance(stability, float)
    assert -1 <= stability <= 1


def test_feature_labels_completeness(catboost_shap_fixture):
    """FEATURE_LABELS contains entries for all columns in X_cat_v2.parquet.

    Hard gate: raises AssertionError (not skip) if real data is missing labels.
    Ensures GDPR Art. 22 compliance with no raw column names in output.
    """
    from pathlib import Path

    # Verify FEATURE_LABELS is a valid dict
    assert isinstance(FEATURE_LABELS, dict), "FEATURE_LABELS must be a dict"
    assert all(isinstance(k, str) for k in FEATURE_LABELS.keys()), "All keys must be strings"
    assert all(isinstance(v, str) for v in FEATURE_LABELS.values()), "All values must be strings"

    # Attempt to load real X_cat_v2 to verify completeness (hard gate)
    X_cat_v2_path = Path("data/processed/X_cat_v2.parquet")

    if X_cat_v2_path.exists():
        X_real = pd.read_parquet(X_cat_v2_path)
        missing_real = [col for col in X_real.columns if col not in FEATURE_LABELS]

        assert not missing_real, (
            f"GATE FAIL: {len(missing_real)} columns missing from FEATURE_LABELS: {missing_real}"
        )


# Integration test (Wave 4)
# Tests: load real model/data, compute SHAP, save figures, write fairness CSV
