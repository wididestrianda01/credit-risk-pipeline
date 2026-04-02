"""
test_utils.py
-------------
Unit tests for credit_engine/utils.py.

Run with
--------
    pytest tests/test_utils.py -v
"""

import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt  # noqa: E402

from credit_engine.utils import (
    evaluate_model,
    gini_coefficient,
    ks_statistic,
    plot_roc_and_pr,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def toy_data() -> tuple[np.ndarray, np.ndarray]:
    """Exact array from the task done-condition. Well-separated, 5 pos / 5 neg."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.7, 0.8, 0.3, 0.9, 0.15, 0.25, 0.6, 0.85])
    return y_true, y_prob


@pytest.fixture
def perfect_data() -> tuple[np.ndarray, np.ndarray]:
    """Perfect separation: all positives score 1.0, negatives 0.0."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    return y_true, y_prob


@pytest.fixture
def random_data() -> tuple[np.ndarray, np.ndarray]:
    """No discrimination: constant probability = prevalence."""
    rng = np.random.default_rng(42)
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_prob = np.full(10, 0.5)
    return y_true, y_prob


@pytest.fixture
def inverse_data() -> tuple[np.ndarray, np.ndarray]:
    """Inverted predictions: high score → non-default."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.9, 0.8, 0.1, 0.2])
    return y_true, y_prob


@pytest.fixture
def imbalanced_data() -> tuple[np.ndarray, np.ndarray]:
    """8% default rate mirroring real dataset distribution."""
    rng = np.random.default_rng(0)
    n = 200
    y_true = (rng.random(n) < 0.08).astype(int)
    y_prob = np.where(y_true == 1, rng.uniform(0.5, 1.0, n), rng.uniform(0.0, 0.5, n))
    return y_true, y_prob


@pytest.fixture
def tiny_clf(toy_data):
    """Tiny fitted LogisticRegression — provides a real predict_proba interface."""
    y_true, y_prob = toy_data
    X = y_prob.reshape(-1, 1)
    clf = LogisticRegression(random_state=42, max_iter=500)
    clf.fit(X, y_true)
    return clf, X, y_true


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test to prevent resource leaks."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# gini_coefficient
# ---------------------------------------------------------------------------

class TestGiniCoefficient:
    def test_toy_data_in_valid_range(self, toy_data):
        y_true, y_prob = toy_data
        gini = gini_coefficient(y_true, y_prob)
        assert 0.5 < gini <= 1.0, f"Expected gini ≥ 0.5, got {gini:.4f}"

    def test_perfect_separation_equals_one(self, perfect_data):
        y_true, y_prob = perfect_data
        gini = gini_coefficient(y_true, y_prob)
        assert abs(gini - 1.0) < 1e-9

    def test_random_predictions_near_zero(self, random_data):
        y_true, y_prob = random_data
        gini = gini_coefficient(y_true, y_prob)
        assert abs(gini) < 0.05, f"Random predictions should give gini ~0, got {gini:.4f}"

    def test_inverse_predictions_negative(self, inverse_data):
        y_true, y_prob = inverse_data
        gini = gini_coefficient(y_true, y_prob)
        assert gini < 0.0, "Inverted predictions should produce negative Gini"

    def test_returns_float(self, toy_data):
        y_true, y_prob = toy_data
        assert isinstance(gini_coefficient(y_true, y_prob), float)

    def test_raises_on_single_class(self):
        y_true = np.array([0, 0, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4])
        with pytest.raises(ValueError):
            gini_coefficient(y_true, y_prob)


# ---------------------------------------------------------------------------
# ks_statistic
# ---------------------------------------------------------------------------

class TestKsStatistic:
    def test_returns_tuple_of_two_floats(self, toy_data):
        y_true, y_prob = toy_data
        result = ks_statistic(y_true, y_prob)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)

    def test_toy_data_ks_above_threshold(self, toy_data):
        y_true, y_prob = toy_data
        ks, thresh = ks_statistic(y_true, y_prob)
        assert 0.3 < ks <= 1.0, f"Expected KS > 0.3 on well-separated data, got {ks:.4f}"

    def test_perfect_separation_ks_equals_one(self, perfect_data):
        y_true, y_prob = perfect_data
        ks, thresh = ks_statistic(y_true, y_prob)
        assert abs(ks - 1.0) < 1e-9

    def test_threshold_within_prob_range(self, toy_data):
        y_true, y_prob = toy_data
        ks, thresh = ks_statistic(y_true, y_prob)
        assert y_prob.min() <= thresh <= y_prob.max()

    def test_ks_value_in_unit_interval(self, toy_data):
        y_true, y_prob = toy_data
        ks, _ = ks_statistic(y_true, y_prob)
        assert 0.0 <= ks <= 1.0

    def test_tied_probabilities_no_crash(self):
        """Tied probabilities must not cause division errors or NaN."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.4, 0.4, 0.6, 0.6, 0.4, 0.6])  # only two unique values
        ks, thresh = ks_statistic(y_true, y_prob)
        assert np.isfinite(ks)
        assert np.isfinite(thresh)

    def test_imbalanced_data_no_crash(self, imbalanced_data):
        y_true, y_prob = imbalanced_data
        ks, thresh = ks_statistic(y_true, y_prob)
        assert np.isfinite(ks)

    def test_raises_when_no_positives(self):
        y_true = np.array([0, 0, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4])
        with pytest.raises(ValueError, match="positive"):
            ks_statistic(y_true, y_prob)

    def test_raises_when_no_negatives(self):
        y_true = np.array([1, 1, 1, 1])
        y_prob = np.array([0.6, 0.7, 0.8, 0.9])
        with pytest.raises(ValueError, match="negative"):
            ks_statistic(y_true, y_prob)


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------

class TestEvaluateModel:
    EXPECTED_KEYS = {"Model", "AUC-ROC", "Gini", "KS", "Brier", "BrierSkill", "AvgPrecision"}

    def test_returns_dict(self, tiny_clf):
        clf, X, y_true = tiny_clf
        result = evaluate_model(clf, X, y_true)
        assert isinstance(result, dict)

    def test_dict_has_all_keys(self, tiny_clf):
        clf, X, y_true = tiny_clf
        result = evaluate_model(clf, X, y_true)
        assert result.keys() == self.EXPECTED_KEYS

    def test_model_name_stored(self, tiny_clf):
        clf, X, y_true = tiny_clf
        result = evaluate_model(clf, X, y_true, model_name="my_model")
        assert result["Model"] == "my_model"

    def test_auc_roc_in_unit_interval(self, tiny_clf):
        clf, X, y_true = tiny_clf
        result = evaluate_model(clf, X, y_true)
        assert 0.0 <= result["AUC-ROC"] <= 1.0

    def test_gini_in_valid_range(self, tiny_clf):
        clf, X, y_true = tiny_clf
        result = evaluate_model(clf, X, y_true)
        assert -1.0 <= result["Gini"] <= 1.0

    def test_ks_in_unit_interval(self, tiny_clf):
        clf, X, y_true = tiny_clf
        result = evaluate_model(clf, X, y_true)
        assert 0.0 <= result["KS"] <= 1.0

    def test_brier_in_unit_interval(self, tiny_clf):
        clf, X, y_true = tiny_clf
        result = evaluate_model(clf, X, y_true)
        assert 0.0 <= result["Brier"] <= 1.0

    def test_brier_skill_at_most_one(self, tiny_clf):
        clf, X, y_true = tiny_clf
        result = evaluate_model(clf, X, y_true)
        assert result["BrierSkill"] <= 1.0

    def test_better_than_random_gives_positive_brier_skill(self, tiny_clf):
        """A fitted model on separable data must have BSS > 0."""
        clf, X, y_true = tiny_clf
        result = evaluate_model(clf, X, y_true)
        assert result["BrierSkill"] > 0.0

    def test_avg_precision_in_unit_interval(self, tiny_clf):
        clf, X, y_true = tiny_clf
        result = evaluate_model(clf, X, y_true)
        assert 0.0 <= result["AvgPrecision"] <= 1.0

    def test_gini_equals_2auc_minus_1(self, tiny_clf):
        clf, X, y_true = tiny_clf
        result = evaluate_model(clf, X, y_true)
        assert abs(result["Gini"] - (2 * result["AUC-ROC"] - 1)) < 1e-9


# ---------------------------------------------------------------------------
# plot_roc_and_pr
# ---------------------------------------------------------------------------

class TestPlotRocAndPr:
    def test_returns_figure(self, tiny_clf):
        clf, X, y_true = tiny_clf
        fig = plot_roc_and_pr(clf, X, y_true)
        assert isinstance(fig, plt.Figure)

    def test_has_exactly_two_axes(self, tiny_clf):
        clf, X, y_true = tiny_clf
        fig = plot_roc_and_pr(clf, X, y_true)
        assert len(fig.axes) == 2

    def test_no_crash_on_random_predictions(self, random_data):
        """plot_roc_and_pr must not crash when the model has no discrimination."""
        y_true, y_prob = random_data

        class ConstantClf:
            def predict_proba(self, X):
                return np.column_stack([1 - y_prob, y_prob])

        fig = plot_roc_and_pr(ConstantClf(), np.zeros((len(y_true), 1)), y_true)
        assert isinstance(fig, plt.Figure)

    def test_save_path_writes_file(self, tiny_clf):
        clf, X, y_true = tiny_clf
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "roc_pr.png"
            plot_roc_and_pr(clf, X, y_true, save_path=str(save_path))
            assert save_path.exists()
            assert save_path.stat().st_size > 0

    def test_no_file_when_save_path_none(self, tiny_clf):
        """Default call must not write any file."""
        clf, X, y_true = tiny_clf
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_roc_and_pr(clf, X, y_true, save_path=None)
            files = list(Path(tmpdir).iterdir())
            assert len(files) == 0

    def test_model_name_used_in_title(self, tiny_clf):
        clf, X, y_true = tiny_clf
        fig = plot_roc_and_pr(clf, X, y_true, model_name="TestModel")
        title_text = fig.texts[0].get_text() if fig.texts else ""
        assert "TestModel" in title_text
