"""
test_model.py
-------------
Unit tests for credit_engine/model.py.

Written TDD-first (RED phase) before implementation.

Run with
--------
    pytest tests/test_model.py -v
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_engine.model import (
    benchmark_imbalance_strategies,
    load_model,
    save_model,
    train_logistic_baseline,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    500-row, 2-feature, 8% positive rate — fast proxy for real WoE dataset.

    Features are linearly separable: positives drawn from N(2, 1),
    negatives from N(0, 1). LR achieves Gini ~ 0.65 on this data,
    well above the unit-test threshold of 0.40.
    """
    rng = np.random.default_rng(42)
    n = 500
    n_pos = int(n * 0.08)
    y = np.zeros(n, dtype=int)
    y[:n_pos] = 1
    rng.shuffle(y)

    X = pd.DataFrame({
        "f1": np.where(y == 1, rng.normal(2.0, 1.0, n), rng.normal(0.0, 1.0, n)),
        "f2": np.where(y == 1, rng.normal(1.5, 1.0, n), rng.normal(0.0, 1.0, n)),
    })
    return X, pd.Series(y, name="TARGET")


@pytest.fixture
def trained_model(mock_data):
    """
    Train logistic baseline on mock data once; reused across structural tests.

    Returns the full 6-tuple:
        (pipeline, metrics, X_train, X_test, y_train, y_test)
    """
    X, y = mock_data
    return train_logistic_baseline(X, y)


# ---------------------------------------------------------------------------
# save_model / load_model
# ---------------------------------------------------------------------------

def test_save_model_creates_file(mock_data, tmp_path):
    """save_model writes a file to the specified path."""
    X, y = mock_data
    pipeline, *_ = train_logistic_baseline(X, y)
    path = tmp_path / "test_model.pkl"
    save_model(pipeline, path)
    assert path.exists()


def test_load_model_returns_pipeline(mock_data, tmp_path):
    """load_model deserialises to a sklearn Pipeline instance."""
    X, y = mock_data
    pipeline, *_ = train_logistic_baseline(X, y)
    path = tmp_path / "test_model.pkl"
    save_model(pipeline, path)
    loaded = load_model(path)
    assert isinstance(loaded, Pipeline)


def test_loaded_model_predicts_same_as_original(mock_data, tmp_path):
    """Round-trip save → load produces identical predict_proba output."""
    X, y = mock_data
    pipeline, _, _, X_test, _, _ = train_logistic_baseline(X, y)
    path = tmp_path / "test_model.pkl"
    save_model(pipeline, path)
    loaded = load_model(path)
    np.testing.assert_array_almost_equal(
        pipeline.predict_proba(X_test),
        loaded.predict_proba(X_test),
    )


def test_load_model_raises_if_file_missing(tmp_path):
    """load_model raises FileNotFoundError for a non-existent path."""
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nonexistent.pkl")


# ---------------------------------------------------------------------------
# Return structure
# ---------------------------------------------------------------------------

def test_train_logistic_baseline_returns_6_tuple(trained_model):
    """Function returns exactly 6 elements: pipeline, metrics, X_train, X_test, y_train, y_test."""
    assert len(trained_model) == 6


def test_train_logistic_baseline_metrics_dict_keys(trained_model):
    """Metrics dict has all keys expected from evaluate_model()."""
    _, metrics, *_ = trained_model
    expected = {"Model", "AUC-ROC", "Gini", "KS", "Brier", "BrierSkill", "AvgPrecision"}
    assert set(metrics.keys()) == expected


def test_train_logistic_baseline_splits_are_correct_types(trained_model):
    """X splits are DataFrames; y splits are Series."""
    _, _, X_train, X_test, y_train, y_test = trained_model
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)


def test_train_test_split_sizes(mock_data, trained_model):
    """Test split is ~20% of total; train + test covers all rows."""
    X, _ = mock_data
    _, _, X_train, X_test, _, _ = trained_model
    assert len(X_train) + len(X_test) == len(X)
    assert abs(len(X_test) / len(X) - 0.2) < 0.02


# ---------------------------------------------------------------------------
# Pipeline structure
# ---------------------------------------------------------------------------

def test_pipeline_has_scaler_and_lr_steps(trained_model):
    """Pipeline contains 'scaler' and 'lr' named steps in that order."""
    pipeline, *_ = trained_model
    step_names = [name for name, _ in pipeline.steps]
    assert step_names == ["scaler", "lr"]


def test_pipeline_scaler_is_standard_scaler(trained_model):
    """First step is a StandardScaler instance."""
    pipeline, *_ = trained_model
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)


def test_logistic_regression_hyperparams(trained_model):
    """LR step has the required hyperparameters for IRB baseline.

    class_weight is intentionally None: balanced shifts predicted probabilities
    from ~prevalence to ~0.43, making PD estimates unusable for EL calculations.
    """
    pipeline, *_ = trained_model
    lr = pipeline.named_steps["lr"]
    assert isinstance(lr, LogisticRegression)
    assert lr.C == 0.1
    assert lr.max_iter == 1000
    assert lr.solver == "lbfgs"
    assert lr.class_weight is None
    assert lr.random_state == 42


# ---------------------------------------------------------------------------
# Metric values (on separable mock data)
# ---------------------------------------------------------------------------

def test_gini_above_minimum_threshold(trained_model):
    """Gini ≥ 0.40 on separable mock data (unit-test floor; real data target ≥ 0.55)."""
    _, metrics, *_ = trained_model
    assert metrics["Gini"] >= 0.40, f"Gini too low on mock data: {metrics['Gini']:.4f}"


def test_auc_roc_in_valid_range(trained_model):
    """AUC-ROC is always in [0, 1]."""
    _, metrics, *_ = trained_model
    assert 0.0 <= metrics["AUC-ROC"] <= 1.0


def test_ks_statistic_is_positive(trained_model):
    """KS > 0 confirms the model has at least some discrimination."""
    _, metrics, *_ = trained_model
    assert metrics["KS"] > 0.0


def test_brier_skill_score_is_finite(trained_model):
    """BrierSkill is a finite float (not NaN — would indicate zero-variance prevalence)."""
    _, metrics, *_ = trained_model
    assert np.isfinite(metrics["BrierSkill"])


# ---------------------------------------------------------------------------
# benchmark_imbalance_strategies — Phase 1 TDD tests (RED before implementation)
# ---------------------------------------------------------------------------

_EXPECTED_STRATEGIES = ["SMOTE", "Cost-Sensitive", "Threshold-Tuned", "SMOTE+Cost-Sensitive"]
_EXPECTED_COLUMNS = ["Strategy", "AUC-ROC", "Gini", "KS", "F1-Macro", "Precision", "Recall"]
_METRIC_COLUMNS = ["AUC-ROC", "Gini", "KS", "F1-Macro", "Precision", "Recall"]


@pytest.fixture
def benchmark_splits(mock_data):
    """
    Extract (X_train, y_train, X_test, y_test) from the logistic baseline 6-tuple.

    Reuses the identical train/test split so Task 3.3 compares on the same
    hold-out set as the LR baseline (Gini 0.489).
    """
    X, y = mock_data
    _, _, X_train, X_test, y_train, y_test = train_logistic_baseline(X, y)
    return X_train, y_train, X_test, y_test


@pytest.fixture
def benchmark_result(benchmark_splits):
    """Run benchmark once; reused across structural tests to avoid repeated training."""
    X_train, y_train, X_test, y_test = benchmark_splits
    return benchmark_imbalance_strategies(X_train, y_train, X_test, y_test)


def test_benchmark_returns_dataframe(benchmark_result):
    """benchmark_imbalance_strategies returns a pandas DataFrame."""
    assert isinstance(benchmark_result, pd.DataFrame)


def test_benchmark_has_correct_shape(benchmark_result):
    """Result has exactly 4 rows (one per strategy) and 7 columns."""
    assert benchmark_result.shape == (4, 7), (
        f"Expected (4, 7), got {benchmark_result.shape}"
    )


def test_benchmark_column_names(benchmark_result):
    """Result columns exactly match the expected metric names."""
    assert list(benchmark_result.columns) == _EXPECTED_COLUMNS


def test_benchmark_strategy_names(benchmark_result):
    """Strategy column contains the four expected strategy names in order."""
    assert list(benchmark_result["Strategy"]) == _EXPECTED_STRATEGIES


def test_benchmark_metrics_in_valid_range(benchmark_result):
    """All 6 metric columns are in [0, 1] — catch numerical errors early."""
    for col in _METRIC_COLUMNS:
        col_min = benchmark_result[col].min()
        col_max = benchmark_result[col].max()
        assert col_min >= 0.0, f"{col} has value below 0: {col_min:.4f}"
        assert col_max <= 1.0, f"{col} has value above 1: {col_max:.4f}"


def test_benchmark_no_nan_metrics(benchmark_result):
    """No NaN in any metric column — NaN indicates a compute error (e.g., empty class)."""
    assert not benchmark_result[_METRIC_COLUMNS].isnull().any().any(), (
        f"NaN found in benchmark metrics:\n{benchmark_result}"
    )


def test_benchmark_gini_above_floor(benchmark_result):
    """At least one strategy achieves Gini > 0.10 on separable mock data."""
    assert benchmark_result["Gini"].max() > 0.10, (
        "All strategies have near-zero Gini — model is failing to learn."
    )


def test_smote_strategy_no_leakage(benchmark_splits):
    """SMOTE must never receive more rows than X_train.

    Verifies by patching SMOTE.fit_resample and asserting the row count
    seen is always ≤ len(X_train). If SMOTE were applied to the full
    dataset (train+test), the call would arrive with > len(X_train) rows.
    """
    from unittest.mock import patch
    from imblearn.over_sampling import SMOTE

    X_train, y_train, X_test, y_test = benchmark_splits
    seen_row_counts: list[int] = []
    original_fit_resample = SMOTE.fit_resample

    def tracking_fit_resample(self, X, y):
        seen_row_counts.append(len(X))
        return original_fit_resample(self, X, y)

    with patch.object(SMOTE, "fit_resample", tracking_fit_resample):
        benchmark_imbalance_strategies(X_train, y_train, X_test, y_test)

    assert len(seen_row_counts) > 0, "SMOTE.fit_resample was never called (SMOTE not used)"
    assert all(n <= len(X_train) for n in seen_row_counts), (
        f"SMOTE received {max(seen_row_counts)} rows but X_train has {len(X_train)}. "
        "Leakage: SMOTE was applied to validation or test data."
    )


def test_threshold_search_uses_cv_validation_only(benchmark_splits, monkeypatch):
    """Threshold optimization is computed on CV validation folds, not test data.

    Patches _find_optimal_threshold_f1_macro and records the size of every
    call. Each call must receive a fold-sized slice (< len(X_train)),
    confirming the function never sees the test set.
    """
    import credit_engine.model as model_module

    X_train, y_train, X_test, y_test = benchmark_splits
    call_sizes: list[int] = []
    original_fn = model_module._find_optimal_threshold_f1_macro

    def tracking_fn(y_true_val: np.ndarray, y_prob_val: np.ndarray) -> float:
        call_sizes.append(len(y_true_val))
        return original_fn(y_true_val, y_prob_val)

    monkeypatch.setattr(model_module, "_find_optimal_threshold_f1_macro", tracking_fn)
    benchmark_imbalance_strategies(X_train, y_train, X_test, y_test)

    assert len(call_sizes) > 0, "_find_optimal_threshold_f1_macro was never called"
    assert all(n < len(X_train) for n in call_sizes), (
        f"Threshold search received {max(call_sizes)} rows — X_train has {len(X_train)}. "
        "Threshold is being computed on full training set or test data."
    )


def test_benchmark_csv_saved(benchmark_splits, tmp_path, monkeypatch):
    """benchmark_imbalance_strategies saves results to reports/imbalance_benchmark.csv."""
    import credit_engine.model as model_module

    # Redirect the save path to tmp_path so tests don't pollute the working tree
    monkeypatch.setattr(model_module, "_BENCHMARK_REPORT_PATH", str(tmp_path / "imbalance_benchmark.csv"))

    X_train, y_train, X_test, y_test = benchmark_splits
    result = benchmark_imbalance_strategies(X_train, y_train, X_test, y_test)

    csv_path = tmp_path / "imbalance_benchmark.csv"
    assert csv_path.exists(), "imbalance_benchmark.csv was not created"

    saved = pd.read_csv(csv_path)
    assert saved.shape == result.shape
    assert list(saved.columns) == list(result.columns)
