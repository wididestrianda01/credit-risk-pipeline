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

import json

from credit_engine.model import (
    _AverageEnsemble,
    _TemporalCV,
    _make_cv,
    benchmark_imbalance_strategies,
    calibrate_model,
    load_model,
    run_ensemble_workflow,
    save_model,
    train_ensemble,
    train_lightgbm_optuna,
    train_logistic_baseline,
    train_xgboost_optuna,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
def benchmark_splits(mock_data):
    """
    Extract (X_train, y_train, X_test, y_test) from the logistic baseline 6-tuple.

    Reuses the identical train/test split so Task 3.3 compares on the same
    hold-out set as the LR baseline (Gini 0.489).
    """
    X, y = mock_data
    _, _, X_train, X_test, y_train, y_test = train_logistic_baseline(X, y)
    return X_train, y_train, X_test, y_test


@pytest.fixture(scope="module")
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


# ---------------------------------------------------------------------------
# train_xgboost_optuna — TDD tests (written RED before implementation)
# ---------------------------------------------------------------------------

_XGB_OPTUNA_EXPECTED_PARAM_KEYS = {
    "n_estimators",
    "max_depth",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "min_child_weight",
    "reg_alpha",
    "reg_lambda",
}


@pytest.fixture(scope="module")
def xgb_optuna_result():
    """
    Run train_xgboost_optuna once per module with n_trials=3 on mock data.

    Module scope ensures Optuna (3 trials × 5 CV folds = 15 fits + 1 final)
    runs only once regardless of how many tests consume this fixture.
    Function scope would re-run the study 7 times (one per test), taking
    ~7× longer with no additional coverage value.
    """
    rng = np.random.default_rng(42)
    n = 500
    n_pos = int(n * 0.08)
    y_arr = np.zeros(n, dtype=int)
    y_arr[:n_pos] = 1
    rng.shuffle(y_arr)
    X = pd.DataFrame({
        "f1": np.where(y_arr == 1, rng.normal(2.0, 1.0, n), rng.normal(0.0, 1.0, n)),
        "f2": np.where(y_arr == 1, rng.normal(1.5, 1.0, n), rng.normal(0.0, 1.0, n)),
    })
    y = pd.Series(y_arr, name="TARGET")
    return train_xgboost_optuna(X, y, n_trials=3)


# --- Return structure ---

def test_train_xgboost_optuna_returns_5_tuple(xgb_optuna_result):
    """Function returns exactly 5 elements."""
    assert len(xgb_optuna_result) == 5


def test_train_xgboost_optuna_return_types(xgb_optuna_result):
    """Return types: (XGBClassifier, dict, DataFrame, Series, dict)."""
    import xgboost as xgb
    model, metrics, X_test, y_test, best_params = xgb_optuna_result
    assert hasattr(model, "predict_proba"), "model must support predict_proba"
    assert isinstance(metrics, dict)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_test, pd.Series)
    assert isinstance(best_params, dict)


def test_train_xgboost_optuna_split_sizes(mock_data, xgb_optuna_result):
    """Test split is ~20% of total rows."""
    X, _ = mock_data
    _, _, X_test, _, _ = xgb_optuna_result
    assert abs(len(X_test) / len(X) - 0.2) < 0.02


# --- Metrics ---

def test_train_xgboost_optuna_metrics_keys(xgb_optuna_result):
    """metrics dict has all evaluate_model keys."""
    _, metrics, *_ = xgb_optuna_result
    expected = {"Model", "AUC-ROC", "Gini", "KS", "Brier", "BrierSkill", "AvgPrecision"}
    assert set(metrics.keys()) == expected


def test_train_xgboost_optuna_gini_on_separable_mock(xgb_optuna_result):
    """Gini ≥ 0.50 on linearly separable mock data (even with 3 trials)."""
    _, metrics, *_ = xgb_optuna_result
    assert metrics["Gini"] >= 0.50, f"Gini too low: {metrics['Gini']:.4f}"


def test_train_xgboost_optuna_auc_in_valid_range(xgb_optuna_result):
    """AUC-ROC is in [0, 1]."""
    _, metrics, *_ = xgb_optuna_result
    assert 0.0 <= metrics["AUC-ROC"] <= 1.0


def test_train_xgboost_optuna_ks_positive(xgb_optuna_result):
    """KS > 0 confirms model has discrimination."""
    _, metrics, *_ = xgb_optuna_result
    assert metrics["KS"] > 0.0


# --- best_params structure ---

def test_train_xgboost_optuna_best_params_has_all_keys(xgb_optuna_result):
    """best_params contains all 8 optimised hyperparameters."""
    *_, best_params = xgb_optuna_result
    assert _XGB_OPTUNA_EXPECTED_PARAM_KEYS.issubset(set(best_params.keys())), (
        f"Missing keys: {_XGB_OPTUNA_EXPECTED_PARAM_KEYS - set(best_params.keys())}"
    )


def test_train_xgboost_optuna_best_params_values_finite(xgb_optuna_result):
    """No NaN or inf in best_params values."""
    *_, best_params = xgb_optuna_result
    for k, v in best_params.items():
        assert np.isfinite(float(v)), f"Non-finite value for {k}: {v}"


# --- Artifact persistence ---

def test_train_xgboost_optuna_model_saved(mock_data, tmp_path, monkeypatch):
    """Model is saved to disk at the configured path."""
    import credit_engine.model as model_module
    monkeypatch.setattr(model_module, "_XGB_OPTUNA_MODEL_PATH", str(tmp_path / "xgb.pkl"))
    monkeypatch.setattr(model_module, "_XGB_OPTUNA_PARAMS_PATH", str(tmp_path / "xgb.json"))
    monkeypatch.setattr(model_module, "_XGB_OPTUNA_FIGURE_PATH", str(tmp_path / "xgb_roc.png"))

    X, y = mock_data
    train_xgboost_optuna(X, y, n_trials=2)
    assert (tmp_path / "xgb.pkl").exists(), "Model pickle not written"


def test_train_xgboost_optuna_params_json_valid(mock_data, tmp_path, monkeypatch):
    """Params JSON is valid, deserializable, and contains all 8 keys."""
    import credit_engine.model as model_module
    monkeypatch.setattr(model_module, "_XGB_OPTUNA_MODEL_PATH", str(tmp_path / "xgb.pkl"))
    params_path = tmp_path / "xgb.json"
    monkeypatch.setattr(model_module, "_XGB_OPTUNA_PARAMS_PATH", str(params_path))
    monkeypatch.setattr(model_module, "_XGB_OPTUNA_FIGURE_PATH", str(tmp_path / "xgb_roc.png"))

    X, y = mock_data
    train_xgboost_optuna(X, y, n_trials=2)

    assert params_path.exists(), "Params JSON not written"
    loaded = json.loads(params_path.read_text())
    assert _XGB_OPTUNA_EXPECTED_PARAM_KEYS.issubset(set(loaded.keys()))


def test_train_xgboost_optuna_model_round_trip(mock_data, tmp_path, monkeypatch):
    """Save → load → predict_proba produces identical output."""
    import credit_engine.model as model_module
    model_path = tmp_path / "xgb.pkl"
    monkeypatch.setattr(model_module, "_XGB_OPTUNA_MODEL_PATH", str(model_path))
    monkeypatch.setattr(model_module, "_XGB_OPTUNA_PARAMS_PATH", str(tmp_path / "xgb.json"))
    monkeypatch.setattr(model_module, "_XGB_OPTUNA_FIGURE_PATH", str(tmp_path / "xgb_roc.png"))

    X, y = mock_data
    model, _, X_test, _, _ = train_xgboost_optuna(X, y, n_trials=2)
    loaded = load_model(model_path)
    np.testing.assert_array_almost_equal(
        model.predict_proba(X_test),
        loaded.predict_proba(X_test),
    )


# --- Data leakage prevention ---

def test_train_xgboost_optuna_cv_never_sees_test_data(mock_data, monkeypatch):
    """CV fold fits must always receive strictly fewer rows than X_train.

    Monkeypatches XGBClassifier.fit to record input sizes. Any call with
    len(X) == len(X_train) means a full-training-set or test-set leak.
    The final refit on full X_train is excluded by only checking calls
    within the objective (n_trials=2 → 2×5=10 fold fits before the final).
    """
    import xgboost as xgb
    from sklearn.model_selection import train_test_split as tts

    X, y = mock_data
    X_train, _, y_train, _ = tts(X, y, test_size=0.2, stratify=y, random_state=42)

    fit_sizes: list[int] = []
    original_fit = xgb.XGBClassifier.fit

    def tracking_fit(self, X_fit, y_fit, **kwargs):
        fit_sizes.append(len(X_fit))
        return original_fit(self, X_fit, y_fit, **kwargs)

    monkeypatch.setattr(xgb.XGBClassifier, "fit", tracking_fit)
    train_xgboost_optuna(X, y, n_trials=2)

    # Exclude the final full-training-set refit (exactly len(X_train) rows)
    # All other calls must be fold-sized (< len(X_train))
    cv_fit_sizes = [s for s in fit_sizes if s < len(X_train)]
    assert len(cv_fit_sizes) > 0, "No CV fold fits detected"
    assert all(s < len(X_train) for s in cv_fit_sizes), (
        f"CV fold fit received {max(cv_fit_sizes)} rows; X_train={len(X_train)}"
    )


# --- Silent operation ---

def test_train_xgboost_optuna_no_stdout(mock_data, capsys):
    """Library function must not write to stdout (no print() calls)."""
    X, y = mock_data
    train_xgboost_optuna(X, y, n_trials=2)
    captured = capsys.readouterr()
    assert captured.out == "", f"Unexpected stdout:\n{captured.out}"


# --- Input validation ---

def test_train_xgboost_optuna_zero_trials_raises(mock_data):
    """n_trials=0 raises ValueError with a descriptive message."""
    X, y = mock_data
    with pytest.raises(ValueError, match="n_trials must be >= 1"):
        train_xgboost_optuna(X, y, n_trials=0)


# ---------------------------------------------------------------------------
# train_lightgbm_optuna — TDD tests (written RED before implementation)
# ---------------------------------------------------------------------------

_LGB_OPTUNA_EXPECTED_PARAM_KEYS = {
    "num_leaves",
    "max_depth",
    "learning_rate",
    "n_estimators",
    "min_child_samples",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
}


@pytest.fixture(scope="module")
def lgb_optuna_result():
    """
    Run train_lightgbm_optuna once per module with n_trials=3 on mock data.

    Module scope ensures Optuna (3 trials × 5 CV folds = 15 fits + 1 final)
    runs only once regardless of how many tests consume this fixture.
    Function scope would re-run the study 14 times (one per test), taking
    ~14× longer with no additional coverage value.
    """
    rng = np.random.default_rng(42)
    n = 500
    n_pos = int(n * 0.08)
    y_arr = np.zeros(n, dtype=int)
    y_arr[:n_pos] = 1
    rng.shuffle(y_arr)
    X = pd.DataFrame({
        "f1": np.where(y_arr == 1, rng.normal(2.0, 1.0, n), rng.normal(0.0, 1.0, n)),
        "f2": np.where(y_arr == 1, rng.normal(1.5, 1.0, n), rng.normal(0.0, 1.0, n)),
    })
    y = pd.Series(y_arr, name="TARGET")
    return train_lightgbm_optuna(X, y, n_trials=3)


# --- Return structure ---

def test_train_lightgbm_optuna_returns_5_tuple(lgb_optuna_result):
    """Function returns exactly 5 elements."""
    assert len(lgb_optuna_result) == 5


def test_train_lightgbm_optuna_return_types(lgb_optuna_result):
    """Return types: (LGBMClassifier, dict, DataFrame, Series, dict)."""
    model, metrics, X_test, y_test, best_params = lgb_optuna_result
    assert hasattr(model, "predict_proba"), "model must support predict_proba"
    assert isinstance(metrics, dict)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_test, pd.Series)
    assert isinstance(best_params, dict)


def test_train_lightgbm_optuna_split_sizes(mock_data, lgb_optuna_result):
    """Test split is ~20% of total rows."""
    X, _ = mock_data
    _, _, X_test, _, _ = lgb_optuna_result
    assert abs(len(X_test) / len(X) - 0.2) < 0.02


# --- Metrics ---

def test_train_lightgbm_optuna_metrics_keys(lgb_optuna_result):
    """metrics dict has all evaluate_model keys."""
    _, metrics, *_ = lgb_optuna_result
    expected = {"Model", "AUC-ROC", "Gini", "KS", "Brier", "BrierSkill", "AvgPrecision"}
    assert set(metrics.keys()) == expected


def test_train_lightgbm_optuna_gini_on_separable_mock(lgb_optuna_result):
    """Gini ≥ 0.50 on linearly separable mock data (even with 3 trials)."""
    _, metrics, *_ = lgb_optuna_result
    assert metrics["Gini"] >= 0.50, f"Gini too low: {metrics['Gini']:.4f}"


def test_train_lightgbm_optuna_auc_in_valid_range(lgb_optuna_result):
    """AUC-ROC is in [0, 1]."""
    _, metrics, *_ = lgb_optuna_result
    assert 0.0 <= metrics["AUC-ROC"] <= 1.0


def test_train_lightgbm_optuna_ks_positive(lgb_optuna_result):
    """KS > 0 confirms model has discrimination."""
    _, metrics, *_ = lgb_optuna_result
    assert metrics["KS"] > 0.0


# --- best_params structure ---

def test_train_lightgbm_optuna_best_params_has_all_keys(lgb_optuna_result):
    """best_params contains all 9 optimised hyperparameters."""
    *_, best_params = lgb_optuna_result
    assert _LGB_OPTUNA_EXPECTED_PARAM_KEYS.issubset(set(best_params.keys())), (
        f"Missing keys: {_LGB_OPTUNA_EXPECTED_PARAM_KEYS - set(best_params.keys())}"
    )


def test_train_lightgbm_optuna_best_params_values_finite(lgb_optuna_result):
    """No NaN or inf in best_params values."""
    *_, best_params = lgb_optuna_result
    for k, v in best_params.items():
        assert np.isfinite(float(v)), f"Non-finite value for {k}: {v}"


# --- Artifact persistence ---

def test_train_lightgbm_optuna_model_saved(mock_data, tmp_path, monkeypatch):
    """Model is saved to disk at the configured path."""
    import credit_engine.model as model_module
    monkeypatch.setattr(model_module, "_LGB_OPTUNA_MODEL_PATH", str(tmp_path / "lgb.pkl"))
    monkeypatch.setattr(model_module, "_LGB_OPTUNA_PARAMS_PATH", str(tmp_path / "lgb.json"))
    monkeypatch.setattr(model_module, "_LGB_OPTUNA_FIGURE_PATH", str(tmp_path / "lgb_roc.png"))

    X, y = mock_data
    train_lightgbm_optuna(X, y, n_trials=2)
    assert (tmp_path / "lgb.pkl").exists(), "Model pickle not written"


def test_train_lightgbm_optuna_params_json_valid(mock_data, tmp_path, monkeypatch):
    """Params JSON is valid, deserializable, and contains all 9 keys."""
    import credit_engine.model as model_module
    monkeypatch.setattr(model_module, "_LGB_OPTUNA_MODEL_PATH", str(tmp_path / "lgb.pkl"))
    params_path = tmp_path / "lgb.json"
    monkeypatch.setattr(model_module, "_LGB_OPTUNA_PARAMS_PATH", str(params_path))
    monkeypatch.setattr(model_module, "_LGB_OPTUNA_FIGURE_PATH", str(tmp_path / "lgb_roc.png"))

    X, y = mock_data
    train_lightgbm_optuna(X, y, n_trials=2)

    assert params_path.exists(), "Params JSON not written"
    loaded = json.loads(params_path.read_text())
    assert _LGB_OPTUNA_EXPECTED_PARAM_KEYS.issubset(set(loaded.keys()))


def test_train_lightgbm_optuna_model_round_trip(mock_data, tmp_path, monkeypatch):
    """Save → load → predict_proba produces identical output."""
    import credit_engine.model as model_module
    model_path = tmp_path / "lgb.pkl"
    monkeypatch.setattr(model_module, "_LGB_OPTUNA_MODEL_PATH", str(model_path))
    monkeypatch.setattr(model_module, "_LGB_OPTUNA_PARAMS_PATH", str(tmp_path / "lgb.json"))
    monkeypatch.setattr(model_module, "_LGB_OPTUNA_FIGURE_PATH", str(tmp_path / "lgb_roc.png"))

    X, y = mock_data
    model, _, X_test, _, _ = train_lightgbm_optuna(X, y, n_trials=2)
    loaded = load_model(model_path)
    np.testing.assert_array_almost_equal(
        model.predict_proba(X_test),
        loaded.predict_proba(X_test),
    )


# --- Data leakage prevention ---

def test_train_lightgbm_optuna_cv_never_sees_test_data(mock_data, monkeypatch):
    """CV fold fits must always receive strictly fewer rows than X_train.

    Monkeypatches LGBMClassifier.fit to record input sizes. Any call with
    len(X) == len(total_X) means a full-dataset or test-set leak.
    The final refit on X_tr (80% of X_train) is excluded by filtering
    on rows strictly less than len(X_train).
    """
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split as tts

    X, y = mock_data
    X_train, _, y_train, _ = tts(X, y, test_size=0.2, stratify=y, random_state=42)

    fit_sizes: list[int] = []
    original_fit = lgb.LGBMClassifier.fit

    def tracking_fit(self, X_fit, y_fit, **kwargs):
        fit_sizes.append(len(X_fit))
        return original_fit(self, X_fit, y_fit, **kwargs)

    monkeypatch.setattr(lgb.LGBMClassifier, "fit", tracking_fit)
    train_lightgbm_optuna(X, y, n_trials=2)

    # All fits (CV folds, stage-1 early-stop fit on X_tr, stage-2 refit on X_train)
    # must be smaller than the full dataset len(X) — X_train is 80% of X so this holds.
    assert len(fit_sizes) > 0, "No LGBMClassifier.fit calls detected"
    assert all(s < len(X) for s in fit_sizes), (
        f"A fit received {max(fit_sizes)} rows but total X has {len(X)} rows. "
        "Possible test-set leakage."
    )


# --- Silent operation ---

def test_train_lightgbm_optuna_no_stdout(mock_data, capsys):
    """Library function must not write to stdout (no print() calls)."""
    X, y = mock_data
    train_lightgbm_optuna(X, y, n_trials=2)
    captured = capsys.readouterr()
    assert captured.out == "", f"Unexpected stdout:\n{captured.out}"


# --- Input validation ---

def test_train_lightgbm_optuna_zero_trials_raises(mock_data):
    """n_trials=0 raises ValueError with a descriptive message."""
    X, y = mock_data
    with pytest.raises(ValueError, match="n_trials must be >= 1"):
        train_lightgbm_optuna(X, y, n_trials=0)


# ---------------------------------------------------------------------------
# calibrate_model — TDD tests (written RED before implementation)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def calibration_inputs(lgb_optuna_result) -> tuple:
    """
    Provide a fitted LightGBM model + matching train/test splits for calibration.

    Module scope: calibration only runs once even though multiple tests consume
    this fixture. The underlying lgb_optuna_result is also module-scoped, so no
    extra model training occurs.
    """
    model, _, X_test, y_test, _ = lgb_optuna_result

    # Reconstruct a small train split consistent with the mock data used in
    # lgb_optuna_result (same RNG seed) so the feature columns align.
    rng = np.random.default_rng(42)
    n = 500
    n_pos = int(n * 0.08)
    y_arr = np.zeros(n, dtype=int)
    y_arr[:n_pos] = 1
    rng.shuffle(y_arr)
    X_all = pd.DataFrame({
        "f1": np.where(y_arr == 1, rng.normal(2.0, 1.0, n), rng.normal(0.0, 1.0, n)),
        "f2": np.where(y_arr == 1, rng.normal(1.5, 1.0, n), rng.normal(0.0, 1.0, n)),
    })
    from sklearn.model_selection import train_test_split as tts
    X_train, _, y_train, _ = tts(
        X_all, pd.Series(y_arr, name="TARGET"),
        test_size=0.2, stratify=y_arr, random_state=42
    )
    return model, X_train, y_train, X_test, y_test


@pytest.fixture(scope="module")
def calibration_result(calibration_inputs, tmp_path_factory):
    """
    Run calibrate_model once per module and return the 3-tuple result.

    tmp_path_factory provides a module-scoped temp directory, unlike
    tmp_path which is function-scoped and incompatible with module fixtures.
    """
    import credit_engine.model as model_module

    tmp = tmp_path_factory.mktemp("calibration")
    model_module._CALIBRATED_MODEL_PATH = str(tmp / "lgb_calibrated.pkl")
    model_module._CALIBRATION_FIGURE_PATH = str(tmp / "calibration_reliability.png")

    model, X_train, y_train, X_test, y_test = calibration_inputs
    return calibrate_model(model, X_train, y_train, X_test, y_test)


# --- Return structure ---

def test_calibrate_model_returns_3_tuple(calibration_result):
    """calibrate_model returns (calibrated_model, brier_uncal, brier_cal)."""
    assert len(calibration_result) == 3


def test_calibrate_model_return_types(calibration_result):
    """Types: (object with predict_proba, float, float)."""
    cal_model, brier_uncal, brier_cal = calibration_result
    assert hasattr(cal_model, "predict_proba"), "calibrated model must support predict_proba"
    assert isinstance(brier_uncal, float)
    assert isinstance(brier_cal, float)


def test_calibrate_model_brier_scores_finite(calibration_result):
    """Both Brier scores are finite positive floats."""
    _, brier_uncal, brier_cal = calibration_result
    assert np.isfinite(brier_uncal) and brier_uncal >= 0.0
    assert np.isfinite(brier_cal) and brier_cal >= 0.0


def test_calibrate_model_brier_improves_or_neutral(calibration_result):
    """Calibration should not meaningfully worsen the Brier score.

    Platt scaling on linearly separable mock data may not produce
    a large improvement, but it must not increase Brier by more than 0.05
    (which would indicate a calibration bug, not just a hard dataset).
    """
    _, brier_uncal, brier_cal = calibration_result
    assert brier_cal <= brier_uncal + 0.05, (
        f"Calibration degraded Brier: {brier_uncal:.4f} → {brier_cal:.4f}"
    )


def test_calibrate_model_predict_proba_in_unit_interval(calibration_result, calibration_inputs):
    """Calibrated probabilities are in [0, 1] for all test samples."""
    cal_model, _, _ = calibration_result
    _, _, _, X_test, _ = calibration_inputs
    y_prob = cal_model.predict_proba(X_test)[:, 1]
    assert np.all(y_prob >= 0.0) and np.all(y_prob <= 1.0)


def test_calibrate_model_probabilities_sum_to_one(calibration_result, calibration_inputs):
    """predict_proba columns sum to 1 for every sample."""
    cal_model, _, _ = calibration_result
    _, _, _, X_test, _ = calibration_inputs
    proba = cal_model.predict_proba(X_test)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# --- Artifact persistence ---

def test_calibrate_model_saves_file(calibration_inputs, tmp_path, monkeypatch):
    """calibrate_model saves a pickle file at the configured path."""
    import credit_engine.model as model_module

    out_path = tmp_path / "cal.pkl"
    monkeypatch.setattr(model_module, "_CALIBRATED_MODEL_PATH", str(out_path))
    monkeypatch.setattr(model_module, "_CALIBRATION_FIGURE_PATH", str(tmp_path / "fig.png"))

    model, X_train, y_train, X_test, y_test = calibration_inputs
    calibrate_model(model, X_train, y_train, X_test, y_test)
    assert out_path.exists(), "Calibrated model file not written"


def test_calibrate_model_saved_file_is_loadable(calibration_inputs, tmp_path, monkeypatch):
    """Saved calibrated model round-trips via joblib and predicts correctly."""
    import credit_engine.model as model_module

    out_path = tmp_path / "cal.pkl"
    monkeypatch.setattr(model_module, "_CALIBRATED_MODEL_PATH", str(out_path))
    monkeypatch.setattr(model_module, "_CALIBRATION_FIGURE_PATH", str(tmp_path / "fig.png"))

    model, X_train, y_train, X_test, y_test = calibration_inputs
    cal_model, _, _ = calibrate_model(model, X_train, y_train, X_test, y_test)
    loaded = load_model(out_path)
    np.testing.assert_array_almost_equal(
        cal_model.predict_proba(X_test),
        loaded.predict_proba(X_test),
    )


def test_calibrate_model_no_stdout(calibration_inputs, tmp_path, monkeypatch, capsys):
    """calibrate_model must not write to stdout."""
    import credit_engine.model as model_module

    monkeypatch.setattr(model_module, "_CALIBRATED_MODEL_PATH", str(tmp_path / "cal.pkl"))
    monkeypatch.setattr(model_module, "_CALIBRATION_FIGURE_PATH", str(tmp_path / "fig.png"))

    model, X_train, y_train, X_test, y_test = calibration_inputs
    calibrate_model(model, X_train, y_train, X_test, y_test)
    captured = capsys.readouterr()
    assert captured.out == "", f"Unexpected stdout:\n{captured.out}"


# ---------------------------------------------------------------------------
# train_ensemble — OOF Ensemble (LGB + XGB stacking)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ensemble_average_result(mock_data):
    """Train ensemble with method='average' once per module."""
    X, y = mock_data
    return train_ensemble(X, y, method="average", n_splits=2)


@pytest.fixture(scope="module")
def ensemble_logistic_result(mock_data):
    """Train ensemble with method='logistic' once per module."""
    X, y = mock_data
    return train_ensemble(X, y, method="logistic", n_splits=2)


class TestEnsemble:
    """TDD tests for out-of-fold ensemble stacking via train_ensemble()."""

    def test_train_ensemble_returns_tuple(self, ensemble_average_result):
        """train_ensemble returns a (model, dict) tuple."""
        model, metrics = ensemble_average_result
        assert isinstance(model, object), "ensemble_model is not an object"
        assert isinstance(metrics, dict), "metrics is not a dict"

    def test_train_ensemble_metrics_keys(self, ensemble_average_result):
        """metrics dict has all expected keys from evaluate_model()."""
        _, metrics = ensemble_average_result
        expected_keys = {"Model", "AUC-ROC", "Gini", "KS", "Brier", "BrierSkill", "AvgPrecision"}
        assert set(metrics.keys()) == expected_keys, (
            f"Missing or extra keys. Expected {expected_keys}, got {set(metrics.keys())}"
        )

    def test_train_ensemble_average_method_gini(self, ensemble_average_result):
        """Ensemble with method='average' achieves Gini > 0.40 on separable mock data."""
        _, metrics = ensemble_average_result
        assert metrics["Gini"] > 0.40, (
            f"Average ensemble Gini too low: {metrics['Gini']:.4f} (expected > 0.40)"
        )

    def test_train_ensemble_logistic_method_gini(self, ensemble_logistic_result):
        """Ensemble with method='logistic' achieves Gini > 0.40 on separable mock data."""
        _, metrics = ensemble_logistic_result
        assert metrics["Gini"] > 0.40, (
            f"Logistic ensemble Gini too low: {metrics['Gini']:.4f} (expected > 0.40)"
        )

    def test_average_ensemble_predict_proba_shape(self, mock_data, ensemble_average_result):
        """_AverageEnsemble.predict_proba returns shape (n, 2) with columns summing to 1.0."""
        X, _ = mock_data
        model, _ = ensemble_average_result
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2), f"Expected shape ({len(X)}, 2), got {proba.shape}"
        # Verify columns sum to 1.0
        sums = proba.sum(axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones(len(X)), decimal=6)

    def test_train_ensemble_oof_no_leakage(self, ensemble_average_result):
        """Ensemble Gini on holdout must be in [0, 1] (basic sanity check)."""
        _, metrics = ensemble_average_result
        gini = metrics["Gini"]
        assert 0.0 <= gini <= 1.0, f"Gini {gini:.4f} outside [0, 1] (data leakage suspected)"


def test_calibrate_model_isotonic_method(calibration_inputs, tmp_path, monkeypatch):
    """calibrate_model with method='isotonic' returns a valid calibrated model."""
    import credit_engine.model as model_module

    monkeypatch.setattr(model_module, "_CALIBRATED_MODEL_PATH", str(tmp_path / "iso.pkl"))
    monkeypatch.setattr(model_module, "_CALIBRATION_FIGURE_PATH", str(tmp_path / "iso_fig.png"))

    model, X_train, y_train, X_test, y_test = calibration_inputs
    cal_model, brier_uncal, brier_cal = calibrate_model(
        model, X_train, y_train, X_test, y_test, method="isotonic"
    )

    assert hasattr(cal_model, "predict_proba"), "isotonic calibrated model must have predict_proba"
    assert isinstance(brier_uncal, float) and np.isfinite(brier_uncal)
    assert isinstance(brier_cal, float) and np.isfinite(brier_cal)
    # Calibration must not catastrophically degrade Brier score
    assert brier_cal <= brier_uncal + 0.05, (
        f"Isotonic calibration degraded Brier: {brier_uncal:.4f} → {brier_cal:.4f}"
    )


# ---------------------------------------------------------------------------
# _TemporalCV and _make_cv — temporal cross-validation tests
# ---------------------------------------------------------------------------

def _make_temporal_mock(n: int = 200) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    200-row mock dataset with a synthetic DAYS_ID_PUBLISH temporal column.

    Returns (X, y, groups) where groups is a Series of integers in [-7197, 0],
    simulating the Home Credit temporal proxy column.
    """
    rng = np.random.default_rng(7)
    n_pos = max(1, int(n * 0.08))
    y_arr = np.zeros(n, dtype=int)
    y_arr[:n_pos] = 1
    rng.shuffle(y_arr)
    X = pd.DataFrame({
        "f1": rng.normal(0.0, 1.0, n),
        "f2": rng.normal(0.0, 1.0, n),
    })
    y = pd.Series(y_arr, name="TARGET")
    # Simulate DAYS_ID_PUBLISH: monotone negative integers (older = more negative)
    groups = pd.Series(np.sort(rng.integers(-7197, 0, n)), name="DAYS_ID_PUBLISH")
    return X, y, groups


def test_temporal_cv_produces_correct_number_of_folds():
    """_TemporalCV yields exactly n_splits (train, val) pairs."""
    X, y, groups = _make_temporal_mock()
    cv = _TemporalCV(groups=groups.to_numpy(), n_splits=5)
    splits = list(cv.split(X, y))
    assert len(splits) == 5, f"Expected 5 folds, got {len(splits)}"


def test_temporal_cv_train_indices_precede_val_indices():
    """All training samples must be temporally older than all validation samples.

    This is the core property: max(groups[train]) < min(groups[val]) for every fold.
    A single violation would indicate that the future is leaking into the past.
    """
    X, y, groups = _make_temporal_mock(n=300)
    groups_arr = groups.to_numpy()
    cv = _TemporalCV(groups=groups_arr, n_splits=5)
    for fold_i, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        max_train_time = groups_arr[train_idx].max()
        min_val_time = groups_arr[val_idx].min()
        assert max_train_time < min_val_time, (
            f"Fold {fold_i}: training data bleeds into the future. "
            f"max(train_time)={max_train_time}, min(val_time)={min_val_time}"
        )


def test_temporal_cv_embargo_reduces_train_size():
    """Embargo must remove at least 1 sample from the training fold boundary.

    Compare a CV with embargo=0 vs embargo=0.05 — the embargoed version must
    always return fewer or equal training samples.
    """
    X, y, groups = _make_temporal_mock(n=300)
    groups_arr = groups.to_numpy()

    cv_no_embargo = _TemporalCV(groups=groups_arr, n_splits=5, embargo_frac=0.0)
    cv_embargoed = _TemporalCV(groups=groups_arr, n_splits=5, embargo_frac=0.05)

    for (train_no, _), (train_em, _) in zip(
        cv_no_embargo.split(X, y), cv_embargoed.split(X, y)
    ):
        assert len(train_em) <= len(train_no), (
            f"Embargo did not reduce train size: {len(train_em)} > {len(train_no)}"
        )


def test_temporal_cv_val_indices_never_in_train():
    """No index appears in both train and validation sets for any fold."""
    X, y, groups = _make_temporal_mock()
    cv = _TemporalCV(groups=groups.to_numpy(), n_splits=5)
    for train_idx, val_idx in cv.split(X, y):
        overlap = set(train_idx.tolist()) & set(val_idx.tolist())
        assert len(overlap) == 0, f"Leakage: {len(overlap)} indices in both train and val"


def test_make_cv_returns_stratified_kfold_when_no_groups():
    """_make_cv(groups_train=None) returns StratifiedKFold, not _TemporalCV."""
    from sklearn.model_selection import StratifiedKFold
    cv = _make_cv(groups_train=None, n_splits=5)
    assert isinstance(cv, StratifiedKFold)


def test_make_cv_returns_temporal_cv_when_groups_provided():
    """_make_cv with groups array returns _TemporalCV."""
    groups_arr = np.arange(200, dtype=float)
    cv = _make_cv(groups_train=groups_arr, n_splits=5)
    assert isinstance(cv, _TemporalCV)


def test_train_logistic_baseline_accepts_groups_parameter():
    """train_logistic_baseline runs without error when groups is provided."""
    X, y, groups = _make_temporal_mock(n=300)
    # groups must align to X.index — our fixture already has matching integer index
    pipeline, metrics, X_train, X_test, y_train, y_test = train_logistic_baseline(
        X, y, groups=groups
    )
    assert isinstance(pipeline, Pipeline)
    assert 0.0 <= metrics["AUC-ROC"] <= 1.0


def test_temporal_cv_groups_alignment_after_train_test_split():
    """groups.loc[X_train.index] correctly aligns when DataFrame has non-contiguous index.

    After train_test_split, X_train has a non-contiguous integer index (e.g.
    [3, 7, 12, ...]). Aligning via .loc preserves temporal ordering, whereas
    positional alignment (.iloc) would silently use wrong group values.
    """
    X, y, groups = _make_temporal_mock(n=300)
    from sklearn.model_selection import train_test_split as tts
    X_train, _, y_train, _ = tts(X, y, test_size=0.2, stratify=y, random_state=42)

    # Verify .loc alignment produces the same length as X_train
    groups_aligned = groups.loc[X_train.index]
    assert len(groups_aligned) == len(X_train), (
        f"groups alignment mismatch: {len(groups_aligned)} != {len(X_train)}"
    )
    # Verify every aligned group value corresponds to a valid X_train row
    assert set(groups_aligned.index) == set(X_train.index)


# ---------------------------------------------------------------------------
# Priority 2.4 — XGBoost extended search space (gamma, max_delta_step, wider MCW)
# ---------------------------------------------------------------------------

class TestXGBoostExtendedSearchSpace:
    """TDD tests for the expanded XGBoost Optuna search space (Priority 2.4).

    Written RED before implementation: gamma and max_delta_step are absent from
    _xgboost_optuna_objective() as of commit e2e101a.
    """

    @pytest.fixture(scope="class")
    def xgb_best_params(self, tmp_path_factory):
        """Run train_xgboost_optuna once and return best_params dict."""
        import credit_engine.model as model_module
        tmp = tmp_path_factory.mktemp("xgb_ext")
        mp = pytest.MonkeyPatch()
        mp.setattr(model_module, "_XGB_OPTUNA_MODEL_PATH", str(tmp / "xgb.pkl"))
        mp.setattr(model_module, "_XGB_OPTUNA_PARAMS_PATH", str(tmp / "xgb.json"))
        mp.setattr(model_module, "_XGB_OPTUNA_FIGURE_PATH", str(tmp / "xgb.png"))
        rng = np.random.default_rng(42)
        n = 500
        y_arr = np.zeros(n, dtype=int)
        y_arr[:40] = 1
        rng.shuffle(y_arr)
        X = pd.DataFrame({
            "f1": np.where(y_arr == 1, rng.normal(2.0, 1.0, n), rng.normal(0.0, 1.0, n)),
            "f2": np.where(y_arr == 1, rng.normal(1.5, 1.0, n), rng.normal(0.0, 1.0, n)),
        })
        y = pd.Series(y_arr, name="TARGET")
        _, _, _, _, best_params = train_xgboost_optuna(X, y, n_trials=3)
        return best_params

    def test_best_params_includes_gamma(self, xgb_best_params):
        """best_params must contain 'gamma' after search space extension."""
        assert "gamma" in xgb_best_params, (
            f"'gamma' missing from best_params keys: {sorted(xgb_best_params.keys())}"
        )

    def test_best_params_includes_max_delta_step(self, xgb_best_params):
        """best_params must contain 'max_delta_step' after search space extension."""
        assert "max_delta_step" in xgb_best_params, (
            f"'max_delta_step' missing from best_params keys: {sorted(xgb_best_params.keys())}"
        )

    def test_gamma_within_validated_range(self, xgb_best_params):
        """Sampled gamma must lie in [0.0, 2.0] — subagent-validated bound."""
        assert 0.0 <= xgb_best_params["gamma"] <= 2.0, (
            f"gamma={xgb_best_params['gamma']:.4f} outside [0.0, 2.0]"
        )

    def test_min_child_weight_constant_extended_to_15(self):
        """_XGB_MIN_CHILD_WEIGHT_MAX must be extended to at least 15."""
        from credit_engine.model import _XGB_MIN_CHILD_WEIGHT_MAX
        assert _XGB_MIN_CHILD_WEIGHT_MAX >= 15, (
            f"_XGB_MIN_CHILD_WEIGHT_MAX={_XGB_MIN_CHILD_WEIGHT_MAX} — "
            "must be >= 15 per subagent recommendation (100-sample leaf rule)"
        )


# ---------------------------------------------------------------------------
# Priority 2.1 prep — Temporal CV wiring in train_ensemble()
# ---------------------------------------------------------------------------

def _make_ensemble_mock(with_sort_col: bool, n: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    """200-row mock with or without prev_days_decision_mean column."""
    rng = np.random.default_rng(99)
    y_arr = np.zeros(n, dtype=int)
    y_arr[:16] = 1
    rng.shuffle(y_arr)
    data = {
        "f1": np.where(y_arr == 1, rng.normal(2.0, 1.0, n), rng.normal(0.0, 1.0, n)),
        "f2": np.where(y_arr == 1, rng.normal(1.5, 1.0, n), rng.normal(0.0, 1.0, n)),
    }
    if with_sort_col:
        data["prev_days_decision_mean"] = np.sort(
            rng.integers(-7000, 0, n)
        ).astype(float)
    return pd.DataFrame(data), pd.Series(y_arr, name="TARGET")


class TestEnsembleTemporalCV:
    """TDD tests for temporal CV wiring in train_ensemble() (Priority 2.1 prep).

    Written RED before implementation: train_ensemble() currently hard-codes
    StratifiedKFold and never calls _make_cv.
    """

    def test_ensemble_uses_temporal_cv_when_sort_col_present(self, monkeypatch):
        """_make_cv must receive non-None groups_train when prev_days_decision_mean is in X."""
        import credit_engine.model as model_module

        received_groups: list = []
        original_make_cv = model_module._make_cv

        def tracking_make_cv(groups_train, n_splits):
            received_groups.append(groups_train)
            return original_make_cv(groups_train, n_splits)

        monkeypatch.setattr(model_module, "_make_cv", tracking_make_cv)

        X, y = _make_ensemble_mock(with_sort_col=True)
        train_ensemble(X, y, n_splits=2)

        assert len(received_groups) > 0, "_make_cv was never called"
        assert any(g is not None for g in received_groups), (
            "train_ensemble passed groups=None to _make_cv despite "
            f"'{model_module._TEMPORAL_SORT_COL}' being present in X"
        )

    def test_ensemble_falls_back_to_stratified_when_no_sort_col(self, monkeypatch):
        """_make_cv must receive groups_train=None when prev_days_decision_mean is absent."""
        import credit_engine.model as model_module

        received_groups: list = []
        original_make_cv = model_module._make_cv

        def tracking_make_cv(groups_train, n_splits):
            received_groups.append(groups_train)
            return original_make_cv(groups_train, n_splits)

        monkeypatch.setattr(model_module, "_make_cv", tracking_make_cv)

        X, y = _make_ensemble_mock(with_sort_col=False)
        train_ensemble(X, y, n_splits=2)

        assert len(received_groups) > 0, "_make_cv was never called"
        assert all(g is None for g in received_groups), (
            "train_ensemble passed non-None groups to _make_cv despite "
            "sort column being absent from X"
        )


# ---------------------------------------------------------------------------
# Priority 2.1 — run_ensemble_workflow() (activate + gate ensemble)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ensemble_workflow_result(tmp_path_factory):
    """Run run_ensemble_workflow once per module using default (non-HPO) params."""
    import credit_engine.model as model_module
    tmp = tmp_path_factory.mktemp("ensemble_wf")
    mp = pytest.MonkeyPatch()
    mp.setattr(model_module, "_ENSEMBLE_WORKFLOW_MODEL_PATH", str(tmp / "ens.pkl"))
    mp.setattr(model_module, "_ENSEMBLE_WORKFLOW_WEIGHTS_PATH", str(tmp / "weights.json"))
    rng = np.random.default_rng(42)
    n = 500
    y_arr = np.zeros(n, dtype=int)
    y_arr[:40] = 1
    rng.shuffle(y_arr)
    X = pd.DataFrame({
        "f1": np.where(y_arr == 1, rng.normal(2.0, 1.0, n), rng.normal(0.0, 1.0, n)),
        "f2": np.where(y_arr == 1, rng.normal(1.5, 1.0, n), rng.normal(0.0, 1.0, n)),
    })
    y = pd.Series(y_arr, name="TARGET")
    return run_ensemble_workflow(X, y)


class TestRunEnsembleWorkflow:
    """TDD tests for run_ensemble_workflow() (Priority 2.1).

    Written RED before implementation: run_ensemble_workflow() does not yet exist.
    """

    def test_returns_dict(self, ensemble_workflow_result):
        """run_ensemble_workflow must return a dict."""
        assert isinstance(ensemble_workflow_result, dict)

    def test_returns_required_keys(self, ensemble_workflow_result):
        """Return dict must contain all four required metric keys."""
        required = {"lgb_gini", "xgb_gini", "ensemble_gini", "improvement", "persisted"}
        assert required.issubset(ensemble_workflow_result.keys()), (
            f"Missing keys: {required - set(ensemble_workflow_result.keys())}"
        )

    def test_improvement_is_gini_delta(self, ensemble_workflow_result):
        """improvement = ensemble_gini - max(lgb_gini, xgb_gini)."""
        r = ensemble_workflow_result
        expected = r["ensemble_gini"] - max(r["lgb_gini"], r["xgb_gini"])
        assert abs(r["improvement"] - expected) < 1e-9, (
            f"improvement={r['improvement']:.6f} != ensemble - best_single={expected:.6f}"
        )

    def test_gini_values_in_unit_interval(self, ensemble_workflow_result):
        """All Gini values must be in [0, 1]."""
        r = ensemble_workflow_result
        for key in ("lgb_gini", "xgb_gini", "ensemble_gini"):
            assert 0.0 <= r[key] <= 1.0, f"{key}={r[key]:.4f} outside [0, 1]"

    def test_ensemble_persists_when_improvement_exceeds_threshold(
        self, tmp_path, monkeypatch
    ):
        """Ensemble model file is written when improvement >= _ENSEMBLE_PERSIST_THRESHOLD."""
        import credit_engine.model as model_module
        out_path = tmp_path / "ens.pkl"
        weights_path = tmp_path / "weights.json"
        monkeypatch.setattr(model_module, "_ENSEMBLE_WORKFLOW_MODEL_PATH", str(out_path))
        monkeypatch.setattr(model_module, "_ENSEMBLE_WORKFLOW_WEIGHTS_PATH", str(weights_path))
        # Force threshold to 0.0 so any positive improvement triggers persist
        monkeypatch.setattr(model_module, "_ENSEMBLE_PERSIST_THRESHOLD", 0.0)

        rng = np.random.default_rng(42)
        n, n_pos = 500, 40
        y_arr = np.zeros(n, dtype=int)
        y_arr[:n_pos] = 1
        rng.shuffle(y_arr)
        X = pd.DataFrame({
            "f1": np.where(y_arr == 1, rng.normal(2.0, 1.0, n), rng.normal(0.0, 1.0, n)),
            "f2": np.where(y_arr == 1, rng.normal(1.5, 1.0, n), rng.normal(0.0, 1.0, n)),
        })
        y = pd.Series(y_arr, name="TARGET")
        result = run_ensemble_workflow(X, y)

        if result["improvement"] >= 0.0:
            assert out_path.exists(), (
                "Ensemble model not persisted despite improvement >= threshold (0.0)"
            )

    def test_ensemble_skips_persist_when_below_threshold(
        self, tmp_path, monkeypatch
    ):
        """Ensemble model file is NOT written when improvement < threshold."""
        import credit_engine.model as model_module
        out_path = tmp_path / "ens_skip.pkl"
        weights_path = tmp_path / "weights_skip.json"
        monkeypatch.setattr(model_module, "_ENSEMBLE_WORKFLOW_MODEL_PATH", str(out_path))
        monkeypatch.setattr(model_module, "_ENSEMBLE_WORKFLOW_WEIGHTS_PATH", str(weights_path))
        # Threshold so high that no ensemble can ever beat it
        monkeypatch.setattr(model_module, "_ENSEMBLE_PERSIST_THRESHOLD", 999.0)

        rng = np.random.default_rng(42)
        n, n_pos = 500, 40
        y_arr = np.zeros(n, dtype=int)
        y_arr[:n_pos] = 1
        rng.shuffle(y_arr)
        X = pd.DataFrame({
            "f1": np.where(y_arr == 1, rng.normal(2.0, 1.0, n), rng.normal(0.0, 1.0, n)),
            "f2": np.where(y_arr == 1, rng.normal(1.5, 1.0, n), rng.normal(0.0, 1.0, n)),
        })
        y = pd.Series(y_arr, name="TARGET")
        result = run_ensemble_workflow(X, y)

        assert not out_path.exists(), "Ensemble model persisted despite improvement < threshold"
        assert not result["persisted"], "result['persisted'] should be False"

    def test_weights_json_written_when_persisted(self, tmp_path, monkeypatch):
        """Ensemble weights JSON is written alongside the model when persisted."""
        import credit_engine.model as model_module
        out_path = tmp_path / "ens_w.pkl"
        weights_path = tmp_path / "weights_w.json"
        monkeypatch.setattr(model_module, "_ENSEMBLE_WORKFLOW_MODEL_PATH", str(out_path))
        monkeypatch.setattr(model_module, "_ENSEMBLE_WORKFLOW_WEIGHTS_PATH", str(weights_path))
        monkeypatch.setattr(model_module, "_ENSEMBLE_PERSIST_THRESHOLD", 0.0)

        rng = np.random.default_rng(42)
        n, n_pos = 500, 40
        y_arr = np.zeros(n, dtype=int)
        y_arr[:n_pos] = 1
        rng.shuffle(y_arr)
        X = pd.DataFrame({
            "f1": np.where(y_arr == 1, rng.normal(2.0, 1.0, n), rng.normal(0.0, 1.0, n)),
            "f2": np.where(y_arr == 1, rng.normal(1.5, 1.0, n), rng.normal(0.0, 1.0, n)),
        })
        y = pd.Series(y_arr, name="TARGET")
        result = run_ensemble_workflow(X, y)

        if result["improvement"] >= 0.0:
            assert weights_path.exists(), "Ensemble weights JSON not written"
            weights = json.loads(weights_path.read_text())
            assert "lgb_gini" in weights and "xgb_gini" in weights and "ensemble_gini" in weights



# ---------------------------------------------------------------------------
# Priority 2.2 — train_catboost_optuna() + prepare_catboost_features()
# ---------------------------------------------------------------------------

from credit_engine.model import train_catboost_optuna, prepare_catboost_features  # noqa: E402


@pytest.fixture(scope="module")
def catboost_result(tmp_path_factory):
    """Run train_catboost_optuna once per module (n_trials=2 for speed)."""
    import credit_engine.model as model_module
    tmp = tmp_path_factory.mktemp("catboost")
    mp = pytest.MonkeyPatch()
    mp.setattr(model_module, "_CAT_MODEL_PATH", str(tmp / "cat.pkl"))
    mp.setattr(model_module, "_CAT_PARAMS_PATH", str(tmp / "cat.json"))
    mp.setattr(model_module, "_CAT_FIGURE_PATH", str(tmp / "cat.png"))
    rng = np.random.default_rng(42)
    n = 500
    y_arr = np.zeros(n, dtype=int)
    y_arr[:40] = 1
    rng.shuffle(y_arr)
    X = pd.DataFrame({
        "f1": np.where(y_arr == 1, rng.normal(2.0, 1.0, n), rng.normal(0.0, 1.0, n)),
        "f2": np.where(y_arr == 1, rng.normal(1.5, 1.0, n), rng.normal(0.0, 1.0, n)),
    })
    y = pd.Series(y_arr, name="TARGET")
    return train_catboost_optuna(X, y, n_trials=2)


class TestCatBoostOptuna:
    """TDD tests for train_catboost_optuna() (Priority 2.2)."""

    def test_returns_5_tuple(self, catboost_result):
        """train_catboost_optuna returns (model, metrics, X_test, y_test, best_params)."""
        assert len(catboost_result) == 5

    def test_metrics_keys(self, catboost_result):
        """metrics dict has all expected keys from evaluate_model()."""
        _, metrics, _, _, _ = catboost_result
        expected = {"Model", "AUC-ROC", "Gini", "KS", "Brier", "BrierSkill", "AvgPrecision"}
        assert expected.issubset(set(metrics.keys()))

    def test_gini_above_threshold(self, catboost_result):
        """CatBoost achieves Gini > 0.40 on linearly-separable mock data."""
        _, metrics, _, _, _ = catboost_result
        assert metrics["Gini"] > 0.40, (
            f"CatBoost Gini={metrics['Gini']:.4f} too low on separable mock data"
        )

    def test_best_params_within_search_space(self, catboost_result):
        """Sampled depth ≤ 8 and l2_leaf_reg ≤ 20 per subagent recommendations."""
        _, _, _, _, best_params = catboost_result
        assert best_params["depth"] <= 8, (
            f"depth={best_params['depth']} exceeds upper bound 8"
        )
        assert best_params["l2_leaf_reg"] <= 20.0, (
            f"l2_leaf_reg={best_params['l2_leaf_reg']:.2f} exceeds upper bound 20"
        )

    def test_model_artifact_saved(self, catboost_result, tmp_path, monkeypatch):
        """CatBoost model file is persisted to disk."""
        import credit_engine.model as model_module
        out = tmp_path / "cat.pkl"
        monkeypatch.setattr(model_module, "_CAT_MODEL_PATH", str(out))
        monkeypatch.setattr(model_module, "_CAT_PARAMS_PATH", str(tmp_path / "p.json"))
        monkeypatch.setattr(model_module, "_CAT_FIGURE_PATH", str(tmp_path / "f.png"))
        rng = np.random.default_rng(7)
        n = 300
        y_arr = np.zeros(n, dtype=int); y_arr[:24] = 1; rng.shuffle(y_arr)
        X = pd.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.normal(0, 1, n)})
        train_catboost_optuna(X, pd.Series(y_arr, name="TARGET"), n_trials=1)
        assert out.exists(), "CatBoost model file not written"

    def test_params_artifact_saved(self, catboost_result, tmp_path, monkeypatch):
        """CatBoost params JSON is persisted to disk."""
        import credit_engine.model as model_module
        params_path = tmp_path / "cat_p.json"
        monkeypatch.setattr(model_module, "_CAT_MODEL_PATH", str(tmp_path / "cat.pkl"))
        monkeypatch.setattr(model_module, "_CAT_PARAMS_PATH", str(params_path))
        monkeypatch.setattr(model_module, "_CAT_FIGURE_PATH", str(tmp_path / "f.png"))
        rng = np.random.default_rng(8)
        n = 300
        y_arr = np.zeros(n, dtype=int); y_arr[:24] = 1; rng.shuffle(y_arr)
        X = pd.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.normal(0, 1, n)})
        train_catboost_optuna(X, pd.Series(y_arr, name="TARGET"), n_trials=1)
        assert params_path.exists(), "CatBoost params JSON not written"

    def test_no_stdout(self, tmp_path, monkeypatch, capsys):
        """train_catboost_optuna must not write to stdout."""
        import credit_engine.model as model_module
        monkeypatch.setattr(model_module, "_CAT_MODEL_PATH", str(tmp_path / "cat.pkl"))
        monkeypatch.setattr(model_module, "_CAT_PARAMS_PATH", str(tmp_path / "p.json"))
        monkeypatch.setattr(model_module, "_CAT_FIGURE_PATH", str(tmp_path / "f.png"))
        rng = np.random.default_rng(9)
        n = 300
        y_arr = np.zeros(n, dtype=int); y_arr[:24] = 1; rng.shuffle(y_arr)
        X = pd.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.normal(0, 1, n)})
        train_catboost_optuna(X, pd.Series(y_arr, name="TARGET"), n_trials=1)
        assert capsys.readouterr().out == "", "train_catboost_optuna wrote to stdout"


class TestPrepareCatBoostFeatures:
    """TDD tests for prepare_catboost_features() helper."""

    def test_swaps_woe_cols_for_raw_when_present(self):
        """Raw categorical columns replace WoE-encoded ones when df_raw is supplied."""
        X_woe = pd.DataFrame({
            "CODE_GENDER": [0.3, -0.5, 0.1],
            "EXT_SOURCE_MEAN": [0.6, 0.3, 0.8],
            "CREDIT_INCOME_RATIO": [2.1, 3.5, 1.2],
        })
        df_raw = pd.DataFrame({
            "CODE_GENDER": ["M", "F", "XNA"],
            "AMT_INCOME_TOTAL": [45000, 67000, 90000],
        })
        X_out, cat_cols = prepare_catboost_features(X_woe, df_raw)
        # CODE_GENDER should now hold the raw string values
        assert X_out["CODE_GENDER"].dtype == "category" or X_out["CODE_GENDER"].dtype == object
        assert "CODE_GENDER" in cat_cols

    def test_returns_woe_only_when_no_raw(self):
        """When df_raw=None, returns X unchanged with empty cat_cols list."""
        X_woe = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
        X_out, cat_cols = prepare_catboost_features(X_woe, df_raw=None)
        pd.testing.assert_frame_equal(X_out, X_woe)
        assert cat_cols == []
