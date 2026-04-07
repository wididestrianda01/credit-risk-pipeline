"""conftest.py — adds repo root to sys.path and exposes src/ as credit_engine."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

# Add project root so `import src` resolves.
sys.path.insert(0, str(Path(__file__).parent))

# Alias src/ as credit_engine so existing imports (from credit_engine.X import ...)
# continue to work while the source tree lives in src/.
import src  # noqa: E402
sys.modules["credit_engine"] = src


# ---------------------------------------------------------------------------
# Wave 0 Fixtures: EXT_SOURCE imputation, combined features, baseline
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ext_source_imputer_model() -> tuple:
    """Mock imputer model for EXT_SOURCE_3 supervised imputation.

    Returns (fitted_mock_imputer, correlation_value) tuple:
    - fitted_mock_imputer: LinearRegression trained on 100 mock rows
    - correlation_value: 0.62 (stable cross-fold imputation)
    """
    rng = np.random.default_rng(42)
    X_train_mock = rng.normal(0, 1, (100, 5))
    y_train_mock = rng.normal(0, 1, 100)

    imputer = LinearRegression()
    imputer.fit(X_train_mock, y_train_mock)

    correlation_value = 0.62
    return (imputer, correlation_value)


@pytest.fixture(scope="module")
def X_combined_features_test() -> pd.DataFrame:
    """Mock combined feature store for testing.

    Returns 1000-row DataFrame with 70 columns:
    - 62 raw features
    - 1 EXT_SOURCE_3_MISSING_FLAG
    - 7 synthetic DFS features
    - All NaN replaced with -999 sentinel
    """
    rng = np.random.default_rng(42)
    n_rows = 1000
    n_raw_features = 62
    n_dfs_features = 7

    # Raw features (62 columns)
    raw_features = {f"raw_feature_{i}": rng.normal(0, 1, n_rows) for i in range(n_raw_features)}

    # EXT_SOURCE_3_MISSING_FLAG
    ext_source_flag = rng.choice([0, 1], size=n_rows, p=[0.7, 0.3])

    # DFS synthetic features (7 columns)
    dfs_features = {f"dfs_feature_{i}": rng.normal(0, 1, n_rows) for i in range(n_dfs_features)}

    # Combine
    X = pd.DataFrame({**raw_features, "EXT_SOURCE_3_MISSING_FLAG": ext_source_flag, **dfs_features})

    # Fill NaN with -999 sentinel
    X = X.fillna(-999.0)

    return X


@pytest.fixture(scope="module")
def baseline_X_raw() -> pd.DataFrame:
    """Mock baseline raw features (no DFS, no imputation).

    Returns 500-row DataFrame with 62 columns:
    - All raw features
    - Some rows have EXT_SOURCE_3 = -999 (missing sentinel)
    """
    rng = np.random.default_rng(42)
    n_rows = 500
    n_features = 62

    X = pd.DataFrame({f"raw_feature_{i}": rng.normal(0, 1, n_rows) for i in range(n_features)})

    # Introduce -999 sentinels for some rows in a feature (simulating missing data)
    missing_mask = rng.choice([False, True], size=n_rows, p=[0.8, 0.2])
    if "raw_feature_0" in X.columns:
        X.loc[missing_mask, "raw_feature_0"] = -999.0

    return X


# ---------------------------------------------------------------------------
# Session-Scoped Fixtures: Test Data Isolation
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def mock_data_dir() -> Path:
    """
    Session-scoped temporary directory for test feature store writes.

    Provides a temporary directory that tests can pass to feature store functions
    via the `output_dir` parameter. This directory is automatically deleted after
    the test session completes.

    Yields
    ------
    Path
        Absolute path to a temporary directory with data/processed/ and models/ subdirectories.

    Usage
    -----
    In a test function:

        def test_build_features(mock_data_dir):
            X = pd.DataFrame({"A": [1, 2, 3]})
            y = pd.Series([0, 1, 0])
            X_out, _ = build_feature_store(X, y, output_dir=mock_data_dir / "data" / "processed")
            assert (mock_data_dir / "data" / "processed" / "X_features.parquet").exists()
    """
    with tempfile.TemporaryDirectory(prefix="test_credit_engine_") as tmpdir:
        tmppath = Path(tmpdir)
        # Pre-create subdirectories expected by feature store functions
        (tmppath / "data" / "processed").mkdir(parents=True, exist_ok=True)
        (tmppath / "models").mkdir(parents=True, exist_ok=True)
        yield tmppath


@pytest.fixture(scope="session")
def mock_data_subdir(mock_data_dir: Path) -> Path:
    """
    Shortcut fixture that returns the data/processed/ subdirectory of mock_data_dir.

    Useful when a test wants to pass the data/ directory directly:

        def test_something(mock_data_subdir):
            X_out, _ = build_feature_store(X, y, output_dir=mock_data_subdir)
    """
    return mock_data_dir / "data" / "processed"
