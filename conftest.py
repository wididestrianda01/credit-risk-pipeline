"""conftest.py — adds repo root to sys.path and provides test fixtures."""
# Register custom pytest marks to suppress PytestUnknownMarkWarning.
# See pytest docs: https://docs.pytest.org/en/stable/how-to/mark.html
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (excluded by -m 'not slow')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests (require full models)")
    config.addinivalue_line("markers", "unit: marks tests as unit tests (isolated, no dependencies)")

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

# Add project root so `import src` resolves.
sys.path.insert(0, str(Path(__file__).parent))


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


# ---------------------------------------------------------------------------
# Wave 0: XGBoost Raw Features HPO (Phase 04.2.3)
# ---------------------------------------------------------------------------


@pytest.fixture
def make_mock_parquet(tmp_path: Path):
    """
    Factory fixture to create mock X_tree_dfs.parquet files with TARGET column.

    This fixture creates mock parquet files suitable for testing
    train_xgboost_optuna(feature_store_path: str, ...) which loads
    parquet from disk instead of accepting DataFrames.

    Parameters
    ----------
    tmp_path : Path
        pytest's temporary directory fixture.

    Yields
    ------
    callable
        A factory function with signature:
            _factory(n_rows: int = 500, n_pos_frac: float = 0.08, n_features: int = 10) -> Path

        Returns the path to the created parquet file.

    Example
    -------
    >>> def test_example(make_mock_parquet):
    ...     parquet_path = make_mock_parquet(n_rows=500, n_features=10)
    ...     model, metrics, X_test, y_test, params = train_xgboost_optuna(
    ...         str(parquet_path), n_trials=2
    ...     )
    ...     assert metrics["Gini"] > 0
    """

    def _factory(
        n_rows: int = 500, n_pos_frac: float = 0.08, n_features: int = 10
    ) -> Path:
        """
        Create a mock X_tree_dfs.parquet file in tmp_path.

        Parameters
        ----------
        n_rows : int, optional
            Number of rows in the mock dataset (default 500).
        n_pos_frac : float, optional
            Fraction of positive (default 1) samples (default 0.08 for imbalance).
        n_features : int, optional
            Number of features to generate (default 10).

        Returns
        -------
        Path
            Path to the created parquet file at tmp_path / "X_tree_dfs.parquet".
        """
        rng = np.random.default_rng(42)
        n_pos = int(n_rows * n_pos_frac)

        # Create target: 8% positive, rest negative
        y_arr = np.zeros(n_rows, dtype=int)
        y_arr[:n_pos] = 1
        rng.shuffle(y_arr)

        # Create linearly separable features
        # Positive samples centered at 1.0, negative at 0.0
        X = pd.DataFrame(
            {
                f"f{i}": np.where(
                    y_arr == 1,
                    rng.normal(1.0, 1.0, n_rows),
                    rng.normal(0.0, 1.0, n_rows),
                )
                for i in range(n_features)
            }
        )

        # Add temporal sort column (required for OOT split in train_xgboost_optuna)
        # SK_ID_CURR is the application ID and is monotonically increasing
        X["SK_ID_CURR"] = np.arange(1, n_rows + 1, dtype=int)

        # Add TARGET column
        X["TARGET"] = y_arr

        # Save to parquet
        path = tmp_path / "X_tree_dfs.parquet"
        X.to_parquet(path)

        return path

    return _factory


# ---------------------------------------------------------------------------
# Wave 1 Fixtures: Delinquency Trajectory Features (Phase 04.2.7)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def df_inst_fixture() -> pd.DataFrame:
    """
    Mock installments_payments table with 50 applicants and varying DPD patterns.

    This fixture provides ~300 rows of instalment payment history data with
    realistic delinquency patterns:
    - Clean history (10 applicants): no DPD
    - Recent 30+DPD (15 applicants): DPD > 30 in last 30 days
    - Historical DPD (10 applicants): DPD > 30 from 180+ days ago
    - Escalating DPD (10 applicants): older clean, recent DPD
    - Single instalment (5 applicants): edge case, only 1 record

    Columns
    -------
    SK_ID_CURR : int
        Applicant ID (1-50)
    DAYS_INSTALMENT : float
        Days scheduled instalment was due (negative, relative to application date)
    DAYS_ENTRY_PAYMENT : float
        Days the payment was actually made (negative)
        DPD = max(0, DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT)
    AMT_INSTALMENT : float
        Amount due for the instalment
    AMT_PAYMENT : float
        Amount actually paid
    """
    rng = np.random.default_rng(42)
    rows = []

    # ==================== Group 1: Clean history (10 applicants) ====================
    for sk_id in range(1, 11):
        # 8-12 instalments, all on-time
        n_instal = rng.integers(8, 13)
        for i in range(n_instal):
            days_instal = -30 * (n_instal - i)  # Spread over last 300 days
            days_entry = days_instal - rng.integers(-5, 6)  # On-time ±5 days (negative)
            amt_instal = 5000 + rng.normal(0, 500)
            amt_payment = amt_instal + rng.normal(0, 200)  # Usually on-time, slight overpayment
            rows.append({
                "SK_ID_CURR": sk_id,
                "DAYS_INSTALMENT": days_instal,
                "DAYS_ENTRY_PAYMENT": days_entry,
                "AMT_INSTALMENT": amt_instal,
                "AMT_PAYMENT": amt_payment,
            })

    # ==================== Group 2: Recent 30+DPD (15 applicants) ====================
    for sk_id in range(11, 26):
        # 6-10 instalments with recent DPD
        n_instal = rng.integers(6, 11)
        for i in range(n_instal):
            days_instal = -30 * (n_instal - i)
            # Last 2-3 instalments have significant DPD (40-60 days late)
            if i >= n_instal - rng.integers(2, 4):
                dpd_days = rng.integers(40, 61)
                days_entry = days_instal + dpd_days  # Late payment
            else:
                days_entry = days_instal - rng.integers(-5, 6)
            amt_instal = 4500 + rng.normal(0, 500)
            amt_payment = amt_instal * rng.uniform(0.7, 1.0)  # Partial or late payment
            rows.append({
                "SK_ID_CURR": sk_id,
                "DAYS_INSTALMENT": days_instal,
                "DAYS_ENTRY_PAYMENT": days_entry,
                "AMT_INSTALMENT": amt_instal,
                "AMT_PAYMENT": amt_payment,
            })

    # ==================== Group 3: Historical DPD (10 applicants) ====================
    for sk_id in range(26, 36):
        # 7-12 instalments, DPD was 180+ days ago, recent on-time
        n_instal = rng.integers(7, 13)
        for i in range(n_instal):
            days_instal = -30 * (n_instal - i)
            # Early instalments (7+ months ago) had DPD
            if i < rng.integers(2, 4):
                dpd_days = rng.integers(30, 90)
                days_entry = days_instal + dpd_days
            else:
                days_entry = days_instal - rng.integers(-5, 6)
            amt_instal = 5500 + rng.normal(0, 500)
            amt_payment = amt_instal + rng.normal(0, 200)
            rows.append({
                "SK_ID_CURR": sk_id,
                "DAYS_INSTALMENT": days_instal,
                "DAYS_ENTRY_PAYMENT": days_entry,
                "AMT_INSTALMENT": amt_instal,
                "AMT_PAYMENT": amt_payment,
            })

    # ==================== Group 4: Escalating DPD (10 applicants) ====================
    for sk_id in range(36, 46):
        # 8-11 instalments, clean early, worsening recent
        n_instal = rng.integers(8, 12)
        for i in range(n_instal):
            days_instal = -30 * (n_instal - i)
            # Escalation: early on-time, recent increasingly late
            if i < n_instal // 2:
                days_entry = days_instal - rng.integers(-5, 6)
            else:
                # Worsening trend: 30, 45, 60 days late
                dpd_days = 30 + 15 * (i - n_instal // 2)
                days_entry = days_instal + min(dpd_days, 90)
            amt_instal = 4800 + rng.normal(0, 500)
            amt_payment = amt_instal * rng.uniform(0.6, 1.0)
            rows.append({
                "SK_ID_CURR": sk_id,
                "DAYS_INSTALMENT": days_instal,
                "DAYS_ENTRY_PAYMENT": days_entry,
                "AMT_INSTALMENT": amt_instal,
                "AMT_PAYMENT": amt_payment,
            })

    # ==================== Group 5: Single instalment (5 applicants) ====================
    for sk_id in range(46, 51):
        # Edge case: only 1 instalment record (new credit or data cutoff)
        days_instal = -30
        days_entry = days_instal - rng.integers(-5, 6)
        amt_instal = 3000 + rng.normal(0, 500)
        amt_payment = amt_instal + rng.normal(0, 200)
        rows.append({
            "SK_ID_CURR": sk_id,
            "DAYS_INSTALMENT": days_instal,
            "DAYS_ENTRY_PAYMENT": days_entry,
            "AMT_INSTALMENT": amt_instal,
            "AMT_PAYMENT": amt_payment,
        })

    df = pd.DataFrame(rows)
    df = df.astype({
        "SK_ID_CURR": int,
        "DAYS_INSTALMENT": float,
        "DAYS_ENTRY_PAYMENT": float,
        "AMT_INSTALMENT": float,
        "AMT_PAYMENT": float,
    })
    return df


@pytest.fixture(scope="module")
def df_bureau_balance_fixture() -> pd.DataFrame:
    """
    Mock bureau_balance table with credit bureau account-level delinquency status.

    This fixture provides ~500 rows of bureau balance history for the same
    50 applicants used in df_inst_fixture. Each row represents a monthly
    snapshot of an account's delinquency status.

    Columns
    -------
    SK_ID_BUREAU : int
        Unique identifier for a credit bureau account (1-200)
    SK_ID_CURR : int
        Applicant ID (1-50), linked to df_inst_fixture
    MONTHS_BALANCE : int
        Month relative to application date (-60 to 0, where 0 is most recent)
    STATUS : str
        Delinquency status encoding:
        - '0': Current (no DPD)
        - '1': 1-30 days past due
        - '2': 31-60 days past due
        - '3': 61-90 days past due
        - '4': 90+ days past due
        - 'X': Closed account
        - 'C': Closed with 0 balance

    Notes
    -----
    Each SK_ID_CURR can have multiple accounts (SK_ID_BUREAU values).
    The STATUS distribution matches DPD patterns from df_inst_fixture:
    - Clean applicants: mostly '0' (current)
    - Recent DPD applicants: '1' or '2' in recent months
    - Historical DPD applicants: '0' in recent months, higher DPD in old months
    """
    rng = np.random.default_rng(42)
    rows = []

    # Create 4-6 accounts per applicant (some have multiple credit accounts)
    for sk_id in range(1, 51):
        n_accounts = rng.integers(4, 7)  # 4-6 accounts per applicant
        for account_num in range(n_accounts):
            bureau_id = sk_id * 10 + account_num  # Simple ID generation
            # Each account has 15-45 monthly snapshots
            n_months = rng.integers(15, 46)
            months = sorted(rng.choice(np.arange(-60, 1), size=n_months, replace=False))

            # Determine this applicant's DPD pattern
            if sk_id <= 10:
                # Clean: mostly '0'
                status_dist = {'0': 0.95, '1': 0.03, '2': 0.02, '3': 0, '4': 0, 'X': 0, 'C': 0}
            elif sk_id <= 25:
                # Recent DPD: recent months high DPD, mix otherwise
                status_dist = {'0': 0.60, '1': 0.20, '2': 0.10, '3': 0.05, '4': 0.03, 'X': 0.01, 'C': 0.01}
            elif sk_id <= 35:
                # Historical DPD: old months high DPD, recent clean
                status_dist = {'0': 0.85, '1': 0.05, '2': 0.05, '3': 0.03, '4': 0.01, 'X': 0.01, 'C': 0}
            elif sk_id <= 45:
                # Escalating: mix of all statuses with worsening trend
                status_dist = {'0': 0.50, '1': 0.20, '2': 0.15, '3': 0.08, '4': 0.05, 'X': 0.01, 'C': 0.01}
            else:
                # New accounts: limited history, mostly clean
                status_dist = {'0': 0.90, '1': 0.05, '2': 0.03, '3': 0.01, '4': 0.01, 'X': 0, 'C': 0}

            for month_balance in months:
                # Status varies slightly with recency for realistic patterns
                if sk_id <= 35 and month_balance < -180 and sk_id > 10:
                    # Historical DPD pattern: high DPD in far past
                    status = rng.choice(['0', '1', '2', '3', '4'], p=[0.4, 0.25, 0.2, 0.1, 0.05])
                else:
                    status = rng.choice(list(status_dist.keys()), p=list(status_dist.values()))

                rows.append({
                    "SK_ID_BUREAU": bureau_id,
                    "SK_ID_CURR": sk_id,
                    "MONTHS_BALANCE": month_balance,
                    "STATUS": status,
                })

    df = pd.DataFrame(rows)
    df = df.astype({
        "SK_ID_BUREAU": int,
        "SK_ID_CURR": int,
        "MONTHS_BALANCE": int,
        "STATUS": str,
    })
    return df


# ---------------------------------------------------------------------------
# Wave 0: SHAP Explainability (Phase 04.3)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def catboost_shap_fixture():
    """
    Real mini-CatBoost model trained on 200 synthetic rows.

    No mocking — tests must verify booster extraction and SHAP value shapes
    against a real model. Training takes < 5 seconds.

    Returns
    -------
    tuple
        (model, explainer, shap_values, X_oot_mini, y_mini)
        where:
        - model: CalibratedClassifierCV(FrozenEstimator(CatBoostClassifier))
        - explainer: shap.TreeExplainer
        - shap_values: shap.Explanation object (n_samples=200, n_features=~20)
        - X_oot_mini: DataFrame (200 × ~20 cols, synthetic features)
        - y_mini: Series (200 labels, binary 0/1)
    """
    import shap
    from catboost import CatBoostClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    rng = np.random.default_rng(42)
    n_samples = 200
    n_features = 20

    # Generate synthetic features
    X_mini = pd.DataFrame({
        f"feature_{i}": rng.normal(0, 1, n_samples)
        for i in range(n_features)
    })

    # Synthetic target: imbalanced binary (8% positive, 92% negative, matching production)
    y_mini = pd.Series(
        rng.choice([0, 1], size=n_samples, p=[0.92, 0.08]),
        name="target"
    )

    # Train raw CatBoost (no calibration yet)
    cat_model = CatBoostClassifier(
        iterations=50,
        depth=4,
        learning_rate=0.1,
        verbose=False,
        random_state=42,
    )
    cat_model.fit(X_mini, y_mini)

    # Wrap in CalibratedClassifierCV with FrozenEstimator (production pattern)
    frozen_estimator = FrozenEstimator(cat_model)
    calibrated_model = CalibratedClassifierCV(frozen_estimator)
    # Platt calibration on same data (for testing; production would use separate fold)
    calibrated_model.fit(X_mini, y_mini)

    # Extract raw booster for SHAP (per D-01)
    raw_booster = calibrated_model.calibrated_classifiers_[0].estimator.estimator

    # Create SHAP explainer with model_output="raw"
    explainer = shap.TreeExplainer(raw_booster, model_output="raw")

    # Compute SHAP values on mini data
    shap_values = explainer(X_mini)

    return (calibrated_model, explainer, shap_values, X_mini, y_mini)


# End of conftest.py
# catboost_shap_fixture is available module-scoped for all test_explain.py tests
