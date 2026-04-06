# Testing Patterns

**Analysis Date:** 2025-02-20

## Test Framework

**Runner:**
- pytest (Python's standard testing framework)
- Config: No explicit config file found; default pytest behavior applied
- Entry point via conftest.py at project root (aliases `src` → `credit_engine`)

**Assertion Library:**
- pytest's built-in assertions + NumPy testing utilities (`np.testing.assert_array_almost_equal`)
- Scikit-learn's cross-validation metrics for model evaluation assertions

**Run Commands:**
```bash
pytest tests/ -v              # Run all tests with verbose output
pytest tests/test_features.py -v  # Run single test module
pytest tests/test_features.py::test_division_by_zero_ratios_zero_denominator -v  # Run single test by name
pytest --cov=src --cov-report=term-missing  # Coverage report (not configured in files)
```

## Test File Organization

**Location:**
- Co-located pattern: `tests/test_<module_name>.py` mirrors `src/<module_name>.py`
- Test modules: `test_data_loader.py`, `test_auto_features.py`, `test_features.py`, `test_utils.py`, `test_model.py`
- Conftest: `conftest.py` at project root (manages imports via alias)

**Naming:**
- Test functions: `test_<function_name>_<behavior>` (e.g., `test_division_by_zero_ratios_zero_denominator`)
- Test classes: `Test<ModuleFunction>` (e.g., `TestGiniCoefficient`)
- Fixtures: lowercase with descriptive names (e.g., `mock_data`, `application_fixture`, `toy_data`)

**Structure:**
```
tests/
├── test_data_loader.py      # 815 lines, 32 tests
├── test_auto_features.py    # 876 lines, ~30 tests
├── test_features.py         # 1650 lines, ~60 tests
├── test_utils.py            # 466 lines, ~32 tests
└── test_model.py            # 1987 lines, ~150+ tests
conftest.py                  # 13 lines: sys.path + src alias
```

## Test Structure

**Module docstring pattern:**
Every test module starts with:
```python
"""
test_<module_name>.py
--------------------
Unit tests for credit_engine/<module_name>.py.

Run with
--------
    pytest tests/test_<module_name>.py -v
"""
```

**Test class organization (example from `test_utils.py`):**
```python
class TestGiniCoefficient:
    def test_toy_data_in_valid_range(self, toy_data):
        y_true, y_prob = toy_data
        gini = gini_coefficient(y_true, y_prob)
        assert 0 <= gini <= 1
    
    def test_perfect_separation_returns_one(self, perfect_data):
        y_true, y_prob = perfect_data
        gini = gini_coefficient(y_true, y_prob)
        assert gini == 1.0
```

**Test function patterns:**
- Arrange-Act-Assert (AAA) structure implicit
- Setup via fixtures (dependency injection)
- Clear assertion messages with context

Example from `test_features.py`:
```python
def test_division_by_zero_ratios_zero_denominator(application_fixture):
    """
    Rows where AMT_INCOME_TOTAL == 0 (row 2) or AMT_ANNUITY == 0 (row 3)
    must produce ratio == 0, not inf or NaN.
    """
    result = engineer_application_features(application_fixture)
    
    # Row 2: AMT_INCOME_TOTAL == 0 → CREDIT_INCOME_RATIO should be 0
    assert result.loc[2, "CREDIT_INCOME_RATIO"] == 0
```

## Fixtures

**Scope Management:**
- `scope="function"` (default): Fresh data for each test
- `scope="module"`: Reused across all tests in a module (marked explicitly)
- `scope="session"`: Shared across entire test suite (rare in this codebase)

**Example fixture definitions (from `test_model.py`):**
```python
@pytest.fixture(scope="module")
def mock_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    500-row, 2-feature, 8% positive rate — fast proxy for real WoE dataset.
    
    Features are linearly separable: positives drawn from N(2, 1),
    negatives from N(0, 1). LR achieves Gini ~ 0.65 on this data.
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
```

**Common fixtures by test module:**

**test_utils.py:**
- `toy_data`: 5 positive, 5 negative examples with clear separation
- `perfect_data`: Perfect separation (all positives 1.0, negatives 0.0)
- `random_data`: No discrimination (constant probability)
- `inverse_data`: Inverted predictions
- `imbalanced_data`: 8% default rate matching real dataset
- `tiny_clf`: Fitted LogisticRegression for predict_proba interface
- `close_figures`: Auto cleanup matplotlib figures after each test

**test_features.py:**
- `sample_df`: Minimal 3-row synthetic DataFrame
- `application_fixture`: 7-row synthetic Home Credit application table with edge cases
  - Row 0: normal applicant
  - Row 1: unemployment sentinel (DAYS_EMPLOYED == 365243)
  - Row 2: zero-income division guard
  - Row 3: zero-annuity division guard
  - Row 4: missing EXT_SOURCE
  - Row 5: partial EXT_SOURCE
  - Row 6: high-risk document flags
- `feature_store_data`: Full pipeline output with WoE mappings

**test_data_loader.py:**
- `_write_synthetic_csvs`: Helper that writes minimal CSV files to temp directory
- No reusable fixtures (each test writes its own synthetic data)

**test_model.py:**
- `mock_data` (module scope): 500 rows, 2 features, 8% positive rate
- `trained_model` (module scope): Reused fitted LogisticRegression baseline
- `benchmark_splits` (module scope): Train/test split from mock_data
- `benchmark_result` (module scope): Cached result of imbalance benchmark

## Mocking

**Framework:** unittest.mock + pytest.MonkeyPatch

**Patterns:**

**1. Patching model fit/predict:**
Used to avoid actual training in speed tests or to control random seeds.

Example from `test_model.py`:
```python
def test_threshold_search_uses_cv_validation_only(mock_data, monkeypatch):
    """Verify that threshold search does not leak training data into validation."""
    X, y = mock_data
    
    # Spy on _find_optimal_threshold_f1_macro calls
    original_find = model._find_optimal_threshold_f1_macro
    calls = []
    
    def spy_find(X_val, y_val, X_train, y_train, seed):
        calls.append((X_val, y_val, X_train, y_train))
        return original_find(X_val, y_val, X_train, y_train, seed)
    
    monkeypatch.setattr(model, "_find_optimal_threshold_f1_macro", spy_find)
    # ... rest of test
```

**2. Patching file I/O:**
Used to avoid writing large files during test runs.

Example from `test_data_loader.py`:
```python
def test_save_training_frame_creates_parquet(tmp_path):
    """save_training_frame writes X_features.parquet and y_target.parquet."""
    X = pd.DataFrame({"col": [1, 2, 3]})
    y = pd.Series([0, 1, 0])
    
    output_dir = tmp_path
    save_training_frame(X, y, output_dir=output_dir)
    
    assert (output_dir / "X_features.parquet").exists()
    assert (output_dir / "y_target.parquet").exists()
```

**3. Patching Optuna trials:**
Used in HPO tests to control trial counts and prevent long runs.

**What to Mock:**
- Expensive external dependencies (database, API, file I/O)
- Random number generation (when determinism is needed)
- Model fitting (in pure logic tests, not in integration tests)
- Callbacks and side effects

**What NOT to Mock:**
- Core algorithm logic (test the real implementation)
- DataFrame operations (pandas is battle-tested)
- NumPy array operations
- sklearn estimator interfaces
- Metrics (gini, KS, AUC)

## Test Types

**Unit Tests (85% of suite):**
- Scope: Single function or method in isolation
- Input: Small synthetic DataFrames or arrays
- Output: Checked against expected values or properties
- Location: Each function tested in `test_<module_name>.py`

Example from `test_features.py`:
```python
def test_division_by_zero_ratios_zero_denominator(application_fixture):
    """Rows with zero denominator must produce ratio == 0, not inf."""
    result = engineer_application_features(application_fixture)
    assert result.loc[2, "CREDIT_INCOME_RATIO"] == 0
```

**Integration Tests (10-15% of suite):**
- Scope: Multiple functions working together (e.g., feature pipeline)
- Input: Larger synthetic datasets (~500 rows)
- Output: Checked for consistency and property preservation

Example from `test_features.py`:
```python
def test_build_feature_store_no_nan_in_output(feature_store_data):
    """Complete feature store pipeline must eliminate all NaN values."""
    X, y = feature_store_data
    X_engineered = engineer_application_features(X)
    X_iv_filtered = select_features_by_iv(X_engineered, y, iv_threshold=0.02)
    
    X_features, woe_map = build_feature_store(X_iv_filtered, y)
    
    assert X_features.isna().sum().sum() == 0, "No NaN should remain after WoE transform"
```

**Property-Based Tests (Emerging pattern):**
- Check invariants rather than exact values
- Example from `test_features.py`:
```python
def test_engineer_application_features_no_nulls(application_fixture):
    """All engineered ratio columns must be non-null after transformation."""
    result = engineer_application_features(application_fixture)
    ratio_cols = ["CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM", "GOODS_CREDIT_RATIO"]
    for col in ratio_cols:
        assert result[col].notna().all(), f"{col} has null values"
```

**E2E Tests (Sparse):**
- Not formalized in test suite
- HPO tests (`test_train_xgboost_optuna`, `test_train_lightgbm_optuna`) are closest to E2E
- Verify full training pipeline on mock data with real models

## Common Testing Patterns

**Async Testing:**
- Not applicable (no async code in pipeline)

**Error Testing:**
Explicit `pytest.raises` context manager for exception validation.

Example from `test_data_loader.py`:
```python
def test_load_data_missing_file_raises(tmp_path):
    """load_data raises FileNotFoundError if application_train.csv is missing."""
    with pytest.raises(FileNotFoundError, match="missing.*application_train.csv"):
        load_data(tmp_path, mode="train")
```

Example from `test_model.py`:
```python
def test_train_xgboost_optuna_rejects_non_dataframe(mock_data):
    """X must be a DataFrame; numpy array raises TypeError."""
    X, y = mock_data
    X_array = X.values  # Convert to numpy array
    
    with pytest.raises(TypeError, match="must be a pd.DataFrame"):
        train_xgboost_optuna(X_array, y)
```

**Data Integrity Testing:**
Custom assertion helpers for joins (e.g., `_assert_no_row_multiplication`).

Example from `test_data_loader.py`:
```python
def test_load_data_no_duplicate_rows_after_join(tmp_path):
    """Multi-table joins must not inflate row count (Cartesian product check)."""
    _write_synthetic_csvs(tmp_path)
    df = load_data(tmp_path, mode="train")
    
    # After 7-table join, row count should match application table
    assert len(df) == 3  # 3 applicants from application_train.csv
```

**Numerical Stability Testing:**
Check for inf, NaN, and boundary conditions.

Example from `test_features.py`:
```python
def test_ext_source_std_all_nan_is_sentinel(application_fixture):
    """When all EXT_SOURCE values are NaN, std should be sentinel, not NaN."""
    result = engineer_application_features(application_fixture)
    # Row 4 has all EXT_SOURCE as NaN
    assert result.loc[4, "EXT_SOURCE_STD"] == -999.0  # Sentinel value
```

**Leakage Detection Testing:**
Explicit tests to catch temporal and data leakage.

Example from `test_model.py`:
```python
def test_smote_strategy_no_leakage(mock_data):
    """SMOTE must not leak training targets into synthetic minority."""
    X, y = mock_data
    
    # Capture SMOTE fit/resample calls
    original_fit_resample = SMOTE.fit_resample
    calls = []
    
    def spy_resample(self, X_train, y_train):
        calls.append((len(X_train), len(y_train)))
        return original_fit_resample(self, X_train, y_train)
    
    monkeypatch.setattr(SMOTE, "fit_resample", spy_resample)
    # ... assert calls don't include test indices
```

## Coverage

**Requirements:** 80%+ line coverage on library code (src/)

**View Coverage:**
```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

**Current status:** ~188 tests across 5 modules (as of 2026-04-05 git log)
- `test_data_loader.py`: 32 tests (~95% coverage)
- `test_features.py`: ~60 tests (~93% coverage on features.py)
- `test_utils.py`: ~32 tests (~97% coverage on utils.py)
- `test_model.py`: 150+ tests (coverage varies by subfunction)
- `test_auto_features.py`: ~30 tests (optional DFS module)

## Test Execution Characteristics

**Speed Profile:**
- Fast suite (unit tests): `test_data_loader.py`, `test_features.py`, `test_utils.py` → complete in <10 seconds
- Slow suite (model training): `test_model.py` → 30-45 minutes due to:
  - Optuna HPO with 50 trials per function
  - Imbalance benchmark with 4 strategies × 5-fold CV
  - Ensemble training
  - Leakage validation tests (4-5 mins each)

**Resource Management:**
- Matplotlib figures closed explicitly after each test: `plt.close("all")` in autouse fixture
- Large DataFrames loaded once per module (scope="module" fixtures)
- Temporary files cleaned by pytest's tmp_path fixture

**Fixture Scope Bug (Historical):**
Fixed in 2026-04-05 session: function-scoped fixtures reused 8× for benchmark tests → suite hung.
Solution: Upgrade to module scope where appropriate + use MonkeyPatch directly in tests.

## Test Data Patterns

**Synthetic data design (from CLAUDE.md):**
- Minimal size: 3-7 rows for unit tests, 500 rows for HPO tests
- Linearly separable: positives N(2, 1), negatives N(0, 1) → LR Gini ~0.65
- 8% positive rate matching real dataset imbalance
- Edge cases embedded: zero denominators, missing values, sentinels, outliers
- Deterministic random seed: `np.random.default_rng(42)` for reproducibility

**DataFrame construction patterns:**
```python
@pytest.fixture
def application_fixture() -> pd.DataFrame:
    """Synthetic Home Credit application table with edge cases."""
    data = {
        "AMT_CREDIT": [500_000, 300_000, ...],
        "DAYS_EMPLOYED": [-2_000, 365_243, ...],  # Row 1: unemployment sentinel
        "EXT_SOURCE_1": [0.6, 0.4, ..., np.nan, ...],  # Row 4: missing
    }
    return pd.DataFrame(data)
```

**Temporary file handling:**
```python
def test_save_training_frame_creates_parquet(tmp_path):
    """Use pytest's tmp_path fixture for isolated file I/O."""
    output_dir = tmp_path  # Automatically cleaned after test
    save_training_frame(X, y, output_dir=output_dir)
    assert (output_dir / "X_features.parquet").exists()
```

## CI/CD Integration

**Test Execution (No CI pipeline configured in repo):**
- Local pytest only (GitHub Actions not detected)
- Pre-commit hooks: nbstripout for .ipynb files (configured via .gitattributes)

**Coverage Enforcement:**
- Manual via pytest flags (not enforced in hooks or CI)

---

*Testing analysis: 2025-02-20*
