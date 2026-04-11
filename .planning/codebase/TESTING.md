# Testing Patterns

**Analysis Date:** 2026-04-11

## Test Framework

**Runner:**
- **pytest** ≥ 8.2 (from `requirements.txt`)
- Config file: `conftest.py` at project root (not `pytest.ini` or `setup.cfg`)
- Markers registered: `pytest.mark.slow`, `pytest.mark.regression`, `pytest.mark.unit` (defined in `conftest.py` lines 4–5)

**Assertion Library:**
- `pytest` built-in assertions: `assert`, `assert not`, `assert x == y`, `assert isinstance()`
- NumPy testing: `np.testing.assert_allclose()`, `np.testing.assert_array_almost_equal()`
- Pytest raises: `pytest.raises(ValueError)`, `pytest.raises(FileNotFoundError)`

**Run Commands:**
```bash
# Run all tests (416 tests total as of 2026-04-11)
pytest tests/ -v

# Fast suite only (skip @pytest.mark.slow tests)
pytest tests/ -v -m "not slow"

# Run a single test by name
pytest tests/test_features.py::test_build_features_returns_dataframe -v

# Run all tests in a module
pytest tests/test_model.py -v

# Coverage report
pytest tests/ --cov=src --cov-report=term-missing --cov-report=html
```

## Test File Organization

**Location:**
- All tests co-located in `tests/` directory (not alongside source)
- Pattern: one test file per source module: `test_features.py` for `src/features.py`
- Shared fixtures in `conftest.py` at project root

**Naming:**
- Test files: `test_<module>.py`
- Test functions: `test_<function>_<scenario>`, e.g., `test_gini_coefficient_returns_float()`, `test_division_by_zero_ratios_zero_denominator()`
- Test classes: `Test<FunctionName>`, e.g., `TestGiniCoefficient`, `TestEngineerApplicationFeatures`
- Classes group related tests for one function or concept

**Structure:**
```
tests/
├── conftest.py              # Shared fixtures + pytest marker registration
├── test_data_loader.py      # Tests for src/data_loader.py
├── test_features.py         # Tests for src/features.py (115K+ lines)
├── test_model.py            # Tests for src/model.py (153K+ lines)
├── test_utils.py            # Tests for src/utils.py
├── test_auto_features.py    # Tests for src/auto_features.py (55K+ lines)
└── test_streak_evaluation.py # Tests for streak delinquency features
```

## Test Structure

**AAA Pattern (Arrange-Act-Assert):**
```python
def test_gini_coefficient_returns_float(toy_data):
    # Arrange: get fixture data
    y_true, y_prob = toy_data
    
    # Act: call function under test
    result = gini_coefficient(y_true, y_prob)
    
    # Assert: verify expected outcome
    assert isinstance(result, float)
```

**Minimal AAA (for simple assertions):**
```python
def test_build_features_no_rows_dropped(sample_df):
    result = build_features(sample_df)
    assert len(result) == len(sample_df)
```

**With Exception Handling:**
```python
def test_raises_on_single_class():
    y_true = np.array([0, 0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError):
        gini_coefficient(y_true, y_prob)
```

**Patterns:**
- Setup via fixtures: dependencies injected as function parameters
- Teardown: `autouse=True` fixtures clean up (e.g., redirect paths, reset state)
- Assertion style: simple comparisons preferred; NumPy testing for numerical arrays

## Mocking

**Framework:**
- `unittest.mock.patch()` (standard library) for monkeypatching
- `pytest.MonkeyPatch` fixture for test-scoped patches
- `pytest.mark.skip()`, `pytest.mark.skipif()` for conditional test skipping

**Patterns observed in `test_model.py`:**

```python
# Session-wide autouse fixture for XGBoost thread safety
@pytest.fixture(scope="session", autouse=True)
def _force_xgb_single_thread():
    """Patch XGBClassifier.__init__ to default nthread=1 for every test."""
    import xgboost as _xgb
    _original = _xgb.XGBClassifier.__init__
    
    @functools.wraps(_original)
    def _patched(self, *args, **kwargs):
        kwargs.setdefault("nthread", 1)
        _original(self, *args, **kwargs)
    
    _xgb.XGBClassifier.__init__ = _patched
    yield
    _xgb.XGBClassifier.__init__ = _original

# Monkeypatch to redirect HPO progress log
@pytest.fixture(autouse=True)
def _redirect_hpo_progress_log(monkeypatch, tmp_path):
    """Redirect _HPO_PROGRESS_LOG_PATH to tmp_path to avoid contaminating production logs."""
    import src.model as _model
    monkeypatch.setattr(_model, "_HPO_PROGRESS_LOG_PATH", str(tmp_path / "hpo_progress.jsonl"))
```

**What to Mock:**
- External dependencies you can't control: file I/O paths, random seeds, environment state
- Heavy operations in unit tests: full Optuna HPO (mock with quick n_trials=2)
- Backend configuration: XGBoost thread pool (monkeypatch before import)

**What NOT to Mock:**
- Core domain logic: if testing a model, fit the model (don't mock predictions)
- Data transformations: actual feature engineering must run
- Library functions you depend on: sklearn, pandas, numpy (they're tested by their packages)

## Fixtures and Factories

**Fixture scope hierarchy (from `conftest.py`):**

```python
# Session scope — created once, reused across entire test suite
@pytest.fixture(scope="session")
def mock_data_dir() -> Path:
    """Temporary directory with data/processed/ and models/ subdirectories."""
    with tempfile.TemporaryDirectory(prefix="test_credit_engine_") as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / "data" / "processed").mkdir(parents=True, exist_ok=True)
        (tmppath / "models").mkdir(parents=True, exist_ok=True)
        yield tmppath

# Module scope — created once, reused across module tests
@pytest.fixture(scope="module")
def mock_data() -> tuple[pd.DataFrame, pd.Series]:
    """500-row, 2-feature, 8% positive rate — linearly separable."""
    rng = np.random.default_rng(42)
    n, n_pos = 500, 40
    y = np.zeros(n, dtype=int)
    y[:n_pos] = 1
    rng.shuffle(y)
    X = pd.DataFrame({
        "f1": np.where(y == 1, rng.normal(2.0, 1.0, n), rng.normal(0.0, 1.0, n)),
        "f2": np.where(y == 1, rng.normal(1.5, 1.0, n), rng.normal(0.0, 1.0, n)),
        "prev_days_decision_mean": np.arange(n, dtype=float),  # Temporal sort column
    })
    return X, pd.Series(y, name="TARGET")

@pytest.fixture(scope="module")
def trained_model(mock_data):
    """Logistic baseline trained once; reused across structural tests."""
    X, y = mock_data
    return train_logistic_baseline(X, y)

# Function scope (default) — fresh per test
@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal synthetic DataFrame for feature tests."""
    return pd.DataFrame({
        "loan_amount": [10_000, 25_000, 5_000],
        "income": [30_000, 60_000, 15_000],
        "age": [35, 52, 28],
        "default_flag": [0, 0, 1],
    })

# Factory fixture — callable, creates many instances
@pytest.fixture
def make_mock_parquet(tmp_path: Path):
    """Factory to create mock X_tree_dfs.parquet files with TARGET column."""
    def _factory(
        n_rows: int = 500, 
        n_pos_frac: float = 0.08, 
        n_features: int = 10
    ) -> Path:
        rng = np.random.default_rng(42)
        n_pos = int(n_rows * n_pos_frac)
        y_arr = np.zeros(n_rows, dtype=int)
        y_arr[:n_pos] = 1
        rng.shuffle(y_arr)
        
        X = pd.DataFrame({
            f"f{i}": np.where(
                y_arr == 1,
                rng.normal(1.0, 1.0, n_rows),
                rng.normal(0.0, 1.0, n_rows),
            )
            for i in range(n_features)
        })
        X["prev_days_decision_mean"] = np.arange(n_rows, dtype=float)
        X["TARGET"] = y_arr
        
        path = tmp_path / "X_tree_dfs.parquet"
        X.to_parquet(path)
        return path
    
    return _factory
```

**Wave 1 fixtures (Phase 04.2.7, from `conftest.py`):**

```python
@pytest.fixture(scope="module")
def df_inst_fixture() -> pd.DataFrame:
    """
    Mock installments_payments with 50 applicants and varying DPD patterns:
    - Clean history (10 applicants)
    - Recent 30+DPD (15 applicants)
    - Historical DPD (10 applicants)
    - Escalating DPD (10 applicants)
    - Single instalment (5 applicants)
    """
    # 300+ rows of payment history with realistic delinquency patterns
    ...

@pytest.fixture(scope="module")
def df_bureau_balance_fixture() -> pd.DataFrame:
    """
    Mock bureau_balance with credit bureau account-level delinquency status.
    ~500 rows of monthly snapshots for the 50 applicants.
    """
    # Bureau STATUS encoding: '0' (current), '1' (1-30 DPD), '2' (31-60 DPD), etc.
    ...
```

**Location:**
- Module-level fixtures: in `conftest.py` (shared across all tests)
- Test-specific fixtures: at top of test file (used by single test module only)
- Synthetic data builders: helper functions like `_write_synthetic_csvs()` called within fixtures

**Scope best practices:**
- `scope="function"` (default): fresh fixture per test — use for data that tests may mutate
- `scope="module"`: fixture created once, reused across module tests — use for expensive setup (model training, HPO)
- **Critical gotcha:** Module-scoped fixtures must return fresh copies on each access (avoid in-place mutations in tests)

## Coverage

**Requirements:**
- Target: 80%+ coverage on `src/`
- Measured with: `pytest --cov=src --cov-report=term-missing`

**View Coverage:**
```bash
# Terminal output with missing lines
pytest tests/ --cov=src --cov-report=term-missing

# HTML report (open in browser)
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

**Blocks to 100%:**
- Stub functions: `roc_curve_plot()`, `calibration_plot()` in `src/utils.py`
- Error paths: file I/O errors in `src/data_loader.py`
- Slow model tests: full HPO runs (use `@pytest.mark.slow` to skip in CI)

## Test Types

**Unit Tests:**
- Scope: Single function in isolation
- Examples: `test_gini_coefficient_returns_float()`, `test_engineer_application_features_creates_all_columns()`, `test_division_by_zero_ratios_zero_denominator()`
- Data: Small synthetic DataFrames (< 1K rows, 7 rows for edge case testing)
- Speed: < 1 second per test
- Assertions: Value correctness, type checking, error raising, column presence
- Location: Most tests in `test_features.py`, `test_utils.py`, `test_data_loader.py`

**Integration Tests:**
- Scope: Multiple functions working together
- Examples: `test_load_data_one_row_per_sk_id_curr()` (reads CSVs + joins), `test_train_logistic_baseline_returns_6_tuple()` (train + evaluate)
- Data: Realistic schemas (~3 applicants, all 7 joined tables or mock feature store)
- Speed: 1–5 seconds per test
- Assertions: End-to-end output shape, column names, no nulls where not expected
- Location: `test_data_loader.py`, `test_model.py`, `test_auto_features.py`

**E2E Tests:**
- Scope: Full pipeline from raw data to model
- Examples: Not implemented in fast test suite (would require full dataset)
- Data: Production dataset (not suitable for fast CI)
- Speed: Minutes to hours
- Assertions: Final Gini ≥ 0.55, feature counts match, model artefacts exist
- Location: Would live in separate `tests/test_e2e.py` (manual execution only)

**Marked tests (@pytest.mark):**
- `@pytest.mark.slow` — tests that take > 5 seconds; excluded by `pytest tests/ -m "not slow"`
- `@pytest.mark.unit` — explicit unit test marker (informational)
- `@pytest.mark.regression` — regression suite marker
- `@pytest.mark.skip(reason="...")` — tests not yet implemented

## Common Patterns

**Error Testing:**
```python
def test_raises_on_single_class():
    """Verify ValueError when only one class in y_true."""
    y_true = np.array([0, 0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError):
        gini_coefficient(y_true, y_prob)

def test_load_model_raises_if_file_missing(tmp_path):
    """Verify FileNotFoundError for non-existent model path."""
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nonexistent.pkl")
```

**Numerical Assertions:**
```python
# Exact equality (for integers, small counts)
assert result == expected

# Approximate equality (for floats)
assert abs(gini - 1.0) < 1e-9
np.testing.assert_allclose(result, expected, rtol=1e-5)

# Range checks
assert 0.3 < ks <= 1.0, f"Expected KS > 0.3, got {ks:.4f}"
assert result.loc[2, "CREDIT_INCOME_RATIO"] == 0
assert result.loc[1, "YEARS_EMPLOYED"] == pytest.approx(0.0)
```

**DataFrame Assertions:**
```python
# Shape
assert len(result) == len(expected)
assert result.shape == (307511, 68)

# Columns present
missing = [c for c in EXPECTED_COLUMNS if c not in result.columns]
assert missing == []

# No nulls in key columns
assert result["SK_ID_CURR"].notna().all()

# Data type
assert isinstance(result, pd.DataFrame)
assert result["AGE_YEARS"].dtype == np.float64

# No inf or extreme values
assert not result["CREDIT_INCOME_RATIO"].isin([np.inf, -np.inf]).any()
assert (result["EMPLOYED_TO_AGE_RATIO"] >= 0).all()
```

**Parametrized Tests (example pattern, not heavily used):**
```python
@pytest.mark.parametrize("value,expected", [
    (0.1, "Low"),
    (0.5, "Medium"),
    (0.9, "High"),
])
def test_risk_category(value, expected):
    assert categorize_risk(value) == expected
```

## Slow vs. Fast Tests

**Fast suite (conftest.py marks registration):**
- Tests without `@pytest.mark.slow`
- Include: unit tests for features, data loader, utils, model structure
- Typical runtime: 30–60 seconds
- Command: `pytest tests/ -v -m "not slow"`

**Slow suite (marked with @pytest.mark.slow):**
- Full model training tests (HPO with n_trials > 0)
- Featuretools DFS build tests
- Typical runtime: 2–5 minutes per test
- Command: `pytest tests/test_model.py::TestXGBoostOptuna::test_train_xgboost_optuna_returns_7_tuple -v` (single slow test)

**Module-scoped fixture gotcha (CRITICAL):**
From `CLAUDE.md`: "Expensive model fixtures (`catboost_result`, `benchmark_result` etc.) must be `scope="module"` — function-scoped causes the suite to hang (each fixture invoked once per test × 8 tests = hundreds of CPU minutes)."

Example from `test_model.py`:
```python
@pytest.fixture(scope="module")  # NOT function scope!
def trained_model(mock_data):
    """Train logistic baseline once; reuse across 8+ tests."""
    X, y = mock_data
    return train_logistic_baseline(X, y)
```

## Test Running

**Full Suite:**
```bash
cd /home/wd/Working\ Folder/Development/credit-risk-pipeline
pytest tests/ -v
# Expected: 416 tests (as of 2026-04-11)
# Fast suite (data_loader, features, utils, auto_features): ~60 seconds
# Slow suite (model training, HPO): ~2 minutes
# Total: ~3 minutes
```

**By Category:**
```bash
# Only feature tests (fast)
pytest tests/test_features.py -v

# Only model tests (includes slow @pytest.mark.slow)
pytest tests/test_model.py -v

# Skip slow tests (fast only)
pytest tests/ -v -m "not slow"

# Single slow test by name
pytest tests/test_model.py::test_train_xgboost_optuna_returns_7_tuple -v
```

**With Coverage:**
```bash
pytest tests/ --cov=src --cov-report=term-missing -v
# Shows uncovered lines at end of report
```

---

*Testing analysis: 2026-04-11*
