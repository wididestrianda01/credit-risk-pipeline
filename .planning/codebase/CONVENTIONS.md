# Coding Conventions

**Analysis Date:** 2025-02-20

## Naming Patterns

**Files:**
- Module names: lowercase with underscores (`data_loader.py`, `features.py`, `utils.py`)
- Test files: `test_<module_name>.py` (e.g., `test_features.py`)

**Functions:**
- Public functions: `snake_case` (e.g., `build_features`, `gini_coefficient`, `train_logistic_baseline`)
- Private functions: `_leading_underscore` (e.g., `_engineer_financial_ratios`, `_make_cv`)
- Private module constants: `_UPPER_SNAKE_CASE` (e.g., `_NAN_SENTINEL`, `_DAYS_EMPLOYED_SENTINEL`)

**Variables:**
- Local variables: `snake_case`
- Boolean variables: prefix with `is_`, `has_`, `should_`, `can_` (e.g., `is_unbalance=True`, `has_data=False`)
- Sentinel values: explicitly named as constants (e.g., `_NAN_SENTINEL: float = -999.0`)

**Types:**
- Type hints on all public and private function signatures (enforced via `from __future__ import annotations`)
- Classes: `PascalCase` (e.g., `CalibratedClassifierCV`, `FrozenEstimator`)
- Dataframe column names: ALL_CAPS with underscores (e.g., `CREDIT_INCOME_RATIO`, `EXT_SOURCE_MEAN`)

## Code Style

**Formatting:**
- PEP 8 compliant (implied by codebase structure)
- Line continuations: implicit line joining inside parentheses
- Operator spacing: follows PEP 8 (space around binary operators)

**Linting:**
- No explicit configuration files found (no .flake8, .pylintrc, setup.cfg, pyproject.toml in root)
- Import sorting observed: stdlib → numpy/pandas → sklearn/lightgbm → credit_engine

**Type Annotations:**
- All function parameters and return types annotated
- Use `from __future__ import annotations` for forward references and clean syntax
- Common type patterns:
  - `pd.DataFrame`, `pd.Series` for data structures
  - `np.ndarray` for numerical arrays
  - `tuple[float, float]` for return tuples (Python 3.9+ syntax)
  - `dict[str, Any]` for dictionaries
  - `Path | str` for file paths (union type)
  - `Literal["train", "test"]` for constrained string values

## Import Organization

**Order:**
1. Standard library imports (`sys`, `warnings`, `json`, `pickle`, `contextlib`)
2. `from __future__ import annotations` (always first after docstring if present)
3. NumPy, Pandas, SciPy imports
4. Scikit-learn imports (grouped by submodule)
5. Specialized libraries (LightGBM, XGBoost, CatBoost, Optuna, SHAP, Featuretools)
6. Matplotlib (with backend initialization in tests: `matplotlib.use("Agg")`)
7. Local credit_engine imports

**Path Aliases:**
- `src/` aliased as `credit_engine` via `conftest.py` → all imports: `from credit_engine.X import ...`
- Absolute imports preferred over relative imports

Example from `src/model.py`:
```python
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from credit_engine.utils import evaluate_model, gini_coefficient
```

## Error Handling

**Patterns:**
- Explicit `raise ValueError`, `KeyError`, `TypeError` with descriptive messages at system boundaries
- Guard conditions before expensive operations (e.g., check `n_trials >= 1` before Optuna loop)
- Input validation in public functions: check array shapes, missing values, value ranges
- No silent failures: always raise or warn

**Examples from `src/model.py`:**
```python
def train_xgboost_optuna(X: pd.DataFrame, y: pd.Series, n_trials: int = 50):
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pd.DataFrame")
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}.")
    if len(np.unique(y)) != 2:
        raise ValueError("y must contain exactly 2 classes (binary classification).")
    if y.sum() == 0:
        raise ValueError("y has no positive samples — cannot compute scale_pos_weight.")
```

**Examples from `src/data_loader.py`:**
```python
if not app_path.exists():
    raise FileNotFoundError(f"Missing {_FILE_APP_TRAIN} in {data_dir}")
if not sk_id_curr.is_unique:
    raise ValueError("SK_ID_CURR is not unique in application table")
```

**Assertion patterns (data integrity checks):**
- `_assert_no_row_multiplication(left_df, result_df, join_name)` — checks that multi-table joins don't inflate row counts
- Used to catch accidental Cartesian product joins

## Logging

**Framework:** No centralized logging module detected. Output to stdout discouraged in library code.

**Patterns Observed:**
- Warnings via `warnings.warn()` for deprecated behavior or data quality issues
- Pandas `SettingWithCopyWarning` suppressed explicitly: `pd.options.mode.copy_on_write = True`
- Silent progress via controlled verbosity in model training (e.g., `verbose=0` in LightGBM, `show_progress=False` in Optuna)

**In tests:**
- `print()` used only for debug output; tests rely on pytest assertions

## Comments

**When to Comment:**
- Mathematical rationale: explain why a specific formula or constant is chosen (e.g., WoE clipping at ±5)
- Domain context: reference regulatory standards (Basel III, GDPR) and sources (Siddiqi, López de Prado)
- Non-obvious decisions: explain tree-friendly sentinel values, handling of structural missingness, temporal embargo rationale
- Example from `src/features.py`:
```python
# WoE clipping bound. ln(dist_non_events / dist_events) is clipped to
# [-_WOE_CLIP, +_WOE_CLIP] to avoid ±inf when a bin contains only events
# or only non-events. ±5 corresponds to an odds ratio of ~150x, already
# extreme for any real feature.
_WOE_CLIP: float = 5.0
```

**Module docstrings:**
- Every module starts with a triple-quoted docstring:
  - First line: module name and brief purpose
  - Blank line
  - Detailed description of responsibility and architecture
  - Usage examples
  - Constants/concepts defined

Example from `src/utils.py`:
```python
"""
utils.py
--------
Shared evaluation metrics and plotting helpers for credit risk models.

Metrics
-------
- gini_coefficient   (= 2 * AUC - 1)
- ks_statistic       (max CDF separation between default / non-default)

Industry benchmarks (Basel III IRB credit scoring)
---------------------------------------------------
KS > 0.30: good separation
Gini > 0.60: good discriminatory power
"""
```

**Function docstrings:**
- Numpy-style docstrings (Parameters, Returns, Raises, Examples sections)
- Examples section shows typical usage
- Example from `src/utils.py`:
```python
def gini_coefficient(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Gini coefficient for binary classification.

    Defined as Gini = 2 × AUC − 1. Ranges from −1 (perfectly inverted
    predictions) through 0 (no discrimination) to 1 (perfect separation).
    The primary regulatory metric in Basel III IRB credit models.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth labels (0 = non-default, 1 = default).
    y_prob : np.ndarray
        Predicted default probabilities in [0, 1].

    Returns
    -------
    float
        Gini coefficient in [−1, 1].

    Raises
    ------
    ValueError
        If ``y_true`` contains only one class.

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_prob = np.array([0.1, 0.2, 0.7, 0.8])
    >>> gini_coefficient(y_true, y_prob)
    1.0
    """
```

## Function Design

**Size Guidelines:**
- Target: 30-50 lines per function
- Private helpers split by concern (e.g., `_engineer_financial_ratios`, `_engineer_demographics`, `_engineer_ext_source` in `features.py`)
- Long functions broken into private `_helpers` to isolate logic

**Parameters:**
- Maximum 5-6 positional parameters (use dataclasses or dicts for larger argument sets)
- Named keyword arguments with `=` for optional parameters
- Use `*args` and `**kwargs` rarely; prefer explicit signature
- Type hints on all parameters

**Return Values:**
- Single return value preferred
- Multiple return values via tuple with type hint: `tuple[Model, dict, pd.DataFrame, ...]`
- Return new DataFrames, never mutate input (immutability pattern)

Example from `src/model.py`:
```python
def train_logistic_baseline(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = _TEST_SIZE,
    random_state: int = _RANDOM_STATE,
) -> tuple[Pipeline, dict, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Train logistic regression on WoE-transformed features.
    
    Returns 6-tuple: (pipeline, metrics, X_train, X_test, y_train, y_test)
    """
```

## Module Design

**Exports (Public API):**
- Public functions listed at module level
- Private functions prefixed with `_`
- No `__all__` lists observed; convention followed implicitly

**Barrel Files:**
- `src/__init__.py` is minimal (13 lines) — no re-exports
- Callers import directly from submodules: `from credit_engine.features import build_features`

**Dependencies:**
- Unidirectional dependency graph: `data_loader` → `features` → `model` → `utils` + `explain`
- `utils` has no internal dependencies (shared evaluation layer)
- `explain.py` is a stub (31 lines)
- `auto_features.py` wraps featuretools; optional dependency handled via try/except

## Immutability & Side Effects

**Pattern:**
- All feature engineering functions: `def func(df: pd.DataFrame) -> pd.DataFrame:`
- Signature indicates transformation, never in-place mutation
- Implementations use `df.copy()` at the start to ensure input DataFrame is never modified
- Example from `src/features.py`:
```python
def _engineer_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """..."""
    out = df.copy()  # Never mutate input
    
    income = out["AMT_INCOME_TOTAL"].to_numpy()
    # ... transformations on out ...
    
    return out
```

**Numerical operations:**
- Use `np.errstate(divide="ignore", invalid="ignore")` to suppress temporary inf/NaN warnings during guarded division
- Replace inf/NaN explicitly: `.replace([np.inf, -np.inf], 0.0).fillna(_NAN_SENTINEL)`

## Constant Management

**Module-level constants (uppercase with leading underscore):**
- File paths: `_FILE_APP_TRAIN = "application_train.csv"`
- Thresholds: `_MISSING_DROP_THRESHOLD = 0.60`
- Sentinel values: `_NAN_SENTINEL = -999.0`
- Model parameters: `_TEST_SIZE: float = 0.2`, `_RANDOM_STATE: int = 42`
- Hyperparameter bounds: `_XGB_N_ESTIMATORS_MIN: int = 100`, `_XGB_N_ESTIMATORS_MAX: int = 1000`

**Rationale documented inline:**
```python
# Temporal CV embargo: strip the last _CV_EMBARGO_FRAC fraction of each
# training fold to prevent serial-correlation leakage across the train/val
# boundary (López de Prado, Advances in Financial Machine Learning, Ch. 7).
# 2% suffices for cross-sectional credit data with long lookback windows.
_CV_EMBARGO_FRAC: float = 0.02
```

## Temporal CV and Grouping

**Pattern Used for Time-Aware Cross-Validation:**
- When `groups` parameter is `None` and a temporal sort column exists (e.g., `prev_days_decision_mean`), auto-detect groups from DataFrame
- `_make_cv()` function returns either `StratifiedKFold` or `_TemporalCV` depending on `groups` argument
- `_TemporalCV` applies embargo fraction to prevent temporal leakage
- Applied consistently in: `train_logistic_baseline`, `train_xgboost_optuna`, `train_lightgbm_optuna`, `train_ensemble`

---

*Convention analysis: 2025-02-20*
