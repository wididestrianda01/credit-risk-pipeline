# Coding Conventions

**Analysis Date:** 2026-04-11

## Naming Patterns

**Files:**
- All source files use lowercase with underscores: `data_loader.py`, `features.py`, `model.py`, `utils.py`, `explain.py`, `auto_features.py`
- Test files follow pattern `test_<module>.py`: `test_data_loader.py`, `test_features.py`, `test_model.py`, `test_utils.py`, `test_auto_features.py`, `test_streak_evaluation.py`

**Functions:**
- All public functions use `snake_case`: `gini_coefficient()`, `ks_statistic()`, `evaluate_model()`, `plot_roc_and_pr()`
- Private helper functions use leading underscore and `snake_case`: `_engineer_financial_ratios()`, `_engineer_demographics()`, `_get_project_root()`, `_write_synthetic_csvs()`
- Public domain functions are verbs or verb phrases: `load_data()`, `build_features()`, `build_training_frame()`, `engineer_application_features()`, `train_xgboost_optuna()`

**Variables:**
- Local variables use `snake_case`: `out`, `income`, `annuity`, `credit`, `ratio_cols`, `rng`, `n_rows`, `n_pos`
- Boolean variables use `is`, `has`, `should`, `can` prefixes: `is_unbalance`, `has_missing`, `should_drop`
- Loop indices: single letters only when context is clear (`i`, `j`), otherwise descriptive (`col`, `row`, `sk_id`)
- Accumulator variables for aggregation: `out` (transformed copy), `result`, `features`

**Types & Classes:**
- Classes use `PascalCase`: `Pipeline`, `StratifiedKFold`, `LogisticRegression`, `_AverageEnsemble`, `_TemporalCV`, `FrozenEstimator`
- Type hints required on all public function signatures (fully typed in `src/model.py`, `src/utils.py`, `src/data_loader.py`, `src/features.py`)
- Type aliases use standard notation: `tuple[float, float]`, `dict[str, Any]`, `list[str]`, `pd.DataFrame`, `pd.Series`

**Constants:**
- All module-level constants use `UPPER_SNAKE_CASE` with leading underscore: `_DAYS_EMPLOYED_SENTINEL`, `_NAN_SENTINEL`, `_WOE_CLIP`, `_IV_VERY_STRONG`, `_PROJECT_ROOT`
- Numeric bounds for search spaces: `_XGB_MAX_DEPTH_MIN`, `_XGB_MAX_DEPTH_MAX`, `_LGB_NUM_LEAVES_MIN`, `_LGB_NUM_LEAVES_MAX`
- File path constants: `_FILE_APP_TRAIN`, `_FILE_BUREAU`, `_FILE_PREV_APP`, `_BENCHMARK_REPORT_PATH`, `_XGB_OPTUNA_MODEL_PATH`
- All module-level constants prefixed with underscore even if technically public (encapsulation pattern)

**Domain-Specific Feature Naming:**
- Aggregated features use source table prefix: `bureau_`, `prev_`, `pos_`, `inst_`, `cc_` (e.g., `bureau_avg_balance`, `inst_days_past_due_mean`)
- Boolean indicators use suffix: `_flag`, `_count`, `_cnt` (e.g., `bureau_overdue_flag`, `EXT_SOURCE_3_MISSING_FLAG`, `HIGH_RISK_DOC_MISSING`)
- Composite features use descriptive names: `CREDIT_INCOME_RATIO`, `EXT_SOURCE_MEAN`, `EMPLOYED_TO_AGE_RATIO`
- WoE-encoded features preserve original names after transformation

## Code Style

**Formatting:**
- No explicit formatter (black/ruff) configured; code follows PEP 8 conventions implicitly
- Line length: ~100 characters (observed in docstrings and code layout)
- Indentation: 4 spaces, no tabs
- Blank lines: 2 lines between top-level definitions, 1 line between methods in a class

**Linting:**
- No explicit linter configuration found
- Code follows PEP 8: `from __future__ import annotations` for forward compatibility, type hints throughout
- Implicit style conventions: avoid `print()` in library code, use explicit return values instead

## Import Organization

**Order (observed in `src/model.py`):**
1. Future imports: `from __future__ import annotations`
2. Environment setup: `os.environ["OMP_NUM_THREADS"] = "1"` (before imports that use it)
3. Backend setup: `matplotlib.use("Agg")` (before pyplot import)
4. Standard library: `import sys`, `import math`, `import json`, `from pathlib import Path`, `import logging`
5. Third-party scientific: `import numpy as np`, `import pandas as pd`, `from sklearn.*`, `import lightgbm as lgb`, `import xgboost as xgb`, `import catboost`, `import optuna`
6. Visualization: `import matplotlib.pyplot as plt`, `import shap`
7. Utility: `import joblib`
8. Local imports: `from src.features import X`, `from src.utils import Y`

**No wildcard imports:** All imports are explicit, e.g., `from src.features import build_features, engineer_application_features`

## Docstrings

**Style:** NumPy-style docstrings (following scikit-learn convention)

**Structure for functions:**
```python
def function_name(param1: int, param2: str) -> float:
    """
    One-line summary ending with period.

    Extended description if needed. May span multiple lines.
    Explains the "why", not the "what" (code speaks for itself).

    Parameters
    ----------
    param1 : int
        Description of param1. Include units if applicable.
    param2 : str
        Description of param2. Mention valid values.

    Returns
    -------
    float
        Description of return value. Include range if bounded.

    Raises
    ------
    ValueError
        When param1 < 0 or param2 is empty.

    Notes
    -----
    Additional context, e.g. algorithmic notes, numerical stability,
    or references to papers.

    Examples
    --------
    >>> function_name(5, "test")
    3.14
    """
```

**Module docstrings (three-part structure, observed in `src/features.py`, `src/data_loader.py`, `src/utils.py`, `src/model.py`):**
```python
"""
module_name.py
--------------
Short summary.

Section Heading (e.g., "Tables", "Architecture", "Usage")
---------
Description of that section.

Usage
-----
    from src.module import function_name
    result = function_name(arg)
"""
```

**Docstring coverage:**
- All public functions have docstrings with Parameters, Returns, Raises sections
- Module-level constants have inline comments: `_DAYS_EMPLOYED_SENTINEL: int = 365_243  # unemployment sentinel`
- Private helper functions have docstrings explaining their single responsibility
- Docstrings reference domain concepts (e.g., "WoE clipping", "IRB scorecard", "Basel III")

## Error Handling

**Patterns:**
- Explicit raise with informative messages: `raise ValueError(f"mode must be 'train' or 'test', got {mode!r}")`
- File operations: `raise FileNotFoundError(f"Could not locate project root (expected to find src/ and tests/)")`
- Data validation: raise at function entry with clear context
- Division by zero: guarded with `np.where(denominator > 0, numerator / denominator, 0.0)`
- Residual inf/NaN: `.replace([np.inf, -np.inf], 0.0).fillna(_NAN_SENTINEL)`

**Edge case handling:**
- All-NaN inputs: explicitly handled with `np.nanmean()`, `np.nanmin()`, result filled with sentinel
- Zero denominators: guarded before division with `np.where()`
- Missing files: validated at function entry; raise before attempting to read
- Empty DataFrames: checked with `len(df) == 0`, result passed upstream

**No try-except swallowing:** Errors propagate unless expected and handled at a higher layer.

## Logging

**Framework:** `logging` module (imported in `src/model.py` line 18)

```python
import logging
logger = logging.getLogger(__name__)
logger.info("HPO trial %d: Gini=%.4f", trial_num, gini_score)
```

**Pattern:** Logging used for HPO progress tracking (written to `reports/hpo_progress.jsonl`), model training milestones, not debug output in feature engineering.

**No print() in library code:** Library functions return values; caller decides logging level.

## Comments

**When to comment:**
- Explain the "why" not the "what" — code should be self-documenting
- Mark regulatory requirements, especially Basel III IRB compliance notes (e.g., temporal OOT split enforcement)
- Explain domain concepts (e.g., "WoE clipping to avoid ±inf when a bin is pure")
- Justify non-obvious constant values with domain reference

**Examples from codebase:**
```python
# Tree-friendly fill value for missing/undefined features.  Using -999 instead
# of 0 or mean avoids shifting the distribution and lets gradient boosting
# models learn a dedicated "missing" split.
_NAN_SENTINEL: float = -999.0

# Regulatory exclusions — columns that must be dropped from tree models per legal compliance.
# CODE_GENDER: GDPR Art. 21 (protection from discrimination), EU Consumer Credit Directive.
_REGULATORY_DROP_COLS: list[str] = ["CODE_GENDER", "thin_file_young"]

# Temporal CV embargo: strip the last _CV_EMBARGO_FRAC fraction of each
# training fold to prevent serial-correlation leakage across the train/val
# boundary (López de Prado, Advances in Financial Machine Learning, Ch. 7).
_CV_EMBARGO_FRAC: float = 0.02
```

## Function Design

**Size:** Keep functions under 50 lines (observed in `src/features.py` — each helper is 20–50 lines); orchestrator functions may exceed 100 lines when delegating to helpers

**Parameters:**
- Use type hints: `def function(df: pd.DataFrame, n_rows: int) -> pd.DataFrame:`
- Required positional first, optional keyword-only after `*`: `def function(df, required, *, optional=None)`
- Avoid excessive defaults; prefer required positional args

**Return values:**
- Single value: return directly, e.g., `return float(2 * auc - 1)`
- Multiple related values: return tuple with clear types: `return (ks_value, threshold_at_ks)` or `(model, metrics, X_train, X_test, y_train, y_test)`
- Data transformation: always return new DataFrame/Series (never mutate input)

**Immutability:** Strictly enforced — all DataFrame transformations create copies:
```python
def _engineer_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()  # Create copy immediately
    # ... transformations on out ...
    return out
```

## Module Design

**Exports (public API):**
- `src/data_loader.py`: `load_data()`, `build_training_frame()`, `save_training_frame()`
- `src/features.py`: `build_features()`, `engineer_application_features()`, `engineer_secondary_features()`, `compute_woe_iv()`, `select_features_by_iv()`, `build_feature_store()`, `apply_feature_store()`, `build_tree_feature_store()`, `engineer_instalment_streaks()`, `engineer_bureau_dpd_trend_3m_vs_12m()`, `compute_knn_target_encoding()`
- `src/auto_features.py`: `build_featuretools_feature_store()`, `apply_featuretools_feature_store()`, `evaluate_dfs_features()`, `filter_dfs_by_iv()`
- `src/model.py`: `train_logistic_baseline()`, `train_xgboost_optuna()`, `train_lightgbm_optuna()`, `train_catboost_optuna()`, `calibrate_model()`, `save_model()`, `load_model()`, `benchmark_imbalance_strategies()`, `train_ensemble()`, `run_ensemble_workflow()`
- `src/utils.py`: `gini_coefficient()`, `ks_statistic()`, `evaluate_model()`, `plot_roc_and_pr()`
- `src/explain.py`: stubs for `compute_shap_values()`, `fairness_report()`

**Barrel files:** No barrel files (no `__init__.py` re-exports) — all imports are direct module-to-module

**Private helpers:**
- Prefixed with `_`: `_engineer_financial_ratios()`, `_load_application()`, `_get_project_root()`, `_write_synthetic_csvs()`
- Grouped at module level in sections marked with comments: `# Private helpers — one concern per function`
- Never imported by external code

**Constants at module level:**
- Sentinel values: `_DAYS_EMPLOYED_SENTINEL`, `_NAN_SENTINEL`, `_WOE_CLIP`
- Hyperparameter bounds: `_XGB_MAX_DEPTH_MIN`, `_LGB_NUM_LEAVES_MAX`, `_CAT_DEPTH_MIN`
- File path references: `_PROJECT_ROOT`, `_HPO_PROGRESS_LOG_PATH`, `_XGB_OPTUNA_MODEL_PATH`
- Magic numbers converted to named constants: `_MISSING_DROP_THRESHOLD`, `_LEAKY_COLUMNS`

## Type Hints

**Requirement:** Type hints on all public function signatures; optional on private helpers.

**Patterns:**
```python
def load_data(data_dir: str | Path, mode: str = "train") -> pd.DataFrame:
    """Load training or test data."""

def gini_coefficient(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Gini = 2 * AUC - 1."""

def train_xgboost_optuna(
    feature_store_path: str, 
    n_trials: int = 50, 
    groups: np.ndarray | None = None
) -> tuple[Any, dict, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
    """Return 7-tuple: (model, metrics, X_train, X_test, y_train, y_test, best_params)."""
```

**Union types:** Use `|` syntax (PEP 604): `str | Path`, `int | float`, `T | None`

**Avoid `Any` in public signatures:** Force specificity, e.g., `dict[str, float]` not `dict[str, Any]`

## Git Commit Conventions

**Format:** `<type>(<scope>): <description>`

**Types:** `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`

**Scope:** Domain area or component name: `(features)`, `(model)`, `(data-loader)`, `(04.2.7)` (phase number)

**Description:** Imperative tense, lowercase, max 72 chars. Include metric result if applicable.

**Examples from codebase:**
- `feat(catboost): Basel CRE36.54 compliant CatBoost HPO, OOT Gini=0.5699, KS=0.4259`
- `fix(model): enforce Basel CRE36.54 temporal OOT split in LGB and CatBoost HPO`
- `feat(04.2.7): implement engineer_inst_late_rate_12m and engineer_inst_late_rate_recent_vs_historical`
- `test(04.2.7): add Wave 1 TDD stubs and conftest fixtures for 7 delinquency features`
- `fix(features): protect Wave 1 features from variance/correlation filters`

**Attribution:** Disabled globally via `~/.claude/settings.json` — no Co-Authored-By trailers.

---

*Convention analysis: 2026-04-11*
