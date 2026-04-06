# Codebase Structure

**Analysis Date:** 2026-04-06

## Directory Layout

```
credit-risk-pipeline/
├── src/                          # Core library (imported as credit_engine via conftest.py alias)
│   ├── __init__.py               # Module docstring, lists all submodules
│   ├── data_loader.py            # 7-table join + dtype enforcement
│   ├── features.py               # Hand-engineered feature pipeline + WoE binning
│   ├── auto_features.py          # Featuretools Deep Feature Synthesis (optional path)
│   ├── model.py                  # Training, HPO, calibration for LGB/XGB/LR/CatBoost
│   ├── utils.py                  # Metrics (Gini, KS, Brier) + plotting helpers
│   └── explain.py                # SHAP explainability stubs (Phase 4)
│
├── app/                          # Deployment endpoints
│   ├── api.py                    # FastAPI /predict, /health (stub)
│   └── streamlit_app.py          # Interactive SHAP dashboard (placeholder)
│
├── data/                         # Raw data (7 CSV tables, read-only)
│   ├── application_train.csv     # Main table: loan details + target (307,511 rows)
│   ├── application_test.csv      # Test set without target (48,744 rows)
│   ├── bureau.csv                # Credit bureau summary per applicant
│   ├── bureau_balance.csv        # Monthly snapshots of bureau accounts
│   ├── previous_application.csv  # Prior applications
│   ├── POS_CASH_balance.csv      # POS/cash loan snapshots
│   ├── installments_payments.py  # Payment history
│   ├── credit_card_balance.csv   # Credit card account snapshots
│   ├── processed/                # Feature store outputs (generated, read on inference)
│   │   ├── X_features.parquet    # 307,511 × 40+ WoE-binned features
│   │   ├── X_train.parquet       # Raw X_train (before feature engineering)
│   │   └── y_train.parquet       # Binary target series
│   └── sample_submission.csv     # Kaggle template (not used)
│
├── models/                       # Persisted models + metadata (generated during training)
│   ├── logistic_baseline.pkl     # Fitted LR(StandardScaler → LogisticRegression)
│   ├── lightgbm_best.pkl         # Fitted LGB (post-Optuna refit)
│   ├── lightgbm_calibrated.pkl   # LGB + Platt sigmoid layer
│   ├── lightgbm_params.json      # Best HPO params for LGB
│   ├── xgboost_best.pkl          # Fitted XGB (post-Optuna refit)
│   ├── xgboost_calibrated.pkl    # XGB + Platt sigmoid layer
│   ├── xgboost_params.json       # Best HPO params for XGB
│   ├── catboost_best.pkl         # Fitted CatBoost model
│   ├── woe_mappings.pkl          # dict[feature_name → pd.Series(bin_edges)]
│   ├── raw_feature_columns.pkl   # list[feature_names] for consistent input ordering
│   ├── featuretools_feature_defs.pkl  # DFS entity relationships + primitives
│   └── featuretools_selected_cols.json # Columns retained post-IV filter (DFS path)
│
├── reports/                      # Results + figures (generated, committed only when significant)
│   ├── figures/                  # PNG plots (ROC, PR, calibration reliability, etc.)
│   │   ├── logistic_roc_pr.png
│   │   ├── xgboost_roc_pr.png
│   │   ├── lightgbm_roc_pr.png
│   │   └── calibration_reliability.png
│   ├── imbalance_benchmark.csv   # 4×7 benchmark table: SMOTE, cost-sensitive, threshold, hybrid
│   ├── model_comparison.csv      # Final evaluation: Gini, KS, Brier per model
│   ├── priority12_eval_results.json  # Full Phase 1&2 metrics
│   ├── lgb_integration_results.json  # LGB config comparison (is_unbalance, scale_pos_weight, etc.)
│   ├── lgb_booster_comparison.json   # GBDT vs DART vs GOSS
│   ├── temporal_analysis_visualization.png  # Target rate drift over application date
│   ├── adversarial_validation.json  # Train/test distribution shift (AUC ≥ 0.65 = problematic)
│   ├── lgb_early_stopping_audit.json # Early stopping variance across n_estimators values
│   ├── lgb_hyperparameter_heatmap.csv # 2D HPO sweep (learning_rate vs depth)
│   └── [various .log files] # Script execution logs (eval_raw_features.py, lgb_integration_v3.log, etc.)
│
├── notebooks/                    # Jupyter analysis (outputs stripped via nbstripout before commit)
│   ├── 01_eda_and_data_quality.ipynb      # 12 EDA sections: null patterns, class imbalance, distributions
│   ├── 02_feature_engineering.ipynb       # IV ranking, WoE binning, correlation heatmap, 5 sanity checks
│   ├── 03_modeling_and_evaluation.ipynb   # Training loops, metric comparison, calibration diagnostics
│   └── 04_explainability_and_fairness.ipynb # SHAP plots, fairness by demographic group (Phase 4)
│
├── scripts/                      # One-off scripts (execution logs, diagnostic runs, not imported by tests)
│   ├── eval_priority12.py        # Full Phase 1&2 evaluation (LR + XGB + LGB)
│   ├── eval_raw_features.py      # Benchmark raw features (before WoE binning)
│   ├── calibrate_xgboost.py      # Fit Platt sigmoid post-training
│   ├── lgb_integration_run.py    # LGB config sweep (is_unbalance, scale_pos_weight, booster_type)
│   ├── lgb_booster_comparison.py # GBDT vs DART vs GOSS comparison
│   ├── lgb_hyperparameter_audit.py # 2D HPO heatmap
│   ├── lgb_early_stopping_audit.py # Variance in early stopping across n_estimators
│   ├── lgb_monotone_constraint_test.py # Monotonicity enforcing (regulatory requirement)
│   ├── train_raw_and_eval.py     # End-to-end raw→train→eval without WoE (comparison)
│   ├── rebuild_feature_store.py  # Regenerate X_features.parquet after feature changes
│   ├── adversarial_validation_check.py # Train/test overlap check
│   ├── validate_gini.py          # Post-training Gini verification
│   └── [others: featuretools, raw feature, calibration variants]
│
├── tests/                        # Pytest test suite (~190 tests, 80%+ coverage target)
│   ├── test_data_loader.py       # 32 tests: load_data, build_training_frame, aggregations
│   ├── test_features.py          # 79 tests: engineer_*, build_feature_store, WoE binning
│   ├── test_auto_features.py     # Featuretools DFS tests
│   ├── test_model.py             # Model training, HPO, ensemble, calibration tests
│   ├── test_utils.py             # 32 tests: gini_coefficient, ks_statistic, evaluate_model
│   └── conftest.py               # (root) pytest fixtures + sys.path setup
│
├── docs/                         # Reference documentation (untracked, internal use)
│   ├── CODEMAPS/                 # Auto-generated code structure maps
│   └── lgb_research_findings.md  # Notes from LGB tuning experiments
│
├── .planning/codebase/           # GSD planning documents (generated by /gsd-map-codebase)
│   ├── ARCHITECTURE.md           # Layer breakdown, data flow, abstractions
│   └── STRUCTURE.md              # This file
│
├── .git/                         # Version control
├── .gitignore                    # Excludes: data/*.csv (raw), models/*.pkl (large), .venv/, __pycache__/
├── .gitattributes                # nbstripout filter: auto-strip notebook outputs on add
├── conftest.py                   # Root pytest config: aliases src → credit_engine
├── requirements.txt              # Python dependencies
├── CLAUDE.md                     # Project context (not committed; private working reference)
├── credit_risk_tasks_list.md     # Session task log (not committed; private working reference)
├── IMPROVEMENT.md                # Running notes on issues fixed (not committed)
└── README.md                     # (if present: project overview)
```

## Directory Purposes

**src/ (Core Library):**
- Purpose: Reusable code imported by tests, notebooks, and scripts
- Contains: Data loading, feature engineering, model training, evaluation
- Key files: `data_loader.py`, `features.py`, `model.py`, `utils.py`
- Aliased as `credit_engine` via conftest.py so imports read: `from credit_engine.X import ...`

**app/ (Deployment):**
- Purpose: Production endpoints and user interfaces
- Contains: FastAPI `/predict` endpoint (stub, Phase 5), Streamlit SHAP dashboard (placeholder, Phase 4)
- Key files: `api.py` (scoring), `streamlit_app.py` (visualization)

**data/ (Raw Input):**
- Purpose: Home Credit dataset — 7 relational tables
- Contains: Never read directly; use `src/data_loader.py` to load and join
- Key file: `data/processed/` subdirectory for generated feature matrices

**models/ (Artifacts):**
- Purpose: Persisted estimators, hyperparameters, metadata
- Contains: `.pkl` files (joblib serialization), `.json` param files
- Key files: `*_best.pkl` (primary models), `woe_mappings.pkl` (feature transform), `raw_feature_columns.pkl` (column order)

**reports/ (Results):**
- Purpose: Metrics, plots, diagnostics from training/evaluation runs
- Contains: JSON result files, CSV benchmark tables, PNG figures, execution logs
- Key files: `figures/` subdirectory, `imbalance_benchmark.csv`, `priority12_eval_results.json`

**notebooks/ (Analysis):**
- Purpose: Exploratory analysis and documentation
- Contains: Jupyter notebooks for EDA, feature derivation, modeling, explainability
- Key files: `01_eda_and_data_quality.ipynb` (mandatory read for domain understanding)
- Convention: Execute from within `notebooks/` directory; outputs stripped before commit via nbstripout

**scripts/ (One-Off Utilities):**
- Purpose: Diagnostic and experimental runs (not part of core pipeline)
- Contains: HPO diagnostics, calibration, DFS experiments, validation checks
- Convention: Read-only from tests; executed standalone via shell for exploration
- Key files: `eval_priority12.py` (full pipeline test), `lgb_integration_run.py` (config sweep)

**tests/ (Validation):**
- Purpose: Unit + integration testing with pytest
- Contains: Test modules organized by src/ module (test_data_loader.py → data_loader.py, etc.)
- Coverage: 80%+ target on src/
- Key files: `conftest.py` (root), test_features.py (largest, 79 tests)

## Key File Locations

**Entry Points:**

- `src/data_loader.py`: `load_data(data_dir)` — raw joined DataFrame
- `src/data_loader.py`: `build_training_frame(data_dir)` — X, y tuple ready for feature engineering
- `src/features.py`: `build_features(df)` — apply all hand-engineered features
- `src/auto_features.py`: `build_featuretools_feature_store()` — DFS auto-aggregation (optional)
- `src/model.py`: `train_xgboost_optuna(X, y)`, `train_lightgbm_optuna(X, y)` — Bayesian HPO + refit
- `scripts/eval_priority12.py`: End-to-end evaluation script (best entry point for full training)

**Configuration:**

- `conftest.py`: Python path setup + module aliasing (src → credit_engine)
- `requirements.txt`: Dependencies (pandas, scikit-learn, lightgbm, xgboost, catboost, optuna, fastapi, etc.)
- `.gitignore`: Exclude raw CSV, pickled models, venv
- `.gitattributes`: nbstripout registration for notebook output stripping

**Core Logic:**

- `src/data_loader.py`: Multi-table join with prefix-namespaced aggregates
- `src/features.py`: WoE binning, IV filtering, feature store persistence
- `src/model.py`: Stratified CV + temporal embargo, Optuna HPO, probability calibration
- `src/utils.py`: Gini, KS, Brier metrics + ROC/PR plotting

**Testing:**

- `tests/test_features.py`: Largest suite (79 tests); tests feature engineering pipeline end-to-end
- `tests/test_model.py`: Model training, ensemble, calibration tests
- `tests/test_utils.py`: Metric computation tests
- `tests/test_data_loader.py`: Data loading and aggregation tests

## Naming Conventions

**Files:**

- Source modules: `lowercase_with_underscores.py` (e.g., `data_loader.py`, `features.py`)
- Test modules: `test_{module}.py` (e.g., `test_features.py`)
- Scripts: `verb_noun_details.py` (e.g., `eval_priority12.py`, `calibrate_xgboost.py`)
- Notebooks: `NN_descriptor.ipynb` (e.g., `01_eda_and_data_quality.ipynb`)
- Models: `{estimator}_{variant}.pkl` (e.g., `lightgbm_best.pkl`, `xgboost_calibrated.pkl`)
- Reports: `{model}_results.json`, `{analysis}_benchmark.csv`

**Directories:**

- Source: `src/` (canonical library code)
- Tests: `tests/` (pytest suite)
- Data: `data/` (raw input)
- Generated: `models/`, `reports/`, `data/processed/` (outputs)
- Utilities: `scripts/`, `notebooks/` (exploration, diagnostics)

**Features (columns after engineering):**

- Prefixed aggregates: `{table}_{metric}` (e.g., `bureau_avg_balance`, `pos_cash_max_instalment`)
- Boolean indicators: `{name}_flag` (e.g., `bureau_overdue_flag`, `high_risk_doc_missing`)
- Ratios: `{numerator}_{denominator}_ratio` or just `{RATIO_NAME}` (e.g., `CREDIT_INCOME_RATIO`)
- Composites: `{source}_{operation}` (e.g., `ext_source_mean`, `ext_source_min`)
- WoE-encoded: `{original_name}` (transformed in-place by `build_feature_store()`)

**Constants (module-level):**

- Threshold/hyperparameter: `_UPPER_SNAKE_CASE` (e.g., `_CV_EMBARGO_FRAC`, `_NAN_SENTINEL`)
- File paths: `_FILE_{TABLE}` (e.g., `_FILE_BUREAU`, `_FILE_INSTALLMENTS`)
- Search bounds: `_{ESTIMATOR}_{PARAM}_{MIN,MAX}` (e.g., `_XGB_MAX_DEPTH_MIN`, `_LGB_LEARNING_RATE_MAX`)

## Where to Add New Code

**New Feature:**

Primary code:
- If domain-specific: add function to `src/features.py` with signature `def _{name}(df: pd.DataFrame) -> pd.DataFrame`
- Call from `build_features()` in sequence
- Add guard for division-by-zero, inf/NaN replacement, -999 fill

Tests:
- Add test class to `tests/test_features.py` with fixtures from `conftest.py`
- Test edge cases: zero denominators, all-NaN rows, structural missingness
- Verify output shape, dtypes, no in-place mutation

**New Model/Estimator:**

Implementation:
- Add training function to `src/model.py` with signature `def train_{estimator}_{variant}(X, y, ...) → (model, metrics, ...)`
- Follow: split data → CV loop → HPO (if applicable) → refit → calibrate → return
- All hyperparameter bounds as `_ESTIMATOR_PARAM_{MIN,MAX}` constants

Tests:
- Add test class to `tests/test_model.py`
- Test: CV loop logic, metric computation, return signatures, artifact persistence

Evaluation:
- Add evaluation call to `scripts/eval_priority12.py`
- Persist results to `reports/{estimator}_results.json`
- Generate plots to `reports/figures/{estimator}_roc_pr.png`

**Utility Function:**

Location:
- If general metric/plot: add to `src/utils.py`
- If model-specific: add to `src/model.py`
- If dataset-specific: add to `src/data_loader.py`

Pattern:
- Type hints on all function signatures
- Numpy-style docstring with Parameters, Returns, Examples
- Return new objects (immutable); never mutate input DataFrame
- No `print()` statements; use `logging` in scripts only

**Experimental/Diagnostic:**

Location:
- Add to `scripts/` directory (not imported by tests)
- Name: `{action}_{subject}_{details}.py` (e.g., `eval_raw_features.py`)
- Convention: standalone execution via `python scripts/{script}.py`

## Special Directories

**data/processed/:**
- Purpose: Intermediate and final feature matrices (generated, rarely committed)
- Generated by: `src/features.py` functions (`build_feature_store()`)
- Committed: Only if rebuilding from raw data changes behavior; document reason in commit message
- Content: X_train.parquet, y_train.parquet, X_features.parquet (WoE-binned final features)

**models/:**
- Purpose: Serialized estimators + metadata (generated, rarely committed)
- Generated by: `src/model.py` training functions (final refit), calibration functions
- Committed: Only if model performance improves Gini by ≥0.005; include metric in commit message
- Cleanup: Remove old variants (keep only `_best.pkl` per estimator)

**reports/:**
- Purpose: Metrics, plots, diagnostic outputs (generated, selectively committed)
- Generated by: Training and evaluation scripts
- Committed: JSON result files, CSV benchmarks, PNG figures if they document findings
- Cleanup: Clear before new runs (except figures/); keep only final versions

**docs/ (untracked):**
- Purpose: Research notes, external references (not committed)
- Usage: Internal documentation during development
- Examples: LGB tuning decisions, dataset paper links, regulatory references

---

*Structure analysis: 2026-04-06*
