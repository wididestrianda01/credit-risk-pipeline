# Codebase Structure

**Analysis Date:** 2026-04-11

## Directory Layout

```
credit-risk-pipeline/
├── src/                          # Core package (functional, immutable operations)
│   ├── __init__.py
│   ├── data_loader.py            # 7-table join, aggregation, dtype enforcement (~350 lines)
│   ├── features.py               # Feature engineering — WoE and raw pipelines (~1,300 lines)
│   ├── auto_features.py          # Featuretools DFS auto-aggregation (~700 lines)
│   ├── model.py                  # Model training, Optuna HPO, calibration, ensemble (~1,400 lines)
│   ├── utils.py                  # Metrics (Gini, KS, BrierSkill), ROC+PR plotting (~350 lines)
│   └── explain.py                # SHAP & fairness stubs (Phase 04.3 pending, ~50 lines)
│
├── app/                          # Deployment layer
│   ├── api.py                    # FastAPI /predict endpoint (Phase 05.1, stubs)
│   └── streamlit_app.py          # Interactive SHAP dashboard (Phase 05.2, placeholder)
│
├── data/                         # Raw datasets and processed artifacts
│   ├── application_train.csv     # 307,511 rows, main application table
│   ├── application_test.csv      # Test set (not used in this pipeline)
│   ├── bureau.csv                # Credit bureau summary per applicant
│   ├── bureau_balance.csv        # Monthly bureau balance history
│   ├── previous_application.csv  # Prior application history
│   ├── POS_CASH_balance.csv      # POS/cash monthly snapshots
│   ├── installments_payments.csv # Payment history per installment
│   ├── credit_card_balance.csv   # Credit card balance snapshots
│   ├── sample_submission.csv     # Kaggle submission template
│   ├── HomeCredit_columns_description.csv
│   └── processed/
│       ├── X_train.parquet       # 307,511 × 195 — joined base features
│       ├── y_train.parquet       # 307,511 × 1 — binary TARGET (8% defaults)
│       ├── X_features.parquet    # 307,511 × 68 — WoE-binned (logistic/interpretability)
│       ├── X_tree_raw.parquet    # 307,511 × 143 — raw engineered (tree models, Wave 1 features)
│       └── X_tree_dfs.parquet    # 307,511 × ~290 — raw + DFS aggregates (experimental)
│
├── models/                       # Fitted model artifacts (joblib .pkl format)
│   ├── xgboost_raw_best.pkl      # XGBoost uncalibrated (Phase 04.2.3)
│   ├── xgboost_raw_calibrated.pkl # PRIMARY — Platt-calibrated XGBoost; OOT Gini=0.5666
│   ├── xgboost_raw_params.json   # Best hyperparameters
│   ├── lightgbm_raw_calibrated.pkl # Basel CRE36.54 compliant; OOT Gini=0.5746
│   ├── lightgbm_raw_params.json  # Best hyperparameters
│   ├── catboost_raw_calibrated.pkl # Basel CRE36.54 compliant; OOT Gini=0.5699
│   ├── catboost_raw_params.json  # Best hyperparameters
│   ├── logistic_baseline.pkl     # Logistic regression baseline; Gini=0.489
│   ├── optuna_studies.db         # SQLite — Optuna trial history (DO NOT DELETE)
│   └── [Other legacy models]     # xgboost_best.pkl, lightgbm_best.pkl, etc. (WoE store, superseded)
│
├── reports/                      # Evaluation artifacts (generated, not tracked)
│   ├── figures/
│   │   ├── xgboost_raw_roc_pr.png
│   │   ├── lightgbm_raw_roc_pr.png
│   │   ├── catboost_raw_roc_pr.png
│   │   └── [EDA figures]
│   ├── xgboost_raw_eval.json     # OOT metrics for XGBoost
│   ├── lgb_feature_store_selection.json # 2-store ablation (raw vs raw+DFS)
│   ├── lgb_compliant_eval.json   # Phase 04.2.4.1 — LGB Basel re-run
│   ├── catboost_compliant_eval.json # Phase 04.2.5.1 — CatBoost Basel re-run
│   ├── imbalance_benchmark.csv   # 4-strategy comparison (SMOTE, cost-sensitive, threshold, hybrid)
│   └── [Other evaluation outputs]
│
├── scripts/                      # Standalone utility scripts (not production code)
│   ├── train_xgboost_raw.py      # CLI wrapper for XGBoost Optuna training
│   ├── run_lgb_wave1_hpo.py      # LGB HPO on X_tree_raw.parquet (Wave 1 features)
│   ├── run_catboost_compliant.py # CatBoost compliant re-run (Basel CRE36.54)
│   ├── rebuild_tree_feature_store_wave1.py # Phase 04.2.7 — feature store rebuild
│   ├── lgb_feature_store_ablation.py # 2-store comparison (raw vs raw+DFS)
│   ├── [Other analysis scripts]  # collect_ceiling_evidence.py, compare_feature_stores.py, etc.
│   └── __pycache__/
│
├── notebooks/                    # Jupyter analysis (not production)
│   ├── 01_eda_and_data_quality.ipynb    # 12 EDA sections, 12 figures
│   ├── 02_feature_engineering.ipynb     # IV ranking, WoE binning, interaction analysis
│   ├── 03_modeling_and_evaluation.ipynb # Model training results (Phase 04)
│   └── 04_explainability_and_fairness.ipynb # SHAP analysis (Phase 04.3, pending)
│
├── tests/                        # Pytest test suite (416+ tests, 80%+ coverage)
│   ├── test_data_loader.py       # 32 tests — join, aggregation, dtype enforcement
│   ├── test_features.py          # 79+ tests — engineering, WoE/IV, feature store
│   ├── test_model.py             # 160+ tests — Optuna HPO, ensemble, calibration, temporal CV
│   ├── test_auto_features.py     # 10+ tests — Featuretools DFS, Woodwork LogicalTypes
│   ├── test_utils.py             # 10 tests — Gini, KS, evaluate_model, plotting
│   └── test_streak_evaluation.py # 7 tests — Instalment streak feature validation
│
├── .planning/                    # GSD codebase documentation
│   ├── codebase/
│   │   ├── ARCHITECTURE.md       # Layers, data flow, abstractions (this session)
│   │   ├── STRUCTURE.md          # Directory layout, naming conventions (this session)
│   │   ├── CONVENTIONS.md        # Code style, import order, patterns
│   │   ├── TESTING.md            # Test framework, fixtures, patterns
│   │   ├── STACK.md              # Technology stack (Python, libraries, runtime)
│   │   ├── INTEGRATIONS.md       # External services, databases, APIs
│   │   └── CONCERNS.md           # Technical debt, risks, fragile areas
│   ├── phases/                   # GSD phase documentation
│   ├── PROJECT.md                # Project overview, goals, milestones
│   └── STATE.md                  # Current phase state, progress tracking
│
├── .claude/                      # Claude Code configuration
│   ├── agents/                   # Custom agents if any
│   └── settings.json             # Model selection, preferences
│
├── .venv/                        # Python virtual environment (not tracked)
├── .git/                         # Git history
├── .pytest_cache/                # Pytest cache (not tracked)
├── .mypy_cache/                  # Type checking cache (not tracked)
├── __pycache__/                  # Python bytecode cache (not tracked)
│
├── conftest.py                   # pytest configuration (adds src/ to sys.path)
├── CLAUDE.md                     # Project instructions (not committed — reference only)
├── credit_risk_tasks_list.md     # Task tracking (not committed — reference only)
├── requirements.txt              # pip dependencies
├── .gitignore                    # Standard: *.pkl, *.parquet, data/, models/, reports/, .env, etc.
├── .gitattributes                # nbstripout filter for .ipynb output stripping
├── .ruff_cache/                  # Ruff linter cache (not tracked)
└── .vscode/                      # VS Code settings (not tracked)
```

## Directory Purposes

**src/:**
- Purpose: Core modeling library (imported as `src.*` directly)
- Contains: Data loading, feature engineering, model training, metrics, SHAP
- Key files: `data_loader.py`, `features.py`, `auto_features.py`, `model.py`, `utils.py`
- Architecture: Functional, immutable — each function takes DataFrame(s) and returns new DataFrame
- Invariant: All paths use `_PROJECT_ROOT` computed from module location for test isolation

**app/:**
- Purpose: Deployment layer (FastAPI + Streamlit)
- Contains: HTTP endpoint, dashboard UI
- Status: Stubs in Phase 3; implementation in Phase 05
- Deployment target: Docker container or local dev (`uvicorn app.api:app --reload`)

**data/:**
- Purpose: Raw Kaggle Home Credit dataset + processed artifacts
- Raw files: Never read directly — always use `src/data_loader.py`
- Processed outputs: Generated by feature engineering, not tracked in git (.gitignore: `processed/*.parquet`)
- Invariant: `data/raw/` not committed; raw CSV tables provided separately

**models/:**
- Purpose: Persisted model artifacts, hyperparameters, study database
- Files: Joblib pickle format; JSON for hyperparams; SQLite for Optuna trials
- Critical: `optuna_studies.db` must NEVER be deleted (contains trial history)
- Artifacts: Generated during training, loaded at inference
- Invariant: All paths anchored to `_PROJECT_ROOT` for absolute safety

**reports/:**
- Purpose: Evaluation results, figures, benchmarks
- Generated: During training runs, not tracked in git
- Contents: CSV metrics, JSON results, PNG figures
- Retention: Used for post-hoc analysis; figures aid regulatory documentation

**scripts/:**
- Purpose: Standalone task scripts for Phase execution
- Not part of core library — utility only
- Pattern: Each script loads data, runs specific task (e.g., HPO), saves results to reports/ or models/
- Usage: `python scripts/train_xgboost_raw.py --feature-store data/processed/X_tree_dfs.parquet --n-trials 50`

**notebooks/:**
- Purpose: Analysis and experimentation (not production)
- Format: Jupyter .ipynb with `nbstripout` auto-output stripping on commit
- Contents: EDA, feature validation, result visualization
- Invariant: Outputs stripped before staging; re-run with `jupyter nbconvert --to notebook --execute notebook.ipynb` if needed

**tests/:**
- Purpose: Pytest suite (416+ tests, 80%+ coverage)
- Organization: One file per src/ module; fixtures in conftest.py
- Execution: `pytest tests/ -v` (all); `pytest tests/ -v -m "not slow"` (fast suite only)
- Invariant: Expensive model fixtures (catboost_result, benchmark_result) scoped to module level to prevent hanging

**.planning/codebase/:**
- Purpose: GSD orchestrator codebase documentation
- Files: ARCHITECTURE.md, STRUCTURE.md, CONVENTIONS.md, TESTING.md, STACK.md, INTEGRATIONS.md, CONCERNS.md
- Committed: Yes
- Updated: On-demand when phase focus changes or architecture evolves
- Audience: GSD planner and executor agents reading during phase planning

## Key File Locations

**Entry Points:**

- `src/data_loader.py::load_data(data_dir)` → raw joined DataFrame
- `src/data_loader.py::build_training_frame(data_dir)` → (X_train, y_train) pair
- `src/features.py::build_features(df)` → raw engineered features
- `src/features.py::build_tree_feature_store(data_dir, output_path)` → X_tree_raw.parquet
- `src/features.py::build_feature_store(data_dir, output_path)` → X_features.parquet (WoE)
- `src/auto_features.py::build_featuretools_feature_store(data_dir, output_path)` → X_tree_dfs.parquet
- `src/model.py::train_xgboost_optuna(feature_store_path, n_trials=50, groups=None)` → (model, metrics, X_test, y_test, best_params)
- `src/model.py::train_lightgbm_optuna(feature_store_path, n_trials=50, groups=None)` → same tuple
- `src/model.py::train_catboost_optuna(feature_store_path, n_trials=50, groups=None)` → same tuple
- `src/model.py::run_ensemble_workflow(X, y)` → ensemble model (persisted if delta_gini >= 0.005)
- `app/api.py::app` → FastAPI application; `/predict` POST endpoint

**Configuration:**

- `conftest.py`: Pytest configuration; adds project root to sys.path
- `requirements.txt`: pip dependencies (pandas, scikit-learn, xgboost, lightgbm, catboost, optuna, featuretools, shap, fastapi, streamlit, etc.)
- `CLAUDE.md`: Project instructions (not committed; reference for humans)
- `credit_risk_tasks_list.md`: Task tracking (not committed; reference for phase coordination)
- `.gitignore`: Excludes *.pkl, *.parquet, data/, models/, reports/, notebooks/.ipynb_checkpoints, .env, etc.
- `.gitattributes`: nbstripout filter for .ipynb output stripping on commit

**Core Logic by Module:**

**src/data_loader.py:**
- `load_data(data_dir, mode)` → loads 7 CSV tables, joins on SK_ID_CURR, aggregates secondary tables (many:1 → 1:1)
- `build_training_frame(data_dir)` → high-level API; returns (X_train, y_train) after feature engineering
- `save_training_frame(X, y, output_dir)` → persists to parquet
- Private helpers: `_load_application()`, `_join_bureau()`, `_join_bureau_balance()`, `_join_previous_application()`, `_join_pos_cash()`, `_join_installments()`, `_join_credit_card()`

**src/features.py:**
- `build_features(df)` → 140+ raw engineered features; chains 5 private helpers
  - `_engineer_financial_ratios()` → CREDIT_INCOME_RATIO, ANNUITY_INCOME_RATIO, CREDIT_TERM, GOODS_CREDIT_RATIO
  - `_engineer_demographics()` → AGE_YEARS, YEARS_EMPLOYED, EMPLOYED_TO_AGE_RATIO
  - `_engineer_documents()` → HIGH_RISK_DOC_MISSING, DOCUMENTS_SUBMITTED
  - `_engineer_ext_source()` → EXT_SOURCE composites (MEAN, MIN, MAX, STD, RANGE) + 5 polynomial terms
  - `engineer_secondary_features()` → 35+ aggregates + 9 cross-table interactions from 5 secondary tables
- `build_feature_store(data_dir, output_path)` → WoE pipeline: raw (130) → engineered (140) → IV-filtered (68)
- `build_tree_feature_store(data_dir, output_path)` → raw pipeline: 140+ features, variance filter only (no WoE)
- `select_features_by_iv(X, y, iv_threshold)` → filters by Information Value

**src/auto_features.py:**
- `build_featuretools_feature_store(data_dir, output_path)` → DFS on 7-table entity set
- `apply_featuretools_feature_store(data_dir, store_path, output_path)` → apply stored feature definitions
- `deduplicate_dfs_features(X_dfs, X_raw, threshold=0.90)` → remove >0.90 correlated pairs
- `evaluate_dfs_features(feature_store_path, n_trials=50)` → Gini gating (delta_gini >= 0.005)

**src/model.py:**
- `train_xgboost_optuna(feature_store_path, n_trials=50, groups=None)` → 8-param HPO (depth, lr, subsample, colsample, min_child_weight, gamma, reg_alpha, reg_lambda)
- `train_lightgbm_optuna(feature_store_path, n_trials=50, groups=None)` → 8-param HPO (depth, lr, num_leaves, min_data_in_leaf, feature_fraction, bagging_fraction, bagging_freq, reg_alpha)
- `train_catboost_optuna(feature_store_path, n_trials=50, groups=None)` → 7-param HPO (depth, lr, l2_leaf_reg, min_data_in_leaf, bagging_temperature, random_strength, subsample)
- `run_ensemble_workflow(X, y)` → loads 3 base models, trains stacking layer, gates on delta_gini
- `calibrate_model(model, X_train, y_train, X_test, y_test)` → Platt sigmoid via FrozenEstimator + CalibratedClassifierCV
- Private helpers: `_make_cv()` (auto-detect temporal or StratifiedKFold), `_fit_temporal_cv()`, `_log_best_trial_info()`

**src/utils.py:**
- `gini_coefficient(y_true, y_prob)` → 2×AUC − 1
- `ks_statistic(y_true, y_prob)` → max CDF separation via scipy.stats.ks_2samp
- `evaluate_model(model, X_test, y_test, model_name)` → full suite (AUC, Gini, KS, Brier, BrierSkill, AvgPrecision)
- `plot_roc_and_pr(model, X_test, y_test, model_name, save_path)` → 2-panel figure (ROC + Precision-Recall) with prevalence baseline

## Naming Conventions

**Files:**

- `data_loader.py`, `features.py`, `model.py`: Module names describe function (lower_snake_case)
- `test_X.py`: One test file per src/ module (test_{module_name}.py)
- `*.pkl`: Model artifacts via joblib
- `*.parquet`: Processed data (faster I/O than CSV for 300K+ rows)
- `*.csv`: Raw source data, config files, metric exports
- `*.json`: Hyperparameters, metrics, configuration

**Directories:**

- All lowercase: `src/`, `app/`, `data/`, `models/`, `reports/`, `scripts/`, `notebooks/`, `tests/`, `.planning/`

**Functions:**

- Public: `build_features()`, `train_xgboost_optuna()` (verb-noun pattern, camelCase)
- Private: `_engineer_financial_ratios()`, `_make_cv()` (underscore prefix, camelCase)
- Helpers: `_compute_woe_mapping_dict()` (verb-noun)

**Variables:**

- Constants: `_CV_EMBARGO_FRAC`, `_NAN_SENTINEL`, `_IV_STRONG` (UPPER_SNAKE_CASE, private if prefixed with _)
- Parameters: `df`, `X`, `y`, `n_trials`, `learning_rate` (snake_case)
- Boolean flags: `is_unbalance`, `allow_writing_files` (is_/allow_ prefix)

**Columns:**

- Aggregated: Prefixed by source table: `bureau_avg_balance`, `pos_cash_max_instalment`, `cc_utilization_std`
- Derived: Suffix for category: `_flag` (boolean), `_ratio` (scaled), `_days` (temporal)
- Raw (from application): No prefix: `AMT_CREDIT`, `DAYS_BIRTH`, `AGE_YEARS`

## Where to Add New Code

**New Feature:**
- Primary code: `src/features.py` (add private helper function + chain into `build_features()` or `engineer_secondary_features()`)
- Tests: `tests/test_features.py` (TDD: write test first, implement after)
- Example: Adding a `DEBT_TO_INCOME_RATIO` feature
  1. Write test in `test_features.py` that calls `engineer_application_features(df)` and asserts `DEBT_TO_INCOME_RATIO` column exists
  2. Add computation in `_engineer_financial_ratios()` or new helper
  3. Chain helper into `build_features()` after existing helpers
  4. Update `build_tree_feature_store()` if feature is included in raw pipeline

**New Model:**
- Implementation: `src/model.py` (follow `train_xgboost_optuna()` pattern: path-based API, Optuna HPO, 2-stage refit, Platt calibration)
- Optuna objective function: Add as `_model_optuna_objective()` with docstring specifying search space
- Return signature: 5-tuple `(model, metrics_dict, X_test, y_test, best_params)` for consistency
- Tests: `tests/test_model.py` (mock_data parquet fixture reused; add tests covering HPO, calibration, ensemble integration)
- Temporal validation: Must follow Basel CRE36.54 workflow (OOT carve before Optuna study creation)

**New Evaluation Metric:**
- Location: `src/utils.py` (add function following `gini_coefficient()` docstring style)
- Integration: Add to `evaluate_model()` return dictionary
- Tests: `tests/test_utils.py`

**New Utility or Helper:**
- Data helpers: `src/data_loader.py` (if related to join/aggregation)
- Feature helpers: `src/features.py` (if related to engineering)
- Model helpers: `src/model.py` (if related to training/CV)
- Metrics: `src/utils.py`
- Inference: `app/api.py` or `app/streamlit_app.py`

**Batch Script:**
- Location: `scripts/{task_name}.py`
- Pattern: Load data → run functions from src/ → save results to reports/ or models/
- CLI args: Use argparse for inputs (e.g., --feature-store, --n-trials)
- Do not import from app/ (keeps app deployment-focused)

## Special Directories

**data/processed/:**
- Purpose: Generated feature matrices and processed data
- Generated: By `save_training_frame()` (X_train.parquet, y_train.parquet) or feature store builders
- Committed: No — .gitignore excludes *.parquet
- Regeneration: Always rebuild on branch checkout to avoid stale features
- Current contents:
  - X_train.parquet: 307,511 × 195 (raw joined)
  - y_train.parquet: 307,511 × 1 (binary TARGET, 8% defaults)
  - X_tree_raw.parquet: 307,511 × 143 (raw engineered, Wave 1 features, primary tree model input)
  - X_tree_dfs.parquet: 307,511 × ~290 (raw + DFS, experimental — confirmed to not improve LGB/XGB in ablation)
  - X_features.parquet: 307,511 × 68 (WoE-binned, logistic regression only)

**models/:**
- Purpose: Persisted model artifacts (generated during training)
- Generated: By `train_*_optuna()` (*.pkl), Optuna trials (params.json)
- Committed: No — .gitignore excludes *.pkl, *.db
- **CRITICAL:** optuna_studies.db must NEVER be deleted; contains trial history for all HPO runs
- Regeneration: Retrain HPO to regenerate best params; studies continue appending to existing database
- Current valid models:
  - xgboost_raw_calibrated.pkl: OOT Gini=0.5666
  - lightgbm_raw_calibrated.pkl: OOT Gini=0.5746 (Basel CRE36.54 compliant)
  - catboost_raw_calibrated.pkl: OOT Gini=0.5699 (Basel CRE36.54 compliant)

**reports/:**
- Purpose: Evaluation results, figures, benchmarks
- Generated: By `plot_roc_and_pr()` (PNG figures), `evaluate_model()` (JSON metrics)
- Committed: No — .gitignore excludes reports/
- Contents: CSV tables for imbalance comparison, JSON for model metrics, PNG for plots

**.planning/codebase/:**
- Purpose: GSD orchestrator codebase documentation
- Files: ARCHITECTURE.md, STRUCTURE.md, CONVENTIONS.md, TESTING.md, STACK.md, INTEGRATIONS.md, CONCERNS.md
- Committed: Yes
- Updated: On-demand by GSD mapper agents; consumed by planner/executor agents during phase execution
- Audience: GSD agents and future developers

**notebooks/:**
- Purpose: Exploratory analysis, NOT production code
- Committed: Yes, with outputs stripped by nbstripout
- Usage: Reference only; re-run with `jupyter nbconvert --to notebook --execute notebook.ipynb` if needed
- Contents: EDA, feature validation, result visualization

---

*Structure analysis: 2026-04-11*
