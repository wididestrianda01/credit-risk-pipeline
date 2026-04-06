# Architecture

**Analysis Date:** 2026-04-06

## Pattern Overview

**Overall:** Functional pipeline with immutable data transformations.

**Key Characteristics:**
- Linear processing chain: raw data → join → engineer → train → score
- Each module accepts DataFrames, returns new DataFrames (no in-place mutation)
- Prefix-namespaced features link back to source tables (e.g., `bureau_`, `prev_`, `pos_`, `cc_`, `inst_`)
- Separation of concerns: data loading, feature engineering, model training, evaluation, explainability
- Multiple feature paths: hand-engineered (canonical) and auto-generated via featuretools
- Tree-based models (LightGBM/XGBoost) as primary; logistic regression as interpretable baseline

## Layers

**Data Loading Layer:**
- Purpose: Join 7 heterogeneous CSV tables and enforce dtypes
- Location: `src/data_loader.py`
- Contains: `load_data()`, `build_training_frame()`, `save_training_frame()`, aggregation helpers per secondary table
- Depends on: pandas, numpy
- Used by: feature engineering pipeline, test fixtures

**Feature Engineering Layer:**
- Purpose: Transform raw joined DataFrame into model-ready feature matrix
- Location: `src/features.py`, `src/auto_features.py`
- Contains: 
  - Hand-engineered: financial ratios, demographics, document flags, EXT_SOURCE composites, cross-table interactions
  - Auto-engineered: featuretools DFS (deep feature synthesis) for automatic aggregation
  - WoE binning and information value (IV) filtering
  - Feature store persistence (pickle + parquet)
- Depends on: pandas, numpy, scikit-learn, featuretools (optional)
- Used by: model training, prediction API

**Model Training Layer:**
- Purpose: Train, tune, and calibrate credit risk classifiers
- Location: `src/model.py`
- Contains: 
  - LightGBM with Optuna Bayesian hyperparameter optimization
  - XGBoost with Optuna Bayesian hyperparameter optimization
  - CatBoost classifier
  - Logistic regression baseline (WoE scoring card)
  - Stratified k-fold cross-validation with temporal embargo
  - Probability calibration (Platt scaling via sklearn)
  - Ensemble voting (temporal CV with out-of-fold predictions)
- Depends on: lightgbm, xgboost, catboost, optuna, scikit-learn, joblib
- Used by: evaluation scripts, inference endpoint

**Evaluation Layer:**
- Purpose: Compute regulatory credit risk metrics and produce publication-ready plots
- Location: `src/utils.py`
- Contains: 
  - `gini_coefficient(y_true, y_prob)` — Gini = 2 × AUC − 1 (primary Basel III metric)
  - `ks_statistic(y_true, y_prob)` — maximum CDF separation
  - `evaluate_model(model, X_test, y_test)` — full metric suite (AUC, KS, Brier, BrierSkill, AvgPrecision)
  - `plot_roc_and_pr(model, X_test, y_test)` — 2-panel ROC + Precision-Recall figure
  - Stubs: `roc_curve_plot()`, `calibration_plot()` (reserved for Phase 4)
- Depends on: scikit-learn, scipy.stats, matplotlib
- Used by: training scripts, reports

**Explainability Layer:**
- Purpose: SHAP-based feature attribution and fairness analysis (Phase 4)
- Location: `src/explain.py`
- Contains: Stubs for `compute_shap_values()`, `fairness_report()`
- Depends on: shap (future)
- Used by: Streamlit dashboard, fairness audit

**Deployment Layer:**
- Purpose: Serve trained models via HTTP endpoints
- Location: `app/api.py`, `app/streamlit_app.py`
- Contains: FastAPI `/predict` endpoint (stub), Streamlit dashboard (placeholder)
- Depends on: fastapi, pydantic, streamlit, joblib
- Used by: external applications, business users

## Data Flow

**Training Pipeline:**

1. **Data Ingestion** (`data_loader.py`)
   - Load 7 raw CSV tables from `data/` directory
   - Join on SK_ID_CURR with structured aggregation (secondary tables → one row per applicant)
   - Enforce dtypes (categorical, int64, float64) and fill missing with -999 sentinel
   - Output: X_train (307,511 × ~100 raw cols), y_train (binary TARGET)

2. **Feature Engineering** (`features.py`)
   - Apply domain-driven transformations: financial ratios, demographics, document flags, composites
   - Apply featuretools auto-aggregation (optional path via `auto_features.py`)
   - Compute Information Value (IV) and filter features (IV ≥ 0.02)
   - WoE binning: transform quantile-discretised features to log-odds space
   - Variance filtering (drop zero-variance features post-binning)
   - Correlation deduplication (keep higher-IV feature when |r| > 0.90)
   - Output: X_features (307,511 × 40–68 final features, depending on path)

3. **Model Training** (`model.py`)
   - 5-fold or 10-fold stratified CV with temporal embargo (López de Prado, Ch. 7)
   - Per fold: train → validate → compute Gini
   - Optuna Bayesian HPO: 50 trials per estimator (LGB, XGB)
   - Final refit on full training set, evaluate on hold-out test set
   - Probability calibration via Platt scaling (sigmoid fit on 30% calibration split)
   - Output: model.pkl, params.json, metrics.json

4. **Evaluation** (`utils.py`)
   - Compute Gini, KS, Brier, BrierSkill on test set
   - Generate ROC + PR curves
   - Persist results to `reports/{model}_results.json`, figures to `reports/figures/`

5. **Inference** (`app/api.py`)
   - Load model.pkl + feature store (WoE mappings, column list)
   - Apply feature pipeline to raw applicant features
   - Score with calibrated model → return PD estimate
   - (Full implementation: Phase 5)

**Feature Store Persistence:**

- `models/woe_mappings.pkl`: dict of `{feature_name: pd.Series(bin_edges)}` for inference-time binning
- `models/raw_feature_columns.pkl`: list of column names (order-sensitive for model input)
- `data/processed/X_features.parquet`: Full 307,511 × 40+ feature matrix (generated post-engineering)

**State Management:**

- **Training-only state:** train/val splits, fold indices, Optuna storage (in-memory)
- **Inference state:** model coefficients/trees (pkl), WoE bin edges (pkl), feature scaler params (in-pipeline)
- **Immutability:** all transforms return new DataFrames; original data never mutated
- **Reproducibility:** fixed `_RANDOM_STATE=42` in all splits, Optuna uses seed for determinism

## Key Abstractions

**Aggregation Functions (data_loader.py):**
- Purpose: Collapse N:M secondary table relationships to 1:1 (per SK_ID_CURR)
- Examples: `_aggregate_bureau_balance()`, `_aggregate_previous_applications()`
- Pattern: pandas groupby → mean/std/min/max/count over temporal snapshots

**Feature Engineering Helpers (features.py):**
- Purpose: Single-concern transformation with edge-case guards
- Examples: `_engineer_financial_ratios()`, `_engineer_demographics()`, `_engineer_documents()`
- Pattern: input DataFrame → output DataFrame, division-by-zero guards, inf→0, NaN→-999

**WoE Transform (features.py):**
- Purpose: Encode categorical/binned numeric features as log-odds
- Functions: `compute_woe_iv()`, `_bin_feature_and_compute_woe()`
- Pattern: Fit on training data (bin edges, WoE per bin); apply on test via `pd.cut()` (never `pd.qcut()`)
- Motivation: LightGBM/XGBoost with WoE-transformed input + logistic regression baseline share same feature space

**Cross-Validation Strategy (_make_cv(), model.py):**
- Purpose: Detect temporal ordering and apply embargo to prevent serial correlation leakage
- Pattern: If `_TEMPORAL_SORT_COL` is in X, sort by it and apply `TimeSeriesSplit(gap=embargo)`; else `StratifiedKFold`
- Embargo fraction: `_CV_EMBARGO_FRAC=0.02` (2% of each fold discarded at train/val boundary)

**Hyperparameter Optimization (model.py):**
- Purpose: Bayesian search with Optuna + reproducible CV loop
- Functions: `train_xgboost_optuna()`, `train_lightgbm_optuna()`, `train_catboost_optuna()`
- Pattern: Optuna trial → split data → CV loop → average metric → return to Optuna
- Objective: Gini coefficient (primary), with fallback to AUC if single-class fold encountered

**Probability Calibration (model.py):**
- Purpose: Fit sigmoid transform to uncalibrated scores → PD for EL = PD × LGD × EAD
- Function: `calibrate_model(model, X_train, y_train, X_test, y_test)`
- Pattern: 70/30 split of X_train → base model frozen, sklearn CalibratedClassifierCV fits sigmoid on 30%
- Preservation: Gini unaffected (monotonic transform); Brier improved (calibration moves probs closer to true empirical frequencies)

## Entry Points

**CLI/Scripts:**
- `scripts/eval_priority12.py`: Full pipeline evaluation (train 3 models, ensemble, persist best)
- `scripts/calibrate_xgboost.py`: Platt-calibrate a fitted model
- `scripts/lgb_integration_run.py`: LightGBM integration test with multiple configs
- `scripts/rebuild_feature_store.py`: Regenerate feature store from raw data

**Jupyter Notebooks:**
- `notebooks/01_eda_and_data_quality.ipynb`: Data profiling, missingness patterns, target imbalance analysis
- `notebooks/02_feature_engineering.ipynb`: Feature derivation, IV ranking, WoE binning visualization
- `notebooks/03_modeling_and_evaluation.ipynb`: Training runs, metric comparison, calibration diagnostics
- `notebooks/04_explainability_and_fairness.ipynb`: SHAP plots, fairness metrics (placeholder, Phase 4)

**API Endpoint:**
- `app/api.py`: FastAPI server with GET `/health` (working) and POST `/predict` (stub, Phase 5)

## Error Handling

**Strategy:** Explicit validation at layer boundaries; raise with context; no silent failures.

**Patterns:**

- **Division by zero:** Guard with `np.where(denom > 0, num / denom, 0.0)` before any ratio
- **Inf/NaN:** Replace inf with 0.0 immediately after computation; fill remaining NaN with -999 (tree-friendly)
- **Single-class folds:** Explicit check in CV loop; if single class detected, skip fold or return 0.0 Gini with warning
- **Missing files:** Raise `FileNotFoundError` with path; let caller decide recovery (don't silently skip)
- **Temporal leakage:** Validate `_TEMPORAL_SORT_COL` is in X; auto-switch to StratifiedKFold if not found
- **Feature mismatch at inference:** Raise if WoE mappings don't match X column names; dump diagnostics to stderr

## Cross-Cutting Concerns

**Logging:** Uses `warnings.warn()` and `print()` sparingly (library code avoids print); scripts use `logging` module for production runs.

**Validation:** Input dtypes checked at module entry points (features.py, model.py); output shapes logged to JSON reports.

**Configuration:** All hyperparameters and thresholds exposed as `_CONSTANT` module-level variables; search spaces documented in docstrings.

**Reproducibility:** `_RANDOM_STATE=42` fixed globally; Optuna uses seed for determinism; temporal CV auto-detected to flag time-awareness requirement.

**Metrics Reporting:** All model results persisted to `reports/{model}_results.json` + CSV tables; figures saved with 300 DPI, no `plt.show()` calls in library code.

---

*Architecture analysis: 2026-04-06*
