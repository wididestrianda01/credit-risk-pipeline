# Architecture

**Analysis Date:** 2026-04-11

## Pattern Overview

**Overall:** Linear transformation pipeline with functional immutability; Basel CRE36.54 temporal validation

**Key Characteristics:**
- Pure function pipeline: each module accepts DataFrames and returns new DataFrames without mutation
- **Mandatory temporal validation:** Train/OOT carving BEFORE Optuna HPO starts; prevents leakage (enforced in code commit 43c9d89)
- Path-based APIs: all model training functions load parquet feature stores and persist models to disk
- Two parallel feature pipelines: WoE-encoded (logistic regression) and raw engineered (tree models)
- 7-table relational schema: joined on SK_ID_CURR, secondary tables aggregated 1:M → 1:1
- Feature column prefixes: structural mapping for SHAP attribution (bureau_, prev_, pos_, inst_, cc_)

## Layers

**Data Ingestion (`src/data_loader.py`):**
- Purpose: Load 7 CSV source tables and aggregate to applicant level (one row per SK_ID_CURR)
- Location: `src/data_loader.py` (~350 lines)
- Contains: `load_data()`, `build_training_frame()`, `save_training_frame()` entry points; 5 secondary-table aggregators
- Depends on: pandas, numpy, pathlib for I/O and aggregation
- Used by: Feature engineering pipeline; test fixtures
- Invariants: Enforces categorical dtype for 16 columns; applies -999 sentinel to missing secondary-table aggregates; returns consistent shape (307,511, ~195)

**Feature Engineering — WoE Pipeline (`src/features.py`):**
- Purpose: Domain-rich feature transformations for interpretability; Weight of Evidence binning for logistic regression
- Location: `src/features.py` (~1,300 lines)
- Contains: `build_features()` (raw engineering), `build_feature_store()` (WoE: 130 raw → 140 engineered → 68 IV-filtered), `build_tree_feature_store()` (raw only: 143+ features, no WoE)
- Depends on: `data_loader` output; sklearn VarianceThreshold; numpy for guarded division
- Used by: Model training pipelines; evaluation
- Outputs:
  - `data/processed/X_features.parquet` — 307,511 × 68 WoE-encoded features (logistic/interpretability)
  - `data/processed/X_tree_raw.parquet` — 307,511 × 143 raw engineered features, no WoE (tree model input)
- Wave 1 features (Phase 04.2.7): 7 new delinquency features (inst_late_rate_12m, inst_late_rate_recent_vs_historical, inst_rolling_30dpd_ratio_3m, inst_delinquency_escalation_flag, inst_days_since_last_30dpd, bureau_dpd_trend_3m_vs_12m, bureau_debt_to_new_credit)
- Invariants: All paths use `_PROJECT_ROOT` for absolute safety; aggregated features prefixed by source table; boolean flags use `_flag` suffix; NaN filled with -999 sentinel; WoE clipped to ±5.0; IV threshold filters at 0.02, 0.1, 0.3

**DFS Auto-Feature Synthesis (`src/auto_features.py`):**
- Purpose: Featuretools Deep Feature Synthesis to generate automatic aggregations from 7-table relational structure
- Location: `src/auto_features.py` (~700 lines)
- Contains: `build_featuretools_feature_store()`, `apply_featuretools_feature_store()`, `deduplicate_dfs_features()`, `evaluate_dfs_features()`
- Depends on: Featuretools + Woodwork LogicalType annotations; `train_xgboost_optuna()` for gating evaluation
- Used by: Tree model feature stores; evaluation pipeline
- Output: `data/processed/X_tree_dfs.parquet` — 307,511 × ~290 raw engineered + DFS auto-aggregates
- Invariants: 15 leaky SK_DPD columns removed (post-origination payment distress, Basel III Article 174); >0.90 correlation pairs deduplicated; gating: DFS persists only if delta_gini >= 0.005 against baseline (evaluated in Phase 04.2.4, found DFS hurts LGB/XGBoost)

**Model Training — Temporal Validation + HPO (`src/model.py`):**
- Purpose: XGBoost, LightGBM, CatBoost training with Basel CRE36.54 compliant temporal CV and Optuna Bayesian HPO
- Location: `src/model.py` (~1,400 lines)
- Contains: `train_xgboost_optuna()`, `train_lightgbm_optuna()`, `train_catboost_optuna()`, `run_ensemble_workflow()`, `calibrate_model()`
- Depends on: sklearn, xgboost, lightgbm, catboost, optuna; sqlalchemy for persistent study backend
- Used by: Scripts, ensemble coordination
- Key signatures (all path-based):
  - `train_xgboost_optuna(feature_store_path: str, n_trials=50, groups=None)` → `(model, metrics_dict, X_test, y_test, best_params)`
  - `train_lightgbm_optuna(feature_store_path: str, n_trials=50, groups=None)` → same tuple
  - `train_catboost_optuna(feature_store_path: str, n_trials=50, groups=None)` → same tuple
  - `run_ensemble_workflow(X, y)` → persists only if `ensemble_gini - max(lgb, xgb, cat) >= 0.005`
- **MANDATORY Basel CRE36.54 temporal validation workflow (enforced in code — commit 43c9d89 fixed LGB/CatBoost):**
  1. Sort full dataset by `prev_days_decision_mean` (NaN rows → seeded random permutation proportional to test fraction)
  2. **Carve OOT = most-recent 20% BEFORE Optuna study creation** — frozen, never accessible during HPO
  3. Optuna HPO on 80% train only: K-fold temporal CV; OOF Gini is trial objective
  4. Select best hyperparameters by OOF Gini; retrain on full 80% (single fit, no CV)
  5. Evaluate on frozen OOT → **OOT Gini is the regulatory metric**
- Violation: Any split happening inside or after `objective()` is non-compliant; Phase 04.2.4 and 04.2.5 invalidated and re-run
- Model artifacts: joblib pickle format; calibration via Platt scaling (FrozenEstimator + CalibratedClassifierCV)
- Current valid models:
  - XGBoost raw+DFS: OOT Gini=0.5666, KS=0.4089 (Phase 04.2.3.2, valid)
  - LightGBM raw: OOT Gini=0.5746, KS=0.4302 (Phase 04.2.4.1, valid — Basel compliant re-run)
  - CatBoost raw: OOT Gini=0.5699, KS=0.4259 (Phase 04.2.5.1, valid — Basel compliant re-run)

**Evaluation Metrics (`src/utils.py`):**
- Purpose: Regulatory-grade metrics (Gini, KS), plotting, fairness evaluation
- Location: `src/utils.py` (~350 lines)
- Contains: `gini_coefficient()`, `ks_statistic()`, `evaluate_model()`, `plot_roc_and_pr()`, `roc_curve_plot()`, `calibration_plot()`
- Depends on: scipy.stats, sklearn.metrics, matplotlib
- Used by: Model training loops, evaluation scripts
- Invariants: Gini = 2×AUC−1; KS via scipy.stats.ks_2samp; BrierSkill = 1 − BS / (prevalence × (1−prevalence)) preferred at 8% imbalance; KS > 0.40 is Basel III strong separation threshold

**Explainability (Stub — `src/explain.py`):**
- Purpose: SHAP TreeExplainer for attribution; demographic parity and equalised odds fairness metrics
- Location: `src/explain.py` (~50 lines, stub)
- Contains: `compute_shap_values()`, `fairness_report()` — not implemented
- Status: Pending Phase 04.3
- Scope: Must support adverse action notices (GDPR Art. 22) and EU AI Act Art. 6 high-risk AI requirements

**API Deployment (`app/api.py`):**
- Purpose: FastAPI REST endpoint for real-time credit scoring
- Location: `app/api.py` (~45 lines)
- Contains: Pydantic request/response models; `/predict` endpoint (stub), `/health` liveness check
- Status: Endpoints stubbed; Phase 05.1
- Future: Load calibrated model, run feature pipeline, return PD + risk band

**Dashboard (`app/streamlit_app.py`):**
- Purpose: Interactive SHAP dashboard and model explainability UI
- Location: `app/streamlit_app.py`
- Status: Placeholder; Phase 05.2
- Future: SHAP waterfall/force plots, feature importance, fairness analysis by demographic group

## Data Flow

**Training Pipeline:**

```
data/raw/ (7 CSV tables, 307,511 rows)
    ↓
src/data_loader.py: load_data() + build_training_frame()
    ↓ (output: X_train.parquet 307511×195, y_train.parquet)
src/features.py: build_tree_feature_store() [+ src/auto_features.py for DFS option]
    ↓ (output: X_tree_raw.parquet 307511×143, X_tree_dfs.parquet 307511×~290)
src/model.py: train_*_optuna(feature_store_path)
    ├─ Step 1: Load parquet, extract TARGET column
    ├─ Step 2: Sort by prev_days_decision_mean (temporal proxy)
    ├─ Step 3: Carve OOT (20%) — FROZEN before Optuna study creation
    ├─ Step 4: Optuna HPO on 80% with OOF temporal CV (K-fold)
    ├─ Step 5: Select best params, retrain on 80% (single fit, no CV)
    └─ Step 6: Evaluate on frozen OOT
    ↓ (output: model.pkl, metrics.json, X_test, y_test)
src/model.py: calibrate_model() [Platt sigmoid via FrozenEstimator]
    ↓ (output: model_calibrated.pkl)
src/utils.py: evaluate_model() + plot_roc_and_pr()
    ↓ (output: evaluation metrics, ROC+PR figure)
src/model.py: run_ensemble_workflow() [Phase 04.2.6]
    ↓ (output: ensemble.pkl if delta_gini >= 0.005)
src/explain.py: compute_shap_values() + fairness_report() [Phase 04.3]
    ↓ (output: SHAP values, fairness audit)
app/api.py: /predict endpoint + app/streamlit_app.py dashboard
```

**Inference Pipeline (Post-Production):**

```
Applicant data (7 table structure or flat features)
    ↓
src/data_loader.py: load_data() → joined frame
    ↓
src/features.py: build_tree_feature_store() or build_features()
    ↓ (output: X_inference, same features as training)
[Loaded calibrated model].predict_proba(X_inference)
    ↓ (output: PD = probability of default)
src/explain.py: compute_shap_values(model, X_inference)
    ↓ (output: SHAP attributions per applicant)
Risk band mapping (PD → LOW/MEDIUM/HIGH)
    ↓
app/api.py: POST /predict response
```

**State Management:**

- **Feature stores (parquet):** Immutable snapshots; rebuilt when engineering changes; version controlled via commit hash
  - X_train.parquet: raw joined matrix (307511×195)
  - X_tree_raw.parquet: raw engineered (307511×143, Wave 1 features included)
  - X_tree_dfs.parquet: raw + DFS aggregates (307511×~290)
- **Models (joblib pickle):** Persisted to `models/` after training; linked to feature store via SQLite Optuna study metadata
  - xgboost_raw_calibrated.pkl: primary XGB model
  - lightgbm_raw_calibrated.pkl: LGB (Basel compliant)
  - catboost_raw_calibrated.pkl: CatBoost (Basel compliant)
- **Metrics (JSON):** Reported per model; OOT metrics are ground truth (OOF is internal trial evaluation only)
- **Optuna studies (SQLite):** `models/optuna_studies.db` persists trial history; studies keyed by model name (xgboost_raw_scalepos, lightgbm_raw_scale, catboost_raw_scalepos)

## Key Abstractions

**Feature Store:**
- Purpose: Immutable parquet snapshots of engineered features + target
- Examples: X_train.parquet, X_tree_raw.parquet, X_tree_dfs.parquet, X_features.parquet
- Pattern: Write-once, read-many; embedded TARGET column for model training; validated on load

**Training Result Tuple:**
- Purpose: Standardized return from `train_*_optuna()` functions
- Structure: `(model, metrics_dict, X_test, y_test, best_params)`
- Usage: Feeds downstream calibration and ensemble; enables integration testing

**Optuna Trial:**
- Purpose: Single hyperparameter configuration evaluation
- Flow: Sample → Objective evaluation (OOF Gini on 80% train only) → Prune → Persist to SQLite
- **Invariant: No OOT access during trial** (checked in code review commit 43c9d89)
- Temporal CV inside trial: K-fold on sorted 80% train with embargo fraction

**Column Prefixes (Feature Attribution):**
- Structural mapping from features back to source tables for SHAP analysis and regulatory traceability
- `bureau_*` ← bureau + bureau_balance
- `prev_*` ← previous_application
- `pos_*` ← POS_CASH_balance
- `inst_*` ← installments_payments
- `cc_*` ← credit_card_balance
- (no prefix) ← application_train

**Temporal Sort Column:**
- `prev_days_decision_mean`: mean days before application when previous applications were decided
- Role: Proxy for applicant vintage; enables temporal CV carving without explicit date columns
- Invariant: NaN rows → seeded random permutation proportional to test fraction (ensures reproducibility)

**Basel CRE36.54 Compliance Gate:**
- Purpose: Ensure temporal validation workflow prevents forward-looking leakage
- Check: OOT carving happens before Optuna study creation (enforced in `train_*_optuna()` signature)
- Audit: Code review commits (43c9d89, b6821ee) document fixes for LGB and CatBoost

## Entry Points

**Data Preparation:**
- Location: `src/data_loader.py::build_training_frame()`
- Triggers: Feature engineering pipeline startup
- Responsibilities: Load 7 CSV tables, enforce dtypes, return clean (X, y) pair

**Feature Engineering:**
- Location: `src/features.py::build_tree_feature_store()` or `build_feature_store()`
- Triggers: Model training pipeline
- Responsibilities: Apply domain transformations, aggregate secondary tables, handle missingness, enforce naming conventions

**Model Training:**
- Location: `src/model.py::train_xgboost_optuna()` / `train_lightgbm_optuna()` / `train_catboost_optuna()`
- Triggers: Scripts (e.g., `scripts/train_xgboost_raw.py`) or Phase executors
- Responsibilities: Load parquet feature store, carve OOT before Optuna, run HPO, calibrate, persist model + metrics

**Ensemble Workflow:**
- Location: `src/model.py::run_ensemble_workflow()`
- Triggers: Phase 04.2.6 (not started — unblocked as of 2026-04-11)
- Responsibilities: Load three trained models, train stacking layer, evaluate gate

**API Serving:**
- Location: `app/api.py` (FastAPI app)
- Triggers: `uvicorn app.api:app --reload`
- Responsibilities: Accept applicant features via POST /predict, run inference pipeline, return PD + risk band

## Error Handling

**Strategy:** Explicit raising with descriptive messages; no silent failures; validation at all system boundaries

**Patterns:**

1. **File I/O:** `FileNotFoundError` if parquet or CSV missing; caught and logged with path
2. **Data validation:** `ValueError` if shape mismatch, missing columns, or dtype inconsistency; raised before processing
3. **Feature engineering:** Guard against division by zero (use `np.where()` or sentinel fill); guard against empty DataFrames
4. **Model training:** `RuntimeError` if Optuna study fails to converge; `ValueError` if OOT carving impossible (e.g., insufficient rows for 20% split)
5. **Metrics:** `ValueError` if single-class labels passed to `gini_coefficient()` or `ks_statistic()`
6. **API:** Pydantic validation errors on request body; 500 responses if inference fails

## Cross-Cutting Concerns

**Logging:** 
- Using `logging` module (not `print()`) in library code
- XGBoost/LightGBM configured with `verbosity=-1` + callback silencers to suppress library output
- Model training progress: Optuna sampler logs trial count and best Gini per N trials
- Early stopping patience: LGB uses two-tier strategy (_LGB_OBJ_EARLY_STOPPING_ROUNDS=20 inside HPO, _LGB_EARLY_STOPPING_ROUNDS=50 for final refit)

**Validation:**
- Feature store shape checked on load (e.g., 307,511 rows expected)
- Column names validated (prefixes, missing columns)
- Target column checked for binary labels (0/1) and class balance (~8% defaults, ~92% non-defaults)
- Path safety: all paths use `_PROJECT_ROOT` computed from module location, never relative paths

**Authentication:**
- Not applicable (batch pipeline); API phase will require API key validation in FastAPI middleware

**Temporal Integrity:**
- **Mandatory:** No data leakage from OOT or test sets into training loops
- **Enforced:** OOT carving **before** Optuna study creation (commit 43c9d89 fix)
- **Verified:** Temporal CV embargo of 2% to prevent serial-correlation leakage (López de Prado, *Advances in Financial Machine Learning*)
- **Regulatory scope:** Basel III IRB CRE36.54 compliance for PD model validation; invalidated models re-trained in Phase 04.2.4.1 and 04.2.5.1

---

*Architecture analysis: 2026-04-11*
