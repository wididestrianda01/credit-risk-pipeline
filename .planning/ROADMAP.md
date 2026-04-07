# Roadmap: Credit Risk Scoring Pipeline

**Created:** 2026-04-07
**Granularity:** Fine (focused phases, clear go/no-go gates)
**Execution:** Sequential (ML phases have hard dependencies)
**Target:** Gini ≥ 0.70 calibrated PD model with full explainability and deployment

---

## Milestone 1 — Tree Model Foundation (Phases 04.2.1–04.2.2)

Establish the correct data pipeline for tree models: raw features without WoE encoding.

### Phase 04.2.1 — Fix Tree Feature Store

**Goal:** Build `X_tree_raw.parquet` — 155+ raw engineered features, full 307K rows, no WoE

**Why this exists:** All prior tree HPO used WoE-encoded 63-feature stores, invalidating XGB/LGB/CatBoost results. Root cause: `prepare_feature_pipelines.py` did an 80/20 pre-split before feature extraction (246K rows instead of 307K) AND applied WoE encoding to tree features.

**Requirements:** DATA-03, TEST-02 (partial)

**Plans:**
4/4 plans executed
- [x] 04.2.1-02-PLAN.md — Build `build_tree_feature_store()` in `src/features.py`
- [x] 04.2.1-03-PLAN.md — Validate `X_tree_raw.parquet`: shape, no WoE, no leakage
- [x] 04.2.1-04-PLAN.md — Add TDD tests for `build_tree_feature_store()`

**Done condition:** `X_tree_raw.parquet` exists with shape (307511, N≥100), columns are raw engineered (no `_woe` suffix), all tests pass

---

### Phase 04.2.2 — DFS Auto-Feature Augmentation

**Goal:** Fix featuretools entity-set and build `X_tree_dfs.parquet` with auto-aggregated features

**Why this exists:** `X_featuretools.parquet` has 0 columns — entity-set was built with empty child tables or wrong column specifications. DFS adds relational cross-table features beyond what manual aggregation provides.

**Requirements:** DATA-04, TEST-02 (complete)

**Plans:**
1. Diagnose `src/auto_features.py` entity-set build failure — trace why child tables produce 0 DFS columns
2. Fix entity-set construction: correct relationship definitions, validate each child table has rows
3. Re-run DFS with corrected entity-set; validate generated features have variance
4. Merge DFS features with `X_tree_raw.parquet` → `X_tree_dfs.parquet`
5. Add TDD tests for corrected DFS pipeline

**Done condition:** `X_tree_dfs.parquet` exists with shape (307511, N>155), DFS columns have non-zero variance, merged cleanly with raw features

---

## Milestone 2 — Tree Model Training (Phases 04.2.3–04.2.5)

Train and optimize all three tree models on the corrected raw+DFS feature store.

### Phase 04.2.3 — XGBoost HPO on Raw Features

**Goal:** Train XGBoost with Optuna HPO on `X_tree_dfs.parquet`; Gini > 0.55 (beat prior 0.5296 on wrong store)

**Requirements:** MODEL-02, CALIB-02, CALIB-03

**Plans:**
1. Update `train_xgboost_optuna()` to accept raw feature store path parameter
2. Continue existing Optuna study from `models/optuna_studies.db` (do NOT restart)
3. Run HPO: 50–100 trials, temporal CV with 2% embargo, `scale_pos_weight = n_neg/n_pos`
4. Apply Platt calibration; save `models/xgboost_raw_calibrated.pkl`
5. Generate reliability diagram + ROC/PR figure
6. Add TDD tests for raw-feature training path

**Done condition:** XGBoost Gini > 0.60 on temporal CV, BrierSkill > 0, model artifact saved with metrics JSON

---

### Phase 04.2.4 — LightGBM HPO on Raw Features

**Goal:** Train LightGBM with Optuna HPO on raw+DFS features; Gini comparable to or exceeding XGBoost

**Why raw features matter for LGB:** LGB's leaf-wise growth and GOSS sampling exploit continuous feature distributions — WoE binning eliminated this advantage entirely. Expected significant improvement over prior Gini=0.4519.

**Requirements:** MODEL-03, CALIB-02 (LGB), CALIB-03 (LGB)

**Plans:**
1. Update `train_lightgbm_optuna()` to use raw feature store
2. Continue/create LGB Optuna study in `models/optuna_studies.db`
3. Run HPO: 50–100 trials, `n_estimators_max=1000`, `is_unbalance=True`, temporal CV
4. Apply Platt calibration; save `models/lightgbm_raw_calibrated.pkl`
5. Generate reliability diagram + ROC/PR figure
6. Add TDD tests for raw-feature LGB training path

**Done condition:** LightGBM Gini > 0.60, BrierSkill > 0, model artifact saved with metrics JSON

---

### Phase 04.2.5 — CatBoost HPO on Raw Features

**Goal:** Train CatBoost with Optuna HPO, leveraging native categorical feature handling

**Why CatBoost matters:** Native ordered boosting + categorical encoding without manual WoE; `prepare_catboost_features()` swaps WoE categoricals back to raw strings.

**Requirements:** MODEL-04, CALIB-02 (CatBoost), CALIB-03 (CatBoost)

**Plans:**
1. Update `train_catboost_optuna()` to use raw feature store with `prepare_catboost_features()`
2. Continue/create CatBoost Optuna study in `models/optuna_studies.db`
3. Run HPO: 50 trials, 5-dim search space, temporal CV, `scale_pos_weight`
4. Apply calibration; save `models/catboost_raw_calibrated.pkl`
5. Generate reliability diagram + ROC/PR figure
6. Add TDD tests

**Done condition:** CatBoost Gini > 0.55, BrierSkill > 0, model artifact saved with metrics JSON

---

## Milestone 3 — Ensemble & Performance Gate (Phase 04.2.6)

### Phase 04.2.6 — Ensemble and Gini Gate

**Goal:** Blend/stack best tree models; achieve Gini ≥ 0.60

**Requirements:** MODEL-05, EVAL-02

**Plans:**
1. Update `train_ensemble()` to use raw-feature model artifacts
2. Run `run_ensemble_workflow()`: OOF stacking of XGB + LGB + (optional) CatBoost
3. Generate full benchmark table: LR, XGB, LGB, CatBoost, Ensemble — Gini, KS, BrierSkill
4. Save `models/ensemble_best.pkl`; record `reports/model_benchmark.csv`
5. **Go/no-go gate:** if ensemble Gini < 0.65, open additional HPO trials before proceeding
6. Add TDD tests for ensemble workflow

**Done condition:** Ensemble Gini ≥ 0.70 OR documented decision to proceed with best available (>0.65)

---

## Milestone 4 — Explainability (Phase 04.3)

### Phase 04.3 — SHAP Explainability and Fairness

**Goal:** Implement `src/explain.py` with global + local SHAP plots and regulatory fairness metrics

**Requirements:** EXPLAIN-01 through EXPLAIN-04, TEST-04

**Plans:**
1. Implement `compute_shap_values(model, X)` using `shap.TreeExplainer`
2. Implement `plot_shap_summary(shap_values, X, save_path)` — global beeswarm + bar
3. Implement `plot_shap_local(shap_values, X, idx, save_path)` — waterfall + force plots
4. Implement `compute_fairness_metrics(model, X, y, sensitive_cols)` — demographic parity + equalised odds
5. Structure SHAP output dict for adverse action notices (top-5 negative factors per applicant)
6. Add TDD tests for all explainability functions

**Done condition:** All 4 EXPLAIN requirements satisfied, figures saved to `reports/figures/`, tests passing

---

## Milestone 5 — Deployment (Phases 05.1–05.2)

### Phase 05.1 — FastAPI Production Endpoint

**Goal:** Production-ready `/predict` endpoint with authentication and SHAP output

**Requirements:** DEPLOY-01 through DEPLOY-03

**Plans:**
1. Replace stub in `app/api.py` — implement POST `/predict` with Pydantic request/response models
2. Add API key authentication middleware
3. Load best calibrated model on startup; serve PD + SHAP top-5 negative factors
4. Add `/health` with model version + uptime
5. Integration tests for API endpoints

**Done condition:** `uvicorn app.api:app` starts, `/predict` returns calibrated PD + SHAP factors, API key required

---

### Phase 05.2 — Streamlit Dashboard

**Goal:** Interactive Streamlit app for applicant risk scoring with SHAP waterfall visualization

**Requirements:** DEPLOY-04

**Plans:**
1. Replace placeholder in `app/streamlit_app.py` — applicant input form (key features as sliders/dropdowns)
2. Call FastAPI `/predict` endpoint or load model directly; display PD with risk tier (Low/Medium/High)
3. Render SHAP waterfall chart for the applicant
4. Add feature contribution table (top 10 positive/negative factors)
5. Manual E2E test: applicant entry → PD output → SHAP waterfall visible

**Done condition:** `streamlit run app/streamlit_app.py` starts, full applicant scoring flow works end-to-end

---

## Milestone 6 — LaTeX Report (Phase 06)

### Phase 06 — Research Report

**Goal:** Complete LaTeX report: methodology, model comparison, fairness analysis, business interpretation

**Requirements:** REPORT-01 through REPORT-03

**Plans:**
1. Write report structure: abstract, data description, methodology, feature engineering, model training
2. Model comparison section: benchmark table, ROC/PR figures, calibration reliability diagrams
3. Fairness and explainability section: SHAP global plots, fairness metric tables
4. Business interpretation: EL calculation example, regulatory compliance summary
5. Compile PDF; verify all figure references resolve

**Done condition:** PDF compiles without errors, all figures referenced, model comparison table present, >15 pages

---

## Phase Dependencies

```
Phase 04.2.1 → Phase 04.2.2 → Phase 04.2.3 → Phase 04.2.4 → Phase 04.2.5 → Phase 04.2.6
                                                                              ↓
                                                                       Phase 04.3
                                                                              ↓
                                                                       Phase 05.1 → Phase 05.2
                                                                              ↓
                                                                       Phase 06
```

All phases are strictly sequential — each builds on prior phase outputs.

## Progress Summary

| Phase | Status | Target |
|-------|--------|--------|
| Phase 1 — Data loading + EDA | ✅ Complete | — |
| Phase 2 — WoE feature engineering | ✅ Complete | — |
| Phase 3 — LR baseline + eval utilities | ✅ Complete | Gini ≥ 0.45 |
| Phase 04.2.1 — Fix raw feature store | ✅ Complete | `X_tree_raw.parquet` (307K×211, 0 NaN, 0 WoE) |
| Phase 04.2.2 — DFS augmentation | 🔲 Not started | `X_tree_dfs.parquet` (307K, >155) |
| Phase 04.2.3 — XGBoost HPO | 🔲 Not started | Gini > 0.55 |
| Phase 04.2.4 — LightGBM HPO | 🔲 Not started | Gini > 0.55 |
| Phase 04.2.5 — CatBoost HPO | 🔲 Not started | Gini > 0.50 |
| Phase 04.2.6 — Ensemble + gate | 🔲 Not started | Gini ≥ 0.60 |
| Phase 04.3 — SHAP + fairness | 🔲 Not started | All EXPLAIN reqs |
| Phase 05.1 — FastAPI endpoint | 🔲 Not started | `/predict` live |
| Phase 05.2 — Streamlit dashboard | 🔲 Not started | E2E flow works |
| Phase 06 — LaTeX report | 🔲 Not started | PDF compiles |

---
*Roadmap created: 2026-04-07*
*Plans added: 2026-04-07*
