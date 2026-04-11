# Roadmap: Credit Risk Scoring Pipeline

**Created:** 2026-04-07 | **Target:** Gini ≥ 0.60 calibrated PD, OOF–OOT gap ≤ 0.05

---

## Completed Phases

| Phase | Status | OOT Gini | Key Result |
|-------|--------|----------|------------|
| Phase 01 — Infrastructure | ✅ | — | Absolute paths, test isolation, import alias |
| Phase 02 — Recurring Infrastructure | ✅ | — | credit_engine removed; all model.py paths anchored |
| Phase 04.2.1 — Fix tree feature store | ✅ | — | `X_tree_raw.parquet` (307K×211, 0 WoE) |
| Phase 04.2.2 — DFS augmentation | ✅ | — | Woodwork entity-set fixed; 9 TDD tests |
| Phase 04.2.3 — XGBoost HPO | ⚠️ invalid | ~~0.9592~~ | SK_DPD leakage; path-based API and 8 HPO improvements valid |
| Phase 04.2.3.1 — SK_DPD removal | ✅ | — | 15 leaky cols removed; clean store rebuilt |
| Phase 04.2.3.2 — Feature engineering + XGB re-run | ✅ | 0.5666 | Gate fail (target 0.60); KS ✓; XGB plateau confirmed |
| Phase 04.2.3.3 — XGB store selection | ✅ superseded | — | Raw+eng wins; DFS adds +0.0001 (noise); absorbed into 04.2.4 |
| Phase 04.2.4 — LGB HPO | ❌ invalidated | — | Basel CRE36.54 OOT contamination; superseded by 04.2.4.1 |
| Phase 04.2.4.1 — LGB compliant re-run | ✅ | 0.5746 | KS=0.4302 ✓; regulatory evidence clean |
| Phase 04.2.5 — CatBoost HPO | ❌ invalidated | — | Basel CRE36.54 OOT contamination; superseded by 04.2.5.1 |
| Phase 04.2.5.1 — CatBoost compliant re-run | ✅ | 0.5699 | KS=0.4259 ✓, Brier=0.0831 ✓ |
| Phase 04.2.7 — Feature Engineering Enhancement | ✅ gate fail | 0.5746 | Wave 1 (7 delinquency features); gate < 0.5845 — no net lift |
| Phase 04.2.6 — Ensemble + Gate | ✅ investigate | 0.5749 | gate=investigate; LGB standalone (0.5746) proceeds as primary |

> **Basel CRE36.54 mandatory workflow (all model training):** Sort by `prev_days_decision_mean` → carve OOT (most-recent 20%, frozen) → Optuna HPO on 80% with OOF CV Gini as objective → retrain on full 80% → evaluate on frozen OOT. Any split inside the Optuna objective closure contaminates results.

---

## Phase 04.2.8 — Split model.py by Model Family ✅ Complete

**Goal:** Decompose `src/model.py` (4,251 lines) into focused modules without changing any public API or test behavior.

**Result:** 5 sibling modules + thin ~438-line facade; 174 tests pass; root cleanup done.
- `src/model_base.py` — constants + shared utilities
- `src/model_xgboost.py`, `src/model_lightgbm.py`, `src/model_catboost.py` — per-algorithm training
- `src/model_ensemble.py` — stacking + gate; `train_ensemble_3model` returns 5-tuple
- `src/model.py` — facade with explicit named re-exports
- Root: `test_hpo_runner.py` + `verify_tree_feature_store.py` moved to `scripts/`; `.gitignore` updated (`*.db`, `*.log`, `catboost_info/`)

**Commits:** `b5548f2` (split), `bfbfe8b` (cleanup)

---

## Phase 04.2.9 — Feature Engineering Expansion

**Goal:** Break through the ~0.575 Gini ceiling by implementing 10+ missing high-signal features, rebuilding the feature store, and re-running all three base model HPOs.

**Why this exists:** Ensemble post-mortem confirmed all three base models converge to the same Gini range regardless of algorithm — this is a feature information ceiling, not an HPO problem. CLAUDE.md documents at least 12 high-signal features from the Phase 2 plan that were not yet implemented.

**Gate:** Any single base model reaches OOT Gini ≥ 0.580 (improvement ≥ 0.004 over LGB best 0.5746)

**Plans:**
1. Audit `X_tree_raw.parquet` columns against CLAUDE.md Phase 2 feature plan — produce gap matrix
2. Implement affordability ratios: `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `CREDIT_TERM`, `GOODS_CREDIT_RATIO` (guard all divisions; clip [0,inf), inf→0, NaN→−999)
3. Implement EXT_SOURCE composites (`EXT_SOURCE_MEAN`, `EXT_SOURCE_MIN`, `EXT_SOURCE_NUM_AVAILABLE`) + employment stability (`YEARS_EMPLOYED`, `EMPLOYED_TO_AGE_RATIO`, `DAYS_EMPLOYED` sentinel clip)
4. Implement document missingness features (`DOCUMENTS_SUBMITTED`, `HIGH_RISK_DOC_MISSING`)
5. Rebuild `X_tree_raw.parquet`; re-run LGB 50-trial HPO gate check
6. Re-run XGBoost + CatBoost HPO on new store; update `model_benchmark.csv`
7. TDD tests for all new feature functions

**Done condition:** `X_tree_raw.parquet` rebuilt with ≥10 new features; ≥1 base model OOT Gini ≥ 0.580; all tests pass; old store backed up as `X_tree_raw_v1.parquet`

**Status:** 🔲 Not started — unblocked after Phase 04.2.7

---

## Phase 04.2.10 — Ensemble Enhancement via Feature Diversity

**Goal:** Re-ensemble using *different* feature stores per base model to create genuine prediction diversity; implement rank-based ensemble and MLP meta-learner; target ensemble OOT Gini ≥ 0.600.

**Why this exists:** Phase 04.2.6 gate=investigate root cause: all three base models trained on identical `X_tree_raw` → OOF correlations ≥ 0.95 → meta-learner had no orthogonal signal. Fix: LGB stays on `X_tree_raw`, XGBoost moves to `X_features` (WoE), CatBoost moves to `X_tree_dfs` (DFS). This creates structurally distinct OOF residuals.

**Gate:** Ensemble OOT Gini ≥ 0.600

**Plans:**
1. Pre-calibrate CatBoost OOF predictions before meta-learner training (uncalibrated OOF BrierSkill −1.268 corrupts logistic fitting)
2. Train XGBoost on `X_features` (WoE-encoded store, 68 cols) — logistic-friendly discrete boundaries
3. Train CatBoost on `X_tree_dfs` (DFS store, ~323 cols) — high-order cross-table aggregates
4. Implement rank-based ensemble: convert scores to percentile ranks → average (collinearity-robust)
5. Implement 2-layer MLP meta-learner (64→32→1, sklearn MLPClassifier) on OOF stack
6. Full ablation across all combo × strategy permutations; update `model_benchmark.csv`; persist best ensemble

**Done condition:** Ensemble OOT Gini ≥ 0.600; `ensemble_v2_ablation.csv` complete; `model_benchmark.csv` updated; best ensemble saved as `models/ensemble_v2_calibrated.pkl` if gate passes

**Status:** 🔲 Not started — depends on Phase 04.2.9

---

## Phase 04.3 — SHAP Explainability and Fairness

**Goal:** Implement `src/explain.py` with global + local SHAP plots and regulatory fairness metrics

**Pre-condition:** Regenerate `models/lightgbm_raw_calibrated.pkl` from `lgb_compliant_eval.json` best_params before starting (current pkl corrupted by test run)

**Plans:**
1. Implement `compute_shap_values(model, X)` using `shap.TreeExplainer`
2. Implement `plot_shap_summary(shap_values, X, save_path)` — global beeswarm + bar
3. Implement `plot_shap_local(shap_values, X, idx, save_path)` — waterfall + force plots
4. Implement `compute_fairness_metrics(model, X, y, sensitive_cols)` — demographic parity + equalised odds
5. Structure SHAP output dict for adverse action notices (top-5 negative factors per applicant)
6. Add TDD tests for all explainability functions

**Requirements:** EXPLAIN-01 through EXPLAIN-04, TEST-04
**Done condition:** All 4 EXPLAIN requirements satisfied, figures saved to `reports/figures/`, tests passing

---

## Phase 05.1 — FastAPI Production Endpoint

**Goal:** Production-ready `/predict` endpoint with authentication and SHAP output

**Plans:**
1. Replace stub in `app/api.py` — POST `/predict` with Pydantic request/response models
2. Add API key authentication middleware
3. Load best calibrated model on startup; serve PD + SHAP top-5 negative factors
4. Add `/health` with model version + uptime
5. Integration tests for API endpoints

**Done condition:** `uvicorn app.api:app` starts, `/predict` returns calibrated PD + SHAP factors, API key required

---

## Phase 05.2 — Streamlit Dashboard

**Goal:** Interactive Streamlit app for applicant risk scoring with SHAP waterfall visualization deployed on Streamlit community free Online

**Plans:**
1. Replace placeholder in `app/streamlit_app.py` — applicant input form (key features as sliders/dropdowns)
2. Call FastAPI `/predict` or load model directly; display PD with risk tier (Low/Medium/High)
3. Render SHAP waterfall chart for the applicant
4. Add feature contribution table (top 10 positive/negative factors)
5. Manual E2E test: applicant entry → PD output → SHAP waterfall visible

**Done condition:** `streamlit run app/streamlit_app.py` starts, full applicant scoring flow works end-to-end

---

## Phase 06 — LaTeX Report

**Goal:** Complete LaTeX report: methodology, model comparison, fairness analysis, business interpretation

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
Phase 01 → Phase 02 → Phase 04.2.1 → 04.2.2 → 04.2.3 → 04.2.3.1 → 04.2.3.2
                                                                          ↓
                                              04.2.4 → 04.2.4.1 → 04.2.5 → 04.2.5.1 → 04.2.6
                                                                                            ↓
                                                                           04.2.7 (contingency — executed; gate fail)
                                                                                            ↓
                                                                           04.2.9 (feature expansion)
                                                                                            ↓
                                                                           04.2.10 (ensemble diversity)
                                                                                            ↓
                                              04.2.8 (model.py refactor, parallel) ──── 04.3 → 05.1 → 05.2 → 06
```

---

## Progress Summary

| Phase | Status | Key Result |
|-------|--------|------------|
| Phase 01 — Infrastructure | ✅ Complete | Test isolation + path safety |
| Phase 02 — Fix Recurring Infrastructure | ✅ Complete | credit_engine removed; paths anchored |
| Phase 04.2.1 — Fix raw feature store | ✅ Complete | `X_tree_raw.parquet` (307K×211) |
| Phase 04.2.2 — DFS augmentation | ✅ Complete | 9 TDD tests; DFS pipeline validated |
| Phase 04.2.3 — XGBoost HPO | ⚠️ Invalid (leaky) | Gini=0.9592; path-based API valid |
| Phase 04.2.3.1 — SK_DPD leakage removal | ✅ Complete | 15 SK_DPD cols removed |
| Phase 04.2.3.2 — Feature engineering + XGB re-run | ⚠️ Gate fail | OOT Gini 0.5666; KS ✓; Brier ✓ |
| Phase 04.2.3.3 — XGB store selection | ✅ Superseded | Raw+eng selected; DFS noise (+0.0001) |
| Phase 04.2.4 — LightGBM HPO | ❌ Invalidated | Basel non-compliant → 04.2.4.1 |
| Phase 04.2.4.1 — LGB Compliant Re-run | ✅ Complete | OOT Gini=0.5746, KS=0.4302 ✓ |
| Phase 04.2.5 — CatBoost HPO | ❌ Invalidated | Basel non-compliant → 04.2.5.1 |
| Phase 04.2.5.1 — CatBoost Compliant Re-run | ✅ Complete | OOT Gini=0.5699, KS=0.4259 ✓ |
| Phase 04.2.7 — Feature Engineering Enhancement | ✅ Gate fail | Wave 1 done; LGB 0.5746 < 0.5845 |
| Phase 04.2.6 — Ensemble + gate | ✅ Complete | gate=investigate; OOT 0.5749; LGB standalone primary |
| Phase 04.2.8 — model.py refactor | ✅ Complete | 5 siblings + facade; 174 tests pass; root cleanup done |
| Phase 04.2.9 — Feature Engineering Expansion | 🔲 Not started | Any base model OOT Gini ≥ 0.580 |
| Phase 04.2.10 — Ensemble Enhancement | 🔲 Not started | Ensemble OOT Gini ≥ 0.600 |
| Phase 04.3 — SHAP + fairness | 🔲 Not started | All EXPLAIN reqs |
| Phase 05.1 — FastAPI endpoint | 🔲 Not started | `/predict` live |
| Phase 05.2 — Streamlit dashboard | 🔲 Not started | E2E flow works |
| Phase 06 — LaTeX report | 🔲 Not started | PDF compiles |

*Roadmap updated: 2026-04-11*
