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
| Phase 04.2.4.1 — LGB compliant re-run | ⚠️ inadmissible | ~~0.5746~~ | KS=0.4302; used `prev_days_decision_mean` sort col — wrong OOT definition |
| Phase 04.2.5 — CatBoost HPO | ❌ invalidated | — | Basel CRE36.54 OOT contamination; superseded by 04.2.5.1 |
| Phase 04.2.5.1 — CatBoost compliant re-run | ✅ | 0.5699 | KS=0.4259 ✓, Brier=0.0831 ✓ |
| Phase 04.2.7 — Feature Engineering Enhancement | ✅ gate fail | 0.5746 | Wave 1 (7 delinquency features); gate < 0.5845 — no net lift |
| Phase 04.2.6 — Ensemble + Gate | ✅ investigate | 0.5749 | gate=investigate; LGB standalone (0.5746) proceeds as primary |

> **Basel CRE36.54 mandatory workflow (all model training):** Sort by `SK_ID_CURR` (monotonically increasing application intake surrogate — not a derived aggregate) → carve OOT (most-recent 20%, frozen) → Optuna HPO on 80% with OOF CV Gini as objective → retrain on full 80% → evaluate on frozen OOT. Any split inside the Optuna objective closure contaminates results.

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

**Status:** ✅ Complete — gate MET; CatBoost v2 OOT Gini=0.5814 (best); LGB=0.5695, XGB=0.5636. All 5 plans done 2026-04-12.

---

## Phase 04.2.10 — Ensemble Enhancement via Feature Diversity

**Goal:** Re-ensemble using *different* feature stores per base model to create genuine prediction diversity; implement rank-based ensemble and MLP meta-learner; target ensemble OOT Gini ≥ 0.600.

**Why this exists:** Phase 04.2.6 gate=investigate root cause: all three base models trained on identical `X_tree_raw` → OOF correlations ≥ 0.95 → meta-learner had no orthogonal signal. Fix: LGB stays on `X_tree_raw`, XGBoost moves to `X_features` (WoE), CatBoost moves to `X_tree_dfs` (DFS). This creates structurally distinct OOF residuals.

**Gate:** Ensemble OOT Gini ≥ 0.600

**Plans:**
0. Use @ensemble_vs_regulation.md to setup how should ensemble to be used. Explain the ensemble justification. If necessary, conduct test to support justification.
1. Pre-calibrate CatBoost OOF predictions before meta-learner training (uncalibrated OOF BrierSkill −1.268 corrupts logistic fitting)
2. Train XGBoost on `X_features` (WoE-encoded store, 68 cols) — logistic-friendly discrete boundaries
3. Train CatBoost on `X_tree_dfs` (DFS store, ~323 cols) — high-order cross-table aggregates
4. Implement rank-based ensemble: convert scores to percentile ranks → average (collinearity-robust)
5. Implement 2-layer MLP meta-learner (64→32→1, sklearn MLPClassifier) on OOF stack
6. Full ablation across all combo × strategy permutations; update `model_benchmark.csv`; persist best ensemble

**Done condition:** Ensemble OOT Gini ≥ 0.600; `ensemble_v2_ablation.csv` complete; `model_benchmark.csv` updated; best ensemble saved as `models/ensemble_v2_calibrated.pkl` if gate passes

**Status:** ✅ Complete — gate=FAIL (best ensemble OOT Gini=0.5681 < 0.580 INVESTIGATE floor)
- ✅ `scripts/train_xgboost_woe.py` — XGB-WoE OOT Gini=0.5519, AUC=0.7734, KS=0.4159
- ✅ `scripts/train_catboost_dfs.py` — CatBoost-DFS OOT Gini=0.5608, AUC=0.7804, KS=0.4275
- ✅ `scripts/run_ensemble_v2.py` — 9-cell ablation; best: LGB+CatBoost-DFS rank_avg OOT Gini=0.5681, KS=0.4362
- CatBoost v2 (OOT Gini=0.5814) remains primary model for SHAP and production serving

---

## Phase 04.3 — SHAP Explainability and Fairness ✅ Complete

**Goal:** Implement `src/explain.py` with global + local SHAP plots and regulatory fairness metrics

**Result (2026-04-14):** All 4 plans executed + inline disparate impact ratio fix committed.
- `src/explain.py`: 6 public functions — `compute_shap_values`, `plot_shap_summary`, `plot_shap_local`, `compute_fairness_metrics`, `get_adverse_action_factors`, `compute_shap_stability`
- `FEATURE_LABELS`: 171-entry module-level dict — GDPR Art. 22 compliant (no raw column names in adverse action output)
- SHAP stability: Spearman rank correlation = 0.9995 (Basel CRE36.54 threshold ≥ 0.90 ✓)
- Fairness: per-attribute disparate impact ratios; Gender DIR ≈ 0.955 (✓ EU AI Act ≥ 0.80); Age DIR ≈ 0.346 (✗ flagged — Young vs Senior gap)
- `reports/figures/`: shap_beeswarm.png, shap_bar.png, shap_waterfall_0.png, shap_force_0.html
- `reports/fairness_metrics.csv`: group_name + demographic_parity + tpr + fpr + 3 `_disparate_impact` cols
- `tests/test_explain.py`: 11 tests (10 fast + 1 slow integration), all passing

**Requirements:** EXPLAIN-01 through EXPLAIN-04, TEST-04 — all satisfied

---

## Phase 04.4 — Fairness-Compliant Feature Stores and Model Retraining ❌ Descoped

**Goal:** Remove EU AI Act Art. 6 / ECHR-prohibited features from all v2 feature stores and retrain XGBoost, LightGBM, and CatBoost to achieve Age DIR ≥ 0.80

**Regulated features to remove:** `AGE_YEARS`, `EMPLOYED_TO_AGE_RATIO`, `CNT_CHILDREN`, `CNT_FAM_MEMBERS`
- `AGE_YEARS` — direct age encoding (EU AI Act Art. 6 protected characteristic)
- `EMPLOYED_TO_AGE_RATIO` — derived from birth year; encodes age directly
- `CNT_CHILDREN` / `CNT_FAM_MEMBERS` — family/parental status (Equal Treatment Directive, ECHR Art. 14)

**Proxy retention strategy:**
- Age signal → retained via `YEARS_EMPLOYED`, `bureau_cnt`, `prev_cnt`, credit history aggregates (legitimate risk signals uncorrelated with birth year)
- Family size signal → absorbed by `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO` (income-to-obligation ratios capture financial load without discriminating on family composition)

**Plans created (2026-04-14):**
- [x] 04.4-01-v3-feature-stores-PLAN.md — TDD: create v3 stores (163/163/167 cols) by dropping regulated cols
- [x] 04.4-02-xgboost-v3-retrain-PLAN.md — Retrain XGBoost on X_xgb_v3; OOT Gini=0.5075
- [x] 04.4-03-lightgbm-v3-retrain-PLAN.md — Retrain LightGBM on X_lgb_v3; OOT Gini=0.5610 (**Phase 05 primary**)
- [x] 04.4-04-catboost-v3-retrain-PLAN.md — Retrain CatBoost on X_cat_v3; OOT Gini=0.5520
- [x] 04.4-05-fairness-eval-and-gate-PLAN.md — Dual-load fairness eval; INVESTIGATE: Age DIR improved (0.449 vs 0.346) but proxy features prevent ≥ 0.80; XGB Gender DIR ✓

**Wave structure:**
- Wave 1: Plan 01 (v3 feature stores)
- Wave 2: Plans 02, 03, 04 (XGB, LGB, CatBoost training — parallel)
- Wave 3: Plan 05 (fairness evaluation + gate)

**Done condition:** All three calibrated models retrained on fairness-compliant stores; Age DIR ≥ 0.80; Gender DIR ≥ 0.80 (no regression); Gini ≥ 0.55 (floor); CLAUDE.md updated with new model paths and metrics

**Requirements:** FAIR-01 through FAIR-06

---

## Phase 04.5 — Project Explanation Notebooks ✅ COMPLETE

**Goal:** Create and update all four project Jupyter notebooks — notebooks 01 and 02 reviewed/augmented where gaps exist, notebooks 03 and 04 built from scratch — to form a complete analytical narrative of the end-to-end credit risk pipeline.

**Completion Summary (2026-04-14):**

- ✅ **04.5-01:** Notebook 01 (EDA and Data Quality) — gap-review completed; temporal sort and feature evolution narrative integrated
- ✅ **04.5-02:** Notebook 02 (Feature Engineering) — tree feature stores (X_tree_raw, v2 stores), temporal trajectory features, feature protection added
- ✅ **04.5-03:** Notebook 03 (Modeling and Evaluation) — 15-cell complete notebook; LR baseline (Gini=0.489), v2 models comparison (XGB/LGB/CatBoost), Basel CRE36.54 temporal validation methodology, CatBoost v2 deployment decision (Gini=0.5814), live inference demo, model card, fairness metrics (Gender DIR=0.955 ✓)
- ✅ **04.5-04:** Notebook 04 (Explainability and Fairness) — 15-cell complete notebook; SHAP global (beeswarm, bar) and local (waterfall) explainability, fairness metrics and disparate impact analysis, GDPR Art. 22 adverse action notice template, EU AI Act Art. 6 high-risk AI compliance, production readiness summary

**All four notebooks:**
- Run end-to-end without errors
- Contain ≥15 cells each (substantive content)
- Follow narrative style: markdown context → code → "What we see" interpretation
- Use artifact-first design: pre-computed figures from Phase 04.3, fairness CSV from Phase 04.3
- Include regulatory framing (GDPR Art. 22, EU AI Act Art. 6, Basel III CRE36.54)
- No TODO stubs, no `plt.show()` calls

**Requirements satisfied:** NB-01, NB-02, NB-03, NB-04

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
                                              04.2.8 (model.py refactor, parallel) ──── 04.3 → 04.4 → 05.1 → 05.2 → 06
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
| Phase 04.2.4.1 — LGB Compliant Re-run | ⚠️ Inadmissible | ~~OOT Gini=0.5746~~ wrong sort col; 0.5695 (v2) is valid baseline |
| Phase 04.2.5 — CatBoost HPO | ❌ Invalidated | Basel non-compliant → 04.2.5.1 |
| Phase 04.2.5.1 — CatBoost Compliant Re-run | ✅ Complete | OOT Gini=0.5699, KS=0.4259 ✓ |
| Phase 04.2.7 — Feature Engineering Enhancement | ✅ Gate fail | Wave 1 done; LGB 0.5746 < 0.5845 |
| Phase 04.2.6 — Ensemble + gate | ✅ Complete | gate=investigate; OOT 0.5749; LGB standalone primary |
| Phase 04.2.8 — model.py refactor | ✅ Complete | 5 siblings + facade; 174 tests pass; root cleanup done |
| Phase 04.2.9 — Feature Engineering Expansion | ✅ Complete | CatBoost v2 OOT Gini=0.5814 ⭐; LGB=0.5695, XGB=0.5636; gate MET |
| Phase 04.2.10 — Ensemble Enhancement | ✅ Complete (gate=FAIL) | Best: LGB+CatBoost-DFS rank_avg, OOT Gini=0.5681; CatBoost v2 (0.5814) is primary |
| Phase 04.3 — SHAP + fairness | ✅ Complete | 11/11 tests; 4 figures; fairness CSV with DIR; SHAP stability=0.9995; GDPR/Basel/EU-AIAct |
| Phase 04.4 — Fairness-Compliant Retraining | ❌ Descoped | Policy revised 2026-04-14: Gender DIR gate only; v2 CatBoost (Gini=0.5814, Gender DIR=0.955 ✓) is production model; v3 stores/models archived |
| Phase 04.5 — Project Explanation Notebooks | ✅ Complete | All 4 plans finished; 4×15-cell notebooks; artifact-first; GDPR/fairness/Basel compliant |
| Phase 05.1 — FastAPI endpoint | 🔲 Not started | `/predict` returns PD + adverse action factors |
| Phase 05.2 — Streamlit dashboard | 🔲 Not started | E2E scoring flow, SHAP waterfall |
| Phase 06 — LaTeX report | 🔲 Not started | PDF compiles, >15 pages |

*Roadmap updated: 2026-04-14*
