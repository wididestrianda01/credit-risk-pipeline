---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
last_updated: "2026-04-11T21:00:00.000000Z"
progress:
  total_phases: 13
  completed_phases: 12
  total_plans: 48
  completed_plans: 48
  percent: 99
---

# Project State

**Project:** Credit Risk Scoring Pipeline
**Last Updated:** 2026-04-11
**Core value:** Calibrated PD feeding EL = PD × LGD × EAD, Gini ≥ 0.60

---

## Current Focus

**Active phase:** Phase 04.2.9 Plan 04 — LGB + XGB HPO on Differentiated Feature Stores

- Phase 04.2.8 complete: model.py split into 5 focused sibling modules; 174 tests pass; root cleanup done
- Primary model: `lightgbm_raw_calibrated.pkl` — must regenerate before SHAP (file is CORRUPTED, 6KB)
- Regeneration params in `reports/lgb_compliant_eval.json`

**Just completed:** Phase 04.2.9 Plan 03 — Build 4 Model-Specific Feature Stores (X_base_v2, X_lgb_v2, X_xgb_v2, X_cat_v2; 8 integration tests pass)

---

## Completed Phases

| Phase | Date | OOT Gini | Gate | Notes |
|-------|------|----------|------|-------|
| Phase 01 — Infrastructure | 2026-04-07 | — | ✅ | Absolute paths, test isolation, import alias |
| Phase 02 — Recurring Infrastructure | 2026-04-10 | — | ✅ | credit_engine removed, dataset/→data/, model.py anchored |
| Phase 04.2.1 — Fix tree feature store | 2026-04-07 | — | ✅ | `X_tree_raw.parquet` (307K×211, 0 WoE) |
| Phase 04.2.2 — DFS augmentation | 2026-04-07 | — | ✅ | 9 TDD tests; Woodwork entity-set fixed |
| Phase 04.2.3 — XGBoost HPO | 2026-04-08 | ~~0.9592~~ | ⚠️ | INVALID — SK_DPD leakage |
| Phase 04.2.3.1 — SK_DPD removal + OOF/OOT | 2026-04-10 | — | ✅ | 15 leaky cols removed; store rebuilt |
| Phase 04.2.3.2 — Feature engineering + XGB re-run | 2026-04-10 | 0.5666 | ⚠️ | Gate fail (target 0.60); KS 0.4089 ✓; XGB plateau |
| Phase 04.2.3.3 — XGB store selection | 2026-04-10 | — | ✅ | Superseded — raw+eng wins; DFS +0.0001 noise |
| Phase 04.2.4 — LGB HPO | 2026-04-11 | — | ❌ | INVALIDATED — Basel CRE36.54 OOT contamination |
| Phase 04.2.4.1 — LGB compliant re-run | 2026-04-11 | 0.5746 | ✅ | KS=0.4302 ✓; regulatory metric clean |
| Phase 04.2.5 — CatBoost HPO | 2026-04-11 | — | ❌ | INVALIDATED — Basel CRE36.54 OOT contamination |
| Phase 04.2.5.1 — CatBoost compliant re-run | 2026-04-11 | 0.5699 | ✅ | KS=0.4259 ✓, Brier=0.0831 ✓ |
| Phase 04.2.7 — Feature Engineering Enhancement | 2026-04-11 | 0.5746 | ⚠️ | Wave 1 (7 features); gate fail (< 0.5845) |
| Phase 04.2.6 — Ensemble + Gate | 2026-04-11 | 0.5749 | ✅ | gate=investigate; LGB standalone proceeds as primary |
| Phase 04.2.8 — Split model.py | 2026-04-11 | — | ✅ | 5 sibling modules + facade; 174 tests pass; root cleanup done |
| Phase 04.2.9.01 — Feature Protection Foundation | 2026-04-11 | — | ✅ | EXT_SOURCE_NUM_AVAILABLE renamed; _PHASE9_PROTECTED (16 features) locked against filters |
| Phase 04.2.9.02 — Wave 2 Temporal Trajectory Features | 2026-04-11 | — | ✅ | 24 functions (10 bureau + 5 inst + 4 CC + 5 prev_app); 21 tests pass; ready for integration |
| Phase 04.2.9.03 — Build 4 Model-Specific Feature Stores | 2026-04-11 | — | ✅ | X_base_v2, X_lgb_v2, X_xgb_v2 (145 cols), X_cat_v2 (149 cols); 8 tests pass; Wave 2 integration deferred |

**Ensemble post-mortem summary (Phase 04.2.6):** All three models trained on `X_tree_raw` produced OOF correlations ≥ 0.95 — no orthogonal signal for meta-learner. Meta-learner coefs: LGB=+3.08, XGB=−1.45, CAT=+1.53. Best combo LGB+CAT avg Gini=0.5754 (+0.0008 vs LGB standalone — below 0.005 threshold). Decision: LGB standalone (OOT Gini=0.5746) is primary model.

---

## Key Files

| File | Status | Notes |
|------|--------|-------|
| `data/processed/X_train.parquet` | ✅ | 307511×195 — raw joined features |
| `data/processed/X_features.parquet` | ✅ | 307511×68 — WoE store for LR only |
| `data/processed/X_tree_raw.parquet` | ✅ | 307511×144 — Wave 1 delinquency features; TARGET embedded |
| `data/processed/X_tree_raw_v1.parquet` | ✅ | Backup of X_tree_raw (Phase 04.2.9.03) |
| `data/processed/X_base_v2.parquet` | ✅ | 307511×145 — X_tree_raw + EXT_SOURCE_NUM_AVAILABLE |
| `data/processed/X_lgb_v2.parquet` | ✅ | 307511×145 — identical to X_base_v2 (raw continuous for LGB) |
| `data/processed/X_xgb_v2.parquet` | ✅ | 307511×145 — X_base_v2 (pragmatic baseline for XGB) |
| `data/processed/X_cat_v2.parquet` | ✅ | 307511×149 — X_base_v2 + 4 categorical columns (CatBoost) |
| `data/processed/X_tree_dfs.parquet` | ✅ | 307511×290 — raw+DFS; raw+eng wins for LGB/XGB |
| `models/optuna_studies.db` | ✅ | Continue existing studies — never restart |
| `models/logistic_baseline.pkl` | ✅ | Gini=0.489, KS=0.361 |
| `models/xgboost_raw_calibrated.pkl` | ✅ | OOT Gini=0.5666, KS=0.4089 |
| `models/lightgbm_raw_calibrated.pkl` | ⚠️ CORRUPTED | 6 KB, n_estimators=2, LR=0.015 — must regenerate before Phase 04.3 |
| `models/catboost_raw_calibrated.pkl` | ✅ | OOT Gini=0.5699, KS=0.4259 |
| `models/ensemble_calibrated.pkl` | ✅ | Platt; Brier=0.0878; gate=investigate (not primary) |
| `reports/lgb_compliant_eval.json` | ✅ | Best params for LGB regeneration |
| `reports/ensemble_weights.json` | ✅ | Meta-learner coefs; gate=investigate |
| `reports/ensemble_ablation.csv` | ✅ | 11-row ranked ablation; best: LGB+CAT avg 0.5754 |
| `reports/model_benchmark.csv` | ✅ | 5-model comparison (LR, XGB, LGB, CatBoost, Ensemble) |
| `src/model.py` | ✅ | Thin facade ~438 lines; re-exports all symbols from 5 sibling modules |
| `src/model_base.py` | ✅ | Constants + shared utilities (_make_cv, calibrate_model, save/load_model) |
| `src/model_xgboost.py` | ✅ | XGBoost training + Optuna HPO |
| `src/model_lightgbm.py` | ✅ | LightGBM training + Optuna HPO |
| `src/model_catboost.py` | ✅ | CatBoost training + Optuna HPO |
| `src/model_ensemble.py` | ✅ | 3-model stacking, meta-learners, ensemble gate |
| `src/explain.py` | 🔲 | Stub — Phase 04.3 |

*Full session-by-session notes: `.planning/SESSION_LOG.md`*
*Project initialized: 2026-04-07*
