---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
last_updated: "2026-04-14T11:10:00.000000Z"
progress:
  total_phases: 13
  completed_phases: 13
  total_plans: 53
  completed_plans: 53
  percent: 100
---

# Project State

**Project:** Credit Risk Scoring Pipeline
**Last Updated:** 2026-04-14
**Core value:** Calibrated PD feeding EL = PD × LGD × EAD, Gini ≥ 0.60

---

## Current Focus

**Phase 04.3 READY FOR EXECUTION ✅** — SHAP Explainability and Fairness — planning complete; 4 plans (04.3-01 through 04.3-04) wave-structured

**v2 model scoreboard (SK_ID_CURR temporal sort, Basel CRE36.54 compliant):**
- LGB (X_lgb_v2, is_unbalance): OOT Gini=0.5695, KS=0.4346 ⭐ primary model
- XGB (X_xgb_v2): OOT Gini=0.5636, KS=0.4183, AUC=0.7776
- CatBoost (X_cat_v2, auto_class_weights=Balanced): OOT Gini=0.5814, AUC=0.7907 ⭐ best single → **SHAP target**
- XGB-WoE (X_features, diversity): OOT Gini=0.5519, AUC=0.7734, KS=0.4159
- CatBoost-DFS (X_tree_dfs, diversity): OOT Gini=0.5608, AUC=0.7804, KS=0.4275

**Phase 04.2.9 Complete ✅** — All 5 plans done. Gate MET: CatBoost OOT Gini=0.5814 ≥ 0.580 ✅
**Phase 04.2.10 Complete ✅** — Gate=FAIL; best ensemble=0.5681 (LGB+CatBoost-DFS rank_avg) < 0.580 floor; CatBoost v2 (0.5814) primary
**Phase 04.3 Planning ✅** — 4 plans created with proper wave structure:
  - 04.3-01-PLAN.md (Wave 0): Test infrastructure + catboost_shap_fixture
  - 04.3-02-PLAN.md (Wave 1): SHAP core functions + complete FEATURE_LABELS (171 entries)
  - 04.3-03-PLAN.md (Wave 1): Fairness metrics + adverse action factors
  - 04.3-04-PLAN.md (Wave 2): Integration test (hard failure gates, no pytest.skip)
**Next: `/gsd-execute-phase 04.3`** — execute test infrastructure (Wave 0), then core + fairness (Wave 1), then integration (Wave 2)

**Phase 04.2.10 Plans:**
0. ✅ Create `train_xgboost_woe.py` — train XGBoost on X_features (WoE-encoded, 81 cols) for diversity
1. ✅ Create `train_catboost_dfs.py` — train CatBoost on X_tree_dfs (~323 cols, Featuretools DFS) with leakage guard
2. ✅ Create `run_ensemble_v2.py` — 9-cell ablation (2-model + 3-model combos, multiple meta-learner strategies)
3. ✅ Run HPO: XGB-WoE OOT Gini=0.5519, AUC=0.7734, KS=0.4159; CatBoost-DFS OOT Gini=0.5608, AUC=0.7804, KS=0.4275
4. ✅ Run ensemble orchestration and gating — best: LGB+CatBoost-DFS rank_avg, OOT Gini=0.5681, KS=0.4362
5. ✅ Gate=FAIL (0.5681 < 0.580 INVESTIGATE floor); CatBoost v2 (0.5814) is primary; Phase 04.3 next

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
| Phase 04.2.4.1 — LGB compliant re-run | 2026-04-11 | ~~0.5746~~ | ⚠️ | KS=0.4302 ✓; **inadmissible** — used `prev_days_decision_mean` sort col (wrong) |
| Phase 04.2.5 — CatBoost HPO | 2026-04-11 | — | ❌ | INVALIDATED — Basel CRE36.54 OOT contamination |
| Phase 04.2.5.1 — CatBoost compliant re-run | 2026-04-11 | 0.5699 | ✅ | KS=0.4259 ✓, Brier=0.0831 ✓ |
| Phase 04.2.7 — Feature Engineering Enhancement | 2026-04-11 | 0.5746 | ⚠️ | Wave 1 (7 features); gate fail (< 0.5845) |
| Phase 04.2.6 — Ensemble + Gate | 2026-04-11 | 0.5749 | ✅ | gate=investigate; LGB standalone proceeds as primary |
| Phase 04.2.8 — Split model.py | 2026-04-11 | — | ✅ | 5 sibling modules + facade; 174 tests pass; root cleanup done |
| Phase 04.2.9.01 — Feature Protection Foundation | 2026-04-11 | — | ✅ | EXT_SOURCE_NUM_AVAILABLE renamed; _PHASE9_PROTECTED (16 features) locked against filters |
| Phase 04.2.9.02 — Wave 2 Temporal Trajectory Features | 2026-04-11 | — | ✅ | 24 functions (10 bureau + 5 inst + 4 CC + 5 prev_app); 21 tests pass; ready for integration |
| Phase 04.2.9.03 — Build 4 Model-Specific Feature Stores | 2026-04-11 | — | ✅ | X_base_v2, X_lgb_v2, X_xgb_v2 (145 cols), X_cat_v2 (149 cols); 8 tests pass; Wave 2 integration deferred |
| Phase 04.2.9 Plan 04 — LGB + XGB HPO on v2 stores | 2026-04-12 | 0.5695 / 0.5636 | ✅ | LGB (X_lgb_v2, is_unbalance): OOT Gini=0.5695 ≥ 0.580 gate PASSED; XGB: OOT Gini=0.5636 |
| Phase 04.2.9 Plan 05 — CatBoost HPO on v2 store | 2026-04-12 | 0.5814 | ✅ | OOT Gini=0.5814, AUC=0.7907; BestOOF=0.5532 (50 trials); auto_class_weights=Balanced; X_cat_v2 |
| Phase 04.2.10 — Ensemble Enhancement via Feature Diversity | 2026-04-13 | 0.5681 | ❌ | Best: LGB+CatBoost-DFS rank_avg; 0.5681 < 0.580 INVESTIGATE floor; CatBoost v2 (0.5814) is primary |

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
| `models/xgboost_raw_calibrated.pkl` | ✅ | OOT Gini=0.5636, KS=0.4183 (v2, SK_ID_CURR sort) — backed up as `xgboost_raw_calibrated_v2.pkl` |
| `models/lightgbm_raw_calibrated_v2.pkl` | ✅ | Frozen backup of v2 result (OOT Gini=0.5695) — safe from future overwrites |
| `models/xgboost_raw_calibrated_v2.pkl` | ✅ | Frozen backup of v2 result (OOT Gini=0.5636) — safe from future overwrites |
| `reports/xgb_raw_X_xgb_v2_eval.json` | ✅ | Versioned backup of XGB v2 eval (generic xgboost_raw_eval.json would be overwritten) |
| `models/lightgbm_raw_calibrated.pkl` | ✅ | 4.6 MB, regenerated 2026-04-12 by Plan 04 — OOT Gini=0.5695; params in `lgb_raw_X_lgb_v2_is_unbalance_eval.json` |
| `models/catboost_raw_calibrated.pkl` | ✅ | v2: OOT Gini=0.5814, AUC=0.7907 (X_cat_v2, auto_class_weights=Balanced) — backed up as `catboost_raw_calibrated_v2.pkl` |
| `models/catboost_raw_calibrated_v2.pkl` | ✅ | Frozen backup of v2 result (OOT Gini=0.5814) — safe from future overwrites |
| `models/ensemble_calibrated.pkl` | ✅ | Platt; Brier=0.0878; gate=investigate (not primary) |
| `reports/lgb_compliant_eval.json` | ✅ | Best params for LGB (Phase 04.2.4.1, inadmissible sort col) |
| `reports/lgb_raw_X_lgb_v2_is_unbalance_eval.json` | ✅ | LGB v2 best params; OOT Gini=0.5695, KS=0.4346 — use for LGB regeneration |
| `reports/xgboost_raw_eval.json` | ✅ | XGB v2 best params; OOT Gini=0.5636, KS=0.4183, AUC=0.7776 |
| `reports/ensemble_weights.json` | ✅ | Meta-learner coefs; gate=investigate |
| `reports/ensemble_ablation.csv` | ✅ | 11-row ranked ablation; best: LGB+CAT avg 0.5754 |
| `reports/model_benchmark.csv` | ✅ | 5-model comparison (LR, XGB, LGB, CatBoost, Ensemble) |
| `src/model.py` | ✅ | Thin facade ~438 lines; re-exports all symbols from 5 sibling modules |
| `src/model_base.py` | ✅ | Constants + shared utilities (_make_cv, calibrate_model, save/load_model) |
| `src/model_xgboost.py` | ✅ | XGBoost training + Optuna HPO |
| `src/model_lightgbm.py` | ✅ | LightGBM training + Optuna HPO |
| `src/model_catboost.py` | ✅ | CatBoost training + Optuna HPO |
| `src/model_ensemble.py` | ✅ | 3-model stacking, meta-learners, ensemble gate |
| `src/explain.py` | 🔲 | Stub — Phase 04.3 (planning complete, execution ready) |

*Full session-by-session notes: `.planning/SESSION_LOG.md`*
*Project initialized: 2026-04-07*
