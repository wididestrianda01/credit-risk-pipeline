---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: unknown
last_updated: "2026-04-08T03:55:58.944Z"
progress:
  total_phases: 3
  completed_phases: 2
  total_plans: 12
  completed_plans: 11
  percent: 92
---

# Project State

**Project:** Credit Risk Scoring Pipeline
**Last Updated:** 2026-04-07

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-04-07)

**Core value:** Calibrated PD feeding EL = PD × LGD × EAD, Gini ≥ 0.70
**Current focus:** Phase 04.2.3 — xgboost-hpo-on-raw-features

---

## Active Phase

**Phase 04.2.3** — XGBoost HPO on Raw Features

- Status: Discussion complete — proceeding to plan-phase
- Last command: `/gsd-discuss-phase 04.2.3 --chains` (completed 2026-04-08)
- Branch: `gsd/phase-04.2.2-dfs-auto-feature-augmentation` (current; new branch for 04.2.3 to be created)

---

## Completed Phases

### Phase 1 — Data Loading and EDA

- Status: ✅ Complete
- Result: `src/data_loader.py` — 7-table join, `X_train.parquet` (307511×195)
- Branch: `phase/1-data-loading` (merged to main)

### Phase 2 — WoE Feature Engineering

- Status: ✅ Complete
- Result: `src/features.py` — 40–68 WoE features, `X_features.parquet` (307511×68)
- Branch: `phase/2-feature-engineering` (merged to main)

### Phase 3 — LR Baseline + Evaluation Utilities

- Status: ✅ Complete
- Result: LR baseline Gini=0.489, KS=0.361; `src/utils.py` (Gini, KS, Brier, BrierSkill); 188 tests
- Branch: `phase/3-model-training`

---

## Session Notes

### 2026-04-07 — Root cause: tree models used wrong dataset

- **Root cause discovered:** All prior XGB/LGB/CatBoost HPO was run on WoE-encoded 63-feature stores. XGB Gini=0.5296 on wrong store (delta=−0.0271 vs 0.5567 baseline, which itself was on wrong store). Results are invalid.
- **Two bugs identified:**
  1. `scripts/prepare_feature_pipelines.py` does an 80/20 pre-split before feature extraction → 246K rows instead of 307K
  2. WoE encoding applied to tree model feature stores — destroys gradient signal
- **Correct path:** `X_train.parquet` (307511×195) already exists → `build_tree_feature_store()` (raw features, variance filter only) → `X_tree_raw.parquet` → DFS augmentation → HPO
- **DFS status:** `X_featuretools.parquet` (179073×0) — entity-set build failed silently, 0 columns generated
- **Optuna:** `models/optuna_studies.db` — always continue existing studies, never restart
- **Key invariants:** Temporal CV `_CV_EMBARGO_FRAC=0.02`, raw features only for trees, Platt calibration required
- **GSD planning initialized:** PROJECT.md, REQUIREMENTS.md, ROADMAP.md, STATE.md created
- **Codebase mapped:** 7 docs in `.planning/codebase/` (1648 lines total)

### 2026-04-07 — Phase 04.2.1 discuss-phase completed

- **Gray area 1 (feature selection for raw store):** Variance filter only (threshold=0.01); no IV filter, no WoE — IV filtering via WoE defeats purpose for trees
- **Gray area 2 (DFS strategy):** Fix entity-set first; run DFS; if >0 features after variance filter, merge with raw store. If DFS produces 0 features after 2 attempts, fall back to manual secondary aggregates.
- **Gray area 3 (model rebuild order):** XGBoost first (most HPO trials already exist in SQLite), then LightGBM, then CatBoost. All 3 must be rebuilt.
- **CONTEXT.md + DISCUSSION-LOG.md:** Recreated 2026-04-07 at `.planning/phases/04.2.1-fix-tree-feature-store/`
- Next command: `/gsd-plan-phase 04.2.1`

### 2026-04-07 — Phase 04.2.2-04 TDD tests complete

- **Task 1:** 3 Woodwork LogicalType fix unit tests added to `test_auto_features.py` — validates `_build_entity_set()` creates EntitySet, asserts `_DEFAULT_AGG_PRIMITIVES` includes `std` and `num_unique`, confirms ≥7 explicit `logical_types=` assignments in source code
- **Task 2:** 5 engineer_time_features tests added to `test_features.py` — synthetic bureau_tables fixture (10 applicants, 0–5 records each, varying DPD patterns), validates 3-month delinquency rate, months-since-last-DPD, credit-age-mean calculations, return type/shape/dtype
- **Task 3:** 1 DFS correlation deduplication regression test added — validates `deduplicate_dfs_features()` removes >0.90 correlated feature pairs
- **Task 4:** Pytest execution: all 9 new tests passing (5.63s), 0 failures, 100% success rate
- **Commit:** `3841b9f` — `test(04.2.2-04): add TDD tests for Woodwork fix, time features, cross-dedup regression`
- **SUMMARY.md:** Created at `.planning/phases/04.2.2-dfs-auto-feature-augmentation/04.2.2-04-SUMMARY.md`

---

## Key Files

| File | Status | Notes |
|------|--------|-------|
| `data/processed/X_train.parquet` | ✅ Exists | 307511×195 — foundation for raw tree store |
| `data/processed/X_features.parquet` | ✅ Exists | 307511×68 — WoE store for LR only |
| `data/processed/X_featuretools.parquet` | ⚠️ Broken | 179073×0 — entity-set failed, 0 DFS columns |
| `data/processed/X_tree_raw.parquet` | ✅ Exists | 307511×211 — raw tree store, Phase 04.2.1 complete |
| `data/processed/X_tree_dfs.parquet` | 🔲 Missing | Target of Phase 04.2.2 |
| `models/optuna_studies.db` | ✅ Exists | Continue existing studies, never restart |
| `models/logistic_baseline.pkl` | ✅ Exists | Gini=0.489, KS=0.361 |
| `models/xgboost_calibrated.pkl` | ⚠️ Invalid | Trained on WoE store — discard |
| `models/lightgbm_best.pkl` | ⚠️ Invalid | Trained on WoE store — discard |
| `src/features.py` | ✅ Valid | `build_tree_feature_store()` complete; backward-compat alias added |
| `src/auto_features.py` | ⚠️ Broken | Entity-set bug to diagnose |
| `src/model.py` | ✅ Valid | Needs raw-store path param on HPO functions |
| `src/explain.py` | 🔲 Stub | Phase 04.3 |

## Accumulated Context

### Roadmap Evolution

- Phase 04.2.1 complete: `build_tree_feature_store()` + `X_tree_raw.parquet` (307511×211) — 2026-04-07
- Phase 1 added: fix project-wide infrastructure issues (credit_engine alias, feature store path safety, test isolation) — 2026-04-07

---
*State initialized: 2026-04-07*
