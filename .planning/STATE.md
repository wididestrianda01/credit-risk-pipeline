---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
last_updated: "2026-04-10T19:00:00.000Z"
progress:
  total_phases: 8
  completed_phases: 8
  total_plans: 41
  completed_plans: 35
  percent: 85
---

# Project State

**Project:** Credit Risk Scoring Pipeline
**Last Updated:** 2026-04-10

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-04-07)

**Core value:** Calibrated PD feeding EL = PD × LGD × EAD, Gini ≥ 0.70
**Current focus:** Phase 04.2.5 — CatBoost HPO on raw+eng and raw+eng+DFS features

---

## Completed Plans

### Phase 04.2.3 — XGBoost HPO on Raw Features

- Status: ✅ Complete (2026-04-08)
- Result: train_xgboost_optuna() path-based API, 8 HPO improvements, Platt calibration, 188 tests migrated, 2 integration tests added
- Summary: `/home/wd/Working Folder/Development/credit-risk-pipeline/.planning/phases/04.2.3-xgboost-hpo-on-raw-features/04.2.3-01-SUMMARY.md`
- Artifacts: `models/xgboost_raw_best.pkl`, `models/xgboost_raw_calibrated.pkl`, `scripts/train_xgboost_raw.py`
- Branch: `gsd/phase-04.2.2-dfs-auto-feature-augmentation` (plan executed on this branch)

## Active Phase

**Phase 04.2.5** — CatBoost HPO on Raw+DFS Features

- Status: ✅ Wave 1 Complete (2026-04-10) | ✅ Wave 2 Complete (2026-04-10)
- Branch: `gsd/phase-04.2.4-lightgbm-hpo-on-raw-features`
- Completion: 2 of 3 waves complete (Wave 1: constants, Wave 2: function rewrite; Wave 3: HPO production runs pending)
- Summary: `/home/wd/Working Folder/Development/credit-risk-pipeline/.planning/phases/04.2.5-catboost-hpo-on-raw-features/04.2.5-02-SUMMARY.md`

## Recently Completed Phases

**Phase 04.2.5 Wave 2** — CatBoost Function Rewrite (COMPLETE — 2026-04-10)

- Status: ✅ Complete (Wave 2 only; Wave 1 constants + Wave 3 HPO runs separate)
- Branch: `gsd/phase-04.2.4-lightgbm-hpo-on-raw-features`
- **Completion summary:**
  - Task 1–2: train_catboost_optuna() signature rewrite from (X, y) to path-based API
  - Task 3–4: Temporal CV auto-detection + Optuna study persistence (TPESampler, HyperbandPruner)
  - Task 5: 2-stage refit (early stopping on 80/20 holdout, full X_train refit)
  - Task 6–8: Evaluation, Platt calibration, artifact saving, test migration
  - **HPO search space (7 parameters):** depth [5,10], learning_rate [0.01,0.2] log, l2_leaf_reg [0.1,30] log, **min_data_in_leaf [5,50]** NEW, bagging_temperature [0,1], random_strength [0,1], bootstrap_type="Bayesian"
  - **Test suite:** All 7 CatBoost tests passing (test_returns_5_tuple, test_metrics_keys, test_gini_above_threshold, test_best_params_within_search_space, test_model_artifact_saved, test_params_artifact_saved, test_no_stdout)
- Artifacts: `models/catboost_raw_calibrated.pkl`, `models/catboost_params.json`, `reports/catboost_raw_eval.json`
- Commit: `1ed98e7` (feat: rewrite train_catboost_optuna path-based API, 2-stage refit, expanded HPO)

**Phase 04.2.4** — LightGBM HPO on Raw Features (COMPLETE — 2026-04-10)

- Status: ⚠️ Complete (OOT gate not met) — 7/7 plans executed
- Branch: `gsd/phase-04.2.4-lightgbm-hpo-on-raw-features`
- **Feature-store ablation results (2026-04-10):** 50-trial LGB HPO, 2-store comparison (raw+eng vs raw+eng+DFS)
  - raw+eng (129 features): OOF Gini 0.5510 | OOT Gini **0.5795** | Gap −0.0286 → REJECT
  - raw+eng+DFS (290 features): OOF Gini 0.5436 | OOT Gini 0.5767 | Gap −0.0331 → REJECT
  - **Winner: raw+eng** — DFS hurts LGB (−0.0028 OOT Gini, wider gap); noise outweighs signal at 160 DFS features
  - OOT Gini target 0.60 not met; OOF plateau at ~0.551 from trial 25 → feature ceiling reached
  - **LGB raw+eng is new overall best single model:** OOT 0.5795 (+0.013 over XGBoost 0.5666)
  - **OOT positive rate drift:** Train 7.5% vs OOT 10.4% (+38%) — rank metrics valid; Platt recalibration at deployment prevalence (~8%) mandatory before EL = PD × LGD × EAD
- Artifacts: `reports/lgb_feature_store_selection.json`, `models/lightgbm_raw_calibrated.pkl`

**Phase 02** — Fix Recurring Infrastructure (COMPLETE — 2026-04-10)

- Status: ✅ Complete — 3/3 plans executed
- Plans: `02-01` (credit_engine alias removal), `02-02` (dataset/ → data/ docstrings), `02-03` (_PROJECT_ROOT paths in model.py)
- Branch: `gsd/phase-02-fix-recurring-infrastructure`
- Commits: `f30e931` (02-01), `f787231` (02-02), `c28eec4` (02-03)

**Phase 04.2.3.2** — Feature Engineering Completion + XGBoost Re-run (COMPLETE — 2026-04-10)

- Status: ✅ Complete — Plan 07 HPO v9 finished, OOT Gini gate not met but KS/Brier targets pass
- Plans: 7 of 7 executed
- **HPO v9 Results (2026-04-10):** 50-trial XGBoost on X_tree_dfs.parquet (study xgboost_raw_v9)
  - OOF Gini: 0.5140 | OOT Gini: 0.5666 | Hold Gini: 0.5343
  - KS: 0.4089 ✓ | Brier: 0.0635 ✓
  - OOT Gini target 0.60 not met (gap: -0.034) — XGBoost plateau reached
  - **Baseline correction:** Prior "0.8594 baseline" was pre-leakage (SK_DPD); true clean baseline ~0.547; HPO v9 is +0.019 improvement
  - DFS adds +0.0001 OOT Gini vs raw features (within noise) — no measurable lift
- **Artifacts:** `models/xgboost_raw_calibrated.pkl`, `reports/xgboost_raw_eval.json`

**Phase 04.2.3.3** — XGBoost Feature-Store Selection (SUPERSEDED — 2026-04-10)

- Status: ✅ Superseded — absorbed into Phase 04.2.3.2 + Phase 04.2.4 findings; no dedicated comparison run needed
- Finding: DFS adds only +0.0001 OOT Gini over raw+eng for XGBoost (within noise); Phase 04.2.4 confirmed DFS hurts LGB (−0.0028, wider gap). Raw+eng (129 features) is the confirmed winning store for all tree models.

**Phase 04.2.3.1** — SK_DPD Leakage Removal + OOF Gini (COMPLETE)

- Status: ✅ Complete — SK_DPD columns removed (commit b6821ee), OOF/OOT Gini added, clean store rebuilt
- Branch: `gsd/phase-04.2.3.1-skdpd-leakage-removal-oof-gini`

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

### 2026-04-10 — Phase 04.2.5 Wave 2 CatBoost function rewrite complete

- **Completion:** train_catboost_optuna() fully rewritten from (X, y) DataFrame API to path-based feature_store_path string API
- **Function signature:** `train_catboost_optuna(feature_store_path: str, n_trials=_CAT_OPTUNA_N_TRIALS, groups=None) → tuple[CatBoostClassifier, dict, pd.DataFrame, pd.Series, dict]`
- **Key features implemented:**
  1. **Path-based loading:** Parquet feature store with TARGET column extraction and validation
  2. **Expanded HPO space:** 7 parameters including NEW min_data_in_leaf [5,50] for imbalanced data (307K × 8% positive)
  3. **Optuna persistence:** SQLite-backed study `catboost_raw_scalepos` with TPESampler(n_startup_trials=20) + HyperbandPruner
  4. **2-stage refit:** Stage 1 (80/20 holdout with early stopping) determines best_iteration_; Stage 2 (full X_train) refits without early stopping
  5. **Temporal CV auto-detection:** Checks for _TEMPORAL_SORT_COL in X_train.columns; supports optional external groups parameter
  6. **Platt calibration:** FrozenEstimator + CalibratedClassifierCV sigmoid scaling
  7. **Artifact persistence:** Model pickle, params JSON, metrics JSON, ROC+PR figure
- **Test migration:** Updated fixture to create parquet feature store; all 7 CatBoost tests passing
- **Integration ready:** Return tuple (model, metrics_dict, X_test, y_test, best_params) feeds run_ensemble_workflow() in Phase 04.2.6
- **Commit:** `1ed98e7` with 8 decision points (D-05–D-22) locked
- **Duration:** ~90 min (continued from previous session context compaction)

### 2026-04-10 — Phase 04.2.4 LightGBM feature-store ablation complete

- **2-store ablation:** raw+eng (129 features, 50 trials) vs raw+eng+DFS (290 features, 50 trials)
- **Winner: raw+eng** — OOT Gini 0.5795 vs DFS 0.5767; DFS hurts LGB (−0.0028, wider gap −0.0331 vs −0.0286)
- **Finding:** Featuretools DFS adds ~161 collinear auto-aggregates that dilute LGB's leaf-wise gradient signal; more features ≠ better for LGB at this dimensionality
- **Feature ceiling:** OOF Gini plateaued at ~0.551 from trial 25 (26 consecutive non-improving trials) — HPO budget exhausted without new signal; new features or architecture needed
- **New best single model:** LGB raw+eng OOT 0.5795 (+0.013 over XGBoost 0.5666)
- **OOT positive rate drift:** Train 7.503% vs OOT 10.351% (+38%) — rank metrics (Gini, KS, AUC) remain valid; Brier/calibration affected; Platt recalibration at ~8% deployment prevalence mandatory in Phase 04.2.6
- **Artifact:** `reports/lgb_feature_store_selection.json`
- **Next:** Phase 04.2.5 CatBoost HPO (raw+eng and raw+eng+DFS) — then ensemble in Phase 04.2.6

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

### 2026-04-08 — Phase 04.2.3-01 XGBoost HPO on Raw Features complete

- **Execution:** 3 task commits (TDD RED→GREEN, test migration, integration tests + CLI)
- **Signature rewrite:** train_xgboost_optuna(feature_store_path: str) — loads parquet, extracts TARGET column, enables integration testing
- **HPO improvements (8 total):** Fixed n_estimators=3000, extended gamma [0,2]→[0,5], log-scale min_child_weight [1,30], log-scale regularization [1e-8,5], tree_method='hist', removed max_delta_step, TPESampler(seed=42)+MedianPruner, early stopping via Optuna user_attrs
- **Platt calibration:** Applied sigmoid on 30% train split; CalibratedClassifierCV artifact saved to `models/xgboost_raw_calibrated.pkl`
- **Test migration:** Updated all 188 existing tests to path-based parquet API using `mock_data_parquet_path` fixture factory; all GREEN
- **TDD validation:** 10 new tests verify parquet loading, TARGET extraction, FileNotFoundError, temporal CV auto-detection, study name isolation, early stopping, tree_method, calibration artifacts
- **Integration tests:** 2 automated tests (full pipeline + BrierSkill>0); 1 production test for X_tree_dfs.parquet (skipif missing)
- **CLI wrapper:** `scripts/train_xgboost_raw.py` with --feature-store and --n-trials arguments
- **Commits:** a002e7a (TDD), 4e9198f (migration), a522893 (integration+CLI)
- **Duration:** 85 min | **Status:** Executing Phase 04.2.4

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
| `data/processed/X_tree_raw.parquet` | ✅ Exists | 307511×211 — raw tree store, Phase 04.2.1 complete |
| `data/processed/X_tree_dfs.parquet` | ✅ Exists | 307511×290 — raw+DFS store (confirmed in Phase 04.2.4 ablation); raw+eng (129) wins over this for LGB and XGBoost |
| `models/optuna_studies.db` | ✅ Exists | Continue existing studies, never restart |
| `models/logistic_baseline.pkl` | ✅ Exists | Gini=0.489, KS=0.361 |
| `models/xgboost_raw_best.pkl` | ✅ Exists | XGBoost on clean raw+DFS features (uncalibrated) |
| `models/xgboost_raw_calibrated.pkl` | ✅ Exists | **Primary model** — Platt-calibrated; OOT Gini=0.5666, KS=0.4089 |
| `models/lightgbm_raw_calibrated.pkl` | ✅ Exists | **Best single model** — Platt-calibrated LGB raw+eng; OOT Gini=0.5795 |
| `models/lightgbm_best.pkl` | ⚠️ Superseded | Trained on WoE store — superseded by lightgbm_raw_calibrated.pkl |
| `reports/lgb_feature_store_selection.json` | ✅ Exists | LGB 2-store ablation results; raw+eng wins over raw+eng+DFS |
| `src/data_loader.py` | ✅ Valid | Docstrings corrected (data/ not dataset/) — Phase 02-02 |
| `src/features.py` | ✅ Valid | All paths use `_PROJECT_ROOT`; docstrings corrected — Phase 02 complete |
| `src/auto_features.py` | ✅ Valid | DFS fixed; 15 SK_DPD leaky columns guarded via `_LEAKY_COLUMNS` |
| `src/model.py` | ✅ Valid | All model/report paths anchored to `_PROJECT_ROOT` — Phase 02-03 |
| `src/explain.py` | 🔲 Stub | Phase 04.3 |
| `tests/` | ✅ Valid | All imports use `from src.X import` — no `credit_engine` alias — Phase 02-01 |

## Accumulated Context

### Roadmap Evolution

- Phase 04.2.1 complete: `build_tree_feature_store()` + `X_tree_raw.parquet` (307511×211) — 2026-04-07
- Phase 01 complete: absolute paths, test isolation, credit_engine alias hardened — 2026-04-07
- Phase 02 complete: credit_engine → src imports in all tests, dataset/ → data/ docstrings, _PROJECT_ROOT paths in model.py — 2026-04-10
- Phase 04.2.3.1 complete: SK_DPD leakage removed, OOF/OOT Gini added, clean store rebuilt — 2026-04-10
- Phase 04.2.3.2 complete: 13 missing features added, Layer 6 filter upgraded, XGBoost HPO v9; OOT Gini=0.5666 (target not met), KS=0.4089 ✓ — 2026-04-10
- Phase 04.2.4 complete: LGB 2-store ablation (raw+eng vs raw+eng+DFS, 50 trials each); raw+eng wins OOT 0.5795 (new best single model +0.013 over XGBoost); DFS confirmed to hurt LGB; OOF ceiling at 0.551 — 2026-04-10

---
*State initialized: 2026-04-07*
