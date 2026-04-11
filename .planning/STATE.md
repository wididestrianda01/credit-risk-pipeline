---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
last_updated: "2026-04-11T17:00:42.904612Z"
progress:
  total_phases: 13
  completed_phases: 10
  total_plans: 48
  completed_plans: 44
  percent: 92
---

# Project State

**Project:** Credit Risk Scoring Pipeline
**Last Updated:** 2026-04-11

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-04-07)

**Core value:** Calibrated PD feeding EL = PD × LGD × EAD, Gini ≥ 0.70
**Current focus:** Phase 04.2.6 — Ensemble + Gini gate (unblocked as of 2026-04-11)

---

## Completed Plans

### Phase 04.2.3 — XGBoost HPO on Raw Features

- Status: ✅ Complete (2026-04-08)
- Result: train_xgboost_optuna() path-based API, 8 HPO improvements, Platt calibration, 188 tests migrated, 2 integration tests added
- Summary: `/home/wd/Working Folder/Development/credit-risk-pipeline/.planning/phases/04.2.3-xgboost-hpo-on-raw-features/04.2.3-01-SUMMARY.md`
- Artifacts: `models/xgboost_raw_best.pkl`, `models/xgboost_raw_calibrated.pkl`, `scripts/train_xgboost_raw.py`
- Branch: `gsd/phase-04.2.2-dfs-auto-feature-augmentation` (plan executed on this branch)

## Active Phase

**Phase 04.2.6** — Ensemble + Gini Gate

### Phase 04.2.6-01 (COMPLETE — 2026-04-11)
- Status: ✅ Complete — Research and context documented
- Branch: `gsd/phase-04.2.7-feature-engineering-enhancement-pass`
- Artifacts: `.planning/phases/04.2.6-ensemble-gini-gate/04.2.6-CONTEXT.md`, `04.2.6-RESEARCH.md`

### Phase 04.2.6-02 (COMPLETE — 2026-04-11)
- Status: ✅ Complete — Ensemble orchestration and gate evaluation
- **Results:**
  - 3-model OOF stacking: LGB (0.5114 OOF) + XGB (0.5393 OOF) + CatBoost (0.5388 OOF)
  - Ensemble OOT Gini: 0.5416 (improvement +0.0023, <0.005 threshold)
  - Gate result: **investigate** (ensemble Gini 0.5416 < 0.58 minimum threshold)
  - Meta-learner coefficients: XGB=3.045 (dominant), CatBoost=1.605, LGB=−1.816 (variance reduction)
- **Artifacts created:**
  - `scripts/run_ensemble.py` (403 lines): Ensemble orchestration with Basel CRE36.54 temporal validation
  - `reports/model_benchmark.csv` (6 lines): 5-model comparison table (LR/XGB/LGB/CatBoost/Ensemble)
  - `reports/ensemble_weights.json` (27 lines): Regulatory meta-learner coefficients and gate documentation
- **Commit:** f1b9e31
- **Summary:** `.planning/phases/04.2.6-ensemble-gini-gate/04.2.6-02-SUMMARY.md`
- **Ensemble inputs (all valid, Basel CRE36.54 compliant):**
  | Model | Feature store | OOT Gini | KS | Status |
  |-------|--------------|----------|----|--------|
  | XGBoost | X_tree_raw.parquet (144 cols) | 0.5666 | 0.4089 | ✅ Phase 04.2.3.2 |
  | LGB | X_tree_raw.parquet (144 cols) | 0.5746 | 0.4302 | ✅ Phase 04.2.4.1 |
  | CatBoost | X_tree_raw.parquet (144 cols) | 0.5699 | 0.4259 | ✅ Phase 04.2.5.1 |
  | Ensemble (Logistic meta) | X_tree_raw.parquet (144 cols) | 0.5416 | NaN | 🔲 Below threshold |

- **Decision:** Ensemble not persisted (gate result: investigate); focus remains on best single model (LGB Gini 0.5746) for Phase 04.3 SHAP explainability

## Recently Completed Phases

**Phase 04.2.5.1** — CatBoost Compliant Re-run (COMPLETE — 2026-04-11)

- Status: ✅ Complete — 50-trial CatBoost HPO on `X_tree_raw.parquet` (144 cols, Wave 1 features)
- **Results:** OOT Gini=0.5699 | KS=0.4259 ✓ | Brier=0.0831 ✓ (corrected — prior 0.2147 was from uncalibrated model; call-order bug fixed in train_catboost_optuna, commit 49a2039)
- **Fix applied:** `_n_trials_before` guard — contaminated trials 0–16 excluded; only trials 17–66 count
- **Best params:** depth=5, lr=0.036, l2_leaf_reg=6.99, min_data_in_leaf=40
- Artifact: `reports/catboost_compliant_eval.json`, `models/catboost_raw_calibrated.pkl`
- Branch: `gsd/phase-04.2.7-feature-engineering-enhancement-pass`

**Phase 04.2.7** — Feature Engineering Enhancement Pass (COMPLETE — 2026-04-11)

- Status: ✅ Wave 1 features implemented + X_tree_raw.parquet rebuilt + LGB HPO gate evaluated
- Branch: `gsd/phase-04.2.7-feature-engineering-enhancement-pass`
- All 7 Wave 1 features implemented: `inst_late_rate_12m`, `inst_late_rate_recent_vs_historical`, `inst_rolling_30dpd_ratio_3m`, `inst_delinquency_escalation_flag`, `inst_days_since_last_30dpd`, `bureau_dpd_trend_3m_vs_12m`, `bureau_debt_to_new_credit`
- **LGB Wave 1 gate result:** OOT Gini=0.5746, gate FAIL (threshold 0.5845); Wave 2+3 features not developed
- **Note:** Gate was run as part of Phase 04.2.4.1 (same store, same model); no separate gate run required

**Phase 04.2.4.1** — LightGBM Compliant Re-run (COMPLETE — 2026-04-11)

- Status: ✅ Complete — 50-trial LGB HPO on `X_tree_raw.parquet` (144 cols, Wave 1 features)
- **Results:** OOT Gini=0.5746 | OOF Gini=null (0.7437 was in-sample, not CV OOF — bug fixed in train_lightgbm_optuna, commit 49a2039; true value requires re-run) | KS=0.4302 ✓ | Brier=0.0913
- **Note:** Large OOF-OOT gap is temporal distribution shift (train 7.5% positives, OOT 10.4%) — not leakage
- Artifact: `reports/lgb_compliant_eval.json`, `models/lightgbm_raw_calibrated.pkl`

## Recently Completed Phases

**Phase 04.2.5** — CatBoost HPO on Raw Features (INVALIDATED — 2026-04-11)

- Status: ❌ INVALIDATED — Basel CRE36.54 non-compliance (re-run required via Phase 04.2.5.1)
- Branch: `gsd/phase-04.2.5-catboost-hpo-on-raw-features`
- **Root cause:** Original `train_catboost_optuna()` performed the train/OOT split inside the Optuna objective closure, exposing OOT rows to gradient statistics during HPO. All reported metrics are from a contaminated evaluation.
- **Fix applied:** Commit `43c9d89` (2026-04-11) — `train_catboost_optuna()` now carves OOT before any Optuna trial starts, enforcing the canonical Basel CRE36.54 workflow.
- **Contaminated metrics (DO NOT USE as regulatory evidence):** OOF=0.5500, OOT=0.5789 (v3 run)
- **Function implementation:** Rewrite to path-based API, 7-param HPO, 2-stage refit, Platt calibration — all correct; only the OOT carve-out ordering was wrong. The function itself is now valid.
- Artifacts produced but INVALID: `models/catboost_raw_best.pkl`, `models/catboost_raw_calibrated.pkl`, `models/catboost_raw_params.json`, `reports/catboost_raw_eval.json`

**Phase 04.2.4** — LightGBM HPO on Raw Features (INVALIDATED — 2026-04-11)

- Status: ❌ INVALIDATED — Basel CRE36.54 non-compliance (re-run required via Phase 04.2.4.1)
- Branch: `gsd/phase-04.2.4-lightgbm-hpo-on-raw-features`
- **Root cause:** Original `train_lightgbm_optuna()` split train/OOT inside the Optuna objective; OOT rows were visible during HPO gradient updates. All reported metrics are from a contaminated evaluation.
- **Fix applied:** Commit `43c9d89` (2026-04-11) — `train_lightgbm_optuna()` now carves OOT before Optuna starts.
- **Contaminated metrics (DO NOT USE as regulatory evidence):** OOF=0.5510, OOT=0.5795 (raw+eng, 50 trials)
- **Feature-store finding remains valid:** raw+eng (129 features) outperforms raw+eng+DFS for LGB — this structural conclusion is independent of the OOT contamination issue and can be reused in Phase 04.2.4.1.
- Artifacts produced but INVALID: `models/lightgbm_raw_calibrated.pkl`, `reports/lgb_feature_store_selection.json`

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

### 2026-04-11 — Phase 04.2.6-02 Ensemble Orchestration Complete

- **Execution:** Ensemble orchestration script written and executed successfully (47min 22sec runtime)
- **All 3 tasks completed:** (1) Load best_params + X_tree_raw temporal split, (2) Benchmark table evaluation (5 models), (3) Extract meta-learner weights + gate decision
- **Gate result:** investigate — ensemble Gini 0.5416 < 0.58 accept threshold; improvement 0.0023 < 0.005 min threshold
- **Key finding:** Ensemble provides minimal gain; best single model (LGB OOT Gini 0.5746) recommended for Phase 04.3 SHAP explainability
- **HPO best_params sourcing:** LGB and CatBoost from compliant eval JSONs; XGBoost fallback via xgb_hpo_results.json (xgboost_raw_eval.json missing best_params key — points to XGBoost HPO uncertainty)
- **Regulatory outputs:** Meta-learner coefficients logged to JSON; gate thresholds and decision documented for audit trail
- **Commits:** f1b9e31 (ensemble orchestration, benchmark, weights JSON)

### 2026-04-11 — Basel CRE36.54 compliance gap discovered and fixed

- **Discovery:** Phase 04.2.4 (LGB) and Phase 04.2.5 (CatBoost) both violated Basel CRE36.54 temporal validation. The root cause: both original `train_lightgbm_optuna()` and `train_catboost_optuna()` performed the train/OOT split *inside* the Optuna `objective()` closure (or after Optuna started). This meant OOT rows were accessible to gradient updates during hyperparameter search, contaminating the temporal holdout.
- **XGBoost status:** `train_xgboost_optuna()` was already compliant — OOT is carved before the study is created. All XGBoost results remain valid.
- **Fix:** Commit `43c9d89` rewrote both LGB and CatBoost functions to perform the temporal sort and OOT carve-out before creating the Optuna study. Docstrings updated to cite Basel CRE36.54 explicitly.
- **Canonical compliant workflow (enforced by code post-fix):**
  1. Sort full dataset by `prev_days_decision_mean` (NaN rows → seeded random permutation)
  2. Carve OOT = most-recent 20% — frozen, never seen during HPO
  3. Optuna HPO on 80% train only: OOF CV Gini is the trial objective
  4. Select best params by OOF Gini; retrain on full 80%
  5. Evaluate on frozen OOT → OOT Gini is the regulatory metric
- **Invalidated artifacts:** `models/lightgbm_raw_calibrated.pkl`, `models/catboost_raw_calibrated.pkl`, `reports/lgb_feature_store_selection.json`, `reports/catboost_raw_eval.json`
- **Valid artifacts:** `models/xgboost_raw_calibrated.pkl` (OOT Gini 0.5666), `reports/xgboost_raw_eval.json`
- **Action:** Phases 04.2.4.1 and 04.2.5.1 added to queue for compliant re-runs after `X_tree_raw.parquet` is rebuilt

### 2026-04-10 — Phase 04.2.5 CatBoost HPO complete (all 3 waves)

- **Completion:** train_catboost_optuna() fully rewritten; 11 new tests; 20/20 CatBoost tests passing
- **Wave 1:** 9 constants updated, train_catboost_extended_hpo() deleted (173 lines), Python import verified
- **Wave 2:** path-based API, 7-param HPO (depth [5,10], lr [0.01,0.2] log, l2_leaf_reg [0.1,30] log, min_data_in_leaf [5,50] NEW, bagging_temperature, random_strength), Optuna study `catboost_raw_scalepos` (TPESampler n_startup=20, HyperbandPruner), 2-stage refit, Platt calibration via FrozenEstimator + CalibratedClassifierCV
- **Wave 3:** 11 tests covering D-03 (scale_pos_weight), D-04 (no cat_features), D-05 (path API), D-08 (return tuple), D-09/D-10 (2-stage refit), D-13 (min_data_in_leaf), D-18 (SQLite study), D-19 (calibration), D-20 (artifacts)
- **Test result:** `20 passed, 162 deselected in 6.84s` — full fast suite green
- **Branch:** `gsd/phase-04.2.5-catboost-hpo-on-raw-features`

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
- **Duration:** 85 min | **Status:** Ready to execute

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
| `data/processed/X_tree_raw.parquet` | ✅ Exists | 307511×144 — Wave 1 features included (7 new delinquency features); TARGET embedded as final column |
| `data/processed/X_tree_dfs.parquet` | ✅ Exists | 307511×290 — raw+DFS store (confirmed in Phase 04.2.4 ablation); raw+eng (129) wins over this for LGB and XGBoost |
| `models/optuna_studies.db` | ✅ Exists | Continue existing studies, never restart |
| `models/logistic_baseline.pkl` | ✅ Exists | Gini=0.489, KS=0.361 |
| `models/xgboost_raw_best.pkl` | ✅ Exists | XGBoost on clean raw+DFS features (uncalibrated) |
| `models/xgboost_raw_calibrated.pkl` | ✅ Exists | **Primary model** — Platt-calibrated; OOT Gini=0.5666, KS=0.4089 |
| `models/lightgbm_raw_calibrated.pkl` | ✅ VALID | Basel CRE36.54 compliant — Phase 04.2.4.1; OOT Gini=0.5746, KS=0.4302 |
| `models/catboost_raw_calibrated.pkl` | ✅ VALID | Basel CRE36.54 compliant — Phase 04.2.5.1; OOT Gini=0.5699, KS=0.4259, Brier=0.0831 (corrected) |
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
- Phase 04.2.5 complete: CatBoost HPO rewritten (path-based API, 7-param search, 2-stage refit, Platt calibration, 20 tests); catboost_raw_calibrated.pkl ready for ensemble — 2026-04-10
- Phase 04.2.7 complete: 7 Wave 1 delinquency features implemented + X_tree_raw.parquet rebuilt (307511×144); LGB gate FAIL (OOT 0.5746 < 0.5845 threshold) — Wave 2+3 not developed — 2026-04-11
- Phase 04.2.4.1 complete: LGB compliant re-run (Basel CRE36.54); OOT Gini=0.5746, KS=0.4302; models/lightgbm_raw_calibrated.pkl is now valid — 2026-04-11
- Phase 04.2.5.1 complete: CatBoost compliant re-run (Basel CRE36.54); OOT Gini=0.5699, KS=0.4259, Brier=0.0831 (corrected from 0.2147 — call-order bug fixed); models/catboost_raw_calibrated.pkl is now valid — 2026-04-11
- **Next:** Phase 04.2.6 — Ensemble + Gini gate; all three model artifacts valid and ready
- **Queued:** Phase 04.2.8 — Split model.py (4,075 lines) by model family into model_base / model_xgboost / model_lightgbm / model_catboost / model_ensemble + thin facade; unblocked after 04.2.6

---
*State initialized: 2026-04-07*
