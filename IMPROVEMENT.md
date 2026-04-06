# Model Improvement Roadmap

**Version:** 3.0 — Completed work archived; forward roadmap only  
**Target:** Gini ≥ 0.60 (checkpoint), ≥ 0.75 (stretch)  
**Current best model:** XGBoost raw — `models/xgboost_raw.pkl` — **Gini 0.5568** (uncalibrated)

---

## Standings

| Model | Feature Path | Gini | Notes |
|---|---|---|---|
| **XGBoost** | Raw (63 features) | **0.5568** | Best — uncalibrated; calibration script ready (not yet run) |
| XGBoost | WoE (68 features) | 0.547 | Calibrated, BrierSkill +0.098 |
| Logistic Regression | WoE (68 features) | 0.489 | Reference baseline |
| LightGBM | WoE (68 features) | 0.452 | Best LGB result — WoE path |
| Ensemble (LGB+XGB) | WoE | 0.544 | Not persisted (Δ < 0.005 threshold) |
| LightGBM | Raw, GOSS, ablation HP | 0.440 | Track A final — GOSS = WoE LGB; algorithm techniques null |
| LightGBM | Raw, `is_unbalance`, leaves≤150 | 0.440 | Ablation — same as GOSS; raw path ceiling for LGB |
| LightGBM | Raw, `scale_pos_weight`, leaves≤300 | 0.411 | First raw attempt — overfit |

---

## Completed Work & Findings

### Feature Engineering
- **WoE feature store (68 features):** IV + correlation filtering; 130 raw → 40 → 68 after Phase 2 rebuild. Path A artifact: `data/processed/X_features.parquet`.
- **Raw feature store (63 features):** `build_raw_feature_store()` — no binning, IV + corr dedup. Path B artifact: `data/processed/X_raw_features.parquet`, 307,511 × 63, zero NaNs.
- **Secondary aggregates:** 17 std/min/max aggregates across all 5 secondary tables.
- **Cross-table interactions:** 9 interactions in `engineer_secondary_features()` (ext_credit_risk, multi_dpd_flag, debt_service_coverage, etc.).
- **EXT_SOURCE polynomial terms:** 5 terms — `EXT_SOURCE_{1,2}_SQ`, `EXT_SOURCE_RATIO_{12,23}`, `EXT_SCORE_FLOOR`.
- **DFS auto-aggregation:** `src/auto_features.py` complete with 44 tests; `scripts/build_featuretools_store.py` ready.

### Model Training
- **Temporal CV:** `_CV_EMBARGO_FRAC=0.02`, auto-detect `prev_days_decision_mean` as group column. Validated: column present in X_raw (63 columns confirmed).
- **LGB Optuna HPO:** 9-dim search, `is_unbalance=True`, n_estimators ceiling raised 500→1000. Standalone LGB: `models/lightgbm_best.pkl`, Gini 0.4519.
- **XGB Optuna HPO:** gamma, max_delta_step, min_child_weight extensions. `models/xgboost_best.pkl`, Gini 0.5470.
- **XGB Platt calibration:** `calibrate_model()` via `FrozenEstimator` pattern. `models/xgboost_calibrated.pkl`, BrierSkill −1.555→+0.098, Gini preserved 0.5470.
- **CatBoost Optuna HPO:** `train_catboost_optuna()` + `prepare_catboost_features()` — code complete, **never run on production data**.
- **OOF ensemble:** `run_ensemble_workflow()` with temporal CV consistency. Not persisted (Δ = −0.003).
- **XGB raw HPO:** `models/xgboost_raw.pkl`, Gini **0.5568** — new best; raw path adopted for XGB.

### Session 2026-04-05 (Current) — Immediate tasks complete

**XGB raw calibration script** — `scripts/calibrate_xgboost_raw.py` created; loads `data/processed/X_raw_features.parquet` + `models/xgboost_raw.pkl`, calls `calibrate_model()`, saves `models/xgboost_raw_calibrated.pkl` + `reports/xgboost_raw_calibration_results.json`. Pre-flight check asserts shape ≥ 100K before running. 3 TDD tests added to `tests/test_model.py` (`TestCalibrateModelRawPath`). **Not yet run on production — next step.**

**LGB algorithm research** — `docs/lgb_research_findings.md` written; 9 high-performing solutions analysed. Key finding: feature engineering accounts for ~95% of Gini improvement (top solutions use 600–1,600 features vs our 63). Highest-ROI confirmed techniques: DART booster, monotone constraints on EXT_SOURCE/AGE/CREDIT_INCOME_RATIO, reduced `min_child_samples` (20–60 vs our 90). Unblocks Track A (A6–A10).

**Track A API extensions** — `train_lightgbm_optuna()` extended with `boosting_type: Literal["gbdt","dart","goss"] = "gbdt"` and `monotone_constraints: dict[str, int] | None = None`. DART-aware early stopping (skipped for DART; fallback to `best_params["n_estimators"]`). DART searches `drop_rate`, GOSS searches `top_rate` and `other_rate`. Constraint dict converted to column-ordered list. Constants added: `_LGB_DART_DROP_RATE_{MIN,MAX}`, `_LGB_GOSS_{TOP,OTHER}_RATE_{MIN,MAX}`. 8 TDD tests in `TestLGBApiExtensions`, all pass. Unblocks A6–A10 scripts.

**Instalment time-series features** — `engineer_instalment_streaks()` added to `src/features.py`; 5 features via vectorised groupby (no loops): `inst_longest_dpd_streak`, `inst_months_since_last_dpd`, `inst_payment_amt_slope`, `inst_payment_ratio_trend`, `inst_recent_vs_historical_dpd`. 12 TDD tests pass. Integrated into `engineer_secondary_features()` — backward-compatible. **Not yet evaluated on production feature store — next step.**

**Adversarial validation utility** — `adversarial_validation_report()` added to `src/utils.py`; `scripts/adversarial_validation_check.py` created. Trains lightweight LGB to separate train vs test rows; verdicts: safe (AUC < 0.55) / investigate (0.55–0.65) / problematic (≥ 0.65). 9 TDD tests pass. **Run this before any feature store rebuild.**

**Test count:** 252 tests passing (up from 188).

---

### LGB Raw Path Investigation (Resolved)
Two-round ablation (2026-04-05) confirmed that raw continuous features combined with simple HPO are structurally insufficient for LightGBM:

| Config | Gini | Finding |
|---|---|---|
| `scale_pos_weight=True`, leaves≤300, 50 trials | 0.411 | `scale_pos_weight=11.4` × wide leaves = narrow minority-class leaves overfit in continuous space |
| `is_unbalance=True`, leaves≤150, 100 trials | 0.440 | +0.029 recovery; best params `num_leaves=125, max_depth=4, min_child_samples=90, reg_alpha=4.25, reg_lambda=9.54` — Optuna chose maximum regularisation |
| WoE baseline | 0.452 | −0.012 gap remains structural |

**Root cause:** WoE pre-encodes target signal as monotone log-odds bins — LGB splits are directly useful. Raw continuous space requires LGB to discover non-linear structure via greedy leaf-wise growth, which overfits under noise without algorithm-level techniques (DART/GOSS/monotone constraints). XGBoost depth-wise symmetric growth is not affected — raw path gains +0.010.

**Resolution:** XGB adopts raw path as default. LGB improvement requires algorithm-level research (DART/GOSS/monotone constraints — see Track A below). `api_extensions` for `boosting_type` and `monotone_constraints` parameters added to `train_lightgbm_optuna()` search space.

---

## Feature Path Architecture

```
Raw CSVs (7 tables, 307K rows)
       ↓
  Feature Engineering (src/features.py, src/data_loader.py)
  → ~130 raw features after engineer_application_features + engineer_secondary_features
       ↓
  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  PATH A: WoE Binning               PATH B: Raw Continuous           │
  build_feature_store()             build_raw_feature_store()        │
  10-bin quantile WoE               no binning, IV + corr dedup      │
  68 features → X_features.parquet  63 features → X_raw.parquet     │
  For: Logistic Regression          For: XGBoost ✅ (Gini 0.5568)   │
       Ensemble meta-learner             LightGBM ⚠ (needs DART/    │
       IRB documentation                GOSS/monotone — Track A)     │
       LightGBM (WoE still best         CatBoost (+ native cats)     │
       until Track A research)          Ensemble base learners       │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘
       ↓ (additive, once DFS run completes)
  PATH C: DFS Auto-Aggregation (src/auto_features.py)
  build_featuretools_feature_store()
  ~400–600 raw DFS → IV + corr filter → ~80–100 features
  → X_featuretools.parquet
  For: LGB (combined raw+DFS), XGB (evaluation pending)
```

---

## Roadmap

### Immediate (no blockers)

**Run XGB raw calibration** — Track C  
Script ready at `scripts/calibrate_xgboost_raw.py`. Execute: `python scripts/calibrate_xgboost_raw.py`. Verify Gini unchanged and BrierSkill improves to positive. Artifacts: `models/xgboost_raw_calibrated.pkl`, `reports/xgboost_raw_calibration_results.json`.  
*Effort: 5 min to run.*

**Run adversarial validation check**  
Script ready at `scripts/adversarial_validation_check.py`. Execute: `python scripts/adversarial_validation_check.py`. Expect verdict "safe" (AUC < 0.55). Output: `reports/adversarial_validation.json`. Required before feature store rebuild.  
*Effort: 5 min to run.*

**Evaluate instalment streaks on production data**  
`engineer_instalment_streaks()` is implemented; rebuild feature store with instalment data supplied to `engineer_secondary_features(df_inst=...)`. Measure XGB Gini delta. Commit new features if delta > +0.01; defer to Track B if ≤ +0.01.  
*Effort: 30 min.*

**LGB algorithm research** — ✅ DONE  
`docs/lgb_research_findings.md` written. 9 solutions analysed. Unblocks Track A (A6–A10).

---

### Track A — LightGBM Algorithm-Level Recovery

**Completed — config, findings, results:**

| Step | Config | Finding | Result |
|---|---|---|---|
| A5 — Research | `docs/lgb_research_findings.md`; 9 solutions | Key techniques: DART, GOSS, monotone constraints, min_child_samples ↓ | Unblocked A6–A10 |
| API extensions | `boosting_type` + `monotone_constraints` in `train_lightgbm_optuna()`; booster-specific HP searched | DART early stopping skipped; GOSS top_rate/other_rate searched | 8 tests pass |
| A6 — Booster comparison | GBDT / DART / GOSS; fixed ablation HP; 5-fold temporal CV | GOSS best; DART −0.031 vs GOSS | GOSS **0.5547**, GBDT 0.5535, DART 0.5240 |
| A7 — Monotone constraints | All 7 features; GOSS; 5-fold CV | `CREDIT_INCOME_RATIO` + `inst_days_past_due_mean` absent from raw store (only 5 active) | delta = −0.001 — **degraded**; constraints not applied |
| A8 — HP grid | 4×4 `num_leaves` × `min_child_samples`; GOSS; 80/20 split | `num_leaves` has no effect (GOSS insensitive to leaf count); Gini plateau across all rows | Gini 0.5568 at `min_child_samples=10/50`; `num_leaves_max` set to 300 |
| A9 — Early stopping | Patience ∈ {20, 50, 100, None}; GOSS; 5-fold CV | Early stopping never triggers — all folds hit n_estimators=500 ceiling | Recommended patience=20; n_estimators must be raised in A10 |
| A10 v1 — Optuna (free search) | GOSS; 100 trials; 10-fold CV; `reg_lambda` ∈ [0,15]; `lr` ∈ [0.01,0.2] | Optuna chose `reg_lambda=1.43` (vs ablation 9.54) + `lr=0.024`; under-regularised at low LR | **Gini 0.4364** — insufficient; search space too permissive |
| A10 v2 — Raised reg/LR floors | `reg_lambda_min`→3.0; `lr_min`→0.03; `n_estimators_max`→1000 | Optuna chose `num_leaves=35`; floor raised to 75 for v3 | **Gini 0.4251** — still insufficient |
| A10 v3 — Raised num_leaves floor | `_LGB_NUM_LEAVES_MIN`→75 | Optuna chose `num_leaves=199, max_depth=3` — `max_depth=3` caps effective nodes at 8 regardless of num_leaves; simpler than A6's `max_depth=4` (16 nodes) | **Gini 0.4100** — degraded further |

| A10 v4 — Ablation HP warm start | `enqueue_trials` added to `train_lightgbm_optuna()`; ablation HP as trial #1; 100 subsequent Optuna trials could not beat warm start | `best_params` = exact ablation HP; Optuna found no improvement | **Gini 0.4400** — warm start was the best trial |

**Track A verdict: CLOSED — null result.**  
A6's reported Gini 0.5547 was a CV metric on the *full dataset* (no held-out test set), inflating the figure by ~0.11 vs proper 80/20 evaluation. When evaluated on a held-out test set with the same HP, GOSS achieves 0.44 — equal to WoE LGB (0.4519) and below XGB raw (0.5568). GOSS/DART/monotone constraints provide no lift over standard LGB on the raw feature path.  
**Primary model remains XGBoost raw: `models/xgboost_raw.pkl`, Gini 0.5568.**

---

### Track B — DFS Feature Evaluation

**Blocker:** DFS generation job must complete (`data/processed/X_featuretools.parquet`).  
**Expected gain:** +0.02–0.04 Gini (additive over raw path)

**B1: Confirm DFS job completion**  
```bash
tail -f /tmp/featuretools_dfs.log
# Look for: "DFS complete in X.X min" + "Selected columns: N"
```

**B2: Evaluate DFS features per model**  
```bash
.venv/bin/python -u scripts/eval_featuretools.py
# Evaluates: featuretools-only + raw+featuretools combined
# Results: reports/featuretools_eval_results.json
```

**B3: Build combined feature set**  
Union `X_raw` + DFS-only columns (drop overlaps), re-apply correlation dedup at |r| > 0.90. Output: `data/processed/X_combined_features.parquet` (~100–130 features).

---

### Track C — XGBoost: Calibration

**C2: Calibrate `models/xgboost_raw.pkl`**  
See "Immediate" section above. Highest ROI task with no blocker.

---

### Track D — CatBoost + 3-Model Ensemble

**Blocker:** LGB baseline must be known (after A10) to assess ensemble diversity value.  
**Expected gain:** +0.01–0.02 Gini from diversity

**D1: CatBoost production evaluation**  
```python
X_cat, cat_cols = prepare_catboost_features(X_raw, df_raw=df_app)
cat_model, metrics, X_test, y_test, params = train_catboost_optuna(X_cat, y, n_trials=30)
# Expected standalone Gini: 0.52–0.56
# Artifact: models/catboost_best.pkl
```

**D2: 3-model ensemble**  
Extend `run_ensemble_workflow()` or create `run_3model_ensemble_workflow()` passing LGB OOF + XGB OOF + CatBoost OOF into a LogisticRegression meta-learner. Persist if improvement ≥ 0.005.

---

### Shared Cross-Model Improvements

These benefit all feature paths and model tracks.

**Instalment time-series features** — ✅ IMPLEMENTED (needs production eval)  
`engineer_instalment_streaks()` added to `src/features.py`. 5 features, 12 tests pass, vectorised (no loops), backward-compatible. Expected gain: +0.01–0.03 Gini. **Next: rebuild feature store with `df_inst=` param and measure Gini delta.**

**EXT_SOURCE imputation via LightGBM regression** — `impute_ext_source_values()` in `src/features.py`  
*Expected gain: +0.01–0.02 Gini. Effort: 2 hrs. After instalment streaks.*

Fit LGB regressors on non-missing rows for each of EXT_SOURCE_1/2/3; predict missing values. Add binary `EXT_SOURCE_{1,2,3}_MISSING` flags alongside imputed values. Leakage rule: regressors fit on `df_train` only. Call early in `engineer_application_features()` before EXT_SOURCE composites are derived.

**Adversarial validation** — ✅ IMPLEMENTED  
`adversarial_validation_report()` in `src/utils.py`; `scripts/adversarial_validation_check.py` ready. AUC < 0.55 = safe; 0.55–0.65 = investigate; > 0.65 = drop or winsorise before HPO. 9 tests pass. Run after each feature store rebuild.

---

## Session Execution Plan

| Session | Tasks | Target | Blocker |
|---|---|---|---|
| **✅ Done** | XGB calibration script (C2) + LGB research (A5) + API extensions + instalment streaks impl + adversarial validation impl + Track A (A6–A10) | Track A closed: GOSS null result; XGB raw 0.5568 remains best | — |
| **Next** | Run XGB raw calibration (C2) + adversarial check + instalment eval on prod data | Calibrated XGB pkl + data integrity confirmed | None |
| **2** | Run XGB calibration + adversarial check + instalment eval on prod data + EXT_SOURCE imputation | Calibrated XGB pkl + data integrity confirmed + Gini delta measured | None |
| **3** | DFS eval (B2–B3) ∥ instalment feature commit (if eval > +0.01) | Combined feature set | DFS job done |
| **4** | LGB integration (A10) → CatBoost eval (D1) → 3-model ensemble (D2) | Final best Gini | A6–A9 done |

**Cumulative expected gain from current best (0.5568):** +0.10–0.18 Gini  
**Realistic ceiling:** ~0.65–0.69

---

## Testing & Validation Gates

Before committing any improvement:

- [ ] `pytest tests/test_features.py -v` passes (no regression in feature tests)
- [ ] New feature store `shape[0] >= 100K` (not silently overwritten by test mock data)
- [ ] Model Gini confirmed on same `X_test, y_test` split with `random_state=42` (reproducible)
- [ ] CV Gini vs test Gini gap < 0.05 (otherwise suspect leakage or shift)
- [ ] No NaN/inf: `assert X.isnull().sum().sum() == 0` before training
- [ ] All model artifacts saved (`models/*.pkl`) and eval results as JSON
- [ ] Adversarial AUC < 0.65 after combined feature set is built

---

## Risk Flags

| Risk | Probability | Mitigation |
|---|---|---|
| A10 integration doesn't beat 0.452 WoE baseline | Medium | If LGB < 0.46 after all techniques: LGB raw path abandoned; pursue DFS (Track B) as primary LGB route |
| DART incompatible with temporal early stopping | Medium | LGB 3.0+ supports both; test in A6 before HPO integration |
| DFS memory overflow | Low | `max_depth=1` keeps feature count manageable; monitor RAM during run |
| DFS features highly correlated | Medium | Correlation dedup at 0.90 threshold; check max \|r\| in output |
| Ensemble not improving over best single | Medium | Add CatBoost as 3rd member for diversity; try Ridge meta-learner |
| XGB raw calibration shifts Gini | Low | Platt scaling is monotone — Gini is rank-based and preserved by construction |

---

## Gini Target Reference

| Gini Band | Status | Key Enablers |
|---|---|---|
| 0.45–0.55 | ✅ Achieved (LGB WoE / LR) | WoE features, Optuna HPO |
| 0.55–0.60 | ✅ Achieved (XGB raw 0.5568) | Raw features, XGB depth-wise growth |
| 0.60–0.65 | Next checkpoint | Combined raw+DFS, 3-model ensemble |
| 0.65–0.70 | Stretch | Instalment streaks, EXT_SOURCE imputation, neural meta-learner |
| **≥ 0.608** | **Top published result (2018)** | All above + DART/GOSS/monotone on LGB |
| ≥ 0.75 | Exceeds prior-art ceiling | External data / deep tabular models (SAINT, TabNet) |
