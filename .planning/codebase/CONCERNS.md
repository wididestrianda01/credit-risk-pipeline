# Codebase Concerns

**Analysis Date:** 2026-04-11

## Critical Issues

### Gini Target Not Met — 20% Performance Gap

**Issue:** Target Gini ≥ 0.70 remains unmet. Current best single models plateau at ~0.575 OOT Gini.

- **Files:** `src/model.py` (train_xgboost_optuna, train_lightgbm_optuna, train_catboost_optuna)
- **Current results:**
  - XGBoost: OOT Gini=0.5666 (X_tree_dfs.parquet, 290 cols) — Phase 04.2.3.2
  - LightGBM: OOT Gini=0.5746 (X_tree_raw.parquet, 144 cols) — Phase 04.2.4.1 ✅ Compliant
  - CatBoost: OOT Gini=0.5699 (X_tree_raw.parquet, 144 cols) — Phase 04.2.5.1 ✅ Compliant
- **Gap:** ~0.125 Gini points below target (17.9% shortfall)
- **Root cause:** Feature signal exhaustion. LGB OOF Gini plateaued at 0.744 from trial 25 onward (26 consecutive non-improving trials in Phase 04.2.4 ablation); Featuretools DFS added 161 auto-aggregates with negligible lift (−0.0028 delta for LGB, within noise). Wave 1 delinquency features (7 new) yielded OOT Gini=0.5746, failing the gate threshold of 0.5845.
- **Impact:** Model cannot meet regulatory approval threshold (Basel III IRB expects Gini ≥ 0.65–0.70 for PD models). Scoring may be rejected in IRB submission.
- **Improvement path:**
  1. **Wave 2 features** (skipped due to gate FAIL): `GOODS_CREDIT_RATIO`, `CREDIT_TERM`, `EXT_SOURCE_MEAN`/`MIN`, `DOCUMENTS_SUBMITTED`, `HIGH_RISK_DOC_MISSING` — these were designed but not implemented because Phase 04.2.7 gate failed (OOT 0.5746 < 0.5845)
  2. **Feature interactions:** Cross-table polynomials (e.g. `EXT_SOURCE_MEAN * CREDIT_INCOME_RATIO`, `DELINQUENCY_FLAG * CREDIT_TERM`)
  3. **Alternative architectures:** Ensemble (Phase 04.2.6, gating rule: `ensemble_gini - max(lgb, xgb) >= 0.005`); expected yield ~0.005 minimum
  4. **Data leak investigation:** Verify secondary tables contain no post-origination information (e.g. confirm `MONTHS_BALANCE >= -6` cutoff excludes application month 0 and later)

### CatBoost Calibration Metrics Anomaly

**Issue:** `catboost_compliant_eval.json` reports OOF Gini=0.0 and BrierSkill=−1.27.

- **Files:** `reports/catboost_compliant_eval.json`, `src/model.py` (~line 1900 for `train_catboost_optuna()`)
- **Symptoms:**
  - OOF Gini=0.0 (invalid metric — should be > 0 for any discriminative model)
  - OOF-OOT gap = −0.5699 (negative, extremely unusual; gap should be positive indicating train overfitting)
  - Brier=0.2147 (very high; 8% prevalence baseline is 0.0736; uncalibrated but still concerning)
  - BrierSkill=−1.27 (worse than random guessing; indicates model inversion or severe miscalibration)
- **Likely cause:** 
  1. CatBoost 2-stage refit (line ~1920) may use incorrect target encoding or feature matrix shape mismatch
  2. Platt calibration sigmoid fit on wrong indices (probability inversion)
  3. TARGET column not properly extracted/aligned during calibration
- **Recommendation:** Audit `train_catboost_optuna()` around lines 1880–1950:
  1. Verify Platt signature matches XGBoost/LGB (load best booster, fit sigmoid on 30% holdout split from X_train, validate output probabilities in [0,1])
  2. Check that OOF fold indices are correctly aligned (no off-by-one in fold assignment)
  3. Log intermediate OOF Gini values for each fold to identify anomalous fold
  4. Consider re-running Phase 04.2.5.1 with debug output enabled

### Large OOF-OOT Gap (LightGBM) — 16.9% Divergence

**Issue:** LGB reports OOF Gini=0.7437 vs OOT Gini=0.5746, gap=0.1691 (169 basis points).

- **Files:** `reports/lgb_compliant_eval.json` (Phase 04.2.4.1), `src/model.py` (~line 1600)
- **Root cause (documented):** Temporal distribution shift — train/OOF 7.503% positive rate vs OOT 10.351% positive rate (+38% elevation). Not data leakage, but structural vintage drift.
- **Risk:**
  - If model deployed at current portfolio 8% default rate, calibration curve will be off-diagonal (fitted on higher prevalence, applied at lower)
  - Brier score will inflate by ~0.015–0.02 points at deployment prevalence
  - Gini and KS (rank metrics) remain valid under prevalence shift, but Brier and calibration degrade
- **Current mitigation:** Platt sigmoid was refit in Phase 04.2.4.1 assuming OOT prevalence (10.4%); Phase 04.2.6 (pending) must recalibrate for deployment prevalence (8%)
- **Safe modification:** Before go-live, validate calibration on held-out cohort at 8% default rate; trigger auto-recalibration if observed positive rate drifts >±1.5%

## Tech Debt

### Monolithic Source Files

**Issue:** Three core modules exceed sustainable size (>2000 lines) making navigation and testing slow.

- **Files:** 
  - `src/model.py` — 4075 lines (model training, CV, calibration, ensemble, XGBoost/LGB/CatBoost HPO all combined)
  - `src/features.py` — 3022 lines (engineering, WoE, IV selection, feature stores, raw + WoE variants)
  - `src/data_loader.py` — 1144 lines (7-table join, aggregations, validation)
- **Impact:** 
  - Hard to navigate (200+ function search hits for `def train`)
  - High test execution time (30–45 min for full suite)
  - Fixture scope complexity (module-scoped fixtures required for memory, risking test interference)
- **Recommendation (future phase):** Extract into focused modules:
  - `src/cv.py`: `_TemporalCV`, `_make_cv()`, CV utilities (200 lines)
  - `src/calibration.py`: `calibrate_model()`, `FrozenEstimator`, Platt sigmoid (150 lines)
  - `src/ensemble.py`: `run_ensemble_workflow()`, OOF stacking (300 lines)
  - `src/woe.py`: WoE binning, IV filtering, Information Value (400 lines from features.py)
  - `src/aggregation.py`: Bureau/previous_application/POS/installment aggregates (600 lines from data_loader.py)

### Incomplete Explainability Module

**Issue:** `src/explain.py` is a 31-line stub with two `# TODO: implement` functions.

- **Files:** `src/explain.py:18–31`
- **Missing:**
  - `compute_shap_values()`: TreeExplainer on XGBoost/LGB, Shapley aggregation, waterfall/force plots
  - `fairness_report()`: Demographic parity, equalised odds, disparate impact ratio by protected group
- **Impact:** 
  - GDPR Art. 22 (right to explanation) unmet — no adverse action notice generation
  - EU AI Act Art. 6 (transparency for high-risk AI) unmet — no interpretability evidence
  - Phase 04.3 blocking (regulatory compliance)
  - Phase 05.2 (Streamlit dashboard) cannot start without SHAP
- **Scope:** ~400–500 lines estimated (SHAP computation, fairness metrics, caching, plotting)
- **Priority:** High — regulatory requirement for production approval
- **Recommendation:** Implement in Phase 04.3 immediately after Phase 04.2.6 (ensemble); parallelise with Phase 05.1 (API skeleton) if feasible

### Incomplete Deployment Modules

**Issue:** `app/api.py` and `app/streamlit_app.py` are non-functional stubs.

- **Files:**
  - `app/api.py` (42 lines) — FastAPI application with stubbed endpoints
  - `app/streamlit_app.py` (30 lines) — Streamlit dashboard with all sections as TODOs
- **API gaps:**
  - POST /predict raises `NotImplementedError`
  - Input schema `ApplicantFeatures` is empty (`pass`)
  - No feature transformation pipeline (reuse `src/features.build_tree_feature_store()`)
  - No model loading or scoring logic
  - Missing error handling (bad request, NaN/inf rejection, feature range validation)
  - No authentication or rate limiting
- **Streamlit gaps:**
  - No data loading, model loading, or SHAP computation
  - All 4 dashboard sections are TODOs (score distribution, global importance, waterfall, fairness)
  - No session_state for caching (will be slow if rebuilt on every interaction)
- **Impact:** Phase 05 (deployment) cannot start; end-to-end scoring flow undefined
- **Scope:** ~200 lines API + ~300 lines Streamlit estimated
- **Blocking:** Phase 05.1 and 05.2; also depends on Phase 04.3 (SHAP) for dashboard features
- **Recommendation:**
  1. Phase 05.1: Define FastAPI contract (request/response schemas matching 144-column raw feature set); implement feature transformation and scoring; add Pydantic validators for input ranges
  2. Phase 05.2: Wire up SHAP computation (lazy-loaded per request or cached); implement fairness metrics dashboard

## Known Risks

### Basel CRE36.54 Temporal Validation Non-Compliance (FIXED)

**Issue:** Phases 04.2.4 and 04.2.5 originally violated Basel III temporal validation (FIXED in commit 43c9d89; 2026-04-11).

- **Files:** `src/model.py` (~lines 1550–1650 LGB, ~lines 1800–1900 CatBoost)
- **Root cause (now fixed):** Original functions performed train/OOT split *inside* the Optuna `objective()` closure, exposing OOT rows to gradient statistics during hyperparameter search
- **Violation impact:** Contaminated OOT Gini metric becomes inadmissible as regulatory evidence
- **Fix applied (commit 43c9d89):**
  1. **Sort full dataset by `prev_days_decision_mean`** (NaN rows → seeded random permutation)
  2. **Carve OOT = most-recent 20% rows BEFORE Optuna study creation** (frozen, never touched during HPO)
  3. Optuna HPO operates on remaining 80% train only; OOF CV Gini is trial objective
  4. Select best_params by OOF Gini; **retrain single model on full 80%** (no CV)
  5. Evaluate on frozen OOT → **OOT Gini is regulatory metric**
- **Invalidated pre-fix artifacts:**
  - `models/lightgbm_raw_calibrated.pkl` (Phase 04.2.4, contaminated OOT 0.5795) ❌
  - `models/catboost_raw_calibrated.pkl` (Phase 04.2.5, contaminated OOT 0.5789) ❌
- **Valid post-fix artifacts:**
  - `models/lightgbm_raw_calibrated.pkl` (Phase 04.2.4.1, clean OOT 0.5746) ✅
  - `models/catboost_raw_calibrated.pkl` (Phase 04.2.5.1, clean OOT 0.5699) ✅
- **Code pattern (CANONICAL for all future HPO):** Carve-out must happen in this order:
  ```python
  # Step 1: Load and sort
  X = pd.read_parquet(feature_store_path)
  if _TEMPORAL_SORT_COL in X.columns:
      X = X.sort_values(_TEMPORAL_SORT_COL, na_position='last')
  
  # Step 2: Carve OOT BEFORE Optuna
  oot_idx = int(len(X) * (1 - _TEST_SIZE))
  X_train, X_test = X[:oot_idx], X[oot_idx:]
  y_train, y_test = y[:oot_idx], y[oot_idx:]
  
  # Step 3: Create Optuna study (operates on X_train only)
  study = optuna.create_study(...)
  study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials)
  ```
- **Safe modification:** All new HPO functions must follow this pattern; code review requirement: verify carve-out happens before study creation

### Featuretools Woodwork LogicalType Registration Gotcha

**Issue:** Featuretools ≥1.x silently produces 0 features if entity columns lack explicit Woodwork `LogicalType` assignment.

- **Files:** `src/auto_features.py` (~lines 170–230 in `_build_entity_set()`)
- **Symptom:** `X_featuretools.parquet` at 179073×0 during Phase 04.2.1 indicated empty DFS output despite valid entity set
- **Root cause:** Featuretools DFS iterates over entity columns; columns without explicit `logical_types=` are treated as Unknown type; aggregations not generated for Unknown types
- **Current state (FIXED):** All 7 entity tables now have explicit assignments:
  ```python
  es.add_dataframe(df, dataframe_name="application",
    logical_types={
      "AMT_CREDIT": Double,
      "CODE_GENDER": Categorical,
      "DAYS_EMPLOYED": Integer,
      ...
    })
  ```
- **Test coverage:** 3 unit tests added in Phase 04.2.2-04 verify LogicalType assignments are present before DFS
- **Risk (future phases):** If new secondary tables added to entity set, must include `logical_types=` for all columns or DFS silently fails
- **Safe modification:** 
  1. Add assertion after DFS: `assert feature_matrix.shape[1] > 1, "DFS produced 0 features — check Woodwork LogicalType assignments"`
  2. Log entity set schema (columns + logical_types) before calling `ft.dfs()` for debugging

### LightGBM Verbosity Suppression Requires Two Layers

**Issue:** `verbosity=-1` alone is insufficient to silence LGB output in test suites.

- **Files:** `src/model.py` (~line 1580 in `train_lightgbm_optuna()`)
- **Root cause:** LGB C++ engine (`verbosity=-1`) silences binary logs, but Python-side callback logging still fires
- **Current state (FIXED):** Both suppressions required:
  ```python
  lgb_params = {..., "verbosity": -1, ...}
  cb_early_stopping = lgb.early_stopping(50, verbose=False)
  cb_log_eval = lgb.log_evaluation(period=0)  # suppress callback printing
  ```
- **Risk (future phases):** Removing `lgb.log_evaluation(period=0)` reintroduces test output spam; test logs become unreadable
- **Safe modification:** Never remove callback suppression when refactoring; if debugging LGB, temporarily enable with `period=10` but revert before commit

### LGB Early Stopping Timeout on Mock Data

**Issue:** `_LGB_OBJ_EARLY_STOPPING_ROUNDS=20` (objective) vs `_LGB_EARLY_STOPPING_ROUNDS=50` (refit) creates hang risk on synthetic data with near-perfect separation.

- **Files:** `src/model.py` (~lines 200–205, ~line 1590)
- **Symptom:** Test suite hangs when mock data has Gini > 0.95 (no realistic default signal); LGB trains full `n_estimators=1000` despite early stopping
- **Root cause:** Early stopping patience never triggers if model continues improving indefinitely on perfectly separable synthetic data
- **Current state:** Constants split into tiers (objective=20, refit=50) + pytest timeout=300s caps hangs
- **Risk (future phases):** If test data generation changes (e.g., more synthetic defaults), early stopping may need adjustment
- **Safe modification:** Capture LGB callback results in test fixture; add assertion: `assert num_leaves < 500, f"Overfitting detected: {num_leaves} leaves"`

## Fragile Areas

### Feature Store Absolute Path Safety (HARDENED)

**Issue:** Feature store functions must use absolute paths to prevent test data from silently overwriting production files.

- **Files:** `src/features.py:1228–1231`, `src/features.py:1492–1495`, `src/auto_features.py` (all write functions)
- **Risk:** Relative path `Path("data/processed/X_features.parquet")` causes test-generated mock data (500 rows) to silently overwrite production store (307K rows)
- **Current state (FIXED):** All functions use `_PROJECT_ROOT = Path(__file__).resolve().parents[1]` to construct absolute paths:
  ```python
  output_path = _PROJECT_ROOT / "data" / "processed" / output_filename
  ```
- **Safe modification:** Never accept user-provided relative paths; always validate output shape before writing:
  ```python
  assert X_final.shape[0] >= 100000, f"Output too small: {X_final.shape[0]} rows (production: 307511)"
  ```

### Temporal CV Relies on Single Column (`prev_days_decision_mean`)

**Issue:** If `prev_days_decision_mean` is missing or all-NaN, auto-detection silently falls back to `StratifiedKFold`, losing temporal ordering with no warning.

- **Files:** `src/model.py` (lines 56, 486–488, 1007–1009, 1337–1344)
- **Risk:** Model trained without temporal awareness (overfitting on future information) but appears valid in metrics
- **Current mitigation:** None; auto-detection is silent
- **Safe modification:** 
  1. Always pass `groups` explicitly in production code; never rely on auto-detection
  2. Add log statement: `if _TEMPORAL_SORT_COL in X.columns: logger.info(f"Temporal CV enabled via {_TEMPORAL_SORT_COL}")`
  3. Add test: Verify models trained with and without temporal CV differ in OOT Gini (temporal CV should be slightly lower due to embargo loss)

### CatBoost Categorical Features Must Be Explicitly Declared

**Issue:** If categorical columns are not registered, CatBoost treats them as continuous, losing categorical split optimization.

- **Files:** `src/model.py` (~line 1820 in `train_catboost_optuna()`)
- **Risk:** Model performance degrades; CatBoost's key advantage (automatic categorical handling) is lost
- **Current mitigation:** Function signature is clear (`cat_features` parameter), but callers may forget
- **Safe modification:** Add assertion: `assert len(cat_features) > 0, "CatBoost requires categorical features; call with cat_features list"`

### Calibration Sigmoid Fit on Small Positive Sample Sizes

**Issue:** Platt calibration uses 30% hold-out from training data. If training set is small or highly imbalanced, calibration set may lack sufficient positives for reliable sigmoid fitting.

- **Files:** `src/model.py` (~lines 1495–1595 in `calibrate_model()`)
- **Risk:** Poorly estimated calibration parameters; confidence intervals unreliable
- **Current state:** Home Credit dataset (307K rows, 8% positives) has 24.6K positives total; 30% calibration set = 7.4K positives (sufficient)
- **Safe modification:** Add assertion: `assert sum(y_calib) >= 50, f"Calibration set too small: {sum(y_calib)} positives (required >= 50)"`

## Scaling Limits

### Feature Store Cardinality at Saturation

**Issue:** DFS augmentation from 129 to 290 features yielded negligible lift, suggesting feature signal exhaustion.

- **Files:** `data/processed/X_tree_raw.parquet` (307511×144), `data/processed/X_tree_dfs.parquet` (307511×290)
- **Evidence (Phase 04.2.4 ablation):**
  - LGB OOT Gini: raw+eng (129 features) = 0.5795 vs raw+eng+DFS (290 features) = 0.5767 (−0.0028 delta, regression)
  - XGBoost OOT Gini: X_tree_dfs = 0.5666 (no separate raw comparison, but expected near-zero lift per LGB precedent)
  - DFS added 161 auto-aggregates; gradient dilution confirmed for leaf-wise models
- **Capacity constraint:** Tree models cannot extract signal from 300+ interdependent features on 307K rows; effective dimensionality ~130
- **Impact:** Adding more manual features (Wave 2, Wave 3) risks further dilution without improvement
- **Scaling path:**
  1. Feature selection: Apply IV filtering to reduce to top 50–80 features (already done for WoE pipeline; apply to raw store)
  2. Domain-driven interactions: 5–10 high-precision cross-table terms instead of exhaustive auto-aggregation (e.g. `EXT_SOURCE_MEAN * CREDIT_INCOME_RATIO`, `DELINQUENCY_RATE * LOAN_TERM`)
  3. Ensemble: Combine 3 diverse feature subsets (raw 129, DFS 290, hand-crafted 50) into meta-learner (Phase 04.2.6)
  4. Dimensionality reduction: PCA or autoencoders (future investigation)

### Model Serialization Size

**Issue:** Total model artifacts approaching memory constraints for serverless deployment.

- **Files:** `models/*.pkl` (joblib format)
- **Current:** XGBoost calibrated (~10 MB), LGB calibrated (~8 MB), CatBoost (~9 MB), ensemble (~15 MB), calibrators + supporting models = 50+ MB total
- **Limit:** Cold-start serverless (AWS Lambda 512MB RAM) may fail if total model load + feature pipeline + dependencies exceed available memory
- **Recommendation (future phase):**
  1. Load only primary model (XGBoost calibrated) on startup; defer ensemble to lazy loading
  2. Profile serialization: `du -h models/` to track sizes across phases
  3. Consider ONNX format (40–50% smaller, cross-framework compatible)

## Test Coverage Gaps

### No Integration Test for Feature-to-Model End-to-End

**Issue:** Unit tests exist for individual functions but no test chains full pipeline (load → engineer → HPO → evaluate → save).

- **Files:** `tests/test_model.py`, `tests/test_features.py`, `tests/test_auto_features.py`
- **Gap:**
  - No test verifies: Load raw data → apply feature engineering → run HPO → evaluate → persist calibrated model pickle
  - No test verifies: Load saved model pickle → apply feature transformation → predict on new applicant
- **Risk:** Silent failures in glue code (path handling, column ordering, dtype mismatches) during production inference
- **Scope:** ~50 lines (1 parametrised integration test with small n_trials=2)
- **Recommendation:** Add `test_full_pipeline_end_to_end()` in `tests/test_model.py`:
  1. Load `X_tree_raw.parquet` subset (1000 rows for speed)
  2. Run `train_xgboost_optuna(..., n_trials=2)`
  3. Assert returned model is joblib-serializable
  4. Save and load from pickle
  5. Call `.predict_proba()` on held-out fold
  6. Assert probabilities in [0,1] and Gini > 0.40

### No Fairness Test Suite

**Issue:** No tests verify demographic parity or equalised odds (required for EU AI Act Art. 6 compliance).

- **Files:** `src/explain.py` (stub), `tests/` (no fairness test file)
- **Gap:**
  - Disparate Impact Ratio (DIR) >= 0.80 by protected group (age quintile, gender) not tested
  - Equalised odds difference < 0.05 for sensitive attributes not tested
  - Brier score stratification by demographic not tested
- **Risk:** Model may exhibit hidden bias against protected groups; regulatory audit would fail
- **Scope:** Requires Phase 04.3 implementation first, then ~100 lines fairness test
- **Blocking:** Depends on Phase 04.3 (SHAP + fairness metrics implementation)

### No Temporal CV Validation Test

**Issue:** `_TemporalCV` splitter has no unit test verifying walk-forward logic and embargo correctness.

- **Files:** `src/model.py` (~lines 580–635), `tests/test_model.py` (missing test)
- **Gap:**
  - No test verifies indices never leak from validation to training
  - No test verifies embargo correctly removes recent training samples
  - No test verifies all rows covered exactly once across 10 folds
  - No test validates walk-forward order (earliest fold first, latest fold last)
- **Risk:** Silent train/validation contamination in future refactorings
- **Scope:** ~30 lines (parametrised test on synthetic sorted X)
- **Recommendation:** Add `test_temporal_cv_no_leakage()` checking:
  1. `.split()` output has no overlapping indices between train and validation
  2. Validation indices are monotonically increasing (walk-forward order)
  3. Sum of train+val indices covers all rows exactly once

## Missing Critical Features

### No Ensemble Validation Gate Test

**Issue:** Phase 04.2.6 ensemble gating rule (`ensemble_gini - max(lgb, xgb) >= 0.005`) has no test coverage.

- **Files:** `src/model.py` (~line 2200 for `run_ensemble_workflow()`), `tests/test_model.py` (missing test)
- **Gap:**
  - No test verifies gate passes when delta=0.005 exactly
  - No test verifies gate fails when delta=0.004
  - No test verifies model is persisted IFF gate passes
- **Risk:** Ensemble may pass gate incorrectly or silently fail to persist
- **Scope:** ~20 lines (parametrised gate test with synthetic predictions)

### No Calibration Asymmetry Detection

**Issue:** Platt sigmoid fit on train may be miscalibrated if train/test default rates differ.

- **Files:** `src/model.py` (~lines 2050–2100 in `calibrate_model()`)
- **Gap:**
  - No test verifies calibrated probabilities match observed OOT default rate ±0.01
  - No test verifies Expected Calibration Error (ECE) < 0.02 on OOT
- **Risk:** LGB OOT positive rate (10.4%) vs deployment (8%) will cause off-diagonal calibration curve; Brier score inflation ~0.015–0.020
- **Scope:** ~35 lines (ECE metric + test assertion)

---

*Concerns audit: 2026-04-11*
