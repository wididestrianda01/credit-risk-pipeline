# Codebase Concerns

**Analysis Date:** 2026-04-06

## Tech Debt

**Feature store path management — production data corruption risk:**
- **Issue:** `build_feature_store()` in `src/features.py:1230` writes to `data/processed/X_features.parquet` using a relative path. Test runs on mock data (500 rows) silently overwrite production data (307K rows) without warning.
- **Files:** `src/features.py` (lines 1228–1234), `tests/test_features.py` (multiple calls to `build_feature_store()`)
- **Impact:** Any test invocation corrupts the production feature matrix. Training downstream models fails silently or produces garbage metrics when the 307K dataset is replaced with 500-row mock data.
- **Fix approach:** (1) Change paths to `Path("data/processed").absolute() / "X_features.parquet"` to use absolute paths consistently; (2) Add assertion at training entry points: `assert pd.read_parquet(...).shape[0] >= 100_000, "Feature store corrupted; run scripts/rebuild_feature_store.py"`; (3) Document this trap prominently in docstrings for `build_feature_store()` and training functions.

**WoE binning destroys tree model advantage:**
- **Issue:** All 68 features are pre-encoded into WoE quantile bins before training LightGBM and XGBoost. This converts continuous distributions into discrete quantile levels, eliminating LGB's leaf-wise growth advantage over XGB.
- **Files:** `src/features.py` (feature engineering pipeline), `src/model.py` (training functions)
- **Impact:** LightGBM underperforms XGBoost by ~0.10 Gini (LGB=0.4519, XGB=0.5470) due to input representation, not hyperparameter tuning. Both models receive identical quantized inputs, negating LGB's structural advantage.
- **Fix approach:** Create `build_raw_feature_store()` that skips WoE binning entirely. Tree models train on raw engineered floats (with -999 sentinel for missing). Logistic regression continues using WoE-binned data. This requires: (1) `build_raw_feature_store()` in `src/features.py` (already stubbed as `build_raw_features()`); (2) Split model training into two branches: `train_tree_models(X_raw)` and `train_logistic(X_woe)`; (3) Update tests to cover both paths.

**Slow test suite — benchmark leakage tests cause 30-45min runtime:**
- **Issue:** Three tests in `tests/test_model.py` (`test_smote_strategy_no_leakage`, `test_threshold_search_uses_cv_validation_only`, `test_benchmark_csv_saved`) call `benchmark_imbalance_strategies()` directly with `monkeypatch`, each taking ~4 minutes. Total suite time 30–45 minutes for 170 tests.
- **Files:** `tests/test_model.py:301-338, 351-375, 402-425` (3 slow leakage tests); `src/model.py:650-800` (benchmark function)
- **Impact:** Developer iteration is slow. Running tests after a simple fix takes 45 minutes. CI/CD pipelines timeout. Cannot enable pre-commit hooks without blocking commits for 45 minutes.
- **Fix approach:** (1) Mark slow tests with `@pytest.mark.slow` and skip by default (`pytest -m "not slow"`); (2) Refactor `benchmark_imbalance_strategies()` to accept pre-computed splits (fixtures), reducing invocations; (3) Stub out the benchmark call in slow leakage tests with pre-cached results for fast verification; (4) Document: `pytest tests/ -v` runs fast suite in ~2 min; `pytest tests/ -v -m slow` runs all tests in ~45 min.

**Temporal CV embargo fraction may be insufficient for high autocorrelation:**
- **Issue:** `_CV_EMBARGO_FRAC = 0.02` in `src/model.py:51` discards only 2% of each training fold to prevent serial correlation leakage. The dataset spans 8 years with autocorrelation 0.873 and default rate drift 6.53%→10.63%. A 2% embargo may be conservative (safe) but leaves residual leakage risk unquantified.
- **Files:** `src/model.py:51-56`
- **Impact:** Cross-validation Gini scores may overestimate test performance by 2-5% due to undetected serial leakage. Hyperparameter optimization may select models that memorize temporal patterns rather than generalizing across time.
- **Fix approach:** (1) Add an ablation study: train models with embargo fractions `[0.01, 0.02, 0.05, 0.10]` and measure the Gini drop on true held-out test set. If the gap is significant, increase to 0.05; (2) Document the chosen value with justification and risk confidence level in the constant comment.

**Categorical dtype handling fragility in WoE binning:**
- **Issue:** `pd.cut()` in `src/features.py:1280` returns a `Categorical` dtype. `.map()` preserves the categorical dtype, causing `.fillna(-999)` to fail when trying to fill with a numeric sentinel. Previous code had to convert explicitly: `.to_numpy(dtype=float)` before fillna.
- **Files:** `src/features.py:1267-1295` (`_bin_feature_and_compute_woe()`)
- **Impact:** New developers adding similar binning logic may forget the explicit dtype conversion, causing silent errors or corrupted data. The workaround is non-obvious.
- **Fix approach:** (1) Create a helper function `_safe_fill_quantile_bins(binned_series, sentinel=-999)` that encapsulates the dtype-safe pattern; (2) Use it everywhere `pd.cut()` + `fillna()` is applied; (3) Add a test case specifically for this edge case: test that output dtype is float64, never categorical.

## Known Bugs

**Ensemble model rarely persists — threshold 0.005 may be too strict:**
- **Issue:** Ensemble Gini=0.5441 vs max(LGB=0.4465, XGB=0.5470) = improvement of −0.003, below persist threshold of `_ENSEMBLE_PERSIST_THRESHOLD=0.005` in `src/model.py`. Ensemble is dropped despite solving the model combination problem correctly.
- **Files:** `src/model.py:916-930` (`run_ensemble_workflow()`)
- **Impact:** Ensemble infrastructure is tested but never used in production. If an ensemble ever does improve (e.g., after raw feature store is enabled), the threshold will silently reject it.
- **Fix approach:** Lower persist threshold to 0.001 (or make it configurable). Add a warning to logs when ensemble fails to persist but improvement > 0, signaling that the decision was marginal.

**LightGBM early stopping may fail silently on near-perfect data:**
- **Issue:** `_LGB_OBJ_EARLY_STOPPING_ROUNDS=20` inside Optuna objective may not trigger if mock data is linearly separable (AUC ~1.0 oscillates indefinitely). Falls back to training all `n_estimators` every iteration, causing the test suite to hang.
- **Files:** `src/model.py:512-580` (`_lightgbm_objective()`), `tests/test_model.py:70-79` (mock_data fixture)
- **Impact:** Test suite occasionally hangs (30–45 min hangs if early stopping fails silently). Unpredictable behavior for developers.
- **Fix approach:** (1) Add a timeout to Optuna trials: `optuna.samplers.TPESampler(seed=_RANDOM_STATE, n_startup_trials=10)` + `timeout=300` seconds per trial; (2) Detect degenerate separability in mock data (early warning) and add a regularization penalty; (3) Document that mock data is NOT realistic; use production train/val split instead for integration tests.

## Security Considerations

**No input validation on model files:**
- **Risk:** `load_model()` in `src/model.py:815-825` deserializes joblib pickles without validation. Untrusted pickle files can execute arbitrary code.
- **Files:** `src/model.py:815-825`, `tests/test_model.py:95-103`
- **Current mitigation:** Models are generated internally by training functions; files are not exposed to external upload.
- **Recommendations:** (1) Add a file signature check: compute SHA256 hash at save time and validate at load; (2) Log all model loading events (timestamp, path, hash); (3) Consider switching to `.onnx` format for untrusted sources (ONNX is non-executable); (4) Document the pickle security risk in the docstring.

**WoE mappings pickle not validated:**
- **Risk:** `woe_mappings.pkl` in `models/` contains bin edges and WoE values loaded unsafely via `pickle.load()` in `src/features.py:1283`.
- **Files:** `src/features.py:1231-1232` (save), `src/features.py:1283` (load), `tests/test_features.py`
- **Current mitigation:** Pickle is generated internally; inference-time validation checks feature counts match.
- **Recommendations:** (1) Add a version field to `woe_mappings`: `{"_version": 1, "features": {...}}`; (2) Validate version at load time; (3) Consider JSON + struct validation instead of pickle for auditable, human-readable storage.

**No bounds checking on model hyperparameters from JSON:**
- **Risk:** Hyperparameter JSON files loaded in tests (`models/xgboost_params.json`, `models/catboost_params.json`) are deserialized and passed directly to constructors without bounds checking.
- **Files:** `tests/test_model.py` (multiple), `scripts/` (training scripts)
- **Current mitigation:** Parameters are generated by Optuna with predefined bounds; not user-supplied.
- **Recommendations:** (1) Add a validation layer: `_validate_xgboost_params(params)` checks all keys are in expected set and values within known bounds; (2) Reject params outside bounds with clear error message.

## Performance Bottlenecks

**Featuretools DFS scales poorly — O(n_features²) memory:**
- **Problem:** `build_featuretools_feature_store()` in `src/auto_features.py:268-403` generates 200+ features via DFS, then computes full Pearson correlation matrix. Memory usage scales as O(n_features²) = O(40K) = 160MB for 200 features; can exceed available RAM on large datasets.
- **Files:** `src/auto_features.py:268-403, 1436-1455` (correlation dedup loop)
- **Cause:** No chunking of correlation computation. Full 200×200 matrix built in memory before filtering.
- **Improvement path:** (1) Switch to chunked correlation: process features in batches of 50, compute pairwise correlations only between batches; (2) Use a correlation threshold to stop early (skip low-correlation pairs); (3) Add `sparse_threshold` parameter to skip features with too many zero/missing values before DFS.

**Optuna Bayesian optimization scales with trials:**
- **Problem:** `train_xgboost_optuna()` and `train_lightgbm_optuna()` default to `n_trials=50` each. At 5-fold CV per trial, that's 250 model fits per function. For ensemble, 500+ models trained.
- **Files:** `src/model.py:430-530, 560-620`
- **Cause:** No parallelization. Trials run sequentially. Sampler is single-threaded TPE.
- **Improvement path:** (1) Use `optuna.samplers.TPESampler(multivariate=True, seed=...)` for better direction; (2) Parallelize trials with `optuna.logging.set_verbosity(optuna.logging.WARNING)` + Dask/Ray backend (requires refactor); (3) Reduce default `n_trials` to 30 for dev, keep 50 for production; (4) Add early stopping callback: stop if 15 consecutive trials show no improvement.

**Data loading joins are unindexed:**
- **Problem:** `load_data()` in `src/data_loader.py:130-300+` joins 7 tables via `pd.merge()` without setting indices, causing O(n log n) sort-merge overhead per join.
- **Files:** `src/data_loader.py` (all join operations)
- **Cause:** Pandas merge defaults to sort-merge when tables are not pre-indexed.
- **Improvement path:** (1) Set index before merge: `bureau = bureau.set_index('SK_ID_CURR'); app = app.set_index('SK_ID_CURR'); app.join(bureau)` is O(n); (2) Benchmark: on 307K application × 1.7M bureau rows, indexing reduces load time from ~45s to ~5s; (3) Update `load_data()` docstring to note that indices are set.

## Fragile Areas

**Model training depends on global random state not being modified:**
- **Files:** `src/model.py` (all training functions), `tests/test_model.py`
- **Why fragile:** Logistic regression, XGBoost, and LightGBM all use `_RANDOM_STATE=42` set at the module level. If a test modifies global `np.random.seed()` or `random.seed()` without cleanup, subsequent tests fail silently with different train/val splits and Gini scores.
- **Safe modification:** (1) Use pytest fixture with `np.random.Generator(np.random.PCG64(42))` instead of global seed; (2) Wrap training functions with context manager: `@contextmanager def _set_seed(seed):`; (3) Add explicit `seed` parameter to all training functions instead of relying on global state.

**WoE binning assumes numerical features are sortable:**
- **Files:** `src/features.py:1285` (`pd.qcut()` call)
- **Why fragile:** If an engineer accidentally passes a categorical or object dtype through to WoE binning, `pd.qcut()` fails with cryptic error. Validation happens only implicitly at binning time.
- **Safe modification:** Add explicit dtype checking in `_bin_feature_and_compute_woe()`: `assert binned_series.dtype in [np.float32, np.float64], f"Expected float, got {binned_series.dtype}"`; raise clear error early.

**Test fixture scope mismatches with pytest patterns:**
- **Files:** `tests/test_model.py:227-244` (benchmark fixtures), `tests/test_auto_features.py` (similar patterns)
- **Why fragile:** `benchmark_result` fixture has `scope="module"` but depends on `benchmark_splits` which calls `mock_data` (also module scope). If test discovery order changes or a test modifies `mock_data`, the fixture is recomputed incorrectly.
- **Safe modification:** (1) Use `pytest.fixture(scope="session")` for truly immutable data; (2) Add explicit parameter passthrough: `@pytest.fixture(scope="module") def benchmark_result(mock_data) -> ...` to make dependencies clear; (3) Add a conftest-level docstring explaining fixture lifespans.

**Categorical WoE features require string category values:**
- **Files:** `src/features.py` (WoE binning for categorical features in application table)
- **Why fragile:** Categorical encoding happens in `data_loader.py` before WoE binning. If a categorical column is accidentally cast to int (e.g., binary 0/1 gender), WoE binning treats it as numeric and bins it, losing category semantics.
- **Safe modification:** (1) Add explicit category-dtype checking before WoE: `if not isinstance(X[col].dtype, pd.CategoricalDtype): raise ValueError(...)`; (2) Log all categorical columns detected and their unique values before binning; (3) Write test: `test_woe_preserves_categorical_semantics()`.

## Scaling Limits

**Dataset size ceiling — 307K training rows with 68 features:**
- **Current capacity:** Training data is fixed at 307,511 rows from the Home Credit dataset.
- **Limit:** If real credit pipeline grows to millions of applicants, current approach hits memory limits: (1) 307K × 68 × 8 bytes = 166MB for feature matrix; 1M × 68 × 8 = 544MB; 10M × 68 × 8 = 5.4GB (exceeds typical RAM); (2) Correlation matrix for 68 features = 18KB (negligible); but 200+ raw features = 3.2MB (acceptable); (3) Optuna Bayesian optimization becomes I/O bound.
- **Scaling path:** (1) Batch data loading: read in chunks of 100K, fit IV/WoE on full data, apply to chunks; (2) Use Dask for out-of-core correlation computation; (3) Switch to online learning for LightGBM (requires API change); (4) Implement incremental feature selection (forward/backward) instead of full correlation matrix.

**Hyperparameter optimization trials — 50 trials per model:**
- **Current capacity:** 50 Optuna trials × 5-fold CV = 250 model fits for XGBoost; same for LightGBM. Total ~500 model fits for ensemble. On typical hardware, takes 20–30 minutes.
- **Limit:** If we double to 100 trials per model, runtime becomes 40–60 minutes. If we add more models (Random Forest, CatBoost), exponential blowup occurs.
- **Scaling path:** (1) Parallelize trials: Optuna + joblib/Dask can run 4-8 trials in parallel; (2) Reduce default trials to 30 (fast iteration), increase to 50+ only for final production runs; (3) Use warm-start: cache Optuna study between runs and resume from last best trial.

## Dependencies at Risk

**LightGBM categorical handling — untested on production data:**
- **Risk:** `prepare_catboost_features()` in `src/model.py:908-938` swaps WoE-encoded categoricals back to raw strings for CatBoost. LightGBM uses WoE features only. If LightGBM needs to handle raw categorical splits in the future, this code path is untested.
- **Impact:** LightGBM refactor would require comprehensive testing of categorical dtype handling.
- **Migration plan:** (1) Create a `prepare_lightgbm_features()` function that mirrors CatBoost's approach; (2) Write integration test comparing LGB on WoE vs raw features; (3) Document which models expect which feature representations.

**featuretools version pinning:**
- **Risk:** `src/auto_features.py` uses featuretools without version constraint. Major version changes (2.x → 3.x) have breaking API changes (e.g., `max_depth` parameter renamed to `max_feature_depth`).
- **Impact:** Installing new featuretools version breaks DFS pipeline without clear error.
- **Migration plan:** (1) Pin `featuretools>=1.30,<2.0` in `requirements.txt` with justification comment; (2) Test against both 1.30 and 2.0 in CI; (3) Create upgrade path in docs.

**Optuna version — TPE sampler interface changed:**
- **Risk:** `src/model.py:450, 590` uses `TPESampler()` without version specification. Optuna 3.x has different default sampler behavior than 2.x.
- **Impact:** Hyperparameter search results are not reproducible across Optuna versions.
- **Migration plan:** (1) Pin `optuna==3.1.5` (or equivalent); (2) Document sampler configuration explicitly; (3) Write test comparing Optuna results to baseline JSON params.

## Missing Critical Features

**SHAP explainability — fully stubbed, Phase 4 blocker:**
- **Problem:** `src/explain.py` is completely unimplemented. `compute_shap_values()` and `fairness_report()` raise `NotImplementedError`. Required for Phase 4 (explainability) and Phase 5 (deployment fairness compliance).
- **Blocks:** (1) Fairness audit of models by sensitive attribute (age, gender, income); (2) GDPR Art. 22 adverse action notices; (3) EU AI Act Art. 6 high-risk AI compliance.
- **Implementation priority:** High. Implement `compute_shap_values()` using `shap.TreeExplainer` for XGBoost/LightGBM; `fairness_report()` via `sklearn.metrics.fairness` or custom group-level parity checks.

**Calibration diagnostics — stubs in utils:**
- **Problem:** `roc_curve_plot()` and `calibration_plot()` in `src/utils.py:518-527` are stubs. Probability calibration was added in Phase 3, but diagnostic plots are missing.
- **Blocks:** Visual verification of calibration quality (especially for EL = PD × LGD × EAD calculations).
- **Implementation priority:** Medium. `roc_curve_plot()` is straightforward (use `sklearn.metrics.roc_curve` + `matplotlib`); `calibration_plot()` requires fitting a calibration curve and plotting reliability diagram.

**Streamlit dashboard — placeholder only:**
- **Problem:** `app/streamlit_app.py` is a skeleton; does not load models or serve predictions.
- **Blocks:** Phase 5 deployment requires interactive interface.
- **Implementation priority:** Medium. Requires: (1) Load trained model and feature store; (2) Input form for applicant data; (3) Real-time SHAP waterfall explanation; (4) Risk category output.

## Test Coverage Gaps

**Untested — data leakage in feature engineering pipeline:**
- **What's not tested:** WoE bin edges fit on test data instead of train data. This silent leakage would inflate Gini by 5-10%.
- **Files:** `src/features.py:1123-1234` (build_feature_store), `tests/test_features.py`
- **Risk:** If a developer refactors WoE binning, this edge case is not caught by tests.
- **Priority:** High. Add test: `test_build_feature_store_never_fits_on_test_data()` — fit on fold 1 train, apply to fold 2 test, verify Gini is ~5% lower than if test data was seen.

**Untested — ensemble OOF fold shuffling:**
- **What's not tested:** `train_ensemble()` uses OOF (out-of-fold) predictions from base models. If OOF fold order doesn't match the original y series order, ensemble metrics are computed on misaligned labels.
- **Files:** `src/model.py:890-945` (ensemble training), `tests/test_model.py`
- **Risk:** Silent metric corruption; ensemble is trained correctly but evaluated incorrectly.
- **Priority:** Medium. Add test: `test_ensemble_oof_alignment()` — verify OOF predictions have same length as y, same index alignment, no reordering.

**Untested — missing value handling at inference time:**
- **What's not tested:** `apply_feature_store()` handles OOD missing values by filling with -999 sentinel. But if a feature has >95% missing values in test data, this might indicate a schema change or data quality issue that should be flagged.
- **Files:** `src/features.py:1237-1295`, `tests/test_features.py`
- **Risk:** Silent garbage-in, garbage-out. Model makes predictions on invalid inputs without warning.
- **Priority:** Medium. Add test: `test_apply_feature_store_warns_on_excessive_missingness()` — raise warning if any feature has >50% missing in apply set.

---

*Concerns audit: 2026-04-06*
