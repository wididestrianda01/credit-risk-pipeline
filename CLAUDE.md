# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

End-to-end credit risk scoring pipeline on the Home Credit Default Risk dataset — multi-table feature engineering, LightGBM/XGBoost models, SHAP explainability, class imbalance handling, and a deployed FastAPI + Streamlit scoring app.

**Target metric:** 
- Gini coefficient ≥ 0.70 on held-out test set (Gini = 2 × AUC − 1).
- Kolmogorov-Smirrov (KS) Statistic ≥ 0.40.
- Brier Score < 0.08.
- The reliability diagram (calibration curve) shows visually whether model's predicted probabilities match observed default rates: near-diagonal. 
- Disparate Impact Ratio — **Gender DIR ≥ 0.80 (gate)**. Age DIR is monitored only; AGE_YEARS is excluded from training features so the model has no direct age signal.

Deliverables: FastAPI scoring endpoint + Streamlit dashboard + GitHub repo + LaTeX report.

## Contributing Notes

When extending this pipeline:
- Always do research and gap analysis before making changes — understand the current state before proposing improvements
- Consider resource efficiency when planning new features or expansions
- Refer to the architecture and domain concepts sections below when making design decisions
- Maintain test coverage above 80% for all new features

## Commands

```bash
# Run all tests
pytest tests/ -v

# Run fast tests only (skip slow model suites)
pytest tests/ -v -m "not slow"

# Run a single test by name
pytest tests/test_features.py::test_build_features_returns_dataframe -v

# Build raw tree feature store (no WoE, ~307K rows × 155+ cols)
python -c "from src.features import build_tree_feature_store; build_tree_feature_store('dataset/', 'data/processed/X_tree_raw.parquet')"

# Build DFS feature store (Featuretools — slow, allow 20–40 min)
python -c "from src.auto_features import build_featuretools_feature_store; build_featuretools_feature_store('data/', 'data/processed/X_tree_dfs.parquet')"

# Train XGBoost on raw+DFS features
python scripts/train_xgboost_raw.py

# Execute EDA notebook (strips outputs via nbstripout filter automatically on commit)
cd notebooks && jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=300 \
  01_eda_and_data_quality.ipynb --output 01_eda_and_data_quality.ipynb

# Start prediction API (development)
uvicorn app.api:app --reload

# Launch SHAP dashboard
streamlit run app/streamlit_app.py
```

> `requirements.txt` exists. Update it when adding new dependencies.
> `nbstripout` is installed and registered via `.gitattributes` — notebook outputs are stripped automatically on `git add`. Run `nbstripout --status` to confirm before any `.ipynb` commit.

## Python Stack

```python
# Feature engineering
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Models
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

# Imbalanced data
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline  # replaces sklearn Pipeline

# Hyperparameter optimisation
import optuna  # Bayesian HPO

# DFS auto-feature generation
import featuretools as ft

# Explainability
import shap  # TreeExplainer for LightGBM/XGBoost

# Calibration
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

# Deployment
from fastapi import FastAPI
import streamlit as st
```

> Apply SMOTE **only to training folds**, never to validation or test data.

## Folder Structure

```
credit-risk-pipeline/
├── src/                       # canonical source package (imported as credit_engine via conftest alias)
│   ├── data_loader.py         # joins 7 tables, handles dtypes
│   ├── features.py            # WoE feature engineering (logistic / interpretability pipeline)
│   ├── auto_features.py       # Featuretools DFS auto-aggregation (tree model pipeline)
│   ├── model.py               # XGBoost/LGB/CatBoost + temporal k-fold CV + Optuna HPO
│   ├── explain.py             # SHAP + fairness analysis
│   └── utils.py               # evaluation metrics, plotting
├── app/
│   ├── api.py                 # FastAPI /predict endpoint
│   └── streamlit_app.py       # interactive SHAP dashboard
├── notebooks/                 # EDA and analysis (do not read .ipynb directly — convert to .py first)
├── reports/                   # evaluation JSON + figures/
├── scripts/                   # one-off production runs (train_xgboost_raw.py etc.)
├── data/                   # 7 CSV tables — never read directly, use data_loader.py
└── tests/                     # test_data_loader, test_features, test_model, test_utils, test_auto_features
```

## Architecture

The pipeline is a linear sequence of pure transformations:

```
data/ (7 CSV tables)
    → src/data_loader.py    # multi-table join + dtype enforcement
    → src/features.py       # WoE feature engineering (logistic / interpretability pipeline)
    → src/auto_features.py  # Featuretools DFS auto-aggregation (tree model pipeline)
    → src/model.py          # XGBoost/LGB/CatBoost + temporal k-fold CV + Optuna HPO
    → src/explain.py        # SHAP values + fairness metrics
    → src/utils.py          # evaluation (Gini, KS, Brier, LogLoss)
    → app/api.py            # FastAPI: POST /predict, GET /health
    → app/streamlit_app.py  # interactive SHAP dashboard
```

Each `src/` module follows a **functional, immutable** pattern: functions accept a DataFrame and return a new DataFrame — no in-place mutation.

### Data loading (`src/data_loader.py`)

`load_data(data_dir)` joins all 7 source tables on loan ID:
`application_train/test`, `bureau`, `bureau_balance`, `previous_application`, `pos_cash_balance`, `installments_payments`, `credit_card_balance`.

Never read files in `data/` directly — always go through `data_loader.py`.

**Processed data** lives in `data/processed/`:
- `X_train.parquet` — raw joined feature matrix (~307K rows, pre-engineering)
- `y_train.parquet` — binary TARGET series
- `X_features.parquet` — 307,511 × 68 WoE-encoded features (logistic/interpretability pipeline)
- `X_tree_raw.parquet` — 307,511 × 155+ raw engineered features, **no WoE** (tree model input)
- `X_tree_dfs.parquet` — 307,511 × ~323 cols (raw + Featuretools DFS auto-aggregates) — primary input to `train_xgboost_optuna`

**Path safety:** All feature store write functions use `_PROJECT_ROOT = Path(__file__).resolve().parents[1]` to construct absolute paths. Never use relative paths like `"data/processed/..."` — test runs on mock data will silently overwrite production files.

**Column prefix conventions** (critical for SHAP attribution and feature selection):
| Prefix | Source table |
|--------|-------------|
| `bureau_` | bureau + bureau_balance |
| `prev_` | previous_application |
| `pos_` | POS_CASH_balance |
| `inst_` | installments_payments |
| `cc_` | credit_card_balance |
| (no prefix) | application_train |

**Key EDA findings** (from `notebooks/01_eda_and_data_quality.ipynb`):
- Target imbalance: ~8% defaults, ~92% non-defaults — SMOTE / cost-sensitive weighting required
- `EXT_SOURCE_1/2/3` are the strongest individual predictors (external credit bureau scores); high missingness (~45–55%) is structural, not random — use `np.nanmean` / `np.nanmin` composites
- `DAYS_EMPLOYED = 365243` is an unemployment sentinel (~18% of rows) — clip to 0 before engineering `YEARS_EMPLOYED`
- `AMT_CREDIT` ↔ `AMT_GOODS_PRICE` correlation ~0.98 — can be replaced with `GOODS_CREDIT_RATIO` for feature reduction
- Secondary table absence (0 bureau records, 0 previous applications) is itself predictive — `bureau_cnt`, `prev_cnt` etc. must be kept as features
- `inst_days_past_due_mean` and `inst_payment_ratio_mean` are expected top-ranking secondary aggregates

### Feature engineering (`src/features.py`)

Entry points: `build_features(df)`, `build_feature_store(data_dir, output_path)`, `build_tree_feature_store(data_dir, output_path)`.

**Two pipelines:**
- **WoE pipeline** (`build_feature_store`): 130 raw → 140 engineered → IV-filtered → 68 final WoE-encoded features. Used for logistic regression and interpretability.
- **Raw pipeline** (`build_tree_feature_store`): 155+ raw engineered features with no WoE encoding. Used as input to `src/auto_features.py` and directly to tree models.

**Naming conventions are structural** — they link features back to source tables for SHAP analysis:
- Aggregated features prefixed by source table: `bureau_avg_balance`, `pos_cash_max_instalment`
- Boolean indicator columns use `_flag` suffix: `bureau_overdue_flag`

**WoE gotcha:** WoE pre-binning destroys LGB's continuous split-finding advantage — LGB and XGB receive identical discretised inputs, eliminating LGB's leaf-wise growth benefit. Always feed raw features to tree models.

**Recommended feature engineering** (informed by EDA):
- `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `CREDIT_TERM`, `GOODS_CREDIT_RATIO` — loan affordability ratios
- `AGE_YEARS`, `YEARS_EMPLOYED` (clip `DAYS_EMPLOYED` sentinel 365243 → 0), `EMPLOYED_TO_AGE_RATIO`
- `EXT_SOURCE_MEAN`, `EXT_SOURCE_MIN` — composite of the 3 external scores (nanmean/nanmin)
- `DOCUMENTS_SUBMITTED`, `HIGH_RISK_DOC_MISSING` — document flag aggregations
- WoE binning for categorical features: `CODE_GENDER`, `NAME_EDUCATION_TYPE`, `NAME_INCOME_TYPE`, `ORGANIZATION_TYPE`
- Division-by-zero guard: clip ratios to `[0, inf)`, replace inf with 0, fill NaN with -999 (tree-friendly sentinel)

### Model training (`src/model.py`)

- Primary: XGBoost on raw+DFS features (`train_xgboost_optuna`) + Platt calibration → `xgboost_raw_calibrated.pkl`
- Secondary: LightGBM (`train_lightgbm_optuna`), CatBoost (`train_catboost_optuna`), ensemble (`run_ensemble_workflow`)
- Imbalance: Cost-Sensitive (`scale_pos_weight = n_neg/n_pos`) — winner of 4-strategy benchmark
- Probability calibration: `calibrate_model(model, X_train, y_train, X_test, y_test)` — `FrozenEstimator` + Platt sigmoid via `CalibratedClassifierCV` (sklearn 1.6: `cv="prefit"` is deprecated, use `FrozenEstimator`)

**⚠️ MANDATORY Basel CRE36.54 temporal validation workflow — all three training functions must follow this exact sequence:**
1. Sort full dataset by `_TEMPORAL_SORT_COL` (`prev_days_decision_mean`); NaN rows → seeded random permutation proportional to `_TEST_SIZE`
2. **Carve OOT first** — most-recent `_TEST_SIZE` (20%) rows; freeze and never touch during HPO
3. Optuna HPO on remaining 80% only: each trial uses OOF cross-validation (K-fold temporal CV); OOF Gini is the trial objective
4. Select best hyperparameters by OOF Gini; **retrain on full 80%** (single fit, no CV)
5. Evaluate on frozen OOT → **OOT Gini is the regulatory metric** (Basel III IRB CRE36.54)

Violation = any split that happens inside or after `objective()` — results are inadmissible as regulatory evidence. `train_xgboost_optuna()` was always compliant. `train_lightgbm_optuna()` and `train_catboost_optuna()` were fixed in commit `43c9d89` (2026-04-11).

**Key function signatures (all path-based, post-fix):**
- `train_xgboost_optuna(feature_store_path: str, n_trials=50, groups=None)` — loads parquet from disk, returns `(model, metrics, X_test, y_test, best_params)`
- `train_lightgbm_optuna(feature_store_path: str, n_trials=50, groups=None)` — path-based API; `prev_days_decision_mean` must be present in the parquet
- `train_catboost_optuna(feature_store_path: str, n_trials=50, groups=None)` — path-based API; 7-param HPO, `bootstrap_type="Bayesian"` required for `bagging_temperature`
- `run_ensemble_workflow(X, y)` → return dict includes `gate_result` key (via `_evaluate_ensemble_gate`); persists only if gate='full_pass' or 'accept_best_available'
- `_evaluate_ensemble_gate(oot_gini, best_single_gini) -> str` — D-12 gate: ≥0.65='full_pass', ≥0.58+lift≥0.005='accept_best_available', else='investigate'
- `scripts/run_ensemble.py` — standalone Basel CRE36.54 compliant orchestration; loads best_params from HPO eval JSONs, runs 3-model stacking, emits benchmark CSV + weights JSON + calibrated pkl
- `save_model(model, path)` / `load_model(path)` — joblib format

**`is_unbalance` vs `scale_pos_weight`:** `is_unbalance=True` adjusts gradient weights AND leaf output values (good for Gini/rank metrics). `scale_pos_weight` adjusts only gradients (better for calibrated PD). Both standalone and ensemble LGB use `is_unbalance=True` — using `scale_pos_weight` inside the ensemble caused OOF rank reversal (OOF Gini 0.5114 vs OOT 0.5746) because the logistic meta-learner penalised LGB based on miscalibrated OOF probs. XGB and CatBoost keep `scale_pos_weight` inside the ensemble.

### DFS auto-aggregation (`src/auto_features.py`)

Entry points: `build_featuretools_feature_store(data_dir, output_path)`, `apply_featuretools_feature_store(data_dir, store_path, output_path)`, `evaluate_dfs_features(feature_store_path, n_trials)`.

**Featuretools Woodwork gotcha:** Empty DFS output (0 columns) means entities were registered without Woodwork `LogicalType` annotations. Fix: assign `Categorical`, `Double`, `Integer`, or `BooleanNullable` to each entity column before calling `ft.dfs()`. This is not optional in Featuretools ≥ 1.x.

**Gating:** `evaluate_dfs_features()` trains a quick XGBoost and gates on `delta_gini >= _MIN_GINI_DELTA (0.005)` — only persists `X_tree_dfs.parquet` if DFS features add measurable lift.

**Output:** `data/processed/X_tree_dfs.parquet` — raw engineered features + DFS aggregates, IV-filtered, correlation-deduplicated. ~323 columns.

### Explainability & fairness (`src/explain.py`)

- SHAP `TreeExplainer` for exact attribution on LightGBM/XGBoost
- Global: beeswarm / bar plots; Local: waterfall / force plots per applicant
- Fairness: demographic parity and equalised odds by sensitive group (age, gender)
- **Regulatory scope:** SHAP output must support adverse action notices (GDPR Art. 22) and EU AI Act Art. 6 high-risk AI requirements — not just visualisations
- **Fairness gate (active):** Gender DIR ≥ 0.80 only. Age DIR is computed and reported but is not a gate — AGE_YEARS is already excluded from all model training features.
- **SHAP target model:** `models/catboost_raw_calibrated_v2.pkl` (v2 CatBoost, Gini=0.5814, Gender DIR=0.955 ✓)

### Evaluation metrics (`src/utils.py`)

- `gini_coefficient(y_true, y_prob) -> float` — Gini = 2 × AUC − 1; raises `ValueError` on single-class input
- `ks_statistic(y_true, y_prob) -> tuple[float, float]` — `(ks_value, threshold_at_ks)` via `scipy.stats.ks_2samp`; KS > 0.30 = good, > 0.40 = strong (Basel III)
- `evaluate_model(model, X_test, y_test, model_name) -> dict` — returns `{Model, AUC-ROC, Gini, KS, Brier, BrierSkill, AvgPrecision}`; BrierSkill = 1 − BS / (prevalence × (1 − prevalence)) preferred over raw Brier at 8% imbalance
- `plot_roc_and_pr(model, X_test, y_test, model_name, save_path) -> Figure` — 2-panel ROC + PR figure; PR panel includes dashed prevalence baseline; never calls `plt.show()`
- `roc_curve_plot`, `calibration_plot` — stubs, reserved for future task

## Implementation Status

> Last updated: 2026-04-15

| Component | Status |
|-----------|--------|
| `src/data_loader.py` | **Complete** — `load_data()` 7-table join, `build_training_frame()`, `save_training_frame()`. +17 std/min/max aggregates across all 5 secondary tables. Docstrings corrected (`data/` not `dataset/`). |
| `src/features.py` | **Complete** — WoE pipeline (81 WoE-encoded features) + `build_tree_feature_store()` (155+ raw features). 9 cross-table interactions + 5 EXT_SOURCE polynomial terms. All paths use `_PROJECT_ROOT`. |
| `src/auto_features.py` | **Complete** — Featuretools DFS on 7-table entity set; Woodwork LogicalType annotations required. `build_featuretools_feature_store`, `apply_featuretools_feature_store`, `evaluate_dfs_features`. 15 SK_DPD leaky columns guarded via `_LEAKY_COLUMNS`. |
| `src/model.py` | **Complete** — thin ~438-line facade; re-exports all public+private symbols from 5 sibling modules. |
| `src/model_base.py` | **Complete** — all constants (`_XGB_*`, `_LGB_*`, `_CAT_*`, `_TEMPORAL_SORT_COL`, `_TEST_SIZE`), shared utilities (`_make_cv`, `calibrate_model`, `save_model`, `load_model`, `apply_ext_source_imputer`). Monkeypatch target for tests. |
| `src/model_xgboost.py` | **Complete** — `train_xgboost_optuna`, `train_xgboost_extended_hpo`, Optuna objective. |
| `src/model_lightgbm.py` | **Complete** — `train_lightgbm_optuna`, `train_lightgbm_extended_hpo`, Optuna objective. |
| `src/model_catboost.py` | **Complete** — `train_catboost_optuna`, `train_catboost_extended_hpo`, Optuna objective. |
| `src/model_ensemble.py` | **Complete** — `run_ensemble_workflow`, `train_ensemble_3model`, `_AverageEnsemble`, `_LogisticEnsemble`, `_LogisticEnsemble3`, ensemble gate. `train_ensemble_3model` returns 5-tuple `(model, metrics, X_test, y_test, base_gini_dict)`. |
| `src/utils.py` | **Complete** — `gini_coefficient`, `ks_statistic`, `evaluate_model`, `plot_roc_and_pr`. |
| `src/explain.py` | **Complete** — 6 functions: `compute_shap_values`, `plot_shap_summary`, `plot_shap_local`, `compute_fairness_metrics`, `get_adverse_action_factors`, `compute_shap_stability`. FEATURE_LABELS 171 entries. AdverseActionFactor TypedDict. Per-attribute disparate impact ratio (EU AI Act Art. 6). GDPR Art.22 + Basel CRE36.54 + EU AI Act compliant. |
| `app/api.py` | **Complete** — POST `/predict` (Pydantic request/response), API key auth, calibrated PD + top-5 SHAP adverse action factors, `/health` endpoint with model version; deployment model: CatBoost v2 |
| `app/streamlit_app.py` | **Complete** — Applicant input form (sidebar), PD + risk band display, SHAP waterfall, top-10 feature contribution table, fairness metrics (Gender/Age DIR), adverse action factors; welcome screen on first load |
| `tests/` | **428 tests** — test_data_loader, test_features, test_model, test_utils, test_auto_features, test_streak_evaluation, test_explain (12: 11 unit/integration + test_v3_fairness_gate) |
| `data/processed/X_features.parquet` | 307,511 × 81 WoE-encoded features (includes TARGET + SK_ID_CURR) |
| `data/processed/X_tree_raw.parquet` | 307,511 × 155+ raw features (no WoE) |
| `data/processed/X_tree_dfs.parquet` | 307,511 × ~323 raw+DFS features — CatBoost-DFS input |
| `data/processed/X_lgb_v2.parquet` | 307,511 × v2 store for LGB (Wave 2 temporal trajectory + protected features) |
| `data/processed/X_xgb_v2.parquet` | 307,511 × 145 cols — v2 store for XGB (Wave 2 temporal trajectory + protected) |
| `data/processed/X_cat_v2.parquet` | 307,511 × 149 cols — v2 store for CatBoost standalone (Wave 2 + protected) |
| `models/xgboost_raw_best.pkl` | XGBoost trained on raw+DFS features (uncalibrated) |
| `models/xgboost_raw_calibrated.pkl` | XGBoost on X_xgb_v2; OOT Gini=0.5636 |
| `models/xgboost_woe_calibrated.pkl` | XGBoost on X_features (WoE); OOT Gini=0.5519, AUC=0.7734 — diversity model for ensemble |
| `models/xgboost_best.pkl` | XGBoost on WoE features; Gini=0.5470 (superseded) |
| `models/lightgbm_raw_calibrated.pkl` | ✅ VALID — regenerated 2026-04-12 (4.6MB); OOT Gini=0.5695, KS=0.4346; params in `reports/lgb_raw_X_lgb_v2_is_unbalance_eval.json` |
| `models/catboost_raw_calibrated.pkl` | ✅ VALID — v2 result (X_cat_v2, auto_class_weights=Balanced, SK_ID_CURR sort); OOT Gini=0.5814, AUC=0.7907; backed up as `catboost_raw_calibrated_v2.pkl` |
| `models/catboost_dfs_calibrated.pkl` | CatBoost on X_tree_dfs (DFS); OOT Gini=0.5608, AUC=0.7804 — diversity model for ensemble |
| `models/ensemble_calibrated.pkl` | Phase 04.2.6 ensemble — gate=investigate; OOT Gini=0.5697, Brier=0.0838 (superseded by Phase 04.2.10) |
| `models/lightgbm_best.pkl` | LGB on WoE features; Gini=0.4519 |
| `models/lightgbm_extended.pkl` | LGB extended HPO (superseded/invalid) |
| `models/catboost_extended.pkl` | CatBoost extended HPO (superseded/invalid) |
| `models/ext_source_imputation_lgb.pkl` | LGB imputer for EXT_SOURCE missing values |
| `models/logistic_baseline.pkl` | `Pipeline(StandardScaler → LR)` on WoE features; Gini=0.489 |
| `reports/xgboost_raw_eval.json` | XGBoost raw evaluation results |
| `reports/xgb_woe_eval.json` | XGBoost WoE evaluation: OOT Gini=0.5519, AUC=0.7734, KS=0.4159 |
| `reports/catboost_dfs_eval.json` | CatBoost DFS evaluation: OOT Gini=0.5608, AUC=0.7804, KS=0.4275 |
| `reports/fairness_metrics.csv` | 5 group rows (Gender M/F, Age Young/Mid/Senior); columns: group_name, demographic_parity, tpr, fpr + per-attribute `{metric}_disparate_impact`; Gender DIR ≈ 0.955 (✓ ≥ 0.80); Age DIR ≈ 0.346 (✗ flagged — Young vs Senior gap) |
| `reports/fairness_metrics_v3.csv` | 15 rows (3 models × 5 groups); v3 INVESTIGATE result — XGB Age DIR=0.449, LGB Age DIR best ≈ 0.45+, all Age DIR < 0.80; XGB Gender DIR=0.851 (✓); v3 age improvement vs v2 baseline (0.346) confirmed; proxy features (temporal credit history, income ratios) carry residual age signal |
| `models/xgboost_v3_calibrated.pkl` | XGBoost v3 (X_xgb_v3, 163 cols, no AGE_YEARS/CODE_GENDER); OOT Gini=0.5075; Age DIR=0.449, Gender DIR=0.851 |
| `models/lightgbm_v3_calibrated.pkl` | LightGBM v3 (X_lgb_v3, 163 cols); OOT Gini=0.5610; archived — Phase 04.4 descoped; v2 CatBoost is deployment model |
| `models/catboost_v3_calibrated.pkl` | CatBoost v3 (X_cat_v3, 167 cols); OOT Gini=0.5520; Gender DIR=0.798 (near-miss) |
| `data/processed/X_xgb_v3.parquet` | 307,511 × 163 cols — XGBoost v3 store (AGE_YEARS, EMPLOYED_TO_AGE_RATIO, CNT_CHILDREN, CNT_FAM_MEMBERS removed) |
| `data/processed/X_lgb_v3.parquet` | 307,511 × 163 cols — LightGBM v3 store (same 4 prohibited features removed) |
| `data/processed/X_cat_v3.parquet` | 307,511 × 167 cols — CatBoost v3 store (same 4 prohibited features removed) |
| `requirements.txt` | Exists and updated |

## Domain Concepts

- **PD / LGD / EAD:** Probability of Default · Loss Given Default · Exposure at Default. Expected Loss = PD × LGD × EAD.
- **KS statistic:** Maximum separation between default and non-default CDF curves — standard industry metric alongside Gini.
- **WoE (Weight of Evidence):** `ln(% non-defaults / % defaults)` per bin. IV > 0.3 = strong predictor. Preferred for logistic regression + regulatory IRB approval.
- **Platt scaling:** Logistic regression fit on model output to calibrate probabilities. Required when PD feeds directly into EL calculations.
- **SMOTE:** Apply only to training data. Generates synthetic minority samples by interpolation — can overfit if applied to the full dataset.

## Codebase Gotchas

These are non-obvious patterns not derivable from reading the code.

**Test suite:** `pytest tests/ -m "not slow"` for the fast suite. Expensive model fixtures (`catboost_result`, `benchmark_result` etc.) must be `scope="module"` — function-scoped causes the suite to hang (each fixture invoked once per test × 8 tests = hundreds of CPU minutes). Use `pytest.MonkeyPatch()` directly when monkeypatching in module-scoped fixtures (the `monkeypatch` built-in fixture is function-scoped and cannot be used inside `scope="module"`).

**LGB verbosity:** LGB requires `verbosity=-1` (C++ engine silencer) AND `lgb.log_evaluation(period=0)` (Python callback) — one alone is insufficient for clean library output.

**LGB early stopping two-tier:** `_LGB_OBJ_EARLY_STOPPING_ROUNDS=20` (inside Optuna objective, fast triage) vs `_LGB_EARLY_STOPPING_ROUNDS=50` (final model refit). On small mock data with near-perfect separability, a large patience never triggers, causing the full `n_estimators` to train every fold → suite hangs.

**Notebook path doubling:** Run `jupyter nbconvert` from inside `notebooks/` (not project root) — running from root doubles the path to `notebooks/notebooks/...`.

**`conftest.py` location:** Lives at project root, not inside `tests/` — pytest must find it before either `test_data_loader.py` or `test_features.py` to set up the `credit_engine` alias.

**`docs/` directory:** Untracked, not committed — investigate before staging.

**`_LogisticEnsemble3` sklearn protocol:** `CalibratedClassifierCV(FrozenEstimator(model))` in sklearn 1.7 requires the inner class to satisfy four checks: (1) `_estimator_type = "classifier"` class attribute so `is_classifier()` returns True; (2) `fit` method (no-op is fine); (3) `predict` method; (4) `classes_` attribute. Items 2–4 can be retrofitted via `__getattr__` for already-pickled instances (pickle restores `__dict__` directly — `__init__` never re-runs on load, so adding attributes there won't help old pickles). Item 1 must be a class attribute — `__getattr__` is not called for class-level lookups.

## Financial Mathematics Standards

- Log returns only, never simple returns
- Annualise daily volatility with `sqrt(252)`, weekly `sqrt(52)`, monthly `sqrt(12)`
- Validate numerical stability in any iterative methods (calibration, SHAP kernel)
