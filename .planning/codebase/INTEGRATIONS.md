# External Integrations

**Analysis Date:** 2026-04-06

## APIs & External Services

**Kaggle:**
- Kaggle API (kaggle >= 1.6)
- Used for: Dataset download (Home Credit Default Risk Kaggle competition dataset)
- SDK/Client: `kaggle` package
- Authentication: Kaggle API token stored in `~/.kaggle/kaggle.json` (not committed, `*.json` in .gitignore)
- Usage: `src/data_loader.py` references dataset downloaded via Kaggle; raw CSV files live in `data/`

**No active external APIs in production code:**
- FastAPI `/predict` endpoint (in `app/api.py`) is not yet implemented (stub raises `NotImplementedError`)
- Streamlit dashboard (in `app/streamlit_app.py`) is not yet implemented (placeholder)

## Data Storage

**Databases:**
- None — no persistent database integration (SQL, MongoDB, etc.)

**File Storage:**
- Local filesystem only
- Raw data: `data/*.csv` (7 Home Credit tables: application_train/test, bureau, bureau_balance, previous_application, POS_CASH_balance, installments_payments, credit_card_balance)
- Processed data: `data/processed/*.parquet`
  - `X_train.parquet` — 307,511 × ~160 feature matrix (raw joined tables)
  - `y_train.parquet` — 307,511 × 1 binary target (DEFAULT)
  - `X_features.parquet` — WoE-encoded features post-IV filter (~68 columns)
  - `X_raw_features.parquet` — Raw engineered features pre-WoE transform
  - `X_featuretools.parquet` — Features from Deep Feature Synthesis (featuretools auto_features)
- Models: `models/*.pkl` (joblib serialization)
  - `logistic_baseline.pkl`
  - `xgboost_best.pkl`, `xgboost_calibrated.pkl`
  - `lightgbm_best.pkl`
  - `catboost_best.pkl` (future)
  - `woe_mappings.pkl` — WoE bin edge storage for inference
- Reports: `reports/*.json`, `reports/*.csv`, `reports/figures/*.png` (matplotlib output)
- Optuna: `optuna.db` (SQLite, .gitignore'd) — stores HPO trial history

**Caching:**
- None — no Redis, Memcached, or distributed cache
- Optuna uses local SQLite database (`optuna.db`) for HPO trial storage and resumption

## Authentication & Identity

**Auth Provider:**
- None in production code
- Kaggle API uses personal token (`~/.kaggle/kaggle.json`) for dataset download only

## Monitoring & Observability

**Error Tracking:**
- None — no Sentry, DataDog, or error monitoring service
- All errors logged via `warnings.warn()` in source code (e.g., `src/model.py`, `src/features.py`)

**Logs:**
- Stdout/stderr only
- pytest captures test output in `.coverage` (coverage artifact)
- Model training scripts write to:
  - `reports/lgb_hyperparameter_heatmap.csv` — LGB Optuna trial metrics
  - `reports/lgb_*.log` — LGB training logs (from scripts)
  - `reports/train_raw_eval.log` — raw feature evaluation logs

## CI/CD & Deployment

**Hosting:**
- Local development: `uvicorn app.api:app --reload` (dev server)
- Local deployment: Streamlit via `streamlit run app/streamlit_app.py`
- No cloud hosting configured (AWS, GCP, Heroku, etc.)

**CI Pipeline:**
- None — no GitHub Actions, GitLab CI, CircleCI, or other CI/CD
- pytest runs locally: `pytest tests/ -v`

## Environment Configuration

**Required env vars:**
- None hardcoded
- Kaggle API token: `~/.kaggle/kaggle.json` (user home directory, not in project)
- Optional: `KAGGLE_CONFIG_DIR` (environment variable) — defaults to `~/.kaggle/`

**Secrets location:**
- `.env` file (if needed) listed in `.gitignore` — use environment variables for sensitive config
- No `.env` file currently in use
- Kaggle credentials live outside project tree (`~/.kaggle/kaggle.json`)

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

## Optional Integrations (Not Yet Implemented)

**featuretools (auto_features.py):**
- Status: Integrated but not in production pipeline
- Package: `featuretools` (not in `requirements.txt`, imported via try/except)
- Usage: `src/auto_features.py` contains `build_featuretools_feature_store()` — runs Deep Feature Synthesis on entity set
- Output: `data/processed/X_featuretools.parquet`
- Note: Requires manual invocation via `scripts/build_featuretools_store.py`; not called in standard training flow

**SHAP (explainability.py):**
- Status: Installed (0.51.0) but implementation incomplete
- Package: `shap` (in requirements.txt)
- Usage: `src/explain.py` stub — `compute_shap_values()` and `fairness_report()` raise `NotImplementedError`
- Planned: TreeExplainer for LightGBM/XGBoost, SHAP waterfall/force plots in Streamlit dashboard

## Data Exchange Formats

**Input:**
- CSV (7 Home Credit tables, ~2.6 GB raw)
- Environment variables (Kaggle API token)

**Output:**
- Parquet (intermediate feature matrices)
- Pickle (joblib models `*.pkl`)
- JSON (evaluation results, calibration metrics, Optuna trial logs)
- CSV (benchmark results, trial metrics)
- PNG (matplotlib figures: ROC, PR, calibration curves, correlation heatmaps)

## Dependency on External Data

**Home Credit Default Risk Dataset:**
- Source: Kaggle competition (https://kaggle.com/c/home-credit-default-risk)
- Size: ~2.6 GB (7 tables, 307K rows, 100+ raw features)
- Download: Kaggle API (`kaggle competitions download -c home-credit-default-risk -p data/`)
- License: Kaggle dataset license (check terms before redistribution)
- Freshness: Static dataset (2019), not updated

---

*Integration audit: 2026-04-06*
