# External Integrations

**Analysis Date:** 2026-04-11

## APIs & External Services

**Kaggle Dataset API:**
- Home Credit Default Risk dataset (7 CSV tables)
  - SDK/Client: `kaggle` 1.6+
  - Auth: `~/.kaggle/kaggle.json` (username/key pair)
  - Usage: `from src.data_loader import load_data` → internally fetches tables from `data/` directory (must pre-download via Kaggle CLI)
  - Tables: `application_train.csv`, `application_test.csv`, `bureau.csv`, `bureau_balance.csv`, `previous_application.csv`, `POS_CASH_balance.csv`, `installments_payments.csv`, `credit_card_balance.csv`

**No external model serving APIs currently integrated** - Models are serialized locally and served via FastAPI (see Deployment section)

## Data Storage

**Databases:**
- None configured (project uses flat CSV + parquet files, no SQL database)

**File Storage:**
- Local filesystem only
  - Input data: `data/*.csv` (7 source tables, gitignored)
  - Feature stores: `data/processed/*.parquet` (X_tree_raw.parquet, X_tree_dfs.parquet, X_features.parquet, y_train.parquet)
    - Path safety: All write operations use `_PROJECT_ROOT` anchoring in `src/features.py`, `src/data_loader.py`, `src/auto_features.py`
  - Models: `models/*.pkl` (joblib serialization, gitignored)
    - Example files: `xgboost_raw_calibrated.pkl`, `lightgbm_raw_calibrated.pkl`, `catboost_raw_calibrated.pkl`, `logistic_baseline.pkl`

**Caching:**
- Hyperparameter optimization study databases (local SQLite):
  - `hpo.db` - Optuna study cache (gitignored)
  - `lgb_hpo.db`, `lgb_raw_hpo.db` - LightGBM trial history
  - Purpose: Resume interrupted HPO trials, track best parameters across runs

## Authentication & Identity

**Auth Provider:**
- Custom / None at API level
  - Kaggle: API key in `~/.kaggle/kaggle.json` (required for dataset download)
  - FastAPI endpoint (`app/api.py`): No auth implemented yet (placeholder stage)

**Current Implementation:**
- Kaggle credentials passed via environment or `~/.kaggle/` config
- API auth: Deferred to Phase 5.1 (implementation pending)

## Monitoring & Observability

**Error Tracking:**
- None configured (project-internal only)

**Logs:**
- Python standard logging (`logging` module imported in `src/model.py`, `app/api.py`)
- Console output: LightGBM requires `verbosity=-1` + `lgb.log_evaluation(period=0)` to suppress C++ chatter
- LGB early stopping verbosity silencer: applied in `_LGB_OBJ_EARLY_STOPPING_ROUNDS=20` (objective) and `_LGB_EARLY_STOPPING_ROUNDS=50` (refit)
- Stdout capture for test suites: via `contextlib.redirect_stdout()` in `src/auto_features.py` and featuretools calls

## CI/CD & Deployment

**Hosting:**
- Not yet deployed (development stage)
  - Target: FastAPI on ASGI server (uvicorn, Gunicorn+uvicorn)
  - Streamlit dashboard: Streamlit Cloud or Docker container

**CI Pipeline:**
- Not configured (no GitHub Actions, GitLab CI, etc.)
  - Local testing: `pytest tests/ -v` or `pytest tests/ -v -m "not slow"`
  - Git workflow: Commits to GSD phase branches, validated locally before pushing to main

**Artifact Management:**
- Models stored as joblib `.pkl` files in `models/` (gitignored, not committed)
- Feature stores (parquets) in `data/processed/` (gitignored, not committed)
- Production deployment requires external artifact storage (S3, GCS, Azure Blob) — not yet configured

## Environment Configuration

**Required env vars:**
- `KAGGLE_USERNAME` - Kaggle account username (for dataset downloads)
- `KAGGLE_KEY` - Kaggle API key (secret)

**Secrets location:**
- `~/.kaggle/kaggle.json` (Kaggle credentials, not tracked)
- `.env` file (if used; currently gitignored, not present in repo)

**Development:**
- Load environment: Create `.env` locally with Kaggle credentials, load via `python-dotenv` in any script that calls `load_data()`

## Webhooks & Callbacks

**Incoming:**
- FastAPI `/predict` endpoint (stub): Will receive applicant feature JSON; currently returns `NotImplementedError`
- GET `/health`: Liveness check (implemented in `app/api.py`)

**Outgoing:**
- None configured (no external callbacks)

## Third-Party Integrations

**Test Fixtures & Mocking:**
- pytest 8.2+ provides mocking via `unittest.mock` (standard library)
- No external mocking services (API mocking done in-process via fixtures in `conftest.py`)

**Notebook Filtering (Git Hooks):**
- `nbstripout` registered in `.gitattributes` - Automatically strips notebook outputs on `git add`
  - Check: `nbstripout --status`
  - Install: `nbstripout --install`
  - Purpose: Keep `.ipynb` files diff-friendly and prevent accidental output commits

## Data Flow & Integration Points

**Ingestion:**
1. User downloads 7 CSV files from Kaggle (manual download via `kaggle` CLI or web UI)
2. Place CSVs in `data/` directory
3. `src/data_loader.load_data()` reads and joins all 7 tables → single modelling DataFrame

**Feature Engineering (two pipelines):**
- **WoE pipeline** (`src/features.build_feature_store()`): Raw → engineered → IV-filtered → 68 WoE-encoded features
- **Tree pipeline** (`src/features.build_tree_feature_store()`): Raw → engineered → 155+ raw features (no WoE)
- **DFS pipeline** (`src/auto_features.build_featuretools_feature_store()`): 7-table EntitySet → auto-aggregates → ~323 features (raw + DFS)

**Model Training:**
- Input: Parquet feature store (`X_tree_raw.parquet` or `X_tree_dfs.parquet`)
- Process: `src/model.train_xgboost_optuna()` / `train_lightgbm_optuna()` / `train_catboost_optuna()`
  - Mandatory Basel CRE36.54 workflow: temporal sort → carve OOT → HPO on 80% → OOF Gini evaluation → retrain on 80% → eval on frozen OOT
- Output: Serialized model (`models/*.pkl`), metrics (`reports/*.json`)

**Inference (future Phase 5):**
- Input: Applicant features (JSON via FastAPI POST /predict)
- Process: Load model (`load_model()`), apply feature pipeline, run inference
- Output: PredictionResponse (probability_of_default, risk_band, gini_at_training)

**Explainability:**
- Input: Trained model + test data
- Process: SHAP TreeExplainer (`src/explain.py` - stub)
- Output: SHAP values, beeswarm/waterfall plots, fairness metrics by demographic group

---

*Integration audit: 2026-04-11*
