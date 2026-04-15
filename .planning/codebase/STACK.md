# Technology Stack

**Analysis Date:** 2026-04-11

## Languages

**Primary:**
- Python 3.14.3 - Core ML pipeline, feature engineering, model training
  - Type annotations enforced throughout via imports in `src/*.py`

## Runtime

**Environment:**
- Python 3.14.3 (specified via `.python-version` check; see `python3 --version`)

**Package Manager:**
- pip (standard Python package manager)
- Lockfile: `requirements.txt` (pinned version constraints)
- Virtual environment: `.venv/` (gitignored, not committed)

## Frameworks

**Core Data & ML:**
- pandas 2.2+ - Data manipulation, feature store I/O (parquet)
- numpy 1.26+ - Numerical operations, array computing
- scikit-learn 1.4+ - Preprocessing, metrics, StratifiedKFold CV
- imbalanced-learn 0.12+ - SMOTE, RandomUnderSampler, ImbPipeline (apply SMOTE to train folds only)

**Tree Models:**
- lightgbm 4.3+ - Primary gradient booster (Basel CRE36.54 compliant OOT Gini=0.5746)
- xgboost 2.0+ - Secondary/benchmark gradient booster
- catboost 1.2+ - Tertiary gradient booster (Bayesian HPO, OOT Gini=0.5699)

**Feature Engineering:**
- featuretools 1.31+ - Deep Feature Synthesis (DFS) auto-aggregation from 7-table relational schema
  - Requires Woodwork LogicalType annotations (Categorical, Double, Integer, BooleanNullable)
  - Entry point: `build_featuretools_feature_store()` in `src/auto_features.py`

**Hyperparameter Optimization:**
- optuna 3.6+ - Bayesian HPO framework for LGB/XGB/CatBoost
  - Mandatory Basel III CRE36.54 workflow: OOF-based HPO on 80% training data, OOT evaluation on frozen 20%
  - Implemented in `train_xgboost_optuna()`, `train_lightgbm_optuna()`, `train_catboost_optuna()` in `src/model.py`

**Explainability:**
- shap 0.45+ - TreeExplainer for LGB/XGB; SHAP values, beeswarm/waterfall/force plots
  - Deployment: `src/explain.py` (stub for Phase 3.5+)

**Deployment:**
- fastapi 0.111+ - REST API for /predict endpoint
  - Run: `uvicorn app.api:app --reload`
  - Entry point: `app/api.py` (health check only; TODO: integrate model loading + feature pipeline)
- uvicorn 0.29+ (with `[standard]` extras) - ASGI server
- pydantic 2.7+ - Request/response validation, BaseModel schemas
- streamlit 1.35+ - Interactive SHAP dashboard
  - Entry point: `app/streamlit_app.py` (placeholder UI)

**Visualization:**
- matplotlib 3.8+ - ROC/PR curves, calibration plots
- seaborn 0.13+ - Statistical plots (correlation heatmaps, feature distributions)

**Testing:**
- pytest 8.2+ - Test runner
  - Command: `pytest tests/ -v` (all tests) or `pytest tests/ -v -m "not slow"` (fast suite only)
  - Config: `conftest.py` at project root (adds `src/` to `sys.path`, provides module-scoped fixtures)
  - Test marks: `@pytest.mark.slow` for expensive model suites

**Data Download:**
- kaggle 1.6+ - Home Credit Default Risk dataset API client
  - Requires: `~/.kaggle/kaggle.json` (Kaggle API credentials)
  - Usage: Download 7 CSV tables to `data/` directory before feature engineering

**Utility:**
- scipy.stats (via scikit-learn) - `ks_2samp` for Kolmogorov-Smirnov statistic
- joblib - Model serialization (`save_model()`, `load_model()` in `src/model.py`)

## Key Dependencies

**Critical (project won't run without):**
- pandas, numpy - Data frames, arrays
- scikit-learn - CV splits, metrics, pipelines
- lightgbm - Primary model training (Basel CRE36.54 compliant)
- optuna - HPO search engine
- fastapi, pydantic - API framework

**Infrastructure (feature engineering):**
- imbalanced-learn - SMOTE for training-only imbalance handling
- featuretools - 7-table DFS auto-aggregation
- shap - TreeExplainer for SHAP values

**Deployment/Dashboard:**
- uvicorn - HTTP server for FastAPI
- streamlit - Interactive web UI

## Configuration

**Environment:**
- `.env` - Not tracked (gitignored); contains secrets (Kaggle API key)
  - Required: `KAGGLE_USERNAME`, `KAGGLE_KEY` (for dataset downloads)
  - Secret management: Use environment variables only, never hardcode

**Build / Version Control:**
- `.gitignore` - Excludes: `data/*.csv`, `models/*.pkl`, `__pycache__/`, `.venv/`, `.env`, `.planning/`
- `.gitattributes` - Registers `nbstripout` filter for automatic notebook output stripping on commit
- `conftest.py` - Pytest configuration: marks definition, sys.path setup, module-scoped fixtures

**Python Environment:**
- `.venv/lib/python3.14/site-packages/` - Virtual environment (local, not committed)
- `requirements.txt` - Pinned dependency versions; update manually on `pip install <pkg>` additions

## Platform Requirements

**Development:**
- Python 3.14.3+
- pip + virtual environment
- 8+ GB RAM (for Featuretools DFS on 307K rows × 7 tables; allow 20–40 min execution)
- Kaggle API credentials in `~/.kaggle/kaggle.json`

**Production:**
- FastAPI-compatible ASGI server (uvicorn, Gunicorn+uvicorn, etc.)
- Streamlit deployment platform (Streamlit Cloud, Docker, cloud VMs)
- Serialized model artifacts: `models/*.pkl` (joblib format)
- Feature store parquets: `data/processed/*.parquet` (X_tree_raw.parquet, X_tree_dfs.parquet, X_features.parquet)

## Dependency Version Constraints

All pinned in `requirements.txt` with `>=` floor version (no upper cap). Key versions in use:
- python: 3.14.3
- numpy: 1.26+
- pandas: 2.2+
- scikit-learn: 1.4+
- lightgbm: 4.3+
- xgboost: 2.0+
- catboost: 1.2+
- optuna: 3.6+
- featuretools: 1.31+
- shap: 0.45+
- fastapi: 0.111+
- streamlit: 1.35+
- pytest: 8.2+

---

*Stack analysis: 2026-04-11*
