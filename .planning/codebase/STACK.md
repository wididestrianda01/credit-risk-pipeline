# Technology Stack

**Analysis Date:** 2026-04-06

## Languages

**Primary:**
- Python 3.14.3 - All source code, modeling, deployment, and testing

**Secondary:**
- Markdown - Documentation (CLAUDE.md, CLAUDE.md, LaTeX reports)

## Runtime

**Environment:**
- Python 3.14.3 (`/usr/bin/python3.14`)
- Virtual environment: `.venv/` with `include-system-site-packages=false`

**Package Manager:**
- pip (bundled with venv)
- Lockfile: `requirements.txt` (pinned versions)

## Frameworks

**Core Data & ML:**
- NumPy 2.4.4 - Numerical computation
- Pandas 2.3.3 - Tabular data manipulation
- scikit-learn 1.7.2 - Feature engineering, metrics, model utilities (LR baseline, StandardScaler, pipelines)

**Models:**
- LightGBM 4.6.0 - Primary gradient boosting classifier
- XGBoost 3.2.0 - Benchmark gradient boosting classifier
- CatBoost 1.2+ - Alternative gradient boosting with categorical feature support
- Logistic Regression - sklearn.linear_model.LogisticRegression, interpretable baseline for IRB scorecards

**Imbalanced Learning:**
- imbalanced-learn 0.14.1 - SMOTE (oversampling), RandomUnderSampler, ImbPipeline

**Hyperparameter Optimization:**
- Optuna 4.8.0 - Bayesian hyperparameter search (TPE sampler, pruning)

**Explainability:**
- SHAP 0.51.0 - TreeExplainer for LightGBM/XGBoost, global/local feature importance, fairness analysis
- Matplotlib 3.8+ - Static visualization
- Seaborn 0.13+ - Statistical visualization

**Deployment:**
- FastAPI 0.111+ - REST API endpoint for real-time credit scoring
- uvicorn 0.29+ - ASGI server (FastAPI host)
- Streamlit 1.35+ - Interactive web dashboard for SHAP analysis
- Pydantic 2.7+ - Request/response validation (BaseModel)

**Testing:**
- pytest 8.2+ - Test runner and assertions
- pytest-cov - Coverage measurement (run with `--cov=src`)

**Data Serialization:**
- joblib - Model persistence (`*.pkl` files)
- pickle - Python object serialization
- Parquet (via Pandas) - Feature matrix storage (`data/processed/*.parquet`)

**Additional Utilities:**
- kaggle 1.6+ - Kaggle API for dataset download (Home Credit Default Risk dataset)
- scipy.stats.ks_2samp - KS statistic computation (Kolmogorov-Smirnov test)

## Key Dependencies

**Critical (Model Training):**
- scikit-learn 1.7.2 - Core feature engineering and model framework
- LightGBM 4.6.0 - Primary production model
- XGBoost 3.2.0 - Benchmark and ensemble member
- Optuna 4.8.0 - Hyperparameter tuning infrastructure
- pandas 2.3.3 - Data manipulation

**Infrastructure (Scoring & Deployment):**
- SHAP 0.51.0 - Explainability (regulatory requirement for adverse action notices)
- FastAPI 0.111 - Production inference API
- Streamlit 1.35 - Dashboard
- imbalanced-learn 0.14.1 - Imbalance handling during training

## Configuration

**Environment:**
- No `.env` file in use (dataset path `data/` is relative, hardcoded)
- `requirements.txt` contains all dependencies with minimum versions
- Virtual environment setup via `python -m venv .venv`
- Project root path resolution: `Path(__file__).parent.parent` in scripts

**Build/Runtime:**
- `conftest.py` - Pytest configuration + module aliasing (`src` → `credit_engine`)
- `.gitignore` - Excludes venv, models, notebooks, data, .env, IDE files
- `.gitattributes` - nbstripout auto-strips Jupyter outputs on commit

## Platform Requirements

**Development:**
- Python 3.14.3
- ~2.6 GB disk for raw dataset + processed parquets + models
- ~8 GB RAM for model training (XGBoost/LightGBM with 24.6K positive samples)
- 2+ CPU cores for parallel feature engineering and model CV

**Production:**
- FastAPI + uvicorn ASGI server
- Streamlit server (lightweight, <500 MB)
- Model inference: fitted LightGBM/XGBoost `*.pkl` loaded via joblib
- Minimal external dependencies: numpy, pandas, sklearn, lightgbm/xgboost

## Test Framework

- pytest 8.2+
- Run all tests: `pytest tests/ -v`
- Run single test: `pytest tests/test_features.py::test_name -v`
- Coverage: `pytest --cov=src --cov-report=term-missing`

---

*Stack analysis: 2026-04-06*
