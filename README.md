# Credit Risk Scoring Pipeline

End-to-end machine learning pipeline for real-time credit risk assessment on the [Home Credit Default Risk dataset](https://www.kaggle.com/competitions/home-credit-default-risk). Combines LightGBM, XGBoost, and CatBoost models with SHAP explainability, fairness auditing, and production-grade API + interactive dashboard.

## Project Overview

This pipeline ingests 7 related credit tables (~307K applicants), engineers 150+ features across multi-table aggregation, trains Basel III IRB-compliant models with Platt calibration, and deploys a **calibrated probability of default (PD)** via FastAPI with GDPR Art. 22 adverse action explanations.

**Key outcomes:**
- **Probability of default (PD)** calibrated for Expected Loss (EL = PD × LGD × EAD) calculations
- **Production model:** CatBoost v2, OOT Gini = 0.5814, Gender disparate impact ratio = 0.955 ✓
- **Explainability:** SHAP TreeExplainer + human-readable adverse action factors
- **Fairness:** GDPR Art. 22 + EU AI Act Art. 6 (high-risk AI) compliant
- **API & Dashboard:** FastAPI /predict endpoint + Streamlit interactive scoring interface

---

## Dataset

### Home Credit Default Risk (Kaggle)

Source: [Home Credit Default Risk — Kaggle Competition](https://www.kaggle.com/competitions/home-credit-default-risk/data)

**7 tables (307K rows × 195 features):**
- `application_train.csv` — main applicant table (demographics, income, credit details)
- `bureau.csv` / `bureau_balance.csv` — bureau credit history + monthly status
- `previous_application.csv` — prior loan applications
- `POS_CASH_balance.csv` — point-of-sale cash balance history
- `installments_payments.csv` — historical payment records
- `credit_card_balance.csv` — credit card balance history

### Download & Setup

1. **Install Kaggle CLI:**
   ```bash
   pip install kaggle
   ```

2. **Get API credentials:** Download `kaggle.json` from [Kaggle Settings → API](https://www.kaggle.com/settings/account) and place in `~/.kaggle/kaggle.json`.

3. **Download dataset:**
   ```bash
   mkdir -p data/raw
   cd data/raw
   kaggle competitions download -c home-credit-default-risk
   unzip home-credit-default-risk.zip
   ```

4. **Expected structure:**
   ```
   data/raw/
   ├── application_train.csv
   ├── application_test.csv
   ├── bureau.csv
   ├── bureau_balance.csv
   ├── previous_application.csv
   ├── POS_CASH_balance.csv
   ├── installments_payments.csv
   ├── credit_card_balance.csv
   └── HomeCredit_columns_description.csv
   ```

---

## Environment Setup

### Requirements

- **Python:** 3.10+
- **OS:** Linux/macOS/Windows

### Installation

1. **Clone repository:**
   ```bash
   git clone <repo-url>
   cd credit-risk-pipeline
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or: venv\Scripts\activate on Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Install nbstripout (for notebook commits):**
   ```bash
   nbstripout --install
   nbstripout --status  # verify installation
   ```

5. **Set up API key (optional, for /predict endpoint):**
   ```bash
   cp .env.example .env
   # Edit .env and set CREDIT_RISK_API_KEY=your-secret-key
   ```

---

## Pipeline Stages

### 1. Data Loading & Preprocessing

Load and join 7 tables on loan ID. Handles missing values, data type enforcement, and temporal sorting for Basel III CRE36.54 compliant validation.

```bash
# Automatic on first import
python -c "from src.data_loader import load_data; df = load_data('data/raw')"
```

**Output:** `data/processed/X_train.parquet` (307,511 × 195 raw features)

---

### 2. Feature Engineering

Two pipelines:

#### **WoE Pipeline** (interpretability, logistic regression)
```bash
python -c "from src.features import build_feature_store; \
  build_feature_store('data/raw', 'data/processed/X_features.parquet')"
```
**Output:** 307,511 × 81 WoE-encoded features

#### **Raw Pipeline** (tree models)
```bash
python -c "from src.features import build_tree_feature_store; \
  build_tree_feature_store('data/raw', 'data/processed/X_tree_raw.parquet')"
```
**Output:** 307,511 × 155+ raw engineered features (no WoE)

---

### 3. Auto-Feature Generation (DFS)

Featuretools deep feature synthesis for hierarchical aggregation:

```bash
python -c "from src.auto_features import build_featuretools_feature_store; \
  build_featuretools_feature_store('data/raw', 'data/processed/X_tree_dfs.parquet')"
```
**Runtime:** 20–40 minutes on standard hardware.

**Output:** 307,511 × ~323 raw + DFS-generated features

---

### 4. Model Training

All three models use **Basel CRE36.54 temporal validation:**
1. Sort by temporal column; carve 20% OOT (out-of-time) for final evaluation
2. Optuna HPO on remaining 80% with k-fold temporal CV
3. Retrain best params on full 80%, then evaluate on frozen OOT
4. Platt sigmoid calibration for PD output

#### **XGBoost**
```bash
python scripts/train_xgboost_raw.py  # uses X_xgb_v2
```
OOT Gini: **0.5636**, KS: 0.4183

#### **LightGBM**
```bash
python scripts/train_lightgbm_raw.py  # uses X_lgb_v2
```
OOT Gini: **0.5695**, KS: 0.4346

#### **CatBoost** (Production model)
```bash
python scripts/train_catboost_raw.py  # uses X_cat_v2
```
OOT Gini: **0.5814**, KS: 0.4147 ⭐

---

### 5. Explainability & Fairness

SHAP TreeExplainer + demographic parity analysis:

```bash
python -c "from src.explain import compute_shap_values, compute_fairness_metrics; \
  # Generates figures/ and fairness_metrics.csv"
```

**Outputs:**
- `reports/figures/shap_beeswarm.png` — global feature importance
- `reports/figures/shap_waterfall_0.png` — per-applicant SHAP contributions
- `reports/fairness_metrics.csv` — disparate impact ratios by demographic group

---

### 6. Run All Tests

```bash
# Full test suite (~12 minutes)
pytest tests/ -v

# Fast tests only (skip expensive model suites)
pytest tests/ -v -m "not slow"

# Test coverage
pytest tests/ --cov=src --cov-report=term-missing
```

**Coverage:** 574 tests, 80%+ coverage maintained

---

### 7. FastAPI Scoring Endpoint

```bash
# Start development server
uvicorn app.api:app --reload

# Or for production
gunicorn app.api:app --workers 4
```

**Endpoints:**

- **`POST /predict`** — Score a single applicant
  - Requires `X-API-Key` header
  - Input: raw applicant fields (all optional, filled with training medians)
  - Output: calibrated PD, risk band, top-5 SHAP adverse action factors

- **`GET /health`** — Liveness check (no auth)
  - Returns: status, model version, uptime

**Example request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "AMT_CREDIT": 500000,
    "AMT_INCOME_TOTAL": 150000,
    "AMT_ANNUITY": 25000,
    "EXT_SOURCE_2": 0.55,
    "DAYS_BIRTH": -14235,
    "DAYS_EMPLOYED": -1825
  }'
```

**Example response:**
```json
{
  "probability_of_default": 0.18,
  "risk_band": "MEDIUM",
  "adverse_action_factors": [
    {
      "rank": 1,
      "feature_name": "EXT_SOURCE_2",
      "human_label": "External Credit Score 2 (bureau, 0–1 scale)",
      "shap_value": 0.31,
      "direction": "increases_risk"
    }
  ],
  "model_version": "catboost_v2",
  "gini_at_training": 0.5814
}
```

**OpenAPI docs:** Visit http://localhost:8000/docs (Swagger UI)

---

### 8. Streamlit Interactive Dashboard

```bash
streamlit run app/streamlit_app.py
```

Opens interactive web interface at http://localhost:8501

**Features:**
- Applicant input form (sidebar)
- Real-time PD scoring + 5-tier risk band
- SHAP waterfall plot (per-applicant feature contributions)
- Top-5 adverse action factors (GDPR Art. 22 compliance)
- Model performance tab (Gini, AUC-ROC, KS, Brier Score)
- Fairness metrics tab (gender/age disparate impact ratios)
- Welcome screen on first load

---

## Deployment — Streamlit Community Cloud

Deploy the interactive dashboard to Streamlit Community Cloud for free, production-grade hosting.

### Prerequisites

1. **GitHub account** — Push your repository to GitHub (public or private)
2. **Hugging Face Hub account** — Host trained model artifacts (free tier available)
3. **Streamlit Community Cloud account** — Sign up at [share.streamlit.io](https://share.streamlit.io)

### Step-by-Step Deployment

#### 1. Prepare the Repository

Ensure these files are committed to GitHub:
- `app/streamlit_app.py` — main dashboard application
- `.streamlit/config.toml` — theme and logger configuration
- `requirements.txt` — Python dependencies (pinned versions)
- `models/catboost_raw_calibrated_v2.pkl` — **model file** (check Hugging Face option below if size > 100 MB)

#### 2. Host Model on Hugging Face Hub (Optional but Recommended)

If your model pickle is large (>100 MB), upload to Hugging Face to avoid GitHub LFS:

```bash
# Install Hugging Face CLI
pip install huggingface-hub

# Login
huggingface-cli login

# Create a repo (on huggingface.co/new)
# Example: https://huggingface.co/your-username/credit-risk-models

# Push model file
huggingface-cli repo create credit-risk-models --private
git clone https://huggingface.co/your-username/credit-risk-models
cp models/catboost_raw_calibrated_v2.pkl credit-risk-models/
cd credit-risk-models && git add . && git commit -m "Add model" && git push
```

Streamlit will fetch the model at runtime using `HF_REPO_ID` and optional `HF_TOKEN`.

#### 3. Connect Repository to Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
2. Click **"New app"** → select your repository
3. Set **Main file path** to `app/streamlit_app.py`
4. Select **Python version:** 3.10 or 3.11
5. Click **"Deploy"** — Streamlit will install dependencies from `requirements.txt`

#### 4. Set secrets in the Community Cloud UI

After deployment, add secrets to Streamlit's encrypted secret store:

1. Go to your app's settings (gear icon in top-right corner)
2. Select **"Secrets"** from the left sidebar
3. Add the following key-value pairs (exact format):

```toml
CREDIT_RISK_API_KEY = "your-production-api-key"
HF_REPO_ID = "your-hf-username/credit-risk-models"
HF_TOKEN = "hf_your_api_token_here"  # Optional if repo is public
```

**Notes:**
- `CREDIT_RISK_API_KEY` — used by the dashboard to authenticate API calls (if using FastAPI backend)
- `HF_REPO_ID` — repository ID on Hugging Face (format: `username/repo-name`)
- `HF_TOKEN` — Hugging Face API token (optional if model repo is public; generate at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens))

#### 5. Local Development with `.streamlit/secrets.toml`

For local testing before deployment:

1. Create `.streamlit/secrets.toml` (gitignored):
   ```toml
   CREDIT_RISK_API_KEY = "your-test-key-here"
   HF_REPO_ID = "your-hf-username/credit-risk-models"
   # HF_TOKEN = "hf_your_api_token_here"
   ```

2. Streamlit reads from this file when running locally:
   ```bash
   streamlit run app/streamlit_app.py
   ```

3. Never commit `.streamlit/secrets.toml` — it's in `.gitignore`

### Troubleshooting

#### "Model file not found" during app startup

**Solution:** Ensure `HF_REPO_ID` is set correctly and the model exists on Hugging Face. If model is in your local repo:

```python
# In app/streamlit_app.py, model loading logic:
import os
from huggingface_hub import hf_hub_download

hf_repo_id = os.environ.get("HF_REPO_ID")
if hf_repo_id:
    model_path = hf_hub_download(repo_id=hf_repo_id, filename="catboost_raw_calibrated_v2.pkl")
else:
    model_path = "models/catboost_raw_calibrated_v2.pkl"
```

#### "API key missing" or "Unauthorized"

**Solution:** Verify `CREDIT_RISK_API_KEY` is set in **Settings → Secrets** (not in code). Restart the app:

1. Go to your app settings (gear icon)
2. Click the restart button or redeploy from GitHub

#### Slow startup time

**Solution:** Hugging Face download on first load (~1–5 min for large models). Consider:
- Caching the downloaded model in Streamlit's cache: `@st.cache_resource`
- Monitoring logs: click **"Manage app"** → **"Logs"** to see download progress

#### App crashes after deployment

**Solution:** Check the logs (**Manage app** → **Logs**) for:
- Missing dependencies (ensure all imports are in `requirements.txt`)
- Model loading errors (verify Hugging Face repo ID and token)
- Environment variable typos (check **Settings → Secrets** for exact key names)

---

## Model Performance

### Primary Model: CatBoost v2

| Metric | Value | Status |
|--------|-------|--------|
| **OOT Gini** | 0.5814 | ✓ |
| **AUC-ROC** | 0.7907 | ✓ |
| **KS Statistic** | 0.4147 | ✓ (Basel III: ≥0.40) |
| **Brier Score** | 0.0662 | ✓ (<0.08) |
| **Feature Store** | X_cat_v2 (149 cols) | v2: protected features |
| **Calibration** | Platt sigmoid | Basel III EL-ready |

### Model Comparison

| Model | OOT Gini | KS | AUC | Notes |
|-------|----------|-----|-----|-------|
| **CatBoost v2** ⭐ | 0.5814 | 0.4147 | 0.7907 | **Production** — deployed in API & dashboard |
| LightGBM v2 | 0.5695 | 0.4346 | — | High rank stability; ensemble diversity |
| XGBoost v2 | 0.5636 | 0.4183 | 0.7776 | Fast training; good calibration |
| CatBoost-DFS | 0.5608 | 0.4275 | 0.7804 | Diversity model (DFS features) |
| XGBoost-WoE | 0.5519 | 0.4159 | 0.7734 | Interpretability model (WoE) |
| Logistic Baseline† | 0.4681 | 0.3486 | 0.7340 | IRB interpretable benchmark (WoE, 80 features) |

**Notes:**
- OOT Gini = 2 × AUC − 1 (Basel III IRB discrimination metric)
- KS = Kolmogorov-Smirnov statistic (default/non-default separation)
- All models Basel CRE36.54 temporal-validation compliant (sort by `prev_days_decision_mean`, most-recent 20% as OOT)
- Ensemble attempted but fell short of 0.58 gate (best: 0.5681)
- CatBoost v2 is single best model and deployment choice
- CatBoost v2 outperforms logistic baseline by **24.2% on Gini** (same OOT split)
- † Logistic Baseline: `Pipeline(StandardScaler → LR, class_weight='balanced')`; Brier=0.1495 (uncalibrated — prevalence shift from 8% train to 6.2% OOT inflates raw score)

---

## Fairness & Regulatory Compliance

### Fairness Metrics

| Metric | Gender | Age | Status |
|--------|--------|-----|--------|
| **Disparate Impact Ratio (DIR)** | 0.955 | monitored | ✓ Gender ≥ 0.80 (gate pass) |
| **Interpretation** | Female/Male approval ratio | Young/Senior approval ratio | Age excluded from training |

**Key details:**
- **Gender DIR = 0.955:** Female applicants 95.5% as likely to be approved as males → gate passes ≥ 0.80 threshold
- **Age DIR:** Monitored only; `AGE_YEARS` is excluded from all model features to prevent direct age discrimination (GDPR Art. 5 fairness principle)
- **Proxy features:** Temporal credit history and income ratios carry residual age signal; accepted as lawful business necessity

### Regulatory Compliance

| Regulation | Requirement | Status |
|------------|-------------|--------|
| **GDPR Art. 22** | Right to explanation for automated decisions | ✓ Top-5 SHAP adverse action factors in API response |
| **EU AI Act Art. 6** | High-risk AI assessment + risk mitigation | ✓ Fairness audit + disparate impact ratios + human-readable explanations |
| **Basel III CRE36.54** | Temporal validation for PD estimation | ✓ OOT Gini = 0.5814 on frozen hold-out set |

**Adverse action notices:**
When applicant is denied or given worse terms, the API provides top-5 contributing factors in human-readable format for regulatory notice requirements.

---

## Project Structure

```
credit-risk-pipeline/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── runtime.txt                        # Python version pin (Streamlit Cloud)
├── conftest.py                        # Root-level pytest fixtures + credit_engine alias
├── .gitignore                         # Standard Python + data safety
├── .env.example                       # API key template
├── .streamlit/
│   └── config.toml                    # Theme + logger configuration (Streamlit Cloud)
│
├── data/
│   ├── raw/                           # Kaggle CSVs (7 tables, not committed)
│   └── processed/                     # Feature stores (parquets, not committed)
│       ├── X_train.parquet            # Raw joined (307K × 195)
│       ├── X_features.parquet         # WoE-encoded (307K × 81)
│       ├── X_tree_raw.parquet         # Raw engineered (307K × 155+)
│       ├── X_lgb_v2.parquet           # LGB feature store (307K × 145)
│       ├── X_xgb_v2.parquet           # XGB feature store (307K × 145)
│       ├── X_cat_v2.parquet           # CatBoost feature store (307K × 149)
│       └── X_tree_dfs.parquet         # Raw + DFS features (307K × ~323)
│
├── models/                            # Trained model artifacts (not committed)
│   ├── catboost_raw_calibrated.pkl    # CatBoost v2 (production) — OOT Gini 0.5814
│   ├── catboost_raw_calibrated_v2.pkl # Frozen backup (same)
│   ├── lightgbm_raw_calibrated.pkl    # LightGBM v2 — OOT Gini 0.5695
│   ├── xgboost_raw_calibrated.pkl     # XGBoost v2 — OOT Gini 0.5636
│   ├── ensemble_calibrated.pkl        # 3-model stacking (not primary)
│   └── optuna_studies.db              # HPO study history
│
├── reports/                           # Evaluation outputs
│   ├── figures/
│   │   ├── shap_beeswarm.png          # SHAP global feature importance
│   │   ├── shap_bar.png               # SHAP mean |value| ranking
│   │   ├── shap_waterfall_0.png       # Per-applicant SHAP breakdown
│   │   └── shap_force_0.html          # Interactive SHAP force plot
│   ├── fairness_metrics.csv           # Disparate impact by demographic group
│   ├── model_benchmark.csv            # 5-model comparison
│   ├── xgboost_raw_eval.json          # XGB v2 best params + metrics
│   ├── lgb_raw_X_lgb_v2_eval.json     # LGB v2 best params + metrics
│   ├── catboost_raw_eval.json         # CatBoost v2 best params + metrics
│   ├── catboost_v2_best_metrics.json  # Canonical CatBoost v2 metrics (read by dashboard)
│   └── lr_woe_oot_eval.json           # Logistic baseline OOT evaluation
│
├── src/                               # Core library (canonical package)
│   ├── data_loader.py                 # 7-table join + dtype enforcement
│   ├── features.py                    # WoE + raw feature engineering
│   ├── auto_features.py               # Featuretools DFS aggregation
│   ├── model.py                       # Thin facade; re-exports submodules
│   ├── model_base.py                  # Constants + shared utilities
│   ├── model_xgboost.py               # XGBoost + Optuna HPO
│   ├── model_lightgbm.py              # LightGBM + Optuna HPO
│   ├── model_catboost.py              # CatBoost + Optuna HPO
│   ├── model_ensemble.py              # 3-model stacking + gate
│   ├── explain.py                     # SHAP + fairness analysis
│   └── utils.py                       # Metrics (Gini, KS, Brier) + plots
│
├── app/                               # Deployment (API + dashboard)
│   ├── api.py                         # FastAPI /predict endpoint
│   └── streamlit_app.py               # Interactive dashboard
│
├── latex/                             # Research report
│   ├── main.tex                       # Full LaTeX source
│   ├── main.pdf                       # Compiled PDF report
│   └── references.bib                 # BibTeX bibliography
│
├── notebooks/                         # Analysis & EDA (outputs stripped on commit)
│   ├── 01_eda_and_data_quality.ipynb  # Dataset exploration + missing patterns
│   ├── 02_feature_engineering.ipynb   # Feature interaction analysis
│   ├── 03_modeling_and_evaluation.ipynb # Model training walkthrough
│   └── 04_explainability_and_fairness.ipynb # SHAP + fairness audit
│
├── scripts/                           # One-off production runs
│   ├── train_xgboost_raw.py           # XGBoost training script
│   ├── train_lightgbm_raw.py          # LightGBM training script
│   ├── train_catboost_raw.py          # CatBoost training script
│   └── run_ensemble.py                # 3-model ensemble orchestration
│
└── tests/                             # Unit + integration tests (574 tests)
    ├── test_data_loader.py            # Data loading validation
    ├── test_features.py               # Feature engineering tests
    ├── test_model.py                  # Model training tests
    ├── test_utils.py                  # Metrics + plotting tests
    ├── test_auto_features.py          # DFS aggregation tests
    ├── test_explain.py                # SHAP + fairness tests
    ├── test_api.py                    # FastAPI endpoint smoke tests
    └── test_streamlit_startup.py      # Streamlit app startup smoke tests
```

---

## Development Workflow

### Running Tests

```bash
# All tests
pytest tests/ -v

# Fast tests only (skip slow model suites)
pytest tests/ -v -m "not slow"

# Single test file
pytest tests/test_features.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ app/ tests/

# Type checking
mypy src/ app/ --ignore-missing-imports

# Linting
ruff check src/ app/ tests/
```

### Committing Notebooks

Notebook outputs are stripped automatically via `.gitattributes` + `nbstripout`:

```bash
# Verify nbstripout is installed and registered
nbstripout --status

# Run and save notebook (outputs will be stripped on git add)
jupyter nbconvert --to notebook --execute notebooks/01_eda_and_data_quality.ipynb

# Stage and commit
git add notebooks/01_eda_and_data_quality.ipynb
git commit -m "docs: update EDA notebook with new findings"
```

---

## Troubleshooting

### "Model file not found" error

Ensure model pickle is in `models/catboost_raw_calibrated.pkl` (or equivalent). If missing, retrain:

```bash
python scripts/train_catboost_raw.py
```

### "API key missing" on /predict

Set CREDIT_RISK_API_KEY environment variable:

```bash
export CREDIT_RISK_API_KEY=your-secret-key
# OR
echo "CREDIT_RISK_API_KEY=your-secret-key" > .env
```

### Slow feature engineering

DFS (Featuretools) can take 20–40 minutes. Run once and reuse:

```bash
python -c "from src.auto_features import build_featuretools_feature_store; \
  build_featuretools_feature_store('data/raw', 'data/processed/X_tree_dfs.parquet')"
```

### Test suite hangs

Ensure fixtures are `scope="module"` (not `scope="function"`). See `conftest.py` for details.

---

## References

### Papers & Standards

- **Basel III IRB CRE36.54:** Regulatory PD discrimination metric (Gini coefficient)
- **GDPR Art. 22:** Right to explanation for automated decision-making
- **EU AI Act Art. 6:** High-risk AI systems — fairness and explainability requirements
- **Platt Scaling:** Sigmoid calibration for probability outputs (Platt, 1999)
- **SHAP:** SHapley Additive exPlanations for model interpretability (Lundberg & Lee, 2017)

### Key Articles

- Gini coefficient and rank-ordering: [Naeem & Parashar, 2007](https://www.google.com/search?q=gini+coefficient+credit+scoring)
- WoE & Information Value: [Naeem Siddiqi, Credit Risk Scorecards]

---

## License

This project is provided under the **MIT License**. See [LICENSE](./LICENSE) for details.

You are free to use, modify, and distribute this code for educational, research, and commercial purposes, provided you include the original license and copyright notice.

---

## Support

For questions or issues:

- Open an issue on GitHub
- Review the source code in `src/` for architecture details
- Review notebooks for usage examples

---

**Last updated:** 2026-04-16

**Current deployment model:** CatBoost v2 (OOT Gini = 0.5814, Gender DIR = 0.955 ✓)
