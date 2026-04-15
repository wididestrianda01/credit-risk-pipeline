"""
streamlit_app.py
----------------
Interactive Streamlit dashboard for credit risk scoring with SHAP explainability.

Run locally:
    streamlit run app/streamlit_app.py

Features:
1. Real-time applicant scoring with calibrated PD (Probability of Default)
2. SHAP-based feature contribution analysis (waterfall plot)
3. Risk band classification (VERY_LOW → VERY_HIGH)
4. Model performance metrics (Gini, KS, Brier, AUC)
5. Fairness analysis (demographic parity, disparate impact ratios)
6. GDPR Art. 22 adverse action factors (regulatory compliance)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.api import (
    _TRAINING_MEDIANS,
    _get_risk_band,
    ApplicantFeaturesRequest,
    _build_inference_features,
)
from src.explain import FEATURE_LABELS
from src.model_base import load_model

# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Credit Risk Scorer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("💳 Credit Risk Scoring — CatBoost v2 (Gini 0.5814)")

# ---------------------------------------------------------------------------
# Model Loading with Caching
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODEL_PATH = _PROJECT_ROOT / "models" / "catboost_raw_calibrated_v2.pkl"


@st.cache_resource
def load_catboost_model():
    """Load calibrated CatBoost model once per session.

    Returns
    -------
    model or None
        Loaded CatBoost model, or None if loading fails.
    """
    try:
        return load_model(str(_MODEL_PATH))
    except FileNotFoundError:
        st.error(f"❌ Model file not found at {_MODEL_PATH}")
        st.info("Expected: `models/catboost_raw_calibrated_v2.pkl`")
        return None
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        return None


model = load_catboost_model()

# Initialize session state for results persistence
if "result_available" not in st.session_state:
    st.session_state.result_available = False
    st.session_state.pd_score = None
    st.session_state.risk_band = None
    st.session_state.X = None
    st.session_state.form_data = None

# ---------------------------------------------------------------------------
# Sidebar Form with 3 Expanders
# ---------------------------------------------------------------------------

st.sidebar.header("📋 Applicant Input Form")

# Store form values in a dict for later submission
form_data: dict[str, Any] = {}

# ===== Expander 1: Loan Terms =====
with st.sidebar.expander("🏦 Loan Terms", expanded=True):
    amt_credit = st.number_input(
        label=FEATURE_LABELS.get("AMT_CREDIT", "AMT_CREDIT"),
        value=int(_TRAINING_MEDIANS["AMT_CREDIT"]),
        min_value=10000,
        max_value=1000000,
        step=10000,
        help="Requested credit amount in currency units",
    )
    form_data["AMT_CREDIT"] = amt_credit

    amt_annuity = st.number_input(
        label=FEATURE_LABELS.get("AMT_ANNUITY", "AMT_ANNUITY"),
        value=int(_TRAINING_MEDIANS["AMT_ANNUITY"]),
        min_value=100,
        max_value=500000,
        step=500,
        help="Monthly annuity/repayment amount",
    )
    form_data["AMT_ANNUITY"] = amt_annuity

    amt_goods_price = st.number_input(
        label=FEATURE_LABELS.get("AMT_GOODS_PRICE", "AMT_GOODS_PRICE"),
        value=int(_TRAINING_MEDIANS.get("AMT_GOODS_PRICE", 450000)),
        min_value=1000,
        max_value=1000000,
        step=10000,
        help="Goods price (for consumer loans)",
    )
    form_data["AMT_GOODS_PRICE"] = amt_goods_price

    credit_term = st.number_input(
        label=FEATURE_LABELS.get("CREDIT_TERM", "CREDIT_TERM"),
        value=int(_TRAINING_MEDIANS["CREDIT_TERM"]),
        min_value=6,
        max_value=72,
        step=1,
        help="Loan term in months",
    )
    form_data["CREDIT_TERM"] = credit_term

    # ORGANIZATION_TYPE selectbox with hardcoded options
    organization_options = [
        "Business Entity Type 3",
        "Business Entity Type 2",
        "Business Entity Type 1",
        "Government",
        "School",
        "Military",
        "Medicine",
        "Police",
        "Trade: type 1",
        "Trade: type 2",
        "Trade: type 3",
        "Transport: type 1",
        "Transport: type 2",
        "Transport: type 3",
        "Transport: type 4",
        "Electricity",
        "Religion",
        "Industry: type 1",
        "Industry: type 2",
        "Industry: type 3",
        "Industry: type 4",
        "Industry: type 5",
        "Industry: type 6",
        "Industry: type 7",
        "Industry: type 8",
        "Industry: type 9",
        "Industry: type 10",
        "Industry: type 11",
        "Industry: type 12",
        "Industry: type 13",
        "Security Agencies",
        "Hotel",
        "Legal Services",
        "Advertising",
        "Cleaning",
        "Insurance",
        "Telecommunications",
        "Restaurant",
        "Realtor",
        "Housing",
        "Bank",
        "Postal",
        "Agriculture",
    ]
    organization_type = st.selectbox(
        label=FEATURE_LABELS.get("ORGANIZATION_TYPE", "ORGANIZATION_TYPE"),
        options=organization_options,
        index=0,
        help="Employer organization type",
    )
    form_data["ORGANIZATION_TYPE"] = organization_type


# ===== Expander 2: Applicant Profile =====
with st.sidebar.expander("👤 Applicant Profile", expanded=False):
    amt_income = st.number_input(
        label=FEATURE_LABELS.get("AMT_INCOME_TOTAL", "AMT_INCOME_TOTAL"),
        value=int(_TRAINING_MEDIANS["AMT_INCOME_TOTAL"]),
        min_value=0,
        max_value=10000000,
        step=10000,
        help="Annual total income in currency units",
    )
    form_data["AMT_INCOME_TOTAL"] = amt_income

    days_birth = st.number_input(
        label=FEATURE_LABELS.get("DAYS_BIRTH", "DAYS_BIRTH"),
        value=int(_TRAINING_MEDIANS.get("DAYS_BIRTH", -14235)),
        min_value=-100 * 365,
        max_value=-16 * 365,
        step=1,
        help="Days since birth (negative integer; e.g., -14235 ≈ 39 years old)",
    )
    form_data["DAYS_BIRTH"] = days_birth

    days_employed = st.number_input(
        label=FEATURE_LABELS.get("DAYS_EMPLOYED", "DAYS_EMPLOYED"),
        value=int(_TRAINING_MEDIANS["DAYS_EMPLOYED"]),
        min_value=-100 * 365,
        max_value=0,
        step=1,
        help="Days employed (negative = duration; 365243 = unemployed sentinel, converted to 0)",
    )
    form_data["DAYS_EMPLOYED"] = days_employed

    cnt_children = st.number_input(
        label=FEATURE_LABELS.get("CNT_CHILDREN", "CNT_CHILDREN"),
        value=int(_TRAINING_MEDIANS["CNT_CHILDREN"]),
        min_value=0,
        max_value=20,
        step=1,
        help="Number of children",
    )
    form_data["CNT_CHILDREN"] = cnt_children

    cnt_fam_members = st.number_input(
        label=FEATURE_LABELS.get("CNT_FAM_MEMBERS", "CNT_FAM_MEMBERS"),
        value=int(_TRAINING_MEDIANS["CNT_FAM_MEMBERS"]),
        min_value=1,
        max_value=20,
        step=1,
        help="Number of family members",
    )
    form_data["CNT_FAM_MEMBERS"] = cnt_fam_members

    # NAME_EDUCATION_TYPE selectbox
    education_options = [
        "Academic degree",
        "Higher education",
        "Incomplete higher",
        "Lower secondary",
        "Secondary / secondary special",
    ]
    education_type = st.selectbox(
        label=FEATURE_LABELS.get("NAME_EDUCATION_TYPE", "NAME_EDUCATION_TYPE"),
        options=education_options,
        index=4,
        help="Applicant's education level",
    )
    form_data["NAME_EDUCATION_TYPE"] = education_type

    # NAME_INCOME_TYPE selectbox
    income_type_options = [
        "Working",
        "Commercial associate",
        "Pensioner",
        "State servant",
        "Self-employed",
        "Businessman",
        "Student",
        "Unemployed",
    ]
    income_type = st.selectbox(
        label=FEATURE_LABELS.get("NAME_INCOME_TYPE", "NAME_INCOME_TYPE"),
        options=income_type_options,
        index=0,
        help="Applicant's income source type",
    )
    form_data["NAME_INCOME_TYPE"] = income_type

    # OCCUPATION_TYPE selectbox
    occupation_options = [
        "Unknown",
        "Laborers",
        "Core staff",
        "Managers",
        "Accountants",
        "Security staff",
        "Drivers",
        "Sales staff",
        "Cleaning staff",
        "Medicine staff",
        "Secretaries",
        "Skilled technical staff",
        "Cooking staff",
        "Private service staff",
        "High skill tech staff",
        "IT staff",
    ]
    occupation_type = st.selectbox(
        label=FEATURE_LABELS.get("OCCUPATION_TYPE", "OCCUPATION_TYPE"),
        options=occupation_options,
        index=0,
        help="Applicant's occupation type",
    )
    form_data["OCCUPATION_TYPE"] = occupation_type

    region_rating = st.slider(
        label=FEATURE_LABELS.get("REGION_RATING_CLIENT", "REGION_RATING_CLIENT"),
        value=int(_TRAINING_MEDIANS["REGION_RATING_CLIENT"]),
        min_value=1,
        max_value=3,
        help="Region credit risk rating (1=low risk, 3=high risk)",
    )
    form_data["REGION_RATING_CLIENT"] = region_rating


# ===== Expander 3: Credit History =====
with st.sidebar.expander("📊 Credit History", expanded=False):
    ext_source_1 = st.slider(
        label=FEATURE_LABELS.get("EXT_SOURCE_1", "EXT_SOURCE_1"),
        value=_TRAINING_MEDIANS.get("EXT_SOURCE_1", 0.5),
        min_value=-999.0,
        max_value=1.0,
        step=0.01,
        help="External credit score 1 (0–1 scale; ~45% missing)",
    )
    form_data["EXT_SOURCE_1"] = ext_source_1

    ext_source_2 = st.slider(
        label=FEATURE_LABELS.get("EXT_SOURCE_2", "EXT_SOURCE_2"),
        value=_TRAINING_MEDIANS["EXT_SOURCE_2"],
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        help="External credit score 2 (0–1 scale; strongest predictor)",
    )
    form_data["EXT_SOURCE_2"] = ext_source_2

    ext_source_3 = st.slider(
        label=FEATURE_LABELS.get("EXT_SOURCE_3", "EXT_SOURCE_3"),
        value=_TRAINING_MEDIANS["EXT_SOURCE_3"],
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        help="External credit score 3 (0–1 scale)",
    )
    form_data["EXT_SOURCE_3"] = ext_source_3

    bureau_cnt = st.number_input(
        label=FEATURE_LABELS.get("bureau_cnt", "bureau_cnt"),
        value=int(_TRAINING_MEDIANS["bureau_cnt"]),
        min_value=0,
        max_value=50,
        step=1,
        help="Count of bureau records",
    )
    form_data["bureau_cnt"] = bureau_cnt

    prev_cnt = st.number_input(
        label=FEATURE_LABELS.get("prev_cnt", "prev_cnt"),
        value=int(_TRAINING_MEDIANS["prev_cnt"]),
        min_value=0,
        max_value=50,
        step=1,
        help="Count of previous applications",
    )
    form_data["prev_cnt"] = prev_cnt

    pos_cnt = st.number_input(
        label=FEATURE_LABELS.get("pos_cnt", "pos_cnt"),
        value=int(_TRAINING_MEDIANS["pos_cnt"]),
        min_value=0,
        max_value=100,
        step=1,
        help="Count of POS cash accounts",
    )
    form_data["pos_cnt"] = pos_cnt

    inst_cnt = st.number_input(
        label=FEATURE_LABELS.get("inst_cnt", "inst_cnt"),
        value=int(_TRAINING_MEDIANS["inst_cnt"]),
        min_value=0,
        max_value=100,
        step=1,
        help="Count of installment payment records",
    )
    form_data["inst_cnt"] = inst_cnt

    cc_cnt = st.number_input(
        label=FEATURE_LABELS.get("cc_cnt", "cc_cnt"),
        value=int(_TRAINING_MEDIANS["cc_cnt"]),
        min_value=0,
        max_value=50,
        step=1,
        help="Count of credit card accounts",
    )
    form_data["cc_cnt"] = cc_cnt


# ===== Score Applicant Button =====
st.sidebar.divider()
score_button = st.sidebar.button(
    "🎯 Score Applicant",
    key="score_button",
    type="primary",
    use_container_width=True,
    help="Click to compute probability of default and SHAP explanation",
)

# ---------------------------------------------------------------------------
# Three-Tab Layout
# ---------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "🎯 Score Applicant",
    "📊 Model Performance",
    "⚖️ Fairness Metrics",
])

# ===== TAB 1: SCORE APPLICANT =====
with tab1:
    st.header("🎯 Score Applicant")

    # Scoring logic: only runs when button is clicked
    if score_button:
        if model is None:
            st.error("❌ Model failed to load. Cannot score applicants.")
            st.stop()

        try:
            # Build feature vector from form inputs
            request = ApplicantFeaturesRequest(**form_data)
            X = _build_inference_features(request)

            # Predict probability of default
            pd_score: float = float(model.predict_proba(X)[0, 1])
            risk_band = _get_risk_band(pd_score)

            # Store in session_state for persistence across tab switches
            st.session_state.pd_score = pd_score
            st.session_state.risk_band = risk_band
            st.session_state.X = X
            st.session_state.form_data = form_data
            st.session_state.result_available = True

            st.success("✅ Scoring complete. Results below:")

        except Exception as e:
            st.error(f"❌ Scoring failed: {str(e)}")
            st.stop()

    # Display results if available, else show welcome message
    if not st.session_state.get("result_available", False):
        st.info(
            "👋 Welcome to the Credit Risk Scorer. "
            "Fill the form on the left and click '🎯 Score Applicant' "
            "to generate a prediction with SHAP waterfall analysis."
        )
    else:
        # Display PD score and risk band
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Probability of Default",
                value=f"{st.session_state.pd_score * 100:.1f}%",
                help="Calibrated PD from CatBoost v2 model",
            )

        with col2:
            st.metric(
                label="Risk Band",
                value=st.session_state.risk_band,
                help="5-tier IRB risk classification",
            )

        with col3:
            st.metric(
                label="OOT Gini",
                value=0.5814,
                help="Model discrimination power (Basel CRE36.54)",
            )

        # Placeholder sections for Plan 02 (SHAP waterfall, adverse action table, etc.)
        st.divider()

        st.subheader("📊 Feature Contribution Analysis (SHAP Waterfall)")
        st.info("⏳ SHAP waterfall plot will be added in Plan 02. Coming soon...")

        st.subheader("📋 Adverse Action Factors (Top 5)")
        st.info("⏳ Adverse action factors table will be added in Plan 02. Coming soon...")


# ===== TAB 2: MODEL PERFORMANCE =====
with tab2:
    st.header("📊 Model Performance Metrics")
    st.info("⏳ Model performance metrics table will be added in Plan 02. Coming soon...")
    st.info(
        "Expected: OOT Gini, KS statistic, AUC-ROC, Brier score, calibration curve."
    )


# ===== TAB 3: FAIRNESS METRICS =====
with tab3:
    st.header("⚖️ Fairness Analysis")
    st.info("⏳ Fairness metrics and demographic parity analysis will be added in Plan 02. Coming soon...")
    st.info(
        "Expected: Gender disparate impact ratio (gate: ≥ 0.80), "
        "Age DIR (monitored only), equalised odds by group."
    )
