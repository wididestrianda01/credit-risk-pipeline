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
import sys
from pathlib import Path
from typing import Any

# Streamlit adds app/ to sys.path, not the project root.
# Insert project root so `from app.api import ...` and `from src.* import ...` resolve.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from app.api import (
    _TRAINING_MEDIANS,
    _get_risk_band,
    ApplicantFeaturesRequest,
    _build_inference_features,
)
from src.explain import FEATURE_LABELS, get_adverse_action_factors
from src.model_base import load_model

# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Credit Risk Scoring",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("💳 Credit Risk Scoring")

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
# Helper Functions
# ---------------------------------------------------------------------------


_RISK_BAND_DISPLAY = {
    "VERY_LOW": "🟢 Very Low Risk",
    "LOW": "🟡 Low Risk",
    "MEDIUM": "🟠 Medium Risk",
    "HIGH": "🔴 High Risk",
    "VERY_HIGH": "⛔ Very High Risk",
}

# Human-readable labels for ORGANIZATION_TYPE (model receives original value via format_func)
_ORG_TYPE_LABELS: dict[str, str] = {
    "Business Entity Type 1": "Private Company — Small (Type 1)",
    "Business Entity Type 2": "Private Company — Medium (Type 2)",
    "Business Entity Type 3": "Private Company — Large (Type 3, most common)",
    "Government": "Government / Public Sector",
    "School": "School / Educational Institution",
    "Military": "Military / Defence",
    "Medicine": "Healthcare / Medical Facility",
    "Police": "Police / Law Enforcement",
    "Trade: type 1": "Retail & Trade — Specialty (Type 1)",
    "Trade: type 2": "Retail & Trade — Type 2",
    "Trade: type 3": "Retail & Trade — General (Type 3)",
    "Transport: type 1": "Transport & Logistics — Type 1",
    "Transport: type 2": "Transport & Logistics — Type 2",
    "Transport: type 3": "Transport & Logistics — Type 3",
    "Transport: type 4": "Transport & Logistics — Type 4",
    "Electricity": "Electricity / Utilities",
    "Religion": "Religious Organisation",
    "Industry: type 1": "Manufacturing / Industry — Type 1",
    "Industry: type 2": "Manufacturing / Industry — Type 2",
    "Industry: type 3": "Manufacturing / Industry — Type 3",
    "Industry: type 4": "Manufacturing / Industry — Type 4",
    "Industry: type 5": "Manufacturing / Industry — Type 5",
    "Industry: type 6": "Manufacturing / Industry — Type 6",
    "Industry: type 7": "Manufacturing / Industry — Type 7",
    "Industry: type 8": "Manufacturing / Industry — Type 8",
    "Industry: type 9": "Manufacturing / Industry — Type 9",
    "Industry: type 10": "Manufacturing / Industry — Type 10",
    "Industry: type 11": "Manufacturing / Industry — Type 11",
    "Industry: type 12": "Manufacturing / Industry — Type 12",
    "Industry: type 13": "Manufacturing / Industry — Type 13",
    "Security Agencies": "Security / Guard Services",
    "Hotel": "Hotel / Hospitality",
    "Legal Services": "Legal Services / Law Firm",
    "Advertising": "Advertising / Marketing",
    "Cleaning": "Cleaning / Facilities Services",
    "Insurance": "Insurance",
    "Telecommunications": "Telecommunications / Telecom",
    "Restaurant": "Restaurant / Food Service",
    "Realtor": "Real Estate / Property Agency",
    "Housing": "Housing / Social Housing",
    "Bank": "Bank / Financial Institution",
    "Postal": "Postal / Courier Service",
    "Agriculture": "Agriculture / Farming",
}


def get_shap_waterfall_figure(shap_explanation: Any, idx: int, X: pd.DataFrame) -> plt.Figure:
    """
    Render SHAP waterfall for a single sample as matplotlib Figure.

    Parameters
    ----------
    shap_explanation : shap.Explanation
        SHAP values from compute_shap_values()
    idx : int
        Row index (0-based) to visualize
    X : pd.DataFrame
        Feature matrix (not used in rendering, but included per signature)

    Returns
    -------
    matplotlib.figure.Figure
        Figure object ready for st.pyplot()

    Notes
    -----
    Adapted from src/explain.py::plot_shap_local() — returns Figure instead of saving.
    Do NOT modify src/explain.py; keep source module report-compatible.
    """
    import shap

    # Validate index
    if idx < 0 or idx >= shap_explanation.shape[0]:
        raise IndexError(f"idx {idx} out of range [0, {shap_explanation.shape[0]})")

    # Render waterfall plot and return figure
    shap.plots.waterfall(shap_explanation[idx], show=False)
    fig = plt.gcf()
    return fig

# ---------------------------------------------------------------------------
# Sidebar Form with 3 Expanders
# ---------------------------------------------------------------------------

st.sidebar.header("📋 Applicant Input Form")

# Store form values in a dict for later submission.
# Age/employment years are collected in human units and converted to
# the model's internal representation (negative day counts) before scoring.
form_data: dict[str, Any] = {}

# ===== Expander 1: Loan Terms =====
with st.sidebar.expander("🏦 Loan Terms", expanded=True):
    amt_credit = st.number_input(
        label="Loan Amount Requested",
        value=int(_TRAINING_MEDIANS["AMT_CREDIT"]),
        min_value=10_000,
        max_value=1_000_000,
        step=10_000,
        help=(
            "Total loan amount the applicant is requesting (in currency units). "
            "Example: enter 500000 for a 500,000 loan. "
            "Typical range: 45,000–1,000,000."
        ),
    )
    form_data["AMT_CREDIT"] = amt_credit

    amt_annuity = st.number_input(
        label="Monthly Repayment Amount",
        value=int(_TRAINING_MEDIANS["AMT_ANNUITY"]),
        min_value=100,
        max_value=500_000,
        step=500,
        help=(
            "Fixed monthly instalment the applicant will pay to repay the loan. "
            "Roughly: Loan Amount ÷ Loan Term in months. "
            "Example: a 500,000 loan over 24 months ≈ 20,833/month."
        ),
    )
    form_data["AMT_ANNUITY"] = amt_annuity

    amt_goods_price = st.number_input(
        label="Purchase Price of Goods",
        value=int(_TRAINING_MEDIANS.get("AMT_GOODS_PRICE", 450_000)),
        min_value=1_000,
        max_value=1_000_000,
        step=10_000,
        help=(
            "For consumer/POS loans: the market price of the item being financed "
            "(e.g. a car worth 400,000 or a refrigerator worth 30,000). "
            "For cash loans with no specific purchase, enter an amount close to the loan amount."
        ),
    )
    form_data["AMT_GOODS_PRICE"] = amt_goods_price

    credit_term = st.number_input(
        label="Loan Term (months)",
        value=int(_TRAINING_MEDIANS["CREDIT_TERM"]),
        min_value=6,
        max_value=72,
        step=1,
        help=(
            "Number of months to repay the loan. "
            "Example: 24 = 2-year loan, 60 = 5-year loan. "
            "Longer terms lower monthly payments but increase total interest paid."
        ),
    )
    form_data["CREDIT_TERM"] = credit_term

    # ORGANIZATION_TYPE — selectbox shows human labels; format_func returns model-compatible value
    organization_type = st.selectbox(
        label="Employer Type / Sector",
        options=list(_ORG_TYPE_LABELS.keys()),
        index=0,
        format_func=lambda x: _ORG_TYPE_LABELS.get(x, x),
        help=(
            "The industry sector of the applicant's current employer. "
            "'Private Company Type 1/2/3' are private-sector companies of increasing size — "
            "Type 3 is the largest and most common category in this dataset. "
            "Select the closest match: e.g. a hospital → Healthcare / Medical Facility; "
            "a supermarket chain → Retail & Trade; a car factory → Manufacturing / Industry."
        ),
    )
    form_data["ORGANIZATION_TYPE"] = organization_type


# ===== Expander 2: Applicant Profile =====
with st.sidebar.expander("👤 Applicant Profile", expanded=False):
    amt_income = st.number_input(
        label="Annual Income",
        value=int(_TRAINING_MEDIANS["AMT_INCOME_TOTAL"]),
        min_value=0,
        max_value=10_000_000,
        step=10_000,
        help=(
            "Total gross annual income from all sources (salary, pension, business, etc.). "
            "Enter the yearly total, not monthly. "
            "Example: if monthly salary is 15,000, enter 180,000."
        ),
    )
    form_data["AMT_INCOME_TOTAL"] = amt_income

    # Age in years — converted to DAYS_BIRTH internally (negative day count)
    _median_age_years = int(abs(_TRAINING_MEDIANS.get("DAYS_BIRTH", -14235)) // 365)
    age_years = st.slider(
        label="Age (years)",
        value=_median_age_years,
        min_value=18,
        max_value=75,
        step=1,
        help=(
            "Applicant's current age in whole years. "
            "The model uses this to assess credit maturity. "
            "Converted internally to days since birth."
        ),
    )
    # Convert years → negative day count (model internal representation)
    form_data["DAYS_BIRTH"] = -int(age_years * 365.25)

    # Years employed — converted to DAYS_EMPLOYED internally
    _median_years_employed = int(abs(_TRAINING_MEDIANS.get("DAYS_EMPLOYED", -1213)) // 365)
    years_employed = st.slider(
        label="Years at Current Employer",
        value=_median_years_employed,
        min_value=0,
        max_value=50,
        step=1,
        help=(
            "How many complete years the applicant has been with their current employer. "
            "Enter 0 if currently unemployed, a student, or a pensioner with no active employment. "
            "Converted internally to days."
        ),
    )
    # 0 years → treated as not currently employed; otherwise convert to negative days
    form_data["DAYS_EMPLOYED"] = 0 if years_employed == 0 else -int(years_employed * 365)

    cnt_children = st.number_input(
        label="Number of Children",
        value=int(_TRAINING_MEDIANS["CNT_CHILDREN"]),
        min_value=0,
        max_value=20,
        step=1,
        help=(
            "Number of children the applicant is financially responsible for. "
            "Does not include adult dependants."
        ),
    )
    form_data["CNT_CHILDREN"] = cnt_children

    cnt_fam_members = st.number_input(
        label="Total Family Members",
        value=int(_TRAINING_MEDIANS["CNT_FAM_MEMBERS"]),
        min_value=1,
        max_value=20,
        step=1,
        help=(
            "Total number of people in the applicant's household, including the applicant. "
            "Example: applicant + spouse + 2 children = 4."
        ),
    )
    form_data["CNT_FAM_MEMBERS"] = cnt_fam_members

    education_options = [
        "Academic degree",
        "Higher education",
        "Incomplete higher",
        "Lower secondary",
        "Secondary / secondary special",
    ]
    education_type = st.selectbox(
        label="Highest Education Level",
        options=education_options,
        index=4,
        help=(
            "Applicant's highest completed level of formal education. "
            "'Secondary / secondary special' = high school or vocational qualification. "
            "'Higher education' = university bachelor's or equivalent. "
            "'Academic degree' = master's, PhD, or equivalent postgraduate."
        ),
    )
    form_data["NAME_EDUCATION_TYPE"] = education_type

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
        label="Employment / Income Type",
        options=income_type_options,
        index=0,
        help=(
            "Primary source of the applicant's income. "
            "'Working' = regular employee. "
            "'Commercial associate' = works for a commercial company. "
            "'State servant' = civil servant / public sector employee. "
            "'Pensioner' = retired and receiving a pension. "
            "'Self-employed' = runs their own business without formal employees."
        ),
    )
    form_data["NAME_INCOME_TYPE"] = income_type

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
        label="Occupation",
        options=occupation_options,
        index=0,
        help=(
            "Applicant's current job role or occupation category. "
            "Select 'Unknown' if the applicant is unemployed, retired, or the occupation does not fit. "
            "'Core staff' = essential/frontline staff (e.g. bank tellers, receptionists). "
            "'Laborers' = manual/physical workers. "
            "'High skill tech staff' = engineers, technicians with specialised training."
        ),
    )
    form_data["OCCUPATION_TYPE"] = occupation_type

    region_rating = st.slider(
        label="Home Region Credit Risk Rating",
        value=int(_TRAINING_MEDIANS["REGION_RATING_CLIENT"]),
        min_value=1,
        max_value=3,
        step=1,
        help=(
            "The lender's internal risk rating of the applicant's home region. "
            "1 = low-risk region (urban, stable economy). "
            "2 = medium-risk region. "
            "3 = high-risk region (rural, higher historical default rates). "
            "If unknown, leave at 2 (the average)."
        ),
    )
    form_data["REGION_RATING_CLIENT"] = region_rating


# ===== Expander 3: Credit History =====
with st.sidebar.expander("📊 Credit History", expanded=False):
    ext_source_1 = st.slider(
        label="External Credit Score 1",
        value=float(_TRAINING_MEDIANS.get("EXT_SOURCE_1", 0.5)),
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        help=(
            "A normalised creditworthiness score from an external credit bureau (scale: 0–1). "
            "Higher = better credit history. "
            "This score is often missing (~45% of applicants); leave at 0.50 if unavailable. "
            "It reflects past repayment behaviour across all lenders."
        ),
    )
    form_data["EXT_SOURCE_1"] = ext_source_1

    ext_source_2 = st.slider(
        label="External Credit Score 2",
        value=float(_TRAINING_MEDIANS["EXT_SOURCE_2"]),
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        help=(
            "A normalised creditworthiness score from a second external bureau (scale: 0–1). "
            "Higher = better credit history. "
            "This is the single strongest predictor in the model — provide it if possible. "
            "It summarises the applicant's overall credit track record."
        ),
    )
    form_data["EXT_SOURCE_2"] = ext_source_2

    ext_source_3 = st.slider(
        label="External Credit Score 3",
        value=float(_TRAINING_MEDIANS["EXT_SOURCE_3"]),
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        help=(
            "A normalised creditworthiness score from a third external source (scale: 0–1). "
            "Higher = better credit history. "
            "Often derived from alternative data (e.g. utility payment records). "
            "Leave at 0.46 (median) if unavailable."
        ),
    )
    form_data["EXT_SOURCE_3"] = ext_source_3

    bureau_cnt = st.number_input(
        label="Number of Credit Bureau Records",
        value=int(_TRAINING_MEDIANS["bureau_cnt"]),
        min_value=0,
        max_value=50,
        step=1,
        help=(
            "Total number of past loans or credit lines recorded at external credit bureaus. "
            "Includes mortgages, car loans, personal loans at any lender — not just this one. "
            "0 means the applicant has no credit history on record."
        ),
    )
    form_data["bureau_cnt"] = bureau_cnt

    prev_cnt = st.number_input(
        label="Previous Applications at This Lender",
        value=int(_TRAINING_MEDIANS["prev_cnt"]),
        min_value=0,
        max_value=50,
        step=1,
        help=(
            "Number of previous loan applications the applicant has made specifically "
            "to this lender (Home Credit). Includes approved, rejected, and cancelled applications."
        ),
    )
    form_data["prev_cnt"] = prev_cnt

    pos_cnt = st.number_input(
        label="Point-of-Sale / Cash Loan Accounts",
        value=int(_TRAINING_MEDIANS["pos_cnt"]),
        min_value=0,
        max_value=100,
        step=1,
        help=(
            "Number of POS (point-of-sale) or cash loan accounts previously held with this lender. "
            "POS loans are short-term consumer loans taken at a retail store checkout. "
            "Enter 0 if the applicant is a first-time borrower here."
        ),
    )
    form_data["pos_cnt"] = pos_cnt

    inst_cnt = st.number_input(
        label="Instalment Payment Records",
        value=int(_TRAINING_MEDIANS["inst_cnt"]),
        min_value=0,
        max_value=100,
        step=1,
        help=(
            "Total number of instalment payment records across all previous loans. "
            "Each monthly payment on a previous loan counts as one record. "
            "Higher counts indicate a longer repayment history."
        ),
    )
    form_data["inst_cnt"] = inst_cnt

    cc_cnt = st.number_input(
        label="Credit Card Accounts",
        value=int(_TRAINING_MEDIANS["cc_cnt"]),
        min_value=0,
        max_value=50,
        step=1,
        help=(
            "Number of active or past credit card accounts held with this lender. "
            "Enter 0 if the applicant has never had a credit card here."
        ),
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
    "📋 Applicant Score",
    "📊 Model Performance",
    "⚖️ Fairness Metrics",
])

# ===== TAB 1: APPLICANT SCORE =====
with tab1:
    st.header("📋 Applicant Score")

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
            "👋 Welcome to the Credit Risk Scoring dashboard. "
            "Fill in the applicant details on the left and click 'Score Applicant' "
            "to generate a Probability of Default with SHAP explanation."
        )
    else:
        # --- Left column: Metrics | Right column: SHAP Waterfall ---
        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric(
                label="Probability of Default",
                value=f"{st.session_state.pd_score * 100:.1f}%",
            )
            st.metric(
                label="Risk Band",
                value=_RISK_BAND_DISPLAY.get(
                    st.session_state.risk_band, st.session_state.risk_band
                ),
            )

        with col2:
            st.markdown("### Top 10 Feature Contributions to Score")

            # Compute SHAP (cached in session_state to avoid recomputation on tab switch)
            if "shap_vals" not in st.session_state:
                from src.explain import compute_shap_values

                st.session_state.shap_vals = compute_shap_values(model, st.session_state.X)

            shap_vals = st.session_state.shap_vals

            # Build a display copy of the SHAP Explanation with human-readable feature names.
            # The original shap_vals (raw names) is kept for adverse action factor lookup.
            import copy

            shap_display = copy.copy(shap_vals)
            if (
                hasattr(shap_display, "feature_names")
                and shap_display.feature_names is not None
            ):
                shap_display.feature_names = [
                    FEATURE_LABELS.get(n, n) for n in shap_display.feature_names
                ]

            # Render waterfall
            try:
                fig = get_shap_waterfall_figure(shap_display, idx=0, X=st.session_state.X)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)  # Free memory
            except Exception as e:
                st.error(f"❌ SHAP waterfall failed: {str(e)}")

        st.divider()

        # --- Adverse Action Factors Table ---
        st.markdown("### Adverse Action Factors (Top 5)")

        try:
            factors = get_adverse_action_factors(
                shap_vals, idx=0, feature_labels=FEATURE_LABELS, top_n=5
            )

            # Format for display: rank, factor (human label), direction, SHAP value
            df_factors = pd.DataFrame(
                [
                    {
                        "Rank": f["rank"],
                        "Factor": f["human_label"],
                        "Impact Direction": "↑ increases risk"
                        if f["direction"] == "increases_risk"
                        else "↓ reduces risk",
                        "SHAP Value": f"{f['shap_value']:.4f}",
                    }
                    for f in factors
                ]
            )

            st.dataframe(df_factors, use_container_width=True)

            st.info("⚠️ Adverse action factors disclosed per GDPR Art. 22 — right to explanation.")
        except Exception as e:
            st.error(f"❌ Adverse action factors failed: {str(e)}")


# ===== TAB 2: MODEL PERFORMANCE =====
with tab2:
    st.header("📊 Model Performance Metrics")

    try:
        # Load evaluation metrics from JSON
        eval_path = Path(__file__).resolve().parent.parent / "reports" / "catboost_raw_eval.json"

        if eval_path.exists():
            with open(eval_path) as f:
                eval_metrics = json.load(f)

            # Display 4 key metrics in columns
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    label="Gini Coefficient (OOT)",
                    value=f"{eval_metrics.get('Gini', 0):.4f}",
                )
                st.caption(
                    "**Rank-ordering power.** Ranges 0–1; higher is better. "
                    "0 = random guessing, 1 = perfect separation. "
                    "Industry benchmark: ≥0.60 is strong. "
                    "Evaluated on the held-out 20% test set (out-of-time, never seen during training)."
                )
            with col2:
                st.metric(
                    label="KS Statistic (OOT)",
                    value=f"{eval_metrics.get('KS', 0):.4f}",
                )
                st.caption(
                    "**Kolmogorov–Smirnov statistic.** Maximum gap between the cumulative "
                    "default rate and non-default rate curves. Ranges 0–1; higher is better. "
                    "≥0.40 = strong discriminating power (Basel III standard)."
                )
            with col3:
                st.metric(
                    label="AUC-ROC (OOT)",
                    value=f"{eval_metrics.get('AUC-ROC', 0):.4f}",
                )
                st.caption(
                    "**Area Under the ROC Curve.** Probability that the model ranks a "
                    "random defaulter above a random non-defaulter. "
                    "0.5 = random, 1.0 = perfect. Related to Gini by: AUC = (Gini + 1) / 2."
                )
            with col4:
                st.metric(
                    label="Brier Score (OOT)",
                    value=f"{eval_metrics.get('Brier', 0):.4f}",
                )
                st.caption(
                    "**Probability calibration accuracy.** Mean squared error between "
                    "predicted default probability and actual outcome. "
                    "Lower = better calibrated. <0.08 = well-calibrated at the 8% default rate."
                )

            st.divider()

            # Basel CRE36.54 attribution (D-14 — exact text per UI-SPEC)
            st.info(
                "📋 **Basel CRE36.54 Compliance:** OOT Gini evaluated on held-out 20% test set "
                "(temporal validation, SK_ID_CURR sort). Gini ≥ 0.60 indicates strong rank-ordering "
                "power on unseen applicants. OOT Gini of 0.5814 is the regulatory discrimination metric."
            )
        else:
            st.error(f"⚠️ Model metrics file not found: {eval_path}")
            st.info("Expected: `reports/catboost_raw_eval.json`")

    except Exception as e:
        st.error(f"❌ Failed to load metrics: {str(e)}")


# ===== TAB 3: FAIRNESS METRICS =====
with tab3:
    st.header("⚖️ Fairness Analysis")

    try:
        # Load fairness metrics from CSV
        fairness_path = Path(__file__).resolve().parent.parent / "reports" / "fairness_metrics.csv"

        if fairness_path.exists():
            df_fair = pd.read_csv(fairness_path)

            # Verify required columns exist
            required_cols = ["group_name", "demographic_parity", "tpr", "fpr"]
            missing_cols = [col for col in required_cols if col not in df_fair.columns]

            if missing_cols:
                st.error(f"❌ CSV is missing columns: {missing_cols}")
                st.info(f"Expected columns: {required_cols}")
            else:
                # Explain column meanings before showing the table
                with st.expander("ℹ️ How to read this table", expanded=True):
                    st.markdown(
                        """
| Column | What it measures |
|--------|-----------------|
| **group_name** | Demographic subgroup (e.g. Gender: Male / Female; Age: Young / Mid / Senior) |
| **demographic_parity** | Average predicted default probability for this group. If groups differ greatly, the model treats them unequally. |
| **tpr** | **True Positive Rate** — fraction of actual defaulters the model correctly flagged as high-risk in this group. Higher = fewer missed defaults. |
| **fpr** | **False Positive Rate** — fraction of non-defaulters incorrectly flagged as high-risk in this group. Lower = fewer unfair rejections. |
| **DIR columns** | **Disparate Impact Ratio** — ratio of the less-favoured group's rate to the more-favoured group's rate. **≥ 0.80 = passes the 80% fairness rule** (EU AI Act Art. 6). |
                        """
                    )

                # Display full fairness table
                st.dataframe(df_fair, use_container_width=True)

                st.divider()

                # Extract Gender DIR (disparate impact ratio) from CSV
                # Format: "Gender: M", "Gender: F" rows; compute DIR = min(F,M) / max(F,M)
                gender_rows = df_fair[df_fair["group_name"].str.contains("Gender", case=False)]

                if len(gender_rows) >= 2:
                    gender_dir_values = gender_rows["demographic_parity"].values
                    gender_dir = min(gender_dir_values) / max(gender_dir_values)

                    # Gate decision (per CONTEXT.md D-16, D-17)
                    if gender_dir >= 0.80:
                        st.success(
                            f"✅ **Gender Disparate Impact Ratio: {gender_dir:.3f}** — Gate PASSED (≥0.80)"
                        )
                    else:
                        st.warning(
                            f"⚠️ **Gender Disparate Impact Ratio: {gender_dir:.3f}** — Gate FAILED (<0.80)"
                        )
                else:
                    st.warning("⚠️ Gender DIR could not be calculated (insufficient gender rows in CSV)")

                st.divider()

                # EU AI Act & GDPR note (D-17 — exact text per UI-SPEC)
                st.info(
                    "📜 **EU AI Act Art. 6 High-Risk AI:** Gender Disparate Impact Ratio gate "
                    "is ≥0.80 (fairness measure for automated decision systems). Age DIR is monitored "
                    "only — AGE_YEARS is excluded from all model training features per GDPR Art. 22 "
                    "(age is a protected attribute)."
                )
        else:
            st.error(f"⚠️ Fairness metrics file not found: {fairness_path}")
            st.info("Expected: `reports/fairness_metrics.csv`")

    except Exception as e:
        st.error(f"❌ Failed to load fairness metrics: {str(e)}")
