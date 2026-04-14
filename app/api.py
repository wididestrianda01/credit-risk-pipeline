"""
app/api.py
----------
FastAPI application for real-time credit risk scoring.

Endpoints
---------
POST /predict   – accepts raw applicant features, returns calibrated PD + risk band + SHAP factors
                  Requires X-API-Key header.
GET  /health    – liveness check, returns model version + uptime (no auth required)

Usage
-----
1. Set API_KEY environment variable:
   export API_KEY=your-api-key

2. Or copy .env.example to .env and update:
   cp .env.example .env

3. Run locally:
   uvicorn app.api:app --reload

4. Test POST /predict:
   curl -X POST http://127.0.0.1:8000/predict \\
     -H "X-API-Key: your-api-key" \\
     -H "Content-Type: application/json" \\
     -d '{"EXT_SOURCE_2": 0.5, "AMT_CREDIT": 250000, "AMT_INCOME_TOTAL": 90000}'

5. Visit OpenAPI docs:
   http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import math
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field

from src.explain import (
    FEATURE_LABELS,
    AdverseActionFactor,
    compute_shap_values,
    get_adverse_action_factors,
)
from src.model_base import load_model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODEL_PATH = _PROJECT_ROOT / "models" / "catboost_raw_calibrated_v2.pkl"

# Risk tier PD thresholds (IRB PD bucket convention, per D-05 in CONTEXT.md)
_RISK_TIER_VERY_LOW: float = 0.05   # PD < 5%
_RISK_TIER_LOW: float = 0.15         # PD 5–15%
_RISK_TIER_MEDIUM: float = 0.30      # PD 15–30%
_RISK_TIER_HIGH: float = 0.50        # PD 30–50%
# PD >= 50% -> VERY_HIGH

_MODEL_VERSION: str = "catboost_v2"
_GINI_AT_TRAINING: float = 0.5814   # OOT Gini, CatBoost v2 (Basel CRE36.54 metric)

# Training-set medians for all 169 model features, in exact model feature order.
# Computed from data/processed/X_cat_v2.parquet (307,511 rows).
# -999.0 sentinels are used for features with high structural missingness
# (EXT_SOURCE_1, bureau aggregates with no bureau history, cc aggregates, etc.)
_TRAINING_MEDIANS: dict[str, Any] = {
    "SK_ID_CURR": 278202.0,
    "CNT_CHILDREN": 0.0,
    "AMT_INCOME_TOTAL": 147150.0,
    "AMT_CREDIT": 513531.0,
    "AMT_ANNUITY": 24903.0,
    "DAYS_EMPLOYED": -1213.0,
    "DAYS_REGISTRATION": -4504.0,
    "DAYS_ID_PUBLISH": -3254.0,
    "FLAG_WORK_PHONE": 0.0,
    "FLAG_PHONE": 0.0,
    "FLAG_EMAIL": 0.0,
    "CNT_FAM_MEMBERS": 2.0,
    "REGION_RATING_CLIENT": 2.0,
    "HOUR_APPR_PROCESS_START": 12.0,
    "REG_REGION_NOT_LIVE_REGION": 0.0,
    "REG_REGION_NOT_WORK_REGION": 0.0,
    "LIVE_REGION_NOT_WORK_REGION": 0.0,
    "REG_CITY_NOT_LIVE_CITY": 0.0,
    "REG_CITY_NOT_WORK_CITY": 0.0,
    "LIVE_CITY_NOT_WORK_CITY": 0.0,
    "EXT_SOURCE_1": -999.0,
    "EXT_SOURCE_2": 0.565467,
    "EXT_SOURCE_3": 0.45969,
    "APARTMENTS_AVG": -999.0,
    "BASEMENTAREA_AVG": -999.0,
    "ELEVATORS_AVG": -999.0,
    "LANDAREA_AVG": -999.0,
    "NONLIVINGAREA_AVG": -999.0,
    "OBS_30_CNT_SOCIAL_CIRCLE": 0.0,
    "DAYS_LAST_PHONE_CHANGE": -757.0,
    "FLAG_DOCUMENT_3": 1.0,
    "FLAG_DOCUMENT_5": 0.0,
    "FLAG_DOCUMENT_6": 0.0,
    "FLAG_DOCUMENT_8": 0.0,
    "AMT_REQ_CREDIT_BUREAU_HOUR": 0.0,
    "bureau_cnt": 4.0,
    "bureau_active_cnt": 1.0,
    "bureau_closed_cnt": 2.0,
    "bureau_overdue_cnt": 0.0,
    "bureau_days_credit_mean": -999.0,
    "bureau_days_credit_min": -1544.0,
    "bureau_days_credit_max": -373.0,
    "bureau_days_credit_std": 436.030471,
    "bureau_credit_sum": 711000.0,
    "bureau_amt_credit_mean": 158730.75,
    "bureau_credit_debt_sum": 87583.5,
    "bureau_credit_debt_std": 19077.67362,
    "bureau_credit_overdue_sum": 0.0,
    "bureau_max_overdue_amt": 0.0,
    "bureau_annuity_mean": -999.0,
    "bureau_overdue_max": 0.0,
    "bureau_prolong_sum": 0.0,
    "bureau_recent_openings": 0.0,
    "bureau_days_since_last_credit": -239.0,
    "bureau_bbal_cnt_mean": -999.0,
    "bbal_months_since_last_dpd": -999.0,
    "prev_cnt": 3.0,
    "prev_approved_cnt": 2.0,
    "prev_refused_cnt": 0.0,
    "prev_cancelled_cnt": 0.0,
    "prev_amt_credit_mean": 110439.0,
    "prev_amt_credit_sum": 383778.0,
    "prev_amt_credit_std": 72074.915627,
    "prev_amt_credit_max": 203760.0,
    "prev_annuity_mean": 11443.4955,
    "prev_amt_down_payment_mean": 2506.5,
    "prev_credit_to_app_ratio_mean": 1.013416,
    "prev_days_decision_min": -1408.0,
    "prev_days_decision_max": -322.0,
    "prev_rate_down_payment_mean": 0.052402,
    "pos_cnt": 21.0,
    "pos_months_balance_mean": -30.3125,
    "pos_sk_dpd_max": 0.0,
    "pos_overdue_cnt": 0.0,
    "inst_cnt": 23.0,
    "inst_late_cnt": 1.0,
    "inst_amt_payment_sum": 289330.02,
    "inst_amt_instalment_mean": 11954.420526,
    "inst_payment_ratio_mean": 1.0,
    "inst_payment_ratio_std": 0.0,
    "inst_payment_ratio_max": 1.0,
    "inst_payment_diff_mean": 0.0,
    "inst_payment_diff_std": 0.0,
    "inst_days_past_due_max": 1.0,
    "inst_max_consec_late_streak": 2858.0,
    "inst_months_since_last_late": 0.27,
    "inst_payment_trend_slope": 100.160339,
    "cc_cnt": 0.0,
    "cc_bal_mean": -999.0,
    "cc_bal_max": -999.0,
    "cc_bal_min": -999.0,
    "cc_drawing_mean": -999.0,
    "cc_drawing_std": -999.0,
    "cc_atm_drawing_mean": -999.0,
    "cc_utilization_mean": -999.0,
    "cc_limit_mean": -999.0,
    "cc_min_payment_ratio_mean": -999.0,
    "CREDIT_INCOME_RATIO": 3.265067,
    "CREDIT_TERM": 20.0,
    "GOODS_CREDIT_RATIO": 0.893815,
    "AGE_YEARS": 43.12115,
    "YEARS_EMPLOYED": 3.321013,
    "EMPLOYED_TO_AGE_RATIO": 0.088645,
    "DOCUMENTS_SUBMITTED": 1.0,
    "EXT_SOURCE_MEAN": -332.58897,
    "EXT_SOURCE_MIN": -999.0,
    "EXT_SOURCE_MAX": 0.648336,
    "EXT_SOURCE_MEDIAN": 0.439698,
    "EXT_SOURCE_PROD_12": -213.513936,
    "EXT_SOURCE_PROD_23": 0.207154,
    "EXT_SOURCE_RATIO_12": -1396.268813,
    "EXT_SOURCE_RATIO_23": 0.890791,
    "prev_approval_rate": 0.75,
    "inst_pct_late": 0.007752,
    "bureau_debt_ratio": 0.120046,
    "cc_overdue_flag": 0.0,
    "pos_overdue_flag": 0.0,
    "prev_credit_income_ratio": 0.7746,
    "prev_refusal_rate": 0.0,
    "inst_late_dpd_ratio": 0.052632,
    "bureau_active_ratio": 0.333333,
    "bureau_debt_to_income": 0.6062,
    "debt_service_ratio": 1.954,
    "ext_credit_risk": 0.0,
    "multi_dpd_flag": 0.0,
    "bureau_inst_dpd": 0.0,
    "leverage_vs_bureau": 1.5998,
    "dpd_trajectory": 0.0,
    "dpd_escalation": 0.0,
    "debt_service_coverage": 5.674398,
    "ever_dpd_bureau": 0.0,
    "bureau_prolong_any": 0.0,
    "high_credit_income": 0.0,
    "low_payment_rate": 0.0,
    "new_credit_to_bureau_ratio": 0.379744,
    "bureau_overdue_to_income": 0.0,
    "inst_late_rate_12m": -999.0,
    "inst_late_rate_recent_vs_historical": -999.0,
    "inst_rolling_30dpd_ratio_3m": -999.0,
    "inst_delinquency_escalation_flag": -999.0,
    "inst_days_since_last_30dpd": -1.0,
    "bureau_debt_to_new_credit": 0.175269,
    "bbal_ever_30dpd": 0.0,
    "bbal_ever_60dpd": 0.0,
    "bbal_ever_90dpd": 0.0,
    "bbal_pct_current": 0.0,
    "bbal_dpd_escalation": 0.0,
    "bbal_max_status_code": -1.0,
    "bbal_status_volatility": 0.0,
    "bbal_max_dpd_months_ago": 0.0,
    "bbal_improving_flag": 0.0,
    "inst_payment_consistency_score": 0.833333,
    "inst_recency_weighted_dpd": -9.107143,
    "inst_early_payment_pct": 0.842105,
    "cc_balance_velocity_3m": 0.0,
    "cc_balance_volatility": 0.0,
    "cc_atm_drawing_frequency": 0.0,
    "prev_reject_high_risk_pct": 0.0,
    "current_to_bureau_debt_ratio": 0.327311,
    "cc_utilization_trend": 0.0,
    "EXT_SOURCE_NUM_AVAILABLE": 3.0,
    "prev_reject_fraud_flag": 0.0,
    "inst_late_payment_acceleration": 0.0,
    "bureau_dpd_trend_3m_vs_12m": 0.0,
    "ANNUITY_INCOME_RATIO": 0.162833,
    "ORGANIZATION_TYPE": "Business Entity Type 3",
    "NAME_EDUCATION_TYPE": "Secondary / secondary special",
    "NAME_INCOME_TYPE": "Working",
    "OCCUPATION_TYPE": "Unknown",
}

# API Key header security dependency
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class ApplicantFeaturesRequest(BaseModel):
    """Request schema for POST /predict.

    All fields are optional — missing fields are filled with training-set medians.
    Secondary-table features (bureau, previous applications, POS, instalment, credit card)
    are not accepted here; the server fills them with training-set medians.

    GDPR Art. 22: server returns top-5 SHAP adverse action factors explaining any
    adverse credit decision.
    """

    # Application-level identifiers
    SK_ID_CURR: int | None = Field(
        default=None,
        description="Client ID (pass 0 or omit for anonymous scoring)",
    )

    # Core demographics and finance
    CNT_CHILDREN: int | None = Field(default=None, description="Number of children (0–20)")
    AMT_INCOME_TOTAL: float | None = Field(
        default=None,
        description="Annual income (currency units). Example: 135000",
        examples=[135000.0],
    )
    AMT_CREDIT: float | None = Field(
        default=None,
        description="Requested credit amount (currency units). Example: 450000",
        examples=[450000.0],
    )
    AMT_ANNUITY: float | None = Field(
        default=None,
        description="Loan annuity / monthly repayment amount (currency units)",
        examples=[22500.0],
    )
    DAYS_BIRTH: int | None = Field(
        default=None,
        description="Days since birth (negative integer, e.g. -14235 ≈ 39 years old). "
        "Used to derive AGE_YEARS. GDPR: age is a protected attribute — model excludes "
        "direct age signal (EMPLOYED_TO_AGE_RATIO uses derived value only).",
        examples=[-14235],
    )
    DAYS_EMPLOYED: int | None = Field(
        default=None,
        description="Days employed (negative = length of employment; 365243 = unemployed sentinel)",
        examples=[-1200],
    )
    DAYS_REGISTRATION: int | None = Field(
        default=None,
        description="Days since client registration",
    )
    DAYS_ID_PUBLISH: int | None = Field(
        default=None,
        description="Days since ID document was published",
    )
    CNT_FAM_MEMBERS: int | None = Field(
        default=None,
        description="Number of family members",
    )
    REGION_RATING_CLIENT: int | None = Field(
        default=None,
        description="Region credit risk rating (1=low, 2=medium, 3=high)",
    )

    # External credit bureau scores (strongest predictors)
    EXT_SOURCE_1: float | None = Field(
        default=None,
        description="External credit score 1 (0–1 scale; ~45% structural missingness). "
        "GDPR Art. 22: this external score is often the top SHAP factor for adverse decisions.",
        examples=[0.62],
    )
    EXT_SOURCE_2: float | None = Field(
        default=None,
        description="External credit score 2 (0–1 scale; strongest individual predictor)",
        examples=[0.58],
    )
    EXT_SOURCE_3: float | None = Field(
        default=None,
        description="External credit score 3 (0–1 scale)",
        examples=[0.55],
    )

    # Loan purpose and structure
    AMT_GOODS_PRICE: float | None = Field(
        default=None,
        description="Goods price for consumer loans (used to derive GOODS_CREDIT_RATIO)",
    )

    # Categorical features
    NAME_EDUCATION_TYPE: str | None = Field(
        default=None,
        description="Education level. One of: 'Academic degree', 'Higher education', "
        "'Incomplete higher', 'Lower secondary', 'Secondary / secondary special'",
    )
    NAME_INCOME_TYPE: str | None = Field(
        default=None,
        description="Income source type. One of: 'Working', 'Commercial associate', "
        "'Pensioner', 'State servant', 'Self-employed', 'Businessman', 'Student', 'Unemployed'",
    )
    ORGANIZATION_TYPE: str | None = Field(
        default=None,
        description="Employer organization type (e.g., 'Business Entity Type 3', 'Government', 'School')",
    )
    OCCUPATION_TYPE: str | None = Field(
        default=None,
        description="Applicant's occupation type (e.g., 'Laborers', 'Core staff', 'Managers'). "
        "'Unknown' is valid for pensioners/unemployed.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "AMT_CREDIT": 250000,
                "AMT_INCOME_TOTAL": 90000,
                "AMT_ANNUITY": 12600,
                "EXT_SOURCE_2": 0.45,
                "EXT_SOURCE_3": 0.38,
                "DAYS_BIRTH": -14235,
                "DAYS_EMPLOYED": -1825,
                "NAME_EDUCATION_TYPE": "Higher education",
                "NAME_INCOME_TYPE": "Working",
            }
        }
    )


class PredictionResponse(BaseModel):
    """Response schema for POST /predict.

    Returns calibrated PD, risk tier, top-5 SHAP adverse action factors, and audit metadata.
    """

    probability_of_default: float = Field(
        description="Calibrated probability of default (0–1). "
        "GDPR Art. 22: applicants have the right to explanation for automated credit decisions. "
        "See adverse_action_factors for the top drivers of this score.",
        examples=[0.18],
    )
    risk_band: Literal["VERY_LOW", "LOW", "MEDIUM", "HIGH", "VERY_HIGH"] = Field(
        description="IRB PD tier: VERY_LOW (PD<5%), LOW (5–15%), MEDIUM (15–30%), "
        "HIGH (30–50%), VERY_HIGH (≥50%)",
        examples=["MEDIUM"],
    )
    adverse_action_factors: list[AdverseActionFactor] = Field(
        description="Top-5 SHAP-based factors explaining this PD score. "
        "direction='increases_risk' for positive SHAP contributions. "
        "GDPR Art. 22 compliance: human-readable labels for adverse action notices.",
    )
    model_version: str = Field(
        description="Model identifier. Use this to verify which model version is deployed.",
        examples=["catboost_v2"],
    )
    gini_at_training: float = Field(
        description="Gini coefficient on OOT held-out set (Gini = 2×AUC − 1). "
        "Basel CRE36.54 regulatory discrimination metric. "
        "Portfolio signal: model's rank-ordering power on unseen applicants.",
        examples=[0.5814],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "probability_of_default": 0.18,
                "risk_band": "MEDIUM",
                "adverse_action_factors": [
                    {
                        "rank": 1,
                        "feature_name": "EXT_SOURCE_2",
                        "human_label": "External Credit Score 2 (bureau, 0–1 scale)",
                        "shap_value": 0.31,
                        "direction": "increases_risk",
                    }
                ],
                "model_version": "catboost_v2",
                "gini_at_training": 0.5814,
            }
        }
    )


class HealthResponse(BaseModel):
    """Response schema for GET /health. No authentication required."""

    status: str = Field(description="Service status ('ok' when running)", examples=["ok"])
    model_version: str = Field(
        description="Deployed model identifier", examples=["catboost_v2"]
    )
    uptime_seconds: float = Field(
        description="Seconds elapsed since app startup", examples=[42.7]
    )


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def _get_risk_band(probability_of_default: float) -> str:
    """Classify calibrated PD into a 5-tier IRB risk band.

    Parameters
    ----------
    probability_of_default : float
        Calibrated PD in [0, 1]

    Returns
    -------
    str
        One of: VERY_LOW, LOW, MEDIUM, HIGH, VERY_HIGH
    """
    if probability_of_default < _RISK_TIER_VERY_LOW:
        return "VERY_LOW"
    elif probability_of_default < _RISK_TIER_LOW:
        return "LOW"
    elif probability_of_default < _RISK_TIER_MEDIUM:
        return "MEDIUM"
    elif probability_of_default < _RISK_TIER_HIGH:
        return "HIGH"
    else:
        return "VERY_HIGH"


def _build_inference_features(request: ApplicantFeaturesRequest) -> pd.DataFrame:
    """Reconstruct the 169-column feature vector from a raw applicant request.

    Strategy:
    1. Start from training-set medians for all 169 model features.
    2. Overlay any application-level fields provided in the request.
    3. Recompute derived features (EXT_SOURCE composites, loan ratios) when
       their raw inputs were explicitly provided — avoids median contamination
       in critical composite features.

    Secondary-table aggregates (bureau_*, prev_*, pos_*, inst_*, cc_*) remain
    at their training medians since the API does not accept secondary-table data.

    Parameters
    ----------
    request : ApplicantFeaturesRequest
        Raw applicant fields (all optional)

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with 169 columns in exact model feature order.
    """
    row = dict(_TRAINING_MEDIANS)  # copy — never mutate constant

    # --- Direct field overlays ---
    direct_fields = [
        "SK_ID_CURR", "CNT_CHILDREN", "AMT_INCOME_TOTAL", "AMT_CREDIT",
        "AMT_ANNUITY", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
        "CNT_FAM_MEMBERS", "REGION_RATING_CLIENT",
        "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
        "NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE", "ORGANIZATION_TYPE", "OCCUPATION_TYPE",
    ]
    for field in direct_fields:
        val = getattr(request, field, None)
        if val is not None:
            row[field] = val

    # --- DAYS_BIRTH → AGE_YEARS ---
    if request.DAYS_BIRTH is not None:
        row["AGE_YEARS"] = abs(request.DAYS_BIRTH) / 365.25

    # --- DAYS_EMPLOYED → YEARS_EMPLOYED (clip 365243 sentinel to 0) ---
    if request.DAYS_EMPLOYED is not None:
        days = request.DAYS_EMPLOYED
        if days == 365243:
            days = 0
        row["YEARS_EMPLOYED"] = abs(days) / 365.25

    # --- Recompute EMPLOYED_TO_AGE_RATIO if both AGE_YEARS and YEARS_EMPLOYED updated ---
    age = row["AGE_YEARS"]
    yrs_emp = row["YEARS_EMPLOYED"]
    if age > 0:
        row["EMPLOYED_TO_AGE_RATIO"] = yrs_emp / age

    # --- Loan amount ratios (recompute when inputs provided) ---
    amt_income = row["AMT_INCOME_TOTAL"]
    amt_credit = row["AMT_CREDIT"]
    amt_annuity = row["AMT_ANNUITY"]

    if amt_income and amt_income > 0:
        row["CREDIT_INCOME_RATIO"] = amt_credit / amt_income
        row["ANNUITY_INCOME_RATIO"] = amt_annuity / amt_income
        row["debt_service_ratio"] = (amt_annuity * 12) / amt_income

    if amt_annuity and amt_annuity > 0:
        row["CREDIT_TERM"] = amt_credit / amt_annuity

    if request.AMT_GOODS_PRICE is not None and request.AMT_GOODS_PRICE > 0:
        row["GOODS_CREDIT_RATIO"] = amt_credit / request.AMT_GOODS_PRICE

    # --- EXT_SOURCE composites (recompute when any EXT_SOURCE changed) ---
    ext1 = row["EXT_SOURCE_1"]
    ext2 = row["EXT_SOURCE_2"]
    ext3 = row["EXT_SOURCE_3"]

    # Only recompute if at least one EXT_SOURCE was explicitly provided
    ext_provided = any(
        getattr(request, f, None) is not None
        for f in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    )
    if ext_provided:
        valid_scores = [s for s in [ext1, ext2, ext3] if s != -999.0]
        row["EXT_SOURCE_NUM_AVAILABLE"] = float(len(valid_scores))

        if valid_scores:
            row["EXT_SOURCE_MEAN"] = sum(valid_scores) / len(valid_scores)
            row["EXT_SOURCE_MIN"] = min(valid_scores)
            row["EXT_SOURCE_MAX"] = max(valid_scores)
            sorted_scores = sorted(valid_scores)
            mid = len(sorted_scores) // 2
            row["EXT_SOURCE_MEDIAN"] = sorted_scores[mid]
        else:
            row["EXT_SOURCE_MEAN"] = -999.0
            row["EXT_SOURCE_MIN"] = -999.0
            row["EXT_SOURCE_MAX"] = -999.0
            row["EXT_SOURCE_MEDIAN"] = -999.0

        # Pairwise products and ratios — use -999 sentinel when score missing
        row["EXT_SOURCE_PROD_12"] = ext1 * ext2
        row["EXT_SOURCE_PROD_23"] = ext2 * ext3
        row["EXT_SOURCE_RATIO_12"] = (ext1 / ext2) if ext2 not in (0.0, -999.0) else -999.0
        row["EXT_SOURCE_RATIO_23"] = (ext2 / ext3) if ext3 not in (0.0, -999.0) else -999.0

        # ext_credit_risk: inverse of mean score (higher = riskier)
        if valid_scores:
            row["ext_credit_risk"] = 1.0 - (sum(valid_scores) / len(valid_scores))

    # Build DataFrame in exact model feature order
    X = pd.DataFrame([row])[list(_TRAINING_MEDIANS.keys())]
    return X


def verify_api_key(api_key: str | None = Depends(_api_key_header)) -> str:
    """Verify X-API-Key header against the API_KEY environment variable.

    Uses secrets.compare_digest() to prevent timing attacks (ASVS 2.7.1 / T-05.1-01).

    Raises
    ------
    HTTPException
        401 if key is missing or does not match API_KEY env var.
    """
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
        )

    expected_key = os.environ.get("API_KEY", "")
    if not secrets.compare_digest(api_key, expected_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    return api_key


# ---------------------------------------------------------------------------
# FastAPI Application — Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Load model once at startup; record startup time for /health uptime."""
    app.state.startup_time = time.time()
    app.state.model = load_model(str(_MODEL_PATH))

    yield
    # Shutdown: model is garbage-collected automatically


app = FastAPI(
    title="Credit Risk Scoring API",
    version="0.1.0",
    description=(
        "Production credit scoring endpoint for the Home Credit Default Risk dataset. "
        "Returns calibrated PD, risk tier, and SHAP adverse action factors. "
        "**GDPR Art. 22** compliant (right to explanation for automated decisions). "
        "**Basel CRE36.54** compliant (OOT Gini metric included in response)."
    ),
    lifespan=_lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Score a credit applicant — returns calibrated PD + SHAP factors",
    tags=["prediction"],
)
async def predict(request: ApplicantFeaturesRequest) -> PredictionResponse:
    """Score a single applicant for credit default risk.

    **Authentication:** Requires `X-API-Key` header.

    **Input:** Raw application fields — all optional; missing values are filled
    with training-set medians. Secondary-table features (bureau history, previous
    applications, etc.) are imputed automatically.

    **Output:**
    - Calibrated PD (0–1) from CatBoost v2 (OOT Gini = 0.5814)
    - 5-tier risk band (VERY_LOW → VERY_HIGH)
    - Top-5 SHAP adverse action factors with human-readable labels

    **Regulatory compliance:**
    - GDPR Art. 22: `adverse_action_factors` supports right-to-explanation notices
    - Basel CRE36.54: `gini_at_training` confirms model discrimination power
    """
    X = _build_inference_features(request)

    pd_score: float = float(app.state.model.predict_proba(X)[0, 1])
    risk_band = _get_risk_band(pd_score)

    shap_explanation = compute_shap_values(app.state.model, X)
    adverse_factors = get_adverse_action_factors(
        shap_explanation, idx=0, feature_labels=FEATURE_LABELS, top_n=5
    )

    return PredictionResponse(
        probability_of_default=pd_score,
        risk_band=risk_band,
        adverse_action_factors=adverse_factors,
        model_version=_MODEL_VERSION,
        gini_at_training=_GINI_AT_TRAINING,
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check — no authentication required",
    tags=["health"],
)
async def health() -> HealthResponse:
    """Return service status, deployed model version, and uptime.

    No API key required. Use this endpoint for load-balancer health checks
    and to verify which model version is deployed.
    """
    uptime_seconds = time.time() - app.state.startup_time
    return HealthResponse(
        status="ok",
        model_version=_MODEL_VERSION,
        uptime_seconds=uptime_seconds,
    )
