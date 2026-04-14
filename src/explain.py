"""
explain.py
----------
SHAP-based explainability and fairness analysis.

Key outputs
-----------
- Global feature importance (beeswarm / bar)
- Local explanations (waterfall / force plots)
- Fairness metrics by sensitive attribute (demographic parity, equalised odds)
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
import pandas as pd


class AdverseActionFactor(TypedDict):
    """Adverse action factor for regulatory compliance (GDPR Art. 22)."""

    feature_name: str  # internal column name (e.g. "EXT_SOURCE_2")
    human_label: str  # mapped from FEATURE_LABELS dict
    shap_value: float  # signed SHAP value for this applicant
    direction: str  # "increases_risk" or "decreases_risk"
    rank: int  # 1 = most influential factor


# Module-level constant: maps internal column names to human-readable descriptions
# CRITICAL: must cover all columns in data/processed/X_cat_v2.parquet for GDPR Art. 22
# See D-06, D-09
FEATURE_LABELS: dict[str, str] = {
    # Application-level demographics and finance
    "SK_ID_CURR": "Client ID",
    "CNT_CHILDREN": "Number of children",
    "AMT_INCOME_TOTAL": "Total annual income (currency)",
    "AMT_CREDIT": "Credit amount requested (currency)",
    "AMT_ANNUITY": "Annuity amount (currency)",
    "DAYS_EMPLOYED": "Days employed (negative = duration, 365243 = unemployed sentinel)",
    "DAYS_REGISTRATION": "Days since ID registration",
    "DAYS_ID_PUBLISH": "Days since ID publication",
    "FLAG_WORK_PHONE": "Has work phone (binary)",
    "FLAG_PHONE": "Has phone (binary)",
    "FLAG_EMAIL": "Has email (binary)",
    "CNT_FAM_MEMBERS": "Number of family members",
    "REGION_RATING_CLIENT": "Region rating by client (1–3)",
    "HOUR_APPR_PROCESS_START": "Hour of application processing start (0–23)",
    "REG_REGION_NOT_LIVE_REGION": "Registration region ≠ living region (binary)",
    "REG_REGION_NOT_WORK_REGION": "Registration region ≠ work region (binary)",
    "LIVE_REGION_NOT_WORK_REGION": "Living region ≠ work region (binary)",
    "REG_CITY_NOT_LIVE_CITY": "Registration city ≠ living city (binary)",
    "REG_CITY_NOT_WORK_CITY": "Registration city ≠ work city (binary)",
    "LIVE_CITY_NOT_WORK_CITY": "Living city ≠ work city (binary)",
    "EXT_SOURCE_1": "External credit score 1 (bureau, 0–1 scale)",
    "EXT_SOURCE_2": "External credit score 2 (bureau, 0–1 scale)",
    "EXT_SOURCE_3": "External credit score 3 (bureau, 0–1 scale)",
    "APARTMENTS_AVG": "Apartment feature (average of building)",
    "BASEMENTAREA_AVG": "Basement area feature (average of building)",
    "ELEVATORS_AVG": "Elevators feature (average of building)",
    "LANDAREA_AVG": "Land area feature (average of building)",
    "NONLIVINGAREA_AVG": "Non-living area feature (average of building)",
    "OBS_30_CNT_SOCIAL_CIRCLE": "Observations from past 30 days in social circle",
    "DAYS_LAST_PHONE_CHANGE": "Days since last phone number change",
    "FLAG_DOCUMENT_3": "Has document type 3 (binary)",
    "FLAG_DOCUMENT_5": "Has document type 5 (binary)",
    "FLAG_DOCUMENT_6": "Has document type 6 (binary)",
    "FLAG_DOCUMENT_8": "Has document type 8 (binary)",
    "AMT_REQ_CREDIT_BUREAU_HOUR": "Credit inquiries in last hour",
    # Bureau aggregates (from bureau + bureau_balance tables)
    "bureau_cnt": "Count of bureau records",
    "bureau_active_cnt": "Count of active bureau accounts",
    "bureau_closed_cnt": "Count of closed bureau accounts",
    "bureau_overdue_cnt": "Count of bureau accounts with overdue",
    "bureau_days_credit_mean": "Mean days since bureau credit opened",
    "bureau_days_credit_min": "Min days since bureau credit opened",
    "bureau_days_credit_max": "Max days since bureau credit opened",
    "bureau_days_credit_std": "Std of days since bureau credit opened",
    "bureau_credit_sum": "Sum of credit amounts (bureau)",
    "bureau_amt_credit_mean": "Mean credit amount (bureau)",
    "bureau_credit_debt_sum": "Sum of remaining debt (bureau)",
    "bureau_credit_debt_std": "Std of remaining debt (bureau)",
    "bureau_credit_overdue_sum": "Sum of overdue amounts (bureau)",
    "bureau_max_overdue_amt": "Maximum overdue amount (bureau)",
    "bureau_annuity_mean": "Mean annuity (bureau)",
    "bureau_overdue_max": "Maximum overdue status code (bureau)",
    "bureau_prolong_sum": "Count of credit line prolongations (bureau)",
    "bureau_recent_openings": "Count of recent credit openings (bureau)",
    "bureau_days_since_last_credit": "Days since last bureau credit opened",
    "bureau_bbal_cnt_mean": "Mean count of bureau balance records per account",
    "bbal_months_since_last_dpd": "Months since last days-past-due status (bureau_balance)",
    # Previous application aggregates
    "prev_cnt": "Count of previous applications",
    "prev_approved_cnt": "Count of approved previous applications",
    "prev_refused_cnt": "Count of refused previous applications",
    "prev_cancelled_cnt": "Count of cancelled previous applications",
    "prev_amt_credit_mean": "Mean credit amount (previous apps)",
    "prev_amt_credit_sum": "Sum of credit amounts (previous apps)",
    "prev_amt_credit_std": "Std of credit amounts (previous apps)",
    "prev_amt_credit_max": "Max credit amount (previous apps)",
    "prev_annuity_mean": "Mean annuity (previous apps)",
    "prev_amt_down_payment_mean": "Mean down payment (previous apps)",
    "prev_credit_to_app_ratio_mean": "Mean credit-to-application ratio (previous apps)",
    "prev_days_decision_min": "Min days since decision (previous apps)",
    "prev_days_decision_mean": "Mean days since decision (previous apps)",
    "prev_days_decision_max": "Max days since decision (previous apps)",
    "prev_rate_down_payment_mean": "Mean down payment rate (previous apps)",
    # POS cash balance aggregates
    "pos_cnt": "Count of POS cash accounts",
    "pos_months_balance_mean": "Mean months of balance history (POS)",
    "pos_sk_dpd_max": "Maximum days past due (POS)",
    "pos_overdue_cnt": "Count of months with overdue (POS)",
    # Installments payments aggregates
    "inst_cnt": "Count of installment payment records",
    "inst_late_cnt": "Count of late payment records",
    "inst_amt_payment_sum": "Sum of payment amounts",
    "inst_amt_instalment_mean": "Mean instalment amount",
    "inst_payment_ratio_mean": "Mean payment-to-instalment ratio",
    "inst_payment_ratio_std": "Std of payment-to-instalment ratio",
    "inst_payment_ratio_max": "Max payment-to-instalment ratio",
    "inst_payment_diff_mean": "Mean difference from scheduled payment",
    "inst_payment_diff_std": "Std of difference from scheduled payment",
    "inst_days_past_due_max": "Maximum days past due (installments)",
    "inst_max_consec_late_streak": "Maximum consecutive late payments",
    "inst_months_since_last_late": "Months since last late payment",
    "inst_payment_trend_slope": "Trend slope of payment ratio over time",
    # Credit card balance aggregates
    "cc_cnt": "Count of credit card accounts",
    "cc_bal_mean": "Mean credit card balance",
    "cc_bal_max": "Max credit card balance",
    "cc_bal_min": "Min credit card balance",
    "cc_drawing_mean": "Mean credit card drawing",
    "cc_drawing_std": "Std of credit card drawing",
    "cc_atm_drawing_mean": "Mean ATM drawing",
    "cc_utilization_mean": "Mean credit utilization ratio",
    "cc_limit_mean": "Mean credit card limit",
    "cc_min_payment_ratio_mean": "Mean minimum payment ratio",
    # Engineered affordability / stability ratios
    "CREDIT_INCOME_RATIO": "Credit amount to annual income ratio",
    "CREDIT_TERM": "Credit term in months",
    "GOODS_CREDIT_RATIO": "Goods price to credit amount ratio",
    "AGE_YEARS": "Age in years (derived from DAYS_BIRTH / -365)",
    "YEARS_EMPLOYED": "Years employed (derived from DAYS_EMPLOYED, sentinel clipped)",
    "EMPLOYED_TO_AGE_RATIO": "Years employed to age ratio",
    "DOCUMENTS_SUBMITTED": "Total document flags submitted",
    # Engineered EXT_SOURCE composites
    "EXT_SOURCE_MEAN": "Mean of EXT_SOURCE_1, 2, 3 (ignoring NaN)",
    "EXT_SOURCE_MIN": "Min of EXT_SOURCE_1, 2, 3 (ignoring NaN)",
    "EXT_SOURCE_MAX": "Max of EXT_SOURCE_1, 2, 3 (ignoring NaN)",
    "EXT_SOURCE_MEDIAN": "Median of EXT_SOURCE_1, 2, 3 (ignoring NaN)",
    "EXT_SOURCE_PROD_12": "Product of EXT_SOURCE_1 × EXT_SOURCE_2",
    "EXT_SOURCE_PROD_23": "Product of EXT_SOURCE_2 × EXT_SOURCE_3",
    "EXT_SOURCE_RATIO_12": "Ratio EXT_SOURCE_1 / EXT_SOURCE_2 (division-safe)",
    "EXT_SOURCE_RATIO_23": "Ratio EXT_SOURCE_2 / EXT_SOURCE_3 (division-safe)",
    # Engineered risk indicators
    "prev_approval_rate": "Approval rate of previous applications",
    "inst_pct_late": "Percentage of late installment payments",
    "bureau_debt_ratio": "Bureau total debt to total credit ratio",
    "cc_overdue_flag": "Has credit card overdue (binary)",
    "pos_overdue_flag": "Has POS overdue (binary)",
    "prev_credit_income_ratio": "Previous credit to income ratio",
    "prev_refusal_rate": "Refusal rate of previous applications",
    "inst_late_dpd_ratio": "Ratio of late days-past-due to total DPD",
    "bureau_active_ratio": "Ratio of active to total bureau accounts",
    "bureau_debt_to_income": "Bureau debt to annual income ratio",
    "debt_service_ratio": "Total debt service to income ratio",
    "ext_credit_risk": "Credit risk score from EXT_SOURCE composites",
    "multi_dpd_flag": "Has days-past-due across multiple sources (binary)",
    "bureau_inst_dpd": "Bureau and installment DPD overlap indicator",
    "leverage_vs_bureau": "Credit leverage vs bureau debt ratio",
    "dpd_trajectory": "Trend in days-past-due over time",
    "dpd_escalation": "Escalation flag for days-past-due (binary)",
    "debt_service_coverage": "Debt service coverage ratio (income / payments)",
    "ever_dpd_bureau": "Ever had bureau DPD (binary)",
    "bureau_prolong_any": "Any bureau credit prolongations (binary)",
    "high_credit_income": "High credit-to-income ratio flag (binary)",
    "low_payment_rate": "Low payment-to-instalment ratio flag (binary)",
    "new_credit_to_bureau_ratio": "New credit to total bureau debt ratio",
    "bureau_overdue_to_income": "Bureau overdue to annual income ratio",
    "inst_late_rate_12m": "Installment late rate in last 12 months",
    "inst_late_rate_recent_vs_historical": "Recent late rate vs historical average",
    "inst_rolling_30dpd_ratio_3m": "Rolling 30+ days past due in last 3 months",
    "inst_delinquency_escalation_flag": "Delinquency escalation flag (binary)",
    "inst_days_since_last_30dpd": "Days since last 30+ DPD event",
    "bureau_debt_to_new_credit": "Bureau total debt to new credit ratio",
    # Bureau balance status indicators
    "bbal_ever_30dpd": "Ever had 30+ DPD status (bureau_balance)",
    "bbal_ever_60dpd": "Ever had 60+ DPD status (bureau_balance)",
    "bbal_ever_90dpd": "Ever had 90+ DPD status (bureau_balance)",
    "bbal_pct_current": "Percentage of months with current status (bureau_balance)",
    "bbal_dpd_escalation": "DPD escalation in bureau_balance (binary)",
    "bbal_max_status_code": "Maximum status code reached (bureau_balance)",
    "bbal_status_volatility": "Volatility of status codes (bureau_balance)",
    "bbal_max_dpd_months_ago": "Months since worst DPD status (bureau_balance)",
    "bbal_improving_flag": "Account improving (DPD trend) (binary)",
    # Installments consistency and recency
    "inst_payment_consistency_score": "Consistency score of payment timings",
    "inst_recency_weighted_dpd": "Recency-weighted DPD (recent overdue weighted higher)",
    "inst_early_payment_pct": "Percentage of early payments",
    # Credit card dynamics
    "cc_balance_velocity_3m": "3-month balance change velocity",
    "cc_balance_volatility": "Volatility of credit card balance over time",
    "cc_atm_drawing_frequency": "Frequency of ATM drawings",
    # Previous application risk
    "prev_reject_high_risk_pct": "Percentage of rejected apps marked high risk",
    "current_to_bureau_debt_ratio": "Current credit to bureau total debt ratio",
    # Remaining trend and risk
    "cc_utilization_trend": "Trend in credit card utilization over time",
    "EXT_SOURCE_NUM_AVAILABLE": "Count of available EXT_SOURCE scores (0–3)",
    "prev_reject_fraud_flag": "Previous application rejected due to fraud (binary)",
    "inst_late_payment_acceleration": "Acceleration of late payment frequency (binary)",
    "bureau_dpd_trend_3m_vs_12m": "Bureau DPD trend: 3-month vs 12-month average",
    "ANNUITY_INCOME_RATIO": "Annuity to annual income ratio",
    # Categorical features (one-hot encoded in final store, but kept as categorical here)
    "ORGANIZATION_TYPE": "Organization type of employer (categorical)",
    "NAME_EDUCATION_TYPE": "Education level (categorical)",
    "NAME_INCOME_TYPE": "Income source type (categorical)",
    "OCCUPATION_TYPE": "Occupation type (categorical)",
    # Target (for reference, may be dropped in prediction)
    "TARGET": "Default target (0 = non-default, 1 = default)",
}


def compute_shap_values(model: object, X: pd.DataFrame) -> Any:
    """
    Extract raw CatBoost booster from CalibratedClassifierCV and compute SHAP values.

    Parameters
    ----------
    model : CalibratedClassifierCV
        Pre-fitted calibrated model wrapping FrozenEstimator(CatBoostClassifier)
    X : DataFrame
        Input features (n_samples, n_features)

    Returns
    -------
    shap.Explanation
        SHAP values of shape (n_samples, n_features), explaining pre-calibration
        log-odds predictions (raw CatBoost output).

    Notes
    -----
    SHAP values explain raw log-odds, not calibrated PD. For adverse action notices,
    top-N risk-increasing factors are extracted from raw SHAP values (per D-04).
    """
    import shap

    # Extract raw booster from calibration wrapper (D-01)
    raw_booster = model.calibrated_classifiers_[0].estimator.estimator

    # TreeExplainer on raw booster with model_output="raw"
    # For CatBoost, "raw" returns log-odds margin, shape (n_samples, n_features)
    explainer = shap.TreeExplainer(raw_booster, model_output="raw")

    # Compute SHAP values — returns Explanation object with .values attr
    shap_values = explainer(X)

    return shap_values


def plot_shap_summary(
    shap_explanation: Any, X: pd.DataFrame, plot_type: str = "dot", save_path: str | None = None
) -> None:
    """
    Global SHAP feature importance: beeswarm ("dot") or bar plot.

    Parameters
    ----------
    shap_explanation : shap.Explanation
        SHAP values from compute_shap_values()
    X : DataFrame
        Input features (same shape as SHAP values)
    plot_type : {"dot", "bar"}, default "dot"
        "dot": beeswarm plot (each point = one sample's SHAP value for one feature)
        "bar": bar plot (each bar = mean |SHAP| per feature)
    save_path : str | None, default None
        If provided, save figure to this path (PNG format)
        If None, display inline (or keep in memory if no display)

    Notes
    -----
    - Never calls plt.show() — caller controls display
    - Closes figure after saving to avoid memory leaks
    """
    import shap
    import matplotlib.pyplot as plt

    # Create appropriate SHAP plot (returns Axes object)
    if plot_type == "dot":
        ax = shap.plots.beeswarm(shap_explanation, show=False)
    elif plot_type == "bar":
        ax = shap.plots.bar(shap_explanation, show=False)
    else:
        raise ValueError(f"plot_type must be 'dot' or 'bar', got {plot_type}")

    # Save if requested; always close the current figure to free memory
    if save_path is not None:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")

    plt.close("all")


def plot_shap_local(
    shap_explanation: Any,
    idx: int,
    X: pd.DataFrame,
    plot_type: str = "waterfall",
    save_path: str | None = None,
) -> None:
    """
    Local SHAP explanation for a single applicant: waterfall or force plot.

    Parameters
    ----------
    shap_explanation : shap.Explanation
        SHAP values from compute_shap_values()
    idx : int
        Row index to explain (0-based)
    X : DataFrame
        Input features (same shape as SHAP values)
    plot_type : {"waterfall", "force"}, default "waterfall"
        "waterfall": waterfall plot (feature contributions) — saved as PNG
        "force": force plot (JavaScript-interactive) — saved as HTML
    save_path : str | None, default None
        If provided, save figure/HTML to this path
        If None, display inline or keep in memory

    Notes
    -----
    - Force plot is HTML (JavaScript); waterfall is PNG (static)
    - Never calls plt.show() — caller controls display
    - Closes matplotlib figures after saving to avoid memory leaks
    """
    import shap
    import matplotlib.pyplot as plt

    # Validate index
    if idx < 0 or idx >= shap_explanation.shape[0]:
        raise IndexError(f"idx {idx} out of range [0, {shap_explanation.shape[0]})")

    if plot_type == "waterfall":
        # Waterfall plot (PNG, static) — returns Axes object
        ax = shap.plots.waterfall(shap_explanation[idx], show=False)

        if save_path is not None:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")

        plt.close("all")

    elif plot_type == "force":
        # Force plot (HTML, interactive) — returns AdditiveForceVisualizer
        force_plot_obj = shap.plots.force(
            shap_explanation.base_values,
            shap_explanation[idx].values,
            X.iloc[idx],
            show=False,
        )

        if save_path is not None:
            # Save HTML directly via the AdditiveForceVisualizer._repr_html_() output
            html_str = force_plot_obj._repr_html_()
            with open(save_path, "w") as f:
                f.write(html_str)

        # Clean up matplotlib figures
        plt.close("all")

    else:
        raise ValueError(f"plot_type must be 'waterfall' or 'force', got {plot_type}")


def compute_fairness_metrics(
    model: object, X: pd.DataFrame, y: pd.Series, sensitive_cols: list[str]
) -> pd.DataFrame:
    """Compute group-level fairness metrics by sensitive attribute."""
    # TODO: implement
    raise NotImplementedError


def get_adverse_action_factors(
    shap_explanation: Any, idx: int, feature_labels: dict[str, str], top_n: int = 5
) -> list[AdverseActionFactor]:
    """Get top-N risk-increasing factors for a single applicant (GDPR Art. 22)."""
    # TODO: implement
    raise NotImplementedError


def compute_shap_stability(shap_train: np.ndarray, shap_oot: np.ndarray) -> float:
    """
    Spearman correlation of mean(|SHAP|) feature rankings between train and OOT.

    Returns correlation coefficient in [-1, 1]. Value >= 0.90 is considered stable.
    """
    # TODO: implement
    raise NotImplementedError


def fairness_report(
    y_true: pd.Series,
    y_pred: pd.Series,
    sensitive_col: pd.Series,
) -> pd.DataFrame:
    """Compute group-level fairness metrics."""
    # TODO: implement
    raise NotImplementedError
