"""
features.py
-----------
All feature engineering functions.  Every transformation used during
training must have a corresponding function here so it can be reused
at inference time.

Convention
----------
- Functions take a DataFrame and return a DataFrame (no side effects).
- Prefix aggregate features with the source table name, e.g. `bureau_`.
- Boolean flags use the suffix `_flag`.
"""

import gc
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold


def _get_project_root() -> Path:
    """
    Locate the project root by finding the directory containing 'src/' and 'tests/'.

    Returns
    -------
    Path
        Absolute path to project root. Raises FileNotFoundError if root not found
        (should never happen in normal execution).

    Notes
    -----
    This function is defensive — it walks up from the current module until it finds
    a directory containing both src/ and tests/ subdirectories. This allows feature
    engineering functions to work correctly regardless of what working directory
    the caller is in.
    """
    current = Path(__file__).parent  # src/ directory
    for candidate in [current.parent] + list(current.parent.parents):
        if (candidate / "src").is_dir() and (candidate / "tests").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate project root (expected to find src/ and tests/)")


_PROJECT_ROOT = _get_project_root()

# Sentinel value used for DAYS_EMPLOYED when the applicant is unemployed.
# Home Credit encodes unemployment as 365243 (a large positive number) instead
# of a negative value like normal employment entries.
_DAYS_EMPLOYED_SENTINEL: int = 365_243

# Tree-friendly fill value for missing/undefined features.  Using -999 instead
# of 0 or mean avoids shifting the distribution and lets gradient boosting
# models learn a dedicated "missing" split.
_NAN_SENTINEL: float = -999.0

# WoE clipping bound.  ln(dist_non_events / dist_events) is clipped to
# [-_WOE_CLIP, +_WOE_CLIP] to avoid ±inf when a bin contains only events
# or only non-events.  ±5 corresponds to an odds ratio of ~150x, already
# extreme for any real feature.
_WOE_CLIP: float = 5.0

# Information Value thresholds (Siddiqi, *Credit Risk Scorecards*).
# Used both for the printed summary and as named bounds so callers can
# reference them without hard-coding magic numbers.
_IV_VERY_STRONG: float = 0.5
_IV_STRONG: float = 0.3
_IV_MEDIUM: float = 0.1
_IV_WEAK: float = 0.02

# Regulatory exclusions — columns that must be dropped from tree models per legal compliance.
# CODE_GENDER: GDPR Art. 21 (protection from discrimination), EU Consumer Credit Directive.
# thin_file_young: EU AI Act Art. 6 (age-gating is prohibited age discrimination).
_REGULATORY_DROP_COLS: list[str] = ["CODE_GENDER", "thin_file_young"]


# ---------------------------------------------------------------------------
# Private helpers — one concern per function
# ---------------------------------------------------------------------------


def _engineer_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute loan-affordability ratios from the application table.

    All divisions are guarded: if the denominator is 0 the ratio is set to 0.
    Residual inf values (e.g. from upstream data anomalies) are replaced with 0.
    Remaining NaN is filled with _NAN_SENTINEL.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain AMT_CREDIT, AMT_INCOME_TOTAL, AMT_ANNUITY, AMT_GOODS_PRICE.

    Returns
    -------
    pd.DataFrame
        Copy of df with four additional ratio columns.
    """
    out = df.copy()

    income = out["AMT_INCOME_TOTAL"].to_numpy()
    annuity = out["AMT_ANNUITY"].to_numpy()
    credit = out["AMT_CREDIT"].to_numpy()
    goods = out["AMT_GOODS_PRICE"].to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        # Higher ratio = applicant is borrowing a larger multiple of their income.
        out["CREDIT_INCOME_RATIO"] = np.where(income > 0, credit / income, 0.0)

        # Higher ratio = annuity payments consume a larger share of monthly income.
        out["ANNUITY_INCOME_RATIO"] = np.where(income > 0, annuity / income, 0.0)

        # Number of months to repay; longer term implies higher total interest cost.
        out["CREDIT_TERM"] = np.where(annuity > 0, credit / annuity, 0.0)

        # Ratio < 1 indicates the loan amount exceeds the goods value (credit risk).
        out["GOODS_CREDIT_RATIO"] = np.where(credit > 0, goods / credit, 0.0)

        # Fraction of credit repaid per payment — lower = longer amortisation, higher interest cost.
        out["payment_rate"] = np.where(credit > 0, annuity / credit, 0.0)

    ratio_cols = ["CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM", "GOODS_CREDIT_RATIO", "payment_rate"]
    for col in ratio_cols:
        out[col] = out[col].replace([np.inf, -np.inf], 0.0).fillna(_NAN_SENTINEL)

    return out


def _engineer_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive age and employment-tenure features from DAYS_BIRTH and DAYS_EMPLOYED.

    DAYS_BIRTH and DAYS_EMPLOYED are stored as negative integers (days before
    application).  DAYS_EMPLOYED == 365243 is a sentinel for unemployment and
    must be clipped to 0 before conversion.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain DAYS_BIRTH and DAYS_EMPLOYED.

    Returns
    -------
    pd.DataFrame
        Copy of df with AGE_YEARS, YEARS_EMPLOYED, EMPLOYED_TO_AGE_RATIO.
    """
    out = df.copy()

    # Convert negative day offsets to positive years.
    age_years = (-out["DAYS_BIRTH"] / 365.25)

    # Clip the unemployment sentinel to 0 before converting.
    days_emp_clipped = out["DAYS_EMPLOYED"].where(
        out["DAYS_EMPLOYED"] != _DAYS_EMPLOYED_SENTINEL, other=0
    )
    # Employment days are negative; negate and cap at 0 to avoid artefacts.
    years_employed = np.maximum(-days_emp_clipped / 365.25, 0.0)

    # Fraction of adult life spent employed; guards against near-zero age.
    employed_to_age = np.where(age_years > 0, years_employed / age_years, 0.0)

    out["AGE_YEARS"] = age_years.fillna(_NAN_SENTINEL)
    out["YEARS_EMPLOYED"] = pd.Series(years_employed, index=out.index).fillna(_NAN_SENTINEL)
    out["EMPLOYED_TO_AGE_RATIO"] = (
        pd.Series(employed_to_age, index=out.index)
        .replace([np.inf, -np.inf], 0.0)
        .fillna(_NAN_SENTINEL)
    )

    return out


def _engineer_documents(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate document-submission flags from FLAG_DOCUMENT_* columns.

    More documents submitted generally signals a more engaged applicant.
    FLAG_DOCUMENT_3 in particular is associated with higher default risk
    when absent (empirical finding from EDA).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain FLAG_DOCUMENT_3 and at least one FLAG_DOCUMENT_* column.

    Returns
    -------
    pd.DataFrame
        Copy of df with DOCUMENTS_SUBMITTED and HIGH_RISK_DOC_MISSING.
    """
    out = df.copy()

    flag_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]

    # Total number of documents the applicant submitted.
    out["DOCUMENTS_SUBMITTED"] = out[flag_cols].sum(axis=1)

    # Missing FLAG_DOCUMENT_3 is a high-risk signal (binary indicator).
    out["HIGH_RISK_DOC_MISSING"] = (out["FLAG_DOCUMENT_3"] == 0).astype(int)

    return out


def _engineer_ext_source(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build composite scores from EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3.

    These are external credit-bureau scores supplied by third parties.
    Their exact semantics are undisclosed, but they are the strongest
    individual predictors in the Home Credit dataset.  High missingness
    (45–55%) is structural — bureaus do not always have records for every
    applicant — so nanmean and nanmin correctly aggregate across the
    available scores without introducing imputation bias.

    Pairwise products let the tree detect "low AND low" joint risk that
    additive terms miss.  EXT_SOURCE_STD measures score inconsistency across
    bureaus — divergent opinions correlate with higher default risk.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3.

    Returns
    -------
    pd.DataFrame
        Copy of df with EXT_SOURCE_{MEAN,MIN,MAX,MEDIAN,STD,RANGE,
        AVAILABLE_CNT,12,13,23}.
    """
    out = df.copy()

    e1 = out["EXT_SOURCE_1"].to_numpy(dtype=float)
    e2 = out["EXT_SOURCE_2"].to_numpy(dtype=float)
    e3 = out["EXT_SOURCE_3"].to_numpy(dtype=float)
    ext = np.column_stack([e1, e2, e3])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ext_mean   = np.nanmean(ext, axis=1)
        ext_min    = np.nanmin(ext, axis=1)
        ext_max    = np.nanmax(ext, axis=1)
        ext_median = np.nanmedian(ext, axis=1)
        ext_std    = np.nanstd(ext, axis=1)

    ext_range     = ext_max - ext_min          # score spread across bureaus
    avail_cnt     = (~np.isnan(ext)).sum(axis=1).astype(float)

    # Pairwise products — NaN if either factor is missing
    prod_12 = np.where(~np.isnan(e1) & ~np.isnan(e2), e1 * e2, np.nan)
    prod_13 = np.where(~np.isnan(e1) & ~np.isnan(e3), e1 * e3, np.nan)
    prod_23 = np.where(~np.isnan(e2) & ~np.isnan(e3), e2 * e3, np.nan)

    def _s(arr: np.ndarray) -> pd.Series:
        return pd.Series(arr, index=out.index)

    out["EXT_SOURCE_MEAN"]          = _s(ext_mean).fillna(_NAN_SENTINEL)
    out["EXT_SOURCE_MIN"]           = _s(ext_min).fillna(_NAN_SENTINEL)
    out["EXT_SOURCE_MAX"]           = _s(ext_max).fillna(_NAN_SENTINEL)
    out["EXT_SOURCE_MEDIAN"]        = _s(ext_median).fillna(_NAN_SENTINEL)
    out["EXT_SOURCE_STD"]           = _s(ext_std).fillna(_NAN_SENTINEL)
    out["EXT_SOURCE_RANGE"]         = _s(ext_range).fillna(_NAN_SENTINEL)
    out["EXT_SOURCE_AVAILABLE_CNT"] = _s(avail_cnt)  # always 0–3, no NaN
    out["EXT_SOURCE_PROD_12"]       = _s(prod_12).fillna(_NAN_SENTINEL)
    out["EXT_SOURCE_PROD_13"]       = _s(prod_13).fillna(_NAN_SENTINEL)
    out["EXT_SOURCE_PROD_23"]       = _s(prod_23).fillna(_NAN_SENTINEL)

    # Polynomial and ratio terms (Task 1.4)
    # Quadratic self-terms capture non-linear risk thresholds — a score of 0.3
    # is not twice as risky as 0.6; the squared term encodes the curvature.
    sq_1 = np.where(~np.isnan(e1), e1 ** 2, np.nan)
    sq_2 = np.where(~np.isnan(e2), e2 ** 2, np.nan)
    # Ratios expose divergence between bureaus — a ratio far from 1 signals
    # that one bureau has very different information about this applicant.
    # Guard: denominator near zero → 0.0 (not a huge number); NaN source → NaN.
    _RATIO_DENOM_FLOOR: float = 1e-4
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_12 = np.where(
            np.isnan(e1) | np.isnan(e2), np.nan,
            np.where(np.abs(e2) > _RATIO_DENOM_FLOOR, e1 / e2, 0.0),
        )
        ratio_23 = np.where(
            np.isnan(e2) | np.isnan(e3), np.nan,
            np.where(np.abs(e3) > _RATIO_DENOM_FLOOR, e2 / e3, 0.0),
        )
    # Joint floor: min × mean — a composite "worst-case average" signal.
    # All-NaN rows produce NaN here; nanmin/nanmean return nan for all-nan input.
    ext_floor = np.where(
        ~np.isnan(ext_min) & ~np.isnan(ext_mean),
        ext_min * ext_mean,
        np.nan,
    )

    out["EXT_SOURCE_1_SQ"]       = _s(sq_1).fillna(_NAN_SENTINEL)
    out["EXT_SOURCE_2_SQ"]       = _s(sq_2).fillna(_NAN_SENTINEL)
    out["EXT_SOURCE_RATIO_12"]   = _s(ratio_12).fillna(_NAN_SENTINEL)
    out["EXT_SOURCE_RATIO_23"]   = _s(ratio_23).fillna(_NAN_SENTINEL)
    out["EXT_SCORE_FLOOR"]       = _s(ext_floor).fillna(_NAN_SENTINEL)

    # Missing-indicator flags — the PRESENCE of a bureau score is itself
    # predictive: thin-file / young applicants often have no EXT_SOURCE_1.
    # These binary columns let the tree model that information explicitly
    # without relying on the sentinel value alone.
    out["EXT_SOURCE_1_missing"]      = _s(np.isnan(e1).astype(float))
    out["EXT_SOURCE_2_missing"]      = _s(np.isnan(e2).astype(float))
    out["EXT_SOURCE_3_missing"]      = _s(np.isnan(e3).astype(float))
    out["ext_source_missing_count"]  = _s(np.isnan(ext).sum(axis=1).astype(float))

    # Triple product: all-three-present "joint good-standing" signal.
    # Distinct from pairwise products — captures the case where all three
    # bureaus simultaneously report a low score (or high).
    prod_123 = np.where(
        ~np.isnan(e1) & ~np.isnan(e2) & ~np.isnan(e3),
        e1 * e2 * e3,
        np.nan,
    )
    out["ext_source_prod"] = _s(prod_123).fillna(_NAN_SENTINEL)

    return out


def engineer_instalment_streaks(df_inst: pd.DataFrame) -> pd.DataFrame:
    """
    Compute instalment time-series features via vectorised groupby operations.

    Captures payment delinquency streaks, recent deterioration, and payment trends
    from instalment-level data. Features link to specific time windows (e.g., last
    6 instalments) and are risk-sensitive (streak length > magnitude).

    Parameters
    ----------
    df_inst : pd.DataFrame
        installments_payments table with columns:
        SK_ID_CURR, DAYS_INSTALMENT, DAYS_ENTRY_PAYMENT, AMT_INSTALMENT, AMT_PAYMENT

        Notes:
        - DAYS_INSTALMENT, DAYS_ENTRY_PAYMENT are negative (days before application).
        - More recent = less negative = higher value.
        - DPD (Days Past Due) = max(0, DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT).

    Returns
    -------
    pd.DataFrame
        One row per SK_ID_CURR (index='SK_ID_CURR') with columns:
        - inst_longest_dpd_streak: max consecutive instalments with DPD > 0
        - inst_months_since_last_dpd: months since most recent DPD; 999 if no DPD
        - inst_payment_amt_slope: OLS slope of AMT_INSTALMENT over time
        - inst_payment_ratio_trend: OLS slope of AMT_PAYMENT / AMT_INSTALMENT
        - inst_recent_vs_historical_dpd: ratio of recent (last 6) to all DPD mean

    All NaN → 0.0 EXCEPT inst_months_since_last_dpd → 999 (far past sentinel).
    inf → 0.0. One-data-point features (single instalment) fill with 0.0.
    """
    # Named constants for readability
    _MONTHS_PAST_SENTINEL: float = 999.0
    _MONTHS_CONVERSION: float = 30.0
    _RECENT_WINDOW: int = 6
    _EPSILON: float = 1e-6

    # Compute DPD = max(0, DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT)
    df = df_inst.copy()
    df["dpd"] = np.maximum(0, df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"])

    results = {}

    # Group by SK_ID_CURR for vectorised processing
    for sk_id, group in df.groupby("SK_ID_CURR", sort=False):
        days_inst = group["DAYS_INSTALMENT"].values
        dpd_vals = group["dpd"].values
        amt_inst = group["AMT_INSTALMENT"].values
        amt_payment = group["AMT_PAYMENT"].values

        # 1. Longest DPD streak: max consecutive instalments with DPD > 0
        # Use cumsum to identify "run" boundaries: DPD > 0 changes from False to True
        is_dpd = (dpd_vals > 0).astype(int)
        # Create run groups: increment counter when is_dpd goes from 0→1
        run_ids = np.cumsum(np.diff(np.concatenate([[0], is_dpd])) > 0)
        if len(is_dpd) > 0:
            # Mask: keep only DPD-positive entries
            run_ids_masked = np.where(is_dpd, run_ids, -1)
            # Count consecutive 1s per run, keeping max
            longest_streak = 0
            if is_dpd.sum() > 0:
                unique_runs = np.unique(run_ids_masked[run_ids_masked >= 0])
                for run_id in unique_runs:
                    streak_len = (run_ids_masked == run_id).sum()
                    longest_streak = max(longest_streak, streak_len)
            results[sk_id] = {
                "inst_longest_dpd_streak": float(longest_streak),
            }
        else:
            results[sk_id] = {"inst_longest_dpd_streak": 0.0}

        # 2. Months since last DPD
        # Find most recent instalment (max DAYS_INSTALMENT) where DPD > 0
        dpd_mask = dpd_vals > 0
        if dpd_mask.any():
            last_dpd_idx = np.argmax(days_inst[dpd_mask])
            last_dpd_day = days_inst[dpd_mask][last_dpd_idx]  # negative value
            most_recent_day = days_inst.max()  # max (least negative)
            # months_ago = abs(most_recent - last_dpd) / 30
            months_ago = abs(most_recent_day - last_dpd_day) / _MONTHS_CONVERSION
            results[sk_id]["inst_months_since_last_dpd"] = months_ago
        else:
            results[sk_id]["inst_months_since_last_dpd"] = _MONTHS_PAST_SENTINEL

        # 3. Payment amount slope: OLS over time (sorted by DAYS_INSTALMENT)
        # Sort by DAYS_INSTALMENT (chronological order)
        sort_idx = np.argsort(days_inst)
        amt_inst_sorted = amt_inst[sort_idx]
        if len(amt_inst_sorted) > 1:
            x = np.arange(len(amt_inst_sorted), dtype=float)
            slope = np.polyfit(x, amt_inst_sorted, 1)[0]
            results[sk_id]["inst_payment_amt_slope"] = slope
        else:
            results[sk_id]["inst_payment_amt_slope"] = 0.0

        # 4. Payment ratio trend: OLS of (AMT_PAYMENT / AMT_INSTALMENT)
        amt_payment_sorted = amt_payment[sort_idx]
        ratio_vals = amt_payment_sorted / (amt_inst_sorted + _EPSILON)
        if len(ratio_vals) > 1:
            slope = np.polyfit(x, ratio_vals, 1)[0]
            results[sk_id]["inst_payment_ratio_trend"] = slope
        else:
            results[sk_id]["inst_payment_ratio_trend"] = 0.0

        # 5. Recent vs historical DPD: mean(last_6_dpd) / (mean(all_dpd) + eps)
        # "Last 6" = 6 most recent instalments (highest DAYS_INSTALMENT values)
        dpd_sorted = dpd_vals[sort_idx]  # Now in chronological order (oldest first)
        if len(dpd_sorted) > 0:
            mean_all_dpd = np.mean(dpd_sorted)
            # Recent = last min(6, len) instalments
            recent_count = min(_RECENT_WINDOW, len(dpd_sorted))
            mean_recent_dpd = np.mean(dpd_sorted[-recent_count:])
            ratio = mean_recent_dpd / (mean_all_dpd + _EPSILON)
            results[sk_id]["inst_recent_vs_historical_dpd"] = ratio
        else:
            results[sk_id]["inst_recent_vs_historical_dpd"] = 0.0

    # Convert to DataFrame indexed by SK_ID_CURR
    result_df = pd.DataFrame.from_dict(results, orient="index")
    result_df.index.name = "SK_ID_CURR"

    # Fill inf → 0.0
    result_df = result_df.replace([np.inf, -np.inf], 0.0)

    # Fill remaining NaN → 0.0 (EXCEPT inst_months_since_last_dpd which uses 999)
    # inst_months_since_last_dpd should already be populated; apply 0 to others
    cols_to_fill_zero = [
        c for c in result_df.columns
        if c != "inst_months_since_last_dpd"
    ]
    result_df[cols_to_fill_zero] = result_df[cols_to_fill_zero].fillna(0.0)
    result_df["inst_months_since_last_dpd"] = (
        result_df["inst_months_since_last_dpd"].fillna(_MONTHS_PAST_SENTINEL)
    )

    return result_df


def engineer_inst_late_rate_12m(df_inst: pd.DataFrame) -> pd.Series:
    """
    Compute fraction of instalments with >30 DPD in the last 365 days.

    Captures recent payment behaviour over a 12-month rolling window.
    Key signal: applicant with recent delinquencies (vs. old ones) is riskier.

    Parameters
    ----------
    df_inst : pd.DataFrame
        installments_payments table with columns:
        SK_ID_CURR, DAYS_INSTALMENT, DAYS_ENTRY_PAYMENT, AMT_INSTALMENT, AMT_PAYMENT.

        DAYS_INSTALMENT and DAYS_ENTRY_PAYMENT are negative (days before application date).
        More recent = less negative.

    Returns
    -------
    pd.Series
        Index = SK_ID_CURR (one row per applicant).
        Values ∈ [0.0, 1.0] (fraction late) or -999.0 (no instalments).
    """
    # DPD = max(0, DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT)
    df = df_inst.copy()
    df["dpd"] = np.maximum(0, df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"])
    df["is_late"] = (df["dpd"] > 30).astype(int)

    # Filter to 12-month window (DAYS_INSTALMENT >= -365)
    df_12m = df[df["DAYS_INSTALMENT"] >= -365].copy()

    # Use vectorized groupby to compute late rate per applicant
    grouped_late = df_12m.groupby("SK_ID_CURR")["is_late"]
    late_rate_12m = grouped_late.sum() / grouped_late.count()

    # Get all unique applicants from the input
    all_ids = df_inst["SK_ID_CURR"].unique()

    # Reindex to include all applicants, fill missing with sentinel
    result = late_rate_12m.reindex(all_ids, fill_value=_NAN_SENTINEL)
    result.index.name = "SK_ID_CURR"

    return result


def engineer_inst_late_rate_recent_vs_historical(df_inst: pd.DataFrame) -> pd.Series:
    """
    Compute trajectory: late rate in 12m window minus all-time late rate.

    Captures whether payment behaviour is worsening (positive) or improving (negative).
    Signal: applicants with worsening trajectory default at higher rates than stagnant delinquents.

    Parameters
    ----------
    df_inst : pd.DataFrame
        installments_payments table (same as engineer_inst_late_rate_12m).

    Returns
    -------
    pd.Series
        Index = SK_ID_CURR.
        Values ∈ [-1.0, 1.0] (late_rate_12m - all_time_late_rate) or -999.0.
    """
    # 12m rate (computed via the function above)
    late_rate_12m = engineer_inst_late_rate_12m(df_inst)

    # All-time rate: compute from entire dataset
    df = df_inst.copy()
    df["dpd"] = np.maximum(0, df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"])
    df["is_late"] = (df["dpd"] > 30).astype(int)

    # Use vectorized groupby to compute all-time late rate per applicant
    grouped_late_all = df.groupby("SK_ID_CURR")["is_late"]
    all_time_rate = grouped_late_all.sum() / grouped_late_all.count()

    # Get all unique applicants from the input
    all_ids = df_inst["SK_ID_CURR"].unique()

    # Reindex to include all applicants, fill missing with sentinel
    all_time_rate = all_time_rate.reindex(all_ids, fill_value=_NAN_SENTINEL)
    all_time_rate.index.name = "SK_ID_CURR"

    # Compute trajectory: 12m - all_time
    # When either rate is -999.0 (missing), the trajectory should also be -999.0
    trajectory = late_rate_12m.sub(all_time_rate, fill_value=_NAN_SENTINEL)

    # Sentinel propagation: if either component was sentinel, result should be sentinel
    mask = (late_rate_12m == _NAN_SENTINEL) | (all_time_rate == _NAN_SENTINEL)
    trajectory[mask] = _NAN_SENTINEL

    return trajectory


def engineer_inst_rolling_30dpd_ratio_3m(df_inst: pd.DataFrame) -> pd.Series:
    """
    Compute fraction of instalments with >30 DPD in the last 90 days.

    Recent delinquency is more predictive of near-term default than historical DPD.

    Parameters
    ----------
    df_inst : pd.DataFrame
        installments_payments table.

    Returns
    -------
    pd.Series
        Index = SK_ID_CURR. Values ∈ [0, 1] or -999.0 sentinel.
        Expected coverage: ~92%.
    """
    df = df_inst.copy()
    df["dpd"] = np.maximum(0, df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"])
    df["is_late"] = (df["dpd"] > 30).astype(int)

    # Filter to 3-month window
    df_3m = df[df["DAYS_INSTALMENT"] >= -90].copy()

    # Use vectorized groupby aggregation
    group_3m = df_3m.groupby("SK_ID_CURR")["is_late"]
    result_dict = {}
    for sk_id, group in df_3m.groupby("SK_ID_CURR"):
        if len(group) == 0:
            result_dict[sk_id] = _NAN_SENTINEL
        else:
            late_cnt = group["is_late"].sum()
            total_cnt = len(group)
            result_dict[sk_id] = float(late_cnt) / float(total_cnt)

    # Ensure all applicants present (fill missing with sentinel)
    all_ids = df_inst["SK_ID_CURR"].unique()
    result = pd.Series(result_dict, dtype=float)
    result = result.reindex(all_ids, fill_value=_NAN_SENTINEL)
    result.index.name = "SK_ID_CURR"
    return result


def engineer_inst_delinquency_escalation_flag(df_inst: pd.DataFrame) -> pd.Series:
    """
    Binary flag: 1 if recent (3m) delinquency rate > historical (6m) rate, else 0.

    Worsening trajectory (escalation) predicts default better than static delinquency.

    Parameters
    ----------
    df_inst : pd.DataFrame
        installments_payments table.

    Returns
    -------
    pd.Series
        Index = SK_ID_CURR. Values ∈ {0, 1, -999}.
        Expected coverage: ~78%.
    """
    df = df_inst.copy()
    df["dpd"] = np.maximum(0, df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"])
    df["is_late"] = (df["dpd"] > 30).astype(int)

    # 3-month window: last 90 days
    df_3m = df[df["DAYS_INSTALMENT"] >= -90].copy()

    # 6-month window: days 90–180 (prior 3 months, non-overlapping)
    df_6m = df[(df["DAYS_INSTALMENT"] >= -180) & (df["DAYS_INSTALMENT"] < -90)].copy()

    # Vectorized: compute rates per applicant using groupby
    # 3m rates
    late_3m = df_3m.groupby("SK_ID_CURR")["is_late"].sum()
    total_3m = df_3m.groupby("SK_ID_CURR").size()
    rate_3m = late_3m / total_3m

    # 6m rates
    late_6m = df_6m.groupby("SK_ID_CURR")["is_late"].sum()
    total_6m = df_6m.groupby("SK_ID_CURR").size()
    rate_6m = late_6m / total_6m

    # Get all applicant IDs
    all_ids = df_inst["SK_ID_CURR"].unique()
    result_series = pd.Series(index=all_ids, dtype=float)

    # Case 1: Both windows present → compare rates
    both_present = result_series.index.isin(rate_3m.index) & result_series.index.isin(rate_6m.index)
    result_series[both_present] = (
        (rate_3m[result_series[both_present].index] > rate_6m[result_series[both_present].index]).astype(float).values
    )

    # Case 2: Only 3m present → flag = 1.0
    only_3m = result_series.index.isin(rate_3m.index) & ~result_series.index.isin(rate_6m.index)
    result_series[only_3m] = 1.0

    # Case 3: Only 6m present → flag = 0.0
    only_6m = ~result_series.index.isin(rate_3m.index) & result_series.index.isin(rate_6m.index)
    result_series[only_6m] = 0.0

    # Case 4: Neither present → _NAN_SENTINEL
    neither = ~result_series.index.isin(rate_3m.index) & ~result_series.index.isin(rate_6m.index)
    result_series[neither] = _NAN_SENTINEL

    result_series.index.name = "SK_ID_CURR"
    return result_series


def engineer_inst_days_since_last_30dpd(df_inst: pd.DataFrame) -> pd.Series:
    """
    Days since the most recent instalment with DPD > 30.

    Recency of delinquency is critical: a default 1 month ago signals higher risk than 12 months ago.

    Parameters
    ----------
    df_inst : pd.DataFrame
        installments_payments table.

    Returns
    -------
    pd.Series
        Index = SK_ID_CURR. Values: days ≥ 0, or -1 (never late), or -999 (no data).
        Expected coverage: ~87%.
    """
    df = df_inst.copy()
    df["dpd"] = np.maximum(0, df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"])

    # Vectorized approach: compute for each applicant in one pass
    # Step 1: Find most recent day (max DAYS_INSTALMENT) for each SK_ID
    most_recent_by_id = df.groupby("SK_ID_CURR")["DAYS_INSTALMENT"].max()

    # Step 2: For DPD > 30 records, find most recent day (max DAYS_INSTALMENT)
    df_late = df[df["dpd"] > 30].copy()
    last_late_by_id = df_late.groupby("SK_ID_CURR")["DAYS_INSTALMENT"].max()

    # Step 3: Check which applicants ever had DPD > 30
    has_late = df.groupby("SK_ID_CURR")["dpd"].apply(lambda x: (x > 30).any())

    # Step 4: Compute result using vectorized logic
    result_series = pd.Series(index=most_recent_by_id.index, dtype=float)

    # Case 1: Never had DPD > 30 → -1.0
    result_series[~has_late] = -1.0

    # Case 2: Had DPD > 30 → days since
    late_ids = has_late[has_late].index
    result_series[late_ids] = (
        most_recent_by_id[late_ids] - last_late_by_id[late_ids]
    ).astype(float)

    result_series.index.name = "SK_ID_CURR"
    return result_series


def engineer_bureau_dpd_trend_3m_vs_12m(df: pd.DataFrame) -> pd.Series:
    """
    Bureau DPD trend: recent (3m) rate minus historical (3-12m) rate.

    Captures whether the applicant's bureau delinquency is worsening or improving.
    Positive trend = worsening (higher risk); negative = improving.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns bureau_bbal_dpd_rate_3m_mean and
        bureau_bbal_dpd_rate_3m_to_12m_mean (pre-aggregated from bureau_balance
        by data_loader.py).

    Returns
    -------
    pd.Series
        Index = df.index. Values ∈ [-1, 1] or -999.0 sentinel.
        Expected coverage: ~97.2%.
    """
    if "bureau_bbal_dpd_rate_3m_mean" not in df.columns or "bureau_bbal_dpd_rate_3m_to_12m_mean" not in df.columns:
        # Return sentinel if columns missing
        return pd.Series(_NAN_SENTINEL, index=df.index, dtype=float)

    rate_3m = df["bureau_bbal_dpd_rate_3m_mean"]
    rate_3m_to_12m = df["bureau_bbal_dpd_rate_3m_to_12m_mean"]

    # Compute trend only where both inputs are valid (not NaN)
    # When either input is NaN, the result will be NaN and get filled with sentinel below
    trend = rate_3m - rate_3m_to_12m

    # Replace inf → -999
    trend = trend.replace([np.inf, -np.inf], _NAN_SENTINEL)

    # Clip to [-1, 1] (both rates are [0, 1], so diff is [-1, 1])
    trend = trend.clip(lower=-1.0, upper=1.0)

    # Fill remaining NaN → -999
    trend = trend.fillna(_NAN_SENTINEL)

    return trend


def engineer_bureau_debt_to_new_credit(df: pd.DataFrame) -> pd.Series:
    """
    Bureau outstanding debt relative to new loan amount.

    Measures how much new credit the applicant is taking relative to existing debt burden.
    High ratio = applicant already heavily indebted; new credit compounds leverage risk.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns bureau_credit_debt_sum and AMT_CREDIT.

    Returns
    -------
    pd.Series
        Index = df.index. Values ≥ 0 or -999.0 sentinel.
        Expected coverage: ~85%.
    """
    if "bureau_credit_debt_sum" not in df.columns or "AMT_CREDIT" not in df.columns:
        return pd.Series(_NAN_SENTINEL, index=df.index, dtype=float)

    debt = df["bureau_credit_debt_sum"]
    credit = df["AMT_CREDIT"]

    # Track which rows had NaN in EITHER input — both missing sources require sentinel
    debt_missing = debt.isna()
    credit_missing = credit.isna()
    either_missing = debt_missing | credit_missing

    # Fill NaN with safe defaults for division only — originals restored via either_missing mask
    debt_filled = debt.fillna(0.0).clip(lower=0.0)
    credit_filled = credit.fillna(0.0).clip(lower=1.0)  # Ensure no division by zero

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = debt_filled / credit_filled

    # Replace inf → -999, NaN → -999
    ratio = pd.Series(ratio, index=df.index, dtype=float)
    ratio = ratio.replace([np.inf, -np.inf], _NAN_SENTINEL)
    ratio = ratio.fillna(_NAN_SENTINEL)

    # Restore sentinel for any row where debt OR credit was originally missing
    ratio[either_missing] = _NAN_SENTINEL

    return ratio


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def engineer_secondary_features(
    df: pd.DataFrame,
    df_inst: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Derive ratio and flag features from secondary table aggregates.

    These features are computed from columns already aggregated by
    data_loader.py (bureau_, prev_, pos_, inst_, cc_ prefixes).
    They add interaction/ratio signal that univariate IV filtering misses.

    Optionally accepts raw instalment-level data to compute time-series features
    (streaks, trends, recent vs historical). If df_inst is provided, the output
    DataFrame will include the 5 instalment streak features.

    All divisions are guarded: if the denominator is 0 the ratio is set to 0.
    Residual inf values are replaced with 0. Remaining NaN is filled with
    _NAN_SENTINEL.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing secondary table aggregate columns (indexed by SK_ID_CURR).
    df_inst : pd.DataFrame | None
        Optional raw instalment_payments table with SK_ID_CURR, DAYS_INSTALMENT,
        DAYS_ENTRY_PAYMENT, AMT_INSTALMENT, AMT_PAYMENT columns. If provided,
        engineer_instalment_streaks() is applied and merged onto df.

    Returns
    -------
    pd.DataFrame
        New DataFrame with all input columns plus derived secondary features.
        No input columns are modified. NaN → _NAN_SENTINEL. inf → 0.

    Features added (if input columns exist):
    - prev_approval_rate: prev_approved_cnt / max(prev_cnt, 1)
    - prev_refusal_rate: prev_refused_cnt / max(prev_cnt, 1)
    - inst_pct_late: inst_late_cnt / max(inst_cnt, 1)
    - inst_late_dpd_ratio: inst_days_past_due_max / (|inst_days_past_due_mean| + 1)
    - bureau_debt_ratio: bureau_credit_debt_sum / max(bureau_credit_sum, 1e-6)
    - bureau_overdue_rate: bureau_overdue_cnt / max(bureau_cnt, 1)
    - bureau_active_ratio: bureau_active_cnt / max(bureau_cnt, 1)
    - bureau_debt_to_income: bureau_credit_debt_sum / max(AMT_INCOME_TOTAL, 1)
    - cc_overdue_flag: (cc_sk_dpd_max > 0).astype(float) [0 if missing]
    - pos_overdue_flag: (pos_sk_dpd_max > 0).astype(float)
    - prev_credit_income_ratio: prev_amt_credit_mean / max(AMT_INCOME_TOTAL, 1), clipped to [0, 100]
    - debt_service_ratio: AMT_ANNUITY / (AMT_INCOME_TOTAL / 12), monthly debt burden

    If df_inst is provided:
    - inst_longest_dpd_streak: max consecutive payments with DPD > 0
    - inst_months_since_last_dpd: months since most recent DPD; 999 if none
    - inst_payment_amt_slope: OLS slope of instalment amounts over time
    - inst_payment_ratio_trend: OLS slope of payment ratio over time
    - inst_recent_vs_historical_dpd: ratio of recent (last 6) to all DPD mean
    """
    out = df.copy()

    # 1. prev_approval_rate: approval history
    if "prev_cnt" in out.columns and "prev_approved_cnt" in out.columns:
        prev_cnt = out["prev_cnt"].to_numpy(dtype=float)
        prev_approved = out["prev_approved_cnt"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["prev_approval_rate"] = np.where(
                prev_cnt > 0, prev_approved / prev_cnt, 0.0
            )
        out["prev_approval_rate"] = (
            out["prev_approval_rate"]
            .replace([np.inf, -np.inf], 0.0)
            .fillna(_NAN_SENTINEL)
        )

    # 2. inst_pct_late: fraction of late payments
    if "inst_cnt" in out.columns and "inst_late_cnt" in out.columns:
        inst_cnt = out["inst_cnt"].to_numpy(dtype=float)
        inst_late = out["inst_late_cnt"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["inst_pct_late"] = np.where(inst_cnt > 0, inst_late / inst_cnt, 0.0)
        out["inst_pct_late"] = (
            out["inst_pct_late"]
            .replace([np.inf, -np.inf], 0.0)
            .fillna(_NAN_SENTINEL)
        )

    # 3. bureau_debt_ratio: bureau leverage
    if "bureau_credit_sum" in out.columns and "bureau_credit_debt_sum" in out.columns:
        bureau_sum = out["bureau_credit_sum"].to_numpy(dtype=float)
        bureau_debt = out["bureau_credit_debt_sum"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["bureau_debt_ratio"] = np.where(
                bureau_sum > 1e-6, bureau_debt / bureau_sum, 0.0
            )
        out["bureau_debt_ratio"] = (
            out["bureau_debt_ratio"]
            .replace([np.inf, -np.inf], 0.0)
            .fillna(_NAN_SENTINEL)
        )

    # 4. cc_overdue_flag: any credit card overdue
    if "cc_sk_dpd_max" in out.columns:
        cc_dpd = out["cc_sk_dpd_max"].fillna(_NAN_SENTINEL)
        # If sentinel, treat as no overdue; otherwise check if > 0
        out["cc_overdue_flag"] = (cc_dpd > 0).astype(np.float64)
        out["cc_overdue_flag"] = out["cc_overdue_flag"].fillna(0.0)

    # 5. pos_overdue_flag: any POS overdue
    if "pos_sk_dpd_max" in out.columns:
        pos_dpd = out["pos_sk_dpd_max"].fillna(_NAN_SENTINEL)
        out["pos_overdue_flag"] = (pos_dpd > 0).astype(np.float64)
        out["pos_overdue_flag"] = out["pos_overdue_flag"].fillna(0.0)

    # 6. prev_credit_income_ratio: prior loan size vs current income
    if (
        "prev_amt_credit_mean" in out.columns
        and "AMT_INCOME_TOTAL" in out.columns
    ):
        prev_credit = out["prev_amt_credit_mean"].to_numpy(dtype=float)
        income = out["AMT_INCOME_TOTAL"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["prev_credit_income_ratio"] = np.where(
                income > 0, prev_credit / income, 0.0
            )
        # Clip to [0, 100] to handle outliers
        out["prev_credit_income_ratio"] = (
            out["prev_credit_income_ratio"]
            .replace([np.inf, -np.inf], 0.0)
            .clip(lower=0, upper=100)
            .fillna(_NAN_SENTINEL)
        )

    # 7. prev_refusal_rate: proportion of previous applications refused
    if "prev_cnt" in out.columns and "prev_refused_cnt" in out.columns:
        prev_cnt = out["prev_cnt"].to_numpy(dtype=float)
        prev_refused = out["prev_refused_cnt"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["prev_refusal_rate"] = np.where(
                prev_cnt > 0, prev_refused / np.maximum(prev_cnt, 1e-10), 0.0
            )
        out["prev_refusal_rate"] = (
            out["prev_refusal_rate"].replace([np.inf, -np.inf], 0.0).fillna(_NAN_SENTINEL)
        )

    # 8. inst_late_dpd_ratio: max DPD scaled by mean DPD (escalation signal)
    if (
        "inst_days_past_due_max" in out.columns
        and "inst_days_past_due_mean" in out.columns
    ):
        denom = (out["inst_days_past_due_mean"].abs() + 1.0).clip(lower=1.0)
        out["inst_late_dpd_ratio"] = (
            out["inst_days_past_due_max"] / denom
        ).replace([np.inf, -np.inf], 0.0).fillna(_NAN_SENTINEL)

    # 9. bureau_overdue_rate: fraction of bureau accounts ever overdue
    if "bureau_cnt" in out.columns and "bureau_overdue_cnt" in out.columns:
        bureau_cnt = out["bureau_cnt"].to_numpy(dtype=float)
        overdue_cnt = out["bureau_overdue_cnt"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["bureau_overdue_rate"] = np.where(
                bureau_cnt > 0, overdue_cnt / np.maximum(bureau_cnt, 1e-10), 0.0
            )
        out["bureau_overdue_rate"] = (
            out["bureau_overdue_rate"].replace([np.inf, -np.inf], 0.0).fillna(_NAN_SENTINEL)
        )

    # 10. bureau_active_ratio: share of bureau accounts currently active
    if "bureau_cnt" in out.columns and "bureau_active_cnt" in out.columns:
        bureau_cnt = out["bureau_cnt"].to_numpy(dtype=float)
        active_cnt = out["bureau_active_cnt"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["bureau_active_ratio"] = np.where(
                bureau_cnt > 0, active_cnt / np.maximum(bureau_cnt, 1e-10), 0.0
            )
        out["bureau_active_ratio"] = (
            out["bureau_active_ratio"].replace([np.inf, -np.inf], 0.0).fillna(_NAN_SENTINEL)
        )

    # 11. bureau_debt_to_income: total outstanding debt relative to declared income
    if "bureau_credit_debt_sum" in out.columns and "AMT_INCOME_TOTAL" in out.columns:
        income = out["AMT_INCOME_TOTAL"].clip(lower=1.0).to_numpy(dtype=float)
        debt = out["bureau_credit_debt_sum"].clip(lower=0.0).to_numpy(dtype=float)
        out["bureau_debt_to_income"] = pd.Series(
            debt / income, index=out.index
        ).clip(upper=50.0).fillna(_NAN_SENTINEL)

    # 12. debt_service_ratio: monthly instalment as fraction of monthly income
    if "AMT_ANNUITY" in out.columns and "AMT_INCOME_TOTAL" in out.columns:
        monthly_income = out["AMT_INCOME_TOTAL"].clip(lower=1.0) / 12.0
        with np.errstate(divide="ignore", invalid="ignore"):
            out["debt_service_ratio"] = (
                out["AMT_ANNUITY"] / monthly_income
            ).replace([np.inf, -np.inf], 0.0).clip(upper=10.0).fillna(_NAN_SENTINEL)

    # -----------------------------------------------------------------------
    # Cross-table interaction features (Task 1.2)
    # -----------------------------------------------------------------------

    # 13. ext_credit_risk: low bureau score × high credit leverage
    # Multiplicative cross-feature: the most cited single interaction in
    # competition solutions — captures the double-risk of a bad external
    # score combined with an already-stretched debt-to-income ratio.
    if "EXT_SOURCE_MEAN" in out.columns and "CREDIT_INCOME_RATIO" in out.columns:
        out["ext_credit_risk"] = (
            out["EXT_SOURCE_MEAN"].clip(lower=0.0)
            * out["CREDIT_INCOME_RATIO"].clip(lower=0.0)
        ).fillna(_NAN_SENTINEL)

    # 14. ext_annuity_risk: worst bureau score × annuity repayment strain
    if "EXT_SOURCE_MIN" in out.columns and "ANNUITY_INCOME_RATIO" in out.columns:
        out["ext_annuity_risk"] = (
            out["EXT_SOURCE_MIN"].clip(lower=0.0)
            * out["ANNUITY_INCOME_RATIO"].clip(lower=0.0)
        ).fillna(_NAN_SENTINEL)

    # 15. multi_dpd_flag: late on 2+ credit products simultaneously
    # Boolean AND across installments and credit card delinquency channels —
    # cross-product defaults are a strong predictor of imminent default.
    if "inst_pct_late" in out.columns and "cc_dpd_rate" in out.columns:
        out["multi_dpd_flag"] = (
            (out["inst_pct_late"] > 0.1) & (out["cc_dpd_rate"] > 0.1)
        ).astype(np.float32).fillna(0.0)
    elif "bureau_overdue_rate" in out.columns and "inst_days_past_due_mean" in out.columns:
        # Fallback when cc_dpd_rate is absent: bureau × installments
        out["multi_dpd_flag"] = (
            (out["bureau_overdue_rate"] > 0.1) & (out["inst_days_past_due_mean"] > 0.0)
        ).astype(np.float32).fillna(0.0)

    # 16. bureau_inst_dpd: bureau overdue rate × mean days past due on instalments
    # Severity (mean DPD days) weighted by frequency (overdue rate) — a single
    # number that summarises how bad AND how often the borrower misses payments.
    if "bureau_overdue_rate" in out.columns and "inst_days_past_due_mean" in out.columns:
        out["bureau_inst_dpd"] = (
            out["bureau_overdue_rate"].clip(lower=0.0)
            * out["inst_days_past_due_mean"].clip(lower=0.0)
        ).fillna(_NAN_SENTINEL)

    # 17. total_debt_exposure: all outstanding debt relative to income
    _has_debt = "bureau_credit_debt_sum" in out.columns and "cc_bal_max" in out.columns
    _has_income = "AMT_INCOME_TOTAL" in out.columns
    if _has_debt and _has_income:
        debt = (
            out["bureau_credit_debt_sum"].clip(lower=0.0)
            + out["cc_bal_max"].clip(lower=0.0)
        )
        income = out["AMT_INCOME_TOTAL"].clip(lower=1.0)
        out["total_debt_exposure"] = (debt / income).clip(lower=0.0, upper=100.0).fillna(_NAN_SENTINEL)

    # 18. leverage_vs_bureau: application leverage amplified by existing bureau debt
    # bureau_debt_to_income must be present (computed in feature 11 above).
    if "bureau_debt_to_income" in out.columns and "CREDIT_INCOME_RATIO" in out.columns:
        out["leverage_vs_bureau"] = (
            out["bureau_debt_to_income"].clip(lower=0.0)
            * out["CREDIT_INCOME_RATIO"].clip(lower=0.0)
        ).clip(lower=0.0).fillna(_NAN_SENTINEL)

    # 19. dpd_trajectory: recent DPD rate minus historical rate — positive = worsening
    _has_6m = "bureau_bbal_dpd_rate_6m_mean" in out.columns
    _has_12m = "bureau_bbal_dpd_rate_12m_mean" in out.columns
    if _has_6m and _has_12m:
        out["dpd_trajectory"] = (
            out["bureau_bbal_dpd_rate_6m_mean"].fillna(0.0)
            - out["bureau_bbal_dpd_rate_12m_mean"].fillna(0.0)
        ).fillna(_NAN_SENTINEL)

    # 20. dpd_escalation: trajectory severity weighted by maximum DPD ever seen
    # Combines direction of delinquency trend with peak severity.
    if "dpd_trajectory" in out.columns and "inst_days_past_due_max" in out.columns:
        out["dpd_escalation"] = (
            out["dpd_trajectory"].clip(lower=0.0)
            * out["inst_days_past_due_max"].clip(lower=0.0)
        ).fillna(_NAN_SENTINEL)

    # 21. debt_service_coverage: income divided by total annuity obligations
    # Higher = more breathing room; low coverage (<1) signals affordability stress.
    _has_annuity = "AMT_ANNUITY" in out.columns
    _has_bureau_annuity = "bureau_annuity_mean" in out.columns
    if _has_income and _has_annuity and _has_bureau_annuity:
        total_annuity = (
            out["AMT_ANNUITY"].clip(lower=0.0)
            + out["bureau_annuity_mean"].clip(lower=0.0)
            + 1.0
        )
        out["debt_service_coverage"] = (
            out["AMT_INCOME_TOTAL"].clip(lower=0.0) / total_annuity
        ).replace([np.inf, -np.inf], 0.0).clip(lower=0.0).fillna(_NAN_SENTINEL)
    elif _has_income and _has_annuity:
        out["debt_service_coverage"] = (
            out["AMT_INCOME_TOTAL"].clip(lower=0.0) / (out["AMT_ANNUITY"].clip(lower=0.0) + 1.0)
        ).replace([np.inf, -np.inf], 0.0).clip(lower=0.0).fillna(_NAN_SENTINEL)

    # -----------------------------------------------------------------------
    # Secondary & Cross-table Features (Phase 04.2.3.2, D-07 through D-19)
    # -----------------------------------------------------------------------

    # D-07: no_bureau_history — thin-file indicator
    if "bureau_cnt" in out.columns:
        out["no_bureau_history"] = (out["bureau_cnt"] == 0).astype(int)

    # D-08: no_prev_applications — thin-file indicator
    if "prev_cnt" in out.columns:
        out["no_prev_applications"] = (out["prev_cnt"] == 0).astype(int)

    # D-09: ever_dpd_bureau — ever had overdue in bureau history
    if "bureau_overdue_cnt" in out.columns:
        out["ever_dpd_bureau"] = (out["bureau_overdue_cnt"] > 0).astype(int)

    # D-10: bureau_prolong_any — debt restructuring signal
    if "bureau_prolong_sum" in out.columns:
        out["bureau_prolong_any"] = (out["bureau_prolong_sum"] > 0).astype(int)

    # D-11: high_credit_income — overstretched indicator
    if "CREDIT_INCOME_RATIO" in out.columns:
        out["high_credit_income"] = (out["CREDIT_INCOME_RATIO"] > 5).astype(int)

    # D-12: low_payment_rate — near-minimum payment indicator
    if "payment_rate" in out.columns:
        out["low_payment_rate"] = (out["payment_rate"] < 0.03).astype(int)

    # D-13: thin_file (REGULATORY REFRAME) — replace thin_file_young
    # thin_file_young (age < 30 AND no_bureau_history) is PROHIBITED under EU AI Act Art. 6
    if "no_bureau_history" in out.columns:
        out["thin_file"] = out["no_bureau_history"].astype(int)

    # D-14: new_credit_to_bureau_ratio — current credit vs existing bureau total
    if "AMT_CREDIT" in out.columns and "bureau_credit_sum" in out.columns:
        bureau_sum = out["bureau_credit_sum"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["new_credit_to_bureau_ratio"] = np.where(
                bureau_sum > 0, out["AMT_CREDIT"].to_numpy(dtype=float) / bureau_sum, np.nan
            )
        out["new_credit_to_bureau_ratio"] = (
            out["new_credit_to_bureau_ratio"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(_NAN_SENTINEL)
        )

    # D-15: annuity_to_prev_annuity_ratio — compare current annuity to historical average
    if "AMT_ANNUITY" in out.columns and "prev_amt_annuity_mean" in out.columns:
        prev_annuity = out["prev_amt_annuity_mean"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["annuity_to_prev_annuity_ratio"] = np.where(
                prev_annuity > 0, out["AMT_ANNUITY"].to_numpy(dtype=float) / prev_annuity, np.nan
            )
        out["annuity_to_prev_annuity_ratio"] = (
            out["annuity_to_prev_annuity_ratio"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(_NAN_SENTINEL)
        )

    # D-16: bureau_overdue_to_income — normalized overdue stress
    if "bureau_overdue_sum" in out.columns and "AMT_INCOME_TOTAL" in out.columns:
        income = out["AMT_INCOME_TOTAL"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["bureau_overdue_to_income"] = np.where(
                income > 0, out["bureau_overdue_sum"].to_numpy(dtype=float) / income, np.nan
            )
        out["bureau_overdue_to_income"] = (
            out["bureau_overdue_to_income"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(_NAN_SENTINEL)
        )

    # D-17: bureau_active_to_prev_apps — ratio of active bureau accounts to previous applications
    if "bureau_active_cnt" in out.columns and "prev_cnt" in out.columns:
        prev_cnt = out["prev_cnt"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["bureau_active_to_prev_apps"] = np.where(
                prev_cnt > 0, out["bureau_active_cnt"].to_numpy(dtype=float) / prev_cnt, np.nan
            )
        out["bureau_active_to_prev_apps"] = (
            out["bureau_active_to_prev_apps"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(_NAN_SENTINEL)
        )

    # D-18: cc_utilisation_to_income — credit card spending relative to income
    if "cc_utilisation_mean" in out.columns and "cc_bal_max" in out.columns and "AMT_INCOME_TOTAL" in out.columns:
        income = out["AMT_INCOME_TOTAL"].to_numpy(dtype=float)
        util = out["cc_utilisation_mean"].to_numpy(dtype=float)
        cc_bal = out["cc_bal_max"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["cc_utilisation_to_income"] = np.where(
                income > 0, (util * cc_bal) / income, np.nan
            )
        out["cc_utilisation_to_income"] = (
            out["cc_utilisation_to_income"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(_NAN_SENTINEL)
        )

    # D-19: bureau_close_rate — ratio of closed to total bureau loans
    if "bureau_closed_cnt" in out.columns and "bureau_cnt" in out.columns:
        bureau_cnt = out["bureau_cnt"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["bureau_close_rate"] = np.where(
                bureau_cnt > 0, out["bureau_closed_cnt"].to_numpy(dtype=float) / bureau_cnt, np.nan
            )
        out["bureau_close_rate"] = (
            out["bureau_close_rate"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(_NAN_SENTINEL)
        )

    # Optional: merge instalment time-series features if raw instalment data is provided
    if df_inst is not None:
        inst_streaks = engineer_instalment_streaks(df_inst)
        out = out.join(inst_streaks, how="left")
        # Fill any missing streak features (e.g., borrowers with no instalments) with 0.0
        streak_cols = inst_streaks.columns.tolist()
        for col in streak_cols:
            if col in out.columns:
                out[col] = out[col].fillna(0.0)

    return out


def engineer_application_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain-engineered features derived from the application table.

    Applies four groups of transformations in sequence:
    financial ratios → demographics → document flags → EXT_SOURCE composites.
    All transformations are immutable (a copy is returned; the input is unchanged).

    Parameters
    ----------
    df : pd.DataFrame
        Raw application DataFrame.  Expected columns include AMT_CREDIT,
        AMT_INCOME_TOTAL, AMT_ANNUITY, AMT_GOODS_PRICE, DAYS_BIRTH,
        DAYS_EMPLOYED, EXT_SOURCE_1/2/3, FLAG_DOCUMENT_*.

    Returns
    -------
    pd.DataFrame
        Copy of df with 19 additional engineered columns:
        CREDIT_INCOME_RATIO, ANNUITY_INCOME_RATIO, CREDIT_TERM,
        GOODS_CREDIT_RATIO, AGE_YEARS, YEARS_EMPLOYED,
        EMPLOYED_TO_AGE_RATIO, DOCUMENTS_SUBMITTED,
        HIGH_RISK_DOC_MISSING, EXT_SOURCE_{MEAN,MIN,MAX,MEDIAN,STD,RANGE,
        AVAILABLE_CNT,PROD_12,PROD_13,PROD_23}.

    Notes
    -----
    EXT_SOURCE features are the strongest predictors in this dataset
    despite undisclosed semantics.  Because their missingness is structural
    (external bureaus do not always have records), nanmean/nanmin avoid
    imputation bias.  Rows where all three sources are absent receive the
    -999 sentinel so tree models can learn a dedicated "no bureau data" split.
    """
    result = _engineer_financial_ratios(df)
    result = _engineer_demographics(result)
    result = _engineer_documents(result)
    result = _engineer_ext_source(result)

    # Drop raw source columns that are fully superseded by engineered equivalents.
    # DAYS_BIRTH → AGE_YEARS (r = 1.0 by construction; keeping both is pure redundancy).
    cols_to_drop = [c for c in ["DAYS_BIRTH"] if c in result.columns]
    return result.drop(columns=cols_to_drop)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full feature engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (typically the joined training frame from data_loader).

    Returns
    -------
    pd.DataFrame
        DataFrame with all engineered features appended.
    """
    result = df
    # Phase 2.1: application-table domain features
    # Only apply if the application-table columns are present.
    if "AMT_CREDIT" in result.columns:
        result = engineer_application_features(result)
    return result


# ---------------------------------------------------------------------------
# Private helpers — WoE / IV computation
# ---------------------------------------------------------------------------


def _compute_binning_table(
    series: pd.Series,
    target: pd.Series,
    bins: int = 10,
) -> tuple[pd.DataFrame, float] | None:
    """
    Compute the per-bin WoE and IV contribution table for a single numeric feature.

    Option-B sentinel handling: rows where ``series == _NAN_SENTINEL`` (or actual
    ``np.nan``) are separated from the quantile bins and reported as a dedicated
    ``'-999 (missing)'`` row.  This is the IRB-standard treatment; mixing missing
    values into the leftmost quantile bin would dilute the WoE signal for both
    populations.

    Parameters
    ----------
    series : pd.Series
        Numeric feature values aligned with ``target`` (same index).
    target : pd.Series
        Binary response (1 = event / default, 0 = non-event).
    bins : int, optional
        Number of quantile buckets for non-sentinel values (default 10).

    Returns
    -------
    tuple[pd.DataFrame, float] or None
        ``(binning_table, total_iv)`` where ``binning_table`` has columns
        ``[bin_range, event_count, non_event_count, woe, iv_contrib]``, or
        ``None`` when the feature carries no information (constant, all-missing).

    Notes
    -----
    Actual ``np.nan`` values are coerced to ``_NAN_SENTINEL`` before binning so
    that raw (un-imputed) features are handled identically to features that were
    already sentinel-filled in Phase 2.1.
    """
    # Coerce actual NaN → sentinel so both kinds of missing are handled uniformly.
    series_filled = series.fillna(_NAN_SENTINEL)

    # Constant or entirely-missing features carry no discriminatory information.
    if series_filled.nunique() <= 1:
        return None

    sentinel_mask: pd.Series = series_filled == _NAN_SENTINEL
    non_s: pd.Series = series_filled[~sentinel_mask]
    non_s_target: pd.Series = target[~sentinel_mask]

    bins_rows: list[dict] = []

    if len(non_s) > 0:
        if non_s.nunique() > 1:
            try:
                binned = pd.qcut(non_s, q=bins, duplicates="drop")
                # Strip index before DataFrame construction to prevent NaN
                # injection from index misalignment after boolean subsetting.
                temp = pd.DataFrame(
                    {"bin": binned.to_numpy(), "y": non_s_target.to_numpy()}
                )
                for bin_label, grp in temp.groupby("bin", observed=True):
                    bins_rows.append(
                        {
                            "bin_range": str(bin_label),
                            "event_count": int(grp["y"].sum()),
                            "non_event_count": int((grp["y"] == 0).sum()),
                        }
                    )
            except ValueError:
                pass  # qcut failed (e.g., too few unique values after drop)
        else:
            # Single unique value in the non-sentinel portion — one bin.
            bins_rows.append(
                {
                    "bin_range": f"constant ({non_s.iloc[0]:.6g})",
                    "event_count": int(non_s_target.sum()),
                    "non_event_count": int((non_s_target == 0).sum()),
                }
            )

    # Sentinel / missing bin (Option B).
    if sentinel_mask.any():
        s_target = target[sentinel_mask]
        bins_rows.append(
            {
                "bin_range": f"{int(_NAN_SENTINEL)} (missing)",
                "event_count": int(s_target.sum()),
                "non_event_count": int((s_target == 0).sum()),
            }
        )

    if not bins_rows:
        return None

    tbl = pd.DataFrame(bins_rows)
    total_e = int(tbl["event_count"].sum())
    total_ne = int(tbl["non_event_count"].sum())

    # A feature where all observations are events (or all non-events) would
    # make dist_events or dist_non_events 0 for every bin — skip it.
    if total_e == 0 or total_ne == 0:
        return None

    woe_vals: list[float] = []
    iv_vals: list[float] = []
    for _, row in tbl.iterrows():
        de = row["event_count"] / total_e
        dne = row["non_event_count"] / total_ne
        if de == 0.0:
            w = _WOE_CLIP        # no defaults in this bin → very safe
        elif dne == 0.0:
            w = -_WOE_CLIP       # all defaults in this bin → very risky
        else:
            w = float(np.clip(np.log(dne / de), -_WOE_CLIP, _WOE_CLIP))
        woe_vals.append(w)
        iv_vals.append((dne - de) * w)

    tbl = tbl.assign(woe=woe_vals, iv_contrib=iv_vals)
    total_iv = float(sum(iv_vals))
    return tbl[["bin_range", "event_count", "non_event_count", "woe", "iv_contrib"]], total_iv


# ---------------------------------------------------------------------------
# Public API — WoE / IV
# ---------------------------------------------------------------------------


def compute_woe_iv(
    df: pd.DataFrame,
    feature: str,
    target: pd.Series,
    bins: int = 10,
) -> tuple[pd.DataFrame, float]:
    """
    Compute Weight of Evidence (WoE) and Information Value (IV) for one feature.

    WoE per bin is defined as::

        WoE_i = ln(dist_non_events_i / dist_events_i)

    where ``dist_events_i = events_in_bin / total_events`` and similarly for
    non-events.  WoE > 0 indicates fewer defaults than average (lower risk);
    WoE < 0 indicates more defaults than average (higher risk).

    IV = sum_i[ (dist_non_events_i - dist_events_i) × WoE_i ] is always ≥ 0
    and measures the total discriminatory power of the feature.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``feature``.
    feature : str
        Column name of the numeric feature to bin.
    target : pd.Series
        Binary response aligned with ``df`` (1 = default, 0 = non-default).
    bins : int, optional
        Number of quantile buckets (default 10).  ``pd.qcut`` with
        ``duplicates='drop'`` may produce fewer bins for sparse distributions.

    Returns
    -------
    binning_table : pd.DataFrame
        Columns: ``bin_range``, ``event_count``, ``non_event_count``,
        ``woe``, ``iv_contrib``.  Empty DataFrame with these columns when the
        feature carries no information (constant, all-missing).
    total_iv : float
        Sum of ``iv_contrib`` across all bins (0.0 for uninformative features).

    Notes
    -----
    **Sentinel handling (Option B, IRB standard):** rows where
    ``feature == -999`` or ``feature is NaN`` are placed in a dedicated
    ``'-999 (missing)'`` bin *before* quantile binning, so structural
    missingness is modelled independently of true feature values.

    **WoE clipping:** extreme bins (zero events or zero non-events) are clipped
    to ±5 rather than using Laplace smoothing.  This is the standard production
    approach for tree-model pipelines (LightGBM / XGBoost) where WoE is used
    for feature selection rather than as a direct model input.

    **Regulatory context:** WoE supports adverse-action notices under GDPR
    Art. 22 and EU AI Act Art. 6 because each bin's WoE has a direct log-odds
    interpretation that non-technical reviewers can verify.
    """
    _EMPTY = pd.DataFrame(
        columns=["bin_range", "event_count", "non_event_count", "woe", "iv_contrib"]
    )
    series = df[feature].reset_index(drop=True)
    aligned_target = target.reset_index(drop=True)
    result = _compute_binning_table(series, aligned_target, bins)
    if result is None:
        return _EMPTY, 0.0
    return result


# ---------------------------------------------------------------------------
# Private helpers — feature store (WoE mapping extraction + transformation)
# ---------------------------------------------------------------------------


def _compute_woe_mapping_dict(
    series: pd.Series,
    binning_table: pd.DataFrame,
    bins: int,
) -> dict:
    """
    Build the WoE mapping entry for one feature.

    Returns a dict with two keys:

    ``bin_edges``
        Sorted list of float cut-points (including left/right extremes) that
        can be passed to ``pd.cut(..., bins=bin_edges)`` at inference time.
        The sentinel bin ``'-999 (missing)'`` is handled separately and is
        never included in ``bin_edges``.

    ``bin_woe_values``
        Dict mapping bin-label string (as produced by ``pd.cut``) → WoE float.
        Also includes the ``'-999 (missing)'`` key when sentinel rows are present.

    Parameters
    ----------
    series : pd.Series
        Non-sentinel values of the feature (sentinel rows already excluded).
    binning_table : pd.DataFrame
        Output of ``_compute_binning_table``: columns
        ``[bin_range, event_count, non_event_count, woe, iv_contrib]``.
    bins : int
        Number of quantile buckets used during training.
    """
    sentinel_label = f"{int(_NAN_SENTINEL)} (missing)"

    # Separate sentinel row from quantile rows
    tbl_q = binning_table[binning_table["bin_range"] != sentinel_label].copy()
    tbl_s = binning_table[binning_table["bin_range"] == sentinel_label].copy()

    bin_woe_values: dict[str, float] = {}

    # Build float bin edges from the non-sentinel portion of the series.
    # pd.qcut on the same data reproduces the identical Interval categories.
    bin_edges: list[float] = []
    if len(tbl_q) > 0 and series.nunique() > 1:
        try:
            _, edge_bins = pd.qcut(series, q=bins, duplicates="drop", retbins=True)
            # Extend edges slightly so pd.cut at inference includes boundary values.
            edge_bins[0] = edge_bins[0] - 1e-9
            bin_edges = [float(e) for e in edge_bins]

            # Map interval label strings to WoE values
            for _, row in tbl_q.iterrows():
                bin_woe_values[row["bin_range"]] = float(row["woe"])
        except ValueError:
            pass  # degenerate feature — no quantile bins possible

    # Add sentinel mapping
    if len(tbl_s) > 0:
        bin_woe_values[sentinel_label] = float(tbl_s.iloc[0]["woe"])

    return {"bin_edges": bin_edges, "bin_woe_values": bin_woe_values}


def _bin_feature_and_compute_woe(
    series: pd.Series,
    target: pd.Series,
    bins: int = 10,
) -> tuple[dict, pd.Series]:
    """
    Bin one feature, compute its WoE mapping, and return the transformed series.

    Parameters
    ----------
    series : pd.Series
        Raw feature values (may contain ``_NAN_SENTINEL`` or actual NaN).
    target : pd.Series
        Binary response aligned with ``series``.
    bins : int, optional
        Number of quantile buckets (default 10).

    Returns
    -------
    woe_entry : dict
        ``{"bin_edges": [...], "bin_woe_values": {...}}`` — the mapping to
        store in ``woe_mappings`` for later use by ``apply_feature_store``.
        Empty dict when the feature carries no information.
    woe_series : pd.Series
        Feature values replaced by their per-bin WoE scores.
        Unresolved values (all-missing, constant) are filled with
        ``_NAN_SENTINEL``.
    """
    series_filled = series.fillna(_NAN_SENTINEL).reset_index(drop=True)
    aligned_target = target.reset_index(drop=True)

    result = _compute_binning_table(series_filled, aligned_target, bins)
    if result is None:
        return {}, pd.Series(_NAN_SENTINEL, index=series.index, dtype=float)

    binning_table, _ = result
    sentinel_label = f"{int(_NAN_SENTINEL)} (missing)"
    sentinel_mask = series_filled == _NAN_SENTINEL

    # Non-sentinel values used for edge extraction
    non_s = series_filled[~sentinel_mask]

    woe_entry = _compute_woe_mapping_dict(non_s, binning_table, bins)

    # Apply WoE transform to produce the output series
    woe_series = pd.Series(_NAN_SENTINEL, index=series.index, dtype=float)

    if woe_entry["bin_edges"]:
        binned = pd.cut(
            series_filled,
            bins=woe_entry["bin_edges"],
            include_lowest=True,
        )
        label_to_woe = {
            label: woe
            for label, woe in woe_entry["bin_woe_values"].items()
            if label != sentinel_label
        }
        # Map Interval objects → float WoE values.
        # Cast to float explicitly before fillna — pd.cut returns a Categorical
        # and fillna on a Categorical requires the fill value to be a known
        # category, which _NAN_SENTINEL (-999) is not.
        mapped = binned.map(lambda iv: label_to_woe.get(str(iv), np.nan))
        woe_series = pd.Series(mapped.to_numpy(dtype=float), index=series.index).fillna(_NAN_SENTINEL)
        woe_series.index = series.index

    # Apply sentinel WoE (if a sentinel bin exists)
    if sentinel_label in woe_entry["bin_woe_values"] and sentinel_mask.any():
        woe_series[sentinel_mask.values] = woe_entry["bin_woe_values"][sentinel_label]

    return woe_entry, woe_series


# ---------------------------------------------------------------------------
# Public API — feature store
# ---------------------------------------------------------------------------


def build_feature_store(
    X: pd.DataFrame,
    y: pd.Series,
    min_iv: float = _IV_WEAK,
    bins: int = 10,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Build the final feature store: engineer features, apply IV filter, WoE
    transform, and variance filter.  Saves artifacts to disk.

    This function is the training-time entry point.  All bin edges are fit
    exclusively on the training data passed here; they are stored in
    ``woe_mappings`` so that ``apply_feature_store`` can apply the exact same
    transformation at inference without re-fitting.

    Parameters
    ----------
    X : pd.DataFrame
        Raw application DataFrame (joined 7-table frame from ``load_data``).
    y : pd.Series
        Binary target (1 = default, 0 = non-default), aligned with ``X``.
    min_iv : float, optional
        Minimum Information Value threshold for feature selection (default 0.02).
    bins : int, optional
        Number of quantile bins per feature (default 10).
    output_dir : str | Path, optional
        Directory to save feature store artifacts. If None, defaults to
        {_PROJECT_ROOT}/data/processed/. Must exist or will be created.

    Returns
    -------
    X_final : pd.DataFrame
        WoE-transformed feature matrix after IV and variance filtering.
        Contains no NaN values (sentinel -999 used for missing/OOD).
    woe_mappings : dict
        ``{feature_name: {"bin_edges": [...], "bin_woe_values": {...}}}``
        Serialised to ``models/woe_mappings.pkl``.

    Notes
    -----
    The 3-line reduction summary printed to stdout is intentional
    user-facing output (consistent with ``select_features_by_iv``).

    **Data leakage prevention:** ``woe_mappings`` contains only bin edges
    derived from training data.  Never call ``pd.qcut`` on test data.
    """
    if output_dir is None:
        output_dir = _PROJECT_ROOT / "data" / "processed"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    X_eng = engineer_application_features(X) if "AMT_CREDIT" in X.columns else X.copy()
    print(f"Raw features: {X_eng.shape[1]}")

    iv_features = select_features_by_iv(X_eng, y, min_iv=min_iv, bins=bins)
    X_iv = X_eng[list(iv_features.keys())].copy()
    print(f"After IV filter (IV >= {min_iv}): {X_iv.shape[1]}")

    # WoE transform each IV-selected feature
    woe_mappings: dict = {}
    transformed_cols: dict[str, pd.Series] = {}
    for feature in X_iv.columns:
        woe_entry, woe_series = _bin_feature_and_compute_woe(X_iv[feature], y, bins)
        if woe_entry:  # skip degenerate features (no bin edges)
            woe_mappings[feature] = woe_entry
            transformed_cols[feature] = woe_series

    X_woe = pd.DataFrame(transformed_cols, index=X_iv.index)

    # Variance filter: drop constant columns then bottom 5% by variance
    variances = X_woe.var()
    positive_var = variances[variances > 0]
    if len(positive_var) == 0:
        X_final = pd.DataFrame(index=X_woe.index)
    else:
        var_threshold = positive_var.quantile(0.05)
        keep_cols = positive_var[positive_var >= var_threshold].index
        X_final = X_woe[keep_cols].copy()

    # Sync woe_mappings to only the columns that survived variance filtering
    woe_mappings = {k: v for k, v in woe_mappings.items() if k in X_final.columns}
    print(f"After variance filter: {X_final.shape[1]}")

    # Correlation deduplication: for pairs with |r| > 0.95, drop the lower-IV feature.
    # This removes near-redundant building measurement variants (_AVG/_MEDI/_MODE)
    # that pass the IV filter individually but carry overlapping information.
    _CORR_THRESHOLD: float = 0.90
    if X_final.shape[1] > 1:
        corr_matrix = X_final.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
        )
        cols_to_drop: set[str] = set()
        # Sort pairs by correlation descending so highest-redundancy is handled first
        high_pairs = (
            upper_tri.stack()
            .loc[lambda s: s > _CORR_THRESHOLD]
            .sort_values(ascending=False)
        )
        iv_lookup = iv_features  # dict {feature: iv} from select_features_by_iv
        for (col_a, col_b), _ in high_pairs.items():
            if col_a in cols_to_drop or col_b in cols_to_drop:
                continue
            # Keep the feature with higher IV; drop the other
            iv_a = iv_lookup.get(col_a, 0.0)
            iv_b = iv_lookup.get(col_b, 0.0)
            cols_to_drop.add(col_b if iv_a >= iv_b else col_a)
        if cols_to_drop:
            X_final = X_final.drop(columns=list(cols_to_drop))
            woe_mappings = {k: v for k, v in woe_mappings.items() if k not in cols_to_drop}
            print(f"After correlation dedup (|r| > {_CORR_THRESHOLD}): {X_final.shape[1]}")

    # Persist artifacts
    X_final.to_parquet(output_dir / "X_features.parquet", index=False)
    models_dir = _PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / "woe_mappings.pkl", "wb") as fh:
        pickle.dump(woe_mappings, fh)

    return X_final, woe_mappings


def apply_feature_store(
    X: pd.DataFrame,
    woe_mappings: dict,
) -> pd.DataFrame:
    """
    Apply stored WoE mappings to transform inference data.

    Uses the bin edges and WoE values computed during training — never
    re-fits on the incoming data.  Values outside training bin edges are
    out-of-distribution and are filled with ``_NAN_SENTINEL`` (-999).

    Parameters
    ----------
    X : pd.DataFrame
        Raw or partially-processed inference DataFrame.  Must contain all
        columns present in ``woe_mappings``.
    woe_mappings : dict
        ``{feature_name: {"bin_edges": [...], "bin_woe_values": {...}}}``
        as produced by ``build_feature_store`` or loaded from
        ``models/woe_mappings.pkl``.

    Returns
    -------
    pd.DataFrame
        Transformed DataFrame with exactly the columns in ``woe_mappings``,
        all values replaced by their WoE scores.  No NaN values remain.

    Notes
    -----
    **Data leakage prevention:** This function only calls ``pd.cut`` with
    stored ``bin_edges``.  It never calls ``pd.qcut`` or any fitting
    operation.
    """
    features = list(woe_mappings.keys())
    X_out = X[features].copy()
    sentinel_label = f"{int(_NAN_SENTINEL)} (missing)"

    for feature, entry in woe_mappings.items():
        series = X_out[feature].fillna(_NAN_SENTINEL)
        sentinel_mask = series == _NAN_SENTINEL

        result = pd.Series(_NAN_SENTINEL, index=X_out.index, dtype=float)

        bin_edges = entry["bin_edges"]
        bin_woe_values = entry["bin_woe_values"]

        if bin_edges:
            label_to_woe = {
                label: woe
                for label, woe in bin_woe_values.items()
                if label != sentinel_label
            }
            binned = pd.cut(series, bins=bin_edges, include_lowest=True)
            mapped = binned.map(lambda iv: label_to_woe.get(str(iv), np.nan))
            result = pd.Series(mapped.to_numpy(dtype=float), index=X_out.index).fillna(_NAN_SENTINEL)
            result.index = X_out.index

        # Override sentinel positions with sentinel WoE (or -999 if no sentinel bin)
        if sentinel_mask.any():
            sentinel_woe = bin_woe_values.get(sentinel_label, _NAN_SENTINEL)
            result[sentinel_mask] = sentinel_woe

        X_out[feature] = result

    return X_out


def select_features_by_iv(
    df: pd.DataFrame,
    target: pd.Series,
    min_iv: float = _IV_WEAK,
    bins: int = 10,
) -> dict[str, float]:
    """
    Rank all numeric features by Information Value and return those above a threshold.

    Iterates over every numeric column in ``df``, calls ``compute_woe_iv`` on
    each, then filters and sorts results.  Non-numeric columns and constant /
    all-missing columns are silently skipped.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix (may contain mixed types; non-numeric columns ignored).
    target : pd.Series
        Binary response (1 = default, 0 = non-default).
    min_iv : float, optional
        Minimum IV to include in the returned dict (default 0.02, i.e. ``_IV_WEAK``).
    bins : int, optional
        Number of quantile buckets passed to ``compute_woe_iv`` (default 10).

    Returns
    -------
    dict[str, float]
        ``{feature_name: iv_value}`` for features with ``iv >= min_iv``,
        sorted descending by IV.

    Notes
    -----
    **Print side-effect (intentional):** This function prints an IV-tier summary
    to stdout.  This is an intentional user-facing output for exploratory analysis
    and is not internal logging.  Future refactors targeting production use should
    replace ``print`` with ``logging.info``.

    **IV tiers** (Siddiqi thresholds):

    +------------------+---------+
    | Tier             | IV      |
    +==================+=========+
    | Very strong      | > 0.5   |
    +------------------+---------+
    | Strong           | 0.3–0.5 |
    +------------------+---------+
    | Medium           | 0.1–0.3 |
    +------------------+---------+
    | Weak             | 0.02–0.1|
    +------------------+---------+
    | Useless          | < 0.02  |
    +------------------+---------+
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    iv_map: dict[str, float] = {}
    for col in numeric_cols:
        try:
            _, iv = compute_woe_iv(df, col, target, bins)
            iv_map[col] = iv
        except Exception:  # noqa: BLE001 — keep scanning even on unexpected errors
            pass

    very_strong = [k for k, v in iv_map.items() if v >= _IV_VERY_STRONG]
    strong = [k for k, v in iv_map.items() if _IV_STRONG <= v < _IV_VERY_STRONG]
    medium = [k for k, v in iv_map.items() if _IV_MEDIUM <= v < _IV_STRONG]
    weak = [k for k, v in iv_map.items() if _IV_WEAK <= v < _IV_MEDIUM]
    useless = [k for k, v in iv_map.items() if v < _IV_WEAK]

    print(f"Very strong (IV >= {_IV_VERY_STRONG}): {len(very_strong)} features")
    print(f"Strong (IV {_IV_STRONG}–{_IV_VERY_STRONG}): {len(strong)} features")
    print(f"Medium (IV {_IV_MEDIUM}–{_IV_STRONG}): {len(medium)} features")
    print(f"Weak (IV {_IV_WEAK}–{_IV_MEDIUM}): {len(weak)} features")
    print(f"Useless (IV < {_IV_WEAK}): {len(useless)} features")

    filtered = {k: v for k, v in iv_map.items() if v >= min_iv}
    sorted_iv = dict(sorted(filtered.items(), key=lambda item: item[1], reverse=True))

    print(f"Total selected (IV >= {min_iv}): {len(sorted_iv)} features")

    return sorted_iv


# ---------------------------------------------------------------------------
# Public API — raw feature store (no WoE transform)
# ---------------------------------------------------------------------------


def build_tree_feature_store(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str | Path | None = None,
    df_inst: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build raw (non-WoE) feature store for tree-based models (XGBoost, LightGBM, CatBoost).

    Applies a simple pipeline: raw feature engineering → numeric filtering → variance filtering.
    Does NOT apply WoE binning or IV filtering. Raw continuous values are preserved,
    allowing tree models to find optimal splits via gradient descent.

    This differs from build_feature_store (which applies WoE for logistic regression)
    and from build_raw_feature_store (which applied IV filtering). For trees, we want
    maximum signal from raw continuous distributions.

    Parameters
    ----------
    X : pd.DataFrame
        Raw application DataFrame (joined 7-table frame from load_data).
    y : pd.Series
        Binary target (1 = default, 0 = non-default), aligned with X.
        Used internally for feature engineering only; not for IV filtering.
    output_dir : str | Path, optional
        Directory to save the raw feature matrix and feature columns list.
        If None, defaults to {_PROJECT_ROOT}/data/processed/.
    df_inst : pd.DataFrame, optional
        Raw installments_payments table. If provided, Wave 1 instalment-based
        features are computed and added to the store.

    Returns
    -------
    X_final : pd.DataFrame
        Raw (continuous-valued) feature matrix after variance filtering only.
        Contains no NaN values (sentinel -999 used). Shape: (307511, N>=100).
    feature_columns : list[str]
        Column names in X_final, to be passed to apply_raw_feature_store.
        Also saved to models/raw_feature_columns.pkl.

    Notes
    -----
    **Data leakage prevention:** Feature engineering is fit only on training data.
    Never call this function on test data. Use apply_raw_feature_store for inference.

    **Variance filter only:** Unlike build_feature_store (WoE + IV) and build_raw_feature_store
    (IV + variance + correlation), this function skips IV and correlation filtering entirely.
    Trees handle correlated features well and can exploit subtle signal in low-IV features.
    This preserves the maximum continuous gradient signal.

    **No WoE binning:** All values remain as raw floats (or -999 for missing).

    **Wave 1 Features:** If df_inst is provided, Wave 1 delinquency trajectory features
    (inst_late_rate_12m, inst_rolling_30dpd_ratio_3m, etc.) are computed and added to the store.
    """
    if output_dir is None:
        output_dir = _PROJECT_ROOT / "data" / "processed"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = _PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    X_eng = engineer_application_features(X) if "AMT_CREDIT" in X.columns else X.copy()
    X_eng = engineer_secondary_features(X_eng)

    # Wave 1 features (Phase 04.2.7 — delinquency trajectory signals)
    if df_inst is not None:
        # Instalment-level features (require raw table)
        X_eng["inst_late_rate_12m"] = engineer_inst_late_rate_12m(df_inst).reindex(X_eng.index, fill_value=_NAN_SENTINEL).values
        X_eng["inst_late_rate_recent_vs_historical"] = engineer_inst_late_rate_recent_vs_historical(df_inst).reindex(X_eng.index, fill_value=_NAN_SENTINEL).values
        X_eng["inst_rolling_30dpd_ratio_3m"] = engineer_inst_rolling_30dpd_ratio_3m(df_inst).reindex(X_eng.index, fill_value=_NAN_SENTINEL).values
        X_eng["inst_delinquency_escalation_flag"] = engineer_inst_delinquency_escalation_flag(df_inst).reindex(X_eng.index, fill_value=_NAN_SENTINEL).values
        X_eng["inst_days_since_last_30dpd"] = engineer_inst_days_since_last_30dpd(df_inst).reindex(X_eng.index, fill_value=_NAN_SENTINEL).values

    # Bureau-level features (operate on aggregated columns already in X_eng)
    X_eng["bureau_dpd_trend_3m_vs_12m"] = engineer_bureau_dpd_trend_3m_vs_12m(X_eng).values
    X_eng["bureau_debt_to_new_credit"] = engineer_bureau_debt_to_new_credit(X_eng).values

    # D-20, D-21: Enforce regulatory compliance
    # Drop CODE_GENDER (GDPR Art. 21) and thin_file_young (EU AI Act Art. 6 age discrimination)
    # Use .drop(..., errors='ignore') to handle cases where columns may not exist
    X_eng = X_eng.drop(columns=_REGULATORY_DROP_COLS, errors="ignore")

    # D-22: ORGANIZATION_TYPE treatment
    # ORGANIZATION_TYPE is passed as integer-encoded categorical to XGBoost (no target encoding here).
    # WoE/logistic pipeline applies smoothing=50 per SR 11-7. Tree pipeline uses raw integers.

    # Keep only numeric columns — categorical columns (dtype 'category' or 'object')
    # cannot be passed to gradient boosting without encoding.
    numeric_cols = X_eng.select_dtypes(include=[np.number]).columns.tolist()
    X_numeric = X_eng[numeric_cols].copy()

    # Replace inf/nan with sentinel
    X_filled = X_numeric.copy()
    X_filled = X_filled.replace([np.inf, -np.inf], np.nan)
    X_filled = X_filled.fillna(_NAN_SENTINEL)

    # Skip IV filter for tree models — preserve all numeric columns with variance
    X_iv = X_filled.copy()

    # Layer 6a: Remove low-variance features (variance < 1%)
    # Use sklearn.feature_selection.VarianceThreshold for reproducible semantics
    # (avoids data-dependent quantile heuristics).
    var_filter = VarianceThreshold(threshold=0.01)
    if X_iv.shape[1] > 0:
        X_var_filtered = var_filter.fit_transform(X_iv)
        X_var = pd.DataFrame(
            X_var_filtered,
            columns=X_iv.columns[var_filter.get_support()],
            index=X_iv.index,
        )
    else:
        X_var = X_iv.copy()

    print(f"After variance filter: {X_var.shape[1]}")

    # Layer 6b: Remove correlated features (|r| > 0.95)
    # Compute Pearson correlation matrix and drop one of each highly-correlated pair.
    if X_var.shape[1] > 1:
        corr_matrix = X_var.corr().abs()

        # Find pairs with |r| > 0.95 (upper triangle only to avoid duplication)
        pairs_to_drop = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.95:
                    pairs_to_drop.append(corr_matrix.columns[j])

        # Keep first of each pair, drop second
        cols_to_drop = list(set(pairs_to_drop))
        X_final = X_var.drop(columns=cols_to_drop, errors="ignore")
    else:
        X_final = X_var.copy()

    print(f"After correlation dedup (|r| > 0.95): {X_final.shape[1]}")

    # Persist artifacts
    X_final.to_parquet(output_dir / "X_tree_raw.parquet", index=False)
    feature_columns = list(X_final.columns)
    with open(models_dir / "raw_feature_columns.pkl", "wb") as fh:
        pickle.dump(feature_columns, fh)

    return X_final, feature_columns


# Backward-compatibility alias — renamed to build_tree_feature_store in Phase 04.2.1
build_raw_feature_store = build_tree_feature_store


def apply_raw_feature_store(
    X: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Apply stored raw feature selection to inference data.

    Uses feature column list from build_tree_feature_store. No re-fitting occurs.
    Missing columns are filled with _NAN_SENTINEL (-999). All inf values replaced
    with nan then filled with sentinel.

    Parameters
    ----------
    X : pd.DataFrame
        Raw or partially-processed inference DataFrame.
    feature_columns : list[str]
        Column list from build_tree_feature_store (or loaded from
        models/raw_feature_columns.pkl).

    Returns
    -------
    pd.DataFrame
        DataFrame with exactly the columns in feature_columns, in that order.
        All NaN/inf replaced with -999. No NaN values remain.

    Notes
    -----
    **Data leakage prevention:** This function never calls select_features_by_iv.
    Feature selection is determined at training time only.
    """
    X_eng = engineer_application_features(X) if "AMT_CREDIT" in X.columns else X.copy()
    X_eng = engineer_secondary_features(X_eng)

    # Replace inf/nan with sentinel
    X_filled = X_eng.copy()
    X_filled = X_filled.replace([np.inf, -np.inf], np.nan)
    X_filled = X_filled.fillna(_NAN_SENTINEL)

    # Select columns (fill missing with -999)
    X_out = pd.DataFrame(index=X_filled.index)
    for col in feature_columns:
        if col in X_filled.columns:
            X_out[col] = X_filled[col]
        else:
            X_out[col] = _NAN_SENTINEL

    return X_out


def engineer_time_features(data_dir: Path | str) -> pd.DataFrame:
    """
    Compute time-window features from raw secondary tables.

    Generates 3 net-new features not already present in X_train.parquet:
    - bbal_dpd_rate_3m: DPD rate in last 3 months (MONTHS_BALANCE >= -3)
    - bbal_months_since_last_dpd: months since last delinquent status in bureau_balance
    - bureau_credit_age_mean: average credit account age in years

    Parameters
    ----------
    data_dir : Path | str
        Directory containing bureau.csv and bureau_balance.csv (and installments_payments.csv
        for potential future use).

    Returns
    -------
    pd.DataFrame
        Index = SK_ID_CURR; Columns = 3 features listed above.
        Missing applicants (not in source tables) are filled with _NAN_SENTINEL (-999).

    Notes
    -----
    Data loading: CSVs are loaded from data_dir; no use of load_data().
    Missing handling: Applicants without records in source tables get -999 sentinel.
    DPD status codes: In bureau_balance STATUS, values 1-5 indicate delinquency;
    C, X, 0 are clean/closed/unknown.
    """
    data_dir = Path(data_dir)

    # Load raw tables
    bbal = pd.read_csv(data_dir / "bureau_balance.csv")
    bureau = pd.read_csv(data_dir / "bureau.csv")

    # -----------------------------------------------------------------------
    # Feature 1: bbal_dpd_rate_3m
    # DPD rate in last 3 months (MONTHS_BALANCE >= -3)
    # -----------------------------------------------------------------------
    # Create SK_ID_CURR mapping from bureau
    bureau_map = bureau[["SK_ID_BUREAU", "SK_ID_CURR"]].drop_duplicates()
    bbal_with_curr = bbal.merge(bureau_map, on="SK_ID_BUREAU", how="left")

    # Filter to 3-month window
    bbal_3m = bbal_with_curr[bbal_with_curr["MONTHS_BALANCE"] >= -3].copy()

    # Mark delinquent statuses: STATUS in ['1', '2', '3', '4', '5']
    bbal_3m["is_dpd"] = bbal_3m["STATUS"].isin(["1", "2", "3", "4", "5"]).astype(float)

    # Compute rate: count(DPD) / count(all) per SK_ID_CURR
    dpd_3m_counts = (
        bbal_3m.groupby("SK_ID_CURR")["is_dpd"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "dpd_count", "count": "total_count"})
    )
    dpd_3m_counts["bbal_dpd_rate_3m"] = (
        dpd_3m_counts["dpd_count"] / dpd_3m_counts["total_count"]
    )
    bbal_dpd_rate_3m = dpd_3m_counts["bbal_dpd_rate_3m"]

    # -----------------------------------------------------------------------
    # Feature 2: bbal_months_since_last_dpd
    # Months since last delinquent status in bureau_balance
    # -----------------------------------------------------------------------
    # Filter to rows with delinquent status
    bbal_dpd = bbal_with_curr[
        bbal_with_curr["STATUS"].isin(["1", "2", "3", "4", "5"])
    ].copy()

    # For each SK_ID_CURR, find the min(MONTHS_BALANCE) among delinquent rows
    # (min because MONTHS_BALANCE is negative; min = most recent = closest to 0)
    # Negate to get "months ago"
    months_since_dpd = (
        bbal_dpd.groupby("SK_ID_CURR")["MONTHS_BALANCE"]
        .apply(lambda x: -x.min() if len(x) > 0 else np.nan)
        .rename("bbal_months_since_last_dpd")
    )

    # -----------------------------------------------------------------------
    # Feature 3: bureau_credit_age_mean
    # Average credit account age in years (from DAYS_CREDIT)
    # -----------------------------------------------------------------------
    # DAYS_CREDIT is negative; convert to absolute value and scale to years
    bureau_with_age = bureau.copy()
    bureau_with_age["credit_age_years"] = np.abs(bureau_with_age["DAYS_CREDIT"]) / 365.25

    # Compute mean age per SK_ID_CURR
    bureau_credit_age_mean = bureau_with_age.groupby("SK_ID_CURR")[
        "credit_age_years"
    ].mean()
    bureau_credit_age_mean = bureau_credit_age_mean.rename("bureau_credit_age_mean")

    # -----------------------------------------------------------------------
    # Combine into result DataFrame
    # -----------------------------------------------------------------------
    # Load all SK_ID_CURR from application_train.csv to ensure complete coverage
    app_train = pd.read_csv(data_dir / "application_train.csv", usecols=["SK_ID_CURR"])
    all_curr_ids = app_train["SK_ID_CURR"].values

    # Create result DataFrame indexed by all SK_ID_CURR
    result = pd.DataFrame(index=pd.Index(all_curr_ids, name="SK_ID_CURR"))

    # Assign features (will introduce NaN for applicants without bureau records)
    result["bbal_dpd_rate_3m"] = bbal_dpd_rate_3m
    result["bbal_months_since_last_dpd"] = months_since_dpd
    result["bureau_credit_age_mean"] = bureau_credit_age_mean

    # Fill all NaN with sentinel value
    result = result.fillna(_NAN_SENTINEL)

    return result


def compute_knn_target_encoding(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    features: list[str],
    k: int = 500,
    n_folds: int = 5,
    random_state: int = 42,
) -> tuple[pd.Series, pd.Series]:
    """
    OOF KNN mean target encoding.

    For the training set, uses out-of-fold predictions to prevent leakage:
    each fold is predicted by a KNN fitted on the remaining folds only.
    For the test set, a single KNN is fitted on the full training set.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix. Must contain all columns in `features`.
    y_train : pd.Series
        Binary target (0/1), indexed as X_train.
    X_test : pd.DataFrame
        Test feature matrix. Must contain all columns in `features`.
    features : list[str]
        Column names defining the KNN neighbourhood space.
        Recommended: ['EXT_SOURCE_MEAN', 'EXT_SOURCE_MIN',
                      'CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO']
    k : int, optional
        Number of neighbours. Default 500.
    n_folds : int, optional
        Number of OOF folds. Default 5.
    random_state : int, optional
        Reproducibility seed for StratifiedKFold. Default 42.

    Returns
    -------
    train_enc : pd.Series
        OOF soft-label encoding for training set, indexed as X_train.
    test_enc : pd.Series
        Soft-label encoding for test set, indexed as X_test.

    Raises
    ------
    ValueError
        If any column in `features` is absent from X_train or X_test.

    Notes
    -----
    **Data leakage prevention:**
      - OOF: Each fold's KNN is fit on (n_train - n_fold_size) rows only.
      - Test: KNN is fit on full training set (len(X_train) rows).
      - NaN handling: Any NaN or -999 sentinel is imputed with column median
        (fit on each training fold separately, applied to val/test).
      - Scaling: StandardScaler fit on each training fold separately.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    # ---------------------
    # Input validation
    # ---------------------
    for feat in features:
        if feat not in X_train.columns:
            raise ValueError(f"Feature '{feat}' not found in X_train columns")
        if feat not in X_test.columns:
            raise ValueError(f"Feature '{feat}' not found in X_test columns")

    # ---------------------
    # Prepare feature subsets
    # ---------------------
    X_train_features = X_train[features].copy()
    X_test_features = X_test[features].copy()

    # ---------------------
    # OOF loop: prevent leakage with StratifiedKFold
    # ---------------------
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    train_enc = pd.Series(index=X_train.index, dtype=float)

    for train_idx, val_idx in skf.split(X_train_features, y_train):
        # Split indices
        X_train_fold = X_train_features.iloc[train_idx].copy()
        y_train_fold = y_train.iloc[train_idx].copy()
        X_val_fold = X_train_features.iloc[val_idx].copy()

        # Impute NaN and sentinel on training fold, fit on training fold
        imputer = SimpleImputer(strategy="median")
        X_train_fold_imputed = pd.DataFrame(
            imputer.fit_transform(X_train_fold.replace(_NAN_SENTINEL, np.nan)),
            columns=features,
            index=X_train_fold.index,
        )

        # Transform validation fold using the fitted imputer
        X_val_fold_imputed = pd.DataFrame(
            imputer.transform(X_val_fold.replace(_NAN_SENTINEL, np.nan)),
            columns=features,
            index=X_val_fold.index,
        )

        # Fit scaler on training fold
        scaler = StandardScaler()
        X_train_fold_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_fold_imputed),
            columns=features,
            index=X_train_fold_imputed.index,
        )

        # Transform validation fold
        X_val_fold_scaled = pd.DataFrame(
            scaler.transform(X_val_fold_imputed),
            columns=features,
            index=X_val_fold_imputed.index,
        )

        # Fit KNN on training fold, predict on validation fold
        knn = KNeighborsClassifier(n_neighbors=k, metric="minkowski", n_jobs=-1)
        knn.fit(X_train_fold_scaled, y_train_fold)
        val_proba = knn.predict_proba(X_val_fold_scaled)[:, 1]

        # Store soft labels (probability of default)
        train_enc.iloc[val_idx] = val_proba

    # ---------------------
    # Test encoding: fit on FULL training set
    # ---------------------
    # Impute NaN and sentinel, fit on full training set
    imputer_full = SimpleImputer(strategy="median")
    X_train_features_imputed = pd.DataFrame(
        imputer_full.fit_transform(X_train_features.replace(_NAN_SENTINEL, np.nan)),
        columns=features,
        index=X_train_features.index,
    )

    # Transform test using the fitted imputer
    X_test_features_imputed = pd.DataFrame(
        imputer_full.transform(X_test_features.replace(_NAN_SENTINEL, np.nan)),
        columns=features,
        index=X_test_features.index,
    )

    # Fit scaler on full training set
    scaler_full = StandardScaler()
    X_train_features_scaled = pd.DataFrame(
        scaler_full.fit_transform(X_train_features_imputed),
        columns=features,
        index=X_train_features_imputed.index,
    )

    # Transform test set
    X_test_features_scaled = pd.DataFrame(
        scaler_full.transform(X_test_features_imputed),
        columns=features,
        index=X_test_features_imputed.index,
    )

    # Fit KNN on full training set, predict on test
    knn_full = KNeighborsClassifier(n_neighbors=k, metric="minkowski", n_jobs=-1)
    knn_full.fit(X_train_features_scaled, y_train)
    test_proba = knn_full.predict_proba(X_test_features_scaled)[:, 1]

    test_enc = pd.Series(test_proba, index=X_test.index, dtype=float)

    # ---------------------
    # Return
    # ---------------------
    train_enc.name = "knn_target_enc"
    test_enc.name = "knn_target_enc"

    return train_enc, test_enc


# ---------------------------------------------------------------------------
# Combined Feature Store Construction (Phase 2)
# ---------------------------------------------------------------------------


def build_combined_feature_store(
    output_path: Path | str | None = None,
    dfs_eval_path: Path | str | None = None,
    imputer_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    Build combined feature store by merging raw + imputed EXT_SOURCE_3 + conditional DFS.

    Loads X_raw_features (307,511 × 62), applies EXT_SOURCE_3 imputation
    (adding EXT_SOURCE_3_MISSING_FLAG), and conditionally includes DFS features
    if the eval decision is "commit" (delta >= 0.01).

    All NaN values are filled with -999 sentinel.

    Parameters
    ----------
    output_path : Path | str, optional
        Output path for combined feature store parquet file.
        If None, defaults to {_PROJECT_ROOT}/data/processed/X_combined_features.parquet.
    dfs_eval_path : Path | str, optional
        Path to DFS evaluation results JSON.
        If None, defaults to {_PROJECT_ROOT}/reports/dfs_eval_results.json.
    imputer_path : Path | str, optional
        Path to fitted EXT_SOURCE_3 imputer model.
        If None, defaults to {_PROJECT_ROOT}/models/ext_source_imputation_lgb.pkl.

    Returns
    -------
    pd.DataFrame
        Combined feature matrix (307,511 rows, >= 65 columns).
        All NaN values replaced with -999 sentinel.
        Includes EXT_SOURCE_3_MISSING_FLAG.

    Notes
    -----
    - If DFS decision is "defer", only raw + imputed features are included.
    - Shape assertion: (307,511 rows, >= 65 columns).
    - Index name: "SK_ID_CURR" (preserved from X_raw).

    Threat Mitigations:
    - T-02-13: Assert DFS decision in {commit, defer}
    - T-02-14: joblib.load() raises if imputer file is corrupted
    - T-02-15: Assert row count == 307,511 throughout
    """
    import json

    from src.model import apply_ext_source_imputer

    if output_path is None:
        output_path = _PROJECT_ROOT / "data" / "processed" / "X_combined_features.parquet"
    else:
        output_path = Path(output_path)

    if dfs_eval_path is None:
        dfs_eval_path = _PROJECT_ROOT / "reports" / "dfs_eval_results.json"
    else:
        dfs_eval_path = Path(dfs_eval_path)

    if imputer_path is None:
        imputer_path = _PROJECT_ROOT / "models" / "ext_source_imputation_lgb.pkl"
    else:
        imputer_path = Path(imputer_path)

    # Step 1: Load X_raw (307,511 × 62)
    X_raw = pd.read_parquet(_PROJECT_ROOT / "data" / "processed" / "X_raw_features.parquet")
    assert (
        X_raw.shape[0] == 307511
    ), f"Raw features shape mismatch: expected 307511 rows, got {X_raw.shape[0]}"
    print(f"Loaded X_raw: {X_raw.shape}")

    # Step 2: Load and apply EXT_SOURCE_3 imputer
    import joblib

    imputer = joblib.load(imputer_path)
    X_imputed = apply_ext_source_imputer(X_raw, imputer, ext_source_col="EXT_SOURCE_3")
    assert (
        X_imputed.shape[0] == 307511
    ), f"After imputation, shape mismatch: {X_imputed.shape[0]} rows"
    assert (
        "EXT_SOURCE_3_MISSING_FLAG" in X_imputed.columns
    ), "EXT_SOURCE_3_MISSING_FLAG not found after imputation"
    print(f"Imputed EXT_SOURCE_3: {X_imputed.shape}")

    # Step 3: Check DFS decision from JSON
    with open(dfs_eval_path, "r") as f:
        dfs_result = json.load(f)
    dfs_decision = dfs_result.get("decision", "defer")
    assert dfs_decision in {
        "commit",
        "defer",
    }, f"Invalid DFS decision: {dfs_decision}"
    print(f"DFS decision: {dfs_decision}")

    # Step 4: If decision="commit", load and merge DFS features
    X_combined = X_imputed.copy()
    if dfs_decision == "commit":
        from src.auto_features import apply_featuretools_feature_store

        X_dfs = apply_featuretools_feature_store(_PROJECT_ROOT / "data" / "processed" / "X_featuretools.parquet")
        # Ensure index alignment
        assert (
            X_dfs.shape[0] == 307511
        ), f"DFS features shape mismatch: {X_dfs.shape[0]} rows"
        X_dfs.index = X_combined.index
        # Merge DFS features (on index)
        X_combined = pd.concat([X_combined, X_dfs], axis=1)
        print(f"Merged DFS features: {X_combined.shape}")

    # Step 5: Fill all NaN with -999 sentinel
    X_combined = X_combined.fillna(_NAN_SENTINEL)
    nan_count_after = X_combined.isna().sum().sum()
    assert (
        nan_count_after == 0
    ), f"NaN sentinel fill failed: {nan_count_after} NaN values remain"

    # Step 6: Validate shape
    assert X_combined.shape[0] == 307511, f"Final shape mismatch: {X_combined.shape[0]} rows"
    # Minimum viable: 62 raw + 1 flag = 63; target when DFS commits: 63+ DFS = 65+
    min_cols_viable = 63 if dfs_decision == "defer" else 65
    assert X_combined.shape[1] >= min_cols_viable, (
        f"Column count below minimum for decision '{dfs_decision}': "
        f"{X_combined.shape[1]} < {min_cols_viable}"
    )
    print(f"Final combined store: {X_combined.shape}")

    # Step 7: Save to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    X_combined.to_parquet(output_path, index=False)
    print(f"Saved combined store to {output_path}")


# ---------------------------------------------------------------------------
# Feature Augmentation Utilities (Phase 04.2 Plan 03)
# ---------------------------------------------------------------------------


def rank_normalize_fold_safe(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    exclude_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert features to within-fold percentile ranks [0, 1].

    Fit min/max statistics on X_train only, then apply to both X_train and
    X_val.  OOD values in X_val are clipped to [0, 1].  Prevents data
    leakage: validation fold never influences normalization statistics.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training fold feature matrix.
    X_val : pd.DataFrame
        Validation fold feature matrix (same columns as X_train).
    exclude_cols : list[str], optional
        Column names to skip (e.g., temporal sort column used for CV groups).
        These columns are passed through unchanged.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (X_train_ranked, X_val_ranked) — both with values in [0, 1] for
        normalised columns; shape and column order preserved.

    Notes
    -----
    **Constant features:** columns where ``col_max == col_min`` are set to 0.5
    in both folds (midpoint of the unit interval).

    **OOD clipping:** validation values outside the training range are clipped
    to [0, 1] so the model always receives bounded inputs.

    **Tree invariance:** gradient-boosted trees (LGB, XGB, CatBoost) are
    monotone-transform invariant, so rank normalisation does not change split
    decisions.  The expected Gini delta is ~0 for these models; rank
    normalisation is tested here to confirm that invariance empirically.
    """
    exclude_cols = set(exclude_cols or [])
    rank_cols = [c for c in X_train.columns if c not in exclude_cols]

    X_train_ranked = X_train.copy()
    X_val_ranked = X_val.copy()

    for col in rank_cols:
        col_min = float(X_train[col].min())
        col_max = float(X_train[col].max())
        col_range = col_max - col_min

        if col_range == 0:
            # Constant feature — set both folds to midpoint
            X_train_ranked[col] = 0.5
            X_val_ranked[col] = 0.5
        else:
            X_train_ranked[col] = (X_train[col] - col_min) / col_range
            X_val_ranked[col] = ((X_val[col] - col_min) / col_range).clip(0.0, 1.0)

    return X_train_ranked, X_val_ranked


def polynomial_interactions_from_shap(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model: object,
    top_n: int = 15,
    iv_gate: float = 0.02,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """
    Generate degree-2 polynomial interactions between top SHAP features.

    Computes mean |SHAP| importance from a fitted tree model, selects the
    top-N features, generates all pairwise products, and retains only those
    with IV >= iv_gate on the training fold.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training fold feature matrix (model must be fitted on this data).
    y_train : pd.Series
        Training fold binary target.
    model : object
        Fitted tree model with ``predict_proba`` method.  Used to compute
        SHAP values via ``shap.TreeExplainer``.
    top_n : int, optional
        Number of top SHAP features to generate interactions from.
    iv_gate : float, optional
        Minimum IV for retaining an interaction column (default 0.02, i.e.,
        "weak" discriminatory power per Siddiqi thresholds).

    Returns
    -------
    X_train_aug : pd.DataFrame
        Training fold with interaction columns appended.
    selected_pairs : list[tuple[str, str]]
        Feature name pairs for interactions that passed the IV gate.
        Pass to ``apply_polynomial_interactions()`` to transform validation fold.

    Notes
    -----
    **Fold safety:** SHAP values and IV statistics are computed exclusively on
    X_train.  Pass ``selected_pairs`` to ``apply_polynomial_interactions`` to
    transform the validation fold using the same pairs — no validation data
    is used to select which interactions to add.

    **Interaction naming:** column ``f"{a}_x_{b}_poly2"`` for each pair (a, b).
    """
    import itertools

    import shap

    # Compute SHAP importance on training fold
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    if isinstance(shap_values, list):
        # Multi-output: use class-1 values (default probability)
        shap_values = shap_values[1]

    shap_importance = np.abs(shap_values).mean(axis=0)
    top_n_actual = min(top_n, len(X_train.columns))
    top_indices = np.argsort(shap_importance)[-top_n_actual:][::-1]
    top_features = [X_train.columns[i] for i in top_indices]

    selected_pairs: list[tuple[str, str]] = []
    X_train_aug = X_train.copy()

    for feat_a, feat_b in itertools.combinations(top_features, 2):
        interaction = X_train[feat_a] * X_train[feat_b]
        col_name = f"{feat_a}_x_{feat_b}_poly2"

        # IV gate: compute IV on a temporary single-column DataFrame
        tmp_df = pd.DataFrame({col_name: interaction})
        _, iv_val = compute_woe_iv(tmp_df, col_name, y_train)

        if iv_val >= iv_gate:
            X_train_aug[col_name] = interaction
            selected_pairs.append((feat_a, feat_b))

    return X_train_aug, selected_pairs


def apply_polynomial_interactions(
    X: pd.DataFrame,
    pairs: list[tuple[str, str]],
) -> pd.DataFrame:
    """
    Apply pre-selected polynomial interaction pairs to a feature matrix.

    Companion to ``polynomial_interactions_from_shap()``.  Call this on the
    validation fold using the ``selected_pairs`` returned by the fit function,
    ensuring the validation fold receives exactly the same interaction columns
    as the training fold — no re-selection, no leakage.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix to augment (e.g., validation fold).
    pairs : list[tuple[str, str]]
        Feature name pairs selected by ``polynomial_interactions_from_shap``.

    Returns
    -------
    pd.DataFrame
        Copy of X with interaction columns appended.  Shape[1] increases by
        ``len(pairs)``.  Shape[0] and index unchanged.
    """
    X_out = X.copy()
    for feat_a, feat_b in pairs:
        col_name = f"{feat_a}_x_{feat_b}_poly2"
        X_out[col_name] = X[feat_a] * X[feat_b]
    return X_out


def pseudo_label_from_predictions(
    y_pred_proba: np.ndarray,
    X_source: pd.DataFrame,
    confidence_threshold: tuple[float, float] = (0.05, 0.95),
    max_synthetic_rows: int | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Create pseudo-labeled rows from high-confidence model predictions.

    Extracts rows from X_source where the predicted probability is below
    the low threshold (pseudo-label 0) or above the high threshold
    (pseudo-label 1).  These synthetic rows can be appended to the training
    fold to augment it with unlabeled data.

    Parameters
    ----------
    y_pred_proba : np.ndarray
        Shape (n,) predicted probabilities from a preliminary model applied
        to X_source.
    X_source : pd.DataFrame
        Feature matrix corresponding to y_pred_proba (e.g., test/holdout set).
        Must NOT overlap with the validation fold used for evaluation.
    confidence_threshold : tuple[float, float], optional
        (low, high) — rows with prob < low → label 0;
        rows with prob > high → label 1.
        Default (0.05, 0.95) captures very high-confidence predictions.
    max_synthetic_rows : int, optional
        Cap on the number of pseudo-labeled rows returned (memory guard).
        When set, a random subsample is drawn without replacement.
    random_state : int, optional
        Seed for the random subsample when max_synthetic_rows is active.

    Returns
    -------
    X_pseudo : pd.DataFrame
        Feature rows with high-confidence pseudo-labels.  Index reset to
        integers starting at 0 (avoid index collision when concatenating
        with training fold).
    y_pseudo : pd.Series
        Binary pseudo-labels (0 or 1) aligned with X_pseudo.

    Notes
    -----
    **Leakage guard:** X_source must be drawn from a data partition that does
    not overlap with the OOF validation fold.  Typically this is either the
    competition test set (no true labels) or a dedicated holdout.  Never use
    the same rows for pseudo-labels and OOF evaluation.
    """
    y_pred_proba = np.asarray(y_pred_proba)
    high_confidence_mask = (y_pred_proba < confidence_threshold[0]) | (
        y_pred_proba >= confidence_threshold[1]
    )

    X_pseudo = X_source.loc[high_confidence_mask].copy()
    y_pseudo_arr = (y_pred_proba[high_confidence_mask] > 0.5).astype(int)
    y_pseudo = pd.Series(y_pseudo_arr, name="TARGET")

    if max_synthetic_rows is not None and len(X_pseudo) > max_synthetic_rows:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X_pseudo), size=max_synthetic_rows, replace=False)
        idx_sorted = np.sort(idx)
        X_pseudo = X_pseudo.iloc[idx_sorted]
        y_pseudo = y_pseudo.iloc[idx_sorted]

    X_pseudo = X_pseudo.reset_index(drop=True)
    y_pseudo = y_pseudo.reset_index(drop=True)

    return X_pseudo, y_pseudo


def build_dfs_feature_store(
    data_dir: Path | str,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build merged tree feature store: raw + DFS + time features.

    Memory-safe orchestration of the Phase 04.2.2 pipeline.
    Uses a DFS checkpoint to allow restarts without re-running DFS,
    and sample-based cross-dedup to avoid materialising the full
    correlation matrix on 307K rows.

    Pipeline steps
    --------------
    1. Load X_tree_raw.parquet
    2. Run DFS via build_featuretools_feature_store (with checkpoint)
       - Internal correlation dedup at 0.90 threshold runs inside this call;
         no second dedup pass is needed or permitted.
    3. Cross-dedup on a 50K-row random sample
    4. Compute time features via engineer_time_features
    5. Merge: pd.concat([X_tree_raw, X_dfs_dedup, X_time], axis=1)
    6. fillna(-999), cast to float32, save

    Parameters
    ----------
    data_dir : Path | str
        Directory containing raw data CSVs (data/ folder).

    Returns
    -------
    X_tree_dfs : pd.DataFrame
        Merged feature matrix, shape (307511, N>155).
        All float32, no NaN (-999 sentinel used).
        Index name = SK_ID_CURR.
    feature_columns : list[str]
        Column names in X_tree_dfs, in order.

    Raises
    ------
    FileNotFoundError
        If X_tree_raw.parquet or required CSV files are missing.
    AssertionError
        If final shape does not satisfy (307511, N>155).
    """
    from src.auto_features import build_featuretools_feature_store

    data_dir = Path(data_dir)
    output_dir = _PROJECT_ROOT / "data" / "processed"
    models_dir = _PROJECT_ROOT / "models"
    checkpoint_path = output_dir / "X_dfs_checkpoint.parquet"

    X_tree_raw_path = output_dir / "X_tree_raw.parquet"
    if not X_tree_raw_path.exists():
        raise FileNotFoundError(
            f"X_tree_raw.parquet not found at {X_tree_raw_path}; run Phase 04.2.1 first"
        )

    # -----------------------------------------------------------------------
    # Step 1: Load X_tree_raw and immediately cast to float32
    # -----------------------------------------------------------------------
    print("Step 1/6: Loading X_tree_raw...")
    X_tree_raw = pd.read_parquet(X_tree_raw_path)

    # If SK_ID_CURR is a column, move it to the index
    if "SK_ID_CURR" in X_tree_raw.columns and X_tree_raw.index.name != "SK_ID_CURR":
        X_tree_raw = X_tree_raw.set_index("SK_ID_CURR")
    elif X_tree_raw.index.name != "SK_ID_CURR":
        # Restore SK_ID_CURR index from application CSV if missing
        _app_ids = pd.read_csv(
            data_dir / "application_train.csv", usecols=["SK_ID_CURR"]
        )["SK_ID_CURR"].values
        X_tree_raw = X_tree_raw.copy()
        X_tree_raw.index = pd.Index(_app_ids, name="SK_ID_CURR")

    X_tree_raw = X_tree_raw.astype("float32")
    print(f"  X_tree_raw: {X_tree_raw.shape}")

    # -----------------------------------------------------------------------
    # Step 2: Run DFS with checkpoint (most memory-intensive step)
    # build_featuretools_feature_store runs internal correlation dedup at
    # corr_threshold=0.90 — a second dedup pass would waste memory and time.
    # -----------------------------------------------------------------------
    print("Step 2/6: Running DFS (or loading checkpoint)...")
    if checkpoint_path.exists():
        print(f"  Checkpoint found — loading from {checkpoint_path}")
        X_dfs = pd.read_parquet(checkpoint_path).astype("float32")
    else:
        y_train_path = output_dir / "y_train.parquet"
        if not y_train_path.exists():
            raise FileNotFoundError(f"y_train.parquet not found at {y_train_path}")
        y_train = pd.read_parquet(y_train_path).squeeze()

        X_dfs, _feature_defs, _selected_cols = build_featuretools_feature_store(
            data_dir=data_dir,
            y_train=y_train,
            output_path=checkpoint_path,  # saves deduped result to disk
            agg_primitives=None,
            max_depth=1,
            iv_threshold=0.02,
            corr_threshold=0.90,
            n_jobs=1,
        )

        # Reload from disk as float32 to free the float64 copy
        del X_dfs, _feature_defs, _selected_cols, y_train
        gc.collect()
        X_dfs = pd.read_parquet(checkpoint_path).astype("float32")

    print(f"  X_dfs after internal dedup: {X_dfs.shape}")

    # -----------------------------------------------------------------------
    # Step 3: Cross-dedup DFS vs raw on a 50K-row sample
    # Avoids materialising the full (307K × N_raw × N_dfs) correlation matrix.
    # -----------------------------------------------------------------------
    print("Step 3/6: Cross-dedup DFS vs raw (50K-row sample)...")
    _SAMPLE_N = 50_000
    _CORR_THRESHOLD = 0.90

    common_idx = X_dfs.index.intersection(X_tree_raw.index)
    if len(common_idx) < _SAMPLE_N:
        sample_idx = common_idx
    else:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(common_idx, size=_SAMPLE_N, replace=False)

    X_dfs_sample = X_dfs.loc[sample_idx]
    X_raw_sample = X_tree_raw.loc[sample_idx]

    # For each DFS column find its maximum absolute correlation with any raw column.
    # corrwith(Series) is used here because Series.corr() only accepts a Series,
    # not a DataFrame — using DataFrame.corrwith() avoids a TypeError.
    cols_to_keep = []
    for col in X_dfs.columns:
        max_corr_with_raw = X_raw_sample.corrwith(X_dfs_sample[col]).abs().max()
        if max_corr_with_raw <= _CORR_THRESHOLD:
            cols_to_keep.append(col)

    del X_dfs_sample, X_raw_sample
    gc.collect()

    X_dfs_dedup = X_dfs[cols_to_keep]
    del X_dfs
    gc.collect()
    print(f"  After cross-dedup: {X_dfs_dedup.shape[1]} DFS columns remain")

    # -----------------------------------------------------------------------
    # Step 4: Compute time features
    # -----------------------------------------------------------------------
    print("Step 4/6: Computing time features...")
    X_time = engineer_time_features(data_dir).astype("float32")
    print(f"  X_time: {X_time.shape}")

    # -----------------------------------------------------------------------
    # Step 5: Merge raw + DFS + time features on SK_ID_CURR index
    # -----------------------------------------------------------------------
    print("Step 5/6: Merging raw + DFS + time features...")
    # Remove DFS columns that are already in X_tree_raw (exact duplicates)
    dfs_cols_to_keep = [col for col in X_dfs_dedup.columns if col not in X_tree_raw.columns]
    X_dfs_dedup_filtered = X_dfs_dedup[dfs_cols_to_keep]
    print(f"  DFS columns after removing duplicates: {X_dfs_dedup_filtered.shape[1]} (was {X_dfs_dedup.shape[1]})")

    # Remove time features that are already in X_tree_raw or DFS
    time_cols_to_keep = [col for col in X_time.columns if col not in X_tree_raw.columns and col not in X_dfs_dedup_filtered.columns]
    X_time_filtered = X_time[time_cols_to_keep]
    print(f"  Time columns after removing duplicates: {X_time_filtered.shape[1]} (was {X_time.shape[1]})")

    X_merged = pd.concat([X_tree_raw, X_dfs_dedup_filtered, X_time_filtered], axis=1)

    del X_tree_raw, X_dfs_dedup, X_time
    gc.collect()
    print(f"  After merge: {X_merged.shape}")

    # -----------------------------------------------------------------------
    # Step 6: Fill NaN, validate, and save
    # -----------------------------------------------------------------------
    print("Step 6/6: Finalising and saving...")
    X_merged = X_merged.fillna(_NAN_SENTINEL)

    assert X_merged.isna().sum().sum() == 0, "NaN values remain after fillna"
    assert X_merged.shape[0] == 307511, f"Expected 307511 rows, got {X_merged.shape[0]}"
    assert X_merged.shape[1] > 155, f"Expected >155 columns, got {X_merged.shape[1]}"
    assert X_merged.index.name == "SK_ID_CURR", f"Index name is {X_merged.index.name}"

    output_dir.mkdir(parents=True, exist_ok=True)
    X_merged.to_parquet(output_dir / "X_tree_dfs.parquet", index=True)
    print(f"  Saved X_tree_dfs.parquet: {X_merged.shape}")

    feature_columns = list(X_merged.columns)
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / "dfs_feature_columns.pkl", "wb") as fh:
        pickle.dump(feature_columns, fh)

    return X_merged, feature_columns
