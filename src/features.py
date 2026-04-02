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

import warnings

import numpy as np
import pandas as pd

# Sentinel value used for DAYS_EMPLOYED when the applicant is unemployed.
# Home Credit encodes unemployment as 365243 (a large positive number) instead
# of a negative value like normal employment entries.
_DAYS_EMPLOYED_SENTINEL: int = 365_243

# Tree-friendly fill value for missing/undefined features.  Using -999 instead
# of 0 or mean avoids shifting the distribution and lets gradient boosting
# models learn a dedicated "missing" split.
_NAN_SENTINEL: float = -999.0


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

    ratio_cols = ["CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM", "GOODS_CREDIT_RATIO"]
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

    Parameters
    ----------
    df : pd.DataFrame
        Must contain EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3.

    Returns
    -------
    pd.DataFrame
        Copy of df with EXT_SOURCE_MEAN and EXT_SOURCE_MIN.
    """
    out = df.copy()

    ext = np.column_stack([
        out["EXT_SOURCE_1"].to_numpy(dtype=float),
        out["EXT_SOURCE_2"].to_numpy(dtype=float),
        out["EXT_SOURCE_3"].to_numpy(dtype=float),
    ])

    # Suppress numpy's "All-NaN slice" warning — we handle that case with fillna.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ext_mean = np.nanmean(ext, axis=1)
        ext_min = np.nanmin(ext, axis=1)

    # Rows where all three sources are NaN produce NaN from nanmean/nanmin.
    out["EXT_SOURCE_MEAN"] = pd.Series(ext_mean, index=out.index).fillna(_NAN_SENTINEL)
    out["EXT_SOURCE_MIN"] = pd.Series(ext_min, index=out.index).fillna(_NAN_SENTINEL)

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
        Copy of df with 11 additional engineered columns:
        CREDIT_INCOME_RATIO, ANNUITY_INCOME_RATIO, CREDIT_TERM,
        GOODS_CREDIT_RATIO, AGE_YEARS, YEARS_EMPLOYED,
        EMPLOYED_TO_AGE_RATIO, DOCUMENTS_SUBMITTED,
        HIGH_RISK_DOC_MISSING, EXT_SOURCE_MEAN, EXT_SOURCE_MIN.

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
    return result


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
