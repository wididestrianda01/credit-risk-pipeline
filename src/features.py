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
