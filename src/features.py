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

import pickle
import warnings
from pathlib import Path

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
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    X_final.to_parquet("data/processed/X_features.parquet", index=False)
    with open("models/woe_mappings.pkl", "wb") as fh:
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


def build_raw_feature_store(
    X: pd.DataFrame,
    y: pd.Series,
    min_iv: float = _IV_WEAK,
    output_path: str = "data/processed/X_raw_features.parquet",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build raw (non-WoE) feature store for gradient boosting models.

    Applies the same pipeline as build_feature_store (engineering, IV filter,
    variance filter, correlation dedup) but SKIPS WoE transformation entirely.
    Raw continuous values are preserved, allowing tree models to find optimal splits.

    Parameters
    ----------
    X : pd.DataFrame
        Raw application DataFrame (joined 7-table frame from load_data).
    y : pd.Series
        Binary target (1 = default, 0 = non-default), aligned with X.
    min_iv : float, optional
        Minimum Information Value threshold for feature selection (default 0.02).
    output_path : str, optional
        Path to save the raw feature matrix as parquet
        (default "data/processed/X_raw_features.parquet").

    Returns
    -------
    X_final : pd.DataFrame
        Raw (continuous-valued) feature matrix after IV, variance, and
        correlation filtering. Contains no NaN values (sentinel -999 used).
    feature_columns : list[str]
        Column names in X_final, to be passed to apply_raw_feature_store.
        Also saved to models/raw_feature_columns.pkl.

    Notes
    -----
    **Data leakage prevention:** IV thresholds are fit only on training data.
    Never call select_features_by_iv on test data. Use apply_raw_feature_store.

    **WoE skipped:** Unlike build_feature_store, this function does NOT apply
    WoE binning. All values remain as raw floats (or -999 for missing).
    This is better for gradient boosting which benefits from continuous distributions.
    """
    X_eng = engineer_application_features(X) if "AMT_CREDIT" in X.columns else X.copy()
    print(f"Raw features: {X_eng.shape[1]}")

    # Replace inf/nan sentinel
    X_filled = X_eng.copy()
    X_filled = X_filled.replace([np.inf, -np.inf], np.nan)
    X_filled = X_filled.fillna(_NAN_SENTINEL)

    # IV filter
    iv_features = select_features_by_iv(X_filled, y, min_iv=min_iv, bins=10)
    X_iv = X_filled[list(iv_features.keys())].copy()
    print(f"After IV filter (IV >= {min_iv}): {X_iv.shape[1]}")

    # Variance filter: drop constant columns then bottom 5% by variance
    variances = X_iv.var()
    positive_var = variances[variances > 0]
    if len(positive_var) == 0:
        X_final = pd.DataFrame(index=X_iv.index)
    else:
        var_threshold = positive_var.quantile(0.05)
        keep_cols = positive_var[positive_var >= var_threshold].index
        X_final = X_iv[keep_cols].copy()

    print(f"After variance filter: {X_final.shape[1]}")

    # Correlation deduplication: for pairs with |r| > 0.90, drop the lower-IV feature.
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
            print(f"After correlation dedup (|r| > {_CORR_THRESHOLD}): {X_final.shape[1]}")

    # Persist artifacts
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    X_final.to_parquet(output_path, index=False)
    feature_columns = list(X_final.columns)
    with open("models/raw_feature_columns.pkl", "wb") as fh:
        pickle.dump(feature_columns, fh)

    return X_final, feature_columns


def apply_raw_feature_store(
    X: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Apply stored raw feature selection to inference data.

    Uses feature column list from build_raw_feature_store. No re-fitting occurs.
    Missing columns are filled with _NAN_SENTINEL (-999). All inf values replaced
    with nan then filled with sentinel.

    Parameters
    ----------
    X : pd.DataFrame
        Raw or partially-processed inference DataFrame.
    feature_columns : list[str]
        Column list from build_raw_feature_store (or loaded from
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
