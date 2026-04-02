"""
test_features.py
----------------
Unit tests for credit_engine/features.py.

Run with
--------
    pytest tests/test_features.py -v
"""

import numpy as np
import pandas as pd
import pytest

from credit_engine.features import (
    build_features,
    compute_woe_iv,
    engineer_application_features,
    select_features_by_iv,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal synthetic DataFrame for feature tests."""
    return pd.DataFrame(
        {
            "loan_amount": [10_000, 25_000, 5_000],
            "income": [30_000, 60_000, 15_000],
            "age": [35, 52, 28],
            "default_flag": [0, 0, 1],
        }
    )


@pytest.fixture
def application_fixture() -> pd.DataFrame:
    """
    Synthetic DataFrame mirroring the Home Credit application table schema.

    Rows are designed to exercise specific edge cases:
      Row 0: normal applicant — all values present, no edge cases
      Row 1: unemployed sentinel — DAYS_EMPLOYED == 365243
      Row 2: zero-income — AMT_INCOME_TOTAL == 0 (division-by-zero guard)
      Row 3: zero-annuity — AMT_ANNUITY == 0 (division-by-zero guard)
      Row 4: missing EXT_SOURCE — all three EXT_SOURCE_* are NaN
      Row 5: partial EXT_SOURCE — only EXT_SOURCE_2 is valid
      Row 6: FLAG_DOCUMENT_3 present — HIGH_RISK_DOC_MISSING should be 0
    """
    data = {
        # Loan financial columns
        "AMT_CREDIT": [500_000, 300_000, 200_000, 150_000, 400_000, 350_000, 600_000],
        "AMT_INCOME_TOTAL": [100_000, 80_000, 0, 90_000, 120_000, 110_000, 200_000],
        "AMT_ANNUITY": [25_000, 15_000, 10_000, 0, 20_000, 18_000, 30_000],
        "AMT_GOODS_PRICE": [450_000, 280_000, 190_000, 140_000, 380_000, 320_000, 550_000],
        # Days (negative = days before application)
        "DAYS_BIRTH": [-10_000, -15_000, -8_000, -12_000, -20_000, -18_000, -14_000],
        "DAYS_EMPLOYED": [
            -2_000,   # Row 0: normal employed
            365_243,  # Row 1: unemployment sentinel
            -500,     # Row 2: short tenure
            -3_000,   # Row 3: normal employed
            -5_000,   # Row 4: long tenure
            -1_200,   # Row 5: moderate tenure
            -4_000,   # Row 6: normal employed
        ],
        # External bureau scores (structural missingness ~45-55%)
        "EXT_SOURCE_1": [0.6, 0.4, 0.7, np.nan, np.nan, np.nan, 0.5],
        "EXT_SOURCE_2": [0.7, 0.5, 0.8, np.nan, np.nan, 0.6, 0.6],
        "EXT_SOURCE_3": [0.5, 0.3, 0.6, np.nan, np.nan, np.nan, 0.7],
        # Document flags (FLAG_DOCUMENT_3 is the high-risk indicator)
        "FLAG_DOCUMENT_2": [0, 1, 0, 1, 0, 1, 1],
        "FLAG_DOCUMENT_3": [1, 0, 1, 0, 1, 0, 1],  # Row 1,3,5 missing = HIGH_RISK
        "FLAG_DOCUMENT_4": [0, 0, 1, 0, 0, 1, 0],
        "FLAG_DOCUMENT_5": [1, 0, 0, 0, 1, 0, 1],
        "FLAG_DOCUMENT_6": [0, 1, 0, 1, 0, 0, 0],
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Existing tests — kept intact
# ---------------------------------------------------------------------------


def test_build_features_returns_dataframe(sample_df):
    result = build_features(sample_df)
    assert isinstance(result, pd.DataFrame)


def test_build_features_no_rows_dropped(sample_df):
    result = build_features(sample_df)
    assert len(result) == len(sample_df)


# ---------------------------------------------------------------------------
# Task 2.1 — engineer_application_features() tests (TDD: RED phase)
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = [
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_TERM",
    "GOODS_CREDIT_RATIO",
    "AGE_YEARS",
    "YEARS_EMPLOYED",
    "EMPLOYED_TO_AGE_RATIO",
    "DOCUMENTS_SUBMITTED",
    "HIGH_RISK_DOC_MISSING",
    "EXT_SOURCE_MEAN",
    "EXT_SOURCE_MIN",
]


def test_engineer_application_features_creates_all_columns(application_fixture):
    """All 11 expected feature columns must be present in the output."""
    result = engineer_application_features(application_fixture)
    missing = [c for c in EXPECTED_COLUMNS if c not in result.columns]
    assert missing == [], f"Missing columns: {missing}"


def test_division_by_zero_ratios_zero_denominator(application_fixture):
    """
    Rows where AMT_INCOME_TOTAL == 0 (row 2) or AMT_ANNUITY == 0 (row 3)
    must produce ratio == 0, not inf or NaN.
    """
    result = engineer_application_features(application_fixture)

    # Row 2: AMT_INCOME_TOTAL == 0 → CREDIT_INCOME_RATIO and ANNUITY_INCOME_RATIO should be 0
    assert result.loc[2, "CREDIT_INCOME_RATIO"] == 0, "Zero-income CREDIT_INCOME_RATIO must be 0"
    assert result.loc[2, "ANNUITY_INCOME_RATIO"] == 0, "Zero-income ANNUITY_INCOME_RATIO must be 0"

    # Row 3: AMT_ANNUITY == 0 → CREDIT_TERM should be 0
    assert result.loc[3, "CREDIT_TERM"] == 0, "Zero-annuity CREDIT_TERM must be 0"

    # No inf values in any ratio column
    ratio_cols = ["CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM", "GOODS_CREDIT_RATIO"]
    for col in ratio_cols:
        assert not result[col].isin([np.inf, -np.inf]).any(), f"{col} must not contain inf"


def test_days_employed_sentinel_clipped(application_fixture):
    """
    DAYS_EMPLOYED == 365243 is the unemployment sentinel (~18% of rows).
    YEARS_EMPLOYED must be 0 for these rows, not ~1000 years.
    """
    result = engineer_application_features(application_fixture)
    # Row 1 has DAYS_EMPLOYED == 365243
    assert result.loc[1, "YEARS_EMPLOYED"] == pytest.approx(0.0), (
        "Unemployment sentinel 365243 must map to YEARS_EMPLOYED == 0"
    )
    # Normal rows must not be zero (row 0: -2000 days → ~5.47 years)
    assert result.loc[0, "YEARS_EMPLOYED"] > 0, "Normal employment must yield positive YEARS_EMPLOYED"


def test_employed_to_age_ratio_safe(application_fixture):
    """
    EMPLOYED_TO_AGE_RATIO must never be inf or NaN, even when edge cases
    produce a near-zero AGE_YEARS (guard via np.where).
    """
    result = engineer_application_features(application_fixture)
    col = result["EMPLOYED_TO_AGE_RATIO"]
    assert not col.isin([np.inf, -np.inf]).any(), "EMPLOYED_TO_AGE_RATIO must not contain inf"
    assert not col.isna().any(), "EMPLOYED_TO_AGE_RATIO must not contain NaN"
    assert (col >= 0).all(), "EMPLOYED_TO_AGE_RATIO must be non-negative"


def test_ext_source_nan_handling(application_fixture):
    """
    Rows with partial or full NaN in EXT_SOURCE_* must be handled:
      - Row 4: all three NaN → EXT_SOURCE_MEAN and EXT_SOURCE_MIN should be -999 (sentinel)
      - Row 5: only EXT_SOURCE_2 valid → nanmean/nanmin must use that value
    """
    result = engineer_application_features(application_fixture)

    # Row 4: all NaN → fill sentinel
    assert result.loc[4, "EXT_SOURCE_MEAN"] == pytest.approx(-999.0), (
        "All-NaN EXT_SOURCE row must use -999 sentinel"
    )
    assert result.loc[4, "EXT_SOURCE_MIN"] == pytest.approx(-999.0), (
        "All-NaN EXT_SOURCE row must use -999 sentinel"
    )

    # Row 5: only EXT_SOURCE_2 == 0.6 is valid
    assert result.loc[5, "EXT_SOURCE_MEAN"] == pytest.approx(0.6), (
        "Single-valid EXT_SOURCE row must return that value as mean"
    )
    assert result.loc[5, "EXT_SOURCE_MIN"] == pytest.approx(0.6), (
        "Single-valid EXT_SOURCE row must return that value as min"
    )


def test_documents_submitted_sum(application_fixture):
    """
    DOCUMENTS_SUBMITTED must equal the sum of all FLAG_DOCUMENT_* columns per row.
    """
    result = engineer_application_features(application_fixture)
    flag_cols = [c for c in application_fixture.columns if c.startswith("FLAG_DOCUMENT_")]
    expected = application_fixture[flag_cols].sum(axis=1)
    pd.testing.assert_series_equal(
        result["DOCUMENTS_SUBMITTED"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_high_risk_doc_missing_logic(application_fixture):
    """
    HIGH_RISK_DOC_MISSING == 1 iff FLAG_DOCUMENT_3 == 0, else 0.
    Rows 1, 3, 5 have FLAG_DOCUMENT_3 == 0 → HIGH_RISK_DOC_MISSING == 1.
    Rows 0, 2, 4, 6 have FLAG_DOCUMENT_3 == 1 → HIGH_RISK_DOC_MISSING == 0.
    """
    result = engineer_application_features(application_fixture)
    for idx, flag_val in enumerate(application_fixture["FLAG_DOCUMENT_3"]):
        expected = 1 if flag_val == 0 else 0
        actual = result.loc[idx, "HIGH_RISK_DOC_MISSING"]
        assert actual == expected, (
            f"Row {idx}: FLAG_DOCUMENT_3={flag_val} → "
            f"expected HIGH_RISK_DOC_MISSING={expected}, got {actual}"
        )


def test_no_nan_in_ratio_columns(application_fixture):
    """
    All RATIO columns must have NaN filled with -999 sentinel.
    No real NaN should remain in the output for ratio features.
    """
    result = engineer_application_features(application_fixture)
    ratio_cols = [c for c in result.columns if "RATIO" in c or c == "CREDIT_TERM"]
    for col in ratio_cols:
        assert not result[col].isna().any(), (
            f"{col} must not contain NaN — should be filled with -999 sentinel"
        )


# ---------------------------------------------------------------------------
# Task 2.2 — compute_woe_iv() and select_features_by_iv() tests (TDD: RED)
# ---------------------------------------------------------------------------


def test_compute_woe_iv_returns_correct_shape():
    """compute_woe_iv returns (DataFrame, float) with the five required columns."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({"score": rng.uniform(0, 1, n)})
    target = pd.Series(rng.binomial(1, 0.3, n))

    tbl, total_iv = compute_woe_iv(df, "score", target, bins=5)

    assert isinstance(tbl, pd.DataFrame)
    assert isinstance(total_iv, float)
    expected_cols = {"bin_range", "event_count", "non_event_count", "woe", "iv_contrib"}
    assert set(tbl.columns) == expected_cols
    # bins=5 with duplicates='drop' may produce fewer rows; at least 1.
    assert 1 <= len(tbl) <= 5


def test_compute_woe_iv_handles_zero_event_bins():
    """
    When a bin has zero events WoE must be clipped to +5 (no defaults = safe).
    When a bin has zero non-events WoE must be clipped to -5 (all defaults = risky).
    No inf or NaN must appear in the binning table.
    """
    # Rows 0-9 are all defaults; rows 10-99 are all non-defaults.
    # With bins=10 each quantile bin holds exactly 10 rows, so:
    #   bin 0 → 10 events, 0 non-events → WoE = -5
    #   bins 1-9 → 0 events, 10 non-events → WoE = +5
    feature = np.arange(100, dtype=float)
    target = pd.Series([1] * 10 + [0] * 90)
    df = pd.DataFrame({"feature": feature})

    tbl, _ = compute_woe_iv(df, "feature", target, bins=10)

    assert np.isfinite(tbl["woe"]).all(), "WoE column must contain only finite values"

    zero_ne = tbl[tbl["non_event_count"] == 0]
    assert len(zero_ne) > 0, "Expected at least one bin with zero non-events"
    assert all(v == -5.0 for v in zero_ne["woe"]), "Zero-non-event bin must have WoE == -5"

    zero_e = tbl[tbl["event_count"] == 0]
    assert len(zero_e) > 0, "Expected at least one bin with zero events"
    assert all(v == 5.0 for v in zero_e["woe"]), "Zero-event bin must have WoE == +5"


def test_compute_woe_iv_finite():
    """All WoE values must be finite, IV non-negative, and IV equals sum of contributions."""
    rng = np.random.default_rng(42)
    n = 1000
    df = pd.DataFrame({"value": rng.uniform(0, 100, n)})
    target = pd.Series(rng.binomial(1, 0.2, n))

    tbl, total_iv = compute_woe_iv(df, "value", target, bins=10)

    assert np.isfinite(tbl["woe"]).all(), "All WoE values must be finite"
    assert total_iv >= 0, "Total IV must be non-negative"
    assert total_iv == pytest.approx(tbl["iv_contrib"].sum(), abs=1e-9)


def test_select_features_by_iv_returns_sorted_dict():
    """select_features_by_iv returns a dict with str keys, float values, sorted descending."""
    rng = np.random.default_rng(42)
    n = 500
    score = rng.uniform(0, 1, n)
    df = pd.DataFrame({
        "score": score,
        "noise": rng.uniform(0, 1, n),
    })
    target = pd.Series((score < 0.3).astype(int))

    iv_dict = select_features_by_iv(df, target, min_iv=0.0, bins=5)

    assert isinstance(iv_dict, dict)
    assert all(isinstance(k, str) for k in iv_dict)
    assert all(isinstance(v, float) for v in iv_dict.values())
    values = list(iv_dict.values())
    assert values == sorted(values, reverse=True), "IV dict must be sorted descending by IV"


def test_select_features_by_iv_skips_non_numeric():
    """Non-numeric (object) columns must be silently excluded from IV computation."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "numeric_a": rng.uniform(0, 1, n),
        "numeric_b": rng.exponential(10, n),
        "category": ["A", "B"] * 100,
        "flag": ["yes", "no"] * 100,
    })
    target = pd.Series(rng.binomial(1, 0.2, n))

    iv_dict = select_features_by_iv(df, target, min_iv=0.0)

    assert "category" not in iv_dict
    assert "flag" not in iv_dict
    assert set(iv_dict.keys()) == {"numeric_a", "numeric_b"}


def test_select_features_by_iv_min_iv_threshold():
    """Only features with IV >= min_iv appear in the output; all returned IVs satisfy the threshold."""
    rng = np.random.default_rng(42)
    n = 1000
    df = pd.DataFrame({
        "a": rng.uniform(0, 1, n),
        "b": rng.uniform(0, 1, n),
        "c": rng.uniform(0, 1, n),
    })
    target = pd.Series(rng.binomial(1, 0.15, n))

    iv_all = select_features_by_iv(df, target, min_iv=0.0, bins=5)
    threshold = max(iv_all.values()) * 0.5

    iv_filtered = select_features_by_iv(df, target, min_iv=threshold, bins=5)

    for col, iv_val in iv_filtered.items():
        assert iv_val >= threshold, f"{col}: IV {iv_val:.6f} is below threshold {threshold:.6f}"
    assert len(iv_filtered) <= len(iv_all)


def test_compute_woe_iv_sentinel_creates_separate_bin():
    """
    Option B: -999 sentinel values form a dedicated '-999 (missing)' bin,
    keeping them separate from the true low-value quantile bins.
    """
    rng = np.random.default_rng(42)
    n = 200
    raw = np.where(rng.random(n) < 0.2, -999.0, rng.uniform(0, 1, n))
    df = pd.DataFrame({"feature": raw})
    target = pd.Series(rng.binomial(1, 0.3, n))

    tbl, _ = compute_woe_iv(df, "feature", target, bins=5)

    assert "-999 (missing)" in tbl["bin_range"].values, (
        "Sentinel -999 values must produce a dedicated '-999 (missing)' bin"
    )
    non_sentinel_bins = tbl[tbl["bin_range"] != "-999 (missing)"]
    assert len(non_sentinel_bins) >= 1, "At least one non-sentinel quantile bin must exist"


def test_compute_woe_iv_does_not_mutate_input():
    """compute_woe_iv must not mutate the input DataFrame or the target Series."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({"feature": rng.uniform(0, 1, n)})
    target = pd.Series(rng.binomial(1, 0.2, n))

    df_copy = df.copy()
    target_copy = target.copy()

    compute_woe_iv(df, "feature", target, bins=5)

    pd.testing.assert_frame_equal(df, df_copy)
    pd.testing.assert_series_equal(target, target_copy)


def test_engineer_application_features_does_not_mutate_input(application_fixture):
    """
    Immutability: the input DataFrame must be unchanged after calling
    engineer_application_features(). No in-place mutations allowed.
    """
    original_columns = set(application_fixture.columns)
    original_values = application_fixture.copy()

    engineer_application_features(application_fixture)

    assert set(application_fixture.columns) == original_columns, (
        "Input DataFrame columns were mutated"
    )
    pd.testing.assert_frame_equal(application_fixture, original_values)
