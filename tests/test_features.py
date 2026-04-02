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

from credit_engine.features import build_features, engineer_application_features


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
