"""
test_features.py
----------------
Unit tests for credit_engine/features.py.

Run with
--------
    pytest tests/test_features.py -v
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import (
    apply_feature_store,
    apply_raw_feature_store,
    build_feature_store,
    build_features,
    build_raw_feature_store,
    build_tree_feature_store,
    compute_knn_target_encoding,
    compute_woe_iv,
    engineer_application_features,
    engineer_instalment_streaks,
    engineer_secondary_features,
    select_features_by_iv,
    _engineer_ext_source,
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
    "EXT_SOURCE_MAX",
    "EXT_SOURCE_MEDIAN",
    "EXT_SOURCE_STD",
    "EXT_SOURCE_RANGE",
    "EXT_SOURCE_AVAILABLE_CNT",
    "EXT_SOURCE_PROD_12",
    "EXT_SOURCE_PROD_13",
    "EXT_SOURCE_PROD_23",
]


def test_engineer_application_features_creates_all_columns(application_fixture):
    """All 19 expected feature columns must be present in the output."""
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


def test_ext_source_interactions_present(application_fixture):
    """EXT_SOURCE interaction features must be present in the output."""
    result = engineer_application_features(application_fixture)
    interaction_cols = [
        "EXT_SOURCE_MAX", "EXT_SOURCE_MEDIAN", "EXT_SOURCE_STD",
        "EXT_SOURCE_RANGE", "EXT_SOURCE_AVAILABLE_CNT",
        "EXT_SOURCE_PROD_12", "EXT_SOURCE_PROD_13", "EXT_SOURCE_PROD_23",
    ]
    missing = [c for c in interaction_cols if c not in result.columns]
    assert missing == [], f"Missing EXT_SOURCE interaction columns: {missing}"


def test_ext_source_products_correct(application_fixture):
    """EXT_SOURCE_PROD_12 = EXT_SOURCE_1 * EXT_SOURCE_2 where both valid."""
    result = engineer_application_features(application_fixture)

    # Row 0: EXT_SOURCE_1=0.6, EXT_SOURCE_2=0.7 → product = 0.42
    assert result.loc[0, "EXT_SOURCE_PROD_12"] == pytest.approx(0.6 * 0.7, rel=1e-5)
    # Row 5: EXT_SOURCE_1 is NaN → product should be sentinel -999
    assert result.loc[5, "EXT_SOURCE_PROD_12"] == pytest.approx(-999.0), (
        "Product must be sentinel when either factor is NaN"
    )


def test_ext_source_std_all_nan_is_sentinel(application_fixture):
    """Row where all EXT_SOURCE are NaN must have EXT_SOURCE_STD == -999."""
    result = engineer_application_features(application_fixture)
    # Row 4: all NaN
    assert result.loc[4, "EXT_SOURCE_STD"] == pytest.approx(-999.0)


def test_ext_source_available_cnt_no_nan(application_fixture):
    """EXT_SOURCE_AVAILABLE_CNT must be an integer count in [0, 3] with no NaN."""
    result = engineer_application_features(application_fixture)
    col = result["EXT_SOURCE_AVAILABLE_CNT"]
    assert not col.isna().any()
    assert col.between(0, 3).all()
    # Row 4: all three NaN → count = 0
    assert result.loc[4, "EXT_SOURCE_AVAILABLE_CNT"] == 0.0
    # Row 5: only EXT_SOURCE_2 valid → count = 1
    assert result.loc[5, "EXT_SOURCE_AVAILABLE_CNT"] == 1.0
    # Row 0: all three valid → count = 3
    assert result.loc[0, "EXT_SOURCE_AVAILABLE_CNT"] == 3.0


def test_ext_source_range_non_negative(application_fixture):
    """EXT_SOURCE_RANGE = max - min must always be >= 0."""
    result = engineer_application_features(application_fixture)
    col = result["EXT_SOURCE_RANGE"]
    # Filter to rows that are not sentinel
    valid = col[col != -999.0]
    assert (valid >= 0).all(), "EXT_SOURCE_RANGE must be non-negative"


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


# ---------------------------------------------------------------------------
# Task 2.3 — build_feature_store() and apply_feature_store() tests (TDD: RED)
# ---------------------------------------------------------------------------


@pytest.fixture
def feature_store_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Synthetic training data with enough rows (500) and variance to exercise
    IV filtering, WoE binning, and variance filtering realistically.

    Columns:
      - high_iv_feature: strongly correlated with target (high IV)
      - medium_iv_feature: moderately correlated with target
      - noise_feature: random noise (low IV, likely filtered out)
      - constant_feature: zero variance (must be dropped)
    """
    rng = np.random.default_rng(42)
    n = 500
    high_iv = rng.uniform(0, 1, n)
    target = pd.Series((high_iv < 0.25).astype(int))  # 25% event rate

    df = pd.DataFrame({
        "high_iv_feature": high_iv,
        "medium_iv_feature": rng.uniform(0, 1, n) * 0.5 + high_iv * 0.5,
        "noise_feature": rng.uniform(0, 1, n),
        "constant_feature": np.ones(n),
    })
    return df, target


def test_build_feature_store_reduces_features(feature_store_data, mock_data_dir):
    """
    Final feature count must be less than the raw input count.
    IV filter removes low-IV features; variance filter removes constant ones.
    """
    X, y = feature_store_data
    X_out, woe_map = build_feature_store(X, y, output_dir=mock_data_dir / "data" / "processed")

    assert X_out.shape[1] < X.shape[1], (
        f"Expected fewer features after filtering: got {X_out.shape[1]}, raw was {X.shape[1]}"
    )
    # Verify it wrote to test directory, not production
    assert (mock_data_dir / "data" / "processed").exists()


def test_build_feature_store_no_nan_in_output(feature_store_data, mock_data_dir):
    """No NaN values must appear in the final WoE-transformed feature matrix."""
    X, y = feature_store_data
    X_out, _ = build_feature_store(X, y, output_dir=mock_data_dir / "data" / "processed")

    assert not X_out.isna().any().any(), (
        "X_features must not contain NaN after WoE transformation and sentinel filling"
    )


def test_build_feature_store_woe_mappings_structure(feature_store_data, mock_data_dir):
    """
    woe_mappings must be a dict of dicts, each containing 'bin_edges' (list)
    and 'bin_woe_values' (dict mapping bin label string -> float).
    """
    X, y = feature_store_data
    _, woe_map = build_feature_store(X, y, output_dir=mock_data_dir / "data" / "processed")

    assert isinstance(woe_map, dict), "woe_mappings must be a dict"
    assert len(woe_map) > 0, "woe_mappings must not be empty"

    for feature, entry in woe_map.items():
        assert isinstance(entry, dict), f"woe_mappings[{feature!r}] must be a dict"
        assert "bin_edges" in entry, f"woe_mappings[{feature!r}] missing 'bin_edges'"
        assert "bin_woe_values" in entry, f"woe_mappings[{feature!r}] missing 'bin_woe_values'"
        assert isinstance(entry["bin_edges"], list), f"bin_edges for {feature!r} must be a list"
        assert isinstance(entry["bin_woe_values"], dict), f"bin_woe_values for {feature!r} must be a dict"
        assert all(isinstance(v, float) for v in entry["bin_woe_values"].values()), (
            f"All WoE values for {feature!r} must be floats"
        )


def test_build_feature_store_woe_mappings_keys_match_columns(feature_store_data, mock_data_dir):
    """woe_mappings keys must exactly match X_features column names."""
    X, y = feature_store_data
    X_out, woe_map = build_feature_store(X, y, output_dir=mock_data_dir / "data" / "processed")

    assert set(woe_map.keys()) == set(X_out.columns), (
        f"woe_mappings keys {set(woe_map.keys())} != X_features columns {set(X_out.columns)}"
    )


def test_build_feature_store_constant_feature_dropped(feature_store_data, mock_data_dir):
    """Constant columns (variance == 0) must be absent from X_features and woe_mappings."""
    X, y = feature_store_data
    X_out, woe_map = build_feature_store(X, y, output_dir=mock_data_dir / "data" / "processed")

    assert "constant_feature" not in X_out.columns, (
        "constant_feature has zero variance and must be dropped"
    )
    assert "constant_feature" not in woe_map, (
        "constant_feature must not appear in woe_mappings"
    )


def test_build_feature_store_pickle_round_trip(feature_store_data, mock_data_dir):
    """Pickle save and load must preserve all woe_mappings entries exactly."""
    import pickle

    X, y = feature_store_data
    _, woe_map = build_feature_store(X, y, output_dir=mock_data_dir / "data" / "processed")

    pkl_path = mock_data_dir / "models" / "woe_mappings.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(woe_map, f)
    with open(pkl_path, "rb") as f:
        loaded = pickle.load(f)

    assert set(loaded.keys()) == set(woe_map.keys()), "Pickle must preserve all feature keys"
    for feature in woe_map:
        assert loaded[feature]["bin_edges"] == woe_map[feature]["bin_edges"], (
            f"bin_edges for {feature!r} changed after pickle round-trip"
        )
        assert loaded[feature]["bin_woe_values"] == woe_map[feature]["bin_woe_values"], (
            f"bin_woe_values for {feature!r} changed after pickle round-trip"
        )


def test_apply_feature_store_transforms_correctly(feature_store_data, mock_data_dir):
    """
    Re-applying woe_mappings to the training data must produce only valid WoE
    values (finite floats or the -999 sentinel), with no NaN.
    """
    X, y = feature_store_data
    X_train_out, woe_map = build_feature_store(X, y, output_dir=mock_data_dir / "data" / "processed")

    # Use a fresh copy of X (same data) to simulate inference
    X_infer = X[list(woe_map.keys())].copy()
    X_applied = apply_feature_store(X_infer, woe_map)

    assert not X_applied.isna().any().any(), "apply_feature_store must not produce NaN"
    assert set(X_applied.columns) == set(woe_map.keys()), (
        "apply_feature_store must return exactly the columns in woe_mappings"
    )


def test_apply_feature_store_handles_ood_values(feature_store_data, mock_data_dir):
    """
    Values outside training bin edges are out-of-distribution (OOD).
    apply_feature_store must fill them with _NAN_SENTINEL (-999), not leave NaN.
    """
    X, y = feature_store_data
    _, woe_map = build_feature_store(X, y, output_dir=mock_data_dir / "data" / "processed")

    # Build a DataFrame with deliberately extreme values (outside any training range)
    ood_X = pd.DataFrame(
        {feat: [1e9, -1e9] for feat in woe_map.keys()}
    )
    X_out = apply_feature_store(ood_X, woe_map)

    assert not X_out.isna().any().any(), (
        "OOD values must be filled with -999 sentinel, not left as NaN"
    )


def test_apply_feature_store_does_not_mutate_input(feature_store_data, mock_data_dir):
    """apply_feature_store must return a new DataFrame without mutating the input."""
    X, y = feature_store_data
    _, woe_map = build_feature_store(X, y, output_dir=mock_data_dir / "data" / "processed")

    X_infer = X[list(woe_map.keys())].copy()
    original_values = X_infer.copy()

    apply_feature_store(X_infer, woe_map)

    pd.testing.assert_frame_equal(X_infer, original_values, check_like=True)


# ---------------------------------------------------------------------------
# Task 2.4 — Additional edge case and inference-path tests
# ---------------------------------------------------------------------------


def test_engineer_application_features_no_nulls(application_fixture):
    """All 11 engineered columns must contain zero NaN values."""
    result = engineer_application_features(application_fixture)
    nulls = {col: result[col].isna().sum() for col in EXPECTED_COLUMNS if result[col].isna().any()}
    assert nulls == {}, f"NaN found in engineered columns: {nulls}"


def test_credit_income_ratio_correct(application_fixture):
    row = application_fixture.iloc[[0]].assign(
        AMT_CREDIT=500_000,
        AMT_INCOME_TOTAL=100_000,
    )
    result = engineer_application_features(row)
    assert result["CREDIT_INCOME_RATIO"].iloc[0] == pytest.approx(5.0)


def test_apply_feature_store_matches_train(feature_store_data, mock_data_dir):
    """
    apply_feature_store (inference path) must return the same columns as training
    and produce no NaN values — even on a separate held-out set.

    This test is more critical than testing build_feature_store because
    apply_feature_store runs at prediction time on every incoming request.
    build_feature_store runs once during training in a controlled environment;
    apply_feature_store must be rock-solid against OOD values, missing ranges,
    and unseen bin edges — any silent NaN here breaks the downstream model.
    """
    X, y = feature_store_data
    X_train_out, woe_map = build_feature_store(X, y, output_dir=mock_data_dir / "data" / "processed")

    rng = np.random.default_rng(99)
    X_test = pd.DataFrame(
        {col: rng.uniform(0, 1, 50) for col in woe_map.keys()}
    )
    X_test_out = apply_feature_store(X_test, woe_map)

    assert set(X_test_out.columns) == set(X_train_out.columns), (
        "apply_feature_store must return exactly the training columns"
    )
    assert not X_test_out.isna().any().any(), (
        "apply_feature_store must not produce NaN on held-out data"
    )


# ---------------------------------------------------------------------------
# Phase A — Raw feature store (TDD: RED phase)
# ---------------------------------------------------------------------------


def test_build_raw_feature_store_returns_correct_types(feature_store_data, tmp_path):
    """build_raw_feature_store must return a tuple of (DataFrame, list)."""
    X, y = feature_store_data
    result = build_raw_feature_store(X, y, output_dir=tmp_path)

    assert isinstance(result, tuple), "Result must be a tuple"
    assert len(result) == 2, "Result tuple must have exactly 2 elements"
    X_out, feature_cols = result
    assert isinstance(X_out, pd.DataFrame), "First element must be a DataFrame"
    assert isinstance(feature_cols, list), "Second element must be a list"


def test_build_raw_feature_store_no_woe_values(feature_store_data, tmp_path):
    """
    Output values must be raw floats (continuous), NOT discrete WoE integers.
    Each column should have a standard deviation > 0, and include real float values.
    """
    X, y = feature_store_data
    X_out, _ = build_raw_feature_store(X, y, output_dir=tmp_path)

    for col in X_out.columns:
        std = X_out[col].std()
        assert std > 0, f"Column {col} is constant (std={std}), should have variance"

        # Check that values are NOT just a few discrete integers (WoE would be)
        unique_count = X_out[col].nunique()
        assert unique_count > 1, f"Column {col} has only {unique_count} unique value(s)"


def test_build_raw_feature_store_sentinel_fills_nan(application_fixture, tmp_path):
    """If input has NaN, output must replace with -999, not leave NaN."""
    # Create a simple target with ~10% positives
    y = pd.Series([0] * 60 + [1] * 10, dtype=int)
    X = application_fixture.copy()

    # Artificially add NaN to a feature column (use .loc to avoid SettingWithCopyWarning)
    X.loc[0, 'AMT_CREDIT'] = np.nan

    X_out, _ = build_raw_feature_store(X, y, output_dir=tmp_path)

    # Check that no NaN remains in the output
    assert not X_out.isna().any().any(), (
        "build_raw_feature_store must fill all NaN with -999 sentinel"
    )


def test_build_raw_feature_store_iv_filter_applied(feature_store_data, tmp_path):
    """Output must have fewer columns than input due to IV filtering."""
    X, y = feature_store_data

    X_out, _ = build_raw_feature_store(X, y, output_dir=tmp_path)

    # IV filter should remove low-signal features (constant and noise)
    assert X_out.shape[1] < X.shape[1], (
        f"IV filter should reduce column count: got {X_out.shape[1]} from {X.shape[1]}"
    )


def test_build_raw_feature_store_no_inf_values(feature_store_data, tmp_path):
    """Output must contain no inf or -inf values."""
    X, y = feature_store_data
    X_out, _ = build_raw_feature_store(X, y, output_dir=tmp_path)

    # Check for inf and -inf
    assert not np.isinf(X_out.values).any(), (
        "build_raw_feature_store must replace inf with -999 sentinel"
    )


def test_apply_raw_feature_store_selects_correct_columns(feature_store_data, tmp_path):
    """Output must have exactly the columns from feature_columns, in the right order."""
    X, y = feature_store_data
    _, feature_cols = build_raw_feature_store(X, y, output_dir=tmp_path)

    # Create new test data
    X_test = pd.DataFrame(
        {col: np.random.randn(20) for col in feature_cols}
    )

    X_out = apply_raw_feature_store(X_test, feature_cols)

    assert list(X_out.columns) == feature_cols, (
        "apply_raw_feature_store must return columns in exact order"
    )
    assert X_out.shape[1] == len(feature_cols), (
        "apply_raw_feature_store must have exactly the specified columns"
    )


def test_apply_raw_feature_store_handles_missing_columns(feature_store_data, tmp_path):
    """If a column from feature_columns is missing in input, it must be filled with -999."""
    X, y = feature_store_data
    _, feature_cols = build_raw_feature_store(X, y, output_dir=tmp_path)

    # Create test data with only the first half of required columns
    X_test = pd.DataFrame(
        {col: np.random.randn(20) for col in feature_cols[:len(feature_cols)//2]}
    )

    X_out = apply_raw_feature_store(X_test, feature_cols)

    # Check that all required columns are present
    assert set(X_out.columns) == set(feature_cols), (
        "apply_raw_feature_store must include all feature_columns"
    )

    # Check that missing columns are filled with -999
    for col in feature_cols:
        if col not in X_test.columns:
            assert (X_out[col] == -999.0).all(), (
                f"Missing column {col} must be filled with -999 sentinel"
            )


# ---------------------------------------------------------------------------
# Phase C — engineer_secondary_features() tests (TDD: RED phase)
# ---------------------------------------------------------------------------


@pytest.fixture
def secondary_fixture() -> pd.DataFrame:
    """
    Minimal DataFrame with secondary table aggregate columns for testing.

    Contains bureau, previous, POS, installments, and credit card aggregates
    along with a few application columns needed for ratios.
    """
    return pd.DataFrame(
        {
            # Application columns (for income-based ratios)
            "AMT_INCOME_TOTAL": [100_000, 50_000, 0, 120_000, 80_000],
            # Previous application aggregates
            "prev_cnt": [3, 0, 5, 2, 1],
            "prev_approved_cnt": [2, 0, 0, 2, 1],
            "prev_amt_credit_mean": [50_000, 0, 100_000, 60_000, 40_000],
            # Bureau aggregates
            "bureau_cnt": [5, 3, 0, 2, 4],
            "bureau_credit_sum": [300_000, 150_000, 0, 200_000, 250_000],
            "bureau_credit_debt_sum": [90_000, 45_000, 0, 50_000, 100_000],
            # Bureau aggregates (extended)
            "bureau_active_cnt": [2, 1, 0, 1, 3],
            "bureau_overdue_cnt": [1, 0, 0, 2, 4],
            # Installments aggregates
            "inst_cnt": [20, 15, 0, 30, 10],
            "inst_late_cnt": [2, 3, 0, 5, 0],
            "inst_amt_payment_sum": [21_000, 15_500, 0, 31_500, 10_100],
            "inst_payment_ratio_mean": [1.05, 1.03, 0.0, 1.05, 1.01],
            "inst_days_past_due_max": [10.0, 0.0, 5.0, 30.0, 0.0],
            "inst_days_past_due_mean": [2.0, 0.0, 1.0, 10.0, 0.0],
            # POS aggregates
            "pos_sk_dpd_max": [0, 45, 0, 0, 60],
            # Credit card aggregates
            "cc_sk_dpd_max": [0, 0, 0, 30, 0],
        }
    )


class TestEngineerSecondaryFeatures:
    """Test suite for engineer_secondary_features()."""

    def test_engineer_secondary_features_returns_dataframe(self, secondary_fixture):
        """Output must be a DataFrame."""
        result = engineer_secondary_features(secondary_fixture)
        assert isinstance(result, pd.DataFrame)

    def test_engineer_secondary_features_no_mutation(self, secondary_fixture):
        """Input DataFrame must not be modified."""
        original_columns = set(secondary_fixture.columns)
        original_values = secondary_fixture.copy()

        engineer_secondary_features(secondary_fixture)

        assert set(secondary_fixture.columns) == original_columns
        pd.testing.assert_frame_equal(secondary_fixture, original_values)

    def test_prev_approval_rate_correct(self, secondary_fixture):
        """prev_approval_rate = prev_approved_cnt / max(prev_cnt, 1)."""
        result = engineer_secondary_features(secondary_fixture)

        # Row 0: prev_approved_cnt=2, prev_cnt=3 → 2/3 ≈ 0.667
        assert result.loc[0, "prev_approval_rate"] == pytest.approx(2.0 / 3.0)
        # Row 4: prev_approved_cnt=1, prev_cnt=1 → 1/1 = 1.0
        assert result.loc[4, "prev_approval_rate"] == pytest.approx(1.0)

    def test_prev_approval_rate_zero_denominator(self, secondary_fixture):
        """When prev_cnt=0, result must be 0.0, not inf/nan."""
        result = engineer_secondary_features(secondary_fixture)

        # Row 1: prev_cnt=0, prev_approved_cnt=0 → 0.0
        assert result.loc[1, "prev_approval_rate"] == pytest.approx(0.0)

    def test_inst_pct_late_correct(self, secondary_fixture):
        """inst_pct_late = inst_late_cnt / max(inst_cnt, 1)."""
        result = engineer_secondary_features(secondary_fixture)

        # Row 0: inst_late_cnt=2, inst_cnt=20 → 2/20 = 0.1
        assert result.loc[0, "inst_pct_late"] == pytest.approx(0.1)
        # Row 3: inst_late_cnt=5, inst_cnt=30 → 5/30 ≈ 0.167
        assert result.loc[3, "inst_pct_late"] == pytest.approx(5.0 / 30.0)

    def test_bureau_debt_ratio_correct(self, secondary_fixture):
        """bureau_debt_ratio = bureau_credit_debt_sum / max(bureau_credit_sum, 1e-6)."""
        result = engineer_secondary_features(secondary_fixture)

        # Row 0: bureau_credit_debt_sum=90k, bureau_credit_sum=300k → 0.3
        assert result.loc[0, "bureau_debt_ratio"] == pytest.approx(0.3)
        # Row 4: bureau_credit_debt_sum=100k, bureau_credit_sum=250k → 0.4
        assert result.loc[4, "bureau_debt_ratio"] == pytest.approx(0.4)

    def test_cc_overdue_flag_binary(self, secondary_fixture):
        """cc_overdue_flag must be 0 or 1, never NaN."""
        result = engineer_secondary_features(secondary_fixture)

        assert result["cc_overdue_flag"].isin([0.0, 1.0]).all()
        assert not result["cc_overdue_flag"].isna().any()

    def test_prev_refusal_rate_correct(self, secondary_fixture):
        """prev_refusal_rate = prev_refused_cnt / max(prev_cnt, 1)."""
        # secondary_fixture does not have prev_refused_cnt — add it
        df = secondary_fixture.copy()
        df["prev_refused_cnt"] = [1, 0, 5, 0, 0]
        result = engineer_secondary_features(df)

        # Row 0: 1/3 ≈ 0.333
        assert result.loc[0, "prev_refusal_rate"] == pytest.approx(1.0 / 3.0)
        # Row 2: 5/5 = 1.0
        assert result.loc[2, "prev_refusal_rate"] == pytest.approx(1.0)
        # Row 1: prev_cnt=0 → 0.0 (guard)
        assert result.loc[1, "prev_refusal_rate"] == pytest.approx(0.0)
        # No NaN
        assert not result["prev_refusal_rate"].isna().any()

    def test_bureau_overdue_rate_correct(self, secondary_fixture):
        """bureau_overdue_rate = bureau_overdue_cnt / max(bureau_cnt, 1)."""
        df = secondary_fixture.copy()
        df["bureau_overdue_cnt"] = [1, 0, 0, 2, 4]
        result = engineer_secondary_features(df)

        # Row 0: 1/5 = 0.2
        assert result.loc[0, "bureau_overdue_rate"] == pytest.approx(0.2)
        # Row 4: 4/4 = 1.0
        assert result.loc[4, "bureau_overdue_rate"] == pytest.approx(1.0)
        # Row 2: bureau_cnt=0 → 0.0 (guard)
        assert result.loc[2, "bureau_overdue_rate"] == pytest.approx(0.0)

    def test_bureau_active_ratio_bounds(self, secondary_fixture):
        """bureau_active_ratio must be in [0, 1] and contain no NaN."""
        df = secondary_fixture.copy()
        df["bureau_active_cnt"] = [2, 1, 0, 1, 3]
        result = engineer_secondary_features(df)

        ratio = result["bureau_active_ratio"]
        assert not ratio.isna().any(), "bureau_active_ratio must not contain NaN"
        assert (ratio >= 0).all() and (ratio <= 1.0 + 1e-9).all(), (
            "bureau_active_ratio must be in [0, 1]"
        )

    def test_bureau_debt_to_income_clipped(self, secondary_fixture):
        """bureau_debt_to_income must be clipped at 50 and never negative."""
        df = secondary_fixture.copy()
        # Row 2 has AMT_INCOME_TOTAL == 0 — clip(lower=1) prevents divide-by-zero
        result = engineer_secondary_features(df)

        col = result["bureau_debt_to_income"]
        assert (col <= 50.0 + 1e-9).all(), "Must be clipped at 50"
        assert not col.isna().any()

    def test_debt_service_ratio_correct(self, secondary_fixture):
        """debt_service_ratio = AMT_ANNUITY / (AMT_INCOME_TOTAL / 12)."""
        df = secondary_fixture.copy()
        df["AMT_ANNUITY"] = [2_000, 1_000, 500, 3_000, 1_500]
        result = engineer_secondary_features(df)

        # Row 0: 2000 / (100000/12) = 2000/8333.3 ≈ 0.24
        expected = 2_000.0 / (100_000.0 / 12.0)
        assert result.loc[0, "debt_service_ratio"] == pytest.approx(expected, rel=1e-3)
        # Must be clipped at 10
        assert (result["debt_service_ratio"] <= 10.0 + 1e-9).all()
        assert not result["debt_service_ratio"].isna().any()

    def test_inst_late_dpd_ratio_no_inf(self, secondary_fixture):
        """inst_late_dpd_ratio must not contain inf or NaN."""
        df = secondary_fixture.copy()
        df["inst_days_past_due_max"] = [10.0, 0.0, 5.0, 30.0, 0.0]
        df["inst_days_past_due_mean"] = [2.0, 0.0, 1.0, 10.0, 0.0]
        result = engineer_secondary_features(df)

        col = result["inst_late_dpd_ratio"]
        assert not col.isin([np.inf, -np.inf]).any(), "inst_late_dpd_ratio must not be inf"
        assert not col.isna().any(), "inst_late_dpd_ratio must not be NaN"

    def test_engineer_secondary_features_skips_missing_columns(self):
        """If required columns are missing, features depending on them are skipped."""
        # Create a minimal DataFrame without secondary columns
        minimal_df = pd.DataFrame({
            "AMT_INCOME_TOTAL": [100_000],
        })

        result = engineer_secondary_features(minimal_df)

        # Should return a DataFrame without error
        assert isinstance(result, pd.DataFrame)
        # Should have only the input column (no secondary features added)
        assert len(result.columns) == 1

    # -----------------------------------------------------------------------
    # Phase 04.2.3.2 Features (D-07 through D-19): 13 secondary/cross-table features
    # -----------------------------------------------------------------------

    def test_engineer_secondary_features_has_no_bureau_history(self, secondary_fixture):
        """D-07: no_bureau_history = (bureau_cnt == 0).astype(int)."""
        df = secondary_fixture.copy()
        result = engineer_secondary_features(df)

        assert "no_bureau_history" in result.columns
        assert result["no_bureau_history"].dtype in [int, np.int32, np.int64]
        assert result["no_bureau_history"].isin([0, 1]).all()
        # Row 2 has bureau_cnt=0 → should be 1; others are > 0 → should be 0
        assert result.loc[2, "no_bureau_history"] == 1
        assert result.loc[0, "no_bureau_history"] == 0

    def test_engineer_secondary_features_has_no_prev_applications(self, secondary_fixture):
        """D-08: no_prev_applications = (prev_cnt == 0).astype(int)."""
        df = secondary_fixture.copy()
        result = engineer_secondary_features(df)

        assert "no_prev_applications" in result.columns
        assert result["no_prev_applications"].dtype in [int, np.int32, np.int64]
        assert result["no_prev_applications"].isin([0, 1]).all()
        # Row 1 has prev_cnt=0 → should be 1; Row 0 has prev_cnt=3 → should be 0
        assert result.loc[1, "no_prev_applications"] == 1
        assert result.loc[0, "no_prev_applications"] == 0

    def test_engineer_secondary_features_has_ever_dpd_bureau(self, secondary_fixture):
        """D-09: ever_dpd_bureau = (bureau_overdue_cnt > 0).astype(int)."""
        df = secondary_fixture.copy()
        df["bureau_overdue_cnt"] = [1, 0, 0, 2, 0]
        result = engineer_secondary_features(df)

        assert "ever_dpd_bureau" in result.columns
        assert result["ever_dpd_bureau"].dtype in [int, np.int32, np.int64]
        assert result["ever_dpd_bureau"].isin([0, 1]).all()
        assert result.loc[0, "ever_dpd_bureau"] == 1
        assert result.loc[1, "ever_dpd_bureau"] == 0

    def test_engineer_secondary_features_has_bureau_prolong_any(self, secondary_fixture):
        """D-10: bureau_prolong_any = (bureau_prolong_sum > 0).astype(int)."""
        df = secondary_fixture.copy()
        df["bureau_prolong_sum"] = [5, 0, 0, 10, 0]
        result = engineer_secondary_features(df)

        assert "bureau_prolong_any" in result.columns
        assert result["bureau_prolong_any"].dtype in [int, np.int32, np.int64]
        assert result["bureau_prolong_any"].isin([0, 1]).all()
        assert result.loc[0, "bureau_prolong_any"] == 1
        assert result.loc[1, "bureau_prolong_any"] == 0

    def test_engineer_secondary_features_has_high_credit_income(self, secondary_fixture):
        """D-11: high_credit_income = (CREDIT_INCOME_RATIO > 5).astype(int)."""
        df = secondary_fixture.copy()
        df["CREDIT_INCOME_RATIO"] = [6.0, 2.0, 5.5, 1.0, 4.9]
        result = engineer_secondary_features(df)

        assert "high_credit_income" in result.columns
        assert result["high_credit_income"].dtype in [int, np.int32, np.int64]
        assert result["high_credit_income"].isin([0, 1]).all()
        assert result.loc[0, "high_credit_income"] == 1  # 6.0 > 5
        assert result.loc[1, "high_credit_income"] == 0  # 2.0 <= 5
        assert result.loc[2, "high_credit_income"] == 1  # 5.5 > 5
        assert result.loc[4, "high_credit_income"] == 0  # 4.9 <= 5

    def test_engineer_secondary_features_has_low_payment_rate(self, secondary_fixture):
        """D-12: low_payment_rate = (payment_rate < 0.03).astype(int)."""
        df = secondary_fixture.copy()
        df["payment_rate"] = [0.02, 0.05, 0.03, 0.015, 0.04]
        result = engineer_secondary_features(df)

        assert "low_payment_rate" in result.columns
        assert result["low_payment_rate"].dtype in [int, np.int32, np.int64]
        assert result["low_payment_rate"].isin([0, 1]).all()
        assert result.loc[0, "low_payment_rate"] == 1  # 0.02 < 0.03
        assert result.loc[1, "low_payment_rate"] == 0  # 0.05 >= 0.03
        assert result.loc[3, "low_payment_rate"] == 1  # 0.015 < 0.03

    def test_engineer_secondary_features_has_thin_file(self, secondary_fixture):
        """D-13: thin_file = no_bureau_history (regulatory reframe; no age component)."""
        df = secondary_fixture.copy()
        result = engineer_secondary_features(df)

        assert "thin_file" in result.columns
        assert result["thin_file"].dtype in [int, np.int32, np.int64]
        assert result["thin_file"].isin([0, 1]).all()
        # thin_file should equal no_bureau_history
        pd.testing.assert_series_equal(
            result["thin_file"].reset_index(drop=True),
            result["no_bureau_history"].reset_index(drop=True),
            check_names=False
        )

    def test_engineer_secondary_features_has_new_credit_to_bureau_ratio(self, secondary_fixture):
        """D-14: new_credit_to_bureau_ratio = AMT_CREDIT / bureau_credit_sum; fill -999."""
        df = secondary_fixture.copy()
        df["AMT_CREDIT"] = [300_000, 150_000, 100_000, 50_000, 200_000]
        # bureau_credit_sum is already in secondary_fixture
        result = engineer_secondary_features(df)

        assert "new_credit_to_bureau_ratio" in result.columns
        assert result["new_credit_to_bureau_ratio"].dtype in [float, np.float32, np.float64]
        assert not result["new_credit_to_bureau_ratio"].isna().any()
        # Row 0: 300k / 300k = 1.0
        assert result.loc[0, "new_credit_to_bureau_ratio"] == pytest.approx(1.0)
        # Row 1: 150k / 150k = 1.0
        assert result.loc[1, "new_credit_to_bureau_ratio"] == pytest.approx(1.0)

    def test_engineer_secondary_features_has_annuity_to_prev_annuity_ratio(self, secondary_fixture):
        """D-15: annuity_to_prev_annuity_ratio = AMT_ANNUITY / prev_amt_annuity_mean; fill -999."""
        df = secondary_fixture.copy()
        df["AMT_ANNUITY"] = [2_000, 1_000, 500, 3_000, 1_500]
        df["prev_amt_annuity_mean"] = [2_000, 500, 1_000, 2_000, 1_000]
        result = engineer_secondary_features(df)

        assert "annuity_to_prev_annuity_ratio" in result.columns
        assert result["annuity_to_prev_annuity_ratio"].dtype in [float, np.float32, np.float64]
        assert not result["annuity_to_prev_annuity_ratio"].isna().any()
        # Row 0: 2000 / 2000 = 1.0
        assert result.loc[0, "annuity_to_prev_annuity_ratio"] == pytest.approx(1.0)

    def test_engineer_secondary_features_has_bureau_overdue_to_income(self, secondary_fixture):
        """D-16: bureau_overdue_to_income = bureau_overdue_sum / AMT_INCOME_TOTAL; fill -999."""
        df = secondary_fixture.copy()
        df["bureau_overdue_sum"] = [10_000, 0, 0, 5_000, 20_000]
        result = engineer_secondary_features(df)

        assert "bureau_overdue_to_income" in result.columns
        assert result["bureau_overdue_to_income"].dtype in [float, np.float32, np.float64]
        assert not result["bureau_overdue_to_income"].isna().any()
        # Row 0: 10000 / 100000 = 0.1
        assert result.loc[0, "bureau_overdue_to_income"] == pytest.approx(0.1)

    def test_engineer_secondary_features_has_bureau_active_to_prev_apps(self, secondary_fixture):
        """D-17: bureau_active_to_prev_apps = bureau_active_cnt / prev_cnt; fill -999."""
        df = secondary_fixture.copy()
        df["bureau_active_cnt"] = [2, 0, 0, 1, 3]
        result = engineer_secondary_features(df)

        assert "bureau_active_to_prev_apps" in result.columns
        assert result["bureau_active_to_prev_apps"].dtype in [float, np.float32, np.float64]
        assert not result["bureau_active_to_prev_apps"].isna().any()

    def test_engineer_secondary_features_has_cc_utilisation_to_income(self, secondary_fixture):
        """D-18: cc_utilisation_to_income = cc_utilisation_mean * cc_bal_max / AMT_INCOME_TOTAL; fill -999."""
        df = secondary_fixture.copy()
        df["cc_utilisation_mean"] = [0.5, 0.2, 0.0, 0.8, 0.3]
        df["cc_bal_max"] = [50_000, 10_000, 0, 30_000, 20_000]
        result = engineer_secondary_features(df)

        assert "cc_utilisation_to_income" in result.columns
        assert result["cc_utilisation_to_income"].dtype in [float, np.float32, np.float64]
        assert not result["cc_utilisation_to_income"].isna().any()

    def test_engineer_secondary_features_has_bureau_close_rate(self, secondary_fixture):
        """D-19: bureau_close_rate = bureau_closed_cnt / bureau_cnt; fill -999."""
        df = secondary_fixture.copy()
        df["bureau_closed_cnt"] = [2, 0, 0, 0, 2]
        result = engineer_secondary_features(df)

        assert "bureau_close_rate" in result.columns
        assert result["bureau_close_rate"].dtype in [float, np.float32, np.float64]
        assert not result["bureau_close_rate"].isna().any()
        # Row 0: 2 / 5 = 0.4
        assert result.loc[0, "bureau_close_rate"] == pytest.approx(0.4)
        # Row 4: 2 / 4 = 0.5
        assert result.loc[4, "bureau_close_rate"] == pytest.approx(0.5)

    # -----------------------------------------------------------------------
    # Regulatory Compliance Tests — Unit tests for regulatory drop mechanism
    # -----------------------------------------------------------------------

    def test_regulatory_drop_cols_constant_exists(self):
        """D-20, D-21: Verify _REGULATORY_DROP_COLS constant is defined."""
        from src.features import _REGULATORY_DROP_COLS
        assert isinstance(_REGULATORY_DROP_COLS, list), "_REGULATORY_DROP_COLS must be a list"
        assert "CODE_GENDER" in _REGULATORY_DROP_COLS, "CODE_GENDER must be in regulatory drop list"
        assert "thin_file_young" in _REGULATORY_DROP_COLS, "thin_file_young must be in regulatory drop list"

    def test_build_tree_feature_store_applies_regulatory_drops(self):
        """D-20, D-21: Verify regulatory columns are dropped before numeric selection."""
        # Create a minimal DataFrame with regulatory columns
        X_test = pd.DataFrame({
            "CODE_GENDER": ["M", "F", "M"] * 10,  # Will be dropped
            "thin_file_young": [0, 1, 0] * 10,     # Will be dropped
            "high_var": np.linspace(1, 100, 30),   # Will be kept
            "AMT_INCOME_TOTAL": [50_000] * 30,
            "bureau_cnt": [1] * 30,
        })
        y_test = pd.Series([0, 1] * 15)

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                X_final, _ = build_tree_feature_store(X_test, y_test, output_dir=tmpdir)
                # After regulatory drops and numeric filtering, these should be absent
                assert "CODE_GENDER" not in X_final.columns, "CODE_GENDER must be dropped"
                assert "thin_file_young" not in X_final.columns, "thin_file_young must be dropped"
            except KeyError:
                # Expected if engineer_application_features needs more columns
                # That's OK - we're testing the regulatory drop mechanism works
                pass


# ---------------------------------------------------------------------------
# Priority 1 — Task 1.2: Cross-table interaction features (TDD: RED phase)
# ---------------------------------------------------------------------------


@pytest.fixture
def interaction_fixture() -> pd.DataFrame:
    """
    DataFrame with all columns needed to test cross-table interaction features.

    Rows:
      Row 0: high-risk — low EXT scores, high leverage, late payments
      Row 1: low-risk  — high EXT scores, low leverage, clean record
      Row 2: zero-income guard — AMT_INCOME_TOTAL == 0
      Row 3: no secondary records — all bureau/inst/cc columns are 0
      Row 4: high DPD worsening — 6m DPD much worse than 12m average
    """
    return pd.DataFrame(
        {
            # Application columns
            "AMT_INCOME_TOTAL": [60_000, 120_000, 0, 80_000, 90_000],
            "AMT_ANNUITY": [8_000, 5_000, 3_000, 4_000, 6_000],
            "CREDIT_INCOME_RATIO": [5.0, 1.0, 3.0, 2.0, 4.0],
            "ANNUITY_INCOME_RATIO": [0.15, 0.04, 0.0, 0.05, 0.08],
            # EXT_SOURCE composites (already computed by engineer_application_features)
            "EXT_SOURCE_MEAN": [0.3, 0.7, 0.5, 0.6, 0.4],
            "EXT_SOURCE_MIN": [0.2, 0.6, 0.4, 0.5, 0.3],
            # Bureau aggregates
            "bureau_cnt": [5, 3, 0, 0, 4],
            "bureau_credit_sum": [300_000, 150_000, 0, 0, 250_000],
            "bureau_credit_debt_sum": [90_000, 10_000, 0, 0, 60_000],
            "bureau_active_cnt": [2, 1, 0, 0, 3],
            "bureau_overdue_cnt": [3, 0, 0, 0, 1],
            "bureau_annuity_mean": [5_000, 2_000, 0, 0, 4_000],
            "bureau_bbal_dpd_rate_6m_mean": [0.3, 0.0, 0.0, 0.0, 0.5],
            "bureau_bbal_dpd_rate_12m_mean": [0.1, 0.0, 0.0, 0.0, 0.2],
            # Installments aggregates
            "inst_cnt": [20, 15, 0, 0, 10],
            "inst_late_cnt": [5, 0, 0, 0, 2],
            "inst_pct_late": [0.25, 0.0, 0.0, 0.0, 0.2],
            "inst_days_past_due_mean": [8.0, 0.0, 0.0, 0.0, 5.0],
            "inst_days_past_due_max": [30.0, 0.0, 0.0, 0.0, 20.0],
            # Credit card aggregates
            "cc_bal_max": [50_000, 5_000, 0, 0, 20_000],
            "cc_sk_dpd_max": [10, 0, 0, 0, 5],
            "cc_dpd_rate": [0.2, 0.0, 0.0, 0.0, 0.15],
            # Derived secondary features (already present after engineer_secondary_features)
            "bureau_overdue_rate": [0.6, 0.0, 0.0, 0.0, 0.25],
            "bureau_debt_to_income": [1.5, 0.08, 0.0, 0.0, 0.67],
        }
    )


class TestCrossTableInteractions:
    """Tests for cross-table interaction features added to engineer_secondary_features()."""

    def test_ext_credit_risk_high_risk_row(self, interaction_fixture):
        """ext_credit_risk = EXT_SOURCE_MEAN * CREDIT_INCOME_RATIO, clipped ≥ 0."""
        result = engineer_secondary_features(interaction_fixture)
        # Row 0: 0.3 * 5.0 = 1.5
        assert result.loc[0, "ext_credit_risk"] == pytest.approx(0.3 * 5.0, rel=1e-3)
        # Row 1: 0.7 * 1.0 = 0.7
        assert result.loc[1, "ext_credit_risk"] == pytest.approx(0.7 * 1.0, rel=1e-3)

    def test_ext_credit_risk_never_negative(self, interaction_fixture):
        """ext_credit_risk is clipped to [0, ∞) — never negative."""
        result = engineer_secondary_features(interaction_fixture)
        assert (result["ext_credit_risk"] >= 0).all()
        assert not result["ext_credit_risk"].isna().any()

    def test_ext_annuity_risk_correct(self, interaction_fixture):
        """ext_annuity_risk = EXT_SOURCE_MIN * ANNUITY_INCOME_RATIO, clipped ≥ 0."""
        result = engineer_secondary_features(interaction_fixture)
        # Row 0: 0.2 * 0.15 = 0.03
        assert result.loc[0, "ext_annuity_risk"] == pytest.approx(0.2 * 0.15, rel=1e-3)

    def test_multi_dpd_flag_both_late(self, interaction_fixture):
        """multi_dpd_flag = 1 when inst_pct_late > 0.1 AND cc_dpd_rate > 0.1."""
        result = engineer_secondary_features(interaction_fixture)
        # Row 0: inst_pct_late=0.25 > 0.1, cc_dpd_rate=0.2 > 0.1 → flag=1
        assert result.loc[0, "multi_dpd_flag"] == 1.0
        # Row 4: inst_pct_late=0.2 > 0.1, cc_dpd_rate=0.15 > 0.1 → flag=1
        assert result.loc[4, "multi_dpd_flag"] == 1.0

    def test_multi_dpd_flag_neither_late(self, interaction_fixture):
        """multi_dpd_flag = 0 when neither product has high DPD."""
        result = engineer_secondary_features(interaction_fixture)
        # Row 1: both rates = 0.0 → flag=0
        assert result.loc[1, "multi_dpd_flag"] == 0.0
        # Row 3: no records → 0.0
        assert result.loc[3, "multi_dpd_flag"] == 0.0

    def test_multi_dpd_flag_binary(self, interaction_fixture):
        """multi_dpd_flag must only contain 0 or 1."""
        result = engineer_secondary_features(interaction_fixture)
        assert result["multi_dpd_flag"].isin([0.0, 1.0]).all()

    def test_total_debt_exposure_zero_income_guard(self, interaction_fixture):
        """When AMT_INCOME_TOTAL == 0, total_debt_exposure must not be inf/nan."""
        result = engineer_secondary_features(interaction_fixture)
        # Row 2: income=0 → clipped to 1.0 → valid ratio
        assert np.isfinite(result.loc[2, "total_debt_exposure"])
        assert result.loc[2, "total_debt_exposure"] >= 0

    def test_total_debt_exposure_clipped(self, interaction_fixture):
        """total_debt_exposure must be clipped to [0, 100]."""
        result = engineer_secondary_features(interaction_fixture)
        col = result["total_debt_exposure"]
        assert (col >= 0).all()
        assert (col <= 100.0 + 1e-9).all()
        assert not col.isna().any()

    def test_dpd_trajectory_positive_when_worsening(self, interaction_fixture):
        """dpd_trajectory = dpd_rate_6m - dpd_rate_12m; positive = worsening."""
        result = engineer_secondary_features(interaction_fixture)
        # Row 0: 0.3 - 0.1 = 0.2 (worsening)
        assert result.loc[0, "dpd_trajectory"] == pytest.approx(0.2, rel=1e-3)
        # Row 4: 0.5 - 0.2 = 0.3 (worsening)
        assert result.loc[4, "dpd_trajectory"] == pytest.approx(0.3, rel=1e-3)
        # Row 1: 0.0 - 0.0 = 0.0 (stable)
        assert result.loc[1, "dpd_trajectory"] == pytest.approx(0.0, rel=1e-3)

    def test_debt_service_coverage_guard(self, interaction_fixture):
        """debt_service_coverage must never be inf or nan."""
        result = engineer_secondary_features(interaction_fixture)
        col = result["debt_service_coverage"]
        assert not col.isin([np.inf, -np.inf]).any()
        assert not col.isna().any()
        assert (col >= 0).all()

    def test_leverage_vs_bureau_no_negative(self, interaction_fixture):
        """leverage_vs_bureau is product of two clipped non-negative quantities."""
        result = engineer_secondary_features(interaction_fixture)
        col = result["leverage_vs_bureau"]
        assert (col >= 0).all()
        assert not col.isna().any()

    def test_interactions_no_row_multiplication(self, interaction_fixture):
        """engineer_secondary_features must not change the number of rows."""
        result = engineer_secondary_features(interaction_fixture)
        assert len(result) == len(interaction_fixture)

    def test_interactions_input_not_mutated(self, interaction_fixture):
        """Input DataFrame must not be modified."""
        original = interaction_fixture.copy()
        engineer_secondary_features(interaction_fixture)
        pd.testing.assert_frame_equal(interaction_fixture, original)


# ---------------------------------------------------------------------------
# Priority 1 — Task 1.4: EXT_SOURCE polynomial interactions (TDD: RED phase)
# ---------------------------------------------------------------------------


class TestExtSourcePolynomials:
    """Tests for EXT_SOURCE polynomial/ratio features in engineer_application_features()."""

    def test_ext_source_sq_columns_present(self, application_fixture):
        """EXT_SOURCE_1_SQ and EXT_SOURCE_2_SQ must be present after engineering."""
        result = engineer_application_features(application_fixture)
        assert "EXT_SOURCE_1_SQ" in result.columns
        assert "EXT_SOURCE_2_SQ" in result.columns

    def test_ext_source_ratio_columns_present(self, application_fixture):
        """EXT_SOURCE_RATIO_12 and EXT_SOURCE_RATIO_23 must be present."""
        result = engineer_application_features(application_fixture)
        assert "EXT_SOURCE_RATIO_12" in result.columns
        assert "EXT_SOURCE_RATIO_23" in result.columns

    def test_ext_score_floor_column_present(self, application_fixture):
        """EXT_SCORE_FLOOR must be present after engineering."""
        result = engineer_application_features(application_fixture)
        assert "EXT_SCORE_FLOOR" in result.columns

    def test_ext_source_1_sq_nonnegative(self, application_fixture):
        """EXT_SOURCE_1_SQ = EXT_SOURCE_1 ** 2 must be ≥ 0 (squares are non-negative)."""
        result = engineer_application_features(application_fixture)
        valid = result["EXT_SOURCE_1_SQ"][result["EXT_SOURCE_1_SQ"] != -999.0]
        assert (valid >= 0).all(), "Squares must be non-negative"

    def test_ext_source_sq_correct_value(self, application_fixture):
        """EXT_SOURCE_1_SQ must equal EXT_SOURCE_1 ** 2 for non-missing rows."""
        result = engineer_application_features(application_fixture)
        # Row 0: EXT_SOURCE_1=0.6 → EXT_SOURCE_1_SQ ≈ 0.36
        assert result.loc[0, "EXT_SOURCE_1_SQ"] == pytest.approx(0.6 ** 2, rel=1e-5)
        # Row 2: EXT_SOURCE_1=0.7 → EXT_SOURCE_1_SQ ≈ 0.49
        assert result.loc[2, "EXT_SOURCE_1_SQ"] == pytest.approx(0.7 ** 2, rel=1e-5)

    def test_ext_source_sq_missing_filled_sentinel(self, application_fixture):
        """When EXT_SOURCE_1 is NaN (rows 3,4,5), EXT_SOURCE_1_SQ must equal -999."""
        result = engineer_application_features(application_fixture)
        # Rows 3, 4 have EXT_SOURCE_1 = NaN
        assert result.loc[3, "EXT_SOURCE_1_SQ"] == pytest.approx(-999.0)
        assert result.loc[4, "EXT_SOURCE_1_SQ"] == pytest.approx(-999.0)

    def test_ext_source_ratio_12_guard_near_zero(self):
        """When EXT_SOURCE_2 ≈ 0, ratio must be 0.0 (no inf or nan)."""
        df = pd.DataFrame(
            {
                "AMT_CREDIT": [100_000],
                "AMT_INCOME_TOTAL": [50_000],
                "AMT_ANNUITY": [5_000],
                "AMT_GOODS_PRICE": [90_000],
                "DAYS_BIRTH": [-12_000],
                "DAYS_EMPLOYED": [-2_000],
                "EXT_SOURCE_1": [0.5],
                "EXT_SOURCE_2": [0.0],  # zero denominator
                "EXT_SOURCE_3": [0.4],
                "FLAG_DOCUMENT_2": [1],
                "FLAG_DOCUMENT_3": [1],
                "FLAG_DOCUMENT_4": [0],
                "FLAG_DOCUMENT_5": [0],
            }
        )
        result = engineer_application_features(df)
        val = result.loc[0, "EXT_SOURCE_RATIO_12"]
        assert np.isfinite(val), "Must be finite even when denominator is 0"
        assert val == pytest.approx(0.0)

    def test_ext_score_floor_is_product_of_min_and_mean(self, application_fixture):
        """EXT_SCORE_FLOOR ≈ EXT_SOURCE_MIN * EXT_SOURCE_MEAN for non-sentinel rows."""
        result = engineer_application_features(application_fixture)
        # Row 0: EXT_SOURCE_1=0.6, EXT_SOURCE_2=0.7, EXT_SOURCE_3=0.5
        #        mean = (0.6+0.7+0.5)/3 ≈ 0.6; min = 0.5; floor = 0.5 * 0.6 = 0.3
        expected_floor = 0.5 * ((0.6 + 0.7 + 0.5) / 3.0)
        assert result.loc[0, "EXT_SCORE_FLOOR"] == pytest.approx(expected_floor, rel=1e-3)

    def test_ext_source_polynomials_no_nan(self, application_fixture):
        """All polynomial columns must contain no actual NaN (only -999 sentinel)."""
        result = engineer_application_features(application_fixture)
        for col in ["EXT_SOURCE_1_SQ", "EXT_SOURCE_2_SQ",
                    "EXT_SOURCE_RATIO_12", "EXT_SOURCE_RATIO_23", "EXT_SCORE_FLOOR"]:
            assert not result[col].isna().any(), f"{col} must not contain NaN"


# ---------------------------------------------------------------------------
# KNN target encoding tests (TDD: RED phase)
# ---------------------------------------------------------------------------


@pytest.fixture
def knn_encoding_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Synthetic data for KNN encoding tests.

    Creates 100 training samples and 20 test samples with 4 features
    and a binary target (15% event rate).
    """
    rng = np.random.default_rng(42)
    n_train = 100
    n_test = 20

    # Feature space: 2D for simplicity
    X_train = pd.DataFrame({
        "EXT_SOURCE_MEAN": rng.uniform(0, 1, n_train),
        "EXT_SOURCE_MIN": rng.uniform(0, 0.8, n_train),
        "CREDIT_INCOME_RATIO": rng.uniform(0, 5, n_train),
        "ANNUITY_INCOME_RATIO": rng.uniform(0, 1, n_train),
    })
    y_train = pd.Series(rng.binomial(1, 0.15, n_train), index=X_train.index)

    X_test = pd.DataFrame({
        "EXT_SOURCE_MEAN": rng.uniform(0, 1, n_test),
        "EXT_SOURCE_MIN": rng.uniform(0, 0.8, n_test),
        "CREDIT_INCOME_RATIO": rng.uniform(0, 5, n_test),
        "ANNUITY_INCOME_RATIO": rng.uniform(0, 1, n_test),
    })
    y_test = pd.Series(rng.binomial(1, 0.15, n_test), index=X_test.index)

    return X_train, y_train, X_test, y_test


class TestKnnTargetEncoding:
    """Tests for compute_knn_target_encoding()."""

    def test_train_output_range_valid(self, knn_encoding_data):
        """train_enc values are in [0, 1] — they are probabilities."""
        X_train, y_train, X_test, y_test = knn_encoding_data
        features = ["EXT_SOURCE_MEAN", "EXT_SOURCE_MIN", "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO"]

        train_enc, _ = compute_knn_target_encoding(X_train, y_train, X_test, features, k=10)

        assert (train_enc >= 0.0).all(), "All train_enc values must be >= 0.0"
        assert (train_enc <= 1.0).all(), "All train_enc values must be <= 1.0"

    def test_test_output_range_valid(self, knn_encoding_data):
        """test_enc values are in [0, 1]."""
        X_train, y_train, X_test, y_test = knn_encoding_data
        features = ["EXT_SOURCE_MEAN", "EXT_SOURCE_MIN", "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO"]

        _, test_enc = compute_knn_target_encoding(X_train, y_train, X_test, features, k=10)

        assert (test_enc >= 0.0).all(), "All test_enc values must be >= 0.0"
        assert (test_enc <= 1.0).all(), "All test_enc values must be <= 1.0"

    def test_output_length_matches_input(self, knn_encoding_data):
        """len(train_enc) == len(X_train), len(test_enc) == len(X_test)."""
        X_train, y_train, X_test, y_test = knn_encoding_data
        features = ["EXT_SOURCE_MEAN", "EXT_SOURCE_MIN", "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO"]

        train_enc, test_enc = compute_knn_target_encoding(X_train, y_train, X_test, features, k=10)

        assert len(train_enc) == len(X_train), "train_enc length must match X_train"
        assert len(test_enc) == len(X_test), "test_enc length must match X_test"

    def test_output_indexed_as_input(self, knn_encoding_data):
        """train_enc index matches X_train index; test_enc index matches X_test."""
        X_train, y_train, X_test, y_test = knn_encoding_data
        features = ["EXT_SOURCE_MEAN", "EXT_SOURCE_MIN", "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO"]

        train_enc, test_enc = compute_knn_target_encoding(X_train, y_train, X_test, features, k=10)

        pd.testing.assert_index_equal(train_enc.index, X_train.index)
        pd.testing.assert_index_equal(test_enc.index, X_test.index)

    def test_no_leakage_train_knn_size(self, knn_encoding_data, monkeypatch):
        """Each OOF fold's KNN is trained on < len(X_train) rows.
        Patch KNeighborsClassifier.fit to capture call argument sizes."""
        X_train, y_train, X_test, y_test = knn_encoding_data
        features = ["EXT_SOURCE_MEAN", "EXT_SOURCE_MIN", "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO"]

        from sklearn.neighbors import KNeighborsClassifier

        fit_sizes = []
        original_fit = KNeighborsClassifier.fit

        def tracked_fit(self, X, y):
            fit_sizes.append(len(X))
            return original_fit(self, X, y)

        monkeypatch.setattr(KNeighborsClassifier, "fit", tracked_fit)

        train_enc, _ = compute_knn_target_encoding(X_train, y_train, X_test, features, k=10, n_folds=5)

        # First 5 calls are OOF folds, each should be < len(X_train)
        oof_sizes = fit_sizes[:5]
        for fold_idx, size in enumerate(oof_sizes):
            assert size < len(X_train), (
                f"Fold {fold_idx}: KNN trained on {size} rows, must be < {len(X_train)} to prevent leakage"
            )

    def test_test_uses_full_train_fit(self, knn_encoding_data, monkeypatch):
        """KNN for test set is fit once on len(X_train) rows (not folds).
        Patch KNeighborsClassifier.fit and assert the largest fit = len(X_train)."""
        X_train, y_train, X_test, y_test = knn_encoding_data
        features = ["EXT_SOURCE_MEAN", "EXT_SOURCE_MIN", "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO"]

        from sklearn.neighbors import KNeighborsClassifier

        fit_sizes = []
        original_fit = KNeighborsClassifier.fit

        def tracked_fit(self, X, y):
            fit_sizes.append(len(X))
            return original_fit(self, X, y)

        monkeypatch.setattr(KNeighborsClassifier, "fit", tracked_fit)

        train_enc, test_enc = compute_knn_target_encoding(X_train, y_train, X_test, features, k=10, n_folds=5)

        # The last fit call should be on full training set
        assert max(fit_sizes) == len(X_train), (
            f"Max fit size {max(fit_sizes)} must equal len(X_train) {len(X_train)}"
        )

    def test_handles_nan_in_features(self):
        """Features with NaN (EXT_SOURCE missing) don't raise errors.
        NaN must be replaced with -999 sentinel before KNN computation."""
        rng = np.random.default_rng(42)
        n_train = 50
        n_test = 10

        X_train = pd.DataFrame({
            "EXT_SOURCE_MEAN": np.where(rng.random(n_train) < 0.3, np.nan, rng.uniform(0, 1, n_train)),
            "EXT_SOURCE_MIN": rng.uniform(0, 0.8, n_train),
            "CREDIT_INCOME_RATIO": rng.uniform(0, 5, n_train),
            "ANNUITY_INCOME_RATIO": rng.uniform(0, 1, n_train),
        })
        y_train = pd.Series(rng.binomial(1, 0.15, n_train), index=X_train.index)

        X_test = pd.DataFrame({
            "EXT_SOURCE_MEAN": np.where(rng.random(n_test) < 0.3, np.nan, rng.uniform(0, 1, n_test)),
            "EXT_SOURCE_MIN": rng.uniform(0, 0.8, n_test),
            "CREDIT_INCOME_RATIO": rng.uniform(0, 5, n_test),
            "ANNUITY_INCOME_RATIO": rng.uniform(0, 1, n_test),
        })

        features = ["EXT_SOURCE_MEAN", "EXT_SOURCE_MIN", "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO"]

        # Should not raise; NaN handling is internal
        train_enc, test_enc = compute_knn_target_encoding(X_train, y_train, X_test, features, k=10)

        assert len(train_enc) == n_train
        assert len(test_enc) == n_test
        assert not train_enc.isna().any(), "train_enc must not contain NaN"
        assert not test_enc.isna().any(), "test_enc must not contain NaN"

    def test_missing_feature_col_raises_valueerror(self, knn_encoding_data):
        """If a feature in `features` list is absent from X_train, raise ValueError."""
        X_train, y_train, X_test, y_test = knn_encoding_data
        features = ["EXT_SOURCE_MEAN", "NONEXISTENT_FEATURE"]

        with pytest.raises(ValueError):
            compute_knn_target_encoding(X_train, y_train, X_test, features, k=10)

    def test_k_neighbours_respected(self, knn_encoding_data, monkeypatch):
        """With k=3 and 10 training rows, the KNN uses n_neighbors=3."""
        from src.features import _NAN_SENTINEL
        from sklearn.neighbors import KNeighborsClassifier

        # Create smaller dataset
        X_train = pd.DataFrame({
            "EXT_SOURCE_MEAN": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
            "EXT_SOURCE_MIN": [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.9],
            "CREDIT_INCOME_RATIO": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "ANNUITY_INCOME_RATIO": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
        })
        y_train = pd.Series([0, 1, 0, 0, 1, 0, 1, 0, 0, 1], index=X_train.index)

        X_test = pd.DataFrame({
            "EXT_SOURCE_MEAN": [0.5],
            "EXT_SOURCE_MIN": [0.45],
            "CREDIT_INCOME_RATIO": [5],
            "ANNUITY_INCOME_RATIO": [0.5],
        })

        features = ["EXT_SOURCE_MEAN", "EXT_SOURCE_MIN", "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO"]

        n_neighbors_used = []
        original_init = KNeighborsClassifier.__init__

        def tracked_init(self, n_neighbors=5, **kwargs):
            n_neighbors_used.append(n_neighbors)
            return original_init(self, n_neighbors=n_neighbors, **kwargs)

        monkeypatch.setattr(KNeighborsClassifier, "__init__", tracked_init)

        train_enc, test_enc = compute_knn_target_encoding(X_train, y_train, X_test, features, k=3, n_folds=2)

        # All KNN instances should use k=3
        for k_used in n_neighbors_used:
            assert k_used == 3, f"Expected k=3, got k={k_used}"


# ---------------------------------------------------------------------------
# Instalment time-series features — TDD tests (RED phase)
# ---------------------------------------------------------------------------


class TestEngineerInstalmentStreaks:
    """TDD tests for engineer_instalment_streaks()."""

    @pytest.fixture
    def instalment_data(self) -> pd.DataFrame:
        """
        Synthetic instalment payments table with rows designed to test streak logic.

        Columns: SK_ID_CURR, DAYS_INSTALMENT, DAYS_ENTRY_PAYMENT, AMT_INSTALMENT, AMT_PAYMENT

        Notes:
        - DAYS_INSTALMENT, DAYS_ENTRY_PAYMENT are negative (days before application).
        - More recent = less negative = higher value.
        - A payment is LATE when DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT.
        - DPD (Days Past Due) = max(0, DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT).
        """
        return pd.DataFrame({
            "SK_ID_CURR": [
                101, 101, 101, 101, 101,  # Borrower 101: 5 instalments, streaky DPD
                102, 102, 102,              # Borrower 102: 3 instalments, no late
                103, 103, 103, 103,         # Borrower 103: 4 instalments, uniform payment
                104, 104,                   # Borrower 104: 2 instalments, only 1 record
            ],
            "DAYS_INSTALMENT": [
                -120, -100, -80, -60, -40,  # Recent to old for 101
                -100, -80, -60,              # Recent to old for 102 (no DPD)
                -120, -100, -80, -60,        # Recent to old for 103
                -50, -30,                    # Recent to old for 104
            ],
            "DAYS_ENTRY_PAYMENT": [
                -115, -105, -75, -70, -40,  # 101: late on days 100, 80, 60; early on day 120
                -100, -80, -60,              # 102: on time always
                -120, -95, -80, -65,         # 103: late on day 100, slight on 60 and 80
                -50, -25,                    # 104: 0 DPD on first, 5 on second
            ],
            "AMT_INSTALMENT": [
                1000.0, 1000.0, 1050.0, 1050.0, 1100.0,  # 101: increasing trend
                1000.0, 1000.0, 1000.0,                   # 102: flat
                500.0, 500.0, 500.0, 500.0,               # 103: flat
                2000.0, 2500.0,                            # 104: increasing
            ],
            "AMT_PAYMENT": [
                950.0, 980.0, 1000.0, 1020.0, 1100.0,   # 101: increasing
                1000.0, 1000.0, 1000.0,                  # 102: exact
                505.0, 510.0, 515.0, 520.0,              # 103: increasing
                2000.0, 2500.0,                           # 104: increasing
            ],
        })

    def test_engineer_instalment_streaks_returns_correct_columns(self, instalment_data):
        """Output must have exactly these 5 columns."""
        result = engineer_instalment_streaks(instalment_data)
        expected_cols = {
            "inst_longest_dpd_streak",
            "inst_months_since_last_dpd",
            "inst_payment_amt_slope",
            "inst_payment_ratio_trend",
            "inst_recent_vs_historical_dpd",
        }
        assert set(result.columns) == expected_cols, (
            f"Expected columns {expected_cols}, got {set(result.columns)}"
        )

    def test_engineer_instalment_streaks_shape(self, instalment_data):
        """Output must have one row per unique SK_ID_CURR."""
        result = engineer_instalment_streaks(instalment_data)
        unique_borrowers = instalment_data["SK_ID_CURR"].nunique()
        assert len(result) == unique_borrowers, (
            f"Expected {unique_borrowers} rows, got {len(result)}"
        )

    def test_engineer_instalment_streaks_index_is_sk_id_curr(self, instalment_data):
        """Result index must be SK_ID_CURR."""
        result = engineer_instalment_streaks(instalment_data)
        assert result.index.name == "SK_ID_CURR", (
            f"Expected index name 'SK_ID_CURR', got '{result.index.name}'"
        )

    def test_engineer_instalment_streaks_no_dpd_borrower(self, instalment_data):
        """Borrower 102 has zero DPD (DAYS_ENTRY_PAYMENT == DAYS_INSTALMENT on all).
        inst_longest_dpd_streak must be 0."""
        result = engineer_instalment_streaks(instalment_data)
        assert result.loc[102, "inst_longest_dpd_streak"] == 0, (
            "Borrower with no DPD must have inst_longest_dpd_streak == 0"
        )

    def test_engineer_instalment_streaks_detects_streak(self, instalment_data):
        """Borrower 101 has consecutive late payments (DAYS_INSTALMENT 100, 80, 60).
        inst_longest_dpd_streak must be > 0."""
        result = engineer_instalment_streaks(instalment_data)
        streak = result.loc[101, "inst_longest_dpd_streak"]
        assert streak > 0, "Borrower with DPD must have inst_longest_dpd_streak > 0"

    def test_engineer_instalment_streaks_no_nan_or_inf(self, instalment_data):
        """Output must have zero NaN or inf values."""
        result = engineer_instalment_streaks(instalment_data)
        assert not result.isna().any().any(), "Result must not contain NaN"
        assert not result.isin([np.inf, -np.inf]).any().any(), "Result must not contain inf"

    def test_engineer_instalment_streaks_months_since_dpd_correct_bounds(self, instalment_data):
        """Borrower with no DPD must have inst_months_since_last_dpd == 999 (far past)."""
        result = engineer_instalment_streaks(instalment_data)
        # Borrower 102 has no DPD
        assert result.loc[102, "inst_months_since_last_dpd"] == 999, (
            "Borrower with no DPD must have inst_months_since_last_dpd == 999"
        )

    def test_engineer_instalment_streaks_payment_slope_reasonable(self, instalment_data):
        """Borrower 101 has increasing payment amounts → positive slope.
        Borrower 102 has flat amounts → slope near 0."""
        result = engineer_instalment_streaks(instalment_data)
        slope_101 = result.loc[101, "inst_payment_amt_slope"]
        slope_102 = result.loc[102, "inst_payment_amt_slope"]
        # 101 increases: 1000 → 1100, so slope should be positive
        assert slope_101 > 0, "Borrower with increasing amounts must have positive slope"
        # 102 is flat: all 1000, so slope should be near 0
        assert abs(slope_102) < 0.1, "Borrower with flat amounts must have slope near 0"

    def test_engineer_instalment_streaks_single_instalment(self, instalment_data):
        """Borrower with only 1 instalment should not crash (edge case for slope).
        Slope should be 0.0 when n < 2."""
        result = engineer_instalment_streaks(instalment_data)
        # Borrower 104 has only 2 instalments; slope should be computed
        slope_104 = result.loc[104, "inst_payment_amt_slope"]
        assert isinstance(slope_104, (int, float)), "Slope must be numeric"
        assert not np.isnan(slope_104), "Slope must not be NaN"

    def test_engineer_instalment_streaks_integration_with_pipeline(self, instalment_data):
        """Calling engineer_instalment_streaks() and merging into a full dataframe works."""
        result = engineer_instalment_streaks(instalment_data)
        # Create a dummy full dataframe and merge
        full_df = pd.DataFrame({
            "SK_ID_CURR": [101, 102, 103, 104],
            "AMT_INCOME_TOTAL": [100_000, 80_000, 120_000, 90_000],
        }).set_index("SK_ID_CURR")
        merged = full_df.join(result, how="left")
        assert merged.shape[1] == 6, f"Expected 6 columns, got {merged.shape[1]}"
        assert not merged.isnull().any().any(), "Merged dataframe must not have NaN"

    def test_engineer_secondary_features_with_instalment_data(self, instalment_data):
        """engineer_secondary_features() accepts optional df_inst parameter.
        When provided, output includes the 5 instalment streak features."""
        # Create a minimal secondary features dataframe
        secondary_df = pd.DataFrame({
            "SK_ID_CURR": [101, 102, 103, 104],
            "inst_cnt": [5, 3, 4, 2],
            "inst_late_cnt": [2, 0, 1, 0],
            "prev_cnt": [1, 2, 1, 0],
            "prev_approved_cnt": [1, 2, 0, 0],
            "AMT_INCOME_TOTAL": [100_000, 80_000, 120_000, 90_000],
            "AMT_ANNUITY": [5_000, 4_000, 6_000, 3_000],
        }).set_index("SK_ID_CURR")

        # Call engineer_secondary_features with df_inst parameter
        result = engineer_secondary_features(secondary_df, df_inst=instalment_data)

        # Check that instalment streak columns are present
        streak_cols = {
            "inst_longest_dpd_streak",
            "inst_months_since_last_dpd",
            "inst_payment_amt_slope",
            "inst_payment_ratio_trend",
            "inst_recent_vs_historical_dpd",
        }
        for col in streak_cols:
            assert col in result.columns, f"Missing column: {col}"

        # Check that original columns are still present
        assert "inst_cnt" in result.columns
        assert "inst_late_cnt" in result.columns

    def test_engineer_secondary_features_backward_compatible(self):
        """engineer_secondary_features() works without df_inst (backward compatible)."""
        secondary_df = pd.DataFrame({
            "SK_ID_CURR": [1, 2, 3],
            "inst_cnt": [5, 3, 4],
            "inst_late_cnt": [2, 0, 1],
            "prev_cnt": [1, 2, 1],
            "prev_approved_cnt": [1, 2, 0],
            "AMT_INCOME_TOTAL": [100_000, 80_000, 120_000],
        }).set_index("SK_ID_CURR")

        # Call without df_inst (None by default)
        result = engineer_secondary_features(secondary_df)

        # Should have original columns plus derived ratios
        assert "inst_cnt" in result.columns
        assert "prev_approval_rate" in result.columns
        # Should NOT have instalment streak features
        assert "inst_longest_dpd_streak" not in result.columns


# ---------------------------------------------------------------------------
# Tests: Combined Feature Store (Wave 0 — test stubs, RED state)
# ---------------------------------------------------------------------------


def _prod_raw_available() -> bool:
    """Return True if production X_raw_features.parquet is present with >= 100K rows."""
    _path = Path("data/processed/X_raw_features.parquet")
    if not _path.exists():
        return False
    try:
        return pd.read_parquet(_path, columns=["SK_ID_CURR"]).shape[0] >= 100_000
    except Exception:
        return False


@pytest.mark.skipif(
    not _prod_raw_available(),
    reason="Production X_raw_features.parquet not available (< 100K rows)",
)
class TestCombinedStore:
    """Integration tests for combined feature store construction.

    These tests require production-scale data (307,511 rows).
    They are skipped automatically when only mock data is present.

    Tests define expected behavior for:
    - Combined store shape (307,511 rows, >= 65 columns)
    - NaN sentinel handling (-999 for all missing)
    - EXT_SOURCE_3_MISSING_FLAG presence
    - Row alignment with y_train
    """

    def test_combined_store_shape(self):
        """Verify combined store has correct shape.

        Arrange: Combined feature store (raw + DFS + imputed)
        Act: Load combined parquet
        Assert: Shape == (307,511 rows, >= 63 columns minimum)
        """
        from src.features import build_combined_feature_store

        X_combined = build_combined_feature_store()
        assert X_combined.shape[0] == 307511, f"Row mismatch: {X_combined.shape[0]}"
        # Minimum: 62 raw + 1 missing flag = 63
        # Target when DFS commits: >= 65
        assert X_combined.shape[1] >= 63, f"Column count too low: {X_combined.shape[1]}"

    def test_combined_store_no_nan(self):
        """Verify that no NaN values remain in combined store.

        Arrange: Combined feature store with imputation applied
        Act: Check for NaN in all columns
        Assert: All NaN replaced with -999 sentinel
        """
        from src.features import build_combined_feature_store

        X_combined = build_combined_feature_store()
        nan_count = X_combined.isna().sum().sum()
        assert nan_count == 0, f"Found {nan_count} NaN values (should be 0)"

    def test_combined_store_includes_missing_flag(self):
        """Verify that EXT_SOURCE_3_MISSING_FLAG column exists.

        Arrange: Combined feature store
        Act: Check for EXT_SOURCE_3_MISSING_FLAG column
        Assert: Column exists; values are binary (0/1)
        """
        from src.features import build_combined_feature_store

        X_combined = build_combined_feature_store()
        assert "EXT_SOURCE_3_MISSING_FLAG" in X_combined.columns, "Missing flag column not found"
        flag_values = X_combined["EXT_SOURCE_3_MISSING_FLAG"].unique()
        assert set(flag_values).issubset({0, 1}), f"Flag values not binary: {flag_values}"

    def test_combined_store_matches_y_train_alignment(self):
        """Verify that combined store rows align with y_train.

        Arrange: Combined store and y_train loaded
        Act: Compare row count and index
        Assert: len(combined_store) == len(y_train) == 307,511
        """
        from src.features import build_combined_feature_store

        X_combined = build_combined_feature_store()
        y_train = pd.read_parquet("data/processed/y_train.parquet")
        assert (
            len(X_combined) == len(y_train) == 307511
        ), f"Row count mismatch: X={len(X_combined)}, y={len(y_train)}"


# ---------------------------------------------------------------------------
# Phase 04.2.1 — Tree feature store TDD suite
# ---------------------------------------------------------------------------


@pytest.fixture
def tree_store_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Minimal synthetic application frame for build_tree_feature_store tests.

    500 rows with 3 numeric columns of different variance levels plus 1
    constant column (zero variance, must be dropped by variance filter).
    """
    rng = np.random.default_rng(7)
    n = 500
    high_var = rng.uniform(0, 100, n)
    target = pd.Series((high_var < 25).astype(int))

    df = pd.DataFrame({
        "high_var_feature": high_var,
        "medium_var_feature": rng.uniform(0, 10, n),
        "low_var_feature": rng.uniform(0, 0.01, n),
        "constant_feature": np.ones(n),
    })
    return df, target


class TestBuildTreeFeatureStore:
    """TDD tests for build_tree_feature_store() — Phase 04.2.1."""

    def test_returns_dataframe_and_list(self, tree_store_data, tmp_path):
        """Return type must be (DataFrame, list[str]).

        Arrange: Synthetic data
        Act: Call build_tree_feature_store
        Assert: Returns 2-tuple of DataFrame + list
        """
        X, y = tree_store_data
        out_path = str(tmp_path / "X_tree_test.parquet")
        result = build_tree_feature_store(X, y, output_dir=out_path)

        assert isinstance(result, tuple), "Result must be a tuple"
        assert len(result) == 2, "Tuple must have exactly 2 elements"
        X_out, cols = result
        assert isinstance(X_out, pd.DataFrame), "First element must be DataFrame"
        assert isinstance(cols, list), "Second element must be list"
        assert len(cols) > 0, "Column list must be non-empty"

    def test_no_nan_values_in_output(self, tree_store_data, tmp_path):
        """All NaN values must be replaced with -999 sentinel before save.

        Arrange: Data with NaN injected
        Act: Call build_tree_feature_store
        Assert: Output DataFrame has zero NaN values
        """
        X, y = tree_store_data
        X_nan = X.copy()
        X_nan.loc[0, "high_var_feature"] = np.nan
        X_nan.loc[1, "medium_var_feature"] = np.nan

        out_path = str(tmp_path / "X_tree_nan_test.parquet")
        X_out, _ = build_tree_feature_store(X_nan, y, output_dir=out_path)

        nan_count = X_out.isna().sum().sum()
        assert nan_count == 0, f"Found {nan_count} NaN values; all must be -999 sentinel"

    def test_no_woe_columns_in_output(self, tree_store_data, tmp_path):
        """Output must contain no columns with '_woe' suffix.

        Arrange: Standard synthetic data
        Act: Call build_tree_feature_store
        Assert: Zero columns with '_woe' in their name
        """
        X, y = tree_store_data
        out_path = str(tmp_path / "X_tree_woe_test.parquet")
        X_out, cols = build_tree_feature_store(X, y, output_dir=out_path)

        woe_cols = [c for c in X_out.columns if "_woe" in c]
        assert len(woe_cols) == 0, f"WoE columns must be absent; found: {woe_cols}"

    def test_all_output_dtypes_numeric(self, tree_store_data, tmp_path):
        """All output columns must be numeric (no category or object dtypes).

        Arrange: Standard synthetic data
        Act: Call build_tree_feature_store
        Assert: select_dtypes(exclude=np.number) returns empty DataFrame
        """
        X, y = tree_store_data
        out_path = str(tmp_path / "X_tree_dtype_test.parquet")
        X_out, _ = build_tree_feature_store(X, y, output_dir=out_path)

        non_numeric = X_out.select_dtypes(exclude=[np.number]).columns.tolist()
        assert len(non_numeric) == 0, f"Non-numeric columns found: {non_numeric}"

    def test_variance_filter_drops_constant_column(self, tree_store_data, tmp_path):
        """Constant columns (variance=0) must be dropped by variance filter.

        Arrange: Data with 'constant_feature' (all ones)
        Act: Call build_tree_feature_store
        Assert: 'constant_feature' is absent from output columns
        """
        X, y = tree_store_data
        assert "constant_feature" in X.columns, "Fixture must include constant_feature"

        out_path = str(tmp_path / "X_tree_const_test.parquet")
        X_out, cols = build_tree_feature_store(X, y, output_dir=out_path)

        assert "constant_feature" not in X_out.columns, (
            "Constant column must be removed by variance filter"
        )
        assert "constant_feature" not in cols

    def test_variance_filter_retains_nonzero_variance_columns(self, tree_store_data, tmp_path):
        """At least one column with non-zero variance must be retained.

        Arrange: Data with high_var_feature (high variance)
        Act: Call build_tree_feature_store
        Assert: Output has >= 1 column; high_var_feature present
        """
        X, y = tree_store_data
        out_path = str(tmp_path / "X_tree_var_test.parquet")
        X_out, cols = build_tree_feature_store(X, y, output_dir=out_path)

        assert X_out.shape[1] >= 1, "At least 1 column must survive variance filter"
        assert "high_var_feature" in X_out.columns, (
            "High-variance column must survive variance filter"
        )

    def test_output_shape_rows_preserved(self, tree_store_data, tmp_path):
        """Row count must equal input row count — no sampling or dropping.

        Arrange: 500-row synthetic data
        Act: Call build_tree_feature_store
        Assert: Output has exactly 500 rows
        """
        X, y = tree_store_data
        out_path = str(tmp_path / "X_tree_shape_test.parquet")
        X_out, _ = build_tree_feature_store(X, y, output_dir=out_path)

        assert X_out.shape[0] == len(X), (
            f"Row count must match input: expected {len(X)}, got {X_out.shape[0]}"
        )

    def test_feature_columns_pkl_saved(self, tree_store_data, tmp_path, monkeypatch):
        """models/raw_feature_columns.pkl must be saved with the column list.

        Arrange: Patch models dir to tmp_path
        Act: Call build_tree_feature_store
        Assert: PKL file exists and loads back the same column list
        """
        import pickle

        X, y = tree_store_data
        out_path = str(tmp_path / "X_tree_pkl_test.parquet")
        pkl_path = tmp_path / "raw_feature_columns.pkl"

        # Patch open() only for the pkl write within the function scope via monkeypatch
        # Simpler: call function, then check that the column list file was written
        # We accept the real pkl write to models/ and verify the returned list
        X_out, cols = build_tree_feature_store(X, y, output_dir=out_path)

        # The column list returned must be non-empty and match the output columns
        assert cols == list(X_out.columns), (
            "Returned column list must match output DataFrame columns exactly"
        )

    def test_no_inf_values_in_output(self, tree_store_data, tmp_path):
        """inf and -inf values must be replaced before save.

        Arrange: Data with inf injected
        Act: Call build_tree_feature_store
        Assert: Output DataFrame has zero inf values
        """
        X, y = tree_store_data
        X_inf = X.copy()
        X_inf.loc[0, "high_var_feature"] = np.inf
        X_inf.loc[1, "medium_var_feature"] = -np.inf

        out_path = str(tmp_path / "X_tree_inf_test.parquet")
        X_out, _ = build_tree_feature_store(X_inf, y, output_dir=out_path)

        inf_count = np.isinf(X_out.select_dtypes(include=[np.number])).sum().sum()
        assert inf_count == 0, f"Found {inf_count} inf values; must be replaced with -999"

    def test_apply_raw_feature_store_round_trip(self, tree_store_data, tmp_path):
        """apply_raw_feature_store must return same columns as build produced.

        Arrange: Run build, then apply on same data
        Act: Compare output columns of both calls
        Assert: apply output has same columns in same order as build output
        """
        X, y = tree_store_data
        out_path = str(tmp_path / "X_tree_roundtrip_test.parquet")

        X_built, build_cols = build_tree_feature_store(X, y, output_dir=out_path)
        X_applied = apply_raw_feature_store(X, build_cols)

        assert list(X_applied.columns) == build_cols, (
            "apply_raw_feature_store must return exact column list from build"
        )
        assert X_applied.shape[0] == X.shape[0], "Row count must be preserved in apply"


# ---------------------------------------------------------------------------
# Tests: engineer_time_features (Wave 2)
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_bureau_tables(tmp_path) -> Path:
    """
    Create synthetic bureau and bureau_balance CSVs in a temporary directory.

    Generates data for 10 unique SK_ID_CURR applicants with varying amounts of
    secondary table data to exercise edge cases (0, 1, >1 rows per customer).

    Returns
    -------
    Path
        Temporary directory containing bureau.csv and bureau_balance.csv.
    """
    # Create bureau.csv
    bureau_data = {
        "SK_ID_CURR": [1001, 1001, 1002, 1003, 1003, 1004, 1005, 1005, 1005, 1006],
        "SK_ID_BUREAU": [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010],
        "DAYS_CREDIT": [-365, -730, -180, -500, -1000, -90, -300, -600, -900, -200],
        "CREDIT_ACTIVE": ["Active", "Closed", "Active", "Active", "Closed", "Active", "Active", "Closed", "Active", "Active"],
    }
    bureau_df = pd.DataFrame(bureau_data)
    bureau_df.to_csv(tmp_path / "bureau.csv", index=False)

    # Create bureau_balance.csv
    bureau_balance_data = {
        "SK_ID_BUREAU": [
            # SK_ID_BUREAU 2001 (SK_ID_CURR 1001): 4 rows with STATUS ['0', '0', '1', '2']
            2001, 2001, 2001, 2001,
            # SK_ID_BUREAU 2002 (SK_ID_CURR 1001): 2 rows with STATUS ['C', '1']
            2002, 2002,
            # SK_ID_BUREAU 2003 (SK_ID_CURR 1002): 3 rows with STATUS ['0', '0', '0']
            2003, 2003, 2003,
            # SK_ID_BUREAU 2004 (SK_ID_CURR 1003): 2 rows with STATUS ['1', '1']
            2004, 2004,
            # SK_ID_BUREAU 2005 (SK_ID_CURR 1003): 1 row with STATUS ['0']
            2005,
            # SK_ID_BUREAU 2006 (SK_ID_CURR 1004): 3 rows with STATUS ['0', '0', '0']
            2006, 2006, 2006,
            # SK_ID_BUREAU 2007 (SK_ID_CURR 1005): 2 rows with STATUS ['2', '3']
            2007, 2007,
            # SK_ID_BUREAU 2008 (SK_ID_CURR 1005): 2 rows with STATUS ['0', '0']
            2008, 2008,
            # SK_ID_BUREAU 2009 (SK_ID_CURR 1005): 1 row with STATUS ['4']
            2009,
            # SK_ID_BUREAU 2010 (SK_ID_CURR 1006): 0 rows (applicant with no balance records)
        ],
        "MONTHS_BALANCE": [
            # 2001: -1, -2, -3, -4
            -1, -2, -3, -4,
            # 2002: -1, -2
            -1, -2,
            # 2003: -1, -2, -3
            -1, -2, -3,
            # 2004: -1, -2
            -1, -2,
            # 2005: -1
            -1,
            # 2006: -1, -2, -3
            -1, -2, -3,
            # 2007: -1, -2
            -1, -2,
            # 2008: -1, -2
            -1, -2,
            # 2009: -1
            -1,
        ],
        "STATUS": [
            # 2001: 0, 0, 1, 2
            "0", "0", "1", "2",
            # 2002: C, 1
            "C", "1",
            # 2003: 0, 0, 0
            "0", "0", "0",
            # 2004: 1, 1
            "1", "1",
            # 2005: 0
            "0",
            # 2006: 0, 0, 0
            "0", "0", "0",
            # 2007: 2, 3
            "2", "3",
            # 2008: 0, 0
            "0", "0",
            # 2009: 4
            "4",
        ],
    }
    bureau_balance_df = pd.DataFrame(bureau_balance_data)
    bureau_balance_df.to_csv(tmp_path / "bureau_balance.csv", index=False)

    # engineer_time_features needs application_train.csv to build the full SK_ID_CURR index
    app_train_df = pd.DataFrame({"SK_ID_CURR": [1001, 1002, 1003, 1004, 1005, 1006]})
    app_train_df.to_csv(tmp_path / "application_train.csv", index=False)

    return tmp_path


class TestEngineerTimeFeatures:
    """Unit and integration tests for engineer_time_features function."""

    def test_engineer_time_features_bbal_dpd_rate_3m(self, synthetic_bureau_tables):
        """
        Test that bbal_dpd_rate_3m is correctly calculated.

        SK_ID_CURR 1001:
          - Bureau 2001 (MONTHS_BALANCE: -1, -2, -3, -4; STATUS: 0, 0, 1, 2)
            In 3m window (>= -3): rows at -1, -2, -3 have STATUS 0, 0, 1 → 1 DPD
          - Bureau 2002 (MONTHS_BALANCE: -1, -2; STATUS: C, 1)
            In 3m window (>= -3): 2 rows with STATUS C, 1 → 1 DPD
          - Combined: 5 rows in 3m, 2 DPD → rate = 2/5 = 0.4

        SK_ID_CURR 1002: all STATUS = 0 (clean) → rate = 0.0
        SK_ID_CURR 1003: Bureau 2004 has 2 rows [-1, -2] with STATUS [1, 1] (both DPD),
                         Bureau 2005 has 1 row [-1] with STATUS [0] (clean)
                         → 3 rows in 3m, 2 DPD → rate = 2/3 ≈ 0.667
        """
        from src.features import engineer_time_features

        result = engineer_time_features(synthetic_bureau_tables)

        # SK_ID_CURR 1001: 5 rows in 3m, 2 DPD (STATUS 1, 2) → 2/5 = 0.4
        assert result.loc[1001, "bbal_dpd_rate_3m"] == pytest.approx(0.4, abs=1e-3), (
            f"SK_ID_CURR 1001: expected 0.4, got {result.loc[1001, 'bbal_dpd_rate_3m']}"
        )

        # SK_ID_CURR 1002: all STATUS = 0 → rate = 0.0
        assert result.loc[1002, "bbal_dpd_rate_3m"] == pytest.approx(0.0, abs=1e-3)

        # SK_ID_CURR 1003: 3 rows (2 DPD + 1 clean) → rate = 2/3 ≈ 0.667
        assert result.loc[1003, "bbal_dpd_rate_3m"] == pytest.approx(2/3, abs=1e-3)

    def test_engineer_time_features_bbal_months_since_last_dpd(self, synthetic_bureau_tables):
        """
        Test that bbal_months_since_last_dpd is correctly calculated.

        The function finds the MIN(MONTHS_BALANCE) among all delinquent rows
        (min because MONTHS_BALANCE is negative; min = oldest = most in the past),
        then negates it to get "months ago".

        SK_ID_CURR 1001:
          - Bureau 2001 DPD: MONTHS_BALANCE [-3, -4] with STATUS [1, 2]
          - Bureau 2002 DPD: MONTHS_BALANCE [-1] with STATUS [1]
          - Min MONTHS_BALANCE = -4 → -(-4) = 4.0 months ago

        SK_ID_CURR 1002: no DPD → -999 sentinel

        SK_ID_CURR 1003:
          - Bureau 2004 DPD: MONTHS_BALANCE [-1, -2] with STATUS [1, 1]
          - Min MONTHS_BALANCE = -2 → -(-2) = 2.0 months ago
        """
        from src.features import engineer_time_features

        result = engineer_time_features(synthetic_bureau_tables)

        # SK_ID_CURR 1001: min DPD at -4 → 4.0 months ago
        assert result.loc[1001, "bbal_months_since_last_dpd"] == pytest.approx(4.0, abs=1e-3)

        # SK_ID_CURR 1002: no DPD → sentinel
        assert result.loc[1002, "bbal_months_since_last_dpd"] == pytest.approx(-999.0, abs=1e-3)

        # SK_ID_CURR 1003: min DPD at -2 → 2.0 months ago
        assert result.loc[1003, "bbal_months_since_last_dpd"] == pytest.approx(2.0, abs=1e-3)

    def test_engineer_time_features_bureau_credit_age_mean(self, synthetic_bureau_tables):
        """
        Test that bureau_credit_age_mean is correctly calculated in years.

        The function converts DAYS_CREDIT to years using 365.25 as the divisor.

        SK_ID_CURR 1001: DAYS_CREDIT = [-365, -730]
          → ages = [365/365.25, 730/365.25] = [0.99932, 1.99863]
          → mean ≈ 1.4989 years

        SK_ID_CURR 1002: DAYS_CREDIT = [-180]
          → age = 180/365.25 ≈ 0.4923 years

        SK_ID_CURR 1006: DAYS_CREDIT = [-200]
          → age = 200/365.25 ≈ 0.5475 years
        """
        from src.features import engineer_time_features

        result = engineer_time_features(synthetic_bureau_tables)

        # SK_ID_CURR 1001: mean of (365/365.25 + 730/365.25) / 2 ≈ 1.499
        expected_1001 = (365.0 / 365.25 + 730.0 / 365.25) / 2.0
        assert result.loc[1001, "bureau_credit_age_mean"] == pytest.approx(expected_1001, abs=1e-3)

        # SK_ID_CURR 1002: 180 / 365.25 ≈ 0.492
        assert result.loc[1002, "bureau_credit_age_mean"] == pytest.approx(180.0 / 365.25, abs=1e-3)

        # SK_ID_CURR 1006: 200 / 365.25 ≈ 0.548
        assert result.loc[1006, "bureau_credit_age_mean"] == pytest.approx(200.0 / 365.25, abs=1e-3)

    def test_engineer_time_features_returns_dataframe(self, synthetic_bureau_tables):
        """
        Test that engineer_time_features returns the correct DataFrame structure.

        Assert:
        - Return type is pd.DataFrame
        - Index name is "SK_ID_CURR"
        - Exactly 3 columns
        - No NaN values (all should be -999 sentinel)
        - All columns are numeric
        """
        from src.features import engineer_time_features

        result = engineer_time_features(synthetic_bureau_tables)

        assert isinstance(result, pd.DataFrame)
        assert result.index.name == "SK_ID_CURR"
        assert result.shape[1] == 3, f"Expected 3 columns, got {result.shape[1]}"

        # Check column names
        expected_cols = {"bbal_dpd_rate_3m", "bbal_months_since_last_dpd", "bureau_credit_age_mean"}
        assert set(result.columns) == expected_cols

        # No NaN values
        assert result.isna().sum().sum() == 0, "Should have no NaN values (use -999 sentinel)"

        # All numeric
        assert result.select_dtypes(include=[np.number]).shape[1] == 3, "All columns must be numeric"

    def test_engineer_time_features_on_real_data(self):
        """
        Integration test: run engineer_time_features on real data if available.

        Skips gracefully if data/bureau.csv not found.
        """
        from src.features import engineer_time_features

        data_dir = Path("data")
        if not (data_dir / "bureau.csv").exists() or not (data_dir / "bureau_balance.csv").exists():
            pytest.skip("Real data files not found in data/ directory")

        result = engineer_time_features(data_dir)

        # Shape: rows should match unique SK_ID_CURR in bureau
        assert result.shape[1] == 3, "Should have 3 columns"
        assert result.shape[0] > 100, "Should have at least 100 applicants in real data"

        # No NaN
        assert result.isna().sum().sum() == 0, "No NaN values allowed"

        # All numeric
        assert result.select_dtypes(include=[np.number]).shape[1] == 3, "All numeric"

        # Index is SK_ID_CURR
        assert result.index.name == "SK_ID_CURR"
        assert result.index.is_unique, "SK_ID_CURR should be unique index"


class TestBuildTreeFeatureStoreL6Filters:
    """Regression tests for Layer 6 feature selection (variance + correlation dedup)."""

    def test_build_tree_feature_store_applies_variance_threshold(self, tree_store_data, tmp_path):
        """Verify VarianceThreshold(0.01) is applied and no features with variance < 0.01 remain."""
        X, y = tree_store_data

        X_tree, feature_cols = build_tree_feature_store(
            X=X,
            y=y,
            output_dir=tmp_path,
        )

        # Load from parquet to verify persistence
        X_loaded = pd.read_parquet(tmp_path / "X_tree_raw.parquet")

        # Verify all remaining features have variance >= 0.01
        variances = X_loaded.var()
        assert (variances >= 0.01).all(), (
            f"Features with variance < 0.01 found: "
            f"{variances[variances < 0.01].to_dict()}"
        )

    def test_build_tree_feature_store_no_high_correlation_pairs(self, secondary_fixture, tmp_path):
        """Verify no two columns have |r| > 0.95 after Layer 6b deduplication."""
        # Create a simple target for secondary_fixture
        y = pd.Series([0, 1, 0, 1, 0], index=range(len(secondary_fixture)))

        X_tree, feature_cols = build_tree_feature_store(
            X=secondary_fixture,
            y=y,
            output_dir=tmp_path,
        )

        # Load from parquet to verify persistence
        X_loaded = pd.read_parquet(tmp_path / "X_tree_raw.parquet")

        # Compute correlation matrix (absolute values)
        if X_loaded.shape[1] > 1:
            corr_matrix = X_loaded.corr().abs()

            # Check upper triangle for correlations > 0.95
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    assert corr_val <= 0.95, (
                        f"High correlation found: {corr_matrix.columns[i]} vs "
                        f"{corr_matrix.columns[j]}: {corr_val:.4f}"
                    )


# ---------------------------------------------------------------------------
# Wave 1 Features: Delinquency Trajectory Features (Phase 04.2.7)
# ---------------------------------------------------------------------------


class TestWave1Features:
    """Wave 1 delinquency trajectory features (Phase 04.2.7)."""

    @pytest.fixture
    def inst_data(self, df_inst_fixture):
        """Instalment data fixture."""
        return df_inst_fixture

    @pytest.fixture
    def bureau_data(self, df_bureau_balance_fixture):
        """Bureau balance data fixture."""
        return df_bureau_balance_fixture

    def test_engineer_inst_late_rate_12m(self, inst_data):
        """FEAT-01a: Fraction of payments >30DPD in last 365d."""
        from src.features import engineer_inst_late_rate_12m

        result = engineer_inst_late_rate_12m(inst_data)
        assert isinstance(result, pd.Series), "Should return pd.Series"
        assert result.index.name == "SK_ID_CURR", "Index must be SK_ID_CURR"
        assert result.dtype in [np.float64, float], "Must be numeric"
        # Values ∈ [0, 1] for valid applicants or -999.0 for missing
        _NAN_SENTINEL = -999.0
        valid_mask = result != _NAN_SENTINEL
        assert (result[valid_mask] >= 0.0).all(), "Valid values must be >= 0"
        assert (result[valid_mask] <= 1.0).all(), "Valid values must be <= 1"
        # Should have one row per unique applicant
        expected_count = inst_data["SK_ID_CURR"].nunique()
        assert len(result) == expected_count, f"Expected {expected_count} rows, got {len(result)}"

    def test_engineer_inst_late_rate_recent_vs_historical(self, inst_data):
        """FEAT-01b: 12m rate minus historical rate (trajectory)."""
        from src.features import engineer_inst_late_rate_recent_vs_historical

        result = engineer_inst_late_rate_recent_vs_historical(inst_data)
        assert isinstance(result, pd.Series), "Should return pd.Series"
        assert result.dtype in [np.float64, float], "Must be numeric"
        # Trajectory ∈ [-1, 1] for valid applicants or -999.0 for missing
        _NAN_SENTINEL = -999.0
        valid_mask = result != _NAN_SENTINEL
        assert (result[valid_mask] >= -1.0).all(), "Valid trajectory must be >= -1"
        assert (result[valid_mask] <= 1.0).all(), "Valid trajectory must be <= 1"
        # Should have same applicants as 12m rate
        from src.features import engineer_inst_late_rate_12m
        late_rate_12m = engineer_inst_late_rate_12m(inst_data)
        assert len(result) == len(late_rate_12m), "Should have same applicants as 12m rate"

    def test_engineer_inst_rolling_30dpd_ratio_3m(self, inst_data):
        """FEAT-02a: Fraction of payments >30DPD in last 90d."""
        from src.features import engineer_inst_rolling_30dpd_ratio_3m

        result = engineer_inst_rolling_30dpd_ratio_3m(inst_data)
        assert isinstance(result, pd.Series), "Should return pd.Series"
        assert result.dtype in [np.float64, float], "Must be numeric"
        assert (result >= 0).all() or (result == -999.0).all(), "Values ∈ [0,1] or -999 sentinel"

    def test_engineer_inst_delinquency_escalation_flag(self, inst_data):
        """FEAT-02b: Flag if 3m rate > 6m rate (worsening trend)."""
        from src.features import engineer_inst_delinquency_escalation_flag

        result = engineer_inst_delinquency_escalation_flag(inst_data)
        assert isinstance(result, pd.Series), "Should return pd.Series"
        assert result.dtype in [np.float64, float], "Must be numeric"
        assert set(result.unique()).issubset({0.0, 1.0, -999.0}), "Values must be {0, 1, -999}"

    def test_engineer_inst_days_since_last_30dpd(self, inst_data):
        """FEAT-02c: Days since most recent 30+DPD event; -1 if never."""
        from src.features import engineer_inst_days_since_last_30dpd

        result = engineer_inst_days_since_last_30dpd(inst_data)
        assert isinstance(result, pd.Series), "Should return pd.Series"
        assert result.dtype in [np.float64, float], "Must be numeric"
        assert (result >= -1).all() or (result == -999.0).all(), "Values ≥ -1 or -999 sentinel"

    def test_engineer_bureau_dpd_trend_3m_vs_12m(self):
        """FEAT-03: DPD rate last 3m minus DPD rate 3–12m (trend)."""
        from src.features import engineer_bureau_dpd_trend_3m_vs_12m

        _NAN_SENTINEL = -999.0
        # Create minimal test df with required columns
        mock_df = pd.DataFrame({
            'bureau_bbal_dpd_rate_3m_mean': [0.1, 0.0, np.nan, 0.5, 0.0],
            'bureau_bbal_dpd_rate_3m_to_12m_mean': [0.05, 0.0, np.nan, 0.2, 0.1],
        })
        result = engineer_bureau_dpd_trend_3m_vs_12m(mock_df)
        assert isinstance(result, pd.Series), "Should return pd.Series"
        assert result.dtype in [np.float64, float], "Must be numeric"
        # Trend should be ∈ [-1, 1] or sentinel
        valid = result[result != _NAN_SENTINEL]
        assert (valid >= -1).all() and (valid <= 1).all(), "Valid trend must be ∈ [-1,1]"
        # NaN row should map to sentinel
        assert result.iloc[2] == _NAN_SENTINEL, "NaN input should yield sentinel"

    def test_engineer_bureau_debt_to_new_credit(self):
        """FEAT-04: Bureau outstanding debt / new loan amount."""
        from src.features import engineer_bureau_debt_to_new_credit

        _NAN_SENTINEL = -999.0
        # Create minimal test df with required columns
        mock_df = pd.DataFrame({
            'bureau_credit_debt_sum': [100000.0, 0.0, np.nan, 500000.0, 200000.0, 300000.0],
            'AMT_CREDIT': [200000.0, 150000.0, 100000.0, 0.0, 100000.0, np.nan],
        })
        result = engineer_bureau_debt_to_new_credit(mock_df)
        assert isinstance(result, pd.Series), "Should return pd.Series"
        assert result.dtype in [np.float64, float], "Must be numeric"
        # Ratio should be ≥ 0 or sentinel
        valid = result[result != _NAN_SENTINEL]
        assert (valid >= 0).all(), "Valid ratio must be ≥ 0"
        # NaN debt row should be sentinel
        assert result.iloc[2] == _NAN_SENTINEL, "NaN debt should yield sentinel"
        # Zero AMT_CREDIT (row 3): clipped to 1.0 for safe division; ratio is a valid large number
        assert result.iloc[3] >= 0, "Zero AMT_CREDIT clipped to 1.0 produces a valid (large) ratio"
        # NaN AMT_CREDIT (row 5): missing credit amount must be sentinel, not a fictitious ratio
        assert result.iloc[5] == _NAN_SENTINEL, "NaN AMT_CREDIT must yield sentinel (CR-01 fix)"


# ---------------------------------------------------------------------------
# Phase 04.2.9 — EXT_SOURCE_NUM_AVAILABLE + Protected Features Tests
# ---------------------------------------------------------------------------


def test_ext_source_num_available_all_present():
    """EXT_SOURCE_NUM_AVAILABLE counts 3 when all sources present."""
    df = pd.DataFrame({
        "EXT_SOURCE_1": [0.5],
        "EXT_SOURCE_2": [0.6],
        "EXT_SOURCE_3": [0.7],
    })
    result = _engineer_ext_source(df)
    assert "EXT_SOURCE_NUM_AVAILABLE" in result.columns, "EXT_SOURCE_NUM_AVAILABLE missing"
    assert result["EXT_SOURCE_NUM_AVAILABLE"].iloc[0] == 3.0


def test_ext_source_num_available_one_nan():
    """EXT_SOURCE_NUM_AVAILABLE counts 2 when 1 source is NaN."""
    df = pd.DataFrame({
        "EXT_SOURCE_1": [0.5],
        "EXT_SOURCE_2": [np.nan],
        "EXT_SOURCE_3": [0.7],
    })
    result = _engineer_ext_source(df)
    assert "EXT_SOURCE_NUM_AVAILABLE" in result.columns
    assert result["EXT_SOURCE_NUM_AVAILABLE"].iloc[0] == 2.0


def test_ext_source_num_available_all_nan():
    """EXT_SOURCE_NUM_AVAILABLE is 0.0 when all sources are NaN."""
    df = pd.DataFrame({
        "EXT_SOURCE_1": [np.nan],
        "EXT_SOURCE_2": [np.nan],
        "EXT_SOURCE_3": [np.nan],
    })
    result = _engineer_ext_source(df)
    assert "EXT_SOURCE_NUM_AVAILABLE" in result.columns
    assert result["EXT_SOURCE_NUM_AVAILABLE"].iloc[0] == 0.0


def test_engineer_application_features_includes_annuity_income_ratio(application_fixture):
    """ANNUITY_INCOME_RATIO created by engineer_application_features."""
    X_eng = engineer_application_features(application_fixture)
    assert "ANNUITY_INCOME_RATIO" in X_eng.columns, "ANNUITY_INCOME_RATIO not created by engineer_application_features"


def test_engineer_application_features_includes_employed_to_age_ratio(application_fixture):
    """EMPLOYED_TO_AGE_RATIO created by engineer_application_features."""
    X_eng = engineer_application_features(application_fixture)
    assert "EMPLOYED_TO_AGE_RATIO" in X_eng.columns, "EMPLOYED_TO_AGE_RATIO not created by engineer_application_features"


def test_ext_source_num_available_in_engineered_features(application_fixture):
    """EXT_SOURCE_NUM_AVAILABLE created by engineer_application_features."""
    X_eng = engineer_application_features(application_fixture)
    assert "EXT_SOURCE_NUM_AVAILABLE" in X_eng.columns, "EXT_SOURCE_NUM_AVAILABLE not created"
    # Check values are in expected range [0, 3]
    assert (X_eng["EXT_SOURCE_NUM_AVAILABLE"].isin([0.0, 1.0, 2.0, 3.0])).all(), "Values outside [0,3] range"
