"""
test_data_loader.py
-------------------
Unit and integration tests for credit_engine/data_loader.py.

Tests use synthetic CSVs written to a temporary directory so they run
without the real dataset and without network access.

Run with
--------
    pytest tests/test_data_loader.py -v
"""

import numpy as np
import pandas as pd
import pytest

from credit_engine.data_loader import build_training_frame, load_data, save_training_frame

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _write_synthetic_csvs(data_dir):
    """Write minimal synthetic CSVs that mirror real dataset schemas."""

    # application_train — minimal 3 applicants
    app_train = pd.DataFrame(
        {
            "SK_ID_CURR": [100001, 100002, 100003],
            "TARGET": [0, 1, 0],
            "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans", "Cash loans"],
            "CODE_GENDER": ["M", "F", "M"],
            "FLAG_OWN_CAR": ["Y", "N", "Y"],
            "FLAG_OWN_REALTY": ["Y", "Y", "N"],
            "CNT_CHILDREN": [0, 2, 1],
            "AMT_INCOME_TOTAL": [135000.0, 202500.0, 67500.0],
            "AMT_CREDIT": [406597.5, 1293502.5, 135000.0],
            "AMT_ANNUITY": [24700.5, 35698.5, 6750.0],
            "AMT_GOODS_PRICE": [351000.0, 1129500.0, 135000.0],
            "DAYS_BIRTH": [-9461, -16765, -19046],
            "DAYS_EMPLOYED": [-637, -1188, -3039],
            "EXT_SOURCE_1": [0.083, 0.311, np.nan],
            "EXT_SOURCE_2": [0.262, 0.622, 0.555],
            "EXT_SOURCE_3": [0.139, np.nan, 0.729],
        }
    )
    app_train.to_csv(data_dir / "application_train.csv", index=False)

    # application_test — same schema, no TARGET
    app_test = app_train.drop(columns=["TARGET"]).copy()
    app_test.to_csv(data_dir / "application_test.csv", index=False)

    # bureau — 2 entries for SK_ID_CURR=100001, 1 for 100002, 0 for 100003
    bureau = pd.DataFrame(
        {
            "SK_ID_CURR": [100001, 100001, 100002],
            "SK_ID_BUREAU": [200001, 200002, 200003],
            "CREDIT_ACTIVE": ["Active", "Closed", "Active"],
            "CREDIT_CURRENCY": ["currency 1", "currency 1", "currency 1"],
            "DAYS_CREDIT": [-497, -1570, -246],
            "CREDIT_DAY_OVERDUE": [0, 0, 5],
            "DAYS_CREDIT_ENDDATE": [731.0, -153.0, np.nan],
            "DAYS_ENDDATE_FACT": [np.nan, -153.0, np.nan],
            "AMT_CREDIT_MAX_OVERDUE": [np.nan, 0.0, np.nan],
            "CNT_CREDIT_PROLONG": [0, 0, 2],
            "AMT_CREDIT_SUM": [225000.0, 464323.5, 808650.0],
            "AMT_CREDIT_SUM_DEBT": [0.0, np.nan, np.nan],
            "AMT_CREDIT_SUM_LIMIT": [np.nan, np.nan, np.nan],
            "AMT_CREDIT_SUM_OVERDUE": [0.0, 0.0, 0.0],
            "CREDIT_TYPE": ["Consumer credit", "Car loan", "Consumer credit"],
            "DAYS_CREDIT_UPDATE": [-103, -2765, -20],
            "AMT_ANNUITY": [np.nan, np.nan, np.nan],
        }
    )
    bureau.to_csv(data_dir / "bureau.csv", index=False)

    # bureau_balance — monthly history for SK_ID_BUREAU 200001 and 200002
    bureau_balance = pd.DataFrame(
        {
            "SK_ID_BUREAU": [200001, 200001, 200001, 200002, 200002],
            "MONTHS_BALANCE": [-1, -2, -3, -1, -2],
            "STATUS": ["0", "0", "C", "1", "0"],
        }
    )
    bureau_balance.to_csv(data_dir / "bureau_balance.csv", index=False)

    # previous_application — 2 entries for SK_ID_CURR=100001, 1 for 100003
    previous_application = pd.DataFrame(
        {
            "SK_ID_PREV": [300001, 300002, 300003],
            "SK_ID_CURR": [100001, 100001, 100003],
            "NAME_CONTRACT_TYPE": ["Cash loans", "Cash loans", "Revolving loans"],
            "AMT_ANNUITY": [3951.0, 11574.0, np.nan],
            "AMT_APPLICATION": [24835.5, 44946.0, 4500.0],
            "AMT_CREDIT": [20250.0, 56970.0, 4500.0],
            "AMT_DOWN_PAYMENT": [np.nan, np.nan, np.nan],
            "AMT_GOODS_PRICE": [24835.5, 44946.0, 4500.0],
            "WEEKDAY_APPR_PROCESS_START": ["WEDNESDAY", "MONDAY", "TUESDAY"],
            "HOUR_APPR_PROCESS_START": [10, 15, 11],
            "FLAG_LAST_APPL_PER_CONTRACT": ["Y", "Y", "Y"],
            "NFLAG_LAST_APPL_IN_DAY": [1, 1, 1],
            "RATE_DOWN_PAYMENT": [np.nan, np.nan, np.nan],
            "RATE_INTEREST_PRIMARY": [np.nan, np.nan, np.nan],
            "RATE_INTEREST_PRIVILEGED": [np.nan, np.nan, np.nan],
            "NAME_CASH_LOAN_PURPOSE": ["XAP", "XAP", "XAP"],
            "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Approved"],
            "DAYS_DECISION": [-73, -164, -128],
            "NAME_PAYMENT_TYPE": [
                "Cash through the bank",
                "Cash through the bank",
                "XNA",
            ],
            "CODE_REJECT_REASON": ["XAP", "HC", "XAP"],
            "NAME_TYPE_SUITE": [np.nan, np.nan, np.nan],
            "NAME_CLIENT_TYPE": ["Repeater", "Repeater", "New"],
            "NAME_GOODS_CATEGORY": ["XNA", "XNA", "XNA"],
            "NAME_PORTFOLIO": ["Cash", "Cash", "Cards"],
            "NAME_PRODUCT_TYPE": ["walk-in", "walk-in", "x-sell"],
            "CHANNEL_TYPE": ["Country-wide", "Country-wide", "Country-wide"],
            "SELLERPLACE_AREA": [-1, -1, -1],
            "NAME_SELLER_INDUSTRY": ["XNA", "XNA", "XNA"],
            "CNT_PAYMENT": [24.0, 12.0, 12.0],
            "NAME_YIELD_GROUP": ["middle", "middle", "low_action"],
        }
    )
    previous_application.to_csv(data_dir / "previous_application.csv", index=False)

    # POS_CASH_balance — 3 months for SK_ID_PREV=300001
    pos_cash = pd.DataFrame(
        {
            "SK_ID_PREV": [300001, 300001, 300001, 300003],
            "SK_ID_CURR": [100001, 100001, 100001, 100003],
            "MONTHS_BALANCE": [-1, -2, -3, -1],
            "CNT_INSTALMENT": [24.0, 24.0, 24.0, 12.0],
            "CNT_INSTALMENT_FUTURE": [23.0, 22.0, 21.0, 11.0],
            "NAME_CONTRACT_STATUS": ["Active", "Active", "Active", "Active"],
            "SK_DPD": [0, 0, 0, 0],
            "SK_DPD_DEF": [0, 0, 0, 0],
        }
    )
    pos_cash.to_csv(data_dir / "POS_CASH_balance.csv", index=False)

    # installments_payments
    installments = pd.DataFrame(
        {
            "SK_ID_PREV": [300001, 300001, 300002, 300003],
            "SK_ID_CURR": [100001, 100001, 100001, 100003],
            "NUM_INSTALMENT_VERSION": [1, 1, 1, 1],
            "NUM_INSTALMENT_NUMBER": [1, 2, 1, 1],
            "DAYS_INSTALMENT": [-319.0, -289.0, -1141.0, -128.0],
            "DAYS_ENTRY_PAYMENT": [-321.0, -291.0, -1143.0, -130.0],
            "AMT_INSTALMENT": [2160.585, 2160.585, 965.91, 375.0],
            "AMT_PAYMENT": [2160.585, 2160.585, 965.91, 375.0],
        }
    )
    installments.to_csv(data_dir / "installments_payments.csv", index=False)

    # credit_card_balance
    cc_balance = pd.DataFrame(
        {
            "SK_ID_PREV": [300002, 300002],
            "SK_ID_CURR": [100001, 100001],
            "MONTHS_BALANCE": [-1, -2],
            "AMT_BALANCE": [0.0, 0.0],
            "AMT_CREDIT_LIMIT_ACTUAL": [135000, 135000],
            "AMT_DRAWINGS_ATM_CURRENT": [0.0, 0.0],
            "AMT_DRAWINGS_CURRENT": [0.0, 0.0],
            "AMT_DRAWINGS_OTHER_CURRENT": [0.0, 0.0],
            "AMT_DRAWINGS_POS_CURRENT": [0.0, 0.0],
            "AMT_INST_MIN_REGULARITY": [0.0, 0.0],
            "AMT_PAYMENT_CURRENT": [0.0, 0.0],
            "AMT_PAYMENT_TOTAL_CURRENT": [0.0, 0.0],
            "AMT_RECEIVABLE_PRINCIPAL": [0.0, 0.0],
            "AMT_RECIVABLE": [0.0, 0.0],
            "AMT_TOTAL_RECEIVABLE": [0.0, 0.0],
            "CNT_DRAWINGS_ATM_CURRENT": [0.0, 0.0],
            "CNT_DRAWINGS_CURRENT": [0.0, 0.0],
            "CNT_DRAWINGS_OTHER_CURRENT": [0.0, 0.0],
            "CNT_DRAWINGS_POS_CURRENT": [0.0, 0.0],
            "CNT_INSTALMENT_MATURE_CUM": [0.0, 0.0],
            "NAME_CONTRACT_STATUS": ["Active", "Active"],
            "SK_DPD": [0, 0],
            "SK_DPD_DEF": [0, 0],
        }
    )
    cc_balance.to_csv(data_dir / "credit_card_balance.csv", index=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def data_dir(tmp_path):
    """Temporary directory with synthetic CSVs matching real dataset schemas."""
    _write_synthetic_csvs(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Contract tests: return shape and structure
# ---------------------------------------------------------------------------


def test_load_data_returns_dataframe(data_dir):
    result = load_data(data_dir)
    assert isinstance(result, pd.DataFrame)


def test_load_data_one_row_per_sk_id_curr(data_dir):
    """No join should multiply application rows."""
    result = load_data(data_dir)
    assert result["SK_ID_CURR"].nunique() == len(result)


def test_load_data_preserves_all_application_rows(data_dir):
    """Left join must not drop any applicant from the main table."""
    app = pd.read_csv(data_dir / "application_train.csv")
    result = load_data(data_dir, mode="train")
    assert set(result["SK_ID_CURR"]) == set(app["SK_ID_CURR"])


def test_load_data_no_duplicate_columns(data_dir):
    """Column names must be unique — no accidental join clashes."""
    result = load_data(data_dir)
    assert len(result.columns) == result.columns.nunique()


def test_load_data_sk_id_curr_no_nulls(data_dir):
    result = load_data(data_dir)
    assert result["SK_ID_CURR"].notna().all()


def test_load_data_sk_id_curr_is_unique(data_dir):
    result = load_data(data_dir)
    assert result["SK_ID_CURR"].is_unique


# ---------------------------------------------------------------------------
# Contract tests: TARGET column
# ---------------------------------------------------------------------------


def test_load_data_train_target_present(data_dir):
    result = load_data(data_dir, mode="train")
    assert "TARGET" in result.columns


def test_load_data_train_target_binary(data_dir):
    result = load_data(data_dir, mode="train")
    assert set(result["TARGET"].dropna().unique()).issubset({0, 1})


def test_load_data_test_mode_no_target(data_dir):
    """Test CSV has no TARGET; result must either omit it or fill NaN."""
    result = load_data(data_dir, mode="test")
    if "TARGET" in result.columns:
        assert result["TARGET"].isna().all()


# ---------------------------------------------------------------------------
# Contract tests: secondary table aggregates present
# ---------------------------------------------------------------------------


def test_load_data_bureau_aggregates_present(data_dir):
    expected = ["bureau_cnt", "bureau_credit_sum", "bureau_overdue_max"]
    result = load_data(data_dir, mode="train")
    for col in expected:
        assert col in result.columns, f"Missing aggregate column: {col}"


def test_load_data_previous_app_aggregates_present(data_dir):
    expected = ["prev_cnt", "prev_approved_cnt", "prev_amt_credit_sum"]
    result = load_data(data_dir, mode="train")
    for col in expected:
        assert col in result.columns, f"Missing aggregate column: {col}"


def test_load_data_pos_cash_aggregates_present(data_dir):
    expected = ["pos_cnt", "pos_sk_dpd_max"]
    result = load_data(data_dir, mode="train")
    for col in expected:
        assert col in result.columns, f"Missing aggregate column: {col}"


def test_load_data_installments_aggregates_present(data_dir):
    expected = ["inst_cnt", "inst_late_cnt", "inst_amt_payment_sum"]
    result = load_data(data_dir, mode="train")
    for col in expected:
        assert col in result.columns, f"Missing aggregate column: {col}"


def test_load_data_credit_card_aggregates_present(data_dir):
    expected = ["cc_cnt", "cc_bal_max", "cc_sk_dpd_max"]
    result = load_data(data_dir, mode="train")
    for col in expected:
        assert col in result.columns, f"Missing aggregate column: {col}"


# ---------------------------------------------------------------------------
# Contract tests: join correctness (verify aggregate values)
# ---------------------------------------------------------------------------


def test_bureau_count_correct(data_dir):
    """SK_ID_CURR=100001 has 2 bureau rows; 100002 has 1; 100003 has 0."""
    result = load_data(data_dir, mode="train")
    result = result.set_index("SK_ID_CURR")
    assert result.loc[100001, "bureau_cnt"] == 2
    assert result.loc[100002, "bureau_cnt"] == 1
    assert result.loc[100003, "bureau_cnt"] == 0


def test_previous_app_count_correct(data_dir):
    """SK_ID_CURR=100001 has 2 previous apps; 100003 has 1; 100002 has 0."""
    result = load_data(data_dir, mode="train").set_index("SK_ID_CURR")
    assert result.loc[100001, "prev_cnt"] == 2
    assert result.loc[100003, "prev_cnt"] == 1
    assert result.loc[100002, "prev_cnt"] == 0


def test_prev_approved_count_correct(data_dir):
    """SK_ID_CURR=100001: 1 Approved + 1 Refused."""
    result = load_data(data_dir, mode="train").set_index("SK_ID_CURR")
    assert result.loc[100001, "prev_approved_cnt"] == 1


def test_inst_late_count_correct(data_dir):
    """All synthetic payments are on-time (DAYS_ENTRY_PAYMENT <= DAYS_INSTALMENT)."""
    result = load_data(data_dir, mode="train").set_index("SK_ID_CURR")
    # payments are 2 days early for SK_ID_CURR=100001
    assert result.loc[100001, "inst_late_cnt"] == 0


# ---------------------------------------------------------------------------
# Contract tests: applicant with no secondary history → NaN/0
# ---------------------------------------------------------------------------


def test_applicant_with_no_bureau_history(data_dir):
    """SK_ID_CURR=100003 has no bureau rows; bureau_cnt must be 0."""
    result = load_data(data_dir, mode="train").set_index("SK_ID_CURR")
    assert result.loc[100003, "bureau_cnt"] == 0


def test_applicant_with_no_cc_history(data_dir):
    """SK_ID_CURR=100002 has no CC rows; cc_cnt must be 0."""
    result = load_data(data_dir, mode="train").set_index("SK_ID_CURR")
    assert result.loc[100002, "cc_cnt"] == 0


# ---------------------------------------------------------------------------
# Contract tests: immutability / determinism
# ---------------------------------------------------------------------------


def test_load_data_deterministic(data_dir):
    """Calling load_data twice produces identical DataFrames."""
    result1 = load_data(data_dir)
    result2 = load_data(data_dir)
    pd.testing.assert_frame_equal(
        result1.reset_index(drop=True), result2.reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Contract tests: error handling
# ---------------------------------------------------------------------------


def test_missing_csv_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path)  # empty directory → no CSVs


def test_invalid_mode_raises_value_error(data_dir):
    with pytest.raises(ValueError, match="mode"):
        load_data(data_dir, mode="validation")


# ---------------------------------------------------------------------------
# Contract tests: build_training_frame
# ---------------------------------------------------------------------------


def test_build_training_frame_returns_tuple(data_dir):
    """build_training_frame returns a (DataFrame, Series) pair."""
    X, y = build_training_frame(data_dir)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


def test_build_training_frame_target_excluded_from_X(data_dir):
    """TARGET must not appear in the feature matrix."""
    X, _ = build_training_frame(data_dir)
    assert "TARGET" not in X.columns


def test_build_training_frame_y_is_binary(data_dir):
    """Target series contains only 0 and 1."""
    _, y = build_training_frame(data_dir)
    assert set(y.unique()).issubset({0, 1})


def test_build_training_frame_aligned_index(data_dir):
    """X and y share the same index."""
    X, y = build_training_frame(data_dir)
    pd.testing.assert_index_equal(X.index, y.index)


def test_build_training_frame_no_numeric_nans(data_dir):
    """All numeric columns in X have no NaN values after sentinel fill."""
    X, _ = build_training_frame(data_dir)
    numeric_cols = X.select_dtypes(include="number").columns
    assert X[numeric_cols].isna().sum().sum() == 0


def test_build_training_frame_sentinel_value_present(data_dir):
    """Columns that had NaNs should contain the -999 sentinel."""
    X, _ = build_training_frame(data_dir)
    # EXT_SOURCE_1 has a NaN in the synthetic data
    if "EXT_SOURCE_1" in X.columns:
        assert (X["EXT_SOURCE_1"] == -999).any()


def test_build_training_frame_drops_high_missingness_columns(data_dir, tmp_path):
    """Columns with > 60 % missing values are dropped."""
    import shutil

    # Copy synthetic CSVs to a new tmp dir and inject a >60%-missing column
    new_dir = tmp_path / "data_drop_test"
    shutil.copytree(data_dir, new_dir)

    app_path = new_dir / "application_train.csv"
    app = pd.read_csv(app_path)
    # Add column that is 100 % NaN — must be dropped
    app["MOSTLY_MISSING"] = np.nan
    app.to_csv(app_path, index=False)

    X, _ = build_training_frame(new_dir)
    assert "MOSTLY_MISSING" not in X.columns


# ---------------------------------------------------------------------------
# Contract tests: save_training_frame
# ---------------------------------------------------------------------------


def test_save_training_frame_creates_parquet_files(data_dir, tmp_path):
    """save_training_frame writes X_train.parquet and y_train.parquet."""
    X, y = build_training_frame(data_dir)
    out_dir = tmp_path / "processed"
    save_training_frame(X, y, out_dir)

    assert (out_dir / "X_train.parquet").exists()
    assert (out_dir / "y_train.parquet").exists()


def test_save_training_frame_roundtrip(data_dir, tmp_path):
    """Data survives a parquet write-read round-trip."""
    X, y = build_training_frame(data_dir)
    out_dir = tmp_path / "processed"
    save_training_frame(X, y, out_dir)

    X_loaded = pd.read_parquet(out_dir / "X_train.parquet")
    y_loaded = pd.read_parquet(out_dir / "y_train.parquet")["TARGET"]

    pd.testing.assert_frame_equal(X, X_loaded)
    pd.testing.assert_series_equal(y, y_loaded)
