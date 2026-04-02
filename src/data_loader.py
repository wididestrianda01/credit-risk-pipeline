"""
data_loader.py
--------------
Joins the 7 raw source tables into a single modelling DataFrame
and enforces correct dtypes throughout.

Tables
------
1. application_train / application_test  – loan application details (main)
2. bureau                                – credit bureau summary per applicant
3. bureau_balance                        – monthly bureau balance history
4. previous_application                  – prior application history
5. POS_CASH_balance                      – POS / cash loan monthly snapshots
6. installments_payments                 – instalment payment history
7. credit_card_balance                   – credit card balance snapshots

Architecture
------------
This module is responsible for:
  - Loading all 7 CSV files
  - Aggregating secondary tables (many-per-applicant) to one row per SK_ID_CURR
  - Enforcing dtypes (categorical, int64, float64)
  - Returning a single clean DataFrame ready for features.py

It does NOT do domain-rich feature engineering — that belongs in features.py.
Aggregates produced here are structural (count, sum, max, mean of raw columns)
to collapse N:M relationships. features.py adds 150+ derived features on top.

Usage
-----
    from credit_engine.data_loader import load_data, build_training_frame, save_training_frame

    # Low-level: raw joined DataFrame
    df = load_data(data_dir="dataset/", mode="train")

    # High-level: clean (X, y) pair ready for feature engineering
    X, y = build_training_frame(data_dir="dataset/")
    save_training_frame(X, y, output_dir="data/processed/")
"""

import math
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# File name constants
# ---------------------------------------------------------------------------

_FILE_APP_TRAIN = "application_train.csv"
_FILE_APP_TEST = "application_test.csv"
_FILE_BUREAU = "bureau.csv"
_FILE_BUREAU_BAL = "bureau_balance.csv"
_FILE_PREV_APP = "previous_application.csv"
_FILE_POS_CASH = "POS_CASH_balance.csv"
_FILE_INSTALLMENTS = "installments_payments.csv"
_FILE_CC_BAL = "credit_card_balance.csv"

_SECONDARY_FILES = [
    _FILE_BUREAU,
    _FILE_BUREAU_BAL,
    _FILE_PREV_APP,
    _FILE_POS_CASH,
    _FILE_INSTALLMENTS,
    _FILE_CC_BAL,
]

# ---------------------------------------------------------------------------
# Categorical columns in the application table
# ---------------------------------------------------------------------------

_CATEGORICAL_APP_COLS = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "WEEKDAY_APPR_PROCESS_START",
    "ORGANIZATION_TYPE",
    "FONDKAPREMONT_MODE",
    "HOUSETYPE_MODE",
    "WALLSMATERIAL_MODE",
    "EMERGENCYSTATE_MODE",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MISSING_DROP_THRESHOLD = 0.60  # drop columns with > 60% missing values
_NAN_SENTINEL = -999  # sentinel fill for tree models (LightGBM/XGBoost)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_data(data_dir: str | Path, mode: str = "train") -> pd.DataFrame:
    """Join all 7 source tables into a single modelling DataFrame.

    Returns one row per SK_ID_CURR. Secondary tables (bureau, previous
    applications, etc.) are aggregated to SK_ID_CURR level before joining
    so the result has no row duplication.

    Parameters
    ----------
    data_dir : str or Path
        Directory that contains all 7 CSV files.
    mode : str, default "train"
        ``"train"`` loads ``application_train.csv`` (includes TARGET column).
        ``"test"`` loads ``application_test.csv`` (no TARGET column).

    Returns
    -------
    pd.DataFrame
        Shape (n_applicants, ~160). One row per SK_ID_CURR.
        Categorical columns cast to ``category`` dtype.
        All secondary-table columns prefixed by source:
        ``bureau_``, ``prev_``, ``pos_``, ``inst_``, ``cc_``.

    Raises
    ------
    FileNotFoundError
        If any required CSV file is missing from ``data_dir``.
    ValueError
        If ``mode`` is not ``"train"`` or ``"test"``.
    """
    data_dir = Path(data_dir)

    if mode not in {"train", "test"}:
        raise ValueError(f"mode must be 'train' or 'test', got {mode!r}")

    _validate_files(data_dir, mode)

    app = _load_application(data_dir, mode)
    app = _join_bureau(app, data_dir)
    app = _join_previous_application(app, data_dir)
    app = _join_pos_cash(app, data_dir)
    app = _join_installments(app, data_dir)
    app = _join_credit_card(app, data_dir)
    app = _enforce_dtypes(app)

    return app


def build_training_frame(
    data_dir: str | Path,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build the flat training matrix from all 7 source tables.

    Calls ``load_data()``, drops high-missingness columns, fills remaining
    numeric NaNs with a sentinel value, and returns a clean ``(X, y)`` pair
    ready for ``features.py``.

    Steps
    -----
    1. Join all 7 tables via ``load_data()``.
    2. Drop columns where > 60 % of values are missing.
    3. Fill remaining numeric NaNs with ``-999`` (sentinel; LightGBM/XGBoost
       treat it as a separate bin rather than imputing a distribution value).
    4. Split TARGET out of the feature matrix.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing the 7 raw CSV files.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix — one row per applicant, TARGET excluded.
    y : pd.Series
        Binary target (0 = repaid, 1 = defaulted), aligned with X.

    Raises
    ------
    KeyError
        If TARGET column is absent (e.g. called with test-mode data).
    """
    data_dir = Path(data_dir)
    df = load_data(data_dir, mode="train")

    if "TARGET" not in df.columns:
        raise KeyError(
            "TARGET column not found. build_training_frame() requires train data."
        )

    # --- drop columns with > 60 % missing -----------------------------------
    n_rows = len(df)
    min_non_null = math.ceil((1.0 - _MISSING_DROP_THRESHOLD) * n_rows)
    df = df.dropna(axis=1, thresh=min_non_null)

    # --- fill remaining numeric NaNs with sentinel --------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    fill_map = {col: _NAN_SENTINEL for col in numeric_cols if df[col].isna().any()}
    if fill_map:
        df = df.fillna(fill_map)

    # --- split target --------------------------------------------------------
    y = df["TARGET"].astype(np.int8)
    X = df.drop(columns=["TARGET"])

    return X, y


def save_training_frame(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str | Path,
) -> None:
    """Persist the training matrix and target to parquet.

    Creates ``output_dir`` if it does not exist.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix returned by ``build_training_frame()``.
    y : pd.Series
        Target series returned by ``build_training_frame()``.
    output_dir : str or Path
        Directory to write ``X_train.parquet`` and ``y_train.parquet``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X.to_parquet(output_dir / "X_train.parquet", index=True)
    y.to_frame().to_parquet(output_dir / "y_train.parquet", index=True)


# ---------------------------------------------------------------------------
# Private helpers: validation
# ---------------------------------------------------------------------------


def _validate_files(data_dir: Path, mode: str) -> None:
    """Raise FileNotFoundError if any required CSV is missing.

    Parameters
    ----------
    data_dir : Path
        Directory to search.
    mode : str
        ``"train"`` or ``"test"``.

    Raises
    ------
    FileNotFoundError
        If a required file is absent.
    """
    app_file = _FILE_APP_TRAIN if mode == "train" else _FILE_APP_TEST
    required = [app_file] + _SECONDARY_FILES
    missing = [f for f in required if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Required CSV files not found in {data_dir}: {missing}"
        )


# ---------------------------------------------------------------------------
# Private helpers: application table
# ---------------------------------------------------------------------------


def _load_application(data_dir: Path, mode: str) -> pd.DataFrame:
    """Load the application table and validate its primary key.

    Parameters
    ----------
    data_dir : Path
        Directory containing the CSV files.
    mode : str
        ``"train"`` or ``"test"``.

    Returns
    -------
    pd.DataFrame
        Application table with SK_ID_CURR as unique primary key.

    Raises
    ------
    ValueError
        If SK_ID_CURR contains duplicates or null values.
    """
    filename = _FILE_APP_TRAIN if mode == "train" else _FILE_APP_TEST
    df = pd.read_csv(data_dir / filename)

    if df["SK_ID_CURR"].isna().any():
        raise ValueError("SK_ID_CURR contains null values in application table")
    if df["SK_ID_CURR"].duplicated().any():
        raise ValueError("SK_ID_CURR is not unique in application table")

    return df


# ---------------------------------------------------------------------------
# Private helpers: bureau
# ---------------------------------------------------------------------------


def _aggregate_bureau_balance(data_dir: Path) -> pd.DataFrame:
    """Aggregate bureau_balance to SK_ID_BUREAU level.

    STATUS codes: C = closed, X = unknown, 0 = no DPD, 1-5 = DPD buckets.
    DPD status is any STATUS not in {'C', 'X', '0'}.

    Parameters
    ----------
    data_dir : Path

    Returns
    -------
    pd.DataFrame
        Columns: SK_ID_BUREAU, bbal_cnt, bbal_dpd_rate.
    """
    bbal = pd.read_csv(data_dir / _FILE_BUREAU_BAL)

    bbal = bbal.assign(
        is_dpd=bbal["STATUS"].isin({"1", "2", "3", "4", "5"}).astype(np.float32)
    )

    agg = (
        bbal.groupby("SK_ID_BUREAU", sort=False)
        .agg(
            bbal_cnt=("MONTHS_BALANCE", "count"),
            bbal_dpd_rate=("is_dpd", "mean"),
        )
        .reset_index()
    )
    return agg


def _join_bureau(app: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Aggregate bureau + bureau_balance to SK_ID_CURR and left-join onto app.

    Aggregate columns added (prefix ``bureau_``):

    - ``bureau_cnt``            : number of bureau entries
    - ``bureau_active_cnt``     : entries with CREDIT_ACTIVE == 'Active'
    - ``bureau_days_credit_mean``: mean days since credit opened
    - ``bureau_days_credit_min`` : most recent bureau credit opened
    - ``bureau_credit_sum``     : total credit amount across bureau entries
    - ``bureau_credit_debt_sum``: total outstanding debt
    - ``bureau_overdue_max``    : maximum days overdue across bureau entries
    - ``bureau_prolong_sum``    : total number of credit prolongations
    - ``bureau_bbal_cnt_mean``  : mean monthly balance record count
    - ``bureau_bbal_dpd_rate_mean``: mean DPD rate across bureau entries

    Parameters
    ----------
    app : pd.DataFrame
        Application table (one row per SK_ID_CURR).
    data_dir : Path

    Returns
    -------
    pd.DataFrame
        ``app`` with bureau aggregate columns appended.
    """
    bureau = pd.read_csv(data_dir / _FILE_BUREAU)
    bbal_agg = _aggregate_bureau_balance(data_dir)

    # Enrich bureau with balance aggregates (left join on SK_ID_BUREAU)
    bureau = bureau.merge(bbal_agg, on="SK_ID_BUREAU", how="left")

    bureau = bureau.assign(
        is_active=(bureau["CREDIT_ACTIVE"] == "Active").astype(np.float32)
    )

    bureau_agg = (
        bureau.groupby("SK_ID_CURR", sort=False)
        .agg(
            bureau_cnt=("SK_ID_BUREAU", "count"),
            bureau_active_cnt=("is_active", "sum"),
            bureau_days_credit_mean=("DAYS_CREDIT", "mean"),
            bureau_days_credit_min=("DAYS_CREDIT", "min"),
            bureau_credit_sum=("AMT_CREDIT_SUM", "sum"),
            bureau_credit_debt_sum=("AMT_CREDIT_SUM_DEBT", "sum"),
            bureau_overdue_max=("CREDIT_DAY_OVERDUE", "max"),
            bureau_prolong_sum=("CNT_CREDIT_PROLONG", "sum"),
            bureau_bbal_cnt_mean=("bbal_cnt", "mean"),
            bureau_bbal_dpd_rate_mean=("bbal_dpd_rate", "mean"),
        )
        .reset_index()
    )

    result = app.merge(bureau_agg, on="SK_ID_CURR", how="left")

    # Applicants with no bureau history → count = 0, not NaN
    result["bureau_cnt"] = result["bureau_cnt"].fillna(0).astype(np.int32)
    result["bureau_active_cnt"] = result["bureau_active_cnt"].fillna(0).astype(np.int32)
    result["bureau_prolong_sum"] = (
        result["bureau_prolong_sum"].fillna(0).astype(np.int32)
    )

    _assert_no_row_multiplication(app, result, "bureau join")
    return result


# ---------------------------------------------------------------------------
# Private helpers: previous application
# ---------------------------------------------------------------------------


def _join_previous_application(app: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Aggregate previous_application to SK_ID_CURR and left-join onto app.

    Aggregate columns added (prefix ``prev_``):

    - ``prev_cnt``               : total previous applications
    - ``prev_approved_cnt``      : applications with status 'Approved'
    - ``prev_refused_cnt``       : applications with status 'Refused'
    - ``prev_amt_credit_mean``   : mean approved credit amount
    - ``prev_amt_credit_sum``    : total approved credit amount
    - ``prev_days_decision_min`` : most recent decision (least negative DAYS)
    - ``prev_rate_down_payment_mean``: mean down payment rate

    Parameters
    ----------
    app : pd.DataFrame
    data_dir : Path

    Returns
    -------
    pd.DataFrame
    """
    prev = pd.read_csv(data_dir / _FILE_PREV_APP)

    prev = prev.assign(
        is_approved=(prev["NAME_CONTRACT_STATUS"] == "Approved").astype(np.float32),
        is_refused=(prev["NAME_CONTRACT_STATUS"] == "Refused").astype(np.float32),
    )

    prev_agg = (
        prev.groupby("SK_ID_CURR", sort=False)
        .agg(
            prev_cnt=("SK_ID_PREV", "count"),
            prev_approved_cnt=("is_approved", "sum"),
            prev_refused_cnt=("is_refused", "sum"),
            prev_amt_credit_mean=("AMT_CREDIT", "mean"),
            prev_amt_credit_sum=("AMT_CREDIT", "sum"),
            prev_days_decision_min=("DAYS_DECISION", "min"),
            prev_rate_down_payment_mean=("RATE_DOWN_PAYMENT", "mean"),
        )
        .reset_index()
    )

    result = app.merge(prev_agg, on="SK_ID_CURR", how="left")

    result["prev_cnt"] = result["prev_cnt"].fillna(0).astype(np.int32)
    result["prev_approved_cnt"] = result["prev_approved_cnt"].fillna(0).astype(np.int32)
    result["prev_refused_cnt"] = result["prev_refused_cnt"].fillna(0).astype(np.int32)

    _assert_no_row_multiplication(app, result, "previous_application join")
    return result


# ---------------------------------------------------------------------------
# Private helpers: POS CASH balance
# ---------------------------------------------------------------------------


def _join_pos_cash(app: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Aggregate POS_CASH_balance to SK_ID_CURR and left-join onto app.

    Aggregate columns added (prefix ``pos_``):

    - ``pos_cnt``                  : total monthly POS/cash snapshots
    - ``pos_months_balance_mean``  : mean MONTHS_BALANCE
    - ``pos_cnt_instalment_mean``  : mean instalment count
    - ``pos_sk_dpd_max``           : max days past due
    - ``pos_sk_dpd_def_max``       : max default days past due

    Parameters
    ----------
    app : pd.DataFrame
    data_dir : Path

    Returns
    -------
    pd.DataFrame
    """
    pos = pd.read_csv(data_dir / _FILE_POS_CASH)

    pos_agg = (
        pos.groupby("SK_ID_CURR", sort=False)
        .agg(
            pos_cnt=("MONTHS_BALANCE", "count"),
            pos_months_balance_mean=("MONTHS_BALANCE", "mean"),
            pos_cnt_instalment_mean=("CNT_INSTALMENT", "mean"),
            pos_sk_dpd_max=("SK_DPD", "max"),
            pos_sk_dpd_def_max=("SK_DPD_DEF", "max"),
        )
        .reset_index()
    )

    result = app.merge(pos_agg, on="SK_ID_CURR", how="left")

    result["pos_cnt"] = result["pos_cnt"].fillna(0).astype(np.int32)

    _assert_no_row_multiplication(app, result, "POS_CASH join")
    return result


# ---------------------------------------------------------------------------
# Private helpers: installments payments
# ---------------------------------------------------------------------------


def _join_installments(app: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Aggregate installments_payments to SK_ID_CURR and left-join onto app.

    A payment is late when DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT
    (both are negative; less negative = more recent / later).

    Aggregate columns added (prefix ``inst_``):

    - ``inst_cnt``               : total instalment payment records
    - ``inst_late_cnt``          : count of late payments
    - ``inst_amt_payment_sum``   : total amount paid
    - ``inst_payment_ratio_mean``: mean (AMT_PAYMENT / AMT_INSTALMENT)
    - ``inst_days_past_due_mean``: mean days past due (positive = late)

    Parameters
    ----------
    app : pd.DataFrame
    data_dir : Path

    Returns
    -------
    pd.DataFrame
    """
    inst = pd.read_csv(data_dir / _FILE_INSTALLMENTS)

    # Late flag: payment date is later (less negative) than scheduled date
    inst = inst.assign(
        is_late=(inst["DAYS_ENTRY_PAYMENT"] > inst["DAYS_INSTALMENT"]).astype(
            np.float32
        ),
        payment_ratio=np.where(
            inst["AMT_INSTALMENT"] != 0,
            inst["AMT_PAYMENT"] / inst["AMT_INSTALMENT"],
            np.nan,
        ),
        days_past_due=inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"],
    )

    inst_agg = (
        inst.groupby("SK_ID_CURR", sort=False)
        .agg(
            inst_cnt=("DAYS_INSTALMENT", "count"),
            inst_late_cnt=("is_late", "sum"),
            inst_amt_payment_sum=("AMT_PAYMENT", "sum"),
            inst_payment_ratio_mean=("payment_ratio", "mean"),
            inst_days_past_due_mean=("days_past_due", "mean"),
        )
        .reset_index()
    )

    result = app.merge(inst_agg, on="SK_ID_CURR", how="left")

    result["inst_cnt"] = result["inst_cnt"].fillna(0).astype(np.int32)
    result["inst_late_cnt"] = result["inst_late_cnt"].fillna(0).astype(np.int32)

    _assert_no_row_multiplication(app, result, "installments join")
    return result


# ---------------------------------------------------------------------------
# Private helpers: credit card balance
# ---------------------------------------------------------------------------


def _join_credit_card(app: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Aggregate credit_card_balance to SK_ID_CURR and left-join onto app.

    Utilisation = AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL.
    Division by zero → NaN (handled via np.where).

    Aggregate columns added (prefix ``cc_``):

    - ``cc_cnt``             : total monthly credit card snapshots
    - ``cc_bal_mean``        : mean balance
    - ``cc_bal_max``         : maximum balance ever
    - ``cc_drawing_mean``    : mean drawings amount
    - ``cc_utilization_mean``: mean credit utilisation ratio
    - ``cc_sk_dpd_max``      : maximum days past due

    Parameters
    ----------
    app : pd.DataFrame
    data_dir : Path

    Returns
    -------
    pd.DataFrame
    """
    cc = pd.read_csv(data_dir / _FILE_CC_BAL)

    cc = cc.assign(
        utilization=np.where(
            cc["AMT_CREDIT_LIMIT_ACTUAL"] > 0,
            cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"],
            np.nan,
        )
    )

    cc_agg = (
        cc.groupby("SK_ID_CURR", sort=False)
        .agg(
            cc_cnt=("MONTHS_BALANCE", "count"),
            cc_bal_mean=("AMT_BALANCE", "mean"),
            cc_bal_max=("AMT_BALANCE", "max"),
            cc_drawing_mean=("AMT_DRAWINGS_CURRENT", "mean"),
            cc_utilization_mean=("utilization", "mean"),
            cc_sk_dpd_max=("SK_DPD", "max"),
        )
        .reset_index()
    )

    result = app.merge(cc_agg, on="SK_ID_CURR", how="left")

    result["cc_cnt"] = result["cc_cnt"].fillna(0).astype(np.int32)

    _assert_no_row_multiplication(app, result, "credit_card join")
    return result


# ---------------------------------------------------------------------------
# Private helpers: dtype enforcement
# ---------------------------------------------------------------------------


def _enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast known categorical columns to ``category`` dtype.

    Only casts columns that are present in the DataFrame (tolerant of
    mode='test' where some columns may differ).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        New DataFrame with categorical columns cast.
    """
    cast_map = {col: "category" for col in _CATEGORICAL_APP_COLS if col in df.columns}
    if not cast_map:
        return df
    return df.astype(cast_map)


# ---------------------------------------------------------------------------
# Private helpers: cardinality guard
# ---------------------------------------------------------------------------


def _assert_no_row_multiplication(
    before: pd.DataFrame,
    after: pd.DataFrame,
    join_label: str,
) -> None:
    """Raise ValueError if a join multiplied the number of rows.

    Parameters
    ----------
    before : pd.DataFrame
        DataFrame before the join.
    after : pd.DataFrame
        DataFrame after the join.
    join_label : str
        Human-readable label for error messages.

    Raises
    ------
    ValueError
        If ``len(after) != len(before)``.
    """
    if len(after) != len(before):
        raise ValueError(
            f"{join_label} changed row count: {len(before)} → {len(after)}. "
            "Ensure secondary table is aggregated to one row per SK_ID_CURR before joining."
        )
