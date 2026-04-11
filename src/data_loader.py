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
    from src.data_loader import load_data, build_training_frame, save_training_frame

    # Low-level: raw joined DataFrame
    df = load_data(data_dir="data/", mode="train")

    # High-level: clean (X, y) pair ready for feature engineering
    X, y = build_training_frame(data_dir="data/")
    save_training_frame(X, y, output_dir="data/processed/")
"""

import math
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.features import _PROJECT_ROOT

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
_SECONDARY_COL_PREFIXES = (  # prefixes of secondary-table columns to sentinel-fill
    "bureau_", "bbal_", "prev_", "pos_", "inst_", "cc_"
)


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
    2. Add binary presence flags for secondary tables (absence itself is predictive).
    3. Fill secondary table NaN columns with sentinel before missingness filter.
    4. Drop columns where > 60 % of values are missing.
    5. Fill remaining numeric NaNs with ``-999`` (sentinel; LightGBM/XGBoost
       treat it as a separate bin rather than imputing a distribution value).
    6. Split TARGET out of the feature matrix.

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

    # --- add binary presence flags (absence is itself predictive) -----------
    # Credit card presence flag
    df = df.assign(cc_has_records=(df["cc_cnt"] > 0).astype(np.int8))

    # Bureau balance presence flag (derived from bureau_bbal_cnt_mean)
    if "bureau_bbal_cnt_mean" in df.columns:
        df = df.assign(
            bureau_has_bbal=(df["bureau_bbal_cnt_mean"] > 0).astype(np.int8)
        )
    else:
        # Fallback if the column doesn't exist (shouldn't happen)
        df = df.assign(bureau_has_bbal=np.int8(0))

    # --- fill secondary table NaN columns with sentinel BEFORE drop ---------
    # Applicants with no records in a secondary table have NaN for all that
    # table's aggregate columns.  Fill with sentinel so these columns survive
    # the 60% missingness filter — absence itself is predictive signal.
    _COUNT_FLAG_COLS = frozenset(
        [
            "bureau_cnt", "bureau_active_cnt", "bureau_closed_cnt",
            "bureau_overdue_cnt", "bureau_prolong_sum",
            "prev_cnt", "prev_approved_cnt", "prev_refused_cnt", "prev_cancelled_cnt",
            "pos_cnt", "pos_overdue_cnt",
            "inst_cnt", "inst_late_cnt",
            "cc_cnt",
            "cc_has_records", "bureau_has_bbal",
        ]
    )
    for col in df.columns:
        if any(col.startswith(p) for p in _SECONDARY_COL_PREFIXES):
            if col not in _COUNT_FLAG_COLS:
                if df[col].isna().any():
                    df = df.assign(**{col: df[col].fillna(_NAN_SENTINEL)})

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
    output_dir: str | Path | None = None,
) -> None:
    """Persist the training matrix and target to parquet.

    Creates ``output_dir`` if it does not exist.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix returned by ``build_training_frame()``.
    y : pd.Series
        Target series returned by ``build_training_frame()``.
    output_dir : str | Path, optional
        Directory to write ``X_train.parquet`` and ``y_train.parquet``.
        If None, defaults to {_PROJECT_ROOT}/data/processed/.
    """
    if output_dir is None:
        output_dir = _PROJECT_ROOT / "data" / "processed"
    else:
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


def _aggregate_bureau_balance(bbal: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bureau_balance to SK_ID_BUREAU level with time-windowed features.

    STATUS codes: C = closed, X = unknown, 0 = no DPD, 1-5 = DPD buckets.
    DPD status is any STATUS not in {'C', 'X', '0'}.
    MONTHS_BALANCE: 0 = most recent, negative integers = older months.

    Time windows:
    - Recent (last 6 months): MONTHS_BALANCE >= -6
    - Historical (months 7-12): MONTHS_BALANCE in [-12, -7]

    Parameters
    ----------
    bbal : pd.DataFrame
        Pre-loaded bureau_balance DataFrame. Caller is responsible for loading.

    Returns
    -------
    pd.DataFrame
        Columns: SK_ID_BUREAU, bbal_cnt, bbal_dpd_rate, bbal_dpd_rate_6m,
        bbal_dpd_rate_12m, bbal_dpd_trend, bbal_max_severity,
        bbal_recent_max_severity, bbal_active_months.
    """
    # bbal pre-loaded by caller — do NOT read CSV here

    # DPD indicator: STATUS in {'1', '2', '3', '4', '5'}
    bbal = bbal.assign(
        is_dpd=bbal["STATUS"].isin({"1", "2", "3", "4", "5"}).astype(np.float32)
    )

    # Status severity: map status codes to numeric severity (0-5)
    # C, X, 0 → 0; '1' → 1, '2' → 2, etc.
    status_map = {"C": 0, "X": 0, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
    bbal = bbal.assign(
        status_severity=bbal["STATUS"].map(status_map).astype(np.float32)
    )

    # Active indicator: STATUS not in {'C', 'X'} (loan was active)
    bbal = bbal.assign(
        is_active=(~bbal["STATUS"].isin({"C", "X"})).astype(np.float32)
    )

    # Time window flags
    bbal = bbal.assign(
        in_recent_3m=(bbal["MONTHS_BALANCE"] >= -3).astype(bool),
        in_recent_6m=(bbal["MONTHS_BALANCE"] >= -6).astype(bool),
        in_historical_12m=(bbal["MONTHS_BALANCE"].between(-12, -7)).astype(bool),
        in_3m_to_12m=(bbal["MONTHS_BALANCE"].between(-12, -3)).astype(bool),
    )

    # Helper: compute windowed mean for a series
    def _windowed_mean(series, mask_col_name, bbal_df):
        """Compute mean of series for rows where mask_col is True."""
        mask = bbal_df.loc[series.index, mask_col_name]
        result = series[mask].mean()
        return result

    def _windowed_max(series, mask_col_name, bbal_df):
        """Compute max of series for rows where mask_col is True."""
        mask = bbal_df.loc[series.index, mask_col_name]
        result = series[mask].max()
        return result

    # Aggregate to SK_ID_BUREAU level
    # Use transform-style aggregation to preserve group context for windowed computations
    groups = bbal.groupby("SK_ID_BUREAU", sort=False)

    agg_dict = {
        "bbal_cnt": ("MONTHS_BALANCE", "count"),
        "bbal_dpd_rate": ("is_dpd", "mean"),
        "bbal_dpd_rate_std": ("is_dpd", "std"),
        "bbal_max_severity": ("status_severity", "max"),
        "bbal_active_months": ("is_active", "sum"),
    }

    agg = groups.agg(**agg_dict).reset_index()

    # Compute windowed metrics separately for each group
    windowed_data = []
    for bureau_id, group in groups:
        dpd_3m = group.loc[group["in_recent_3m"], "is_dpd"].mean()
        dpd_6m = group.loc[group["in_recent_6m"], "is_dpd"].mean()
        dpd_12m = group.loc[group["in_historical_12m"], "is_dpd"].mean()
        dpd_3m_to_12m = group.loc[group["in_3m_to_12m"], "is_dpd"].mean()
        severity_recent = group.loc[group["in_recent_6m"], "status_severity"].max()

        windowed_data.append({
            "SK_ID_BUREAU": bureau_id,
            "bbal_dpd_rate_3m": dpd_3m,
            "bbal_dpd_rate_6m": dpd_6m,
            "bbal_dpd_rate_12m": dpd_12m,
            "bbal_dpd_rate_3m_to_12m": dpd_3m_to_12m,
            "bbal_recent_max_severity": severity_recent,
        })

    windowed_df = pd.DataFrame(windowed_data)
    agg = agg.merge(windowed_df, on="SK_ID_BUREAU", how="left")

    # Compute trend: dpd_rate_6m - dpd_rate_12m (positive = worsening)
    agg = agg.assign(
        bbal_dpd_trend=agg["bbal_dpd_rate_6m"] - agg["bbal_dpd_rate_12m"]
    )

    return agg


def _join_bureau(app: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Aggregate bureau + bureau_balance to SK_ID_CURR and left-join onto app.

    Aggregate columns added (prefix ``bureau_``):

    - ``bureau_cnt``                   : number of bureau entries
    - ``bureau_active_cnt``            : entries with CREDIT_ACTIVE == 'Active'
    - ``bureau_days_credit_mean``      : mean days since credit opened
    - ``bureau_days_credit_min``       : most recent bureau credit opened
    - ``bureau_credit_sum``            : total credit amount across bureau entries
    - ``bureau_credit_debt_sum``       : total outstanding debt
    - ``bureau_overdue_max``           : maximum days overdue across bureau entries
    - ``bureau_prolong_sum``           : total number of credit prolongations
    - ``bureau_bbal_cnt_mean``         : mean monthly balance record count
    - ``bureau_bbal_dpd_rate_mean``    : mean DPD rate across bureau entries
    - ``bureau_bbal_dpd_rate_3m_mean`` : mean DPD rate (last 3 months, -3m to 0)
    - ``bureau_bbal_dpd_rate_6m_mean`` : mean DPD rate (last 6 months)
    - ``bureau_bbal_dpd_rate_12m_mean``: mean DPD rate (months 7-12 ago, -12m to -6m)
    - ``bureau_bbal_dpd_rate_3m_to_12m_mean``: mean DPD rate (3m to 12m, -12m to -3m)
    - ``bureau_bbal_dpd_trend_mean``   : mean trend (6m - 12m)
    - ``bureau_bbal_max_severity_mean``: mean max severity across bureau entries
    - ``bureau_bbal_recent_max_severity_mean``: mean recent max severity
    - ``bureau_bbal_active_months_mean``: mean active months per bureau entry

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

    # Load bureau_balance once for both aggregation helpers (D-07 memory protocol)
    import gc
    bbal = pd.read_csv(data_dir / _FILE_BUREAU_BAL)
    bbal_agg = _aggregate_bureau_balance(bbal)

    # Enrich bureau with balance aggregates (left join on SK_ID_BUREAU)
    bureau = bureau.merge(bbal_agg, on="SK_ID_BUREAU", how="left")

    bureau = bureau.assign(
        is_active=(bureau["CREDIT_ACTIVE"] == "Active").astype(np.float32),
        is_closed=(bureau["CREDIT_ACTIVE"] == "Closed").astype(np.float32),
        is_overdue=(bureau["CREDIT_DAY_OVERDUE"] > 0).astype(np.float32),
        opened_in_last_year=(bureau["DAYS_CREDIT"] >= -365).astype(int),
    )

    bureau_agg = (
        bureau.groupby("SK_ID_CURR", sort=False)
        .agg(
            bureau_cnt=("SK_ID_BUREAU", "count"),
            bureau_active_cnt=("is_active", "sum"),
            bureau_closed_cnt=("is_closed", "sum"),
            bureau_overdue_cnt=("is_overdue", "sum"),
            bureau_days_credit_mean=("DAYS_CREDIT", "mean"),
            bureau_days_credit_min=("DAYS_CREDIT", "min"),
            bureau_days_credit_max=("DAYS_CREDIT", "max"),
            bureau_days_credit_std=("DAYS_CREDIT", "std"),
            bureau_credit_sum=("AMT_CREDIT_SUM", "sum"),
            bureau_amt_credit_mean=("AMT_CREDIT_SUM", "mean"),
            bureau_credit_debt_sum=("AMT_CREDIT_SUM_DEBT", "sum"),
            bureau_credit_debt_std=("AMT_CREDIT_SUM_DEBT", "std"),
            bureau_credit_debt_max=("AMT_CREDIT_SUM_DEBT", "max"),
            bureau_credit_overdue_sum=("AMT_CREDIT_SUM_OVERDUE", "sum"),
            bureau_overdue_sum=("AMT_CREDIT_SUM_OVERDUE", "sum"),
            bureau_max_overdue_amt=("AMT_CREDIT_MAX_OVERDUE", "max"),
            bureau_annuity_mean=("AMT_ANNUITY", "mean"),
            bureau_overdue_max=("CREDIT_DAY_OVERDUE", "max"),
            bureau_prolong_sum=("CNT_CREDIT_PROLONG", "sum"),
            bureau_recent_openings=("opened_in_last_year", "sum"),
            bureau_days_since_last_credit=("DAYS_CREDIT", "max"),
            bureau_bbal_cnt_mean=("bbal_cnt", "mean"),
            bureau_bbal_dpd_rate_mean=("bbal_dpd_rate", "mean"),
            bureau_bbal_dpd_rate_std_mean=("bbal_dpd_rate_std", "mean"),
            bureau_bbal_dpd_rate_3m_mean=("bbal_dpd_rate_3m", "mean"),
            bureau_bbal_dpd_rate_6m_mean=("bbal_dpd_rate_6m", "mean"),
            bureau_bbal_dpd_rate_12m_mean=("bbal_dpd_rate_12m", "mean"),
            bureau_bbal_dpd_rate_3m_to_12m_mean=("bbal_dpd_rate_3m_to_12m", "mean"),
            bureau_bbal_dpd_trend_mean=("bbal_dpd_trend", "mean"),
            bureau_bbal_max_severity_mean=("bbal_max_severity", "mean"),
            bureau_bbal_recent_max_severity_mean=("bbal_recent_max_severity", "mean"),
            bureau_bbal_active_months_mean=("bbal_active_months", "mean"),
        )
        .reset_index()
    )

    result = app.merge(bureau_agg, on="SK_ID_CURR", how="left")

    # Applicants with no bureau history → count = 0, not NaN
    result["bureau_cnt"] = result["bureau_cnt"].fillna(0).astype(np.int32)
    result["bureau_active_cnt"] = result["bureau_active_cnt"].fillna(0).astype(np.int32)
    result["bureau_closed_cnt"] = result["bureau_closed_cnt"].fillna(0).astype(np.int32)
    result["bureau_overdue_cnt"] = result["bureau_overdue_cnt"].fillna(0).astype(np.int32)
    result["bureau_prolong_sum"] = (
        result["bureau_prolong_sum"].fillna(0).astype(np.int32)
    )
    result["bureau_recent_openings"] = result["bureau_recent_openings"].fillna(0).astype(np.int32)
    result["bureau_overdue_sum"] = result["bureau_overdue_sum"].fillna(0)
    result["bureau_amt_credit_mean"] = result["bureau_amt_credit_mean"].fillna(0)
    result["bureau_days_since_last_credit"] = result["bureau_days_since_last_credit"].fillna(0)

    # Reuse same bbal for DPD recency — no second CSV load (D-07 memory protocol)
    dpd_recency = _compute_bureau_dpd_recency(bbal, bureau)
    del bbal  # Release 359 MB before returning
    gc.collect()
    result = result.merge(dpd_recency, on="SK_ID_CURR", how="left")

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
        is_cancelled=(prev["NAME_CONTRACT_STATUS"] == "Canceled").astype(np.float32),
        credit_to_app_ratio=np.where(
            prev["AMT_APPLICATION"] > 0,
            prev["AMT_CREDIT"] / prev["AMT_APPLICATION"],
            np.nan,
        ),
    )

    prev_agg = (
        prev.groupby("SK_ID_CURR", sort=False)
        .agg(
            prev_cnt=("SK_ID_PREV", "count"),
            prev_approved_cnt=("is_approved", "sum"),
            prev_refused_cnt=("is_refused", "sum"),
            prev_cancelled_cnt=("is_cancelled", "sum"),
            prev_amt_credit_mean=("AMT_CREDIT", "mean"),
            prev_amt_credit_sum=("AMT_CREDIT", "sum"),
            prev_amt_credit_std=("AMT_CREDIT", "std"),
            prev_amt_credit_max=("AMT_CREDIT", "max"),
            prev_amt_application_mean=("AMT_APPLICATION", "mean"),
            prev_annuity_mean=("AMT_ANNUITY", "mean"),
            prev_amt_annuity_mean=("AMT_ANNUITY", "mean"),
            prev_amt_down_payment_mean=("AMT_DOWN_PAYMENT", "mean"),
            prev_credit_to_app_ratio_mean=("credit_to_app_ratio", "mean"),
            prev_days_decision_min=("DAYS_DECISION", "min"),
            prev_days_decision_mean=("DAYS_DECISION", "mean"),
            prev_days_decision_max=("DAYS_DECISION", "max"),
            prev_cnt_payment_mean=("CNT_PAYMENT", "mean"),
            prev_rate_down_payment_mean=("RATE_DOWN_PAYMENT", "mean"),
        )
        .reset_index()
    )

    result = app.merge(prev_agg, on="SK_ID_CURR", how="left")

    result["prev_cnt"] = result["prev_cnt"].fillna(0).astype(np.int32)
    result["prev_approved_cnt"] = result["prev_approved_cnt"].fillna(0).astype(np.int32)
    result["prev_refused_cnt"] = result["prev_refused_cnt"].fillna(0).astype(np.int32)
    result["prev_cancelled_cnt"] = result["prev_cancelled_cnt"].fillna(0).astype(np.int32)
    if "prev_amt_annuity_mean" in result.columns:
        result["prev_amt_annuity_mean"] = result["prev_amt_annuity_mean"].fillna(0)
    if "prev_amt_down_payment_mean" in result.columns:
        result["prev_amt_down_payment_mean"] = result["prev_amt_down_payment_mean"].fillna(0)

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

    pos = pos.assign(
        is_overdue=(pos["SK_DPD"] > 0).astype(np.float32),
        is_active=(pos["NAME_CONTRACT_STATUS"] == "Active").astype(np.float32),
        is_completed=(pos["NAME_CONTRACT_STATUS"] == "Completed").astype(np.float32),
    )

    pos_agg = (
        pos.groupby("SK_ID_CURR", sort=False)
        .agg(
            pos_cnt=("MONTHS_BALANCE", "count"),
            pos_months_balance_mean=("MONTHS_BALANCE", "mean"),
            pos_cnt_instalment_mean=("CNT_INSTALMENT", "mean"),
            pos_cnt_instalment_std=("CNT_INSTALMENT", "std"),
            pos_sk_dpd_max=("SK_DPD", "max"),
            pos_sk_dpd_std=("SK_DPD", "std"),
            pos_sk_dpd_mean=("SK_DPD", "mean"),
            pos_sk_dpd_def_max=("SK_DPD_DEF", "max"),
            pos_overdue_cnt=("is_overdue", "sum"),
            pos_overdue_rate=("is_overdue", "mean"),
            pos_active_cnt=("is_active", "sum"),
            pos_completed_cnt=("is_completed", "sum"),
        )
        .reset_index()
    )

    result = app.merge(pos_agg, on="SK_ID_CURR", how="left")

    result["pos_cnt"] = result["pos_cnt"].fillna(0).astype(np.int32)
    result["pos_overdue_cnt"] = result["pos_overdue_cnt"].fillna(0).astype(np.int32)

    _assert_no_row_multiplication(app, result, "POS_CASH join")
    return result


# ---------------------------------------------------------------------------
# Private helpers: time-series features (Phase A2)
# ---------------------------------------------------------------------------


def _compute_installment_streaks(inst: pd.DataFrame) -> pd.DataFrame:
    """Compute consecutive late payment streaks from installments_payments.

    A payment is LATE when DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT (both negative;
    less negative = more recent / later payment).

    Returns
    -------
    pd.DataFrame
        Columns: inst_max_consec_late_streak, inst_months_since_last_late.
        Indexed by SK_ID_CURR.
    """
    inst = inst.copy()

    # Late flag: payment date > scheduled date (less negative = later)
    inst = inst.assign(
        is_late=(inst["DAYS_ENTRY_PAYMENT"] > inst["DAYS_INSTALMENT"]).astype(np.int8)
    )

    # Sort by applicant and schedule date (oldest first)
    inst = inst.sort_values(["SK_ID_CURR", "DAYS_INSTALMENT"]).reset_index(drop=True)

    # Compute consecutive streak using cumsum trick:
    # Within each applicant, identify groups where is_late status changes
    shifted = inst.groupby("SK_ID_CURR", sort=False)["is_late"].shift()
    inst["is_late_changed"] = inst["is_late"] != shifted
    inst["streak_group"] = inst.groupby("SK_ID_CURR", sort=False)["is_late_changed"].cumsum()

    # Within each streak_group, count consecutive records (cumcount)
    inst["streak_count"] = inst.groupby(
        ["streak_group"], sort=False
    ).cumcount() + 1

    # Only keep counts where is_late == 1 (late payments)
    # For non-late groups, replace with 0
    inst["streak_length"] = np.where(
        inst["is_late"] == 1,
        inst["streak_count"],
        0
    )

    # Max consecutive late streak per applicant
    max_streak = (
        inst.groupby("SK_ID_CURR", sort=False)["streak_length"]
        .max()
        .fillna(0)
        .astype(np.int32)
    )

    # Most recent late payment: find max DAYS_INSTALMENT (least negative = most recent)
    # where is_late == 1, then compute months as abs(DAYS_INSTALMENT) / 30
    late_payments = inst[inst["is_late"] == 1].copy()
    most_recent_late = (
        late_payments.groupby("SK_ID_CURR", sort=False)["DAYS_INSTALMENT"]
        .max()  # least negative = most recent
    )
    months_since_late = (-most_recent_late / 30.0).round(2)

    # Combine results
    result = pd.DataFrame({
        'inst_max_consec_late_streak': max_streak,
        'inst_months_since_last_late': months_since_late,
    })

    return result


def _compute_bureau_dpd_recency(
    bbal: pd.DataFrame,
    bureau: pd.DataFrame,
) -> pd.DataFrame:
    """Compute DPD recency features from bureau_balance + bureau.

    STATUS codes: C = closed, X = unknown, 0 = no DPD, 1-5 = DPD buckets.
    DPD = True when STATUS in {'1', '2', '3', '4', '5'}.
    MONTHS_BALANCE: 0 = most recent, negative = older months.

    Returns
    -------
    pd.DataFrame
        Columns: bbal_months_since_last_dpd, bbal_dpd_last_3m_rate,
        bbal_dpd_last_6m_vs_prior_rate.
        Indexed by SK_ID_CURR (aggregated from SK_ID_BUREAU level).
    """
    bbal = bbal.copy()

    # DPD indicator: STATUS in {'1', '2', '3', '4', '5'}
    bbal = bbal.assign(
        is_dpd=bbal["STATUS"].isin({"1", "2", "3", "4", "5"}).astype(np.int8)
    )

    # Add time window flags
    bbal = bbal.assign(
        in_last_3m=(bbal["MONTHS_BALANCE"] >= -3).astype(bool),
        in_last_6m=(bbal["MONTHS_BALANCE"] >= -6).astype(bool),
        in_prior_6m=(bbal["MONTHS_BALANCE"] < -6).astype(bool),
    )

    # --- Months since last DPD ---
    # Most recent DPD (max MONTHS_BALANCE = least negative)
    dpd_records = bbal[bbal["is_dpd"] == 1].copy()
    most_recent_dpd = (
        dpd_records.groupby("SK_ID_BUREAU", sort=False)["MONTHS_BALANCE"]
        .max()
    )
    months_since_dpd_bureau = (-most_recent_dpd).round(2)

    # --- DPD rates (windowed) ---
    # Last 3m rate per bureau
    dpd_3m_rate = (
        bbal[bbal["in_last_3m"]]
        .groupby("SK_ID_BUREAU", sort=False)["is_dpd"]
        .mean()
        .fillna(0.0)
    )

    # Last 6m and prior 6m rates
    dpd_last_6m_rate = (
        bbal[bbal["in_last_6m"]]
        .groupby("SK_ID_BUREAU", sort=False)["is_dpd"]
        .mean()
        .fillna(0.0)
    )
    dpd_prior_6m_rate = (
        bbal[bbal["in_prior_6m"]]
        .groupby("SK_ID_BUREAU", sort=False)["is_dpd"]
        .mean()
        .fillna(0.0)
    )

    # Trajectory: last_6m - prior_6m (positive = worsening)
    dpd_trajectory = (dpd_last_6m_rate - dpd_prior_6m_rate).round(3)

    # Combine bureau-level results
    bureau_features = pd.DataFrame({
        'bbal_months_since_last_dpd': months_since_dpd_bureau,
        'bbal_dpd_last_3m_rate': dpd_3m_rate,
        'bbal_dpd_last_6m_vs_prior_rate': dpd_trajectory,
    }).reset_index()

    # Aggregate to SK_ID_CURR by mean (average across multiple bureau entries)
    bureau = bureau[['SK_ID_CURR', 'SK_ID_BUREAU']].copy()
    merged = bureau.merge(bureau_features, on='SK_ID_BUREAU', how='left')

    result = (
        merged.groupby("SK_ID_CURR", sort=False)
        .agg({
            'bbal_months_since_last_dpd': 'mean',
            'bbal_dpd_last_3m_rate': 'mean',
            'bbal_dpd_last_6m_vs_prior_rate': 'mean',
        })
        .round(3)
    )

    return result


def _compute_payment_amount_trend(inst: pd.DataFrame) -> pd.DataFrame:
    """Compute payment amount trend (slope) from installments_payments.

    Uses linear regression: AMT_PAYMENT ~ time_order.
    Time order increases from oldest (most negative DAYS_ENTRY_PAYMENT) to newest
    (least negative DAYS_ENTRY_PAYMENT).
    Applicants with < 3 payments get slope = 0.0.

    Returns
    -------
    pd.DataFrame
        Column: inst_payment_trend_slope.
        Indexed by SK_ID_CURR.
    """
    inst = inst.copy()
    inst = inst.sort_values(["SK_ID_CURR", "DAYS_ENTRY_PAYMENT"]).reset_index(drop=True)

    def compute_trend(group):
        """Fit linear regression within one applicant's payments."""
        if len(group) < 3:
            return 0.0

        # Create time ordering (0, 1, 2, ...) from oldest to newest
        x = np.arange(len(group), dtype=np.float64)
        y = group["AMT_PAYMENT"].values.astype(np.float64)

        # Linear regression: y = a*x + b; return slope a
        try:
            coeffs = np.polyfit(x, y, deg=1)
            return float(coeffs[0])
        except (np.linalg.LinAlgError, ValueError):
            return 0.0

    slope = (
        inst.groupby("SK_ID_CURR", sort=False)
        .apply(compute_trend, include_groups=False)
        .astype(np.float32)
    )

    result = pd.DataFrame({
        'inst_payment_trend_slope': slope,
    })

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

    inst = inst.assign(
        **{k: v for k, v in [
            ("payment_diff", inst["AMT_INSTALMENT"] - inst["AMT_PAYMENT"]),
        ]}
    )

    inst_agg = (
        inst.groupby("SK_ID_CURR", sort=False)
        .agg(
            inst_cnt=("DAYS_INSTALMENT", "count"),
            inst_late_cnt=("is_late", "sum"),
            inst_amt_payment_sum=("AMT_PAYMENT", "sum"),
            inst_amt_instalment_mean=("AMT_INSTALMENT", "mean"),
            inst_payment_ratio_mean=("payment_ratio", "mean"),
            inst_payment_ratio_std=("payment_ratio", "std"),
            inst_payment_ratio_min=("payment_ratio", "min"),
            inst_payment_ratio_max=("payment_ratio", "max"),
            inst_payment_diff_mean=("payment_diff", "mean"),
            inst_payment_diff_std=("payment_diff", "std"),
            inst_days_past_due_mean=("days_past_due", "mean"),
            inst_days_past_due_max=("days_past_due", "max"),
            inst_days_past_due_std=("days_past_due", "std"),
        )
        .reset_index()
    )

    result = app.merge(inst_agg, on="SK_ID_CURR", how="left")

    result["inst_cnt"] = result["inst_cnt"].fillna(0).astype(np.int32)
    result["inst_late_cnt"] = result["inst_late_cnt"].fillna(0).astype(np.int32)

    # Phase A2: Add time-series features
    streak_features = _compute_installment_streaks(inst)
    trend_features = _compute_payment_amount_trend(inst)

    result = result.merge(streak_features, on="SK_ID_CURR", how="left")
    result = result.merge(trend_features, on="SK_ID_CURR", how="left")

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
        ),
        is_overdue=(cc["SK_DPD"] > 0).astype(np.float32),
        min_payment_ratio=np.where(
            cc["AMT_INST_MIN_REGULARITY"] > 0,
            cc["AMT_PAYMENT_CURRENT"] / cc["AMT_INST_MIN_REGULARITY"],
            np.nan,
        ),
    )

    cc_agg = (
        cc.groupby("SK_ID_CURR", sort=False)
        .agg(
            cc_cnt=("MONTHS_BALANCE", "count"),
            cc_bal_mean=("AMT_BALANCE", "mean"),
            cc_bal_max=("AMT_BALANCE", "max"),
            cc_bal_std=("AMT_BALANCE", "std"),
            cc_bal_min=("AMT_BALANCE", "min"),
            cc_drawing_mean=("AMT_DRAWINGS_CURRENT", "mean"),
            cc_drawing_std=("AMT_DRAWINGS_CURRENT", "std"),
            cc_atm_drawing_mean=("AMT_DRAWINGS_ATM_CURRENT", "mean"),
            cc_utilization_mean=("utilization", "mean"),
            cc_utilization_max=("utilization", "max"),
            cc_utilization_std=("utilization", "std"),
            cc_limit_mean=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
            cc_sk_dpd_max=("SK_DPD", "max"),
            cc_sk_dpd_mean=("SK_DPD", "mean"),
            cc_dpd_rate=("is_overdue", "mean"),
            cc_min_payment_ratio_mean=("min_payment_ratio", "mean"),
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


# ---------------------------------------------------------------------------
# Public helper: load raw secondary tables for Wave 2 feature engineering
# ---------------------------------------------------------------------------


def load_secondary_raw(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """
    Load raw secondary tables needed for Wave 2 temporal trajectory features.

    Prepares each table with the index structure expected by the Wave 2 private
    functions in ``src/features.py``.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing the raw CSV files (same as used by ``load_data``).

    Returns
    -------
    dict with keys:
        - ``"bbal"`` : bureau_balance merged with bureau for SK_ID_CURR mapping.
          MultiIndex with named levels ``["SK_ID_CURR", "MONTHS_BALANCE"]``.
          Contains STATUS column.
        - ``"inst"`` : installments_payments with SK_ID_CURR as index.
          Contains DAYS_ENTRY_PAYMENT, DAYS_INSTALMENT, AMT_PAYMENT, AMT_INSTALMENT.
        - ``"cc"``   : credit_card_balance with SK_ID_CURR as index.
          AMT_CREDIT_LIMIT_ACTUAL renamed to AMT_CREDIT_LIMIT.
          Contains AMT_BALANCE, MONTHS_BALANCE, AMT_DRAWINGS_ATM_CURRENT.
        - ``"prev"`` : previous_application with SK_ID_CURR as index.
          Contains CODE_REJECT_REASON, NAME_GOODS_CATEGORY, RATE_INTEREST_PRIMARY.
    """
    data_dir = Path(data_dir)

    # --- bureau_balance: needs SK_ID_BUREAU → SK_ID_CURR mapping via bureau.csv ---
    df_bureau = pd.read_csv(
        data_dir / _FILE_BUREAU,
        usecols=["SK_ID_CURR", "SK_ID_BUREAU"],
    )
    df_bbal_raw = pd.read_csv(
        data_dir / _FILE_BUREAU_BAL,
        usecols=["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"],
    )
    df_bbal = df_bbal_raw.merge(
        df_bureau[["SK_ID_BUREAU", "SK_ID_CURR"]],
        on="SK_ID_BUREAU",
        how="left",
    )
    df_bbal = df_bbal.dropna(subset=["SK_ID_CURR"])
    df_bbal["SK_ID_CURR"] = df_bbal["SK_ID_CURR"].astype(int)
    df_bbal = df_bbal.set_index(["SK_ID_CURR", "MONTHS_BALANCE"])
    df_bbal.index.names = ["SK_ID_CURR", "MONTHS_BALANCE"]

    # --- installments_payments ---
    df_inst = pd.read_csv(
        data_dir / _FILE_INSTALLMENTS,
        usecols=["SK_ID_CURR", "DAYS_ENTRY_PAYMENT", "DAYS_INSTALMENT",
                 "AMT_PAYMENT", "AMT_INSTALMENT"],
    )
    # Compute days_past_due required by _inst_payment_consistency_score and
    # _inst_recency_weighted_dpd; positive = late, negative = paid early.
    df_inst["days_past_due"] = df_inst["DAYS_ENTRY_PAYMENT"] - df_inst["DAYS_INSTALMENT"]
    df_inst = df_inst.set_index("SK_ID_CURR")

    # --- credit_card_balance ---
    df_cc = pd.read_csv(
        data_dir / _FILE_CC_BAL,
        usecols=["SK_ID_CURR", "MONTHS_BALANCE", "AMT_BALANCE",
                 "AMT_CREDIT_LIMIT_ACTUAL", "AMT_DRAWINGS_ATM_CURRENT"],
    )
    df_cc = df_cc.rename(columns={"AMT_CREDIT_LIMIT_ACTUAL": "AMT_CREDIT_LIMIT"})
    df_cc = df_cc.set_index("SK_ID_CURR")

    # --- previous_application ---
    df_prev = pd.read_csv(
        data_dir / _FILE_PREV_APP,
        usecols=["SK_ID_CURR", "CODE_REJECT_REASON",
                 "NAME_GOODS_CATEGORY", "RATE_INTEREST_PRIMARY"],
    )
    df_prev = df_prev.set_index("SK_ID_CURR")

    return {
        "bbal": df_bbal,
        "inst": df_inst,
        "cc": df_cc,
        "prev": df_prev,
    }
