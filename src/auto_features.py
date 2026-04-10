"""
auto_features.py
----------------
Featuretools Deep Feature Synthesis (DFS) auto-aggregation.
Generates features from the 7-table relational structure without
manual specification of aggregate functions.

Entry points:
  - build_featuretools_feature_store: DFS on train data, IV + correlation filter
  - apply_featuretools_feature_store: Apply feature definitions to test data
  - deduplicate_dfs_features: Remove highly correlated feature pairs
  - evaluate_dfs_features: Evaluate DFS features with Gini delta gating
"""

import contextlib
import io
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import featuretools as ft
except ImportError:
    ft = None  # type: ignore

try:
    from woodwork.logical_types import Categorical, Double, Integer, BooleanNullable
except ImportError:
    Categorical = Double = Integer = BooleanNullable = None  # type: ignore

from src.features import select_features_by_iv
from src.model import train_xgboost_optuna
from src.utils import gini_coefficient

# Constants
_DEFAULT_AGG_PRIMITIVES = [
    "count",
    "mean",
    "sum",
    "std",
    "num_unique",
]
_NAN_SENTINEL = -999.0

# Leaky SK_DPD columns to be removed (post-origination payment distress, Basel III Article 174)
_LEAKY_SKDPD_COLS: list[str] = [
    # Raw SK_DPD aggregates (pos_cash / credit_card) — directly from target-period records
    "pos_sk_dpd_max", "pos_sk_dpd_std", "pos_sk_dpd_mean", "pos_sk_dpd_def_max",
    "cc_sk_dpd_max", "cc_sk_dpd_mean", "cc_dpd_rate",
    # Featuretools DFS aggregates of SK_DPD columns (SUM, MEAN, STD variants)
    "SUM(credit_card.SK_DPD)", "SUM(credit_card.SK_DPD_DEF)",
    "MEAN(credit_card.SK_DPD)", "MEAN(credit_card.SK_DPD_DEF)",
    "STD(credit_card.SK_DPD)", "STD(credit_card.SK_DPD_DEF)",
    "SUM(pos_cash.SK_DPD)", "SUM(pos_cash.SK_DPD_DEF)",
    "MEAN(pos_cash.SK_DPD)", "MEAN(pos_cash.SK_DPD_DEF)",
    "STD(pos_cash.SK_DPD)", "STD(pos_cash.SK_DPD_DEF)",
    "SUM(previous_application.credit_card.SK_DPD)", "SUM(previous_application.credit_card.SK_DPD_DEF)",
    "MEAN(previous_application.credit_card.SK_DPD)", "MEAN(previous_application.credit_card.SK_DPD_DEF)",
    "STD(previous_application.credit_card.SK_DPD)", "STD(previous_application.credit_card.SK_DPD_DEF)",
    "SUM(previous_application.pos_cash.SK_DPD)", "SUM(previous_application.pos_cash.SK_DPD_DEF)",
    "MEAN(previous_application.pos_cash.SK_DPD)", "MEAN(previous_application.pos_cash.SK_DPD_DEF)",
    "STD(previous_application.pos_cash.SK_DPD)", "STD(previous_application.pos_cash.SK_DPD_DEF)",
    # Bureau-balance DPD (MONTHS_BALANCE >= -6 or all-months — includes application month 0)
    "bureau_bbal_dpd_rate_mean",     # mean across ALL months including month 0
    "bureau_bbal_dpd_rate_std_mean", # std across ALL months including month 0
    "bureau_bbal_dpd_rate_6m_mean",  # MB >= -6, includes month 0
    "bureau_bbal_dpd_trend_mean",    # 6m - 12m; 6m component is leaky
    "bbal_months_since_last_dpd",    # uses all records including month 0
    "bbal_dpd_last_3m_rate",         # MB >= -3, includes month 0
    "bbal_dpd_last_6m_vs_prior_rate",# recent window includes month 0
    # Cross-table features derived from leaky inputs
    "multi_dpd_flag",    # primary path uses cc_dpd_rate (already leaky)
    "dpd_escalation",    # derived from bureau_bbal_dpd_rate_6m_mean (leaky)
]

# File names (canonical in Home Credit dataset)
_FILE_APP_TRAIN = "application_train.csv"
_FILE_APP_TEST = "application_test.csv"
_FILE_BUREAU = "bureau.csv"
_FILE_BUREAU_BAL = "bureau_balance.csv"
_FILE_PREV_APP = "previous_application.csv"
_FILE_POS_CASH = "POS_CASH_balance.csv"
_FILE_INSTALLMENTS = "installments_payments.csv"
_FILE_CC_BAL = "credit_card_balance.csv"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_entity_tables(
    data_dir: Path | str,
    train_ids: list[int],
) -> dict[str, pd.DataFrame]:
    """
    Load 7-table relational dataset from CSVs and filter to train_ids.

    Loads application, bureau, bureau_balance, previous_application,
    POS_CASH_balance, installments_payments, and credit_card_balance tables.
    All secondary tables are filtered to rows where SK_ID_CURR is in train_ids.
    bureau_balance is filtered via its parent bureau table.

    Parameters
    ----------
    data_dir : Path | str
        Directory containing CSV files.
    train_ids : list[int]
        List of SK_ID_CURR values to filter secondary tables.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: "application", "bureau", "bureau_balance", "previous_application",
        "pos_cash", "installments", "credit_card".
    """
    data_dir = Path(data_dir)

    # Load application table
    app_path = data_dir / _FILE_APP_TRAIN
    if not app_path.exists():
        raise FileNotFoundError(f"Missing {_FILE_APP_TRAIN} in {data_dir}")

    application = pd.read_csv(app_path)
    # Filter application to train_ids
    application = application[application["SK_ID_CURR"].isin(train_ids)]

    # Load secondary tables and filter to train_ids
    bureau_path = data_dir / _FILE_BUREAU
    bureau = pd.read_csv(bureau_path)
    bureau = bureau[bureau["SK_ID_CURR"].isin(train_ids)]

    bureau_balance_path = data_dir / _FILE_BUREAU_BAL
    bureau_balance = pd.read_csv(bureau_balance_path)
    # Filter to SK_ID_BUREAU in filtered bureau
    valid_bureaus = set(bureau["SK_ID_BUREAU"])
    bureau_balance = bureau_balance[bureau_balance["SK_ID_BUREAU"].isin(valid_bureaus)]

    prev_app_path = data_dir / _FILE_PREV_APP
    previous_application = pd.read_csv(prev_app_path)
    previous_application = previous_application[
        previous_application["SK_ID_CURR"].isin(train_ids)
    ]

    pos_cash_path = data_dir / _FILE_POS_CASH
    pos_cash = pd.read_csv(pos_cash_path)
    pos_cash = pos_cash[pos_cash["SK_ID_CURR"].isin(train_ids)]
    # D-04: Filter to historical months only (MONTHS_BALANCE < 0 excludes application month = 0)
    pos_cash = pos_cash[pos_cash["MONTHS_BALANCE"] < 0]

    installments_path = data_dir / _FILE_INSTALLMENTS
    installments = pd.read_csv(installments_path)
    installments = installments[installments["SK_ID_CURR"].isin(train_ids)]

    cc_balance_path = data_dir / _FILE_CC_BAL
    credit_card = pd.read_csv(cc_balance_path)
    credit_card = credit_card[credit_card["SK_ID_CURR"].isin(train_ids)]
    # D-04: Filter to historical months only (MONTHS_BALANCE < 0 excludes application month = 0)
    credit_card = credit_card[credit_card["MONTHS_BALANCE"] < 0]

    return {
        "application": application,
        "bureau": bureau,
        "bureau_balance": bureau_balance,
        "previous_application": previous_application,
        "pos_cash": pos_cash,
        "installments": installments,
        "credit_card": credit_card,
    }


def _build_entity_set(tables: dict[str, pd.DataFrame]) -> Any:
    """
    Construct a featuretools EntitySet from loaded tables.

    Configures the 7-table relational structure with foreign keys and
    synthetic indices where needed (for child tables without unique PKs).

    Adds manual aggregations for STATUS-based DPD and installment recency features
    before EntitySet construction to ensure they are available for DFS.

    Parameters
    ----------
    tables : dict[str, pd.DataFrame]
        Dictionary returned by _load_entity_tables.

    Returns
    -------
    ft.EntitySet
        EntitySet with ID "home_credit" and all relationships configured.

    Raises
    ------
    ImportError
        If featuretools is not installed.
    """
    if ft is None:
        raise ImportError("featuretools is required for this function")

    # --- Task 1: Add STATUS-based bureau_balance DPD aggregations ---
    # Pre-compute DPD flags at bureau level (historical closed loans only)
    bureau_balance = tables["bureau_balance"].copy()
    bureau_balance_hist = bureau_balance[bureau_balance["MONTHS_BALANCE"] < 0].copy()

    # STATUS ∈ ["1","2","3","4","5"] = 1-5+ months overdue
    bureau_balance_hist["is_dpd"] = bureau_balance_hist["STATUS"].isin(
        ["1", "2", "3", "4", "5"]
    ).astype(int)

    # Aggregate to SK_ID_BUREAU level
    bb_agg = bureau_balance_hist.groupby("SK_ID_BUREAU").agg(
        bb_dpd_mean=("is_dpd", "mean"),
        bb_dpd_max=("is_dpd", "max"),
    ).reset_index()

    # Merge into bureau DataFrame
    bureau = tables["bureau"].copy()
    bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")
    bureau[["bb_dpd_mean", "bb_dpd_max"]] = bureau[
        ["bb_dpd_mean", "bb_dpd_max"]
    ].fillna(0)

    # Update tables dictionary (immutable pattern: create copies, don't mutate original)
    tables = tables.copy()
    tables["bureau"] = bureau
    tables["bureau_balance"] = bureau_balance

    # --- Task 2 & 3: Add installment recency aggregations and skew ---
    installments = tables["installments"].copy()

    # Derive base columns if not already present
    if "days_late" not in installments.columns:
        # days_late = max(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT, 0)
        installments["days_late"] = (
            installments["DAYS_ENTRY_PAYMENT"] - installments["DAYS_INSTALMENT"]
        ).clip(lower=0)

    if "payment_diff" not in installments.columns:
        # payment_diff = AMT_INSTALMENT - AMT_PAYMENT
        installments["payment_diff"] = (
            installments["AMT_INSTALMENT"] - installments["AMT_PAYMENT"]
        )

    # Create a rank column for recency (most recent first: rank 0 is most recent)
    installments["rank"] = installments.groupby("SK_ID_CURR").cumcount(ascending=False)
    inst_recent = installments[installments["rank"] < 12].copy()  # Most recent 12

    # Recency aggregations
    inst_recent_agg = inst_recent.groupby("SK_ID_CURR").agg(
        inst_recent_late_mean=("days_late", "mean"),
        inst_recent_underpay=("payment_diff", "mean"),
    ).reset_index()

    # For trend, compute overall mean separately
    inst_all_agg = installments.groupby("SK_ID_CURR").agg(
        inst_days_late_mean=("days_late", "mean"),
    ).reset_index()

    inst_recent_agg = inst_recent_agg.merge(inst_all_agg, on="SK_ID_CURR", how="left")
    inst_recent_agg["inst_late_trend"] = (
        inst_recent_agg["inst_recent_late_mean"]
        - inst_recent_agg["inst_days_late_mean"]
    )
    inst_recent_agg = inst_recent_agg.drop(columns=["inst_days_late_mean"])

    # Worst-case underpayment
    inst_payment_max = installments.groupby("SK_ID_CURR").agg(
        inst_payment_diff_max=("payment_diff", "max"),
    ).reset_index()

    # Distribution shape of payment delays
    try:
        from scipy.stats import skew

        inst_skew = installments.groupby("SK_ID_CURR").agg(
            inst_late_skew=("days_late", lambda x: skew(x) if len(x) > 1 else 0),
        ).reset_index()
    except ImportError:
        # Fallback: compute simple skew manually or use 0
        inst_skew = installments.groupby("SK_ID_CURR").agg(
            inst_late_skew=("days_late", lambda x: 0),
        ).reset_index()

    # Merge recency aggregations back into installments
    installments = installments.merge(inst_recent_agg, on="SK_ID_CURR", how="left")
    installments = installments.merge(inst_payment_max, on="SK_ID_CURR", how="left")
    installments = installments.merge(inst_skew, on="SK_ID_CURR", how="left")

    # Fill NaN values with 0 for aggregations that may not have matched rows
    recency_cols = [
        "inst_recent_late_mean",
        "inst_recent_underpay",
        "inst_late_trend",
        "inst_payment_diff_max",
        "inst_late_skew",
    ]
    for col in recency_cols:
        if col in installments.columns:
            installments[col] = installments[col].fillna(0)

    tables["installments"] = installments

    es = ft.EntitySet(id="home_credit")

    # Application (primary table, has SK_ID_CURR as PK)
    logical_types_application = {
        "SK_ID_CURR": Integer,
        "NAME_CONTRACT_TYPE": Categorical,
        "CODE_GENDER": Categorical,
        "FLAG_OWN_CAR": Categorical,
        "FLAG_OWN_REALTY": Categorical,
        "CNT_CHILDREN": Integer,
        "AMT_INCOME_TOTAL": Double,
        "AMT_CREDIT": Double,
        "AMT_ANNUITY": Double,
        "AMT_GOODS_PRICE": Double,
        "NAME_TYPE_SUITE": Categorical,
        "NAME_INCOME_TYPE": Categorical,
        "NAME_EDUCATION_TYPE": Categorical,
        "NAME_FAMILY_STATUS": Categorical,
        "NAME_HOUSING_TYPE": Categorical,
        "REGION_POPULATION_RELATIVE": Double,
        "DAYS_BIRTH": Integer,
        "DAYS_EMPLOYED": Integer,
        "DAYS_REGISTRATION": Integer,
        "DAYS_ID_PUBLISH": Integer,
        "OWN_CAR_AGE": Double,
        "FLAG_MOBIL": Integer,
        "FLAG_EMP_PHONE": Integer,
        "FLAG_WORK_PHONE": Integer,
        "FLAG_CONT_MOBILE": Integer,
        "FLAG_PHONE": Integer,
        "FLAG_EMAIL": Integer,
        "OCCUPATION_TYPE": Categorical,
        "CNT_FAM_MEMBERS": Double,  # Has 2 NaN values, so use Double instead of Integer
        "REGION_RATING_CLIENT": Integer,
        "REGION_RATING_CLIENT_W_CITY": Integer,
        "WEEKDAY_APPR_PROCESS_START": Categorical,
        "HOUR_APPR_PROCESS_START": Integer,
        "REG_REGION_NOT_LIVE_REGION": Integer,
        "REG_REGION_NOT_WORK_REGION": Integer,
        "LIVE_REGION_NOT_WORK_REGION": Integer,
        "REG_CITY_NOT_LIVE_CITY": Integer,
        "REG_CITY_NOT_WORK_CITY": Integer,
        "LIVE_CITY_NOT_WORK_CITY": Integer,
        "ORGANIZATION_TYPE": Categorical,
        "EXT_SOURCE_1": Double,
        "EXT_SOURCE_2": Double,
        "EXT_SOURCE_3": Double,
        "APARTMENTS_AVG": Double,
        "BASEMENTAREA_AVG": Double,
        "YEARS_BEGINEXPLUATATION_AVG": Double,
        "YEARS_BUILD_AVG": Double,
        "COMMONAREA_AVG": Double,
        "ELEVATORS_AVG": Double,
        "ENTRANCES_AVG": Double,
        "FLOORSMAX_AVG": Double,
        "FLOORSMIN_AVG": Double,
        "LANDAREA_AVG": Double,
        "LIVINGAPARTMENTS_AVG": Double,
        "LIVINGAREA_AVG": Double,
        "NONLIVINGAPARTMENTS_AVG": Double,
        "NONLIVINGAREA_AVG": Double,
        "APARTMENTS_MODE": Double,
        "BASEMENTAREA_MODE": Double,
        "YEARS_BEGINEXPLUATATION_MODE": Double,
        "YEARS_BUILD_MODE": Double,
        "COMMONAREA_MODE": Double,
        "ELEVATORS_MODE": Double,
        "ENTRANCES_MODE": Double,
        "FLOORSMAX_MODE": Double,
        "FLOORSMIN_MODE": Double,
        "LANDAREA_MODE": Double,
        "LIVINGAPARTMENTS_MODE": Double,
        "LIVINGAREA_MODE": Double,
        "NONLIVINGAPARTMENTS_MODE": Double,
        "NONLIVINGAREA_MODE": Double,
        "APARTMENTS_MEDI": Double,
        "BASEMENTAREA_MEDI": Double,
        "YEARS_BEGINEXPLUATATION_MEDI": Double,
        "YEARS_BUILD_MEDI": Double,
        "COMMONAREA_MEDI": Double,
        "ELEVATORS_MEDI": Double,
        "ENTRANCES_MEDI": Double,
        "FLOORSMAX_MEDI": Double,
        "FLOORSMIN_MEDI": Double,
        "LANDAREA_MEDI": Double,
        "LIVINGAPARTMENTS_MEDI": Double,
        "LIVINGAREA_MEDI": Double,
        "NONLIVINGAPARTMENTS_MEDI": Double,
        "NONLIVINGAREA_MEDI": Double,
        "FONDKAPREMONT_MODE": Categorical,
        "HOUSETYPE_MODE": Categorical,
        "TOTALAREA_MODE": Double,
        "WALLSMATERIAL_MODE": Categorical,
        "EMERGENCYSTATE_MODE": Categorical,
        "OBS_30_CNT_SOCIAL_CIRCLE": Double,  # Has 1021 NaN values, use Double instead of Integer
        "DEF_30_CNT_SOCIAL_CIRCLE": Double,  # Has 1021 NaN values, use Double instead of Integer
        "OBS_60_CNT_SOCIAL_CIRCLE": Double,  # Has 1021 NaN values, use Double instead of Integer
        "DEF_60_CNT_SOCIAL_CIRCLE": Double,  # Has 1021 NaN values, use Double instead of Integer
        "DAYS_LAST_PHONE_CHANGE": Double,  # Has 1 NaN value, use Double instead of Integer
        "FLAG_DOCUMENT_2": Integer,
        "FLAG_DOCUMENT_3": Integer,
        "FLAG_DOCUMENT_4": Integer,
        "FLAG_DOCUMENT_5": Integer,
        "FLAG_DOCUMENT_6": Integer,
        "FLAG_DOCUMENT_7": Integer,
        "FLAG_DOCUMENT_8": Integer,
        "FLAG_DOCUMENT_9": Integer,
        "FLAG_DOCUMENT_10": Integer,
        "FLAG_DOCUMENT_11": Integer,
        "FLAG_DOCUMENT_12": Integer,
        "FLAG_DOCUMENT_13": Integer,
        "FLAG_DOCUMENT_14": Integer,
        "FLAG_DOCUMENT_15": Integer,
        "FLAG_DOCUMENT_16": Integer,
        "FLAG_DOCUMENT_17": Integer,
        "FLAG_DOCUMENT_18": Integer,
        "FLAG_DOCUMENT_19": Integer,
        "FLAG_DOCUMENT_20": Integer,
        "FLAG_DOCUMENT_21": Integer,
        "AMT_REQ_CREDIT_BUREAU_HOUR": Double,  # Has 41519 NaN values, use Double instead of Integer
        "AMT_REQ_CREDIT_BUREAU_DAY": Double,  # Has 41519 NaN values, use Double instead of Integer
        "AMT_REQ_CREDIT_BUREAU_WEEK": Double,  # Has 41519 NaN values, use Double instead of Integer
        "AMT_REQ_CREDIT_BUREAU_MON": Double,  # Has 41519 NaN values, use Double instead of Integer
        "AMT_REQ_CREDIT_BUREAU_QRT": Double,  # Has 41519 NaN values, use Double instead of Integer
        "AMT_REQ_CREDIT_BUREAU_YEAR": Double,  # Has 41519 NaN values, use Double instead of Integer
    }
    logical_types_application = {
        k: v for k, v in logical_types_application.items() if k in tables["application"].columns
    }
    es = es.add_dataframe(
        dataframe_name="application",
        dataframe=tables["application"],
        index="SK_ID_CURR",
        logical_types=logical_types_application,
    )

    # Bureau (has SK_ID_BUREAU as PK)
    logical_types_bureau = {
        "SK_ID_CURR": Integer,
        "SK_ID_BUREAU": Integer,
        "CREDIT_ACTIVE": Categorical,
        "CREDIT_CURRENCY": Categorical,
        "DAYS_CREDIT": Integer,
        "CREDIT_DAY_OVERDUE": Integer,
        "DAYS_CREDIT_ENDDATE": Double,
        "DAYS_ENDDATE_FACT": Double,
        "AMT_CREDIT_MAX_OVERDUE": Double,
        "CNT_CREDIT_PROLONG": Integer,
        "AMT_CREDIT_SUM": Double,
        "AMT_CREDIT_SUM_DEBT": Double,
        "AMT_CREDIT_SUM_LIMIT": Double,
        "AMT_CREDIT_SUM_OVERDUE": Double,
        "CREDIT_TYPE": Categorical,
        "DAYS_CREDIT_UPDATE": Integer,
        "AMT_ANNUITY": Double,
        "bb_dpd_mean": Double,
        "bb_dpd_max": Double,
    }
    logical_types_bureau = {
        k: v for k, v in logical_types_bureau.items() if k in tables["bureau"].columns
    }
    es = es.add_dataframe(
        dataframe_name="bureau",
        dataframe=tables["bureau"],
        index="SK_ID_BUREAU",
        logical_types=logical_types_bureau,
    )

    # Bureau balance (no PK, create synthetic index)
    logical_types_bureau_balance = {
        "SK_ID_BUREAU": Integer,
        "MONTHS_BALANCE": Integer,
        "STATUS": Categorical,
    }
    logical_types_bureau_balance = {
        k: v for k, v in logical_types_bureau_balance.items() if k in tables["bureau_balance"].columns
    }
    es = es.add_dataframe(
        dataframe_name="bureau_balance",
        dataframe=tables["bureau_balance"],
        index="bbal_id",
        make_index=True,
        logical_types=logical_types_bureau_balance,
    )

    # Previous application (has SK_ID_PREV as PK)
    logical_types_previous_application = {
        "SK_ID_PREV": Integer,
        "SK_ID_CURR": Integer,
        "NAME_CONTRACT_TYPE": Categorical,
        "AMT_ANNUITY": Double,
        "AMT_APPLICATION": Double,
        "AMT_CREDIT": Double,
        "AMT_DOWN_PAYMENT": Double,
        "AMT_GOODS_PRICE": Double,
        "WEEKDAY_APPR_PROCESS_START": Categorical,
        "HOUR_APPR_PROCESS_START": Integer,
        "FLAG_LAST_APPL_PER_CONTRACT": Categorical,
        "NFLAG_LAST_APPL_IN_DAY": Integer,
        "RATE_DOWN_PAYMENT": Double,
        "RATE_INTEREST_PRIMARY": Double,
        "RATE_INTEREST_PRIVILEGED": Double,
        "NAME_CASH_LOAN_PURPOSE": Categorical,
        "NAME_CONTRACT_STATUS": Categorical,
        "DAYS_DECISION": Integer,
        "NAME_PAYMENT_TYPE": Categorical,
        "CODE_REJECT_REASON": Categorical,
        "NAME_TYPE_SUITE": Categorical,
        "NAME_CLIENT_TYPE": Categorical,
        "NAME_GOODS_CATEGORY": Categorical,
        "NAME_PORTFOLIO": Categorical,
        "NAME_PRODUCT_TYPE": Categorical,
        "CHANNEL_TYPE": Categorical,
        "SELLERPLACE_AREA": Integer,
        "NAME_SELLER_INDUSTRY": Categorical,
        "CNT_PAYMENT": Double,
        "NAME_YIELD_GROUP": Categorical,
        "PRODUCT_COMBINATION": Categorical,
        "DAYS_FIRST_DRAWING": Double,
        "DAYS_FIRST_DUE": Double,
        "DAYS_LAST_DUE_1ST_VERSION": Double,
        "DAYS_LAST_DUE": Double,
        "DAYS_TERMINATION": Double,
        "NFLAG_INSURED_ON_APPROVAL": Double,
    }
    logical_types_previous_application = {
        k: v for k, v in logical_types_previous_application.items() if k in tables["previous_application"].columns
    }
    es = es.add_dataframe(
        dataframe_name="previous_application",
        dataframe=tables["previous_application"],
        index="SK_ID_PREV",
        logical_types=logical_types_previous_application,
    )

    # POS_CASH (no PK, create synthetic index)
    logical_types_pos_cash = {
        "SK_ID_PREV": Integer,
        "SK_ID_CURR": Integer,
        "MONTHS_BALANCE": Integer,
        "CNT_INSTALMENT": Double,
        "CNT_INSTALMENT_FUTURE": Double,
        "NAME_CONTRACT_STATUS": Categorical,
        "SK_DPD": Integer,
        "SK_DPD_DEF": Integer,
    }
    logical_types_pos_cash = {
        k: v for k, v in logical_types_pos_cash.items() if k in tables["pos_cash"].columns
    }
    es = es.add_dataframe(
        dataframe_name="pos_cash",
        dataframe=tables["pos_cash"],
        index="pos_id",
        make_index=True,
        logical_types=logical_types_pos_cash,
    )

    # Installments (no PK, create synthetic index)
    logical_types_installments = {
        "SK_ID_PREV": Integer,
        "SK_ID_CURR": Integer,
        "NUM_INSTALMENT_VERSION": Double,
        "NUM_INSTALMENT_NUMBER": Integer,
        "DAYS_INSTALMENT": Double,
        "DAYS_ENTRY_PAYMENT": Double,
        "AMT_INSTALMENT": Double,
        "AMT_PAYMENT": Double,
        "days_late": Double,
        "payment_diff": Double,
        "rank": Integer,
        "inst_recent_late_mean": Double,
        "inst_recent_underpay": Double,
        "inst_late_trend": Double,
        "inst_payment_diff_max": Double,
        "inst_late_skew": Double,
    }
    logical_types_installments = {
        k: v for k, v in logical_types_installments.items() if k in tables["installments"].columns
    }
    es = es.add_dataframe(
        dataframe_name="installments",
        dataframe=tables["installments"],
        index="inst_id",
        make_index=True,
        logical_types=logical_types_installments,
    )

    # Credit card (no PK, create synthetic index)
    logical_types_credit_card = {
        "SK_ID_PREV": Integer,
        "SK_ID_CURR": Integer,
        "MONTHS_BALANCE": Integer,
        "AMT_BALANCE": Double,
        "AMT_CREDIT_LIMIT_ACTUAL": Integer,
        "AMT_DRAWINGS_ATM_CURRENT": Double,
        "AMT_DRAWINGS_CURRENT": Double,
        "AMT_DRAWINGS_OTHER_CURRENT": Double,
        "AMT_DRAWINGS_POS_CURRENT": Double,
        "AMT_INST_MIN_REGULARITY": Double,
        "AMT_PAYMENT_CURRENT": Double,
        "AMT_PAYMENT_TOTAL_CURRENT": Double,
        "AMT_RECEIVABLE_PRINCIPAL": Double,
        "AMT_RECIVABLE": Double,
        "AMT_TOTAL_RECEIVABLE": Double,
        "CNT_DRAWINGS_ATM_CURRENT": Double,
        "CNT_DRAWINGS_CURRENT": Integer,
        "CNT_DRAWINGS_OTHER_CURRENT": Double,
        "CNT_DRAWINGS_POS_CURRENT": Double,
        "CNT_INSTALMENT_MATURE_CUM": Double,
        "NAME_CONTRACT_STATUS": Categorical,
        "SK_DPD": Integer,
        "SK_DPD_DEF": Integer,
    }
    logical_types_credit_card = {
        k: v for k, v in logical_types_credit_card.items() if k in tables["credit_card"].columns
    }
    es = es.add_dataframe(
        dataframe_name="credit_card",
        dataframe=tables["credit_card"],
        index="cc_id",
        make_index=True,
        logical_types=logical_types_credit_card,
    )

    # Define relationships
    # application -> bureau
    es = es.add_relationship(
        parent_dataframe_name="application",
        parent_column_name="SK_ID_CURR",
        child_dataframe_name="bureau",
        child_column_name="SK_ID_CURR",
    )

    # bureau -> bureau_balance
    es = es.add_relationship(
        parent_dataframe_name="bureau",
        parent_column_name="SK_ID_BUREAU",
        child_dataframe_name="bureau_balance",
        child_column_name="SK_ID_BUREAU",
    )

    # application -> previous_application
    es = es.add_relationship(
        parent_dataframe_name="application",
        parent_column_name="SK_ID_CURR",
        child_dataframe_name="previous_application",
        child_column_name="SK_ID_CURR",
    )

    # application -> pos_cash (direct by SK_ID_CURR)
    es = es.add_relationship(
        parent_dataframe_name="application",
        parent_column_name="SK_ID_CURR",
        child_dataframe_name="pos_cash",
        child_column_name="SK_ID_CURR",
    )

    # previous_application -> pos_cash (by SK_ID_PREV)
    es = es.add_relationship(
        parent_dataframe_name="previous_application",
        parent_column_name="SK_ID_PREV",
        child_dataframe_name="pos_cash",
        child_column_name="SK_ID_PREV",
    )

    # application -> installments (direct by SK_ID_CURR)
    es = es.add_relationship(
        parent_dataframe_name="application",
        parent_column_name="SK_ID_CURR",
        child_dataframe_name="installments",
        child_column_name="SK_ID_CURR",
    )

    # previous_application -> installments (by SK_ID_PREV)
    es = es.add_relationship(
        parent_dataframe_name="previous_application",
        parent_column_name="SK_ID_PREV",
        child_dataframe_name="installments",
        child_column_name="SK_ID_PREV",
    )

    # application -> credit_card (direct by SK_ID_CURR)
    es = es.add_relationship(
        parent_dataframe_name="application",
        parent_column_name="SK_ID_CURR",
        child_dataframe_name="credit_card",
        child_column_name="SK_ID_CURR",
    )

    # previous_application -> credit_card (by SK_ID_PREV)
    es = es.add_relationship(
        parent_dataframe_name="previous_application",
        parent_column_name="SK_ID_PREV",
        child_dataframe_name="credit_card",
        child_column_name="SK_ID_PREV",
    )

    return es


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_featuretools_feature_store(
    data_dir: Path | str,
    y_train: pd.Series,
    output_path: Path | str | None = None,
    agg_primitives: list[str] | None = None,
    max_depth: int = 1,
    iv_threshold: float = 0.02,
    corr_threshold: float = 0.95,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, list[Any], list[str]]:
    """
    Build automated feature store via featuretools DFS.

    Loads train data, builds EntitySet, runs DFS, applies IV + correlation
    filtering, and optionally saves selected features to parquet.

    Parameters
    ----------
    data_dir : Path | str
        Directory containing application_train.csv and other CSVs.
    y_train : pd.Series
        Target series with index = SK_ID_CURR values (defines train set).
    output_path : Path | str | None, optional
        If provided, save only the selected-column subset to this parquet path.
    agg_primitives : list[str] | None, optional
        Aggregate primitives passed to DFS (default: mean, std, min, max, count, skew, median).
    max_depth : int, optional
        Maximum depth for DFS (default 1, i.e. direct aggregates only).
    iv_threshold : float, optional
        Minimum IV to include feature (default 0.02).
    corr_threshold : float, optional
        Correlation threshold for deduplication (default 0.95 per feature.md Layer 6).
    n_jobs : int, optional
        Number of jobs for DFS (default 1).

    Returns
    -------
    tuple[pd.DataFrame, list[Any], list[str]]
        - feature_matrix: Selected-column subset of DFS output (same as parquet)
        - feature_defs: Feature definitions from DFS (for apply function)
        - selected_cols: List of column names passing IV + correlation filters

    Raises
    ------
    ValueError
        If y_train is empty or iv_threshold is negative.
    FileNotFoundError
        If required CSV files are missing.
    """
    if len(y_train) == 0:
        raise ValueError("y_train cannot be empty")

    if iv_threshold < 0:
        raise ValueError(f"iv_threshold must be >= 0, got {iv_threshold}")

    data_dir = Path(data_dir)
    if agg_primitives is None:
        agg_primitives = _DEFAULT_AGG_PRIMITIVES

    # Extract SK_ID_CURR values from application_train.csv.
    # y_train may have a positional integer index (not SK_ID_CURR), so we read
    # the actual loan IDs directly from the source file and align by row order.
    app_ids_series = pd.read_csv(
        data_dir / _FILE_APP_TRAIN, usecols=["SK_ID_CURR"]
    )["SK_ID_CURR"].reset_index(drop=True)

    if len(app_ids_series) != len(y_train):
        raise ValueError(
            f"application_train.csv has {len(app_ids_series)} rows but "
            f"y_train has {len(y_train)} rows — cannot align by row order"
        )

    train_ids = app_ids_series.tolist()

    # Build a properly SK_ID_CURR indexed y for IV filtering so index alignment
    # between feature_matrix (SK_ID_CURR index) and labels is correct.
    y_indexed = pd.Series(y_train.values, index=app_ids_series.values, name=y_train.name)

    # Load and build EntitySet
    tables = _load_entity_tables(data_dir, train_ids)

    # Drop TARGET from application before building EntitySet (it's a label, not a feature)
    if "TARGET" in tables["application"].columns:
        tables["application"] = tables["application"].drop(columns=["TARGET"])

    entity_set = _build_entity_set(tables)

    # Suppress FutureWarnings from featuretools
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)

        # Run DFS
        feature_matrix, feature_defs = ft.dfs(
            entityset=entity_set,
            target_dataframe_name="application",
            agg_primitives=agg_primitives,
            trans_primitives=[],
            max_depth=max_depth,
            n_jobs=n_jobs,
            verbose=False,
        )

    # Diagnostic step: print dtype distribution after DFS (verify Woodwork fix working)
    print("Feature matrix dtypes after DFS:")
    print(feature_matrix.dtypes.value_counts())
    print(f"Total columns: {feature_matrix.shape[1]}")
    numeric_cols_initial = feature_matrix.select_dtypes(include=["number"]).columns
    print(f"Numeric columns (before filtering): {len(numeric_cols_initial)}")
    print(f"Non-numeric columns (to be dropped): {feature_matrix.shape[1] - len(numeric_cols_initial)}")

    # Post-process: numeric only, inf -> sentinel, NaN -> sentinel
    all_cols = feature_matrix.columns.tolist()
    numeric_cols = feature_matrix.select_dtypes(include=["number"]).columns.tolist()
    dropped = [c for c in all_cols if c not in numeric_cols]
    if dropped:
        warnings.warn(
            f"Dropping {len(dropped)} non-numeric DFS columns (e.g. {dropped[:3]})",
            UserWarning,
            stacklevel=2,
        )
    feature_matrix = feature_matrix[numeric_cols].copy()
    feature_matrix = feature_matrix.replace([np.inf, -np.inf], _NAN_SENTINEL)
    feature_matrix = feature_matrix.fillna(_NAN_SENTINEL)

    # Ensure index name is SK_ID_CURR (DFS preserves application index)
    feature_matrix.index.name = "SK_ID_CURR"

    # Per D-02: Skip IV filter (inappropriate for tree models).
    # Use correlation deduplication instead on all DFS features.
    all_dfs_cols = feature_matrix.columns.tolist()

    # Correlation deduplication on all DFS features
    if len(all_dfs_cols) > 1:
        corr_matrix = feature_matrix[all_dfs_cols].corr().abs()
        to_drop = set()
        for i, col_a in enumerate(all_dfs_cols):
            if col_a in to_drop:
                continue
            for col_b in all_dfs_cols[i + 1 :]:
                if col_b in to_drop:
                    continue
                if corr_matrix.loc[col_a, col_b] > corr_threshold:
                    # Keep the first one (arbitrary, but consistent)
                    to_drop.add(col_b)

        selected_cols = [c for c in all_dfs_cols if c not in to_drop]
    else:
        selected_cols = all_dfs_cols

    # Belt-and-suspenders guard: drop any remaining leaky SK_DPD columns (D-02)
    feature_matrix = feature_matrix.drop(columns=_LEAKY_SKDPD_COLS, errors='ignore')
    selected_cols = [c for c in selected_cols if c not in _LEAKY_SKDPD_COLS]

    # Resolve _x/_y duplicate columns introduced when DFS entity columns overlap
    # with hand-engineered features: keep _x (first occurrence), drop _y.
    xy_x_cols = [c for c in feature_matrix.columns if c.endswith('_x')]
    xy_y_cols = [c for c in feature_matrix.columns if c.endswith('_y')]
    xy_x_bases = {c[:-2] for c in xy_x_cols}
    xy_y_bases = {c[:-2] for c in xy_y_cols}
    paired_bases = xy_x_bases & xy_y_bases
    if paired_bases:
        rename_map = {f"{b}_x": b for b in paired_bases}
        drop_y = [f"{b}_y" for b in paired_bases]
        feature_matrix = feature_matrix.drop(columns=drop_y, errors='ignore')
        feature_matrix = feature_matrix.rename(columns=rename_map)
        selected_cols = [
            rename_map.get(c, c) for c in selected_cols if c not in drop_y
        ]

    # Drop FLAG_DOCUMENT_* columns — sparse binary flags with negligible IV for tree models
    flag_doc_cols = [c for c in feature_matrix.columns if 'FLAG_DOCUMENT' in c]
    if flag_doc_cols:
        feature_matrix = feature_matrix.drop(columns=flag_doc_cols)
        selected_cols = [c for c in selected_cols if c not in flag_doc_cols]

    # Save selected features if output_path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Add TARGET column to feature matrix before saving (required for train_xgboost_optuna)
        output_df = feature_matrix[selected_cols].copy()
        output_df['TARGET'] = y_indexed.loc[output_df.index].values
        output_df.to_parquet(output_path)
        # When saving, return only selected columns (without TARGET)
        return feature_matrix[selected_cols], feature_defs, selected_cols

    return feature_matrix, feature_defs, selected_cols


def apply_featuretools_feature_store(
    data_dir: Path | str,
    feature_defs: list[Any],
    selected_cols: list[str],
    mode: str = "test",
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Apply trained feature definitions to inference data.

    Loads test (or train) data, rebuilds EntitySet, runs DFS with the same
    feature definitions, and post-processes.

    Parameters
    ----------
    data_dir : Path | str
        Directory containing application_test.csv (or application_train.csv).
    feature_defs : list[Any]
        Feature definitions from build_featuretools_feature_store.
    selected_cols : list[str]
        Column list from build_featuretools_feature_store.
    mode : str, optional
        "test" (default) uses application_test.csv; "train" uses application_train.csv.
    n_jobs : int, optional
        Number of jobs for DFS (default 1).

    Returns
    -------
    pd.DataFrame
        Feature matrix with selected columns only, matching the training data shape
        and column order.

    Raises
    ------
    ValueError
        If feature_defs is empty or mode is not "test" or "train".
    FileNotFoundError
        If CSV files are missing.
    """
    if not feature_defs:
        raise ValueError("feature_defs cannot be empty")

    if mode not in ("test", "train"):
        raise ValueError(f'mode must be "test" or "train", got {mode}')

    if ft is None:
        raise ImportError("featuretools is required for this function")

    data_dir = Path(data_dir)

    # Choose CSV file based on mode
    if mode == "test":
        app_filename = _FILE_APP_TEST
    else:
        app_filename = _FILE_APP_TRAIN

    app_path = data_dir / app_filename
    if not app_path.exists():
        raise FileNotFoundError(f"Missing {app_filename} in {data_dir}")

    application = pd.read_csv(app_path)

    # Load all secondary tables (no filtering by train_ids)
    bureau_path = data_dir / _FILE_BUREAU
    bureau = pd.read_csv(bureau_path)

    bureau_balance_path = data_dir / _FILE_BUREAU_BAL
    bureau_balance = pd.read_csv(bureau_balance_path)

    prev_app_path = data_dir / _FILE_PREV_APP
    previous_application = pd.read_csv(prev_app_path)

    pos_cash_path = data_dir / _FILE_POS_CASH
    pos_cash = pd.read_csv(pos_cash_path)

    installments_path = data_dir / _FILE_INSTALLMENTS
    installments = pd.read_csv(installments_path)

    cc_balance_path = data_dir / _FILE_CC_BAL
    credit_card = pd.read_csv(cc_balance_path)

    # Rebuild EntitySet
    tables = {
        "application": application,
        "bureau": bureau,
        "bureau_balance": bureau_balance,
        "previous_application": previous_application,
        "pos_cash": pos_cash,
        "installments": installments,
        "credit_card": credit_card,
    }

    # Drop TARGET if present
    if "TARGET" in tables["application"].columns:
        tables["application"] = tables["application"].drop(columns=["TARGET"])

    entity_set = _build_entity_set(tables)

    # Suppress FutureWarnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)

        # Apply feature definitions
        feature_matrix = ft.calculate_feature_matrix(
            feature_defs,
            entityset=entity_set,
            n_jobs=n_jobs,
            verbose=False,
        )

    # Post-process
    numeric_cols = feature_matrix.select_dtypes(include=["number"]).columns.tolist()
    feature_matrix = feature_matrix[numeric_cols].copy()
    feature_matrix = feature_matrix.replace([np.inf, -np.inf], _NAN_SENTINEL)
    feature_matrix = feature_matrix.fillna(_NAN_SENTINEL)

    # Ensure index name
    feature_matrix.index.name = "SK_ID_CURR"

    # Return only selected columns
    return feature_matrix[selected_cols]


def deduplicate_dfs_features(
    X_dfs: pd.DataFrame,
    feature_importance: dict[str, float] | None = None,
    corr_threshold: float = 0.95,
) -> list[str]:
    """
    Identify and remove highly correlated DFS feature pairs.

    Computes the absolute correlation matrix and identifies pairs with
    |r| > threshold. For each pair, keeps the feature with higher importance
    (or the first one if importance dict is not provided).

    Parameters
    ----------
    X_dfs : pd.DataFrame
        DFS feature matrix.
    feature_importance : dict[str, float] | None, optional
        Feature importance scores (e.g. from tree importance). If provided,
        uses importance to decide which feature to drop. If None, drops the
        second feature in each correlated pair.
    corr_threshold : float, optional
        Correlation threshold for deduplication (default 0.95 per feature.md Layer 6).

    Returns
    -------
    list[str]
        Column names to keep (after deduplication).
    """
    all_cols = X_dfs.columns.tolist()

    if len(all_cols) <= 1:
        return all_cols

    corr_matrix = X_dfs.corr().abs()
    to_drop = set()

    for i, col_a in enumerate(all_cols):
        if col_a in to_drop:
            continue
        for col_b in all_cols[i + 1 :]:
            if col_b in to_drop:
                continue
            if corr_matrix.loc[col_a, col_b] > corr_threshold:
                # Drop the feature with lower importance (or col_b if not provided)
                if feature_importance is not None:
                    imp_a = feature_importance.get(col_a, 0.0)
                    imp_b = feature_importance.get(col_b, 0.0)
                    if imp_a >= imp_b:
                        to_drop.add(col_b)
                    else:
                        to_drop.add(col_a)
                else:
                    # No importance info: keep the first one
                    to_drop.add(col_b)

    return [c for c in all_cols if c not in to_drop]


def evaluate_dfs_features(
    X_raw: pd.DataFrame,
    X_dfs: pd.DataFrame,
    y: pd.Series,
    output_path: Path | str | None = None,
    n_trials: int = 50,
    corr_threshold: float = 0.95,
) -> dict[str, Any]:
    """
    Evaluate DFS features by comparing Gini on raw vs combined feature sets.

    Trains XGBoost on raw features (baseline), deduplicates DFS features,
    trains XGBoost on combined (raw + DFS), and computes Gini delta.
    Commits DFS features only if delta >= 0.01.

    Parameters
    ----------
    X_raw : pd.DataFrame
        Raw feature matrix (baseline).
    X_dfs : pd.DataFrame
        DFS-generated feature matrix.
    y : pd.Series
        Binary target series.
    output_path : Path | str | None, optional
        If provided, save evaluation results to JSON at this path.
    n_trials : int, optional
        Number of Optuna trials for XGBoost HPO (default 50).
    corr_threshold : float, optional
        Correlation threshold for DFS feature deduplication (default 0.95 per feature.md).

    Returns
    -------
    dict[str, Any]
        Keys:
        - "raw_gini": Gini coefficient on raw features
        - "dfs_gini": Gini coefficient on combined (raw + dedup DFS) features
        - "gini_delta": dfs_gini - raw_gini
        - "decision": "commit" if delta >= 0.01 else "defer"
        - "raw_features": Number of raw features
        - "dfs_features": Number of DFS features before dedup
        - "dfs_features_dedup": Number of DFS features after dedup
    """
    # Baseline: XGBoost on raw features
    model_raw, metrics_raw, X_test_raw, y_test, _ = train_xgboost_optuna(
        X=X_raw,
        y=y,
        n_trials=n_trials,
    )

    raw_gini = gini_coefficient(y_test, model_raw.predict_proba(X_test_raw)[:, 1])

    # Deduplicate DFS features (no importance info, so uses default)
    dfs_cols_dedup = deduplicate_dfs_features(
        X_dfs,
        feature_importance=None,
        corr_threshold=corr_threshold,
    )

    X_dfs_dedup = X_dfs[dfs_cols_dedup].copy()

    # Combined: raw + dedup DFS
    X_combined = pd.concat([X_raw, X_dfs_dedup], axis=1)

    # Align indices and train on combined
    model_combined, _, X_test_combined, _, _ = train_xgboost_optuna(
        X=X_combined,
        y=y,
        n_trials=n_trials,
    )

    dfs_gini = gini_coefficient(y_test, model_combined.predict_proba(X_test_combined)[:, 1])

    # Compute delta and decision
    gini_delta = dfs_gini - raw_gini
    decision = "commit" if gini_delta >= 0.01 else "defer"

    result = {
        "raw_gini": float(raw_gini),
        "dfs_gini": float(dfs_gini),
        "gini_delta": float(gini_delta),
        "decision": decision,
        "raw_features": X_raw.shape[1],
        "dfs_features": X_dfs.shape[1],
        "dfs_features_dedup": len(dfs_cols_dedup),
    }

    # Save to JSON if path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

    return result
