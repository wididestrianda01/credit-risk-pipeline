"""
auto_features.py
----------------
Featuretools Deep Feature Synthesis (DFS) auto-aggregation.
Generates features from the 7-table relational structure without
manual specification of aggregate functions.

Entry points:
  - build_featuretools_feature_store: DFS on train data, IV + correlation filter
  - apply_featuretools_feature_store: Apply feature definitions to test data
"""

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import featuretools as ft
except ImportError:
    ft = None  # type: ignore

from credit_engine.features import select_features_by_iv

# Constants
_DEFAULT_AGG_PRIMITIVES = ["mean", "std", "min", "max", "count", "skew", "median"]
_NAN_SENTINEL = -999.0

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

    installments_path = data_dir / _FILE_INSTALLMENTS
    installments = pd.read_csv(installments_path)
    installments = installments[installments["SK_ID_CURR"].isin(train_ids)]

    cc_balance_path = data_dir / _FILE_CC_BAL
    credit_card = pd.read_csv(cc_balance_path)
    credit_card = credit_card[credit_card["SK_ID_CURR"].isin(train_ids)]

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

    es = ft.EntitySet(id="home_credit")

    # Application (primary table, has SK_ID_CURR as PK)
    es = es.add_dataframe(
        dataframe_name="application",
        dataframe=tables["application"],
        index="SK_ID_CURR",
    )

    # Bureau (has SK_ID_BUREAU as PK)
    es = es.add_dataframe(
        dataframe_name="bureau",
        dataframe=tables["bureau"],
        index="SK_ID_BUREAU",
    )

    # Bureau balance (no PK, create synthetic index)
    es = es.add_dataframe(
        dataframe_name="bureau_balance",
        dataframe=tables["bureau_balance"],
        index="bbal_id",
        make_index=True,
    )

    # Previous application (has SK_ID_PREV as PK)
    es = es.add_dataframe(
        dataframe_name="previous_application",
        dataframe=tables["previous_application"],
        index="SK_ID_PREV",
    )

    # POS_CASH (no PK, create synthetic index)
    es = es.add_dataframe(
        dataframe_name="pos_cash",
        dataframe=tables["pos_cash"],
        index="pos_id",
        make_index=True,
    )

    # Installments (no PK, create synthetic index)
    es = es.add_dataframe(
        dataframe_name="installments",
        dataframe=tables["installments"],
        index="inst_id",
        make_index=True,
    )

    # Credit card (no PK, create synthetic index)
    es = es.add_dataframe(
        dataframe_name="credit_card",
        dataframe=tables["credit_card"],
        index="cc_id",
        make_index=True,
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
    corr_threshold: float = 0.90,
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
        Correlation threshold for deduplication (default 0.90).
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

    # Load and build EntitySet
    train_ids = list(y_train.index)
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

    # Post-process: numeric only, inf -> 0, NaN -> sentinel
    numeric_cols = feature_matrix.select_dtypes(include=["number"]).columns.tolist()
    feature_matrix = feature_matrix[numeric_cols].copy()
    feature_matrix = feature_matrix.replace([np.inf, -np.inf], _NAN_SENTINEL)
    feature_matrix = feature_matrix.fillna(_NAN_SENTINEL)

    # Ensure index name is SK_ID_CURR (DFS preserves application index)
    feature_matrix.index.name = "SK_ID_CURR"

    # Apply IV filter (suppress print output from select_features_by_iv)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        iv_dict = select_features_by_iv(
            feature_matrix,
            y_train,
            min_iv=iv_threshold,
            bins=10,
        )

    iv_cols = list(iv_dict.keys())

    # Correlation deduplication on IV-filtered features only
    if len(iv_cols) > 1:
        corr_matrix = feature_matrix[iv_cols].corr().abs()
        to_drop = set()
        for i, col_a in enumerate(iv_cols):
            if col_a in to_drop:
                continue
            for col_b in iv_cols[i + 1 :]:
                if col_b in to_drop:
                    continue
                if corr_matrix.loc[col_a, col_b] > corr_threshold:
                    to_drop.add(col_b)  # keep col_a (higher IV comes first)

        selected_cols = [c for c in iv_cols if c not in to_drop]
    else:
        selected_cols = iv_cols

    # Save selected features if output_path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        feature_matrix[selected_cols].to_parquet(output_path)
        # When saving, return only selected columns
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
    feature_defs : list
        Feature definitions from build_featuretools_feature_store.
    selected_cols : list[str]
        Selected column names (from build output).
    mode : str, optional
        "test" (default) or "train".
    n_jobs : int, optional
        Number of jobs for DFS (default 1).

    Returns
    -------
    pd.DataFrame
        Feature matrix with columns = selected_cols, no NaN/inf values.

    Raises
    ------
    ValueError
        If mode is not "test" or "train".
    """
    if mode not in ("test", "train"):
        raise ValueError(f"mode must be 'test' or 'train', got {mode!r}")

    data_dir = Path(data_dir)

    # Load application data
    if mode == "train":
        app_path = data_dir / _FILE_APP_TRAIN
    else:  # mode == "test"
        app_path = data_dir / _FILE_APP_TEST

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
