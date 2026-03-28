"""
test_features.py
----------------
Unit tests for credit_engine/features.py.

Run with
--------
    pytest tests/test_features.py -v
"""

import pandas as pd
import pytest
from credit_engine.features import build_features


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


def test_build_features_returns_dataframe(sample_df):
    result = build_features(sample_df)
    assert isinstance(result, pd.DataFrame)


def test_build_features_no_rows_dropped(sample_df):
    result = build_features(sample_df)
    assert len(result) == len(sample_df)


# TODO: add tests for each individual feature function
