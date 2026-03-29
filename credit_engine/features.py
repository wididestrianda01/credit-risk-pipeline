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

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full feature engineering pipeline."""
    # TODO: call individual feature functions in order
    return df
