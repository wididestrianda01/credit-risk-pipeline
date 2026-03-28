"""
data_loader.py
--------------
Joins the 7 raw source tables into a single modelling DataFrame
and enforces correct dtypes throughout.

Tables (update paths as needed)
--------------------------------
1. applications      – loan application details
2. bureau            – credit bureau summary
3. bureau_balance    – monthly bureau balance history
4. previous_app      – prior application history
5. pos_cash          – POS / cash loan monthly snapshots
6. installments      – instalment payment history
7. credit_card_bal   – credit card balance snapshots

Usage
-----
    from credit_engine.data_loader import load_data
    df = load_data(data_dir="data/raw/")
"""

from pathlib import Path
import pandas as pd


def load_data(data_dir: str | Path) -> pd.DataFrame:
    """Join all source tables and return a clean modelling DataFrame."""
    data_dir = Path(data_dir)
    # TODO: implement joins and dtype enforcement
    raise NotImplementedError
