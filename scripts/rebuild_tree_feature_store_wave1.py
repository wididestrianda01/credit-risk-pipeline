#!/usr/bin/env python3
"""
Rebuild X_tree_raw.parquet with Wave 1 features integrated.

This script orchestrates the rebuild of the tree feature store with the following
new Wave 1 delinquency trajectory features:
1. inst_late_rate_12m
2. inst_late_rate_recent_vs_historical
3. inst_rolling_30dpd_ratio_3m
4. inst_delinquency_escalation_flag
5. inst_days_since_last_30dpd
6. bureau_dpd_trend_3m_vs_12m
7. bureau_debt_to_new_credit
"""

import sys
from pathlib import Path

# Setup path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

import pandas as pd
from data_loader import build_training_frame
from features import build_tree_feature_store

def main():
    print("=" * 80)
    print("REBUILDING X_tree_raw.parquet WITH WAVE 1 FEATURES")
    print("=" * 80)

    # Step 1: Load data
    print("\n[1/3] Loading data from data/...")
    data_dir = project_root / 'data'
    X_train, y_train = build_training_frame(str(data_dir))
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")

    # Step 2: Load raw instalment_payments for Wave 1 features
    print("\n[2/3] Loading raw installments_payments table...")
    df_inst = pd.read_csv(data_dir / "installments_payments.csv")
    print(f"  installments_payments shape: {df_inst.shape}")

    # Step 3: Rebuild X_tree_raw with Wave 1 features
    print("\n[3/3] Building X_tree_raw (hand-engineered features + Wave 1)...")
    X_tree_raw, feature_cols_raw = build_tree_feature_store(
        X_train, y_train,
        output_dir=str(project_root / 'data' / 'processed'),
        df_inst=df_inst
    )
    print(f"  X_tree_raw shape: {X_tree_raw.shape}")
    print(f"  Feature columns: {len(feature_cols_raw)}")

    # Step 4: Verify Wave 1 features are present
    print("\n[4/4] Verifying Wave 1 features...")
    expected_wave1 = [
        'inst_late_rate_12m',
        'inst_late_rate_recent_vs_historical',
        'inst_rolling_30dpd_ratio_3m',
        'inst_delinquency_escalation_flag',
        'inst_days_since_last_30dpd',
        'bureau_dpd_trend_3m_vs_12m',
        'bureau_debt_to_new_credit'
    ]

    missing_wave1 = [c for c in expected_wave1 if c not in X_tree_raw.columns]
    if missing_wave1:
        print(f"  ✗ Missing Wave 1 columns: {missing_wave1}")
        return 1
    else:
        print(f"  ✓ All 7 Wave 1 features present")
        for col in expected_wave1:
            sentinel_count = (X_tree_raw[col] == -999.0).sum()
            nan_count = X_tree_raw[col].isna().sum()
            non_sentinel = X_tree_raw[col][(X_tree_raw[col] != -999.0) & (X_tree_raw[col].notna())].count()
            print(f"    {col}: {sentinel_count} sentinel, {nan_count} NaN, {non_sentinel} valid")

    print("\n" + "=" * 80)
    print(f"SUCCESS: X_tree_raw.parquet rebuilt with {X_tree_raw.shape[1]} columns")
    print(f"Output: data/processed/X_tree_raw.parquet ({X_tree_raw.shape})")
    print("=" * 80)
    return 0

if __name__ == "__main__":
    sys.exit(main())
