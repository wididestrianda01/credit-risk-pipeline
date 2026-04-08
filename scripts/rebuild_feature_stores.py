#!/usr/bin/env python3
"""
Rebuild X_tree_raw.parquet and X_tree_dfs.parquet with leakage removal.

This script orchestrates the rebuild of the two main feature stores:
1. X_tree_raw: Hand-engineered raw features from CSVs
2. X_tree_dfs: DFS auto-aggregates merged with X_tree_raw

Both stores include the leakage guard (drop _LEAKY_SKDPD_COLS) before saving.
"""

import sys
from pathlib import Path

# Setup path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

import conftest  # Setup credit_engine alias
import pandas as pd
from data_loader import load_data
from features import build_tree_feature_store
from auto_features import build_featuretools_feature_store, _LEAKY_SKDPD_COLS

def main():
    print("=" * 80)
    print("REBUILDING FEATURE STORES (X_tree_raw + X_tree_dfs)")
    print("=" * 80)

    # Step 1: Load data
    print("\n[1/3] Loading data from data/...")
    X_train, y_train = load_data('data/')
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")

    # Step 2: Rebuild X_tree_raw
    print("\n[2/3] Building X_tree_raw (hand-engineered features)...")
    X_tree_raw, feature_cols_raw = build_tree_feature_store(
        X_train, y_train, output_dir='data/processed/'
    )
    print(f"  X_tree_raw shape: {X_tree_raw.shape}")
    print(f"  Feature columns: {len(feature_cols_raw)}")

    # Verify no leaky columns in raw store
    leaked_raw = [c for c in X_tree_raw.columns if c in _LEAKY_SKDPD_COLS]
    if leaked_raw:
        print(f"  WARNING: Found leaky columns in X_tree_raw: {leaked_raw}")
    else:
        print(f"  OK: No leaky columns in X_tree_raw")

    # Verify temporal sort column present
    if 'prev_days_decision_mean' not in X_tree_raw.columns:
        print(f"  ERROR: Missing temporal sort column 'prev_days_decision_mean'")
        return 1
    print(f"  OK: Temporal sort column 'prev_days_decision_mean' present")

    # Step 3: Rebuild X_tree_dfs (DFS merged with X_tree_raw)
    print("\n[3/3] Building X_tree_dfs (DFS merged with X_tree_raw)...")

    # Build DFS features
    print("  [3.1] Running DFS on entity set...")
    X_dfs, feature_defs, selected_cols = build_featuretools_feature_store(
        data_dir='data/',
        y_train=y_train,
        output_path=None,  # Don't save yet; we need to merge first
        max_depth=1,
        iv_threshold=0.02,
        corr_threshold=0.90,
        n_jobs=1
    )
    print(f"  DFS output shape (before merge): {X_dfs.shape}")

    # Load X_tree_raw from disk (already saved in Step 2)
    print("  [3.2] Loading X_tree_raw for merge...")
    X_tree_raw_loaded = pd.read_parquet('data/processed/X_tree_raw.parquet')
    print(f"  X_tree_raw shape: {X_tree_raw_loaded.shape}")

    # Apply leakage guard to X_tree_raw before merge
    print("  [3.3] Applying leakage guard to X_tree_raw...")
    X_tree_raw_clean = X_tree_raw_loaded.drop(columns=_LEAKY_SKDPD_COLS, errors='ignore')
    print(f"  X_tree_raw shape after guard: {X_tree_raw_clean.shape}")

    # Merge DFS with X_tree_raw on SK_ID_CURR index
    print("  [3.4] Merging DFS with X_tree_raw...")
    # Both should have SK_ID_CURR as index
    X_merged = X_dfs.merge(
        X_tree_raw_clean,
        left_index=True,
        right_index=True,
        how='left'
    )
    print(f"  Merged shape: {X_merged.shape}")

    # Verify row count preserved
    if X_merged.shape[0] != 307511:
        print(f"  ERROR: Row loss during merge: {X_merged.shape[0]} != 307511")
        return 1
    print(f"  OK: Row count preserved (307511 rows)")

    # Apply final leakage guard
    print("  [3.5] Applying final leakage guard to merged output...")
    X_merged_clean = X_merged.drop(columns=_LEAKY_SKDPD_COLS, errors='ignore')

    # Verify no leaky columns
    leaked_merged = [c for c in X_merged_clean.columns if c in _LEAKY_SKDPD_COLS]
    if leaked_merged:
        print(f"  ERROR: Leaky columns in merged store: {leaked_merged}")
        return 1
    print(f"  OK: No leaky columns in merged output")

    # Verify temporal sort column present
    if 'prev_days_decision_mean' not in X_merged_clean.columns:
        print(f"  ERROR: Missing temporal sort column in merged output")
        return 1
    print(f"  OK: Temporal sort column present in merged output")

    # Save merged parquet
    print("  [3.6] Saving X_tree_dfs.parquet...")
    output_path = Path('data/processed/X_tree_dfs.parquet')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Add TARGET column to merged output (required by train_xgboost_optuna)
    y_indexed = pd.Series(y_train.values, index=X_train.index.tolist(), name='TARGET')
    X_merged_with_target = X_merged_clean.copy()
    X_merged_with_target['TARGET'] = y_indexed.loc[X_merged_clean.index].values
    X_merged_with_target.to_parquet(output_path)
    print(f"  Saved to {output_path}")
    print(f"  Final shape (with TARGET): {X_merged_with_target.shape}")

    print("\n" + "=" * 80)
    print("REBUILD COMPLETE")
    print("=" * 80)
    print(f"X_tree_raw.parquet: {Path('data/processed/X_tree_raw.parquet').stat().st_size / 1e6:.1f} MB")
    print(f"X_tree_dfs.parquet:  {Path('data/processed/X_tree_dfs.parquet').stat().st_size / 1e6:.1f} MB")

    return 0

if __name__ == '__main__':
    sys.exit(main())
