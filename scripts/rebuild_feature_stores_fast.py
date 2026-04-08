#!/usr/bin/env python3
"""
Fast rebuild of X_tree_raw.parquet and X_tree_dfs.parquet.

Uses pre-built X_train.parquet instead of reloading from CSVs.
"""

import sys
from pathlib import Path

# Setup path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

import conftest  # Setup credit_engine alias
import pandas as pd
from features import build_tree_feature_store
from auto_features import build_featuretools_feature_store, _LEAKY_SKDPD_COLS

def main():
    print("=" * 80)
    print("FAST REBUILD OF FEATURE STORES")
    print("=" * 80)

    # Step 1: Load from existing X_train.parquet + y_train.parquet
    print("\n[1/3] Loading X_train and y_train from parquets...")
    X_train = pd.read_parquet('data/processed/X_train.parquet')
    y_train = pd.read_parquet('data/processed/y_train.parquet').squeeze()
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")

    # Step 2: Build X_tree_raw
    print("\n[2/3] Building X_tree_raw (hand-engineered features)...")
    X_tree_raw, feature_cols_raw = build_tree_feature_store(
        X_train, y_train, output_dir='data/processed/'
    )
    print(f"  X_tree_raw shape: {X_tree_raw.shape}")
    print(f"  Feature columns: {len(feature_cols_raw)}")

    # Verify temporal sort column present
    if 'prev_days_decision_mean' not in X_tree_raw.columns:
        print(f"  ERROR: Missing 'prev_days_decision_mean'")
        return 1
    print(f"  OK: Temporal sort column present")

    # Step 3: Build X_tree_dfs (DFS merged with X_tree_raw)
    print("\n[3/3] Building X_tree_dfs (DFS + X_tree_raw merge)...")

    print("  [3.1] Running DFS...")
    X_dfs, feature_defs, selected_cols = build_featuretools_feature_store(
        data_dir='data/',
        y_train=y_train,
        output_path=None,  # Don't save yet
        max_depth=1,
        iv_threshold=0.02,
        corr_threshold=0.90,
        n_jobs=1
    )
    print(f"  DFS shape: {X_dfs.shape}")

    # Load X_tree_raw from disk
    print("  [3.2] Loading X_tree_raw for merge...")
    X_tree_raw_loaded = pd.read_parquet('data/processed/X_tree_raw.parquet')

    # Apply leakage guard before merge
    print("  [3.3] Applying leakage guards...")
    X_tree_raw_clean = X_tree_raw_loaded.drop(columns=_LEAKY_SKDPD_COLS, errors='ignore')
    X_dfs_clean = X_dfs.drop(columns=_LEAKY_SKDPD_COLS, errors='ignore')

    # Merge
    print("  [3.4] Merging DFS with X_tree_raw...")
    X_merged = X_dfs_clean.merge(
        X_tree_raw_clean,
        left_index=True,
        right_index=True,
        how='left'
    )
    print(f"  Merged shape: {X_merged.shape}")

    if X_merged.shape[0] != 307511:
        print(f"  ERROR: Row loss: {X_merged.shape[0]} != 307511")
        return 1
    print(f"  OK: Row count preserved")

    # Final guard
    print("  [3.5] Applying final leakage guard...")
    X_final = X_merged.drop(columns=_LEAKY_SKDPD_COLS, errors='ignore')

    leaked = [c for c in X_final.columns if c in _LEAKY_SKDPD_COLS]
    if leaked:
        print(f"  ERROR: Leaky columns: {leaked}")
        return 1
    print(f"  OK: No leaky columns ({X_final.shape[1]} columns)")

    if 'prev_days_decision_mean' not in X_final.columns:
        print(f"  ERROR: Missing temporal column")
        return 1
    print(f"  OK: Temporal column present")

    # Save
    print("  [3.6] Saving X_tree_dfs.parquet...")
    output_path = Path('data/processed/X_tree_dfs.parquet')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add TARGET column (required by train_xgboost_optuna)
    # X_final has SK_ID_CURR as index (from merge), y_train has positional index
    # We need to align by position first
    X_with_target = X_final.copy()
    # Reset both to get positional alignment
    X_positions = list(range(len(X_final)))
    y_positions = y_train.iloc[X_positions].values
    X_with_target['TARGET'] = y_positions
    X_with_target.to_parquet(output_path)
    print(f"  Saved {output_path}")

    print("\n" + "=" * 80)
    print("REBUILD COMPLETE")
    print("=" * 80)

    return 0

if __name__ == '__main__':
    sys.exit(main())
