#!/usr/bin/env python3
"""
Verification script for build_tree_feature_store() function.
Loads production data and builds the raw feature matrix.
"""
import pandas as pd
import sys
from pathlib import Path

# Import from credit_engine alias
try:
    from credit_engine.features import build_tree_feature_store
except ImportError:
    # Fallback: add src to path manually using portable path construction
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    from src.features import build_tree_feature_store

print("Loading input data...")
X_train = pd.read_parquet("data/processed/X_train.parquet")
y_train = pd.read_parquet("data/processed/y_train.parquet").squeeze()

print(f"Input X_train shape: {X_train.shape}")
print(f"Input y_train shape: {y_train.shape}")
print(f"Input has NaN values: {X_train.isna().sum().sum()}")

print("\nCalling build_tree_feature_store()...")
X_tree, cols = build_tree_feature_store(X_train, y_train)

print(f"\nOutput X_tree shape: {X_tree.shape}")
print(f"Output columns: {len(cols)}")
print(f"Output has NaN values: {X_tree.isna().sum().sum()}")
print(f"Output has inf values: {(X_tree == float('inf')).sum().sum() + (X_tree == float('-inf')).sum().sum()}")

# Verify output file exists
import os
if os.path.exists("data/processed/X_tree_raw.parquet"):
    file_size = os.path.getsize("data/processed/X_tree_raw.parquet")
    print(f"\nOutput file created: data/processed/X_tree_raw.parquet")
    print(f"File size: {file_size / 1024 / 1024:.1f} MB")

    # Verify by reading back
    X_verify = pd.read_parquet("data/processed/X_tree_raw.parquet")
    print(f"Verification read shape: {X_verify.shape}")
    print(f"Verification NaN count: {X_verify.isna().sum().sum()}")
else:
    print("ERROR: Output file not created!")
    sys.exit(1)

# Check for WoE columns
woe_cols = [c for c in X_tree.columns if '_woe' in c.lower()]
if woe_cols:
    print(f"\nERROR: Found WoE columns in output: {woe_cols}")
    sys.exit(1)
else:
    print("\nSUCCESS: No WoE columns found in output")

print("\n=== VERIFICATION PASSED ===")
