#!/usr/bin/env python3
"""
Rebuild X_raw_features.parquet from X_train.parquet.

This script addresses the common pattern where test runs on mock data
silently overwrite production parquet via relative path writes.
It regenerates the feature store from the canonical X_train.parquet.
"""

import pandas as pd
from pathlib import Path

def rebuild_feature_store():
    """Load X_train.parquet and save as X_raw_features.parquet."""
    data_dir = Path("data/processed")

    # Load the canonical training data
    print("Loading X_train.parquet...")
    X_train = pd.read_parquet(data_dir / "X_train.parquet")
    print(f"  Shape: {X_train.shape}")

    # Save as X_raw_features.parquet
    output_path = data_dir / "X_raw_features.parquet"
    print(f"Saving to {output_path}...")
    X_train.to_parquet(output_path)

    # Verify
    X_verify = pd.read_parquet(output_path)
    print(f"  Verified shape: {X_verify.shape}")
    print(f"  Columns: {X_verify.shape[1]}")

    assert X_verify.shape[0] >= 100_000, f"FAILED: {X_verify.shape[0]} < 100K"
    print(f"✓ Feature store rebuilt successfully: {X_verify.shape}")

if __name__ == "__main__":
    rebuild_feature_store()
