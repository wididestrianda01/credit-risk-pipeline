"""
rebuild_feature_store.py
------------------------
Rebuild the production feature store from X_train.parquet (307,511 rows).
This must be done before retraining models after any feature engineering changes.
"""

import sys
from pathlib import Path

# Ensure we're in the right directory
import os
os.chdir(Path(__file__).parent.parent)

import pandas as pd

# Direct import from src module
sys.path.insert(0, str(Path.cwd() / 'src'))
from features import build_feature_store

# Ensure output directory exists
Path("reports").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

print("=" * 70)
print("Rebuild Feature Store (Production Data)")
print("=" * 70)
print()

# Load raw training data
print("Loading raw training data...")
X_train = pd.read_parquet('data/processed/X_train.parquet')
y_train = pd.read_parquet('data/processed/y_train.parquet').squeeze()

print(f"  X_train shape: {X_train.shape}")
print(f"  y_train shape: {y_train.shape}")
print(f"  Default rate: {y_train.mean():.4f} ({y_train.sum()} positives)")
print()

# Build feature store
print("Building feature store...")
print("  Engineering features...")
print("  Computing IV values...")
print("  Binning and WoE transforming...")
print("  Applying variance and correlation filters...")
print()

try:
    X_final, woe_mappings = build_feature_store(X_train, y_train)

    print()
    print("=" * 70)
    print("Feature Store Rebuild Complete")
    print("=" * 70)
    print(f"Final feature matrix: {X_final.shape}")
    print(f"  Columns: {list(X_final.columns)[:10]}...")
    print(f"  WoE mappings saved: {len(woe_mappings)} features")
    print()

    # Verify no NaNs
    nan_count = X_final.isna().sum().sum()
    print(f"Validation:")
    print(f"  NaN values in X_final: {nan_count} ✓" if nan_count == 0 else f"  NaN values in X_final: {nan_count} ✗")
    print()

    print("Artifacts persisted:")
    print("  ✓ data/processed/X_features.parquet")
    print("  ✓ models/woe_mappings.pkl")
    print()

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
