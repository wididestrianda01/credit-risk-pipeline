#!/usr/bin/env python3
"""
Run adversarial validation on train/test split with gating logic.

Detects distribution shift between training and test sets.
Implements decision logic from Phase 1 CONTEXT.md:
  - verdict "safe" (AUC < 0.55): proceed with no gate
  - verdict "investigate" (0.55 <= AUC < 0.65): print shifted features, block Phase 2
  - verdict "problematic" (AUC >= 0.65): log, emit WARNING, halt
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import adversarial_validation_report

def run_adversarial_validation():
    """Load train/test split and run adversarial validation with gating logic."""
    # Load feature matrix and target
    print("Loading feature store...")
    X = pd.read_parquet("data/processed/X_raw_features.parquet")
    y = pd.read_parquet("data/processed/y_train.parquet").squeeze()
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")

    # Create deterministic 80/20 stratified split (must match xgboost_raw baseline)
    print("Creating deterministic 80/20 stratified split (random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"  y_train default rate: {y_train.mean():.4f}")
    print(f"  y_test default rate: {y_test.mean():.4f}")

    # Run adversarial validation
    print("\nRunning adversarial validation (LGB 5-fold CV)...")
    report = adversarial_validation_report(
        X_train, X_test,
        output_path="reports/adversarial_validation.json",
        n_top_features=10
    )

    # Print results
    auc = report["auc"]
    verdict = report["verdict"]
    shifted_features = report["shifted_features"]

    print(f"\n{'='*70}")
    print(f"ADVERSARIAL VALIDATION REPORT")
    print(f"{'='*70}")
    print(f"AUC (train vs test classifier): {auc:.4f}")
    print(f"Verdict: {verdict.upper()}")
    print(f"Shifted features (top 10 by importance):")
    for i, feat in enumerate(shifted_features, 1):
        print(f"  {i}. {feat}")
    print(f"{'='*70}\n")

    # Implement gating logic (D-01, D-02, D-03)
    if verdict == "safe":
        print("✓ Train/test distributions are safe. Proceeding to Phase 2.")
        return

    elif verdict == "investigate":
        print("⚠️ INVESTIGATE verdict: Moderate distribution shift detected.")
        print(f"Top shifted features: {shifted_features}")
        raise ValueError(
            "Adversarial validation verdict: investigate. "
            "User sign-off required before Phase 2 proceeds."
        )

    elif verdict == "problematic":
        print("❌ PROBLEMATIC verdict: Significant distribution shift detected.")
        print(f"Shifted features (top 10): {shifted_features}")
        raise ValueError(
            "Adversarial validation verdict: problematic. "
            "Feature-level audit required before Phase 2. "
            "See reports/adversarial_validation.json for details."
        )

if __name__ == "__main__":
    try:
        run_adversarial_validation()
    except ValueError as e:
        print(f"\nBlocking error: {e}")
        sys.exit(1)
