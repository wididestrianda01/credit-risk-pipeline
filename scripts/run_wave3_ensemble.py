#!/usr/bin/env python3
"""
Wave 3: Run both ensemble variants and select winner.
Saves comparison to reports/ensemble_variants_comparison.json.
"""
import pandas as pd
import numpy as np
import json
import sys

# Add project root to path for imports
sys.path.insert(0, ".")
from conftest import *
from credit_engine.model import train_ensemble_variant_a, train_ensemble_variant_b


def main():
    # Load feature matrices from Wave 2 (these are training fold only from 80/20 split)
    print("Loading per-model feature matrices from Wave 2...")
    X_lgb = pd.read_parquet("data/processed/X_lgb_features.parquet")
    X_xgb = pd.read_parquet("data/processed/X_xgb_features.parquet")
    X_cat = pd.read_parquet("data/processed/X_cat_features.parquet")

    # Load full target and feature matrices to perform split consistently
    y_full = pd.read_parquet("data/processed/y_train.parquet").squeeze()
    X_raw = pd.read_parquet("data/processed/X_raw_features.parquet")

    # Perform same 80/20 split to get aligned y_tr and y_te
    from sklearn.model_selection import train_test_split
    X_dummy_tr, X_dummy_te, y_tr, y_te = train_test_split(
        X_raw, y_full, test_size=0.2, random_state=42, stratify=y_full
    )
    # Now y_tr aligns with the 246K rows in X_lgb/X_xgb/X_cat

    # For Logistic Regression, use X_lgb (target-encoded categorical features)
    X_lr = X_lgb.copy()

    print(f"X_lgb shape: {X_lgb.shape}")
    print(f"X_xgb shape: {X_xgb.shape}")
    print(f"X_cat shape: {X_cat.shape}")
    print(f"X_lr shape: {X_lr.shape}")
    print(f"y_tr shape: {y_tr.shape}")
    print(f"y_te shape: {y_te.shape}")

    # Run Variant A (4-model logistic stack)
    print("\n" + "=" * 80)
    print("Running Ensemble Variant A (4-model logistic stack)...")
    print("=" * 80)
    try:
        meta_a, metrics_a = train_ensemble_variant_a(X_lgb, X_xgb, X_cat, X_lr, y_tr)
        print(f"\nVariant A Results:")
        print(f"  LGB Gini: {metrics_a['lgb_gini']:.4f}")
        print(f"  XGB Gini: {metrics_a['xgb_gini']:.4f}")
        print(f"  CatBoost Gini: {metrics_a['cat_gini']:.4f}")
        print(f"  LR Gini: {metrics_a['lr_gini']:.4f}")
        print(f"  Ensemble Gini: {metrics_a['ensemble_gini']:.4f}")
        print(f"  Improvement: {metrics_a['improvement']:+.4f}")
        print(f"  Persisted: {metrics_a['persisted']}")
    except Exception as e:
        print(f"ERROR in Variant A: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Run Variant B (3-model Ridge stack)
    print("\n" + "=" * 80)
    print("Running Ensemble Variant B (3-model Ridge stack)...")
    print("=" * 80)
    try:
        meta_b, metrics_b = train_ensemble_variant_b(X_lgb, X_xgb, X_cat, y_tr)
        print(f"\nVariant B Results:")
        print(f"  LGB Gini: {metrics_b['lgb_gini']:.4f}")
        print(f"  XGB Gini: {metrics_b['xgb_gini']:.4f}")
        print(f"  CatBoost Gini: {metrics_b['cat_gini']:.4f}")
        print(f"  Ensemble Gini: {metrics_b['ensemble_gini']:.4f}")
        print(f"  Improvement: {metrics_b['improvement']:+.4f}")
        print(f"  Persisted: {metrics_b['persisted']}")
        print(f"  Meta-Learner Alpha: {metrics_b['meta_alpha']}")
    except Exception as e:
        print(f"ERROR in Variant B: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Select winner
    print("\n" + "=" * 80)
    if metrics_a["ensemble_gini"] > metrics_b["ensemble_gini"]:
        winner = "variant_a"
        best_metrics = metrics_a
        print(f"WINNER: Ensemble Variant A")
    else:
        winner = "variant_b"
        best_metrics = metrics_b
        print(f"WINNER: Ensemble Variant B")
    print(f"Best Ensemble Gini: {best_metrics['ensemble_gini']:.4f}")
    print(f"Improvement over best base: {best_metrics['improvement']:+.4f}")
    print("=" * 80)

    # Save comparison
    comparison = {
        "variant_a": metrics_a,
        "variant_b": metrics_b,
        "winner": winner,
        "best_ensemble_gini": best_metrics["ensemble_gini"],
        "best_improvement": best_metrics["improvement"],
        "persisted": best_metrics["persisted"],
        "reason": f"Variant {winner.split('_')[1].upper()} Gini {best_metrics['ensemble_gini']:.4f} > "
        + (
            f"Variant B Gini {metrics_b['ensemble_gini']:.4f}"
            if winner == "variant_a"
            else f"Variant A Gini {metrics_a['ensemble_gini']:.4f}"
        ),
    }

    with open("reports/ensemble_variants_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    print(f"\nComparison saved to reports/ensemble_variants_comparison.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
