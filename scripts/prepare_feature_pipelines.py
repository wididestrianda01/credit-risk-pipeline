#!/usr/bin/env python
"""
Prepare per-model feature matrices for Wave 2 (per-model HPO & gap closure).

Generates:
1. LGB features: raw continuous + optional target encoding variant
2. XGB features: raw continuous + optional interactions
3. CatBoost features: native categorical columns
4. DFS features: auto-generated aggregations with IV filtering

Evaluates each variant and records results to reports/feature_pipeline_comparison.json.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from conftest import *  # noqa: F401, F403

from credit_engine.features import select_features_by_iv
from credit_engine.model import (
    apply_target_encoding_fold_safe,
    filter_dfs_by_iv,
    train_lightgbm_extended_hpo,
)
from credit_engine.utils import evaluate_model


def main():
    print("=" * 80)
    print("Phase 04.1 Wave 2: Per-Model Feature Pipeline Preparation")
    print("=" * 80)

    # Load data
    print("\n[1] Loading training data...")
    X_raw = pd.read_parquet("data/processed/X_raw_features.parquet")
    X_train_orig = pd.read_parquet("data/processed/X_train.parquet")
    y = pd.read_parquet("data/processed/y_train.parquet").squeeze()

    print(f"  X_raw shape: {X_raw.shape}")
    print(f"  X_train_orig shape: {X_train_orig.shape}")
    print(f"  y shape: {y.shape}")

    # Standard train/test split (same as Wave 1 for consistency)
    from sklearn.model_selection import train_test_split

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    X_orig_tr, X_orig_te, _, _ = train_test_split(
        X_train_orig, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  X_tr shape: {X_tr.shape}, X_te shape: {X_te.shape}")
    print(f"  Positive rate: {y_tr.mean():.2%}")

    results = {}
    feature_shapes = {}

    # ===== Task 1: LightGBM Feature Pipeline (raw vs target-encoded) =====
    print("\n[2] LightGBM Feature Pipeline: Comparing raw vs target-encoded...")

    cat_cols = [
        "CODE_GENDER",
        "NAME_EDUCATION_TYPE",
        "NAME_INCOME_TYPE",
        "ORGANIZATION_TYPE",
    ]

    # Baseline: raw features
    print("  Training LGB baseline (raw features, n_trials=1)...")
    try:
        lgb_raw_model = train_lightgbm_extended_hpo(X_tr, y_tr, n_trials=1)
        lgb_raw_eval = evaluate_model(lgb_raw_model, X_te, y_te, "LGB_Raw")
        lgb_raw_gini = lgb_raw_eval["Gini"]
        print(f"    ✓ LGB raw Gini: {lgb_raw_gini:.4f}")
    except Exception as e:
        print(f"    ✗ Error training LGB raw: {e}")
        lgb_raw_gini = None

    # With target encoding
    if "CODE_GENDER" in X_orig_tr.columns:
        print("  Training LGB with target encoding (n_trials=1)...")
        try:
            X_tr_te, X_te_te = apply_target_encoding_fold_safe(
                X_orig_tr[cat_cols], y_tr, X_orig_te[cat_cols], cat_cols
            )
            X_tr_with_te = pd.concat([X_tr, X_tr_te], axis=1)
            X_te_with_te = pd.concat([X_te, X_te_te], axis=1)

            lgb_te_model = train_lightgbm_extended_hpo(X_tr_with_te, y_tr, n_trials=1)
            lgb_te_eval = evaluate_model(lgb_te_model, X_te_with_te, y_te, "LGB_TE")
            lgb_te_gini = lgb_te_eval["Gini"]
            print(f"    ✓ LGB with TE Gini: {lgb_te_gini:.4f}")

            # Keep best
            if lgb_te_gini > lgb_raw_gini:
                X_lgb_final = X_tr_with_te
                lgb_best = "target_encoded"
                lgb_final_gini = lgb_te_gini
                lgb_delta = lgb_te_gini - lgb_raw_gini
                print(f"    → Selected: target_encoded (+{lgb_delta:.4f})")
            else:
                X_lgb_final = X_tr
                lgb_best = "raw"
                lgb_final_gini = lgb_raw_gini
                lgb_delta = 0.0
                print(f"    → Selected: raw (TE delta: {lgb_te_gini - lgb_raw_gini:.4f})")

        except Exception as e:
            print(f"    ✗ Error with target encoding: {e}")
            X_lgb_final = X_tr
            lgb_best = "raw"
            lgb_final_gini = lgb_raw_gini
            lgb_te_gini = None
            lgb_delta = None
    else:
        X_lgb_final = X_tr
        lgb_best = "raw"
        lgb_final_gini = lgb_raw_gini
        lgb_te_gini = None
        lgb_delta = None

    X_lgb_final.to_parquet("data/processed/X_lgb_features.parquet", index=False)
    feature_shapes["X_lgb_features"] = X_lgb_final.shape
    results["lightgbm"] = {
        "raw_gini": float(lgb_raw_gini) if lgb_raw_gini else None,
        "te_gini": float(lgb_te_gini) if lgb_te_gini else None,
        "selected": lgb_best,
        "final_gini": float(lgb_final_gini) if lgb_final_gini else None,
        "te_delta": float(lgb_delta) if lgb_delta else None,
    }

    # ===== Task 2: XGBoost Feature Pipeline =====
    print("\n[3] XGBoost Feature Pipeline: Using raw features...")
    X_xgb_final = X_tr.copy()
    X_xgb_final.to_parquet("data/processed/X_xgb_features.parquet", index=False)
    feature_shapes["X_xgb_features"] = X_xgb_final.shape
    results["xgboost"] = {"pipeline": "raw_continuous_63", "shape": list(X_xgb_final.shape)}

    # ===== Task 3: CatBoost Feature Pipeline =====
    print("\n[4] CatBoost Feature Pipeline: Using native categorical features...")
    if "CODE_GENDER" in X_orig_tr.columns:
        X_cat_tr = pd.concat([X_tr, X_orig_tr[cat_cols]], axis=1)
    else:
        X_cat_tr = X_tr.copy()
    X_cat_tr.to_parquet("data/processed/X_cat_features.parquet", index=False)
    feature_shapes["X_cat_features"] = X_cat_tr.shape
    results["catboost"] = {
        "pipeline": "raw_continuous_63_plus_native_categorical_4",
        "shape": list(X_cat_tr.shape),
    }

    # ===== Task 4: DFS Features (Optional) =====
    print("\n[5] DFS Features: Auto-generated aggregations with IV filter...")
    print("  Note: Full DFS on 307K rows is memory-intensive.")
    print("  In production, run: python scripts/generate_dfs_features.py")
    print("  For now, creating placeholder...")

    # Placeholder for DFS features (full run deferred to production script)
    dfs_placeholder = {"status": "deferred", "reason": "memory_intensive_on_full_dataset"}

    results["dfs"] = dfs_placeholder
    results["feature_matrix_shapes"] = {
        k: list(v) if isinstance(v, tuple) else v for k, v in feature_shapes.items()
    }

    # Save comparison results
    print("\n[6] Saving results to reports/feature_pipeline_comparison.json...")
    output_path = Path("reports/feature_pipeline_comparison.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"  ✓ Saved to {output_path}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"LightGBM:")
    print(f"  Raw Gini:         {results['lightgbm']['raw_gini']}")
    print(f"  TE Gini:          {results['lightgbm']['te_gini']}")
    print(f"  Selected:         {results['lightgbm']['selected']}")
    print(f"  Delta:            {results['lightgbm']['te_delta']}")
    print(f"\nXGBoost:")
    print(f"  Pipeline:         {results['xgboost']['pipeline']}")
    print(f"  Shape:            {results['xgboost']['shape']}")
    print(f"\nCatBoost:")
    print(f"  Pipeline:         {results['catboost']['pipeline']}")
    print(f"  Shape:            {results['catboost']['shape']}")
    print(f"\nDFS:")
    print(f"  Status:           {results['dfs']['status']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
