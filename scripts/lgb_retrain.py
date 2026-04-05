"""
lgb_retrain.py
--------------
Retrain LightGBM after the scale_pos_weight fix.
Measures impact on Gini coefficient compared to baseline models.

REQUIREMENTS:
- Production feature store must be pre-built: data/processed/X_features.parquet
- If X_features is stale (< 307511 rows), regenerate it first with rebuild_feature_store.py

EXPECTED GINI IMPROVEMENT:
- Old LGB (is_unbalance=True): ~0.4465
- New LGB (scale_pos_weight):   ~0.53-0.56
- Improvement:                  +0.08 to +0.11
"""

import sys
from pathlib import Path

import pandas as pd
import json

# Setup imports - conftest pattern
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import src
sys.modules['credit_engine'] = src

from src.model import train_lightgbm_optuna

# Ensure output directory exists
Path("reports").mkdir(exist_ok=True)

print("=" * 70)
print("LightGBM Retrain: scale_pos_weight Cost-Sensitive Fix")
print("=" * 70)
print()

# Load production data
print("Loading production data...")
X_path = Path('data/processed/X_features.parquet')
y_path = Path('data/processed/y_train.parquet')

if not X_path.exists():
    print(f"ERROR: {X_path} not found")
    print("  Please run: python scripts/rebuild_feature_store.py")
    sys.exit(1)

X_final = pd.read_parquet(X_path)
y = pd.read_parquet(y_path).squeeze()

print(f"  X shape: {X_final.shape}")
print(f"  y shape: {y.shape}")
print(f"  Default rate: {y.mean():.4f} ({y.sum()} positives, {(y==0).sum()} negatives)")

# Validate that we have production data (not mock)
if X_final.shape[0] < 100000:
    print()
    print("WARNING: Feature store appears to be mock data (< 100K rows)")
    print(f"  Current: {X_final.shape[0]} rows")
    print("  Please regenerate with: python scripts/rebuild_feature_store.py")
    sys.exit(1)

print()
print("=" * 70)
print("Training LightGBM with Optuna (n_trials=50)")
print("=" * 70)
print()

try:
    lgb_model, lgb_metrics, lgb_X_test, lgb_y_test, lgb_best_params = train_lightgbm_optuna(
        X_final, y, n_trials=50
    )

    print()
    print("=" * 70)
    print("RESULTS: LightGBM Retrain (scale_pos_weight Fix)")
    print("=" * 70)
    print()
    print(f"Gini:        {lgb_metrics['Gini']:.4f}")
    print(f"AUC-ROC:     {lgb_metrics['AUC-ROC']:.4f}")
    print(f"KS:          {lgb_metrics['KS']:.4f}")
    print(f"Brier:       {lgb_metrics['Brier']:.4f}")
    print(f"BrierSkill:  {lgb_metrics['BrierSkill']:.4f}")
    print(f"AvgPrecision:{lgb_metrics['AvgPrecision']:.4f}")
    print()

    print("Baseline Comparisons:")
    print(f"  LR baseline        (0.4890): {lgb_metrics['Gini'] - 0.4890:+.4f}")
    print(f"  Old LGB            (0.4465): {lgb_metrics['Gini'] - 0.4465:+.4f}")
    print(f"  XGBoost            (0.5470): {lgb_metrics['Gini'] - 0.5470:+.4f}")
    print()

    # Evaluate improvement
    improvement_over_old_lgb = lgb_metrics['Gini'] - 0.4465
    improvement_pct = (improvement_over_old_lgb / 0.4465) * 100

    if lgb_metrics['Gini'] >= 0.53:
        status = "✓ EXCELLENT"
    elif lgb_metrics['Gini'] >= 0.51:
        status = "✓ GOOD"
    elif lgb_metrics['Gini'] >= 0.50:
        status = "⚠ MARGINAL"
    else:
        status = "✗ BELOW EXPECTED"

    print(f"Status: {status}")
    print(f"  New Gini: {lgb_metrics['Gini']:.4f}")
    print(f"  Improvement vs old: {improvement_over_old_lgb:+.4f} ({improvement_pct:+.1f}%)")
    print()

    print("Best Hyperparameters:")
    for key, val in lgb_best_params.items():
        if isinstance(val, float):
            print(f"  {key:25s}: {val:.6f}" if val < 1 else f"  {key:25s}: {val:.4f}")
        else:
            print(f"  {key:25s}: {val}")
    print()

    # Save results
    results = {
        "model": "LightGBM_scale_pos_weight_fix",
        "timestamp": pd.Timestamp.now().isoformat(),
        "gini": lgb_metrics['Gini'],
        "auc_roc": lgb_metrics['AUC-ROC'],
        "ks": lgb_metrics['KS'],
        "brier": lgb_metrics['Brier'],
        "brier_skill": lgb_metrics['BrierSkill'],
        "avg_precision": lgb_metrics['AvgPrecision'],
        "best_params": lgb_best_params,
        "comparison": {
            "vs_lr_baseline": lgb_metrics['Gini'] - 0.4890,
            "vs_old_lgb": lgb_metrics['Gini'] - 0.4465,
            "vs_xgboost": lgb_metrics['Gini'] - 0.5470,
        },
        "improvement": {
            "vs_old_lgb_absolute": improvement_over_old_lgb,
            "vs_old_lgb_percent": improvement_pct,
        },
        "test_set_stats": {
            "n_test": len(lgb_y_test),
            "default_rate": float(lgb_y_test.mean()),
        },
        "feature_store": {
            "shape": list(X_final.shape),
            "features": list(X_final.columns),
        }
    }

    results_path = Path('reports/lgb_retrain_results.json')
    with results_path.open('w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")
    print()

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
