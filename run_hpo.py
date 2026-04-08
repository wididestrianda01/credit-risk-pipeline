#!/usr/bin/env python3
"""
Run 50-trial XGBoost Optuna HPO with explicit output and error handling.
"""
import sys
from pathlib import Path

# Setup imports
sys.path.insert(0, str(Path.cwd()))
import src
sys.modules["credit_engine"] = src

import json
import time
from credit_engine.model import train_xgboost_optuna

feature_store_path = "data/processed/X_tree_dfs.parquet"
start_time = time.time()

print("[START] 50-trial XGBoost Optuna HPO", flush=True)
print(f"[INFO] Feature store: {feature_store_path}", flush=True)
print(f"[INFO] Expected runtime: ~2–3 hours", flush=True)
print(f"[INFO] Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
print("", flush=True)

try:
    print("[EXEC] Calling train_xgboost_optuna(n_trials=50)...", flush=True)
    model, metrics, X_test, y_test, best_params, oof_predictions = train_xgboost_optuna(
        feature_store_path=feature_store_path,
        n_trials=50,
        groups=None,
    )

    elapsed = time.time() - start_time
    print("\n[COMPLETE] 50-TRIAL HPO FINISHED", flush=True)
    print(f"[INFO] Duration: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)", flush=True)
    print(f"[INFO] End time: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("", flush=True)

    # Print metrics
    print("[METRICS]", flush=True)
    for key in ['oof_gini', 'oot_gini', 'Gini', 'AUC-ROC', 'BrierSkill', 'KS', 'Brier', 'AvgPrecision', 'best_trial']:
        val = metrics.get(key)
        if val is not None:
            if isinstance(val, (int, float)):
                print(f"  {key}: {val:.4f}", flush=True)
            else:
                print(f"  {key}: {val}", flush=True)

    # Save final metrics
    with open('reports/xgboost_raw_eval.json', 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print("", flush=True)
    print("[SAVE] Final metrics saved to reports/xgboost_raw_eval.json", flush=True)
    print("[SUCCESS] HPO execution complete", flush=True)

except Exception as e:
    print(f"\n[ERROR] HPO FAILED: {e}", flush=True)
    import traceback
    traceback.print_exc(file=sys.stdout)
    sys.exit(1)
