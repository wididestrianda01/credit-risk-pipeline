#!/usr/bin/env python3
"""
Performance evaluation for Priority 1 & 2 improvements.

Regenerates feature store from production data, trains LightGBM and XGBoost
with Optuna HPO, runs ensemble workflow, and summarizes results.

Execution: python scripts/eval_priority12.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

# Set up working directory and path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Create the credit_engine alias (required for internal imports in model.py, features.py, etc.)
import src
sys.modules["credit_engine"] = src

# Now import modules
from src import features, model


def main():
    """Run full evaluation pipeline."""
    print("=" * 80)
    print("PRIORITY 1 & 2 EVALUATION: Feature Store + HPO + Ensemble")
    print("=" * 80)
    print()

    # --- Step 1: Load production data ---
    print("Step 1: Loading production data...")
    X_raw = pd.read_parquet(project_root / "data/processed/X_train.parquet")
    y = pd.read_parquet(project_root / "data/processed/y_train.parquet").squeeze()
    print(f"  Loaded X_raw: {X_raw.shape}")
    print(f"  Loaded y: {y.shape}")
    print()

    # --- Step 2: Regenerate feature store ---
    print("Step 2: Regenerating feature store...")
    X_final, woe_mappings = features.build_feature_store(X_raw, y)
    print(f"  Feature store shape: {X_final.shape}")
    print(f"  Features ({len(X_final.columns)}): {list(X_final.columns)}")
    print(f"  WoE mappings persisted to: models/woe_mappings.pkl")
    print(f"  Feature matrix persisted to: data/processed/X_features.parquet")
    print()

    # --- Step 3: Train LightGBM with Optuna HPO ---
    print("Step 3: Training LightGBM with Optuna HPO (30 trials)...")
    lgb_model, lgb_metrics, X_test_lgb, y_test_lgb, lgb_best_params = (
        model.train_lightgbm_optuna(X_final, y, n_trials=30)
    )
    print(f"  ✓ LightGBM trained")
    print(f"    Gini:        {lgb_metrics['Gini']:.4f}")
    print(f"    AUC-ROC:     {lgb_metrics['AUC-ROC']:.4f}")
    print(f"    KS:          {lgb_metrics['KS']:.4f}")
    print(f"    Brier:       {lgb_metrics['Brier']:.4f}")
    print(f"    BrierSkill:  {lgb_metrics['BrierSkill']:.4f}")
    print(f"    AvgPrecision:{lgb_metrics['AvgPrecision']:.4f}")
    print(f"  Best params: {lgb_best_params}")
    print(f"  Model saved to: models/lightgbm_best.pkl")
    print(f"  Figure saved to: reports/figures/lightgbm_roc_pr.png")
    print()

    # --- Step 4: Train XGBoost with Optuna HPO ---
    print("Step 4: Training XGBoost with Optuna HPO (30 trials)...")
    xgb_model, xgb_metrics, X_test_xgb, y_test_xgb, xgb_best_params = (
        model.train_xgboost_optuna(X_final, y, n_trials=30)
    )
    print(f"  ✓ XGBoost trained")
    print(f"    Gini:        {xgb_metrics['Gini']:.4f}")
    print(f"    AUC-ROC:     {xgb_metrics['AUC-ROC']:.4f}")
    print(f"    KS:          {xgb_metrics['KS']:.4f}")
    print(f"    Brier:       {xgb_metrics['Brier']:.4f}")
    print(f"    BrierSkill:  {xgb_metrics['BrierSkill']:.4f}")
    print(f"    AvgPrecision:{xgb_metrics['AvgPrecision']:.4f}")
    print(f"  Best params: {xgb_best_params}")
    print(f"  Model saved to: models/xgboost_best.pkl")
    print(f"  Figure saved to: reports/figures/xgboost_roc_pr.png")
    print()

    # --- Step 5: Run ensemble workflow ---
    print("Step 5: Running ensemble workflow...")
    ensemble_result = model.run_ensemble_workflow(
        X_final, y, lgb_params=lgb_best_params, xgb_params=xgb_best_params
    )
    print(f"  ✓ Ensemble trained")
    print(f"    LightGBM Gini:   {ensemble_result['lgb_gini']:.4f}")
    print(f"    XGBoost Gini:    {ensemble_result['xgb_gini']:.4f}")
    print(f"    Ensemble Gini:   {ensemble_result['ensemble_gini']:.4f}")
    print(f"    Improvement:     {ensemble_result['improvement']:.4f}")
    print(f"    Persisted:       {ensemble_result['persisted']}")
    print()

    # --- Step 6: Build summary results ---
    print("=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)

    summary = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "feature_store": {
            "shape": X_final.shape,
            "columns": len(X_final.columns),
            "feature_names": list(X_final.columns),
        },
        "lightgbm": {
            "gini": float(lgb_metrics["Gini"]),
            "auc_roc": float(lgb_metrics["AUC-ROC"]),
            "ks": float(lgb_metrics["KS"]),
            "brier": float(lgb_metrics["Brier"]),
            "brier_skill": float(lgb_metrics["BrierSkill"]),
            "avg_precision": float(lgb_metrics["AvgPrecision"]),
            "best_params": lgb_best_params,
        },
        "xgboost": {
            "gini": float(xgb_metrics["Gini"]),
            "auc_roc": float(xgb_metrics["AUC-ROC"]),
            "ks": float(xgb_metrics["KS"]),
            "brier": float(xgb_metrics["Brier"]),
            "brier_skill": float(xgb_metrics["BrierSkill"]),
            "avg_precision": float(xgb_metrics["AvgPrecision"]),
            "best_params": xgb_best_params,
        },
        "ensemble": {
            "lgb_gini": float(ensemble_result["lgb_gini"]),
            "xgb_gini": float(ensemble_result["xgb_gini"]),
            "ensemble_gini": float(ensemble_result["ensemble_gini"]),
            "improvement": float(ensemble_result["improvement"]),
            "persisted": bool(ensemble_result["persisted"]),
        },
    }

    # Print formatted table
    print("\n" + "=" * 80)
    print("METRIC COMPARISON TABLE")
    print("=" * 80)
    print()
    print(f"{'Model':<20} {'Gini':<12} {'AUC-ROC':<12} {'KS':<12} {'BrierSkill':<12}")
    print("-" * 68)
    print(
        f"{'LightGBM':<20} {lgb_metrics['Gini']:>11.4f} {lgb_metrics['AUC-ROC']:>11.4f} "
        f"{lgb_metrics['KS']:>11.4f} {lgb_metrics['BrierSkill']:>11.4f}"
    )
    print(
        f"{'XGBoost':<20} {xgb_metrics['Gini']:>11.4f} {xgb_metrics['AUC-ROC']:>11.4f} "
        f"{xgb_metrics['KS']:>11.4f} {xgb_metrics['BrierSkill']:>11.4f}"
    )
    print(
        f"{'Ensemble':<20} {ensemble_result['ensemble_gini']:>11.4f} {'—':<11} "
        f"{'—':<11} {'—':<11}"
    )
    print()

    # Improvement analysis
    best_single = max(lgb_metrics["Gini"], xgb_metrics["Gini"])
    ensemble_gini = ensemble_result["ensemble_gini"]
    improvement = ensemble_gini - best_single

    print("ENSEMBLE PERFORMANCE")
    print("-" * 68)
    print(f"Best single model Gini:     {best_single:.4f}")
    print(f"Ensemble Gini:              {ensemble_gini:.4f}")
    print(f"Improvement:                {improvement:+.4f}")
    print(f"Ensemble persisted:         {ensemble_result['persisted']}")
    print()

    # --- Step 7: Save results ---
    results_path = project_root / "reports/priority12_eval_results.json"
    print(f"Saving results to: {results_path}")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print("✓ Results saved")
    print()

    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
