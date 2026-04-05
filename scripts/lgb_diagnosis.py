#!/usr/bin/env python
"""
Diagnostic script: isolate is_unbalance effect on LightGBM Gini.

Hypothesis: is_unbalance=True shifts leaf output values to over-index on
positive class, compressing majority class probability range → hurts Gini.

Expected result: scale_pos_weight should outperform is_unbalance=True.
"""

import sys
import json as _json
from pathlib import Path

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils import gini_coefficient

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
REPORT_DIR = Path(__file__).parent.parent / "reports"

BEST_PARAMS = {
    "num_leaves": 21,
    "max_depth": 11,
    "learning_rate": 0.04819169994726929,
    "n_estimators": 243,
    "min_child_samples": 57,
    "subsample": 0.6487149710509457,
    "colsample_bytree": 0.6477091967603724,
    "reg_alpha": 2.0870697236084186,
    "reg_lambda": 1.3102005475277416,
}

RANDOM_STATE = 42
TEST_SIZE = 0.2

# =============================================================================
# PART 1: DIAGNOSIS — is_unbalance vs scale_pos_weight vs no handling
# =============================================================================

print("=" * 80)
print("PART 1: Isolating is_unbalance effect")
print("=" * 80)

# Load feature store
X_final = pd.read_parquet(DATA_DIR / "X_features.parquet")
y = pd.read_parquet(DATA_DIR / "y_train.parquet").squeeze()

print(f"\nDataset shape: X={X_final.shape}, y={y.shape}")
print(f"Default rate: {y.mean():.2%}")

# Stratified train/test split (identical to train_lightgbm_optuna)
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

print(f"\nTrain split: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Test split:  X_test={X_test.shape}, y_test={y_test.shape}")

n_neg = int((y_train == 0).sum())
n_pos = int((y_train == 1).sum())
scale_pos_weight = n_neg / n_pos

print(f"\nImbalance ratio (n_neg / n_pos): {scale_pos_weight:.2f}")

# *** Model 1: is_unbalance=True (current behavior) ***
print("\n" + "-" * 80)
print("Training Model 1: is_unbalance=True")
print("-" * 80)

m_imbal = lgb.LGBMClassifier(
    **BEST_PARAMS,
    is_unbalance=True,
    verbosity=-1,
    random_state=RANDOM_STATE,
)
m_imbal.fit(X_train, y_train)
y_prob_imbal = m_imbal.predict_proba(X_test)[:, 1]
g_imbal = gini_coefficient(y_test, y_prob_imbal)

print(f"Gini (is_unbalance=True):  {g_imbal:.4f}")
print(f"Prob range: [{y_prob_imbal.min():.4f}, {y_prob_imbal.max():.4f}]")
print(f"Mean prob:  {y_prob_imbal.mean():.4f}")

# *** Model 2: scale_pos_weight (XGBoost-style) ***
print("\n" + "-" * 80)
print(f"Training Model 2: scale_pos_weight={scale_pos_weight:.2f}")
print("-" * 80)

m_spw = lgb.LGBMClassifier(
    **BEST_PARAMS,
    scale_pos_weight=scale_pos_weight,
    verbosity=-1,
    random_state=RANDOM_STATE,
)
m_spw.fit(X_train, y_train)
y_prob_spw = m_spw.predict_proba(X_test)[:, 1]
g_spw = gini_coefficient(y_test, y_prob_spw)

print(f"Gini (scale_pos_weight):   {g_spw:.4f}")
print(f"Prob range: [{y_prob_spw.min():.4f}, {y_prob_spw.max():.4f}]")
print(f"Mean prob:  {y_prob_spw.mean():.4f}")

# *** Model 3: No imbalance handling (baseline) ***
print("\n" + "-" * 80)
print("Training Model 3: No imbalance handling")
print("-" * 80)

m_raw = lgb.LGBMClassifier(
    **BEST_PARAMS,
    verbosity=-1,
    random_state=RANDOM_STATE,
)
m_raw.fit(X_train, y_train)
y_prob_raw = m_raw.predict_proba(X_test)[:, 1]
g_raw = gini_coefficient(y_test, y_prob_raw)

print(f"Gini (no handling):        {g_raw:.4f}")
print(f"Prob range: [{y_prob_raw.min():.4f}, {y_prob_raw.max():.4f}]")
print(f"Mean prob:  {y_prob_raw.mean():.4f}")

# Summary
print("\n" + "=" * 80)
print("DIAGNOSIS SUMMARY")
print("=" * 80)
print(f"Gini (is_unbalance=True):  {g_imbal:.4f} ← current behavior")
print(f"Gini (scale_pos_weight):   {g_spw:.4f}")
print(f"Gini (no handling):        {g_raw:.4f}")

best_imbalance_strategy = None
best_gini_diagnosis = g_imbal
if g_spw > best_gini_diagnosis:
    best_gini_diagnosis = g_spw
    best_imbalance_strategy = "scale_pos_weight"
if g_raw > best_gini_diagnosis:
    best_gini_diagnosis = g_raw
    best_imbalance_strategy = "none"

if best_imbalance_strategy != "none":
    print(f"\n✓ RECOMMENDATION: Switch to {best_imbalance_strategy} strategy")
else:
    print(f"\n✓ RECOMMENDATION: Use no imbalance handling (current is_unbalance is hurting)")

diagnosis_result = {
    "gini_is_unbalance": float(g_imbal),
    "gini_scale_pos_weight": float(g_spw),
    "gini_no_handling": float(g_raw),
    "best_strategy": best_imbalance_strategy,
    "root_cause": (
        "is_unbalance=True adjusts both loss weights and leaf output values, "
        "shifting predicted probabilities toward positive class and compressing "
        "majority class probability range. This hurts Gini (rank-based metric). "
        "scale_pos_weight only adjusts gradient weights, preserving probability separation."
    ),
}

# =============================================================================
# PART 2: RETRAIN with best strategy + Optuna (n_trials=50)
# =============================================================================

print("\n" + "=" * 80)
print("PART 2: Retraining LightGBM with Optuna (n_trials=50)")
print("=" * 80)

# Import the training function from src.model
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model import train_lightgbm_optuna

print("\nCalling train_lightgbm_optuna(X, y, n_trials=50)...")
print("This may take 10-20 minutes...")

lgb_model, lgb_metrics, lgb_X_test, lgb_y_test, lgb_best_params = (
    train_lightgbm_optuna(X_final, y, n_trials=50)
)

print("\n" + "-" * 80)
print("Retraining results:")
print("-" * 80)
print(f"Gini:          {lgb_metrics['Gini']:.4f}")
print(f"AUC-ROC:       {lgb_metrics['AUC-ROC']:.4f}")
print(f"KS:            {lgb_metrics['KS']:.4f}")
print(f"BrierSkill:    {lgb_metrics['BrierSkill']:.4f}")
print(f"\nBest hyperparameters:")
for key, val in lgb_best_params.items():
    if isinstance(val, float):
        print(f"  {key}: {val:.6f}")
    else:
        print(f"  {key}: {val}")

retrain_result = {
    "gini": float(lgb_metrics["Gini"]),
    "auc_roc": float(lgb_metrics["AUC-ROC"]),
    "ks": float(lgb_metrics["KS"]),
    "brier_skill": float(lgb_metrics["BrierSkill"]),
    "best_params": lgb_best_params,
}

# =============================================================================
# PART 3: SAVE RESULTS
# =============================================================================

print("\n" + "=" * 80)
print("PART 3: Saving results")
print("=" * 80)

REPORT_DIR.mkdir(parents=True, exist_ok=True)
results_path = REPORT_DIR / "lgb_diagnosis_results.json"

full_results = {
    "diagnosis": diagnosis_result,
    "retrained": retrain_result,
}

with results_path.open("w") as fh:
    _json.dump(full_results, fh, indent=2)

print(f"\n✓ Results saved to: {results_path}")

# Final summary
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"\nDiagnosis:")
print(f"  - is_unbalance=True Gini:    {g_imbal:.4f}")
print(f"  - scale_pos_weight Gini:     {g_spw:.4f}")
print(f"  - No handling Gini:          {g_raw:.4f}")
print(f"  - Best strategy:             {best_imbalance_strategy}")

print(f"\nRetrained LightGBM (n_trials=50):")
print(f"  - Gini:                      {lgb_metrics['Gini']:.4f}")
print(f"  - AUC-ROC:                   {lgb_metrics['AUC-ROC']:.4f}")

baseline_gini = 0.489
print(f"\nComparison to LR baseline (Gini={baseline_gini:.4f}):")
print(f"  - Improvement: {lgb_metrics['Gini'] - baseline_gini:+.4f}")

if lgb_metrics["Gini"] >= baseline_gini:
    print(f"\n✓ SUCCESS: Retrained LGB exceeds LR baseline")
else:
    print(f"\n⚠ CONCERN: Retrained LGB still below LR baseline")
    print(f"  This may indicate WoE feature representation ceiling for tree models.")
