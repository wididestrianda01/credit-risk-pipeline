#!/usr/bin/env python
"""Run CatBoost HPO on Wave 1 rebuilt feature store — Basel CRE36.54 compliant.

Phase 04.2.5.1 — CatBoost compliant re-run.
Prerequisite: reports/lgb_compliant_eval.json must exist (Phase 04.2.4.1 done).
"""
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import psutil

FEATURE_STORE = "data/processed/X_tree_raw.parquet"
OUTPUT_REPORT = "reports/catboost_compliant_eval.json"
N_TRIALS = 50

print("=" * 60)
print("CatBoost HPO — BASEL CRE36.54 COMPLIANT RE-RUN (Phase 04.2.5.1)")
print("=" * 60)
print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Feature store: {FEATURE_STORE}")
print(f"Trials: {N_TRIALS}")
print()

# --- Pre-flight: confirm Phase 04.2.4.1 is complete ---
lgb_report = Path("reports/lgb_compliant_eval.json")
assert lgb_report.exists(), (
    "reports/lgb_compliant_eval.json not found — Phase 04.2.4.1 must complete first"
)
with open(lgb_report) as f:
    lgb_result = json.load(f)
lgb_oot = lgb_result.get("oot_gini", 0)
assert lgb_oot > 0.40, f"LGB compliant re-run not valid (OOT Gini={lgb_oot:.4f})"
print(f"LGB Phase 04.2.4.1 confirmed: OOT Gini={lgb_oot:.4f}")
print()

# --- Memory check ---
mem = psutil.virtual_memory()
available_gb = mem.available / 1e9
print(f"Available RAM: {available_gb:.1f} GB ({100 - mem.percent:.0f}% free)")
assert available_gb >= 6.0, f"Insufficient RAM: need >=6 GB, have {available_gb:.1f} GB"

# --- Verify feature store ---
X_check = pd.read_parquet(FEATURE_STORE)
print(f"Feature store shape: {X_check.shape}")
assert X_check.shape[0] > 300_000, "Feature store not rebuilt — run Phase 04.2.7 first"
assert "TARGET" in X_check.columns, "TARGET column missing from feature store"
wave1 = [
    "inst_late_rate_12m",
    "inst_late_rate_recent_vs_historical",
    "inst_rolling_30dpd_ratio_3m",
    "inst_delinquency_escalation_flag",
    "inst_days_since_last_30dpd",
    "bureau_dpd_trend_3m_vs_12m",
    "bureau_debt_to_new_credit",
]
missing = [c for c in wave1 if c not in X_check.columns]
assert not missing, f"Wave 1 columns missing: {missing}"
print(f"Wave 1 columns: all 7 present")
print(f"Memory usage: {X_check.memory_usage(deep=True).sum() / 1e9:.2f} GB")
del X_check
gc.collect()
print()

# --- Run compliant HPO ---
from src.model import train_catboost_optuna

start_time = time.time()

model, metrics, X_test, y_test, best_params = train_catboost_optuna(
    feature_store_path=FEATURE_STORE,
    n_trials=N_TRIALS,
)

elapsed_min = (time.time() - start_time) / 60
print(f"\nHPO complete in {elapsed_min:.1f} minutes")

oof_gini = metrics.get("oof_gini", 0.0)
oot_gini = metrics.get("oot_gini", 0.0)
gap = oof_gini - oot_gini
ks = metrics.get("KS", metrics.get("ks"))
brier = metrics.get("Brier", metrics.get("brier"))
brier_skill = metrics.get("BrierSkill", metrics.get("brier_skill"))

print()
print("=" * 60)
print("PHASE 04.2.5.1 RESULTS")
print("=" * 60)
print(f"OOF Gini:     {oof_gini:.4f}")
print(f"OOT Gini:     {oot_gini:.4f}")
print(f"OOF-OOT gap:  {gap:.4f}  {'✓' if gap <= 0.05 else '✗ OVERFIT WARNING'}")
print(f"KS:           {ks:.4f}" if ks else "KS:           N/A")
print(f"Brier:        {brier:.4f}" if brier else "Brier:        N/A")
print(f"BrierSkill:   {brier_skill:.4f}" if brier_skill else "BrierSkill:   N/A")
print(f"LGB baseline: {lgb_oot:.4f}")
print(f"CatBoost vs LGB: {oot_gini - lgb_oot:+.4f}")
print()

# Leakage guard
assert oot_gini <= 0.85, (
    f"OOT Gini={oot_gini:.4f} suspiciously high — STOP and investigate leakage"
)

# --- Save report ---
report = {
    "phase": "04.2.5.1",
    "model": "catboost",
    "feature_store": FEATURE_STORE,
    "n_trials": N_TRIALS,
    "oot_gini": oot_gini,
    "oof_gini": oof_gini,
    "oof_oot_gap": gap,
    "ks": ks,
    "brier": brier,
    "brier_skill": brier_skill,
    "lgb_oot_gini": lgb_oot,
    "best_params": best_params,
    "elapsed_minutes": round(elapsed_min, 1),
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
}

Path(OUTPUT_REPORT).parent.mkdir(exist_ok=True)
with open(OUTPUT_REPORT, "w") as f:
    json.dump(report, f, indent=2)

print(f"Report saved: {OUTPUT_REPORT}")
print("Done.")
