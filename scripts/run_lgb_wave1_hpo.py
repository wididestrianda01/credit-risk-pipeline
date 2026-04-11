#!/usr/bin/env python
"""Run LGB HPO on Wave 1 rebuilt feature store and evaluate gate.

Phase 04.2.7 — Wave 4 gate evaluation.
Gate: OOT Gini >= 0.5845 (+0.005 over LGB baseline 0.5795).
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
OUTPUT_REPORT = "reports/lgb_wave1_eval.json"
N_TRIALS = 50
BASELINE_OOT_GINI = 0.5795
GATE_THRESHOLD = 0.5845

WAVE1_FEATURES = [
    "inst_late_rate_12m",
    "inst_late_rate_recent_vs_historical",
    "inst_rolling_30dpd_ratio_3m",
    "inst_delinquency_escalation_flag",
    "inst_days_since_last_30dpd",
    "bureau_dpd_trend_3m_vs_12m",
    "bureau_debt_to_new_credit",
]

print("=" * 60)
print("LGB HPO — WAVE 1 GATE EVALUATION (Phase 04.2.7)")
print("=" * 60)
print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Feature store: {FEATURE_STORE}")
print(f"Trials: {N_TRIALS}")
print(f"Gate threshold: OOT Gini >= {GATE_THRESHOLD}")
print()

# --- Pre-flight checks ---
mem = psutil.virtual_memory()
available_gb = mem.available / 1e9
print(f"Available RAM: {available_gb:.1f} GB ({100 - mem.percent:.0f}% free)")
assert available_gb >= 6.0, f"Insufficient RAM: need >=6 GB, have {available_gb:.1f} GB"

X_check = pd.read_parquet(FEATURE_STORE)
print(f"Feature store shape: {X_check.shape}")
missing = [c for c in WAVE1_FEATURES if c not in X_check.columns]
assert not missing, f"Wave 1 columns missing: {missing}"
print(f"Wave 1 columns: all 7 present")
print(f"Memory usage: {X_check.memory_usage(deep=True).sum() / 1e9:.2f} GB")
del X_check
gc.collect()
print()

# --- Run HPO ---
from src.model import train_lightgbm_optuna, _OPTUNA_DB_PATH
import optuna

# Compute remaining trials to allow safe resume after interruption
_study_name = "lgb_wave1_143col_scale_pos_weight"
# Fresh study — 143-col Wave 1 store; incompatible with prior 141-col study.
# Never resume the old "lgb_raw_Xtreeraw_scale_pos_weight" study (22 trials on stale store).
remaining = N_TRIALS
print(f"New study '{_study_name}': running {remaining} trials")

start_time = time.time()

model, metrics, X_test, y_test, best_params = train_lightgbm_optuna(
    feature_store_path=FEATURE_STORE,
    n_trials=remaining,
    imbalance_strategy="scale_pos_weight",
)

elapsed_min = (time.time() - start_time) / 60
print(f"\nHPO complete in {elapsed_min:.1f} minutes")

# --- Gate evaluation ---
oot_gini = metrics.get("oot_gini", 0.0)
oof_gini = metrics.get("oof_gini", None)
ks = metrics.get("KS", metrics.get("ks", None))
brier = metrics.get("Brier", metrics.get("brier", None))
improvement = oot_gini - BASELINE_OOT_GINI
gate_pass = oot_gini >= GATE_THRESHOLD

print()
print("=" * 60)
print("WAVE 1 GATE EVALUATION")
print("=" * 60)
print(f"OOT Gini:        {oot_gini:.4f}")
print(f"OOF Gini:        {oof_gini:.4f}" if oof_gini else "OOF Gini:        N/A")
print(f"KS:              {ks:.4f}" if ks else "KS:              N/A")
print(f"Brier:           {brier:.4f}" if brier else "Brier:           N/A")
print(f"Baseline (LGB):  {BASELINE_OOT_GINI:.4f}")
print(f"Gate threshold:  {GATE_THRESHOLD:.4f}")
print(f"Improvement:     {improvement:+.4f}")
print(f"Gate:            {'PASS ✓' if gate_pass else 'FAIL ✗'}")
print()

# --- Save report ---
report = {
    "phase": "04.2.7",
    "wave": 1,
    "feature_store": FEATURE_STORE,
    "model": "lightgbm",
    "n_trials": N_TRIALS,
    "baseline_oot_gini": BASELINE_OOT_GINI,
    "oot_gini": oot_gini,
    "oof_gini": oof_gini,
    "ks": ks,
    "brier": brier,
    "improvement": improvement,
    "gate_threshold": GATE_THRESHOLD,
    "gate_pass": gate_pass,
    "best_params": best_params,
    "elapsed_minutes": round(elapsed_min, 1),
}

Path(OUTPUT_REPORT).parent.mkdir(exist_ok=True)
with open(OUTPUT_REPORT, "w") as f:
    json.dump(report, f, indent=2)

print(f"Report saved: {OUTPUT_REPORT}")

# --- Ceiling analysis if gate fails ---
if not gate_pass:
    import lightgbm as lgb

    print("\nGate FAILED — running feature importance ceiling analysis...")
    try:
        booster = model.estimator_.booster_ if hasattr(model, "estimator_") else model.booster_
        importances = pd.DataFrame(
            {
                "feature": booster.feature_name(),
                "importance_gain": booster.feature_importance(importance_type="gain"),
                "importance_split": booster.feature_importance(importance_type="split"),
            }
        ).sort_values("importance_gain", ascending=False).reset_index(drop=True)

        wave1_imp = importances[importances["feature"].isin(WAVE1_FEATURES)].copy()
        wave1_imp["rank"] = wave1_imp.index + 1

        ceiling_path = "reports/wave1_ceiling_analysis.md"
        with open(ceiling_path, "w") as f:
            f.write("# Wave 1 Feature Ceiling Analysis\n\n")
            f.write(f"**Gate Result:** FAIL\n\n")
            f.write(f"**OOT Gini:** {oot_gini:.4f} (target: {GATE_THRESHOLD}, baseline: {BASELINE_OOT_GINI}, delta: {improvement:+.4f})\n\n")
            f.write("## Wave 1 Feature Importances (gain-ranked among all features)\n\n")
            f.write(wave1_imp[["feature", "rank", "importance_gain", "importance_split"]].to_markdown(index=False))
            f.write("\n\n## Next Step\n\nProceed to Phase 04.2.6 (Ensemble) — stacking may extract remaining lift via model diversity.\n")

        print(f"Ceiling analysis saved: {ceiling_path}")
        print("\nWave 1 feature rankings:")
        print(wave1_imp[["feature", "rank", "importance_gain"]].to_string(index=False))
    except Exception as e:
        print(f"Warning: could not compute ceiling analysis: {e}")

print("\nDone.")
