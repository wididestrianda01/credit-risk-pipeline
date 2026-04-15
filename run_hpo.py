#!/usr/bin/env python
"""Run full 50-trial XGBoost HPO on enriched feature store."""
import json
import sys
import time
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

# Import after path setup
from src.model import train_xgboost_optuna

print("=" * 60)
print("FULL 50-TRIAL XGBOOST HPO ON ENRICHED STORE")
print("=" * 60)
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Feature store: data/processed/X_tree_dfs.parquet")
print(f"Optuna study: xgboost_raw_v2 (resumable)")
print(f"Trials: 50 (continuing from prior runs if any)")
print()

start_time = time.time()

try:
    # Run full 50-trial HPO
    model, metrics_dict, X_test, y_test, best_params, oof_predictions = train_xgboost_optuna(
        feature_store_path="data/processed/X_tree_dfs.parquet",
        n_trials=50,  # Full HPO
        groups=None  # Use default temporal split
    )

    elapsed = time.time() - start_time
    elapsed_min = elapsed / 60

    # Extract metrics
    oof_gini = metrics_dict.get("oof_gini")
    oot_gini = metrics_dict.get("oot_gini")
    gini_holdout = metrics_dict.get("Gini")
    gap = oof_gini - oot_gini

    print()
    print("=" * 60)
    print("HPO COMPLETE")
    print("=" * 60)
    print(f"Elapsed time: {elapsed_min:.1f} minutes ({elapsed_min/60:.1f} hours)")
    print()
    print("FINAL METRICS:")
    print(f"  OOF Gini (development):      {oof_gini:.4f}")
    print(f"  OOT Gini (temporal):         {oot_gini:.4f}")
    print(f"  Gini (holdout test set):     {gini_holdout:.4f}")
    print(f"  OOF–OOT gap:                 {gap:.4f}")
    print()

    # Apply tiered validation gates (D-16 from Phase 04.2.3.1)
    print("GATE VALIDATION:")

    # Gate 1: OOT Gini > 0.80 = REJECT (leakage)
    if oot_gini > 0.80:
        print(f"  ✗ REJECT: oot_gini > 0.80 ({oot_gini:.4f}) — leakage detected")
        raise ValueError(f"Leakage gate failed: oot_gini={oot_gini:.4f}")
    else:
        print(f"  ✓ oot_gini ≤ 0.80 ({oot_gini:.4f}) — no leakage")

    # Gate 2: OOT Gini ≥ 0.60 = ACCEPT primary gate
    if oot_gini < 0.60:
        print(f"  ✗ REJECT: oot_gini < 0.60 ({oot_gini:.4f}) — insufficient signal")
        raise ValueError(f"OOT threshold not met: oot_gini={oot_gini:.4f} < 0.60")
    else:
        print(f"  ✓ oot_gini ≥ 0.60 ({oot_gini:.4f}) — primary gate PASSED")

    # Gate 3: OOF Gini < 0.75 = plausible (not leaked)
    if oof_gini >= 0.75:
        print(f"  ⚠️  WARNING: oof_gini ≥ 0.75 ({oof_gini:.4f}) — possible residual leakage")
        print("      (proceeding, but monitor calibration performance)")
    else:
        print(f"  ✓ oof_gini < 0.75 ({oof_gini:.4f}) — plausible development set")

    # Gate 4: Gap ≤ 0.10 = good (≤ 0.05 = excellent)
    if gap > 0.10:
        print(f"  ⚠️  WARNING: gap > 0.10 ({gap:.4f}) — temporal drift")
        print("      (may need stronger regularisation in next phase)")
    elif gap > 0.05:
        print(f"  ✓ gap in [0.05, 0.10] ({gap:.4f}) — conditional accept (light regularisation advised)")
    else:
        print(f"  ✓ gap ≤ 0.05 ({gap:.4f}) — excellent temporal stability")

    print()
    print("OVERALL DECISION: ✓ MODEL ACCEPTED (OOT Gini > 0.60, no leakage)")
    print()

    # Save final evaluation
    eval_output = {
        "phase": "04.2.3.2",
        "model": "xgboost_raw_calibrated",
        "feature_store": "X_tree_dfs.parquet (enriched: Plans 01-05)",
        "trials": 50,
        "elapsed_minutes": round(elapsed_min, 1),
        "oof_gini": round(float(oof_gini), 4),
        "oot_gini": round(float(oot_gini), 4),
        "Gini": round(float(gini_holdout), 4),
        "oof_oot_gap": round(float(gap), 4),
        "best_params": best_params,
        "gates": {
            "oot_gini > 0.80": oot_gini <= 0.80,
            "oot_gini >= 0.60": oot_gini >= 0.60,
            "oof_gini < 0.75": oof_gini < 0.75,
            "gap <= 0.10": gap <= 0.10,
        },
        "decision": "ACCEPT" if (oot_gini >= 0.60 and oot_gini <= 0.80) else "REJECT",
    }

    output_path = Path("reports/xgboost_raw_eval.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(eval_output, f, indent=2)

    print(f"Evaluation saved to: {output_path}")
    print()

    # Verify model artifact was saved
    model_path = Path("models/xgboost_raw_calibrated.pkl")
    assert model_path.exists(), f"Model artifact missing: {model_path}"
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"✓ Model artifact saved: {model_path} ({model_size_mb:.1f} MB)")

    print()
    print("=" * 60)
    print("PHASE 04.2.3.2 COMPLETE — Ready for Phase 04.2.4 (LightGBM)")
    print("=" * 60)

except Exception as e:
    print()
    print("=" * 60)
    print(f"✗ HPO FAILED: {e}")
    print("=" * 60)
    import traceback
    traceback.print_exc()

    print()
    print("TROUBLESHOOTING:")
    print("1. Check X_tree_dfs.parquet exists and is readable")
    print("2. Verify prev_days_decision_mean column is present (OOT sort key)")
    print("3. Check available disk/memory: df -h, free -h")
    print("4. Re-run Plan 06 sanity check if uncertain about data quality")
    print()
    raise
