"""Test file to run HPO via pytest."""
import json
import time
from pathlib import Path
import pytest

def test_xgboost_full_hpo_50_trials():
    """Run full 50-trial XGBoost HPO on enriched feature store."""
    from credit_engine.model import train_xgboost_optuna

    print("\n" + "=" * 60)
    print("FULL 50-TRIAL XGBOOST HPO ON ENRICHED STORE")
    print("=" * 60)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Feature store: data/processed/X_tree_dfs.parquet")
    print(f"Optuna study: xgboost_raw_v2 (resumable)")
    print(f"Trials: 50")
    print()

    start_time = time.time()

    # Run HPO
    model, metrics_dict, X_test, y_test, best_params, oof_predictions = train_xgboost_optuna(
        feature_store_path="data/processed/X_tree_dfs.parquet",
        n_trials=50,
        groups=None
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

    # Gate validation
    print("GATE VALIDATION:")

    if oot_gini > 0.80:
        raise ValueError(f"Leakage detected: oot_gini={oot_gini:.4f} > 0.80")
    print(f"  ✓ oot_gini ≤ 0.80 ({oot_gini:.4f}) — no leakage")

    if oot_gini < 0.60:
        raise ValueError(f"OOT threshold not met: oot_gini={oot_gini:.4f} < 0.60")
    print(f"  ✓ oot_gini ≥ 0.60 ({oot_gini:.4f}) — primary gate PASSED")

    if oof_gini < 0.75:
        print(f"  ✓ oof_gini < 0.75 ({oof_gini:.4f}) — plausible development set")
    else:
        print(f"  ⚠️  WARNING: oof_gini ≥ 0.75 ({oof_gini:.4f}) — possible residual leakage")

    if gap <= 0.05:
        print(f"  ✓ gap ≤ 0.05 ({gap:.4f}) — excellent temporal stability")
    elif gap <= 0.10:
        print(f"  ✓ gap ≤ 0.10 ({gap:.4f}) — good temporal stability")
    else:
        print(f"  ⚠️  WARNING: gap > 0.10 ({gap:.4f}) — temporal drift")

    print()
    print("OVERALL DECISION: ✓ MODEL ACCEPTED (OOT Gini > 0.60, no leakage)")
    print()

    # Save evaluation
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

    # Verify model
    model_path = Path("models/xgboost_raw_calibrated.pkl")
    assert model_path.exists(), f"Model missing: {model_path}"
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"✓ Model artifact saved: {model_path} ({model_size_mb:.1f} MB)")

    print()
    print("=" * 60)
    print("PHASE 04.2.3.2 COMPLETE — Ready for Phase 04.2.4 (LightGBM)")
    print("=" * 60)
    print()

    # Assertions for testing
    assert oot_gini >= 0.60, f"Primary gate failed: OOT Gini {oot_gini:.4f} < 0.60"
    assert oot_gini <= 0.80, f"Leakage gate failed: OOT Gini {oot_gini:.4f} > 0.80"
    assert oof_gini < 0.75, f"OOF Gini {oof_gini:.4f} >= 0.75 indicates possible leakage"
    assert gap <= 0.10, f"Gap {gap:.4f} > 0.10 indicates temporal drift"

