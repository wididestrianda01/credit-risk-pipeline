"""
Plan 02-07 execution: fresh XGBoost HPO on raw vs combined features.
Saves intermediate JSON after each training run so results survive any timeout.
"""
import json
import sys
import time
from pathlib import Path

# Add project root to path so credit_engine alias works
sys.path.insert(0, str(Path(__file__).parent.parent))
import conftest  # noqa: F401 — registers credit_engine alias

import pandas as pd

REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

RAW_RESULT_FILE = REPORTS / "xgb_raw_baseline_task2.json"
COMBINED_RESULT_FILE = REPORTS / "xgb_combined_baseline_task3.json"
GATE_FILE = REPORTS / "combined_features_eval.json"

N_TRIALS = 50


def extract_gini(metrics: dict) -> float:
    """Pull Gini from various key conventions used in the codebase."""
    for key in ("Gini", "gini", "gini_coefficient"):
        if key in metrics:
            return float(metrics[key])
    raise KeyError(f"No Gini key found in metrics: {list(metrics.keys())}")


def run_task2() -> dict:
    from credit_engine.model import train_xgboost_optuna

    print("=== Task 2: XGBoost on raw 62-feature baseline ===", flush=True)
    X_raw = pd.read_parquet("data/processed/X_raw_features.parquet")
    y = pd.read_parquet("data/processed/y_train.parquet").squeeze()
    assert X_raw.shape == (307511, 62), f"Unexpected shape: {X_raw.shape}"

    t0 = time.time()
    result = train_xgboost_optuna(X_raw, y, n_trials=N_TRIALS)
    elapsed = (time.time() - t0) / 60

    # Handle both 2-tuple and 5-tuple return signatures
    model_raw, metrics_raw = result[0], result[1]
    gini_raw = extract_gini(metrics_raw)
    print(f"Raw Gini: {gini_raw:.4f} (training: {elapsed:.1f} min)", flush=True)

    payload = {
        "gini_raw": gini_raw,
        "metrics": {k: float(v) if isinstance(v, (int, float)) else v
                    for k, v in metrics_raw.items()},
        "training_minutes": round(elapsed, 2),
    }
    RAW_RESULT_FILE.write_text(json.dumps(payload, indent=2))
    print(f"Saved: {RAW_RESULT_FILE}", flush=True)
    return payload


def run_task3(gini_raw: float) -> dict:
    from credit_engine.model import train_xgboost_optuna

    print("=== Task 3: XGBoost on combined 63-feature store ===", flush=True)
    X_combined = pd.read_parquet("data/processed/X_combined_features.parquet")
    y = pd.read_parquet("data/processed/y_train.parquet").squeeze()
    assert X_combined.shape == (307511, 63), f"Unexpected shape: {X_combined.shape}"

    t0 = time.time()
    result = train_xgboost_optuna(X_combined, y, n_trials=N_TRIALS)
    elapsed = (time.time() - t0) / 60

    model_comb, metrics_comb = result[0], result[1]
    gini_combined = extract_gini(metrics_comb)
    gini_delta = gini_combined - gini_raw
    print(f"Combined Gini: {gini_combined:.4f}, Delta: {gini_delta:+.6f} (training: {elapsed:.1f} min)", flush=True)

    payload = {
        "gini_combined": gini_combined,
        "gini_delta": gini_delta,
        "metrics": {k: float(v) if isinstance(v, (int, float)) else v
                    for k, v in metrics_comb.items()},
        "training_minutes": round(elapsed, 2),
    }
    COMBINED_RESULT_FILE.write_text(json.dumps(payload, indent=2))
    print(f"Saved: {COMBINED_RESULT_FILE}", flush=True)
    return payload


def run_task4(raw: dict, combined: dict) -> dict:
    print("=== Task 4: Gate evaluation ===", flush=True)
    gini_raw = raw["gini_raw"]
    gini_combined = combined["gini_combined"]
    gini_delta = combined["gini_delta"]
    gate_passed = gini_delta >= 0.0

    results = {
        "gini_raw": gini_raw,
        "gini_combined": gini_combined,
        "gini_delta": gini_delta,
        "n_raw_features": 62,
        "n_combined_features": 63,
        "phase_2_gate_passed": gate_passed,
        "validation_notes": {
            "delta_requirement": "gini_delta >= 0 (combined must not regress vs raw)",
            "delta_value": gini_delta,
            "delta_valid": gate_passed,
            "adversarial_auc": 0.5012,
            "adversarial_verdict": "safe",
            "explanation": (
                f"Corrected imputer (trained on 59 features, best_corr=0.9994) rebuilt combined store. "
                f"Gini delta = {gini_delta:+.6f}. "
                f"Phase 2 gate {'PASSED' if gate_passed else 'FAILED'} "
                f"(threshold: >= 0). "
                f"Adversarial AUC = 0.5012 < 0.55 (PASSED, from Plan 02-05)."
            ),
        },
        "training_minutes": {
            "raw_baseline": raw["training_minutes"],
            "combined_store": combined["training_minutes"],
        },
    }

    if not gate_passed:
        results["remediation_options"] = [
            "Option A: Tune EXT_SOURCE_3 imputer (more Optuna trials, wider LR/depth search)",
            "Option B: Ensemble imputation (LGB + XGB + mean voting)",
            "Option C: Revert to raw-only — drop EXT_SOURCE_3 imputation entirely",
            "Option D: Test MISSING_FLAG alone (without imputed values) as a binary feature",
        ]

    GATE_FILE.write_text(json.dumps(results, indent=2))
    status = "PASSED" if gate_passed else "FAILED"
    print(f"\n{'=' * 50}", flush=True)
    print(f"Phase 2 gate: {status}", flush=True)
    print(f"  Gini delta: {gini_delta:+.6f} (raw={gini_raw:.4f}, combined={gini_combined:.4f})", flush=True)
    print(f"  Adversarial AUC: 0.5012 (PASSED)", flush=True)
    print(f"Saved: {GATE_FILE}", flush=True)
    return results


if __name__ == "__main__":
    # Task 2 — skip if already saved (allows resuming after partial failure)
    if RAW_RESULT_FILE.exists():
        print(f"Resuming: {RAW_RESULT_FILE} already exists, skipping Task 2", flush=True)
        raw_payload = json.loads(RAW_RESULT_FILE.read_text())
    else:
        raw_payload = run_task2()

    # Task 3 — skip if already saved
    if COMBINED_RESULT_FILE.exists():
        print(f"Resuming: {COMBINED_RESULT_FILE} already exists, skipping Task 3", flush=True)
        combined_payload = json.loads(COMBINED_RESULT_FILE.read_text())
    else:
        combined_payload = run_task3(raw_payload["gini_raw"])

    # Task 4
    gate = run_task4(raw_payload, combined_payload)
    print("\nAll tasks complete.", flush=True)
