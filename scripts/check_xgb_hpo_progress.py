#!/usr/bin/env python3
"""
Check XGBoost HPO progress from Optuna DB.

Usage:
    python scripts/check_xgb_hpo_progress.py

Shows:
    - Total trials completed
    - Best AUC/Gini so far
    - Estimated time remaining
    - Non-regression status
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

if "credit_engine" not in sys.modules:
    import src
    sys.modules["credit_engine"] = src

import optuna

OPTUNA_DB_PATH = "sqlite:///models/optuna_studies.db"
OPTUNA_STUDY_NAME = "xgboost_extended_study"
TOTAL_TRIALS_TARGET = 100
TIME_PER_TRIAL_MIN = 23.6

def main():
    storage = optuna.storages.RDBStorage(OPTUNA_DB_PATH)
    try:
        study = optuna.load_study(study_name=OPTUNA_STUDY_NAME, storage=storage)
    except KeyError:
        print("ERROR: Study not found in DB")
        return 1

    total = len(study.trials)
    completed = sum(1 for t in study.trials if t.value is not None)
    running = total - completed

    print("=" * 70)
    print("XGBoost HPO Progress")
    print("=" * 70)
    print(f"Total trials submitted: {total}/{TOTAL_TRIALS_TARGET}")
    print(f"Completed: {completed}")
    print(f"In progress: {running}")

    if completed > 0:
        values = [t.value for t in study.trials if t.value is not None]
        best_auc = max(values)
        best_gini = 2 * best_auc - 1
        worst_auc = min(values)

        print(f"\nResults so far:")
        print(f"  Best AUC: {best_auc:.6f}")
        print(f"  Best Gini: {best_gini:.6f}")
        print(f"  Worst AUC: {worst_auc:.6f}")
        print(f"  Average AUC: {sum(values)/len(values):.6f}")

        # Non-regression check
        baseline = 0.5567
        if best_gini >= baseline - 0.001:
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
        print(f"\nNon-regression vs Phase 04.1 baseline ({baseline:.4f}):")
        print(f"  Status: {status}")
        print(f"  Delta: {best_gini - baseline:+.6f}")

    # Time estimate
    trials_remaining = TOTAL_TRIALS_TARGET - total
    if trials_remaining > 0:
        eta_hours = trials_remaining * TIME_PER_TRIAL_MIN / 60
        print(f"\nEstimated time remaining:")
        print(f"  {eta_hours:.1f} hours (~{int(eta_hours)}h {int((eta_hours%1)*60)}m)")
    else:
        print("\nAll trials submitted! Waiting for completion of final trials...")

    # Check for results JSON
    results_path = Path("reports/xgb_hpo_results.json")
    if results_path.exists():
        print(f"\n✓ Results JSON exists: {results_path}")
    else:
        print(f"\n⧗ Results JSON pending (will be created on completion)")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    exit(main())
