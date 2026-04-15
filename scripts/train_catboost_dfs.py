#!/usr/bin/env python3
"""
Train CatBoost with Optuna HPO on DFS-augmented feature store (X_tree_dfs.parquet).

This script trains CatBoost on the DFS (Featuretools) aggregated features for
Phase 04.2.10 ensemble diversity. DFS features provide comprehensive
multi-table auto-engineered signals, complementing raw and WoE features.

Produces:
  - models/catboost_dfs_calibrated.pkl
  - models/catboost_dfs_params.json
  - reports/catboost_dfs_eval.json

Usage:
  python scripts/train_catboost_dfs.py [--feature-store PATH] [--n-trials N]

Example:
  python scripts/train_catboost_dfs.py --feature-store data/processed/X_tree_dfs.parquet --n-trials 50
"""

import argparse
import json
import sys
from pathlib import Path

# Bootstrap credit_engine alias when running outside pytest
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
import src as _src  # noqa: E402
if "credit_engine" not in sys.modules:
    sys.modules["credit_engine"] = _src

from credit_engine.model import train_catboost_optuna  # noqa: E402
import pandas as pd


def check_leaky_columns(feature_store_path: str) -> None:
    """
    Verify that no SK_DPD leaky columns exist in the feature store.

    Parameters
    ----------
    feature_store_path : str
        Path to parquet file.

    Raises
    ------
    ValueError
        If any leaky columns are detected.
    """
    df = pd.read_parquet(feature_store_path)

    # Known SK_DPD leaky column patterns (observation-time delinquency status).
    # NOTE: inst_days_past_due_* is NOT leaky — it is historical payment behaviour
    # aggregated from installments_payments.DAYS_PAST_DUE (backward-looking).
    # Only the SK_DPD columns from POS/CC tables represent current-status leakage.
    leaky_patterns = [
        "SK_DPD",       # Direct credit bureau delinquency flag (observation-time)
        "prev_sk_dpd",  # Previous application delinquency flag
        "cc_sk_dpd",    # Credit card current delinquency flag
        "pos_sk_dpd",   # POS current delinquency flag
    ]

    leaky_found = []
    for col in df.columns:
        for pattern in leaky_patterns:
            if pattern.lower() in col.lower():
                leaky_found.append(col)
                break

    if leaky_found:
        raise ValueError(
            f"Leaky SK_DPD columns detected in {feature_store_path}: {leaky_found}. "
            "These must be removed before training to ensure regulatory compliance."
        )

    print(f"✓ No SK_DPD leaky columns detected in feature store ({df.shape[1]} features)")


def main():
    parser = argparse.ArgumentParser(
        description="Train CatBoost with Optuna HPO on DFS-augmented features."
    )
    parser.add_argument(
        "--feature-store",
        type=str,
        default="data/processed/X_tree_dfs.parquet",
        help="Path to DFS feature store parquet file (default: data/processed/X_tree_dfs.parquet)"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials (default: 50)"
    )

    args = parser.parse_args()

    feature_store_path = Path(args.feature_store)
    n_trials = args.n_trials

    # Validate inputs
    if not feature_store_path.exists():
        print(f"ERROR: Feature store not found: {feature_store_path}", file=sys.stderr)
        sys.exit(1)

    if n_trials < 1:
        print(f"ERROR: n_trials must be >= 1, got {n_trials}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading feature store from {feature_store_path}...")

    # Check for leaky columns
    try:
        check_leaky_columns(str(feature_store_path))
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Running Optuna HPO with {n_trials} trials...")
    print(f"Per-trial progress: tail -f reports/hpo_progress.jsonl")
    import sys as _sys
    _sys.stdout.flush()

    exit_code = 0
    try:
        model, metrics, X_test, y_test, best_params = train_catboost_optuna(
            str(feature_store_path), n_trials=n_trials
        )

        print("\n" + "="*60)
        print("CatBoost HPO Complete (DFS Features)")
        print("="*60)
        print(f"Gini:        {metrics.get('Gini', 'N/A'):.4f}")
        print(f"AUC-ROC:     {metrics.get('AUC-ROC', 'N/A'):.4f}")
        print(f"KS:          {metrics.get('KS', 'N/A'):.4f}")
        print(f"BrierSkill:  {metrics.get('BrierSkill', 'N/A'):.4f}")
        print("="*60)

        oot_gini = metrics.get('oot_gini', 0.0)
        holdout_gini = metrics.get('Gini', 0.0)

        print(f"OOT Gini:    {oot_gini:.4f}")
        print(f"Baseline (CatBoost v2): 0.5814")
        print("Note: Individual Gini may be lower than baseline (diversity is the goal)")
        print("="*60)

        # Write evaluation results
        eval_results = {
            "Store": "X_tree_dfs",
            "Strategy": "scale_pos_weight",
            "Model": "CatBoost (DFS, Calibrated)",
            "AUC-ROC": metrics.get("AUC-ROC"),
            "Gini": holdout_gini,
            "OOT_Gini": oot_gini,
            "KS": metrics.get("KS"),
            "Brier": metrics.get("Brier"),
            "BrierSkill": metrics.get("BrierSkill"),
            "AvgPrecision": metrics.get("AvgPrecision"),
        }
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        with open(reports_dir / "catboost_dfs_eval.json", "w") as f:
            json.dump(eval_results, f, indent=2)

        # Write best params
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        with open(models_dir / "catboost_dfs_params.json", "w") as f:
            json.dump(best_params, f, indent=2)

        print(f"\nResults written to:")
        print(f"  - reports/catboost_dfs_eval.json")
        print(f"  - models/catboost_dfs_params.json")

        print("\nArtifacts saved:")
        print("  - models/catboost_dfs_calibrated.pkl")
        print("  - models/catboost_dfs_params.json")
        print("  - reports/catboost_dfs_eval.json")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
