#!/usr/bin/env python3
"""
Train XGBoost with Optuna HPO on WoE-encoded feature store (X_features.parquet).

This script trains XGBoost on the WoE-encoded features for Phase 04.2.10
ensemble diversity. WoE features provide a different signal than raw features,
so ensemble predictions benefit from the structural diversity.

Produces:
  - models/xgboost_woe_calibrated.pkl
  - models/xgboost_woe_params.json
  - reports/xgb_woe_eval.json

Usage:
  python scripts/train_xgboost_woe.py [--feature-store PATH] [--n-trials N]

Example:
  python scripts/train_xgboost_woe.py --feature-store data/processed/X_features.parquet --n-trials 50
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

from credit_engine.model import train_xgboost_optuna  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Train XGBoost with Optuna HPO on WoE-encoded features."
    )
    parser.add_argument(
        "--feature-store",
        type=str,
        default="data/processed/X_features.parquet",
        help="Path to WoE feature store parquet file (default: data/processed/X_features.parquet)"
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
    print(f"Running Optuna HPO with {n_trials} trials...")
    print(f"Per-trial progress: tail -f reports/hpo_progress.jsonl")
    import sys as _sys
    _sys.stdout.flush()

    exit_code = 0
    try:
        model, metrics, X_test, y_test, best_params, _oof_preds = train_xgboost_optuna(
            str(feature_store_path), n_trials=n_trials
        )

        print("\n" + "="*60)
        print("XGBoost HPO Complete (WoE Features)")
        print("="*60)
        print(f"Gini:        {metrics.get('Gini', 'N/A'):.4f}")
        print(f"AUC-ROC:     {metrics.get('AUC-ROC', 'N/A'):.4f}")
        print(f"KS:          {metrics.get('KS', 'N/A'):.4f}")
        print(f"BrierSkill:  {metrics.get('BrierSkill', 'N/A'):.4f}")
        print("="*60)

        oot_gini = metrics.get('oot_gini', 0.0)
        holdout_gini = metrics.get('Gini', 0.0)

        print(f"OOT Gini:    {oot_gini:.4f}")
        print(f"Baseline (XGB raw): 0.5636")
        print("="*60)

        # Write evaluation results
        eval_results = {
            "Store": "X_features",
            "Strategy": "scale_pos_weight",
            "Model": "XGBoost (WoE, Calibrated)",
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
        with open(reports_dir / "xgb_woe_eval.json", "w") as f:
            json.dump(eval_results, f, indent=2)

        # Write best params
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        with open(models_dir / "xgboost_woe_params.json", "w") as f:
            json.dump(best_params, f, indent=2)

        print(f"\nResults written to:")
        print(f"  - reports/xgb_woe_eval.json")
        print(f"  - models/xgboost_woe_params.json")

        print("\nArtifacts saved:")
        print("  - models/xgboost_woe_calibrated.pkl")
        print("  - models/xgboost_woe_params.json")
        print("  - reports/xgb_woe_eval.json")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
