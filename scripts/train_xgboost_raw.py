#!/usr/bin/env python3
"""
Train XGBoost with Optuna HPO on raw + DFS feature store (X_tree_dfs.parquet).

Produces:
  - models/xgboost_raw_best.pkl
  - models/xgboost_raw_calibrated.pkl (Platt-scaled)
  - models/xgboost_raw_params.json
  - reports/xgboost_raw_eval.json
  - reports/figures/xgboost_raw_roc_pr.png
  - reports/figures/xgboost_raw_calibration.png

Usage:
  python scripts/train_xgboost_raw.py [--feature-store PATH] [--n-trials N]

Example:
  python scripts/train_xgboost_raw.py --feature-store data/processed/X_tree_dfs.parquet --n-trials 100
"""

import argparse
import json
import sys
from pathlib import Path

# Bootstrap credit_engine alias when running outside pytest.
# src/model.py imports from credit_engine.utils at module level, so the alias
# must exist before the import — same setup as conftest.py.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
import src as _src  # noqa: E402
if "credit_engine" not in sys.modules:
    sys.modules["credit_engine"] = _src

from credit_engine.model import train_xgboost_optuna  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost with Optuna HPO on raw features.")
    parser.add_argument(
        "--feature-store",
        type=str,
        default="data/processed/X_tree_dfs.parquet",
        help="Path to feature store parquet file (default: data/processed/X_tree_dfs.parquet)"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of Optuna trials (default: 100)"
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
    print(f"Full run log:       tail -f reports/hpo_v7_run.log  (if launched via nohup)")
    import sys as _sys; _sys.stdout.flush()

    exit_code = 0
    try:
        model, metrics, X_test, y_test, params, oof_preds = train_xgboost_optuna(
            str(feature_store_path), n_trials=n_trials
        )

        print("\n" + "="*60)
        print("XGBoost HPO Complete (Raw Features)")
        print("="*60)
        print(f"Gini:        {metrics.get('Gini', 'N/A'):.4f}")
        print(f"AUC-ROC:     {metrics.get('AUC-ROC', 'N/A'):.4f}")
        print(f"KS:          {metrics.get('KS', 'N/A'):.4f}")
        print(f"BrierSkill:  {metrics.get('BrierSkill', 'N/A'):.4f}")
        print("="*60)

        # Extract OOF/OOT metrics and apply tiered gap policy (D-12)
        oof_gini = metrics.get('oof_gini')
        oot_gini = metrics.get('oot_gini')
        holdout_gini = metrics.get('Gini', 0.0)

        if oof_gini is not None and oot_gini is not None:
            gap = oof_gini - oot_gini
            print(f"OOF Gini:    {oof_gini:.4f}")
            print(f"OOT Gini:    {oot_gini:.4f}")
            print(f"Gap (OOF-OOT): {gap:.4f}")
            print("="*60)

            # Tiered gap policy (D-12)
            if oot_gini <= 0.60:
                disposition = "REJECT"
                reason = f"OOT Gini {oot_gini:.4f} below minimum 0.60 threshold"
                print(f"\nREJECT: {reason}")
                exit_code = 1
            elif gap <= 0.05:
                disposition = "ACCEPT"
                reason = f"Gap {gap:.4f} within optimal range (<=0.05)"
                print(f"\nACCEPT: gap={gap:.4f}, oot_gini={oot_gini:.4f}")
                exit_code = 0
            elif gap <= 0.10:
                disposition = "CONDITIONAL_ACCEPT"
                reason = f"Gap {gap:.4f} in acceptable range (0.05-0.10) — documented drift"
                print(f"\nCONDITIONAL ACCEPT: gap={gap:.4f} (0.05-0.10 range)")
                exit_code = 0
            else:
                disposition = "REJECT"
                reason = f"Gap {gap:.4f} exceeds acceptable range (>0.10)"
                print(f"\nREJECT: {reason}")
                exit_code = 1
        else:
            disposition = "UNKNOWN"
            reason = "OOF/OOT metrics not available"
            gap = None
            exit_code = 0

        # Write evaluation results
        eval_results = {
            "Model": metrics.get("Model", "XGBoost (Raw, Calibrated)"),
            "AUC-ROC": metrics.get("AUC-ROC"),
            "Gini": holdout_gini,
            "KS": metrics.get("KS"),
            "Brier": metrics.get("Brier"),
            "BrierSkill": metrics.get("BrierSkill"),
            "AvgPrecision": metrics.get("AvgPrecision"),
            "oof_gini": oof_gini,
            "oot_gini": oot_gini,
            "gap": gap,
            "disposition": disposition,
            "reason": reason,
        }
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        with open(reports_dir / "xgboost_raw_eval.json", "w") as f:
            json.dump(eval_results, f, indent=2)
        print(f"\nResults written to reports/xgboost_raw_eval.json")

        print("\nArtifacts saved:")
        print("  - models/xgboost_raw_best.pkl")
        print("  - models/xgboost_raw_calibrated.pkl")
        print("  - models/xgboost_raw_params.json")
        print("  - reports/xgboost_raw_eval.json")
        print("  - reports/figures/xgboost_raw_roc_pr.png")
        print("  - reports/figures/xgboost_raw_calibration.png")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
