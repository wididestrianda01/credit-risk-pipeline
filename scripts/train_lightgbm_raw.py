#!/usr/bin/env python3
"""
Train LightGBM with Optuna HPO on raw features or execute full ablation.

Produces (single-run mode):
  - models/lightgbm_raw_calibrated.pkl
  - reports/lgb_raw_{store_tag}_{strategy}_metrics.json
  - reports/figures/lgb_raw_calibration_plot.png

Produces (ablation mode):
  - 9× metrics JSON files (one per store × strategy combination)
  - reports/lgb_raw_ablation_comparison.csv (comparison table)
  - models/lightgbm_raw_calibrated.pkl (best model)

Usage:
  # Single run with custom feature store and strategy
  python scripts/train_lightgbm_raw.py --feature-store data/processed/X_tree_dfs.parquet \\
    --strategy scale_pos_weight --n-trials 50

  # Full 9-cell ablation (3 stores × 3 strategies)
  python scripts/train_lightgbm_raw.py --ablation --n-trials 50

Example:
  python scripts/train_lightgbm_raw.py --ablation --n-trials 10  # Quick test
  python scripts/train_lightgbm_raw.py --ablation --n-trials 50  # Full ablation (~2-4 hours)
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

from credit_engine.model import train_lightgbm_optuna, run_lightgbm_ablation_workflow  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Train LightGBM with Optuna HPO on raw features or execute full ablation"
    )
    parser.add_argument(
        "--feature-store",
        type=str,
        default=None,
        help="Path to feature store parquet file (e.g., data/processed/X_tree_dfs.parquet). "
             "Required for single-run mode; ignored in ablation mode."
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials per run (default: 50)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="scale_pos_weight",
        choices=["scale_pos_weight", "is_unbalance", "smote"],
        help="Imbalance strategy for single-run mode (default: scale_pos_weight)"
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run full 9-cell ablation (3 stores × 3 strategies) instead of single run"
    )

    args = parser.parse_args()

    n_trials = args.n_trials

    # Validate n_trials
    if n_trials < 1:
        print(f"ERROR: n_trials must be >= 1, got {n_trials}", file=sys.stderr)
        sys.exit(1)

    exit_code = 0

    try:
        if args.ablation:
            # --- Ablation mode: run 9 cells ---
            print(f"Running full LightGBM ablation (3 stores × 3 strategies = 9 cells)")
            print(f"Number of trials per cell: {n_trials}")
            print(f"Total trials: {n_trials * 9}")
            print(f"Estimated runtime: 2-4 hours depending on hardware")
            print("="*60)
            import sys as _sys; _sys.stdout.flush()

            results = run_lightgbm_ablation_workflow(n_trials=n_trials)

            # Verify comparison table was created
            comparison_path = Path("reports") / "lgb_raw_ablation_comparison.csv"
            if comparison_path.exists():
                import pandas as pd
                comparison_df = pd.read_csv(comparison_path)
                print("\n" + "="*60)
                print("Ablation Complete — Comparison Table")
                print("="*60)
                print(comparison_df[["Store", "Strategy", "OOT_Gini", "BrierSkill"]].to_string(index=False))
                print("="*60)

                # Extract best model info
                if len(comparison_df) > 0:
                    best_idx = comparison_df["OOT_Gini"].idxmax()
                    best_row = comparison_df.iloc[best_idx]
                    print(f"\nBest Model: {best_row['Store']} + {best_row['Strategy']}")
                    print(f"OOT Gini: {best_row['OOT_Gini']:.4f}")
                    print(f"BrierSkill: {best_row['BrierSkill']:.4f}")

                    # Verify gates
                    if best_row["OOT_Gini"] > 0.60:
                        print(f"✓ OOT Gini gate PASSED (> 0.60)")
                        exit_code = 0
                    else:
                        print(f"✗ OOT Gini gate FAILED ({best_row['OOT_Gini']:.4f} <= 0.60)")
                        exit_code = 1

            print("\nArtifacts saved:")
            print("  - reports/lgb_raw_ablation_comparison.csv")
            print("  - reports/lgb_raw_Xtrain_*.json (3 strategy variants)")
            print("  - reports/lgb_raw_Xtreeraw_*.json (3 strategy variants)")
            print("  - reports/lgb_raw_Xtreeds_*.json (3 strategy variants)")
            print("  - models/lightgbm_raw_calibrated.pkl (best model)")
            print("  - reports/figures/lgb_raw_calibration_plot.png (best model)")

        else:
            # --- Single-run mode ---
            if args.feature_store is None:
                parser.error("--feature-store required for single-run mode (or use --ablation)")

            feature_store_path = Path(args.feature_store)

            # Validate inputs
            if not feature_store_path.exists():
                print(f"ERROR: Feature store not found: {feature_store_path}", file=sys.stderr)
                sys.exit(1)

            print(f"Loading feature store from {feature_store_path}...")
            print(f"Imbalance strategy: {args.strategy}")
            print(f"Running Optuna HPO with {n_trials} trials...")
            print("="*60)
            import sys as _sys; _sys.stdout.flush()

            model, metrics, X_test, y_test, best_params = train_lightgbm_optuna(
                feature_store_path=str(feature_store_path),
                n_trials=n_trials,
                imbalance_strategy=args.strategy,
            )

            print("\n" + "="*60)
            print("LightGBM HPO Complete (Raw Features)")
            print("="*60)
            print(f"Store:       {feature_store_path.stem}")
            print(f"Strategy:    {args.strategy}")
            print(f"Gini:        {metrics.get('Gini', 'N/A'):.4f}")
            print(f"OOT Gini:    {metrics.get('oot_gini', 'N/A'):.4f}")
            print(f"AUC-ROC:     {metrics.get('AUC-ROC', 'N/A'):.4f}")
            print(f"KS:          {metrics.get('KS', 'N/A'):.4f}")
            print(f"BrierSkill:  {metrics.get('BrierSkill', 'N/A'):.4f}")
            print("="*60)

            # Write evaluation results
            store_name = feature_store_path.stem
            store_tag_map = {
                "X_train": "Xtrain",
                "X_tree_raw": "Xtreeraw",
                "X_tree_dfs": "Xtreeds",
            }
            store_tag = store_tag_map.get(store_name, store_name)
            eval_filename = f"lgb_raw_{store_tag}_{args.strategy}_eval.json"

            eval_results = {
                "Store": store_name,
                "Strategy": args.strategy,
                "Model": metrics.get("Model", "LGB (Raw, Calibrated)"),
                "AUC-ROC": metrics.get("AUC-ROC"),
                "Gini": metrics.get("Gini"),
                "OOT_Gini": metrics.get("oot_gini"),
                "OOF_Gini": metrics.get("oof_gini"),
                "KS": metrics.get("KS"),
                "Brier": metrics.get("Brier"),
                "BrierSkill": metrics.get("BrierSkill"),
                "AvgPrecision": metrics.get("AvgPrecision"),
            }
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)
            with open(reports_dir / eval_filename, "w") as f:
                json.dump(eval_results, f, indent=2)
            print(f"\nResults written to reports/{eval_filename}")

            # Verify gates
            oot_gini = metrics.get("oot_gini", 0.0)
            brier_skill = metrics.get("BrierSkill", 0.0)

            if oot_gini > 0.60:
                print(f"✓ OOT Gini gate PASSED (> 0.60)")
            else:
                print(f"✗ OOT Gini gate FAILED ({oot_gini:.4f} <= 0.60)")
                exit_code = 1

            if brier_skill > 0:
                print(f"✓ BrierSkill gate PASSED (> 0)")
            else:
                print(f"✗ BrierSkill gate FAILED (<= 0)")
                exit_code = 1

            print("\nArtifacts saved:")
            print("  - models/lightgbm_raw_calibrated.pkl")
            print(f"  - reports/{eval_filename}")
            print("  - reports/figures/lgb_raw_calibration_plot.png")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
