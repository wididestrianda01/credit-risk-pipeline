"""
Full 50-trial XGBoost HPO on X_tree_dfs.parquet — study xgboost_raw_v9.

Changes vs v8:
  - n_estimators: 3000 → 1000 (prevents runaway trials)
  - early_stopping_rounds: 100 → 50 (faster pruning)
  - Study: xgboost_raw_v9 (fresh, no contamination from v8)

Gates (same as Plan 07):
  - OOT Gini > 0.60
  - OOF Gini < 0.75 (no leakage)
  - OOF-OOT gap ≤ 0.05

Usage:
    python scripts/run_xgb_hpo_v9.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

# conftest alias: credit_engine → src
import src  # noqa: E402
sys.modules.setdefault("credit_engine", src)
for _mod in ["utils", "data_loader", "features", "auto_features", "model", "explain"]:
    full = f"src.{_mod}"
    alias = f"credit_engine.{_mod}"
    try:
        import importlib
        sys.modules.setdefault(alias, importlib.import_module(full))
    except ModuleNotFoundError:
        pass

from src.model import train_xgboost_optuna  # noqa: E402

_FEATURE_STORE = str(_PROJECT_ROOT / "data" / "processed" / "X_tree_dfs.parquet")
_N_TRIALS = 50
_CALIBRATED_OUT = str(_PROJECT_ROOT / "models" / "xgboost_raw_calibrated.pkl")
_EVAL_OUT = str(_PROJECT_ROOT / "reports" / "xgboost_raw_eval.json")
_HPO_LOG = str(_PROJECT_ROOT / "reports" / "hpo_progress.jsonl")

# Gate thresholds (Plan 07 spec)
_OOT_GINI_GATE = 0.60
_OOF_GINI_MAX = 0.75
_OOF_OOT_GAP_MAX = 0.05


def main() -> None:
    print(f"Starting XGBoost HPO v9 — {_N_TRIALS} trials on {_FEATURE_STORE}")
    print(f"Artifacts: {_CALIBRATED_OUT}")

    model, metrics, X_test, y_test, best_params, oof_preds = train_xgboost_optuna(
        feature_store_path=_FEATURE_STORE,
        n_trials=_N_TRIALS,
        progress_log_path=_HPO_LOG,
    )

    oof_gini = metrics.get("oof_gini", 0.0)
    oot_gini = metrics.get("oot_gini", 0.0)
    holdout_gini = metrics.get("Gini", 0.0)

    print(f"\nResults:")
    print(f"  OOF Gini:  {oof_gini:.4f}")
    print(f"  OOT Gini:  {oot_gini:.4f}")
    print(f"  Hold Gini: {holdout_gini:.4f}")
    print(f"  KS:        {metrics.get('KS', 0.0):.4f}")
    print(f"  Brier:     {metrics.get('Brier', 0.0):.4f}")

    # Gate verification
    gap = abs(oof_gini - oot_gini)
    gates = {
        "oot_gini_gate": oot_gini > _OOT_GINI_GATE,
        "oof_leakage_gate": oof_gini < _OOF_GINI_MAX,
        "gap_gate": gap <= _OOF_OOT_GAP_MAX,
    }
    print(f"\nGates:")
    for gate, passed in gates.items():
        print(f"  {gate}: {'PASS' if passed else 'FAIL'}")

    # Save metrics
    Path(_EVAL_OUT).write_text(json.dumps(metrics, indent=2))
    print(f"\nEval saved to {_EVAL_OUT}")

    # Save model
    import joblib
    joblib.dump(model, _CALIBRATED_OUT)
    print(f"Model saved to {_CALIBRATED_OUT}")

    if all(gates.values()):
        print("\nAll gates PASS — Plan 07 complete.")
    else:
        failed = [k for k, v in gates.items() if not v]
        print(f"\nFailed gates: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
