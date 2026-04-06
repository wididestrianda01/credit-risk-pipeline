"""
lgb_raw_ablation.py
-------------------
Ablation (a): LGB on raw features with is_unbalance=True, num_leaves_max=150.

Hypothesis: scale_pos_weight=11.4 × num_leaves_max=300 caused overfitting.
This run reverts to is_unbalance=True (WoE baseline's imbalance handling)
while keeping raw continuous features.

Expected: Gini > 0.452 (WoE baseline) if hypothesis is confirmed.
Artifact: models/lightgbm_raw_ablation_a.pkl
Results:  reports/lgb_raw_ablation_a.json
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import src  # noqa: E402
sys.modules["credit_engine"] = src

from credit_engine.model import train_lightgbm_optuna  # noqa: E402
from credit_engine.utils import evaluate_model          # noqa: E402

_RAW_FEATURES_PATH = project_root / "data" / "processed" / "X_raw_features.parquet"
_Y_PATH            = project_root / "data" / "processed" / "y_train.parquet"
_REPORTS_DIR       = project_root / "reports"
_MODELS_DIR        = project_root / "models"

_LGB_BASELINE_GINI = 0.452   # WoE path, is_unbalance=True
_LGB_RAW_GINI      = 0.4107  # raw path, scale_pos_weight=True, num_leaves_max=300
_N_TRIALS          = 100


def main() -> None:
    print("=" * 70)
    print("LGB Raw Ablation (a): is_unbalance=True, num_leaves_max=150")
    print("=" * 70)

    X_raw = pd.read_parquet(_RAW_FEATURES_PATH)
    y     = pd.read_parquet(_Y_PATH).squeeze()

    common_idx = X_raw.index.intersection(y.index)
    X_raw = X_raw.loc[common_idx]
    y     = y.loc[common_idx]
    print(f"X_raw: {X_raw.shape}  |  y default rate: {y.mean():.2%}")

    print(f"\nTraining LightGBM ({_N_TRIALS} Optuna trials, is_unbalance=True, num_leaves_max=150)...")
    model, metrics, X_test, y_test, best_params = train_lightgbm_optuna(
        X_raw, y,
        n_trials=_N_TRIALS,
        use_scale_pos_weight=False,   # ← ablation: reverts to is_unbalance=True
        num_leaves_max=150,           # ← ablation: matches WoE ceiling
    )
    ev    = evaluate_model(model, X_test, y_test, "LightGBM_raw_ablation_a")
    gini  = ev["Gini"]

    print(f"\n  Gini: {gini:.4f}")
    print(f"  vs WoE baseline ({_LGB_BASELINE_GINI:.4f}):  {gini - _LGB_BASELINE_GINI:+.4f}")
    print(f"  vs raw attempt  ({_LGB_RAW_GINI:.4f}):  {gini - _LGB_RAW_GINI:+.4f}")

    joblib.dump(model, _MODELS_DIR / "lightgbm_raw_ablation_a.pkl")
    print("  Saved: models/lightgbm_raw_ablation_a.pkl")

    results = {
        "gini":                   round(gini, 6),
        "auc":                    round(ev["AUC-ROC"], 6),
        "delta_vs_woe_baseline":  round(gini - _LGB_BASELINE_GINI, 6),
        "delta_vs_raw_attempt":   round(gini - _LGB_RAW_GINI, 6),
        "config": {
            "use_scale_pos_weight": False,
            "num_leaves_max":       150,
            "n_trials":             _N_TRIALS,
            "n_features":           X_raw.shape[1],
        },
        "best_params": best_params,
        "hypothesis": (
            "scale_pos_weight=11.4 x num_leaves_max=300 caused overfitting. "
            "Ablation: is_unbalance=True, num_leaves_max=150."
        ),
        "verdict": (
            "CONFIRMED" if gini > _LGB_BASELINE_GINI
            else "NOT_CONFIRMED — investigate further"
        ),
    }

    out_path = _REPORTS_DIR / "lgb_raw_ablation_a.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"  Results: {out_path.name}")

    print("\n" + "=" * 70)
    print("VERDICT:", results["verdict"])
    print("=" * 70)


if __name__ == "__main__":
    main()
