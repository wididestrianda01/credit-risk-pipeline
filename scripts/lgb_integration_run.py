"""
lgb_integration_run.py — Step A10
------------------------------------
Integration run combining best findings from A6–A9:
  - Best booster (from A6)
  - Monotone constraints if they improved Gini by ≥ 0.005 in A7
  - Optimal num_leaves + min_child_samples bounds (from A8)
  - Recommended patience (from A9)

Runs 100 Optuna trials under 10-fold temporal CV on X_raw_features.parquet.

Success threshold: Gini > 0.48
  - Gini > 0.50  → Track A highly successful; run per-technique ablation
  - 0.48–0.50    → Track A marginal; investigate which technique contributed
  - < 0.48       → Track A insufficient on raw path; pursue additional diagnostics
                   before abandoning (run with single best technique to isolate root cause)

Output:
  reports/lgb_integration_results.json
  models/lightgbm_best_v2.pkl

Usage
-----
    .venv/bin/python scripts/lgb_integration_run.py [--n-trials N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Alias src as credit_engine for consistency with conftest.py
sys.path.insert(0, str(Path(__file__).parent.parent))
import src  # noqa: E402
sys.modules["credit_engine"] = src

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_X_PATH = "data/processed/X_raw_features.parquet"
_Y_PATH = "data/processed/y_train.parquet"

_A6_PATH = "reports/lgb_booster_comparison.json"
_A7_PATH = "reports/lgb_monotone_constraint_test.json"
_A8_PATH = "reports/lgb_hyperparameter_heatmap.csv"
_A9_PATH = "reports/lgb_early_stopping_audit.json"
_ABLATION_PATH = "reports/lgb_raw_ablation_a.json"

_MODEL_OUTPUT = "models/lightgbm_best_v2.pkl"
_RESULTS_OUTPUT = "reports/lgb_integration_results.json"

_MIN_ROWS = 100_000
_N_TRIALS_DEFAULT = 100
_CV_N_SPLITS = 5   # matches A6 ablation regime; 10-fold produced noisier AUC estimates that biased Optuna toward shallow trees
_NUM_LEAVES_MAX_DEFAULT = 300

# Monotone constraints (all 7 directional features)
MONOTONE_CONSTRAINTS: dict[str, int] = {
    "AGE_YEARS": 1,
    "YEARS_EMPLOYED": 1,
    "CREDIT_INCOME_RATIO": -1,
    "EXT_SOURCE_1": 1,
    "EXT_SOURCE_2": 1,
    "EXT_SOURCE_3": 1,
    "inst_days_past_due_mean": -1,
}


# ---------------------------------------------------------------------------
# Config resolution from prior step outputs
# ---------------------------------------------------------------------------

def _load_a6(X: pd.DataFrame) -> str:
    """Resolve best booster from A6 results."""
    a6_path = Path(_A6_PATH)
    if not a6_path.exists():
        print(f"A6 results not found at {_A6_PATH} — defaulting to 'gbdt'")
        return "gbdt"
    with a6_path.open() as fh:
        data = json.load(fh)
    best = data.get("best_booster", "gbdt")
    best_gini = data.get("best_mean_gini", 0.0)
    print(f"A6: best booster = {best.upper()} (Gini={best_gini:.4f})")
    return best


def _load_a7(X: pd.DataFrame) -> dict[str, int] | None:
    """Resolve monotone constraints from A7 results (apply if delta >= 0.005)."""
    a7_path = Path(_A7_PATH)
    if not a7_path.exists():
        print(f"A7 results not found at {_A7_PATH} — applying all constraints by default")
        return {k: v for k, v in MONOTONE_CONSTRAINTS.items() if k in X.columns}
    with a7_path.open() as fh:
        data = json.load(fh)
    delta = data.get("delta_gini", 0.0)
    verdict = data.get("verdict", "unknown")
    print(f"A7: delta_gini={delta:+.4f} ({verdict})")
    if delta >= 0.005:
        active = data.get("active_constraints", MONOTONE_CONSTRAINTS)
        print(f"  Applying monotone constraints (delta >= 0.005): {len(active)} features")
        return active
    print(f"  Skipping monotone constraints (delta {delta:+.4f} < 0.005 threshold)")
    return None


def _load_a8(default_num_leaves_max: int) -> dict:
    """Resolve num_leaves bounds from A8 heatmap."""
    a8_path = Path(_A8_PATH)
    if not a8_path.exists():
        print(f"A8 results not found — using default num_leaves_max={default_num_leaves_max}")
        return {"num_leaves_max": default_num_leaves_max}
    pivot = pd.read_csv(a8_path, index_col=0)
    # Find best cell
    best_val = pivot.values.max()
    best_row_idx, best_col_idx = divmod(pivot.values.argmax(), pivot.shape[1])
    best_nl = int(pivot.index[best_row_idx])
    best_mc = int(pivot.columns[best_col_idx])
    print(f"A8: best grid cell — num_leaves={best_nl}, min_child_samples={best_mc} (Gini={best_val:.4f})")
    # Expand num_leaves_max to cover the best cell + headroom
    nl_max = max(best_nl + 50, default_num_leaves_max)
    return {
        "num_leaves_max": nl_max,
        "a8_best_num_leaves": best_nl,
        "a8_best_min_child_samples": best_mc,
    }


def _load_a9() -> int | None:
    """Resolve recommended patience from A9."""
    a9_path = Path(_A9_PATH)
    if not a9_path.exists():
        print(f"A9 results not found — using default patience=50")
        return 50
    with a9_path.open() as fh:
        data = json.load(fh)
    patience = data.get("recommended_patience")
    print(f"A9: recommended patience = {patience}")
    return patience


def _load_ablation_warmstart() -> list[dict] | None:
    """
    Build a warm-start trial from the ablation best params.

    The ablation used GBDT; its HP (num_leaves=125, max_depth=4, reg_lambda≈9.54)
    transferred well to GOSS in A6 (Gini 0.5547). Adding GOSS-specific defaults
    gives Optuna a strong prior anchored near the known optimum so TPE does not
    waste trials in the shallow-tree region.
    """
    ablation_path = Path(_ABLATION_PATH)
    if not ablation_path.exists():
        print(f"Ablation results not found at {_ABLATION_PATH} — no warm start")
        return None
    with ablation_path.open() as fh:
        data = json.load(fh)
    bp = data.get("best_params", {})
    warmstart: dict = {
        "num_leaves": int(bp.get("num_leaves", 125)),
        "max_depth": int(bp.get("max_depth", 4)),
        "learning_rate": float(bp.get("learning_rate", 0.031)),
        "n_estimators": int(bp.get("n_estimators", 947)),
        "min_child_samples": int(bp.get("min_child_samples", 90)),
        "subsample": float(bp.get("subsample", 0.94)),
        "colsample_bytree": float(bp.get("colsample_bytree", 0.62)),
        "reg_alpha": float(bp.get("reg_alpha", 4.25)),
        "reg_lambda": float(bp.get("reg_lambda", 9.54)),
        # GOSS-specific defaults (median of typical search range)
        "top_rate": 0.20,
        "other_rate": 0.10,
    }
    gini = data.get("gini", 0.0)
    print(f"Ablation warm start: num_leaves={warmstart['num_leaves']}, "
          f"max_depth={warmstart['max_depth']}, "
          f"reg_lambda={warmstart['reg_lambda']:.2f} "
          f"(source Gini={gini:.4f} GBDT → {0.5547:.4f} GOSS in A6)")
    return [warmstart]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_integration(n_trials: int = _N_TRIALS_DEFAULT) -> dict:
    """
    Full integration run combining A6–A9 findings.

    Parameters
    ----------
    n_trials : int
        Number of Optuna trials. Default 100.

    Returns
    -------
    dict
        Final evaluation metrics and configuration used.
    """
    import credit_engine.model as model_module
    from credit_engine.model import train_lightgbm_optuna

    print("=" * 60)
    print("LightGBM integration run")
    print("=" * 60)

    print("\nLoading data...")
    X = pd.read_parquet(_X_PATH)
    y = pd.read_parquet(_Y_PATH).squeeze()

    assert X.shape[0] >= _MIN_ROWS, (
        f"Pre-flight: {X.shape[0]} rows < {_MIN_ROWS}"
    )
    assert X.isnull().sum().sum() == 0, "NaN values in X."
    print(f"Data: {X.shape[0]:,} rows × {X.shape[1]} cols")

    # --- Resolve config from A6–A9 ---
    print("\nResolving config from A6–A9...")
    booster = _load_a6(X)
    monotone_constraints = _load_a7(X)
    a8_config = _load_a8(_NUM_LEAVES_MAX_DEFAULT)
    patience = _load_a9()
    warmstart = _load_ablation_warmstart()
    num_leaves_max = a8_config["num_leaves_max"]

    print(f"\nIntegration config:")
    print(f"  booster         = {booster.upper()}")
    print(f"  num_leaves_max  = {num_leaves_max}")
    print(f"  patience        = {patience}")
    print(f"  constraints     = {len(monotone_constraints) if monotone_constraints else 0} features")
    print(f"  n_trials        = {n_trials}")
    print(f"  cv_folds        = {_CV_N_SPLITS}")
    print(f"  warm start      = {'yes (ablation HP)' if warmstart else 'no'}")

    # Patch patience into the module constant for this run
    # (train_lightgbm_optuna uses _LGB_EARLY_STOPPING_ROUNDS for final refit)
    original_es = model_module._LGB_EARLY_STOPPING_ROUNDS
    if patience is not None:
        model_module._LGB_EARLY_STOPPING_ROUNDS = patience

    try:
        final_model, metrics, X_test, y_test, best_params = train_lightgbm_optuna(
            X, y,
            n_trials=n_trials,
            num_leaves_max=num_leaves_max,
            boosting_type=booster,
            monotone_constraints=monotone_constraints,
            enqueue_trials=warmstart,
        )
    finally:
        # Restore original constant regardless of outcome
        model_module._LGB_EARLY_STOPPING_ROUNDS = original_es

    gini = metrics.get("Gini", 0.0)
    print(f"\nFinal Gini: {gini:.4f}")

    # Determine verdict
    if gini > 0.50:
        verdict = "success_high"
        advice = "Track A highly successful. Run per-technique ablation to isolate contributions."
    elif gini > 0.48:
        verdict = "success_marginal"
        advice = "Track A marginal. Investigate which technique contributed most."
    else:
        verdict = "insufficient"
        advice = (
            "Gini below threshold. Run additional diagnostics (isolate each technique) "
            "before deciding whether to abandon raw path."
        )

    print(f"Verdict: {verdict}")
    print(f"  {advice}")

    # Save model
    from credit_engine.model import save_model
    out_model_path = Path(_MODEL_OUTPUT)
    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(final_model, _MODEL_OUTPUT)
    print(f"Model saved to {_MODEL_OUTPUT}")

    result = {
        "gini": round(float(gini), 6),
        "metrics": metrics,
        "best_params": {k: (float(v) if isinstance(v, (float, int)) else v) for k, v in best_params.items()},
        "config": {
            "booster": booster,
            "num_leaves_max": num_leaves_max,
            "patience": patience,
            "n_trials": n_trials,
            "cv_folds": _CV_N_SPLITS,
            "monotone_constraints": monotone_constraints,
            "a8_config": a8_config,
        },
        "verdict": verdict,
        "advice": advice,
    }

    out_path = Path(_RESULTS_OUTPUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(result, fh, indent=2)
    print(f"Results saved to {_RESULTS_OUTPUT}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightGBM integration run")
    parser.add_argument("--n-trials", type=int, default=_N_TRIALS_DEFAULT,
                        help=f"Number of Optuna trials (default {_N_TRIALS_DEFAULT})")
    args = parser.parse_args()
    run_integration(n_trials=args.n_trials)
