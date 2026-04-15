#!/usr/bin/env python
"""
evaluate_v3_fairness.py
-----------------------
Dual-load fairness evaluation for all three EU AI Act–compliant v3 models.

Dual-load pattern:
- v3 parquet (163/167 cols, no AGE_YEARS): used for model prediction
- X_train.parquet: provides CODE_GENDER (SK_ID_CURR-aligned)
- v2 parquet: provides AGE_YEARS (SK_ID_CURR-aligned)

compute_fairness_metrics() internally drops sensitive_cols before calling
predict_proba, so the model sees exactly its v3 training features while
the function groups predictions by age/gender from the side-loaded sources.

Gate criteria (EU AI Act Art. 6 / ECHR Art. 14):
- Age DIR  >= 0.80 (primary gate — removes direct age discrimination)
- Gender DIR >= 0.80 (must not regress from v2 baseline of 0.956)

v2 baseline:
- Gender DIR = 0.956 (passed)
- Age DIR    = 0.346 (failed — Young vs Senior gap drove by age features)

Output:
- reports/fairness_metrics_v3.csv   (one row per group per model)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.explain import compute_fairness_metrics
from src.model_base import _TEMPORAL_SORT_COL, _TEST_SIZE, load_model

_V2_BASELINE_AGE_DIR: float = 0.346
_V2_BASELINE_GENDER_DIR: float = 0.956
_DIR_THRESHOLD: float = 0.80

_MODELS: dict[str, Path] = {
    "xgboost": _PROJECT_ROOT / "models" / "xgboost_v3_calibrated.pkl",
    "lightgbm": _PROJECT_ROOT / "models" / "lightgbm_v3_calibrated.pkl",
    "catboost": _PROJECT_ROOT / "models" / "catboost_v3_calibrated.pkl",
}

_STORES_V3: dict[str, Path] = {
    "xgboost": _PROJECT_ROOT / "data" / "processed" / "X_xgb_v3.parquet",
    "lightgbm": _PROJECT_ROOT / "data" / "processed" / "X_lgb_v3.parquet",
    "catboost": _PROJECT_ROOT / "data" / "processed" / "X_cat_v3.parquet",
}

_STORES_V2: dict[str, Path] = {
    "xgboost": _PROJECT_ROOT / "data" / "processed" / "X_xgb_v2.parquet",
    "lightgbm": _PROJECT_ROOT / "data" / "processed" / "X_lgb_v2.parquet",
    "catboost": _PROJECT_ROOT / "data" / "processed" / "X_cat_v2.parquet",
}

_X_TRAIN_PATH = _PROJECT_ROOT / "data" / "processed" / "X_train.parquet"
_FAIRNESS_CSV_PATH = _PROJECT_ROOT / "reports" / "fairness_metrics_v3.csv"


def _load_external_demographics(sk_ids: np.ndarray) -> tuple[pd.Series, pd.Series]:
    """
    Load CODE_GENDER from X_train.parquet and AGE_YEARS from X_xgb_v2.parquet,
    aligned to the given SK_ID_CURR values.

    Parameters
    ----------
    sk_ids : array of SK_ID_CURR values for the OOT set

    Returns
    -------
    code_gender : Series (index=sk_ids)
    age_years   : Series (index=sk_ids)
    """
    # CODE_GENDER from raw joined data (only store that retains raw string column)
    X_raw = pd.read_parquet(_X_TRAIN_PATH, columns=["SK_ID_CURR", "CODE_GENDER"])
    X_raw = X_raw.set_index("SK_ID_CURR")
    code_gender = X_raw.loc[sk_ids, "CODE_GENDER"]

    # AGE_YEARS from any v2 store (all three are identical for this column)
    X_v2 = pd.read_parquet(_STORES_V2["xgboost"], columns=["SK_ID_CURR", "AGE_YEARS"])
    X_v2 = X_v2.set_index("SK_ID_CURR")
    age_years = X_v2.loc[sk_ids, "AGE_YEARS"]

    return code_gender, age_years


def evaluate_model_fairness(
    model_name: str,
    model: object,
) -> pd.DataFrame:
    """
    Evaluate fairness for a single v3 model using the dual-load pattern.

    Returns
    -------
    DataFrame from compute_fairness_metrics with an extra 'model' column.
    """
    print(f"\n{'='*70}")
    print(f"Evaluating {model_name.upper()} v3")
    print(f"{'='*70}")

    # --- 1. Load v3 store, pop TARGET, temporal sort ----------------------
    X = pd.read_parquet(_STORES_V3[model_name])
    print(f"  v3 store loaded: {X.shape}")

    assert "AGE_YEARS" not in X.columns, f"{model_name} v3 still has AGE_YEARS — wrong store"
    assert "CODE_GENDER" not in X.columns, f"{model_name} v3 still has CODE_GENDER — wrong store"

    y = X.pop("TARGET")
    sort_idx = X[_TEMPORAL_SORT_COL].argsort()
    X = X.iloc[sort_idx].reset_index(drop=True)
    y = y.iloc[sort_idx].reset_index(drop=True)

    # --- 2. Carve OOT (same split used during training) -------------------
    test_start = int(len(X) * (1.0 - _TEST_SIZE))
    X_oot = X.iloc[test_start:].copy()
    y_oot = y.iloc[test_start:]

    sk_ids_oot = X_oot[_TEMPORAL_SORT_COL].values

    # Drop sort key — not a model feature
    X_oot = X_oot.drop(columns=[_TEMPORAL_SORT_COL])
    print(f"  OOT set: {X_oot.shape}  |  default rate: {y_oot.mean():.4f}")

    # --- 3. Side-load protected attributes (NOT from v3 model input) ------
    code_gender, age_years = _load_external_demographics(sk_ids_oot)

    # Build fairness frame: model features + protected grouping columns
    # compute_fairness_metrics will strip protected cols before predict_proba
    X_fairness = X_oot.copy()
    X_fairness["CODE_GENDER"] = code_gender.values
    X_fairness["AGE_YEARS"] = age_years.values
    print(f"  Fairness frame: {X_fairness.shape}  (+CODE_GENDER, +AGE_YEARS side-loaded)")

    # --- 4. Compute fairness metrics using existing infrastructure --------
    fairness_df = compute_fairness_metrics(
        model=model,
        X=X_fairness,
        y=y_oot,
        sensitive_cols=["CODE_GENDER", "AGE_YEARS"],
    )

    # --- 5. Extract and print DIRs ----------------------------------------
    age_rows = fairness_df[fairness_df["group_name"].str.startswith("Age:")]
    gender_rows = fairness_df[fairness_df["group_name"].str.startswith("Gender:")]

    age_dir = age_rows["demographic_parity_disparate_impact"].iloc[0] if len(age_rows) > 0 else 0.0
    gender_dir = gender_rows["demographic_parity_disparate_impact"].iloc[0] if len(gender_rows) > 0 else 0.0

    age_pass = age_dir >= _DIR_THRESHOLD
    gender_pass = gender_dir >= _DIR_THRESHOLD

    print(f"\n  Age DIR:    {age_dir:.4f}  {'✓ PASS' if age_pass else '✗ BELOW THRESHOLD'}")
    print(f"  Gender DIR: {gender_dir:.4f}  {'✓ PASS' if gender_pass else '✗ BELOW THRESHOLD'}")
    print(f"  Gate:       {'✓ PASS' if (age_pass and gender_pass) else '✗ INVESTIGATE'}")

    fairness_df = fairness_df.copy()
    fairness_df.insert(0, "model", model_name)
    return fairness_df


def main() -> None:
    """Run fairness evaluation for all three v3 models and produce the gate report."""

    print("Loading v3 models...")
    models: dict[str, object] = {}
    for name, path in _MODELS.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} model not found: {path}")
        models[name] = load_model(str(path))
        print(f"  {name}: {path.name} loaded")

    all_results: list[pd.DataFrame] = []
    gate_status: dict[str, bool] = {}
    age_dirs: dict[str, float] = {}

    for model_name, model in models.items():
        fairness_df = evaluate_model_fairness(model_name, model)
        all_results.append(fairness_df)

        age_rows = fairness_df[fairness_df["group_name"].str.startswith("Age:")]
        gender_rows = fairness_df[fairness_df["group_name"].str.startswith("Gender:")]

        age_dir = age_rows["demographic_parity_disparate_impact"].iloc[0] if len(age_rows) > 0 else 0.0
        gender_dir = gender_rows["demographic_parity_disparate_impact"].iloc[0] if len(gender_rows) > 0 else 0.0

        age_dirs[model_name] = age_dir
        gate_status[model_name] = (age_dir >= _DIR_THRESHOLD) and (gender_dir >= _DIR_THRESHOLD)

    # --- Save fairness CSV ------------------------------------------------
    results_df = pd.concat(all_results, ignore_index=True)
    _FAIRNESS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(_FAIRNESS_CSV_PATH, index=False)
    print(f"\n✓ Fairness metrics saved: {_FAIRNESS_CSV_PATH}")

    # --- Print gate summary -----------------------------------------------
    print(f"\n{'='*70}")
    print("PHASE 04.4 FAIRNESS GATE RESULTS")
    print(f"{'='*70}")
    print(f"  Threshold: Age DIR ≥ {_DIR_THRESHOLD:.2f}  |  Gender DIR ≥ {_DIR_THRESHOLD:.2f}")
    print(f"\n  {'Model':<12} {'Age DIR':>9} {'Gender DIR':>11} {'Gate':>10}")
    print(f"  {'-'*12} {'-'*9} {'-'*11} {'-'*10}")

    passing_models: list[str] = []
    for df in all_results:
        name = df["model"].iloc[0]
        age_rows = df[df["group_name"].str.startswith("Age:")]
        gender_rows = df[df["group_name"].str.startswith("Gender:")]
        a = age_rows["demographic_parity_disparate_impact"].iloc[0] if len(age_rows) > 0 else 0.0
        g = gender_rows["demographic_parity_disparate_impact"].iloc[0] if len(gender_rows) > 0 else 0.0
        gate = gate_status[name]
        if gate:
            passing_models.append(name)
        print(f"  {name:<12} {a:>9.4f} {g:>11.4f} {'✓ PASS':>10}" if gate else
              f"  {name:<12} {a:>9.4f} {g:>11.4f} {'✗ FAIL':>10}")

    print(f"\n  v2 baseline — Age DIR: {_V2_BASELINE_AGE_DIR:.3f}  |  Gender DIR: {_V2_BASELINE_GENDER_DIR:.3f}")
    print(f"\n{'='*70}")

    if passing_models:
        best = max(passing_models, key=lambda n: age_dirs[n])
        print(f"GATE: PASS — {len(passing_models)} model(s) meet EU AI Act Art. 6 fairness criteria")
        print(f"Recommended for Phase 05.1: {best.upper()} v3 (highest Age DIR)")
    else:
        print("GATE: INVESTIGATE — No v3 models meet Age DIR ≥ 0.80")
        print("Recommendation: Investigate proxy features or rollback to v2 with enhanced fairness mitigations")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
