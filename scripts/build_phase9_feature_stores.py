"""
build_phase9_feature_stores.py
-------------------------------
Rebuild all 4 model-specific feature stores (v2) for Phase 04.2.9.

This script integrates Wave 2 temporal trajectory features (24 features from
secondary tables) into the tree feature store, then saves the 4 stores:
  - X_base_v2.parquet  (145 + Wave 2 = ~169 cols)
  - X_lgb_v2.parquet   (identical to X_base_v2)
  - X_xgb_v2.parquet   (identical to X_base_v2)
  - X_cat_v2.parquet   (X_base_v2 + 4 categorical string cols = ~173 cols)

Wave 2 features added:
  bbal_*  : bureau_balance DPD trajectory (10 features)
  inst_*  : installment payment velocity  (5 features)
  cc_*    : credit card balance trajectory (4 features)
  prev_*  : previous application signals  (4 features)
  current_to_bureau_debt_ratio            (1 feature)

Usage
-----
    python scripts/build_phase9_feature_stores.py [--data-dir data/]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_loader import build_training_frame, load_secondary_raw
from src.features import build_tree_feature_store

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CAT_COLS = [
    "ORGANIZATION_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_INCOME_TYPE",
    "OCCUPATION_TYPE",
]

_OUTPUT_DIR = _PROJECT_ROOT / "data" / "processed"


def _build_cat_store(X_base: pd.DataFrame) -> pd.DataFrame:
    """Add 4 categorical string columns to base store for CatBoost."""
    X_cat = X_base.copy()
    for col in _CAT_COLS:
        if col in X_cat.columns:
            X_cat[col] = X_cat[col].astype(str).replace("nan", "Unknown")
    return X_cat


def main(data_dir: str) -> None:
    data_path = Path(data_dir)
    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Load X / y
    # ------------------------------------------------------------------
    print("Loading training frame …")
    X_train, y_train = build_training_frame(data_dir=str(data_path))
    print(f"  X_train shape: {X_train.shape}, y_train positives: {y_train.sum()}")

    # ------------------------------------------------------------------
    # 2. Load raw secondary tables for Wave 2
    # ------------------------------------------------------------------
    print("Loading secondary raw tables (Wave 2) …")
    t1 = time.time()
    raw = load_secondary_raw(data_path)
    print(f"  bbal: {raw['bbal'].shape}  inst: {raw['inst'].shape}  "
          f"cc: {raw['cc'].shape}  prev: {raw['prev'].shape}  "
          f"[{time.time() - t1:.1f}s]")

    # Also load df_inst in Wave-1 format (SK_ID_CURR as column) for Wave 1 features
    import pandas as _pd
    df_inst_wave1 = _pd.read_csv(
        data_path / "installments_payments.csv",
        usecols=["SK_ID_CURR", "DAYS_ENTRY_PAYMENT", "DAYS_INSTALMENT",
                 "AMT_PAYMENT", "AMT_INSTALMENT"],
    )

    # ------------------------------------------------------------------
    # 3. Build base store (Wave 1 + Wave 2 + EXT_SOURCE_NUM_AVAILABLE)
    # ------------------------------------------------------------------
    print("Building X_base_v2 (Wave 1 + Wave 2 features) …")
    t2 = time.time()
    X_base, _ = build_tree_feature_store(
        X=X_train,
        y=y_train,
        output_dir=None,          # skip auto-write to X_tree_raw.parquet
        df_inst=df_inst_wave1,    # Wave 1
        df_bbal=raw["bbal"],      # Wave 2
        df_cc=raw["cc"],          # Wave 2
        df_prev=raw["prev"],      # Wave 2
    )
    print(f"  X_base shape: {X_base.shape}  [{time.time() - t2:.1f}s]")

    # Add EXT_SOURCE_NUM_AVAILABLE (Phase 04.2.9.01 protected feature)
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    _ext_present = [c for c in ext_cols if c in X_base.columns]
    if _ext_present:
        X_base["EXT_SOURCE_NUM_AVAILABLE"] = X_base[_ext_present].notna().sum(axis=1).astype(float)
    elif "EXT_SOURCE_NUM_AVAILABLE" not in X_base.columns:
        # Derive from original X_train if stripped by feature pipeline
        X_base["EXT_SOURCE_NUM_AVAILABLE"] = (
            X_train[[c for c in ext_cols if c in X_train.columns]]
            .notna()
            .sum(axis=1)
            .reindex(X_base.index)
            .fillna(0.0)
            .astype(float)
        )
    print(f"  EXT_SOURCE_NUM_AVAILABLE: {X_base['EXT_SOURCE_NUM_AVAILABLE'].value_counts().to_dict()}")

    # Embed TARGET for train_*_optuna compatibility
    X_base_with_target = X_base.copy()
    X_base_with_target["TARGET"] = y_train.reindex(X_base.index).values

    # ------------------------------------------------------------------
    # 4. Save X_base_v2
    # ------------------------------------------------------------------
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path_base = _OUTPUT_DIR / "X_base_v2.parquet"
    X_base_with_target.to_parquet(path_base, index=False)
    print(f"  Saved X_base_v2.parquet  {X_base_with_target.shape}")

    # ------------------------------------------------------------------
    # 5. X_lgb_v2 = X_base_v2 (raw continuous preferred by LGB leaf-wise)
    # ------------------------------------------------------------------
    path_lgb = _OUTPUT_DIR / "X_lgb_v2.parquet"
    X_base_with_target.to_parquet(path_lgb, index=False)
    print(f"  Saved X_lgb_v2.parquet  {X_base_with_target.shape}  (== X_base_v2)")

    # ------------------------------------------------------------------
    # 6. X_xgb_v2 = X_base_v2 (pragmatic baseline, same as LGB for now)
    # ------------------------------------------------------------------
    path_xgb = _OUTPUT_DIR / "X_xgb_v2.parquet"
    X_base_with_target.to_parquet(path_xgb, index=False)
    print(f"  Saved X_xgb_v2.parquet  {X_base_with_target.shape}  (== X_base_v2)")

    # ------------------------------------------------------------------
    # 7. X_cat_v2 = X_base_v2 + 4 categorical string columns
    # ------------------------------------------------------------------
    # Re-derive categorical cols from X_train (they were dropped as non-numeric)
    X_cat = X_base_with_target.copy()
    for col in _CAT_COLS:
        if col in X_train.columns:
            X_cat[col] = (
                X_train[col]
                .reindex(X_base.index)
                .astype(str)
                .replace("nan", "Unknown")
            )
        else:
            X_cat[col] = "Unknown"

    path_cat = _OUTPUT_DIR / "X_cat_v2.parquet"
    X_cat.to_parquet(path_cat, index=False)
    print(f"  Saved X_cat_v2.parquet  {X_cat.shape}  (base + {len(_CAT_COLS)} categorical cols)")

    # ------------------------------------------------------------------
    # 8. Validation checks
    # ------------------------------------------------------------------
    print("\nValidation checks …")

    wave2_cols = [
        "bbal_ever_30dpd", "bbal_ever_60dpd", "bbal_ever_90dpd",
        "bbal_pct_current", "bbal_dpd_escalation",
        "inst_payment_consistency_score", "inst_recency_weighted_dpd",
        "inst_late_payment_acceleration",
        "cc_balance_velocity_3m", "cc_utilization_trend",
        "prev_reject_fraud_flag", "current_to_bureau_debt_ratio",
    ]
    for col in wave2_cols:
        present = col in X_base.columns
        print(f"  {'✓' if present else '✗'} {col}")

    assert X_base_with_target.isnull().sum().sum() == 0, "NaN found in X_base_v2"
    assert not np.isinf(X_base_with_target.select_dtypes("number").values).any(), "Inf found in X_base_v2"
    assert "SK_DPD" not in X_base.columns, "Leaky SK_DPD found in X_base_v2"
    assert "SK_DPD_DEF" not in X_base.columns, "Leaky SK_DPD_DEF found in X_base_v2"
    for col in _CAT_COLS:
        assert col in X_cat.columns, f"Missing categorical column: {col}"
        assert X_cat[col].dtype == object, f"Categorical column not object dtype: {col}"
        assert (X_cat[col] == "nan").sum() == 0, f"'nan' string found in {col}"

    print(f"\nDone in {time.time() - t0:.1f}s")
    print(f"X_base_v2:  {X_base_with_target.shape}")
    print(f"X_cat_v2:   {X_cat.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild Phase 04.2.9 feature stores")
    parser.add_argument("--data-dir", default="data/", help="Directory with raw CSV files")
    args = parser.parse_args()
    main(args.data_dir)
