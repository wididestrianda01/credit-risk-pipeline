"""
build_featuretools_store.py
---------------------------
Run Deep Feature Synthesis on the 7-table Home Credit entity set.
Generates aggregated features via featuretools, applies IV + correlation
dedup, and saves the result to data/processed/X_featuretools.parquet.

Must be run AFTER X_raw_features.parquet and y_train.parquet exist.
Expected runtime: 10–30 min depending on n_jobs and max_depth.

Usage
-----
    python -u scripts/build_featuretools_store.py
"""

import sys
import time
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import src  # noqa: E402
sys.modules["credit_engine"] = src

from credit_engine.auto_features import build_featuretools_feature_store  # noqa: E402

_DATA_DIR = project_root / "data"
_Y_PATH = project_root / "data" / "processed" / "y_train.parquet"
_OUTPUT_PATH = project_root / "data" / "processed" / "X_featuretools.parquet"
_FEATURE_DEFS_PATH = project_root / "models" / "featuretools_feature_defs.pkl"
_SELECTED_COLS_PATH = project_root / "models" / "featuretools_selected_cols.json"

_AGG_PRIMITIVES = ["mean", "std", "min", "max", "count", "skew", "median"]
_MAX_DEPTH = 1
_IV_THRESHOLD = 0.02
_CORR_THRESHOLD = 0.90
_N_JOBS = -1


def main() -> None:
    import json
    import joblib

    print("=" * 70)
    print("Featuretools Deep Feature Synthesis")
    print("=" * 70)
    print(f"  Data dir:   {_DATA_DIR}")
    print(f"  max_depth:  {_MAX_DEPTH}")
    print(f"  primitives: {_AGG_PRIMITIVES}")
    print(f"  IV >= {_IV_THRESHOLD}, |r| threshold {_CORR_THRESHOLD}")
    print(f"  n_jobs:     {_N_JOBS}")

    y_train = pd.read_parquet(_Y_PATH).squeeze()
    print(f"\ny_train loaded: {len(y_train):,} rows, default rate {y_train.mean():.2%}")

    t0 = time.time()
    feature_matrix, feature_defs, selected_cols = build_featuretools_feature_store(
        data_dir=_DATA_DIR,
        y_train=y_train,
        output_path=_OUTPUT_PATH,
        agg_primitives=_AGG_PRIMITIVES,
        max_depth=_MAX_DEPTH,
        iv_threshold=_IV_THRESHOLD,
        corr_threshold=_CORR_THRESHOLD,
        n_jobs=_N_JOBS,
    )
    elapsed = time.time() - t0

    joblib.dump(feature_defs, _FEATURE_DEFS_PATH)
    _SELECTED_COLS_PATH.write_text(json.dumps(selected_cols, indent=2))

    print(f"\nDFS complete in {elapsed / 60:.1f} min")
    print(f"  Feature matrix shape:  {feature_matrix.shape}")
    print(f"  Selected columns:      {len(selected_cols)}")
    print(f"  Saved parquet:         {_OUTPUT_PATH.name}")
    print(f"  Saved feature defs:    {_FEATURE_DEFS_PATH.name}")
    print(f"  Saved selected cols:   {_SELECTED_COLS_PATH.name}")
    print("\nTop 10 selected features:")
    for col in selected_cols[:10]:
        print(f"    {col}")
    print("=" * 70)


if __name__ == "__main__":
    main()
