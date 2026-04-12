"""
Plan 00 prerequisite: Rebuild data/processed/X_features.parquet

The WoE store is currently a corrupt 500×2 test artifact. This script:
1. Loads all 7 tables via build_training_frame (returns X, y)
2. Calls build_feature_store(X, y) to fit WoE bins on full training data
3. Writes X_features.parquet (~307,511 × 68 WoE-encoded features)

Expected runtime: 10–20 min (7-table join + WoE fitting on 307K rows)
"""
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_loader import build_training_frame
from src.features import build_feature_store

DATA_DIR = _PROJECT_ROOT / "data"
OUTPUT_PATH = _PROJECT_ROOT / "data" / "processed" / "X_features.parquet"


def main() -> None:
    t0 = time.time()
    print(f"[rebuild_woe_store] Loading 7-table training frame from {DATA_DIR} ...", flush=True)
    X, y = build_training_frame(str(DATA_DIR))
    print(f"[rebuild_woe_store] Loaded: X={X.shape}, y={y.shape} — {time.time()-t0:.1f}s", flush=True)

    print("[rebuild_woe_store] Building WoE feature store ...", flush=True)
    X_final, woe_mappings = build_feature_store(X, y, output_dir=OUTPUT_PATH.parent)
    elapsed = time.time() - t0
    print(
        f"[rebuild_woe_store] Done: X_features={X_final.shape}, "
        f"features={len(woe_mappings)}, elapsed={elapsed:.1f}s",
        flush=True,
    )

    # Verify output
    import pandas as pd
    df_check = pd.read_parquet(OUTPUT_PATH)
    print(f"[rebuild_woe_store] Verified on disk: {df_check.shape}", flush=True)


if __name__ == "__main__":
    main()
