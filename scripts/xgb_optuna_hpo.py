#!/usr/bin/env python3
"""
XGBoost 100-trial Bayesian hyperparameter optimization via Optuna.

Implements Phase 04.2 Plan 02: Exhaustive XGBoost search space exploration
using Optuna's TPE sampler with persistence to SQLite. Ensures non-regression
via seed trial from Phase 04.1 baseline and enforces temporal CV for OOF
generation per Wave 0 diagnosis.

Command:
    python scripts/xgb_optuna_hpo.py

Output:
    - models/xgboost_hpo_best.pkl: Best XGBoost model from 100 trials
    - reports/xgb_hpo_results.json: HPO results, convergence curve, trial history
    - models/optuna_studies.db: Updated with 100 completed trials
"""

import json
import sys
from pathlib import Path

import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup conftest alias if not already present
if "credit_engine" not in sys.modules:
    import src
    sys.modules["credit_engine"] = src

# Now import from credit_engine (which is aliased to src)
from credit_engine.model import (
    _xgboost_optuna_objective,
    _make_cv,
    evaluate_model,
    save_model,
)

# ============================================================================
# Constants
# ============================================================================

OPTUNA_DB_PATH = "sqlite:///models/optuna_studies.db"
OPTUNA_STUDY_NAME = "xgboost_extended_study"  # Matches LGB/CatBoost naming from Phase 04.1
TOTAL_TRIALS_TARGET = 100
N_TRIALS_STARTUP = 5
RANDOM_STATE = 42

# Feature and target paths
X_FEATURES_PATH = "data/processed/X_xgb_features.parquet"
Y_TRAIN_PATH = "data/processed/y_train.parquet"

# Output paths
HPO_RESULTS_PATH = "reports/xgb_hpo_results.json"
HPO_MODEL_PATH = "models/xgboost_hpo_best.pkl"

# ============================================================================
# Main Script
# ============================================================================


def main():
    """Run XGBoost 100-trial Bayesian HPO with Optuna persistence."""

    print("[XGBoost HPO] Loading feature data...")
    X = pd.read_parquet(X_FEATURES_PATH)
    y = pd.read_parquet(Y_TRAIN_PATH).squeeze()

    # Align shapes: X_xgb_features is a subset (246008 rows), match y to it
    y = y.iloc[:X.shape[0]]

    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Default rate: {y.sum() / len(y):.4f}")

    # =========================================================================
    # Split data for HPO
    # =========================================================================

    print(f"\n[XGBoost HPO] Performing train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape: {X_test.shape}")

    # Extract groups from temporal sort column for CV
    groups_train = X_train["prev_days_decision_mean"].to_numpy()
    scale_pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())
    cv = _make_cv(groups_train, n_splits=5)

    print(f"  scale_pos_weight: {scale_pos_weight:.4f}")

    # =========================================================================
    # Initialize or load Optuna study
    # =========================================================================

    print(f"\n[XGBoost HPO] Initializing Optuna study from {OPTUNA_DB_PATH}")

    storage = optuna.storages.RDBStorage(OPTUNA_DB_PATH)

    try:
        study = optuna.load_study(
            study_name=OPTUNA_STUDY_NAME,
            storage=storage
        )
        print(f"  Loaded existing study: {len(study.trials)} trials completed")
    except KeyError:
        # Study doesn't exist; create new
        print(f"  Study does not exist; creating new...")
        study = optuna.create_study(
            study_name=OPTUNA_STUDY_NAME,
            storage=storage,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=RANDOM_STATE,
                n_startup_trials=N_TRIALS_STARTUP
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=20
            )
        )
        print(f"  Created new study with TPE sampler")

    # Ensure non-regression: if this is a fresh study, seed with Phase 04.1
    # baseline. The seed is already in the DB from Wave 0 (04.2 Plan 01).
    if len(study.trials) == 0:
        print(f"  WARNING: Study is empty; seeding with baseline params...")
        baseline_params = {
            "n_estimators": 400,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "gamma": 0.5,
            "max_delta_step": 0,
            "reg_alpha": 0.5,
            "reg_lambda": 2.0,
        }
        study.enqueue_trial(baseline_params)
        print(f"  Seeded trial 0 with baseline params")

    # =========================================================================
    # Run HPO to reach 100 total trials
    # =========================================================================

    n_trials_remaining = TOTAL_TRIALS_TARGET - len(study.trials)
    print(f"\n[XGBoost HPO] Running {n_trials_remaining} trials "
          f"(from {len(study.trials)} → {TOTAL_TRIALS_TARGET})...")

    if n_trials_remaining > 0:
        def objective(trial: optuna.Trial) -> float:
            """Optuna objective: OOF AUC from XGBoost Bayesian HPO."""
            return _xgboost_optuna_objective(
                trial, X_train, y_train, scale_pos_weight, cv
            )

        study.optimize(
            objective,
            n_trials=n_trials_remaining,
            show_progress_bar=True
        )
        print(f"  Completed {len(study.trials)} total trials")
    else:
        print(f"  Target reached; no additional trials needed")

    # =========================================================================
    # Extract best trial and retrain
    # =========================================================================

    print(f"\n[XGBoost HPO] Extracting best trial results...")
    best_trial = study.best_trial
    print(f"  Best trial: #{best_trial.number}, value={best_trial.value:.4f}")

    best_params = best_trial.params
    print(f"  Best params: {json.dumps(best_params, indent=4)}")

    # Retrain on full X_train with best params
    best_model = xgb.XGBClassifier(
        **best_params,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        use_label_encoder=False,
        verbosity=0,
        random_state=RANDOM_STATE,
    )
    best_model.fit(X_train, y_train)

    # Evaluate on test set
    metrics = evaluate_model(best_model, X_test, y_test, "XGBoost")
    print(f"\n[XGBoost HPO] Test set metrics:")
    print(f"  Gini: {metrics.get('Gini', 'N/A'):.4f}")
    print(f"  AUC-ROC: {metrics.get('AUC-ROC', 'N/A'):.4f}")
    print(f"  KS: {metrics.get('KS', 'N/A'):.4f}")

    # =========================================================================
    # Compute convergence curve
    # =========================================================================

    print(f"\n[XGBoost HPO] Computing convergence metrics...")
    convergence = {}

    # Extract best value at each trial for convergence tracking
    trial_values = [t.value for t in study.trials if t.value is not None]
    trial_best_values = []
    current_best = float('-inf')
    for val in trial_values:
        current_best = max(current_best, val)
        trial_best_values.append(current_best)

    # Median best value at key trial counts
    if len(trial_best_values) >= 50:
        convergence["median_best_at_trial_50"] = trial_best_values[49]
    if len(trial_best_values) >= 75:
        convergence["median_best_at_trial_75"] = trial_best_values[74]
    if len(trial_best_values) >= 100:
        convergence["final_best_at_trial_100"] = trial_best_values[99]
    else:
        convergence["final_best_at_trial_100"] = trial_best_values[-1]

    print(f"  Convergence curve: {json.dumps(convergence, indent=4)}")

    # =========================================================================
    # Compute non-regression check
    # =========================================================================

    phase_04_1_baseline = 0.5567  # XGB standalone Gini from Phase 04.1
    oof_gini = 2 * best_trial.value - 1  # Convert AUC to Gini

    non_regression_passed = oof_gini >= (phase_04_1_baseline - 0.001)
    improvement = oof_gini - phase_04_1_baseline

    print(f"\n[XGBoost HPO] Non-regression check:")
    print(f"  Phase 04.1 baseline (standalone XGB): {phase_04_1_baseline:.4f}")
    print(f"  Best trial Gini: {oof_gini:.4f}")
    print(f"  Improvement: {improvement:+.4f}")
    print(f"  Passed: {non_regression_passed}")

    # =========================================================================
    # Save artifacts
    # =========================================================================

    print(f"\n[XGBoost HPO] Saving artifacts...")

    # Save model
    save_model(best_model, HPO_MODEL_PATH)
    print(f"  Saved model to {HPO_MODEL_PATH}")

    # Save results JSON
    results = {
        "phase": "04.2",
        "plan": "02",
        "hpo_type": "xgboost_100_trial_optuna",
        "total_trials": len(study.trials),
        "best_trial_number": best_trial.number,
        "best_oof_auc": float(best_trial.value),
        "best_oof_gini": float(oof_gini),
        "best_test_gini": float(metrics.get('Gini', 0.0)),
        "best_test_auc": float(metrics.get('AUC-ROC', 0.0)),
        "best_params": best_params,
        "improvement_vs_phase_04_1_baseline": float(improvement),
        "convergence": convergence,
        "model_file": HPO_MODEL_PATH,
        "non_regression_check": {
            "seed_trial_value": phase_04_1_baseline,
            "best_trial_value": float(oof_gini),
            "passed": non_regression_passed,
            "tolerance": 0.001,
        }
    }

    Path(HPO_RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(HPO_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved results to {HPO_RESULTS_PATH}")

    # =========================================================================
    # Summary
    # =========================================================================

    print(f"\n{'='*70}")
    print(f"[XGBoost HPO] COMPLETE")
    print(f"{'='*70}")
    print(f"Total trials: {len(study.trials)}")
    print(f"Best trial #{best_trial.number}: OOF Gini = {oof_gini:.4f}")
    print(f"Improvement vs Phase 04.1: {improvement:+.4f}")
    print(f"Non-regression: {'PASS' if non_regression_passed else 'FAIL'}")
    print(f"\nArtifacts:")
    print(f"  - Model: {HPO_MODEL_PATH}")
    print(f"  - Results: {HPO_RESULTS_PATH}")
    print(f"  - Optuna DB: {OPTUNA_DB_PATH.replace('sqlite:///', '')}")

    return results


if __name__ == "__main__":
    main()
