#!/usr/bin/env python3
"""Optuna hyperparameter tuning for LightGBM, optimizing top-decile precision.

Runs 50 trials over the search space defined in the ML improvement plan.
Saves best params to data/ml/lgbm_best_params.json for use by train_ml_model.py.

Usage:
    python scripts/tune_lightgbm.py
    python scripts/tune_lightgbm.py --binary
    python scripts/tune_lightgbm.py --trials 100 --dataset data/ml/training_dataset.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tradingagents.ml.feature_engineering import FEATURE_COLUMNS
from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)

OUTPUT_PATH = Path("data/ml/lgbm_best_params.json")


def remap_binary(y: np.ndarray) -> np.ndarray:
    """Collapse {-1,0,1} → {0,1}: LOSS+TIMEOUT → 0, WIN → 1."""
    return (y == 1).astype(int)


def time_split(df: pd.DataFrame, val_start: str = "2024-07-01") -> tuple:
    """Time-based split — same logic as train_ml_model.py."""
    df["date"] = pd.to_datetime(df["date"])
    val_start_dt = pd.Timestamp(val_start)
    train = df[df["date"] < val_start_dt]
    val = df[df["date"] >= val_start_dt]

    X_train = train[FEATURE_COLUMNS].values
    y_train = train["label"].values.astype(int)
    X_val = val[FEATURE_COLUMNS].values
    y_val = val["label"].values.astype(int)

    logger.info(f"Train: {len(train)} | Val: {len(val)}")
    return X_train, y_train, X_val, y_val


def top_decile_precision(y_true: np.ndarray, win_probs: np.ndarray) -> float:
    """Fraction of actual wins in the top 10% by predicted P(WIN)."""
    threshold = np.percentile(win_probs, 90)
    mask = win_probs >= threshold
    if mask.sum() == 0:
        return 0.0
    return float((y_true[mask] == 1).mean())


def objective(trial, X_train, y_train, X_val, y_val, binary: bool) -> float:
    """Optuna objective: maximize top-decile precision on validation set."""
    import lightgbm as lgb

    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
        "subsample_freq": 1,
        "verbose": -1,
        "seed": 42,
    }

    if binary:
        from collections import Counter
        class_counts = Counter(y_train)
        n_pos = class_counts.get(1, 1)
        n_neg = class_counts.get(0, 1)
        params["objective"] = "binary"
        params["metric"] = "binary_logloss"
        params["scale_pos_weight"] = n_neg / n_pos

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLUMNS)
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_COLUMNS, reference=train_data)

        booster = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
        )

        win_probs = booster.predict(X_val)  # shape (n,) — P(WIN)
        return top_decile_precision(y_val, win_probs)

    else:
        from collections import Counter
        y_train_mapped = y_train + 1
        y_val_mapped = y_val + 1
        class_counts = Counter(y_train_mapped)
        total = len(y_train_mapped)
        n_classes = len(class_counts)
        class_weight = {c: total / (n_classes * count) for c, count in class_counts.items()}
        sample_weights = np.array([class_weight[y] for y in y_train_mapped])

        params["objective"] = "multiclass"
        params["num_class"] = 3
        params["metric"] = "multi_logloss"

        train_data = lgb.Dataset(X_train, label=y_train_mapped, weight=sample_weights, feature_name=FEATURE_COLUMNS)
        val_data = lgb.Dataset(X_val, label=y_val_mapped, feature_name=FEATURE_COLUMNS, reference=train_data)

        booster = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
        )

        # probas shape (n, 3): class 2 = WIN (mapped from 1)
        probas = booster.predict(X_val)
        win_probs = probas[:, 2]
        return top_decile_precision(y_val, win_probs)


def main():
    parser = argparse.ArgumentParser(description="Optuna LightGBM hyperparameter tuning")
    parser.add_argument("--dataset", type=str, default="data/ml/training_dataset.parquet")
    parser.add_argument("--binary", action="store_true",
                        help="Binary mode: collapse LOSS+TIMEOUT → 0, WIN → 1")
    parser.add_argument("--trials", type=int, default=50,
                        help="Number of Optuna trials (default: 50)")
    parser.add_argument("--val-start", type=str, default="2024-07-01")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    try:
        import optuna
        import lightgbm  # noqa: F401
    except ImportError as e:
        logger.error(f"Missing dependency: {e}. Install with: pip install optuna lightgbm")
        sys.exit(1)

    # Suppress Optuna's verbose logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    logger.info(f"Loading dataset from {args.dataset}")
    df = pd.read_parquet(args.dataset)

    X_train, y_train, X_val, y_val = time_split(df, val_start=args.val_start)

    if args.binary:
        y_train = remap_binary(y_train)
        y_val = remap_binary(y_val)
        logger.info(f"Binary mode. WIN% train={y_train.mean():.1%} val={y_val.mean():.1%}")

    logger.info(f"Starting Optuna study: {args.trials} trials, objective=top_decile_precision")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))

    def _objective(trial):
        return objective(trial, X_train, y_train, X_val, y_val, binary=args.binary)

    study.optimize(_objective, n_trials=args.trials, show_progress_bar=True)

    best = study.best_trial
    logger.info(f"\nBest trial #{best.number}: top_decile_precision={best.value:.4f}")
    logger.info(f"Best params: {best.params}")

    # Save best params (without framework-specific keys — those are set by train_lightgbm)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(best.params, f, indent=2)
    logger.info(f"Best params saved to {output_path}")

    # Print top-5 trials
    logger.info("\nTop 5 trials by top_decile_precision:")
    sorted_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:5]
    for i, t in enumerate(sorted_trials):
        logger.info(f"  #{i+1} trial={t.number} precision={t.value:.4f} lr={t.params.get('learning_rate', '?'):.4f}")


if __name__ == "__main__":
    main()
