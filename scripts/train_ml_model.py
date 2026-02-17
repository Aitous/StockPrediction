#!/usr/bin/env python3
"""Train ML model on the generated dataset.

Supports TabPFN (recommended, requires GPU or API) and LightGBM (fallback).
Uses time-based train/validation split to prevent data leakage.

Usage:
    python scripts/train_ml_model.py
    python scripts/train_ml_model.py --model lightgbm
    python scripts/train_ml_model.py --model tabpfn --dataset data/ml/training_dataset.parquet
    python scripts/train_ml_model.py --max-train-samples 5000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tradingagents.ml.feature_engineering import FEATURE_COLUMNS
from tradingagents.ml.predictor import LGBMWrapper, MLPredictor
from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)

DATA_DIR = Path("data/ml")
LABEL_NAMES = {-1: "LOSS", 0: "TIMEOUT", 1: "WIN"}


def load_dataset(path: str) -> pd.DataFrame:
    """Load and validate the training dataset."""
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} samples from {path}")

    # Validate columns
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if "label" not in df.columns:
        raise ValueError("Missing 'label' column")
    if "date" not in df.columns:
        raise ValueError("Missing 'date' column")

    # Show label distribution
    for label, name in LABEL_NAMES.items():
        count = (df["label"] == label).sum()
        pct = count / len(df) * 100
        logger.info(f"  {name:>7} ({label:+d}): {count:>7} ({pct:.1f}%)")

    return df


def time_split(
    df: pd.DataFrame,
    val_start: str = "2024-07-01",
    max_train_samples: int | None = None,
) -> tuple:
    """Split dataset by time — train on older data, validate on newer."""
    df["date"] = pd.to_datetime(df["date"])
    val_start_dt = pd.Timestamp(val_start)

    train = df[df["date"] < val_start_dt].copy()
    val = df[df["date"] >= val_start_dt].copy()

    if max_train_samples is not None and len(train) > max_train_samples:
        train = train.sort_values("date").tail(max_train_samples)
        logger.info(
            f"Limiting training samples to most recent {max_train_samples} "
            f"before {val_start}"
        )

    logger.info(f"Time-based split at {val_start}:")
    logger.info(f"  Train: {len(train)} samples ({train['date'].min().date()} to {train['date'].max().date()})")
    logger.info(f"  Val:   {len(val)} samples ({val['date'].min().date()} to {val['date'].max().date()})")

    X_train = train[FEATURE_COLUMNS].values
    y_train = train["label"].values.astype(int)
    X_val = val[FEATURE_COLUMNS].values
    y_val = val["label"].values.astype(int)

    return X_train, y_train, X_val, y_val


def train_tabpfn(X_train, y_train, X_val, y_val):
    """Train using TabPFN foundation model."""
    try:
        from tabpfn import TabPFNClassifier
    except ImportError:
        logger.error("TabPFN not installed. Install with: pip install tabpfn")
        logger.error("Falling back to LightGBM...")
        return train_lightgbm(X_train, y_train, X_val, y_val)

    logger.info("Training TabPFN classifier...")

    # TabPFN handles NaN values natively
    # For large datasets, subsample training data (TabPFN works best with <10K samples)
    max_train = 10_000
    if len(X_train) > max_train:
        logger.info(f"Subsampling training data: {len(X_train)} → {max_train}")
        idx = np.random.RandomState(42).choice(len(X_train), max_train, replace=False)
        X_train_sub = X_train[idx]
        y_train_sub = y_train[idx]
    else:
        X_train_sub = X_train
        y_train_sub = y_train

    try:
        clf = TabPFNClassifier()
        clf.fit(X_train_sub, y_train_sub)
        return clf, "tabpfn"
    except Exception as e:
        logger.error(f"TabPFN training failed: {e}")
        logger.error("Falling back to LightGBM...")
        return train_lightgbm(X_train, y_train, X_val, y_val)


def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train using LightGBM (fallback when TabPFN unavailable)."""
    try:
        import lightgbm as lgb
    except ImportError:
        logger.error("LightGBM not installed. Install with: pip install lightgbm")
        sys.exit(1)

    logger.info("Training LightGBM classifier...")

    # Remap labels: {-1, 0, 1} → {0, 1, 2} for LightGBM
    y_train_mapped = y_train + 1  # -1→0, 0→1, 1→2
    y_val_mapped = y_val + 1

    # Compute class weights to handle imbalanced labels
    from collections import Counter

    class_counts = Counter(y_train_mapped)
    total = len(y_train_mapped)
    n_classes = len(class_counts)
    class_weight = {c: total / (n_classes * count) for c, count in class_counts.items()}
    sample_weights = np.array([class_weight[y] for y in y_train_mapped])

    train_data = lgb.Dataset(X_train, label=y_train_mapped, weight=sample_weights, feature_name=FEATURE_COLUMNS)
    val_data = lgb.Dataset(X_val, label=y_val_mapped, feature_name=FEATURE_COLUMNS, reference=train_data)

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        # Lower LR + more rounds = smoother learning on noisy data
        "learning_rate": 0.01,
        # More capacity to find feature interactions
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 100,
        # Aggressive subsampling to reduce overfitting on noise
        "subsample": 0.7,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        # Stronger regularization for financial data
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "min_gain_to_split": 0.01,
        "path_smooth": 1.0,
        "verbose": -1,
        "seed": 42,
    }

    callbacks = [
        lgb.log_evaluation(period=100),
        lgb.early_stopping(stopping_rounds=100),
    ]

    booster = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    # Wrap in sklearn-compatible interface
    clf = LGBMWrapper(booster, y_train)

    return clf, "lightgbm"


def evaluate(model, X_val, y_val, model_type: str) -> dict:
    """Evaluate model and return metrics dict."""
    if isinstance(X_val, np.ndarray):
        X_df = pd.DataFrame(X_val, columns=FEATURE_COLUMNS)
    else:
        X_df = X_val

    y_pred = model.predict(X_df)
    probas = model.predict_proba(X_df)

    accuracy = accuracy_score(y_val, y_pred)
    report = classification_report(
        y_val, y_pred,
        target_names=["LOSS (-1)", "TIMEOUT (0)", "WIN (+1)"],
        output_dict=True,
    )
    cm = confusion_matrix(y_val, y_pred)

    # Win-class specific metrics
    win_mask = y_val == 1
    if win_mask.sum() > 0:
        win_probs = probas[win_mask]
        win_col_idx = list(model.classes_).index(1)
        avg_win_prob_for_actual_wins = float(win_probs[:, win_col_idx].mean())
    else:
        avg_win_prob_for_actual_wins = 0.0

    # High-confidence win precision
    win_col_idx = list(model.classes_).index(1)
    high_conf_mask = probas[:, win_col_idx] >= 0.6
    if high_conf_mask.sum() > 0:
        high_conf_precision = float((y_val[high_conf_mask] == 1).mean())
        high_conf_count = int(high_conf_mask.sum())
    else:
        high_conf_precision = 0.0
        high_conf_count = 0

    # Calibration analysis: do higher P(WIN) quintiles actually win more?
    win_probs_all = probas[:, win_col_idx]
    quintile_labels = pd.qcut(win_probs_all, q=5, labels=False, duplicates="drop")
    calibration = {}
    for q in sorted(set(quintile_labels)):
        mask = quintile_labels == q
        q_probs = win_probs_all[mask]
        q_actual_win_rate = float((y_val[mask] == 1).mean())
        q_actual_loss_rate = float((y_val[mask] == -1).mean())
        calibration[f"Q{q+1}"] = {
            "mean_predicted_win_prob": round(float(q_probs.mean()), 4),
            "actual_win_rate": round(q_actual_win_rate, 4),
            "actual_loss_rate": round(q_actual_loss_rate, 4),
            "count": int(mask.sum()),
        }

    # Top decile (top 10% by P(WIN)) — most actionable metric
    top_decile_threshold = np.percentile(win_probs_all, 90)
    top_decile_mask = win_probs_all >= top_decile_threshold
    top_decile_win_rate = float((y_val[top_decile_mask] == 1).mean()) if top_decile_mask.sum() > 0 else 0.0
    top_decile_loss_rate = float((y_val[top_decile_mask] == -1).mean()) if top_decile_mask.sum() > 0 else 0.0

    metrics = {
        "model_type": model_type,
        "accuracy": round(accuracy, 4),
        "per_class": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in report.items() if isinstance(v, dict)},
        "confusion_matrix": cm.tolist(),
        "avg_win_prob_for_actual_wins": round(avg_win_prob_for_actual_wins, 4),
        "high_confidence_win_precision": round(high_conf_precision, 4),
        "high_confidence_win_count": high_conf_count,
        "calibration_quintiles": calibration,
        "top_decile_win_rate": round(top_decile_win_rate, 4),
        "top_decile_loss_rate": round(top_decile_loss_rate, 4),
        "top_decile_threshold": round(float(top_decile_threshold), 4),
        "top_decile_count": int(top_decile_mask.sum()),
        "val_samples": len(y_val),
    }

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Model: {model_type}")
    logger.info(f"Overall Accuracy: {accuracy:.1%}")
    logger.info(f"\nPer-class metrics:")
    logger.info(f"{'':>15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    for label, name in [(-1, "LOSS"), (0, "TIMEOUT"), (1, "WIN")]:
        key = f"{name} ({label:+d})"
        if key in report:
            r = report[key]
            logger.info(f"{name:>15} {r['precision']:>10.3f} {r['recall']:>10.3f} {r['f1-score']:>10.3f} {r['support']:>10.0f}")

    logger.info(f"\nConfusion Matrix (rows=actual, cols=predicted):")
    logger.info(f"{'':>10} {'LOSS':>8} {'TIMEOUT':>8} {'WIN':>8}")
    for i, name in enumerate(["LOSS", "TIMEOUT", "WIN"]):
        logger.info(f"{name:>10} {cm[i][0]:>8} {cm[i][1]:>8} {cm[i][2]:>8}")

    logger.info(f"\nWin-class insights:")
    logger.info(f"  Avg P(WIN) for actual winners: {avg_win_prob_for_actual_wins:.1%}")
    logger.info(f"  High-confidence (>60%) precision: {high_conf_precision:.1%} ({high_conf_count} samples)")

    logger.info("\nCalibration (does higher P(WIN) = more actual wins?):")
    logger.info(f"{'Quintile':>10} {'Avg P(WIN)':>12} {'Actual WIN%':>12} {'Actual LOSS%':>13} {'Count':>8}")
    for q_name, q_data in calibration.items():
        logger.info(
            f"{q_name:>10} {q_data['mean_predicted_win_prob']:>12.1%} "
            f"{q_data['actual_win_rate']:>12.1%} {q_data['actual_loss_rate']:>13.1%} "
            f"{q_data['count']:>8}"
        )

    logger.info("\nTop decile (top 10% by P(WIN)):")
    logger.info(f"  Threshold: P(WIN) >= {top_decile_threshold:.1%}")
    logger.info(f"  Actual win rate: {top_decile_win_rate:.1%} ({int(top_decile_mask.sum())} samples)")
    logger.info(f"  Actual loss rate: {top_decile_loss_rate:.1%}")
    baseline_win = float((y_val == 1).mean())
    logger.info(f"  Baseline win rate: {baseline_win:.1%}")
    if baseline_win > 0:
        logger.info(f"  Lift over baseline: {top_decile_win_rate / baseline_win:.2f}x")
    logger.info(f"{'='*60}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train ML model for win probability")
    parser.add_argument("--dataset", type=str, default="data/ml/training_dataset.parquet")
    parser.add_argument("--model", type=str, choices=["tabpfn", "lightgbm", "auto", "both"], default="auto",
                        help="Model type (auto tries TabPFN first, falls back to LightGBM; both trains and compares both)")
    parser.add_argument("--val-start", type=str, default="2024-07-01",
                        help="Validation split date (default: 2024-07-01)")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Limit training samples to the most recent N before val-start")
    parser.add_argument("--output-dir", type=str, default="data/ml")
    args = parser.parse_args()

    if args.max_train_samples is not None and args.max_train_samples <= 0:
        logger.error("--max-train-samples must be a positive integer")
        sys.exit(1)

    # Set up file logging
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "training.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"Logs will be written to {log_file}")

    # Load dataset
    df = load_dataset(args.dataset)

    # Split
    X_train, y_train, X_val, y_val = time_split(
        df,
        val_start=args.val_start,
        max_train_samples=args.max_train_samples,
    )

    if len(X_val) == 0:
        logger.error(f"No validation data after {args.val_start} — adjust --val-start")
        sys.exit(1)

    # Train
    if args.model == "both":
        logger.info("\n" + "="*60)
        logger.info("COMPARISON MODE: Training both TabPFN and LightGBM")
        logger.info("="*60)

        # Train TabPFN
        logger.info("\n--- Training TabPFN ---")
        model_tabpfn, _ = train_tabpfn(X_train, y_train, X_val, y_val)
        metrics_tabpfn = evaluate(model_tabpfn, X_val, y_val, "tabpfn")

        # Train LightGBM
        logger.info("\n--- Training LightGBM ---")
        model_lightgbm, _ = train_lightgbm(X_train, y_train, X_val, y_val)
        metrics_lightgbm = evaluate(model_lightgbm, X_val, y_val, "lightgbm")

        # Save both models
        predictor_tabpfn = MLPredictor(model=model_tabpfn, feature_columns=FEATURE_COLUMNS, model_type="tabpfn")
        model_path_tabpfn = predictor_tabpfn.save(args.output_dir, suffix="_tabpfn")
        logger.info(f"TabPFN model saved to {model_path_tabpfn}")

        predictor_lightgbm = MLPredictor(model=model_lightgbm, feature_columns=FEATURE_COLUMNS, model_type="lightgbm")
        model_path_lightgbm = predictor_lightgbm.save(args.output_dir, suffix="_lightgbm")
        logger.info(f"LightGBM model saved to {model_path_lightgbm}")

        # Save comparison metrics
        comparison_metrics = {
            "comparison": True,
            "tabpfn": metrics_tabpfn,
            "lightgbm": metrics_lightgbm,
        }
        metrics_path = os.path.join(args.output_dir, "metrics_comparison.json")
        with open(metrics_path, "w") as f:
            json.dump(comparison_metrics, f, indent=2)
        logger.info(f"Comparison metrics saved to {metrics_path}")

        # Print summary
        logger.info("\n" + "="*60)
        logger.info("COMPARISON SUMMARY")
        logger.info("="*60)
        logger.info(f"TabPFN Accuracy: {metrics_tabpfn['accuracy']:.1%}")
        logger.info(f"LightGBM Accuracy: {metrics_lightgbm['accuracy']:.1%}")
        logger.info(f"\nTabPFN Top Decile Win Rate: {metrics_tabpfn['top_decile_win_rate']:.1%}")
        logger.info(f"LightGBM Top Decile Win Rate: {metrics_lightgbm['top_decile_win_rate']:.1%}")
        logger.info("="*60)
    else:
        if args.model == "tabpfn" or args.model == "auto":
            model, model_type = train_tabpfn(X_train, y_train, X_val, y_val)
        else:
            model, model_type = train_lightgbm(X_train, y_train, X_val, y_val)

        # Evaluate
        metrics = evaluate(model, X_val, y_val, model_type)

        # Save model
        predictor = MLPredictor(model=model, feature_columns=FEATURE_COLUMNS, model_type=model_type)
        model_path = predictor.save(args.output_dir)
        logger.info(f"Model saved to {model_path}")

        # Save metrics
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
