#!/usr/bin/env python3
"""Ensemble multiple trained ML models via soft voting or stacking.

Loads trained models, computes soft-vote ensemble (average win_prob),
optionally fits a LogisticRegression meta-model (stacking), and saves
an ensemble_config.json for use by EnsemblePredictor.

Usage:
    # Soft voting
    python scripts/ensemble_models.py \
        --models data/ml/tabpfn_model_tabpfn.pkl data/ml/tabpfn_model_lightgbm.pkl \
        --val-data data/ml/training_dataset.parquet

    # With stacking
    python scripts/ensemble_models.py \
        --models data/ml/tabpfn_model_tabpfn.pkl data/ml/tabpfn_model_lightgbm.pkl \
        --val-data data/ml/training_dataset.parquet \
        --stack
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tradingagents.ml.feature_engineering import FEATURE_COLUMNS
from tradingagents.ml.predictor import MLPredictor
from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)


# These wrapper classes must be defined here so pickle can find them when
# deserializing models that were saved from train_ml_model.py (__main__).
class CatBoostMulticlassWrapper:
    def __init__(self, model):
        self.model = model
        self.classes_ = np.array([-1, 0, 1])

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.nan_to_num(X, nan=0.0)
        return self.model.predict_proba(X)

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]


class XGBoostMulticlassWrapper:
    def __init__(self, model):
        self.model = model
        self.classes_ = np.array([-1, 0, 1])

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict_proba(X)

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

ENSEMBLE_CONFIG_PATH = Path("data/ml/ensemble_config.json")
STACKING_MODEL_PATH = Path("data/ml/stacking_model.pkl")


def load_model(path: str) -> MLPredictor:
    """Load a single MLPredictor from disk."""
    with open(path, "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]
    feature_columns = saved.get("feature_columns", FEATURE_COLUMNS)
    model_type = saved.get("model_type", "unknown")
    logger.info(f"Loaded {model_type} model from {path}")
    return MLPredictor(model=model, feature_columns=feature_columns, model_type=model_type)


def get_win_probs(predictor: MLPredictor, X: np.ndarray) -> np.ndarray:
    """Extract P(WIN) vector from a predictor for a feature matrix."""
    X_df = pd.DataFrame(X, columns=predictor.feature_columns)
    probas = predictor.model.predict_proba(X_df)
    classes = list(predictor.model.classes_)

    # Find WIN column: class 1 in {-1,0,1} space or class 1 in {0,1} binary space
    if 1 in classes:
        win_idx = classes.index(1)
        return probas[:, win_idx]
    elif len(classes) == 2:
        # Binary model with classes [0, 1]
        return probas[:, 1]
    else:
        # Fallback: last column
        return probas[:, -1]


def top_decile_precision(y_true: np.ndarray, win_probs: np.ndarray) -> float:
    threshold = np.percentile(win_probs, 90)
    mask = win_probs >= threshold
    if mask.sum() == 0:
        return 0.0
    # Support both binary (1) and 3-class (1) as WIN label
    return float((y_true[mask] == 1).mean())


def evaluate_ensemble(
    y_val: np.ndarray,
    win_probs: np.ndarray,
    label: str = "ensemble",
) -> dict:
    """Compute and print ensemble evaluation metrics."""
    tdp = top_decile_precision(y_val, win_probs)
    threshold = np.percentile(win_probs, 90)
    top_mask = win_probs >= threshold
    lift = tdp / float((y_val == 1).mean()) if (y_val == 1).mean() > 0 else 0.0

    high_conf_mask = win_probs >= 0.6
    high_conf_count = int(high_conf_mask.sum())
    high_conf_prec = float((y_val[high_conf_mask] == 1).mean()) if high_conf_count > 0 else 0.0

    logger.info(f"\n{'='*50}")
    logger.info(f"Ensemble: {label}")
    logger.info(f"  Top-decile threshold: P(WIN) >= {threshold:.3f}")
    logger.info(f"  Top-decile precision: {tdp:.4f} ({int(top_mask.sum())} samples)")
    logger.info(f"  Lift over baseline:   {lift:.3f}x")
    logger.info(f"  High-conf (>60%):     {high_conf_prec:.4f} precision ({high_conf_count} samples)")
    logger.info(f"{'='*50}")

    return {
        "label": label,
        "top_decile_precision": round(tdp, 4),
        "top_decile_count": int(top_mask.sum()),
        "lift": round(lift, 4),
        "high_confidence_precision": round(high_conf_prec, 4),
        "high_confidence_count": high_conf_count,
    }


def time_split(df: pd.DataFrame, val_start: str = "2024-07-01") -> tuple:
    df["date"] = pd.to_datetime(df["date"])
    val_dt = pd.Timestamp(val_start)
    train = df[df["date"] < val_dt]
    val = df[df["date"] >= val_dt]

    X_train = train[FEATURE_COLUMNS].values
    y_train = train["label"].values.astype(int)
    X_val = val[FEATURE_COLUMNS].values
    y_val = val["label"].values.astype(int)

    logger.info(f"Train: {len(train)} | Val: {len(val)}")
    return X_train, y_train, X_val, y_val


def main():
    parser = argparse.ArgumentParser(description="Ensemble multiple ML models")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Paths to trained model .pkl files")
    parser.add_argument("--val-data", type=str, default="data/ml/training_dataset.parquet",
                        help="Path to dataset parquet for evaluation")
    parser.add_argument("--val-start", type=str, default="2024-07-01")
    parser.add_argument("--stack", action="store_true",
                        help="Also fit a LogisticRegression stacking meta-model")
    parser.add_argument("--output-config", type=str, default=str(ENSEMBLE_CONFIG_PATH))
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Optional per-model weights for soft voting (must match --models count)")
    args = parser.parse_args()

    if args.weights is not None and len(args.weights) != len(args.models):
        logger.error("--weights must have the same number of values as --models")
        sys.exit(1)

    # Load models
    predictors: List[MLPredictor] = []
    for path in args.models:
        try:
            p = load_model(path)
            predictors.append(p)
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            sys.exit(1)

    # Load dataset and split
    logger.info(f"Loading val data from {args.val_data}")
    df = pd.read_parquet(args.val_data)
    X_train, y_train, X_val, y_val = time_split(df, val_start=args.val_start)

    # --- Soft voting ---
    logger.info(f"\nComputing soft-vote ensemble ({len(predictors)} models)...")
    win_prob_matrix_val = np.stack(
        [get_win_probs(p, X_val) for p in predictors], axis=1
    )  # shape (n_val, n_models)

    weights = np.array(args.weights) if args.weights else np.ones(len(predictors))
    weights = weights / weights.sum()
    soft_vote_probs = win_prob_matrix_val @ weights  # weighted average

    # Evaluate individual models for comparison
    logger.info("\nIndividual model performance:")
    individual_metrics = []
    for i, (predictor, path) in enumerate(zip(predictors, args.models)):
        win_probs_i = win_prob_matrix_val[:, i]
        m = evaluate_ensemble(y_val, win_probs_i, label=f"{predictor.model_type} [{Path(path).name}]")
        individual_metrics.append(m)

    # Evaluate soft-vote ensemble
    soft_metrics = evaluate_ensemble(y_val, soft_vote_probs, label="Soft-Vote Ensemble")

    stacking_metrics = None
    stacking_model = None

    # --- Stacking (optional) ---
    if args.stack:
        logger.info("\nFitting LogisticRegression stacking meta-model...")

        # Collect train-set base model probabilities via cross_val_predict (5-fold)
        win_prob_matrix_train = np.stack(
            [get_win_probs(p, X_train) for p in predictors], axis=1
        )

        meta = LogisticRegression(C=1.0, class_weight="balanced", random_state=42, max_iter=500)

        # Cross-val predictions on train set — avoids leakage
        meta_train_probs = cross_val_predict(
            meta,
            win_prob_matrix_train,
            y_train,
            cv=5,
            method="predict_proba",
        )

        # Fit final meta-model on all train data
        meta.fit(win_prob_matrix_train, y_train)

        # Predict on val set
        stacked_probs_val = meta.predict_proba(win_prob_matrix_val)
        # WIN class index
        meta_classes = list(meta.classes_)
        win_idx = meta_classes.index(1) if 1 in meta_classes else -1
        stacked_win_probs = stacked_probs_val[:, win_idx] if win_idx >= 0 else stacked_probs_val[:, 1]

        stacking_metrics = evaluate_ensemble(y_val, stacked_win_probs, label="Stacking Ensemble")
        stacking_model = meta

        # Save stacking model
        STACKING_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(STACKING_MODEL_PATH, "wb") as f:
            pickle.dump({"model": meta, "classes": meta_classes}, f)
        logger.info(f"Stacking model saved to {STACKING_MODEL_PATH}")

    # --- Print comparison summary ---
    logger.info("\n" + "="*60)
    logger.info("ENSEMBLE COMPARISON SUMMARY")
    logger.info("="*60)
    logger.info(f"{'Source':>35} {'Top-Decile Prec':>16} {'Lift':>6}")
    for m in individual_metrics:
        logger.info(f"{m['label']:>35} {m['top_decile_precision']:>16.4f} {m['lift']:>6.3f}x")
    logger.info(f"{'Soft-Vote Ensemble':>35} {soft_metrics['top_decile_precision']:>16.4f} {soft_metrics['lift']:>6.3f}x")
    if stacking_metrics:
        logger.info(f"{'Stacking Ensemble':>35} {stacking_metrics['top_decile_precision']:>16.4f} {stacking_metrics['lift']:>6.3f}x")
    logger.info("="*60)

    # --- Save ensemble config ---
    config = {
        "model_paths": args.models,
        "model_types": [p.model_type for p in predictors],
        "weights": weights.tolist(),
        "has_stacking": args.stack and stacking_model is not None,
        "stacking_model_path": str(STACKING_MODEL_PATH) if args.stack else None,
        "val_metrics": {
            "soft_vote": soft_metrics,
            "individual": individual_metrics,
        },
    }
    if stacking_metrics:
        config["val_metrics"]["stacking"] = stacking_metrics

    output_path = Path(args.output_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"\nEnsemble config saved to {output_path}")


if __name__ == "__main__":
    main()
