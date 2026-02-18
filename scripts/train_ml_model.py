#!/usr/bin/env python3
"""Train ML model on the generated dataset.

Supports TabPFN (recommended, requires GPU or API) and LightGBM (fallback).
Uses time-based train/validation split to prevent data leakage.

Usage:
    python scripts/train_ml_model.py
    python scripts/train_ml_model.py --model lightgbm
    python scripts/train_ml_model.py --binary --model all
    python scripts/train_ml_model.py --model tabpfn --dataset data/ml/training_dataset.parquet
    python scripts/train_ml_model.py --max-train-samples 5000
    python scripts/train_ml_model.py --binary --profit-target 0.07 --stop-loss 0.02 --hold-days 5
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
LGBM_BEST_PARAMS_PATH = DATA_DIR / "lgbm_best_params.json"


def remap_binary(y: np.ndarray) -> np.ndarray:
    """Collapse 3-class labels {-1, 0, 1} → binary {0, 1}.

    LOSS (-1) and TIMEOUT (0) both map to 0 (not-WIN).
    WIN (1) maps to 1.
    """
    return (y == 1).astype(int)


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


def train_lightgbm(X_train, y_train, X_val, y_val, binary: bool = False, params_override: dict | None = None):
    """Train using LightGBM (fallback when TabPFN unavailable).

    Args:
        binary: If True, expects y already remapped to {0,1}; trains binary classifier.
        params_override: Optional dict of hyperparameters to override defaults (from Optuna).
    """
    try:
        import lightgbm as lgb
    except ImportError:
        logger.error("LightGBM not installed. Install with: pip install lightgbm")
        sys.exit(1)

    logger.info("Training LightGBM classifier...")

    if binary:
        # Binary classification: y is already {0, 1}
        from collections import Counter
        class_counts = Counter(y_train)
        n_pos = class_counts.get(1, 1)
        n_neg = class_counts.get(0, 1)
        scale_pos = n_neg / n_pos

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLUMNS)
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_COLUMNS, reference=train_data)

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.01,
            "num_leaves": 63,
            "max_depth": 8,
            "min_child_samples": 100,
            "subsample": 0.7,
            "subsample_freq": 1,
            "colsample_bytree": 0.7,
            "reg_alpha": 1.0,
            "reg_lambda": 1.0,
            "min_gain_to_split": 0.01,
            "path_smooth": 1.0,
            "is_unbalance": False,
            "scale_pos_weight": scale_pos,
            "verbose": -1,
            "seed": 42,
        }

        if params_override:
            params.update(params_override)

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

        clf = LGBMBinaryWrapper(booster)

    else:
        # 3-class: remap labels {-1, 0, 1} → {0, 1, 2}
        y_train_mapped = y_train + 1
        y_val_mapped = y_val + 1

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
            "learning_rate": 0.01,
            "num_leaves": 63,
            "max_depth": 8,
            "min_child_samples": 100,
            "subsample": 0.7,
            "subsample_freq": 1,
            "colsample_bytree": 0.7,
            "reg_alpha": 1.0,
            "reg_lambda": 1.0,
            "min_gain_to_split": 0.01,
            "path_smooth": 1.0,
            "verbose": -1,
            "seed": 42,
        }

        if params_override:
            params.update(params_override)

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

        clf = LGBMWrapper(booster, y_train)

    return clf, "lightgbm"


def train_catboost(X_train, y_train, X_val, y_val, binary: bool = False):
    """Train using CatBoost classifier.

    Args:
        binary: If True, trains binary classifier (WIN vs not-WIN).
    """
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        logger.error("CatBoost not installed. Install with: pip install catboost")
        sys.exit(1)

    logger.info("Training CatBoost classifier...")

    # Replace NaN with 0 — CatBoost handles missing values, but explicit is safer
    X_train_clean = np.nan_to_num(X_train, nan=0.0)
    X_val_clean = np.nan_to_num(X_val, nan=0.0)

    if binary:
        clf = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.03,
            depth=6,
            auto_class_weights="Balanced",
            eval_metric="AUC",
            early_stopping_rounds=100,
            random_seed=42,
            verbose=100,
            loss_function="Logloss",
        )
        clf.fit(
            X_train_clean, y_train,
            eval_set=(X_val_clean, y_val),
        )
    else:
        # 3-class: remap {-1,0,1} → {0,1,2} then restore
        y_train_mapped = y_train + 1
        y_val_mapped = y_val + 1
        clf_raw = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.03,
            depth=6,
            auto_class_weights="Balanced",
            eval_metric="Accuracy",
            early_stopping_rounds=100,
            random_seed=42,
            verbose=100,
            loss_function="MultiClass",
        )
        clf_raw.fit(
            X_train_clean, y_train_mapped,
            eval_set=(X_val_clean, y_val_mapped),
        )
        clf = CatBoostMulticlassWrapper(clf_raw)

    return clf, "catboost"


def train_xgboost(X_train, y_train, X_val, y_val, binary: bool = False):
    """Train using XGBoost classifier.

    Args:
        binary: If True, uses binary:logistic objective; else multi:softprob.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        logger.error("XGBoost not installed. Install with: pip install xgboost")
        sys.exit(1)

    logger.info("Training XGBoost classifier...")

    from collections import Counter

    if binary:
        class_counts = Counter(y_train)
        n_pos = class_counts.get(1, 1)
        n_neg = class_counts.get(0, 1)
        scale_pos = n_neg / n_pos

        clf = XGBClassifier(
            objective="binary:logistic",
            n_estimators=2000,
            learning_rate=0.01,
            max_depth=6,
            min_child_weight=10,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            early_stopping_rounds=100,
            random_state=42,
            verbosity=0,
        )
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100,
        )
    else:
        # 3-class: remap {-1,0,1} → {0,1,2}
        y_train_mapped = y_train + 1
        y_val_mapped = y_val + 1

        class_counts = Counter(y_train_mapped)
        total = len(y_train_mapped)
        n_classes = len(class_counts)
        class_weight = {c: total / (n_classes * count) for c, count in class_counts.items()}
        sample_weights = np.array([class_weight[y] for y in y_train_mapped])

        clf_raw = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=2000,
            learning_rate=0.01,
            max_depth=6,
            min_child_weight=10,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=1.0,
            eval_metric="mlogloss",
            early_stopping_rounds=100,
            random_state=42,
            verbosity=0,
        )
        clf_raw.fit(
            X_train, y_train_mapped,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val_mapped)],
            verbose=100,
        )
        clf = XGBoostMulticlassWrapper(clf_raw)

    return clf, "xgboost"


class LGBMBinaryWrapper:
    """Sklearn-compatible wrapper for binary LightGBM booster."""

    def __init__(self, booster):
        self.booster = booster
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        win_prob = self.booster.predict(X)  # shape (n,) for binary
        return np.column_stack([1 - win_prob, win_prob])

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]


class CatBoostMulticlassWrapper:
    """Sklearn-compatible wrapper for 3-class CatBoost (remaps {0,1,2} → {-1,0,1})."""

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
        mapped = np.argmax(probas, axis=1)
        return self.classes_[mapped]


class XGBoostMulticlassWrapper:
    """Sklearn-compatible wrapper for 3-class XGBoost (remaps {0,1,2} → {-1,0,1})."""

    def __init__(self, model):
        self.model = model
        self.classes_ = np.array([-1, 0, 1])

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict_proba(X)

    def predict(self, X):
        probas = self.predict_proba(X)
        mapped = np.argmax(probas, axis=1)
        return self.classes_[mapped]


def evaluate(model, X_val, y_val, model_type: str, binary: bool = False) -> dict:
    """Evaluate model and return metrics dict.

    Args:
        binary: If True, y_val is {0,1} and we suppress TIMEOUT row.
    """
    if isinstance(X_val, np.ndarray):
        X_df = pd.DataFrame(X_val, columns=FEATURE_COLUMNS)
    else:
        X_df = X_val

    y_pred = model.predict(X_df)
    probas = model.predict_proba(X_df)

    accuracy = accuracy_score(y_val, y_pred)

    if binary:
        target_names = ["NOT_WIN (0)", "WIN (1)"]
        report = classification_report(
            y_val, y_pred,
            target_names=target_names,
            output_dict=True,
        )
        cm = confusion_matrix(y_val, y_pred)

        # Win column is index 1 (class label 1)
        win_col_idx = 1

        # Top-decile precision: of the top-10% by P(WIN), what fraction are true wins?
        win_probs_all = probas[:, win_col_idx]
        top_decile_threshold = np.percentile(win_probs_all, 90)
        top_decile_mask = win_probs_all >= top_decile_threshold
        top_decile_win_rate = float((y_val[top_decile_mask] == 1).mean()) if top_decile_mask.sum() > 0 else 0.0
        top_decile_loss_rate = float((y_val[top_decile_mask] == 0).mean()) if top_decile_mask.sum() > 0 else 0.0
        top_decile_precision = top_decile_win_rate  # precision = win rate in top decile

        # High-confidence precision
        high_conf_mask = win_probs_all >= 0.6
        high_conf_precision = float((y_val[high_conf_mask] == 1).mean()) if high_conf_mask.sum() > 0 else 0.0
        high_conf_count = int(high_conf_mask.sum())

        # Avg P(WIN) for actual winners
        win_mask = y_val == 1
        avg_win_prob_for_actual_wins = float(win_probs_all[win_mask].mean()) if win_mask.sum() > 0 else 0.0

        # Calibration
        quintile_labels = pd.qcut(win_probs_all, q=5, labels=False, duplicates="drop")
        calibration = {}
        for q in sorted(set(quintile_labels)):
            mask = quintile_labels == q
            q_probs = win_probs_all[mask]
            calibration[f"Q{q+1}"] = {
                "mean_predicted_win_prob": round(float(q_probs.mean()), 4),
                "actual_win_rate": round(float((y_val[mask] == 1).mean()), 4),
                "count": int(mask.sum()),
            }

        metrics = {
            "model_type": model_type,
            "binary": True,
            "accuracy": round(accuracy, 4),
            "per_class": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in report.items() if isinstance(v, dict)},
            "confusion_matrix": cm.tolist(),
            "avg_win_prob_for_actual_wins": round(avg_win_prob_for_actual_wins, 4),
            "high_confidence_win_precision": round(high_conf_precision, 4),
            "high_confidence_win_count": high_conf_count,
            "calibration_quintiles": calibration,
            "top_decile_win_rate": round(top_decile_win_rate, 4),
            "top_decile_precision": round(top_decile_precision, 4),
            "top_decile_loss_rate": round(top_decile_loss_rate, 4),
            "top_decile_threshold": round(float(top_decile_threshold), 4),
            "top_decile_count": int(top_decile_mask.sum()),
            "val_samples": len(y_val),
        }

        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info(f"Model: {model_type} [BINARY]")
        logger.info(f"Overall Accuracy: {accuracy:.1%}")
        logger.info(f"\nPer-class metrics:")
        logger.info(f"{'':>15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        for name in ["NOT_WIN (0)", "WIN (1)"]:
            if name in report:
                r = report[name]
                logger.info(f"{name:>15} {r['precision']:>10.3f} {r['recall']:>10.3f} {r['f1-score']:>10.3f} {r['support']:>10.0f}")
        logger.info(f"\nWin-class insights:")
        logger.info(f"  Avg P(WIN) for actual winners: {avg_win_prob_for_actual_wins:.1%}")
        logger.info(f"  High-confidence (>60%) precision: {high_conf_precision:.1%} ({high_conf_count} samples)")
        logger.info(f"\nTop decile (top 10% by P(WIN)):")
        logger.info(f"  Threshold: P(WIN) >= {top_decile_threshold:.1%}")
        logger.info(f"  Top-decile precision: {top_decile_precision:.1%}")
        logger.info(f"  Actual win rate in top decile: {top_decile_win_rate:.1%} ({int(top_decile_mask.sum())} samples)")
        baseline_win = float((y_val == 1).mean())
        logger.info(f"  Baseline win rate: {baseline_win:.1%}")
        if baseline_win > 0:
            logger.info(f"  Lift over baseline: {top_decile_win_rate / baseline_win:.2f}x")
        logger.info(f"{'='*60}")

    else:
        # 3-class evaluation
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

        win_col_idx = list(model.classes_).index(1)
        win_probs_all = probas[:, win_col_idx]

        # High-confidence win precision
        high_conf_mask = win_probs_all >= 0.6
        high_conf_precision = float((y_val[high_conf_mask] == 1).mean()) if high_conf_mask.sum() > 0 else 0.0
        high_conf_count = int(high_conf_mask.sum())

        # Calibration
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

        # Top decile
        top_decile_threshold = np.percentile(win_probs_all, 90)
        top_decile_mask = win_probs_all >= top_decile_threshold
        top_decile_win_rate = float((y_val[top_decile_mask] == 1).mean()) if top_decile_mask.sum() > 0 else 0.0
        top_decile_loss_rate = float((y_val[top_decile_mask] == -1).mean()) if top_decile_mask.sum() > 0 else 0.0
        # top_decile_precision: same as win rate in top decile for 3-class
        top_decile_precision = top_decile_win_rate

        metrics = {
            "model_type": model_type,
            "binary": False,
            "accuracy": round(accuracy, 4),
            "per_class": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in report.items() if isinstance(v, dict)},
            "confusion_matrix": cm.tolist(),
            "avg_win_prob_for_actual_wins": round(avg_win_prob_for_actual_wins, 4),
            "high_confidence_win_precision": round(high_conf_precision, 4),
            "high_confidence_win_count": high_conf_count,
            "calibration_quintiles": calibration,
            "top_decile_win_rate": round(top_decile_win_rate, 4),
            "top_decile_precision": round(top_decile_precision, 4),
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
        logger.info(f"  Top-decile precision: {top_decile_precision:.1%}")
        logger.info(f"  Actual win rate: {top_decile_win_rate:.1%} ({int(top_decile_mask.sum())} samples)")
        logger.info(f"  Actual loss rate: {top_decile_loss_rate:.1%}")
        baseline_win = float((y_val == 1).mean())
        logger.info(f"  Baseline win rate: {baseline_win:.1%}")
        if baseline_win > 0:
            logger.info(f"  Lift over baseline: {top_decile_win_rate / baseline_win:.2f}x")
        logger.info(f"{'='*60}")

    return metrics


def _load_lgbm_best_params() -> dict | None:
    """Load best LightGBM params from Optuna output if available."""
    if LGBM_BEST_PARAMS_PATH.exists():
        with open(LGBM_BEST_PARAMS_PATH) as f:
            params = json.load(f)
        logger.info(f"Loaded LightGBM best params from {LGBM_BEST_PARAMS_PATH}")
        return params
    return None


def main():
    parser = argparse.ArgumentParser(description="Train ML model for win probability")
    parser.add_argument("--dataset", type=str, default="data/ml/training_dataset.parquet")
    parser.add_argument(
        "--model",
        type=str,
        choices=["tabpfn", "lightgbm", "catboost", "xgboost", "auto", "both", "all"],
        default="auto",
        help=(
            "Model type. 'auto' tries TabPFN first; 'both' trains TabPFN+LightGBM; "
            "'all' trains all 4 models (TabPFN, LightGBM, CatBoost, XGBoost)"
        ),
    )
    parser.add_argument("--binary", action="store_true",
                        help="Binary mode: collapse LOSS+TIMEOUT → 0, WIN → 1")
    parser.add_argument("--profit-target", type=float, default=0.05,
                        help="Triple-barrier profit target (default: 0.05 = 5%%)")
    parser.add_argument("--stop-loss", type=float, default=0.03,
                        help="Triple-barrier stop loss (default: 0.03 = 3%%)")
    parser.add_argument("--hold-days", type=int, default=7,
                        help="Triple-barrier max holding days (default: 7)")
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

    if args.binary:
        logger.info(f"Binary mode: LOSS+TIMEOUT→0, WIN→1")
    logger.info(f"Label params: profit_target={args.profit_target}, stop_loss={args.stop_loss}, hold_days={args.hold_days}")

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

    # Apply binary remapping if requested
    if args.binary:
        y_train = remap_binary(y_train)
        y_val = remap_binary(y_val)
        win_pct = y_train.mean() * 100
        logger.info(f"Binary labels applied. WIN% in train: {win_pct:.1f}%")

    # Load Optuna best params for LightGBM if available
    lgbm_params_override = _load_lgbm_best_params() if args.model in ("lightgbm", "auto", "both", "all") else None

    # Train
    if args.model == "all":
        logger.info("\n" + "="*60)
        logger.info("ALL-MODELS MODE: Training TabPFN, LightGBM, CatBoost, XGBoost")
        logger.info("="*60)

        all_metrics = {}
        all_predictors = {}

        for model_name, train_fn in [
            ("tabpfn", lambda: train_tabpfn(X_train, y_train, X_val, y_val)),
            ("lightgbm", lambda: train_lightgbm(X_train, y_train, X_val, y_val, binary=args.binary, params_override=lgbm_params_override)),
            ("catboost", lambda: train_catboost(X_train, y_train, X_val, y_val, binary=args.binary)),
            ("xgboost", lambda: train_xgboost(X_train, y_train, X_val, y_val, binary=args.binary)),
        ]:
            logger.info(f"\n--- Training {model_name.upper()} ---")
            try:
                m, mt = train_fn()
                metrics = evaluate(m, X_val, y_val, mt, binary=args.binary)
                all_metrics[model_name] = metrics

                predictor = MLPredictor(model=m, feature_columns=FEATURE_COLUMNS, model_type=mt)
                model_path = predictor.save(args.output_dir, suffix=f"_{model_name}")
                logger.info(f"{model_name} model saved to {model_path}")
                all_predictors[model_name] = predictor
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                all_metrics[model_name] = {"error": str(e)}

        # Save all metrics
        metrics_path = os.path.join(args.output_dir, "metrics_comparison.json")
        with open(metrics_path, "w") as f:
            json.dump({"comparison": True, **all_metrics}, f, indent=2)
        logger.info(f"Comparison metrics saved to {metrics_path}")

        # Print comparison table
        logger.info("\n" + "="*60)
        logger.info("COMPARISON SUMMARY")
        logger.info("="*60)
        logger.info(f"{'Model':>12} {'Accuracy':>10} {'Top-Decile Prec':>17} {'High-Conf Count':>16}")
        for name, m in all_metrics.items():
            if "error" not in m:
                logger.info(
                    f"{name:>12} {m['accuracy']:>10.1%} "
                    f"{m.get('top_decile_precision', m.get('top_decile_win_rate', 0)):>17.1%} "
                    f"{m.get('high_confidence_win_count', 0):>16}"
                )
        logger.info("="*60)

    elif args.model == "both":
        logger.info("\n" + "="*60)
        logger.info("COMPARISON MODE: Training both TabPFN and LightGBM")
        logger.info("="*60)

        logger.info("\n--- Training TabPFN ---")
        model_tabpfn, _ = train_tabpfn(X_train, y_train, X_val, y_val)
        metrics_tabpfn = evaluate(model_tabpfn, X_val, y_val, "tabpfn", binary=args.binary)

        logger.info("\n--- Training LightGBM ---")
        model_lightgbm, _ = train_lightgbm(X_train, y_train, X_val, y_val, binary=args.binary, params_override=lgbm_params_override)
        metrics_lightgbm = evaluate(model_lightgbm, X_val, y_val, "lightgbm", binary=args.binary)

        predictor_tabpfn = MLPredictor(model=model_tabpfn, feature_columns=FEATURE_COLUMNS, model_type="tabpfn")
        model_path_tabpfn = predictor_tabpfn.save(args.output_dir, suffix="_tabpfn")
        logger.info(f"TabPFN model saved to {model_path_tabpfn}")

        predictor_lightgbm = MLPredictor(model=model_lightgbm, feature_columns=FEATURE_COLUMNS, model_type="lightgbm")
        model_path_lightgbm = predictor_lightgbm.save(args.output_dir, suffix="_lightgbm")
        logger.info(f"LightGBM model saved to {model_path_lightgbm}")

        comparison_metrics = {
            "comparison": True,
            "tabpfn": metrics_tabpfn,
            "lightgbm": metrics_lightgbm,
        }
        metrics_path = os.path.join(args.output_dir, "metrics_comparison.json")
        with open(metrics_path, "w") as f:
            json.dump(comparison_metrics, f, indent=2)
        logger.info(f"Comparison metrics saved to {metrics_path}")

        logger.info("\n" + "="*60)
        logger.info("COMPARISON SUMMARY")
        logger.info("="*60)
        logger.info(f"{'Model':>12} {'Accuracy':>10} {'Top-Decile Prec':>17}")
        for name, m in [("tabpfn", metrics_tabpfn), ("lightgbm", metrics_lightgbm)]:
            logger.info(f"{name:>12} {m['accuracy']:>10.1%} {m.get('top_decile_precision', 0):>17.1%}")
        logger.info("="*60)

    else:
        if args.model == "tabpfn" or args.model == "auto":
            model, model_type = train_tabpfn(X_train, y_train, X_val, y_val)
        elif args.model == "lightgbm":
            model, model_type = train_lightgbm(X_train, y_train, X_val, y_val, binary=args.binary, params_override=lgbm_params_override)
        elif args.model == "catboost":
            model, model_type = train_catboost(X_train, y_train, X_val, y_val, binary=args.binary)
        elif args.model == "xgboost":
            model, model_type = train_xgboost(X_train, y_train, X_val, y_val, binary=args.binary)

        metrics = evaluate(model, X_val, y_val, model_type, binary=args.binary)

        predictor = MLPredictor(model=model, feature_columns=FEATURE_COLUMNS, model_type=model_type)
        model_path = predictor.save(args.output_dir)
        logger.info(f"Model saved to {model_path}")

        metrics_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
