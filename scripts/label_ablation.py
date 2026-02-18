#!/usr/bin/env python3
"""Label parameter ablation study over triple-barrier labeling configurations.

Runs 27 configs (profit_target × stop_loss × hold_days) and records how each
affects top-decile precision when training a binary LightGBM model.
Uses existing parquet data — no re-fetching required.

Output: data/ml/label_ablation_results.csv (sorted by top_decile_precision desc)

Usage:
    python scripts/label_ablation.py
    python scripts/label_ablation.py --dataset data/ml/training_dataset.parquet
    python scripts/label_ablation.py --val-start 2024-07-01
"""

from __future__ import annotations

import argparse
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tradingagents.ml.feature_engineering import FEATURE_COLUMNS, apply_triple_barrier_labels
from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)

OUTPUT_PATH = Path("data/ml/label_ablation_results.csv")

# Grid search space
PROFIT_TARGETS = [0.03, 0.05, 0.07]
STOP_LOSSES = [0.02, 0.03, 0.05]
HOLD_DAYS = [5, 7, 10]

# Fixed baseline LightGBM params (no Optuna — isolate label effect)
BASELINE_LGBM_PARAMS = {
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
    "verbose": -1,
    "seed": 42,
}


def top_decile_precision(y_true: np.ndarray, win_probs: np.ndarray) -> float:
    threshold = np.percentile(win_probs, 90)
    mask = win_probs >= threshold
    return float((y_true[mask] == 1).mean()) if mask.sum() > 0 else 0.0


def top_decile_win_rate(y_true: np.ndarray, win_probs: np.ndarray) -> float:
    """Alias for top_decile_precision in binary context."""
    return top_decile_precision(y_true, win_probs)


def compute_lift(y_true: np.ndarray, win_probs: np.ndarray) -> float:
    baseline = float((y_true == 1).mean())
    if baseline == 0:
        return 0.0
    return top_decile_precision(y_true, win_probs) / baseline


def run_single_config(
    df_base: pd.DataFrame,
    profit_target: float,
    stop_loss: float,
    hold_days: int,
    val_start: str,
) -> dict:
    """Re-label data with given triple-barrier params and train binary LightGBM.

    Returns a dict of metrics for this config.
    """
    import lightgbm as lgb

    # Re-apply triple-barrier labels per ticker
    # df_base must have 'ticker', 'date', 'close', and all FEATURE_COLUMNS
    relabeled_rows = []

    for ticker, group in df_base.groupby("ticker"):
        group = group.sort_values("date").copy()
        close = group.set_index("date")["close"]

        new_labels = apply_triple_barrier_labels(
            close,
            profit_target=profit_target,
            stop_loss=stop_loss,
            max_holding_days=hold_days,
        )

        group = group.set_index("date")
        group["label"] = new_labels
        group = group.dropna(subset=["label"])
        group["label"] = group["label"].astype(int)
        relabeled_rows.append(group.reset_index())

    if not relabeled_rows:
        logger.warning(f"No rows after re-labeling for pt={profit_target} sl={stop_loss} hd={hold_days}")
        return {}

    df_new = pd.concat(relabeled_rows, ignore_index=True)
    df_new["date"] = pd.to_datetime(df_new["date"])

    # Time split
    val_dt = pd.Timestamp(val_start)
    train = df_new[df_new["date"] < val_dt]
    val = df_new[df_new["date"] >= val_dt]

    if len(train) == 0 or len(val) == 0:
        logger.warning(f"Empty split for pt={profit_target} sl={stop_loss} hd={hold_days}")
        return {}

    X_train = train[FEATURE_COLUMNS].values
    y_train_raw = train["label"].values.astype(int)
    X_val = val[FEATURE_COLUMNS].values
    y_val_raw = val["label"].values.astype(int)

    # Binary: WIN → 1, LOSS+TIMEOUT → 0
    y_train = (y_train_raw == 1).astype(int)
    y_val = (y_val_raw == 1).astype(int)

    # Class balance
    from collections import Counter
    train_counts = Counter(y_train)
    val_counts = Counter(y_val)
    n_pos = train_counts.get(1, 1)
    n_neg = train_counts.get(0, 1)

    params = dict(BASELINE_LGBM_PARAMS)
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

    win_probs = booster.predict(X_val)
    tdp = top_decile_precision(y_val, win_probs)
    tdwr = tdp  # same thing in binary
    lift = compute_lift(y_val, win_probs)

    # Label distribution on val
    raw_val_counts = Counter(y_val_raw)
    total_val = len(y_val_raw)
    win_pct = raw_val_counts.get(1, 0) / total_val * 100
    loss_pct = raw_val_counts.get(-1, 0) / total_val * 100
    timeout_pct = raw_val_counts.get(0, 0) / total_val * 100

    from sklearn.metrics import accuracy_score
    y_pred = (win_probs >= 0.5).astype(int)
    acc = accuracy_score(y_val, y_pred)

    return {
        "profit_target": profit_target,
        "stop_loss": stop_loss,
        "hold_days": hold_days,
        "top_decile_precision": round(tdp, 4),
        "top_decile_win_rate": round(tdwr, 4),
        "lift": round(lift, 4),
        "accuracy": round(acc, 4),
        "val_win_pct": round(win_pct, 2),
        "val_loss_pct": round(loss_pct, 2),
        "val_timeout_pct": round(timeout_pct, 2),
        "val_samples": len(y_val),
        "train_samples": len(y_train),
    }


def main():
    parser = argparse.ArgumentParser(description="Label parameter ablation study")
    parser.add_argument("--dataset", type=str, default="data/ml/training_dataset.parquet")
    parser.add_argument("--val-start", type=str, default="2024-07-01")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    try:
        import lightgbm  # noqa: F401
    except ImportError:
        logger.error("LightGBM required. Install with: pip install lightgbm")
        sys.exit(1)

    logger.info(f"Loading dataset from {args.dataset}")
    df = pd.read_parquet(args.dataset)

    # Validate required columns
    required = FEATURE_COLUMNS + ["label", "date", "ticker", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        # 'close' might be missing if dataset doesn't store raw prices
        if "close" in missing:
            logger.error(
                "Dataset missing 'close' column — needed to re-apply triple-barrier labels. "
                "Re-run build_ml_dataset.py with close price preservation."
            )
            sys.exit(1)
        raise ValueError(f"Missing columns: {missing}")

    configs = list(product(PROFIT_TARGETS, STOP_LOSSES, HOLD_DAYS))
    n_total = len(configs)
    logger.info(f"Running {n_total} label configs ({len(PROFIT_TARGETS)} profit_targets × {len(STOP_LOSSES)} stop_losses × {len(HOLD_DAYS)} hold_days)")

    results = []
    for i, (pt, sl, hd) in enumerate(configs, 1):
        t0 = time.time()
        logger.info(f"\n[{i}/{n_total}] profit_target={pt} stop_loss={sl} hold_days={hd}")
        try:
            row = run_single_config(df, pt, sl, hd, val_start=args.val_start)
            if row:
                results.append(row)
                elapsed = time.time() - t0
                logger.info(
                    f"  → top_decile_precision={row['top_decile_precision']:.4f} "
                    f"lift={row['lift']:.3f}x  ({elapsed:.1f}s)"
                )
        except Exception as e:
            logger.error(f"  Config failed: {e}")

    if not results:
        logger.error("No results collected — check dataset and column names")
        sys.exit(1)

    results_df = pd.DataFrame(results).sort_values("top_decile_precision", ascending=False)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to {output_path}")

    # Print top-5 configs
    logger.info("\n" + "="*70)
    logger.info("TOP 5 CONFIGS BY TOP-DECILE PRECISION")
    logger.info("="*70)
    logger.info(f"{'pt':>6} {'sl':>6} {'hd':>4} {'TDP':>8} {'Lift':>6} {'WIN%':>6} {'LOSS%':>6} {'TMOUT%':>7}")
    for _, row in results_df.head(5).iterrows():
        logger.info(
            f"{row['profit_target']:>6.2f} {row['stop_loss']:>6.2f} {row['hold_days']:>4.0f} "
            f"{row['top_decile_precision']:>8.4f} {row['lift']:>6.3f}x "
            f"{row['val_win_pct']:>6.1f} {row['val_loss_pct']:>6.1f} {row['val_timeout_pct']:>7.1f}"
        )
    logger.info("="*70)

    winner = results_df.iloc[0]
    logger.info(f"\nWinner config:")
    logger.info(f"  profit_target = {winner['profit_target']}")
    logger.info(f"  stop_loss     = {winner['stop_loss']}")
    logger.info(f"  hold_days     = {winner['hold_days']:.0f}")
    logger.info(f"  top_decile_precision = {winner['top_decile_precision']:.4f}")
    logger.info(f"  lift                 = {winner['lift']:.3f}x")


if __name__ == "__main__":
    main()
