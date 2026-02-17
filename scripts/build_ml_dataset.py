#!/usr/bin/env python3
"""Build ML training dataset from historical OHLCV data.

Fetches price data for a universe of liquid stocks, computes features
locally via stockstats, and applies triple-barrier labels.

Usage:
    python scripts/build_ml_dataset.py
    python scripts/build_ml_dataset.py --stocks 100 --years 2
    python scripts/build_ml_dataset.py --ticker-file data/tickers_top50.txt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tradingagents.ml.feature_engineering import (
    FEATURE_COLUMNS,
    MIN_HISTORY_ROWS,
    apply_triple_barrier_labels,
    compute_features_bulk,
)
from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)

# Default universe: S&P 500 most liquid by volume (top ~200)
# Can be overridden via --ticker-file
DEFAULT_TICKERS = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL", "CRM",
    "AMD", "INTC", "CSCO", "ADBE", "NFLX", "QCOM", "TXN", "AMAT", "MU", "LRCX",
    "KLAC", "MRVL", "SNPS", "CDNS", "PANW", "CRWD", "FTNT", "NOW", "UBER", "ABNB",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "SCHW", "BLK", "AXP", "USB",
    "PNC", "TFC", "COF", "BK", "STT", "FITB", "HBAN", "RF", "CFG", "KEY",
    # Healthcare
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "GILD", "ISRG", "VRTX", "REGN", "MDT", "SYK", "BSX", "EW", "ZTS",
    # Consumer
    "WMT", "PG", "KO", "PEP", "COST", "MCD", "NKE", "SBUX", "TGT", "LOW",
    "HD", "TJX", "ROST", "DG", "DLTR", "EL", "CL", "KMB", "GIS", "K",
    # Energy
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY", "DVN",
    "HAL", "FANG", "HES", "BKR", "KMI", "WMB", "OKE", "ET", "TRGP", "LNG",
    # Industrials
    "CAT", "DE", "UNP", "UPS", "HON", "RTX", "BA", "LMT", "GD", "NOC",
    "GE", "MMM", "EMR", "ITW", "PH", "ROK", "ETN", "SWK", "CMI", "PCAR",
    # Materials & Utilities
    "LIN", "APD", "ECL", "SHW", "DD", "NEM", "FCX", "VMC", "MLM", "NUE",
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "WEC", "ES",
    # REITs & Telecom
    "AMT", "PLD", "CCI", "EQIX", "SPG", "O", "PSA", "DLR", "WELL", "AVB",
    "T", "VZ", "TMUS", "CHTR", "CMCSA",
    # High-volatility / popular retail
    "COIN", "MARA", "RIOT", "PLTR", "SOFI", "HOOD", "RBLX", "SNAP", "PINS", "SQ",
    "SHOP", "SE", "ROKU", "DKNG", "PENN", "WYNN", "MGM", "LVS", "DASH", "TTD",
    # Biotech
    "MRNA", "BNTX", "BIIB", "SGEN", "ALNY", "BMRN", "EXAS", "DXCM", "HZNP", "INCY",
]

OUTPUT_DIR = Path("data/ml")


def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch OHLCV data for a single ticker via yfinance."""
    from tradingagents.dataflows.y_finance import download_history

    df = download_history(
        ticker,
        start=start,
        end=end,
        multi_level_index=False,
        progress=False,
        auto_adjust=True,
    )

    if df.empty:
        return df

    df = df.reset_index()
    return df


def get_market_cap(ticker: str) -> float | None:
    """Get current market cap for a ticker (snapshot — used as static feature)."""
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info
        return info.get("marketCap")
    except Exception:
        return None


def process_ticker(
    ticker: str,
    start: str,
    end: str,
    profit_target: float,
    stop_loss: float,
    max_holding_days: int,
    market_cap: float | None = None,
) -> pd.DataFrame | None:
    """Process a single ticker: fetch data, compute features, apply labels."""
    try:
        ohlcv = fetch_ohlcv(ticker, start, end)
        if ohlcv.empty or len(ohlcv) < MIN_HISTORY_ROWS + max_holding_days:
            logger.debug(f"{ticker}: insufficient data ({len(ohlcv)} rows), skipping")
            return None

        # Compute features
        features = compute_features_bulk(ohlcv, market_cap=market_cap)
        if features.empty:
            logger.debug(f"{ticker}: feature computation failed, skipping")
            return None

        # Compute triple-barrier labels
        close = ohlcv.set_index("Date")["Close"] if "Date" in ohlcv.columns else ohlcv["Close"]
        if isinstance(close.index, pd.DatetimeIndex):
            pass
        else:
            close.index = pd.to_datetime(close.index)

        labels = apply_triple_barrier_labels(
            close,
            profit_target=profit_target,
            stop_loss=stop_loss,
            max_holding_days=max_holding_days,
        )

        # Align features and labels by date
        combined = features.join(labels, how="inner")

        # Drop rows with NaN features or labels
        combined = combined.dropna(subset=["label"] + FEATURE_COLUMNS)

        if combined.empty:
            logger.debug(f"{ticker}: no valid rows after alignment, skipping")
            return None

        # Add metadata columns
        combined["ticker"] = ticker
        combined["date"] = combined.index

        logger.info(
            f"{ticker}: {len(combined)} samples "
            f"(WIN={int((combined['label'] == 1).sum())}, "
            f"LOSS={int((combined['label'] == -1).sum())}, "
            f"TIMEOUT={int((combined['label'] == 0).sum())})"
        )

        return combined

    except Exception as e:
        logger.warning(f"{ticker}: error processing — {e}")
        return None


def build_dataset(
    tickers: list[str],
    start: str = "2022-01-01",
    end: str = "2025-12-31",
    profit_target: float = 0.05,
    stop_loss: float = 0.03,
    max_holding_days: int = 7,
) -> pd.DataFrame:
    """Build the full training dataset across all tickers."""
    all_data = []
    total = len(tickers)

    logger.info(f"Building ML dataset: {total} tickers, {start} to {end}")
    logger.info(
        f"Triple-barrier: +{profit_target*100:.0f}% profit, "
        f"-{stop_loss*100:.0f}% stop, {max_holding_days}d timeout"
    )

    # Batch-fetch market caps
    logger.info("Fetching market caps...")
    market_caps = {}
    for ticker in tickers:
        market_caps[ticker] = get_market_cap(ticker)
        time.sleep(0.05)  # rate limit courtesy

    for i, ticker in enumerate(tickers):
        logger.info(f"[{i+1}/{total}] Processing {ticker}...")
        result = process_ticker(
            ticker=ticker,
            start=start,
            end=end,
            profit_target=profit_target,
            stop_loss=stop_loss,
            max_holding_days=max_holding_days,
            market_cap=market_caps.get(ticker),
        )
        if result is not None:
            all_data.append(result)

        # Brief pause between tickers to be polite to yfinance
        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {i+1}/{total} tickers processed, pausing 2s...")
            time.sleep(2)

    if not all_data:
        logger.error("No data collected — check tickers and date range")
        return pd.DataFrame()

    dataset = pd.concat(all_data, ignore_index=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset built: {len(dataset)} total samples from {len(all_data)} tickers")
    logger.info(f"Label distribution:")
    logger.info(f"  WIN  (+1): {int((dataset['label'] == 1).sum()):>7} ({(dataset['label'] == 1).mean()*100:.1f}%)")
    logger.info(f"  LOSS (-1): {int((dataset['label'] == -1).sum()):>7} ({(dataset['label'] == -1).mean()*100:.1f}%)")
    logger.info(f"  TIMEOUT:   {int((dataset['label'] == 0).sum()):>7} ({(dataset['label'] == 0).mean()*100:.1f}%)")
    logger.info(f"Features: {len(FEATURE_COLUMNS)}")
    logger.info(f"{'='*60}")

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Build ML training dataset")
    parser.add_argument("--stocks", type=int, default=None, help="Limit to N stocks from default universe")
    parser.add_argument("--ticker-file", type=str, default=None, help="File with tickers (one per line)")
    parser.add_argument("--start", type=str, default="2022-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--profit-target", type=float, default=0.05, help="Profit target fraction (default: 0.05)")
    parser.add_argument("--stop-loss", type=float, default=0.03, help="Stop loss fraction (default: 0.03)")
    parser.add_argument("--holding-days", type=int, default=7, help="Max holding days (default: 7)")
    parser.add_argument("--output", type=str, default=None, help="Output parquet path")
    args = parser.parse_args()

    # Determine ticker list
    if args.ticker_file:
        with open(args.ticker_file) as f:
            tickers = [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]
        logger.info(f"Loaded {len(tickers)} tickers from {args.ticker_file}")
    else:
        tickers = DEFAULT_TICKERS
        if args.stocks:
            tickers = tickers[: args.stocks]

    # Build dataset
    dataset = build_dataset(
        tickers=tickers,
        start=args.start,
        end=args.end,
        profit_target=args.profit_target,
        stop_loss=args.stop_loss,
        max_holding_days=args.holding_days,
    )

    if dataset.empty:
        logger.error("Empty dataset — aborting")
        sys.exit(1)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = args.output or str(OUTPUT_DIR / "training_dataset.parquet")
    dataset.to_parquet(output_path, index=False)
    logger.info(f"Saved dataset to {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
