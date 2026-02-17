"""Shared feature extraction for ML model — used by both training and inference.

All 20 features are computed locally from OHLCV data via stockstats + pandas.
Zero API calls required for indicator computation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from stockstats import wrap

from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)

# Canonical feature list — order matters for model consistency
FEATURE_COLUMNS: List[str] = [
    # Base indicators (20)
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "atr_pct",
    "bb_width_pct",
    "bb_position",
    "adx",
    "mfi",
    "stoch_k",
    "volume_ratio_5d",
    "volume_ratio_20d",
    "return_1d",
    "return_5d",
    "return_20d",
    "sma50_distance",
    "sma200_distance",
    "high_low_range",
    "gap_pct",
    "log_market_cap",
    # Interaction & derived features (10)
    "momentum_x_compression",  # strong trend + tight bands = breakout signal
    "rsi_momentum",  # RSI rate of change (acceleration)
    "volume_price_confirm",  # volume surge + positive return = confirmed move
    "trend_alignment",  # SMA50 and SMA200 agree on direction
    "volatility_regime",  # ATR percentile rank (0-1) within own history
    "mean_reversion_signal",  # oversold RSI + below lower BB
    "breakout_signal",  # above upper BB + high volume
    "macd_strength",  # MACD histogram normalized by ATR
    "return_volatility_ratio",  # Sharpe-like: return_5d / atr_pct
    "trend_momentum_score",  # combined trend + momentum z-score
]

# Minimum rows of OHLCV history needed before features are valid
# (200-day SMA needs 200 rows of prior data)
MIN_HISTORY_ROWS = 210


def compute_features_bulk(ohlcv: pd.DataFrame, market_cap: Optional[float] = None) -> pd.DataFrame:
    """Compute all 20 ML features for every row in an OHLCV DataFrame.

    Args:
        ohlcv: DataFrame with columns: Date, Open, High, Low, Close, Volume.
               Must be sorted by Date ascending.
        market_cap: Market capitalization in USD. If None, log_market_cap = NaN.

    Returns:
        DataFrame indexed by Date with one column per feature.
        Rows with insufficient history (first ~210) will have NaN values.
    """
    if ohlcv.empty or len(ohlcv) < MIN_HISTORY_ROWS:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    df = ohlcv.copy()

    # Ensure Date column is available and set as index
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

    # Normalize column names (yfinance sometimes returns Title Case)
    col_map = {}
    for col in df.columns:
        lower = col.lower()
        if lower == "open":
            col_map[col] = "Open"
        elif lower == "high":
            col_map[col] = "High"
        elif lower == "low":
            col_map[col] = "Low"
        elif lower in ("close", "adj close"):
            col_map[col] = "Close"
        elif lower == "volume":
            col_map[col] = "Volume"
    df = df.rename(columns=col_map)

    # Need these columns
    for required in ("Open", "High", "Low", "Close", "Volume"):
        if required not in df.columns:
            logger.warning(f"Missing column {required} in OHLCV data")
            return pd.DataFrame(columns=FEATURE_COLUMNS)

    close = df["Close"]
    volume = df["Volume"]

    # --- Stockstats indicators ---
    ss = wrap(df.copy())

    features = pd.DataFrame(index=df.index)

    # 1. RSI (14-period)
    features["rsi_14"] = ss["rsi_14"]

    # 2-4. MACD (12, 26, 9)
    features["macd"] = ss["macd"]
    features["macd_signal"] = ss["macds"]
    features["macd_hist"] = ss["macdh"]

    # 5. ATR as percentage of price
    atr = ss["atr_14"]
    features["atr_pct"] = (atr / close) * 100

    # 6. Bollinger Band width as percentage
    bb_upper = ss["boll_ub"]
    bb_lower = ss["boll_lb"]
    bb_middle = ss["boll"]
    features["bb_width_pct"] = ((bb_upper - bb_lower) / bb_middle) * 100

    # 7. Position within Bollinger Bands (0 = lower band, 1 = upper band)
    bb_range = bb_upper - bb_lower
    features["bb_position"] = np.where(
        bb_range > 0, (close - bb_lower) / bb_range, 0.5
    )

    # 8. ADX (trend strength)
    features["adx"] = ss["dx_14"]

    # 9. Money Flow Index
    features["mfi"] = ss["mfi_14"]

    # 10. Stochastic %K
    features["stoch_k"] = ss["kdjk"]

    # --- Pandas-computed features ---

    # 11-12. Volume ratios
    vol_ma_5 = volume.rolling(5).mean()
    vol_ma_20 = volume.rolling(20).mean()
    features["volume_ratio_5d"] = volume / vol_ma_5.replace(0, np.nan)
    features["volume_ratio_20d"] = volume / vol_ma_20.replace(0, np.nan)

    # 13-15. Historical returns (looking backward — no data leakage)
    features["return_1d"] = close.pct_change(1, fill_method=None) * 100
    features["return_5d"] = close.pct_change(5, fill_method=None) * 100
    features["return_20d"] = close.pct_change(20, fill_method=None) * 100

    # 16-17. Distance from moving averages
    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()
    features["sma50_distance"] = ((close - sma_50) / sma_50) * 100
    features["sma200_distance"] = ((close - sma_200) / sma_200) * 100

    # 18. High-Low range as percentage of close
    features["high_low_range"] = ((df["High"] - df["Low"]) / close) * 100

    # 19. Gap percentage (open vs previous close)
    prev_close = close.shift(1)
    features["gap_pct"] = ((df["Open"] - prev_close) / prev_close) * 100

    # 20. Log market cap (static per stock)
    if market_cap and market_cap > 0:
        features["log_market_cap"] = np.log10(market_cap)
    else:
        features["log_market_cap"] = np.nan

    # --- Interaction & derived features (10) ---

    # 21. Momentum × Compression: strong trend direction + tight Bollinger = breakout setup
    #     High absolute MACD + low BB width = coiled spring
    features["momentum_x_compression"] = features["macd_hist"].abs() / features["bb_width_pct"].replace(0, np.nan)

    # 22. RSI momentum: 5-day rate of change of RSI (acceleration of momentum)
    features["rsi_momentum"] = features["rsi_14"] - features["rsi_14"].shift(5)

    # 23. Volume-price confirmation: volume surge accompanied by price move
    features["volume_price_confirm"] = features["volume_ratio_5d"] * features["return_1d"]

    # 24. Trend alignment: both SMAs agree (1 = aligned bullish, -1 = aligned bearish)
    features["trend_alignment"] = np.sign(features["sma50_distance"]) * np.sign(features["sma200_distance"])

    # 25. Volatility regime: ATR percentile within rolling 60-day window (0-1)
    atr_pct_series = features["atr_pct"]
    features["volatility_regime"] = atr_pct_series.rolling(60).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5,
        raw=False,
    )

    # 26. Mean reversion signal: oversold RSI + price below lower Bollinger
    features["mean_reversion_signal"] = (
        (100 - features["rsi_14"]) / 100  # inversed RSI (higher = more oversold)
    ) * (1 - features["bb_position"].clip(0, 1))  # below lower band amplifies signal

    # 27. Breakout signal: above upper BB + high volume ratio
    features["breakout_signal"] = (
        features["bb_position"].clip(0, 2) * features["volume_ratio_20d"]
    )

    # 28. MACD strength: histogram normalized by volatility
    features["macd_strength"] = features["macd_hist"] / features["atr_pct"].replace(0, np.nan)

    # 29. Return/Volatility ratio: Sharpe-like metric
    features["return_volatility_ratio"] = features["return_5d"] / features["atr_pct"].replace(0, np.nan)

    # 30. Trend-momentum composite score
    features["trend_momentum_score"] = (
        features["sma50_distance"] * 0.4
        + features["rsi_14"].sub(50) * 0.3  # RSI centered at 50
        + features["macd_hist"] * 0.3
    )

    return features[FEATURE_COLUMNS]


def compute_features_single(
    ohlcv: pd.DataFrame,
    date: str,
    market_cap: Optional[float] = None,
) -> Optional[Dict[str, float]]:
    """Compute features for a single date. Used during live inference.

    Args:
        ohlcv: Full OHLCV DataFrame (needs ~210 rows of history before `date`).
        date: Target date string (YYYY-MM-DD).
        market_cap: Market cap in USD.

    Returns:
        Dict mapping feature name → value, or None if insufficient data.
    """
    features_df = compute_features_bulk(ohlcv, market_cap=market_cap)
    if features_df.empty:
        return None

    date_ts = pd.Timestamp(date)
    # Find the closest date on or before the target
    valid = features_df.index[features_df.index <= date_ts]
    if len(valid) == 0:
        return None

    row = features_df.loc[valid[-1]]
    if row.isna().all():
        return None

    return row.to_dict()


def compute_features_from_enriched_candidate(cand: Dict) -> Optional[Dict[str, float]]:
    """Extract ML features from an already-enriched discovery candidate.

    During live inference, the enrichment pipeline has already computed
    many of the values we need. This avoids redundant computation.

    Args:
        cand: Enriched candidate dict from filter.py.

    Returns:
        Dict of feature values, or None if critical fields are missing.
    """
    features: Dict[str, float] = {}

    # Features already available on enriched candidates
    features["rsi_14"] = cand.get("rsi_value", np.nan)
    features["atr_pct"] = cand.get("atr_pct", np.nan)
    features["bb_width_pct"] = cand.get("bb_width_pct", np.nan)
    features["volume_ratio_20d"] = cand.get("volume_ratio", np.nan)

    # Market cap
    market_cap_bil = cand.get("market_cap_bil")
    if market_cap_bil and market_cap_bil > 0:
        features["log_market_cap"] = np.log10(market_cap_bil * 1e9)
    else:
        features["log_market_cap"] = np.nan

    # Intraday return as proxy for return_1d
    features["return_1d"] = cand.get("intraday_change_pct", np.nan)

    # Short interest as a signal (use as proxy where we lack full OHLCV)
    short_pct = cand.get("short_interest_pct")
    if short_pct is not None:
        features["log_market_cap"] = features.get("log_market_cap", np.nan)

    # For features not directly available on enriched candidates,
    # we need to fetch OHLCV and compute. This is the "full path".
    # Return None to signal the caller should use compute_features_single() instead.
    missing = [f for f in FEATURE_COLUMNS if f not in features or np.isnan(features.get(f, np.nan))]
    if len(missing) > 5:
        # Too many missing — need full OHLCV computation
        return None

    # Fill remaining with NaN (TabPFN handles missing values)
    for col in FEATURE_COLUMNS:
        if col not in features:
            features[col] = np.nan

    return features


def apply_triple_barrier_labels(
    close_prices: pd.Series,
    profit_target: float = 0.05,
    stop_loss: float = 0.03,
    max_holding_days: int = 7,
) -> pd.Series:
    """Apply triple-barrier labeling to a series of close prices.

    For each day, looks forward up to `max_holding_days` trading days:
      +1 (WIN):     Price hits +profit_target first
      -1 (LOSS):    Price hits -stop_loss first
       0 (TIMEOUT): Neither barrier hit within the window

    Args:
        close_prices: Series of daily close prices, indexed by date.
        profit_target: Upside target as fraction (0.05 = 5%).
        stop_loss: Downside limit as fraction (0.03 = 3%).
        max_holding_days: Maximum forward-looking trading days.

    Returns:
        Series of labels (+1, -1, 0) aligned with close_prices index.
        Last `max_holding_days` rows will be NaN (can't look forward).
    """
    prices = close_prices.values
    n = len(prices)
    labels = np.full(n, np.nan)

    for i in range(n - max_holding_days):
        entry = prices[i]
        upper = entry * (1 + profit_target)
        lower = entry * (1 - stop_loss)

        label = 0  # default: timeout
        for j in range(1, max_holding_days + 1):
            future_price = prices[i + j]
            if future_price >= upper:
                label = 1  # hit profit target
                break
            elif future_price <= lower:
                label = -1  # hit stop loss
                break

        labels[i] = label

    return pd.Series(labels, index=close_prices.index, name="label")
