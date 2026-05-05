from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


SUPPORTED = {
    "RSI_14",
    "RSI_21",
    "ATR_14",
    "ATR_200",
    "MACD_12_26_9",
    "MACDh_12_26_9",
    "MACD_hist",
    "Vol20",
    "Vol50",
    "dist_ma20",
    "body_pct",
    "range_pct",
}


def compute_indicators(df: pd.DataFrame, indicators: list[str]) -> pd.DataFrame:
    out = df.copy()
    close = out["close"].astype(float)

    for ind in indicators:
        if ind not in SUPPORTED:
            continue
        if ind == "RSI_14":
            out[ind] = rsi(close, 14)
        elif ind == "RSI_21":
            out[ind] = rsi(close, 21)
        elif ind == "ATR_14":
            out[ind] = atr(out, 14)
        elif ind == "ATR_200":
            out[ind] = atr(out, 200)
        elif ind == "MACD_12_26_9":
            macd_line, signal_line, _hist = macd(close, 12, 26, 9)
            out["MACD_12_26_9"] = macd_line
            out["MACDs_12_26_9"] = signal_line
        elif ind == "MACDh_12_26_9":
            macd_line, signal_line, hist = macd(close, 12, 26, 9)
            out["MACD_12_26_9"] = macd_line
            out["MACDs_12_26_9"] = signal_line
            out["MACDh_12_26_9"] = hist
        elif ind == "MACD_hist":
            macd_line, signal_line, hist = macd(close, 12, 26, 9)
            out["MACD_12_26_9"] = macd_line
            out["MACDs_12_26_9"] = signal_line
            out["MACD_hist"] = hist
        elif ind == "Vol20":
            out[ind] = out["volume"].astype(float).rolling(20, min_periods=20).mean()
        elif ind == "Vol50":
            out[ind] = out["volume"].astype(float).rolling(50, min_periods=50).mean()
        elif ind == "dist_ma20":
            ma20 = close.rolling(20, min_periods=20).mean()
            out[ind] = (close - ma20) / ma20.replace(0.0, np.nan)
        elif ind == "body_pct":
            o = out["open"].astype(float)
            out[ind] = (close - o) / o.replace(0.0, np.nan)
        elif ind == "range_pct":
            o = out["open"].astype(float)
            out[ind] = (out["high"].astype(float) - out["low"].astype(float)) / o.replace(0.0, np.nan)

    return out

