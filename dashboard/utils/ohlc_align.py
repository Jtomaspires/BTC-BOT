from __future__ import annotations

import numpy as np
import pandas as pd


def align_ohlc_for_engine(
    signals_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Alinhamento bit-exato com ``INF/run_walkforward`` (test e val).

    Ver comentário em ``dashboard.utils.backtest_lite`` (histórico).
    """
    raw = signals_df["signal"].to_numpy(dtype=np.float64)
    o0 = signals_df["open"].to_numpy(dtype=np.float64)
    h0 = signals_df["high"].to_numpy(dtype=np.float64)
    l0 = signals_df["low"].to_numpy(dtype=np.float64)
    c0 = signals_df["close"].to_numpy(dtype=np.float64)

    if raw.shape[0] == 0:
        raise ValueError("signals_df is empty")
    if not (o0.shape[0] == h0.shape[0] == l0.shape[0] == c0.shape[0]):
        raise ValueError("OHLC columns must all have the same length in signals_df")

    if o0.shape[0] == raw.shape[0] + 1:
        return raw, o0, h0, l0, c0

    if o0.shape[0] != raw.shape[0]:
        raise ValueError(
            f"Inconsistent signal vs OHLC row counts: signal={raw.shape[0]}, OHLC={o0.shape[0]} "
            "(expected equal rows, or OHLC = signal + 1 if already aligned)."
        )

    o = np.concatenate([o0, [o0[-1]]])
    h = np.concatenate([h0, [h0[-1]]])
    l = np.concatenate([l0, [l0[-1]]])
    c = np.concatenate([c0, [c0[-1]]])
    return raw, o, h, l, c
