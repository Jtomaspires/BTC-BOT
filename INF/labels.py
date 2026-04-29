"""
Construção de labels supervisionados para treino.
"""

from __future__ import annotations

import numpy as np


def build_triple_barrier_labels(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    *,
    tp_pct: float,
    sl_pct: float,
    horizon: int,
) -> np.ndarray:
    """
    Labels de 3 classes:
    - 0: timeout (nenhuma barreira tocada no horizonte)
    - 1: TP tocado primeiro (long)
    - 2: SL tocado primeiro (short)

    Os últimos `horizon` elementos são marcados como -1 (inválidos).
    """
    c = np.asarray(closes, dtype=np.float64).reshape(-1)
    h = np.asarray(highs, dtype=np.float64).reshape(-1)
    l = np.asarray(lows, dtype=np.float64).reshape(-1)

    if not (len(c) == len(h) == len(l)):
        raise ValueError("closes/highs/lows devem ter o mesmo comprimento")
    if len(c) == 0:
        raise ValueError("séries OHLC não podem ser vazias")
    if tp_pct <= 0 or sl_pct <= 0:
        raise ValueError("tp_pct e sl_pct devem ser > 0")
    if horizon <= 0:
        raise ValueError("horizon deve ser > 0")

    n = len(c)
    labels = np.zeros(n, dtype=np.int64)
    valid_upto = max(0, n - horizon)
    for i in range(valid_upto):
        entry = float(c[i])
        tp_price = entry * (1.0 + float(tp_pct))
        sl_price = entry * (1.0 - float(sl_pct))
        for j in range(i + 1, min(n, i + horizon + 1)):
            if h[j] >= tp_price:
                labels[i] = 1
                break
            if l[j] <= sl_price:
                labels[i] = 2
                break

    labels[valid_upto:] = -1
    return labels
