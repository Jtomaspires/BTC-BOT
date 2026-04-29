from __future__ import annotations

import importlib.util
from pathlib import Path

_bootstrap_path = Path(__file__).resolve().parent.parent / "bootstrap_sys_path.py"
_spec = importlib.util.spec_from_file_location("_nn_dashboard_bootstrap", _bootstrap_path)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(importlib.util.module_from_spec(_spec))

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from INF.backtest_engine import run_backtest_grid
from INF.metrics import max_drawdown_info, sharpe_ratio


@dataclass(frozen=True)
class CellMetrics:
    equity: float
    sharpe: float
    max_dd: float
    robustness: float


def _neighbor_mean(mat: np.ndarray) -> np.ndarray:
    """Mean of 8-neighborhood (excluding self); NaNs ignored."""
    h, w = mat.shape
    out = np.full((h, w), np.nan, dtype=np.float64)
    for i in range(h):
        for j in range(w):
            vals = []
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ii, jj = i + di, j + dj
                    if 0 <= ii < h and 0 <= jj < w:
                        v = mat[ii, jj]
                        if np.isfinite(v):
                            vals.append(float(v))
            out[i, j] = float(np.mean(vals)) if vals else np.nan
    return out


def _align_ohlc_for_engine(
    signals_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Alinhamento bit-exato com ``INF/run_walkforward`` (test e val).

    Convenção do ``run_walkforward._save_signals_csv``:
    - Salva ``n`` linhas OHLC e ``n`` sinais (1 linha por sinal).
    - Os ``n`` opens do CSV == ``opens[0..n-1]`` que o engine **realmente usa** no loop.

    Convenção do ``run_single_backtest`` (engine):
    - Exige ``len(opens) == len(signal) + 1``, mas só lê ``opens[0..n-1]``.
    - A última barra ``opens[n]`` é apenas para a invariante de tamanho — nunca acessada
      no loop nem no fecho final (que usa ``c[n-1]``).

    Logo, basta **apender** uma barra fantasma (duplicar a última do CSV); o resultado
    do backtest é numericamente idêntico ao INF.
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


def run_grid(
    *,
    signals_df: pd.DataFrame,
    sl_values: Iterable[float],
    tp_values: Iterable[float],
    threshold_values: Iterable[float],
    trailing_stop_values: Iterable[float],
    window_dir: Optional[Path] = None,  # mantido por compatibilidade; já não é necessário
    taker_fee: float = 0.00055,
    position_notional: float = 1000.0,
    periods_per_year: float = 365.0 * 24.0,
) -> dict[tuple[float, float, float, float], CellMetrics]:
    """
    Runs a SL/TP grid using precomputed signals.csv (no model inference).

    Resultado **bit-exato** com ``INF/run_walkforward`` quando ``signal_threshold``,
    ``taker_fee``, ``position_notional`` e ``trailing_stop_points`` são iguais aos do
    config — ver ``_align_ohlc_for_engine`` para o porquê.

    Returns a dict keyed by (sl, tp, threshold, trailing_stop).
    Robustness é a média da vizinhança 8 da equity por (threshold, trailing_stop).
    """
    del window_dir  # noqa: F841 — argumento legado, sem uso após alinhamento bit-exato

    required = {"signal", "open", "high", "low", "close"}
    missing = required - set(signals_df.columns)
    if missing:
        raise ValueError(f"signals_df missing columns: {sorted(missing)}")

    raw, o, h, l, c = _align_ohlc_for_engine(signals_df)

    sls = [float(x) for x in sl_values]
    tps = [float(x) for x in tp_values]
    thrs = [float(x) for x in threshold_values]
    trails = [float(x) for x in trailing_stop_values]

    out: dict[tuple[float, float, float, float], CellMetrics] = {}

    for thr in thrs:
        for trail in trails:
            grid = run_backtest_grid(
                raw,
                o,
                h,
                l,
                c,
                sl_points=sls,
                tp_points=tps,
                trailing_stop_points=[trail],
                taker_fee=float(taker_fee),
                position_notional=float(position_notional),
                signal_threshold=float(thr),
            )
            # Build equity matrix for robustness scoring.
            equity_mat = np.full((len(sls), len(tps)), np.nan, dtype=np.float64)
            cell_tmp: dict[tuple[float, float], tuple[float, float, float]] = {}
            for (sl, tp, _trail), res in grid.items():
                eq_final = float(res.equities[-1]) if len(res.equities) else float(position_notional)
                sh = float(sharpe_ratio(res.equities, periods_per_year=periods_per_year))
                dd = float(max_drawdown_info(res.equities).get("max_drawdown", 0.0))
                cell_tmp[(float(sl), float(tp))] = (eq_final, sh, dd)
                i = sls.index(float(sl))
                j = tps.index(float(tp))
                equity_mat[i, j] = eq_final

            robust = _neighbor_mean(equity_mat)
            for i, sl in enumerate(sls):
                for j, tp in enumerate(tps):
                    eq_final, sh, dd = cell_tmp.get((sl, tp), (float("nan"), float("nan"), float("nan")))
                    out[(sl, tp, thr, trail)] = CellMetrics(
                        equity=eq_final,
                        sharpe=sh,
                        max_dd=dd,
                        robustness=float(robust[i, j]) if np.isfinite(robust[i, j]) else float("nan"),
                    )

    return out

