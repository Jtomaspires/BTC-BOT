"""
Métricas puras para resultados de backtest.

Convenções:
- `max_drawdown` retorna fração positiva (ex.: 0.15 = 15%).
- `win_rate` retorna fração em [0, 1].
- `sharpe_ratio` usa retornos simples derivados de equity.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


def _to_1d(name: str, values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} deve ser 1D; recebido shape={arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} não pode ser vazio")
    return arr


def equity_returns(equity: Sequence[float]) -> np.ndarray:
    eq = _to_1d("equity", equity)
    if eq.size < 2:
        return np.asarray([], dtype=np.float64)
    prev = eq[:-1]
    curr = eq[1:]
    out = np.zeros_like(curr, dtype=np.float64)
    mask = prev != 0.0
    out[mask] = (curr[mask] - prev[mask]) / prev[mask]
    return out


def sharpe_ratio(
    equity: Sequence[float],
    periods_per_year: float,
    *,
    rf: float = 0.0,
) -> float:
    if periods_per_year <= 0:
        raise ValueError("periods_per_year deve ser > 0")
    rets = equity_returns(equity)
    if rets.size < 2:
        return 0.0
    rf_per_period = rf / periods_per_year
    excess = rets - rf_per_period
    std = float(np.std(excess, ddof=1))
    if std <= 0.0:
        return 0.0
    mean = float(np.mean(excess))
    return float(np.sqrt(periods_per_year) * mean / std)


def max_drawdown(equity: Sequence[float]) -> float:
    eq = _to_1d("equity", equity)
    peaks = np.maximum.accumulate(eq)
    drawdowns = (peaks - eq) / peaks
    drawdowns = np.where(peaks == 0.0, 0.0, drawdowns)
    return float(np.max(drawdowns))


def max_drawdown_info(equity: Sequence[float]) -> Dict[str, Any]:
    """
    Alinhado ao `calculate_max_drawdown` de `CNN/CNN_BTC/backtest/main.ipynb`.

    Devolve fracção e localização do peak/trough (para anotar gráficos).
    """
    eq = _to_1d("equity", equity)
    peaks = np.maximum.accumulate(eq)
    dd = np.where(peaks == 0.0, 0.0, (peaks - eq) / peaks)
    trough_idx = int(np.argmax(dd))
    peak_idx = int(np.argmax(eq[: trough_idx + 1])) if trough_idx >= 0 else 0
    return {
        "max_drawdown": float(dd[trough_idx]) if dd.size else 0.0,
        "peak_index": peak_idx,
        "trough_index": trough_idx,
        "peak_value": float(eq[peak_idx]),
        "trough_value": float(eq[trough_idx]),
    }


def win_rate(trade_pnls: Sequence[float]) -> float:
    pnl = np.asarray(trade_pnls, dtype=np.float64)
    if pnl.size == 0:
        return 0.0
    return float(np.mean(pnl > 0.0))


def total_pnl(trade_pnls: Sequence[float]) -> float:
    return float(np.sum(np.asarray(trade_pnls, dtype=np.float64)))


def num_trades(trade_pnls: Sequence[float]) -> int:
    return int(np.asarray(trade_pnls, dtype=np.float64).size)


def avg_trade(trade_pnls: Sequence[float]) -> float:
    pnl = np.asarray(trade_pnls, dtype=np.float64)
    if pnl.size == 0:
        return 0.0
    return float(np.mean(pnl))


def profit_factor(trade_pnls: Sequence[float]) -> float:
    pnl = np.asarray(trade_pnls, dtype=np.float64)
    wins = float(np.sum(pnl[pnl > 0.0]))
    losses = float(np.sum(np.abs(pnl[pnl < 0.0])))
    if losses == 0.0:
        return float("inf") if wins > 0.0 else 0.0
    return wins / losses


def _extract_result_data(result: Any) -> Dict[str, Any]:
    if is_dataclass(result):
        data = asdict(result)
    elif isinstance(result, Mapping):
        data = dict(result)
    else:
        data = {
            "equities": getattr(result, "equities"),
            "trade_pnls": getattr(result, "trade_pnls"),
            "total_fees": getattr(result, "total_fees", 0.0),
        }
        for key in (
            "entries",
            "completed_trades",
            "sl_hits",
            "tp_hits",
            "num_longs",
            "num_shorts",
            "sl_points",
            "tp_points",
            "trailing_stop_points",
        ):
            if hasattr(result, key):
                data[key] = getattr(result, key)
    if "equities" not in data or "trade_pnls" not in data:
        raise ValueError("cada resultado deve fornecer 'equities' e 'trade_pnls'")
    return data


def _split_grid_key(key: Tuple[float, ...]) -> tuple[float, float, float]:
    if len(key) == 2:
        sl, tp = key
        return float(sl), float(tp), 0.0
    if len(key) == 3:
        sl, tp, trailing = key
        return float(sl), float(tp), float(trailing)
    raise ValueError(f"grid key inválida: {key!r}. Esperado (sl, tp) ou (sl, tp, trailing_stop)")


def summarize_window(
    results_by_sl_tp: Mapping[Tuple[float, ...], Any],
    *,
    periods_per_year: float,
) -> pd.DataFrame:
    rows = []
    for grid_key, result in results_by_sl_tp.items():
        sl, tp, trailing_stop = _split_grid_key(tuple(grid_key))
        data = _extract_result_data(result)
        eq = np.asarray(data["equities"], dtype=np.float64)
        pnl = np.asarray(data["trade_pnls"], dtype=np.float64)
        rows.append(
            {
                "sl": float(sl),
                "tp": float(tp),
                "trailing_stop": float(data.get("trailing_stop_points", trailing_stop)),
                "final_equity": float(eq[-1]) if eq.size else 0.0,
                "sharpe": sharpe_ratio(eq, periods_per_year=periods_per_year),
                "max_drawdown": max_drawdown(eq) if eq.size else 0.0,
                "win_rate": win_rate(pnl),
                "total_pnl": total_pnl(pnl),
                "num_trades": num_trades(pnl),
                "avg_trade": avg_trade(pnl),
                "profit_factor": profit_factor(pnl),
                "total_fees": float(data.get("total_fees", 0.0)),
                "entries": int(data.get("entries", 0)),
                "completed_trades": int(data.get("completed_trades", 0)),
                "sl_hits": int(data.get("sl_hits", 0)),
                "tp_hits": int(data.get("tp_hits", 0)),
                "num_longs": int(data.get("num_longs", 0)),
                "num_shorts": int(data.get("num_shorts", 0)),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "sl",
                "tp",
                "trailing_stop",
                "final_equity",
                "sharpe",
                "max_drawdown",
                "win_rate",
                "total_pnl",
                "num_trades",
                "avg_trade",
                "profit_factor",
                "total_fees",
                "entries",
                "completed_trades",
                "sl_hits",
                "tp_hits",
                "num_longs",
                "num_shorts",
            ]
        )
    df = pd.DataFrame(rows)
    return df.sort_values(["sl", "tp", "trailing_stop"], kind="stable").reset_index(drop=True)


def select_best_grid_params(
    grid_results: Mapping[Tuple[float, ...], Any],
    metric: str,
    *,
    periods_per_year: float,
) -> Tuple[float, float, float, float]:
    """
    Selecciona o melhor conjunto (SL, TP, trailing stop) por métrica.

    Métricas suportadas:
    - `sharpe` (maior melhor)
    - `final_equity` (maior melhor)
    - `max_drawdown` (menor melhor)
    - `total_pnl` (maior melhor)
    """
    if not grid_results:
        raise ValueError("grid_results não pode ser vazio")

    key = metric.strip().lower()
    allowed = {"sharpe", "final_equity", "max_drawdown", "total_pnl"}
    if key not in allowed:
        raise ValueError(f"metric inválida: {metric!r}. Use uma de {sorted(allowed)}")

    best_sl = None
    best_tp = None
    best_trailing = None
    best_score = None

    for grid_key, result in grid_results.items():
        sl, tp, trailing_stop = _split_grid_key(tuple(grid_key))
        data = _extract_result_data(result)
        eq = np.asarray(data["equities"], dtype=np.float64)
        pnl = np.asarray(data["trade_pnls"], dtype=np.float64)

        if key == "sharpe":
            score = sharpe_ratio(eq, periods_per_year=periods_per_year)
        elif key == "final_equity":
            score = float(eq[-1]) if eq.size else 0.0
        elif key == "max_drawdown":
            score = -max_drawdown(eq) if eq.size else 0.0
        else:  # total_pnl
            score = total_pnl(pnl)

        if best_score is None:
            best_sl, best_tp, best_trailing, best_score = (
                float(sl),
                float(tp),
                float(trailing_stop),
                float(score),
            )
            continue

        # Desempate determinístico: maior score, depois maior SL, maior TP, maior trailing.
        if (score > best_score) or (
            np.isclose(score, best_score)
            and (float(sl), float(tp), float(trailing_stop)) > (best_sl, best_tp, best_trailing)
        ):
            best_sl, best_tp, best_trailing, best_score = (
                float(sl),
                float(tp),
                float(trailing_stop),
                float(score),
            )

    assert best_sl is not None and best_tp is not None and best_trailing is not None and best_score is not None
    return best_sl, best_tp, best_trailing, best_score


def select_best_sl_tp(
    grid_results: Mapping[Tuple[float, ...], Any],
    metric: str,
    *,
    periods_per_year: float,
) -> Tuple[float, float, float]:
    """
    Compatibilidade legacy: devolve apenas (SL, TP, score).
    """
    best_sl, best_tp, _, best_score = select_best_grid_params(
        grid_results,
        metric=metric,
        periods_per_year=periods_per_year,
    )
    return best_sl, best_tp, best_score
