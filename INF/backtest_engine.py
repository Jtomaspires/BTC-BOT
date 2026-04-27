"""
Motor de backtest single-pair alinhado ao `trading_backtest` do notebook CNN_BTC.

Notas de paridade importantes:
- Ordem intra-barra: SL e depois TP (`if` / `elif`), usando `low/high` da barra.
- Se SL e TP estiverem ambos dentro do range da barra, prevalece SL.
- `pending_entry` evita reentrada no mesmo candle após fecho por SL/TP.
- No fim da série, `pending_entry` não é executado (sem preço futuro).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from models import get_action


@dataclass(frozen=True)
class BacktestResult:
    sl_points: float
    tp_points: float
    trailing_stop_points: float
    equities: np.ndarray
    trade_pnls: List[float]
    total_fees: float
    entries: int
    completed_trades: int
    sl_hits: int
    tp_hits: int
    num_longs: int
    num_shorts: int
    fee_log: List[float]
    actions_log: List[int]


def _as_1d_array(name: str, values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} deve ser 1D; recebido shape={arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} não pode ser vazio")
    return arr


def run_single_backtest(
    raw_signal_per_bar: Sequence[float],
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    *,
    sl_points: float,
    tp_points: float,
    taker_fee: float,
    position_notional: float,
    signal_threshold: float,
    trailing_stop_points: float = 0.0,
) -> BacktestResult:
    """
    Simula um único par (SL, TP) com a mesma semântica do notebook.

    Convenção de tamanhos (igual ao notebook):
    - O loop corre `n = len(opens) - 1` barras.
    - `raw_signal_per_bar` deve ter comprimento `n`.
    """
    o = _as_1d_array("opens", opens)
    h = _as_1d_array("highs", highs)
    l = _as_1d_array("lows", lows)
    c = _as_1d_array("closes", closes)
    s = _as_1d_array("raw_signal_per_bar", raw_signal_per_bar)

    if not (len(o) == len(h) == len(l) == len(c)):
        raise ValueError("opens/highs/lows/closes devem ter o mesmo comprimento")
    if len(o) < 2:
        raise ValueError("série OHLC deve ter pelo menos 2 barras")

    n = len(o) - 1
    if len(s) != n:
        raise ValueError(
            "raw_signal_per_bar deve ter comprimento len(opens)-1 "
            f"(esperado {n}, recebido {len(s)})"
        )
    if sl_points <= 0 or tp_points <= 0:
        raise ValueError("sl_points e tp_points devem ser > 0")
    if trailing_stop_points < 0:
        raise ValueError("trailing_stop_points deve ser >= 0")
    if taker_fee < 0:
        raise ValueError("taker_fee deve ser >= 0")
    if position_notional <= 0:
        raise ValueError("position_notional deve ser > 0")

    cash = float(position_notional)
    position = 0
    entry_price: Optional[float] = None
    high_watermark: Optional[float] = None
    low_watermark: Optional[float] = None
    pending_entry: Optional[int] = None

    equities: List[float] = []
    fee_log: List[float] = []
    actions_log: List[int] = []

    total_fees = 0.0
    entries = 0
    completed_trades = 0
    sl_hits = 0
    tp_hits = 0
    num_longs = 0
    num_shorts = 0
    trade_pnls: List[float] = []

    fee_per_leg = float(position_notional) * float(taker_fee)

    def realize_to(price: float) -> None:
        nonlocal cash, entry_price, position, completed_trades, total_fees, high_watermark, low_watermark
        if position == 0 or entry_price is None:
            return
        pct = (price - entry_price) / entry_price
        pnl = float(position_notional) * (pct if position == 1 else -pct)
        cash += pnl - fee_per_leg
        total_fees += fee_per_leg
        trade_pnls.append(pnl - fee_per_leg)
        entry_price = None
        high_watermark = None
        low_watermark = None
        position = 0
        completed_trades += 1

    def mark_to_market(close_price: float) -> float:
        if position == 0 or entry_price is None:
            return cash
        pct = (close_price - entry_price) / entry_price
        return cash + float(position_notional) * (pct if position == 1 else -pct)

    for i in range(n):
        curr_open = float(o[i])
        curr_high = float(h[i])
        curr_low = float(l[i])
        curr_close = float(c[i])
        bar_fee = 0.0
        closed_by_sl_tp_this_bar = False

        desired = get_action([float(s[i])], threshold=float(signal_threshold))

        if position == 0 and pending_entry is not None:
            cash -= fee_per_leg
            bar_fee += fee_per_leg
            total_fees += fee_per_leg
            entry_price = curr_open
            high_watermark = curr_open
            low_watermark = curr_open
            position = pending_entry
            entries += 1
            num_longs += pending_entry == 1
            num_shorts += pending_entry == -1
            pending_entry = None

        if position != 0 and entry_price is not None:
            sl_price = entry_price - sl_points if position == 1 else entry_price + sl_points
            tp_price = entry_price + tp_points if position == 1 else entry_price - tp_points
            if trailing_stop_points > 0:
                if position == 1 and high_watermark is not None:
                    sl_price = max(sl_price, high_watermark - trailing_stop_points)
                elif position == -1 and low_watermark is not None:
                    sl_price = min(sl_price, low_watermark + trailing_stop_points)

            if (position == 1 and curr_low <= sl_price) or (position == -1 and curr_high >= sl_price):
                realize_to(sl_price)
                sl_hits += 1
                closed_by_sl_tp_this_bar = True
            elif (position == 1 and curr_high >= tp_price) or (
                position == -1 and curr_low <= tp_price
            ):
                realize_to(tp_price)
                tp_hits += 1
                closed_by_sl_tp_this_bar = True

        if position != 0:
            if high_watermark is None or low_watermark is None:
                high_watermark = curr_open
                low_watermark = curr_open
            high_watermark = max(high_watermark, curr_high)
            low_watermark = min(low_watermark, curr_low)

        if position == 0 and desired != 0:
            if closed_by_sl_tp_this_bar:
                pending_entry = desired
            else:
                cash -= fee_per_leg
                bar_fee += fee_per_leg
                total_fees += fee_per_leg
                entry_price = curr_open
                high_watermark = curr_open
                low_watermark = curr_open
                position = desired
                entries += 1
                num_longs += desired == 1
                num_shorts += desired == -1

        equities.append(mark_to_market(curr_close))
        fee_log.append(bar_fee)
        actions_log.append(position)

    if position != 0:
        realize_to(float(c[len(equities) - 1]))

    return BacktestResult(
        sl_points=float(sl_points),
        tp_points=float(tp_points),
        trailing_stop_points=float(trailing_stop_points),
        equities=np.asarray(equities, dtype=np.float64),
        trade_pnls=trade_pnls,
        total_fees=float(total_fees),
        entries=int(entries),
        completed_trades=int(completed_trades),
        sl_hits=int(sl_hits),
        tp_hits=int(tp_hits),
        num_longs=int(num_longs),
        num_shorts=int(num_shorts),
        fee_log=fee_log,
        actions_log=actions_log,
    )


def run_backtest_grid(
    raw_signal_per_bar: Sequence[float],
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    *,
    sl_points: Sequence[float],
    tp_points: Sequence[float],
    trailing_stop_points: Sequence[float] | None = None,
    taker_fee: float,
    position_notional: float,
    signal_threshold: float,
) -> Dict[Tuple[float, float, float], BacktestResult]:
    """
    Corre o produto cartesiano de `sl_points x tp_points x trailing_stop_points`.
    """
    if not sl_points or not tp_points:
        raise ValueError("sl_points e tp_points devem conter pelo menos 1 valor")
    trailing_values = list(trailing_stop_points) if trailing_stop_points is not None else [0.0]
    if not trailing_values:
        raise ValueError("trailing_stop_points deve conter pelo menos 1 valor quando definido")

    results: Dict[Tuple[float, float, float], BacktestResult] = {}
    for sl, tp, trailing in product(sl_points, tp_points, trailing_values):
        key = (float(sl), float(tp), float(trailing))
        results[key] = run_single_backtest(
            raw_signal_per_bar,
            opens,
            highs,
            lows,
            closes,
            sl_points=float(sl),
            tp_points=float(tp),
            taker_fee=taker_fee,
            position_notional=position_notional,
            signal_threshold=signal_threshold,
            trailing_stop_points=float(trailing),
        )
    return results
