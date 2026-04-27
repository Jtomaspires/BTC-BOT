from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_WF = Path(__file__).resolve().parent.parent
if str(_WF) not in sys.path:
    sys.path.insert(0, str(_WF))

import backtest_engine as BE


def _reference_trading_backtest(
    all_raw_signals,
    opens,
    highs,
    lows,
    closes,
    *,
    sl_points: float,
    tp_points: float,
    threshold: float,
    pos_size: float,
    taker_fee: float,
):
    """Referência 1x1 fiel à lógica do notebook CNN_BTC."""
    cash = float(pos_size)
    position = 0
    entry_price = None

    equities = []
    fee_log = []
    actions_log = []
    total_fees = 0.0
    entries = 0
    completed_trades = 0
    sl_hits = 0
    tp_hits = 0
    num_longs = 0
    num_shorts = 0
    trade_pnls = []

    def fee_per_leg():
        return pos_size * taker_fee

    def realize_to(price):
        nonlocal cash, entry_price, position, completed_trades, total_fees
        if position == 0 or entry_price is None:
            return
        pct = (price - entry_price) / entry_price
        pnl = pos_size * (pct if position == 1 else -pct)
        fee = fee_per_leg()
        cash += pnl - fee
        total_fees += fee
        trade_pnls.append(pnl - fee)
        entry_price = None
        position = 0
        completed_trades += 1

    def mark_to_market(close_price):
        if position == 0 or entry_price is None:
            return cash
        pct = (close_price - entry_price) / entry_price
        return cash + pos_size * (pct if position == 1 else -pct)

    n = len(opens) - 1
    pending_entry = None
    for i in range(n):
        curr_open = opens[i]
        curr_high = highs[i]
        curr_low = lows[i]
        curr_close = closes[i]
        bar_fee = 0.0
        closed_by_sl_tp_this_bar = False

        raw_values = [arr[i] for arr in all_raw_signals]
        mean_signal = np.mean(raw_values)
        if mean_signal > threshold:
            desired = 1
        elif mean_signal < -threshold:
            desired = -1
        else:
            desired = 0

        if position == 0 and pending_entry is not None:
            fee = fee_per_leg()
            cash -= fee
            bar_fee += fee
            total_fees += fee
            entry_price = curr_open
            position = pending_entry
            entries += 1
            num_longs += pending_entry == 1
            num_shorts += pending_entry == -1
            pending_entry = None

        if position != 0 and entry_price is not None:
            sl_price = entry_price - sl_points if position == 1 else entry_price + sl_points
            tp_price = entry_price + tp_points if position == 1 else entry_price - tp_points

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

        if position == 0 and desired != 0:
            if closed_by_sl_tp_this_bar:
                pending_entry = desired
            else:
                fee = fee_per_leg()
                cash -= fee
                bar_fee += fee
                total_fees += fee
                entry_price = curr_open
                position = desired
                entries += 1
                num_longs += desired == 1
                num_shorts += desired == -1

        equity = mark_to_market(curr_close)
        equities.append(equity)
        fee_log.append(bar_fee)
        actions_log.append(position)

    if position != 0:
        realize_to(closes[len(equities) - 1])

    return {
        "equities": np.asarray(equities, dtype=np.float64),
        "trade_pnls": list(trade_pnls),
        "total_fees": float(total_fees),
        "entries": int(entries),
        "completed_trades": int(completed_trades),
        "sl_hits": int(sl_hits),
        "tp_hits": int(tp_hits),
        "num_longs": int(num_longs),
        "num_shorts": int(num_shorts),
        "fee_log": list(fee_log),
        "actions_log": list(actions_log),
    }


class TestBacktestEngineParity(unittest.TestCase):
    def _run_both(self, raw, o, h, l, c, *, sl=2.0, tp=4.0, th=0.5, pos=1000.0, fee=0.00055):
        ref = _reference_trading_backtest(
            [np.asarray(raw, dtype=np.float64)],
            np.asarray(o, dtype=np.float64),
            np.asarray(h, dtype=np.float64),
            np.asarray(l, dtype=np.float64),
            np.asarray(c, dtype=np.float64),
            sl_points=sl,
            tp_points=tp,
            threshold=th,
            pos_size=pos,
            taker_fee=fee,
        )
        eng = BE.run_single_backtest(
            raw,
            o,
            h,
            l,
            c,
            sl_points=sl,
            tp_points=tp,
            taker_fee=fee,
            position_notional=pos,
            signal_threshold=th,
        )
        return ref, eng

    def test_parity_random_like_series(self):
        rng = np.random.default_rng(42)
        n = 120
        opens = 100 + np.cumsum(rng.normal(0, 0.4, size=n + 1))
        highs = opens + np.abs(rng.normal(0.8, 0.2, size=n + 1))
        lows = opens - np.abs(rng.normal(0.8, 0.2, size=n + 1))
        closes = opens + rng.normal(0, 0.3, size=n + 1)
        raw = rng.normal(0, 1, size=n)

        ref, eng = self._run_both(raw, opens, highs, lows, closes, sl=1.5, tp=2.5, th=0.2)

        self.assertTrue(np.allclose(eng.equities, ref["equities"], atol=1e-9))
        self.assertEqual(eng.entries, ref["entries"])
        self.assertEqual(eng.completed_trades, ref["completed_trades"])
        self.assertEqual(eng.sl_hits, ref["sl_hits"])
        self.assertEqual(eng.tp_hits, ref["tp_hits"])
        self.assertEqual(eng.num_longs, ref["num_longs"])
        self.assertEqual(eng.num_shorts, ref["num_shorts"])
        self.assertTrue(np.allclose(np.asarray(eng.trade_pnls), np.asarray(ref["trade_pnls"]), atol=1e-9))
        self.assertAlmostEqual(eng.total_fees, ref["total_fees"], places=12)
        self.assertTrue(np.allclose(np.asarray(eng.fee_log), np.asarray(ref["fee_log"]), atol=1e-12))
        self.assertEqual(list(eng.actions_log), ref["actions_log"])

    def test_sl_precedence_when_both_levels_touch(self):
        # i=0 abre long; i=1 toca simultaneamente SL e TP. Deve contar SL (if antes de elif).
        opens = [100.0, 100.0, 100.0]
        highs = [100.5, 105.0, 100.0]
        lows = [99.5, 97.0, 100.0]
        closes = [100.0, 100.0, 100.0]
        raw = [1.0, 0.0]

        ref, eng = self._run_both(raw, opens, highs, lows, closes, sl=2.0, tp=2.0, th=0.5)
        self.assertTrue(np.allclose(eng.equities, ref["equities"], atol=1e-9))
        self.assertEqual(eng.sl_hits, 1)
        self.assertEqual(eng.tp_hits, 0)

    def test_pending_entry_last_bar_is_not_opened(self):
        # No último i do loop: SL fecha e desired cria pending_entry, que deve ser descartado.
        opens = [100.0, 100.0, 100.0, 100.0]
        highs = [101.0, 101.0, 101.0, 101.0]
        lows = [99.0, 99.0, 97.0, 99.0]
        closes = [100.0, 100.0, 100.0, 100.0]
        raw = [1.0, 1.0, 1.0]

        ref, eng = self._run_both(raw, opens, highs, lows, closes, sl=2.0, tp=5.0, th=0.5)
        self.assertTrue(np.allclose(eng.equities, ref["equities"], atol=1e-9))
        self.assertEqual(eng.entries, ref["entries"])
        self.assertEqual(eng.entries, 1)
        self.assertEqual(eng.actions_log[-1], 0)
        self.assertAlmostEqual(eng.total_fees, ref["total_fees"], places=12)
        self.assertEqual(len(eng.equities), len(opens) - 1)

    def test_run_backtest_grid_product(self):
        opens = [100.0, 100.0, 100.0, 100.0]
        highs = [101.0, 101.0, 101.0, 101.0]
        lows = [99.0, 99.0, 99.0, 99.0]
        closes = [100.0, 100.0, 100.0, 100.0]
        raw = [0.0, 0.0, 0.0]
        out = BE.run_backtest_grid(
            raw,
            opens,
            highs,
            lows,
            closes,
            sl_points=[1.0, 2.0],
            tp_points=[3.0, 4.0],
            taker_fee=0.00055,
            position_notional=1000.0,
            signal_threshold=0.5,
        )
        self.assertEqual(len(out), 4)
        self.assertIn((1.0, 3.0, 0.0), out)
        self.assertIn((2.0, 4.0, 0.0), out)
        for v in out.values():
            self.assertIsInstance(v, BE.BacktestResult)

    def test_run_backtest_grid_includes_trailing_product(self):
        opens = [100.0, 100.0, 100.0, 100.0]
        highs = [101.0, 101.0, 101.0, 101.0]
        lows = [99.0, 99.0, 99.0, 99.0]
        closes = [100.0, 100.0, 100.0, 100.0]
        raw = [0.0, 0.0, 0.0]
        out = BE.run_backtest_grid(
            raw,
            opens,
            highs,
            lows,
            closes,
            sl_points=[1.0],
            tp_points=[3.0],
            trailing_stop_points=[0.0, 2.0],
            taker_fee=0.00055,
            position_notional=1000.0,
            signal_threshold=0.5,
        )
        self.assertEqual(len(out), 2)
        self.assertIn((1.0, 3.0, 0.0), out)
        self.assertIn((1.0, 3.0, 2.0), out)

    def test_trailing_stop_can_close_long_above_entry(self):
        opens = [100.0, 110.0, 110.0, 110.0]
        highs = [101.0, 130.0, 111.0, 111.0]
        lows = [99.0, 109.0, 109.0, 109.0]
        closes = [100.0, 125.0, 110.0, 110.0]
        raw = [1.0, 1.0, 1.0]
        result = BE.run_single_backtest(
            raw,
            opens,
            highs,
            lows,
            closes,
            sl_points=20.0,
            tp_points=100.0,
            taker_fee=0.0,
            position_notional=1000.0,
            signal_threshold=0.5,
            trailing_stop_points=10.0,
        )
        self.assertEqual(result.sl_hits, 1)
        self.assertAlmostEqual(result.trade_pnls[0], 200.0, places=9)
        self.assertEqual(result.trailing_stop_points, 10.0)


if __name__ == "__main__":
    unittest.main()
