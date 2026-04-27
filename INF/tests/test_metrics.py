from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_WF = Path(__file__).resolve().parent.parent
if str(_WF) not in sys.path:
    sys.path.insert(0, str(_WF))

import metrics as MT


class TestMetricsCore(unittest.TestCase):
    def test_equity_returns(self):
        eq = np.asarray([100.0, 110.0, 99.0], dtype=np.float64)
        out = MT.equity_returns(eq)
        self.assertTrue(np.allclose(out, np.asarray([0.1, -0.1], dtype=np.float64), atol=1e-12))

    def test_max_drawdown_known_case(self):
        # Pico 120 -> vale 90 => DD = 25%.
        eq = np.asarray([100.0, 120.0, 90.0, 110.0], dtype=np.float64)
        self.assertAlmostEqual(MT.max_drawdown(eq), 0.25, places=12)

    def test_max_drawdown_info_localises_peak_and_trough(self):
        # Pico em idx=3 (150), vale em idx=5 (90) => DD = 40%.
        eq = np.asarray([100.0, 110.0, 130.0, 150.0, 120.0, 90.0, 140.0], dtype=np.float64)
        info = MT.max_drawdown_info(eq)
        self.assertAlmostEqual(info["max_drawdown"], (150.0 - 90.0) / 150.0, places=12)
        self.assertEqual(info["peak_index"], 3)
        self.assertEqual(info["trough_index"], 5)
        self.assertAlmostEqual(info["peak_value"], 150.0, places=12)
        self.assertAlmostEqual(info["trough_value"], 90.0, places=12)

    def test_win_rate(self):
        self.assertAlmostEqual(MT.win_rate([1.0, -1.0, 2.0, 0.0]), 0.5, places=12)
        self.assertEqual(MT.win_rate([]), 0.0)

    def test_sharpe_smoke(self):
        eq_up = np.asarray([100, 102, 104, 106, 108], dtype=np.float64)
        eq_flat = np.asarray([100, 100, 100, 100], dtype=np.float64)
        self.assertGreater(MT.sharpe_ratio(eq_up, periods_per_year=365.0), 0.0)
        self.assertEqual(MT.sharpe_ratio(eq_flat, periods_per_year=365.0), 0.0)

    def test_profit_factor(self):
        self.assertAlmostEqual(MT.profit_factor([3.0, -1.0, 2.0, -1.0]), 2.5, places=12)
        self.assertEqual(MT.profit_factor([]), 0.0)
        self.assertTrue(np.isinf(MT.profit_factor([1.0, 2.0])))


class TestSummaryAndBestSelection(unittest.TestCase):
    def _fake_grid(self):
        return {
            (1.0, 3.0): {
                "equities": np.asarray([100.0, 101.0, 100.5], dtype=np.float64),
                "trade_pnls": [1.0, -0.5],
                "total_fees": 1.1,
                "entries": 2,
                "completed_trades": 2,
            },
            (2.0, 4.0): {
                "equities": np.asarray([100.0, 102.0, 104.0], dtype=np.float64),
                "trade_pnls": [2.0, 2.0],
                "total_fees": 1.1,
                "entries": 2,
                "completed_trades": 2,
            },
            (3.0, 5.0): {
                "equities": np.asarray([100.0, 99.0, 98.0], dtype=np.float64),
                "trade_pnls": [-1.0, -1.0],
                "total_fees": 1.1,
                "entries": 2,
                "completed_trades": 2,
            },
        }

    def test_summarize_window(self):
        df = MT.summarize_window(self._fake_grid(), periods_per_year=365.0)
        self.assertEqual(len(df), 3)
        self.assertIn("sl", df.columns)
        self.assertIn("tp", df.columns)
        self.assertIn("trailing_stop", df.columns)
        self.assertIn("final_equity", df.columns)
        self.assertIn("sharpe", df.columns)
        self.assertIn("max_drawdown", df.columns)
        self.assertIn("win_rate", df.columns)

    def test_summarize_window_with_trailing_key(self):
        grid = {
            (1.0, 3.0, 0.0): {"equities": [100.0, 101.0], "trade_pnls": [1.0]},
            (1.0, 3.0, 2.0): {"equities": [100.0, 102.0], "trade_pnls": [2.0]},
        }
        df = MT.summarize_window(grid, periods_per_year=365.0)
        self.assertEqual(df["trailing_stop"].tolist(), [0.0, 2.0])
        sl, tp, trailing, score = MT.select_best_grid_params(
            grid, metric="final_equity", periods_per_year=365.0
        )
        self.assertEqual((sl, tp, trailing), (1.0, 3.0, 2.0))
        self.assertAlmostEqual(score, 102.0, places=12)

    def test_select_best_final_equity(self):
        sl, tp, score = MT.select_best_sl_tp(
            self._fake_grid(), metric="final_equity", periods_per_year=365.0
        )
        self.assertEqual((sl, tp), (2.0, 4.0))
        self.assertAlmostEqual(score, 104.0, places=12)

    def test_select_best_total_pnl(self):
        sl, tp, _ = MT.select_best_sl_tp(self._fake_grid(), metric="total_pnl", periods_per_year=365.0)
        self.assertEqual((sl, tp), (2.0, 4.0))

    def test_select_best_max_drawdown(self):
        sl, tp, _ = MT.select_best_sl_tp(
            self._fake_grid(), metric="max_drawdown", periods_per_year=365.0
        )
        self.assertEqual((sl, tp), (2.0, 4.0))

    def test_select_best_tie_breaker(self):
        grid = {
            (1.0, 3.0): {"equities": [100.0, 110.0], "trade_pnls": [10.0]},
            (2.0, 2.0): {"equities": [100.0, 110.0], "trade_pnls": [10.0]},
        }
        sl, tp, score = MT.select_best_sl_tp(grid, metric="final_equity", periods_per_year=365.0)
        self.assertEqual((sl, tp), (2.0, 2.0))
        self.assertAlmostEqual(score, 110.0, places=12)


if __name__ == "__main__":
    unittest.main()
