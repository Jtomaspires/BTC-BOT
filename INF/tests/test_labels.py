from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_WF = Path(__file__).resolve().parent.parent
if str(_WF) not in sys.path:
    sys.path.insert(0, str(_WF))

import labels as LB


class TestTripleBarrierLabels(unittest.TestCase):
    def test_tp_hit_first_is_long_class(self):
        closes = np.asarray([100.0, 100.0, 100.0, 100.0], dtype=np.float64)
        highs = np.asarray([100.0, 103.0, 100.0, 100.0], dtype=np.float64)
        lows = np.asarray([100.0, 99.0, 99.0, 99.0], dtype=np.float64)
        y = LB.build_triple_barrier_labels(closes, highs, lows, tp_pct=0.02, sl_pct=0.02, horizon=2)
        self.assertEqual(int(y[0]), 1)

    def test_sl_hit_first_is_short_class(self):
        closes = np.asarray([100.0, 100.0, 100.0, 100.0], dtype=np.float64)
        highs = np.asarray([100.0, 101.0, 101.0, 101.0], dtype=np.float64)
        lows = np.asarray([100.0, 97.0, 99.0, 99.0], dtype=np.float64)
        y = LB.build_triple_barrier_labels(closes, highs, lows, tp_pct=0.02, sl_pct=0.02, horizon=2)
        self.assertEqual(int(y[0]), 2)

    def test_timeout_is_zero(self):
        closes = np.asarray([100.0, 100.0, 100.0, 100.0], dtype=np.float64)
        highs = np.asarray([100.0, 101.0, 101.0, 101.0], dtype=np.float64)
        lows = np.asarray([100.0, 99.5, 99.5, 99.5], dtype=np.float64)
        y = LB.build_triple_barrier_labels(closes, highs, lows, tp_pct=0.02, sl_pct=0.02, horizon=2)
        self.assertEqual(int(y[0]), 0)

    def test_last_horizon_labels_are_invalid(self):
        closes = np.asarray([100.0, 101.0, 102.0, 103.0, 104.0], dtype=np.float64)
        highs = closes + 0.2
        lows = closes - 0.2
        y = LB.build_triple_barrier_labels(closes, highs, lows, tp_pct=0.02, sl_pct=0.02, horizon=3)
        self.assertTrue(np.all(y[-3:] == -1))


if __name__ == "__main__":
    unittest.main()
