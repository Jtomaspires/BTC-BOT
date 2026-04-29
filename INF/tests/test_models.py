"""Forward pass e ``get_action`` — ``num_features`` alinhado a ``features.NUM_FEATURES``."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

_WF = Path(__file__).resolve().parent.parent
if str(_WF) not in sys.path:
    sys.path.insert(0, str(_WF))

import features as F
import models as M


class TestModelForward(unittest.TestCase):
    def setUp(self):
        self.nf = F.NUM_FEATURES
        self.batch, self.seq = 4, 48
        torch.manual_seed(0)
        self.x = torch.randn(self.batch, self.seq, self.nf, dtype=torch.float32)

    def _assert_out_shape(self, arch: str):
        m = M.get_model(arch, num_features=self.nf)
        m.eval()
        with torch.no_grad():
            y = m(self.x)
        self.assertEqual(y.shape, (self.batch, self.nf))

    def test_conv1d(self):
        self._assert_out_shape("conv1d")

    def test_lstm(self):
        self._assert_out_shape("lstm")

    def test_hybrid(self):
        self._assert_out_shape("hybrid")

    def test_unknown_arch(self):
        with self.assertRaises(ValueError):
            M.get_model("transformer", num_features=self.nf)

    def test_output_dim_override(self):
        m = M.get_model("conv1d", num_features=self.nf, output_dim=3)
        m.eval()
        with torch.no_grad():
            y = m(self.x)
        self.assertEqual(y.shape, (self.batch, 3))


class TestGetAction(unittest.TestCase):
    def test_threshold(self):
        self.assertEqual(M.get_action([0.001, 0.002], 0.0007), 1)
        self.assertEqual(M.get_action([-0.001, -0.002], 0.0007), -1)
        self.assertEqual(M.get_action([0.0001, -0.0001], 0.0007), 0)

    def test_empty(self):
        self.assertEqual(M.get_action([], 0.5), 0)


class TestLogitsToSignal(unittest.TestCase):
    def test_logits_to_signal_shape_and_range(self):
        logits = torch.tensor(
            [
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        s = M.logits_to_signal(logits)
        self.assertEqual(s.shape, (3,))
        self.assertGreater(s[0], 0.0)
        self.assertLess(s[1], 0.0)
        self.assertAlmostEqual(float(s[2]), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
