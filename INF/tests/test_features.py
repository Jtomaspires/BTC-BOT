"""
Paridade de ``build_features`` / scalers com a lógica de
``CNN/CNN_BTC/fit_scalers_bootstrap.py`` num slice fixo (``split_info.json``).

Correr a partir da raiz do repositório::

    python -m unittest discover -s INF/tests -p "test_*.py" -v
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler

_WF = Path(__file__).resolve().parent.parent
_REPO = _WF.parent
if str(_WF) not in sys.path:
    sys.path.insert(0, str(_WF))

import features as F


def _raw_feature_arrays(opens, highs, lows, closes, volumes):
    """Mesmas expressões que ``fit_scalers_bootstrap.py``."""
    feature1 = (closes - opens) / opens
    feature2 = (highs - opens) / opens
    feature3 = (highs - closes) / closes
    feature4 = (lows - opens) / opens
    feature5 = (lows - closes) / closes
    feature6 = (highs - lows) / opens
    feature7 = (highs - lows) / closes
    feature8 = volumes
    return [
        feature1,
        feature2,
        feature3,
        feature4,
        feature5,
        feature6,
        feature7,
        feature8,
    ]


def _fit_scalers_bootstrap_style(opens, highs, lows, closes, volumes):
    feats = _raw_feature_arrays(opens, highs, lows, closes, volumes)
    scalers = [MaxAbsScaler() for _ in range(8)]
    for i in range(8):
        scalers[i].fit(feats[i].reshape(-1, 1))
    return scalers


def _transform_with_scalers(opens, highs, lows, closes, volumes, scalers):
    feats = _raw_feature_arrays(opens, highs, lows, closes, volumes)
    scaled_fts = []
    for i in range(8):
        scaled_fts.append(scalers[i].transform(feats[i].reshape(-1, 1)).flatten())
    return np.stack(scaled_fts, axis=-1)


class TestBuildFeaturesParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        split_path = _REPO / "CNN" / "CNN_BTC" / "artifacts" / "split_info.json"
        if not split_path.is_file():
            raise unittest.SkipTest(f"Sem split_info: {split_path}")
        with open(split_path, "r", encoding="utf-8") as f:
            sp = json.load(f)
        cls.train_start = int(sp.get("train_start", 0))
        cls.train_end = int(sp["train_end"])

        for rel in (
            Path("CNN") / "CNN_BTC" / "data" / "BTCUSDT-1h-data.csv",
            Path("CNN") / "data" / "raw" / "BTCUSDT-1h-data.csv",
        ):
            p = _REPO / rel
            if p.is_file():
                cls.csv_path = p
                break
        else:
            raise unittest.SkipTest("CSV BTCUSDT não encontrado nas paths conhecidas")

        cls.df = pd.read_csv(cls.csv_path)

    def test_num_features_constant(self):
        self.assertEqual(F.NUM_FEATURES, 8)

    def test_shape_full_series_fit(self):
        sub = self.df.iloc[:1000]
        o = np.asarray(sub["open"].values)
        h = np.asarray(sub["high"].values)
        low = np.asarray(sub["low"].values)
        c = np.asarray(sub["close"].values)
        v = np.asarray(sub["volume"].values)
        scaled, n_feat, _ = F.build_features(o, h, low, c, v, train_scalers=None)
        self.assertEqual(n_feat, 8)
        self.assertEqual(scaled.shape, (1000, 8))

    def test_parity_fit_on_train_slice(self):
        """Fit no mesmo intervalo que ``fit_scalers_bootstrap.py``."""
        sub = self.df.iloc[self.train_start : self.train_end]
        o = np.asarray(sub["open"].values)
        h = np.asarray(sub["high"].values)
        low = np.asarray(sub["low"].values)
        c = np.asarray(sub["close"].values)
        v = np.asarray(sub["volume"].values)

        ref_scalers = _fit_scalers_bootstrap_style(o, h, low, c, v)
        ref_scaled = _transform_with_scalers(o, h, low, c, v, ref_scalers)

        scaled, _, out_scalers = F.build_features(o, h, low, c, v, train_scalers=None)

        self.assertTrue(np.allclose(scaled, ref_scaled, rtol=0, atol=1e-12))
        for i in range(8):
            self.assertTrue(
                np.allclose(
                    out_scalers[i].scale_,
                    ref_scalers[i].scale_,
                    rtol=0,
                    atol=1e-12,
                ),
                msg=f"scaler channel {i}",
            )

    def test_transform_only_matches_bootstrap_on_holdout(self):
        """Scalers fit no treino; transform na fatia seguinte (como no notebook)."""
        tr = self.df.iloc[self.train_start : self.train_end]
        ho = self.df.iloc[self.train_end : self.train_end + 2000]

        def cols(d):
            return (
                np.asarray(d["open"].values),
                np.asarray(d["high"].values),
                np.asarray(d["low"].values),
                np.asarray(d["close"].values),
                np.asarray(d["volume"].values),
            )

        o0, h0, l0, c0, v0 = cols(tr)
        ref_scalers = _fit_scalers_bootstrap_style(o0, h0, l0, c0, v0)

        o1, h1, l1, c1, v1 = cols(ho)
        ref_hold = _transform_with_scalers(o1, h1, l1, c1, v1, ref_scalers)

        _, _, fitted = F.build_features(o0, h0, l0, c0, v0, train_scalers=None)
        scaled_hold, _, _ = F.build_features(o1, h1, l1, c1, v1, train_scalers=fitted)

        self.assertTrue(np.allclose(scaled_hold, ref_hold, rtol=0, atol=1e-12))


class TestMakeScalers(unittest.TestCase):
    def test_len(self):
        self.assertEqual(len(F.make_scalers(8)), 8)
        self.assertEqual(len(F.make_scalers()), F.NUM_FEATURES)


if __name__ == "__main__":
    unittest.main()
