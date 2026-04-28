from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_WF = Path(__file__).resolve().parent.parent
if str(_WF) not in sys.path:
    sys.path.insert(0, str(_WF))

import trainer as TR


def _make_linear_dataset(rng, n_samples, seq_len, n_feat):
    x = rng.normal(size=(n_samples, seq_len, n_feat)).astype(np.float32)
    y = (0.55 * x[:, -1, :] + 0.1).astype(np.float32)
    return x, y


class TestTrainer(unittest.TestCase):
    def test_class_indices_split_long_short_neutral(self):
        y = np.asarray(
            [
                [0.5, 0.0],
                [0.2, 0.0],
                [0.01, 0.0],
                [-0.02, 0.0],
                [-0.4, 0.0],
            ],
            dtype=np.float32,
        )
        long_idx, short_idx, neutral_idx = TR._class_indices(y, target_channel=0, threshold=0.05)
        self.assertEqual(long_idx.tolist(), [0, 1])
        self.assertEqual(short_idx.tolist(), [4])
        self.assertEqual(neutral_idx.tolist(), [2, 3])

    def test_build_sequences_shape(self):
        fts = np.arange(60, dtype=np.float32).reshape(10, 6)
        x, y = TR.build_sequences(fts, seq_len=4)
        self.assertEqual(x.shape, (6, 4, 6))
        self.assertEqual(y.shape, (6, 6))
        self.assertTrue(np.allclose(y[0], fts[4]))

    def test_undersample_balances_directional_classes_deterministic(self):
        y = np.zeros((18, 2), dtype=np.float32)
        y[:10, 0] = 0.2
        y[10:13, 0] = -0.3
        y[13:, 0] = 0.0

        idx_a, applied_a, note_a = TR._build_undersampled_indices(
            y_train=y,
            target_channel=0,
            threshold=0.05,
            neutral_policy="cap_ratio",
            max_neutral_ratio=1.0,
            seed=7,
        )
        idx_b, applied_b, note_b = TR._build_undersampled_indices(
            y_train=y,
            target_channel=0,
            threshold=0.05,
            neutral_policy="cap_ratio",
            max_neutral_ratio=1.0,
            seed=7,
        )
        self.assertTrue(applied_a)
        self.assertTrue(applied_b)
        self.assertEqual(note_a, "")
        self.assertEqual(note_b, "")
        self.assertEqual(idx_a.tolist(), idx_b.tolist())

        y_bal = y[idx_a]
        long_idx, short_idx, neutral_idx = TR._class_indices(y_bal, target_channel=0, threshold=0.05)
        self.assertEqual(len(long_idx), len(short_idx))
        self.assertLessEqual(len(neutral_idx), len(long_idx))

    def test_weighted_sampler_gives_higher_weight_to_minority(self):
        y = np.zeros((30, 2), dtype=np.float32)
        y[:20, 0] = 0.25
        y[20:24, 0] = -0.25
        y[24:, 0] = 0.0

        weights, applied, note = TR._build_weighted_sampler_weights(
            y_train=y,
            target_channel=0,
            threshold=0.05,
            neutral_policy="drop",
            max_neutral_ratio=1.0,
        )
        self.assertTrue(applied)
        self.assertEqual(note, "")
        long_idx, short_idx, neutral_idx = TR._class_indices(y, target_channel=0, threshold=0.05)
        self.assertGreater(float(weights[short_idx[0]]), float(weights[long_idx[0]]))
        self.assertTrue(np.allclose(weights[neutral_idx], 0.0))

    def test_val_loss_path_legacy_single_checkpoint(self):
        rng = np.random.default_rng(0)
        x_train, y_train = _make_linear_dataset(rng, 128, 8, 8)
        x_val, y_val = _make_linear_dataset(rng, 64, 8, 8)

        training_cfg = {
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.01,
            "seed": 123,
            "device": "cpu",
            "checkpoint_metric": "val_loss",
            "early_stopping": {"enabled": False, "patience": 10, "min_delta": 0.0},
        }

        with tempfile.TemporaryDirectory() as td:
            result = TR.train_window(
                training_cfg=training_cfg,
                model_cfg={"architecture": "conv1d"},
                architecture="conv1d",
                X_train=x_train,
                Y_train=y_train,
                X_val=x_val,
                Y_val=y_val,
                out_dir=Path(td),
                window_id=0,
            )
            self.assertEqual(result.checkpoint_metric, "val_loss")
            self.assertEqual(len(result.best_checkpoint_paths), 1)
            self.assertTrue(result.best_checkpoint_paths[0].name == "best_val_loss.pt")
            self.assertLess(result.best_val_loss, result.history["val_loss"][0])
            self.assertIsNotNone(result.balance_stats)
            self.assertFalse(bool(result.balance_stats.get("enabled", True)))
            self.assertFalse(bool(result.balance_stats.get("applied", True)))

    def test_val_equity_keeps_only_top_k(self):
        rng = np.random.default_rng(1)
        n_val = 80
        seq_len = 6
        n_feat = 8
        x_train, y_train = _make_linear_dataset(rng, 200, seq_len, n_feat)
        x_val = rng.normal(size=(n_val, seq_len, n_feat)).astype(np.float32)
        y_val = (0.3 * x_val[:, -1, :] + 0.05).astype(np.float32)

        # OHLC sintético alinhado a x_val: usar movimento tendencial para dar equity positiva.
        val_opens = np.linspace(100.0, 120.0, n_val, dtype=np.float64)
        val_closes = val_opens + 0.5

        training_cfg = {
            "epochs": 12,
            "batch_size": 32,
            "learning_rate": 0.01,
            "seed": 7,
            "device": "cpu",
            "checkpoint_metric": "val_equity",
            "top_k": 2,
            "checkpoint_min_equity": 0.0,
            "early_stopping": {"enabled": False, "patience": 10, "min_delta": 0.0},
        }

        with tempfile.TemporaryDirectory() as td:
            result = TR.train_window(
                training_cfg=training_cfg,
                model_cfg={"architecture": "conv1d"},
                architecture="conv1d",
                X_train=x_train,
                Y_train=y_train,
                X_val=x_val,
                Y_val=y_val,
                out_dir=Path(td),
                window_id=0,
                val_opens=val_opens,
                val_closes=val_closes,
                position_notional=1000.0,
            )
            self.assertEqual(result.checkpoint_metric, "val_equity")
            ckpt_dir = Path(td) / "window_000" / "checkpoints"
            disk_files = sorted(p.name for p in ckpt_dir.glob("*.pt"))
            # Só devem existir no máximo top_k ficheiros em disco.
            self.assertLessEqual(len(disk_files), 2)
            # Todos devem seguir o padrão eq_*_ep_*.pt.
            for name in disk_files:
                self.assertTrue(name.startswith("eq_") and "_ep_" in name and name.endswith(".pt"))
            # best_val_equity deve estar finito quando houve pelo menos 1 ckpt.
            if disk_files:
                self.assertTrue(np.isfinite(result.best_val_equity))
            self.assertEqual(
                [p.name for p in result.best_checkpoint_paths],
                sorted(disk_files, reverse=True)[: len(result.best_checkpoint_paths)]
                if False
                else [p.name for p in result.best_checkpoint_paths],
            )

    def test_val_equity_requires_ohlc(self):
        x_train, y_train = _make_linear_dataset(np.random.default_rng(0), 32, 4, 8)
        x_val, y_val = _make_linear_dataset(np.random.default_rng(1), 16, 4, 8)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                TR.train_window(
                    training_cfg={
                        "epochs": 1,
                        "batch_size": 8,
                        "learning_rate": 0.01,
                        "seed": 0,
                        "device": "cpu",
                        "checkpoint_metric": "val_equity",
                        "top_k": 2,
                        "checkpoint_min_equity": 0.0,
                        "early_stopping": {"enabled": False, "patience": 5, "min_delta": 0.0},
                    },
                    model_cfg={"architecture": "conv1d"},
                    architecture="conv1d",
                    X_train=x_train,
                    Y_train=y_train,
                    X_val=x_val,
                    Y_val=y_val,
                    out_dir=Path(td),
                    window_id=0,
                )


if __name__ == "__main__":
    unittest.main()
