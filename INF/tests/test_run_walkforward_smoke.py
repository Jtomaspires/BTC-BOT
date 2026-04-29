from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_WF = Path(__file__).resolve().parent.parent
if str(_WF) not in sys.path:
    sys.path.insert(0, str(_WF))

import run_walkforward as RW


class TestRunWalkforwardSmoke(unittest.TestCase):
    def test_smoke_one_window_outputs_summary_and_checkpoint(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML não instalado no ambiente de teste")

        rng = np.random.default_rng(1)
        n = 180
        timestamps = np.arange(n, dtype=np.int64)
        base = 100 + np.cumsum(rng.normal(0, 0.2, size=n))
        opens = base
        highs = base + np.abs(rng.normal(0.4, 0.1, size=n))
        lows = base - np.abs(rng.normal(0.4, 0.1, size=n))
        closes = base + rng.normal(0, 0.1, size=n)
        volumes = np.abs(rng.normal(1000, 50, size=n))
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
            }
        )

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            csv_path = td_path / "sample.csv"
            df.to_csv(csv_path, index=False)

            cfg = {
                "data": {"csv_path": str(csv_path), "pair": "TEST", "timeframe": "1h"},
                "preprocess": {"seq_len": 8},
                "walkforward": {
                    "train_size": 90,
                    "val_size": 45,
                    "step_size": 45,
                    "anchor": 0,
                    "min_train_rows": None,
                    "test_size": None,
                    "max_windows": 1,
                },
                "training": {
                    "epochs": 2,
                    "batch_size": 16,
                    "learning_rate": 0.01,
                    "seed": 0,
                    "device": "cpu",
                    "checkpoint_metric": "val_loss",
                    "early_stopping": {"enabled": False, "patience": 10, "min_delta": 0.0},
                },
                "model": {"architecture": "conv1d", "warm_start_checkpoint": None},
                "backtest": {
                    "taker_fee": 0.00055,
                    "position_notional": 1000.0,
                    "signal_threshold": 0.0007,
                    "sl_points": [100.0],
                    "tp_points": [200.0],
                    "sl_tp_grid": None,
                    "grid_select_metric": "sharpe",
                },
                "outputs": {"output_dir": str(td_path / "outputs")},
            }

            import yaml

            cfg_path = td_path / "config.yaml"
            with cfg_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)

            summary_df, run_dir = RW.run_walkforward(cfg_path)
            self.assertFalse(summary_df.empty)
            self.assertTrue((run_dir / "summary_all_windows.csv").exists())
            self.assertTrue(
                (run_dir / "window_000" / "checkpoints" / "conv1d" / "best_val_loss.pt").exists()
            )

    def test_smoke_val_equity_topk_and_plot(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML não instalado no ambiente de teste")

        rng = np.random.default_rng(11)
        n = 220
        timestamps = np.arange(n, dtype=np.int64)
        base = 100 + np.linspace(0.0, 4.0, n) + np.cumsum(rng.normal(0, 0.15, size=n))
        opens = base
        highs = base + np.abs(rng.normal(0.4, 0.1, size=n))
        lows = base - np.abs(rng.normal(0.4, 0.1, size=n))
        closes = base + rng.normal(0, 0.1, size=n)
        volumes = np.abs(rng.normal(1000, 50, size=n))
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
            }
        )

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            csv_path = td_path / "sample.csv"
            df.to_csv(csv_path, index=False)

            cfg = {
                "data": {"csv_path": str(csv_path), "pair": "TEST", "timeframe": "1h"},
                "preprocess": {"seq_len": 8},
                "walkforward": {
                    "train_size": 120,
                    "val_size": 50,
                    "step_size": 50,
                    "anchor": 0,
                    "min_train_rows": None,
                    "test_size": None,
                    "max_windows": 1,
                },
                "training": {
                    "epochs": 3,
                    "batch_size": 16,
                    "learning_rate": 0.01,
                    "seed": 0,
                    "device": "cpu",
                    "checkpoint_metric": "val_equity",
                    "top_k": 2,
                    "checkpoint_min_equity": 0.0,
                    "early_stopping": {"enabled": False, "patience": 10, "min_delta": 0.0},
                },
                "model": {"architecture": "conv1d", "warm_start_checkpoint": None},
                "backtest": {
                    "taker_fee": 0.00055,
                    "position_notional": 1000.0,
                    "signal_threshold": 0.0007,
                    "sl_points": [100.0],
                    "tp_points": [200.0],
                    "sl_tp_grid": None,
                    "grid_select_metric": "sharpe",
                },
                "outputs": {"output_dir": str(td_path / "outputs")},
            }

            import yaml

            cfg_path = td_path / "config.yaml"
            with cfg_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)

            summary_df, run_dir = RW.run_walkforward(cfg_path)
            self.assertFalse(summary_df.empty)
            self.assertIn("num_checkpoints", summary_df.columns)
            self.assertIn("architectures", summary_df.columns)
            # Só top-K (=2) ficheiros .pt em disco.
            ckpt_dir = run_dir / "window_000" / "checkpoints" / "conv1d"
            pt_files = sorted(ckpt_dir.glob("*.pt"))
            self.assertLessEqual(len(pt_files), 2)
            self.assertGreaterEqual(len(pt_files), 1)
            for p in pt_files:
                self.assertTrue(p.name.startswith("eq_") and "_ep_" in p.name)
            # Plot buy & hold deve existir.
            png_files = list((run_dir / "window_000").glob("equity_best_*.png"))
            self.assertEqual(len(png_files), 1)
            self.assertGreater(png_files[0].stat().st_size, 0)

    def test_smoke_multi_architectures_checkpoints_per_architecture(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML não instalado no ambiente de teste")

        rng = np.random.default_rng(21)
        n = 220
        timestamps = np.arange(n, dtype=np.int64)
        base = 100 + np.cumsum(rng.normal(0, 0.15, size=n))
        opens = base
        highs = base + np.abs(rng.normal(0.4, 0.1, size=n))
        lows = base - np.abs(rng.normal(0.4, 0.1, size=n))
        closes = base + rng.normal(0, 0.1, size=n)
        volumes = np.abs(rng.normal(1000, 50, size=n))
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
            }
        )

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            csv_path = td_path / "sample.csv"
            df.to_csv(csv_path, index=False)

            cfg = {
                "data": {"csv_path": str(csv_path), "pair": "TEST", "timeframe": "1h"},
                "preprocess": {"seq_len": 8},
                "walkforward": {
                    "train_size": 120,
                    "val_size": 50,
                    "step_size": 50,
                    "anchor": 0,
                    "min_train_rows": None,
                    "test_size": None,
                    "max_windows": 1,
                },
                "training": {
                    "epochs": 2,
                    "batch_size": 16,
                    "learning_rate": 0.01,
                    "seed": 0,
                    "device": "cpu",
                    "checkpoint_metric": "val_equity",
                    "top_k": 1,
                    "checkpoint_min_equity": 0.0,
                    "early_stopping": {"enabled": False, "patience": 10, "min_delta": 0.0},
                },
                "model": {"architectures": ["conv1d", "lstm", "hybrid"]},
                "backtest": {
                    "taker_fee": 0.00055,
                    "position_notional": 1000.0,
                    "signal_threshold": 0.0007,
                    "sl_points": [100.0],
                    "tp_points": [200.0],
                    "sl_tp_grid": None,
                    "grid_select_metric": "sharpe",
                },
                "outputs": {"output_dir": str(td_path / "outputs")},
            }

            import yaml

            cfg_path = td_path / "config.yaml"
            with cfg_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)

            summary_df, run_dir = RW.run_walkforward(cfg_path)
            self.assertFalse(summary_df.empty)
            self.assertEqual(summary_df["architectures"].iloc[0], "conv1d,lstm,hybrid")
            self.assertIn("num_ckpts_conv1d", summary_df.columns)
            self.assertIn("num_ckpts_lstm", summary_df.columns)
            self.assertIn("num_ckpts_hybrid", summary_df.columns)
            for arch in ["conv1d", "lstm", "hybrid"]:
                arch_dir = run_dir / "window_000" / "checkpoints" / arch
                self.assertTrue(arch_dir.exists())
                self.assertLessEqual(len(list(arch_dir.glob("*.pt"))), 1)

    def test_recent_rows_adds_train_context_and_preserves_windows(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML não instalado no ambiente de teste")

        rng = np.random.default_rng(31)
        n = 60
        timestamps = np.arange(n, dtype=np.int64)
        base = 100 + np.cumsum(rng.normal(0, 0.2, size=n))
        opens = base
        highs = base + np.abs(rng.normal(0.4, 0.1, size=n))
        lows = base - np.abs(rng.normal(0.4, 0.1, size=n))
        closes = base + rng.normal(0, 0.1, size=n)
        volumes = np.abs(rng.normal(1000, 50, size=n))
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
            }
        )

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            csv_path = td_path / "sample.csv"
            df.to_csv(csv_path, index=False)

            cfg = {
                "data": {
                    "csv_path": str(csv_path),
                    "pair": "TEST",
                    "timeframe": "1h",
                    "recent_rows": 20,
                },
                "preprocess": {"seq_len": 4},
                "walkforward": {
                    "train_size": 20,
                    "val_size": 10,
                    "step_size": 10,
                    "anchor": 0,
                    "min_train_rows": None,
                    "test_size": None,
                    "max_windows": None,
                },
                "training": {
                    "epochs": 1,
                    "batch_size": 8,
                    "learning_rate": 0.01,
                    "seed": 0,
                    "device": "cpu",
                    "checkpoint_metric": "val_loss",
                    "early_stopping": {"enabled": False, "patience": 10, "min_delta": 0.0},
                },
                "model": {"architecture": "conv1d", "warm_start_checkpoint": None},
                "backtest": {
                    "taker_fee": 0.00055,
                    "position_notional": 1000.0,
                    "signal_threshold": 0.0007,
                    "sl_points": [100.0],
                    "tp_points": [200.0],
                    "sl_tp_grid": None,
                    "grid_select_metric": "sharpe",
                },
                "outputs": {"output_dir": str(td_path / "outputs")},
            }

            import yaml

            cfg_path = td_path / "config.yaml"
            with cfg_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)

            summary_df, _ = RW.run_walkforward(cfg_path)
            # Esperado: keep_rows = recent_rows + train_size = 40
            # n_windows = floor((40 - (20 + 10)) / 10) + 1 = 2
            self.assertEqual(sorted(summary_df["window_id"].unique().tolist()), [0, 1])

    def test_multi_experiments_outputs_and_registry(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML não instalado no ambiente de teste")

        rng = np.random.default_rng(41)
        n = 220
        timestamps = np.arange(n, dtype=np.int64)
        base = 100 + np.cumsum(rng.normal(0, 0.15, size=n))
        opens = base
        highs = base + np.abs(rng.normal(0.4, 0.1, size=n))
        lows = base - np.abs(rng.normal(0.4, 0.1, size=n))
        closes = base + rng.normal(0, 0.1, size=n)
        volumes = np.abs(rng.normal(1000, 50, size=n))
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
            }
        )

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            csv_path = td_path / "sample.csv"
            df.to_csv(csv_path, index=False)

            cfg = {
                "config_name": "suite_a",
                "data": {"csv_path": str(csv_path), "pair": "TEST", "timeframe": "1h"},
                "preprocess": {"seq_len": 8},
                "walkforward": {
                    "train_size": 120,
                    "val_size": 50,
                    "step_size": 50,
                    "anchor": 0,
                    "min_train_rows": None,
                    "test_size": 50,
                    "max_windows": 1,
                },
                "training": {
                    "epochs": 2,
                    "batch_size": 16,
                    "learning_rate": 0.01,
                    "seed": 0,
                    "device": "cpu",
                    "checkpoint_metric": "val_loss",
                    "early_stopping": {"enabled": False, "patience": 10, "min_delta": 0.0},
                },
                "model": {"architecture": "conv1d", "warm_start_checkpoint": None},
                "backtest": {
                    "taker_fee": 0.00055,
                    "position_notional": 1000.0,
                    "signal_threshold": 0.0007,
                    "sl_points": [100.0],
                    "tp_points": [200.0],
                    "sl_tp_grid": None,
                    "grid_select_metric": "sharpe",
                },
                "experiments": [
                    {"name": "base"},
                    {"name": "short_train", "training": {"epochs": 1}},
                ],
                "outputs": {"output_dir": str(td_path / "outputs")},
            }

            import yaml

            cfg_path = td_path / "config.yaml"
            with cfg_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)

            summary_df, run_dir = RW.run_walkforward(cfg_path)
            self.assertFalse(summary_df.empty)
            self.assertIn("experiment_name", summary_df.columns)
            self.assertIn("config_name", summary_df.columns)
            self.assertEqual(sorted(summary_df["experiment_name"].unique().tolist()), ["base", "short_train"])

            base_dir = run_dir / "experiments" / "base"
            short_dir = run_dir / "experiments" / "short_train"
            self.assertTrue((base_dir / "summary_all_windows.csv").exists())
            self.assertTrue((short_dir / "summary_all_windows.csv").exists())
            self.assertTrue((base_dir / "run_summary.csv").exists())
            self.assertTrue((short_dir / "run_summary.csv").exists())

            registry = pd.read_csv(td_path / "outputs" / "runs_summary.csv")
            self.assertEqual(len(registry), 2)
            self.assertEqual(sorted(registry["experiment_name"].tolist()), ["base", "short_train"])
            self.assertTrue((td_path / "outputs" / "comparison_by_config.csv").exists())

    def test_smoke_triple_barrier_cross_entropy(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML não instalado no ambiente de teste")

        rng = np.random.default_rng(51)
        n = 260
        timestamps = np.arange(n, dtype=np.int64)
        base = 100 + np.cumsum(rng.normal(0, 0.25, size=n))
        opens = base
        highs = base + np.abs(rng.normal(0.6, 0.15, size=n))
        lows = base - np.abs(rng.normal(0.6, 0.15, size=n))
        closes = base + rng.normal(0, 0.12, size=n)
        volumes = np.abs(rng.normal(1000, 50, size=n))
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
            }
        )

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            csv_path = td_path / "sample.csv"
            df.to_csv(csv_path, index=False)

            cfg = {
                "data": {"csv_path": str(csv_path), "pair": "TEST", "timeframe": "1h"},
                "preprocess": {"seq_len": 8},
                "target": {"type": "triple_barrier", "tp_pct": 0.01, "sl_pct": 0.01, "horizon": 6},
                "walkforward": {
                    "train_size": 150,
                    "val_size": 60,
                    "step_size": 60,
                    "anchor": 0,
                    "min_train_rows": None,
                    "test_size": None,
                    "max_windows": 1,
                },
                "training": {
                    "epochs": 2,
                    "batch_size": 16,
                    "learning_rate": 0.01,
                    "seed": 0,
                    "device": "cpu",
                    "loss": "cross_entropy",
                    "class_balance": "weighted",
                    "checkpoint_metric": "val_equity",
                    "top_k": 1,
                    "checkpoint_min_equity": 0.0,
                    "early_stopping": {"enabled": False, "patience": 10, "min_delta": 0.0},
                },
                "model": {"architecture": "conv1d", "output_dim": 3, "warm_start_checkpoint": None},
                "backtest": {
                    "taker_fee": 0.00055,
                    "position_notional": 1000.0,
                    "signal_threshold": 0.0007,
                    "sl_points": [100.0],
                    "tp_points": [200.0],
                    "sl_tp_grid": None,
                    "grid_select_metric": "sharpe",
                },
                "outputs": {"output_dir": str(td_path / "outputs")},
            }

            import yaml

            cfg_path = td_path / "config.yaml"
            with cfg_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)

            summary_df, run_dir = RW.run_walkforward(cfg_path)
            self.assertFalse(summary_df.empty)
            self.assertIn("bal_method_conv1d", summary_df.columns)
            self.assertEqual(str(summary_df["bal_method_conv1d"].iloc[0]), "weighted")
            ckpt_dir = run_dir / "window_000" / "checkpoints" / "conv1d"
            pt_files = sorted(ckpt_dir.glob("*.pt"))
            self.assertGreaterEqual(len(pt_files), 1)
            self.assertLessEqual(len(pt_files), 1)


if __name__ == "__main__":
    unittest.main()
