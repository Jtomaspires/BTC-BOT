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

import reporter as RP


class TestReporter(unittest.TestCase):
    def test_save_metrics_csv_roundtrip(self):
        df = pd.DataFrame(
            [
                {
                    "window_id": 0,
                    "sl": 150.0,
                    "tp": 300.0,
                    "sharpe": 1.1,
                    "max_drawdown": 0.2,
                    "final_equity": 1050.0,
                    "num_trades": 10,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            path = RP.save_metrics_csv(df, Path(td) / "window_000" / "metrics.csv")
            self.assertTrue(path.exists())
            got = pd.read_csv(path)
            self.assertEqual(list(got.columns), list(df.columns))
            self.assertEqual(len(got), 1)

    def test_plot_equity_curve_creates_png(self):
        eq = np.asarray([1000.0, 1002.0, 1001.0, 1010.0], dtype=np.float64)
        with tempfile.TemporaryDirectory() as td:
            out = RP.plot_equity_curve(eq, Path(td) / "eq.png", title="teste")
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 0)

    def test_plot_equity_with_buyhold_creates_png(self):
        eq = np.asarray(
            [1000.0, 1020.0, 1015.0, 980.0, 995.0, 1030.0, 1010.0, 1050.0], dtype=np.float64
        )
        opens = np.asarray(
            [100.0, 101.0, 102.0, 101.5, 101.8, 103.0, 102.5, 104.0], dtype=np.float64
        )
        max_dd_info = {
            "max_drawdown": 0.039,
            "peak_index": 1,
            "trough_index": 3,
            "peak_value": 1020.0,
            "trough_value": 980.0,
        }
        with tempfile.TemporaryDirectory() as td:
            out = RP.plot_equity_with_buyhold(
                equities=eq,
                opens=opens,
                sharpe=0.1234,
                max_dd_info=max_dd_info,
                path=Path(td) / "eq_bh.png",
                position_notional=1000.0,
            )
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 0)

    def test_save_run_summary_row_creates_one_line_files(self):
        summary_df = pd.DataFrame(
            [
                {
                    "window_id": 0,
                    "eval_split": "test",
                    "final_equity": 1100.0,
                    "sharpe": 1.2,
                    "max_drawdown": 0.1,
                },
                {
                    "window_id": 1,
                    "eval_split": "test",
                    "final_equity": 900.0,
                    "sharpe": -0.2,
                    "max_drawdown": 0.2,
                },
            ]
        )
        cfg = {
            "data": {"pair": "ETHUSDT", "timeframe": "1h", "csv_path": "CNN/data/raw/ETHUSDT-1h-data.csv"},
            "walkforward": {"train_size": 5000, "val_size": 700, "test_size": 700, "step_size": 700},
            "training": {"epochs": 100, "checkpoint_metric": "val_equity"},
            "model": {"architectures": ["conv1d", "lstm", "hybrid"]},
            "backtest": {"position_notional": 1000.0, "grid_select_metric": "sharpe", "signal_threshold": 0.006},
        }
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            run_dir = base / "outputs" / "run_foo"
            out_root = base / "outputs"
            run_file, registry_file = RP.save_run_summary_row(
                summary_df,
                run_id="run_foo",
                cfg=cfg,
                features_used=["body_over_open", "volume"],
                run_dir=run_dir,
                output_root=out_root,
                config_name="eth_1h_baseline",
                experiment_name="base",
                batch_id="run_foo",
            )
            self.assertTrue(run_file.exists())
            self.assertTrue(registry_file.exists())
            self.assertTrue((out_root / "comparison_by_config.csv").exists())

            run_df = pd.read_csv(run_file)
            self.assertEqual(len(run_df), 1)
            self.assertEqual(run_df.loc[0, "run_id"], "run_foo")
            self.assertEqual(run_df.loc[0, "config_name"], "eth_1h_baseline")
            self.assertEqual(run_df.loc[0, "experiment_name"], "base")
            self.assertAlmostEqual(float(run_df.loc[0, "avg_return"]), 0.0, places=9)
            self.assertIn("000_roi", run_df.columns)
            self.assertIn("000_dd", run_df.columns)
            self.assertIn("001_roi", run_df.columns)
            self.assertIn("001_dd", run_df.columns)
            self.assertLess(run_df.columns.get_loc("000_roi"), run_df.columns.get_loc("000_dd"))
            self.assertLess(run_df.columns.get_loc("000_dd"), run_df.columns.get_loc("001_roi"))
            self.assertEqual(run_df.columns[-1], "features")
            self.assertEqual(str(run_df.loc[0, "features"]), "body_over_open,volume")

    def test_registry_dedup_is_by_run_and_experiment(self):
        summary_df = pd.DataFrame(
            [{"window_id": 0, "eval_split": "test", "final_equity": 1100.0, "sharpe": 1.0, "max_drawdown": 0.1}]
        )
        cfg = {
            "data": {"pair": "ETHUSDT", "timeframe": "1h", "csv_path": "x.csv"},
            "walkforward": {"train_size": 1, "val_size": 1, "test_size": 1, "step_size": 1},
            "training": {"epochs": 1, "checkpoint_metric": "val_equity"},
            "model": {"architectures": ["conv1d"]},
            "backtest": {"position_notional": 1000.0, "grid_select_metric": "sharpe", "signal_threshold": 0.0},
        }
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            out_root = base / "outputs"
            RP.save_run_summary_row(
                summary_df,
                run_id="run_1",
                cfg=cfg,
                features_used=["f1"],
                run_dir=out_root / "run_1" / "exp_a",
                output_root=out_root,
                config_name="cfg",
                experiment_name="exp_a",
                batch_id="run_1",
            )
            RP.save_run_summary_row(
                summary_df,
                run_id="run_1",
                cfg=cfg,
                features_used=["f1"],
                run_dir=out_root / "run_1" / "exp_b",
                output_root=out_root,
                config_name="cfg",
                experiment_name="exp_b",
                batch_id="run_1",
            )
            # mesmo run+experiment substitui (não duplica)
            RP.save_run_summary_row(
                summary_df,
                run_id="run_1",
                cfg=cfg,
                features_used=["f1"],
                run_dir=out_root / "run_1" / "exp_a",
                output_root=out_root,
                config_name="cfg",
                experiment_name="exp_a",
                batch_id="run_1",
            )
            reg = pd.read_csv(out_root / "runs_summary.csv")
            self.assertEqual(len(reg), 2)
            self.assertEqual(sorted(reg["experiment_name"].tolist()), ["exp_a", "exp_b"])


if __name__ == "__main__":
    unittest.main()
