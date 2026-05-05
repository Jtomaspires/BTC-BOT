from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Iterable, Literal

_bootstrap_path = Path(__file__).resolve().parent.parent / "bootstrap_sys_path.py"
_spec = importlib.util.spec_from_file_location("_nn_dashboard_bootstrap", _bootstrap_path)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(importlib.util.module_from_spec(_spec))

import numpy as np
import pandas as pd

from INF.metrics import max_drawdown_info, sharpe_ratio

from dashboard.utils.loader import resolve_signals_path
from dashboard.utils.regime import compute_atr_ratio
from dashboard.utils.trades import replay_signals_full

SignalSplit = Literal["val", "test", "primary"]


def _load_signals_df(window_dir: Path, split: SignalSplit) -> pd.DataFrame | None:
    p, _ = resolve_signals_path(window_dir, split)  # type: ignore[arg-type]
    if not p.exists():
        return None
    return pd.read_csv(p)


def compute_robustness_surface(
    exp_dir: Path,
    *,
    split: SignalSplit,
    thr_values: Iterable[float],
    sl_values: Iterable[float],
    tp_values: Iterable[float],
    trail_values: Iterable[float],
    taker_fee: float = 0.00055,
    position_notional: float = 1000.0,
    periods_per_year: float = 365.0 * 24.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Por janela em ``exp_dir/window_*``: grelha completa (thr×sl×tp×trail), rank de sharpe
    intra-janela, métricas agregadas por combinação.

    Devolve (detail_df, agg_df).
    """
    thr_l = [float(x) for x in thr_values]
    sl_l = [float(x) for x in sl_values]
    tp_l = [float(x) for x in tp_values]
    tr_l = [float(x) for x in trail_values]

    win_dirs = sorted([p for p in Path(exp_dir).glob("window_*") if p.is_dir()], key=lambda x: x.name)
    rows: list[dict] = []

    for wd in win_dirs:
        try:
            wid = int(wd.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        sig = _load_signals_df(wd, split)
        if sig is None or sig.empty:
            continue

        ohlc = sig[["open", "high", "low", "close"]].astype(float)
        atr_r = compute_atr_ratio(ohlc, fast=14, slow=200)
        atr_mean = float(np.nanmean(atr_r.to_numpy()))

        grid_sharpes: list[float] = []
        for thr in thr_l:
            for sl in sl_l:
                for tp in tp_l:
                    for trail in tr_l:
                        rep = replay_signals_full(
                            sig,
                            sl_points=sl,
                            tp_points=tp,
                            signal_threshold=thr,
                            trailing_stop_points=trail,
                            taker_fee=taker_fee,
                            position_notional=position_notional,
                            window_id=wid,
                            split=str(split),
                        )
                        eq = rep.equities
                        fin = float(eq[-1]) if len(eq) else float(position_notional)
                        roi = (fin / float(position_notional) - 1.0) * 100.0
                        sh = float(sharpe_ratio(eq, periods_per_year=periods_per_year))
                        dd = float(max_drawdown_info(eq).get("max_drawdown", 0.0))
                        n_tr = int(len(rep.trades))

                        rows.append(
                            {
                                "window_id": wid,
                                "thr": thr,
                                "sl": sl,
                                "tp": tp,
                                "trail": trail,
                                "sharpe": sh,
                                "roi_pct": roi,
                                "max_dd": dd,
                                "n_trades": n_tr,
                                "atr_mean": atr_mean,
                            }
                        )
                        grid_sharpes.append(sh)

        # rank_pct intra-window (over rows just appended for this window)
        n_g = len(grid_sharpes)
        if n_g == 0:
            continue
        idx_start = len(rows) - n_g
        arr_s = np.asarray(grid_sharpes, dtype=np.float64)
        ranks = np.argsort(np.argsort(arr_s))  # 0..n-1 order
        rank_pct = ranks / max(1, (n_g - 1)) if n_g > 1 else np.ones(n_g)
        for k in range(n_g):
            rows[idx_start + k]["rank_pct_sharpe"] = float(rank_pct[k])

    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()

    gcols = ["thr", "sl", "tp", "trail"]
    agg = (
        detail.groupby(gcols, as_index=False)
        .agg(
            avg_sharpe=("sharpe", "mean"),
            avg_roi=("roi_pct", "mean"),
            avg_dd=("max_dd", "mean"),
            avg_n_trades=("n_trades", "mean"),
        )
    )
    pos = detail.groupby(gcols)["sharpe"].apply(lambda s: float(np.mean(s > 0.0))).reset_index(name="frac_positive")

    tq = (
        detail.groupby(gcols)["rank_pct_sharpe"]
        .apply(lambda s: float(np.mean(s >= 0.75)))
        .reset_index(name="pct_top_quartile")
    )
    agg = agg.merge(pos, on=gcols).merge(tq, on=gcols)
    agg["pct_positive"] = (agg["frac_positive"] * 100.0).round(2)
    agg = agg.drop(columns=["frac_positive"])
    agg["robustness"] = agg["avg_sharpe"] * agg["pct_top_quartile"]
    agg = agg.sort_values("robustness", ascending=False).reset_index(drop=True)
    return detail, agg


def suggested_atr_filter_threshold(
    detail: pd.DataFrame,
    *,
    top_key: tuple[float, float, float, float],
) -> float | None:
    """
    Para a combinação top, procura o maior atr_mean entre janelas com sharpe < 0;
    sugere evitar trading acima desse nivel (heurística do plano).
    """
    thr, sl, tp, tr = top_key
    sub = detail[
        (detail["thr"] == thr)
        & (detail["sl"] == sl)
        & (detail["tp"] == tp)
        & (detail["trail"] == tr)
    ]
    if sub.empty:
        return None
    bad = sub[sub["sharpe"] < 0.0]
    if bad.empty:
        return None
    return float(bad["atr_mean"].max())
