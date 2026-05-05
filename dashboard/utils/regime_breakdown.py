from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from dashboard.utils.regime import compute_atr_arrays, entry_mask_from_hardstop
from dashboard.utils.trades import replay_signals_dynamic

SignalSplit = Literal["val", "test", "primary", "cmp"]


def compute_atr_dynamic_trades_for_window(
    signals_df: pd.DataFrame,
    *,
    atr_period: int,
    slow_period: int = 200,
    hardstop: float | None,
    sl_mults: tuple[float, ...],
    tp_mults: tuple[float, ...],
    trail_mults: tuple[float, ...],
    signal_threshold: float,
    taker_fee: float,
    position_notional: float,
    ohlcv_context_df: pd.DataFrame | None,
    window_id: int,
    split: SignalSplit,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    arr = compute_atr_arrays(
        signals_df,
        atr_period=int(atr_period),
        slow_period=int(slow_period),
        ohlcv_context_df=ohlcv_context_df,
    )
    atr_fast_dec = np.asarray(arr["atr_fast_dec"], dtype=np.float64)
    atr_ratio_dec = np.asarray(arr["atr_ratio_dec"], dtype=np.float64)

    warmup = int(atr_period)
    mask = entry_mask_from_hardstop(atr_ratio_dec, hardstop=hardstop, warmup=warmup, atr_fast=atr_fast_dec)

    frames: list[pd.DataFrame] = []
    for sm in sl_mults:
        for tm in tp_mults:
            for trl in trail_mults:
                sl_arr = atr_fast_dec * float(sm)
                tp_arr = atr_fast_dec * float(tm)
                sl_safe = np.where(np.isfinite(sl_arr) & (sl_arr > 0), sl_arr, 1e-9)
                tp_safe = np.where(np.isfinite(tp_arr) & (tp_arr > 0), tp_arr, 1e-9)
                trail_arr: np.ndarray | None = None
                if float(trl) > 0.0:
                    raw_trail = atr_fast_dec * float(trl)
                    trail_arr = np.where(np.isfinite(raw_trail) & (raw_trail > 0), raw_trail, 0.0)

                rep = replay_signals_dynamic(
                    signals_df,
                    sl_points_per_bar=sl_safe,
                    tp_points_per_bar=tp_safe,
                    signal_threshold=float(signal_threshold),
                    trailing_stop_per_bar=trail_arr,
                    taker_fee=float(taker_fee),
                    position_notional=float(position_notional),
                    entry_mask=mask,
                    window_id=int(window_id),
                    split=str(split),
                )
                tdf = rep.trades.copy()
                if tdf.empty:
                    continue
                tdf["sl_mult"] = float(sm)
                tdf["tp_mult"] = float(tm)
                tdf["trail_mult"] = float(trl)
                tdf["window_id"] = int(window_id)
                tdf["split"] = str(split)
                frames.append(tdf)

    if not frames:
        return pd.DataFrame(), atr_ratio_dec, atr_fast_dec
    return pd.concat(frames, ignore_index=True, sort=False), atr_ratio_dec, atr_fast_dec


def tag_regime_bins(
    trades_df: pd.DataFrame,
    atr_ratio_dec_by_window: dict[int, np.ndarray],
    atr_fast_dec_by_window: dict[int, np.ndarray],
    *,
    ratio_bin_w: float,
    fast_bin_w: float,
) -> pd.DataFrame:
    if trades_df.empty:
        return trades_df.copy()
    out = trades_df.copy()
    ratio_vals = np.full(len(out), np.nan, dtype=np.float64)
    fast_vals = np.full(len(out), np.nan, dtype=np.float64)

    for wid, idx in out.groupby("window_id").groups.items():
        pos = np.asarray(list(idx), dtype=np.int64)
        entry_idx = out.iloc[pos]["entry_idx"].astype(int).to_numpy()
        ratio_arr = atr_ratio_dec_by_window.get(int(wid))
        fast_arr = atr_fast_dec_by_window.get(int(wid))
        if ratio_arr is not None:
            ok = (entry_idx >= 0) & (entry_idx < len(ratio_arr))
            ratio_vals[pos[ok]] = ratio_arr[entry_idx[ok]]
        if fast_arr is not None:
            ok = (entry_idx >= 0) & (entry_idx < len(fast_arr))
            fast_vals[pos[ok]] = fast_arr[entry_idx[ok]]

    out["atr_ratio_entry"] = ratio_vals
    out["atr_fast_entry"] = fast_vals
    out["ratio_bin"] = np.floor(ratio_vals / float(ratio_bin_w)) * float(ratio_bin_w)
    out["fast_bin"] = np.floor(fast_vals / float(fast_bin_w)) * float(fast_bin_w)
    out.loc[~np.isfinite(out["ratio_bin"].to_numpy(dtype=np.float64)), "ratio_bin"] = np.nan
    out.loc[~np.isfinite(out["fast_bin"].to_numpy(dtype=np.float64)), "fast_bin"] = np.nan
    return out


def best_combo_per_bin_per_window(
    trades_df: pd.DataFrame,
    *,
    bin_col: str,
    metric_col: str,
    min_trades: int,
    position_notional: float,
) -> pd.DataFrame:
    if trades_df.empty or bin_col not in trades_df.columns:
        return pd.DataFrame()
    df = trades_df.dropna(subset=[bin_col]).copy()
    if df.empty:
        return pd.DataFrame()

    grp = (
        df.groupby(["window_id", bin_col, "sl_mult", "tp_mult", "trail_mult"], dropna=False)
        .agg(
            n_trades=("pnl_net", "size"),
            pnl_sum=("pnl_net", "sum"),
            pnl_mean=("pnl_net", "mean"),
            pnl_std=("pnl_net", lambda s: float(np.std(s.astype(float), ddof=0))),
            win_rate=("pnl_net", lambda s: float((s.astype(float) > 0.0).mean() * 100.0)),
        )
        .reset_index()
    )
    grp["roi_pct"] = (
        (100.0 * grp["pnl_sum"].astype(float) / float(position_notional)) if position_notional else np.nan
    )
    grp["sharpe"] = np.where(
        (grp["n_trades"].astype(float) > 1.0) & (grp["pnl_std"].astype(float) > 0.0),
        (grp["pnl_mean"].astype(float) / grp["pnl_std"].astype(float)) * np.sqrt(grp["n_trades"].astype(float)),
        np.nan,
    )
    grp = grp.drop(columns=["pnl_sum", "pnl_mean", "pnl_std"])
    grp = grp[grp["n_trades"] >= int(min_trades)].copy()
    if grp.empty:
        return pd.DataFrame()
    grp = grp[np.isfinite(grp[metric_col].to_numpy(dtype=np.float64))]
    if grp.empty:
        return pd.DataFrame()

    idx = grp.groupby(["window_id", bin_col], dropna=False)[metric_col].idxmax()
    out = grp.loc[idx].copy().sort_values(["window_id", bin_col]).reset_index(drop=True)
    out["metric_name"] = str(metric_col)
    out["metric_value"] = out[metric_col].astype(float)
    return out[
        [
            "window_id",
            bin_col,
            "sl_mult",
            "tp_mult",
            "trail_mult",
            "metric_name",
            "metric_value",
            "n_trades",
            "roi_pct",
            "sharpe",
            "win_rate",
        ]
    ]


def consensus_best_combo(
    per_window_best_df: pd.DataFrame,
    *,
    bin_col: str,
) -> pd.DataFrame:
    if per_window_best_df.empty or bin_col not in per_window_best_df.columns:
        return pd.DataFrame()
    df = per_window_best_df.copy()

    votes = (
        df.groupby([bin_col, "sl_mult", "tp_mult", "trail_mult"], dropna=False)
        .agg(
            n_windows_chose=("window_id", "nunique"),
            avg_metric=("metric_value", "mean"),
            median_metric=("metric_value", "median"),
            avg_n_trades=("n_trades", "mean"),
        )
        .reset_index()
    )
    total_w = (
        df.groupby(bin_col, dropna=False)["window_id"]
        .nunique()
        .reset_index(name="total_windows")
    )
    votes = votes.merge(total_w, on=bin_col, how="left")
    votes["pct_windows"] = np.where(
        votes["total_windows"] > 0,
        100.0 * votes["n_windows_chose"].astype(float) / votes["total_windows"].astype(float),
        np.nan,
    )
    votes = votes.sort_values(
        [bin_col, "n_windows_chose", "avg_metric"],
        ascending=[True, False, False],
    )
    out = votes.groupby(bin_col, dropna=False).head(1).reset_index(drop=True)
    return out

