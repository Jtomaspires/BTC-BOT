from __future__ import annotations

import importlib.util
from pathlib import Path

_bootstrap_path = Path(__file__).resolve().parent.parent / "bootstrap_sys_path.py"
_spec = importlib.util.spec_from_file_location("_nn_dashboard_bootstrap", _bootstrap_path)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(importlib.util.module_from_spec(_spec))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from INF.metrics import max_drawdown_info, sharpe_ratio

from dashboard.utils.loader import (
    experiment_picker_options,
    load_config_resolved,
    load_runs_summary,
    resolve_exp_dir_like_heatmap,
    resolve_signals_path,
    run_id_select_options,
)
from dashboard.utils.parse_params import parse_float_list
from dashboard.utils.paths import REPO_ROOT
from dashboard.utils.regime import compute_atr_arrays, entry_mask_from_hardstop
from dashboard.utils.regime_breakdown import compute_atr_dynamic_trades_for_window, tag_regime_bins
from dashboard.utils.state import get_selection_defaults, set_selection
from dashboard.utils.trades import replay_signals_dynamic


st.set_page_config(page_title="ATR robustness", layout="wide")
st.title("ATR robustness — ATR multipliers across windows")
st.caption("Grid de SL/TP/TR como múltiplos de ATR, agregado por múltiplas janelas (estilo Settings Comparator).")

defaults = get_selection_defaults()
runs_df = load_runs_summary()
run_ids = run_id_select_options(runs_df)
if not run_ids:
    st.error("Sem runs.")
    st.stop()

_run_default = str(defaults.run_id).strip() if defaults.run_id else None
run_id = st.selectbox(
    "run_id (pastas em outputs; como o heatmap)",
    run_ids,
    index=(run_ids.index(_run_default) if _run_default in run_ids else 0),
)
run_id = str(run_id).strip()
run_dir = (REPO_ROOT / "INF" / "outputs" / run_id).resolve()

csv_exps = (
    runs_df.loc[runs_df["run_id"].astype(str).str.strip() == run_id, "experiment_name"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)
exp_opts = experiment_picker_options(run_dir, csv_exps)
experiment = st.selectbox(
    "experiment (pastas em experiments/ + runs_summary)",
    exp_opts,
    index=(
        exp_opts.index(str(defaults.experiment).strip())
        if defaults.experiment and str(defaults.experiment).strip() in exp_opts
        else 0
    ),
)
experiment = str(experiment).strip()
exp_dir = resolve_exp_dir_like_heatmap(run_dir, experiment)

win_dirs = sorted([p for p in exp_dir.glob("window_*") if p.is_dir()], key=lambda x: int(x.name.split("_")[1]))
win_ids_all = [int(p.name.split("_")[1]) for p in win_dirs]
if not win_ids_all:
    st.warning("Sem pastas window_* neste experimento.")
    st.stop()

split_opts = ["val", "test"]
split_ix = split_opts.index(defaults.split) if defaults.split in split_opts else 0
split = st.radio("split", split_opts, index=split_ix, horizontal=True)

sel_w = st.multiselect(
    "window_ids (máx. 24)",
    win_ids_all,
    default=win_ids_all[: min(24, len(win_ids_all))],
    max_selections=24,
)
set_selection(run_id=run_id, experiment=experiment, split=split)

cfg = load_config_resolved(run_dir) or {}
bt = (cfg.get("backtest") or {}) if isinstance(cfg, dict) else {}
data_cfg = (cfg.get("data") or {}) if isinstance(cfg, dict) else {}
fee = float(bt.get("taker_fee", 0.00055))
nom = float(bt.get("position_notional", 1000.0))
base_thr = float(bt.get("signal_threshold", 0.007))
raw_csv_path = str(data_cfg.get("csv_path", "")).strip() if isinstance(data_cfg, dict) else ""

col1, col2, col3 = st.columns(3)
with col1:
    thr = float(
        st.number_input(
            "signal_threshold (1 valor)",
            value=float(base_thr),
            min_value=0.0,
            step=0.0005,
            format="%.6f",
        )
    )
    atr_period = int(st.number_input("atr_period", value=14, min_value=5, max_value=200, step=1))
with col2:
    sl_mult_text = st.text_input("sl_mult_values", "0.5,0.7,1.0,1.2,1.5,2.0,2.5")
    tp_mult_text = st.text_input("tp_mult_values", "2.0,2.5,3.0,3.5,4.0,4.5,5.0")
with col3:
    trail_mult_text = st.text_input("trail_mult_values (0 = sem trailing)", "0,0.5,1.0,1.5")
    hardstop_text = st.selectbox("ATR hardstop (ratio ATR14/ATR200)", ["off", "1.0", "1.2", "1.5", "2.0"], index=0)

sl_mults = tuple(sorted(set(parse_float_list(sl_mult_text))))
tp_mults = tuple(sorted(set(parse_float_list(tp_mult_text))))
trail_mults = tuple(sorted(set(max(0.0, v) for v in parse_float_list(trail_mult_text)))) or (0.0,)

paths: list[str] = []
for w in sorted(sel_w):
    wd = exp_dir / f"window_{int(w):03d}"
    p_sig, _ = resolve_signals_path(wd, split)  # type: ignore[arg-type]
    if p_sig.exists():
        paths.append(str(p_sig.resolve()))
if not paths:
    st.warning("Sem CSVs de sinais para as janelas escolhidas.")
    st.stop()


@st.cache_data(show_spinner=True)
def _grid_across_windows_cached(
    signals_paths: tuple[str, ...],
    raw_csv_path: str,
    atr_period: int,
    hardstop_text: str,
    sl_mults: tuple[float, ...],
    tp_mults: tuple[float, ...],
    trail_mults: tuple[float, ...],
    signal_threshold: float,
    taker_fee: float,
    position_notional: float,
    split: str,
) -> pd.DataFrame:
    ctx_df: pd.DataFrame | None = None
    rp = Path(raw_csv_path) if str(raw_csv_path).strip() else None
    if rp is not None:
        if not rp.is_absolute():
            rp = (REPO_ROOT / rp).resolve()
        if rp.exists():
            try:
                ctx_df = pd.read_csv(rp)
            except Exception:
                ctx_df = None

    hs = None if hardstop_text == "off" else float(hardstop_text)
    rows = []
    ppy = 365.0 * 24.0
    for spath in signals_paths:
        p_sig = Path(spath)
        wid = int(p_sig.parent.name.split("_")[1])
        sig = pd.read_csv(p_sig)
        arr = compute_atr_arrays(sig, atr_period=int(atr_period), slow_period=200, ohlcv_context_df=ctx_df)
        atr_fast_dec = np.asarray(arr["atr_fast_dec"], dtype=np.float64)
        atr_ratio_dec = np.asarray(arr["atr_ratio_dec"], dtype=np.float64)
        mask = entry_mask_from_hardstop(atr_ratio_dec, hardstop=hs, warmup=int(atr_period), atr_fast=atr_fast_dec)

        for sm in sl_mults:
            for tm in tp_mults:
                for trl in trail_mults:
                    sl_arr = atr_fast_dec * float(sm)
                    tp_arr = atr_fast_dec * float(tm)
                    sl_safe = np.where(np.isfinite(sl_arr) & (sl_arr > 0), sl_arr, 1e-9)
                    tp_safe = np.where(np.isfinite(tp_arr) & (tp_arr > 0), tp_arr, 1e-9)
                    trail_arr = None
                    if float(trl) > 0:
                        raw_tr = atr_fast_dec * float(trl)
                        trail_arr = np.where(np.isfinite(raw_tr) & (raw_tr > 0), raw_tr, 0.0)

                    rep = replay_signals_dynamic(
                        sig,
                        sl_points_per_bar=sl_safe,
                        tp_points_per_bar=tp_safe,
                        signal_threshold=float(signal_threshold),
                        trailing_stop_per_bar=trail_arr,
                        taker_fee=float(taker_fee),
                        position_notional=float(position_notional),
                        entry_mask=mask,
                        window_id=wid,
                        split=str(split),
                    )
                    eq = rep.equities
                    fin = float(eq[-1]) if len(eq) else float(position_notional)
                    roi = (fin / float(position_notional) - 1.0) * 100.0
                    sh = float(sharpe_ratio(eq, periods_per_year=ppy))
                    dd = float(max_drawdown_info(eq).get("max_drawdown", 0.0))
                    rows.append(
                        {
                            "window_id": wid,
                            "sl_mult": float(sm),
                            "tp_mult": float(tm),
                            "trail_mult": float(trl),
                            "roi_pct": float(roi),
                            "sharpe": float(sh),
                            "max_dd": float(dd),
                            "n_trades": int(len(rep.trades)),
                            "entries_blocked": int(rep.entries_blocked),
                        }
                    )
    return pd.DataFrame(rows)


res_df = _grid_across_windows_cached(
    tuple(sorted(paths)),
    raw_csv_path,
    atr_period,
    hardstop_text,
    sl_mults,
    tp_mults,
    trail_mults,
    thr,
    fee,
    nom,
    split,
)
if res_df.empty:
    st.warning("Sem resultados.")
    st.stop()

agg = (
    res_df.groupby(["sl_mult", "tp_mult", "trail_mult"], as_index=False)
    .agg(
        avg_roi=("roi_pct", "mean"),
        avg_sharpe=("sharpe", "mean"),
        avg_dd=("max_dd", "mean"),
        avg_trades=("n_trades", "mean"),
        avg_entries_blocked=("entries_blocked", "mean"),
        pct_windows_positive=("roi_pct", lambda s: float((s > 0).mean() * 100.0)),
    )
    .sort_values("avg_sharpe", ascending=False)
    .reset_index(drop=True)
)

st.subheader("Summary (across selected windows)")
st.dataframe(agg.head(30), width="stretch")

st.subheader("ROI% por janela (top-10 combos por avg_sharpe)")
topk = agg.head(10)[["sl_mult", "tp_mult", "trail_mult"]]
keys = set(tuple(r) for r in topk.to_numpy())
sub = res_df[res_df.apply(lambda r: (r["sl_mult"], r["tp_mult"], r["trail_mult"]) in keys, axis=1)].copy()
sub["label"] = sub.apply(lambda r: f"SL={r['sl_mult']:g} TP={r['tp_mult']:g} TR={r['trail_mult']:g}", axis=1)
fig = go.Figure()
for lab in sorted(sub["label"].unique()):
    s = sub[sub["label"] == lab].sort_values("window_id")
    fig.add_trace(go.Bar(x=s["window_id"], y=s["roi_pct"], name=lab))
fig.add_hline(y=0, line_dash="solid", line_color="gray")
fig.update_layout(barmode="group", height=430, xaxis_title="window_id", yaxis_title="ROI %")
st.plotly_chart(fig, width="stretch")


def _best_combo_by_bin_global(
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
        df.groupby([bin_col, "sl_mult", "tp_mult", "trail_mult"], dropna=False)
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

    idx = grp.groupby(bin_col, dropna=False)[metric_col].idxmax()
    out = grp.loc[idx].copy().sort_values(bin_col).reset_index(drop=True)
    out["metric_name"] = str(metric_col)
    out["metric_value"] = out[metric_col].astype(float)
    return out[
        [
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


@st.cache_data(show_spinner=True)
def _atr_breakdown_trades_cached(
    signals_paths: tuple[str, ...],
    split: str,
    raw_csv_path: str,
    atr_period: int,
    hardstop_text: str,
    sl_mults: tuple[float, ...],
    tp_mults: tuple[float, ...],
    trail_mults: tuple[float, ...],
    signal_threshold: float,
    taker_fee: float,
    position_notional: float,
    ratio_bin_w: float,
    fast_bin_w: float,
) -> pd.DataFrame:
    ctx_df: pd.DataFrame | None = None
    rp = Path(raw_csv_path) if str(raw_csv_path).strip() else None
    if rp is not None:
        if not rp.is_absolute():
            rp = (REPO_ROOT / rp).resolve()
        if rp.exists():
            try:
                ctx_df = pd.read_csv(rp)
            except Exception:
                ctx_df = None

    hs = None if hardstop_text == "off" else float(hardstop_text)
    all_trades: list[pd.DataFrame] = []
    ratio_by_window: dict[int, np.ndarray] = {}
    fast_by_window: dict[int, np.ndarray] = {}

    for spath in signals_paths:
        p_sig = Path(spath)
        try:
            wid = int(p_sig.parent.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        sig = pd.read_csv(p_sig)
        tdf, ratio_dec, fast_dec = compute_atr_dynamic_trades_for_window(
            sig,
            atr_period=int(atr_period),
            slow_period=200,
            hardstop=hs,
            sl_mults=sl_mults,
            tp_mults=tp_mults,
            trail_mults=trail_mults,
            signal_threshold=float(signal_threshold),
            taker_fee=float(taker_fee),
            position_notional=float(position_notional),
            ohlcv_context_df=ctx_df,
            window_id=int(wid),
            split=str(split),
        )
        ratio_by_window[int(wid)] = ratio_dec
        fast_by_window[int(wid)] = fast_dec
        if not tdf.empty:
            all_trades.append(tdf)

    if not all_trades:
        return pd.DataFrame()

    tagged = tag_regime_bins(
        pd.concat(all_trades, ignore_index=True, sort=False),
        ratio_by_window,
        fast_by_window,
        ratio_bin_w=float(ratio_bin_w),
        fast_bin_w=float(fast_bin_w),
    )
    return tagged


with st.expander("Regime Breakdown — melhor combo por bin (across selected windows)", expanded=False):
    ratio_bin_w = float(
        st.slider("ATR ratio bin width", min_value=0.05, max_value=0.50, value=0.10, step=0.05, format="%.2f")
    )
    fast_bin_w = float(
        st.slider("ATR_14 level bin width", min_value=1.0, max_value=20.0, value=5.0, step=1.0, format="%.1f")
    )
    metric_label = st.selectbox("Metric to select best combo", ["roi_pct", "sharpe", "win_rate"], index=1)
    min_trades = int(st.number_input("Minimum trades per bin", min_value=1, max_value=1000, value=3, step=1))

    tagged = _atr_breakdown_trades_cached(
        tuple(sorted(paths)),
        split,
        raw_csv_path,
        atr_period,
        hardstop_text,
        sl_mults,
        tp_mults,
        trail_mults,
        thr,
        fee,
        nom,
        ratio_bin_w,
        fast_bin_w,
    )
    if tagged is None or not isinstance(tagged, pd.DataFrame) or tagged.empty:
        st.info("Sem trades suficientes para construir regime breakdown com as janelas/combos actuais.")
    else:
        best_ratio = _best_combo_by_bin_global(
            tagged,
            bin_col="ratio_bin",
            metric_col=metric_label,
            min_trades=min_trades,
            position_notional=nom,
        )
        best_fast = _best_combo_by_bin_global(
            tagged,
            bin_col="fast_bin",
            metric_col=metric_label,
            min_trades=min_trades,
            position_notional=nom,
        )

        st.markdown("**Best combo por ATR ratio bin**")
        if best_ratio.empty:
            st.caption("Sem bins com dados suficientes para ATR ratio.")
        else:
            st.dataframe(best_ratio, width="stretch")

        st.markdown("**Best combo por ATR_14 level bin**")
        if best_fast.empty:
            st.caption("Sem bins com dados suficientes para ATR_14.")
        else:
            st.dataframe(best_fast, width="stretch")

