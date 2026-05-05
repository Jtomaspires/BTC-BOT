from __future__ import annotations

import importlib.util
from pathlib import Path

_bootstrap_path = Path(__file__).resolve().parent.parent / "bootstrap_sys_path.py"
_spec = importlib.util.spec_from_file_location("_nn_dashboard_bootstrap", _bootstrap_path)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(importlib.util.module_from_spec(_spec))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from INF.metrics import max_drawdown_info, sharpe_ratio

from dashboard.utils.loader import (
    experiment_picker_options,
    list_experiment_subdir_names,
    load_config_resolved,
    load_runs_summary,
    load_signals_for_window,
    resolve_exp_dir_like_heatmap,
    resolve_signals_path,
    run_id_select_options,
)
from dashboard.utils.parse_params import parse_float_list
from dashboard.utils.paths import REPO_ROOT
from dashboard.utils.regime import compute_atr_arrays, entry_mask_from_hardstop
from dashboard.utils.state import get_selection_defaults, set_selection
from dashboard.utils.trades import replay_signals_dynamic


def _roi_cols_ordered(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if isinstance(c, str) and len(c) == 7 and c.endswith("_roi")]
    return sorted(cols, key=lambda x: int(x[:3]))


def _window_ids_worst_first(row: pd.Series, roi_cols: list[str]) -> list[int]:
    pairs: list[tuple[int, float]] = []
    for c in roi_cols:
        v = row.get(c)
        if pd.notna(v):
            pairs.append((int(c[:3]), float(v)))
    pairs.sort(key=lambda x: x[1])
    return [p[0] for p in pairs]


def _tag_regime_bins(
    trades_df: pd.DataFrame,
    atr_ratio_dec: np.ndarray,
    atr_fast_dec: np.ndarray,
    ratio_bin_w: float,
    fast_bin_w: float,
) -> pd.DataFrame:
    if trades_df.empty:
        return trades_df.copy()
    out = trades_df.copy()
    entry_idx = out["entry_idx"].astype(int).to_numpy()
    ratio_vals = np.full(len(out), np.nan, dtype=np.float64)
    fast_vals = np.full(len(out), np.nan, dtype=np.float64)
    valid = (entry_idx >= 0) & (entry_idx < len(atr_ratio_dec))
    ratio_vals[valid] = atr_ratio_dec[entry_idx[valid]]
    valid_f = (entry_idx >= 0) & (entry_idx < len(atr_fast_dec))
    fast_vals[valid_f] = atr_fast_dec[entry_idx[valid_f]]

    out["atr_ratio_entry"] = ratio_vals
    out["atr_fast_entry"] = fast_vals
    out["ratio_bin"] = np.floor(ratio_vals / float(ratio_bin_w)) * float(ratio_bin_w)
    out["fast_bin"] = np.floor(fast_vals / float(fast_bin_w)) * float(fast_bin_w)
    out.loc[~np.isfinite(out["ratio_bin"].to_numpy(dtype=np.float64)), "ratio_bin"] = np.nan
    out.loc[~np.isfinite(out["fast_bin"].to_numpy(dtype=np.float64)), "fast_bin"] = np.nan
    return out


def _best_combo_by_bin(
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
    out["metric_name"] = metric_col
    out["metric_value"] = out[metric_col].astype(float)
    return out[[bin_col, "sl_mult", "tp_mult", "trail_mult", "metric_name", "metric_value", "n_trades", "roi_pct", "sharpe", "win_rate"]]


@st.cache_data(show_spinner=True)
def _replay_atr_grid_cached(
    signals_path: str,
    config_thr: float,
    taker_fee: float,
    position_notional: float,
    raw_ohlcv_path: str,
    atr_period: int,
    hardstop_text: str,
    sl_mults: tuple[float, ...],
    tp_mults: tuple[float, ...],
    trail_mults: tuple[float, ...],
) -> dict:
    sig = pd.read_csv(signals_path)
    ctx_df: pd.DataFrame | None = None
    raw_path = Path(raw_ohlcv_path) if str(raw_ohlcv_path).strip() else None
    if raw_path is not None:
        if not raw_path.is_absolute():
            raw_path = (REPO_ROOT / raw_path).resolve()
        if raw_path.exists():
            try:
                ctx_df = pd.read_csv(raw_path)
            except Exception:
                ctx_df = None
    arr = compute_atr_arrays(sig, atr_period=atr_period, slow_period=200, ohlcv_context_df=ctx_df)
    atr_fast = arr["atr_fast"]
    atr_ratio = arr["atr_ratio"]
    atr_fast_dec = arr["atr_fast_dec"]
    atr_ratio_dec = arr["atr_ratio_dec"]
    n = len(sig)

    warmup = int(atr_period)
    hardstop = None if hardstop_text == "off" else float(hardstop_text)
    mask_off = entry_mask_from_hardstop(atr_ratio_dec, hardstop=None, warmup=warmup, atr_fast=atr_fast_dec)
    mask_hs = entry_mask_from_hardstop(atr_ratio_dec, hardstop=hardstop, warmup=warmup, atr_fast=atr_fast_dec)
    active_mask = mask_hs if hardstop is not None else mask_off

    signal_vals = sig["signal"].astype(float).to_numpy()
    signal_events = (np.abs(signal_vals) > float(config_thr)) & mask_off
    blocked_events = signal_events & (~active_mask)
    blocked_pct = float(100.0 * blocked_events.sum() / signal_events.sum()) if signal_events.any() else 0.0

    close = sig["close"].astype(float).to_numpy()
    if close.size and close[0] != 0:
        buy_hold = float(position_notional) * (close / close[0])
    else:
        buy_hold = np.full(n, position_notional)

    out: dict[str, object] = {
        "x_index": np.arange(n, dtype=np.int64),
        "buy_hold": buy_hold.astype(np.float64),
        "atr_fast": atr_fast,
        "atr_ratio": atr_ratio,
        "atr_fast_dec": atr_fast_dec,
        "atr_ratio_dec": atr_ratio_dec,
        "hardstop": hardstop,
        "blocked_pct": blocked_pct,
        "metrics": [],
        "curves": {},
        "trades_list": [],
    }

    for sm in sl_mults:
        for tm in tp_mults:
            for trl in trail_mults:
                # Decisão sem lookahead: SL/TP/trail no bar i são dimensionados
                # a partir do ATR conhecido no fim de i-1 (atr_fast_dec).
                sl_arr = atr_fast_dec * float(sm)
                tp_arr = atr_fast_dec * float(tm)
                sl_safe = np.where(np.isfinite(sl_arr) & (sl_arr > 0), sl_arr, 1e-9)
                tp_safe = np.where(np.isfinite(tp_arr) & (tp_arr > 0), tp_arr, 1e-9)

                trail_arr: np.ndarray | None = None
                if float(trl) > 0:
                    raw_trail = atr_fast_dec * float(trl)
                    trail_arr = np.where(np.isfinite(raw_trail) & (raw_trail > 0), raw_trail, 0.0)

                rep_base = replay_signals_dynamic(
                    sig,
                    sl_points_per_bar=sl_safe,
                    tp_points_per_bar=tp_safe,
                    signal_threshold=float(config_thr),
                    trailing_stop_per_bar=trail_arr,
                    taker_fee=float(taker_fee),
                    position_notional=float(position_notional),
                    entry_mask=mask_off,
                )
                rep = rep_base
                if hardstop is not None:
                    rep = replay_signals_dynamic(
                        sig,
                        sl_points_per_bar=sl_safe,
                        tp_points_per_bar=tp_safe,
                        signal_threshold=float(config_thr),
                        trailing_stop_per_bar=trail_arr,
                        taker_fee=float(taker_fee),
                        position_notional=float(position_notional),
                        entry_mask=active_mask,
                    )

                eq = rep.equities
                fin = float(eq[-1]) if len(eq) else float(position_notional)
                roi = (fin / float(position_notional) - 1.0) * 100.0
                sh = float(sharpe_ratio(eq, periods_per_year=365.0 * 24.0))
                dd = float(max_drawdown_info(eq).get("max_drawdown", 0.0))

                tr_df = rep.trades
                if tr_df.empty:
                    avg_sl_d = 0.0
                    avg_tp_d = 0.0
                else:
                    avg_sl_d = float(
                        np.nanmean((tr_df["sl_dist"].astype(float) / tr_df["entry_price"].astype(float)) * position_notional)
                    )
                    avg_tp_d = float(
                        np.nanmean((tr_df["tp_dist"].astype(float) / tr_df["entry_price"].astype(float)) * position_notional)
                    )

                key = f"{sm:g}|{tm:g}|{trl:g}"
                out["curves"][key] = eq
                tr_payload = rep.trades.copy()
                if not tr_payload.empty:
                    tr_payload["sl_mult"] = float(sm)
                    tr_payload["tp_mult"] = float(tm)
                    tr_payload["trail_mult"] = float(trl)
                out["trades_list"].append(
                    {
                        "sl_mult": float(sm),
                        "tp_mult": float(tm),
                        "trail_mult": float(trl),
                        "trades_df": tr_payload,
                    }
                )
                out["metrics"].append(
                    {
                        "sl_mult": float(sm),
                        "tp_mult": float(tm),
                        "trail_mult": float(trl),
                        "RR": float(tm / sm) if sm != 0 else np.nan,
                        "ROI%": float(roi),
                        "sharpe": float(sh),
                        "max_dd": float(dd),
                        "n_trades": int(len(rep.trades)),
                        "avg_SL_$": float(avg_sl_d),
                        "avg_TP_$": float(avg_tp_d),
                        "trades_blocked": int(max(0, len(rep_base.trades) - len(rep.trades))),
                        "entries_blocked": int(rep.entries_blocked),
                    }
                )
    return out


st.set_page_config(page_title="ATR Dynamic Explorer", layout="wide")
st.title("ATR Dynamic Explorer — threshold fixo, SL/TP por ATR")

defaults = get_selection_defaults()
runs_df = load_runs_summary()
run_ids = run_id_select_options(runs_df)
if not run_ids:
    st.error("Não há pastas `run_*` em INF/outputs nem run_id no runs_summary.")
    st.stop()

_run_default = str(defaults.run_id).strip() if defaults.run_id else None
run_id = st.selectbox(
    "run_id (pastas em outputs; mesmo critério que o heatmap)",
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

row_df = runs_df[
    (runs_df["run_id"].astype(str).str.strip() == run_id)
    & (runs_df["experiment_name"].astype(str).str.strip() == experiment)
]
if row_df.empty and experiment != ".":
    row_df = runs_df[(runs_df["run_id"].astype(str).str.strip() == run_id)]
roi_cols = _roi_cols_ordered(runs_df)
if row_df.empty:
    st.error("Sem linha no runs_summary para este par run/experiment.")
    st.stop()
summary_row = row_df.iloc[0]
order_w = _window_ids_worst_first(summary_row, roi_cols)

exp_dir = resolve_exp_dir_like_heatmap(run_dir, experiment)
win_dirs = sorted([p for p in exp_dir.glob("window_*") if p.is_dir()], key=lambda x: int(x.name.split("_")[1]))
win_ids_avail = [int(p.name.split("_")[1]) for p in win_dirs]
ordered = [w for w in order_w if w in win_ids_avail]
rest = [w for w in win_ids_avail if w not in ordered]
win_pick_order = ordered + sorted(rest)

if not win_pick_order:
    hint = ""
    sub = list_experiment_subdir_names(run_dir)
    if sub:
        hint = f" Pastas em `experiments/`: {', '.join(sub)}."
    st.warning(
        f"Não há pastas `window_*` em `{exp_dir}`. Verifica o nome do experimento ou corre o walk-forward.{hint}"
    )
    st.stop()

split_opts = ["val", "test"]
selected_split_csv = (
    str(summary_row.get("selected_split", "")).strip().lower() if "selected_split" in summary_row.index else ""
)
default_split = selected_split_csv if selected_split_csv in split_opts else "val"
split_ix = split_opts.index(default_split)
if defaults.split in split_opts:
    split_ix = split_opts.index(defaults.split)


def _win_index() -> int:
    if defaults.window in win_pick_order:
        return win_pick_order.index(defaults.window)
    return 0


window_id = st.selectbox(
    "window_id (pior ROI primeiro no runs_summary)",
    win_pick_order,
    index=min(_win_index(), len(win_pick_order) - 1),
    format_func=lambda w: f"{int(w):03d}",
)
if window_id is None:
    window_id = win_pick_order[0]

split = st.radio("split", split_opts, index=split_ix, horizontal=True)
set_selection(run_id=run_id, experiment=experiment, window=int(window_id), split=split)

cfg = load_config_resolved(run_dir) or {}
bt = (cfg.get("backtest") or {}) if isinstance(cfg, dict) else {}
data_cfg = (cfg.get("data") or {}) if isinstance(cfg, dict) else {}
config_thr = float(bt.get("signal_threshold", 0.007)) if isinstance(bt, dict) else 0.007
taker_fee = float(bt.get("taker_fee", 0.00055))
position_notional = float(bt.get("position_notional", 1000.0))
raw_csv_path = str(data_cfg.get("csv_path", "")).strip() if isinstance(data_cfg, dict) else ""

thr = float(
    st.number_input(
        "signal_threshold (1 valor)",
        value=float(config_thr),
        min_value=0.0,
        step=0.0005,
        format="%.6f",
    )
)
sl_mult_text = st.text_input("sl_mult_values", "0.5,0.7,1.0,1.2,1.5,2.0,2.5")
tp_mult_text = st.text_input("tp_mult_values", "2.0,2.5,3.0,3.5,4.0,4.5,5.0")
trail_mult_text = st.text_input("trail_mult_values (0 = sem trailing)", "0,0.5,1.0,1.5")
atr_period = int(st.number_input("atr_period", value=14, min_value=5, max_value=200, step=1))
hardstop = st.selectbox("ATR hardstop (ratio ATR14/ATR200)", ["off", "1.0", "1.2", "1.5", "2.0"], index=0)

window_dir = exp_dir / f"window_{int(window_id):03d}"
signals = load_signals_for_window(window_dir, split)  # type: ignore[arg-type]
if signals is None or signals.empty:
    st.warning(f"Sem sinais em {window_dir} para split={split}.")
    st.stop()

sig_p, _ = resolve_signals_path(window_dir, split)  # type: ignore[arg-type]
sig_path = str(sig_p.resolve())
sl_mults = tuple(sorted(set(parse_float_list(sl_mult_text))))
tp_mults = tuple(sorted(set(parse_float_list(tp_mult_text))))
trail_mults = tuple(sorted(set(max(0.0, v) for v in parse_float_list(trail_mult_text))))
if not trail_mults:
    trail_mults = (0.0,)
payload = _replay_atr_grid_cached(
    sig_path,
    thr,
    taker_fee,
    position_notional,
    raw_csv_path,
    atr_period,
    hardstop,
    sl_mults,
    tp_mults,
    trail_mults,
)

metrics_df = pd.DataFrame(payload["metrics"]).sort_values("sharpe", ascending=False).reset_index(drop=True)

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    row_heights=[0.65, 0.35],
    vertical_spacing=0.05,
    specs=[[{}], [{"secondary_y": True}]],
)
x = payload["x_index"]
curve_keys = [f"{r['sl_mult']:g}|{r['tp_mult']:g}|{r['trail_mult']:g}" for _, r in metrics_df.iterrows()]
fin_eq = [payload["curves"][k][-1] for k in curve_keys]
rank = np.argsort(np.argsort(fin_eq))
nc = max(1, len(fin_eq) - 1)
colors = [px.colors.sample_colorscale("RdYlGn", r)[0] for r in (rank / nc)]

for i, (_, row) in enumerate(metrics_df.iterrows()):
    key = f"{row['sl_mult']:g}|{row['tp_mult']:g}|{row['trail_mult']:g}"
    y = payload["curves"][key]
    trail_label = f" | TR={row['trail_mult']:g}×ATR" if float(row["trail_mult"]) > 0 else ""
    hover_cd = np.column_stack(
        [
            np.full(len(x), float(row["sl_mult"]), dtype=np.float64),
            np.full(len(x), float(row["tp_mult"]), dtype=np.float64),
            np.full(len(x), float(row["trail_mult"]), dtype=np.float64),
            np.full(len(x), float(row["ROI%"]), dtype=np.float64),
            np.full(len(x), float(row["sharpe"]), dtype=np.float64),
        ]
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=f"SL={row['sl_mult']:g}×ATR | TP={row['tp_mult']:g}×ATR{trail_label} | {row['ROI%']:+.1f}%",
            line=dict(color=colors[i], width=1.4),
            customdata=hover_cd,
            hovertemplate=(
                "bar=%{x}<br>"
                "equity=%{y:.2f}<br>"
                "SL=%{customdata[0]:.3g}x ATR<br>"
                "TP=%{customdata[1]:.3g}x ATR<br>"
                "TR=%{customdata[2]:.3g}x ATR<br>"
                "ROI=%{customdata[3]:+.2f}%<br>"
                "Sharpe=%{customdata[4]:.3f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

fig.add_trace(
    go.Scatter(
        x=x,
        y=payload["buy_hold"],
        mode="lines",
        name="Buy & hold",
        line=dict(color="rgba(128,128,128,0.8)", dash="dash"),
    ),
    row=1,
    col=1,
)

atr_ratio = np.asarray(payload["atr_ratio"], dtype=np.float64)
atr_fast = np.asarray(payload["atr_fast"], dtype=np.float64)
fig.add_trace(go.Scatter(x=x, y=atr_ratio, name="ATR_ratio", line=dict(color="steelblue")), row=2, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=x, y=atr_fast, name="ATR_14", line=dict(color="orange")), row=2, col=1, secondary_y=True)
fig.add_hline(y=0.8, line_dash="dash", line_color="green", row=2, col=1)
fig.add_hline(y=1.2, line_dash="dash", line_color="orange", row=2, col=1)

hs = payload["hardstop"]
if hs is not None:
    hs_val = float(hs)
    fig.add_hline(y=hs_val, line_dash="dash", line_color="red", row=2, col=1)
    blocked = np.asarray(np.isfinite(atr_ratio) & (atr_ratio > hs_val), dtype=bool)
    start = None
    for i, b in enumerate(blocked):
        if b and start is None:
            start = i
        if (not b) and start is not None:
            fig.add_vrect(x0=start, x1=i, fillcolor="rgba(255,0,0,0.08)", line_width=0)
            start = None
    if start is not None:
        fig.add_vrect(x0=start, x1=len(blocked) - 1, fillcolor="rgba(255,0,0,0.08)", line_width=0)

fig.update_layout(height=520, hovermode="closest", title="ATR-dynamic grid: equity + regime")
fig.update_yaxes(title_text="Equity", row=1, col=1)
fig.update_yaxes(title_text="ATR_ratio", row=2, col=1, secondary_y=False)
fig.update_yaxes(title_text="ATR_14", row=2, col=1, secondary_y=True)
st.plotly_chart(fig, use_container_width=True)

show_cols = [
    "sl_mult",
    "tp_mult",
    "trail_mult",
    "RR",
    "ROI%",
    "sharpe",
    "max_dd",
    "n_trades",
    "avg_SL_$",
    "avg_TP_$",
    "trades_blocked",
    "entries_blocked",
]
st.dataframe(metrics_df[show_cols], use_container_width=True)

if not metrics_df.empty:
    best = metrics_df.iloc[0]
    rr = float(best["RR"])
    trail_note = f" Trail={best['trail_mult']:g}×ATR." if float(best["trail_mult"]) > 0 else " Sem trailing."
    st.success(
        f"Best combo: SL={best['sl_mult']:g}×ATR / TP={best['tp_mult']:g}×ATR "
        f"(RR 1:{rr:.1f}){trail_note} → {best['ROI%']:+.1f}% ROI, sharpe {best['sharpe']:.2f}, "
        f"max_dd {100.0*best['max_dd']:.1f}%. ATR hardstop blocked {float(payload['blocked_pct']):.0f}% of signals. "
        f"Avg SL ${best['avg_SL_$']:.0f}, Avg TP ${best['avg_TP_$']:.0f}."
    )

with st.expander("Regime Breakdown — melhor combo por bin", expanded=False):
    ratio_bin_w = float(
        st.slider("ATR ratio bin width", min_value=0.05, max_value=0.50, value=0.10, step=0.05, format="%.2f")
    )
    fast_bin_w = float(
        st.slider("ATR_14 level bin width", min_value=1.0, max_value=20.0, value=5.0, step=1.0, format="%.1f")
    )
    metric_label = st.selectbox(
        "Metric to select best combo",
        ["roi_pct", "sharpe", "win_rate"],
        index=1,
    )
    min_trades = int(st.number_input("Minimum trades per bin", min_value=1, max_value=1000, value=3, step=1))

    trades_list = payload.get("trades_list", [])
    trades_frames: list[pd.DataFrame] = []
    for item in trades_list:
        tdf = item.get("trades_df")
        if isinstance(tdf, pd.DataFrame) and not tdf.empty:
            trades_frames.append(tdf)

    if not trades_frames:
        st.info("Sem trades para construir regime breakdown nesta janela/split.")
    else:
        all_trades = pd.concat(trades_frames, ignore_index=True, sort=False)
        tagged = _tag_regime_bins(
            all_trades,
            np.asarray(payload["atr_ratio_dec"], dtype=np.float64),
            np.asarray(payload["atr_fast_dec"], dtype=np.float64),
            ratio_bin_w=ratio_bin_w,
            fast_bin_w=fast_bin_w,
        )

        best_ratio = _best_combo_by_bin(
            tagged,
            bin_col="ratio_bin",
            metric_col=metric_label,
            min_trades=min_trades,
            position_notional=position_notional,
        )
        best_fast = _best_combo_by_bin(
            tagged,
            bin_col="fast_bin",
            metric_col=metric_label,
            min_trades=min_trades,
            position_notional=position_notional,
        )

        st.markdown("**Best combo por ATR ratio bin**")
        if best_ratio.empty:
            st.caption("Sem bins com dados suficientes para ATR ratio.")
        else:
            st.dataframe(best_ratio, use_container_width=True)

        st.markdown("**Best combo por ATR_14 level bin**")
        if best_fast.empty:
            st.caption("Sem bins com dados suficientes para ATR_14.")
        else:
            st.dataframe(best_fast, use_container_width=True)

st.caption("Nota: este modo ATR-dinâmico não existe no `metrics.csv` original do INF (só combinações fixas).")
