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

from dashboard.utils.indicators import compute_indicators
from dashboard.utils.loader import (
    experiment_picker_options,
    list_experiment_subdir_names,
    load_config_resolved,
    load_runs_summary,
    load_signals_for_window,
    load_window_metrics,
    resolve_exp_dir_like_heatmap,
    resolve_signals_path,
    run_id_select_options,
)
from dashboard.utils.parse_params import parse_float_list, scalar_float_from_yaml
from dashboard.utils.paths import REPO_ROOT
from dashboard.utils.regime import classify_regime, compute_atr_ratio
from dashboard.utils.state import get_selection_defaults, set_selection


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


@st.cache_data(show_spinner=True)
def _replay_thresholds_cached(
    signals_path: str,
    sl: float,
    tp: float,
    trail: float,
    taker_fee: float,
    position_notional: float,
    thresholds: tuple[float, ...],
) -> dict:
    from dashboard.utils.trades import replay_signals_full

    sig = pd.read_csv(signals_path)
    out: dict[str, object] = {"thresholds": [], "curves": {}, "metrics": []}
    n_bars = len(sig)
    c0 = sig["close"].astype(float).to_numpy()
    buy_hold = float(position_notional) * (c0 / c0[0]) if c0.size and c0[0] != 0 else np.full(n_bars, position_notional)
    out["buy_hold"] = buy_hold.astype(np.float64)
    out["x_index"] = np.arange(n_bars, dtype=np.int64)

    for thr in thresholds:
        rep = replay_signals_full(
            sig,
            sl_points=sl,
            tp_points=tp,
            signal_threshold=float(thr),
            trailing_stop_points=trail,
            taker_fee=taker_fee,
            position_notional=position_notional,
        )
        eq = rep.equities
        fin = float(eq[-1]) if len(eq) else float(position_notional)
        roi_pct = (fin / float(position_notional) - 1.0) * 100.0
        from INF.metrics import max_drawdown_info, sharpe_ratio

        sh = float(sharpe_ratio(eq, periods_per_year=365.0 * 24.0))
        dd = float(max_drawdown_info(eq).get("max_drawdown", 0.0))
        out["thresholds"].append(float(thr))
        out["curves"][float(thr)] = eq
        out["metrics"].append(
            {
                "thr": float(thr),
                "roi_pct": roi_pct,
                "sharpe": sh,
                "max_dd": dd,
                "n_trades": int(len(rep.trades)),
                "tp_hits": int(rep.tp_hits),
                "sl_hits": int(rep.sl_hits),
            }
        )
    return out


st.set_page_config(page_title="Threshold Analyzer", layout="wide")
st.title("Threshold Analyzer — uma janela, equity vs threshold")

defaults = get_selection_defaults()
runs_df = load_runs_summary()
run_ids = run_id_select_options(runs_df)
if not run_ids:
    st.error("Não há pastas ``run_*`` em INF/outputs nem run_id no runs_summary.")
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
    index=(exp_opts.index(str(defaults.experiment).strip()) if defaults.experiment and str(defaults.experiment).strip() in exp_opts else 0),
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
selected_split_csv = str(summary_row.get("selected_split", "")).strip().lower() if "selected_split" in summary_row.index else ""
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
default_sl = scalar_float_from_yaml(bt.get("sl_points", 50), 50.0) if isinstance(bt, dict) else 50.0
default_tp = scalar_float_from_yaml(bt.get("tp_points", 100), 100.0) if isinstance(bt, dict) else 100.0
default_trail = scalar_float_from_yaml(bt.get("trailing_stop_points", [0.0]), 0.0) if isinstance(bt, dict) else 0.0
config_thr = float(bt.get("signal_threshold", 0.007)) if isinstance(bt, dict) else 0.007

st.caption(
    f"Defaults do `config.resolved.yaml`: thr={config_thr:g}, sl={default_sl:g}, tp={default_tp:g}, "
    f"trail={default_trail:g} · `selected_split` no `runs_summary`: **{selected_split_csv or '—'}**."
)

thr_text = st.text_input(
    "threshold_values (vírgula)",
    "0.001,0.003,0.005,0.007,0.008,0.010,0.012",
)
sl = float(st.text_input("sl_points", str(default_sl)))
tp = float(st.text_input("tp_points", str(default_tp)))
trail = float(st.text_input("trailing_stop", str(default_trail)))

taker_fee = float(bt.get("taker_fee", 0.00055))
position_notional = float(bt.get("position_notional", 1000.0))

overlay = st.multiselect(
    "Indicadores (painel inferior)",
    ["ATR_ratio", "RSI_14", "MACD_hist", "Vol20"],
    default=["ATR_ratio", "RSI_14"],
)

window_dir = exp_dir / f"window_{int(window_id):03d}"
signals = load_signals_for_window(window_dir, split)  # type: ignore[arg-type]
if signals is None or signals.empty:
    st.warning(f"Sem sinais em {window_dir} para split={split}.")
    st.stop()

sig_p, _ = resolve_signals_path(window_dir, split)  # type: ignore[arg-type]
sig_path = str(sig_p.resolve())

thr_list = parse_float_list(thr_text)
thr_tuple = tuple(sorted(set(thr_list)))
payload = _replay_thresholds_cached(sig_path, sl, tp, trail, taker_fee, position_notional, thr_tuple)

metrics_df = pd.DataFrame(payload["metrics"])
metrics_df = metrics_df.sort_values("sharpe", ascending=False).reset_index(drop=True)

ohlc = signals[["open", "high", "low", "close"]].copy()
if "volume" not in signals.columns:
    ohlc["volume"] = 0.0
else:
    ohlc["volume"] = signals["volume"]
inds_df = compute_indicators(ohlc, ["RSI_14", "MACDh_12_26_9", "MACD_hist", "Vol20"])
atr_r = compute_atr_ratio(ohlc, 14, 200)
inds_df["ATR_ratio"] = atr_r
macd_col = "MACD_hist" if "MACD_hist" in inds_df.columns else "MACDh_12_26_9"

fin_eq = [payload["curves"][t][-1] for t in payload["thresholds"]]  # type: ignore[index]
rank = np.argsort(np.argsort(fin_eq))
n_c = max(1, len(fin_eq) - 1)
color_seq = [px.colors.sample_colorscale("RdYlGn", r)[0] for r in (rank / n_c)]

fig_eq = go.Figure()
x = payload["x_index"]
for i, thr in enumerate(payload["thresholds"]):
    y = payload["curves"][thr]
    m = metrics_df.loc[metrics_df["thr"] == thr].iloc[0]
    fig_eq.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=f"thr={thr:g} | ROI={m['roi_pct']:.1f}% | n={int(m['n_trades'])}",
            line=dict(color=color_seq[i], width=1.5),
            hovertemplate="thr=%{customdata[0]:.4f}<br>equity=%{y:.2f}<br>ROI%%=%{customdata[1]:.2f}<extra></extra>",
            customdata=np.column_stack([np.full(len(y), thr), (y / position_notional - 1.0) * 100.0]),
        )
    )
bh = payload["buy_hold"]
fig_eq.add_trace(
    go.Scatter(
        x=x,
        y=bh,
        mode="lines",
        name="Buy & hold",
        line=dict(color="rgba(128,128,128,0.8)", dash="dash"),
        hovertemplate="buy_hold=%{y:.2f}<extra></extra>",
    )
)
fig_eq.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), title="Equity por threshold", hovermode="x unified")

if not overlay:
    st.plotly_chart(fig_eq, use_container_width=True)
else:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.06)
    for tr in fig_eq.data:
        fig.add_trace(tr, row=1, col=1)
    x = np.arange(len(signals), dtype=int)
    for name in overlay:
        if name == "ATR_ratio":
            yv = inds_df["ATR_ratio"].to_numpy()
            fig.add_trace(go.Scatter(x=x, y=yv, name="ATR_ratio", line=dict(color="steelblue")), row=2, col=1)
            fig.add_hline(y=0.8, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_hline(y=1.2, line_dash="dash", line_color="red", row=2, col=1)
        elif name == "RSI_14" and "RSI_14" in inds_df:
            fig.add_trace(
                go.Scatter(x=x, y=inds_df["RSI_14"], name="RSI_14", line=dict(color="purple")), row=2, col=1
            )
            fig.add_hline(y=30, line_dash="dot", line_color="gray", row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="gray", row=2, col=1)
        elif name == "MACD_hist" and macd_col in inds_df:
            hv = inds_df[macd_col].fillna(0.0).to_numpy()
            fig.add_trace(
                go.Bar(x=x, y=hv, name="MACD hist", marker_color=np.where(hv >= 0, "green", "red")), row=2, col=1
            )
        elif name == "Vol20" and "Vol20" in inds_df:
            fig.add_trace(
                go.Scatter(x=x, y=inds_df["Vol20"], name="Vol20", line=dict(color="orange")),
                row=2,
                col=1,
            )
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10), title="Equity + indicadores", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

_is_config = (
    np.isclose(metrics_df["thr"], config_thr, atol=1e-9, rtol=0)
    & np.isclose(sl, default_sl, atol=1e-9, rtol=0)
    & np.isclose(tp, default_tp, atol=1e-9, rtol=0)
    & np.isclose(trail, default_trail, atol=1e-9, rtol=0)
)
metrics_df.insert(0, "is_config_default", _is_config)


def _hl_default(row):  # type: ignore[no-untyped-def]
    return ["background-color: rgba(255, 215, 0, 0.18)" if row.get("is_config_default") else "" for _ in row]


st.dataframe(metrics_df.style.apply(_hl_default, axis=1), use_container_width=True)

with st.expander("Linha de referência do `metrics.csv` (mesma janela)", expanded=False):
    mref = load_window_metrics(window_dir)
    if mref is None or mref.empty:
        st.write("Não encontrei `metrics.csv` nesta janela.")
    else:
        sub = mref.copy()
        for col, val in (("sl", sl), ("tp", tp), ("trailing_stop", trail)):
            if col in sub.columns:
                sub = sub[np.isclose(sub[col].astype(float), float(val), atol=1e-6, rtol=0)]
        if "eval_split" in sub.columns:
            sub = sub[sub["eval_split"].astype(str).str.lower() == split]
        keep_cols = [
            c
            for c in [
                "window_id",
                "eval_split",
                "sl",
                "tp",
                "trailing_stop",
                "final_equity",
                "total_pnl",
                "num_trades",
                "sl_hits",
                "tp_hits",
                "sharpe",
                "max_drawdown",
                "is_best",
            ]
            if c in sub.columns
        ]
        if sub.empty:
            st.write(
                f"Sem linha em `metrics.csv` para split={split}, sl={sl:g}, tp={tp:g}, trail={trail:g}. "
                "O INF só grava a combinação do config; mexer em SL/TP/trail aqui afasta-te do que está em disco."
            )
        else:
            st.caption(
                "Compara com a linha **is_config_default** acima — devem coincidir bit-exact (igual ao motor INF)."
            )
            st.dataframe(sub[keep_cols], use_container_width=True)

if not metrics_df.empty:
    best = metrics_df.iloc[0]
    atr_m = float(np.nanmean(atr_r.to_numpy()))
    _, rnames = classify_regime(pd.Series([atr_m]))
    regime_name = str(rnames.iloc[0]) if len(rnames) else "unknown"
    n_pos = int((metrics_df["roi_pct"] > 0).sum())
    st.info(
        f"Threshold **{best['thr']:.4f}** → ROI **{best['roi_pct']:.1f}%** "
        f"({int(best['n_trades'])} trades, melhor sharpe nesta grelha). "
        f"ATR regime (média janela): **{regime_name}** (avg ratio {atr_m:.2f}). "
        f"{n_pos}/{len(metrics_df)} thresholds com ROI>0."
    )

focus = st.number_input("Definir janela para outras páginas (session)", min_value=0, value=int(window_id), step=1)
if st.button("Gravar window_id na sessão"):
    set_selection(window=int(focus))
    st.success(f"window_id={int(focus)} guardado.")
