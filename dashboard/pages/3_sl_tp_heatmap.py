from __future__ import annotations

import importlib.util
from pathlib import Path

_bootstrap_path = Path(__file__).resolve().parent.parent / "bootstrap_sys_path.py"
_spec = importlib.util.spec_from_file_location("_nn_dashboard_bootstrap", _bootstrap_path)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(importlib.util.module_from_spec(_spec))

import numpy as np
import plotly.express as px
import streamlit as st

from dashboard.utils.backtest_lite import run_grid
from dashboard.utils.loader import list_run_dirs, load_config_resolved, load_signals
from dashboard.utils.paths import REPO_ROOT
from dashboard.utils.state import get_selection_defaults, set_selection


st.set_page_config(page_title="SL/TP heatmap", layout="wide")
st.title("SL/TP heatmap — signals.csv")

defaults = get_selection_defaults()

run_refs = list_run_dirs()
run_ids = [r.run_id for r in run_refs]
if not run_ids:
    st.error("Não encontrei runs em INF/outputs/run_*.")
    st.stop()

run_id = st.selectbox("run_id", run_ids, index=(run_ids.index(defaults.run_id) if defaults.run_id in run_ids else len(run_ids) - 1))
run_dir = (REPO_ROOT / "INF" / "outputs" / run_id).resolve()

exp_root = run_dir / "experiments"
exp_names = sorted([p.name for p in exp_root.iterdir() if p.is_dir()]) if exp_root.exists() else ["."]
exp = st.selectbox("experiment dir", exp_names, index=(exp_names.index(defaults.experiment) if defaults.experiment in exp_names else 0))
exp_dir = run_dir if exp == "." else (exp_root / exp)

win_dirs = sorted([p for p in exp_dir.glob("window_*") if p.is_dir()])
win_ids = [int(p.name.split("_")[1]) for p in win_dirs]
window_id = st.selectbox("window", win_ids, index=(win_ids.index(defaults.window) if defaults.window in win_ids else 0))
set_selection(run_id=run_id, experiment=exp, window=int(window_id))

window_dir = exp_dir / f"window_{int(window_id):03d}"
signals = load_signals(window_dir)
if signals is None or signals.empty:
    st.warning(f"signals.csv não encontrado em {window_dir}. (Pré-requisito do pipeline)")
    st.stop()

st.caption(f"Loaded {len(signals)} rows from signals.csv")

cfg_resolved = load_config_resolved(run_dir) or {}
bt_cfg = (cfg_resolved.get("backtest") or {}) if isinstance(cfg_resolved, dict) else {}
default_thr = float(bt_cfg.get("signal_threshold", 0.007))
default_taker_fee = float(bt_cfg.get("taker_fee", 0.00055))
default_position_notional = float(bt_cfg.get("position_notional", 1000.0))
default_trailing_list = bt_cfg.get("trailing_stop_points", [0.0])
if isinstance(default_trailing_list, (list, tuple)) and default_trailing_list:
    default_trailing_text = ",".join(f"{float(v):g}" for v in default_trailing_list)
else:
    default_trailing_text = "0"

st.caption(
    f"INF backtest defaults from config.resolved.yaml — "
    f"signal_threshold={default_thr:g}, taker_fee={default_taker_fee:g}, "
    f"position_notional={default_position_notional:g}, trailing_stop_points={default_trailing_text}"
)

colA, colB, colC = st.columns(3)
with colA:
    sl_min = st.number_input("SL min", min_value=0.1, value=2.0, step=0.5)
    sl_max = st.number_input("SL max", min_value=0.1, value=10.0, step=0.5)
    sl_step = st.number_input("SL step", min_value=0.1, value=1.0, step=0.5)
with colB:
    tp_min = st.number_input("TP min", min_value=0.1, value=10.0, step=1.0)
    tp_max = st.number_input("TP max", min_value=0.1, value=100.0, step=1.0)
    tp_step = st.number_input("TP step", min_value=0.1, value=5.0, step=1.0)
with colC:
    thr_text = st.text_input("threshold_values (comma)", value=f"{default_thr:g}")
    ts_text = st.text_input("trailing_stop_values (comma)", value=default_trailing_text)
    taker_fee = st.number_input(
        "taker_fee", min_value=0.0, value=default_taker_fee, step=0.0001, format="%.5f"
    )
    position_notional = st.number_input(
        "position_notional", min_value=1.0, value=default_position_notional, step=100.0
    )


def _parse_floats(s: str) -> list[float]:
    vals = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return vals


sl_values = np.arange(sl_min, sl_max + 1e-9, sl_step).tolist()
tp_values = np.arange(tp_min, tp_max + 1e-9, tp_step).tolist()
thr_values = _parse_floats(thr_text)
ts_values = _parse_floats(ts_text)

if st.button("Run grid"):
    with st.spinner("Running grid..."):
        out = run_grid(
            signals_df=signals,
            sl_values=sl_values,
            tp_values=tp_values,
            threshold_values=thr_values,
            trailing_stop_values=ts_values,
            taker_fee=float(taker_fee),
            position_notional=float(position_notional),
        )

    # Show only first (thr,ts) for now in UI (keep simple).
    thr0 = thr_values[0] if thr_values else 0.0
    ts0 = ts_values[0] if ts_values else 0.0

    eq_mat = np.full((len(sl_values), len(tp_values)), np.nan)
    sh_mat = np.full((len(sl_values), len(tp_values)), np.nan)
    rb_mat = np.full((len(sl_values), len(tp_values)), np.nan)

    for i, sl in enumerate(sl_values):
        for j, tp in enumerate(tp_values):
            m = out.get((float(sl), float(tp), float(thr0), float(ts0)))
            if m is None:
                continue
            eq_mat[i, j] = m.equity
            sh_mat[i, j] = m.sharpe
            rb_mat[i, j] = m.robustness

    tabs = st.tabs(["Equity heatmap", "Sharpe heatmap"])
    with tabs[0]:
        fig = px.imshow(eq_mat, x=[f"{tp:g}" for tp in tp_values], y=[f"{sl:g}" for sl in sl_values], aspect="auto")
        fig.update_layout(xaxis_title="TP", yaxis_title="SL")
        st.plotly_chart(fig, width="stretch")
    with tabs[1]:
        fig = px.imshow(sh_mat, x=[f"{tp:g}" for tp in tp_values], y=[f"{sl:g}" for sl in sl_values], aspect="auto")
        fig.update_layout(xaxis_title="TP", yaxis_title="SL")
        st.plotly_chart(fig, width="stretch")

    # Best by sharpe
    best_idx = np.unravel_index(np.nanargmax(sh_mat), sh_mat.shape)
    best_sl = sl_values[int(best_idx[0])]
    best_tp = tp_values[int(best_idx[1])]
    best_sh = float(sh_mat[best_idx])
    st.subheader("Best params (by sharpe)")
    st.write({"sl": best_sl, "tp": best_tp, "threshold": thr0, "trailing_stop": ts0, "sharpe": best_sh})

    st.text_area(
        "Export snippet",
        value=(
            "backtest:\n"
            f"  sl_points: [{best_sl}]\n"
            f"  tp_points: [{best_tp}]\n"
            f"  signal_threshold: {thr0}\n"
            f"  trailing_stop_points: [{ts0}]\n"
        ),
        height=140,
    )

