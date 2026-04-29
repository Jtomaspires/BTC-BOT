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

from dashboard.utils.indicators import SUPPORTED, compute_indicators
from dashboard.utils.loader import find_experiment_dir, list_run_dirs, load_config_resolved, load_runs_summary
from dashboard.utils.paths import REPO_ROOT
from dashboard.utils.state import get_selection_defaults, set_selection


st.set_page_config(page_title="Window analysis", layout="wide")
st.title("Window analysis — forensics")

results = load_runs_summary()
defaults = get_selection_defaults()

run_ids = sorted(results["run_id"].dropna().astype(str).unique().tolist()) if "run_id" in results.columns else []
exp_names = sorted(results["experiment_name"].dropna().astype(str).unique().tolist()) if "experiment_name" in results.columns else []

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    run_id = st.selectbox(
        "run_id",
        run_ids,
        index=(run_ids.index(defaults.run_id) if defaults.run_id in run_ids else (len(run_ids) - 1 if run_ids else 0)),
    )
with col2:
    exp = st.selectbox(
        "experiment",
        exp_names,
        index=(exp_names.index(defaults.experiment) if defaults.experiment in exp_names else 0),
    )

roi_cols = [c for c in results.columns if isinstance(c, str) and c.endswith("_roi") and len(c) == 7]
row = results[(results["run_id"].astype(str).str.strip() == str(run_id).strip()) & (results["experiment_name"].astype(str) == exp)]
if row.empty:
    st.warning("Não encontrei esta combinação no runs_summary.csv. A usar o último match por experiment.")
    row = results[results["experiment_name"].astype(str) == exp].tail(1)
row = row.iloc[0]

win_rows = []
for c in roi_cols:
    w = int(c.split("_")[0])
    win_rows.append(
        {
            "window": w,
            "roi": float(row.get(c)) if pd.notna(row.get(c)) else np.nan,
            "dd": float(row.get(f"{w:03d}_dd")) if pd.notna(row.get(f"{w:03d}_dd")) else np.nan,
        }
    )
win_df = pd.DataFrame(win_rows).sort_values("roi", ascending=True, na_position="last")

with col3:
    options = win_df["window"].astype(int).tolist()
    default_w = int(defaults.window) if defaults.window in options else int(options[0] if options else 0)
    window_id = st.selectbox("window (worst-first)", options, index=options.index(default_w) if options else 0)

set_selection(run_id=str(run_id), experiment=str(exp), window=int(window_id))

st.subheader("Piores janelas (ordenadas por ROI)")
st.dataframe(win_df, width="stretch", height=240)

st.subheader("Market slice + indicadores")

run_dir = (REPO_ROOT / "INF" / "outputs" / str(run_id)).resolve()
cfg = load_config_resolved(run_dir)
if not cfg:
    st.error(f"config.resolved.yaml não encontrado em {run_dir}")
    st.stop()

data_cfg = cfg.get("data", {}) or {}
wf_cfg = cfg.get("walkforward", {}) or {}

csv_path = Path(data_cfg.get("csv_path", ""))
if not csv_path.is_absolute():
    csv_path = (REPO_ROOT / csv_path).resolve()
market = pd.read_csv(csv_path)

recent_rows = int(data_cfg.get("recent_rows", 0) or 0)
if recent_rows > 0:
    market = market.tail(recent_rows).reset_index(drop=True)

train_size = int(wf_cfg.get("train_size", 0) or 0)
val_size = int(wf_cfg.get("val_size", 0) or 0)
test_size = int(wf_cfg.get("test_size", 0) or 0)
step_size = int(wf_cfg.get("step_size", 0) or 0)
anchor = int(wf_cfg.get("anchor", 0) or 0)

start = anchor + int(window_id) * step_size
train_end = start + train_size
val_end = train_end + val_size
test_end = val_end + (test_size if test_size > 0 else 0)

sl = slice(val_end, test_end if test_size > 0 else val_end)
slice_df = market.iloc[sl].copy()

inds = st.multiselect(
    "Indicators overlay",
    sorted(SUPPORTED),
    default=["ATR_14", "RSI_14"],
)
if inds:
    slice_df = compute_indicators(slice_df, inds)

price_fig = go.Figure()
price_fig.add_trace(
    go.Candlestick(
        x=slice_df.index,
        open=slice_df["open"],
        high=slice_df["high"],
        low=slice_df["low"],
        close=slice_df["close"],
        name="OHLC",
    )
)
price_fig.update_layout(height=500, xaxis_rangeslider_visible=False, title=f"Window {window_id} (rows {val_end}:{test_end})")

for ind in inds:
    if ind in slice_df.columns:
        price_fig.add_trace(go.Scatter(x=slice_df.index, y=slice_df[ind], name=ind, yaxis="y2"))

price_fig.update_layout(
    yaxis=dict(title="price"),
    yaxis2=dict(title="indicator", overlaying="y", side="right", showgrid=False),
)

st.plotly_chart(price_fig, width="stretch")

roi_val = float(win_df[win_df["window"] == int(window_id)]["roi"].iloc[0])
dd_val = float(win_df[win_df["window"] == int(window_id)]["dd"].iloc[0]) if "dd" in win_df.columns else float("nan")
mc1, mc2 = st.columns(2)
mc1.metric("ROI", f"{roi_val:.2%}" if np.isfinite(roi_val) else "n/a")
mc2.metric("DD", f"{dd_val:.2%}" if np.isfinite(dd_val) else "n/a")

st.caption("Nota: o mapping window→slice aqui usa apenas walkforward sizes do config. Se quiseres timestamps exatos, podemos usar a coluna timestamp (quando existir) para mostrar datas reais.")

