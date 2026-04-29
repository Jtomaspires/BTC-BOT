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

from dashboard.utils.loader import load_runs_summary
from dashboard.utils.state import set_selection


st.set_page_config(page_title="Timeline", layout="wide")
st.title("Timeline — windows over time (proxy)")

df = load_runs_summary()
roi_cols = [c for c in df.columns if isinstance(c, str) and c.endswith("_roi") and len(c) == 7]

if df.empty or not roi_cols:
    st.error("Sem dados suficientes em runs_summary.csv.")
    st.stop()

run_ids = sorted(df["run_id"].dropna().astype(str).unique().tolist())
exp_names = sorted(df["experiment_name"].dropna().astype(str).unique().tolist())

run_id = st.selectbox("run_id", run_ids, index=len(run_ids) - 1)
exp = st.selectbox("experiment", exp_names, index=0)

row = df[(df["run_id"].astype(str) == run_id) & (df["experiment_name"].astype(str) == exp)]
if row.empty:
    row = df[df["experiment_name"].astype(str) == exp].tail(1)
row = row.iloc[0]

timeline = []
for c in roi_cols:
    w = int(c.split("_")[0])
    roi = row.get(c)
    if np.isfinite(roi):
        timeline.append({"window": w, "roi": float(roi)})
timeline = sorted(timeline, key=lambda x: x["window"])

tl_df = px.data.tips()  # placeholder for structure (we'll replace below)
import pandas as pd

tl_df = pd.DataFrame(timeline)
fig = px.bar(tl_df, x="window", y="roi", color=np.where(tl_df["roi"] >= 0, "pos", "neg"), color_discrete_map={"pos": "green", "neg": "red"})
fig.update_layout(xaxis_title="window", yaxis_title="ROI", showlegend=False)
st.plotly_chart(fig, width="stretch")

st.caption("Click-through: escolhe uma janela e salta para a página 2.")
window = st.number_input("selected window", min_value=int(tl_df["window"].min()), max_value=int(tl_df["window"].max()), value=int(tl_df["window"].min()))
if st.button("Go to window analysis"):
    set_selection(run_id=run_id, experiment=exp, window=int(window))
    st.switch_page("dashboard/pages/2_window_analysis.py")

