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
import streamlit as st

from dashboard.utils.loader import load_runs_summary
from dashboard.utils.state import set_selection


st.set_page_config(page_title="Overview", layout="wide")
st.title("Overview — experiments")

df = load_runs_summary()

roi_cols = [c for c in df.columns if isinstance(c, str) and c.endswith("_roi") and len(c) == 7]
if not roi_cols:
    st.error("Não encontrei colunas NNN_roi no runs_summary.csv.")
    st.stop()

roi_mat = df[roi_cols].to_numpy(dtype=float)
df = df.copy()
df["pct_windows_positive"] = np.nanmean(roi_mat > 0.0, axis=1) * 100.0
df["worst_window"] = np.nanargmin(roi_mat, axis=1)
df["best_window"] = np.nanargmax(roi_mat, axis=1)

cols = [
    c
    for c in (
        "run_id",
        "config_name",
        "experiment_name",
        "avg_return",
        "avg_drawdown",
        "avg_sharpe",
        "pct_windows_positive",
        "worst_window",
        "best_window",
    )
    if c in df.columns
]

st.subheader("Tabela")
st.dataframe(df[cols].sort_values(["avg_return", "pct_windows_positive"], ascending=[False, False]), width="stretch")

st.subheader("ROI médio por experiment")
if "experiment_name" in df.columns and "avg_return" in df.columns:
    grouped = (
        df.groupby("experiment_name", dropna=False, as_index=False)["avg_return"]
        .mean()
        .sort_values("avg_return", ascending=False)
    )
    grouped["color"] = np.where(grouped["avg_return"] >= 0, "positive", "negative")
    fig = px.bar(grouped, x="experiment_name", y="avg_return", color="color", color_discrete_map={"positive": "green", "negative": "red"})
    fig.update_layout(xaxis_title="experiment", yaxis_title="avg_return")
    st.plotly_chart(fig, width="stretch")

st.subheader("Distribuição de ROI por janela")
long = df[["experiment_name"] + roi_cols].melt(id_vars=["experiment_name"], var_name="window", value_name="roi")
long["window"] = long["window"].str.replace("_roi", "", regex=False)
fig2 = px.box(long, x="experiment_name", y="roi", points="outliers")
fig2.update_layout(xaxis_title="experiment", yaxis_title="window ROI")
st.plotly_chart(fig2, width="stretch")

st.subheader("Atalho")
st.caption("Define seleção global (session_state) para usar noutras páginas.")
run_id = st.selectbox("run_id", sorted(df["run_id"].dropna().astype(str).unique().tolist()) if "run_id" in df.columns else [], index=0)
exp = st.selectbox("experiment", sorted(df["experiment_name"].dropna().astype(str).unique().tolist()) if "experiment_name" in df.columns else [], index=0)
win = st.number_input("window", min_value=0, max_value=999, value=0, step=1)
if st.button("Set selection"):
    set_selection(run_id=run_id, experiment=exp, window=int(win))
    st.success("Selection saved.")

