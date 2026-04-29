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


st.set_page_config(page_title="Compare", layout="wide")
st.title("Compare — experiment A vs B")

df = load_runs_summary()
roi_cols = [c for c in df.columns if isinstance(c, str) and c.endswith("_roi") and len(c) == 7]
if df.empty or not roi_cols:
    st.error("Sem colunas NNN_roi em runs_summary.csv.")
    st.stop()

exp_names = sorted(df["experiment_name"].dropna().astype(str).unique().tolist())
run_ids = sorted(df["run_id"].dropna().astype(str).unique().tolist())

colA, colB = st.columns(2)
with colA:
    run_a = st.selectbox("run_id A", run_ids, index=len(run_ids) - 1)
    exp_a = st.selectbox("experiment A", exp_names, index=0, key="expA")
with colB:
    run_b = st.selectbox("run_id B", run_ids, index=len(run_ids) - 1)
    exp_b = st.selectbox("experiment B", exp_names, index=min(1, len(exp_names) - 1), key="expB")


def extract_row(run_id: str, exp: str) -> pd.Series:
    row = df[(df["run_id"].astype(str) == run_id) & (df["experiment_name"].astype(str) == exp)]
    if row.empty:
        row = df[df["experiment_name"].astype(str) == exp].tail(1)
    return row.iloc[0]


rA = extract_row(run_a, exp_a)
rB = extract_row(run_b, exp_b)

rows = []
for c in roi_cols:
    w = int(c.split("_")[0])
    a = float(rA.get(c)) if pd.notna(rA.get(c)) else np.nan
    b = float(rB.get(c)) if pd.notna(rB.get(c)) else np.nan
    rows.append({"window": w, "roi_A": a, "roi_B": b, "delta": a - b if np.isfinite(a) and np.isfinite(b) else np.nan})

comp = pd.DataFrame(rows).sort_values("window")
st.dataframe(comp, width="stretch", height=240)

fig = px.bar(comp.melt(id_vars=["window"], value_vars=["roi_A", "roi_B"], var_name="series", value_name="roi"), x="window", y="roi", color="series", barmode="group")
st.plotly_chart(fig, width="stretch")

fig2 = px.bar(comp, x="window", y="delta", color=np.where(comp["delta"] >= 0, "A_wins", "B_wins"), color_discrete_map={"A_wins": "green", "B_wins": "red"})
fig2.update_layout(showlegend=False)
st.plotly_chart(fig2, width="stretch")

st.subheader("Resumo")
valid = comp.dropna(subset=["delta"])
st.write(
    {
        "A_wins": int((valid["delta"] > 0).sum()),
        "B_wins": int((valid["delta"] < 0).sum()),
        "ties": int((valid["delta"] == 0).sum()),
    }
)

thr = st.slider("Meaningful delta threshold", min_value=0.0, max_value=0.2, value=0.05, step=0.01)
big = valid[valid["delta"].abs() > thr].sort_values("delta", key=lambda s: s.abs(), ascending=False)
st.subheader(f"Janelas com |delta| > {thr:.2%}")
st.dataframe(big, width="stretch")

window = st.number_input("Inspect window", min_value=int(comp["window"].min()), max_value=int(comp["window"].max()), value=int(comp["window"].min()))
col1, col2 = st.columns(2)
with col1:
    if st.button("Inspect A"):
        set_selection(run_id=run_a, experiment=exp_a, window=int(window))
        st.switch_page("dashboard/pages/2_window_analysis.py")
with col2:
    if st.button("Inspect B"):
        set_selection(run_id=run_b, experiment=exp_b, window=int(window))
        st.switch_page("dashboard/pages/2_window_analysis.py")

