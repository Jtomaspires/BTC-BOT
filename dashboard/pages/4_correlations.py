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
import streamlit as st

from dashboard.utils.indicators import SUPPORTED, compute_indicators
from dashboard.utils.loader import load_runs_summary
from dashboard.utils.paths import REPO_ROOT
from dashboard.utils.state import get_selection_defaults


st.set_page_config(page_title="Correlations", layout="wide")
st.title("Correlations — indicators vs ROI")

results = load_runs_summary()
roi_cols = [c for c in results.columns if isinstance(c, str) and c.endswith("_roi") and len(c) == 7]
defaults = get_selection_defaults()

if "run_id" not in results.columns or "experiment_name" not in results.columns:
    st.error("runs_summary.csv precisa de colunas run_id e experiment_name.")
    st.stop()

indicator = st.selectbox("Indicator", sorted(SUPPORTED), index=sorted(SUPPORTED).index("ATR_14") if "ATR_14" in SUPPORTED else 0)

# Optional filters driven by session_state defaults.
all_exps = sorted(results["experiment_name"].dropna().astype(str).unique().tolist())
default_exp = [defaults.experiment] if defaults.experiment in all_exps else []
exp_filter = st.multiselect("Filter experiments", all_exps, default=default_exp)
if exp_filter:
    results = results[results["experiment_name"].astype(str).isin(exp_filter)].copy()

all_runs = sorted(results["run_id"].dropna().astype(str).unique().tolist())
run_default_idx = all_runs.index(defaults.run_id) if defaults.run_id in all_runs else (len(all_runs) - 1 if all_runs else 0)
run_filter = st.selectbox("Filter run_id (optional)", ["(all)"] + all_runs, index=(run_default_idx + 1 if defaults.run_id in all_runs else 0))
if run_filter != "(all)":
    results = results[results["run_id"].astype(str) == str(run_filter)].copy()

# Market CSV path: take most common data_csv_path (best-effort).
if "data_csv_path" not in results.columns:
    st.error("runs_summary.csv não tem data_csv_path.")
    st.stop()

common_csv = results["data_csv_path"].dropna().astype(str).value_counts().index[0]
csv_path = Path(common_csv)
if not csv_path.is_absolute():
    csv_path = (REPO_ROOT / csv_path).resolve()
market = pd.read_csv(csv_path)

recent_rows = None
if "train_size" in results.columns and "val_size" in results.columns and "test_size" in results.columns and "step_size" in results.columns:
    pass

# Prepare per-window ROI long form.
long_rows = []
for _, r in results.iterrows():
    run_id = str(r["run_id"])
    exp = str(r["experiment_name"])
    for c in roi_cols:
        w = int(c.split("_")[0])
        roi = r.get(c)
        if pd.isna(roi):
            continue
        long_rows.append({"run_id": run_id, "experiment_name": exp, "window": w, "roi": float(roi)})
roi_long = pd.DataFrame(long_rows)

st.caption(f"Market CSV: {csv_path}")

# Compute indicator per window using the first row's WF params per (run_id, experiment).
# This is a best-effort fallback; page 2 uses config.resolved.yaml for exact params.
st.info("Nota: esta página usa sizes do próprio runs_summary.csv para mapear janelas (fallback). Para maior rigor, podemos ler config.resolved.yaml por run.")

group_cols = ["run_id", "experiment_name"]
wf_cols = ["train_size", "val_size", "test_size", "step_size", "anchor"]
missing_wf = [c for c in wf_cols if c not in results.columns]
if missing_wf:
    results = results.assign(**{c: np.nan for c in missing_wf})

meta = results[group_cols + wf_cols].drop_duplicates(subset=group_cols, keep="last")


def _safe_int(val: object, default: int = 0) -> int:
    if val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)) or pd.isna(val):
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def window_slice(row_meta: pd.Series, window: int) -> slice:
    train_size = _safe_int(row_meta.get("train_size"), 0)
    val_size = _safe_int(row_meta.get("val_size"), 0)
    test_size = _safe_int(row_meta.get("test_size"), 0)
    step_size = _safe_int(row_meta.get("step_size"), 0)
    anchor = _safe_int(row_meta.get("anchor"), 0)
    start = anchor + window * step_size
    val_end = start + train_size + val_size
    test_end = val_end + (test_size if test_size > 0 else 0)
    return slice(val_end, test_end if test_size > 0 else val_end)


ind_values = []
for _, m in meta.iterrows():
    run_id = str(m["run_id"])
    exp = str(m["experiment_name"])
    for w in sorted(roi_long[roi_long["run_id"] == run_id]["window"].unique().tolist()):
        sl = window_slice(m, int(w))
        window_df = market.iloc[sl].copy()
        if window_df.empty:
            continue
        window_df = compute_indicators(window_df, [indicator])
        if indicator not in window_df.columns:
            continue
        ind_mean = float(np.nanmean(window_df[indicator].to_numpy(dtype=float)))
        ind_values.append({"run_id": run_id, "experiment_name": exp, "window": int(w), "indicator_mean": ind_mean})

ind_df = pd.DataFrame(ind_values)
merged = roi_long.merge(ind_df, on=["run_id", "experiment_name", "window"], how="inner")

if merged.empty:
    st.warning("Não consegui construir dados suficientes para correlação.")
    st.stop()

st.subheader("Scatter + regressão")
fig = px.scatter(merged, x="indicator_mean", y="roi", color="experiment_name", hover_data=["run_id", "window"])

# Regression line + R²
x = merged["indicator_mean"].to_numpy(dtype=float)
y = merged["roi"].to_numpy(dtype=float)
mask = np.isfinite(x) & np.isfinite(y)
if mask.sum() >= 3:
    a, b = np.polyfit(x[mask], y[mask], 1)
    yhat = a * x[mask] + b
    ss_res = float(np.sum((y[mask] - yhat) ** 2))
    ss_tot = float(np.sum((y[mask] - np.mean(y[mask])) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    xs = np.linspace(float(np.min(x[mask])), float(np.max(x[mask])), 50)
    fig.add_trace(go.Scatter(x=xs, y=a * xs + b, name="fit", mode="lines"))
    fig.update_layout(title=f"{indicator} vs ROI (R²={r2:.3f})")

st.plotly_chart(fig, width="stretch")

st.subheader("Ranking (single-indicator)")
st.write(
    merged.groupby("experiment_name", dropna=False)
    .apply(lambda g: float(np.corrcoef(g["indicator_mean"], g["roi"])[0, 1]) if len(g) >= 3 else np.nan)
    .rename("corr")
    .reset_index()
    .sort_values("corr", key=lambda s: s.abs(), ascending=False)
)

