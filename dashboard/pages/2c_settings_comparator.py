from __future__ import annotations

import importlib.util
import json
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
from dashboard.utils.parse_params import scalar_float_from_yaml
from dashboard.utils.paths import REPO_ROOT
from dashboard.utils.regime import compute_atr_arrays, entry_mask_from_hardstop
from dashboard.utils.state import get_selection_defaults, set_selection
from dashboard.utils.trades import replay_signals_dynamic, replay_signals_full

st.set_page_config(page_title="Settings Comparator", layout="wide")
st.title("Settings Comparator — vários settings × várias janelas")

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
base_thr = float(bt.get("signal_threshold", 0.007))
base_sl = scalar_float_from_yaml(bt.get("sl_points", 50), 50.0)
base_tp = scalar_float_from_yaml(bt.get("tp_points", 100), 100.0)
trl = bt.get("trailing_stop_points", [10.0])
base_trail = float(trl[0]) if isinstance(trl, (list, tuple)) and trl else 10.0
base_fee = float(bt.get("taker_fee", 0.00055))
base_notional = float(bt.get("position_notional", 1000.0))

st.subheader("Settings (até 6)")
if "cmp_settings" not in st.session_state:
    st.session_state["cmp_settings"] = [
        {"mode": "fixed", "thr": base_thr, "sl": base_sl, "tp": base_tp, "trail": base_trail},
        {"mode": "fixed", "thr": base_thr + 0.001, "sl": base_sl, "tp": base_tp, "trail": base_trail},
        {"mode": "fixed", "thr": max(0.001, base_thr - 0.001), "sl": base_sl * 1.2, "tp": base_tp, "trail": base_trail},
    ]

st.caption("Add fixed setting")
fa, fb = st.columns(2)
with fa:
    f_thr = st.text_input("fixed.thr", str(base_thr))
    f_sl = st.text_input("fixed.sl", str(base_sl))
with fb:
    f_tp = st.text_input("fixed.tp", str(base_tp))
    f_tr = st.text_input("fixed.trail", str(base_trail))

st.caption("Add ATR-dynamic setting")
da, db, dc = st.columns(3)
with da:
    d_thr = st.text_input("dyn.thr", str(base_thr))
    d_sm = st.text_input("dyn.sl_mult", "0.7")
with db:
    d_tm = st.text_input("dyn.tp_mult", "1.5")
    d_ap = st.text_input("dyn.atr_period", "14")
with dc:
    d_hs = st.selectbox("dyn.hardstop", ["off", "1.0", "1.2", "1.5", "2.0"], index=0)
    d_tr = st.text_input("dyn.trail", str(base_trail))

if st.button("Add fixed setting") and len(st.session_state["cmp_settings"]) < 6:
    st.session_state["cmp_settings"].append(
        {"mode": "fixed", "thr": float(f_thr), "sl": float(f_sl), "tp": float(f_tp), "trail": float(f_tr)}
    )
if st.button("Add ATR-dynamic setting") and len(st.session_state["cmp_settings"]) < 6:
    st.session_state["cmp_settings"].append(
        {
            "mode": "atr_dynamic",
            "thr": float(d_thr),
            "sl_mult": float(d_sm),
            "tp_mult": float(d_tm),
            "atr_period": int(float(d_ap)),
            "hardstop": str(d_hs),
            "trail": float(d_tr),
        }
    )
if st.button("Reset defaults"):
    st.session_state["cmp_settings"] = [
        {"mode": "fixed", "thr": base_thr, "sl": base_sl, "tp": base_tp, "trail": base_trail},
        {"mode": "fixed", "thr": base_thr + 0.001, "sl": base_sl, "tp": base_tp, "trail": base_trail},
        {"mode": "atr_dynamic", "thr": base_thr, "sl_mult": 0.7, "tp_mult": 1.5, "atr_period": 14, "hardstop": "1.2", "trail": base_trail},
    ]

settings = st.session_state["cmp_settings"][:6]
for i, s in enumerate(settings):
    if s.get("mode") == "atr_dynamic":
        st.caption(
            f"#{i}: mode=atr_dynamic thr={s['thr']} sl_mult={s['sl_mult']} tp_mult={s['tp_mult']} "
            f"atr_p={s['atr_period']} hs={s['hardstop']} trail={s['trail']}"
        )
    else:
        st.caption(f"#{i}: mode=fixed thr={s['thr']} sl={s['sl']} tp={s['tp']} trail={s['trail']}")


@st.cache_data(show_spinner=True)
def _cmp_runs(signals_paths: tuple[str, ...], settings_key: str, fee: float, notional: float) -> pd.DataFrame:
    sets = json.loads(settings_key)
    rows = []
    p_py = 365.0 * 24.0
    cfg = load_config_resolved(run_dir) or {}
    data_cfg = (cfg.get("data") or {}) if isinstance(cfg, dict) else {}
    raw_csv_path = str(data_cfg.get("csv_path", "")).strip() if isinstance(data_cfg, dict) else ""
    ctx_df: pd.DataFrame | None = None
    if raw_csv_path:
        p = Path(raw_csv_path)
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        if p.exists():
            try:
                ctx_df = pd.read_csv(p)
            except Exception:
                ctx_df = None
    for spath in signals_paths:
        wid = int(Path(spath).parent.name.split("_")[1])
        sig = pd.read_csv(spath)
        atr_ref = compute_atr_arrays(sig, atr_period=14, slow_period=200, ohlcv_context_df=ctx_df)
        atr_m = float(np.nanmean(np.asarray(atr_ref["atr_ratio"], dtype=np.float64)))
        regime = "mid_vol"
        if np.isfinite(atr_m):
            if atr_m < 0.8:
                regime = "low_vol"
            elif atr_m >= 1.2:
                regime = "high_vol"

        for si, stg in enumerate(sets):
            mode = stg.get("mode", "fixed")
            if mode == "atr_dynamic":
                atr_period = int(stg.get("atr_period", 14))
                arr = compute_atr_arrays(sig, atr_period=atr_period, slow_period=200, ohlcv_context_df=ctx_df)
                atr_fast_dec = arr["atr_fast_dec"]
                atr_ratio_dec = arr["atr_ratio_dec"]
                warmup = atr_period
                hs_text = str(stg.get("hardstop", "off"))
                hs_val = None if hs_text == "off" else float(hs_text)
                # Decisão sem lookahead: usa ATR (fast/ratio) com lag-1.
                mask = entry_mask_from_hardstop(
                    atr_ratio_dec, hardstop=hs_val, warmup=warmup, atr_fast=atr_fast_dec
                )
                sl_arr = np.where(np.isfinite(atr_fast_dec) & (atr_fast_dec > 0), atr_fast_dec * float(stg["sl_mult"]), 1e-9)
                tp_arr = np.where(np.isfinite(atr_fast_dec) & (atr_fast_dec > 0), atr_fast_dec * float(stg["tp_mult"]), 1e-9)
                rep = replay_signals_dynamic(
                    sig,
                    sl_points_per_bar=sl_arr,
                    tp_points_per_bar=tp_arr,
                    signal_threshold=float(stg["thr"]),
                    trailing_stop_points=float(stg["trail"]),
                    taker_fee=fee,
                    position_notional=notional,
                    entry_mask=mask,
                    window_id=wid,
                    split="cmp",
                )
                label = f"S{si} dyn sl×{stg['sl_mult']} tp×{stg['tp_mult']} hs={hs_text}"
            else:
                rep = replay_signals_full(
                    sig,
                    sl_points=float(stg["sl"]),
                    tp_points=float(stg["tp"]),
                    signal_threshold=float(stg["thr"]),
                    trailing_stop_points=float(stg["trail"]),
                    taker_fee=fee,
                    position_notional=notional,
                    window_id=wid,
                    split="cmp",
                )
                label = f"S{si} fix sl={stg['sl']} tp={stg['tp']}"

            eq = rep.equities
            fin = float(eq[-1]) if len(eq) else float(notional)
            roi = (fin / float(notional) - 1.0) * 100.0
            sh = float(sharpe_ratio(eq, periods_per_year=p_py))
            dd = float(max_drawdown_info(eq).get("max_drawdown", 0.0))
            rows.append(
                {
                    "window_id": wid,
                    "setting_idx": si,
                    "setting_label": label,
                    "mode": mode,
                    "roi_pct": roi,
                    "sharpe": sh,
                    "max_dd": dd,
                    "n_trades": int(len(rep.trades)),
                    "entries_blocked": int(rep.entries_blocked),
                    "atr_mean": atr_m,
                    "regime": regime,
                }
            )
    return pd.DataFrame(rows)


paths: list[str] = []
for w in sorted(sel_w):
    wd = exp_dir / f"window_{int(w):03d}"
    p, _ = resolve_signals_path(wd, split)  # type: ignore[arg-type]
    if p.exists():
        paths.append(str(p.resolve()))

if not paths or not settings:
    st.warning("Escolhe janelas com ficheiros de sinais e pelo menos um setting.")
    st.stop()

sk = json.dumps(settings, sort_keys=True)
res_df = _cmp_runs(tuple(sorted(paths)), sk, base_fee, base_notional)
res_df = res_df.sort_values(["window_id", "setting_idx"])

w_ids = sorted(res_df["window_id"].unique())
fig = go.Figure()
for si in sorted(res_df["setting_idx"].unique()):
    sub = res_df[res_df["setting_idx"] == si].set_index("window_id").reindex(w_ids)
    label = str(sub["setting_label"].dropna().iloc[0]) if len(sub["setting_label"].dropna()) else f"S{si}"
    fig.add_trace(go.Bar(x=w_ids, y=sub["roi_pct"].to_numpy(), name=label))
fig.add_hline(y=0, line_dash="solid", line_color="gray")
fig.update_layout(barmode="group", title="ROI % por janela", height=430, xaxis_title="window_id")
st.plotly_chart(fig, use_container_width=True)

st.caption("Faixa de regime (ATR ratio médio por janela):")
reg_colors = {"low_vol": "#2ca02c", "mid_vol": "#ff7f0e", "high_vol": "#d62728"}
strip = [
    reg_colors.get(
        str(res_df.loc[res_df["window_id"] == w, "regime"].iloc[0] if len(res_df.loc[res_df["window_id"] == w]) else "mid_vol"),
        "#7f7f7f",
    )
    for w in w_ids
]
fig2 = go.Figure(data=[go.Bar(x=w_ids, y=[1] * len(w_ids), marker_color=strip, showlegend=False)])
fig2.update_layout(height=80, yaxis_visible=False, margin=dict(t=10, b=10))
st.plotly_chart(fig2, use_container_width=True)

opts = list(range(len(settings)))
a_ix = st.selectbox("setting A (delta)", opts, index=0)
b_ix = st.selectbox("setting B (delta)", opts, index=min(1, len(settings) - 1))
metrics_for_hm = ["roi_pct", "sharpe", "max_dd"]
da = res_df[res_df["setting_idx"] == a_ix].set_index("window_id")
db = res_df[res_df["setting_idx"] == b_ix].set_index("window_id")
mat = []
for m in metrics_for_hm:
    row = []
    for w in w_ids:
        va = da.loc[w, m] if w in da.index else np.nan
        vb = db.loc[w, m] if w in db.index else np.nan
        row.append(float(va - vb) if np.isfinite(va) and np.isfinite(vb) else np.nan)
    mat.append(row)
z = np.asarray(mat, dtype=np.float64)
fig_h = go.Figure(
    data=go.Heatmap(z=z, x=[str(w) for w in w_ids], y=metrics_for_hm, colorscale="RdYlGn", zmid=0.0, colorbar=dict(title="A−B"))
)
fig_h.update_layout(title=f"Delta (A=S{a_ix} vs B=S{b_ix})", height=220)
st.plotly_chart(fig_h, use_container_width=True)

summ = []
best_per_w = res_df.loc[res_df.groupby("window_id")["sharpe"].idxmax()]
for si in sorted(res_df["setting_idx"].unique()):
    sub = res_df[res_df["setting_idx"] == si]
    wins = int((best_per_w["setting_idx"] == si).sum())
    mode = str(sub["mode"].iloc[0]) if len(sub) else "unknown"
    label = str(sub["setting_label"].iloc[0]) if len(sub) else f"S{si}"
    summ.append(
        {
            "setting": si,
            "label": label,
            "mode": mode,
            "avg_roi": float(sub["roi_pct"].mean()),
            "avg_sharpe": float(sub["sharpe"].mean()),
            "avg_dd": float(sub["max_dd"].mean()),
            "avg_trades": float(sub["n_trades"].mean()),
            "avg_entries_blocked": float(sub["entries_blocked"].mean()),
            "pct_windows_positive": float((sub["roi_pct"] > 0).mean() * 100.0),
            "pct_windows_best_sharpe": float(wins / max(1, len(w_ids)) * 100.0),
        }
    )
st.dataframe(pd.DataFrame(summ), use_container_width=True)

if summ:
    best_si = int(max(summ, key=lambda s: s["pct_windows_best_sharpe"])["setting"])
else:
    best_si = 0
n_wins = int((best_per_w["setting_idx"] == best_si).sum()) if len(best_per_w) else 0
bw = best_per_w[best_per_w["setting_idx"] == best_si]
dom_reg = str(bw["regime"].mode().iloc[0]) if len(bw) else "—"
worst_all = res_df.groupby("window_id")["roi_pct"].max()
worst_ids = worst_all[worst_all < 0].index.tolist() if len(worst_all) else []
st.info(
    f"Setting **S{best_si}** lidera em **{n_wins}/{len(w_ids)}** janelas (melhor sharpe). "
    f"Regime mais frequente nessas vitórias: **{dom_reg}**. "
    f"Janelas onde o melhor entre settings ainda tem ROI negativo: **{worst_ids[:8]}**."
)

drill = st.selectbox("Drill-down: gravar window_id na sessão para a página Threshold", w_ids, key="drill_w")
if st.button("Gravar window na sessão"):
    set_selection(window=int(drill))
    st.success(f"selected_window={int(drill)}")
