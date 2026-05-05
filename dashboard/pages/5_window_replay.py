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


def _profit_factor(trades_df: pd.DataFrame) -> float:
    if trades_df.empty or "pnl_net" not in trades_df.columns:
        return 0.0
    pnl = trades_df["pnl_net"].astype(float)
    gross_win = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    if gross_loss <= 0:
        return float("inf") if gross_win > 0 else 0.0
    return gross_win / gross_loss


def _safe_window_id_from_path(signals_path: str) -> int:
    p = Path(signals_path)
    try:
        return int(p.parent.name.split("_")[1])
    except (IndexError, ValueError):
        return -1


def _bins_from_ratio(atr_ratio_dec: np.ndarray, *, bin_w: float) -> np.ndarray:
    r = np.asarray(atr_ratio_dec, dtype=np.float64)
    out = np.full(r.shape[0], np.nan, dtype=np.float64)
    ok = np.isfinite(r)
    if not ok.any():
        return out
    w = float(bin_w)
    out[ok] = np.floor(r[ok] / w) * w
    return out


def _profile_rows_to_mult_arrays(
    *,
    atr_ratio_dec: np.ndarray,
    bin_w: float,
    rows: tuple[tuple[float, float, float, float], ...],
    defaults: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Constrói arrays por barra:
    - ratio_bin[i] = floor(atr_ratio_dec[i]/bin_w)*bin_w
    - sl_mult[i], tp_mult[i], trail_mult[i] segundo tabela; fallback para defaults.
    """
    ratio_bin = _bins_from_ratio(atr_ratio_dec, bin_w=float(bin_w))
    sl0, tp0, tr0 = (float(defaults[0]), float(defaults[1]), float(defaults[2]))
    sl_mult = np.full_like(ratio_bin, sl0, dtype=np.float64)
    tp_mult = np.full_like(ratio_bin, tp0, dtype=np.float64)
    tr_mult = np.full_like(ratio_bin, tr0, dtype=np.float64)

    if len(rows) == 0:
        return ratio_bin, sl_mult, tp_mult, tr_mult

    # Map ratio_bin -> mults
    mapping: dict[float, tuple[float, float, float]] = {}
    for rb, sm, tm, tr in rows:
        if not np.isfinite(float(rb)):
            continue
        mapping[float(rb)] = (float(sm), float(tm), float(tr))

    if not mapping:
        return ratio_bin, sl_mult, tp_mult, tr_mult

    uniq = np.unique(ratio_bin[np.isfinite(ratio_bin)])
    for rb in uniq:
        vals = mapping.get(float(rb))
        if vals is None:
            continue
        m = ratio_bin == float(rb)
        sl_mult[m] = float(vals[0])
        tp_mult[m] = float(vals[1])
        tr_mult[m] = float(vals[2])

    return ratio_bin, sl_mult, tp_mult, tr_mult


@st.cache_data(show_spinner=True)
def _replay_per_window_cached(
    *,
    signals_paths: tuple[str, ...],
    metrics_paths: tuple[str, ...],
    raw_csv_path: str,
    split: str,
    signal_threshold: float,
    original_threshold: float,
    atr_period: int,
    slow_period: int,
    hardstop_text: str,
    ratio_bin_w: float,
    default_mults: tuple[float, float, float],
    profiles: tuple[tuple[str, tuple[tuple[float, float, float, float], ...]], ...],
    taker_fee: float,
    position_notional: float,
) -> tuple[dict[int, dict], pd.DataFrame]:
    ctx_df: pd.DataFrame | None = None
    raw_path = Path(raw_csv_path) if str(raw_csv_path).strip() else None
    if raw_path is not None:
        if not raw_path.is_absolute():
            raw_path = (REPO_ROOT / raw_path).resolve()
        if raw_path.exists():
            try:
                ctx_df = pd.read_csv(raw_path)
            except Exception:
                ctx_df = None

    hardstop = None if hardstop_text == "off" else float(hardstop_text)
    ppy = 365.0 * 24.0
    per_window: dict[int, dict] = {}
    rows: list[dict[str, float | int | str]] = []

    for sig_path, met_path in zip(signals_paths, metrics_paths):
        sig = pd.read_csv(sig_path)
        wid = _safe_window_id_from_path(sig_path)
        if wid < 0:
            continue

        arr = compute_atr_arrays(
            sig,
            atr_period=int(atr_period),
            slow_period=int(slow_period),
            ohlcv_context_df=ctx_df,
        )
        atr_fast_dec = np.asarray(arr["atr_fast_dec"], dtype=np.float64)
        atr_ratio_dec = np.asarray(arr["atr_ratio_dec"], dtype=np.float64)
        atr_ratio_plot = np.asarray(arr["atr_ratio"], dtype=np.float64)
        atr_fast_plot = np.asarray(arr["atr_fast"], dtype=np.float64)
        mask = entry_mask_from_hardstop(
            atr_ratio_dec,
            hardstop=hardstop,
            warmup=int(atr_period),
            atr_fast=atr_fast_dec,
        )

        if "timestamp" in sig.columns:
            x_vals = sig["timestamp"].astype(str).tolist()
        elif "time" in sig.columns:
            x_vals = sig["time"].astype(str).tolist()
        else:
            x_vals = list(range(len(sig)))

        profiles_payload: list[dict] = []
        for prof_name, prof_rows in profiles:
            ratio_bin, sl_mult_arr, tp_mult_arr, tr_mult_arr = _profile_rows_to_mult_arrays(
                atr_ratio_dec=atr_ratio_dec,
                bin_w=float(ratio_bin_w),
                rows=prof_rows,
                defaults=default_mults,
            )

            sl_arr = atr_fast_dec * sl_mult_arr
            tp_arr = atr_fast_dec * tp_mult_arr
            sl_safe = np.where(np.isfinite(sl_arr) & (sl_arr > 0), sl_arr, 1e-9)
            tp_safe = np.where(np.isfinite(tp_arr) & (tp_arr > 0), tp_arr, 1e-9)

            raw_trail = atr_fast_dec * tr_mult_arr
            trail_safe = np.where(np.isfinite(raw_trail) & (raw_trail > 0), raw_trail, 0.0)

            rep = replay_signals_dynamic(
                sig,
                sl_points_per_bar=sl_safe,
                tp_points_per_bar=tp_safe,
                signal_threshold=float(signal_threshold),
                trailing_stop_per_bar=trail_safe,
                taker_fee=float(taker_fee),
                position_notional=float(position_notional),
                entry_mask=mask,
                window_id=int(wid),
                split=str(split),
            )
            eq = np.asarray(rep.equities, dtype=np.float64)
            final_equity = float(eq[-1]) if eq.size else float(position_notional)
            roi_pct = (final_equity / float(position_notional) - 1.0) * 100.0
            sh = float(sharpe_ratio(eq, periods_per_year=ppy))
            dd = float(max_drawdown_info(eq).get("max_drawdown", 0.0))
            trades_df = rep.trades.copy()
            n_trades = int(len(trades_df))
            win_rate = (
                float((trades_df["pnl_net"].astype(float) > 0).mean() * 100.0) if n_trades else 0.0
            )
            total_pnl = float(trades_df["pnl_net"].astype(float).sum()) if n_trades else 0.0
            total_fees = float(trades_df["fees"].astype(float).sum()) if n_trades else 0.0
            sl_hits = int((trades_df.get("exit_reason") == "sl").sum()) if n_trades else 0
            tp_hits = int((trades_df.get("exit_reason") == "tp").sum()) if n_trades else 0
            pf = _profit_factor(trades_df)

            profiles_payload.append(
                {
                    "profile": str(prof_name),
                    "equity": eq.tolist(),
                    "roi_pct": float(roi_pct),
                    "sharpe": float(sh),
                    "n_trades": int(n_trades),
                    "ratio_bin": ratio_bin.tolist(),
                    "sl_mult": sl_mult_arr.tolist(),
                    "tp_mult": tp_mult_arr.tolist(),
                    "trail_mult": tr_mult_arr.tolist(),
                }
            )
            rows.append(
                {
                    "window_id": int(wid),
                    "split": str(split),
                    "profile": str(prof_name),
                    "roi_pct": float(roi_pct),
                    "final_equity": float(final_equity),
                    "sharpe": float(sh),
                    "max_dd": float(dd),
                    "n_trades": int(n_trades),
                    "sl_hits": int(sl_hits),
                    "tp_hits": int(tp_hits),
                    "win_rate": float(win_rate),
                    "total_pnl": float(total_pnl),
                    "profit_factor": float(pf),
                    "entries_blocked": int(rep.entries_blocked),
                    "total_fees": float(total_fees),
                }
            )

        original_payload: dict[str, float | list[float]] = {
            "sl": float("nan"),
            "tp": float("nan"),
            "trail": float("nan"),
            "equity": [],
            "roi_pct": float("nan"),
        }
        metrics_path = Path(met_path)
        if metrics_path.exists():
            mdf = pd.read_csv(metrics_path)
            if not mdf.empty:
                if "eval_split" in mdf.columns:
                    sel = mdf[mdf["eval_split"].astype(str).str.lower() == str(split).lower()]
                    mrow = sel.iloc[0] if not sel.empty else mdf.iloc[0]
                else:
                    mrow = mdf.iloc[0]
                sl_orig = scalar_float_from_yaml(mrow.get("sl"), 0.0)
                tp_orig = scalar_float_from_yaml(mrow.get("tp"), 0.0)
                tr_orig = scalar_float_from_yaml(mrow.get("trailing_stop"), 0.0)
                rep_o = replay_signals_full(
                    sig,
                    sl_points=float(sl_orig),
                    tp_points=float(tp_orig),
                    signal_threshold=float(original_threshold),
                    trailing_stop_points=float(tr_orig),
                    taker_fee=float(taker_fee),
                    position_notional=float(position_notional),
                    window_id=int(wid),
                    split=str(split),
                )
                eq_o = np.asarray(rep_o.equities, dtype=np.float64)
                final_o = float(eq_o[-1]) if eq_o.size else float(position_notional)
                roi_o = (final_o / float(position_notional) - 1.0) * 100.0
                original_payload = {
                    "sl": float(sl_orig),
                    "tp": float(tp_orig),
                    "trail": float(tr_orig),
                    "equity": eq_o.tolist(),
                    "roi_pct": float(roi_o),
                }

        per_window[int(wid)] = {
            "x": x_vals,
            "profiles": profiles_payload,
            "original": original_payload,
            "atr_ratio": atr_ratio_plot.tolist(),
            "atr_fast": atr_fast_plot.tolist(),
            "hardstop": hardstop,
        }

    return per_window, pd.DataFrame(rows)


st.set_page_config(page_title="Window replay", layout="wide")
st.title("Window replay — dynamic ATR backtest inspector")
st.caption("Corre uma grelha ATR-dinâmica por janela e sobrepõe a curva original reproduzida a partir do metrics.csv.")

defaults = get_selection_defaults()
runs_df = load_runs_summary()
run_ids = run_id_select_options(runs_df)
if not run_ids:
    st.error("Sem runs.")
    st.stop()

_run_default = str(defaults.run_id).strip() if defaults.run_id else None
run_id = st.selectbox(
    "run_id",
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
    "experiment",
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
split_ix = split_opts.index(defaults.split) if defaults.split in split_opts else 1
split = st.radio("split", split_opts, index=split_ix, horizontal=True)

sel_w = st.multiselect("window_ids", win_ids_all, default=win_ids_all)
if not sel_w:
    st.warning("Seleciona pelo menos uma janela.")
    st.stop()
set_selection(run_id=run_id, experiment=experiment, split=split)

cfg = load_config_resolved(run_dir) or {}
bt = (cfg.get("backtest") or {}) if isinstance(cfg, dict) else {}
data_cfg = (cfg.get("data") or {}) if isinstance(cfg, dict) else {}
fee = float(bt.get("taker_fee", 0.00055))
nom = float(bt.get("position_notional", 1000.0))
orig_thr = float(bt.get("signal_threshold", 0.007))
raw_csv_path = str(data_cfg.get("csv_path", "")).strip() if isinstance(data_cfg, dict) else ""

col1, col2, col3 = st.columns(3)
with col1:
    thr = float(
        st.number_input(
            "signal_threshold (novo replay)",
            value=float(orig_thr),
            min_value=0.0,
            step=0.0005,
            format="%.6f",
        )
    )
    atr_period = int(st.number_input("atr_period", value=14, min_value=5, max_value=200, step=1))
    slow_period = int(st.number_input("slow_period", value=200, min_value=50, max_value=400, step=10))
with col2:
    st.markdown("**Defaults (fallback se bin não existir)**")
    sl_default = float(st.number_input("default_sl_mult", value=1.0, min_value=0.05, step=0.05, format="%.3f"))
    tp_default = float(st.number_input("default_tp_mult", value=3.0, min_value=0.05, step=0.05, format="%.3f"))
with col3:
    tr_default = float(st.number_input("default_trail_mult (0 = sem trailing)", value=0.5, min_value=0.0, step=0.1, format="%.3f"))
    hardstop_text = st.selectbox("ATR hardstop (ratio ATR14/ATR200)", ["off", "1.0", "1.2", "1.5", "2.0"], index=0)

ratio_bin_w = float(st.number_input("ratio_bin_width", value=0.10, min_value=0.01, step=0.01, format="%.2f"))
default_mults = (float(sl_default), float(tp_default), float(tr_default))

profiles_key = "window_replay_profiles"
if profiles_key not in st.session_state:
    st.session_state[profiles_key] = [
        {
            "name": "profile_A",
            "rows": pd.DataFrame(
                [
                    {"ratio_bin": 0.0, "sl_mult": sl_default, "tp_mult": tp_default, "trail_mult": tr_default},
                    {"ratio_bin": 0.5, "sl_mult": sl_default, "tp_mult": tp_default, "trail_mult": tr_default},
                    {"ratio_bin": 1.0, "sl_mult": sl_default, "tp_mult": tp_default, "trail_mult": tr_default},
                ]
            ),
        }
    ]

st.subheader("Profiles (ratio_bin → SL/TP/TR)")
st.caption("Cada profile define múltiplos por bin do `ATR_ratio_dec` (lag-1). Se um bin não existir, usa os defaults.")

prof_cols = st.columns([2, 2, 6])
with prof_cols[0]:
    new_name = st.text_input("novo profile", "profile_B")
with prof_cols[1]:
    if st.button("Add profile", width="stretch"):
        st.session_state[profiles_key].append(
            {
                "name": str(new_name).strip() or f"profile_{len(st.session_state[profiles_key]) + 1}",
                "rows": pd.DataFrame(
                    [
                        {"ratio_bin": 0.0, "sl_mult": sl_default, "tp_mult": tp_default, "trail_mult": tr_default},
                        {"ratio_bin": 1.0, "sl_mult": sl_default, "tp_mult": tp_default, "trail_mult": tr_default},
                    ]
                ),
            }
        )

remove_idx = None
for i, prof in enumerate(list(st.session_state[profiles_key])):
    st.markdown(f"**{prof['name']}**")
    ed = st.data_editor(
        prof["rows"],
        num_rows="dynamic",
        use_container_width=True,
        key=f"profile_editor_{i}",
        column_config={
            "ratio_bin": st.column_config.NumberColumn("ratio_bin", format="%.3g"),
            "sl_mult": st.column_config.NumberColumn("sl_mult", format="%.3g"),
            "tp_mult": st.column_config.NumberColumn("tp_mult", format="%.3g"),
            "trail_mult": st.column_config.NumberColumn("trail_mult", format="%.3g"),
        },
    )
    prof["rows"] = ed
    c1, c2 = st.columns([1, 5])
    with c1:
        if st.button("Remove", key=f"remove_profile_{i}"):
            remove_idx = i
    st.divider()

if remove_idx is not None and 0 <= int(remove_idx) < len(st.session_state[profiles_key]):
    st.session_state[profiles_key].pop(int(remove_idx))

profiles_serialized: list[tuple[str, tuple[tuple[float, float, float, float], ...]]] = []
for prof in st.session_state[profiles_key]:
    name = str(prof.get("name", "")).strip() or "profile"
    df = prof.get("rows")
    if not isinstance(df, pd.DataFrame) or df.empty:
        rows_t = tuple()
    else:
        dff = df.copy()
        for col in ("ratio_bin", "sl_mult", "tp_mult", "trail_mult"):
            if col not in dff.columns:
                dff[col] = np.nan
        dff = dff.dropna(subset=["ratio_bin"]).copy()
        dff["ratio_bin"] = dff["ratio_bin"].astype(float)
        dff = dff.sort_values("ratio_bin")
        rows_t = tuple(
            (float(r["ratio_bin"]), float(r["sl_mult"]), float(r["tp_mult"]), float(r["trail_mult"]))
            for _, r in dff.iterrows()
        )
    profiles_serialized.append((name, rows_t))

profiles_tuple = tuple(profiles_serialized) if profiles_serialized else (("profile", tuple()),)

paths: list[str] = []
metrics_paths: list[str] = []
for w in sorted(sel_w):
    wd = exp_dir / f"window_{int(w):03d}"
    p_sig, _ = resolve_signals_path(wd, split)  # type: ignore[arg-type]
    if p_sig.exists():
        paths.append(str(p_sig.resolve()))
        metrics_paths.append(str((wd / "metrics.csv").resolve()))
if not paths:
    st.warning("Sem CSVs de sinais para as janelas escolhidas.")
    st.stop()

per_window, metrics_long_df = _replay_per_window_cached(
    signals_paths=tuple(paths),
    metrics_paths=tuple(metrics_paths),
    raw_csv_path=raw_csv_path,
    split=split,
    signal_threshold=thr,
    original_threshold=orig_thr,
    atr_period=atr_period,
    slow_period=slow_period,
    hardstop_text=hardstop_text,
    ratio_bin_w=ratio_bin_w,
    default_mults=default_mults,
    profiles=profiles_tuple,
    taker_fee=fee,
    position_notional=nom,
)

if metrics_long_df.empty or not per_window:
    st.warning("Sem resultados para os parâmetros atuais.")
    st.stop()

selected_windows = sorted([int(w) for w in sel_w if int(w) in per_window])
if not selected_windows:
    st.warning("Sem resultados nas janelas selecionadas.")
    st.stop()

idx_key = "replay_win_idx"
if idx_key not in st.session_state:
    st.session_state[idx_key] = 0
st.session_state[idx_key] = int(np.clip(int(st.session_state[idx_key]), 0, len(selected_windows) - 1))

col_prev, col_pick, col_next = st.columns([1, 4, 1])
with col_prev:
    if st.button("<", width="stretch"):
        st.session_state[idx_key] = max(0, int(st.session_state[idx_key]) - 1)
with col_next:
    if st.button(">", width="stretch"):
        st.session_state[idx_key] = min(len(selected_windows) - 1, int(st.session_state[idx_key]) + 1)

with col_pick:
    current_window = st.selectbox(
        "Janela",
        selected_windows,
        index=int(st.session_state[idx_key]),
        format_func=lambda w: f"window_{int(w):03d}",
    )
st.session_state[idx_key] = selected_windows.index(int(current_window))
set_selection(window=int(current_window))

payload = per_window[int(current_window)]
profiles = payload.get("profiles", [])
x_vals = payload.get("x", [])
orig = payload.get("original", {})

if profiles:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.05,
        specs=[[{}], [{"secondary_y": True}]],
    )
    roi_vals = [float(p.get("roi_pct", 0.0)) for p in profiles]
    rank = np.argsort(np.argsort(roi_vals))
    denom = max(1, len(roi_vals) - 1)
    colors = [px.colors.sample_colorscale("RdYlGn", float(r) / float(denom))[0] for r in rank]

    for i, p in enumerate(sorted(profiles, key=lambda d: float(d.get("roi_pct", 0.0)), reverse=True)):
        y = p.get("equity", [])
        slm = np.asarray(p.get("sl_mult", []), dtype=np.float64)
        tpm = np.asarray(p.get("tp_mult", []), dtype=np.float64)
        trm = np.asarray(p.get("trail_mult", []), dtype=np.float64)
        if len(y) != len(slm) or len(y) != len(tpm) or len(y) != len(trm):
            slm = np.full(len(y), np.nan, dtype=np.float64)
            tpm = np.full(len(y), np.nan, dtype=np.float64)
            trm = np.full(len(y), np.nan, dtype=np.float64)
        customdata = np.column_stack(
            [
                slm,
                tpm,
                trm,
                np.full(len(y), float(p.get("roi_pct", 0.0)), dtype=np.float64),
                np.full(len(y), float(p.get("sharpe", 0.0)), dtype=np.float64),
            ]
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y,
                mode="lines",
                name=(
                    f"{str(p.get('profile','profile'))} | "
                    f"ROI={float(p.get('roi_pct', 0.0)):+.1f}%"
                ),
                line=dict(color=colors[min(i, len(colors) - 1)], width=1.5),
                customdata=customdata,
                hovertemplate=(
                    "x=%{x}<br>"
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

    y_orig = orig.get("equity", [])
    if isinstance(y_orig, list) and y_orig:
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_orig,
                mode="lines",
                name=(
                    f"Original (sl={float(orig.get('sl', np.nan)):g}, "
                    f"tp={float(orig.get('tp', np.nan)):g}, "
                    f"trail={float(orig.get('trail', np.nan)):g}) | "
                    f"ROI={float(orig.get('roi_pct', np.nan)):+.1f}%"
                ),
                line=dict(color="black", width=2.0, dash="dash"),
            ),
            row=1,
            col=1,
        )

    fig.add_hline(y=float(nom), line_dash="dot", line_color="gray", row=1, col=1)

    atr_ratio = np.asarray(payload.get("atr_ratio", []), dtype=np.float64)
    atr_fast = np.asarray(payload.get("atr_fast", []), dtype=np.float64)
    if atr_ratio.size and atr_ratio.size == len(x_vals):
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=atr_ratio,
                name="ATR_ratio",
                line=dict(color="steelblue"),
            ),
            row=2,
            col=1,
            secondary_y=False,
        )
    if atr_fast.size and atr_fast.size == len(x_vals):
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=atr_fast,
                name="ATR_14",
                line=dict(color="orange"),
            ),
            row=2,
            col=1,
            secondary_y=True,
        )
    fig.add_hline(y=0.8, line_dash="dash", line_color="green", row=2, col=1, secondary_y=False)
    fig.add_hline(y=1.2, line_dash="dash", line_color="orange", row=2, col=1, secondary_y=False)

    hs_panel = payload.get("hardstop")
    if hs_panel is not None:
        hs_val = float(hs_panel)
        fig.add_hline(y=hs_val, line_dash="dash", line_color="red", row=2, col=1, secondary_y=False)
        if atr_ratio.size == len(x_vals) and len(x_vals) > 0:
            blocked = np.asarray(np.isfinite(atr_ratio) & (atr_ratio > hs_val), dtype=bool)
            start: int | None = None
            for j, b in enumerate(blocked):
                if b and start is None:
                    start = int(j)
                if (not b) and start is not None:
                    x0 = x_vals[start] if start < len(x_vals) else start
                    x1 = x_vals[j - 1] if j - 1 < len(x_vals) else j - 1
                    fig.add_vrect(
                        x0=x0, x1=x1, fillcolor="rgba(255,0,0,0.08)", line_width=0, row=2, col=1
                    )
                    start = None
            if start is not None and len(x_vals):
                x0 = x_vals[start] if start < len(x_vals) else start
                x1 = x_vals[len(x_vals) - 1]
                fig.add_vrect(
                    x0=x0, x1=x1, fillcolor="rgba(255,0,0,0.08)", line_width=0, row=2, col=1
                )

    fig.update_layout(
        height=620,
        hovermode="closest",
        title=f"Equity + ATR regime — window_{int(current_window):03d}",
    )
    fig.update_yaxes(title_text="Equity", row=1, col=1)
    fig.update_xaxes(title_text="bar/time", row=2, col=1)
    fig.update_yaxes(title_text="ATR_ratio", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="ATR_14", row=2, col=1, secondary_y=True)
    st.plotly_chart(fig, width="stretch")
else:
    st.info("Sem profiles para a janela atual.")

st.subheader("Stats por janela/combinação")
show_all_windows = st.checkbox("Mostrar todas as janelas", value=True)
tbl = metrics_long_df.copy()
if not show_all_windows:
    tbl = tbl[tbl["window_id"].astype(int) == int(current_window)].copy()

tbl = tbl.sort_values(["window_id", "sharpe"], ascending=[True, False]).reset_index(drop=True)
tbl = tbl[
    [
        "window_id",
        "split",
        "profile",
        "roi_pct",
        "final_equity",
        "sharpe",
        "max_dd",
        "n_trades",
        "sl_hits",
        "tp_hits",
        "win_rate",
        "profit_factor",
        "total_pnl",
        "total_fees",
        "entries_blocked",
    ]
]

st.dataframe(
    tbl.style.format(
        {
            "roi_pct": "{:+.2f}%",
            "final_equity": "{:.2f}",
            "sharpe": "{:.3f}",
            "max_dd": "{:.2%}",
            "win_rate": "{:.1f}%",
            "profit_factor": "{:.3f}",
            "total_pnl": "{:.2f}",
            "total_fees": "{:.2f}",
        }
    ),
    width="stretch",
)
