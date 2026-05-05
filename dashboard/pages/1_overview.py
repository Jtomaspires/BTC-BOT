from __future__ import annotations

import importlib.util
import re
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

st.caption(
    "O CSV guarda ROI/DD como **razão** (ex.: 0,0503 = 5,03%). Abaixo podes ver o equivalente em percentagem "
    "estilo PT, e agregados sobre as **últimas N janelas com ROI válido** por linha (ordem 000 → …), "
    "para não misturar runs curtos com colunas vazias no fim do CSV."
)


def fmt_pt_pct_ratio(x: float | None) -> str:
    """Formata razão (0.0503) como percentagem com vírgula decimal (5,03%)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    v = float(x) * 100.0
    s = f"{v:.2f}".replace(".", ",")
    return f"{s}%"


def fmt_pt_number(x: float | None, *, decimals: int = 4) -> str:
    """Número com vírgula decimal (ex. quociente ROI/DD)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    s = f"{float(x):.{decimals}f}".replace(".", ",")
    return s


def window_ids_sorted(roi_cols: list[str]) -> list[int]:
    out: list[int] = []
    for c in roi_cols:
        m = re.match(r"^(\d{3})_roi$", str(c))
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


df = load_runs_summary()

roi_cols = [c for c in df.columns if isinstance(c, str) and c.endswith("_roi") and len(c) == 7]
if not roi_cols:
    st.error("Não encontrei colunas NNN_roi no runs_summary.csv.")
    st.stop()

roi_cols = sorted(roi_cols, key=lambda c: int(str(c)[:3]))
window_ids = window_ids_sorted(roi_cols)
n_windows = len(window_ids)

roi_mat = df[roi_cols].to_numpy(dtype=float)
df = df.copy()
df["pct_windows_positive"] = np.nanmean(roi_mat > 0.0, axis=1) * 100.0
df["worst_window"] = np.nanargmin(roi_mat, axis=1)
df["best_window"] = np.nanargmax(roi_mat, axis=1)

dd_cols_ordered = [f"{int(str(c)[:3]):03d}_dd" for c in roi_cols]
dd_cols_ordered = [c for c in dd_cols_ordered if c in df.columns]
dd_mat = df[dd_cols_ordered].to_numpy(dtype=float) if dd_cols_ordered else None

with st.sidebar:
    st.subheader("Últimas N janelas")
    default_n = min(12, n_windows) if n_windows else 1
    last_n = st.slider(
        "N (para médias ROI/DD nas últimas N janelas)",
        min_value=1,
        max_value=max(1, n_windows),
        value=default_n,
        help="Por linha: média só sobre as últimas N janelas **com ROI válido** (ignora NaN no fim do CSV).",
    )
    st.subheader("Resumo — runs e linhas")
    all_run_ids = sorted(df["run_id"].dropna().astype(str).unique().tolist()) if "run_id" in df.columns else []
    selected_runs = st.multiselect(
        "Runs (run_id) a incluir",
        options=all_run_ids,
        default=all_run_ids,
        help="Escolhe quais backtests entram na tabela de resumo.",
    )
    cap_windows = st.checkbox(
        "Só runs com ≤ X janelas com dados",
        value=True,
        help="Útil quando o CSV tem muitas colunas NNN_roi mas só alguns runs longos; "
        "a maioria (~24 janelas) fica abaixo do limiar.",
    )
    max_windows_data = st.number_input(
        "X (máx. janelas com ROI válido)",
        min_value=1,
        max_value=500,
        value=30,
        step=1,
        disabled=not cap_windows,
    )
    max_summary_rows = st.number_input(
        "Máx. linhas na tabela resumo",
        min_value=5,
        max_value=500,
        value=50,
        step=5,
    )
    st.subheader("Tabela por janela")
    show_all_windows = st.checkbox("Mostrar todas as janelas na tabela larga", value=False)
    if show_all_windows:
        visible_windows = window_ids
    else:
        visible_windows = window_ids[-last_n:] if window_ids else []

n_rows = len(df)
_roi_tail = np.full(n_rows, np.nan, dtype=np.float64)
_dd_tail = np.full(n_rows, np.nan, dtype=np.float64)
_roi_over_dd_tail = np.full(n_rows, np.nan, dtype=np.float64)
_n_data = np.zeros(n_rows, dtype=np.int32)
eps = 1e-12
for i in range(n_rows):
    row_roi = roi_mat[i, :]
    idx = np.flatnonzero(np.isfinite(row_roi))
    _n_data[i] = int(idx.size)
    if idx.size == 0:
        continue
    tail = idx[-last_n:]
    _roi_tail[i] = float(np.nanmean(row_roi[tail]))
    if dd_mat is not None:
        _dd_tail[i] = float(np.nanmean(dd_mat[i, tail]))
        if (
            np.isfinite(_roi_tail[i])
            and np.isfinite(_dd_tail[i])
            and abs(float(_dd_tail[i])) > eps
        ):
            _roi_over_dd_tail[i] = float(_roi_tail[i]) / float(_dd_tail[i])
df["_roi_mean_last_n"] = _roi_tail
df["_dd_mean_last_n"] = _dd_tail
df["_roi_over_dd_last_n"] = _roi_over_dd_tail
df["_n_windows_data"] = _n_data

summary_base = [
    "run_id",
    "config_name",
    "experiment_name",
    "_n_windows_data",
    "avg_return",
    "avg_drawdown",
    "avg_sharpe",
    "pct_windows_positive",
    "worst_window",
    "best_window",
    "_roi_mean_last_n",
    "_dd_mean_last_n",
    "_roi_over_dd_last_n",
]
summary_cols = [c for c in summary_base if c in df.columns]

with st.expander("Resumo (métricas + últimas N janelas)", expanded=True):
    filt = df
    if "run_id" in filt.columns:
        if not selected_runs:
            st.warning("Selecciona pelo menos um run_id na sidebar.")
            filt = filt.iloc[0:0]
        else:
            filt = filt[filt["run_id"].astype(str).isin(selected_runs)]
    if cap_windows and "_n_windows_data" in filt.columns and len(filt):
        filt = filt[filt["_n_windows_data"] <= int(max_windows_data)]
    view = filt[summary_cols].sort_values(
        ["avg_return", "pct_windows_positive"],
        ascending=[False, False],
        kind="stable",
    ).head(int(max_summary_rows))
    disp = view.copy()
    for c in ("avg_return", "avg_drawdown", "_roi_mean_last_n", "_dd_mean_last_n"):
        if c in disp.columns:
            disp[c + " (%)"] = disp[c].map(fmt_pt_pct_ratio)
    drop_numeric_pct = [c for c in ("avg_return", "avg_drawdown", "_roi_mean_last_n", "_dd_mean_last_n") if c in disp.columns]
    disp = disp.drop(columns=drop_numeric_pct, errors="ignore")
    if "_roi_over_dd_last_n" in disp.columns:
        disp["ROI/DD (últ. N)"] = disp["_roi_over_dd_last_n"].map(lambda x: fmt_pt_number(x, decimals=4))
        disp = disp.drop(columns=["_roi_over_dd_last_n"], errors="ignore")
    st.caption(
        "**ROI/DD (últ. N):** (média do ROI nas últimas N janelas com dados) ÷ (média do max DD nas **mesmas** janelas). "
        "Valores em razão (como o CSV); quanto maior, melhor retorno por unidade de drawdown nesse bloco."
    )
    st.dataframe(disp, width="stretch", hide_index=True)

with st.expander("Gráficos", expanded=True):
    st.markdown("##### ROI médio por experiment")
    if "experiment_name" in df.columns and "avg_return" in df.columns:
        grouped = (
            df.groupby("experiment_name", dropna=False, as_index=False)["avg_return"]
            .mean()
            .sort_values("avg_return", ascending=False)
        )
        grouped["color"] = np.where(grouped["avg_return"] >= 0, "positive", "negative")
        fig = px.bar(
            grouped,
            x="experiment_name",
            y="avg_return",
            color="color",
            color_discrete_map={"positive": "green", "negative": "red"},
        )
        fig.update_layout(xaxis_title="experiment", yaxis_title="avg_return (razão)")
        fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig, width="stretch")

    st.markdown("##### Distribuição de ROI por janela")
    long = df[["experiment_name"] + roi_cols].melt(id_vars=["experiment_name"], var_name="window", value_name="roi")
    long["window"] = long["window"].str.replace("_roi", "", regex=False)
    fig2 = px.box(long, x="experiment_name", y="roi", points="outliers")
    fig2.update_layout(xaxis_title="experiment", yaxis_title="window ROI (razão)")
    fig2.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig2, width="stretch")

with st.expander("Detalhe por janela (ROI / DD formatados)", expanded=False):
    if not visible_windows:
        st.info("Sem janelas seleccionadas.")
    else:
        pair_cols: list[str] = []
        for w in visible_windows:
            r, d = f"{w:03d}_roi", f"{w:03d}_dd"
            if r in df.columns:
                pair_cols.append(r)
            if d in df.columns:
                pair_cols.append(d)
        meta = [c for c in ("run_id", "config_name", "experiment_name") if c in df.columns]
        extra_sort = ["avg_return"] if "avg_return" in df.columns else []
        tbl = df[meta + pair_cols + extra_sort].copy()
        if "avg_return" in tbl.columns:
            tbl = tbl.sort_values("avg_return", ascending=False, kind="stable").drop(columns=["avg_return"])
        elif meta:
            tbl = tbl.sort_values(meta[0], kind="stable")
        disp2 = tbl.copy()
        for c in pair_cols:
            if c.endswith("_roi") or c.endswith("_dd"):
                disp2[c] = disp2[c].map(fmt_pt_pct_ratio)
        st.dataframe(disp2, width="stretch", hide_index=True)

st.divider()
st.subheader("Atalho")
st.caption("Define seleção global (session_state) para usar noutras páginas.")
run_id = st.selectbox(
    "run_id",
    sorted(df["run_id"].dropna().astype(str).unique().tolist()) if "run_id" in df.columns else [],
    index=0,
)
exp = st.selectbox(
    "experiment",
    sorted(df["experiment_name"].dropna().astype(str).unique().tolist()) if "experiment_name" in df.columns else [],
    index=0,
)
win = st.number_input("window", min_value=0, max_value=999, value=0, step=1)
if st.button("Set selection"):
    set_selection(run_id=run_id, experiment=exp, window=int(win))
    st.success("Selection saved.")

with st.expander("Nota: colunas colapsáveis no Streamlit", expanded=False):
    st.markdown(
        """
O **Streamlit** não tem equivalente directo ao *outline* / colunas colapsáveis do Excel dentro
de um único `st.dataframe`.

**Alternativas comuns:**

1. **`st.expander`** (como acima) — agrupar blocos de tabela ou métricas.
2. **Multiselect / slider** — escolher quais janelas ou que subset de colunas mostrar.
3. **`streamlit-aggrid`** — tabelas estilo Excel com *column groups* e pin/filtros (dependência extra).
4. **Várias abas** (`st.tabs`) — ex.: *Resumo* | *Por janela* | *Export*.
"""
    )
