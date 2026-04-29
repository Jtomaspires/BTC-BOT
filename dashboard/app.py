from __future__ import annotations

import importlib.util
from pathlib import Path

_bootstrap_path = Path(__file__).resolve().parent / "bootstrap_sys_path.py"
_spec = importlib.util.spec_from_file_location("_nn_dashboard_bootstrap", _bootstrap_path)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(importlib.util.module_from_spec(_spec))

import streamlit as st

from dashboard.utils.state import ensure_session_keys


st.set_page_config(
    page_title="NN — Walk-forward Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

ensure_session_keys()

st.title("Walk-forward Dashboard")
st.caption("Explora resultados existentes em INF/outputs de forma interactiva.")

st.markdown(
    """
Use as páginas no menu do Streamlit (sidebar) para navegar:

- Overview (experiments)
- Window analysis (forensics)
- SL/TP heatmap (signals.csv)
- Correlations
- Timeline
- Compare
"""
)

