from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import yaml

from .paths import OUTPUTS_DIR, REPO_ROOT


ROI_COL_RE = re.compile(r"^(?P<w>\d{3})_roi$")
DD_COL_RE = re.compile(r"^(?P<w>\d{3})_dd$")
LEGACY_RET_RE = re.compile(r"^window_(?P<w>\d+)_(return|roi)$")
LEGACY_DD_RE = re.compile(r"^window_(?P<w>\d+)_(drawdown|dd)$")


@dataclass(frozen=True)
class RunRef:
    run_id: str
    run_dir: Path


def _coerce_outputs_dir() -> Path:
    return OUTPUTS_DIR


@st.cache_data(show_spinner=False)
def load_runs_summary(path: Optional[str] = None) -> pd.DataFrame:
    p = Path(path) if path is not None else (_coerce_outputs_dir() / "runs_summary.csv")
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"runs_summary.csv not found: {p}")
    df = pd.read_csv(p)
    return normalize_results_df(df)


def normalize_results_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Fold legacy window_* columns into NNN_roi/NNN_dd (keep NNN_* as canonical).
    for col in list(out.columns):
        c = str(col)
        m_ret = LEGACY_RET_RE.match(c)
        if m_ret:
            w = int(m_ret.group("w"))
            new_col = f"{w:03d}_roi"
            if new_col not in out.columns:
                out[new_col] = np.nan
            out[new_col] = out[new_col].combine_first(out[c])
            out = out.drop(columns=[c])
            continue
        m_dd = LEGACY_DD_RE.match(c)
        if m_dd:
            w = int(m_dd.group("w"))
            new_col = f"{w:03d}_dd"
            if new_col not in out.columns:
                out[new_col] = np.nan
            out[new_col] = out[new_col].combine_first(out[c])
            out = out.drop(columns=[c])

    # Ensure numeric types where possible.
    for col in out.columns:
        if ROI_COL_RE.match(str(col)) or DD_COL_RE.match(str(col)) or str(col).startswith("avg_"):
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def list_run_dirs(outputs_dir: Optional[Path] = None) -> list[RunRef]:
    root = outputs_dir or _coerce_outputs_dir()
    if not root.exists():
        return []
    refs: list[RunRef] = []
    for p in sorted(root.glob("run_*")):
        if p.is_dir():
            refs.append(RunRef(run_id=p.name, run_dir=p))
    return refs


def load_config_resolved(run_dir: Path) -> Optional[dict]:
    cfg = run_dir / "config.resolved.yaml"
    if not cfg.exists():
        return None
    with cfg.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_experiment_dir(run_dir: Path, experiment_name: str) -> Path:
    # In multi-experiment mode: run_dir/experiments/<slug>/...
    # Otherwise: artifacts live directly under run_dir.
    exp_root = run_dir / "experiments"
    if not exp_root.exists():
        return run_dir

    # Prefer exact match directory name; fallback to slugified match.
    direct = exp_root / experiment_name
    if direct.is_dir():
        return direct
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", experiment_name).strip("_").lower()
    for cand in exp_root.iterdir():
        if cand.is_dir() and cand.name.lower() == slug:
            return cand
    return exp_root / slug


@st.cache_data(show_spinner=False)
def load_summary_all_windows(exp_dir: Path) -> Optional[pd.DataFrame]:
    p = exp_dir / "summary_all_windows.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def load_window_metrics(window_dir: Path) -> Optional[pd.DataFrame]:
    p = window_dir / "metrics.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def load_signals(window_dir: Path) -> Optional[pd.DataFrame]:
    p = window_dir / "signals.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)

