from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

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
    for col in ("run_id", "experiment_name", "batch_id"):
        if col in out.columns:
            out[col] = out[col].map(lambda x: str(x).strip() if pd.notna(x) else x)
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


def run_id_select_options(runs_df: Optional[pd.DataFrame] = None) -> list[str]:
    """
    Lista de run_id para selectboxes: **ordem igual ao heatmap** (pastas ``run_*`` no disco),
    depois entradas só-no-CSV. Evita ``run_id`` com espaços do CSV a apontar para pastas
    inexistentes (ex.: ``outputs/   run_xxx``).
    """
    from_fs = [r.run_id for r in list_run_dirs()]
    fs_set = set(from_fs)
    extra: list[str] = []
    if runs_df is not None and "run_id" in runs_df.columns:
        for x in runs_df["run_id"].dropna().unique():
            s = str(x).strip()
            if s and s not in fs_set:
                extra.append(s)
        extra = sorted(set(extra))
    return list(from_fs) + extra


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

    name = str(experiment_name).strip()
    # Prefer exact match directory name; fallback to slugified match.
    direct = exp_root / name
    if direct.is_dir():
        return direct

    # Case-insensitive match (Windows / naming drift vs runs_summary).
    name_lower = name.lower()
    for cand in exp_root.iterdir():
        if cand.is_dir() and cand.name.lower() == name_lower:
            return cand

    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    for cand in exp_root.iterdir():
        if cand.is_dir() and cand.name.lower() == slug:
            return cand
    return exp_root / slug


def list_experiment_subdir_names(run_dir: Path) -> list[str]:
    """Nomes das pastas em ``run_dir/experiments`` (vazio se não existir)."""
    exp_root = run_dir / "experiments"
    if not exp_root.is_dir():
        return []
    return sorted(p.name for p in exp_root.iterdir() if p.is_dir())


def experiment_picker_options(run_dir: Path, csv_experiment_names: list[str]) -> list[str]:
    """
    Opções do selectbox: união do que está no disco (``experiments/*``) e do CSV.
    Igual à ideia de ``3_sl_tp_heatmap``: o disco é fonte de verdade para pastas.
    """
    disk = list_experiment_subdir_names(run_dir)
    csv_clean = sorted(
        {str(x).strip() for x in csv_experiment_names if str(x).strip() and str(x).strip() != "."}
    )
    if disk:
        return sorted(set(disk) | set(csv_clean))
    return csv_clean if csv_clean else ["."]


def resolve_exp_dir_like_heatmap(run_dir: Path, experiment_name: str) -> Path:
    """
    O mesmo que ``3_sl_tp_heatmap``:
    ``exp_dir = run_dir`` se ``experiment == '.'``, senão ``run_dir / 'experiments' / nome``.

    Se a pasta directa não existir, recua para ``find_experiment_dir`` (slug / case).
    """
    run_dir = run_dir.resolve()
    exp = str(experiment_name).strip()
    if exp == ".":
        return run_dir
    exp_root = run_dir / "experiments"
    direct = exp_root / exp
    if direct.is_dir():
        return direct.resolve()
    return find_experiment_dir(run_dir, exp)


def resolve_experiment_dir(run_dir: Path, experiment_name: str) -> Path:
    """Alias de :func:`resolve_exp_dir_like_heatmap` (pastas como no heatmap)."""
    return resolve_exp_dir_like_heatmap(run_dir, experiment_name)


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


SignalSplit = Literal["val", "test", "primary"]


def resolve_signals_path(window_dir: Path, split: SignalSplit) -> tuple[Path, str]:
    """
    Resolve which CSV to load for a window directory.

    Returns (path, label) where label is a short description for UI captions.
    """
    wd = Path(window_dir)
    val_p = wd / "signals_val.csv"
    test_p = wd / "signals_test.csv"
    primary_p = wd / "signals.csv"

    if split == "val":
        return val_p, "signals_val.csv"
    if split == "test":
        if test_p.exists():
            return test_p, "signals_test.csv"
        return primary_p, "signals.csv (fallback — signals_test.csv missing)"
    return primary_p, "signals.csv"


@st.cache_data(show_spinner=False)
def load_signals_for_window(window_dir: Path, split: SignalSplit) -> Optional[pd.DataFrame]:
    """
    Load precomputed signals for the given walk-forward window.

    - ``val``: validation slice only (``signals_val.csv``).
    - ``test``: ``signals_test.csv`` when present, else ``signals.csv``.
    - ``primary``: ``signals.csv`` (stable name; often a copy of test when test exists).
    """
    p, _ = resolve_signals_path(window_dir, split)
    if not p.exists():
        return None
    return pd.read_csv(p)

