"""Executed via importlib before any `import dashboard`; stdlib only."""
from __future__ import annotations

import sys
from pathlib import Path


def ensure_sys_path() -> Path:
    """Put repo root and INF/ on sys.path (same targets as dashboard.utils.paths)."""
    dash_dir = Path(__file__).resolve().parent
    root = dash_dir
    for cand in (dash_dir, *dash_dir.parents):
        if (cand / "INF").is_dir():
            root = cand
            break
    for p in (root, root / "INF"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    return root


ensure_sys_path()
