from __future__ import annotations

import sys
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for cand in (p, *p.parents):
        if (cand / "INF").is_dir():
            return cand
    return p


def ensure_repo_on_path() -> Path:
    """
    Ensure repo root and INF/ are on sys.path.

    Streamlit executes scripts under dashboard/, which puts only `dashboard/` on sys.path.
    Adding repo root enables `import dashboard...`.
    Adding INF/ preserves the project's flat imports (`models`, `features`, ...) used when
    cwd is INF/ for training scripts.
    """
    root = find_repo_root()
    for p in (root, root / "INF"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    return root


REPO_ROOT = ensure_repo_on_path()
INF_DIR = REPO_ROOT / "INF"
OUTPUTS_DIR = INF_DIR / "outputs"

