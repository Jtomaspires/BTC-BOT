from __future__ import annotations

import sys
from pathlib import Path

# Repo root + INF on path (dashboard imports `models`, `INF.metrics`, …)
_dash = Path(__file__).resolve().parent.parent
_root = _dash.parent
for p in (_root, _root / "INF"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
