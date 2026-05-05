from __future__ import annotations

import importlib.util
from pathlib import Path

_bootstrap_path = Path(__file__).resolve().parent.parent / "bootstrap_sys_path.py"
_spec = importlib.util.spec_from_file_location("_nn_dashboard_bootstrap", _bootstrap_path)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(importlib.util.module_from_spec(_spec))

from dataclasses import dataclass
from typing import Iterable, Literal, Optional

import numpy as np
import pandas as pd

from dashboard.utils.trades import replay_signals_full

from INF.metrics import max_drawdown_info, sharpe_ratio


@dataclass(frozen=True)
class CellMetrics:
    equity: float
    sharpe: float
    max_dd: float
    robustness: float


def _neighbor_mean(mat: np.ndarray) -> np.ndarray:
    """Mean of 8-neighborhood (excluding self); NaNs ignored."""
    h, w = mat.shape
    out = np.full((h, w), np.nan, dtype=np.float64)
    for i in range(h):
        for j in range(w):
            vals = []
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ii, jj = i + di, j + dj
                    if 0 <= ii < h and 0 <= jj < w:
                        v = mat[ii, jj]
                        if np.isfinite(v):
                            vals.append(float(v))
            out[i, j] = float(np.mean(vals)) if vals else np.nan
    return out


def run_grid(
    *,
    signals_df: pd.DataFrame,
    sl_values: Iterable[float],
    tp_values: Iterable[float],
    threshold_values: Iterable[float],
    trailing_stop_values: Iterable[float],
    window_dir: Optional[Path] = None,  # mantido por compatibilidade; já não é necessário
    taker_fee: float = 0.00055,
    position_notional: float = 1000.0,
    periods_per_year: float = 365.0 * 24.0,
) -> dict[tuple[float, float, float, float], CellMetrics]:
    """
    Runs a SL/TP grid using precomputed signals.csv (no model inference).

    Resultado **bit-exato** com ``INF/run_walkforward`` quando ``signal_threshold``,
    ``taker_fee``, ``position_notional`` e ``trailing_stop_points`` são iguais aos do
    config — ver ``dashboard.utils.ohlc_align.align_ohlc_for_engine`` para o porquê.

    Returns a dict keyed by (sl, tp, threshold, trailing_stop).
    Robustness é a média da vizinhança 8 da equity por (threshold, trailing_stop).
    """
    del window_dir  # noqa: F841 — argumento legado, sem uso após alinhamento bit-exato

    required = {"signal", "open", "high", "low", "close"}
    missing = required - set(signals_df.columns)
    if missing:
        raise ValueError(f"signals_df missing columns: {sorted(missing)}")

    sls = [float(x) for x in sl_values]
    tps = [float(x) for x in tp_values]
    thrs = [float(x) for x in threshold_values]
    trails = [float(x) for x in trailing_stop_values]

    out: dict[tuple[float, float, float, float], CellMetrics] = {}

    for thr in thrs:
        for trail in trails:
            equity_mat = np.full((len(sls), len(tps)), np.nan, dtype=np.float64)
            cell_tmp: dict[tuple[float, float], tuple[float, float, float]] = {}
            for i, sl in enumerate(sls):
                for j, tp in enumerate(tps):
                    rep = replay_signals_full(
                        signals_df,
                        sl_points=float(sl),
                        tp_points=float(tp),
                        signal_threshold=float(thr),
                        trailing_stop_points=float(trail),
                        taker_fee=float(taker_fee),
                        position_notional=float(position_notional),
                    )
                    res_eq = rep.equities
                    eq_final = float(res_eq[-1]) if len(res_eq) else float(position_notional)
                    sh = float(sharpe_ratio(res_eq, periods_per_year=periods_per_year))
                    dd = float(max_drawdown_info(res_eq).get("max_drawdown", 0.0))
                    cell_tmp[(float(sl), float(tp))] = (eq_final, sh, dd)
                    equity_mat[i, j] = eq_final

            robust = _neighbor_mean(equity_mat)
            for i, sl in enumerate(sls):
                for j, tp in enumerate(tps):
                    eq_final, sh, dd = cell_tmp.get((sl, tp), (float("nan"), float("nan"), float("nan")))
                    out[(sl, tp, thr, trail)] = CellMetrics(
                        equity=eq_final,
                        sharpe=sh,
                        max_dd=dd,
                        robustness=float(robust[i, j]) if np.isfinite(robust[i, j]) else float("nan"),
                    )

    return out


GridMetric = Literal["sharpe", "equity", "robustness"]


def _metric_scalar(m: CellMetrics, metric: str) -> float:
    key = str(metric).strip().lower()
    if key == "sharpe":
        return float(m.sharpe)
    if key in {"equity", "final_equity"}:
        return float(m.equity)
    if key in {"robustness", "robust"}:
        return float(m.robustness)
    raise ValueError(
        f"metric inválido: {metric!r}. Use 'sharpe', 'equity' ou 'robustness'."
    )


def find_best_cell(
    out: dict[tuple[float, float, float, float], CellMetrics],
    metric: GridMetric | str = "sharpe",
) -> tuple[tuple[float, float, float, float], CellMetrics, float]:
    """
    Melhor célula do grid completo (SL, TP, threshold, trailing).

    Desempate determinístico: maior score; em empate aproximado (``rtol=1e-9``),
    escolhe a chave lexicográfica maior ``(sl, tp, thr, trail)``.
    """
    if not out:
        raise ValueError("grid vazio")

    best_key: tuple[float, float, float, float] | None = None
    best_m: CellMetrics | None = None
    best_score: float | None = None

    for key, m in out.items():
        score = _metric_scalar(m, metric)
        if not np.isfinite(score):
            continue
        sl, tp, thr, trail = (float(key[0]), float(key[1]), float(key[2]), float(key[3]))
        tup_key = (sl, tp, thr, trail)

        if best_score is None:
            best_key, best_m, best_score = tup_key, m, score
            continue

        assert best_key is not None
        if score > best_score or (
            np.isclose(score, best_score, rtol=1e-9, atol=0.0) and tup_key > best_key
        ):
            best_key, best_m, best_score = tup_key, m, score

    if best_key is None or best_m is None or best_score is None:
        raise ValueError("Nenhuma célula com métrica finita no grid.")
    return best_key, best_m, float(best_score)
