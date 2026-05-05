from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

_bootstrap_path = Path(__file__).resolve().parent.parent / "bootstrap_sys_path.py"
_spec = importlib.util.spec_from_file_location("_nn_dashboard_bootstrap", _bootstrap_path)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(importlib.util.module_from_spec(_spec))

import numpy as np
import pandas as pd

from models import get_action

from dashboard.utils.ohlc_align import align_ohlc_for_engine as _align_ohlc_for_engine
from dashboard.utils.indicators import compute_indicators
from dashboard.utils.regime import classify_regime, compute_regime_features


@dataclass(frozen=True)
class ReplayResult:
    trades: pd.DataFrame
    equities: np.ndarray
    sl_hits: int
    tp_hits: int
    entries: int
    completed_trades: int
    entries_blocked: int = 0


def _ts_at(signals_df: pd.DataFrame, i: int) -> Any:
    if "timestamp" in signals_df.columns:
        return signals_df.iloc[i]["timestamp"]
    if "time" in signals_df.columns:
        return signals_df.iloc[i]["time"]
    return pd.NA


def replay_signals_full(
    signals_df: pd.DataFrame,
    *,
    sl_points: float,
    tp_points: float,
    signal_threshold: float,
    trailing_stop_points: float = 0.0,
    taker_fee: float,
    position_notional: float,
    window_id: Optional[int] = None,
    split: Optional[str] = None,
) -> ReplayResult:
    """
    Replica o loop de ``INF.backtest_engine.run_single_backtest`` e devolve
    trades + curva de equity (paridade com o motor).
    """
    raw, o, h, l, c = _align_ohlc_for_engine(signals_df)
    n = len(raw)
    fee_per_leg = float(position_notional) * float(taker_fee)

    cash = float(position_notional)
    position = 0
    entry_price: Optional[float] = None
    high_watermark: Optional[float] = None
    low_watermark: Optional[float] = None
    pending_entry: Optional[int] = None

    equities: list[float] = []
    sl_hits = 0
    tp_hits = 0
    entries = 0
    completed_trades = 0

    entry_bar: Optional[int] = None
    trades_rows: list[dict[str, Any]] = []

    def mark_to_market(close_price: float) -> float:
        if position == 0 or entry_price is None:
            return cash
        pct = (close_price - entry_price) / entry_price
        return cash + float(position_notional) * (pct if position == 1 else -pct)

    def realize_to(price: float, reason: str, bar_idx: int) -> None:
        nonlocal cash, entry_price, position, completed_trades, high_watermark, low_watermark
        nonlocal entry_bar
        if position == 0 or entry_price is None or entry_bar is None:
            return
        pct = (price - entry_price) / entry_price
        pnl_gross = float(position_notional) * (pct if position == 1 else -pct)
        cash += pnl_gross - fee_per_leg
        side = position
        eb = int(entry_bar)
        tr = {
            "window_id": window_id,
            "split": split,
            "entry_idx": eb,
            "exit_idx": int(bar_idx),
            "entry_ts": _ts_at(signals_df, eb),
            "exit_ts": _ts_at(signals_df, int(bar_idx)),
            "position": int(side),
            "entry_price": float(entry_price),
            "exit_price": float(price),
            "pnl_gross": float(pnl_gross),
            "fees": float(2.0 * fee_per_leg),
            "pnl_net": float(pnl_gross - fee_per_leg),
            "exit_reason": reason,
            "bars_in_trade": int(bar_idx - eb + 1),
            "signal_at_entry": float(raw[eb]),
            "abs_signal_at_entry": float(abs(raw[eb])),
            "sl": float(sl_points),
            "tp": float(tp_points),
            "trailing_stop": float(trailing_stop_points),
            "signal_threshold": float(signal_threshold),
            "taker_fee": float(taker_fee),
            "position_notional": float(position_notional),
        }
        trades_rows.append(tr)
        entry_price = None
        high_watermark = None
        low_watermark = None
        position = 0
        entry_bar = None
        completed_trades += 1

    for i in range(n):
        curr_open = float(o[i])
        curr_high = float(h[i])
        curr_low = float(l[i])
        curr_close = float(c[i])
        closed_by_sl_tp_this_bar = False

        desired = get_action([float(raw[i])], threshold=float(signal_threshold))

        if position == 0 and pending_entry is not None:
            cash -= fee_per_leg
            entry_price = curr_open
            high_watermark = curr_open
            low_watermark = curr_open
            position = pending_entry
            entry_bar = i
            entries += 1
            pending_entry = None

        if position != 0 and entry_price is not None:
            sl_price = entry_price - sl_points if position == 1 else entry_price + sl_points
            tp_price = entry_price + tp_points if position == 1 else entry_price - tp_points
            if trailing_stop_points > 0:
                if position == 1 and high_watermark is not None:
                    sl_price = max(sl_price, high_watermark - trailing_stop_points)
                elif position == -1 and low_watermark is not None:
                    sl_price = min(sl_price, low_watermark + trailing_stop_points)

            if (position == 1 and curr_low <= sl_price) or (position == -1 and curr_high >= sl_price):
                realize_to(sl_price, "sl", i)
                sl_hits += 1
                closed_by_sl_tp_this_bar = True
            elif (position == 1 and curr_high >= tp_price) or (
                position == -1 and curr_low <= tp_price
            ):
                realize_to(tp_price, "tp", i)
                tp_hits += 1
                closed_by_sl_tp_this_bar = True

        if position != 0:
            if high_watermark is None or low_watermark is None:
                high_watermark = curr_open
                low_watermark = curr_open
            high_watermark = max(high_watermark, curr_high)
            low_watermark = min(low_watermark, curr_low)

        if position == 0 and desired != 0:
            if closed_by_sl_tp_this_bar:
                pending_entry = desired
            else:
                cash -= fee_per_leg
                entry_price = curr_open
                high_watermark = curr_open
                low_watermark = curr_open
                position = desired
                entry_bar = i
                entries += 1

        equities.append(mark_to_market(curr_close))

    if position != 0:
        realize_to(float(c[len(equities) - 1]), "eos", len(equities) - 1)

    tdf = pd.DataFrame(trades_rows)
    eq = np.asarray(equities, dtype=np.float64)
    return ReplayResult(
        trades=tdf,
        equities=eq,
        sl_hits=int(sl_hits),
        tp_hits=int(tp_hits),
        entries=int(entries),
        completed_trades=int(completed_trades),
        entries_blocked=0,
    )


def replay_signals_dynamic(
    signals_df: pd.DataFrame,
    *,
    sl_points_per_bar: np.ndarray,
    tp_points_per_bar: np.ndarray,
    signal_threshold: float,
    trailing_stop_points: float = 0.0,
    trailing_stop_per_bar: np.ndarray | None = None,
    taker_fee: float,
    position_notional: float,
    entry_mask: np.ndarray | None = None,
    window_id: Optional[int] = None,
    split: Optional[str] = None,
) -> ReplayResult:
    """
    Variante de replay com SL/TP dinâmicos por barra e máscara de entrada.

    - ``sl_points_per_bar[i]`` / ``tp_points_per_bar[i]`` define distância em pontos
      para entradas abertas na barra ``i``.
    - ``trailing_stop_per_bar[i]``: se fornecido, o trailing stop da trade é lido no
      bar de entrada e mantido fixo durante toda a trade. Tem prioridade sobre
      ``trailing_stop_points``.
    - ``entry_mask[i] == False`` bloqueia qualquer entrada (imediata ou pendente) na barra ``i``.
    """
    raw, o, h, l, c = _align_ohlc_for_engine(signals_df)
    n = len(raw)

    sl_arr = np.asarray(sl_points_per_bar, dtype=np.float64)
    tp_arr = np.asarray(tp_points_per_bar, dtype=np.float64)
    if sl_arr.shape[0] != n or tp_arr.shape[0] != n:
        raise ValueError(
            f"sl/tp arrays devem ter len={n}; recebido sl={sl_arr.shape[0]} tp={tp_arr.shape[0]}"
        )
    if np.any(~np.isfinite(sl_arr)) or np.any(~np.isfinite(tp_arr)):
        raise ValueError("sl/tp arrays contêm valores não finitos")
    if np.any(sl_arr <= 0.0) or np.any(tp_arr <= 0.0):
        raise ValueError("sl/tp arrays devem ser > 0 em todas as barras")

    trail_arr: np.ndarray | None = None
    if trailing_stop_per_bar is not None:
        trail_arr = np.asarray(trailing_stop_per_bar, dtype=np.float64)
        if trail_arr.shape[0] != n:
            raise ValueError(f"trailing_stop_per_bar deve ter len={n}; recebido {trail_arr.shape[0]}")

    if entry_mask is None:
        mask = np.ones(n, dtype=bool)
    else:
        mask = np.asarray(entry_mask, dtype=bool)
        if mask.shape[0] != n:
            raise ValueError(f"entry_mask deve ter len={n}; recebido {mask.shape[0]}")

    fee_per_leg = float(position_notional) * float(taker_fee)
    cash = float(position_notional)
    position = 0
    entry_price: Optional[float] = None
    high_watermark: Optional[float] = None
    low_watermark: Optional[float] = None
    pending_entry: Optional[int] = None
    entry_bar: Optional[int] = None
    sl_dist: Optional[float] = None
    tp_dist: Optional[float] = None
    entry_trail_dist: Optional[float] = None

    equities: list[float] = []
    sl_hits = 0
    tp_hits = 0
    entries = 0
    completed_trades = 0
    entries_blocked = 0
    trades_rows: list[dict[str, Any]] = []

    def mark_to_market(close_price: float) -> float:
        if position == 0 or entry_price is None:
            return cash
        pct = (close_price - entry_price) / entry_price
        return cash + float(position_notional) * (pct if position == 1 else -pct)

    def realize_to(price: float, reason: str, bar_idx: int) -> None:
        nonlocal cash, entry_price, position, completed_trades, high_watermark, low_watermark
        nonlocal entry_bar, sl_dist, tp_dist, entry_trail_dist
        if position == 0 or entry_price is None or entry_bar is None:
            return
        pct = (price - entry_price) / entry_price
        pnl_gross = float(position_notional) * (pct if position == 1 else -pct)
        cash += pnl_gross - fee_per_leg
        side = position
        eb = int(entry_bar)
        tr = {
            "window_id": window_id,
            "split": split,
            "entry_idx": eb,
            "exit_idx": int(bar_idx),
            "entry_ts": _ts_at(signals_df, eb),
            "exit_ts": _ts_at(signals_df, int(bar_idx)),
            "position": int(side),
            "entry_price": float(entry_price),
            "exit_price": float(price),
            "pnl_gross": float(pnl_gross),
            "fees": float(2.0 * fee_per_leg),
            "pnl_net": float(pnl_gross - fee_per_leg),
            "exit_reason": reason,
            "bars_in_trade": int(bar_idx - eb + 1),
            "signal_at_entry": float(raw[eb]),
            "abs_signal_at_entry": float(abs(raw[eb])),
            "sl": float(sl_dist) if sl_dist is not None else np.nan,
            "tp": float(tp_dist) if tp_dist is not None else np.nan,
            "sl_dist": float(sl_dist) if sl_dist is not None else np.nan,
            "tp_dist": float(tp_dist) if tp_dist is not None else np.nan,
            "trailing_stop": float(entry_trail_dist) if entry_trail_dist is not None else float(trailing_stop_points),
            "signal_threshold": float(signal_threshold),
            "taker_fee": float(taker_fee),
            "position_notional": float(position_notional),
        }
        trades_rows.append(tr)
        entry_price = None
        high_watermark = None
        low_watermark = None
        position = 0
        entry_bar = None
        sl_dist = None
        tp_dist = None
        entry_trail_dist = None
        completed_trades += 1

    for i in range(n):
        curr_open = float(o[i])
        curr_high = float(h[i])
        curr_low = float(l[i])
        curr_close = float(c[i])
        closed_by_sl_tp_this_bar = False

        desired = get_action([float(raw[i])], threshold=float(signal_threshold))
        allow_entry_i = bool(mask[i])

        if pending_entry is not None and not allow_entry_i:
            pending_entry = None
            entries_blocked += 1

        if position == 0 and pending_entry is not None and allow_entry_i:
            cash -= fee_per_leg
            entry_price = curr_open
            high_watermark = curr_open
            low_watermark = curr_open
            position = pending_entry
            entry_bar = i
            sl_dist = float(sl_arr[i])
            tp_dist = float(tp_arr[i])
            entry_trail_dist = float(trail_arr[i]) if trail_arr is not None else None
            entries += 1
            pending_entry = None

        if position != 0 and entry_price is not None:
            assert sl_dist is not None and tp_dist is not None
            sl_price = entry_price - sl_dist if position == 1 else entry_price + sl_dist
            tp_price = entry_price + tp_dist if position == 1 else entry_price - tp_dist
            active_trail = entry_trail_dist if entry_trail_dist is not None else trailing_stop_points
            if active_trail > 0:
                if position == 1 and high_watermark is not None:
                    sl_price = max(sl_price, high_watermark - active_trail)
                elif position == -1 and low_watermark is not None:
                    sl_price = min(sl_price, low_watermark + active_trail)

            if (position == 1 and curr_low <= sl_price) or (position == -1 and curr_high >= sl_price):
                realize_to(sl_price, "sl", i)
                sl_hits += 1
                closed_by_sl_tp_this_bar = True
            elif (position == 1 and curr_high >= tp_price) or (
                position == -1 and curr_low <= tp_price
            ):
                realize_to(tp_price, "tp", i)
                tp_hits += 1
                closed_by_sl_tp_this_bar = True

        if position != 0:
            if high_watermark is None or low_watermark is None:
                high_watermark = curr_open
                low_watermark = curr_open
            high_watermark = max(high_watermark, curr_high)
            low_watermark = min(low_watermark, curr_low)

        if position == 0 and desired != 0:
            if not allow_entry_i:
                entries_blocked += 1
            elif closed_by_sl_tp_this_bar:
                pending_entry = desired
            else:
                cash -= fee_per_leg
                entry_price = curr_open
                high_watermark = curr_open
                low_watermark = curr_open
                position = desired
                entry_bar = i
                sl_dist = float(sl_arr[i])
                tp_dist = float(tp_arr[i])
                entries += 1

        equities.append(mark_to_market(curr_close))

    if position != 0:
        realize_to(float(c[len(equities) - 1]), "eos", len(equities) - 1)

    tdf = pd.DataFrame(trades_rows)
    eq = np.asarray(equities, dtype=np.float64)
    return ReplayResult(
        trades=tdf,
        equities=eq,
        sl_hits=int(sl_hits),
        tp_hits=int(tp_hits),
        entries=int(entries),
        completed_trades=int(completed_trades),
        entries_blocked=int(entries_blocked),
    )


def replay_signals_to_trades(
    signals_df: pd.DataFrame,
    *,
    sl_points: float,
    tp_points: float,
    signal_threshold: float,
    trailing_stop_points: float = 0.0,
    taker_fee: float,
    position_notional: float,
    window_id: Optional[int] = None,
    split: Optional[str] = None,
) -> pd.DataFrame:
    return replay_signals_full(
        signals_df,
        sl_points=float(sl_points),
        tp_points=float(tp_points),
        signal_threshold=float(signal_threshold),
        trailing_stop_points=float(trailing_stop_points),
        taker_fee=float(taker_fee),
        position_notional=float(position_notional),
        window_id=window_id,
        split=split,
    ).trades


def attach_features_at_entry(
    trades_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    indicators: Optional[list[str]] = None,
) -> pd.DataFrame:
    if trades_df.empty:
        return trades_df
    inds = indicators or ["ATR_14", "ATR_200", "RSI_14", "MACDh_12_26_9", "volume"]
    need_cols = set(inds) - {"volume"}
    sig = signals_df.copy()
    if need_cols & {"ATR_14", "ATR_200", "RSI_14", "MACDh_12_26_9", "MACD_hist", "MACD_12_26_9"}:
        compute_list = [x for x in inds if x != "volume"]
        sig = compute_indicators(sig, compute_list)
    if "volume" in inds and "volume" in sig.columns:
        pass  # already present
    out = trades_df.copy()
    ei = out["entry_idx"].astype(int)

    for col in inds:
        if col in sig.columns:
            out[f"entry_{col}"] = sig.iloc[ei][col].to_numpy()

    return out


def attach_regime(trades_df: pd.DataFrame, signals_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return trades_df
    feats = compute_regime_features(signals_df[["open", "high", "low", "close"]].copy())
    ei = trades_df["entry_idx"].astype(int)
    out = trades_df.copy()
    out["atr_ratio_at_entry"] = feats["atr_ratio"].iloc[ei].to_numpy()
    out["regime_name"] = feats["regime_name"].iloc[ei].to_numpy()
    labels, _ = classify_regime(feats["atr_ratio"])
    out["regime_label"] = labels.iloc[ei].to_numpy()
    return out


def build_trades_detailed(
    signals_df: pd.DataFrame,
    *,
    sl_points: float,
    tp_points: float,
    signal_threshold: float,
    trailing_stop_points: float,
    taker_fee: float,
    position_notional: float,
    window_id: Optional[int] = None,
    split: Optional[str] = None,
    feature_indicators: Optional[list[str]] = None,
) -> pd.DataFrame:
    tdf = replay_signals_to_trades(
        signals_df,
        sl_points=sl_points,
        tp_points=tp_points,
        signal_threshold=signal_threshold,
        trailing_stop_points=trailing_stop_points,
        taker_fee=taker_fee,
        position_notional=position_notional,
        window_id=window_id,
        split=split,
    )
    tdf = attach_features_at_entry(tdf, signals_df, feature_indicators)
    tdf = attach_regime(tdf, signals_df)
    return tdf
