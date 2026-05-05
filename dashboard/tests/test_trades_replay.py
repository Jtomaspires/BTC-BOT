from __future__ import annotations

import numpy as np
import pandas as pd

from INF.backtest_engine import run_single_backtest

from dashboard.utils.ohlc_align import align_ohlc_for_engine
from dashboard.utils.trades import replay_signals_dynamic, replay_signals_full


def _synthetic_signals(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    price = 100 + np.cumsum(rng.normal(0, 0.5, size=n))
    o = price + rng.normal(0, 0.1, size=n)
    h = np.maximum(o, price) + np.abs(rng.normal(0, 0.2, size=n))
    l = np.minimum(o, price) - np.abs(rng.normal(0, 0.2, size=n))
    c = price
    sig = rng.normal(0, 0.02, size=n)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "signal": sig, "volume": rng.random(n) * 1e3})


def test_replay_parity_with_engine() -> None:
    sig = _synthetic_signals(200)
    raw, o, h, l, c = align_ohlc_for_engine(sig)
    sl, tp, thr, trail = 1.5, 2.5, 0.008, 0.0
    fee, nom = 0.00055, 1000.0

    eng = run_single_backtest(
        raw,
        o,
        h,
        l,
        c,
        sl_points=sl,
        tp_points=tp,
        taker_fee=fee,
        position_notional=nom,
        signal_threshold=thr,
        trailing_stop_points=trail,
    )
    rep = replay_signals_full(
        sig,
        sl_points=sl,
        tp_points=tp,
        signal_threshold=thr,
        trailing_stop_points=trail,
        taker_fee=fee,
        position_notional=nom,
    )

    np.testing.assert_allclose(rep.equities, eng.equities, rtol=0, atol=1e-9)
    assert rep.sl_hits == eng.sl_hits
    assert rep.tp_hits == eng.tp_hits
    assert rep.entries == eng.entries
    assert rep.completed_trades == eng.completed_trades
    np.testing.assert_allclose(
        rep.trades["pnl_net"].to_numpy(),
        np.asarray(eng.trade_pnls, dtype=np.float64),
        rtol=0,
        atol=1e-9,
    )


def test_replay_dynamic_matches_static_when_arrays_constant() -> None:
    sig = _synthetic_signals(200)
    sl, tp, thr, trail = 4.0, 75.0, 0.007, 10.0
    fee, nom = 0.00055, 1000.0
    n = len(sig)
    sl_arr = np.full(n, sl, dtype=np.float64)
    tp_arr = np.full(n, tp, dtype=np.float64)
    mask = np.ones(n, dtype=bool)

    rep_static = replay_signals_full(
        sig,
        sl_points=sl,
        tp_points=tp,
        signal_threshold=thr,
        trailing_stop_points=trail,
        taker_fee=fee,
        position_notional=nom,
    )
    rep_dynamic = replay_signals_dynamic(
        sig,
        sl_points_per_bar=sl_arr,
        tp_points_per_bar=tp_arr,
        signal_threshold=thr,
        trailing_stop_points=trail,
        taker_fee=fee,
        position_notional=nom,
        entry_mask=mask,
    )

    np.testing.assert_allclose(rep_dynamic.equities, rep_static.equities, rtol=0, atol=1e-9)
    assert rep_dynamic.sl_hits == rep_static.sl_hits
    assert rep_dynamic.tp_hits == rep_static.tp_hits
    assert rep_dynamic.entries == rep_static.entries
    assert rep_dynamic.completed_trades == rep_static.completed_trades
    np.testing.assert_allclose(
        rep_dynamic.trades["pnl_net"].to_numpy(),
        rep_static.trades["pnl_net"].to_numpy(),
        rtol=0,
        atol=1e-9,
    )
