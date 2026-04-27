"""
Feature engineering modular para walk-forward.

Suporta:
- Features base de price-action (compatibilidade com o setup anterior).
- Seleção de subset por nome via config.
- Indicadores técnicos adicionais (RSI, MACD, ATR, etc.) sem look-ahead.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler

DEFAULT_FEATURE_NAMES = [
    "body_over_open",
    "high_minus_open_over_open",
    "high_minus_close_over_close",
    "low_minus_open_over_open",
    "low_minus_close_over_close",
    "range_over_open",
    "range_over_close",
    "volume",
]
# Compatibilidade com call-sites antigos.
FEATURE_NAMES = list(DEFAULT_FEATURE_NAMES)
NUM_FEATURES = len(DEFAULT_FEATURE_NAMES)

FEATURE_ALIASES = {
    # aliases comuns
    "body_pct": "body_over_open",
    "upper_wick": "high_minus_close_over_close",
    "lower_wick": "low_minus_open_over_open",
    "range_pct": "range_over_open",
}


def _sanitize_feature(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0.0, np.nan)


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # Quando não há perdas no período, RSI tende para 100.
    rsi = rsi.where(~((avg_loss == 0.0) & (avg_gain > 0.0)), 100.0)
    return rsi.fillna(0.0)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean().fillna(0.0)


def _build_feature_catalog(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
) -> Dict[str, np.ndarray]:
    o = pd.Series(np.asarray(opens, dtype=np.float64))
    h = pd.Series(np.asarray(highs, dtype=np.float64))
    l = pd.Series(np.asarray(lows, dtype=np.float64))
    c = pd.Series(np.asarray(closes, dtype=np.float64))
    v = pd.Series(np.asarray(volumes, dtype=np.float64))

    roll_low_14 = l.rolling(window=14, min_periods=14).min()
    roll_high_14 = h.rolling(window=14, min_periods=14).max()
    so = 100.0 * _safe_div(c - roll_low_14, roll_high_14 - roll_low_14)

    roi_2 = c.pct_change(2)
    roi_3 = c.pct_change(3)
    roi_7 = c.pct_change(7)

    ema12 = c.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = c.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    macd_hist = macd - macd_signal

    differenced = c.pct_change()
    previous_differenced = differenced.shift(1)

    vol20 = _safe_div(c.rolling(window=20, min_periods=20).std(), c.rolling(window=20, min_periods=20).mean())
    vol50 = _safe_div(c.rolling(window=50, min_periods=50).std(), c.rolling(window=50, min_periods=50).mean())

    roll_min_60 = c.rolling(window=60, min_periods=60).min()
    roll_max_60 = c.rolling(window=60, min_periods=60).max()
    pos_range_60 = _safe_div(c - roll_min_60, roll_max_60 - roll_min_60)
    neg_range_60 = _safe_div(roll_max_60 - c, roll_max_60 - roll_min_60)

    catalog_series: Dict[str, pd.Series] = {
        # Features base (compat)
        "body_over_open": _safe_div(c - o, o),
        "high_minus_open_over_open": _safe_div(h - o, o),
        "high_minus_close_over_close": _safe_div(h - c, c),
        "low_minus_open_over_open": _safe_div(l - o, o),
        "low_minus_close_over_close": _safe_div(l - c, c),
        "range_over_open": _safe_div(h - l, o),
        "range_over_close": _safe_div(h - l, c),
        "volume": v,
        # OHLCV raw opcionais
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume_raw": v,
        # Indicadores pedidos
        "SO": so,
        "RSI_1": _rsi(c, 2),
        "RSI_2": _rsi(c, 3),
        "RSI_3": _rsi(c, 4),
        "RSI_14": _rsi(c, 14),
        "RSI_21": _rsi(c, 21),
        "2 Day ROI": roi_2,
        "3 Day ROI": roi_3,
        "7 Day ROI": roi_7,
        "MACD_12_26_9": macd,
        "MACDh_12_26_9": macd_hist,
        "MACDs_12_26_9": macd_signal,
        "Previous_differenced": previous_differenced,
        "Vol20": vol20,
        "Vol50": vol50,
        "PosRange60": pos_range_60,
        "NegRange60": neg_range_60,
        "ATR_14": _atr(h, l, c, 14),
    }
    return {k: _sanitize_feature(s.values) for k, s in catalog_series.items()}


def _resolve_feature_name(name: str) -> str:
    raw = str(name).strip()
    if not raw:
        return raw
    return FEATURE_ALIASES.get(raw, raw)


def resolve_feature_list(selected_features: Optional[Iterable[str]]) -> List[str]:
    if selected_features is None:
        return list(DEFAULT_FEATURE_NAMES)
    out: List[str] = []
    seen: set[str] = set()
    for feat in selected_features:
        resolved = _resolve_feature_name(str(feat))
        if not resolved:
            continue
        if resolved not in seen:
            out.append(resolved)
            seen.add(resolved)
    if not out:
        raise ValueError("A lista de features selecionadas está vazia")
    return out


def make_scalers(num_features: int = NUM_FEATURES) -> List[MaxAbsScaler]:
    return [MaxAbsScaler() for _ in range(num_features)]


def build_features(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    train_scalers: Optional[List[MaxAbsScaler]] = None,
    selected_features: Optional[Iterable[str]] = None,
) -> Tuple[np.ndarray, int, List[MaxAbsScaler]]:
    """
    Se ``train_scalers is None``: cria scalers e faz fit_transform (treino).
    Caso contrário: aplica transform com os scalers do treino (val/test).

    `selected_features` controla quais colunas entram na matriz final.
    """
    feature_names = resolve_feature_list(selected_features)
    catalog = _build_feature_catalog(opens, highs, lows, closes, volumes)
    missing = [f for f in feature_names if f not in catalog]
    if missing:
        available = ", ".join(sorted(catalog.keys()))
        raise ValueError(
            "Features desconhecidas no config: "
            f"{missing}. Disponíveis: {available}"
        )

    features = [catalog[name] for name in feature_names]
    num_features = len(features)
    scaled_fts: List[np.ndarray] = []

    if train_scalers is None:
        train_scalers = [MaxAbsScaler() for _ in range(num_features)]
        is_train = True
    else:
        if len(train_scalers) != num_features:
            raise ValueError(
                "Número de scalers incompatível com features selecionadas: "
                f"scalers={len(train_scalers)}, features={num_features}"
            )
        is_train = False

    for i in range(num_features):
        scaler = train_scalers[i]
        col = features[i].reshape(-1, 1)
        if is_train:
            scaled_ft = scaler.fit_transform(col)
        else:
            scaled_ft = scaler.transform(col)
        scaled_fts.append(scaled_ft.flatten())

    scaled_features = np.stack(scaled_fts, axis=-1)
    num_features = scaled_features.shape[-1]
    return scaled_features, num_features, train_scalers
