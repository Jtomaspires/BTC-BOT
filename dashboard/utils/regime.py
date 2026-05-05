from __future__ import annotations

import numpy as np
import pandas as pd

from dashboard.utils.indicators import atr


def compute_atr_ratio(ohlcv_df: pd.DataFrame, fast: int = 14, slow: int = 200) -> pd.Series:
    """ATR(fast) / ATR(slow); NaN onde slow ainda não aqueceu."""
    hi = atr(ohlcv_df, int(fast))
    lo = atr(ohlcv_df, int(slow))
    return hi / lo.replace(0.0, np.nan)


def classify_regime(
    atr_ratio: pd.Series | np.ndarray,
    *,
    low_thr: float = 0.8,
    high_thr: float = 1.2,
) -> tuple[pd.Series, pd.Series]:
    """
    Devolve (regime_label, regime_name).
    label: 0=low_vol, 1=mid_vol, 2=high_vol; NaN onde ratio é NaN.
    """
    s = atr_ratio if isinstance(atr_ratio, pd.Series) else pd.Series(atr_ratio)
    labels = pd.Series(np.nan, index=s.index, dtype=np.float64)
    names = pd.Series(pd.NA, index=s.index, dtype="string")

    mask = s.notna()
    low_m = mask & (s <= low_thr)
    high_m = mask & (s >= high_thr)
    mid_m = mask & ~low_m & ~high_m

    labels = labels.astype(np.float64)
    labels[low_m] = 0.0
    labels[mid_m] = 1.0
    labels[high_m] = 2.0

    names[low_m] = "low_vol"
    names[mid_m] = "mid_vol"
    names[high_m] = "high_vol"
    return labels, names


def compute_regime_features(ohlc_df: pd.DataFrame) -> pd.DataFrame:
    """Colunas ATR_14, ATR_200, atr_ratio (para join por índice de barra)."""
    out = ohlc_df.copy()
    out["ATR_14"] = atr(ohlc_df, 14)
    out["ATR_200"] = atr(ohlc_df, 200)
    out["atr_ratio"] = out["ATR_14"] / out["ATR_200"].replace(0.0, np.nan)
    _, rnames = classify_regime(out["atr_ratio"])
    out["regime_name"] = rnames
    return out


def compute_atr_arrays(
    signals_df: pd.DataFrame,
    *,
    atr_period: int = 14,
    slow_period: int = 200,
    ohlcv_context_df: pd.DataFrame | None = None,
) -> dict[str, np.ndarray]:
    """
    ATR arrays alinhados por barra (mesmo índice de ``signals_df``).

    Quando ``ohlcv_context_df`` é fornecido e ambos os DataFrames têm ``timestamp``,
    usa as barras anteriores ao início da janela para "pré-aquecer" ATRs (sobretudo ATR lento),
    evitando NaNs no arranque da janela.
    """
    base = signals_df[["open", "high", "low", "close"]].astype(float).copy()
    lead = pd.DataFrame(columns=["open", "high", "low", "close"])
    warmup = max(int(atr_period), int(slow_period))

    if ohlcv_context_df is not None and "timestamp" in signals_df.columns and "timestamp" in ohlcv_context_df.columns:
        sig_ts = pd.to_datetime(signals_df["timestamp"], errors="coerce")
        if len(sig_ts):
            start_ts = sig_ts.iloc[0]
            if pd.notna(start_ts):
                ctx = ohlcv_context_df.copy()
                need_cols = {"open", "high", "low", "close"}
                if need_cols.issubset(ctx.columns):
                    ctx_ts = pd.to_datetime(ctx["timestamp"], errors="coerce")
                    lead = (
                        ctx.loc[ctx_ts < start_ts, ["open", "high", "low", "close"]]
                        .astype(float)
                        .tail(warmup)
                    )

    joined = pd.concat([lead, base], axis=0, ignore_index=True)
    atr_fast_full = atr(joined, int(atr_period))
    atr_slow_full = atr(joined, int(slow_period))
    cut = int(len(lead))
    atr_fast = atr_fast_full.iloc[cut:].reset_index(drop=True)
    atr_slow = atr_slow_full.iloc[cut:].reset_index(drop=True)
    atr_ratio = atr_fast / atr_slow.replace(0.0, np.nan)

    fast_np = atr_fast.to_numpy(dtype=np.float64)
    slow_np = atr_slow.to_numpy(dtype=np.float64)
    ratio_np = atr_ratio.to_numpy(dtype=np.float64)

    def _lag1(a: np.ndarray) -> np.ndarray:
        out = np.empty_like(a, dtype=np.float64)
        if out.size == 0:
            return out
        out[0] = np.nan
        out[1:] = a[:-1]
        return out

    return {
        "atr_fast": fast_np,
        "atr_slow": slow_np,
        "atr_ratio": ratio_np,
        # Versões com lag-1 — usar para decisões (entry/exit) para evitar lookahead.
        # Em barra i representam o ATR conhecido no fim de i-1.
        "atr_fast_dec": _lag1(fast_np),
        "atr_slow_dec": _lag1(slow_np),
        "atr_ratio_dec": _lag1(ratio_np),
    }


def entry_mask_from_hardstop(
    atr_ratio: np.ndarray,
    *,
    hardstop: float | None,
    warmup: int,
    atr_fast: np.ndarray | None = None,
) -> np.ndarray:
    """
    Máscara de entrada por barra.

    Lógica separada por concern:
    - ``warmup``: bloqueia as primeiras N barras (tamanho do ATR rápido, ex. 14).
    - Barras onde ``atr_fast`` não é finito ficam bloqueadas (sem preço de stop válido).
    - Se ``hardstop`` activo: bloqueia barras onde ``atr_ratio`` é finito E > hardstop.
      Barras onde ``atr_ratio`` é NaN (ATR_200 ainda não aqueceu) **não** são bloqueadas
      pelo hardstop — sem informação de regime assumimos livre para entrar.
    """
    ratio = np.asarray(atr_ratio, dtype=np.float64)
    n = ratio.shape[0]

    # Por defeito: tudo permitido
    mask = np.ones(n, dtype=bool)

    # Warmup do ATR rápido
    if warmup > 0:
        mask[: min(warmup, n)] = False

    # Sem preço de stop válido → não pode entrar
    if atr_fast is not None:
        fast = np.asarray(atr_fast, dtype=np.float64)
        mask &= np.isfinite(fast) & (fast > 0)

    # Hardstop: só bloqueia quando ratio é CONHECIDO E acima do limite
    if hardstop is not None:
        hs = float(hardstop)
        mask &= ~(np.isfinite(ratio) & (ratio > hs))

    return mask.astype(bool)
