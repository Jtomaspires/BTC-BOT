"""
Walk-forward janelas sobre OHLCV em CSV. Sem features, torch ou backtest.

Convenções
----------
- Por janela k: train = [train_start, train_end), val = [val_start, val_end),
  sem sobreposição entre treino e validação na mesma janela.
- O índice inicial do treino da janela k é: anchor + k * step_size.
- step_size desloca o início do treino para a janela seguinte.
- Se test_size > 0: teste = [val_end, val_end + test_size) (OOS após val),
  contíguo à validação; train/val/test continuam sem overlap.
- No fim do DataFrame, se não couber train_size + val_size (+ test_size),
  essa janela e seguintes não são emitidas (nada de fatias parciais).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

REQUIRED_COLS = ("open", "high", "low", "close", "volume")
OPTIONAL_COLS = ("timestamp",)


@dataclass(frozen=True)
class WalkWindow:
    """Uma janela walk-forward com índices em linhas do DataFrame (labels 0..len-1)."""

    window_id: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: Optional[int] = None
    test_end: Optional[int] = None

    @property
    def n_train(self) -> int:
        return self.train_end - self.train_start

    @property
    def n_val(self) -> int:
        return self.val_end - self.val_start

    @property
    def n_test(self) -> int:
        if self.test_start is None or self.test_end is None:
            return 0
        return self.test_end - self.test_start


def default_project_root() -> Path:
    """Raiz do repositório NN (pai de INF/)."""
    return Path(__file__).resolve().parents[1]


def load_ohlcv(path: str | Path, *, project_root: Optional[Path] = None) -> pd.DataFrame:
    """
    Lê o CSV e valida colunas mínimas.

    Parameters
    ----------
    path
        Caminho absoluto, ou relativo a `project_root` (por omissão raiz NN).
    project_root
        Pasta usada para resolver paths relativos (típico: raiz do repo).
    """
    p = Path(path)
    if not p.is_absolute():
        root = project_root if project_root is not None else default_project_root()
        p = (root / p).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"CSV não encontrado: {p}")

    df = pd.read_csv(p)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltam colunas obrigatórias {missing} em {p}. Colunas: {list(df.columns)}")

    for col in REQUIRED_COLS:
        if df[col].isnull().any():
            raise ValueError(f"Coluna '{col}' contém NaN em {p}")

    return df


def iter_walkforward_windows(
    df: pd.DataFrame,
    train_size: int,
    val_size: int,
    step_size: int,
    *,
    test_size: Optional[int] = None,
    anchor: int = 0,
) -> Iterator[WalkWindow]:
    """
    Gera janelas walk-forward enquanto couber um bloco completo em ``len(df)``.

    Parameters
    ----------
    df
        DataFrame completo (índices implícitos 0 .. len-1).
    train_size, val_size
        Comprimentos em número de barras (linhas).
    step_size
        Deslocamento do ``train_start`` entre janelas consecutivas.
    test_size
        Se não None e > 0, acrescenta fatia de teste imediatamente após ``val_end``.
    anchor
        ``train_start`` da janela 0; janela k tem ``train_start = anchor + k * step_size``.

    Yields
    ------
    WalkWindow
        Índices half-open [start, end) como no pandas ``iloc``.
    """
    if train_size <= 0 or val_size <= 0 or step_size <= 0:
        raise ValueError("train_size, val_size e step_size devem ser > 0")
    if anchor < 0:
        raise ValueError("anchor deve ser >= 0")
    if test_size is not None and test_size < 0:
        raise ValueError("test_size deve ser null ou >= 0")

    n = len(df)
    ts = test_size if test_size is not None else 0
    block = train_size + val_size + (ts if ts > 0 else 0)

    k = 0
    while True:
        train_start = anchor + k * step_size
        train_end = train_start + train_size
        val_start = train_end
        val_end = val_start + val_size

        if ts > 0:
            test_start = val_end
            test_end = test_start + ts
            if train_start < 0 or test_end > n:
                break
            yield WalkWindow(
                window_id=k,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
            )
        else:
            if train_start < 0 or val_end > n:
                break
            yield WalkWindow(
                window_id=k,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=None,
                test_end=None,
            )
        k += 1
