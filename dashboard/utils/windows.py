from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from INF.data_loader import iter_walkforward_windows


@dataclass(frozen=True)
class WindowSlice:
    window_id: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: Optional[int]
    test_end: Optional[int]

    @property
    def has_test(self) -> bool:
        return self.test_start is not None and self.test_end is not None and self.test_end > self.test_start


def compute_window_slices(
    df: pd.DataFrame,
    *,
    train_size: int,
    val_size: int,
    step_size: int,
    test_size: int | None,
    anchor: int = 0,
) -> list[WindowSlice]:
    slices: list[WindowSlice] = []
    for w in iter_walkforward_windows(
        df,
        train_size=int(train_size),
        val_size=int(val_size),
        step_size=int(step_size),
        test_size=int(test_size) if test_size is not None else None,
        anchor=int(anchor),
    ):
        slices.append(
            WindowSlice(
                window_id=int(w.window_id),
                train_start=int(w.train_start),
                train_end=int(w.train_end),
                val_start=int(w.val_start),
                val_end=int(w.val_end),
                test_start=int(w.test_start) if w.test_start is not None else None,
                test_end=int(w.test_end) if w.test_end is not None else None,
            )
        )
    return slices

