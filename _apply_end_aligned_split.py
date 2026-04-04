"""Patch training main.ipynb cells for end-aligned split. Run from repo root: python _apply_end_aligned_split.py"""
from __future__ import annotations

import json
from pathlib import Path

OLD = """TRAIN_START = 52500
TRAIN_END = 61500
VAL_END = 63000
assert 0 <= TRAIN_START < TRAIN_END < VAL_END <= len(df), f"{TRAIN_START=} {TRAIN_END=} {VAL_END=} len(df)={len(df)}"

df_train = df.iloc[TRAIN_START:TRAIN_END].copy()
df_val = df.iloc[TRAIN_END:VAL_END].copy()

print(f"Train rows [{TRAIN_START}:{TRAIN_END}]: {len(df_train)}")
print(f"Val rows   [{TRAIN_END}:{VAL_END}]: {len(df_val)} (equity + checkpoints; not holdout)")
print(f"Holdout    [{VAL_END}:] reserved for CNN/backtest (not used in this notebook)")"""

NEW = """TRAIN_BARS = 9000
VAL_BARS = 1500
HOLDOUT_BARS = 6000
L = len(df)
min_rows = TRAIN_BARS + VAL_BARS + HOLDOUT_BARS
assert L >= min_rows, f"need at least {min_rows} rows, got {L}"
total_rows = L
val_end = L - HOLDOUT_BARS
train_end = val_end - VAL_BARS
train_start = train_end - TRAIN_BARS
TRAIN_START = train_start
TRAIN_END = train_end
VAL_END = val_end
assert 0 <= TRAIN_START < TRAIN_END < VAL_END <= L, f"{TRAIN_START=} {TRAIN_END=} {VAL_END=} len(df)={L}"

df_train = df.iloc[TRAIN_START:TRAIN_END].copy()
df_val = df.iloc[TRAIN_END:VAL_END].copy()

print(f"Train rows [{TRAIN_START}:{TRAIN_END}]: {len(df_train)}")
print(f"Val rows   [{TRAIN_END}:{VAL_END}]: {len(df_val)} (equity + checkpoints; not holdout)")
print(f"Holdout    [{VAL_END}:] reserved for CNN/backtest (not used in this notebook)")"""

OLD_INFO = 'split_info = {"train_start": TRAIN_START, "train_end": TRAIN_END, "val_end": VAL_END, "total_rows": int(len(df))}'
NEW_INFO = 'split_info = {"train_start": TRAIN_START, "train_end": TRAIN_END, "val_end": VAL_END, "total_rows": total_rows}'


def set_cell_source(cell: dict, text: str) -> None:
    lines = text.split("\n")
    if not lines:
        cell["source"] = []
        return
    cell["source"] = [ln + "\n" for ln in lines[:-1]] + [lines[-1] + "\n"]


def patch_nb(path: Path) -> bool:
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        orig = src
        if OLD in src:
            src = src.replace(OLD, NEW)
        if OLD_INFO in src:
            src = src.replace(OLD_INFO, NEW_INFO)
        if src != orig:
            set_cell_source(cell, src)
            changed = True
    if changed:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return changed


def main() -> int:
    root = Path(__file__).resolve().parent
    targets = [
        root / "CNN_ETH/LSTM_model_training/LSTM_model_training/main.ipynb",
        root / "CNN_ETH/hybrid_model_training/main.ipynb",
        root / "CNN_ETH/CONV1D_model_training/CONV1D_model_training/main.ipynb",
        root / "CNN/LSTM_model_training/LSTM_model_training/main.ipynb",
        root / "CNN/hybrid_model_training/main.ipynb",
        root / "CNN/CONV1D_model_training/CONV1D_model_training/main.ipynb",
        root / "CNN_LINK/LSTM_model_training/LSTM_model_training/main.ipynb",
        root / "CNN_LINK/hybrid_model_training/main.ipynb",
        root / "CNN_LINK/CONV1D_model_training/CONV1D_model_training/main.ipynb",
    ]
    for p in targets:
        if not p.is_file():
            print("skip missing", p)
            continue
        if patch_nb(p):
            print("patched", p)
        else:
            print("no change", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
