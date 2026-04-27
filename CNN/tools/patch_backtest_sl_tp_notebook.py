"""
Idempotent patch: parametrizar trading_backtest, heatmap só seleção, célula BEST_SL/BEST_TP.
Alinha com CNN/backtest/main.ipynb (defaults 150/300, mesma grelha).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

NEW_HEAD = """def trading_backtest(
    all_raw_signals, opens, highs, lows, closes, phase_label: str,
    *,
    SL_POINTS: float = 150.0,
    TP_POINTS: float = 300.0,
    verbose: bool = True,
):
    TAKER_FEE = 0.00055
    pos_size = 1000.0
    THRESHOLD = 0.0007
"""

OLD_HEAD_RE = re.compile(
    r"def trading_backtest\(all_raw_signals, opens, highs, lows, closes, phase_label: str\):\n"
    r"    TAKER_FEE = 0\.00055\n"
    r"    pos_size = 1000\.0\n"
    r"    SL_POINTS = [\d.]+\n"
    r"    TP_POINTS = [\d.]+\n"
    r"    THRESHOLD = 0\.0007\n",
)

OLD_TAIL = """    print(f"===== {phase_label} =====")
    print(f"Entries: {entries} | Completed: {completed_trades}")
    print(f"Longs: {num_longs} | Shorts: {num_shorts}")
    print(f"SL hits: {sl_hits} | TP hits: {tp_hits}")
    print(f"Win rate: {win_rate:.1f}% | Avg PnL/trade: ${avg_pnl:.2f}")
    print(f"Total fees: ${total_fees:.2f}")
    print(f"Final equity: ${equities[-1]:.2f}")
    return equities


raw_sel = ensemble_raw_signals(X_sel_t, m_sel)
equities_selection = trading_backtest(raw_sel, sel_opens, sel_highs, sel_lows, sel_closes, "selection [train_end:val_end]")

raw_hold = ensemble_raw_signals(X_hold_t, m_hold)
equities_holdout = trading_backtest(raw_hold, ho_opens, ho_highs, ho_lows, ho_closes, "holdout [val_end:end]")
"""

NEW_TAIL = """    if verbose:
        print(f"===== {phase_label} =====")
        print(f"Entries: {entries} | Completed: {completed_trades}")
        print(f"Longs: {num_longs} | Shorts: {num_shorts}")
        print(f"SL hits: {sl_hits} | TP hits: {tp_hits}")
        print(f"Win rate: {win_rate:.1f}% | Avg PnL/trade: ${avg_pnl:.2f}")
        print(f"Total fees: ${total_fees:.2f}")
        print(f"Final equity: ${equities[-1]:.2f}")
    return equities


raw_sel = ensemble_raw_signals(X_sel_t, m_sel)
raw_hold = ensemble_raw_signals(X_hold_t, m_hold)


"""

HEATMAP_MD = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "**SL/TP heatmap (só seleção)**\n",
        "\n",
        "Grelha apenas na janela de seleção; não uses o holdout para escolher SL/TP. "
        "Depois fixa `BEST_SL` / `BEST_TP` na célula seguinte.\n",
    ],
}
HEATMAP_CODE = {
    "cell_type": "code",
    "metadata": {},
    "outputs": [],
    "source": [
        "# SL/TP grid (selection only; never raw_hold).\n",
        "SL_values = [50, 100, 150, 200, 300, 500]\n",
        "TP_values = [100, 200, 300, 500, 750, 1000]\n",
        "\n",
        "results = np.empty((len(SL_values), len(TP_values)))\n",
        "for i, sl in enumerate(SL_values):\n",
        "    for j, tp in enumerate(TP_values):\n",
        "        eq = trading_backtest(\n",
        "            raw_sel, sel_opens, sel_highs, sel_lows, sel_closes,\n",
        '            "selection SL/TP grid (silent)",\n',
        "            SL_POINTS=float(sl), TP_POINTS=float(tp), verbose=False,\n",
        "        )\n",
        "        results[i, j] = eq[-1]\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(8, 5))\n",
        'im = ax.imshow(results, aspect="auto", origin="lower")\n',
        "ax.set_xticks(range(len(TP_values)), labels=[str(x) for x in TP_values])\n",
        "ax.set_yticks(range(len(SL_values)), labels=[str(x) for x in SL_values])\n",
        'ax.set_xlabel("TP (points)")\n',
        'ax.set_ylabel("SL (points)")\n',
        'ax.set_title("Final equity on selection window (heatmap)")\n',
        'plt.colorbar(im, ax=ax, label="Final equity ($)")\n',
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "best_flat = np.nanargmax(results)\n",
        "bi, bj = np.unravel_index(best_flat, results.shape)\n",
        'print(f"Max final equity grid cell: SL={SL_values[bi]}, TP={TP_values[bj]} -> ${results[bi, bj]:.2f}")\n',
    ],
}
HOLDOUT_MD = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "**Backtest final (seleção + holdout)**\n",
        "\n",
        "Define `BEST_SL` e `BEST_TP` **após** veres o heatmap; recalcula a curva de "
        "seleção e corre o holdout **uma vez** com os mesmos valores.\n",
    ],
}
HOLDOUT_CODE = {
    "cell_type": "code",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Preencher após heatmap (evita correr holdout antes de escolheres SL/TP).\n",
        "BEST_SL = 150.0  # preencher após heatmap\n",
        "BEST_TP = 300.0  # preencher após heatmap\n",
        "\n",
        "equities_selection = trading_backtest(\n",
        "    raw_sel, sel_opens, sel_highs, sel_lows, sel_closes,\n",
        '    "selection [train_end:val_end]",\n',
        "    SL_POINTS=BEST_SL, TP_POINTS=BEST_TP,\n",
        "    verbose=True,\n",
        ")\n",
        "equities_holdout = trading_backtest(\n",
        "    raw_hold, ho_opens, ho_highs, ho_lows, ho_closes,\n",
        '    "HOLDOUT FINAL",\n',
        "    SL_POINTS=BEST_SL, TP_POINTS=BEST_TP,\n",
        "    verbose=True,\n",
        ")\n",
    ],
}


def patch_notebook(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    if "SL_POINTS: float = 150.0" in raw and "# SL/TP grid (selection only" in raw:
        return "skip (already patched)"

    nb = json.loads(raw)
    idx_main = None
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        full = "".join(cell.get("source", []))
        if "def trading_backtest" in full and "ensemble_raw_signals" in full:
            idx_main = i
            break
    if idx_main is None:
        return "error: trading_backtest cell not found"

    cell = nb["cells"][idx_main]
    full = "".join(cell["source"])

    m = OLD_HEAD_RE.search(full)
    if not m:
        if "SL_POINTS: float = 150.0" in full:
            full2 = full
        else:
            return "error: OLD_HEAD pattern not found"
    else:
        full2 = OLD_HEAD_RE.sub(NEW_HEAD, full, count=1)

    if OLD_TAIL in full2:
        patched = full2.replace(OLD_TAIL, NEW_TAIL, 1)
    elif (
        "SL_POINTS: float = 150.0" in full2
        and "raw_hold = ensemble_raw_signals(X_hold_t, m_hold)" in full2
        and "equities_selection = trading_backtest(raw_sel" not in full2
    ):
        patched = full2
    else:
        return "error: OLD_TAIL missing and main cell not already refactored"

    cell["source"] = [line + "\n" for line in patched.split("\n")]
    if cell["source"] and cell["source"][-1] == "\n":
        cell["source"].pop()

    if "# SL/TP grid (selection only" not in json.dumps(nb):
        nb["cells"] = (
            nb["cells"][: idx_main + 1]
            + [HEATMAP_MD, HEATMAP_CODE, HOLDOUT_MD, HOLDOUT_CODE]
            + nb["cells"][idx_main + 1 :]
        )

    path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    return "ok"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    targets = [
        root / "CNN_ETH" / "backtest" / "main.ipynb",
        root / "CNN_LINK" / "backtest" / "main.ipynb",
        root / "CNN_PAXG" / "backtest" / "main.ipynb",
        root / "CNN_SOL" / "backtest" / "main.ipynb",
        root / "CNN_XRP" / "backtest" / "main.ipynb",
    ]
    for p in targets:
        if not p.is_file():
            print(p, "MISSING")
            continue
        r = patch_notebook(p)
        print(p.relative_to(root), r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
