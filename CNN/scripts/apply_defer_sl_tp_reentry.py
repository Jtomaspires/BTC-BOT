#!/usr/bin/env python3
"""Apply defer SL/TP re-entry logic (pending_entry) to CNN_* backtests and PORTFOLIO compounding."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --- Single-pair trading_backtest: exact block from CNN_ETH (THRESHOLD / SL_POINTS / TP_POINTS) ---
OLD_SINGLE = """    n = len(opens) - 1
    for i in range(n):
        curr_open = opens[i]
        curr_high = highs[i]
        curr_low = lows[i]
        curr_close = closes[i]
        bar_fee = 0.0

        raw_values = [arr[i] for arr in all_raw_signals]
        mean_signal = np.mean(raw_values)

        if mean_signal > THRESHOLD:
            desired = 1
        elif mean_signal < -THRESHOLD:
            desired = -1
        else:
            desired = 0

        if position != 0 and entry_price is not None:
            sl_price = entry_price - SL_POINTS if position == 1 else entry_price + SL_POINTS
            tp_price = entry_price + TP_POINTS if position == 1 else entry_price - TP_POINTS

            if (position == 1 and curr_low <= sl_price) or (position == -1 and curr_high >= sl_price):
                realize_to(sl_price)
                sl_hits += 1

            elif (position == 1 and curr_high >= tp_price) or (position == -1 and curr_low <= tp_price):
                realize_to(tp_price)
                tp_hits += 1

        if position == 0 and desired != 0:
            fee = taker_fee()
            cash -= fee
            bar_fee += fee
            total_fees += fee
            entry_price = curr_open
            position = desired
            entries += 1
            num_longs += desired == 1
            num_shorts += desired == -1

        equity = mark_to_market(curr_close)
        equities.append(equity)
        fee_log.append(bar_fee)
        actions_log.append(position)

"""

NEW_SINGLE = """    n = len(opens) - 1
    pending_entry = None
    for i in range(n):
        curr_open = opens[i]
        curr_high = highs[i]
        curr_low = lows[i]
        curr_close = closes[i]
        bar_fee = 0.0
        closed_by_sl_tp_this_bar = False

        raw_values = [arr[i] for arr in all_raw_signals]
        mean_signal = np.mean(raw_values)

        if mean_signal > THRESHOLD:
            desired = 1
        elif mean_signal < -THRESHOLD:
            desired = -1
        else:
            desired = 0

        if position == 0 and pending_entry is not None:
            fee = taker_fee()
            cash -= fee
            bar_fee += fee
            total_fees += fee
            entry_price = curr_open
            position = pending_entry
            entries += 1
            num_longs += pending_entry == 1
            num_shorts += pending_entry == -1
            pending_entry = None

        if position != 0 and entry_price is not None:
            sl_price = entry_price - SL_POINTS if position == 1 else entry_price + SL_POINTS
            tp_price = entry_price + TP_POINTS if position == 1 else entry_price - TP_POINTS

            if (position == 1 and curr_low <= sl_price) or (position == -1 and curr_high >= sl_price):
                realize_to(sl_price)
                sl_hits += 1
                closed_by_sl_tp_this_bar = True

            elif (position == 1 and curr_high >= tp_price) or (position == -1 and curr_low <= tp_price):
                realize_to(tp_price)
                tp_hits += 1
                closed_by_sl_tp_this_bar = True

        if position == 0 and desired != 0:
            if closed_by_sl_tp_this_bar:
                pending_entry = desired
            else:
                fee = taker_fee()
                cash -= fee
                bar_fee += fee
                total_fees += fee
                entry_price = curr_open
                position = desired
                entries += 1
                num_longs += desired == 1
                num_shorts += desired == -1

        equity = mark_to_market(curr_close)
        equities.append(equity)
        fee_log.append(bar_fee)
        actions_log.append(position)

"""

MARKDOWN_SNIP = (
    "\n"
    "Para evitar mini-lookahead, se uma posição fecha por SL/TP dentro da barra atual, "
    "uma nova entrada na direção do sinal fica em fila (`pending_entry`) e só é executada "
    "no **open da barra seguinte**. Assim, o backtest não volta a entrar no `curr_open` "
    "de uma barra cujo movimento intra-barra já foi parcialmente observado.\n"
)

SINGLE_PAIR_NOTEBOOKS = [
    ROOT / "CNN" / "backtest" / "main.ipynb",
    ROOT / "CNN_SOL" / "backtest" / "main.ipynb",
    ROOT / "CNN_LINK" / "backtest" / "main.ipynb",
    ROOT / "CNN_XRP" / "backtest" / "main.ipynb",
    ROOT / "CNN_PAXG" / "backtest" / "main.ipynb",
]

PORTFOLIO_NB = ROOT / "PORTFOLIO" / "backtest_4pairs" / "main.ipynb"

# Portfolio compounding: replace inner loop from "for i in range(n):" through bars_log.append / equities.append
OLD_PF_LOOP_START = """    for i in range(n):
        o = [legs[k][\"opens\"][i] for k in range(K)]
        h = [legs[k][\"highs\"][i] for k in range(K)]
        l_ = [legs[k][\"lows\"][i] for k in range(K)]
        c = [legs[k][\"closes\"][i] for k in range(K)]

        # 1) Independent signals (same order as single-pair backtest)
        desired = [0] * K
        mean_signals = [0.0] * K
        for k in range(K):
            raw_values = [arr[i] for arr in legs[k][\"all_raw_signals\"]]
            mean_signal = float(np.mean(raw_values))
            mean_signals[k] = mean_signal
            th = legs[k][\"threshold\"]
            if mean_signal > th:
                desired[k] = 1
            elif mean_signal < -th:
                desired[k] = -1

        # 2) SL / TP
        for k in range(K):
            if pos[k] == 0 or entry_price[k] is None:
                continue
            sl_p = legs[k][\"sl_points\"]
            tp_p = legs[k][\"tp_points\"]
            ep = entry_price[k]
            sl_price = ep - sl_p if pos[k] == 1 else ep + sl_p
            tp_price = ep + tp_p if pos[k] == 1 else ep - tp_p
            ch, cl = h[k], l_[k]
            if (pos[k] == 1 and cl <= sl_price) or (pos[k] == -1 and ch >= sl_price):
                realize_leg(k, sl_price)
                sl_hits[k] += 1
            elif (pos[k] == 1 and ch >= tp_price) or (pos[k] == -1 and cl <= tp_price):
                realize_leg(k, tp_price)
                tp_hits[k] += 1

        # 3) Equity after exits; slice = E / K at bar open (compounding)
        E = total_equity_at(o)
        slice_sz = max(E, 0.0) / float(K)

        # 4) New entries (same slice for any leg that opens this bar)
        for k in range(K):
            if pos[k] == 0 and desired[k] != 0 and slice_sz > 0:
                fee = slice_sz * taker_fee
                cash -= fee
                total_fees += fee
                entry_price[k] = o[k]
                pos[k] = desired[k]
                N[k] = slice_sz
                entries[k] += 1

        eq_close = total_equity_at(c)
"""

NEW_PF_LOOP_START = """    pending_entry = [None] * K
    for i in range(n):
        o = [legs[k][\"opens\"][i] for k in range(K)]
        h = [legs[k][\"highs\"][i] for k in range(K)]
        l_ = [legs[k][\"lows\"][i] for k in range(K)]
        c = [legs[k][\"closes\"][i] for k in range(K)]

        closed_by_sl_tp = [False] * K

        # 1) Independent signals (same order as single-pair backtest)
        desired = [0] * K
        mean_signals = [0.0] * K
        for k in range(K):
            raw_values = [arr[i] for arr in legs[k][\"all_raw_signals\"]]
            mean_signal = float(np.mean(raw_values))
            mean_signals[k] = mean_signal
            th = legs[k][\"threshold\"]
            if mean_signal > th:
                desired[k] = 1
            elif mean_signal < -th:
                desired[k] = -1

        # 1b) Deferred entries from prior bar (open of this bar, before SL/TP)
        E_pre = total_equity_at(o)
        slice_pre = max(E_pre, 0.0) / float(K)
        for k in range(K):
            if pos[k] == 0 and pending_entry[k] is not None and slice_pre > 0:
                fee = slice_pre * taker_fee
                cash -= fee
                total_fees += fee
                entry_price[k] = o[k]
                pos[k] = pending_entry[k]
                N[k] = slice_pre
                entries[k] += 1
                pending_entry[k] = None

        # 2) SL / TP
        for k in range(K):
            if pos[k] == 0 or entry_price[k] is None:
                continue
            sl_p = legs[k][\"sl_points\"]
            tp_p = legs[k][\"tp_points\"]
            ep = entry_price[k]
            sl_price = ep - sl_p if pos[k] == 1 else ep + sl_p
            tp_price = ep + tp_p if pos[k] == 1 else ep - tp_p
            ch, cl = h[k], l_[k]
            if (pos[k] == 1 and cl <= sl_price) or (pos[k] == -1 and ch >= sl_price):
                realize_leg(k, sl_price)
                sl_hits[k] += 1
                closed_by_sl_tp[k] = True
            elif (pos[k] == 1 and ch >= tp_price) or (pos[k] == -1 and cl <= tp_price):
                realize_leg(k, tp_price)
                tp_hits[k] += 1
                closed_by_sl_tp[k] = True

        # 3) Equity after exits; slice = E / K at bar open (compounding)
        E = total_equity_at(o)
        slice_sz = max(E, 0.0) / float(K)

        # 4) New entries — defer to next bar if this leg just closed by SL/TP
        for k in range(K):
            if pos[k] == 0 and desired[k] != 0 and slice_sz > 0:
                if closed_by_sl_tp[k]:
                    pending_entry[k] = desired[k]
                else:
                    fee = slice_sz * taker_fee
                    cash -= fee
                    total_fees += fee
                    entry_price[k] = o[k]
                    pos[k] = desired[k]
                    N[k] = slice_sz
                    entries[k] += 1

        eq_close = total_equity_at(c)
"""

def patch_single_pair_cell(source: list[str]) -> tuple[list[str], bool]:
    text = "".join(source)
    if "pending_entry" in text and "closed_by_sl_tp_this_bar" in text:
        return source, False
    if OLD_SINGLE not in text:
        return source, False
    text = text.replace(OLD_SINGLE, NEW_SINGLE, 1)
    return _source_lines(text), True


def _source_lines(text: str) -> list[str]:
    if not text.endswith("\n"):
        text = text + "\n"
    lines = text.splitlines(keepends=True)
    return lines if lines else [""]


def patch_markdown_heatmap(cells: list[dict]) -> bool:
    changed = False
    for cell in cells:
        if cell.get("cell_type") != "markdown":
            continue
        src = cell.get("source", [])
        if not src:
            continue
        text = "".join(src)
        if "**SL/TP heatmap" not in text and "SL/TP heatmap" not in text:
            continue
        if "pending_entry" in text and "mini-lookahead" in text:
            continue
        if "Para evitar mini-lookahead" in text:
            continue
        cell["source"] = _source_lines(text.rstrip("\n") + MARKDOWN_SNIP)
        changed = True
        break
    return changed


def patch_portfolio_compound(source: list[str]) -> tuple[list[str], bool]:
    text = "".join(source)
    if "pending_entry = [None] * K" in text:
        return source, False
    if OLD_PF_LOOP_START not in text:
        return source, False
    text = text.replace(OLD_PF_LOOP_START, NEW_PF_LOOP_START, 1)
    # Move pending_entry init: it should be once before loop, not inside realize_leg area.
    # NEW_PF_LOOP_START already has `pending_entry = [None] * K` right before for i.
    # Remove duplicate if we accidentally had `pending_entry` after pos = [...] in original - we didn't.
    return _source_lines(text), True


def process_notebook(path: Path, *, is_portfolio: bool = False) -> list[str]:
    msgs: list[str] = []
    data = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in data["cells"]:
        if cell.get("cell_type") != "code":
            continue
        text = "".join(cell.get("source", []))
        if "def trading_backtest(" in text and "def trading_backtest_portfolio" not in text:
            new_src, did = patch_single_pair_cell(cell["source"])
            if did:
                cell["source"] = new_src
                changed = True
                msgs.append(f"  patched trading_backtest in {path.name}")
        if is_portfolio and "def trading_backtest_portfolio_compounding" in text:
            new_src, did = patch_portfolio_compound(cell["source"])
            if did:
                cell["source"] = new_src
                changed = True
                msgs.append(f"  patched trading_backtest_portfolio_compounding in {path.name}")
    if not is_portfolio and patch_markdown_heatmap(data["cells"]):
        changed = True
        msgs.append(f"  patched heatmap markdown in {path.name}")
    if changed:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return msgs


def main() -> None:
    for nb in SINGLE_PAIR_NOTEBOOKS:
        if not nb.exists():
            print(f"SKIP missing {nb}")
            continue
        out = process_notebook(nb, is_portfolio=False)
        print(nb)
        for m in out:
            print(m)
        if not out:
            print("  (no changes — already patched or pattern mismatch)")

    if PORTFOLIO_NB.exists():
        out = process_notebook(PORTFOLIO_NB, is_portfolio=True)
        print(PORTFOLIO_NB)
        for m in out:
            print(m)
        if not out:
            print("  (portfolio single-pair cell unchanged)")
    else:
        print(f"SKIP missing {PORTFOLIO_NB}")

    eth = ROOT / "CNN_ETH" / "backtest" / "main.ipynb"
    if eth.exists():
        out = process_notebook(eth, is_portfolio=False)
        print(eth)
        for m in out:
            print(m)
        if not out:
            print("  (CNN_ETH already has defer)")


if __name__ == "__main__":
    main()
