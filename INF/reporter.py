"""
Persistência de resultados e visualização para o run walk-forward.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_metrics_csv(df: pd.DataFrame, path: Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def plot_equity_curve(
    equity: np.ndarray | Sequence[float],
    path: Path,
    *,
    title: str | None = None,
) -> Path:
    eq = np.asarray(equity, dtype=np.float64).reshape(-1)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(eq, linewidth=1.2)
    ax.set_xlabel("bar")
    ax.set_ylabel("equity")
    ax.set_title(title or "Equity Curve")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_equity_with_buyhold(
    *,
    equities: np.ndarray | Sequence[float],
    opens: np.ndarray | Sequence[float],
    sharpe: float,
    max_dd_info: Mapping[str, float],
    path: Path,
    position_notional: float = 1000.0,
) -> Path:
    """
    Gráfico espelhado do ``plot_equity`` de ``CNN/CNN_BTC/backtest/main.ipynb``:

    - Curva de equity do modelo (azul)
    - Curva Buy & Hold (verde) calibrada para o mesmo capital inicial
    - Marcação do peak e do trough do Max Drawdown em vermelho + anotação
    - Título: ``Equity: X, Sharpe Ratio Y, Drawdown Z%``
    """
    eq = np.asarray(equities, dtype=np.float64).reshape(-1)
    op = np.asarray(opens, dtype=np.float64).reshape(-1)[: len(eq)]
    if eq.size == 0 or op.size == 0:
        raise ValueError("equities e opens não podem ser vazios")
    if op[0] == 0.0:
        raise ValueError("primeiro open é 0 — buy & hold não calculável")
    buy_and_hold = float(position_notional) * (op / op[0])

    chart_title = (
        f"Equity: {eq[-1]:.0f}, Sharpe Ratio {float(sharpe):.4f}, "
        f"Drawdown {float(max_dd_info.get('max_drawdown', 0.0)):.2%}"
    )

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        buy_and_hold,
        color="green",
        alpha=0.6,
        lw=1.5,
        label=f"Buy & Hold  (final: {buy_and_hold[-1]:.0f})",
    )
    ax.plot(
        eq,
        color="blue",
        lw=1.5,
        label=f"Model Equity  (final: {eq[-1]:.0f})",
    )

    peak_idx = int(max_dd_info.get("peak_index", 0))
    trough_idx = int(max_dd_info.get("trough_index", 0))
    peak_val = float(max_dd_info.get("peak_value", eq[peak_idx] if peak_idx < eq.size else eq[0]))
    trough_val = float(
        max_dd_info.get("trough_value", eq[trough_idx] if trough_idx < eq.size else eq[-1])
    )

    ax.scatter([peak_idx, trough_idx], [peak_val, trough_val], color="red", zorder=5, label="Max Drawdown")
    ax.annotate(
        f"Max Drawdown: {float(max_dd_info.get('max_drawdown', 0.0)):.2%}",
        xy=(trough_idx, trough_val),
        xytext=(trough_idx + max(1, int(eq.size * 0.02)), trough_val * 0.9),
        arrowprops=dict(facecolor="red", shrink=0.05),
    )

    ax.set_title(chart_title)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value (USD)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_summary_all_windows(dfs: list[pd.DataFrame], path: Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not dfs:
        pd.DataFrame().to_csv(out, index=False)
        return out
    merged = pd.concat(dfs, axis=0, ignore_index=True, sort=False)
    merged.to_csv(out, index=False)
    return out


def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("Sem resultados para reportar.")
        return
    cols = [c for c in ("window_id", "sl", "tp", "trailing_stop", "final_equity", "sharpe", "max_drawdown") if c in df]
    print(df[cols].head(10).to_string(index=False))
    if "sharpe" in df.columns:
        best_idx = int(df["sharpe"].astype(float).idxmax())
        best = df.iloc[best_idx]
        print(
            "Melhor por sharpe:",
            f"window={best.get('window_id', 'n/a')}, sl={best.get('sl')}, tp={best.get('tp')},",
            f"sharpe={float(best['sharpe']):.6f}",
        )


_RUN_SUMMARY_BASE_COLUMNS = [
    "run_id",
    "batch_id",
    "config_name",
    "experiment_name",
    "created_at",
    "selected_split",
    "n_windows",
    "avg_return",
    "avg_drawdown",
    "avg_sharpe",
    "pair",
    "timeframe",
    "data_csv_path",
    "train_size",
    "val_size",
    "test_size",
    "step_size",
    "epochs",
    "checkpoint_metric",
    "architectures",
    "grid_select_metric",
    "signal_threshold",
    "trailing_stop_points",
]


def _short_window_column(column: str) -> str:
    old_match = re.match(r"^window_(\d+)_(return|drawdown)$", column)
    if old_match:
        metric = "roi" if old_match.group(2) == "return" else "dd"
        return f"{int(old_match.group(1)):03d}_{metric}"

    short_match = re.match(r"^(\d+)_(roi|dd)$", column)
    if short_match:
        return f"{int(short_match.group(1)):03d}_{short_match.group(2)}"

    return column


def _normalize_run_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for old_col in list(out.columns):
        new_col = _short_window_column(str(old_col))
        if new_col == old_col:
            continue
        if new_col in out.columns:
            out[new_col] = out[new_col].combine_first(out[old_col])
            out = out.drop(columns=[old_col])
        else:
            out = out.rename(columns={old_col: new_col})
    # Safety: in case legacy columns slip through, drop them explicitly.
    legacy_cols = [
        c
        for c in out.columns
        if re.match(r"^window_\d+_(return|drawdown)$", str(c))
    ]
    if legacy_cols:
        out = out.drop(columns=legacy_cols)
    return out


def _ordered_run_summary_columns(columns: Sequence[str]) -> list[str]:
    existing = [str(c) for c in columns]
    base = [c for c in _RUN_SUMMARY_BASE_COLUMNS if c in existing]
    window_ids = sorted(
        {
            int(match.group(1))
            for col in existing
            if (match := re.match(r"^(\d+)_(roi|dd)$", col))
        }
    )
    window_metrics = [
        col
        for window_id in window_ids
        for col in (f"{window_id:03d}_roi", f"{window_id:03d}_dd")
        if col in existing
    ]
    handled = set(base) | set(window_metrics) | {"features"}
    # Exclude any legacy columns from ordering (should have been normalized/dropped anyway).
    extras = [
        c
        for c in existing
        if c not in handled and not re.match(r"^window_\d+_(return|drawdown)$", c)
    ]
    features = ["features"] if "features" in existing else []
    return base + window_metrics + extras + features


def build_run_summary_row(
    summary_df: pd.DataFrame,
    *,
    run_id: str,
    cfg: Mapping[str, Any],
    features_used: Sequence[str],
    config_name: str | None = None,
    experiment_name: str | None = None,
    batch_id: str | None = None,
) -> dict[str, Any]:
    if summary_df.empty:
        raise ValueError("summary_df não pode ser vazio")

    data_cfg = cfg.get("data", {}) or {}
    wf_cfg = cfg.get("walkforward", {}) or {}
    train_cfg = cfg.get("training", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    backtest_cfg = cfg.get("backtest", {}) or {}

    if "eval_split" in summary_df.columns and (summary_df["eval_split"] == "test").any():
        rows = summary_df[summary_df["eval_split"] == "test"].copy()
        selected_split = "test"
    elif "eval_split" in summary_df.columns:
        rows = summary_df[summary_df["eval_split"] == "val"].copy()
        if "is_best" in rows.columns:
            best_rows = rows[rows["is_best"] == True]  # noqa: E712 - pandas
            if not best_rows.empty:
                rows = best_rows
        selected_split = "val"
    else:
        rows = summary_df.copy()
        selected_split = "unknown"

    if "window_id" in rows.columns:
        rows = rows.sort_values("window_id", kind="stable")

    position_notional = float(backtest_cfg.get("position_notional", 1000.0))
    final_equity = rows["final_equity"].astype(float).to_numpy()
    returns = (final_equity - position_notional) / position_notional
    avg_return = float(np.mean(returns)) if returns.size else 0.0

    out: dict[str, Any] = {
        "run_id": run_id,
        "batch_id": str(batch_id or run_id),
        "config_name": str(config_name or ""),
        "experiment_name": str(experiment_name or ""),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "selected_split": selected_split,
        "n_windows": int(len(rows)),
        "avg_return": avg_return,
        "avg_drawdown": float(np.mean(rows["max_drawdown"].astype(float))) if "max_drawdown" in rows else 0.0,
        "avg_sharpe": float(np.mean(rows["sharpe"].astype(float))) if "sharpe" in rows else 0.0,
        "pair": str(data_cfg.get("pair", "")),
        "timeframe": str(data_cfg.get("timeframe", "")),
        "data_csv_path": str(data_cfg.get("csv_path", "")),
        "train_size": wf_cfg.get("train_size"),
        "val_size": wf_cfg.get("val_size"),
        "test_size": wf_cfg.get("test_size"),
        "step_size": wf_cfg.get("step_size"),
        "epochs": train_cfg.get("epochs"),
        "checkpoint_metric": str(train_cfg.get("checkpoint_metric", "")),
        "architectures": ",".join(model_cfg.get("architectures", []))
        if isinstance(model_cfg.get("architectures"), list)
        else str(model_cfg.get("architecture", "")),
        "features": ",".join(str(x) for x in features_used),
        "grid_select_metric": str(backtest_cfg.get("grid_select_metric", "")),
        "signal_threshold": backtest_cfg.get("signal_threshold"),
        "trailing_stop_points": ",".join(str(x) for x in backtest_cfg.get("trailing_stop_points", [0.0])),
    }

    if "window_id" in rows.columns:
        for _, row in rows.iterrows():
            w_id = int(row["window_id"])
            eq = float(row["final_equity"])
            w_return = (eq - position_notional) / position_notional
            out[f"{w_id:03d}_roi"] = w_return
            if "max_drawdown" in rows.columns:
                out[f"{w_id:03d}_dd"] = float(row["max_drawdown"])

    return out


def save_comparison_by_config(registry_file: Path, output_root: Path) -> Path:
    src = Path(registry_file)
    out = Path(output_root) / "comparison_by_config.csv"
    if not src.exists():
        pd.DataFrame().to_csv(out, index=False)
        return out

    df = pd.read_csv(src)
    if df.empty:
        df.to_csv(out, index=False)
        return out

    for col in ("config_name", "experiment_name", "avg_return", "avg_drawdown", "avg_sharpe"):
        if col not in df.columns:
            df[col] = np.nan if col.startswith("avg_") else ""

    grouped = (
        df.groupby(["config_name", "experiment_name"], dropna=False, as_index=False)
        .agg(
            runs=("run_id", "nunique"),
            mean_avg_return=("avg_return", "mean"),
            mean_avg_drawdown=("avg_drawdown", "mean"),
            mean_avg_sharpe=("avg_sharpe", "mean"),
            last_run_at=("created_at", "max"),
        )
        .sort_values(["mean_avg_return", "mean_avg_sharpe"], ascending=[False, False], kind="stable")
    )
    grouped.to_csv(out, index=False)
    return out


def save_run_summary_row(
    summary_df: pd.DataFrame,
    *,
    run_id: str,
    cfg: Mapping[str, Any],
    features_used: Sequence[str],
    run_dir: Path,
    output_root: Path,
    config_name: str | None = None,
    experiment_name: str | None = None,
    batch_id: str | None = None,
) -> tuple[Path, Path]:
    row = build_run_summary_row(
        summary_df,
        run_id=run_id,
        cfg=cfg,
        features_used=features_used,
        config_name=config_name,
        experiment_name=experiment_name,
        batch_id=batch_id,
    )
    one_row_df = pd.DataFrame([row])
    one_row_df = _normalize_run_summary_columns(one_row_df)
    one_row_df = one_row_df.reindex(columns=_ordered_run_summary_columns(one_row_df.columns))

    run_file = Path(run_dir) / "run_summary.csv"
    run_file.parent.mkdir(parents=True, exist_ok=True)
    one_row_df.to_csv(run_file, index=False)

    registry_file = Path(output_root) / "runs_summary.csv"
    if registry_file.exists():
        prev = _normalize_run_summary_columns(pd.read_csv(registry_file))
        merged = pd.concat([prev, one_row_df], axis=0, ignore_index=True, sort=False)
        dedup_cols = [c for c in ("run_id", "experiment_name") if c in merged.columns]
        if dedup_cols:
            merged = merged.drop_duplicates(subset=dedup_cols, keep="last")
        merged = _normalize_run_summary_columns(merged)
        merged = merged.reindex(columns=_ordered_run_summary_columns(merged.columns))
        merged.to_csv(registry_file, index=False)
    else:
        one_row_df.to_csv(registry_file, index=False)

    save_comparison_by_config(registry_file, output_root)
    return run_file, registry_file
