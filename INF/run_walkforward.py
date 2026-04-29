"""
Orquestrador da pipeline walk-forward.
"""

from __future__ import annotations

import argparse
import copy
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from backtest_engine import BacktestResult, run_backtest_grid, run_single_backtest
from data_loader import default_project_root, iter_walkforward_windows, load_ohlcv
from features import build_features, resolve_feature_list
from labels import build_triple_barrier_labels
from metrics import max_drawdown_info, select_best_grid_params, sharpe_ratio, summarize_window
from models import get_action, get_model, logits_to_signal
from reporter import (
    plot_equity_curve,
    plot_equity_with_buyhold,
    print_summary,
    save_run_summary_row,
    save_metrics_csv,
    save_summary_all_windows,
)
from trainer import build_sequences, train_window


def _resolve_architectures(model_cfg: Mapping[str, Any]) -> list[str]:
    raw = model_cfg.get("architectures")
    if raw is None:
        return [str(model_cfg.get("architecture", "conv1d")).strip().lower()]
    if not isinstance(raw, list) or not raw:
        raise ValueError("model.architectures deve ser uma lista não vazia")
    archs: list[str] = []
    for item in raw:
        a = str(item).strip().lower()
        if not a:
            continue
        if a not in archs:
            archs.append(a)
    if not archs:
        raise ValueError("model.architectures não contém arquiteturas válidas")
    return archs


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML não instalado. Instale com: pip install pyyaml") from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config inválido em {path}: esperado mapeamento no topo.")
    return data


def _save_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(payload), f, sort_keys=False, allow_unicode=False)


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (default_project_root() / p).resolve()


def _periods_per_year(timeframe: str | None) -> float:
    tf = (timeframe or "").strip().lower()
    mapping = {"1h": 365.0 * 24.0, "4h": 365.0 * 6.0, "1d": 365.0}
    return mapping.get(tf, 365.0 * 24.0)


def _device_for_inference(device_cfg: str) -> torch.device:
    key = str(device_cfg).strip().lower()
    if key == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if key in {"cpu", "cuda"}:
        if key == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(key)
    return torch.device("cpu")


def _grid_results_from_cfg(
    *,
    raw_signal_per_bar: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    backtest_cfg: Mapping[str, Any],
) -> dict[tuple[float, float, float], BacktestResult]:
    sl_tp_grid = backtest_cfg.get("sl_tp_grid")
    if sl_tp_grid:
        out: dict[tuple[float, float, float], BacktestResult] = {}
        for item in sl_tp_grid:
            sl = float(item["sl"])
            tp = float(item["tp"])
            trailing = float(item.get("trailing_stop", item.get("trailing_stop_points", 0.0)))
            out[(sl, tp, trailing)] = run_single_backtest(
                raw_signal_per_bar,
                opens,
                highs,
                lows,
                closes,
                sl_points=sl,
                tp_points=tp,
                taker_fee=float(backtest_cfg["taker_fee"]),
                position_notional=float(backtest_cfg["position_notional"]),
                signal_threshold=float(backtest_cfg["signal_threshold"]),
                trailing_stop_points=trailing,
            )
        return out

    return run_backtest_grid(
        raw_signal_per_bar,
        opens,
        highs,
        lows,
        closes,
        sl_points=backtest_cfg["sl_points"],
        tp_points=backtest_cfg["tp_points"],
        trailing_stop_points=backtest_cfg.get("trailing_stop_points", [0.0]),
        taker_fee=float(backtest_cfg["taker_fee"]),
        position_notional=float(backtest_cfg["position_notional"]),
        signal_threshold=float(backtest_cfg["signal_threshold"]),
    )


def _save_signals_csv(
    *,
    out_path: Path,
    df_slice: pd.DataFrame,
    seq_len: int,
    raw_signal_per_bar: np.ndarray,
    signal_threshold: float,
    eval_split: str,
    aligned_mask: np.ndarray | None = None,
) -> Path:
    """
    Persiste sinais por barra para análise rápida (dashboard) sem recarregar modelo.

    Convenção igual ao backtest:
    - `raw_signal_per_bar` tem comprimento `n = len(opens) - 1`
    - OHLCV é alinhado com `df_slice[col][seq_len:]` e truncado para `n` barras
      (a última barra do OHLC não tem `signal` associado).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if len(df_slice) <= seq_len + 1:
        raise ValueError("df_slice demasiado pequeno para seq_len + 1 barras")

    opens = df_slice["open"].to_numpy(dtype=np.float64)[seq_len:]
    highs = df_slice["high"].to_numpy(dtype=np.float64)[seq_len:]
    lows = df_slice["low"].to_numpy(dtype=np.float64)[seq_len:]
    closes = df_slice["close"].to_numpy(dtype=np.float64)[seq_len:]
    volumes = df_slice["volume"].to_numpy(dtype=np.float64)[seq_len:]
    if aligned_mask is not None:
        mask = np.asarray(aligned_mask, dtype=bool).reshape(-1)
        if mask.shape[0] != opens.shape[0]:
            raise ValueError(
                f"aligned_mask length {mask.shape[0]} != aligned OHLC length {opens.shape[0]}"
            )
        opens = opens[mask]
        highs = highs[mask]
        lows = lows[mask]
        closes = closes[mask]
        volumes = volumes[mask]

    n = len(opens) - 1
    sig = np.asarray(raw_signal_per_bar, dtype=np.float64).reshape(-1)[:n]
    if len(sig) != n:
        raise ValueError(f"raw_signal_per_bar length {len(sig)} != n {n}")

    ts = None
    if "timestamp" in df_slice.columns:
        ts_full = df_slice["timestamp"].to_numpy()[seq_len:]
        if aligned_mask is not None:
            ts_full = ts_full[np.asarray(aligned_mask, dtype=bool).reshape(-1)]
        ts = ts_full[:n]
    else:
        ts = np.arange(n, dtype=np.int64)

    desired = np.asarray([get_action([float(x)], threshold=float(signal_threshold)) for x in sig], dtype=np.int64)

    out_df = pd.DataFrame(
        {
            "timestamp": ts,
            "eval_split": str(eval_split),
            "signal": sig,
            "desired": desired,
            "open": opens[:n],
            "high": highs[:n],
            "low": lows[:n],
            "close": closes[:n],
            "volume": volumes[:n],
        }
    )
    out_df.to_csv(out_path, index=False)
    return out_path


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dict(base))
    for key, val in override.items():
        if isinstance(val, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(dict(out[key]), val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _slugify(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip())
    cleaned = cleaned.strip("._")
    return cleaned or "experiment"


def _resolve_config_name(config_abs: Path, cfg: Mapping[str, Any]) -> str:
    raw = str(cfg.get("config_name", "")).strip()
    return raw or config_abs.stem


def _resolve_experiments(cfg: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    experiments_raw = cfg.get("experiments")
    if experiments_raw is None:
        default_name = str(cfg.get("experiment_name", cfg.get("name", "default"))).strip() or "default"
        return [(default_name, dict(cfg))]

    if not isinstance(experiments_raw, list) or not experiments_raw:
        raise ValueError("`experiments` deve ser uma lista não vazia quando definido")

    base_cfg = {k: v for k, v in dict(cfg).items() if k != "experiments"}
    out: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for idx, item in enumerate(experiments_raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"experiments[{idx}] deve ser um mapeamento")
        raw_name = str(item.get("name", "")).strip()
        if not raw_name:
            raise ValueError(f"experiments[{idx}] precisa de `name` não vazio")
        if raw_name in seen:
            raise ValueError(f"Nome de experiment repetido: {raw_name!r}")
        seen.add(raw_name)
        override = {k: v for k, v in dict(item).items() if k != "name"}
        merged = _deep_merge(base_cfg, override)
        out.append((raw_name, merged))
    return out


def _resolve_feature_list_from_cfg(cfg: Mapping[str, Any]) -> list[str]:
    raw = cfg.get("features")
    if raw is None:
        return resolve_feature_list(None)
    if isinstance(raw, Mapping):
        raw = raw.get("list")
    if raw is None:
        return resolve_feature_list(None)
    if not isinstance(raw, list):
        raise ValueError("`features` deve ser lista ou mapeamento com `features.list`")
    return resolve_feature_list(raw)


def _resolve_training_cfg(training_cfg: Mapping[str, Any], backtest_cfg: Mapping[str, Any]) -> dict[str, Any]:
    resolved = copy.deepcopy(dict(training_cfg))
    class_balance = resolved.get("class_balance")
    if isinstance(class_balance, Mapping):
        cb = dict(class_balance)
        threshold_raw = cb.get("threshold", "auto")
        if isinstance(threshold_raw, str) and threshold_raw.strip().lower() == "auto":
            cb["threshold"] = float(backtest_cfg.get("signal_threshold", 0.0))
        resolved["class_balance"] = cb
    return resolved


def _resolve_target_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = cfg.get("target")
    if raw is None:
        return {"type": "next_features"}
    if not isinstance(raw, Mapping):
        raise ValueError("target deve ser um mapeamento")
    target_type = str(raw.get("type", "next_features")).strip().lower()
    if target_type == "next_features":
        return {"type": "next_features"}
    if target_type != "triple_barrier":
        raise ValueError("target.type deve ser 'next_features' ou 'triple_barrier'")

    tp_pct = float(raw.get("tp_pct", 0.0))
    sl_pct = float(raw.get("sl_pct", 0.0))
    horizon = int(raw.get("horizon", 0))
    if tp_pct <= 0 or sl_pct <= 0 or horizon <= 0:
        raise ValueError("target triple_barrier requer tp_pct>0, sl_pct>0 e horizon>0")
    return {"type": "triple_barrier", "tp_pct": tp_pct, "sl_pct": sl_pct, "horizon": horizon}


def _attach_balance_columns(
    df: pd.DataFrame,
    *,
    architectures: Sequence[str],
    balance_stats_by_arch: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    out = df.copy()
    agg_before_long = 0
    agg_before_short = 0
    agg_before_neutral = 0
    agg_after_long = 0
    agg_after_short = 0
    agg_after_neutral = 0
    enabled_any = False
    applied_any = False
    sampler_any = False
    methods: list[str] = []
    statuses: list[str] = []

    for arch in architectures:
        key = _slugify(arch)
        stats = dict(balance_stats_by_arch.get(arch, {}))
        enabled = bool(stats.get("enabled", False))
        applied = bool(stats.get("applied", False))
        sampler_enabled = bool(stats.get("sampler_enabled", False))
        status = str(stats.get("status", "disabled"))
        method = str(stats.get("method", "weighted_sampler"))
        threshold = float(stats.get("threshold", 0.0))
        before_long = int(stats.get("before_count_long", 0))
        before_short = int(stats.get("before_count_short", 0))
        before_neutral = int(stats.get("before_count_neutral", 0))
        after_long = int(stats.get("after_count_long", 0))
        after_short = int(stats.get("after_count_short", 0))
        after_neutral = int(stats.get("after_count_neutral", 0))

        out[f"bal_enabled_{key}"] = enabled
        out[f"bal_applied_{key}"] = applied
        out[f"bal_sampler_{key}"] = sampler_enabled
        out[f"bal_method_{key}"] = method
        out[f"bal_status_{key}"] = status
        out[f"bal_threshold_{key}"] = threshold
        out[f"bal_before_long_{key}"] = before_long
        out[f"bal_before_short_{key}"] = before_short
        out[f"bal_before_neutral_{key}"] = before_neutral
        out[f"bal_after_long_{key}"] = after_long
        out[f"bal_after_short_{key}"] = after_short
        out[f"bal_after_neutral_{key}"] = after_neutral

        agg_before_long += before_long
        agg_before_short += before_short
        agg_before_neutral += before_neutral
        agg_after_long += after_long
        agg_after_short += after_short
        agg_after_neutral += after_neutral
        enabled_any = enabled_any or enabled
        applied_any = applied_any or applied
        sampler_any = sampler_any or sampler_enabled
        methods.append(method)
        statuses.append(status)

    out["bal_enabled_any"] = enabled_any
    out["bal_applied_any"] = applied_any
    out["bal_sampler_any"] = sampler_any
    out["bal_methods"] = ",".join(methods)
    out["bal_statuses"] = ",".join(statuses)
    out["bal_before_long_total"] = agg_before_long
    out["bal_before_short_total"] = agg_before_short
    out["bal_before_neutral_total"] = agg_before_neutral
    out["bal_after_long_total"] = agg_after_long
    out["bal_after_short_total"] = agg_after_short
    out["bal_after_neutral_total"] = agg_after_neutral
    return out


def _run_single_experiment(
    cfg: Mapping[str, Any],
    *,
    run_dir: Path,
    output_root: Path,
    run_id: str,
    config_name: str,
    experiment_name: str,
    max_windows_override: Optional[int],
) -> pd.DataFrame:
    data_cfg = cfg["data"]
    preprocess_cfg = cfg["preprocess"]
    walk_cfg = cfg["walkforward"]
    train_cfg = _resolve_training_cfg(cfg["training"], cfg["backtest"])
    model_cfg = cfg["model"]
    backtest_cfg = cfg["backtest"]
    target_cfg = _resolve_target_cfg(cfg)
    features_used = _resolve_feature_list_from_cfg(cfg)
    train_cfg["target_type"] = str(target_cfg.get("type", "next_features"))

    resolved_cfg = copy.deepcopy(dict(cfg))
    resolved_cfg["training"] = copy.deepcopy(dict(train_cfg))
    resolved_cfg["features"] = {"list": list(features_used)}
    resolved_cfg["target"] = copy.deepcopy(dict(target_cfg))
    _save_yaml(run_dir / "config.resolved.yaml", resolved_cfg)

    seq_len = int(preprocess_cfg["seq_len"])
    if seq_len <= 0:
        raise ValueError("preprocess.seq_len deve ser > 0")

    max_windows_cfg = walk_cfg.get("max_windows")
    max_windows = max_windows_override if max_windows_override is not None else max_windows_cfg
    if max_windows is not None:
        max_windows = int(max_windows)
        if max_windows <= 0:
            raise ValueError("max_windows deve ser null ou inteiro > 0")

    df = load_ohlcv(data_cfg["csv_path"])
    recent_rows = int(data_cfg.get("recent_rows", 0))
    if recent_rows < 0:
        raise ValueError("data.recent_rows deve ser >= 0")
    if recent_rows > 0:
        train_size = int(walk_cfg["train_size"])
        test_size_cfg = walk_cfg.get("test_size")
        test_size = int(test_size_cfg) if test_size_cfg is not None else 0
        # Mantém o período recente para avaliação e adiciona contexto suficiente
        # para o treino da primeira janela.
        keep_rows = recent_rows + train_size + (test_size if test_size > 0 else 0)
        df = df.tail(keep_rows).reset_index(drop=True)
    periods_per_year = _periods_per_year(data_cfg.get("timeframe"))

    rows_per_window: list[pd.DataFrame] = []
    architectures = _resolve_architectures(model_cfg)
    windows_iter = iter_walkforward_windows(
        df,
        train_size=int(walk_cfg["train_size"]),
        val_size=int(walk_cfg["val_size"]),
        step_size=int(walk_cfg["step_size"]),
        test_size=walk_cfg.get("test_size"),
        anchor=int(walk_cfg.get("anchor", 0)),
    )

    for w in windows_iter:
        if max_windows is not None and w.window_id >= max_windows:
            break

        train_df = df.iloc[w.train_start : w.train_end]
        val_df = df.iloc[w.val_start : w.val_end]
        test_df = (
            df.iloc[w.test_start : w.test_end]
            if (w.test_start is not None and w.test_end is not None)
            else None
        )
        has_test = test_df is not None

        train_features, _, train_scalers = build_features(
            train_df["open"].to_numpy(dtype=np.float64),
            train_df["high"].to_numpy(dtype=np.float64),
            train_df["low"].to_numpy(dtype=np.float64),
            train_df["close"].to_numpy(dtype=np.float64),
            train_df["volume"].to_numpy(dtype=np.float64),
            train_scalers=None,
            selected_features=features_used,
        )
        val_features, _, _ = build_features(
            val_df["open"].to_numpy(dtype=np.float64),
            val_df["high"].to_numpy(dtype=np.float64),
            val_df["low"].to_numpy(dtype=np.float64),
            val_df["close"].to_numpy(dtype=np.float64),
            val_df["volume"].to_numpy(dtype=np.float64),
            train_scalers=train_scalers,
            selected_features=features_used,
        )
        test_features = None
        if has_test and test_df is not None:
            test_features, _, _ = build_features(
                test_df["open"].to_numpy(dtype=np.float64),
                test_df["high"].to_numpy(dtype=np.float64),
                test_df["low"].to_numpy(dtype=np.float64),
                test_df["close"].to_numpy(dtype=np.float64),
                test_df["volume"].to_numpy(dtype=np.float64),
                train_scalers=train_scalers,
                selected_features=features_used,
            )

        if str(target_cfg["type"]) == "triple_barrier":
            train_labels = build_triple_barrier_labels(
                train_df["close"].to_numpy(dtype=np.float64),
                train_df["high"].to_numpy(dtype=np.float64),
                train_df["low"].to_numpy(dtype=np.float64),
                tp_pct=float(target_cfg["tp_pct"]),
                sl_pct=float(target_cfg["sl_pct"]),
                horizon=int(target_cfg["horizon"]),
            )
            val_labels = build_triple_barrier_labels(
                val_df["close"].to_numpy(dtype=np.float64),
                val_df["high"].to_numpy(dtype=np.float64),
                val_df["low"].to_numpy(dtype=np.float64),
                tp_pct=float(target_cfg["tp_pct"]),
                sl_pct=float(target_cfg["sl_pct"]),
                horizon=int(target_cfg["horizon"]),
            )
            x_train, y_train = build_sequences(train_features, seq_len=seq_len, labels=train_labels)
            x_val_raw, y_val_raw = build_sequences(val_features, seq_len=seq_len, labels=val_labels)
            valid_train = y_train >= 0
            valid_val = y_val_raw >= 0
            x_train = x_train[valid_train]
            y_train = y_train[valid_train]
            x_val = x_val_raw[valid_val]
            y_val = y_val_raw[valid_val]
            if len(x_train) == 0 or len(x_val) == 0:
                raise ValueError(
                    f"Sem amostras válidas após filtro triple_barrier na janela {w.window_id}. "
                    "Ajuste horizon/train_size/val_size."
                )
        else:
            x_train, y_train = build_sequences(train_features, seq_len=seq_len)
            x_val, y_val = build_sequences(val_features, seq_len=seq_len)

        x_test = None
        if has_test and test_features is not None:
            x_test, _ = build_sequences(test_features, seq_len=seq_len)
        val_opens_full = val_df["open"].to_numpy(dtype=np.float64)[seq_len:]
        val_closes_full = val_df["close"].to_numpy(dtype=np.float64)[seq_len:]
        if str(target_cfg["type"]) == "triple_barrier":
            val_opens_aligned = val_opens_full[valid_val]
            val_closes_aligned = val_closes_full[valid_val]
        else:
            val_opens_aligned = val_opens_full
            val_closes_aligned = val_closes_full

        device = _device_for_inference(str(train_cfg.get("device", "auto")))
        x_val_t = torch.from_numpy(x_val.astype(np.float32)).to(device)
        x_test_t = (
            torch.from_numpy(x_test.astype(np.float32)).to(device)
            if (has_test and x_test is not None)
            else None
        )
        raw_signal_val_per_ckpt: list[np.ndarray] = []
        raw_signal_test_per_ckpt: list[np.ndarray] = []
        checkpoints_by_arch: dict[str, int] = {}
        balance_stats_by_arch: dict[str, dict[str, Any]] = {}
        for arch in architectures:
            train_result = train_window(
                training_cfg=train_cfg,
                model_cfg=model_cfg,
                architecture=arch,
                X_train=x_train,
                Y_train=y_train,
                X_val=x_val,
                Y_val=y_val,
                out_dir=run_dir,
                window_id=w.window_id,
                checkpoint_subdir=arch,
                val_opens=val_opens_aligned,
                val_closes=val_closes_aligned,
                position_notional=float(backtest_cfg["position_notional"]),
                signal_threshold=float(backtest_cfg["signal_threshold"]),
            )
            checkpoints_by_arch[arch] = len(train_result.best_checkpoint_paths)
            balance_stats_by_arch[arch] = dict(train_result.balance_stats or {})
            for ckpt_path in train_result.best_checkpoint_paths:
                model = get_model(
                    arch,
                    num_features=train_result.num_features,
                    output_dim=train_result.output_dim,
                ).to(device)
                try:
                    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
                except TypeError:
                    checkpoint = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                with torch.no_grad():
                    preds_val = model(x_val_t).cpu().numpy()
                if str(target_cfg["type"]) == "triple_barrier":
                    raw_signal_val_per_ckpt.append(logits_to_signal(preds_val))
                else:
                    # Paridade com CNN `ensemble_raw_signals`: usa primeiro canal (feature 1).
                    raw_signal_val_per_ckpt.append(preds_val[:, 0].astype(np.float64))
                if x_test_t is not None:
                    with torch.no_grad():
                        preds_test = model(x_test_t).cpu().numpy()
                    if str(target_cfg["type"]) == "triple_barrier":
                        raw_signal_test_per_ckpt.append(logits_to_signal(preds_test))
                    else:
                        raw_signal_test_per_ckpt.append(preds_test[:, 0].astype(np.float64))
                del model

        if not raw_signal_val_per_ckpt:
            raise RuntimeError("Nenhum checkpoint gerou predições para validação nesta janela")

        # Média entre checkpoints (ensemble) antes do threshold — equivalente a
        # ``mean_signal`` em ``CNN trading_backtest`` porque a regra é linear.
        raw_signal_val = np.mean(np.stack(raw_signal_val_per_ckpt, axis=0), axis=0)

        # Alinhamento CNN (``preprocess_data`` + ``trading_backtest``):
        # opens/highs/lows/closes fatiados com ``[seq_len:]`` e loop corre
        # ``n = len(opens) - 1``; a última predição (sem barra futura) é descartada.
        open_val = val_df["open"].to_numpy(dtype=np.float64)[seq_len:]
        high_val = val_df["high"].to_numpy(dtype=np.float64)[seq_len:]
        low_val = val_df["low"].to_numpy(dtype=np.float64)[seq_len:]
        close_val = val_df["close"].to_numpy(dtype=np.float64)[seq_len:]
        if str(target_cfg["type"]) == "triple_barrier":
            open_val = open_val[valid_val]
            high_val = high_val[valid_val]
            low_val = low_val[valid_val]
            close_val = close_val[valid_val]
        raw_signal_val = raw_signal_val[: len(open_val) - 1]
        if len(raw_signal_val) != (len(open_val) - 1):
            raise RuntimeError(
                "Alinhamento inválido entre raw_signal_val e OHLC da validação: "
                f"raw={len(raw_signal_val)}, ohlc_n={len(open_val)}"
            )

        val_grid_results = _grid_results_from_cfg(
            raw_signal_per_bar=raw_signal_val,
            opens=open_val,
            highs=high_val,
            lows=low_val,
            closes=close_val,
            backtest_cfg=backtest_cfg,
        )

        val_df_metrics = summarize_window(val_grid_results, periods_per_year=periods_per_year)
        val_df_metrics.insert(0, "window_id", int(w.window_id))
        val_df_metrics.insert(1, "eval_split", "val")
        val_df_metrics.insert(2, "architectures", ",".join(architectures))
        val_df_metrics.insert(3, "num_checkpoints", len(raw_signal_val_per_ckpt))
        for arch in architectures:
            val_df_metrics[f"num_ckpts_{arch}"] = int(checkpoints_by_arch.get(arch, 0))
        val_df_metrics = _attach_balance_columns(
            val_df_metrics,
            architectures=architectures,
            balance_stats_by_arch=balance_stats_by_arch,
        )
        best_sl, best_tp, best_trailing_stop, best_score = select_best_grid_params(
            val_grid_results,
            metric=str(backtest_cfg.get("grid_select_metric", "sharpe")),
            periods_per_year=periods_per_year,
        )
        val_df_metrics["is_best"] = (
            (val_df_metrics["sl"] == best_sl)
            & (val_df_metrics["tp"] == best_tp)
            & (val_df_metrics["trailing_stop"] == best_trailing_stop)
        )
        val_df_metrics["best_score"] = np.where(val_df_metrics["is_best"], best_score, np.nan)

        per_split_metrics = [val_df_metrics]
        best_result_for_plot = val_grid_results[(best_sl, best_tp, best_trailing_stop)]
        opens_for_plot = open_val

        if has_test and test_df is not None:
            if not raw_signal_test_per_ckpt:
                raise RuntimeError("Nenhum checkpoint gerou predições para teste nesta janela")
            raw_signal_test = np.mean(np.stack(raw_signal_test_per_ckpt, axis=0), axis=0)
            open_test = test_df["open"].to_numpy(dtype=np.float64)[seq_len:]
            high_test = test_df["high"].to_numpy(dtype=np.float64)[seq_len:]
            low_test = test_df["low"].to_numpy(dtype=np.float64)[seq_len:]
            close_test = test_df["close"].to_numpy(dtype=np.float64)[seq_len:]
            raw_signal_test = raw_signal_test[: len(open_test) - 1]
            if len(raw_signal_test) != (len(open_test) - 1):
                raise RuntimeError(
                    "Alinhamento inválido entre raw_signal_test e OHLC do teste: "
                    f"raw={len(raw_signal_test)}, ohlc_n={len(open_test)}"
                )

            test_result = run_single_backtest(
                raw_signal_test,
                open_test,
                high_test,
                low_test,
                close_test,
                sl_points=best_sl,
                tp_points=best_tp,
                taker_fee=float(backtest_cfg["taker_fee"]),
                position_notional=float(backtest_cfg["position_notional"]),
                signal_threshold=float(backtest_cfg["signal_threshold"]),
                trailing_stop_points=best_trailing_stop,
            )
            test_df_metrics = summarize_window(
                {(best_sl, best_tp, best_trailing_stop): test_result},
                periods_per_year=periods_per_year,
            )
            test_df_metrics.insert(0, "window_id", int(w.window_id))
            test_df_metrics.insert(1, "eval_split", "test")
            test_df_metrics.insert(2, "architectures", ",".join(architectures))
            test_df_metrics.insert(3, "num_checkpoints", len(raw_signal_test_per_ckpt))
            for arch in architectures:
                test_df_metrics[f"num_ckpts_{arch}"] = int(checkpoints_by_arch.get(arch, 0))
            test_df_metrics = _attach_balance_columns(
                test_df_metrics,
                architectures=architectures,
                balance_stats_by_arch=balance_stats_by_arch,
            )
            test_df_metrics["is_best"] = True
            test_df_metrics["best_score"] = best_score
            per_split_metrics.append(test_df_metrics)
            best_result_for_plot = test_result
            opens_for_plot = open_test

        win_df = pd.concat(per_split_metrics, axis=0, ignore_index=True, sort=False)

        win_dir = run_dir / f"window_{w.window_id:03d}"
        # Persist signals for fast dashboard SL/TP grids (prefer test when available).
        # We always write signals_val.csv; when test exists we also write signals_test.csv and
        # set signals.csv to the test split; otherwise signals.csv points to val split.
        _save_signals_csv(
            out_path=win_dir / "signals_val.csv",
            df_slice=val_df,
            seq_len=seq_len,
            raw_signal_per_bar=raw_signal_val,
            signal_threshold=float(backtest_cfg["signal_threshold"]),
            eval_split="val",
            aligned_mask=(valid_val if str(target_cfg["type"]) == "triple_barrier" else None),
        )
        primary_src = win_dir / "signals_val.csv"
        if has_test and test_df is not None:
            _save_signals_csv(
                out_path=win_dir / "signals_test.csv",
                df_slice=test_df,
                seq_len=seq_len,
                raw_signal_per_bar=raw_signal_test,
                signal_threshold=float(backtest_cfg["signal_threshold"]),
                eval_split="test",
            )
            primary_src = win_dir / "signals_test.csv"
        # Copy (not symlink) to a stable name `signals.csv`.
        try:
            import shutil

            shutil.copyfile(primary_src, win_dir / "signals.csv")
        except OSError:
            # fallback: write again under signals.csv
            src_df = pd.read_csv(primary_src)
            src_df.to_csv(win_dir / "signals.csv", index=False)

        save_metrics_csv(win_df, win_dir / "metrics.csv")
        best_equity = best_result_for_plot.equities
        dd_info = max_drawdown_info(best_equity)
        best_sharpe = sharpe_ratio(best_equity, periods_per_year=periods_per_year)
        plot_equity_with_buyhold(
            equities=best_equity,
            opens=opens_for_plot,
            sharpe=best_sharpe,
            max_dd_info=dd_info,
            path=win_dir / f"equity_best_sl{best_sl:g}_tp{best_tp:g}_trail{best_trailing_stop:g}.png",
        )
        rows_per_window.append(win_df)

    if not rows_per_window:
        raise RuntimeError("Nenhuma janela walk-forward foi executada. Ajuste train/val/step/max_windows.")

    summary_path = run_dir / "summary_all_windows.csv"
    save_summary_all_windows(rows_per_window, summary_path)
    summary_df = pd.concat(rows_per_window, axis=0, ignore_index=True, sort=False)
    print_summary(summary_df)
    save_run_summary_row(
        summary_df,
        run_id=run_id,
        cfg=cfg,
        features_used=features_used,
        run_dir=run_dir,
        output_root=output_root,
        config_name=config_name,
        experiment_name=experiment_name,
        batch_id=run_id,
    )
    return summary_df


def run_walkforward(config_path: str | Path, *, max_windows_override: Optional[int] = None) -> tuple[pd.DataFrame, Path]:
    config_abs = _resolve_path(config_path)
    cfg = _load_yaml(config_abs)
    outputs_cfg = cfg["outputs"]
    output_root = _resolve_path(outputs_cfg["output_dir"])
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _save_yaml(run_dir / "config.resolved.yaml", cfg)

    config_name = _resolve_config_name(config_abs, cfg)
    experiments = _resolve_experiments(cfg)
    multi_mode = cfg.get("experiments") is not None
    all_results: list[pd.DataFrame] = []

    for experiment_name, experiment_cfg in experiments:
        exp_dir = run_dir / "experiments" / _slugify(experiment_name) if multi_mode else run_dir
        exp_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 72}\nEXPERIMENT: {experiment_name}\n{'=' * 72}")
        exp_df = _run_single_experiment(
            experiment_cfg,
            run_dir=exp_dir,
            output_root=output_root,
            run_id=run_id,
            config_name=config_name,
            experiment_name=experiment_name,
            max_windows_override=max_windows_override,
        )
        exp_df = exp_df.copy()
        exp_df.insert(0, "experiment_name", experiment_name)
        exp_df.insert(1, "config_name", config_name)
        all_results.append(exp_df)

    merged = pd.concat(all_results, axis=0, ignore_index=True, sort=False)
    return merged, run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa pipeline walk-forward em INF/")
    parser.add_argument(
        "--config",
        type=str,
        default="INF/config.yaml",
        help="Caminho para config.yaml (absoluto ou relativo à raiz do repo)",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Override opcional de walkforward.max_windows",
    )
    args = parser.parse_args()
    run_walkforward(args.config, max_windows_override=args.max_windows)


if __name__ == "__main__":
    main()
