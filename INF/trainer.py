"""
Treino de uma única janela walk-forward.

Política de checkpointing alinhada ao sandbox CNN (`CNN/CNN_BTC/*`):

- A cada epoch computa **equity na slice de validação** tal como o notebook:
  `actions = preds_val[:, 0] > 0`, bar-a-bar, pos_size = position_notional.
  Long se action=1, short se action=0; pnl = |pos * pct_change|. Sem SL/TP,
  sem fees nesta métrica (idêntico a ``CNN/CNN_BTC/CONV1D_model_training``).
- Guarda `.pt` com o nome ``eq_{equity:.0f}_ep_{epoch:03d}.pt`` apenas quando
  ``val_equity > checkpoint_min_equity`` (default = ``position_notional``).
- Mantém em disco apenas os **top-K** checkpoints por equity (rolling); os
  restantes são apagados no próprio loop (resposta directa ao pedido:
  “guarda apenas os 2/3 melhores, não todos os outros”).
- Ainda regista val_loss (MSE) em ``history`` para logs.

Suporte legacy: ``training.checkpoint_metric: val_loss`` mantém o
comportamento antigo (ficheiro único `best_val_loss.pt`, sem top-K).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from models import get_model, logits_to_signal


@dataclass(frozen=True)
class TrainResult:
    best_checkpoint_paths: List[Path]
    best_val_loss: float
    best_val_equity: float
    history: Dict[str, list[float]]
    num_features: int
    output_dim: int
    best_epoch: int
    checkpoint_metric: str = "val_equity"
    loss_name: str = "mse"
    target_type: str = "next_features"
    balance_stats: Dict[str, Any] | None = None

    @property
    def best_checkpoint_path(self) -> Path:
        """Primeiro (melhor) checkpoint — compat com call-sites antigos."""
        if not self.best_checkpoint_paths:
            raise RuntimeError("Sem checkpoints gravados neste treino")
        return self.best_checkpoint_paths[0]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_sequences(
    features: np.ndarray,
    seq_len: int,
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"features deve ser 2D (n_rows, n_features); recebido {arr.shape}")
    if seq_len <= 0:
        raise ValueError("seq_len deve ser > 0")
    n_rows = arr.shape[0]
    n_samples = n_rows - seq_len
    if n_samples <= 0:
        raise ValueError(
            f"Dados insuficientes para seq_len={seq_len}. n_rows={n_rows} precisa > seq_len."
        )

    x = np.stack([arr[i : i + seq_len] for i in range(n_samples)], axis=0).astype(np.float32)
    if labels is None:
        y = arr[seq_len:].astype(np.float32)
        return x, y

    y_arr = np.asarray(labels).reshape(-1)
    if y_arr.shape[0] != n_rows:
        raise ValueError(
            "labels deve ter o mesmo comprimento de features "
            f"(labels={y_arr.shape[0]}, features={n_rows})"
        )
    y_cls = y_arr[seq_len:].astype(np.int64)
    return x, y_cls


def _choose_device(device_cfg: str) -> torch.device:
    key = str(device_cfg).strip().lower()
    if key == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if key in {"cpu", "cuda"}:
        if key == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(key)
    raise ValueError(f"device inválido: {device_cfg!r}. Use auto/cpu/cuda.")


def _to_tensor(values, name: str, ndim: int) -> torch.Tensor:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != ndim:
        raise ValueError(f"{name} deve ser {ndim}D; recebido shape={arr.shape}")
    return torch.from_numpy(arr)


def _resolve_class_balance_cfg(training_cfg: Mapping[str, Any], *, fallback_threshold: float) -> dict[str, Any]:
    raw = training_cfg.get("class_balance")
    if raw is None:
        return {
            "enabled": False,
            "method": "weighted_sampler",
            "target_channel": 0,
            "threshold": float(fallback_threshold),
            "neutral_policy": "keep",
            "max_neutral_ratio": 1.0,
        }
    if isinstance(raw, str):
        mode = raw.strip().lower()
        if mode in {"weighted", "weighted_sampler"}:
            raw = {"enabled": True, "method": "weighted_sampler"}
        elif mode in {"off", "none", "disabled"}:
            raw = {"enabled": False}
        else:
            raise ValueError(
                "training.class_balance string inválido; use weighted/weighted_sampler/off/none/disabled"
            )
    if not isinstance(raw, Mapping):
        raise ValueError("training.class_balance deve ser string ou mapeamento quando definido")

    enabled = bool(raw.get("enabled", False))
    method = str(raw.get("method", "weighted_sampler")).strip().lower()
    if method not in {"weighted_sampler", "undersample"}:
        raise ValueError("training.class_balance.method deve ser 'weighted_sampler' ou 'undersample'")

    target_channel = int(raw.get("target_channel", 0))
    if target_channel < 0:
        raise ValueError("training.class_balance.target_channel deve ser >= 0")

    threshold_raw = raw.get("threshold", "auto")
    if isinstance(threshold_raw, str):
        if threshold_raw.strip().lower() == "auto":
            threshold = float(fallback_threshold)
        else:
            threshold = float(threshold_raw)
    else:
        threshold = float(threshold_raw)
    if threshold < 0:
        raise ValueError("training.class_balance.threshold deve ser >= 0")

    neutral_policy = str(raw.get("neutral_policy", "keep")).strip().lower()
    if neutral_policy not in {"keep", "drop", "cap_ratio"}:
        raise ValueError("training.class_balance.neutral_policy deve ser keep/drop/cap_ratio")

    max_neutral_ratio = float(raw.get("max_neutral_ratio", 1.0))
    if max_neutral_ratio < 0:
        raise ValueError("training.class_balance.max_neutral_ratio deve ser >= 0")

    return {
        "enabled": enabled,
        "method": method,
        "target_channel": target_channel,
        "threshold": threshold,
        "neutral_policy": neutral_policy,
        "max_neutral_ratio": max_neutral_ratio,
    }


def _class_indices(y: np.ndarray, *, target_channel: int, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(y, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Y_train deve ser 2D; recebido shape={arr.shape}")
    if target_channel >= arr.shape[1]:
        raise ValueError(
            f"training.class_balance.target_channel={target_channel} fora do intervalo "
            f"[0, {arr.shape[1] - 1}]"
        )
    t = arr[:, target_channel].astype(np.float64)
    long_idx = np.where(t > float(threshold))[0].astype(np.int64)
    short_idx = np.where(t < -float(threshold))[0].astype(np.int64)
    neutral_idx = np.where(np.abs(t) <= float(threshold))[0].astype(np.int64)
    return long_idx, short_idx, neutral_idx


def _class_count_stats(long_idx: np.ndarray, short_idx: np.ndarray, neutral_idx: np.ndarray) -> dict[str, Any]:
    total = int(len(long_idx) + len(short_idx) + len(neutral_idx))
    if total <= 0:
        return {
            "count_total": 0,
            "count_long": 0,
            "count_short": 0,
            "count_neutral": 0,
            "ratio_long": 0.0,
            "ratio_short": 0.0,
            "ratio_neutral": 0.0,
        }
    return {
        "count_total": total,
        "count_long": int(len(long_idx)),
        "count_short": int(len(short_idx)),
        "count_neutral": int(len(neutral_idx)),
        "ratio_long": float(len(long_idx) / total),
        "ratio_short": float(len(short_idx) / total),
        "ratio_neutral": float(len(neutral_idx) / total),
    }


def _apply_neutral_policy(
    neutral_idx: np.ndarray,
    *,
    neutral_policy: str,
    max_neutral_ratio: float,
    reference_class_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if neutral_policy == "drop":
        return np.asarray([], dtype=np.int64)
    if neutral_policy == "keep":
        return neutral_idx
    cap = int(np.floor(max_neutral_ratio * max(reference_class_size, 0)))
    if cap <= 0:
        return np.asarray([], dtype=np.int64)
    if len(neutral_idx) <= cap:
        return neutral_idx
    return np.asarray(rng.choice(neutral_idx, size=cap, replace=False), dtype=np.int64)


def _build_undersampled_indices(
    *,
    y_train: np.ndarray,
    target_channel: int,
    threshold: float,
    neutral_policy: str,
    max_neutral_ratio: float,
    seed: int,
) -> tuple[np.ndarray, bool, str]:
    long_idx, short_idx, neutral_idx = _class_indices(
        y_train,
        target_channel=target_channel,
        threshold=threshold,
    )
    if len(long_idx) == 0 or len(short_idx) == 0:
        keep = np.arange(y_train.shape[0], dtype=np.int64)
        return keep, False, "insufficient_directional_samples"

    rng = np.random.default_rng(seed)
    min_class = min(len(long_idx), len(short_idx))
    if len(long_idx) > min_class:
        long_idx = np.asarray(rng.choice(long_idx, size=min_class, replace=False), dtype=np.int64)
    if len(short_idx) > min_class:
        short_idx = np.asarray(rng.choice(short_idx, size=min_class, replace=False), dtype=np.int64)
    neutral_keep = _apply_neutral_policy(
        neutral_idx,
        neutral_policy=neutral_policy,
        max_neutral_ratio=max_neutral_ratio,
        reference_class_size=min_class,
        rng=rng,
    )
    balanced_idx = np.concatenate([long_idx, short_idx, neutral_keep], axis=0)
    rng.shuffle(balanced_idx)
    return balanced_idx.astype(np.int64), True, ""


def _build_weighted_sampler_weights(
    *,
    y_train: np.ndarray,
    target_channel: int,
    threshold: float,
    neutral_policy: str,
    max_neutral_ratio: float,
) -> tuple[np.ndarray, bool, str]:
    long_idx, short_idx, neutral_idx = _class_indices(
        y_train,
        target_channel=target_channel,
        threshold=threshold,
    )
    if len(long_idx) == 0 or len(short_idx) == 0:
        weights = np.ones(y_train.shape[0], dtype=np.float32)
        return weights, False, "insufficient_directional_samples"

    weights = np.zeros(y_train.shape[0], dtype=np.float32)
    weights[long_idx] = np.float32(1.0 / len(long_idx))
    weights[short_idx] = np.float32(1.0 / len(short_idx))

    min_class = min(len(long_idx), len(short_idx))
    neutral_mass = 0.0
    if neutral_policy == "keep":
        neutral_mass = float(len(neutral_idx) / max(min_class, 1))
    elif neutral_policy == "cap_ratio":
        keep_mass = float(len(neutral_idx) / max(min_class, 1))
        neutral_mass = float(min(keep_mass, max_neutral_ratio))

    if len(neutral_idx) > 0 and neutral_mass > 0:
        weights[neutral_idx] = np.float32(neutral_mass / len(neutral_idx))
    if not np.any(weights > 0):
        weights[:] = 1.0
        return weights, False, "all_weights_zero_fallback"
    return weights, True, ""


def _prefix_stats(stats: Mapping[str, Any], *, prefix: str) -> dict[str, Any]:
    return {f"{prefix}_{k}": v for k, v in stats.items()}


def _resolve_loss_name(training_cfg: Mapping[str, Any]) -> str:
    loss_name = str(training_cfg.get("loss", "mse")).strip().lower()
    if loss_name not in {"mse", "cross_entropy"}:
        raise ValueError("training.loss deve ser 'mse' ou 'cross_entropy'")
    return loss_name


def _cross_entropy_class_weights(y_train: np.ndarray) -> np.ndarray:
    labels = np.asarray(y_train, dtype=np.int64).reshape(-1)
    if labels.size == 0:
        raise ValueError("Y_train vazio para cálculo de class weights")
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    if np.any(counts <= 0):
        return np.asarray([], dtype=np.float32)
    weights = 1.0 / counts
    weights = weights / np.sum(weights)
    return weights.astype(np.float32)


def compute_val_equity(
    preds_first_channel: np.ndarray,
    val_opens: np.ndarray,
    val_closes: np.ndarray,
    *,
    position_notional: float,
    signal_threshold: float = 0.0,
) -> float:
    """
    Equity bar-a-bar alinhada à decisão do ``trading_backtest`` do CNN:

    - ``mean_signal[i] = preds_first_channel[i]`` (já é a média do ensemble
      no call-site; aqui é 1 checkpoint por epoch).
    - Se ``|mean_signal| <= signal_threshold`` → **no-trade** (equity inalterada).
    - Caso contrário: long se ``mean_signal > +threshold``, short se
      ``mean_signal < -threshold``. ``pnl = |pos_size * pct_change|`` (sem
      SL/TP e sem fees — paridade com o CNN ``training loop`` mantida; o
      selector de checkpoints só acrescenta o threshold para convergir com
      o backtest final).

    ``signal_threshold=0.0`` recupera o comportamento legacy (long se
    ``pred > 0``, short caso contrário).
    """
    preds = np.asarray(preds_first_channel, dtype=np.float64).reshape(-1)
    opens = np.asarray(val_opens, dtype=np.float64).reshape(-1)
    closes = np.asarray(val_closes, dtype=np.float64).reshape(-1)
    if not (len(preds) == len(opens) == len(closes)):
        raise ValueError(
            "preds_first_channel, val_opens e val_closes devem ter o mesmo comprimento"
        )
    if signal_threshold < 0:
        raise ValueError("signal_threshold deve ser >= 0")

    equity = float(position_notional)
    pos_size = float(position_notional)
    thr = float(signal_threshold)
    legacy = thr == 0.0
    n = len(preds) - 1
    for i in range(n):
        curr_open = float(opens[i])
        curr_close = float(closes[i])
        if curr_open == 0.0:
            continue
        signal = float(preds[i])
        if legacy:
            action = 1 if signal > 0 else -1
        else:
            if signal > thr:
                action = 1
            elif signal < -thr:
                action = -1
            else:
                continue  # no-trade quando sinal está dentro do threshold

        pct_change = (curr_close - curr_open) / curr_open
        pnl = abs(pos_size * pct_change)
        if action == 1:
            if pct_change > 0:
                equity += pnl
            elif pct_change < 0:
                equity -= pnl
        else:
            if pct_change > 0:
                equity -= pnl
            elif pct_change < 0:
                equity += pnl
    return float(equity)


def _delete_outside_top_k(ckpt_dir: Path, keep: Sequence[Path]) -> None:
    keep_names = {Path(p).name for p in keep}
    for p in ckpt_dir.iterdir():
        if not p.is_file() or p.suffix != ".pt":
            continue
        if p.name not in keep_names:
            try:
                p.unlink()
            except OSError:
                pass


def _evaluate_val_loss(model: nn.Module, x_val: torch.Tensor, y_val: torch.Tensor, criterion: nn.Module, device: torch.device) -> tuple[float, np.ndarray]:
    model.eval()
    with torch.no_grad():
        preds = model(x_val.to(device))
        loss = float(criterion(preds, y_val.to(device)).item())
        preds_np = preds.detach().cpu().numpy()
    return loss, preds_np


def train_window(
    *,
    training_cfg: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
    architecture: str,
    X_train,
    Y_train,
    X_val,
    Y_val,
    out_dir: Path,
    window_id: int,
    checkpoint_subdir: Optional[str] = None,
    val_opens: Optional[np.ndarray] = None,
    val_closes: Optional[np.ndarray] = None,
    position_notional: float = 1000.0,
    signal_threshold: float = 0.0,
) -> TrainResult:
    """
    Treina uma janela.

    Se ``checkpoint_metric = "val_equity"`` (default CNN-alinhado), precisa de
    ``val_opens`` e ``val_closes`` (OHLC alinhado às predições, ou seja,
    fatias da val slice após ``seq_len``).
    """
    epochs = int(training_cfg.get("epochs", 1))
    if epochs <= 0:
        raise ValueError("training.epochs deve ser > 0")
    batch_size = int(training_cfg.get("batch_size", 32))
    if batch_size <= 0:
        raise ValueError("training.batch_size deve ser > 0")
    learning_rate = float(training_cfg.get("learning_rate", 1e-3))
    if learning_rate <= 0:
        raise ValueError("training.learning_rate deve ser > 0")

    checkpoint_metric = str(training_cfg.get("checkpoint_metric", "val_equity")).strip().lower()
    if checkpoint_metric not in {"val_loss", "val_equity"}:
        raise ValueError("checkpoint_metric tem de ser 'val_loss' ou 'val_equity'")

    top_k = int(training_cfg.get("top_k", model_cfg.get("top_k", 3)))
    if top_k < 1:
        raise ValueError("training.top_k deve ser >= 1")
    min_equity = float(
        training_cfg.get(
            "checkpoint_min_equity",
            model_cfg.get("checkpoint_min_equity", position_notional),
        )
    )

    seed = int(training_cfg.get("seed", 0))
    set_seed(seed)
    device = _choose_device(str(training_cfg.get("device", "auto")))

    loss_name = _resolve_loss_name(training_cfg)
    target_type = str(training_cfg.get("target_type", "next_features"))
    x_train_np = np.asarray(X_train, dtype=np.float32)
    x_val_np = np.asarray(X_val, dtype=np.float32)
    y_train_np_raw = np.asarray(Y_train)
    y_val_np_raw = np.asarray(Y_val)
    if x_train_np.ndim != 3:
        raise ValueError(f"X_train deve ser 3D; recebido shape={x_train_np.shape}")
    if x_val_np.ndim != 3:
        raise ValueError(f"X_val deve ser 3D; recebido shape={x_val_np.shape}")
    if loss_name == "cross_entropy":
        y_train_np = np.asarray(y_train_np_raw, dtype=np.int64).reshape(-1)
        y_val_np = np.asarray(y_val_np_raw, dtype=np.int64).reshape(-1)
    else:
        y_train_np = np.asarray(y_train_np_raw, dtype=np.float32)
        y_val_np = np.asarray(y_val_np_raw, dtype=np.float32)
        if y_train_np.ndim != 2:
            raise ValueError(f"Y_train deve ser 2D; recebido shape={y_train_np.shape}")
        if y_val_np.ndim != 2:
            raise ValueError(f"Y_val deve ser 2D; recebido shape={y_val_np.shape}")

    if len(x_train_np) != len(y_train_np):
        raise ValueError("X_train e Y_train devem ter o mesmo número de amostras")
    if len(x_val_np) != len(y_val_np):
        raise ValueError("X_val e Y_val devem ter o mesmo número de amostras")
    if len(x_train_np) == 0 or len(x_val_np) == 0:
        raise ValueError("Treino e validação precisam de pelo menos 1 amostra")

    sampler: WeightedRandomSampler | None = None
    sampler_enabled = False
    class_weights_t: torch.Tensor | None = None
    x_train_bal = x_train_np
    balance_stats: Dict[str, Any]
    if loss_name == "cross_entropy":
        if np.any((y_train_np < 0) | (y_train_np > 2)) or np.any((y_val_np < 0) | (y_val_np > 2)):
            raise ValueError("No modo cross_entropy, labels devem estar no intervalo [0, 2]")
        class_balance_raw = training_cfg.get("class_balance")
        weighted_enabled = str(class_balance_raw).strip().lower() in {"weighted", "true", "1"}
        if isinstance(class_balance_raw, Mapping):
            weighted_enabled = bool(class_balance_raw.get("enabled", False))
        before_counts = np.bincount(y_train_np, minlength=3).astype(np.int64)
        if weighted_enabled:
            class_weights = _cross_entropy_class_weights(y_train_np)
            if class_weights.size == 3:
                class_weights_t = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
        balance_stats = {
            "enabled": bool(weighted_enabled),
            "applied": class_weights_t is not None,
            "status": "ok" if class_weights_t is not None else ("class_missing" if weighted_enabled else "disabled"),
            "method": "weighted",
            "target_channel": 0,
            "threshold": 0.0,
            "neutral_policy": "keep",
            "max_neutral_ratio": 1.0,
            "sampler_enabled": False,
            "before_count_total": int(np.sum(before_counts)),
            "before_count_long": int(before_counts[1]),
            "before_count_short": int(before_counts[2]),
            "before_count_neutral": int(before_counts[0]),
            "after_count_total": int(np.sum(before_counts)),
            "after_count_long": int(before_counts[1]),
            "after_count_short": int(before_counts[2]),
            "after_count_neutral": int(before_counts[0]),
        }
    else:
        balance_cfg = _resolve_class_balance_cfg(
            training_cfg,
            fallback_threshold=float(signal_threshold),
        )
        long_before, short_before, neutral_before = _class_indices(
            y_train_np,
            target_channel=int(balance_cfg["target_channel"]),
            threshold=float(balance_cfg["threshold"]),
        )
        before_stats = _class_count_stats(long_before, short_before, neutral_before)

        y_train_bal = y_train_np
        applied = False
        status = "disabled"
        if bool(balance_cfg["enabled"]):
            if str(balance_cfg["method"]) == "undersample":
                idx, applied, note = _build_undersampled_indices(
                    y_train=y_train_np,
                    target_channel=int(balance_cfg["target_channel"]),
                    threshold=float(balance_cfg["threshold"]),
                    neutral_policy=str(balance_cfg["neutral_policy"]),
                    max_neutral_ratio=float(balance_cfg["max_neutral_ratio"]),
                    seed=seed,
                )
                x_train_bal = x_train_np[idx]
                y_train_bal = y_train_np[idx]
                status = note if note else "ok"
            else:
                weights, applied, note = _build_weighted_sampler_weights(
                    y_train=y_train_np,
                    target_channel=int(balance_cfg["target_channel"]),
                    threshold=float(balance_cfg["threshold"]),
                    neutral_policy=str(balance_cfg["neutral_policy"]),
                    max_neutral_ratio=float(balance_cfg["max_neutral_ratio"]),
                )
                if applied:
                    weight_t = torch.as_tensor(weights, dtype=torch.float32)
                    sampler_generator = torch.Generator()
                    sampler_generator.manual_seed(seed)
                    sampler = WeightedRandomSampler(
                        weight_t,
                        num_samples=len(weight_t),
                        replacement=True,
                        generator=sampler_generator,
                    )
                    sampler_enabled = True
                status = note if note else "ok"

        long_after, short_after, neutral_after = _class_indices(
            y_train_bal,
            target_channel=int(balance_cfg["target_channel"]),
            threshold=float(balance_cfg["threshold"]),
        )
        after_stats = _class_count_stats(long_after, short_after, neutral_after)
        balance_stats = {
            "enabled": bool(balance_cfg["enabled"]),
            "applied": bool(applied),
            "status": status,
            "method": str(balance_cfg["method"]),
            "target_channel": int(balance_cfg["target_channel"]),
            "threshold": float(balance_cfg["threshold"]),
            "neutral_policy": str(balance_cfg["neutral_policy"]),
            "max_neutral_ratio": float(balance_cfg["max_neutral_ratio"]),
            "sampler_enabled": bool(sampler_enabled),
        }
        balance_stats.update(_prefix_stats(before_stats, prefix="before"))
        balance_stats.update(_prefix_stats(after_stats, prefix="after"))
        if len(x_train_bal) == 0:
            raise ValueError("class_balance removeu todas as amostras de treino; ajuste a configuração")

    x_train = _to_tensor(x_train_bal, "X_train", 3)
    x_val = _to_tensor(x_val_np, "X_val", 3)
    if loss_name == "cross_entropy":
        y_train = torch.as_tensor(y_train_np, dtype=torch.long)
        y_val = torch.as_tensor(y_val_np, dtype=torch.long)
    else:
        y_train = _to_tensor(y_train_bal, "Y_train", 2)
        y_val = _to_tensor(y_val_np, "Y_val", 2)

    if checkpoint_metric == "val_equity":
        if val_opens is None or val_closes is None:
            raise ValueError(
                "checkpoint_metric=val_equity requer val_opens e val_closes "
                "alinhados às amostras X_val/Y_val (pós seq_len)."
            )
        if not (len(val_opens) == len(val_closes) == len(x_val)):
            raise ValueError(
                "val_opens/val_closes devem ter o mesmo comprimento de X_val "
                f"(recebido opens={len(val_opens)}, closes={len(val_closes)}, x_val={len(x_val)})"
            )

    num_features = int(x_train.shape[-1])
    output_dim = int(model_cfg.get("output_dim", 3)) if loss_name == "cross_entropy" else int(num_features)
    arch = architecture or str(model_cfg.get("architecture", "conv1d"))
    model = get_model(arch, num_features, output_dim=output_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion: nn.Module = (
        nn.CrossEntropyLoss(weight=class_weights_t)
        if loss_name == "cross_entropy"
        else nn.MSELoss()
    )
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=False,
    )

    history: Dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_equity": []}
    best_val_loss = float("inf")
    best_val_equity = -float("inf")
    best_epoch = -1

    early_cfg = training_cfg.get("early_stopping", {}) or {}
    early_enabled = bool(early_cfg.get("enabled", False))
    early_patience = int(early_cfg.get("patience", 10))
    early_min_delta = float(early_cfg.get("min_delta", 0.0))
    stale_epochs = 0

    run_window_dir = Path(out_dir) / f"window_{window_id:03d}"
    ckpt_dir = run_window_dir / "checkpoints"
    if checkpoint_subdir:
        ckpt_dir = ckpt_dir / str(checkpoint_subdir).strip()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Heap de tamanho K: (equity, epoch, path). Ordenado desc por equity.
    top_k_entries: List[tuple[float, int, Path]] = []

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        epoch_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        history["train_loss"].append(epoch_train_loss)

        val_loss, preds_val = _evaluate_val_loss(model, x_val, y_val, criterion, device)
        history["val_loss"].append(val_loss)

        val_equity = float("nan")
        if checkpoint_metric == "val_equity":
            val_signal = logits_to_signal(preds_val) if loss_name == "cross_entropy" else preds_val[:, 0]
            val_equity = compute_val_equity(
                val_signal,
                np.asarray(val_opens, dtype=np.float64),
                np.asarray(val_closes, dtype=np.float64),
                position_notional=position_notional,
                signal_threshold=float(signal_threshold),
            )
        history["val_equity"].append(val_equity)

        improved = False
        if checkpoint_metric == "val_loss":
            if val_loss < (best_val_loss - early_min_delta):
                improved = True
                best_val_loss = val_loss
                best_epoch = epoch
                best_ckpt_path = ckpt_dir / "best_val_loss.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": int(epoch),
                        "val_loss": float(val_loss),
                        "architecture": arch,
                        "num_features": int(num_features),
                        "output_dim": int(output_dim),
                        "loss": loss_name,
                        "target_type": target_type,
                    },
                    best_ckpt_path,
                )
                top_k_entries = [(-val_loss, epoch, best_ckpt_path)]
        else:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            if val_equity > best_val_equity:
                best_val_equity = val_equity
                best_epoch = epoch
                improved = True

            if val_equity > min_equity:
                ckpt_path = ckpt_dir / f"eq_{val_equity:.0f}_ep_{epoch:03d}.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": int(epoch),
                        "val_equity": float(val_equity),
                        "val_loss": float(val_loss),
                        "architecture": arch,
                        "num_features": int(num_features),
                        "output_dim": int(output_dim),
                        "loss": loss_name,
                        "target_type": target_type,
                    },
                    ckpt_path,
                )
                top_k_entries.append((val_equity, epoch, ckpt_path))
                top_k_entries.sort(key=lambda t: (t[0], t[1]), reverse=True)
                top_k_entries = top_k_entries[:top_k]
                _delete_outside_top_k(ckpt_dir, [p for _, _, p in top_k_entries])

        if improved:
            stale_epochs = 0
        else:
            stale_epochs += 1
        if early_enabled and stale_epochs >= early_patience:
            break

    if not top_k_entries:
        raise RuntimeError(
            "Nenhum checkpoint foi gravado. "
            "Se checkpoint_metric=val_equity, talvez a equity nunca tenha passado de "
            f"checkpoint_min_equity={min_equity}. Considere reduzir esse limite, "
            "aumentar epochs, ou usar checkpoint_metric=val_loss."
        )

    return TrainResult(
        best_checkpoint_paths=[p for _, _, p in top_k_entries],
        best_val_loss=float(best_val_loss),
        best_val_equity=float(best_val_equity) if np.isfinite(best_val_equity) else float("nan"),
        history=history,
        num_features=num_features,
        output_dim=output_dim,
        best_epoch=best_epoch,
        checkpoint_metric=checkpoint_metric,
        loss_name=loss_name,
        target_type=target_type,
        balance_stats=balance_stats,
    )
