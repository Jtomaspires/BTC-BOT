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
from torch.utils.data import DataLoader, TensorDataset

from models import get_model


@dataclass(frozen=True)
class TrainResult:
    best_checkpoint_paths: List[Path]
    best_val_loss: float
    best_val_equity: float
    history: Dict[str, list[float]]
    num_features: int
    best_epoch: int
    checkpoint_metric: str = "val_equity"

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


def build_sequences(features: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
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
    y = arr[seq_len:].astype(np.float32)
    return x, y


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

    x_train = _to_tensor(X_train, "X_train", 3)
    y_train = _to_tensor(Y_train, "Y_train", 2)
    x_val = _to_tensor(X_val, "X_val", 3)
    y_val = _to_tensor(Y_val, "Y_val", 2)

    if len(x_train) != len(y_train):
        raise ValueError("X_train e Y_train devem ter o mesmo número de amostras")
    if len(x_val) != len(y_val):
        raise ValueError("X_val e Y_val devem ter o mesmo número de amostras")
    if len(x_train) == 0 or len(x_val) == 0:
        raise ValueError("Treino e validação precisam de pelo menos 1 amostra")

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
    arch = architecture or str(model_cfg.get("architecture", "conv1d"))
    model = get_model(arch, num_features).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion: nn.Module = nn.MSELoss()
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
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
            val_equity = compute_val_equity(
                preds_val[:, 0],
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
        best_epoch=best_epoch,
        checkpoint_metric=checkpoint_metric,
    )
