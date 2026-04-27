"""
Arquiteturas espelhadas de `CNN/CNN_BTC/backtest/main.ipynb` (Model_1/2/3),
com ``num_features`` injectável (no notebook era variável global).

Input: ``(batch, seq_len, num_features)`` — igual ao notebook.
"""

from __future__ import annotations

from typing import Dict, Sequence, Type

import torch
import torch.nn as nn

from features import NUM_FEATURES


class Model_1(nn.Module):
    """Conv1D — chave ``conv1d``."""

    def __init__(self, num_features: int = NUM_FEATURES):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=num_features,
            out_channels=32,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1,
        )
        self.act2 = nn.GELU()
        self.fc_out = nn.Linear(64, num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = x.permute(0, 2, 1)
        x = x[:, -1, :]
        x = self.fc_out(x)
        return x


class Model_2(nn.Module):
    """LSTM — chave ``lstm``."""

    def __init__(self, num_features: int = NUM_FEATURES):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.fc_out = nn.Linear(64, num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc_out(x)
        return x


class Model_3(nn.Module):
    """Conv1D + LSTM — chave ``hybrid``."""

    def __init__(self, num_features: int = NUM_FEATURES):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=num_features,
            out_channels=64,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.act1 = nn.GELU()
        self.lstm = nn.LSTM(64, 32, batch_first=True)
        self.fc_out = nn.Linear(32, num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.act1(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc_out(x)
        return x


ARCH_MAP: Dict[str, Type[nn.Module]] = {
    "conv1d": Model_1,
    "lstm": Model_2,
    "hybrid": Model_3,
}


def get_model(architecture: str, num_features: int = NUM_FEATURES) -> nn.Module:
    key = architecture.lower().strip()
    if key not in ARCH_MAP:
        raise ValueError(f"Arquitetura desconhecida: {architecture!r}. Usar uma de {sorted(ARCH_MAP)}.")
    return ARCH_MAP[key](num_features=num_features)


def get_action(raw_signals: Sequence[float], threshold: float) -> int:
    """
    Mesma regra que o ensemble no backtest: média dos sinais vs ``threshold``.
    """
    if not raw_signals:
        return 0
    mean_signal = float(sum(raw_signals) / len(raw_signals))
    if mean_signal > threshold:
        return 1
    if mean_signal < -threshold:
        return -1
    return 0
