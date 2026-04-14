import json

import numpy as np
import torch
import torch.nn as nn

from .load_scalers import build_features, seq_len, num_features, CNN_ETH_ROOT

THRESHOLD = 0.0007


class Model_1(nn.Module):
    """Conv1D architecture (matches CNN_ETH CONV1D_model_training)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(num_features, 32, kernel_size=3, padding=1, stride=1)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.act2 = nn.GELU()
        self.fc_out = nn.Linear(64, num_features)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = x.permute(0, 2, 1)
        x = x[:, -1, :]
        return self.fc_out(x)


class Model_2(nn.Module):
    """LSTM architecture (matches CNN_ETH LSTM_model_training)."""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(num_features, 64, num_layers=1, batch_first=True)
        self.fc_out = nn.Linear(64, num_features)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.fc_out(x)


class Model_3(nn.Module):
    """Hybrid Conv1D + LSTM architecture (matches CNN_ETH hybrid_model_training)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(num_features, 64, kernel_size=3, padding=1, stride=1)
        self.act1 = nn.GELU()
        self.lstm = nn.LSTM(64, 32, batch_first=True)
        self.fc_out = nn.Linear(32, num_features)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.act1(self.conv1(x))
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.fc_out(x)


ARCH_MAP = {"conv1d": Model_1, "lstm": Model_2, "hybrid": Model_3}

MODELS = []


def _validate_manifest(manifest: object) -> dict:
    """
    Validate that manifest is dict[str, list[str]] with known architecture keys.
    Raises ValueError with a descriptive message on any schema violation.
    """
    known_archs = set(ARCH_MAP.keys())

    if not isinstance(manifest, dict):
        raise ValueError(
            f"manifest.json must be a JSON object (dict), got {type(manifest).__name__}."
        )

    for key, value in manifest.items():
        if not isinstance(key, str):
            raise ValueError(
                f"manifest.json keys must be strings, got {type(key).__name__!r} for key {key!r}."
            )
        if key not in known_archs:
            raise ValueError(
                f"Unknown architecture key {key!r} in manifest.json. "
                f"Supported keys: {sorted(known_archs)}."
            )
        if not isinstance(value, list):
            raise ValueError(
                f"manifest.json[{key!r}] must be a list of path strings, "
                f"got {type(value).__name__}."
            )
        for i, item in enumerate(value):
            if not isinstance(item, str):
                raise ValueError(
                    f"manifest.json[{key!r}][{i}] must be a string path, "
                    f"got {type(item).__name__}: {item!r}."
                )

    return manifest


def load_model():
    """Load all checkpoints listed in CNN_ETH/artifacts/manifest.json."""
    manifest_path = CNN_ETH_ROOT / "artifacts" / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    manifest = _validate_manifest(raw)

    for arch_name, rel_paths in manifest.items():
        arch_cls = ARCH_MAP[arch_name]  # key already validated
        for rel_path in rel_paths:
            full_path = CNN_ETH_ROOT / rel_path
            if not full_path.exists():
                print(f"Checkpoint not found: {full_path} — skipping.")
                continue
            model = arch_cls()
            model.load_state_dict(
                torch.load(str(full_path), map_location="cpu", weights_only=False)
            )
            model.eval()
            MODELS.append(model)
            print(f"Loaded [{arch_name}] {full_path.name}")

    if not MODELS:
        raise FileNotFoundError(
            "No checkpoints were loaded. Check CNN_ETH/artifacts/manifest.json and model paths."
        )


def _preprocess(opens, highs, lows, closes, volumes, train_scalers):
    """Scale raw OHLCV arrays into a (1, seq_len, num_features) tensor."""
    features, _, _ = build_features(opens, highs, lows, closes, volumes, train_scalers)
    return np.expand_dims(features, axis=0).astype(np.float32)


def get_action(opens, highs, lows, closes, volumes, train_scalers):
    """
    Run ensemble inference and return:
      +1  → long signal
      -1  → short signal
       0  → neutral (no trade)
    Logic mirrors CNN_ETH/backtest/main.ipynb:
      mean of y_pred[:, 0] across all models, compared to THRESHOLD=0.0007.
    """
    x = torch.from_numpy(_preprocess(opens, highs, lows, closes, volumes, train_scalers))

    raw_values = []
    with torch.no_grad():
        for model in MODELS:
            y_pred = model(x).detach().cpu().numpy()
            raw_values.append(float(y_pred[0, 0]))

    if not raw_values:
        return 0

    mean_signal = float(np.mean(raw_values))

    if mean_signal > THRESHOLD:
        return 1
    elif mean_signal < -THRESHOLD:
        return -1
    else:
        return 0
