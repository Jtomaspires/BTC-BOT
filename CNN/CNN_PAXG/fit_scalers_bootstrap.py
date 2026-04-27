"""Fit MaxAbsScaler list on train slice and write artifacts/scalers.pkl.
Uses the same 8 features as the training/backtest notebooks.

Respects artifacts/split_info.json:
- train_start (optional; defaults to 0)
- train_end (required)
"""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler

CNN_ROOT = Path(__file__).resolve().parent
DATA_CSV = CNN_ROOT / "data" / "PAXGUSDT-1h-data.csv"
ARTIFACTS = CNN_ROOT / "artifacts"

df = pd.read_csv(DATA_CSV)
with open(ARTIFACTS / "split_info.json", "r", encoding="utf-8") as f:
    import json

    sp = json.load(f)
train_start = int(sp.get("train_start", 0))
train_end = int(sp["train_end"])
sub = df.iloc[train_start:train_end]
opens = sub["open"].values
highs = sub["high"].values
lows = sub["low"].values
closes = sub["close"].values
volumes = sub["volume"].values

feature1 = (closes - opens) / opens
feature2 = (highs - opens) / opens
feature3 = (highs - closes) / closes
feature4 = (lows - opens) / opens
feature5 = (lows - closes) / closes
feature6 = (highs - lows) / opens
feature7 = (highs - lows) / closes
feature8 = volumes
features = [feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8]

scalers = [MaxAbsScaler() for _ in range(8)]
for i in range(8):
    scalers[i].fit(features[i].reshape(-1, 1))

ARTIFACTS.mkdir(parents=True, exist_ok=True)
with open(ARTIFACTS / "scalers.pkl", "wb") as f:
    pickle.dump(scalers, f)
print("Wrote", ARTIFACTS / "scalers.pkl")
