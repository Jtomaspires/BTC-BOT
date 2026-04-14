import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MaxAbsScaler

# CNN_ETH root: three levels up from this file (NN/CNN_ETH)
CNN_ETH_ROOT = Path(__file__).parent.parent.parent / "CNN_ETH"
_ARTIFACTS = CNN_ETH_ROOT / "artifacts"

seq_len: int = 48
num_features: int = 8


def build_features(opens, highs, lows, closes, volumes, train_scalers=None):
    feature1 = (closes - opens) / opens
    feature2 = (highs - opens) / opens
    feature3 = (highs - closes) / closes
    feature4 = (lows - opens) / opens
    feature5 = (lows - closes) / closes
    feature6 = (highs - lows) / opens
    feature7 = (highs - lows) / closes
    feature8 = volumes

    features = [
        feature1, feature2, feature3, feature4,
        feature5, feature6, feature7, feature8,
    ]
    n = len(features)

    is_train = train_scalers is None
    if is_train:
        train_scalers = [MaxAbsScaler() for _ in range(n)]

    scaled_fts = []
    for i in range(n):
        scaler = train_scalers[i]
        if is_train:
            scaled_ft = scaler.fit_transform(features[i].reshape(-1, 1))
        else:
            scaled_ft = scaler.transform(features[i].reshape(-1, 1))
        scaled_fts.append(scaled_ft.flatten())

    scaled_features = np.stack(scaled_fts, axis=-1)
    return scaled_features, n, train_scalers


with open(_ARTIFACTS / "scalers.pkl", "rb") as _f:
    train_scalers = pickle.load(_f)

print("Scalers loaded from CNN_ETH/artifacts/scalers.pkl")
