"""One-off patcher for CNN_SOL sandbox notebooks. Run from repo: python CNN_SOL/_patch_notebooks.py"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# End-aligned split (same as training notebooks); with L=69000 -> 52500/61500/63000
SPLIT_END_ALIGNED = """TRAIN_BARS = 9000
VAL_BARS = 1500
HOLDOUT_BARS = 6000
L = len(df)
min_rows = TRAIN_BARS + VAL_BARS + HOLDOUT_BARS
assert L >= min_rows, f"need at least {min_rows} rows, got {L}"
total_rows = L
val_end = L - HOLDOUT_BARS
train_end = val_end - VAL_BARS
train_start = train_end - TRAIN_BARS
TRAIN_START = train_start
TRAIN_END = train_end
VAL_END = val_end
assert 0 <= TRAIN_START < TRAIN_END < VAL_END <= L, f"{TRAIN_START=} {TRAIN_END=} {VAL_END=} len(df)={L}"

df_train = df.iloc[TRAIN_START:TRAIN_END].copy()
df_val = df.iloc[TRAIN_END:VAL_END].copy()

print(f"Train rows [{TRAIN_START}:{TRAIN_END}]: {len(df_train)}")
print(f"Val rows   [{TRAIN_END}:{VAL_END}]: {len(df_val)} (equity + checkpoints; not holdout)")
print(f"Holdout    [{VAL_END}:] reserved for CNN/backtest (not used in this notebook)")"""


def load_nb(rel: str):
    p = ROOT / rel
    return json.loads(p.read_text(encoding="utf-8")), p


def save_nb(nb, p: Path):
    p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def replace_in_cell_sources(nb, old: str, new: str, must_contain: str | None = None) -> int:
    n = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = cell["source"]
        if isinstance(src, str):
            text = src
        else:
            text = "".join(src)
        if must_contain and must_contain not in text:
            continue
        if old not in text:
            continue
        text = text.replace(old, new)
        cell["source"] = [line + "\n" for line in text.split("\n")]
        if cell["source"] and cell["source"][-1] == "\n":
            cell["source"].pop()
        n += 1
    return n


def set_cell_source(nb, idx: int, new_text: str):
    lines = new_text.split("\n")
    nb["cells"][idx]["source"] = [ln + "\n" for ln in lines[:-1]] + ([lines[-1] + "\n"] if lines else [])


def append_artifacts_cell(nb):
    art_lines = [
        "import json",
        "import re",
        "",
        'split_info = {"train_start": TRAIN_START, "train_end": TRAIN_END, "val_end": VAL_END, "total_rows": total_rows}',
        'with open(ARTIFACTS / "split_info.json", "w", encoding="utf-8") as f:',
        "    json.dump(split_info, f, indent=2)",
        "",
        'with open(ARTIFACTS / "scalers.pkl", "wb") as f:',
        "    pickle.dump(train_scalers, f)",
        "",
        'models_dir = Path("models").resolve()',
        "pt_files = []",
        "for fname in os.listdir(models_dir):",
        '    if not fname.endswith(".pt"):',
        "        continue",
        '    m = re.match(r"^eq_(\\d+)_ep_", fname)',
        "    if not m:",
        "        continue",
        "    eq = float(m.group(1))",
        "    rel_path = os.path.relpath(str(models_dir / fname), str(CNN_ROOT)).replace(os.sep, \"/\")",
        "    pt_files.append((eq, rel_path))",
        "",
        "pt_files.sort(key=lambda x: -x[0])",
        "top3 = [p for _, p in pt_files[:3]]",
        "",
        'manifest_path = ARTIFACTS / "manifest.json"',
        "manifest = {}",
        "if manifest_path.is_file():",
        '    with open(manifest_path, "r", encoding="utf-8") as f:',
        "        manifest = json.load(f)",
        "manifest[ARCH_MANIFEST_KEY] = top3",
        'with open(manifest_path, "w", encoding="utf-8") as f:',
        "    json.dump(manifest, f, indent=2)",
        "",
        'print("Written:", ARTIFACTS / "split_info.json", ARTIFACTS / "scalers.pkl", manifest_path)',
        'print("manifest key:", ARCH_MANIFEST_KEY, "top3:", top3)',
    ]
    art = "\n".join(art_lines)
    nb["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in art.split("\n")[:-1]] + [art.split("\n")[-1] + "\n"],
        }
    )


def patch_lstm():
    nb, p = load_nb("LSTM_model_training/LSTM_model_training/main.ipynb")
    c0 = """import os
from pathlib import Path
from datetime import datetime
import pickle
import random
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler

import torch
import torch.nn as nn
import torch.nn.functional as F


def _find_cnn_root():
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        if (p / "data" / "SOLUSDT-1h-data.csv").is_file():
            return p
        cand = p / "CNN_SOL" / "data" / "SOLUSDT-1h-data.csv"
        if cand.is_file():
            return (p / "CNN_SOL").resolve()
    return here


CNN_ROOT = _find_cnn_root()
ARTIFACTS = CNN_ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
DATA_CSV = CNN_ROOT / "data" / "SOLUSDT-1h-data.csv"
ARCH_MANIFEST_KEY = "lstm"

os.makedirs("models", exist_ok=True)

def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

seed = 0
set_all_seeds(seed)"""
    set_cell_source(nb, 0, c0)
    set_cell_source(nb, 1, "df = pd.read_csv(DATA_CSV)\nprint(df)")
    set_cell_source(nb, 5, SPLIT_END_ALIGNED)
    set_cell_source(
        nb,
        12,
        """# Sequence Length
seq_len = 48

# Preprocess data (fit scalers on train only; transform val slice for epoch equity)
X_train, Y_train, num_features, _, _, train_scalers = preprocess_data(seq_len, df_train)
X_val, Y_val, num_features, val_opens, val_closes, _ = preprocess_data(seq_len, df_val, train_scalers)
""",
    )
    set_cell_source(
        nb,
        13,
        """m_train = X_train.shape[0]
m_val = X_val.shape[0]
print(f"m_train: {m_train}")
print(f"m_val: {m_val}")
print(f"X_train shape: {X_train.shape}")
print(f"Y_train shape: {Y_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"Y_val shape: {Y_val.shape}")""",
    )
    set_cell_source(
        nb,
        15,
        """device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print("my deive: ", device)

X_train = torch.from_numpy(X_train.astype(np.float32)).to(device, dtype=torch.float32)
Y_train = torch.from_numpy(Y_train.astype(np.float32)).to(device, dtype=torch.float32)

X_val = torch.from_numpy(X_val.astype(np.float32)).to(device, dtype=torch.float32)
Y_val = torch.from_numpy(Y_val.astype(np.float32)).to(device, dtype=torch.float32)

Y_pred_val = torch.zeros([m_val, num_features], device=device, dtype=torch.float32)""",
    )
    n = replace_in_cell_sources(nb, "m_test", "m_val", must_contain="for epoch in range")
    assert n >= 1, "epoch loop m_test"
    n2 = replace_in_cell_sources(nb, "X_test", "X_val", must_contain="for epoch in range")
    n3 = replace_in_cell_sources(nb, "Y_test", "Y_val", must_contain="for epoch in range")
    n4 = replace_in_cell_sources(nb, "Y_pred_test", "Y_pred_val", must_contain="for epoch in range")
    n5 = replace_in_cell_sources(nb, "backtest_opens", "val_opens", must_contain="for epoch in range")
    n6 = replace_in_cell_sources(nb, "backtest_closes", "val_closes", must_contain="for epoch in range")
    # doc comments in epoch cell
    replace_in_cell_sources(nb, "# Predict on test data", "# Predict on val slice (selection period)", must_contain="for epoch in range")
    replace_in_cell_sources(nb, "# Convert 'Y_pred_test'", "# Convert 'Y_pred_val'", must_contain="for epoch in range")
    append_artifacts_cell(nb)
    save_nb(nb, p)
    print("patched LSTM", p)


def patch_conv1d():
    nb, p = load_nb("CONV1D_model_training/CONV1D_model_training/main.ipynb")
    # find cell indices by content
    def find_cell(pred):
        for i, c in enumerate(nb["cells"]):
            if c.get("cell_type") != "code":
                continue
            t = "".join(c.get("source", []))
            if pred(t):
                return i, t
        raise RuntimeError("cell not found")

    i0, _ = find_cell(lambda t: t.strip().startswith("import os") and "set_all_seeds" in t)
    i_df, _ = find_cell(lambda t: "read_csv" in t and ("DATA_CSV" in t or "BTCUSDT" in t or "SOLUSDT" in t))
    i_split, _ = find_cell(lambda t: "TRAIN_BARS" in t and "df_train" in t and "df_val" in t)
    i_pre, _ = find_cell(lambda t: "X_train, Y_train" in t and "preprocess_data(seq_len, df_train)" in t)
    i_m, _ = find_cell(lambda t: "m_train = X_train.shape[0]" in t and "m_test" in t)
    i_dev, _ = find_cell(lambda t: "Y_pred_test = torch.zeros" in t)
    i_epoch, _ = find_cell(lambda t: "for epoch in range(epoch, num_epochs):" in t)

    c0 = nb["cells"][i0]["source"]
    t0 = "".join(c0)
    if "def _find_cnn_root" not in t0:
        t0 = t0.replace(
            "import os\n",
            "import os\nfrom pathlib import Path\n",
        )
        t0 = t0.replace(
            "import torch.nn.functional as F\n\nos.makedirs",
            """import torch.nn.functional as F


def _find_cnn_root():
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        if (p / "data" / "SOLUSDT-1h-data.csv").is_file():
            return p
        cand = p / "CNN_SOL" / "data" / "SOLUSDT-1h-data.csv"
        if cand.is_file():
            return (p / "CNN_SOL").resolve()
    return here


CNN_ROOT = _find_cnn_root()
ARTIFACTS = CNN_ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
DATA_CSV = CNN_ROOT / "data" / "SOLUSDT-1h-data.csv"
ARCH_MANIFEST_KEY = "conv1d"

os.makedirs""",
        )
        nb["cells"][i0]["source"] = [x + "\n" for x in t0.split("\n")[:-1]] + [t0.split("\n")[-1] + "\n"]

    set_cell_source(nb, i_df, "df = pd.read_csv(DATA_CSV)\nprint(df)")
    set_cell_source(nb, i_split, SPLIT_END_ALIGNED)
    set_cell_source(
        nb,
        i_pre,
        """# Sequence Length
seq_len = 48

# Preprocess data (fit scalers on train only; transform val slice for epoch equity)
X_train, Y_train, num_features, _, _, train_scalers = preprocess_data(seq_len, df_train)
X_val, Y_val, num_features, val_opens, val_closes, _ = preprocess_data(seq_len, df_val, train_scalers)
""",
    )
    t_m = "".join(nb["cells"][i_m]["source"])
    t_m = t_m.replace("m_test = X_test.shape[0]", "m_val = X_val.shape[0]")
    t_m = t_m.replace("m_test:", "m_val:")
    t_m = t_m.replace("X_test shape", "X_val shape")
    t_m = t_m.replace("Y_test shape", "Y_val shape")
    t_m = t_m.replace("{m_test}", "{m_val}")
    t_m = t_m.replace("{X_test.shape}", "{X_val.shape}")
    t_m = t_m.replace("{Y_test.shape}", "{Y_val.shape}")
    nb["cells"][i_m]["source"] = [x + "\n" for x in t_m.split("\n")[:-1]] + [t_m.split("\n")[-1] + "\n"]

    set_cell_source(
        nb,
        i_dev,
        """device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print("my deive: ", device)

X_train = torch.from_numpy(X_train.astype(np.float32)).to(device, dtype=torch.float32)
Y_train = torch.from_numpy(Y_train.astype(np.float32)).to(device, dtype=torch.float32)

X_val = torch.from_numpy(X_val.astype(np.float32)).to(device, dtype=torch.float32)
Y_val = torch.from_numpy(Y_val.astype(np.float32)).to(device, dtype=torch.float32)

Y_pred_val = torch.zeros([m_val, num_features], device=device, dtype=torch.float32)""",
    )

    replace_in_cell_sources(nb, "m_test", "m_val", must_contain="for epoch in range")
    replace_in_cell_sources(nb, "X_test", "X_val", must_contain="for epoch in range")
    replace_in_cell_sources(nb, "Y_test", "Y_val", must_contain="for epoch in range")
    replace_in_cell_sources(nb, "Y_pred_test", "Y_pred_val", must_contain="for epoch in range")
    replace_in_cell_sources(nb, "backtest_opens", "val_opens", must_contain="for epoch in range")
    replace_in_cell_sources(nb, "backtest_closes", "val_closes", must_contain="for epoch in range")
    replace_in_cell_sources(nb, "# Predict on test data", "# Predict on val slice (selection period)", must_contain="for epoch in range")
    replace_in_cell_sources(nb, "# Convert 'Y_pred_test'", "# Convert 'Y_pred_val'", must_contain="for epoch in range")

    append_artifacts_cell(nb)
    save_nb(nb, p)
    print("patched CONV1D", p)


def patch_hybrid():
    nb, p = load_nb("hybrid_model_training/main.ipynb")

    def find_cell(pred):
        for i, c in enumerate(nb["cells"]):
            if c.get("cell_type") != "code":
                continue
            t = "".join(c.get("source", []))
            if pred(t):
                return i, t
        raise RuntimeError("cell not found")

    i0, t0 = find_cell(lambda t: t.strip().startswith("import os") and "set_all_seeds" in t)
    if "def _find_cnn_root" not in t0:
        t0 = t0.replace("import os\n", "import os\nfrom pathlib import Path\n")
        t0 = t0.replace(
            "import torch.nn.functional as F\n\nos.makedirs",
            """import torch.nn.functional as F


def _find_cnn_root():
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        if (p / "data" / "SOLUSDT-1h-data.csv").is_file():
            return p
        cand = p / "CNN_SOL" / "data" / "SOLUSDT-1h-data.csv"
        if cand.is_file():
            return (p / "CNN_SOL").resolve()
    return here


CNN_ROOT = _find_cnn_root()
ARTIFACTS = CNN_ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
DATA_CSV = CNN_ROOT / "data" / "SOLUSDT-1h-data.csv"
ARCH_MANIFEST_KEY = "hybrid"

os.makedirs""",
        )
        nb["cells"][i0]["source"] = [x + "\n" for x in t0.split("\n")[:-1]] + [t0.split("\n")[-1] + "\n"]

    i_df, _ = find_cell(lambda t: "read_csv" in t and ("DATA_CSV" in t or "BTCUSDT" in t or "SOLUSDT" in t))
    i_split, _ = find_cell(lambda t: "TRAIN_BARS" in t and "df_train" in t and "df_val" in t)
    i_pre, _ = find_cell(lambda t: "X_train, Y_train" in t and "preprocess_data(seq_len, df_train)" in t)
    i_m, _ = find_cell(lambda t: "m_train = X_train.shape[0]" in t and "m_test" in t)
    i_dev, _ = find_cell(lambda t: "Y_pred_test = torch.zeros" in t)
    i_epoch, _ = find_cell(lambda t: "for epoch in range(epoch, num_epochs):" in t)

    set_cell_source(nb, i_df, "df = pd.read_csv(DATA_CSV)\nprint(df)")
    set_cell_source(nb, i_split, SPLIT_END_ALIGNED)
    set_cell_source(
        nb,
        i_pre,
        """# Sequence Length
seq_len = 48

# Preprocess data (fit scalers on train only; transform val slice for epoch equity)
X_train, Y_train, num_features, _, _, train_scalers = preprocess_data(seq_len, df_train)
X_val, Y_val, num_features, val_opens, val_closes, _ = preprocess_data(seq_len, df_val, train_scalers)
""",
    )
    t_m = "".join(nb["cells"][i_m]["source"])
    t_m = t_m.replace("m_test = X_test.shape[0]", "m_val = X_val.shape[0]")
    t_m = t_m.replace("m_test:", "m_val:")
    t_m = t_m.replace("X_test shape", "X_val shape")
    t_m = t_m.replace("Y_test shape", "Y_val shape")
    nb["cells"][i_m]["source"] = [x + "\n" for x in t_m.split("\n")[:-1]] + [t_m.split("\n")[-1] + "\n"]

    set_cell_source(
        nb,
        i_dev,
        """device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print("my deive: ", device)

X_train = torch.from_numpy(X_train.astype(np.float32)).to(device, dtype=torch.float32)
Y_train = torch.from_numpy(Y_train.astype(np.float32)).to(device, dtype=torch.float32)

X_val = torch.from_numpy(X_val.astype(np.float32)).to(device, dtype=torch.float32)
Y_val = torch.from_numpy(Y_val.astype(np.float32)).to(device, dtype=torch.float32)

Y_pred_val = torch.zeros([m_val, num_features], device=device, dtype=torch.float32)""",
    )

    replace_in_cell_sources(nb, "m_test", "m_val", must_contain="for epoch in range")
    replace_in_cell_sources(nb, "X_test", "X_val", must_contain="for epoch in range")
    replace_in_cell_sources(nb, "Y_test", "Y_val", must_contain="for epoch in range")
    replace_in_cell_sources(nb, "Y_pred_test", "Y_pred_val", must_contain="for epoch in range")
    replace_in_cell_sources(nb, "backtest_opens", "val_opens", must_contain="for epoch in range")
    replace_in_cell_sources(nb, "backtest_closes", "val_closes", must_contain="for epoch in range")
    replace_in_cell_sources(nb, "# Predict on test data", "# Predict on val slice (selection period)", must_contain="for epoch in range")
    replace_in_cell_sources(nb, "# Convert 'Y_pred_test'", "# Convert 'Y_pred_val'", must_contain="for epoch in range")

    append_artifacts_cell(nb)
    save_nb(nb, p)
    print("patched hybrid", p)


def patch_backtest():
    nb, p = load_nb("backtest/main.ipynb")
    c0 = """import os
from pathlib import Path
from datetime import datetime
import pickle
import random
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler

import torch
import torch.nn as nn
import torch.nn.functional as F


def _find_cnn_root():
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        if (p / "data" / "SOLUSDT-1h-data.csv").is_file():
            return p
        cand = p / "CNN_SOL" / "data" / "SOLUSDT-1h-data.csv"
        if cand.is_file():
            return (p / "CNN_SOL").resolve()
    return here


CNN_ROOT = _find_cnn_root()
ARTIFACTS = CNN_ROOT / "artifacts"
DATA_CSV = CNN_ROOT / "data" / "SOLUSDT-1h-data.csv"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("ensemble_models", exist_ok=True)

def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

seed = 0
set_all_seeds(seed)"""
    set_cell_source(nb, 0, c0)
    set_cell_source(nb, 1, "df = pd.read_csv(DATA_CSV)\nprint(df)")
    c5 = """import json

with open(ARTIFACTS / "split_info.json", "r", encoding="utf-8") as f:
    sp = json.load(f)
train_end = int(sp["train_end"])
val_end = int(sp["val_end"])
total_rows = int(sp.get("total_rows", len(df)))

with open(ARTIFACTS / "scalers.pkl", "rb") as f:
    train_scalers = pickle.load(f)

df_sel = df.iloc[train_end:val_end].copy()
df_hold = df.iloc[val_end:total_rows].copy()

print(f"Selection (test) slice [{train_end}:{val_end}] len={len(df_sel)}")
print(f"Holdout (backtest) slice [{val_end}:{total_rows}] len={len(df_hold)}")"""
    set_cell_source(nb, 5, c5)
    c10 = """# Sequence Length
seq_len = 48

# Transform only — scalers loaded from artifacts (no fit on post-train rows)
X_sel, Y_sel, num_features, sel_opens, sel_highs, sel_lows, sel_closes, _ = preprocess_data(
    seq_len, df_sel, train_scalers
)
X_hold, Y_hold, num_features, ho_opens, ho_highs, ho_lows, ho_closes, _ = preprocess_data(
    seq_len, df_hold, train_scalers
)
"""
    set_cell_source(nb, 10, c10)
    c11 = """m_sel = X_sel.shape[0]
m_hold = X_hold.shape[0]
print(f"m_sel: {m_sel}")
print(f"m_hold: {m_hold}")
print(f"X_sel shape: {X_sel.shape}")
print(f"X_hold shape: {X_hold.shape}")
print(f"num_features: {num_features}")"""
    set_cell_source(nb, 11, c11)
    c13 = """device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("my device: ", device)

X_sel_t = torch.from_numpy(X_sel.astype(np.float32)).to(device, dtype=torch.float32)
X_hold_t = torch.from_numpy(X_hold.astype(np.float32)).to(device, dtype=torch.float32)
"""
    set_cell_source(nb, 13, c13)

    c18 = r'''import json

manifest_path = ARTIFACTS / "manifest.json"
with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

ARCH_MAP = {"conv1d": Model_1, "lstm": Model_2, "hybrid": Model_3}

MODELS_INFOS = []
for arch_name in ["conv1d", "lstm", "hybrid"]:
    if arch_name not in manifest:
        continue
    rel_paths = manifest[arch_name]
    abs_paths = [str(CNN_ROOT / Path(*rp.split("/"))) for rp in rel_paths]
    MODELS_INFOS.append({"paths": abs_paths, "architecture": ARCH_MAP[arch_name]})

if not MODELS_INFOS:
    raise FileNotFoundError(
        "No entries in manifest.json — run training notebooks (artifact cells) first."
    )


def ensemble_raw_signals(X_t: torch.Tensor, m_len: int):
    out = []
    for model_info in MODELS_INFOS:
        architecture = model_info["architecture"]
        for model_path in model_info["paths"]:
            model = architecture().to(device)
            model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
            model.eval()
            Y_pred = torch.zeros([m_len, num_features], device=device, dtype=torch.float32)
            with torch.no_grad():
                chunks = 16
                while True:
                    try:
                        mchunk = max(1, m_len // chunks)
                        for j in range(chunks + 1):
                            start = mchunk * j
                            if start >= m_len:
                                break
                            end = min(mchunk * (j + 1), m_len)
                            Y_pred[start:end] = model(X_t[start:end])
                        break
                    except Exception as err:
                        print(f"Chunks ({chunks}) err: {err}")
                        chunks += 2
            out.append(Y_pred.detach().cpu().numpy()[:, 0].copy())
    return out


def trading_backtest(all_raw_signals, opens, highs, lows, closes, phase_label: str):
    TAKER_FEE = 0.00055
    pos_size = 1000.0
    SL_POINTS = 50.0
    TP_POINTS = 500.0
    THRESHOLD = 0.0008

    cash = 1000.0
    position = 0
    entry_price = None

    equities = []
    fee_log = []
    actions_log = []

    total_fees = 0.0
    entries = 0
    completed_trades = 0
    sl_hits = 0
    tp_hits = 0
    num_longs = 0
    num_shorts = 0
    trade_pnls = []

    def taker_fee():
        return pos_size * TAKER_FEE

    def realize_to(price):
        nonlocal cash, entry_price, position, completed_trades, total_fees, trade_pnls
        if position == 0 or entry_price is None:
            return
        pct = (price - entry_price) / entry_price
        pnl = pos_size * (pct if position == 1 else -pct)
        fee = taker_fee()
        cash += pnl - fee
        total_fees += fee
        trade_pnls.append(pnl - fee)
        entry_price = None
        position = 0
        completed_trades += 1

    def mark_to_market(close_price):
        if position == 0 or entry_price is None:
            return cash
        pct = (close_price - entry_price) / entry_price
        return cash + pos_size * (pct if position == 1 else -pct)

    n = len(opens) - 1
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

            if (position == 1 and curr_high >= tp_price) or (position == -1 and curr_low <= tp_price):
                realize_to(tp_price)
                tp_hits += 1

            elif (position == 1 and curr_low <= sl_price) or (position == -1 and curr_high >= sl_price):
                realize_to(sl_price)
                sl_hits += 1

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

    if position != 0:
        realize_to(closes[len(equities) - 1])

    win_trades = sum(1 for p in trade_pnls if p > 0)
    win_rate = win_trades / len(trade_pnls) * 100 if trade_pnls else 0
    avg_pnl = np.mean(trade_pnls) if trade_pnls else 0

    print(f"===== {phase_label} =====")
    print(f"Entries: {entries} | Completed: {completed_trades}")
    print(f"Longs: {num_longs} | Shorts: {num_shorts}")
    print(f"SL hits: {sl_hits} | TP hits: {tp_hits}")
    print(f"Win rate: {win_rate:.1f}% | Avg PnL/trade: ${avg_pnl:.2f}")
    print(f"Total fees: ${total_fees:.2f}")
    print(f"Final equity: ${equities[-1]:.2f}")


raw_sel = ensemble_raw_signals(X_sel_t, m_sel)
trading_backtest(raw_sel, sel_opens, sel_highs, sel_lows, sel_closes, "selection [train_end:val_end]")

raw_hold = ensemble_raw_signals(X_hold_t, m_hold)
trading_backtest(raw_hold, ho_opens, ho_highs, ho_lows, ho_closes, "holdout [val_end:end]")
'''
    set_cell_source(nb, 18, c18)
    save_nb(nb, p)
    print("patched backtest", p)


if __name__ == "__main__":
    patch_lstm()
    patch_conv1d()
    patch_hybrid()
    patch_backtest()
    print("done")
