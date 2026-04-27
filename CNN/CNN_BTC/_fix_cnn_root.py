import json
from pathlib import Path

OLD = """def _find_cnn_root():
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        if (p / "data" / "BTCUSDT-1h-data.csv").is_file():
            return p
    return here"""

NEW = """def _find_cnn_root():
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        if (p / "data" / "BTCUSDT-1h-data.csv").is_file():
            return p
        cand = p / "CNN" / "data" / "BTCUSDT-1h-data.csv"
        if cand.is_file():
            return (p / "CNN").resolve()
    return here"""

root = Path(__file__).resolve().parent
for p in root.rglob("main.ipynb"):
    nb = json.loads(p.read_text(encoding="utf-8"))
    changed = False
    for c in nb.get("cells", []):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if OLD not in src:
            continue
        new_src = src.replace(OLD, NEW)
        c["source"] = [ln + "\n" for ln in new_src.split("\n")[:-1]] + [new_src.split("\n")[-1] + "\n"]
        changed = True
    if changed:
        p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print("updated", p)
