from __future__ import annotations

import argparse
from pathlib import Path


def clean_notebook(path: Path) -> None:
    """
    Make a notebook smaller / tool-friendly:
    - clears all outputs
    - sets execution_count to None
    """
    import nbformat  # type: ignore

    nb = nbformat.read(path, as_version=4)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    nbformat.write(nb, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook", type=Path)
    args = ap.parse_args()
    clean_notebook(args.notebook)
    print(f"Cleaned outputs: {args.notebook}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

