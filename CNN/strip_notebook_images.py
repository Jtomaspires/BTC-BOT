from __future__ import annotations

import argparse
import re
from pathlib import Path


def strip_image_png_lines(path: Path) -> int:
    """
    Replace any JSON line that contains an embedded image/png base64 payload
    with an empty payload, to keep notebooks small and (re)valid.
    """
    text = path.read_text(encoding="utf-8", errors="replace")

    # Replace: "image/png": "<anything until end-of-line>"
    # This works even if the JSON is already broken by a missing closing quote,
    # because it only matches until the newline.
    new_text, n = re.subn(
        r'("image/png"\s*:\s*)"[^\n]*',
        r'\1"",',
        text,
        flags=re.IGNORECASE,
    )

    if n:
        path.write_text(new_text, encoding="utf-8", newline="\n")
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook", type=Path)
    args = ap.parse_args()
    n = strip_image_png_lines(args.notebook)
    print(f"Stripped {n} embedded image/png line(s) in {args.notebook}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

