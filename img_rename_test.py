#!/usr/bin/env python3
"""
Copy JPGs from INPUT_DIR to OUTPUT_DIR with names YYYYMMDDhhmmss.jpg,
1 minute apart. Order: filename sort (case-insensitive).

Edit CONFIG below only.
"""

from __future__ import annotations

import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG — set everything here
# ---------------------------------------------------------------------------

INPUT_DIR = Path("/home/kyber/projects/digital energy/dataset/raw/skippd_images_utc/test")
OUTPUT_DIR = Path("/home/kyber/projects/digital energy/dataset/skyimg")

# First image (after sort) gets this time. Use naive local time you intend for filenames.
REFERENCE_TIME = datetime(2025, 1, 1, 0, 0, 0)  # year, month, day, hour, minute, second

# True  -> 1st image = REFERENCE_TIME, 2nd = +1 min, 3rd = +2 min, ...
# False -> 1st image = REFERENCE_TIME, 2nd = -1 min, 3rd = -2 min, ...
TIME_GOES_FORWARD = True

# ---------------------------------------------------------------------------
# end CONFIG
# ---------------------------------------------------------------------------


def collect_jpgs(input_dir: Path) -> list[Path]:
    files = [
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".jpg"
    ]
    return sorted(files, key=lambda x: x.name.lower())


def main() -> int:
    input_dir = INPUT_DIR.expanduser().resolve()
    output_dir = OUTPUT_DIR.expanduser().resolve()

    if not input_dir.is_dir():
        print(f"error: not a directory: {input_dir}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    jpgs = collect_jpgs(input_dir)
    if not jpgs:
        print(f"error: no .jpg files in {input_dir}", file=sys.stderr)
        return 1

    step = timedelta(minutes=1) if TIME_GOES_FORWARD else -timedelta(minutes=1)
    ref = REFERENCE_TIME

    for i, src in enumerate(jpgs):
        t = ref + i * step
        name = t.strftime("%Y%m%d%H%M%S") + ".jpg"
        dst = output_dir / name
        if dst.exists():
            print(f"error: exists, not overwriting: {dst}", file=sys.stderr)
            return 1
        shutil.copy2(src, dst)
        print(f"{src.name} -> {name}")

    print(f"Done. {len(jpgs)} file(s) -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())