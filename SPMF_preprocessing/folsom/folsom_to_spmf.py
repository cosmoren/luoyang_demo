#!/usr/bin/env python3
"""
Folsom → SPMF preprocessing.

Each job is a standalone function. Pick **one** task in ``main()`` (see
``ACTIVE_TASK``) and set the matching paths in this file — nothing runs in batch
unless you ask for it.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal

import pandas as pd

# =============================================================================
# Main control — set exactly one task (only that branch runs).
# =============================================================================

ActiveTask = Literal["none", "flatten_jpegs", "merge_nwp_csvs", "rename_jpg_date_time"]

ACTIVE_TASK: ActiveTask = "rename_jpg_date_time"

# --- Task: flatten_jpegs ------------------------------------------------------
FOLSOM_INPUT_DIR = Path("/home/kyber/projects/digital energy/folsom_ds/original/sky image/raw")
SPMF_OUTPUT_DIR = Path("/home/kyber/projects/digital energy/folsom_ds/processed/sky")
OVERWRITE_EXISTING = False

# --- Task: merge_nwp_csvs (4 CSVs → 1 averaged CSV) --------------------------
NWP_AVG_INPUT_DIR = Path("/home/kyber/projects/digital energy/folsom_ds/original/NAM forecast/raw")
NWP_AVG_OUTPUT_DIR = Path("/home/kyber/projects/digital energy/folsom_ds/processed/NWP")
NWP_AVG_OUTPUT_FILENAME = "nwp_merged_averaged.csv"

# Folsom NAM row-2 headers (full order): reftime, valtime, dwsw, cloud_cover,
# precipitation, pressure, wind-u, wind-v, temperature, rel_humidity
NWP_KEY_COLS = ["reftime", "valtime"]
NWP_NUMERIC_COLS = [
    "dwsw",
    "cloud_cover",
    "precipitation",
    "pressure",
    "wind-u",
    "wind-v",
    "temperature",
    "rel_humidity",
]
NWP_EXPECTED_FILE_COUNT = 4
# Row 1 in the file is junk; row 2 holds the real header (pandas ``header`` is 0-based).
NWP_CSV_HEADER_ROW = 1

# If True: after averaging, ``cloud_cover`` and ``rel_humidity`` are rounded to the nearest int.
# If False: they stay float means like the other numeric columns.
NWP_ROUND_CLOUD_COVER_AND_REL_HUMIDITY = True

# --- Task: rename_jpg_date_time (``20140101_000011.jpg`` -> ``20140101000011.jpg``) -
RENAME_JPG_DATE_TIME_DIR = Path("/home/kyber/projects/digital energy/folsom_ds/processed/sky")


# =============================================================================
# Imagery: nested JPEG tree → flat folder
# =============================================================================


def _unique_dest_path(output_dir: Path, preferred_name: str) -> Path:
    """Pick a destination filename that does not already exist."""
    dest = output_dir / preferred_name
    if not dest.exists():
        return dest
    stem = Path(preferred_name).stem
    suffix = Path(preferred_name).suffix
    n = 2
    while True:
        candidate = output_dir / f"{stem}_{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def flatten_nested_jpegs_to_dir(
    input_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> int:
    """Copy every ``.jpg`` under ``input_dir`` (recursive) into a single ``output_dir``."""
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_res = output_dir.resolve()

    def _is_under_output(path: Path) -> bool:
        try:
            path.resolve().relative_to(out_res)
            return True
        except ValueError:
            return False

    count = 0
    for src in input_dir.rglob("*"):
        if not src.is_file() or src.suffix.lower() != ".jpg":
            continue
        if _is_under_output(src):
            continue

        dest = output_dir / src.name
        if dest.exists():
            if overwrite:
                print(f"[name overlap] {src.name!r} (overwrite)")
            else:
                print(f"[name overlap] {src.name!r}")
                dest = _unique_dest_path(output_dir, src.name)
                if dest.name != src.name:
                    print(f"[name overlap] -> {dest.name!r} <- {src}")

        shutil.copy2(src, dest)
        count += 1

    return count


def rename_jpg_remove_date_time_underscore(folder: Path) -> int:
    """
    In ``folder`` (non-recursive), rename each ``.jpg`` whose stem contains ``_``:
    remove the **first** underscore so ``20140101_000011.jpg`` becomes
    ``20140101000011.jpg``.

    Skips files with no underscore in the stem, or if the target name already exists.
    Returns the number of files renamed.
    """
    folder = folder.resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    n = 0
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".jpg":
            continue
        stem = path.stem
        if "_" not in stem:
            continue
        new_stem = stem.replace("_", "", 1)
        new_path = path.with_name(new_stem + path.suffix)
        if new_path.resolve() == path.resolve():
            continue
        if new_path.exists():
            print(f"[skip] {path.name!r} -> {new_path.name!r} (target already exists)")
            continue
        path.rename(new_path)
        n += 1
    return n


# =============================================================================
# NWP: merge 4 CSVs by timestamp keys, average weather columns
# =============================================================================


def merge_and_average_nwp_csvs() -> Path:
    """
    Read exactly ``NWP_EXPECTED_FILE_COUNT`` CSV files from ``NWP_AVG_INPUT_DIR``,
    using ``NWP_CSV_HEADER_ROW`` so line 1 is skipped and column names come from
    line 2. Align rows on ``reftime`` + ``valtime``, average ``NWP_NUMERIC_COLS``,
    optionally round ``cloud_cover`` / ``rel_humidity`` (see
    ``NWP_ROUND_CLOUD_COVER_AND_REL_HUMIDITY``), and write ``NWP_AVG_OUTPUT_FILENAME``
    under ``NWP_AVG_OUTPUT_DIR``.

    Returns the path to the written CSV.
    """
    in_dir = NWP_AVG_INPUT_DIR.resolve()
    out_dir = NWP_AVG_OUTPUT_DIR.resolve()

    if not in_dir.is_dir():
        raise FileNotFoundError(f"NWP input folder not found: {in_dir}")

    paths = sorted(in_dir.glob("*.csv"))
    if len(paths) != NWP_EXPECTED_FILE_COUNT:
        raise ValueError(
            f"Expected exactly {NWP_EXPECTED_FILE_COUNT} *.csv files in {in_dir}, "
            f"found {len(paths)}: {[p.name for p in paths]}"
        )

    required = NWP_KEY_COLS + NWP_NUMERIC_COLS
    frames: list[pd.DataFrame] = []

    for p in paths:
        df = pd.read_csv(p, header=NWP_CSV_HEADER_ROW)
        df.columns = df.columns.str.strip()
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{p.name}: missing required columns: {missing}")

        df = df[required].copy()
        # Normalise timestamps so the same instant matches across files
        for k in NWP_KEY_COLS:
            df[k] = pd.to_datetime(df[k], utc=True, errors="raise")

        dup = df.duplicated(subset=NWP_KEY_COLS, keep=False)
        if dup.any():
            raise ValueError(
                f"{p.name}: duplicate (reftime, valtime) rows present; "
                f"cannot average unambiguously."
            )

        for col in NWP_NUMERIC_COLS:
            df[col] = pd.to_numeric(df[col], errors="raise")

        frames.append(df.set_index(NWP_KEY_COLS))

    # Same set of keys in every file (order may differ)
    key_sets = [set(f.index) for f in frames]
    if len({frozenset(s) for s in key_sets}) != 1:
        only0 = key_sets[0] - key_sets[1]
        only1 = key_sets[1] - key_sets[0]
        raise ValueError(
            "NWP CSVs are not aligned: (reftime, valtime) sets differ between files. "
            f"Example keys only in file 0 vs 1: {list(only0)[:5]} ...; "
            f"only in file 1 vs 0: {list(only1)[:5]} ..."
        )

    # Same row count per file already implied by identical key sets + no dupes
    idx = frames[0].index.sort_values()
    aligned = [f.reindex(idx) for f in frames]
    for i, f in enumerate(aligned):
        if f.isna().any().any():
            raise ValueError(
                f"After aligning on sorted keys, file {paths[i].name!r} has missing rows "
                f"(unexpected hole in index)."
            )

    # Mean across the four files, column-wise
    summed = aligned[0][NWP_NUMERIC_COLS].copy()
    for f in aligned[1:]:
        summed = summed + f[NWP_NUMERIC_COLS]
    averaged = summed / float(len(aligned))

    if NWP_ROUND_CLOUD_COVER_AND_REL_HUMIDITY:
        for col in ("cloud_cover", "rel_humidity"):
            averaged[col] = averaged[col].round(0).astype("int64")

    out_df = averaged.reset_index()
    # Write timestamps without timezone offset (no trailing "+00:00" in the CSV).
    for k in NWP_KEY_COLS:
        out_df[k] = pd.to_datetime(out_df[k], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / NWP_AVG_OUTPUT_FILENAME
    out_df.to_csv(out_path, index=False)
    return out_path


# =============================================================================
# Entry (optional name kept for callers who import it)
# =============================================================================


def folsom_to_spmf() -> None:
    """Backward-compatible alias: run the task selected by ``ACTIVE_TASK``."""
    main()


def main() -> None:
    if ACTIVE_TASK == "none":
        print("No task selected: set ACTIVE_TASK in folsom_to_spmf.py.")
        return

    if ACTIVE_TASK == "flatten_jpegs":
        n = flatten_nested_jpegs_to_dir(
            FOLSOM_INPUT_DIR,
            SPMF_OUTPUT_DIR,
            overwrite=OVERWRITE_EXISTING,
        )
        print(f"flatten_jpegs: copied {n} .jpg -> {SPMF_OUTPUT_DIR.resolve()}")
        return

    if ACTIVE_TASK == "merge_nwp_csvs":
        out = merge_and_average_nwp_csvs()
        print(f"merge_nwp_csvs: wrote {out}")
        return

    if ACTIVE_TASK == "rename_jpg_date_time":
        n = rename_jpg_remove_date_time_underscore(RENAME_JPG_DATE_TIME_DIR)
        print(f"rename_jpg_date_time: renamed {n} file(s) in {RENAME_JPG_DATE_TIME_DIR.resolve()}")
        return

    raise ValueError(f"Unknown ACTIVE_TASK: {ACTIVE_TASK!r}")


if __name__ == "__main__":
    main()
