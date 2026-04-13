"""
Helpers for per-station PV CSVs (5-minute rows, ``collectTime`` sorted).

``load_csv`` / ``list_csv_files`` are used by ``training.train``; inverter column name and
valid-state code are shared constants for masks and filtering.
"""

from pathlib import Path

import pandas as pd

INVERTER_STATE_COL = "inverter_state"
VALID_STATE = 512


def load_csv(csv_path: Path | str) -> pd.DataFrame:
    """Load a single device CSV; ensure collectTime is parsed and sorted."""
    df = pd.read_csv(csv_path)
    if "collectTime" in df.columns:
        df["collectTime"] = pd.to_datetime(df["collectTime"])
        df = df.sort_values("collectTime").reset_index(drop=True)
    return df


def list_csv_files(
    start_idx: int = 0,
    end_idx: int | None = None,
    data_dir: Path | str | None = None,
) -> list[Path]:
    """
    Sorted ``*.csv`` paths under ``data_dir``.

    - Default ``end_idx=None``: return all files from ``start_idx`` through the last file (any count).
    - If ``end_idx`` is an int, it is the **inclusive** last index in the sorted list (``start_idx`` … ``end_idx``).
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")
    all_csvs: list[Path] = sorted(data_dir.glob("*.csv"))
    n = len(all_csvs)
    if n == 0:
        return []
    if start_idx < 0:
        raise ValueError("start_idx must be >= 0")
    if start_idx >= n:
        raise ValueError(f"start_idx ({start_idx}) out of range (n={n})")
    if end_idx is None:
        return all_csvs[start_idx:]
    if end_idx < start_idx:
        raise ValueError("require start_idx <= end_idx")
    if end_idx >= n:
        raise ValueError(
            f"end_idx ({end_idx}) out of range (n={n}); valid inclusive indices are 0..{n - 1}"
        )
    return all_csvs[start_idx : end_idx + 1]
