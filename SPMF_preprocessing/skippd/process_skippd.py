#!/usr/bin/env python3
"""
Generate SKIPPD train/test metadata and build merged outputs.

Outputs written in current directory:
- train_metadata.csv
- test_metadata.csv
- NE_skippd_ori.csv  (1-minute completed table)
- NE_skippd.csv      (5-minute subsample)
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from datasets import load_dataset


INPUT_SPLITS = ("train", "test")
STATION_DEV = "NE=skippd"
LATITUDE_DEVICE = 37.4275
LONGITUDE_DEVICE = -122.1697
OUTPUT_FILE_ORI = "NE_skippd_ori.csv"
OUTPUT_FILE_5MIN = "NE_skippd.csv"

OUTPUT_COLS = [
    "stationCode",
    "latitude_device",
    "longitude_device",
    "collectTime",
    "devDn",
    "inverter_state",
    "active_power",
]

# Dataset wall times are Pacific (PST/PDT); naive values are interpreted as America/Los_Angeles.
PACIFIC = ZoneInfo("America/Los_Angeles")


def _parse_to_utc(ts: object) -> datetime:
    if isinstance(ts, datetime):
        dt = ts
    else:
        s = str(ts).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=PACIFIC)
    return dt.astimezone(timezone.utc)


def format_timestamp_filename(ts: object) -> str:
    """UTC wall clock YYYYMMDDHHMMSS for filenames."""
    return _parse_to_utc(ts).replace(tzinfo=None).strftime("%Y%m%d%H%M%S")


def format_timestamp_csv(ts: object) -> str:
    """UTC wall clock for CSV: YYYY-MM-DD HH:MM:SS (no T/Z; values are still UTC)."""
    dt = _parse_to_utc(ts).replace(tzinfo=None)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def generate_metadata_csvs(base_dir: Path) -> tuple[Path, Path]:
    dataset = load_dataset("solarbench/SKIPPD", cache_dir="/data")
    output_paths: dict[str, Path] = {}

    for split in INPUT_SPLITS:
        output_path = base_dir / f"{split}_metadata.csv"
        with output_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "timestamp_utc", "pv"])

            for sample in dataset[split]:
                ts_raw = sample["time"]
                ts_fmt = format_timestamp_filename(ts_raw)
                ts_csv = format_timestamp_csv(ts_raw)
                writer.writerow([f"{ts_fmt}.png", ts_csv, sample["pv"]])

        output_paths[split] = output_path
        print(f"{split} metadata saved: {output_path}")

    return output_paths["train"], output_paths["test"]


def merge_and_fill(base_dir: Path) -> tuple[Path, Path]:
    frames = []
    for split in INPUT_SPLITS:
        file_path = base_dir / f"{split}_metadata.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing input file: {file_path}")
        frames.append(pd.read_csv(file_path))

    df = pd.concat(frames, ignore_index=True)

    required_cols = {"filename", "timestamp_utc", "pv"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input CSVs: {sorted(missing)}")

    df = df.rename(columns={"timestamp_utc": "collectTime", "pv": "active_power"})

    df["collectTime"] = pd.to_datetime(df["collectTime"], errors="coerce")
    if df["collectTime"].isna().any():
        bad_count = int(df["collectTime"].isna().sum())
        raise ValueError(f"Found {bad_count} invalid collectTime values.")

    df["active_power"] = pd.to_numeric(df["active_power"], errors="coerce")
    df["inverter_state"] = 512
    df["stationCode"] = STATION_DEV
    df["devDn"] = STATION_DEV
    df["latitude_device"] = LATITUDE_DEVICE
    df["longitude_device"] = LONGITUDE_DEVICE
    df = df[OUTPUT_COLS]

    min_minute = df["collectTime"].min().floor("min")
    max_minute = df["collectTime"].max().floor("min")
    full_minutes = pd.date_range(min_minute, max_minute, freq="1min")
    existing_timestamps = set(df["collectTime"])
    missing_minutes = [ts for ts in full_minutes if ts not in existing_timestamps]

    fill_df = pd.DataFrame({"collectTime": missing_minutes})
    if not fill_df.empty:
        fill_df["active_power"] = 0
        fill_df["inverter_state"] = 0
        fill_df["stationCode"] = STATION_DEV
        fill_df["devDn"] = STATION_DEV
        fill_df["latitude_device"] = LATITUDE_DEVICE
        fill_df["longitude_device"] = LONGITUDE_DEVICE
        out = pd.concat([df, fill_df], ignore_index=True)
    else:
        out = df.copy()

    out = out[OUTPUT_COLS].sort_values("collectTime")

    output_path_ori = base_dir / OUTPUT_FILE_ORI
    out.to_csv(output_path_ori, index=False)

    out_5min = out[
        (out["collectTime"].dt.second == 0) & (out["collectTime"].dt.minute % 5 == 0)
    ].sort_values("collectTime")
    output_path_5min = base_dir / OUTPUT_FILE_5MIN
    out_5min.to_csv(output_path_5min, index=False)

    return output_path_ori, output_path_5min


def main() -> None:
    base_dir = Path.cwd()
    train_path, test_path = generate_metadata_csvs(base_dir)
    print(f"Wrote: {train_path}")
    print(f"Wrote: {test_path}")

    output_path_ori, output_path_5min = merge_and_fill(base_dir)
    print(f"Wrote: {output_path_ori}")
    print(f"Wrote: {output_path_5min}")


if __name__ == "__main__":
    main()
