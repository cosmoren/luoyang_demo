#!/usr/bin/env python3
"""
Convert raw SKIPPD PV CSV files into SPMF-style PV CSVs.

Default input:
    ~/datasets/skippd_raw/{2017,2018,2019}/{year}_pv_raw.csv

Default output:
    ~/datasets/skippd_SPMF/pv/skippd_raw_2017_2019_merged.csv
    ~/datasets/skippd_SPMF/pv/NE_skippd_raw_ori.csv
    ~/datasets/skippd_SPMF/pv/NE_skippd_raw.csv

The raw Date column is interpreted as America/Los_Angeles local civil time
(including DST). SPMF collectTime and utcTime are written as UTC wall-clock
strings without timezone suffixes, matching the convention used by the
existing SKIPPD SPMF preprocessing scripts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_ROOT = Path("~/datasets/skippd_raw").expanduser()
DEFAULT_OUTPUT_DIR = Path("~/datasets/skippd_SPMF/pv").expanduser()
DEFAULT_YEARS = (2017, 2018, 2019)

STATION_DEV = "NE=skippd_raw"
LATITUDE_DEVICE = 37.4275
LONGITUDE_DEVICE = -122.1697
LOCAL_TZ = "America/Los_Angeles"

MERGED_OUTPUT = "skippd_raw_2017_2019_merged.csv"
OUTPUT_FILE_ORI = "NE_skippd_raw_ori.csv"
OUTPUT_FILE_5MIN = "NE_skippd_raw.csv"

SPMF_OUTPUT_COLS = [
    "stationCode",
    "latitude_device",
    "longitude_device",
    "localTime",
    "collectTime",
    "utcTime",
    "devDn",
    "inverter_state",
    "active_power",
]

MERGED_OUTPUT_COLS = [
    "source_file",
    "localTime",
    "active_power",
    "utcTime",
]


def _format_dt(values: pd.Series | pd.DatetimeIndex) -> pd.Series:
    return pd.Series(values).dt.strftime("%Y-%m-%d %H:%M:%S")


def _read_raw_csv(
    path: Path,
    *,
    date_col: str,
    power_col: str | None,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing raw CSV: {path}")

    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"{path}: missing date column {date_col!r}; columns={list(df.columns)!r}")

    actual_power_col = power_col
    if actual_power_col is None:
        candidates = [c for c in df.columns if c != date_col]
        if len(candidates) != 1:
            raise ValueError(
                f"{path}: could not infer power column; pass --power-col. "
                f"columns={list(df.columns)!r}"
            )
        actual_power_col = candidates[0]
    if actual_power_col not in df.columns:
        raise ValueError(
            f"{path}: missing power column {actual_power_col!r}; columns={list(df.columns)!r}"
        )

    local_time = pd.to_datetime(df[date_col], errors="coerce")
    if local_time.isna().any():
        bad_count = int(local_time.isna().sum())
        raise ValueError(f"{path}: found {bad_count} invalid local timestamps in {date_col!r}")

    active_power = pd.to_numeric(df[actual_power_col], errors="coerce")
    if active_power.isna().any():
        bad_count = int(active_power.isna().sum())
        raise ValueError(f"{path}: found {bad_count} invalid power values in {actual_power_col!r}")

    # Raw timestamps are sorted local civil time. During the fall-back hour,
    # each minute appears twice: first in daylight time, then in standard time.
    duplicated_local = local_time.duplicated(keep=False)
    second_duplicate = local_time.duplicated(keep="first")
    ambiguous = (duplicated_local & ~second_duplicate).to_numpy()
    local_aware = local_time.dt.tz_localize(
        LOCAL_TZ,
        ambiguous=ambiguous,
        nonexistent="shift_forward",
    )
    utc_aware = local_aware.dt.tz_convert("UTC")
    utc_naive = utc_aware.dt.tz_localize(None)

    out = pd.DataFrame(
        {
            "source_file": path.name,
            "local_dt": local_time,
            "utc_dt": utc_naive,
            "active_power": active_power.astype(float),
        }
    )
    return out


def load_and_merge_raw(
    input_root: Path,
    *,
    years: tuple[int, ...],
    date_col: str,
    power_col: str | None,
    clip_negative_power: bool,
) -> pd.DataFrame:
    frames = []
    for year in years:
        path = input_root / str(year) / f"{year}_pv_raw.csv"
        frames.append(_read_raw_csv(path, date_col=date_col, power_col=power_col))

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values("utc_dt", kind="stable").reset_index(drop=True)
    if merged["utc_dt"].duplicated().any():
        dup_count = int(merged["utc_dt"].duplicated().sum())
        raise ValueError(f"Found {dup_count} duplicate UTC timestamps after timezone conversion.")

    if clip_negative_power:
        merged["active_power"] = merged["active_power"].clip(lower=0.0)

    return merged


def _add_spmf_metadata(df: pd.DataFrame, *, station_dev: str) -> pd.DataFrame:
    out = df.copy()
    out["stationCode"] = station_dev
    out["latitude_device"] = LATITUDE_DEVICE
    out["longitude_device"] = LONGITUDE_DEVICE
    out["devDn"] = station_dev
    return out


def make_spmf_table(
    merged: pd.DataFrame,
    *,
    station_dev: str,
    fill_missing: bool,
) -> pd.DataFrame:
    if merged.empty:
        raise ValueError("No raw rows were loaded.")

    base = merged[["utc_dt", "active_power"]].copy()
    base["inverter_state"] = 512
    base = base.sort_values("utc_dt", kind="stable").set_index("utc_dt")

    if fill_missing:
        full_index = pd.date_range(base.index.min(), base.index.max(), freq="1min")
        base = base.reindex(full_index)
        missing = base["active_power"].isna()
        base.loc[missing, "active_power"] = 0.0
        base.loc[missing, "inverter_state"] = 0

    utc_naive = pd.DatetimeIndex(base.index)
    local_naive = utc_naive.tz_localize("UTC").tz_convert(LOCAL_TZ).tz_localize(None)

    out = pd.DataFrame(
        {
            "localTime": _format_dt(local_naive).to_numpy(),
            "collectTime": _format_dt(utc_naive).to_numpy(),
            "utcTime": _format_dt(utc_naive).to_numpy(),
            "inverter_state": base["inverter_state"].astype(int).to_numpy(),
            "active_power": base["active_power"].astype(float).to_numpy(),
        }
    )
    out = _add_spmf_metadata(out, station_dev=station_dev)
    return out[SPMF_OUTPUT_COLS]


def make_merged_output(merged: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "source_file": merged["source_file"],
            "localTime": _format_dt(merged["local_dt"]).to_numpy(),
            "active_power": merged["active_power"].astype(float).to_numpy(),
            "utcTime": _format_dt(merged["utc_dt"]).to_numpy(),
        }
    )
    return out[MERGED_OUTPUT_COLS]


def write_outputs(
    merged: pd.DataFrame,
    spmf_1min: pd.DataFrame,
    output_dir: Path,
    *,
    freq_minutes: int,
    dry_run: bool,
) -> tuple[Path, Path, Path]:
    merged_path = output_dir / MERGED_OUTPUT
    ori_path = output_dir / OUTPUT_FILE_ORI
    freq_path = output_dir / OUTPUT_FILE_5MIN

    collect_time = pd.to_datetime(spmf_1min["collectTime"], errors="raise")
    spmf_freq = spmf_1min[
        (collect_time.dt.second == 0) & (collect_time.dt.minute % freq_minutes == 0)
    ].copy()

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        make_merged_output(merged).to_csv(merged_path, index=False)
        spmf_1min.to_csv(ori_path, index=False)
        spmf_freq.to_csv(freq_path, index=False)

    print(f"merged rows: {len(merged):,}")
    print(f"spmf 1min rows: {len(spmf_1min):,}")
    print(f"spmf {freq_minutes}min rows: {len(spmf_freq):,}")
    print(f"time range UTC: {spmf_1min['utcTime'].iloc[0]} -> {spmf_1min['utcTime'].iloc[-1]}")
    if dry_run:
        print(f"dry run: would write {merged_path}")
        print(f"dry run: would write {ori_path}")
        print(f"dry run: would write {freq_path}")
    else:
        print(f"wrote: {merged_path}")
        print(f"wrote: {ori_path}")
        print(f"wrote: {freq_path}")

    return merged_path, ori_path, freq_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ~/datasets/skippd_raw 2017-2019 PV CSVs to SPMF CSV format."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--years", type=int, nargs="+", default=list(DEFAULT_YEARS))
    parser.add_argument("--date-col", default="Date")
    parser.add_argument(
        "--power-col",
        default=None,
        help="Power column name. Defaults to the single non-Date column.",
    )
    parser.add_argument("--station-dev", default=STATION_DEV)
    parser.add_argument("--freq-minutes", type=int, default=5)
    parser.add_argument(
        "--no-fill-missing",
        action="store_true",
        help="Do not fill missing UTC minutes with active_power=0.",
    )
    parser.add_argument(
        "--clip-negative-power",
        action="store_true",
        help="Clip negative active_power values to 0 before writing outputs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.freq_minutes <= 0:
        raise ValueError("--freq-minutes must be positive.")

    input_root = args.input_root.expanduser()
    output_dir = args.output_dir.expanduser()
    years = tuple(args.years)

    merged = load_and_merge_raw(
        input_root,
        years=years,
        date_col=args.date_col,
        power_col=args.power_col,
        clip_negative_power=args.clip_negative_power,
    )
    spmf_1min = make_spmf_table(
        merged,
        station_dev=args.station_dev,
        fill_missing=not args.no_fill_missing,
    )
    write_outputs(
        merged,
        spmf_1min,
        output_dir,
        freq_minutes=args.freq_minutes,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
