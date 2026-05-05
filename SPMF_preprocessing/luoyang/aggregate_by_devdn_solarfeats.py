#!/usr/bin/env python3
"""
Aggregate 365 daily CSV files by devDn:
- One output CSV per unique devDn (e.g. 639 files if 639 unique devDn).
- Drop columns B, C, D (plantName, plantAddress, devName).
- Fill missing 5-min slots with 0 for data columns.
- Time step: every 5 minutes; full year 2025.
- Input collectTime is interpreted as UTC+8 (China); output collectTime is UTC.
- Each output row also carries solar-geometry / time-of-year fields computed
  from collectTime (UTC) + device lat/lon via pvlib:
  solar_zenith, solar_azimuth, local_solar_time, day_of_year, hour_of_day.
  day_of_year and hour_of_day are taken from local solar time.
- After per-devDn files, writes NE_total.csv: row-wise aggregate (see aggregate_ne_total).

Each daily file is read once; staging CSVs on disk then one devDn at a time in RAM
for the final grid. Staging is deleted when done.

Dependencies: pandas, pvlib (used for solar position / equation of time).

CLI:
  python aggregate_by_devdn.py [--input DIR] [--output DIR]
  Defaults: input = script directory, output = <input>/aggregated_by_devDn
"""

import argparse
import csv
import os
import re
import shutil
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Set

import pandas as pd
import pvlib

# Paths already reported as empty (no header row); avoid duplicate stderr lines per devDn pass.
_logged_empty_csv_paths: Set[Path] = set()

NE_TOTAL_FILENAME = "NE_total.csv"
STAGING_DIRNAME = "_staging_by_devdn"

# Source tables use China wall time (UTC+8, no DST).
CN_UTC_OFFSET = timedelta(hours=8)

# Output 5-min grid bounds, expressed in China local wall time (naive).
# Single source of truth: change here to retarget another year / span.
GRID_STEP = timedelta(minutes=5)
GRID_START_CN_LOCAL = datetime(2025, 1, 1, 0, 0, 0)
GRID_END_CN_LOCAL = datetime(2025, 12, 31, 23, 55, 0)

HEADER = [
    "stationCode", "latitude_device", "longitude_device", "capacity",
    "collectTime", "devDn", "inverter_state", "efficiency", "temperature",
    "power_factor", "elec_freq", "active_power", "reactive_power",
    "day_cap", "mppt_power", "total_cap", "mppt_total_cap",
]
NUMERIC_COL_NAMES = [
    "latitude_device", "longitude_device", "capacity", "inverter_state",
    "efficiency", "temperature", "power_factor", "elec_freq", "active_power",
    "reactive_power", "day_cap", "mppt_power", "total_cap", "mppt_total_cap",
]

# Extra time/solar-geometry columns appended only to the final per-devDn CSVs
# and NE_total.csv (staging files keep the original HEADER).
SOLAR_EXTRA_COLS = [
    "solar_zenith", "solar_azimuth", "local_solar_time",
    "day_of_year", "hour_of_day",
]
OUTPUT_HEADER = HEADER + SOLAR_EXTRA_COLS


def format_ts(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Solar geometry via pvlib. We compute zenith/azimuth and equation-of-time
# vectorized over the full 2025 5-min UTC grid for each (lat, lon) pair, then
# look up per-row values by index. Apparent local solar time is derived as
#     LST = UTC + 4 * lon (min) + EoT (min).
#
# Azimuth convention follows pvlib: 0 = N, 90 = E, 180 = S, 270 = W.
# Zenith is pvlib's `apparent_zenith` (refraction-corrected).
# ---------------------------------------------------------------------------
def utc_index_for_grid() -> pd.DatetimeIndex:
    """tz-aware UTC DatetimeIndex aligned with full_5min_index() / ordered_utc_collect_time_keys()."""
    return pd.date_range(
        start=cn_local_naive_to_utc_naive(GRID_START_CN_LOCAL),
        end=cn_local_naive_to_utc_naive(GRID_END_CN_LOCAL),
        freq=GRID_STEP,
        tz="UTC",
    )


_solar_cache: dict = {}


def compute_solar_arrays(utc_index: pd.DatetimeIndex, lat: float, lon: float) -> dict:
    """Vectorized per-(lat, lon) solar arrays aligned with utc_index. Cached by rounded coords."""
    key = (round(float(lat), 5), round(float(lon), 5))
    cached = _solar_cache.get(key)
    if cached is not None:
        return cached

    sp = pvlib.solarposition.get_solarposition(utc_index, lat, lon)
    eot_min = sp["equation_of_time"].to_numpy()
    zenith = sp["apparent_zenith"].to_numpy()
    azimuth = sp["azimuth"].to_numpy()

    offset = pd.to_timedelta(4.0 * float(lon) + eot_min, unit="min")
    utc_naive = utc_index.tz_convert("UTC").tz_localize(None)
    lst_index = utc_naive + offset

    arrays = {
        "zenith": zenith,
        "azimuth": azimuth,
        "lst_strs": lst_index.strftime("%Y-%m-%d %H:%M:%S").tolist(),
        "day_of_year": lst_index.dayofyear.to_numpy(),
        "hour_of_day": (
            lst_index.hour
            + lst_index.minute / 60.0
            + lst_index.second / 3600.0
        ).to_numpy(),
    }
    _solar_cache[key] = arrays
    return arrays


def solar_fields_at(arrays: dict, idx: int) -> dict:
    """Format the 5 solar/temporal output fields for the row at array index `idx`."""
    return {
        "solar_zenith": f"{float(arrays['zenith'][idx]):.4f}",
        "solar_azimuth": f"{float(arrays['azimuth'][idx]):.4f}",
        "local_solar_time": arrays["lst_strs"][idx],
        "day_of_year": str(int(arrays["day_of_year"][idx])),
        "hour_of_day": f"{float(arrays['hour_of_day'][idx]):.4f}",
    }


def cn_local_naive_to_utc_naive(dt_cn_naive):
    """Interpret naive datetime as China local (UTC+8) and return naive UTC datetime."""
    return dt_cn_naive - CN_UTC_OFFSET


def collect_time_cn_str_to_utc_str(ct_cn: str) -> str:
    """Parse collectTime string (China local) and return the same instant as UTC string."""
    dt_cn = datetime.strptime(ct_cn.strip(), "%Y-%m-%d %H:%M:%S")
    return format_ts(cn_local_naive_to_utc_naive(dt_cn))


def grid_cn_time_to_utc_str(t_cn_naive):
    """5-min grid tick (naive = China local) -> UTC collectTime string."""
    return format_ts(cn_local_naive_to_utc_naive(t_cn_naive))


def full_5min_index():
    """Yield all 5-min ticks (naive China local wall clock) within the grid bounds."""
    t = GRID_START_CN_LOCAL
    while t <= GRID_END_CN_LOCAL:
        yield t
        t += GRID_STEP


def ordered_utc_collect_time_keys():
    return [grid_cn_time_to_utc_str(t) for t in full_5min_index()]


def get_sorted_csv_paths(input_dir: Path):
    """Return paths to 组串式逆变器-YYYY-MM-DD.csv sorted by date."""
    pattern = re.compile(r"组串式逆变器-(\d{4}-\d{2}-\d{2})\.csv")
    files = []
    for f in input_dir.glob("组串式逆变器-*.csv"):
        m = pattern.match(f.name)
        if m:
            files.append((m.group(1), f))
    files.sort(key=lambda x: x[0])
    return [f for _, f in files]


def safe_devdn_filename(devdn):
    safe = str(devdn).replace("=", "_").replace("/", "_").replace("\\", "_")
    return f"{safe}.csv"


def normalize_header(header):
    """Strip BOM and whitespace so 'stationCode' etc. match."""
    return [h.strip().lstrip("\ufeff") for h in header]


def read_csv_header_row(reader):
    """Return normalized header row, or None if the file has no rows."""
    try:
        return normalize_header(next(reader))
    except StopIteration:
        return None


def log_skip_empty_csv(path: Path) -> None:
    """Warn once per file (stderr) when a CSV has no header row."""
    key = path.resolve()
    if key in _logged_empty_csv_paths:
        return
    _logged_empty_csv_paths.add(key)
    print(f"Skipping empty CSV (no header row): {path}", file=sys.stderr)


def build_keep_map(header):
    header = normalize_header(header)
    idx_by_name = {h: i for i, h in enumerate(header)}
    return [(name, idx_by_name[name]) for name in HEADER if name in idx_by_name]


def read_row_without_bcd(keep_map, row):
    out = {name: "" for name in HEADER}
    for name, i in keep_map:
        val = (row[i].strip() if i < len(row) and row[i] else "") or ""
        # Fill missing values with 0 for numeric columns
        if name in NUMERIC_COL_NAMES and not val:
            val = "0"
        out[name] = val
    return out


def fanout_daily_rows_to_staging(csv_paths, staging_dir: Path) -> Set[str]:
    """
    Single pass: read each daily file once; append each row to a per-devDn staging CSV (UTC collectTime).
    Returns the set of devDn values that received at least one data row.
    """
    keep_map_cache = {}
    staging_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[str, csv.DictWriter] = {}
    handles: dict[str, object] = {}
    seen: Set[str] = set()

    def get_writer(devdn: str) -> csv.DictWriter:
        if devdn not in writers:
            spath = staging_dir / safe_devdn_filename(devdn)
            fh = open(spath, "w", encoding="utf-8", newline="")
            handles[devdn] = fh
            w = csv.DictWriter(fh, fieldnames=HEADER)
            w.writeheader()
            writers[devdn] = w
        return writers[devdn]

    try:
        for path in csv_paths:
            with open(path, "r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                header = read_csv_header_row(reader)
                if header is None:
                    log_skip_empty_csv(path)
                    continue
                keep_map = keep_map_cache.get(path)
                if keep_map is None:
                    keep_map = build_keep_map(header)
                    keep_map_cache[path] = keep_map
                devdn_idx = next((i for i, h in enumerate(header) if h == "devDn"), -1)
                if devdn_idx < 0:
                    continue
                for row in reader:
                    if len(row) <= devdn_idx:
                        continue
                    devdn = row[devdn_idx].strip()
                    if not devdn:
                        continue
                    kept = read_row_without_bcd(keep_map, row)
                    ct = kept.get("collectTime", "").strip()
                    if not ct:
                        continue
                    utc_key = collect_time_cn_str_to_utc_str(ct)
                    kept["collectTime"] = utc_key
                    get_writer(devdn).writerow(kept)
                    seen.add(devdn)
    finally:
        for fh in handles.values():
            fh.close()

    return seen


def load_staging_to_time_to_row(staging_path: Path):
    """Read one staging CSV into time_to_row (UTC key -> row) and capture first data row for fill metadata."""
    time_to_row = {}
    first_row = None
    if not staging_path.is_file():
        return time_to_row, {k: "" for k in HEADER}
    with open(staging_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ct = row.get("collectTime", "").strip()
            if not ct:
                continue
            time_to_row[ct] = row
            if first_row is None:
                first_row = row
    return time_to_row, first_row or {k: "" for k in HEADER}


def parse_float(s):
    if s is None or s == "":
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def fmt_number(x, as_int=False):
    if as_int:
        return str(int(round(x)))
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    s = f"{x:.10f}".rstrip("0").rstrip(".")
    return s if s else "0"


def zero_like_row():
    return {k: "0" if k in NUMERIC_COL_NAMES else "" for k in HEADER}


def ne_total_partial_blank():
    """Running aggregate for one UTC collectTime (one row added per station file)."""
    return {
        "n": 0,
        "lat_sum": 0.0,
        "lat_n": 0,
        "lon_sum": 0.0,
        "lon_n": 0,
        "cap_sum": 0.0,
        "has_512": False,
        "eff_sum": 0.0,
        "temp_sum": 0.0,
        "pf_sum": 0.0,
        "efq_sum": 0.0,
        "react_sum": 0.0,
        "ap_sum": 0.0,
        "day_sum": 0.0,
        "mppt_sum": 0.0,
        "totc_sum": 0.0,
        "mppt_tot_sum": 0.0,
    }


def ne_total_partial_add(partials: dict, utc_key: str, row: dict) -> None:
    """Incorporate one station row into the running totals for utc_key."""
    p = partials.setdefault(utc_key, ne_total_partial_blank())
    p["n"] += 1
    lat = parse_float(row["latitude_device"])
    if lat != -1.0:
        p["lat_sum"] += lat
        p["lat_n"] += 1
    lon = parse_float(row["longitude_device"])
    if lon != -1.0:
        p["lon_sum"] += lon
        p["lon_n"] += 1
    p["cap_sum"] += parse_float(row["capacity"])
    if int(round(parse_float(row["inverter_state"]))) == 512:
        p["has_512"] = True
    p["eff_sum"] += parse_float(row["efficiency"])
    p["temp_sum"] += parse_float(row["temperature"])
    p["pf_sum"] += parse_float(row["power_factor"])
    p["efq_sum"] += parse_float(row["elec_freq"])
    p["react_sum"] += parse_float(row["reactive_power"])
    p["ap_sum"] += parse_float(row["active_power"])
    p["day_sum"] += parse_float(row["day_cap"])
    p["mppt_sum"] += parse_float(row["mppt_power"])
    p["totc_sum"] += parse_float(row["total_cap"])
    p["mppt_tot_sum"] += parse_float(row["mppt_total_cap"])


def ne_total_partial_finalize(p: dict, utc_key: str) -> dict:
    """Turn running totals into one NE=total row (same rules as former aggregate_ne_total_row)."""
    n = p["n"]
    if n == 0:
        out = zero_like_row()
        out["stationCode"] = "NE=total"
        out["devDn"] = "NE=total"
        out["collectTime"] = utc_key
        return out
    inv = 512 if p["has_512"] else 0
    return {
        "stationCode": "NE=total",
        "latitude_device": fmt_number(
            (p["lat_sum"] / p["lat_n"]) if p["lat_n"] else 0.0
        ),
        "longitude_device": fmt_number(
            (p["lon_sum"] / p["lon_n"]) if p["lon_n"] else 0.0
        ),
        "capacity": fmt_number(p["cap_sum"] / n),
        "collectTime": utc_key,
        "devDn": "NE=total",
        "inverter_state": str(inv),
        "efficiency": fmt_number(p["eff_sum"] / n),
        "temperature": fmt_number(p["temp_sum"] / n),
        "power_factor": fmt_number(p["pf_sum"] / n),
        "elec_freq": fmt_number(p["efq_sum"] / n),
        "active_power": fmt_number(p["ap_sum"]),
        "reactive_power": fmt_number(p["react_sum"] / n),
        "day_cap": fmt_number(p["day_sum"]),
        "mppt_power": fmt_number(p["mppt_sum"] / n),
        "total_cap": fmt_number(p["totc_sum"] / n),
        "mppt_total_cap": fmt_number(p["mppt_tot_sum"] / n),
    }


def write_ne_total(output_dir: Path):
    """Combine all per-devDn CSVs (excluding NE_total.csv) into NE_total.csv.

    Reads one device file at a time; peak RAM is O(number of UTC timesteps), not O(stations × timesteps).
    """
    dev_paths = sorted(
        p for p in output_dir.glob("*.csv") if p.name != NE_TOTAL_FILENAME
    )
    if not dev_paths:
        print("No per-devDn CSVs found; skip NE_total.csv")
        return

    n_files = len(dev_paths)
    print(
        f"Building {NE_TOTAL_FILENAME} from {n_files} device files "
        f"(streaming, one file at a time)..."
    )
    partials: dict = {}

    for path in dev_paths:
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ct = row.get("collectTime", "").strip()
                if not ct:
                    continue
                ne_total_partial_add(partials, ct, row)

    # Representative (lat, lon) for solar geometry: weighted mean of valid
    # per-timestep contributions. lat/lon are essentially constant across all
    # devices in one plant, so a single (lat, lon) is fine for the whole file.
    lat_sum_total = sum(p["lat_sum"] for p in partials.values())
    lat_n_total = sum(p["lat_n"] for p in partials.values())
    lon_sum_total = sum(p["lon_sum"] for p in partials.values())
    lon_n_total = sum(p["lon_n"] for p in partials.values())
    ref_lat = (lat_sum_total / lat_n_total) if lat_n_total else 0.0
    ref_lon = (lon_sum_total / lon_n_total) if lon_n_total else 0.0
    sun = compute_solar_arrays(utc_index_for_grid(), ref_lat, ref_lon)

    keys = ordered_utc_collect_time_keys()
    out_path = output_dir / NE_TOTAL_FILENAME
    with open(out_path, "w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=OUTPUT_HEADER)
        writer.writeheader()
        for i, utc_key in enumerate(keys):
            p = partials.get(utc_key)
            n_have = 0 if p is None else p["n"]
            need = n_files - n_have
            if need < 0:
                raise ValueError(
                    f"NE_total: more than {n_files} rows for {utc_key!r} "
                    f"(duplicate collectTime in one device file?)"
                )
            if need > 0:
                z = zero_like_row()
                z["collectTime"] = utc_key
                for _ in range(need):
                    ne_total_partial_add(partials, utc_key, z)
                p = partials[utc_key]
            row = ne_total_partial_finalize(p, utc_key)
            row.update(solar_fields_at(sun, i))
            writer.writerow(row)

    print(f"Wrote {out_path}")


def main(input_dir: Path, output_dir: Path):
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    os.makedirs(output_dir, exist_ok=True)

    csv_paths = get_sorted_csv_paths(input_dir)
    print(f"Found {len(csv_paths)} daily CSV files")

    staging_dir = output_dir / STAGING_DIRNAME
    if staging_dir.is_dir():
        shutil.rmtree(staging_dir)

    print("Pass 1/2: single read of each daily file, fan-out to staging (per devDn)...")
    unique_devdns_set = fanout_daily_rows_to_staging(csv_paths, staging_dir)
    unique_devdns = sorted(unique_devdns_set)
    print(f"Unique devDn count: {len(unique_devdns)}")

    n_points = sum(1 for _ in full_5min_index())
    print(f"Full time index: {n_points} points (5-min for 2025, output collectTime in UTC)")

    zero_row = {k: "0" if k in NUMERIC_COL_NAMES else "" for k in HEADER}

    utc_idx = utc_index_for_grid()  # built once; shared across devices and NE_total

    try:
        print("Pass 2/2: fill 5-min grid from staging (one devDn at a time)...")
        for idx, devdn in enumerate(unique_devdns):
            print(f"Processing devDn {idx + 1}/{len(unique_devdns)}: {devdn}")

            time_to_row, first_row = load_staging_to_time_to_row(
                staging_dir / safe_devdn_filename(devdn)
            )

            # Pick a stable (lat, lon) for this device for solar geometry. Prefer
            # first_row, but fall back to scanning rows if it's missing/sentinel.
            dev_lat = parse_float(first_row.get("latitude_device", ""))
            dev_lon = parse_float(first_row.get("longitude_device", ""))
            if dev_lat in (0.0, -1.0) or dev_lon in (0.0, -1.0):
                for r in time_to_row.values():
                    lat_v = parse_float(r.get("latitude_device", ""))
                    lon_v = parse_float(r.get("longitude_device", ""))
                    if lat_v not in (0.0, -1.0) and lon_v not in (0.0, -1.0):
                        dev_lat, dev_lon = lat_v, lon_v
                        break

            sun = compute_solar_arrays(utc_idx, dev_lat, dev_lon)

            out_path = output_dir / safe_devdn_filename(devdn)
            with open(out_path, "w", encoding="utf-8", newline="") as out:
                writer = csv.DictWriter(out, fieldnames=OUTPUT_HEADER)
                writer.writeheader()
                for i, t in enumerate(full_5min_index()):
                    ts_str = grid_cn_time_to_utc_str(t)
                    if ts_str in time_to_row:
                        row = dict(time_to_row[ts_str])
                    else:
                        row = zero_row.copy()
                        row["collectTime"] = ts_str
                        row["stationCode"] = first_row.get("stationCode") or ""
                        row["devDn"] = devdn
                        row["latitude_device"] = first_row.get("latitude_device") or "0"
                        row["longitude_device"] = first_row.get("longitude_device") or "0"
                    row.update(solar_fields_at(sun, i))
                    writer.writerow(row)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    write_ne_total(output_dir)

    print(f"Done. Output directory: {output_dir}")
    print(f"Per-devDn files: {len(unique_devdns)}, plus {NE_TOTAL_FILENAME}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Aggregate daily 组串式逆变器 CSVs by devDn; output UTC collectTime and NE_total.csv."
    )
    p.add_argument(
        "-i",
        "--input",
        type=Path,
        default='/data/2015_all_station/',
        help="Directory containing 组串式逆变器-YYYY-MM-DD.csv files (default: script directory).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default='/data/data/luoayng_data_626',
        help="Directory for per-devDn CSVs and NE_total.csv (default: <input>/aggregated_by_devDn).",
    )
    args = p.parse_args()
    input_dir = args.input.expanduser()
    output_dir = args.output.expanduser() if args.output is not None else input_dir / "aggregated_by_devDn"
    return input_dir, output_dir


if __name__ == "__main__":
    main(*parse_args())
