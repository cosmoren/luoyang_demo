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
- ``weather_score``: station-level PV slope-jitter in [0, 1] (~1 = slopes jump violently /
  cloudy-unstable, ~0 = slopes change smoothly / sunny); identical across all devDn at
  the same ``collectTime``. In a ±1h window, consecutive 5-min slopes are compared;
  smooth slope evolution scores low, erratic slope changes score high. Night: 0.

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

import numpy as np
import pandas as pd
import pvlib

# Paths already reported as empty (no header row); avoid duplicate stderr lines per devDn pass.
_logged_empty_csv_paths: Set[Path] = set()

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
# (staging files keep the original HEADER).
SOLAR_EXTRA_COLS = [
    "solar_zenith", "solar_azimuth", "local_solar_time",
    "day_of_year", "hour_of_day", "p_cs", "kt", "kt_mask", "p_mean",
    "weather_score",
]
OUTPUT_HEADER = HEADER + SOLAR_EXTRA_COLS

# Minimum normalized clear-sky POA (``p_cs``) to treat a timestep as daytime.
# Shared by per-inverter ``kt_mask`` and station-level ``weather_score``.
P_CS_DAYTIME_THRESHOLD = 0.1

# ``weather_score``: ±1h window on the 5-min grid (12 steps each side → 25 points).
WEATHER_WINDOW_HALF_STEPS = 12
WEATHER_WINDOW_SIZE = 2 * WEATHER_WINDOW_HALF_STEPS + 1
WEATHER_WINDOW_MIN_PERIODS = 13
WEATHER_NORM_PERCENTILE = 95.0


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


def compute_clearsky_power(lat: float, lon: float, utc_index: pd.DatetimeIndex) -> pd.Series:
    """Normalized clear-sky POA (``poa_global / 1000``), clipped to [0, 1.2]."""
    tilt = abs(lat)  # unknown tilt: rough latitude-tilt assumption
    azimuth = 180 if lat >= 0 else 0  # north: south-facing; south: north-facing
    solpos = pvlib.solarposition.get_solarposition(utc_index, lat, lon)

    loc = pvlib.location.Location(lat, lon)
    # 1. Clear-sky GHI/DNI/DHI
    cs = loc.get_clearsky(utc_index, model="ineichen")

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=cs["dni"],
        ghi=cs["ghi"],
        dhi=cs["dhi"],
        albedo=0.2,
    )

    poa_clear = poa["poa_global"].clip(lower=0)
    p_cs = (poa_clear / 1000).clip(0, 1.2)

    return p_cs


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

    p_cs = compute_clearsky_power(lat, lon, utc_index)

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
        "p_cs": p_cs.to_numpy(),
    }
    _solar_cache[key] = arrays
    return arrays


def solar_fields_at(arrays: dict, idx: int) -> dict:
    """Format solar/temporal output fields (incl. ``p_cs``) for the row at array index ``idx``."""
    return {
        "solar_zenith": f"{float(arrays['zenith'][idx]):.4f}",
        "solar_azimuth": f"{float(arrays['azimuth'][idx]):.4f}",
        "local_solar_time": arrays["lst_strs"][idx],
        "day_of_year": str(int(arrays["day_of_year"][idx])),
        "hour_of_day": f"{float(arrays['hour_of_day'][idx]):.4f}",
        "p_cs": f"{float(arrays['p_cs'][idx]):.6f}",
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


def load_active_power_series(staging_path: Path, utc_keys: list) -> np.ndarray:
    """Read one staging CSV into an ``active_power`` array aligned with ``utc_keys``."""
    time_to_row, _ = load_staging_to_time_to_row(staging_path)
    return np.array(
        [
            parse_float(time_to_row.get(ts, {}).get("active_power", "0"))
            for ts in utc_keys
        ],
        dtype=np.float64,
    )


def _slope_jump_std(segment: np.ndarray) -> float:
    """Std of consecutive slope changes within a power segment (>=3 points)."""
    if segment.size < 3:
        return np.nan
    slopes = np.diff(segment)
    if slopes.size < 2:
        return np.nan
    return float(np.std(np.diff(slopes)))


def compute_station_weather_score(
    unique_devdns: list,
    staging_dir: Path,
    p_cs_arr: np.ndarray,
    utc_keys: list,
) -> np.ndarray:
    """
    Station-level PV slope-jitter in [0, 1], shared across all devDn.

    For each timestep ``t`` on the 5-min grid, take station power in [t-1h, t+1h],
    compute slopes between adjacent samples, then measure how much those slopes jump
    from one interval to the next (std of slope differences). Smooth clear-sky ramps
    change slope gradually → low score; cloud-driven up/down swings → high score.

        station_power[t] = sum_i active_power_i(t)
        slopes[k] = P[k+1] - P[k]   (within window)
        jumps[k] = slopes[k+1] - slopes[k]
        raw[t] = std(jumps) / (p_cs[t] * sum_i p_mean_i + eps)
        weather_score[t] = clip(raw[t] / p95_daytime, 0, 1) * kt_mask[t]

    Night timesteps (``p_cs <= P_CS_DAYTIME_THRESHOLD``) are set to 0.
    """
    n = len(utc_keys)
    station_power = np.zeros(n, dtype=np.float64)
    station_scale = 0.0  # sum_i p_mean_i

    for devdn in unique_devdns:
        active_power_arr = load_active_power_series(
            staging_dir / safe_devdn_filename(devdn), utc_keys
        )
        station_power += active_power_arr
        station_scale += float(active_power_arr.mean())

    expected = p_cs_arr * station_scale
    scale = np.maximum(expected, 1e-6)
    hw = WEATHER_WINDOW_HALF_STEPS
    raw = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        lo = max(0, i - hw)
        hi = min(n, i + hw + 1)
        seg = station_power[lo:hi]
        jump_std = _slope_jump_std(seg)
        if np.isnan(jump_std):
            continue
        raw[i] = jump_std / scale[i]

    kt_mask = (p_cs_arr > P_CS_DAYTIME_THRESHOLD).astype(np.float64)
    daytime = kt_mask > 0
    if not np.any(daytime):
        return np.zeros(n, dtype=np.float64)

    valid_day = daytime & np.isfinite(raw)
    if not np.any(valid_day):
        return np.zeros(n, dtype=np.float64)

    p95 = float(np.percentile(raw[valid_day], WEATHER_NORM_PERCENTILE))
    weather_score = np.clip(raw / (p95 + 1e-6), 0.0, 1.0)
    weather_score = np.nan_to_num(weather_score, nan=0.0) * kt_mask
    return weather_score


def main(input_dir: Path, output_dir: Path, lat: float, lon: float):
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    os.makedirs(output_dir, exist_ok=True)

    csv_paths = get_sorted_csv_paths(input_dir)
    print(f"Found {len(csv_paths)} daily CSV files")
    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}\n"
            f"Expected daily files matching 组串式逆变器-YYYY-MM-DD.csv "
            f"(e.g. ~/datasets/2025_all_station)."
        )
    if not csv_paths:
        hint = ""
        alt = Path.home() / "datasets" / "2025_all_station"
        if alt.is_dir() and alt != input_dir:
            hint = f"\nHint: found data at {alt} ({len(get_sorted_csv_paths(alt))} daily files)."
        raise FileNotFoundError(
            f"No 组串式逆变器-YYYY-MM-DD.csv files under {input_dir}.{hint}"
        )

    staging_dir = output_dir / STAGING_DIRNAME
    if staging_dir.is_dir():
        shutil.rmtree(staging_dir, ignore_errors=True)
        if staging_dir.is_dir():
            raise OSError(
                f"Could not remove existing staging directory: {staging_dir}. "
                "Stop other aggregate runs and delete it manually."
            )

    print("Pass 1/2: single read of each daily file, fan-out to staging (per devDn)...")
    unique_devdns_set = fanout_daily_rows_to_staging(csv_paths, staging_dir)
    unique_devdns = sorted(unique_devdns_set)
    print(f"Unique devDn count: {len(unique_devdns)}")

    n_points = sum(1 for _ in full_5min_index())
    print(f"Full time index: {n_points} points (5-min for 2025, output collectTime in UTC)")

    zero_row = {k: "0" if k in NUMERIC_COL_NAMES else "" for k in HEADER}

    utc_idx = utc_index_for_grid()  # built once; shared across all per-devDn outputs

    try:
        sun = compute_solar_arrays(utc_idx, lat, lon)
        p_cs_arr = np.asarray(sun["p_cs"], dtype=np.float64)
        utc_keys = ordered_utc_collect_time_keys()

        print("Pass 2a/2: computing station-level weather_score (shared across all devDn)...")
        weather_score_arr = compute_station_weather_score(
            unique_devdns, staging_dir, p_cs_arr, utc_keys
        )

        print("Pass 2b/2: fill 5-min grid from staging (one devDn at a time)...")
        for idx, devdn in enumerate(unique_devdns):
            print(f"Processing devDn {idx + 1}/{len(unique_devdns)}: {devdn}")

            time_to_row, first_row = load_staging_to_time_to_row(
                staging_dir / safe_devdn_filename(devdn)
            )

            active_power_arr = load_active_power_series(
                staging_dir / safe_devdn_filename(devdn), utc_keys
            )
            if active_power_arr.shape != p_cs_arr.shape:
                raise ValueError(
                    f"shape mismatch for {devdn}: active_power={active_power_arr.shape}, "
                    f"p_cs={p_cs_arr.shape}"
                )

            p_mean = float(active_power_arr.mean())
            kt_mask = (p_cs_arr > P_CS_DAYTIME_THRESHOLD).astype(np.float64)
            kt = active_power_arr / (p_cs_arr * p_mean + 1e-6)
            kt = kt * kt_mask

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
                        row["latitude_device"] = fmt_number(lat)
                        row["longitude_device"] = fmt_number(lon)
                    row.update(solar_fields_at(sun, i))
                    row["kt"] = f"{float(kt[i]):.6f}"
                    row["kt_mask"] = str(int(kt_mask[i]))
                    row["p_mean"] = f"{p_mean:.6f}"
                    row["weather_score"] = f"{float(weather_score_arr[i]):.6f}"
                    writer.writerow(row)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    print(f"Done. Output directory: {output_dir}")
    print(f"Per-devDn files: {len(unique_devdns)}")


def parse_args():
    home_datasets = Path.home() / "datasets"
    p = argparse.ArgumentParser(
        description="Aggregate daily 组串式逆变器 CSVs by devDn; output UTC collectTime per devDn."
    )
    p.add_argument(
        "-i",
        "--input",
        type=Path,
        default=home_datasets / "2025_all_station",
        help="Directory containing 组串式逆变器-YYYY-MM-DD.csv files (default: ~/datasets/2025_all_station).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=home_datasets / "luoyang_data_626",
        help="Directory for per-devDn CSVs (default: ~/datasets/luoyang_data_626).",
    )
    p.add_argument(
        "--lat",
        type=float,
        default=34.69984,
        help="Site latitude (decimal degrees) used for solar geometry of all devices (default: 34.68).",
    )
    p.add_argument(
        "--lon",
        type=float,
        default=112.28440,
        help="Site longitude (decimal degrees) used for solar geometry of all devices (default: 112.45).",
    )
    args = p.parse_args()
    input_dir = args.input.expanduser()
    output_dir = args.output.expanduser() if args.output is not None else input_dir / "aggregated_by_devDn"
    return input_dir, output_dir, args.lat, args.lon


if __name__ == "__main__":
    main(*parse_args())
