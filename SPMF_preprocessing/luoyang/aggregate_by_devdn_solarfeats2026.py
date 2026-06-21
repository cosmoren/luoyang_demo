#!/usr/bin/env python3
"""
Convert SolarNet Luoyang per-plant long power CSVs to pv_kt-like per-devDn CSVs.

Input: one file per plant under power_575/, e.g.:
  plant_guangfu333858040_power_long.csv

Output: one file per plant under output dir, e.g.:
  NE_333858040.csv

Rules:
- Source ``dtime`` is Beijing local time (UTC+8), kept in output.
- Output ``collectTime`` is UTC0 converted from ``dtime``.
- ``devDn`` uses ``NE=<PlantID>``.
- Keep source columns (e.g. observe_power, weather_ghi) using original names.
- Add solar/kt fields similar to pv_ktuni:
  solar_zenith, solar_azimuth, local_solar_time, day_of_year, hour_of_day, p_cs, kt, kt_mask, p_mean.
"""

from __future__ import annotations

import argparse
import re
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

CN_UTC_OFFSET = timedelta(hours=8)
P_CS_DAYTIME_THRESHOLD = 0.1
DROP_OUTPUT_COLS = {
    "Power1",
    "Power2",
    "ActualPower",
    "stationCode",
    "latitude_device",
    "longitude_device",
    "capacity",
    "inverter_state",
    "efficiency",
    "temperature",
    "power_factor",
    "elec_freq",
    "active_power",
    "reactive_power",
    "day_cap",
    "mppt_power",
    "total_cap",
    "mppt_total_cap",
}

BASE_HEADER = [
    "collectTime",
    "devDn",
    "solar_zenith",
    "solar_azimuth",
    "local_solar_time",
    "day_of_year",
    "hour_of_day",
    "p_cs",
    "kt",
    "kt_mask",
    "p_mean",
    "dtime",
]


def _format_ts(dt: pd.Timestamp) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _compute_clearsky_power(
    lat: float, lon: float, utc_index: pd.DatetimeIndex
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    tilt = abs(lat)
    azimuth = 180 if lat >= 0 else 0

    solpos = pvlib.solarposition.get_solarposition(utc_index, lat, lon)
    zenith = solpos["apparent_zenith"].to_numpy()
    az = solpos["azimuth"].to_numpy()
    eot_min = solpos["equation_of_time"].to_numpy()

    loc = pvlib.location.Location(lat, lon)
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
    p_cs = (poa["poa_global"].clip(lower=0) / 1000.0).clip(0, 1.2).to_numpy()

    utc_naive = utc_index.tz_convert("UTC").tz_localize(None)
    lst_index = utc_naive + pd.to_timedelta(4.0 * lon + eot_min, unit="min")
    lst_strs = lst_index.strftime("%Y-%m-%d %H:%M:%S").tolist()
    day_of_year = lst_index.dayofyear.to_numpy()
    hour_of_day = (
        lst_index.hour + lst_index.minute / 60.0 + lst_index.second / 3600.0
    ).to_numpy()

    return zenith, az, p_cs, lst_strs, day_of_year, hour_of_day


def _source_files(input_dir: Path) -> list[Path]:
    pat = re.compile(r"plant_guangfu(\d+)_power_long\.csv$")
    out: list[tuple[int, Path]] = []
    for p in input_dir.glob("plant_guangfu*_power_long.csv"):
        m = pat.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return [p for _, p in out]


def _plant_id_from_path(path: Path) -> str:
    m = re.match(r"plant_guangfu(\d+)_power_long\.csv$", path.name)
    if m is None:
        raise ValueError(f"unexpected source file name: {path.name}")
    return m.group(1)


def _build_output_df(df: pd.DataFrame, plant_id: str, lat: float, lon: float) -> pd.DataFrame:
    if "dtime" not in df.columns:
        raise KeyError("source CSV missing required column: dtime")
    if "final_power" not in df.columns:
        raise KeyError("source CSV missing required column: final_power")

    out = pd.DataFrame()
    dtime_local = pd.to_datetime(df["dtime"], errors="coerce")
    if dtime_local.isna().any():
        bad = int(dtime_local.isna().sum())
        raise ValueError(f"{plant_id}: dtime parse failed for {bad} rows")

    collect_utc = dtime_local - CN_UTC_OFFSET
    utc_index = pd.DatetimeIndex(collect_utc).tz_localize("UTC")
    (
        zenith,
        azimuth,
        p_cs_arr,
        lst_strs,
        day_of_year,
        hour_of_day,
    ) = _compute_clearsky_power(lat, lon, utc_index)

    active_power = pd.to_numeric(df["final_power"], errors="coerce").fillna(0.0).to_numpy()
    p_mean = float(np.mean(active_power))
    kt_mask = (p_cs_arr > P_CS_DAYTIME_THRESHOLD).astype(np.int32)
    kt = active_power / (p_cs_arr * p_mean + 1e-6)
    kt = kt * kt_mask

    dev_dn = f"NE={plant_id}"
    out["collectTime"] = [_format_ts(t) for t in collect_utc]
    out["devDn"] = dev_dn
    out["solar_zenith"] = np.round(zenith, 4)
    out["solar_azimuth"] = np.round(azimuth, 4)
    out["local_solar_time"] = lst_strs
    out["day_of_year"] = day_of_year.astype(np.int32)
    out["hour_of_day"] = np.round(hour_of_day, 4)
    out["p_cs"] = np.round(p_cs_arr, 6)
    out["kt"] = np.round(kt, 6)
    out["kt_mask"] = kt_mask.astype(np.int32)
    out["p_mean"] = np.round(np.full(len(out), p_mean, dtype=np.float64), 6)
    out["dtime"] = df["dtime"].astype(str)

    passthrough_cols = [
        c for c in df.columns if c not in {"dtime"} and c not in DROP_OUTPUT_COLS
    ]
    for c in passthrough_cols:
        if c in out.columns:
            # keep our mapped target-style columns if name collides
            continue
        else:
            out[c] = df[c]

    ordered = BASE_HEADER + [c for c in out.columns if c not in BASE_HEADER]
    out = out[ordered]
    return out


def convert_all(input_dir: Path, output_dir: Path, lat: float, lon: float) -> None:
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = _source_files(input_dir)
    print(f"Found {len(files)} source files under {input_dir}")
    if not files:
        raise FileNotFoundError("no plant_guangfu*_power_long.csv files found")

    for i, path in enumerate(files, start=1):
        plant_id = _plant_id_from_path(path)
        print(f"[{i}/{len(files)}] processing {path.name}")
        df = pd.read_csv(path)
        out_df = _build_output_df(df, plant_id=plant_id, lat=lat, lon=lon)
        out_path = output_dir / f"NE_{plant_id}.csv"
        out_df.to_csv(out_path, index=False)

    print(f"Done. Wrote {len(files)} files to: {output_dir}")


def parse_args() -> tuple[Path, Path, float, float]:
    home = Path.home() / "datasets"
    parser = argparse.ArgumentParser(
        description="Convert SolarNet per-plant power CSVs to pv_kt-like per-devDn CSVs."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=home / "SolarNet-LuoYang-v1" / "processed_pv" / "power_575",
        help="Source folder with plant_guangfu*_power_long.csv files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=home / "luoyang_2026_SPMF" / "pv_test",
        help="Output folder for converted NE_<PlantID>.csv files.",
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=34.69984,
        help="Site latitude for solar features.",
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=112.28440,
        help="Site longitude for solar features.",
    )
    args = parser.parse_args()
    return args.input, args.output, args.lat, args.lon


if __name__ == "__main__":
    convert_all(*parse_args())

