#!/usr/bin/env python3
"""
Convert Luoyang 2026 whole-plant power CSV to one pv_kt-like NE_total CSV.

Input (single file):
  /mnt/nfs/slurm/home/yuan/datasets/SolarNet-LuoYang-v1/processed_pv/whole_plant_power_governed.csv

Rules:
- Source ``dtime`` is Beijing local time (UTC+8).
- Output ``collectTime`` is UTC converted from ``dtime``.
- Add solar/kt fields:
  solar_zenith, solar_azimuth, local_solar_time, day_of_year, hour_of_day,
  p_cs, kt, kt_mask, p_mean.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

CN_UTC_OFFSET = timedelta(hours=8)
P_CS_DAYTIME_THRESHOLD = 0.1
DEFAULT_INPUT = Path(
    "/mnt/nfs/slurm/home/yuan/datasets/SolarNet-LuoYang-v1/processed_pv/whole_plant_power_governed.csv"
)
DEFAULT_OUTPUT = Path.home() / "datasets" / "luoyang_2026_SPMF" / "pv_test" / "NE_total.csv"

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
    lat: float,
    lon: float,
    utc_index: pd.DatetimeIndex,
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


def _build_output_df(df: pd.DataFrame, *, dev_dn: str, lat: float, lon: float) -> pd.DataFrame:
    if "dtime" not in df.columns:
        raise KeyError("source CSV missing required column: dtime")
    if "final_power" not in df.columns:
        raise KeyError("source CSV missing required column: final_power")

    dtime_local = pd.to_datetime(df["dtime"], errors="coerce")
    if dtime_local.isna().any():
        bad = int(dtime_local.isna().sum())
        raise ValueError(f"dtime parse failed for {bad} rows")

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

    active_power = pd.to_numeric(df["final_power"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    kt_mask = (p_cs_arr > P_CS_DAYTIME_THRESHOLD).astype(np.int32)

    # Calibrate p_mean so daytime kt has P90 ~ 1:
    # ratio = final_power / p_cs  (for kt_mask==1), then p_mean = Q90(ratio).
    # Fallback to global mean if daytime points are unavailable or quantile is invalid.
    day_sel = kt_mask == 1
    ratio = np.divide(
        active_power,
        p_cs_arr,
        out=np.full_like(active_power, np.nan, dtype=np.float64),
        where=(p_cs_arr > 0),
    )
    day_ratio = ratio[day_sel]
    day_ratio = day_ratio[np.isfinite(day_ratio)]
    if day_ratio.size > 0:
        p_mean = float(np.quantile(day_ratio, 0.9))
    else:
        p_mean = float(np.mean(active_power))
    if (not np.isfinite(p_mean)) or p_mean <= 0:
        p_mean = max(float(np.mean(active_power)), 1e-6)

    denom = p_cs_arr * p_mean
    kt = np.divide(
        active_power,
        denom,
        out=np.zeros_like(active_power, dtype=np.float64),
        where=(denom > 0),
    )
    kt = kt * kt_mask

    out = pd.DataFrame()
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

    passthrough_cols = [c for c in df.columns if c not in {"dtime"} and c not in DROP_OUTPUT_COLS]
    for c in passthrough_cols:
        if c in out.columns:
            continue
        out[c] = df[c]

    ordered = BASE_HEADER + [c for c in out.columns if c not in BASE_HEADER]
    return out[ordered]


def convert_one(input_csv: Path, output_csv: Path, *, dev_dn: str, lat: float, lon: float) -> None:
    input_csv = input_csv.expanduser().resolve()
    output_csv = output_csv.expanduser().resolve()
    if not input_csv.is_file():
        raise FileNotFoundError(f"input CSV not found: {input_csv}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {input_csv}")
    df = pd.read_csv(input_csv)
    out_df = _build_output_df(df, dev_dn=dev_dn, lat=lat, lon=lon)
    out_df.to_csv(output_csv, index=False)
    print(f"Done. Wrote {len(out_df)} rows to: {output_csv}")


def parse_args() -> tuple[Path, Path, str, float, float]:
    parser = argparse.ArgumentParser(
        description="Convert whole_plant_power_governed.csv to pv_kt-like NE_total.csv."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input whole-plant CSV (Beijing dtime).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path (single file).",
    )
    parser.add_argument(
        "--devdn",
        type=str,
        default="NE=total",
        help="Output devDn value for all rows.",
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
    return args.input, args.output, args.devdn, args.lat, args.lon


if __name__ == "__main__":
    in_csv, out_csv, devdn, lat, lon = parse_args()
    convert_one(in_csv, out_csv, dev_dn=devdn, lat=lat, lon=lon)
