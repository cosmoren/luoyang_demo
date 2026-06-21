#!/usr/bin/env python3
"""
Augment YLJ parquet rows with solar/clear-sky features and kt ratios.

Per row (anchor: ``timestamp_win``), this script appends:
- past (672 x 15min): zenith/azimuth/day_of_year/hour_of_day/local_solar_time,
  CS_GHI/CS_DNI/CS_DHI/CS_power, delta_t (minutes)
- future (192 x 15min): same keys with ``_predict`` suffix, plus delta_t_predict
- one global p_mean estimated from the whole dataset
- kt = observe_power / (CS_power * p_mean)
- kt_predict = observe_power_future / (CS_power_predict * p_mean)

Timezone policy:
- naive ``timestamp_win`` is treated as Asia/Shanghai, then converted to UTC.

CS_power definition:
- clear-sky POA global / 1000
- POA orientation: tilt=latitude, azimuth=180 (south-facing)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pvlib
from pvlib import solarposition


# ----------------------------
# Script defaults / constants
# ----------------------------
# Input parquet must contain:
# - timestamp_win
# - observe_power (past sequence)
# - observe_power_future (future sequence)
#
# Output parquet keeps original columns and appends:
# - solar/clear-sky features for past + future windows
# - kt / kt_predict
# - one global p_mean (same value for all rows)
#
# If --p-mean is not provided, p_mean is estimated from data using
# a robust percentile rule (not max) to reduce outlier impact.
#
# Run examples:
# 1) Auto-estimate p_mean
# micromamba run -n SimVP python SPMF_preprocessing/ylj/ylj_parquet_solarfeats.py \
#   --input-parquet ~/datasets/ylj_raw/ds_v322_2024.parquet \
#   --output-parquet ~/datasets/ylj_raw/ds_v322_2024_with_solarfeats.parquet \
#   --lat 29.9 --lon 100.5
#
# 2) Use a fixed p_mean
# micromamba run -n SimVP python SPMF_preprocessing/ylj/ylj_parquet_solarfeats.py \
#   --input-parquet ~/datasets/ylj_raw/ds_v322_2024.parquet \
#   --output-parquet ~/datasets/ylj_raw/ds_v322_2024_with_solarfeats.parquet \
#   --lat 29.9 --lon 100.5 \
#   --p-mean 1.23

DEFAULT_INPUT = Path("/mnt/nfs/slurm/home/yuan/datasets/ylj_raw/ds_v322_2024.parquet")
DEFAULT_ANCHOR_TZ = "Asia/Shanghai"
DEFAULT_LAT = 29.9
DEFAULT_LON = 100.5
# Panel tilt used by clear-sky POA calculation (degrees).
# Keep this as a top-level constant so it is easy to change in one place.
DEFAULT_SURFACE_TILT_DEG = 29.9
DEFAULT_YEAR = 2024
DEFAULT_FREQ_MIN = 15
DEFAULT_PAST_LEN = 672
DEFAULT_FUTURE_LEN = 192
DEFAULT_EPS = 1e-6
DEFAULT_KT_TARGET_MAX = 1.1
DEFAULT_KT_TARGET_QUANTILE = 99.0
DEFAULT_RATIO_SAMPLES_CAP = 2_000_000


def parse_coord(text: str, *, is_lat: bool) -> float:
    """
    Parse coordinate from formats like:
    - "29.9"
    - "29.9N" / "100.5E"
    - "-121.17"
    """
    s = str(text).strip().upper()
    m = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*([NSEW])?\s*$", s)
    if not m:
        raise argparse.ArgumentTypeError(f"Invalid coordinate: {text!r}")
    value = float(m.group(1))
    hemi = m.group(2)
    if hemi in ("S", "W"):
        value = -abs(value)
    elif hemi in ("N", "E"):
        value = abs(value)
    if is_lat and not (-90.0 <= value <= 90.0):
        raise argparse.ArgumentTypeError(f"Latitude out of range [-90, 90]: {value}")
    if (not is_lat) and not (-180.0 <= value <= 180.0):
        raise argparse.ArgumentTypeError(f"Longitude out of range [-180, 180]: {value}")
    return value


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add solar/clear-sky features to YLJ parquet rows.")
    p.add_argument("--input-parquet", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-parquet", type=Path, default=None)
    p.add_argument("--lat", type=lambda s: parse_coord(s, is_lat=True), default=DEFAULT_LAT)
    p.add_argument("--lon", type=lambda s: parse_coord(s, is_lat=False), default=DEFAULT_LON)
    p.add_argument("--year", type=int, default=DEFAULT_YEAR)
    p.add_argument("--anchor-timezone", type=str, default=DEFAULT_ANCHOR_TZ)
    p.add_argument("--freq-min", type=int, default=DEFAULT_FREQ_MIN)
    p.add_argument("--past-len", type=int, default=DEFAULT_PAST_LEN)
    p.add_argument("--future-len", type=int, default=DEFAULT_FUTURE_LEN)
    p.add_argument("--eps", type=float, default=DEFAULT_EPS)
    p.add_argument(
        "--p-mean",
        type=float,
        default=None,
        help="Optional fixed global p_mean. If omitted, p_mean is estimated from data.",
    )
    return p.parse_args()


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_with_solarfeats.parquet")


def _normalize_anchor_series(raw: pd.Series, anchor_tz: str) -> pd.Series:
    ts = pd.to_datetime(raw, errors="coerce")
    if ts.isna().any():
        bad = int(ts.isna().sum())
        raise ValueError(f"timestamp_win has {bad} unparseable rows")

    # pandas stores either all-tz-aware or all-naive for a datetime Series.
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(anchor_tz).dt.tz_convert("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")
    return ts


def _utc_to_local_solar_time_pvlib(utc_times: pd.DatetimeIndex, longitude: float) -> pd.DatetimeIndex:
    if utc_times.tz is not None:
        utc_naive = utc_times.tz_convert("UTC").tz_localize(None)
    else:
        utc_naive = utc_times
    lmst_offset_hours = longitude / 15.0
    dayofyear = utc_naive.dayofyear
    eot_minutes = solarposition.equation_of_time_spencer71(dayofyear)
    local_solar = utc_naive + pd.Timedelta(hours=lmst_offset_hours) + pd.to_timedelta(eot_minutes, unit="m")
    return local_solar


def _build_lookup(
    *,
    lat: float,
    lon: float,
    surface_tilt_deg: float,
    year: int,
    freq_min: int,
    required_start: pd.Timestamp,
    required_end: pd.Timestamp,
) -> pd.DataFrame:
    year_start = pd.Timestamp(f"{year}-01-01 00:00:00", tz="UTC")
    next_year_start = pd.Timestamp(f"{year + 1}-01-01 00:00:00", tz="UTC")
    year_end = next_year_start - pd.Timedelta(minutes=freq_min)

    start = min(required_start.tz_convert("UTC"), year_start)
    end = max(required_end.tz_convert("UTC"), year_end)
    grid_utc = pd.date_range(start=start, end=end, freq=f"{freq_min}min", tz="UTC")

    solpos = solarposition.get_solarposition(grid_utc, lat, lon)
    cs = pvlib.location.Location(lat, lon).get_clearsky(grid_utc, model="ineichen")
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=float(surface_tilt_deg),
        surface_azimuth=180.0,
        dni=np.asarray(cs["dni"].values, dtype=np.float64),
        ghi=np.asarray(cs["ghi"].values, dtype=np.float64),
        dhi=np.asarray(cs["dhi"].values, dtype=np.float64),
        solar_zenith=np.asarray(solpos["apparent_zenith"].values, dtype=np.float64),
        solar_azimuth=np.asarray(solpos["azimuth"].values, dtype=np.float64),
    )
    local_solar = _utc_to_local_solar_time_pvlib(grid_utc, lon)
    hour_of_day = (
        local_solar.hour.values
        + local_solar.minute.values / 60.0
        + local_solar.second.values / 3600.0
    ).astype(np.float32)

    poa_global = np.asarray(poa["poa_global"], dtype=np.float32)

    out = pd.DataFrame(
        {
            "time_utc": grid_utc,
            "local_solar_time": local_solar.tz_localize(None),
            "zenith": np.asarray(solpos["zenith"].values, dtype=np.float32),
            "azimuth": np.asarray(solpos["azimuth"].values, dtype=np.float32),
            "day_of_year": local_solar.dayofyear.values.astype(np.int16),
            "hour_of_day": hour_of_day,
            "CS_GHI": np.asarray(cs["ghi"].values, dtype=np.float32),
            "CS_DNI": np.asarray(cs["dni"].values, dtype=np.float32),
            "CS_DHI": np.asarray(cs["dhi"].values, dtype=np.float32),
            "CS_power": (poa_global / 1000.0).astype(np.float32),
        }
    )
    out.index = out["time_utc"]
    return out


def _as_float_array(cell: Any, expected_len: int, name: str, row_idx: int) -> np.ndarray:
    arr = np.asarray(cell, dtype=np.float32).reshape(-1)
    if arr.shape[0] != expected_len:
        raise ValueError(
            f"row {row_idx}: {name} length={arr.shape[0]} (expected {expected_len})"
        )
    return arr


def _try_as_float_array(cell: Any, expected_len: int) -> np.ndarray | None:
    try:
        arr = np.asarray(cell, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.shape[0] != expected_len:
        return None
    return arr


def _to_minute_offsets(length: int, freq_min: int, *, future: bool) -> np.ndarray:
    if future:
        return np.arange(1, length + 1, dtype=np.int64) * int(freq_min)
    return np.arange(-(length - 1), 1, dtype=np.int64) * int(freq_min)


def main() -> None:
    args = parse_args()

    input_path = args.input_parquet.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")
    output_path = (
        args.output_parquet.expanduser().resolve()
        if args.output_parquet
        else _default_output_path(input_path)
    )

    print(f"[ylj] input: {input_path}")
    print(f"[ylj] output: {output_path}")
    print(f"[ylj] site: lat={args.lat}, lon={args.lon}")
    print(f"[ylj] cs_power_tilt_deg={DEFAULT_SURFACE_TILT_DEG}")

    df = pd.read_parquet(input_path)
    required_cols = ("timestamp_win", "observe_power", "observe_power_future")
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    anchors_utc = _normalize_anchor_series(df["timestamp_win"], args.anchor_timezone)
    freq = pd.Timedelta(minutes=int(args.freq_min))

    required_start = anchors_utc.min() - (int(args.past_len) - 1) * freq
    required_end = anchors_utc.max() + int(args.future_len) * freq
    lookup = _build_lookup(
        lat=float(args.lat),
        lon=float(args.lon),
        surface_tilt_deg=float(DEFAULT_SURFACE_TILT_DEG),
        year=int(args.year),
        freq_min=int(args.freq_min),
        required_start=required_start,
        required_end=required_end,
    )

    index_map = {int(ts.value): i for i, ts in enumerate(lookup.index)}
    lookup_cols = {
        "zenith": lookup["zenith"].to_numpy(dtype=np.float32, copy=False),
        "azimuth": lookup["azimuth"].to_numpy(dtype=np.float32, copy=False),
        "day_of_year": lookup["day_of_year"].to_numpy(dtype=np.float32, copy=False),
        "hour_of_day": lookup["hour_of_day"].to_numpy(dtype=np.float32, copy=False),
        "CS_GHI": lookup["CS_GHI"].to_numpy(dtype=np.float32, copy=False),
        "CS_DNI": lookup["CS_DNI"].to_numpy(dtype=np.float32, copy=False),
        "CS_DHI": lookup["CS_DHI"].to_numpy(dtype=np.float32, copy=False),
        "CS_power": lookup["CS_power"].to_numpy(dtype=np.float32, copy=False),
    }
    time_utc_str = lookup.index.strftime("%Y-%m-%d %H:%M:%S").to_numpy()
    local_solar_str = lookup["local_solar_time"].dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy()

    past_off_min = _to_minute_offsets(int(args.past_len), int(args.freq_min), future=False)
    fut_off_min = _to_minute_offsets(int(args.future_len), int(args.freq_min), future=True)
    past_off_ns = past_off_min * 60 * 1_000_000_000
    fut_off_ns = fut_off_min * 60 * 1_000_000_000
    anchor_ns = anchors_utc.astype("int64").to_numpy()

    n = len(df)
    out_cols: dict[str, list[Any]] = {
        "timestamp_utc": [],
        "zenith": [],
        "azimuth": [],
        "day_of_year": [],
        "hour_of_day": [],
        "local_solar_time": [],
        "CS_GHI": [],
        "CS_DNI": [],
        "CS_DHI": [],
        "CS_power": [],
        "delta_t": [],
        "timestamp_utc_predict": [],
        "zenith_predict": [],
        "azimuth_predict": [],
        "day_of_year_predict": [],
        "hour_of_day_predict": [],
        "local_solar_time_predict": [],
        "CS_GHI_predict": [],
        "CS_DNI_predict": [],
        "CS_DHI_predict": [],
        "CS_power_predict": [],
        "delta_t_predict": [],
        "kt": [],
        "kt_predict": [],
    }

    # Pass 1: collect valid rows and optionally estimate one global p_mean.
    # Robust strategy (when not provided): use q-th percentile of (observe_power / CS_power),
    # so a few outliers do not shrink most kt values.
    dropped_rows: list[int] = []
    kept_indices: list[int] = []
    ratio_chunks: list[np.ndarray] = []
    ratio_kept = 0
    rng = np.random.default_rng(0)

    for i in range(n):
        obs = _try_as_float_array(df.iloc[i]["observe_power"], int(args.past_len))
        obs_f = _try_as_float_array(df.iloc[i]["observe_power_future"], int(args.future_len))
        if obs is None or obs_f is None:
            dropped_rows.append(i)
            continue

        a_ns = int(anchor_ns[i])
        past_ns = a_ns + past_off_ns
        fut_ns = a_ns + fut_off_ns
        try:
            past_idx = np.asarray([index_map[int(x)] for x in past_ns], dtype=np.int64)
            fut_idx = np.asarray([index_map[int(x)] for x in fut_ns], dtype=np.int64)
        except KeyError as exc:
            raise KeyError(f"row {i}: timestamp window out of lookup range around {anchors_utc.iloc[i]}") from exc

        if args.p_mean is None:
            cs_p = lookup_cols["CS_power"][past_idx].astype(np.float32, copy=False)
            cs_pf = lookup_cols["CS_power"][fut_idx].astype(np.float32, copy=False)
            m1 = cs_p > float(args.eps)
            m2 = cs_pf > float(args.eps)
            if bool(np.any(m1)):
                ratio1 = np.nan_to_num(obs[m1] / cs_p[m1], nan=0.0, posinf=0.0, neginf=0.0)
                ratio1 = ratio1[np.isfinite(ratio1) & (ratio1 >= 0.0)]
                if ratio1.size:
                    remain = max(0, int(DEFAULT_RATIO_SAMPLES_CAP) - int(ratio_kept))
                    if remain > 0:
                        if ratio1.size > remain:
                            ratio1 = ratio1[rng.choice(ratio1.size, size=remain, replace=False)]
                        ratio_chunks.append(ratio1.astype(np.float32, copy=False))
                        ratio_kept += int(ratio1.size)
            if bool(np.any(m2)):
                ratio2 = np.nan_to_num(obs_f[m2] / cs_pf[m2], nan=0.0, posinf=0.0, neginf=0.0)
                ratio2 = ratio2[np.isfinite(ratio2) & (ratio2 >= 0.0)]
                if ratio2.size:
                    remain = max(0, int(DEFAULT_RATIO_SAMPLES_CAP) - int(ratio_kept))
                    if remain > 0:
                        if ratio2.size > remain:
                            ratio2 = ratio2[rng.choice(ratio2.size, size=remain, replace=False)]
                        ratio_chunks.append(ratio2.astype(np.float32, copy=False))
                        ratio_kept += int(ratio2.size)
        kept_indices.append(i)

    if not kept_indices:
        raise ValueError("No valid rows left after dropping invalid observe_power lengths.")
    if args.p_mean is not None:
        p_mean = max(float(args.p_mean), float(args.eps))
        print(f"[ylj] using provided p_mean={p_mean:.6f}")
    else:
        if not ratio_chunks:
            raise ValueError("Unable to estimate p_mean: no valid (observe_power / CS_power) samples found.")
        ratio_all = np.concatenate(ratio_chunks, axis=0)
        ratio_q = float(np.percentile(ratio_all, float(DEFAULT_KT_TARGET_QUANTILE)))
        p_mean = max(ratio_q / float(DEFAULT_KT_TARGET_MAX), float(args.eps))
        print(
            f"[ylj] estimated global p_mean={p_mean:.6f} "
            f"(q{DEFAULT_KT_TARGET_QUANTILE} ratio={ratio_q:.6f}, target_kt_max={DEFAULT_KT_TARGET_MAX}, "
            f"ratio_samples={ratio_all.size})"
        )

    clipped_denom_count = 0
    for out_idx, i in enumerate(kept_indices, start=1):
        obs = _try_as_float_array(df.iloc[i]["observe_power"], int(args.past_len))
        obs_f = _try_as_float_array(df.iloc[i]["observe_power_future"], int(args.future_len))
        assert obs is not None and obs_f is not None

        a_ns = int(anchor_ns[i])
        past_ns = a_ns + past_off_ns
        fut_ns = a_ns + fut_off_ns
        past_idx = np.asarray([index_map[int(x)] for x in past_ns], dtype=np.int64)
        fut_idx = np.asarray([index_map[int(x)] for x in fut_ns], dtype=np.int64)

        for k in ("zenith", "azimuth", "day_of_year", "hour_of_day", "CS_GHI", "CS_DNI", "CS_DHI", "CS_power"):
            out_cols[k].append(lookup_cols[k][past_idx].astype(np.float32, copy=False).tolist())
            out_cols[f"{k}_predict"].append(lookup_cols[k][fut_idx].astype(np.float32, copy=False).tolist())

        out_cols["timestamp_utc"].append(time_utc_str[past_idx].tolist())
        out_cols["timestamp_utc_predict"].append(time_utc_str[fut_idx].tolist())
        out_cols["local_solar_time"].append(local_solar_str[past_idx].tolist())
        out_cols["local_solar_time_predict"].append(local_solar_str[fut_idx].tolist())
        out_cols["delta_t"].append(past_off_min.astype(np.float32, copy=False).tolist())
        out_cols["delta_t_predict"].append(fut_off_min.astype(np.float32, copy=False).tolist())

        cs_p = lookup_cols["CS_power"][past_idx].astype(np.float32, copy=False)
        cs_pf = lookup_cols["CS_power"][fut_idx].astype(np.float32, copy=False)
        denom_p = np.maximum(cs_p * float(p_mean), float(args.eps))
        denom_pf = np.maximum(cs_pf * float(p_mean), float(args.eps))
        clipped_denom_count += int((cs_p * float(p_mean) < float(args.eps)).sum() + (cs_pf * float(p_mean) < float(args.eps)).sum())

        kt = np.nan_to_num(obs / denom_p, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        kt_pred = np.nan_to_num(obs_f / denom_pf, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        out_cols["kt"].append(kt.tolist())
        out_cols["kt_predict"].append(kt_pred.tolist())

        if out_idx % 1000 == 0:
            print(f"[ylj] processed {out_idx}/{len(kept_indices)} valid rows")

    out_df = df.iloc[kept_indices].copy()
    for k, v in out_cols.items():
        out_df[k] = v
    out_df["p_mean"] = float(p_mean)

    '''
    if len(out_df) > 0:
        first = out_df.iloc[0]
        print(f"[ylj:first] kt={first.get('kt')}")
        print(f"[ylj:first] kt_predict={first.get('kt_predict')}")
        print(f"[ylj:first] p_mean={first.get('p_mean')}")
    '''

    # Lightweight validation on generated columns.
    for k in ("zenith", "delta_t", "kt"):
        lengths = out_df[k].map(lambda x: len(x) if isinstance(x, (list, tuple, np.ndarray)) else -1)
        if int(lengths.min()) != int(args.past_len) or int(lengths.max()) != int(args.past_len):
            raise ValueError(f"{k} has non-uniform past length")
    for k in ("zenith_predict", "delta_t_predict", "kt_predict"):
        lengths = out_df[k].map(lambda x: len(x) if isinstance(x, (list, tuple, np.ndarray)) else -1)
        if int(lengths.min()) != int(args.future_len) or int(lengths.max()) != int(args.future_len):
            raise ValueError(f"{k} has non-uniform future length")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)
    print(f"[ylj] wrote: {output_path}")
    print(
        f"[ylj] rows_in={len(df)} rows_out={len(out_df)} dropped_rows={len(dropped_rows)} "
        f"clipped_cs_power_points={clipped_denom_count} p_mean={p_mean:.6f}"
    )
    if dropped_rows:
        print(f"[ylj] dropped row sample (first 10): {dropped_rows[:10]}")


if __name__ == "__main__":
    main()
