"""
Batch-crop Himawari L2 cloud NetCDF files with ``nc_crop_mask`` and save all samples into one ``.zarr``.

Default input: ``/home/hw1/data/himawari2025L2cloud_raw``. Output is one Zarr store folder (default
``.../himawari2025L2cloud_all.zarr``) containing:
  - ``merged`` with shape ``(time, y, x, channel)``
  - ``ts_datetime`` on the ``time`` axis (UTC+0)
Unreadable or corrupt NC/HDF files are skipped with a log line. Site lat/lon must be provided manually
via command-line arguments.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from pvlib import solarposition

import numpy as np
import pandas as pd
import xarray as xr

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from preprocessing.himawari_infer import parse_time_from_nc_name
from preprocessing.nc_processing import nc_crop_mask

_NETCDF_SUFFIXES = {".nc", ".nc4"}
_RE_14 = re.compile(r"\d{14}")


def utc_to_local_solar_time_pvlib(utc_times: pd.DatetimeIndex, longitude: float) -> pd.DatetimeIndex:
    """Convert UTC to local (apparent) solar time using longitude and pvlib equation of time."""
    if utc_times.tz is not None:
        utc_naive = utc_times.tz_convert("UTC").tz_localize(None)
    else:
        utc_naive = utc_times
    lmst_offset_hours = longitude / 15.0
    dayofyear = utc_naive.dayofyear
    eot_minutes = solarposition.equation_of_time_spencer71(dayofyear)
    local_solar = utc_naive + pd.Timedelta(hours=lmst_offset_hours) + pd.to_timedelta(eot_minutes, unit="m")
    return local_solar


def observation_stem_from_nc_path(path: Path) -> str | None:
    """
    Return ``YYYYMMDDHHMMSS`` (UTC) from filename, or None if not parseable.

    Prefer JAXA-style ``NC_H09_YYYYMMDD_HHMM_...``; else first substring of 14 digits that is a valid
    wall-clock timestamp in the file stem.
    """
    t = parse_time_from_nc_name(path.name)
    if t is not None:
        return t.strftime("%Y%m%d%H%M%S")
    stem = path.stem
    for m in _RE_14.finditer(stem):
        s14 = m.group(0)
        try:
            datetime.strptime(s14, "%Y%m%d%H%M%S")
            return s14
        except ValueError:
            continue
    return None


def iter_netcdf_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in _NETCDF_SUFFIXES:
            files.append(p)
    return files


def main() -> None:
    default_in = Path("/media/kyber/f166835a-9fa7-4966-844b-7eb1285d3654/himawari2025L2cloud_raw")
    default_out = Path("/media/kyber/f166835a-9fa7-4966-844b-7eb1285d3654/himawari2025L2cloud_all.zarr")

    parser = argparse.ArgumentParser(description="Crop L2 cloud NC to site ROI and write one aggregated .zarr")
    parser.add_argument(
        "--in-dir",
        type=Path,
        default=default_in,
        help=f"Root directory to scan for .nc / .nc4 (default: {default_in})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_out,
        help=f"Output Zarr store folder path (default: {default_out})",
    )
    parser.add_argument("--lat", type=float, required=True, help="Center latitude")
    parser.add_argument("--lon", type=float, required=True, help="Center longitude")
    parser.add_argument(
        "--size-km",
        type=float,
        default=500.0,
        help="Half-box size in km (passed to nc_crop_mask)",
    )
    parser.add_argument(
        "--resample-size",
        type=int,
        default=100,
        help="Output square side length (passed to nc_crop_mask)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip entire run if output Zarr store already exists",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Log progress every N files (default: 100)",
    )
    args = parser.parse_args()

    in_dir: Path = args.in_dir.resolve()
    if not in_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {in_dir}")

    lat = float(args.lat)
    lon = float(args.lon)

    paths = iter_netcdf_files(in_dir)
    if not paths:
        print(f"No .nc / .nc4 files under {in_dir}")
        return

    print(f"Found {len(paths)} NetCDF file(s) under {in_dir}")

    pairs: list[tuple[Path, str]] = []
    n_skipped_name = 0
    for p in paths:
        stem = observation_stem_from_nc_path(p)
        if stem is None:
            print(f"  skip (unparseable name): {p}")
            n_skipped_name += 1
            continue
        pairs.append((p, stem))

    if not pairs:
        print("No files with parseable observation time in filename.")
        return

    out_store = args.out_dir.resolve()
    out_store.parent.mkdir(parents=True, exist_ok=True)
    if args.skip_existing and out_store.exists():
        print(f"Output already exists, skip: {out_store}")
        return

    print(f"Using lat={lat}, lon={lon}, size_km={args.size_km}, resample_size={args.resample_size}")
    print(f"Output zarr store: {out_store}")

    n_written = 0
    n_skipped_bad = 0
    n_seen = 0
    failed_hdf: list[str] = []
    merged_list: list[np.ndarray] = []
    ts_datetime_list: list[datetime] = []

    for i, (nc_path, _) in enumerate(pairs, 1):
        n_seen += 1
        if i == 1 or i % args.log_every == 0:
            print(f"[{i}/{len(pairs)}] process: {nc_path.name}", flush=True)
        try:
            merged, ts_datetime, ts_string = nc_crop_mask(
                str(nc_path),
                lat0=lat,
                lon0=lon,
                size_km=args.size_km,
                resample_size=args.resample_size,
            )

            merged_list.append(merged.astype(np.float32))
            ts_datetime_list.append(ts_datetime)
            n_written += 1
        except OSError as e:
            print(f"         SKIP (HDF/NetCDF read error): {e}", flush=True)
            failed_hdf.append(str(nc_path))
            n_skipped_bad += 1
        except Exception as e:
            print(f"         SKIP ({type(e).__name__}): {e}", flush=True)
            n_skipped_bad += 1

    if not merged_list:
        print("No valid samples to write.")
        return

    merged_arr = np.stack(merged_list, axis=0).astype(np.float32)
    ts_index = pd.DatetimeIndex(ts_datetime_list)
    ts_datetime_utc = ts_index.to_numpy(dtype="datetime64[ns]")
    local_solar = utc_to_local_solar_time_pvlib(ts_index, lon)
    solpos = solarposition.get_solarposition(ts_index, lat, lon)
    azimuth_arr = solpos["azimuth"].to_numpy(dtype=np.float32)
    zenith_arr = solpos["zenith"].to_numpy(dtype=np.float32)
    day_of_year_arr = local_solar.dayofyear.to_numpy(dtype=np.int32)
    hour_of_day_arr = (
        local_solar.hour
        + local_solar.minute / 60.0
        + local_solar.second / 3600.0
    ).to_numpy(dtype=np.float32)
    local_solar_time_arr = local_solar.to_numpy(dtype="datetime64[ns]")

    ds_out = xr.Dataset(
        data_vars={
            "images": (("time_utc", "channel", "H", "W"), merged_arr.transpose(0, 3, 1, 2)),
            "zenith": (("time_utc",), zenith_arr),
            "azimuth": (("time_utc",), azimuth_arr),
            "day_of_year": (("time_utc",), day_of_year_arr),
            "hour_of_day": (("time_utc",), hour_of_day_arr),
            "local_solar_time": (("time_utc",), local_solar_time_arr),
        },
        coords={
            "time_utc": ts_datetime_utc,
            "channel": ["CLOT", "CLOT_MASK"],
        },
    )
    ds_out["time_utc"].attrs["timezone"] = "UTC+0"
    ds_out.attrs["latitude"] = lat
    ds_out.attrs["longitude"] = lon
    ds_out.to_zarr(str(out_store), mode="w")

    print(
        f"Done. seen={n_seen} written={n_written} skipped_bad={n_skipped_bad} "
        f"skipped_unparseable_name={n_skipped_name} output={out_store}"
    )
    if failed_hdf:
        print("HDF/NetCDF failed files:")
        for p in failed_hdf:
            print(f"  {p}")


if __name__ == "__main__":
    main()
