"""
One-shot converter: Folsom sky JPGs -> xarray Dataset -> Zarr archive.

This script mirrors the JPG scan/decode behavior of ``scripts/build_sky_zarr.py``,
but writes through xarray's ``Dataset.to_zarr`` path and includes split solar
feature variables and uses datetime64[ns] for time variables.

Output schema (xarray-native):
    <out>/
      time_utc (coord)       datetime64[ns], sorted ascending (treated as UTC)
      images                 uint8, shape (time_utc, channel, y, x)
      local_solar_time       datetime64[ns], shape (time_utc,)
      azimuth                float32, shape (time_utc,)
      zenith                 float32, shape (time_utc,)
      day_of_year            int16, shape (time_utc,)
      hour_of_day            float32, shape (time_utc,)

Example:
    python scripts/build_sky_xarray_zarr.py \
        --src "/data/folsom/sky_sample" \
        --out "/data/folsom/sky_sample_xr.zarr" \
        --spatial-size 224 \
        --chunk-frames 120 \
        --compressor zstd --clevel 3
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from numcodecs import Blosc
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.solar_encoder import compute_solar_features


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--src", type=Path, required=True, help="Directory containing YYYYMMDDHHMMSS.jpg files.")
    p.add_argument("--out", type=Path, required=True, help="Target .zarr path (will be created).")
    p.add_argument("--spatial-size", type=int, default=224, help="Resize JPGs to spatial_size x spatial_size.")
    p.add_argument(
        "--chunk-frames",
        type=int,
        default=120,
        help="Frames per Zarr chunk along the time axis.",
    )
    p.add_argument(
        "--compressor",
        choices=["zstd", "lz4", "zlib", "blosclz", "snappy", "none"],
        default="zstd",
        help="Blosc inner codec; 'none' disables compression.",
    )
    p.add_argument("--clevel", type=int, default=3, help="Compression level (1..9 typical).")
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only convert the first N JPGs (for quick tests).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete --out if it already exists.",
    )
    p.add_argument(
        "--site-config",
        type=Path,
        default=PROJECT_ROOT / "config" / "conf_folsom.yaml",
        help="YAML file used to read default site latitude/longitude.",
    )
    p.add_argument("--latitude", type=float, default=None, help="Site latitude. Defaults to site.latitude in --site-config.")
    p.add_argument(
        "--longitude",
        type=float,
        default=None,
        help="Site longitude. Defaults to site.longitude in --site-config.",
    )
    return p.parse_args()


def _scan_jpgs(src: Path) -> tuple[list[pd.Timestamp], list[Path]]:
    """Only keep YYYYMMDDHHMMSS.jpg files and return them sorted by time."""
    if not src.is_dir():
        raise FileNotFoundError(f"--src is not a directory: {src}")
    times: list[pd.Timestamp] = []
    paths: list[Path] = []
    for p in src.iterdir():
        if not p.is_file() or p.suffix.lower() != ".jpg":
            continue
        stem = p.stem.strip()
        if len(stem) != 14 or not stem.isdigit():
            continue
        try:
            t = pd.to_datetime(stem, format="%Y%m%d%H%M%S", errors="raise")
        except Exception:
            continue
        times.append(pd.Timestamp(t))
        paths.append(p)
    if not times:
        raise RuntimeError(f"no YYYYMMDDHHMMSS.jpg files found under {src}")
    order = np.argsort(np.asarray([t.value for t in times], dtype=np.int64), kind="mergesort")
    times = [times[int(i)] for i in order]
    paths = [paths[int(i)] for i in order]
    return times, paths


def _resize_jpg_to_chw_uint8(path: Path, spatial: int) -> np.ndarray:
    """Decode + resize one JPG; returns (3, H, W) uint8 or zeros on failure."""
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS
    try:
        with Image.open(path) as im:
            im = im.convert("RGB").resize((spatial, spatial), resample)
            arr = np.asarray(im, dtype=np.uint8)
        return np.transpose(arr, (2, 0, 1)).copy()
    except Exception:
        return np.zeros((3, spatial, spatial), dtype=np.uint8)


def _build_compressor(name: str, clevel: int):
    if name == "none":
        return None
    return Blosc(cname=name, clevel=int(clevel), shuffle=Blosc.BITSHUFFLE)


def _resolve_site_latlon(args: argparse.Namespace) -> tuple[float, float]:
    if args.latitude is not None and args.longitude is not None:
        return float(args.latitude), float(args.longitude)
    cfg_path = args.site_config.expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"site config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        conf = yaml.safe_load(f) or {}
    site = conf.get("site", {})
    lat = args.latitude if args.latitude is not None else site.get("latitude")
    lon = args.longitude if args.longitude is not None else site.get("longitude")
    if lat is None or lon is None:
        raise ValueError(
            "could not resolve latitude/longitude; pass --latitude/--longitude "
            f"or define site.latitude/site.longitude in {cfg_path}"
        )
    return float(lat), float(lon)


def main() -> int:
    args = _parse_args()
    src: Path = args.src.expanduser().resolve()
    out: Path = args.out.expanduser().resolve()
    spatial = int(args.spatial_size)
    chunk_frames = int(args.chunk_frames)
    if spatial < 1:
        print("--spatial-size must be >= 1", file=sys.stderr)
        return 2
    if chunk_frames < 1:
        print("--chunk-frames must be >= 1", file=sys.stderr)
        return 2

    latitude, longitude = _resolve_site_latlon(args)

    if out.exists():
        if not args.overwrite:
            print(f"--out already exists: {out}\n  pass --overwrite to replace.", file=sys.stderr)
            return 2
        shutil.rmtree(out)

    print(f"[build_sky_xarray_zarr] scanning {src} ...")
    times, paths = _scan_jpgs(src)
    n_total = len(times)
    if args.limit and args.limit > 0:
        n_total = min(n_total, int(args.limit))
        times = times[:n_total]
        paths = paths[:n_total]
    print(
        f"[build_sky_xarray_zarr] {n_total:,} JPGs to convert; "
        f"spatial={spatial} chunk_frames={chunk_frames} "
        f"lat={latitude:.6f} lon={longitude:.6f}"
    )

    print("[build_sky_xarray_zarr] preparing time + solar feature arrays ...")
    t0 = time.monotonic()
    next_log = t0 + 5.0

    time_coord = pd.DatetimeIndex(times).astype("datetime64[ns]")
    solar_features = compute_solar_features(time_coord, latitude, longitude)
    local_solar_time = np.asarray(
        [pd.Timestamp(row["local_solar_time"]).to_datetime64() for row in solar_features],
        dtype="datetime64[ns]",
    )
    azimuth = np.asarray([row["azimuth"] for row in solar_features], dtype=np.float32)
    zenith = np.asarray([row["zenith"] for row in solar_features], dtype=np.float32)
    day_of_year = np.asarray([row["day_of_year"] for row in solar_features], dtype=np.int16)
    hour_of_day = np.asarray([row["hour_of_day"] for row in solar_features], dtype=np.float32)

    compressor = _build_compressor(args.compressor, args.clevel)
    one_d_chunk = min(n_total, 1 << 16)
    encoding = {
        "images": {
            "dtype": "uint8",
            "chunks": (chunk_frames, 3, spatial, spatial),
            "compressor": compressor,
        },
        "azimuth": {
            "dtype": "float32",
            "chunks": (one_d_chunk,),
        },
        "zenith": {
            "dtype": "float32",
            "chunks": (one_d_chunk,),
        },
        "day_of_year": {
            "dtype": "int16",
            "chunks": (one_d_chunk,),
        },
        "hour_of_day": {
            "dtype": "float32",
            "chunks": (one_d_chunk,),
        },
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    print("[build_sky_xarray_zarr] phase A: streaming images + numeric vars to zarr ...")
    block = np.empty((chunk_frames, 3, spatial, spatial), dtype=np.uint8)
    written = 0
    first_chunk = True
    for chunk_start in range(0, n_total, chunk_frames):
        chunk_end = min(chunk_start + chunk_frames, n_total)
        cur_size = chunk_end - chunk_start
        for j in range(cur_size):
            block[j] = _resize_jpg_to_chw_uint8(paths[chunk_start + j], spatial)

        chunk_ds = xr.Dataset(
            data_vars={
                "images": (
                    ("time_utc", "channel", "y", "x"),
                    block[:cur_size],
                ),
                "azimuth": (("time_utc",), azimuth[chunk_start:chunk_end]),
                "zenith": (("time_utc",), zenith[chunk_start:chunk_end]),
                "day_of_year": (("time_utc",), day_of_year[chunk_start:chunk_end]),
                "hour_of_day": (("time_utc",), hour_of_day[chunk_start:chunk_end]),
            },
            coords={
                "time_utc": time_coord[chunk_start:chunk_end],
            },
        )
        if first_chunk:
            chunk_ds.attrs = {
                "spatial_size": spatial,
                "chunk_frames": chunk_frames,
                "compressor": args.compressor,
                "clevel": int(args.clevel),
                "source_dir": str(src),
                "time_convention": "UTC (stored as plain datetime64[ns] values)",
                "local_solar_time_note": "Apparent local solar time; not UTC wall-clock.",
                "latitude": float(latitude),
                "longitude": float(longitude),
                "build_time_utc": pd.Timestamp.utcnow().isoformat(),
            }
            chunk_ds.to_zarr(str(out), mode="w", encoding=encoding)
            first_chunk = False
        else:
            chunk_ds.to_zarr(str(out), mode="a", append_dim="time_utc")

        written = chunk_end
        now = time.monotonic()
        if now >= next_log or chunk_end == n_total:
            elapsed = now - t0
            rate = written / elapsed if elapsed > 0 else 0.0
            eta = (n_total - written) / rate if rate > 0 else 0.0
            print(
                f"  ... {written:,}/{n_total:,} frames  "
                f"({rate:6.1f} img/s, elapsed {elapsed:5.1f}s, eta {eta:5.1f}s)"
            )
            next_log = now + 5.0

    print("[build_sky_xarray_zarr] phase B: writing datetime var local_solar_time ...")
    datetime_ds = xr.Dataset(
        data_vars={
            "local_solar_time": (("time_utc",), local_solar_time),
        },
        coords={
            "time_utc": time_coord,
        },
    )
    datetime_ds.to_zarr(
        str(out),
        mode="a",
        encoding={
            "local_solar_time": {
                "chunks": (one_d_chunk,),
            }
        },
    )

    elapsed = time.monotonic() - t0
    print(
        f"[build_sky_xarray_zarr] done: {n_total:,} frames in {elapsed:.1f}s "
        f"-> {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
