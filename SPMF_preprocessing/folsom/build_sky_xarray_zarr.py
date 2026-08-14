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
      zenith                 float32, shape (time_utc,)  [pvlib apparent_zenith via
                             compute_solar_features; key name stays ``zenith``]
      day_of_year            int16, shape (time_utc,)
      hour_of_day            float32, shape (time_utc,)
      image_valid            uint8, shape (time_utc,)  [required; --image-valid-csv]
      sun_u                  float32, shape (time_utc,) [required; --sun-centers-csv]
      sun_v                  float32, shape (time_utc,) [required; --sun-centers-csv]
      sun_valid              uint8, shape (time_utc,)  [required; --sun-centers-csv]

Solar geometry comes from ``modules.solar_encoder.compute_solar_features`` (apparent
zenith). Existing stores written before that switch may still hold geometric zenith
until regenerated.

Example:
    python scripts/build_sky_xarray_zarr.py \
        --src "/data/folsom/sky_sample" \
        --out "/data/folsom/sky_sample_xr.zarr" \
        --spatial-size 224 \
        --chunk-frames 120 \
        --compressor zstd --clevel 3 \
        --image-valid-csv "/data/folsom/image_valid.csv" \
        --sun-centers-csv "/data/folsom/sun_centers.csv"
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
    p.add_argument(
        "--image-valid-csv",
        type=Path,
        required=True,
        help=(
            "Required CSV with columns timestamp,mask (0/1). "
            "timestamp format YYYYMMDDHHMMSS matching JPG stems. "
            "Must cover exactly the same stamps as the image set (fail-fast). "
            "Always written as uint8 image_valid along time_utc."
        ),
    )
    p.add_argument(
        "--sun-centers-csv",
        type=Path,
        required=True,
        help=(
            "Required CSV with columns datetime_utc,above_horizon,u_sun,v_sun "
            "(frame_id ignored). datetime_utc like '2014-01-01 00:00:11' must "
            "match JPG stems exactly after parse (fail-fast). "
            "Writes float32 sun_u/sun_v and uint8 sun_valid along time_utc."
        ),
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


def _image_stamps_from_paths(paths: list[Path]) -> list[str]:
    """JPG stem stamps (YYYYMMDDHHMMSS) in image timeline order."""
    return [p.stem.strip() for p in paths]


def _load_image_valid_aligned(csv_path: Path, image_stamps: list[str]) -> np.ndarray:
    """
    Load CSV (timestamp, mask) and return uint8 image_valid aligned to image_stamps.

    Fail-fast: row count and stamp set must match the image list exactly.
    """
    csv_path = csv_path.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"--image-valid-csv not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"timestamp", "mask"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"--image-valid-csv missing required columns {sorted(missing_cols)}; "
            f"got columns {list(df.columns)}"
        )

    n_images = len(image_stamps)
    n_csv = len(df)
    if n_csv != n_images:
        raise ValueError(
            f"--image-valid-csv row count ({n_csv:,}) != image count ({n_images:,}); "
            "refusing to write image_valid with a partial join"
        )

    csv_stamps = df["timestamp"].astype(str).str.strip()
    bad = csv_stamps.map(lambda s: len(s) != 14 or not s.isdigit())
    if bool(bad.any()):
        examples = csv_stamps[bad].head(5).tolist()
        raise ValueError(
            f"--image-valid-csv timestamp must be YYYYMMDDHHMMSS (14 digits); "
            f"bad examples: {examples}"
        )

    csv_stamp_list = csv_stamps.tolist()
    image_set = set(image_stamps)
    csv_set = set(csv_stamp_list)
    if len(csv_stamp_list) != len(csv_set):
        raise ValueError(
            f"--image-valid-csv has duplicate timestamps "
            f"({n_csv:,} rows, {len(csv_set):,} unique)"
        )
    if len(image_stamps) != len(image_set):
        raise ValueError(
            f"image stamp list has duplicates "
            f"({n_images:,} images, {len(image_set):,} unique)"
        )
    if csv_set != image_set:
        only_csv = sorted(csv_set - image_set)[:10]
        only_img = sorted(image_set - csv_set)[:10]
        raise ValueError(
            "--image-valid-csv stamp set does not exactly match image stamps; "
            f"only_in_csv={only_csv} only_in_images={only_img}"
        )

    mask_vals = pd.to_numeric(df["mask"], errors="coerce")
    if mask_vals.isna().any():
        raise ValueError("--image-valid-csv mask column has non-numeric values")
    uniq = set(np.unique(mask_vals.to_numpy()))
    if not uniq.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"--image-valid-csv mask must be 0 or 1; got values {sorted(uniq)}")

    by_stamp = dict(zip(csv_stamp_list, mask_vals.astype(np.uint8).tolist()))
    return np.asarray([by_stamp[s] for s in image_stamps], dtype=np.uint8)


def _datetime_utc_to_stamps(series: pd.Series) -> list[str]:
    """Parse spaced UTC datetimes into YYYYMMDDHHMMSS compact stamps."""
    parsed = pd.to_datetime(series, errors="coerce", utc=False)
    if parsed.isna().any():
        bad = series[parsed.isna()].astype(str).head(5).tolist()
        raise ValueError(
            f"--sun-centers-csv datetime_utc has unparseable values; examples: {bad}"
        )
    # Naive wall times treated as UTC stamps matching JPG stems (no tz shift).
    return [pd.Timestamp(t).strftime("%Y%m%d%H%M%S") for t in parsed]


def _load_sun_centers_aligned(
    csv_path: Path, image_stamps: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load sun-centers CSV and return (sun_u, sun_v, sun_valid) aligned to image_stamps.

    Fail-fast: row count and stamp set must match the image list exactly.
    """
    csv_path = csv_path.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"--sun-centers-csv not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"datetime_utc", "above_horizon", "u_sun", "v_sun"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"--sun-centers-csv missing required columns {sorted(missing_cols)}; "
            f"got columns {list(df.columns)}"
        )

    n_images = len(image_stamps)
    n_csv = len(df)
    if n_csv != n_images:
        raise ValueError(
            f"--sun-centers-csv row count ({n_csv:,}) != image count ({n_images:,}); "
            "refusing to write sun_* with a partial join"
        )

    csv_stamp_list = _datetime_utc_to_stamps(df["datetime_utc"])
    image_set = set(image_stamps)
    csv_set = set(csv_stamp_list)
    if len(csv_stamp_list) != len(csv_set):
        raise ValueError(
            f"--sun-centers-csv has duplicate datetime stamps "
            f"({n_csv:,} rows, {len(csv_set):,} unique)"
        )
    if len(image_stamps) != len(image_set):
        raise ValueError(
            f"image stamp list has duplicates "
            f"({n_images:,} images, {len(image_set):,} unique)"
        )
    if csv_set != image_set:
        only_csv = sorted(csv_set - image_set)[:10]
        only_img = sorted(image_set - csv_set)[:10]
        raise ValueError(
            "--sun-centers-csv stamp set does not exactly match image stamps; "
            f"only_in_csv={only_csv} only_in_images={only_img}"
        )

    # above_horizon -> sun_valid uint8 0/1 (handles bool or True/False strings)
    ah = df["above_horizon"]
    if ah.dtype == object or pd.api.types.is_string_dtype(ah):
        ah_norm = ah.astype(str).str.strip().str.lower()
        mapping = {"true": 1, "false": 0, "1": 1, "0": 0}
        if not ah_norm.isin(mapping).all():
            bad = ah[~ah_norm.isin(mapping)].head(5).tolist()
            raise ValueError(
                f"--sun-centers-csv above_horizon must be True/False or 0/1; "
                f"bad examples: {bad}"
            )
        sun_valid_by_stamp = dict(
            zip(csv_stamp_list, ah_norm.map(mapping).astype(np.uint8).tolist())
        )
    else:
        ah_num = ah.astype(int)
        uniq = set(np.unique(ah_num.to_numpy()))
        if not uniq.issubset({0, 1}):
            raise ValueError(
                f"--sun-centers-csv above_horizon must be 0/1 or bool; got {sorted(uniq)}"
            )
        sun_valid_by_stamp = dict(zip(csv_stamp_list, ah_num.astype(np.uint8).tolist()))

    u_vals = pd.to_numeric(df["u_sun"], errors="coerce")
    v_vals = pd.to_numeric(df["v_sun"], errors="coerce")
    # Non-numeric garbage (not NaN) would already be coerced; allow NaN when sun invalid.
    sun_u_by_stamp = dict(zip(csv_stamp_list, u_vals.astype(np.float32).tolist()))
    sun_v_by_stamp = dict(zip(csv_stamp_list, v_vals.astype(np.float32).tolist()))

    sun_u = np.asarray([sun_u_by_stamp[s] for s in image_stamps], dtype=np.float32)
    sun_v = np.asarray([sun_v_by_stamp[s] for s in image_stamps], dtype=np.float32)
    sun_valid = np.asarray([sun_valid_by_stamp[s] for s in image_stamps], dtype=np.uint8)
    return sun_u, sun_v, sun_valid


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

    image_stamps = _image_stamps_from_paths(paths)
    print(f"[build_sky_xarray_zarr] loading image_valid from {args.image_valid_csv} ...")
    image_valid = _load_image_valid_aligned(args.image_valid_csv, image_stamps)
    n_ok = int(image_valid.sum())
    print(
        f"[build_sky_xarray_zarr] image_valid aligned: "
        f"{n_ok:,}/{n_total:,} valid (1), {n_total - n_ok:,} invalid (0)"
    )

    print(f"[build_sky_xarray_zarr] loading sun centers from {args.sun_centers_csv} ...")
    sun_u, sun_v, sun_valid = _load_sun_centers_aligned(args.sun_centers_csv, image_stamps)
    n_sun_ok = int(sun_valid.sum())
    print(
        f"[build_sky_xarray_zarr] sun centers aligned: "
        f"{n_sun_ok:,}/{n_total:,} above horizon (sun_valid=1)"
    )

    print("[build_sky_xarray_zarr] preparing time + solar feature arrays ...")
    t0 = time.monotonic()
    next_log = t0 + 5.0

    time_coord = pd.DatetimeIndex(times).astype("datetime64[ns]")
    solar_features = compute_solar_features(time_coord, latitude, longitude)
    local_solar_time = np.asarray(solar_features["local_solar_time"], dtype="datetime64[ns]")
    azimuth = np.asarray(solar_features["azimuth"], dtype=np.float32)
    zenith = np.asarray(solar_features["zenith"], dtype=np.float32)
    day_of_year = np.asarray(solar_features["day_of_year"], dtype=np.int16)
    hour_of_day = np.asarray(solar_features["hour_of_day"], dtype=np.float32)

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
        "image_valid": {
            "dtype": "uint8",
            "chunks": (one_d_chunk,),
        },
        "sun_u": {
            "dtype": "float32",
            "chunks": (one_d_chunk,),
        },
        "sun_v": {
            "dtype": "float32",
            "chunks": (one_d_chunk,),
        },
        "sun_valid": {
            "dtype": "uint8",
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
                "image_valid": (("time_utc",), image_valid[chunk_start:chunk_end]),
                "sun_u": (("time_utc",), sun_u[chunk_start:chunk_end]),
                "sun_v": (("time_utc",), sun_v[chunk_start:chunk_end]),
                "sun_valid": (("time_utc",), sun_valid[chunk_start:chunk_end]),
            },
            coords={
                "time_utc": time_coord[chunk_start:chunk_end],
            },
        )
        if first_chunk:
            chunk_ds["image_valid"].attrs = {
                "description": "Per-frame sky image validity flag from CSV mask column",
                "flag_values": "0,1",
                "flag_meanings": "invalid valid",
            }
            chunk_ds["sun_u"].attrs = {
                "description": "Projected sun center u (pixels) from sun-centers CSV",
            }
            chunk_ds["sun_v"].attrs = {
                "description": "Projected sun center v (pixels) from sun-centers CSV",
            }
            chunk_ds["sun_valid"].attrs = {
                "description": "Sun above-horizon flag from CSV above_horizon column",
                "flag_values": "0,1",
                "flag_meanings": "below_horizon above_horizon",
            }
            chunk_ds["zenith"].attrs = {
                "description": (
                    "Sun zenith from compute_solar_features "
                    "(pvlib apparent_zenith, degrees); variable name stays zenith"
                ),
                "units": "degrees",
            }
            chunk_ds["azimuth"].attrs = {
                "description": "Sun azimuth from compute_solar_features (degrees)",
                "units": "degrees",
            }
            chunk_ds.attrs = {
                "spatial_size": spatial,
                "chunk_frames": chunk_frames,
                "compressor": args.compressor,
                "clevel": int(args.clevel),
                "source_dir": str(src),
                "image_valid_csv": str(Path(args.image_valid_csv).expanduser().resolve()),
                "sun_centers_csv": str(Path(args.sun_centers_csv).expanduser().resolve()),
                "time_convention": "UTC (stored as plain datetime64[ns] values)",
                "local_solar_time_note": "Apparent local solar time; not UTC wall-clock.",
                "zenith_note": (
                    "Stored under name zenith; value is pvlib apparent_zenith "
                    "via modules.solar_encoder.compute_solar_features."
                ),
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