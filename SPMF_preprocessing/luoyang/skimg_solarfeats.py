"""
Convert UTC-timestamped sky images into a zarr dataset with:
1) `images`: RGB tensors with shape (time, 3, H, W)
2) `solarfeats`: [solar_zenith, solar_azimuth, day_of_year, hour_of_day]
3) `local_solar_time`: local apparent solar time in unix nanoseconds

Input image filenames must follow `yyyymmddhhmmss` (UTC+0), e.g. `20250101000000.jpg`.
All images are resized to `--resize-size H W` before writing.
If `--fill-missing` is enabled, missing 1-minute timestamps between min/max observed
time are filled with all-zero RGB frames.

Run examples:
  # Fill missing timestamps with zero frames (default behavior)
  micromamba run -n SimVP python SPMF_preprocessing/luoyang/empty_skimg_solarfeats.py \
    --input-dir /path/to/images \
    --output-zarr /path/to/output.zarr \
    --lat 34.68 --lon 112.45 \
    --resize-size 224 224 \
    --fill-missing

  # Keep only existing frames (no timestamp filling)
  micromamba run -n SimVP python SPMF_preprocessing/luoyang/empty_skimg_solarfeats.py \
    --input-dir /path/to/images \
    --output-zarr /path/to/output.zarr \
    --lat 34.68 --lon 112.45 \
    --resize-size 224 224 \
    --no-fill-missing
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image
from pvlib import solarposition

TIMESTAMP_RE = re.compile(r"^\d{14}$")
CHANNEL_NAMES = ["R", "G", "B"]
SOLAR_FEATURE_NAMES = [
    "solar_zenith",
    "solar_azimuth",
    "day_of_year",
    "hour_of_day",
]


def format_utc_ts(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S")


def utc_to_local_solar_time_pvlib(utc_times: pd.DatetimeIndex, longitude: float) -> pd.DatetimeIndex:
    """Convert UTC timestamps to local apparent solar time with pvlib equation-of-time."""
    if utc_times.tz is not None:
        utc_naive = utc_times.tz_convert("UTC").tz_localize(None)
    else:
        utc_naive = utc_times

    lmst_offset_hours = longitude / 15.0
    day_of_year = utc_naive.dayofyear
    eot_minutes = solarposition.equation_of_time_spencer71(day_of_year)
    return utc_naive + pd.Timedelta(hours=lmst_offset_hours) + pd.to_timedelta(eot_minutes, unit="m")


def parse_utc_from_stem(stem: str) -> pd.Timestamp:
    if not TIMESTAMP_RE.match(stem):
        raise ValueError(f"Invalid timestamp stem: {stem}")
    dt = pd.to_datetime(stem, format="%Y%m%d%H%M%S", utc=True)
    return pd.Timestamp(dt)


def collect_images(input_dir: Path) -> dict[pd.Timestamp, Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    ts_to_path: dict[pd.Timestamp, Path] = {}
    for p in sorted(input_dir.iterdir()):
        if not p.is_file():
            continue
        stem = p.stem.strip()
        if not TIMESTAMP_RE.match(stem):
            continue
        ts = parse_utc_from_stem(stem)
        if ts in ts_to_path:
            raise ValueError(
                f"Duplicate timestamp {stem} from files: {ts_to_path[ts].name} and {p.name}"
            )
        ts_to_path[ts] = p

    if not ts_to_path:
        raise ValueError(
            f"No image files with yyyymmddhhmmss stems found under {input_dir}"
        )
    return ts_to_path


def build_time_index(ts_to_path: dict[pd.Timestamp, Path], fill_missing: bool) -> pd.DatetimeIndex:
    observed = sorted(ts_to_path.keys())
    if fill_missing:
        start = observed[0]
        end = observed[-1]
        return pd.date_range(start=start, end=end, freq="1min", tz="UTC")
    return pd.DatetimeIndex(observed)


def read_rgb(path: Path, image_h: int, image_w: int) -> np.ndarray:
    img = np.asarray(Image.open(path))
    if img.ndim == 2:
        raise ValueError(f"Image must be 3-channel RGB, got grayscale: {path}")
    if img.ndim != 3:
        raise ValueError(f"Unsupported image ndim={img.ndim} for file: {path}")
    if img.shape[2] < 3:
        raise ValueError(f"Image must have at least 3 channels, got {img.shape[2]} for file: {path}")
    pil_rgb = Image.fromarray(img[:, :, :3]).convert("RGB")
    if pil_rgb.size != (image_w, image_h):
        pil_rgb = pil_rgb.resize((image_w, image_h), Image.BILINEAR)
    rgb = np.asarray(pil_rgb, dtype=np.uint8)
    return np.transpose(rgb, (2, 0, 1))


def compute_solarfeats(ts_chunk: pd.DatetimeIndex, lat: float, lon: float) -> tuple[np.ndarray, np.ndarray]:
    solpos = solarposition.get_solarposition(ts_chunk, latitude=lat, longitude=lon)
    zenith_arr = solpos["apparent_zenith"].to_numpy(dtype=np.float32)
    azimuth_arr = solpos["azimuth"].to_numpy(dtype=np.float32)

    local_solar_time = utc_to_local_solar_time_pvlib(ts_chunk, lon)
    local_solar_time_arr = local_solar_time.asi8.astype(np.int64)
    day_of_year_arr = local_solar_time.dayofyear.to_numpy(dtype=np.float32)
    hour_of_day_arr = (
        local_solar_time.hour
        + local_solar_time.minute / 60.0
        + local_solar_time.second / 3600.0
    ).to_numpy(dtype=np.float32)

    solarfeats = np.stack(
        [zenith_arr, azimuth_arr, day_of_year_arr, hour_of_day_arr],
        axis=1,
    ).astype(np.float32)
    return solarfeats, local_solar_time_arr


def write_zarr(
    ts_index_utc: pd.DatetimeIndex,
    ts_to_path: dict[pd.Timestamp, Path],
    out_store: Path,
    lat: float,
    lon: float,
    fill_missing: bool,
    chunk_size: int,
    image_h: int,
    image_w: int,
) -> None:
    out_store.parent.mkdir(parents=True, exist_ok=True)

    zero_rgb = np.zeros((3, image_h, image_w), dtype=np.uint8)
    total = len(ts_index_utc)
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        ts_chunk = ts_index_utc[start:stop]
        print(f"writing chunk {start}:{stop}/{total}")

        images = np.zeros((len(ts_chunk), 3, image_h, image_w), dtype=np.uint8)
        for i, ts in enumerate(ts_chunk):
            path = ts_to_path.get(pd.Timestamp(ts))
            if path is None:
                images[i] = zero_rgb
            else:
                images[i] = read_rgb(path, image_h=image_h, image_w=image_w)

        solarfeats_arr, local_solar_time_arr = compute_solarfeats(ts_chunk, lat, lon)
        ts_datetime_utc = ts_chunk.tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[ns]")

        ds_chunk = xr.Dataset(
            data_vars={
                "images": (("time_utc", "channel", "H", "W"), images),
                "solarfeats": (("time_utc", "solar_feature"), solarfeats_arr),
                "local_solar_time": (("time_utc",), local_solar_time_arr),
            },
            coords={
                "time_utc": ts_datetime_utc,
                "channel": CHANNEL_NAMES,
                "solar_feature": SOLAR_FEATURE_NAMES,
            },
        )
        ds_chunk["time_utc"].attrs["timezone"] = "UTC+0"
        ds_chunk["local_solar_time"].attrs["unit"] = "unix_ns"
        ds_chunk.attrs["latitude"] = float(lat)
        ds_chunk.attrs["longitude"] = float(lon)
        ds_chunk.attrs["fill_missing"] = bool(fill_missing)
        ds_chunk.attrs["image_shape"] = [3, image_h, image_w]
        ds_chunk.attrs["image_dtype"] = "uint8"

        if start == 0:
            ds_chunk.to_zarr(str(out_store), mode="w")
        else:
            ds_chunk.to_zarr(str(out_store), mode="a", append_dim="time_utc")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read UTC image files named yyyymmddhhmmss, convert to zarr with "
            "images and solarfeats, optionally filling missing 1-min frames with zeros."
        )
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing image files named yyyymmddhhmmss.* (UTC+0).",
    )
    p.add_argument(
        "--output-zarr",
        type=Path,
        required=True,
        help="Output zarr store path.",
    )
    p.add_argument(
        "--lat",
        type=float,
        default=34.68,
        help="Latitude used for solar geometry.",
    )
    p.add_argument(
        "--lon",
        type=float,
        default=112.45,
        help="Longitude used for solar geometry.",
    )
    p.add_argument(
        "--fill-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to fill missing 1-min timestamps with all-zero RGB frames.",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=30,
        help="Number of timesteps per zarr append chunk.",
    )
    p.add_argument(
        "--resize-size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=(224, 224),
        help="Resize all images to this size before writing (default: 224 224).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    image_h, image_w = args.resize_size
    if image_h <= 0 or image_w <= 0:
        raise ValueError(f"--resize-size must be positive, got: {image_h} {image_w}")

    ts_to_path = collect_images(args.input_dir.expanduser())
    ts_index_utc = build_time_index(ts_to_path, fill_missing=args.fill_missing)
    observed_count = len(ts_to_path)
    written_count = len(ts_index_utc)
    filled_count = max(0, written_count - observed_count)
    range_start = format_utc_ts(pd.Timestamp(ts_index_utc[0]))
    range_end = format_utc_ts(pd.Timestamp(ts_index_utc[-1]))

    print(
        f"found {observed_count} images, writing {written_count} frames "
        f"(fill_missing={args.fill_missing}, resize={image_h}x{image_w})"
    )
    write_zarr(
        ts_index_utc=ts_index_utc,
        ts_to_path=ts_to_path,
        out_store=args.output_zarr.expanduser(),
        lat=args.lat,
        lon=args.lon,
        fill_missing=args.fill_missing,
        chunk_size=args.chunk_size,
        image_h=image_h,
        image_w=image_w,
    )
    print("summary:")
    print(f"- coverage_utc: {range_start} -> {range_end}")
    print(f"- original_images: {observed_count}")
    print(f"- filled_images: {filled_count}")
    print(f"- total_written_frames: {written_count}")
    print(f"done: {args.output_zarr}")


if __name__ == "__main__":
    main()