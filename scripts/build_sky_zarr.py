"""
One-shot converter: Folsom sky JPGs -> a single Zarr archive.

The output mirrors the layout the runtime reader expects
(:mod:`dataloader.folsom_sky_zarr`):

    <out>/
      images/        uint8, shape (N, 3, H, W), chunks (chunk_frames, 3, H, W)
      timestamps/    int64 ns, shape (N,), sorted ascending

The two knobs that matter most for RAM/throughput experiments are
``--chunk-frames`` (how many frames live in one compressed block) and the
compressor settings (``--compressor`` + ``--clevel``).

Example:
    python scripts/build_sky_zarr.py \
        --src "/data/folsom/sky_sample" \
        --out "/data/folsom/sky_sample.zarr" \
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
import zarr
from numcodecs import Blosc
from PIL import Image


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--src", type=Path, required=True, help="Directory containing YYYYMMDDHHMMSS.jpg files.")
    p.add_argument("--out", type=Path, required=True, help="Target .zarr path (will be created).")
    p.add_argument("--spatial-size", type=int, default=224, help="Resize JPGs to spatial_size x spatial_size.")
    p.add_argument(
        "--chunk-frames",
        type=int,
        default=120,
        help="Frames per Zarr chunk along the time axis (main RAM/throughput knob).",
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
    return p.parse_args()


def _scan_jpgs(src: Path) -> tuple[list[pd.Timestamp], list[Path]]:
    """Mirror dataloader.folsom._scan_sky_index: only YYYYMMDDHHMMSS.jpg files, sorted by time."""
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
    """Decode + resize a single JPEG; returns (3, H, W) uint8 or zeros on failure."""
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

    if out.exists():
        if not args.overwrite:
            print(f"--out already exists: {out}\n  pass --overwrite to replace.", file=sys.stderr)
            return 2
        shutil.rmtree(out)

    print(f"[build_sky_zarr] scanning {src} ...")
    times, paths = _scan_jpgs(src)
    n_total = len(times)
    if args.limit and args.limit > 0:
        n_total = min(n_total, int(args.limit))
        times = times[:n_total]
        paths = paths[:n_total]
    print(f"[build_sky_zarr] {n_total:,} JPGs to convert; spatial={spatial} chunk_frames={chunk_frames}")

    out.parent.mkdir(parents=True, exist_ok=True)
    grp = zarr.open_group(str(out), mode="w")
    compressor = _build_compressor(args.compressor, args.clevel)

    images = grp.create_dataset(
        "images",
        shape=(n_total, 3, spatial, spatial),
        chunks=(chunk_frames, 3, spatial, spatial),
        dtype="uint8",
        compressor=compressor,
    )
    ts_arr = grp.create_dataset(
        "timestamps",
        shape=(n_total,),
        chunks=(min(n_total, 1 << 16),),
        dtype="int64",
    )
    ts_arr[:] = np.asarray([int(t.value) for t in times], dtype=np.int64)

    grp.attrs["spatial_size"] = spatial
    grp.attrs["chunk_frames"] = chunk_frames
    grp.attrs["compressor"] = args.compressor
    grp.attrs["clevel"] = int(args.clevel)
    grp.attrs["source_dir"] = str(src)
    grp.attrs["build_time_utc"] = pd.Timestamp.utcnow().isoformat()

    print("[build_sky_zarr] writing image chunks ...")
    t0 = time.monotonic()
    next_log = t0 + 5.0
    block = np.empty((chunk_frames, 3, spatial, spatial), dtype=np.uint8)
    written = 0
    for chunk_start in range(0, n_total, chunk_frames):
        chunk_end = min(chunk_start + chunk_frames, n_total)
        cur_size = chunk_end - chunk_start
        for j in range(cur_size):
            block[j] = _resize_jpg_to_chw_uint8(paths[chunk_start + j], spatial)
        images[chunk_start:chunk_end] = block[:cur_size]
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

    elapsed = time.monotonic() - t0
    print(
        f"[build_sky_zarr] done: {n_total:,} frames in {elapsed:.1f}s "
        f"-> {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
