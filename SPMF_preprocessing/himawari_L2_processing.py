"""
Batch-crop Himawari L2 cloud NetCDF files with ``nc_crop_mask`` and save float32 arrays as ``.npy``.

Default input: ``/home/hw1/data/himawari2025L2cloud_raw``. All ``.npy`` go to one folder (default ``.../himawari2025L2cloud_npy``), same **basename** as the NetCDF file (only extension ``.npy``). Duplicate names under different subdirs overwrite—see scan warning. Unreadable or corrupt NC/HDF files are skipped with a log line. Site lat/lon from ``config/conf.yaml`` unless overridden.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from preprocessing.nc_processing import nc_crop_mask

_NETCDF_SUFFIXES = {".nc", ".nc4"}


def _load_site_lat_lon() -> tuple[float, float]:
    conf_path = _ROOT / "config" / "conf.yaml"
    with open(conf_path) as f:
        conf = yaml.safe_load(f)
    site = conf.get("site") or {}
    lat = site.get("latitude")
    lon = site.get("longitude")
    if lat is None or lon is None:
        raise KeyError("config/conf.yaml must define site.latitude and site.longitude")
    return float(lat), float(lon)


def iter_netcdf_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in _NETCDF_SUFFIXES:
            files.append(p)
    return files


def main() -> None:
    default_in = Path("/home/hw1/data/himawari2025L2cloud_raw")
    default_out = Path("/home/hw1/data/himawari2025L2cloud_npy")
    lat0, lon0 = _load_site_lat_lon()

    parser = argparse.ArgumentParser(description="Crop L2 cloud NC to site ROI and write .npy")
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
        help=f"Single directory for all .npy files (default: {default_out})",
    )
    parser.add_argument("--lat", type=float, default=None, help="Override center latitude")
    parser.add_argument("--lon", type=float, default=None, help="Override center longitude")
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
        help="Skip if target .npy already exists",
    )
    args = parser.parse_args()

    in_dir: Path = args.in_dir.resolve()
    if not in_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {in_dir}")

    lat = float(args.lat) if args.lat is not None else lat0
    lon = float(args.lon) if args.lon is not None else lon0

    paths = iter_netcdf_files(in_dir)
    if not paths:
        print(f"No .nc / .nc4 files under {in_dir}")
        return

    out_names = [p.with_suffix(".npy").name for p in paths]
    dup_counts = Counter(out_names)
    collisions = sorted(n for n, c in dup_counts.items() if c > 1)
    if collisions:
        print(
            "Warning: same output basename appears in multiple paths (later overwrites earlier): "
            + ", ".join(collisions[:20])
            + (" ..." if len(collisions) > 20 else "")
        )

    print(f"Found {len(paths)} NetCDF file(s) under {in_dir}")
    print(f"Using lat={lat}, lon={lon}, size_km={args.size_km}, resample_size={args.resample_size}")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    n_written = 0
    n_skipped_exist = 0
    n_skipped_bad = 0

    for i, nc_path in enumerate(paths, 1):
        out_path = out_dir / nc_path.with_suffix(".npy").name

        if args.skip_existing and out_path.is_file():
            print(f"[{i}/{len(paths)}] skip (exists): {out_path.name}")
            n_skipped_exist += 1
            continue

        print(f"[{i}/{len(paths)}] {nc_path.name} -> {out_path.name}")
        try:
            merged = nc_crop_mask(
                str(nc_path),
                lat0=lat,
                lon0=lon,
                size_km=args.size_km,
                resample_size=args.resample_size,
            )
            np.save(out_path, merged)
            n_written += 1
        except OSError as e:
            print(f"         SKIP (HDF/NetCDF read error): {e}")
            n_skipped_bad += 1
        except Exception as e:
            print(f"         SKIP ({type(e).__name__}): {e}")
            n_skipped_bad += 1

    print(
        f"Done. written={n_written} skipped_bad={n_skipped_bad} skipped_existing={n_skipped_exist}"
    )


if __name__ == "__main__":
    main()
