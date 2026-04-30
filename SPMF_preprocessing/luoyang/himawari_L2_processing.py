"""
Batch-crop Himawari L2 cloud NetCDF files with ``nc_crop_mask`` and save float32 arrays as ``.npy``.

Default input: ``/home/hw1/data/himawari2025L2cloud_raw``. All ``.npy`` go to one folder (default
``.../himawari2025L2cloud_npy``) as ``YYYYMMDDHHMMSS.npy``: observation time is parsed from the NetCDF
filename (standard ``NC_H09_YYYYMMDD_HHMM_...`` pattern, or the first valid 14-digit timestamp in the
stem). Minute-only names get ``SS=00``. Duplicate stems get ``_2``, ``_3``, … before ``.npy``.
Unreadable or corrupt NC/HDF files are skipped with a log line. Site lat/lon from ``config/conf.yaml``
unless overridden.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from preprocessing.himawari_infer import parse_time_from_nc_name
from preprocessing.nc_processing import nc_crop_mask

_NETCDF_SUFFIXES = {".nc", ".nc4"}
_RE_14 = re.compile(r"\d{14}")


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
    default_in = Path("/media/kyber/f166835a-9fa7-4966-844b-7eb1285d3654/himawari2025L2cloud_raw")
    default_out = Path("/media/kyber/f166835a-9fa7-4966-844b-7eb1285d3654/himawari2025L2cloud_npy")
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

    dup_counts = Counter(st for _, st in pairs)
    collisions = sorted(s for s, c in dup_counts.items() if c > 1)
    if collisions:
        print(
            "Note: duplicate observation stems (suffix _2, _3, … in output): "
            + ", ".join(collisions[:20])
            + (" ..." if len(collisions) > 20 else "")
        )

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    per_stem_idx: dict[str, int] = {}
    jobs: list[tuple[Path, Path]] = []
    for nc_path, stem in pairs:
        n = per_stem_idx.get(stem, 0)
        per_stem_idx[stem] = n + 1
        if n == 0:
            out_path = out_dir / f"{stem}.npy"
        else:
            out_path = out_dir / f"{stem}_{n + 1}.npy"
        jobs.append((nc_path, out_path))

    print(f"Using lat={lat}, lon={lon}, size_km={args.size_km}, resample_size={args.resample_size}")
    print(f"Output directory: {out_dir}")

    n_written = 0
    n_skipped_exist = 0
    n_skipped_bad = 0

    for i, (nc_path, out_path) in enumerate(jobs, 1):
        if args.skip_existing and out_path.is_file():
            print(f"[{i}/{len(jobs)}] skip (exists): {out_path.name}")
            n_skipped_exist += 1
            continue

        print(f"[{i}/{len(jobs)}] {nc_path.name} -> {out_path.name}")
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
        f"Done. written={n_written} skipped_bad={n_skipped_bad} "
        f"skipped_existing={n_skipped_exist} skipped_unparseable_name={n_skipped_name}"
    )


if __name__ == "__main__":
    main()
