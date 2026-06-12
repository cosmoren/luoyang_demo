"""
Stream-process NCEI GridSat-CONUS goes15 imagery into Folsom-centered float16 .npy shards.

For each 15-min UTC anchor, this driver:
  1. Picks the file URL closest to the anchor (via :mod:`gridsat_download`).
  2. Downloads it to a scratch dir (parallel, default 8 streams), retrying once on transient errors.
  3. Opens the netCDF, slices a 100x100 box centered on Folsom (CA) in lat/lon coords (resolved from
     each file's coordinate arrays - no hardcoded indices).
  4. Normalizes ch1/ch4/ch3 -> [0,1] with the agreed clip ranges, stacks (3, 100, 100) float16.
  5. Writes ``{out_root}/YYYY/MM/goes15_YYYYMMDD_HHMM.npy`` and deletes the raw .nc.
  6. Appends a per-frame manifest record with status, offset, per-channel stats, and clip rates.

Manifest is a JSONL at ``{out_root}/manifest_<start>_<end>.jsonl``. Every anchor produces exactly one
record; missing/error anchors are recorded with a status string (see ``Status`` constants).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests

# HDF5 / netCDF4 are not thread-safe; serialize all file opens with this lock.
_NC_LOCK = threading.Lock()

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from gridsat_download import AnchorMatch, build_anchor_matches  # noqa: E402

# ---------------------------------------------------------------------------- #
# constants

FOLSOM_LAT = 38.642
FOLSOM_LON = -121.148  # West, i.e. negative
CROP = 100  # H = W = 100

CH_VIS_RAW = "ch1"  # 0.65 um VIS reflectance, dimensionless
CH_IR_RAW = "ch4"  # 10.7 um IR window, Kelvin
CH_WV_RAW = "ch3"  # 6.5 um water vapor, Kelvin

VIS_CLIP = (0.0, 1.2)
IR_CLIP = (180.0, 330.0)
WV_CLIP = (190.0, 270.0)


class Status:
    OK = "ok"
    LISTING_MISS = "listing_miss"
    DOWNLOAD_ERROR = "download_error"
    PARSE_ERROR = "parse_error"
    EMPTY = "empty"  # ch1/3/4 all-fill / all-NaN


# ---------------------------------------------------------------------------- #
# helpers

def _shard_path(out_root: Path, anchor: datetime) -> Path:
    return out_root / f"{anchor:%Y}" / f"{anchor:%m}" / f"goes15_{anchor:%Y%m%d_%H%M}.npy"


def _normalize(channel: np.ndarray, lo: float, hi: float) -> tuple[np.ndarray, int, int]:
    """Clip to [lo, hi] and rescale to [0, 1] as float32. Returns (data, n_clipped_lo, n_clipped_hi).

    NaNs are filled with 0.0 after normalization (i.e. treated as the bottom of the range).
    """
    arr = channel.astype(np.float32, copy=True)
    nan_mask = np.isnan(arr)
    arr[nan_mask] = lo
    n_lo = int(np.sum(arr < lo))
    n_hi = int(np.sum(arr > hi))
    np.clip(arr, lo, hi, out=arr)
    arr -= lo
    arr /= (hi - lo)
    arr[nan_mask] = 0.0
    return arr, n_lo, n_hi


def _crop_indices(lat_arr: np.ndarray, lon_arr: np.ndarray) -> tuple[slice, slice, float, float]:
    """Return (lat_slice, lon_slice, picked_lat, picked_lon) for a 100x100 box centered on Folsom."""
    ilat = int(np.argmin(np.abs(lat_arr - FOLSOM_LAT)))
    ilon = int(np.argmin(np.abs(lon_arr - FOLSOM_LON)))
    half = CROP // 2
    lat_lo, lat_hi = ilat - half, ilat + half
    lon_lo, lon_hi = ilon - half, ilon + half
    if lat_lo < 0 or lat_hi > len(lat_arr) or lon_lo < 0 or lon_hi > len(lon_arr):
        raise ValueError(
            f"crop window out of bounds: lat[{lat_lo}:{lat_hi}] lon[{lon_lo}:{lon_hi}] "
            f"vs lat={len(lat_arr)} lon={len(lon_arr)}"
        )
    return slice(lat_lo, lat_hi), slice(lon_lo, lon_hi), float(lat_arr[ilat]), float(lon_arr[ilon])


def _http_download(session: requests.Session, url: str, dst: Path, timeout: float = 60.0) -> None:
    """Stream-download URL to dst. Atomic via temp file. Raises on HTTP errors."""
    tmp = dst.with_suffix(dst.suffix + ".part")
    with session.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
    tmp.rename(dst)


# ---------------------------------------------------------------------------- #
# per-task pipeline

@dataclass
class FrameRecord:
    anchor_iso: str
    status: str
    file_iso: str | None = None
    offset_min: float | None = None
    url: str | None = None
    out_path: str | None = None
    picked_lat: float | None = None
    picked_lon: float | None = None
    error: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__)


def _process_one(
    match: AnchorMatch,
    out_root: Path,
    scratch: Path,
    session: requests.Session,
    skip_existing: bool,
) -> FrameRecord:
    rec = FrameRecord(
        anchor_iso=match.anchor_iso,
        status=Status.OK,
        file_iso=match.file_iso,
        offset_min=match.offset_min,
        url=match.url,
    )

    if match.url is None:
        rec.status = Status.LISTING_MISS
        return rec

    anchor_dt = datetime.strptime(match.anchor_iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    out_path = _shard_path(out_root, anchor_dt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rec.out_path = str(out_path)

    if skip_existing and out_path.is_file():
        # Don't re-download. Still want stats? Keep cheap and skip; recompute by re-running without --skip-existing.
        rec.status = Status.OK
        rec.stats = {"reused_existing": True}
        return rec

    nc_path = scratch / f"goes15_{anchor_dt:%Y%m%d_%H%M}.nc"

    # download with one retry
    last_err: Exception | None = None
    for attempt in (1, 2):
        try:
            _http_download(session, match.url, nc_path, timeout=120.0)
            last_err = None
            break
        except Exception as e:  # network / HTTP / disk
            last_err = e
            if attempt == 1:
                time.sleep(1.5)
            else:
                rec.status = Status.DOWNLOAD_ERROR
                rec.error = f"{type(e).__name__}: {e}"
                if nc_path.exists():
                    try:
                        nc_path.unlink()
                    except OSError:
                        pass
                return rec

    # parse + crop + normalize (serialized: HDF5/netCDF4 isn't thread-safe)
    try:
        import xarray as xr  # local import keeps worker startup cheap

        with _NC_LOCK:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ds = xr.open_dataset(nc_path, decode_timedelta=False)
            try:
                lat_arr = ds["lat"].values.astype(np.float32)
                lon_arr = ds["lon"].values.astype(np.float32)
                lat_sl, lon_sl, picked_lat, picked_lon = _crop_indices(lat_arr, lon_arr)
                rec.picked_lat = picked_lat
                rec.picked_lon = picked_lon

                def _read_channel(name: str) -> np.ndarray:
                    da = ds[name]
                    # Most files are (time=1, lat, lon)
                    arr = da.isel(time=0).values if "time" in da.dims else da.values
                    arr = np.asarray(arr)[lat_sl, lon_sl]
                    return arr

                vis_raw = _read_channel(CH_VIS_RAW)  # ch1 -> channel 0
                ir_raw = _read_channel(CH_IR_RAW)   # ch4 -> channel 1
                wv_raw = _read_channel(CH_WV_RAW)   # ch3 -> channel 2
            finally:
                ds.close()
    except Exception as e:
        rec.status = Status.PARSE_ERROR
        rec.error = f"{type(e).__name__}: {e}"
        if nc_path.exists():
            try:
                nc_path.unlink()
            except OSError:
                pass
        return rec

    # detect empty (all-NaN OR completely uninformative i.e. variance == 0 across all 3 channels)
    is_empty = all(
        (np.isnan(a).all() or float(np.nanstd(a)) == 0.0)
        for a in (vis_raw, ir_raw, wv_raw)
    )

    vis_n, vis_lo, vis_hi = _normalize(vis_raw, *VIS_CLIP)
    ir_n, ir_lo, ir_hi = _normalize(ir_raw, *IR_CLIP)
    wv_n, wv_lo, wv_hi = _normalize(wv_raw, *WV_CLIP)

    stack = np.stack([vis_n, ir_n, wv_n], axis=0).astype(np.float16)
    assert stack.shape == (3, CROP, CROP), f"unexpected shape {stack.shape}"

    if is_empty:
        rec.status = Status.EMPTY
    else:
        rec.status = Status.OK

    # always save (even empties) - the report will flag them
    np.save(out_path, stack)

    rec.stats = {
        "shape": list(stack.shape),
        "dtype": str(stack.dtype),
        "ch0_raw": _summary(vis_raw),
        "ch1_raw": _summary(ir_raw),
        "ch2_raw": _summary(wv_raw),
        "ch0_norm": _summary(vis_n),
        "ch1_norm": _summary(ir_n),
        "ch2_norm": _summary(wv_n),
        "clip_counts": {
            "ch0_below_lo": vis_lo, "ch0_above_hi": vis_hi,
            "ch1_below_lo": ir_lo, "ch1_above_hi": ir_hi,
            "ch2_below_lo": wv_lo, "ch2_above_hi": wv_hi,
        },
    }

    if nc_path.exists():
        try:
            nc_path.unlink()
        except OSError:
            pass

    return rec


def _summary(arr: np.ndarray) -> dict[str, float]:
    a = arr
    if a.size == 0 or np.isnan(a).all():
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "median": float("nan")}
    return {
        "min": float(np.nanmin(a)),
        "max": float(np.nanmax(a)),
        "mean": float(np.nanmean(a)),
        "median": float(np.nanmedian(a)),
    }


# ---------------------------------------------------------------------------- #
# CLI

def main() -> None:
    p = argparse.ArgumentParser(description="Download + crop NCEI GridSat goes15 -> Folsom .npy shards")
    p.add_argument("--start", required=True, help="UTC start YYYY-MM-DD (inclusive)")
    p.add_argument("--end", required=True, help="UTC end YYYY-MM-DD (inclusive)")
    p.add_argument("--out-root", type=Path, default=Path("/work/folsom_dataset/sat_goes_gridsat"))
    p.add_argument("--scratch", type=Path, default=Path("/tmp/folsom_gridsat_dl"))
    p.add_argument("--manifest", type=Path, default=None, help="Optional JSONL manifest path")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--tolerance-min", type=float, default=7.0)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_excl = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)

    args.out_root.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)

    print(f"[gridsat] listing anchors {start.date()}..{end_excl.date()} (exclusive)")
    sess = requests.Session()
    matches = build_anchor_matches(start, end_excl, tolerance_min=args.tolerance_min, session=sess)
    n_anchors = len(matches)
    n_with_url = sum(1 for m in matches if m.url)
    print(f"[gridsat] {n_anchors} anchors total, {n_with_url} with a candidate file")

    manifest_path = args.manifest or (
        args.out_root / f"manifest_{start:%Y%m%d}_{(end_excl - timedelta(days=1)):%Y%m%d}.jsonl"
    )
    print(f"[gridsat] manifest -> {manifest_path}")

    t0 = time.time()
    n_done = 0
    n_ok = 0
    status_counts: dict[str, int] = {}
    with manifest_path.open("w") as fout, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(_process_one, m, args.out_root, args.scratch, sess, args.skip_existing): m
            for m in matches
        }
        for fut in as_completed(futs):
            try:
                rec = fut.result()
            except Exception as e:
                m = futs[fut]
                rec = FrameRecord(anchor_iso=m.anchor_iso, status="worker_crash",
                                  file_iso=m.file_iso, offset_min=m.offset_min, url=m.url,
                                  error=f"{type(e).__name__}: {e}")
            fout.write(rec.to_json() + "\n")
            fout.flush()
            n_done += 1
            status_counts[rec.status] = status_counts.get(rec.status, 0) + 1
            if rec.status == Status.OK:
                n_ok += 1
            if n_done % 50 == 0 or n_done == n_anchors:
                el = time.time() - t0
                rate = n_done / el if el > 0 else 0
                print(
                    f"[gridsat] {n_done}/{n_anchors}  ok={n_ok}  "
                    f"status={status_counts}  {rate:.1f} fps  {el:.0f}s",
                    flush=True,
                )

    el = time.time() - t0
    print(f"[gridsat] done in {el:.0f}s  ok={n_ok}/{n_anchors}  status_counts={status_counts}")
    print(f"[gridsat] manifest written: {manifest_path}")


if __name__ == "__main__":
    main()
