#!/usr/bin/env python3
"""
The function create_himawari_tensor() is used to create a tensor from the himawari real time downlading
data folder. The input is the path to the himawari real time downlading data folder, the latitude and 
longitude of the center of the crop, the time span in minutes and the number of frames to include in 
the tensor. The output is the latest timestamp and the tensor. The tensor size is 
(N, 3, resample_size, resample_size). The second channel is the pixel level mask, and the third channel 
is frame level mask.

Read .nc files from the himawari data folder and return the latest timestamp and tensor
from the filenames (e.g. NC_H09_YYYYMMDD_HHMM_L2CLP010_FLDK.*.nc).
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, List, Any

# Ensure this directory is on the path so nc_processing can be found when run from project root
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from nc_processing import nc_crop_mask

import numpy as np
import matplotlib.pyplot as plt

def parse_time_from_nc_name(fname: str) -> Optional[datetime]:
    """Extract timestamp from NC filename (parts 2 and 3 = YYYYMMDD, HHMM)."""
    try:
        parts = Path(fname).stem.split("_")
        if len(parts) < 4:
            return None
        dt_str = parts[2] + parts[3]  # YYYYMMDDHHMM
        return datetime.strptime(dt_str, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    except (ValueError, IndexError):
        return None


def select_files(
    data_dir: Path, time_span: int, frame_num: int
) -> Tuple[Optional[datetime], List[Optional[str]]]:
    """
    Scan data_dir for .nc files. Build time slots at time_span-minute steps back from the
    latest down to the oldest file. For each slot, use the file name if present, else None.
    Return (latest timestamp, last frame_num slots in ascending time order).
    time_span unit: minutes.
    """
    if not data_dir.is_dir():
        return None, []
    # Collect (timestamp, file_name) for each matching .nc file
    timed_files = []
    for path in data_dir.iterdir():
        if path.is_file() and path.suffix.lower() == ".nc" and "FLDK" in path.name:
            t = parse_time_from_nc_name(path.name)
            if t is not None:
                timed_files.append((t, path.name))
    if not timed_files:
        return None, []
    # Sort by timestamp (oldest first)
    timed_files.sort(key=lambda x: x[0])
    oldest = timed_files[0][0]
    latest = timed_files[-1][0]
    time_to_name = {t: name for t, name in timed_files}
    # Build slot times: latest, latest-time_span, ... until we reach or pass oldest
    slot_times = []
    t = latest
    while t >= oldest:
        slot_times.append(t)
        t = t - timedelta(minutes=time_span)
    slot_times.sort()
    # Fill filename or None for each slot
    selected = [time_to_name.get(slot_t) for slot_t in slot_times]
    # Keep only the last frame_num frames (most recent)
    selected = selected[-frame_num:] if frame_num > 0 else []
    return latest, selected


def get_crop_bounds(lat0, lon0, size_km):
    dlat = size_km / 111 / 2
    dlon = size_km / (111 * np.cos(np.deg2rad(lat0))) / 2

    lat_min = lat0 - dlat
    lat_max = lat0 + dlat
    lon_min = lon0 - dlon
    lon_max = lon0 + dlon

    return lat_min, lat_max, lon_min, lon_max   

def nc_files_to_tensor(
    data_dir: Path, file_names: List[Optional[str]], lat0: float, lon0: float, size_km: float = 500, resample_size: int = 100
) -> np.ndarray:
    """
    Read each .nc file by file_name from data_dir; extract CLOT and merge to 3-channel, resample.
    Returns array of shape (N, 3, resample_size, resample_size). Missing frames are zeros.
    """
    lat_min, lat_max, lon_min, lon_max = get_crop_bounds(lat0, lon0, size_km=500)

    result = []
    for file_name in file_names:
        if file_name is None:
            result.append(np.zeros((resample_size, resample_size, 3), dtype=np.float32))
            continue
        path = data_dir / file_name
        if not path.is_file():
            result.append(np.zeros((resample_size, resample_size, 3), dtype=np.float32))
            continue
        try:
            merged = nc_crop_mask(path, lat0, lon0, size_km, resample_size)
            result.append(merged)
        except Exception:
            result.append(np.zeros((resample_size, resample_size, 3), dtype=np.float32))

    stacked = np.stack(result, axis=0) if result else np.zeros((0, resample_size, resample_size, 3), dtype=np.float32)
    tensor_np = np.transpose(stacked, (0, 3, 1, 2))  # (N, resample_size, resample_size, 3) -> (N, 3, resample_size, resample_size)
    return tensor_np

def create_himawari_tensor(path: Path, lat0: float, lon0: float, time_span: int = 10, frame_num: int = 12, size_km: float = 500, resample_size: int = 100):
    # Read .nc files from the Himawari real time downloading path and convert to tensor
    latest, file_names = select_files(path, time_span, frame_num)
    tensor_np = nc_files_to_tensor(path, file_names, lat0=lat0, lon0=lon0, size_km=size_km, resample_size=resample_size)
    return latest, tensor_np

if __name__ == "__main__":
    path = Path(os.path.expanduser("~/workspace/data/himawari"))
    latest, tensor_np = create_himawari_tensor(path, time_span=10, frame_num=12, lat0=34, lon0=112, size_km=500, resample_size=100)
    print(latest)
    print(tensor_np.shape)
