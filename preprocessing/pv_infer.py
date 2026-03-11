from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd


def parse_time_from_csv_name(fname: str) -> Optional[datetime]:
    """Parse YYYYMMDD_HHMM from CSV filename (stem). Returns None if invalid."""
    try:
        stem = Path(fname).stem
        return datetime.strptime(stem, "%Y%m%d_%H%M")
    except (ValueError, TypeError):
        return None


def get_latest_pv_data(path: Path, num: int, time_span_minutes: int = 5) -> Tuple[List[datetime], List[Optional[Path]]]:
    """
    Scan path for CSV files named YYYYMMDD_HHMM.csv to find the latest time.
    Generate num timestamps at time_span_minutes (default 5) intervals, with the last one = latest_dt.
    Return (timestamps, paths). If a file for a slot does not exist, the corresponding path is None.
    If no latest file is found, return ([], []).
    """
    path = Path(path)
    if not path.is_dir():
        return [], []
    latest_dt = None
    for f in path.glob("*.csv"):
        dt = parse_time_from_csv_name(f.name)
        if dt is not None and (latest_dt is None or dt > latest_dt):
            latest_dt = dt
    if latest_dt is None:
        return [], []

    # num timestamps: latest_dt - (num-1)*span, ..., latest_dt - span, latest_dt
    span = timedelta(minutes=time_span_minutes)
    timestamps = [latest_dt - (num - 1 - i) * span for i in range(num)]
    paths: List[Optional[Path]] = []
    for t in timestamps:
        fpath = path / (t.strftime("%Y%m%d_%H%M") + ".csv")
        paths.append(fpath if fpath.is_file() else None)

    pv_dict = read_pv_csvs_to_dict(paths)

    return timestamps, paths, pv_dict


def read_pv_csvs_to_dict(paths: List[Optional[Path]]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Read each CSV in paths one by one. For each devDn, build arrays of inverter_state and active_power
    with one value per path slot. If paths[i] is None (file missing), use 0 for that slot.
    Return dict: devDn -> {"inverter_state": np.array, "active_power": np.array}, both shape (len(paths),).
    """
    n = len(paths)
    # Collect all devDn from all files
    all_dev_dns = set()
    file_data = []  # list of (devDn -> {inverter_state, active_power}) per file
    for p in paths:
        if p is None or not Path(p).is_file():
            file_data.append(None)
            continue
        df = pd.read_csv(p)
        if "devDn" not in df.columns:
            file_data.append(None)
            continue
        row_map = {}
        for _, row in df.iterrows():
            dev_dn = row.get("devDn")
            if pd.isna(dev_dn):
                continue
            dev_dn = str(dev_dn)
            all_dev_dns.add(dev_dn)
            row_map[dev_dn] = {
                "inverter_state": row.get("inverter_state"),
                "active_power": row.get("active_power"),
            }
        file_data.append(row_map)

    result: Dict[str, Dict[str, np.ndarray]] = {}
    for dev_dn in all_dev_dns:
        inv_arr = np.zeros(n, dtype=np.float64)
        pow_arr = np.zeros(n, dtype=np.float64)
        for i, data in enumerate(file_data):
            if data is None:
                inv_arr[i] = 0
                pow_arr[i] = 0
            elif dev_dn in data:
                v = data[dev_dn]
                inv_val = pd.to_numeric(v["inverter_state"], errors="coerce")
                pow_val = pd.to_numeric(v["active_power"], errors="coerce")
                inv_arr[i] = 0 if pd.isna(inv_val) else inv_val
                pow_arr[i] = 0 if pd.isna(pow_val) else pow_val
            else:
                inv_arr[i] = 0
                pow_arr[i] = 0
        result[dev_dn] = {"inverter_state": inv_arr, "active_power": pow_arr}
    return result

def create_pv_dataframe(path, num=12):
    timestamps, paths, pv_dict = get_latest_pv_data(path, num)   
    return timestamps, paths, pv_dict
