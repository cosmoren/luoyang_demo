"""
Data loader for luoyang_data (per-device CSVs in datasets/luoyang_data/).

Each CSV is one power station. For training, use iter_batches(): each iteration randomly
selects N stations and yields one (12×5) input and (192×5) output per station—no dataset
built in advance.

Each sample:
- Input: 12 consecutive rows at 5-min spacing; at least one row must have inverter_state (col G) == 512.
- Output: 192 rows at 15-min spacing (48 hours); at least one row must have inverter_state == 512.
- Continuity: the last of the 12 input rows is immediately followed by the first of the 192 output rows
  (no gap in the 5-min time series).
"""

import pickle
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from datetime import datetime
from scipy.ndimage import zoom

# Default path to luoyang_data directory (relative to project root)
DEFAULT_LUOYANG_DATA_DIR = Path(__file__).resolve().parent / "datasets" / "luoyang_data"

# Column name for inverter state (column G in the CSV)
INVERTER_STATE_COL = "inverter_state"
VALID_STATE = 512

# Default columns for input and output (stationCode and devDn omitted; saved as station_id in sample dict, format NE=XXXXXXX)
DEFAULT_COLUMNS = [
    "collectTime",
    "inverter_state",
    "active_power",
]

# Input: 12 rows, 5-min spacing
INPUT_LEN = 576
INPUT_INTERVAL_MIN = 5

# Output: 192 rows, 15-min spacing (48 hours)
OUTPUT_LEN = 192
OUTPUT_INTERVAL_MIN = 15
# In 5-min rows: one 15-min step = 3 rows
OUTPUT_STRIDE = OUTPUT_INTERVAL_MIN // INPUT_INTERVAL_MIN  # 3


def _is_valid_sample(
    df: pd.DataFrame,
    start_input: int,
    end_input: int,
    inverter_col: str = INVERTER_STATE_COL,
) -> bool:
    """True if input window and output window each have at least one inverter_state == 512."""
    inv = pd.to_numeric(df[inverter_col], errors="coerce").fillna(0).astype(int)
    valid = inv == VALID_STATE
    if not valid.iloc[start_input : end_input + 1].any():
        return False
    output_indices = [end_input + 1 + k * OUTPUT_STRIDE for k in range(OUTPUT_LEN)]
    if not valid.iloc[output_indices].any():
        return False
    return True


def _find_valid_sample_ranges(
    df: pd.DataFrame,
    inverter_col: str = INVERTER_STATE_COL,
) -> list[tuple[int, int]]:
    """
    Find all (start_input, end_input) index pairs that satisfy:
    - Input: 12 consecutive rows ending at end_input; at least one has inverter_state == 512.
    - Output: 192 rows at 15-min spacing starting at end_input+1; at least one has inverter_state == 512.
    - end_input + 1 + (OUTPUT_LEN - 1) * OUTPUT_STRIDE < len(df).

    Returns list of (start_input, end_input) where input = df.iloc[start_input:end_input+1] (12 rows), 
    output = df.iloc[end_input+1 : end_input+1 + OUTPUT_LEN*OUTPUT_STRIDE : OUTPUT_STRIDE].
    """
    n = len(df)
    min_end = INPUT_LEN - 1
    # Last output index = end_input + 1 + (OUTPUT_LEN-1)*OUTPUT_STRIDE must be <= n-1
    max_end = n - 2 - (OUTPUT_LEN - 1) * OUTPUT_STRIDE
    if max_end < min_end:
        return []

    inv = pd.to_numeric(df[inverter_col], errors="coerce").fillna(0).astype(int)
    valid_state = inv == VALID_STATE
    valid_arr = valid_state.values

    end_inputs = np.arange(min_end, max_end + 1, dtype=np.intp)
    output_indices_2d = end_inputs[:, None] + 1 + np.arange(OUTPUT_LEN, dtype=np.intp) * OUTPUT_STRIDE
    valid_output = valid_arr[output_indices_2d].any(axis=1)

    result = []
    for i, end_input in enumerate(end_inputs):
        if not valid_output[i]:
            continue
        start_input = end_input - INPUT_LEN + 1
        input_slice = slice(start_input, end_input + 1)
        if not valid_state.iloc[input_slice].any():
            continue
        result.append((start_input, end_input))
    return result


def load_csv(csv_path: Path | str) -> pd.DataFrame:
    """Load a single device CSV; ensure collectTime is parsed and sorted."""
    df = pd.read_csv(csv_path)
    if "collectTime" in df.columns:
        df["collectTime"] = pd.to_datetime(df["collectTime"])
        df = df.sort_values("collectTime").reset_index(drop=True)
    return df


def aggregate_all_to_csv(
    data_dir: Path | str | None = None,
    output_path: Path | str | None = None,
) -> Path:
    """
    Aggregate all CSV files in luoyang_data/ into a single CSV under datasets/.

    Reads each file in data_dir, adds a column 'station_id' from the filename (e.g. NE_346617581),
    concatenates, sorts by collectTime then station_id, and saves to output_path.
    """
    data_dir = Path(data_dir or DEFAULT_LUOYANG_DATA_DIR)
    if output_path is None:
        output_path = data_dir.parent / "luoyang_data_aggregated.csv"
    output_path = Path(output_path)

    frames: list[pd.DataFrame] = []
    for csv_path in sorted(data_dir.glob("*.csv")):
        df = load_csv(csv_path)
        df["station_id"] = csv_path.stem.replace("_", "=")  # NE_XXXXXXX -> NE=XXXXXXX (same as devDn format)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    out = pd.concat(frames, axis=0, ignore_index=True)
    out = out.sort_values(by=["collectTime", "station_id"]).reset_index(drop=True)
    out.to_csv(output_path, index=False)
    return output_path


def _dataframe_to_array(
    df: pd.DataFrame,
    columns: list[str],
) -> np.ndarray:
    """
    Convert a DataFrame slice to array (dtype object: strings + floats).
    - collectTime: kept as string (from CSV).
    - inverter_state, active_power: numeric as-is, float32.
    """
    out = []
    for c in columns:
        if c not in df.columns:
            continue
        if c == "collectTime":
            out.append(df["collectTime"].astype(str).to_numpy(dtype=object)[:, None])
        else:
            out.append(pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.float32).to_numpy()[:, None])
    if not out:
        return np.empty((len(df), 0), dtype=object)
    return np.hstack(out)


def _extract_one_sample(
    df: pd.DataFrame,
    start_input: int,
    end_input: int,
    cols_in: list[str],
    cols_out: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract one (12, 3) input and (192, 3) output from loaded df; returns (X, Y). collectTime is string."""
    cols_in_f = [c for c in cols_in if c in df.columns]
    cols_out_f = [c for c in cols_out if c in df.columns]
    X = _dataframe_to_array(df.iloc[start_input : end_input + 1], cols_in_f)
    output_indices = [end_input + 1 + k * OUTPUT_STRIDE for k in range(OUTPUT_LEN)]
    Y = _dataframe_to_array(df.iloc[output_indices], cols_out_f)
    return X, Y


def iter_batches(
    batch_size: int,
    data_dir: Path | str | None = None,
    input_columns: list[str] | None = None,
    output_columns: list[str] | None = None,
    inverter_col: str = INVERTER_STATE_COL,
    rng: np.random.Generator | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray, tuple[str, str, str, str] | None]]:
    """
    For training: each iteration randomly selects N stations (CSV files), reads only those N files,
    and yields one (12×5) input and (192×5) output per station. No full-dataset scan; only N CSVs
    are read per batch (e.g. N=5 → 5 files in memory).

    Yields (X_batch, Y_batch, time_strs) where:
    - X_batch: (N, 12, 3) dtype object — col 0 (collectTime) string; 1,2 (inverter_state, active_power) float32. station_id in meta.
    - Y_batch: (N, 192, 3) dtype object — same column layout
    - time_strs: (input_start, input_end, output_start, output_end) as strings from CSV for sample 0, or None
    """
    data_dir = Path(data_dir or DEFAULT_LUOYANG_DATA_DIR)
    if not data_dir.is_dir():
        return
    all_paths = sorted(data_dir.glob("*.csv"))
    if not all_paths:
        return
    rng = rng or np.random.default_rng()
    cols_in = input_columns if input_columns is not None else DEFAULT_COLUMNS
    cols_out = output_columns if output_columns is not None else DEFAULT_COLUMNS
    n_paths = len(all_paths)
    # Lazy cache: path -> list of (start, end); populated only when we load that path
    ranges_cache: dict[Path, list[tuple[int, int]]] = {}
    max_retries = max(20, batch_size * 3)  # cap retries when path has no valid samples

    while True:
        # Randomly select N path indices (with replacement so batch size is always N)
        chosen_idx = rng.integers(0, n_paths, size=batch_size)
        # Group by path so we read each file at most once per batch (only N files in memory)
        path_to_batch_positions: dict[int, list[int]] = {}
        for batch_pos, path_idx in enumerate(chosen_idx):
            path_to_batch_positions.setdefault(int(path_idx), []).append(batch_pos)

        batch_X: list[np.ndarray | None] = [None] * batch_size
        batch_Y: list[np.ndarray | None] = [None] * batch_size
        positions_filled = set()
        time_strs: tuple[str, str, str, str] | None = None  # (inp_start, inp_end, out_start, out_end) from CSV for sample 0

        for path_idx, positions in path_to_batch_positions.items():
            csv_path = all_paths[path_idx]
            df = load_csv(csv_path)
            if inverter_col not in df.columns:
                continue
            if csv_path not in ranges_cache:
                ranges_cache[csv_path] = _find_valid_sample_ranges(df, inverter_col=inverter_col)
            ranges = ranges_cache[csv_path]
            if not ranges:
                continue
            for batch_pos in positions:
                start_input, end_input = ranges[rng.integers(0, len(ranges))]
                if batch_pos == 0 and "collectTime" in df.columns:
                    out_last_idx = end_input + 1 + (OUTPUT_LEN - 1) * OUTPUT_STRIDE
                    tc = df["collectTime"]
                    time_strs = (
                        str(tc.iloc[start_input]),
                        str(tc.iloc[end_input]),
                        str(tc.iloc[end_input + 1]),
                        str(tc.iloc[out_last_idx]),
                    )
                X, Y = _extract_one_sample(df, start_input, end_input, cols_in, cols_out)
                batch_X[batch_pos] = X
                batch_Y[batch_pos] = Y
                positions_filled.add(batch_pos)

        # If some positions unfilled (path had no valid samples), retry with other random paths
        positions_unfilled = set(range(batch_size)) - positions_filled
        tries = 0
        while positions_unfilled and tries < max_retries:
            path_idx = int(rng.integers(0, n_paths))
            csv_path = all_paths[path_idx]
            df = load_csv(csv_path)
            if inverter_col not in df.columns:
                tries += 1
                continue
            if csv_path not in ranges_cache:
                ranges_cache[csv_path] = _find_valid_sample_ranges(df, inverter_col=inverter_col)
            ranges = ranges_cache[csv_path]
            if not ranges:
                tries += 1
                continue
            for batch_pos in list(positions_unfilled):
                start_input, end_input = ranges[rng.integers(0, len(ranges))]
                if time_strs is None and batch_pos == 0 and "collectTime" in df.columns:
                    out_last_idx = end_input + 1 + (OUTPUT_LEN - 1) * OUTPUT_STRIDE
                    tc = df["collectTime"]
                    time_strs = (
                        str(tc.iloc[start_input]),
                        str(tc.iloc[end_input]),
                        str(tc.iloc[end_input + 1]),
                        str(tc.iloc[out_last_idx]),
                    )
                X, Y = _extract_one_sample(df, start_input, end_input, cols_in, cols_out)
                batch_X[batch_pos] = X
                batch_Y[batch_pos] = Y
                positions_filled.add(batch_pos)
                positions_unfilled.discard(batch_pos)
            tries += 1

        # Fill any still-missing positions with first valid sample; if none valid, skip batch
        first_ok = next((i for i in range(batch_size) if batch_X[i] is not None), None)
        if first_ok is None:
            continue
        if any(x is None for x in batch_X):
            fill_x, fill_y = batch_X[first_ok], batch_Y[first_ok]
            for i in range(batch_size):
                if batch_X[i] is None:
                    batch_X[i], batch_Y[i] = fill_x.copy(), fill_y.copy()
        X_batch = np.stack(batch_X, axis=0)
        Y_batch = np.stack(batch_Y, axis=0)
        yield X_batch, Y_batch, time_strs


def build_train_test_splits(
    data_dir: Path | str | None = None,
    test_start: str = "2025-01-01 00:00:00",
    test_end: str = "2025-03-31 23:59:59.999999",
    max_train_per_file: int = 200,
    max_test_per_file: int = 200,
    split: str = "train",
    verbose: bool = True,
) -> list[dict]:
    """
    Go over all CSV files sequentially by time; classify each valid sample as training or testing.
    - Testing: all input and output collectTime in [test_start, test_end] (test_start default: 2025-01-01 00:00:00).
    - Training: otherwise (all before test_start or all after test_end).
    - Discard: if input or output collectTime spans March to April (has timestamps in both March and April).

    Does not pre-scan for valid samples. Uniformly samples max_train_per_file (2500) and
    max_test_per_file (3200) end_input indices per CSV; for each candidate, validates on
    extract (inverter_state); if invalid, skips. May yield fewer than 2500/3200 per file.
    Saves to two pkl files. Returns (n_train, n_test).
    """
    data_dir = Path(data_dir or DEFAULT_LUOYANG_DATA_DIR)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")
    t_start = pd.Timestamp(test_start)
    t_end = pd.Timestamp(test_end)

    train_list: list[dict] = []
    test_list: list[dict] = []
    cols_in = cols_out = DEFAULT_COLUMNS
    all_csvs = sorted(data_dir.glob("*.csv"))
    n_csvs = len(all_csvs)

    def spans_march_april(ts_min: pd.Timestamp, ts_max: pd.Timestamp) -> bool:
        months = {ts_min.month, ts_max.month}
        return 3 in months and 4 in months

    for file_idx, csv_path in enumerate(all_csvs):
        if verbose:
            print(f"  [{file_idx + 1}/{n_csvs}] {csv_path.name} ...", end=" ", flush=True)
        df = load_csv(csv_path)
        if INVERTER_STATE_COL not in df.columns:
            if verbose:
                print("skip (no inverter_state)")
            continue
        n = len(df)
        min_end = INPUT_LEN - 1
        # Last output index = end_input + 1 + (OUTPUT_LEN-1)*OUTPUT_STRIDE must be <= n-1
        max_end = n - 2 - (OUTPUT_LEN - 1) * OUTPUT_STRIDE
        if max_end < min_end:
            if verbose:
                print("skip (too short)")
            continue
        tc = df["collectTime"]

        # Uniformly sample candidate end_input indices (no validity pre-check)
        train_candidates = np.linspace(min_end, max_end, max_train_per_file, dtype=np.intp)
        test_candidates = np.linspace(min_end, max_end, max_test_per_file, dtype=np.intp)

        n_train_added, n_test_added = 0, 0
        
        if  split == "train":
            for end_input in train_candidates:
                start_input = end_input - INPUT_LEN + 1
                if not _is_valid_sample(df, start_input, end_input, INVERTER_STATE_COL):
                    continue
                input_times = tc.iloc[start_input : end_input + 1]
                output_indices = [end_input + 1 + k * OUTPUT_STRIDE for k in range(OUTPUT_LEN)]
                output_times = tc.iloc[output_indices]
                inp_min, inp_max = input_times.min(), input_times.max()
                out_min, out_max = output_times.min(), output_times.max()
                if spans_march_april(inp_min, inp_max) or spans_march_april(out_min, out_max):
                    continue
                in_window = (inp_min >= t_start and inp_max <= t_end and
                            out_min >= t_start and out_max <= t_end)
                if in_window:
                    continue  # training candidate must be outside test window
                X, Y = _extract_one_sample(df, start_input, end_input, cols_in, cols_out)
                station_code = str(df["stationCode"].iloc[start_input]) if "stationCode" in df.columns else ""
                train_list.append({
                    "X": X, "Y": Y,
                    "station_id": csv_path.stem.replace("_", "="),
                    "stationCode": station_code,
                    "start_input": start_input, "end_input": end_input,
                })

                n_train_added += 1

        elif split == "test":
            for end_input in test_candidates:
                start_input = end_input - INPUT_LEN + 1
                if not _is_valid_sample(df, start_input, end_input, INVERTER_STATE_COL):
                    continue
                input_times = tc.iloc[start_input : end_input + 1]
                output_indices = [end_input + 1 + k * OUTPUT_STRIDE for k in range(OUTPUT_LEN)]
                output_times = tc.iloc[output_indices]
                inp_min, inp_max = input_times.min(), input_times.max()
                out_min, out_max = output_times.min(), output_times.max()
                if spans_march_april(inp_min, inp_max) or spans_march_april(out_min, out_max):
                    continue
                in_window = (inp_min >= t_start and inp_max <= t_end and
                            out_min >= t_start and out_max <= t_end)
                if not in_window:
                    continue  # test candidate must be in test window
                X, Y = _extract_one_sample(df, start_input, end_input, cols_in, cols_out)
                station_code = str(df["stationCode"].iloc[start_input]) if "stationCode" in df.columns else ""
                test_list.append({
                    "X": X, "Y": Y,
                    "station_id": csv_path.stem.replace("_", "="),
                    "stationCode": station_code,
                    "start_input": start_input, "end_input": end_input,
                })
                n_test_added += 1
        
    if split == "train":
        return train_list
    elif split == "test":
        return test_list
    else:
        raise ValueError(f"Invalid split: {split}")

# ===== ADDED sat and NWP utils =====
def build_sat_index(sat_root):
    sat_root = Path(sat_root)
    sat_time_to_file = {}
    for npy_path in sat_root.rglob("*.npy"):
        name = npy_path.name
        # 假设格式：20250101_0000.npy
        try:
            t = pd.to_datetime(name.replace(".npy", ""), format="%Y%m%d_%H%M")
            sat_time_to_file[t] = str(npy_path)
        except Exception:
            continue
    sat_times = sorted(sat_time_to_file.keys())
    print(f"[SAT] indexed {len(sat_times)} npy files")
    return sat_time_to_file, sat_times

def build_sat_sequence(t0, sat_time_to_file, sat_times, T=10):
    X_sat_list = []
    X_sat_mask = []
    for k in range(T):
        tk = t0 - pd.Timedelta(minutes=15 * (T - 1 - k))
        sat_time = find_latest_sat_time(tk, sat_times)
        if sat_time is None:
            img = np.zeros((224, 224, 3), dtype=np.float32)
            found = 0
        else:
            path = sat_time_to_file.get(pd.Timestamp(sat_time), None)
            if path is None:
                img = np.zeros((224, 224, 3), dtype=np.float32)
                found = 0
            else:
                img, found = load_sat_frame(path)
        X_sat_list.append(img)
        X_sat_mask.append(found)

    return np.stack(X_sat_list), np.asarray(X_sat_mask)

def find_latest_sat_time(t0, sat_times):
    if len(sat_times) == 0:
        return None
    t0 = pd.Timestamp(t0).to_pydatetime()
    sat_times = [pd.Timestamp(t).to_pydatetime() for t in sat_times]
    idx = np.searchsorted(sat_times, t0, side="right") - 1
    if idx < 0:
        return None
    return sat_times[idx]

def load_sat_frame(path):
    try:
        img = np.load(path)  # already (H, W, 3)
        img = np.squeeze(img).astype(np.float32)
        # ===== ensure shape =====
        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(f"Unexpected shape: {img.shape}")
        # ===== resize to 224x224 if needed =====
        if img.shape[0] != 224 or img.shape[1] != 224:
            zoom_factors = (
                224 / img.shape[0],
                224 / img.shape[1],
                1,
            )
            img = zoom(img, zoom_factors, order=0)
        return img, 1
    except Exception:
        return np.zeros((224, 224, 3), dtype=np.float32), 0

def build_nwp_index(csv_path):
    # issue_time -> dataframe slice
    df = pd.read_csv(csv_path, parse_dates=["start_time", "forecast_time"])
    issue_map = {}
    issue_times = []
    for issue_time, group in df.groupby("start_time"):
        issue_map[issue_time] = group.sort_values("forecast_time")
        issue_times.append(issue_time)
    issue_times = sorted(issue_times)
    return issue_map, issue_times

# ===== build sat and nwp samples =====
def build_sat_nwp_sample(
    t0,
    sat_time_to_file,
    sat_times,
    solar_issue_map,
    solar_issue_times,
    wind_issue_map,
    wind_issue_times,
    horizon_steps=192,
):
    # ========= 统一时间 =========
    t0 = pd.Timestamp(t0)
    if t0.tzinfo is not None:
        t0 = t0.tz_convert(None)
    # ========= forecast timeline =========
    target_times = [
        t0 + pd.Timedelta(minutes=15 * (i + 1))
        for i in range(horizon_steps)
    ]
    target_times = pd.to_datetime(target_times)
    # ========= helper =========
    def find_latest_issue(issue_times, t0):
        issue_times = pd.to_datetime(issue_times)
        issue_times = [t.tz_localize(None) if t.tzinfo else t for t in issue_times]
        idx = np.searchsorted(issue_times, t0, side="right") - 1
        if idx < 0:
            return None
        return issue_times[idx]

    # ========= SOLAR =========
    solar_block = np.zeros((horizon_steps, 1), dtype=np.float32)  # ssrd
    if solar_issue_map is not None:
        issue_time = find_latest_issue(solar_issue_times, t0)
        if issue_time is not None:
            df = solar_issue_map[issue_time].copy()
            df["forecast_time"] = pd.to_datetime(df["forecast_time"])
            df["forecast_time"] = df["forecast_time"].dt.tz_localize(None)
            cols = []
            if "ssrd" in df.columns:
                cols.append("ssrd")
            df = df.set_index("forecast_time")[cols]
            full_index = df.index.union(target_times)
            df = df.reindex(full_index).sort_index()
            df = df.interpolate(method="time")
            max_gap = pd.Timedelta("1H")
            prev_valid = df.index.to_series().where(df.notna().any(axis=1)).ffill()
            next_valid = df.index.to_series().where(df.notna().any(axis=1)).bfill()
            gap = next_valid - prev_valid
            df = df.where(gap <= max_gap)
            df = df.loc[target_times]
            df = df.fillna(0.0)
            for i, name in enumerate(["ssrd"]):
                if name in df.columns:
                    solar_block[:, i] = df[name].values.astype(np.float32)
    # ========= WIND =========
    wind_block = np.zeros((horizon_steps, 5), dtype=np.float32)  # t2m, u10, v10, u100, v100
    if wind_issue_map is not None:
        issue_time = find_latest_issue(wind_issue_times, t0)
        if issue_time is not None:
            df = wind_issue_map[issue_time].copy()
            df["forecast_time"] = pd.to_datetime(df["forecast_time"])
            df["forecast_time"] = df["forecast_time"].dt.tz_localize(None)
            cols = []
            for c in ["t2m", "u10", "v10", "u100", "v100"]:
                if c in df.columns:
                    cols.append(c)
            df = df.set_index("forecast_time")[cols]
            full_index = df.index.union(target_times)
            df = df.reindex(full_index).sort_index()
            df = df.interpolate(method="time")
            max_gap = pd.Timedelta("1H")
            prev_valid = df.index.to_series().where(df.notna().any(axis=1)).ffill()
            next_valid = df.index.to_series().where(df.notna().any(axis=1)).bfill()
            gap = next_valid - prev_valid
            df = df.where(gap <= max_gap)
            df = df.loc[target_times]
            df = df.fillna(0.0)
            for i, name in enumerate(["t2m", "u10", "v10", "u100", "v100"]):
                if name in df.columns:
                    wind_block[:, i] = df[name].values.astype(np.float32)
    # ========= CONCAT =========
    X_fcst = np.concatenate([solar_block, wind_block], axis=1)  # [T, 6]
    X_fcst = np.transpose(X_fcst, (1, 0)) # [6, T]
    # ========= SAT =========
    X_sat, X_sat_mask = build_sat_sequence(
        t0,
        sat_time_to_file,
        sat_times
    )
    X_sat = np.transpose(X_sat, (0, 3, 1, 2)) # [12, 3, 224, 224]
    return X_sat, X_sat_mask, X_fcst

if __name__ == "__main__":
    # Build train/test splits and save to pkl; print progress and final counts
    data_dir = DEFAULT_LUOYANG_DATA_DIR
    train_pkl = data_dir.parent / "training_samples.pkl"
    test_pkl = data_dir.parent / "testing_samples.pkl"
    print("Building train/test splits ...")
    print(f"  Data dir:    {data_dir}")
    print(f"  Test window: 2025-01-01 00:00:00 to 2025-03-31 23:59:59")
    print(f"  Max training per CSV: 2500 (uniformly sampled if more)")
    print(f"  Max testing per CSV:  3200 (uniformly sampled if more)")
    print(f"  Train pkl:   {train_pkl}")
    print(f"  Test pkl:    {test_pkl}")
    print("Progress:")
    n_train, n_test = build_train_test_splits(
        data_dir=data_dir,
        train_pkl=train_pkl,
        test_pkl=test_pkl,
        verbose=True,
    )
    print("Done.")
    print(f"  Training samples: {n_train}  (saved to {train_pkl})")
    print(f"  Testing samples:  {n_test}   (saved to {test_pkl})")

    # # Training-style: iter_batches, no dataset built in advance
    # data_dir = DEFAULT_LUOYANG_DATA_DIR
    # print(f"Data dir: {data_dir}")
    # batch_iter = iter_batches(4, data_dir=data_dir, rng=np.random.default_rng(42))
    # for i, (X_batch, Y_batch, time_strs) in enumerate(batch_iter):
    #     if i >= 1:
    #         break
    #     print(f"  Batch {i}: X {X_batch.shape}, Y {Y_batch.shape} (one entry per station)")
    #     if time_strs is not None:
    #         inp_start, inp_end, out_start, out_end = time_strs
    #         print(f"    input  timestamps: {inp_start} -> {inp_end} (12 x 5min)")
    #         print(f"    output timestamps: {out_start} -> {out_end} (192 x 15min)")
    #     # Output input values for batch 1: 1x12x5 (12 rows, 5 values per row)
    #     inp = X_batch[0]  # (12, 3)
    #     print(f"  Input for batch 1 (1x12x5):")
    #     print(f"    cols: {DEFAULT_COLUMNS}")
    #     for row in range(inp.shape[0]):
    #         print(f"    row {row}: {list(inp[row])}")
    #     # Output first batch output entry: 1x192x5 (192 rows, 5 values per row), same format as input
    #     out = Y_batch[0]  # (192, 3)
    #     print(f"  Output for batch 1 (1x192x5):")
    #     print(f"    cols: {DEFAULT_COLUMNS}")
    #     for row in range(out.shape[0]):
    #         print(f"    row {row}: {list(out[row])}")
    # print("  (use iter_batches(batch_size, data_dir) for training; no preloading)")
