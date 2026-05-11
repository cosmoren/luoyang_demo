"""Luoyang PV dataset and loader utilities for training."""
import sys
from datetime import timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from pvlib import solarposition
from torch.utils.data import Dataset
from modules.solar_encoder import compute_solar_features, solar_features_encoder, delta_time_encoder, extract_solar_features
import time
import xarray as xr

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config_utils import get_resolved_paths

# --- PV per-station CSV helpers (5-minute rows, ``collectTime`` sorted) ---

INVERTER_STATE_COL = "inverter_state"
VALID_STATE = 512

def load_csv(csv_path: Path | str) -> pd.DataFrame:
    """Load a single device CSV; ensure collectTime is parsed and sorted."""
    df = pd.read_csv(csv_path)
    if "collectTime" in df.columns:
        df["collectTime"] = pd.to_datetime(df["collectTime"])
        df = df.sort_values("collectTime").reset_index(drop=True)
    return df


def list_csv_files(
    start_idx: int = 0,
    end_idx: int | None = None,
    data_dir: Path | str | None = None,
) -> list[Path]:
    """
    Sorted ``*.csv`` paths under ``data_dir``.

    - Default ``end_idx=None``: return all files from ``start_idx`` through the last file (any count).
    - If ``end_idx`` is an int, it is the **inclusive** last index in the sorted list (``start_idx`` … ``end_idx``).
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")
    all_csvs: list[Path] = sorted(data_dir.glob("*.csv"))
    n = len(all_csvs)
    if n == 0:
        return []
    if start_idx < 0:
        raise ValueError("start_idx must be >= 0")
    if start_idx >= n:
        raise ValueError(f"start_idx ({start_idx}) out of range (n={n})")
    if end_idx is None:
        return all_csvs[start_idx:]
    if end_idx < start_idx:
        raise ValueError("require start_idx <= end_idx")
    if end_idx >= n:
        raise ValueError(
            f"end_idx ({end_idx}) out of range (n={n}); valid inclusive indices are 0..{n - 1}"
        )
    return all_csvs[start_idx : end_idx + 1]


# ``nwp_interp`` columns: ssrd (solar) then wind fields, in this order.
_NWP_INTERP_STACK_COLS = ("ssrd", "msl", "t2m", "u10", "v10", "u100", "v100")
_NWP_SOLAR_INTERP_COLS = _NWP_INTERP_STACK_COLS[:1]
_NWP_WIND_INTERP_COLS = _NWP_INTERP_STACK_COLS[1:]
_NS_PER_HOUR = 3_600_000_000_000
_NS_PER_DAY = 86_400_000_000_000


def _sanitize_nwp_interp(nwp_interp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Replace non-finite values in ``nwp_interp`` (shape ``[T, 7]``) with 0.

    Returns ``(nwp_clean, nwp_mask)`` where ``nwp_mask`` has shape ``[T, 1]``, dtype float32:
    ``1.0`` if that timestep had any nan/inf (hence replacement), else ``0.0``.
    """
    x = np.asarray(nwp_interp, dtype=np.float64)
    bad = ~np.isfinite(x)
    x_clean = np.where(bad, 0.0, x).astype(np.float32, copy=False)
    row_bad = bad.any(axis=1).astype(np.float32).reshape(-1, 1)
    return x_clean, row_bad


def _precompute_nwp_blocks(df, value_cols):
    """Group an NWP CSV by ``start_time`` and pre-build per-column interpolation tables.

    Returns ``None`` if ``df`` is ``None``. Otherwise returns
    ``dict[start_ns:int -> dict[col:str -> (xp_ns:np.ndarray[float64], fp:np.ndarray[float64])]]``
    where each ``(xp_ns, fp)`` pair has been sorted by ``forecast_time`` (asc), de-duplicated
    keeping the last occurrence per ``forecast_time`` (matches ``drop_duplicates(keep="last")``),
    and stripped of NaN ``y`` rows. Done once at dataset construction so that per-sample
    ``interpolate_nwp_features`` calls become ``O(T)`` instead of ``O(N_rows * T)``.
    """
    if df is None:
        return None
    if "start_time" not in df.columns or "forecast_time" not in df.columns:
        return {}

    start_ns_all = pd.to_datetime(df["start_time"], utc=True).astype("int64").to_numpy()
    ft_ns_all = pd.to_datetime(df["forecast_time"], utc=True).astype("int64").to_numpy()
    n_total = start_ns_all.shape[0]
    if n_total == 0:
        return {}

    cols_all: dict[str, np.ndarray] = {}
    for col in value_cols:
        if col in df.columns:
            cols_all[col] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)

    order = np.argsort(start_ns_all, kind="stable")
    s_sorted = start_ns_all[order]
    unique_start, idx_first = np.unique(s_sorted, return_index=True)
    bounds = np.append(idx_first, n_total)

    blocks: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for i, st in enumerate(unique_start):
        rows = order[bounds[i]:bounds[i + 1]]
        ft_grp = ft_ns_all[rows]
        col_dict: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for col, arr in cols_all.items():
            y = arr[rows]
            valid = ~np.isnan(y)
            if not valid.any():
                col_dict[col] = (
                    np.empty(0, dtype=np.float64),
                    np.empty(0, dtype=np.float64),
                )
                continue
            ft_v = ft_grp[valid]
            y_v = y[valid]
            sort_idx = np.argsort(ft_v, kind="stable")
            ft_s = ft_v[sort_idx]
            y_s = y_v[sort_idx]
            keep = np.empty_like(ft_s, dtype=bool)
            keep[:-1] = ft_s[1:] != ft_s[:-1]
            keep[-1] = True
            col_dict[col] = (
                ft_s[keep].astype(np.float64),
                y_s[keep].astype(np.float64),
            )
        blocks[int(st)] = col_dict
    return blocks


def _interp_nwp_col(blocks, start_ns: int, col: str, xq_ns_f: np.ndarray) -> np.ndarray:
    """Linearly interpolate one column from a precomputed block; NaN-vector if missing."""
    n = xq_ns_f.shape[0]
    if blocks is None:
        return np.full(n, np.nan, dtype=np.float64)
    block = blocks.get(start_ns)
    if block is None:
        return np.full(n, np.nan, dtype=np.float64)
    pair = block.get(col)
    if pair is None or pair[0].size == 0:
        return np.full(n, np.nan, dtype=np.float64)
    xp_ns_f, fp_f = pair
    return np.interp(xq_ns_f, xp_ns_f, fp_f)

def interpolate_nwp_features(nwp_solar_blocks, nwp_wind_blocks, forecast_timestamps_utc, dhour=12):
    """
    Use ``forecast_timestamps_utc[0]`` minus ``dhour`` to get ``t_ref_utc``, then ``prev_noon_utc`` (last UTC
    noon before ``t_ref_utc``). Look up the matching solar/wind block (precomputed per ``start_time``), then
    linearly interpolate ``ssrd`` and ``msl``, ``t2m``, ``u10``, ``v10``, ``u100``, ``v100`` along ``forecast_time``
    onto every ``forecast_timestamps_utc`` (``np.nan`` if no data). Those series are stacked as ``nwp_interp``
    with shape ``[n, 7]`` (then nan/inf → 0 with ``_sanitize_nwp_interp``).

    ``nwp_solar_blocks`` / ``nwp_wind_blocks`` are produced by ``_precompute_nwp_blocks``. ``None`` means the
    corresponding NWP CSV was not loaded; in that case the function returns ``None`` (preserving original
    behavior). An empty dict ({}) means the CSV was loaded but had no rows / matching block — in that case
    the function still produces a [n, 8] zero+mask=1 array (preserving original "no match → all zeros" path).
    """
    if nwp_solar_blocks is None or nwp_wind_blocks is None:
        return None
    if not forecast_timestamps_utc:
        return None

    xq_ns_f = pd.DatetimeIndex(forecast_timestamps_utc).asi8.astype(np.float64)

    t0_ns = int(pd.Timestamp(forecast_timestamps_utc[0]).value)
    t_ref_ns = t0_ns - int(dhour) * _NS_PER_HOUR
    midnight_ns = (t_ref_ns // _NS_PER_DAY) * _NS_PER_DAY
    noon_ns = midnight_ns + 12 * _NS_PER_HOUR
    prev_noon_ns = noon_ns if noon_ns < t_ref_ns else (noon_ns - _NS_PER_DAY)

    ssrd_interp = _interp_nwp_col(nwp_solar_blocks, prev_noon_ns, "ssrd", xq_ns_f)
    wind_arrs = [
        _interp_nwp_col(nwp_wind_blocks, prev_noon_ns, c, xq_ns_f)
        for c in _NWP_WIND_INTERP_COLS
    ]

    nwp_interp = np.column_stack([ssrd_interp] + wind_arrs)
    nwp_interp_clean, nwp_mask = _sanitize_nwp_interp(nwp_interp)
    nwp_interp_wmask = np.concatenate([nwp_interp_clean, nwp_mask], axis=1)  # [T_out, 8]

    return nwp_interp_wmask


class PVDataset(Dataset):
    """
    **Chronological split (per CSV, same for all files sharing the reference row count):** row indices
    are partitioned in fixed proportions: first **60%** train, next **10%** validation, last **30%** test.
    Let ``split_train_end = int(n * 0.6)`` and ``split_val_end = int(n * 0.7)``. An anchor is used for
    **train** only if the full X+Y window lies in ``[0, split_train_end)`` (last Y row ``< split_train_end``);
    for **val** only if the window lies in ``[split_train_end, split_val_end)``;
    for **test** only if the window lies in ``[split_val_end, n)``.
    Sky/sat are still loaded by timestamp for each sample.

    The ``pv_train_time_fraction`` constructor argument is kept for call-site compatibility but **not** used
    for these boundaries (splits are fixed at 60% / 10% / 30%).

    Train: ``__len__`` = number of CSVs; ``__getitem__`` picks a **random** train-segment anchor where Y has at
    least one ``inverter_state == VALID_STATE``.

    Val (``split="val"``): anchors in the val row band are taken every ``val_anchor_stride_min`` (CSV row axis,
    no Y validity filter). Test (``split="test"``): same idea with ``test_anchor_stride_min``.
    ``__len__`` = ``len(sample_files) * num_{val|test}_windows``. For each window, the **reference** last-X
    ``collectTime`` from the first CSV is stored; each inverter CSV resolves the row index that matches that
    time (within ``test_collect_time_match_tolerance_min``) so all devices share the same wall times.

    Sky images (under ``data_dir`` / ``paths.sky_image_path``, ``YYYYMMDDHHMMSS_12.jpg``): history sequence
    ends at the last X ``collectTime``; forecast sequence starts at the first Y ``collectTime``;
    step size is ``skyimg_time_resolution_min`` (independent of PV CSV spacing).
    Missing files → black ``(3, H, W)`` tensors resized to ``training.skyimg_spatial_size``.

    All timestamps in this dataloader are treated as **UTC**: PV CSV ``collectTime`` rows, sky JPEG
    filenames (``YYYYMMDDHHMMSS_12.jpg``), and Himawari NPY filenames
    (``NC_H09_YYYYMMDD_HHMM_L2CLP010_FLDK.02401_02401.npy``) all use UTC wall time. ``collectTime``
    is floored to 10-minute boundaries for the satellite key timestep. History/forecast windows step
    in UTC (sizes from constructor kwargs). Sat arrays are float32 HWC per ``satimg_npy_shape_hwc``;
    missing files → zeros.

    The ``config_path`` argument is the dataset YAML for this instance (e.g.
    ``config/datasets/conf_luoyang.yaml``). It supplies ``paths.data_dir``,
    ``paths.pv_device_path`` (PV device Excel), and (optionally) ``paths.nwp_path`` used to load
    ``solar.csv`` / ``wind.csv``. Site coordinates are **not** taken from this YAML; they are
    read from ``<data_dir>/info.yaml`` (``site.latitude`` / ``site.longitude``) so that the
    same dataset folder yields the same coordinates regardless of which config selects it.

    If ``paths.nwp_path`` is set (relative to ``paths.data_dir`` unless absolute) and ``solar.csv`` /
    ``wind.csv`` exist and load successfully, they populate ``nwp_solar_df`` and ``nwp_wind_df``;
    otherwise (missing key, missing ``data_dir``, missing files, or read error) both stay ``None``
    without raising.
    """

    def __init__(
        self,
        config_path: str | Path,
        pv_dir: str,
        skyimg_dir: str,
        satimg_dir: str,
        *,
        split: str,
        csv_interval_min: int,
        pv_input_interval_min: int,
        pv_input_len: int,
        pv_output_interval_min: int,
        pv_output_len: int,
        pv_train_time_fraction: float,
        test_anchor_stride_min: int,
        val_anchor_stride_min: int,
        test_collect_time_match_tolerance_min: int,
        skyimg_window_size: int,
        skyimg_time_resolution_min: int,
        skyimg_spatial_size: int,
        satimg_window_size: int,
        satimg_time_resolution_min: int,
        satimg_npy_shape_hwc: tuple[int, int, int],
    ):
        self._config_path = Path(config_path).resolve()
        if not self._config_path.is_file():
            raise FileNotFoundError(f"PVDataset config_path not found: {self._config_path}")
        if split not in ("train", "val", "test"):
            raise ValueError("split must be 'train', 'val', or 'test'")
        self.split = split
        if skyimg_window_size < 1:
            raise ValueError("skyimg_window_size must be >= 1")
        self.skyimg_window_size = skyimg_window_size
        if satimg_window_size < 1:
            raise ValueError("satimg_window_size must be >= 1")
        self.satimg_window_size = satimg_window_size
        if csv_interval_min <= 0 or pv_input_interval_min % csv_interval_min:
            raise ValueError("pv_input_interval_min must be a positive multiple of csv_interval_min")
        if pv_output_interval_min % csv_interval_min:
            raise ValueError("pv_output_interval_min must be a positive multiple of csv_interval_min")
        self._sx = pv_input_interval_min // csv_interval_min
        self._sy = pv_output_interval_min // csv_interval_min
        self.pv_input_len = pv_input_len
        self.pv_output_len = pv_output_len
        self.pv_output_interval_min = pv_output_interval_min
        tf = float(pv_train_time_fraction)
        if not (0.0 < tf < 1.0):
            raise ValueError("pv_train_time_fraction must be strictly between 0 and 1")
        self._pv_train_time_fraction = tf
        if skyimg_time_resolution_min <= 0:
            raise ValueError("skyimg_time_resolution_min must be positive")
        self._skyimg_dt_min = skyimg_time_resolution_min
        self._skyimg_dir = Path(skyimg_dir).resolve()
        if skyimg_spatial_size < 1:
            raise ValueError("skyimg_spatial_size must be >= 1")
        self._skyimg_spatial_size = int(skyimg_spatial_size)
        if len(satimg_npy_shape_hwc) != 3 or any(x < 1 for x in satimg_npy_shape_hwc):
            raise ValueError("satimg_npy_shape_hwc must be three positive ints (H, W, C)")
        self._satimg_npy_shape_hwc = tuple(int(x) for x in satimg_npy_shape_hwc)
        if satimg_time_resolution_min <= 0:
            raise ValueError("satimg_time_resolution_min must be positive")
        self._satimg_dt_min = satimg_time_resolution_min
        self._satimg_dir = Path(satimg_dir).resolve()
        self.satimg_ds = xr.open_zarr(self._satimg_dir)
        self.skyimg_ds = xr.open_zarr(self._skyimg_dir)

        if test_anchor_stride_min <= 0 or test_anchor_stride_min % csv_interval_min:
            raise ValueError(
                "test_anchor_stride_min must be a positive multiple of csv_interval_min "
                f"(got {test_anchor_stride_min}, csv_interval_min={csv_interval_min})"
            )
        self._test_anchor_stride_rows = test_anchor_stride_min // csv_interval_min
        if val_anchor_stride_min <= 0 or val_anchor_stride_min % csv_interval_min:
            raise ValueError(
                "val_anchor_stride_min must be a positive multiple of csv_interval_min "
                f"(got {val_anchor_stride_min}, csv_interval_min={csv_interval_min})"
            )
        self._val_anchor_stride_rows = val_anchor_stride_min // csv_interval_min
        tol_m = int(test_collect_time_match_tolerance_min)
        if tol_m < 0:
            raise ValueError("test_collect_time_match_tolerance_min must be >= 0")
        self._test_collect_time_match_tolerance_min = tol_m
        self._test_collect_tolerance_ns = tol_m * 60 * 1_000_000_000

        with open(self._config_path) as f:
            conf = yaml.safe_load(f) or {}
        paths = get_resolved_paths(conf, _PROJECT_ROOT)
        paths_cfg = conf.get("paths", {}) or {}
        data_dir = paths.get("data_dir")

        nwp_raw = paths_cfg.get("nwp_path")
        self.nwp_solar_df = None
        self.nwp_wind_df = None
        if nwp_raw is not None and str(nwp_raw).strip() != "" and data_dir is not None:
            try:
                nwp_p = Path(nwp_raw)
                nwp_dir = nwp_p.resolve() if nwp_p.is_absolute() else (data_dir / nwp_p).resolve()
                solar_csv = nwp_dir / "solar.csv"
                wind_csv = nwp_dir / "wind.csv"
                if solar_csv.is_file() and wind_csv.is_file():
                    self.nwp_solar_df = pd.read_csv(solar_csv)
                    self.nwp_wind_df = pd.read_csv(wind_csv)
            except Exception:
                self.nwp_solar_df = None
                self.nwp_wind_df = None
        # Precompute per-start_time interpolation tables once so per-sample interpolate_nwp_features
        # avoids repeated to_datetime/groupby/sort/dropna over the full CSV (the previous CPU hot spot).
        self._nwp_solar_blocks = _precompute_nwp_blocks(self.nwp_solar_df, _NWP_SOLAR_INTERP_COLS)
        self._nwp_wind_blocks = _precompute_nwp_blocks(self.nwp_wind_df, _NWP_WIND_INTERP_COLS)

        pv_device_path = paths.get("pv_device_path")
        if pv_device_path is None:
            raise KeyError(
                f"dataset config paths.pv_device_path is required (in {self._config_path})"
            )

        # Site coordinates live with the dataset itself (``<data_dir>/info.yaml``), not in the
        # per-instance dataset config — same dataset folder reused across configs ⇒ same site.
        if data_dir is None:
            raise KeyError(
                f"dataset config paths.data_dir is required (in {self._config_path})"
            )
        info_path = Path(data_dir) / "info.yaml"
        if not info_path.is_file():
            raise FileNotFoundError(
                f"dataset info file not found: {info_path} "
                f"(expected ``site.latitude`` / ``site.longitude``)"
            )
        with open(info_path) as f:
            info = yaml.safe_load(f) or {}
        site = info.get("site", {}) or {}
        self.latitude = site.get("latitude")
        self.longitude = site.get("longitude")
        if self.latitude is None or self.longitude is None:
            raise KeyError(
                f"{info_path} must define both site.latitude and site.longitude"
            )

        pv_device_df = pd.read_excel(pv_device_path)
        self.devDn_list = pv_device_df["devDn"].dropna().unique().tolist()

        self.sample_files = list_csv_files(data_dir=pv_dir)
        if not self.sample_files:
            raise FileNotFoundError(f"No CSV files in {pv_dir!r}")

        ref_df = load_csv(self.sample_files[0])
        n = len(ref_df)
        lx, ly = self.pv_input_len, self.pv_output_len
        sx, sy = self._sx, self._sy
        amin = (lx - 1) * sx
        y_last_off = sy * ly
        amax = n - 1 - y_last_off
        if n == 0 or amin > amax:
            raise RuntimeError(
                f"reference CSV {self.sample_files[0].name}: no anchor fits bounds "
                f"(n={n}, need {amin}<=anchor<={amax})"
            )
        anchors = np.arange(amin, amax + 1, dtype=np.intp)
        y_off = sy + np.arange(ly, dtype=np.intp) * sy
        x_tail = (-(lx - 1) * sx + np.arange(lx, dtype=np.intp) * sx).reshape(1, -1)
        self._csv_row_count = n
        self._y_idx_per_anchor = anchors[:, None] + y_off[None, :]
        self._x_idx_per_anchor = anchors[:, None] + x_tail
        self._x_tail_1d = (-(lx - 1) * sx + np.arange(lx, dtype=np.intp) * sx).astype(np.intp, copy=False)
        self._y_off_1d = (sy + np.arange(ly, dtype=np.intp) * sy).astype(np.intp, copy=False)

        split_train_end = int(n * 0.6)
        split_val_end = int(n * 0.7)
        if not (0 < split_train_end < split_val_end < n):
            raise ValueError(
                f"fixed 60%/10%/30% row split invalid for n={n}: "
                f"split_train_end={split_train_end}, split_val_end={split_val_end}"
            )
        min_row = anchors - (lx - 1) * sx
        max_row = anchors + ly * sy
        self._train_anchor_mask = max_row < split_train_end
        self._val_anchor_mask = (min_row >= split_train_end) & (max_row < split_val_end)
        self._test_anchor_mask = min_row >= split_val_end
        if self.split == "train" and not bool(self._train_anchor_mask.any()):
            raise RuntimeError(
                f"split=train: no anchor fits entirely in the first {split_train_end} rows (60% of n={n}); "
                "shorten windows or check data length"
            )
        if self.split == "val" and not bool(self._val_anchor_mask.any()):
            raise RuntimeError(
                f"split=val: no anchor fits entirely in rows [{split_train_end}, {split_val_end}) "
                f"(10% val band); adjust window lengths or stride"
            )
        if self.split == "test" and not bool(self._test_anchor_mask.any()):
            raise RuntimeError(
                f"split=test: no anchor fits entirely from row {split_val_end} onward (last 30%); "
                "adjust window lengths"
            )

        val_stride = self._val_anchor_stride_rows
        val_anchor_positions = np.nonzero(self._val_anchor_mask)[0]
        self._val_r_indices = val_anchor_positions[::val_stride].astype(np.intp, copy=False)
        self._num_val_windows = int(self._val_r_indices.size)
        if self.split == "val" and self._num_val_windows == 0:
            raise RuntimeError(
                "split=val: no val anchors after stride subsampling "
                "(reduce val_anchor_stride_min or widen the val segment)"
            )

        test_stride = self._test_anchor_stride_rows
        test_anchor_positions = np.nonzero(self._test_anchor_mask)[0]
        self._test_r_indices = test_anchor_positions[::test_stride].astype(np.intp, copy=False)
        self._num_test_windows = int(self._test_r_indices.size)
        if self.split == "test" and self._num_test_windows == 0:
            raise RuntimeError(
                "split=test: no test anchors after time split and stride "
                "(reduce test_anchor_stride_min or widen the test segment)"
            )

        if self.split in ("val", "test"):
            ct_ref = pd.to_datetime(ref_df["collectTime"], errors="coerce")
            if ct_ref.isna().any():
                raise ValueError(f"reference CSV {self.sample_files[0].name}: NaT in collectTime")
            if self.split == "val":
                last_x_rows_val = anchors[self._val_r_indices]
                self._val_last_x_time_ref = [
                    pd.Timestamp(ct_ref.iloc[int(row)]) for row in last_x_rows_val
                ]
            else:
                self._val_last_x_time_ref = None
            if self.split == "test":
                last_x_rows = anchors[self._test_r_indices]
                self._test_last_x_time_ref = [
                    pd.Timestamp(ct_ref.iloc[int(row)]) for row in last_x_rows
                ]
            else:
                self._test_last_x_time_ref = None
        else:
            self._val_last_x_time_ref = None
            self._test_last_x_time_ref = None

        valid_files: list[Path] = []
        self._csv_cache: dict[str, pd.DataFrame] = {}
        for i, p in enumerate(self.sample_files):
            print(f"Processing file {i+1} of {len(self.sample_files)}: {p.name}")
            df = ref_df if i == 0 else load_csv(p)
            if len(df) != n:
                continue
            if INVERTER_STATE_COL not in df.columns:
                continue
            cache_key = p.resolve().as_posix()
            if self.split == "train":
                inv = (
                    pd.to_numeric(df[INVERTER_STATE_COL], errors="coerce").fillna(0).astype(int).values
                    == VALID_STATE
                )
                ok = inv[self._y_idx_per_anchor].any(axis=1) & self._train_anchor_mask
                if ok.any():
                    valid_files.append(p)
                    self._csv_cache[cache_key] = df
            else:
                valid_files.append(p)
                self._csv_cache[cache_key] = df

        self.sample_files = valid_files
        if self.split == "val":
            _tw = f", val_windows_per_file={self._num_val_windows}"
        elif self.split == "test":
            _tw = f", test_windows_per_file={self._num_test_windows}"
        else:
            _tw = ""
        print(f"Valid files: {len(self.sample_files)} ({self.split}{_tw})")
        if not self.sample_files:
            raise RuntimeError(
                f"No CSV files left after prefilter for split={self.split!r} in pv_dir={pv_dir!r}"
                + (
                    f" (train requires at least one Y row with {INVERTER_STATE_COL}=={VALID_STATE})"
                    if self.split == "train"
                    else ""
                )
            )

    def __len__(self):
        if self.split == "train":
            return len(self.sample_files)
        if self.split == "val":
            return len(self.sample_files) * self._num_val_windows
        return len(self.sample_files) * self._num_test_windows

    def _row_index_for_collect_time_match(
        self,
        collect_time: pd.Series,
        target_ts: pd.Timestamp,
        csv_name: str,
    ) -> int:
        """Map ``collect_time`` row to index whose timestamp is nearest ``target_ts`` within tolerance (ns)."""
        ct = pd.to_datetime(collect_time, errors="coerce")
        if bool(ct.isna().any()):
            raise ValueError(f"{csv_name}: NaT in collectTime")
        if not bool(ct.is_monotonic_increasing):
            raise ValueError(f"{csv_name}: collectTime must be sorted non-decreasing for test alignment")
        times_ns = ct.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        tgt = pd.Timestamp(target_ts)
        if tgt.tzinfo is not None:
            tgt = tgt.tz_convert("UTC").tz_localize(None)
        tgt_ns = np.datetime64(tgt.to_datetime64(), "ns").astype(np.int64)

        pos = int(np.searchsorted(times_ns, tgt_ns, side="left"))
        candidates: list[int] = []
        if pos > 0:
            candidates.append(pos - 1)
        if pos < len(times_ns):
            candidates.append(pos)
        if not candidates:
            raise RuntimeError(f"{csv_name}: empty collectTime series")

        best_j = candidates[0]
        best_d = abs(int(times_ns[best_j] - tgt_ns))
        for j in candidates[1:]:
            d = abs(int(times_ns[j] - tgt_ns))
            if d < best_d:
                best_d = d
                best_j = j

        if best_d > self._test_collect_tolerance_ns:
            raise ValueError(
                f"{csv_name}: no collectTime within {self._test_collect_time_match_tolerance_min} min of "
                f"{target_ts!r} (nearest row {best_j}, delta_ns={best_d})"
            )
        return int(best_j)

    @staticmethod
    def _to_utc_timestamps(values) -> list[pd.Timestamp]:
        """
        Normalize a sequence to UTC-aware ``pd.Timestamp`` (same format for forecast / sat / sky).

        Naive inputs are interpreted as UTC; tz-aware inputs are converted to UTC.
        """
        out: list[pd.Timestamp] = []
        for v in values:
            t = pd.Timestamp(v)
            if t.tz is None:
                t = t.tz_localize("UTC")
            out.append(t.tz_convert("UTC"))
        return out

    @staticmethod
    def _sky_filename_ts(ts_raw) -> pd.Timestamp:
        """Naive UTC timestamp for sky filenames; seconds floored to 0."""
        ts = pd.Timestamp(ts_raw)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return ts.replace(second=0, microsecond=0, nanosecond=0)

    def _sky_jpg_path(self, ts: pd.Timestamp) -> Path:
        stem = ts.strftime("%Y%m%d%H%M%S")
        return self._skyimg_dir / f"{stem}.jpg"

    def _black_sky_tensor(self) -> torch.Tensor:
        s = self._skyimg_spatial_size
        return torch.zeros((3, s, s), dtype=torch.float32)

    def _load_sky_uint8_from_disk(self, path: Path) -> torch.Tensor | None:
        """Load and resize a sky JPG as ``[3, s, s]`` uint8 tensor; ``None`` on failure."""
        try:
            if not path.is_file():
                return None
            try:
                resample = Image.Resampling.LANCZOS
            except AttributeError:
                resample = Image.LANCZOS
            s = self._skyimg_spatial_size
            with Image.open(path) as im:
                im = im.convert("RGB")
                im = im.resize((s, s), resample)
                arr = np.asarray(im, dtype=np.uint8)
            return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        except Exception:
            return None

    def _load_sky_tensor(self, path: Path) -> torch.Tensor:
        """Return ``[3, s, s]`` float32 in ``[0, 1]``; falls back to zeros on missing/invalid file."""
        u8 = self._load_sky_uint8_from_disk(path)
        if u8 is not None:
            return u8.to(torch.float32) / 255.0
        return self._shared_black_sky

    def _history_sky_frame_times(self, t_end: pd.Timestamp) -> list[pd.Timestamp]:
        t_end = self._sky_filename_ts(t_end)
        w = self.skyimg_window_size
        return [
            t_end - timedelta(minutes=(w - 1 - i) * self._skyimg_dt_min) for i in range(w)
        ]

    def _forecast_sky_frame_times(self, t_start: pd.Timestamp) -> list[pd.Timestamp]:
        t_start = self._sky_filename_ts(t_start)
        w = self.skyimg_window_size
        return [t_start + timedelta(minutes=i * self._skyimg_dt_min) for i in range(w)]

    def _stack_sky_frames(self, frame_times_utc: list[pd.Timestamp]) -> torch.Tensor:
        """``frame_times_utc``: UTC-aware timestamps (same convention as ``forecast_timestamps_utc``)."""
        return torch.stack(
            [
                self._load_sky_tensor(self._sky_jpg_path(self._sky_filename_ts(t)))
                for t in frame_times_utc
            ],
            dim=0,
        )

    @staticmethod
    def _satimg_floor_utc_10min(ts_utc_naive: pd.Timestamp) -> pd.Timestamp:
        u = pd.Timestamp(ts_utc_naive)
        m = (int(u.minute) // 10) * 10
        return u.replace(minute=m, second=0, microsecond=0, nanosecond=0)

    def _satimg_utc_key_time(self, ts_raw) -> pd.Timestamp:
        """UTC naive timestamp floored to 10 min; ``ts_raw`` is already UTC (naive or aware)."""
        ts = pd.Timestamp(ts_raw)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return self._satimg_floor_utc_10min(ts)

    def _satimg_npy_path(self, t_utc_naive: pd.Timestamp) -> Path:
        u = self._satimg_floor_utc_10min(t_utc_naive)
        stem = (
            f"NC_H09_{u.strftime('%Y%m%d')}_{u.strftime('%H%M')}_L2CLP010_FLDK.02401_02401"
        )
        return self._satimg_dir / f"{stem}.npy"

    def _dummy_satimg_tensor(self) -> torch.Tensor:
        h, w, c = self._satimg_npy_shape_hwc
        return torch.zeros((c, h, w), dtype=torch.float32)

    def _load_satimg_float32_from_disk(self, path: Path) -> torch.Tensor | None:
        """Load NPY as ``[C, H, W]`` float32 tensor; ``None`` on failure or shape/dtype mismatch."""
        try:
            if not path.is_file():
                return None
            arr = np.load(path, allow_pickle=False)
            if arr.shape != self._satimg_npy_shape_hwc or arr.dtype != np.float32:
                return None
            return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        except Exception:
            return None

    def _load_satimg_tensor(self, path: Path) -> torch.Tensor:
        """Return ``[C, H, W]`` float32; falls back to zeros on missing/invalid file."""
        t = self._load_satimg_float32_from_disk(path)
        if t is not None:
            return t
        return self._shared_dummy_sat

    def _history_satimg_frame_utc_times(self, t_end_raw) -> list[pd.Timestamp]:
        t_end_utc = self._satimg_utc_key_time(t_end_raw)
        w = self.satimg_window_size
        return [
            self._satimg_floor_utc_10min(
                t_end_utc - timedelta(minutes=(w - 1 - i) * self._satimg_dt_min)
            )
            for i in range(w)
        ]

    def _forecast_satimg_frame_utc_times(self, t_start_raw) -> list[pd.Timestamp]:
        t0_utc = self._satimg_utc_key_time(t_start_raw)
        w = self.satimg_window_size
        return [
            self._satimg_floor_utc_10min(t0_utc + timedelta(minutes=i * self._satimg_dt_min))
            for i in range(w)
        ]

    def _stack_satimg_frames(self, frame_times_utc: list[pd.Timestamp]) -> torch.Tensor:
        """``frame_times_utc``: UTC-aware timestamps; NPY keys use naive UTC floored to 10 min."""
        return torch.stack(
            [
                self._load_satimg_tensor(
                    self._satimg_npy_path(
                        pd.Timestamp(t).tz_convert("UTC").tz_localize(None)
                    )
                )
                for t in frame_times_utc
            ],
            dim=0,
        )

    def _build_sample(
        self,
        df: pd.DataFrame,
        dev_idx: torch.Tensor,
        r: int,
        *,
        anchor_last_row: int | None = None,
    ) -> dict:
        t0 = time.time()
        if anchor_last_row is not None:
            j = int(anchor_last_row)
            x_idx = j + self._x_tail_1d
            y_idx_1d = j + self._y_off_1d
            if int(x_idx[0]) < 0 or int(y_idx_1d[-1]) >= len(df):
                raise ValueError(
                    f"anchor_last_row={j} out of bounds for len(df)={len(df)} "
                    f"(x_idx range [{int(x_idx[0])}, {int(x_idx[-1])}], y_idx range [{int(y_idx_1d[0])}, {int(y_idx_1d[-1])}])"
                )
        else:
            x_idx = self._x_idx_per_anchor[r]
            y_idx_1d = self._y_idx_per_anchor[r]
        sub_x = df.iloc[x_idx]
        sub_y = df.iloc[y_idx_1d]

        # Same as forecast_timestamps_utc / sat_timestamps_utc / skimg_timestamps_utc: UTC-aware pd.Timestamp
        timestamps = self._to_utc_timestamps(list(sub_x["collectTime"]))
        time0_utc = timestamps[-1]

        inv_x = pd.to_numeric(sub_x[INVERTER_STATE_COL], errors="coerce").fillna(0).astype(np.int32).values
        pow_x = pd.to_numeric(sub_x["active_power"], errors="coerce").fillna(0).values.astype(np.float32) / 50.0
        pv_mask = torch.from_numpy((inv_x == VALID_STATE).astype(np.float32)).unsqueeze(0)
        pv = torch.from_numpy(pow_x.astype(np.float32)).unsqueeze(0)

        pv_solar_features = extract_solar_features(sub_x)
        # pv_solar_features = compute_solar_features(timestamps, self.latitude, self.longitude)
        pv_timefeats = solar_features_encoder(pv_solar_features)
        pv_dtimefeats = delta_time_encoder(timestamps, time0_utc)
        pv_timefeats = torch.cat([pv_timefeats, pv_dtimefeats.unsqueeze(1)], dim=1)

        forecast_timestamps_utc = self._to_utc_timestamps(list(sub_y["collectTime"]))
        # forecast_timestamps_utc = [
        #    time0_utc + pd.Timedelta(minutes=self.pv_output_interval_min * (i + 1))
        #    for i in range(self.pv_output_len)
        #]
        forecast_solar_features = extract_solar_features(sub_y)
        #forecast_solar_features = compute_solar_features(forecast_timestamps_utc, self.latitude, self.longitude)
        forecast_timefeats = solar_features_encoder(forecast_solar_features)
        forecast_dtimefeats = delta_time_encoder(forecast_timestamps_utc, time0_utc)
        forecast_timefeats = torch.cat([forecast_timefeats, forecast_dtimefeats.unsqueeze(1)], dim=1)

        # Interpolate NWP features on forecast_timestamps_utc → [T_out, 7] (ssrd + wind scalars)
        nwp_out = interpolate_nwp_features(self._nwp_solar_blocks, self._nwp_wind_blocks, forecast_timestamps_utc)
        if nwp_out is None:
            nwp_tensor = None
        else:
            nwp_tensor = torch.from_numpy(np.asarray(nwp_out, dtype=np.float32))

        inv_y = pd.to_numeric(sub_y[INVERTER_STATE_COL], errors="coerce").fillna(0).astype(np.int32).values
        pow_y = pd.to_numeric(sub_y["active_power"], errors="coerce").fillna(0).values.astype(np.float32)
        target_pv = torch.from_numpy((pow_y / 50.0).astype(np.float32))
        target_mask = torch.from_numpy((inv_y == VALID_STATE).astype(np.float32))

        # Select satellite images from the window
        sat_t0 = time0_utc - timedelta(minutes=(245+30))
        sat_t1 = time0_utc - timedelta(minutes=30)
        sat_t0 = pd.Timestamp(sat_t0).tz_convert("UTC").tz_localize(None)
        sat_t1 = pd.Timestamp(sat_t1).tz_convert("UTC").tz_localize(None)
        sat_data = self.satimg_ds.sel(time_utc=slice(sat_t0, sat_t1))
        sat_solar_features = {'azimuth': sat_data['azimuth'].values, 
                              'zenith': sat_data['zenith'].values, 
                              'day_of_year': sat_data['day_of_year'].values, 
                              'hour_of_day': sat_data['hour_of_day'].values}
        sat_timestamps_utc = sat_data['time_utc'].values

        sat_timefeats = solar_features_encoder(sat_solar_features)
        sat_dtimefeats = delta_time_encoder(sat_timestamps_utc, time0_utc)
        sat_timefeats = torch.cat([sat_timefeats, sat_dtimefeats.unsqueeze(1)], dim=1)
        sat_tensor = torch.from_numpy(
            np.asarray(sat_data['images'].values, dtype=np.float32)
        )
        ones_channel = torch.ones(
            (sat_tensor.shape[0], 1, sat_tensor.shape[2], sat_tensor.shape[3]),
            dtype=sat_tensor.dtype,
        )
        sat_tensor = torch.cat([sat_tensor, ones_channel], dim=1)

        if sat_timefeats.shape[0]>=self.satimg_window_size:
            sat_timefeats = sat_timefeats[-self.satimg_window_size:, :]
        else:
            sat_timefeats = torch.cat([torch.zeros(self.satimg_window_size-sat_timefeats.shape[0], sat_timefeats.shape[1]), sat_timefeats], dim=0)

        if sat_tensor.shape[0] >= self.satimg_window_size:
            sat_tensor = sat_tensor[-self.satimg_window_size:]
        else:
            pad = torch.zeros(
                (self.satimg_window_size - sat_tensor.shape[0],) + tuple(sat_tensor.shape[1:]),
                dtype=sat_tensor.dtype,
            )
            sat_tensor = torch.cat([pad, sat_tensor], dim=0)

        # Select sky images from the window
        sky_t0 = time0_utc - timedelta(minutes=(30))
        sky_t1 = time0_utc - timedelta(minutes=0)
        sky_t0 = pd.Timestamp(sky_t0).tz_convert("UTC").tz_localize(None)
        sky_t1 = pd.Timestamp(sky_t1).tz_convert("UTC").tz_localize(None)
        sky_data = self.skyimg_ds.sel(time_utc=slice(sky_t0, sky_t1))
        sky_solar_features = {'azimuth': sky_data['azimuth'].values, 
                              'zenith': sky_data['zenith'].values, 
                              'day_of_year': sky_data['day_of_year'].values, 
                              'hour_of_day': sky_data['hour_of_day'].values}
        
        sky_timestamps_utc = sky_data['time_utc'].values

        sky_timefeats = solar_features_encoder(sky_solar_features)
        sky_dtimefeats = delta_time_encoder(sky_timestamps_utc, time0_utc)
        sky_timefeats = torch.cat([sky_timefeats, sky_dtimefeats.unsqueeze(1)], dim=1)
        sky_tensor = torch.from_numpy(
            np.asarray(sky_data['images'].values, dtype=np.float32)
        )

        return {
            "dev_idx": dev_idx,
            "pv": pv,
            "pv_mask": pv_mask,
            "pv_timefeats": pv_timefeats,
            "forecast_timefeats": forecast_timefeats,
            "sat_tensor": sat_tensor,
            "sat_timefeats": sat_timefeats,
            "skimg_tensor": None,
            "skimg_timefeats": None,
            "nwp_tensor": nwp_tensor,
            "target_pv": target_pv,
            "target_mask": target_mask,
        }


    def __getitem__(self, idx):
        if self.split == "train":
            sample_path = self.sample_files[idx]
            r_fixed: int | None = None
        elif self.split == "val":
            nw = self._num_val_windows
            sample_path = self.sample_files[idx // nw]
            r_fixed = int(self._val_r_indices[idx % nw])
        else:
            nw = self._num_test_windows
            sample_path = self.sample_files[idx // nw]
            r_fixed = int(self._test_r_indices[idx % nw])

        devDn = sample_path.stem.replace("_", "=")
        try:
            dev_idx = torch.tensor(self.devDn_list.index(devDn), dtype=torch.long)
        except ValueError as e:
            raise ValueError(f"devDn {devDn!r} from {sample_path.name} not in pv_device list") from e

        cache_key = sample_path.resolve().as_posix()
        try:
            df = self._csv_cache[cache_key]
        except KeyError as e:
            raise RuntimeError(
                f"CSV not in in-memory cache for {sample_path.name!r} (key={cache_key!r}); "
                "this path must be in PVDataset.sample_files from __init__"
            ) from e
        if INVERTER_STATE_COL not in df.columns:
            raise ValueError(f"missing {INVERTER_STATE_COL} in {sample_path}")

        n = len(df)
        if n != self._csv_row_count:
            raise ValueError(
                f"{sample_path.name}: expected {self._csv_row_count} rows (same as reference CSV), got {n}"
            )

        if self.split == "train":
            inv = (
                pd.to_numeric(df[INVERTER_STATE_COL], errors="coerce").fillna(0).astype(int).values
                == VALID_STATE
            )
            valid_mask = inv[self._y_idx_per_anchor].any(axis=1) & self._train_anchor_mask
            valid_rows = np.nonzero(valid_mask)[0]
            if valid_rows.size == 0:
                raise RuntimeError(
                    f"{sample_path.name}: no anchor with at least one Y row where "
                    f"{INVERTER_STATE_COL}=={VALID_STATE}"
                )
            r = int(np.random.choice(valid_rows))
        else:
            assert r_fixed is not None
            if self.split == "val":
                assert self._val_last_x_time_ref is not None
                nw = self._num_val_windows
                time_ref_list = self._val_last_x_time_ref
            else:
                assert self._test_last_x_time_ref is not None
                nw = self._num_test_windows
                time_ref_list = self._test_last_x_time_ref
            k = idx % nw
            T_ref = time_ref_list[k]
            j = self._row_index_for_collect_time_match(df["collectTime"], T_ref, sample_path.name)
            return self._build_sample(df, dev_idx, r_fixed, anchor_last_row=j)

        return self._build_sample(df, dev_idx, r)


def collate_batched(batch):
    """Stack list of samples into one dict of tensors with batch dim B in front."""
    if not batch:
        raise ValueError("empty batch")

    def _stack(key: str) -> torch.Tensor:
        return torch.stack([s[key] for s in batch])

    out: dict = {
        "dev_idx": _stack("dev_idx"),
        "pv": _stack("pv"),
        "pv_mask": _stack("pv_mask"),
        "pv_timefeats": _stack("pv_timefeats"),
        "forecast_timefeats": _stack("forecast_timefeats"),
        "target_pv": _stack("target_pv"),
        "target_mask": _stack("target_mask"),
    }
    for key in ("nwp_tensor", "sat_tensor", "sat_timefeats", "skimg_tensor", "skimg_timefeats"):
        vals = [s[key] for s in batch]
        if vals[0] is None:
            if not all(v is None for v in vals):
                raise ValueError(f"collate_batched: mixed None and tensor for {key!r}")
            out[key] = None
        else:
            out[key] = torch.stack(vals)
    return out

