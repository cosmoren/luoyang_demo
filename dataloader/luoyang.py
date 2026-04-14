"""Luoyang PV dataset and loader utilities for training."""
import sys
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from pvlib import solarposition
from torch.utils.data import Dataset
from modules.solar_encoder import compute_solar_features, solar_features_encoder, delta_time_encoder

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config_utils import get_resolved_paths
from training.training_conf import CONF_PATH, get_training_hparams_from_conf, load_config
from training.data_loader import (
    INVERTER_STATE_COL,
    VALID_STATE,
    list_csv_files,
    load_csv,
)


def _load_training_defaults() -> dict:
    conf = load_config()
    return get_training_hparams_from_conf(conf)


_TRAINING_HPARAM_DEFAULTS = _load_training_defaults()

# ``nwp_interp`` columns: ssrd (solar) then wind fields, in this order.
_NWP_INTERP_STACK_COLS = ("ssrd", "msl", "t2m", "u10", "v10", "u100", "v100")
_NWP_WIND_INTERP_COLS = _NWP_INTERP_STACK_COLS[1:]


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


def _linear_interp_nwp_column(
    block: pd.DataFrame,
    forecast_timestamps_utc: list,
    value_col: str,
) -> np.ndarray:
    """Linearly interpolate ``value_col`` vs ``forecast_time`` in ``block`` onto ``forecast_timestamps_utc`` (UTC)."""
    n = len(forecast_timestamps_utc)
    if block.empty or "forecast_time" not in block.columns or value_col not in block.columns:
        return np.full(n, np.nan, dtype=np.float64)
    ft = pd.to_datetime(block["forecast_time"], utc=True)
    y = pd.to_numeric(block[value_col], errors="coerce")
    tab = pd.DataFrame({"ft": ft, "y": y}).dropna(subset=["y"])
    if tab.empty:
        return np.full(n, np.nan, dtype=np.float64)
    tab = tab.sort_values("ft").drop_duplicates(subset=["ft"], keep="last")
    xp = pd.DatetimeIndex(tab["ft"]).asi8.astype(np.float64)
    fp = tab["y"].to_numpy(dtype=np.float64)
    xq = pd.DatetimeIndex(pd.to_datetime(forecast_timestamps_utc, utc=True)).asi8.astype(np.float64)
    return np.interp(xq, xp, fp)


def interpolate_nwp_features(nwp_solar_df, nwp_wind_df, forecast_timestamps_utc, dhour=12):
    """
    Use ``forecast_timestamps_utc[0]`` minus ``dhour`` to get ``t_ref_utc``, then ``prev_noon_utc`` (last UTC
    noon before ``t_ref_utc``). Slice solar/wind rows with ``start_time == prev_noon_utc``. Linearly
    interpolate ``ssrd`` from the solar block and ``msl``, ``t2m``, ``u10``, ``v10``, ``u100``, ``v100`` from
    the wind block along ``forecast_time`` (UTC) onto every ``forecast_timestamps_utc`` (``np.nan`` if no data).
    Those series are stacked as ``nwp_interp`` with shape ``[n, 7]`` (then nan/inf → 0 with ``_sanitize_nwp_interp``).
    """
    if nwp_solar_df is None or nwp_wind_df is None:
        return None
    if not forecast_timestamps_utc:
        return None

    delta = pd.Timedelta(hours=float(dhour))
    t_ref = pd.Timestamp(forecast_timestamps_utc[0]) - delta
    if t_ref.tzinfo is None:
        t_ref_utc = t_ref.tz_localize("UTC")
    else:
        t_ref_utc = t_ref.tz_convert("UTC")

    midnight_utc = t_ref_utc.normalize()
    noon_same_day = midnight_utc + pd.Timedelta(hours=12)
    if noon_same_day < t_ref_utc:
        prev_noon_utc = noon_same_day
    else:
        prev_noon_utc = noon_same_day - pd.Timedelta(days=1)

    solar_st = pd.to_datetime(nwp_solar_df["start_time"], utc=True)
    nwp_solar_block = nwp_solar_df.loc[solar_st == prev_noon_utc].copy()
    wind_st = pd.to_datetime(nwp_wind_df["start_time"], utc=True)
    nwp_wind_block = nwp_wind_df.loc[wind_st == prev_noon_utc].copy()

    ssrd_interp = _linear_interp_nwp_column(nwp_solar_block, forecast_timestamps_utc, "ssrd")
    wind_interp = {
        col: _linear_interp_nwp_column(nwp_wind_block, forecast_timestamps_utc, col)
        for col in _NWP_WIND_INTERP_COLS
    }

    nwp_interp = np.column_stack([ssrd_interp] + [wind_interp[c] for c in _NWP_WIND_INTERP_COLS])
    nwp_interp_clean, nwp_mask = _sanitize_nwp_interp(nwp_interp)
    nwp_interp_wmask = np.concatenate([nwp_interp_clean, nwp_mask], axis=1) # [T_out, 8]  (ssrd, msl, t2m, u10, v10, u100, v100, mask) mask is 1 if any nan/inf, 0 otherwise

    return nwp_interp_wmask


class PVDataset(Dataset):
    """
    **Chronological split (per CSV, same for all files sharing the reference row count):** row indices
    ``[0, split_idx)`` are the train segment, ``[split_idx, n)`` the test segment, with
    ``split_idx = int(n * pv_train_time_fraction)``. An anchor is used for **train** only if the full
    X+Y window lies in the train segment (last Y row ``< split_idx``); for **test** only if the full
    window lies in the test segment (first X row ``>= split_idx``). Sky/sat are still loaded by
    timestamp for each sample.

    Train: ``__len__`` = number of CSVs; ``__getitem__`` picks a **random** train-segment anchor where Y has at
    least one ``inverter_state == VALID_STATE``.

    Test (``split="test"``): ``__len__`` = ``len(sample_files) * num_test_windows``; test-segment anchors are
    subsampled every ``test_anchor_stride_min`` along the CSV row axis (no Y validity filter). For each test
    window, the **reference** last-X ``collectTime`` is stored; each inverter CSV resolves the row index that
    matches that time (within ``test_collect_time_match_tolerance_min``) so all devices share the same wall times.

    Sky images (under ``data_dir`` / ``paths.sky_image_path``, ``YYYYMMDDHHMMSS_12.jpg``): history sequence
    ends at the last X ``collectTime``; forecast sequence starts at the first Y ``collectTime``;
    step size is ``skyimg_time_resolution_min`` (independent of PV CSV spacing).
    Missing files → black ``(3, H, W)`` tensors resized to ``training.skyimg_spatial_size``.

    Himawari NPY (``satimg``): names ``NC_H09_YYYYMMDD_HHMM_L2CLP010_FLDK.02401_02401.npy`` with
    ``YYYYMMDD_HHMM`` in UTC; ``collectTime`` is interpreted as Asia/Shanghai local, converted to UTC,
    then floored to 10-minute boundaries for the key timestep. History/forecast windows step in UTC
    (sizes from ``conf.yaml`` ``training``). Arrays are float32 HWC per ``training.satimg_npy_shape_hwc``;
    missing files → zeros.

    If ``conf paths.nwp_path`` is set (relative to ``paths.data_dir`` unless absolute) and ``solar.csv`` /
    ``wind.csv`` exist and load successfully, they populate ``nwp_solar_df`` and ``nwp_wind_df``; otherwise
    (missing key, missing ``data_dir``, missing files, or read error) both stay ``None`` without raising.
    """

    def __init__(
        self,
        pv_dir: str,
        skyimg_dir: str,
        satimg_dir: str,
        *,
        split: str = "train",
        csv_interval_min: int = _TRAINING_HPARAM_DEFAULTS["csv_interval_min"],
        pv_input_interval_min: int = _TRAINING_HPARAM_DEFAULTS["pv_input_interval_min"],
        pv_input_len: int = _TRAINING_HPARAM_DEFAULTS["pv_input_len"],
        pv_output_interval_min: int = _TRAINING_HPARAM_DEFAULTS["pv_output_interval_min"],
        pv_output_len: int = _TRAINING_HPARAM_DEFAULTS["pv_output_len"],
        pv_train_time_fraction: float = _TRAINING_HPARAM_DEFAULTS["pv_train_time_fraction"],
        test_anchor_stride_min: int = _TRAINING_HPARAM_DEFAULTS["test_anchor_stride_min"],
        test_collect_time_match_tolerance_min: int = _TRAINING_HPARAM_DEFAULTS[
            "test_collect_time_match_tolerance_min"
        ],
        skyimg_window_size: int = _TRAINING_HPARAM_DEFAULTS["skyimg_window_size"],
        skyimg_time_resolution_min: int = _TRAINING_HPARAM_DEFAULTS["skyimg_time_resolution_min"],
        skyimg_spatial_size: int = _TRAINING_HPARAM_DEFAULTS["skyimg_spatial_size"],
        satimg_window_size: int = _TRAINING_HPARAM_DEFAULTS["satimg_window_size"],
        satimg_time_resolution_min: int = _TRAINING_HPARAM_DEFAULTS["satimg_time_resolution_min"],
        satimg_npy_shape_hwc: tuple[int, int, int] = _TRAINING_HPARAM_DEFAULTS["satimg_npy_shape_hwc"],
    ):
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
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
        if test_anchor_stride_min <= 0 or test_anchor_stride_min % csv_interval_min:
            raise ValueError(
                "test_anchor_stride_min must be a positive multiple of csv_interval_min "
                f"(got {test_anchor_stride_min}, csv_interval_min={csv_interval_min})"
            )
        self._test_anchor_stride_rows = test_anchor_stride_min // csv_interval_min
        tol_m = int(test_collect_time_match_tolerance_min)
        if tol_m < 0:
            raise ValueError("test_collect_time_match_tolerance_min must be >= 0")
        self._test_collect_time_match_tolerance_min = tol_m
        self._test_collect_tolerance_ns = tol_m * 60 * 1_000_000_000

        with open(CONF_PATH) as f:
            conf = yaml.safe_load(f)
        paths = get_resolved_paths(conf, _PROJECT_ROOT)
        paths_cfg = conf.get("paths", {}) or {}
        nwp_raw = paths_cfg.get("nwp_path")
        self.nwp_solar_df = None
        self.nwp_wind_df = None
        if nwp_raw is not None and str(nwp_raw).strip() != "":
            data_dir = paths.get("data_dir")
            if data_dir is not None:
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

        pv_device_path = paths["pv_device_path"]

        site = conf.get("site", {})
        self.latitude = site.get("latitude")
        self.longitude = site.get("longitude")

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

        split_idx = int(n * self._pv_train_time_fraction)
        if split_idx <= 0 or split_idx >= n:
            raise ValueError(
                f"pv_train_time_fraction={self._pv_train_time_fraction} gives split_idx={split_idx} for n={n}; "
                "need 0 < split_idx < n"
            )
        min_row = anchors - (lx - 1) * sx
        max_row = anchors + ly * sy
        self._train_anchor_mask = max_row < split_idx
        self._test_anchor_mask = min_row >= split_idx
        if self.split == "train" and not bool(self._train_anchor_mask.any()):
            raise RuntimeError(
                f"split=train: no anchor fits entirely in the first {split_idx} rows "
                f"(pv_train_time_fraction={self._pv_train_time_fraction}); shorten windows or adjust fraction"
            )
        if self.split == "test" and not bool(self._test_anchor_mask.any()):
            raise RuntimeError(
                f"split=test: no anchor fits entirely from row {split_idx} onward; "
                "adjust pv_train_time_fraction or window lengths"
            )

        n_anchors = len(anchors)
        test_anchor_positions = np.nonzero(self._test_anchor_mask)[0]
        stride = self._test_anchor_stride_rows
        self._test_r_indices = test_anchor_positions[::stride].astype(np.intp, copy=False)
        self._num_test_windows = int(self._test_r_indices.size)
        if self.split == "test" and self._num_test_windows == 0:
            raise RuntimeError(
                "split=test: no test anchors after time split and stride "
                "(reduce test_anchor_stride_min or widen the test segment)"
            )

        if self.split == "test":
            last_x_rows = anchors[self._test_r_indices]
            ct_ref = pd.to_datetime(ref_df["collectTime"], errors="coerce")
            if ct_ref.isna().any():
                raise ValueError(f"reference CSV {self.sample_files[0].name}: NaT in collectTime")
            self._test_last_x_time_ref = [pd.Timestamp(ct_ref.iloc[int(row)]) for row in last_x_rows]
        else:
            self._test_last_x_time_ref = None

        valid_files: list[Path] = []
        for i, p in enumerate(self.sample_files):
            print(f"Processing file {i+1} of {len(self.sample_files)}: {p.name}")
            df = ref_df if i == 0 else load_csv(p)
            if len(df) != n:
                continue
            if INVERTER_STATE_COL not in df.columns:
                continue
            if self.split == "train":
                inv = (
                    pd.to_numeric(df[INVERTER_STATE_COL], errors="coerce").fillna(0).astype(int).values
                    == VALID_STATE
                )
                ok = inv[self._y_idx_per_anchor].any(axis=1) & self._train_anchor_mask
                if ok.any():
                    valid_files.append(p)
            else:
                valid_files.append(p)

        self.sample_files = valid_files
        _tw = (
            f", test_windows_per_file={self._num_test_windows}" if self.split == "test" else ""
        )
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
    def _to_utc_timestamps(values, *, naive_tz: str) -> list[pd.Timestamp]:
        """
        Normalize a sequence to UTC-aware ``pd.Timestamp`` (same format for forecast / sat / sky).

        - If an element is naive, it is interpreted as wall time in ``naive_tz`` (``UTC`` or ``Asia/Shanghai``).
        - If timezone-aware, it is converted to UTC.
        """
        out: list[pd.Timestamp] = []
        for v in values:
            t = pd.Timestamp(v)
            if t.tz is None:
                t = t.tz_localize(naive_tz, ambiguous=True)
            out.append(t.tz_convert("UTC"))
        return out

    @staticmethod
    def _sky_collect_ts_local(ts_raw) -> pd.Timestamp:
        """Local-time timestamp for sky filenames; seconds floored to 0."""
        ts = pd.Timestamp(ts_raw)
        if ts.tzinfo is not None:
            ts = ts.tz_convert(ZoneInfo("Asia/Shanghai")).replace(tzinfo=None)
        return ts.replace(second=0, microsecond=0, nanosecond=0)

    def _sky_jpg_path(self, ts: pd.Timestamp) -> Path:
        stem = ts.strftime("%Y%m%d%H%M%S") + "_12"
        return self._skyimg_dir / f"{stem}.jpg"

    def _black_sky_tensor(self) -> torch.Tensor:
        s = self._skyimg_spatial_size
        return torch.zeros((3, s, s), dtype=torch.float32)

    def _load_sky_tensor(self, path: Path) -> torch.Tensor:
        try:
            if path.is_file():
                try:
                    resample = Image.Resampling.LANCZOS
                except AttributeError:
                    resample = Image.LANCZOS
                s = self._skyimg_spatial_size
                with Image.open(path) as im:
                    im = im.convert("RGB")
                    im = im.resize((s, s), resample)
                    arr = np.asarray(im, dtype=np.float32) / 255.0
                return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        except Exception:
            pass
        return self._black_sky_tensor()

    def _history_sky_frame_times(self, t_end: pd.Timestamp) -> list[pd.Timestamp]:
        t_end = self._sky_collect_ts_local(t_end)
        w = self.skyimg_window_size
        return [
            t_end - timedelta(minutes=(w - 1 - i) * self._skyimg_dt_min) for i in range(w)
        ]

    def _forecast_sky_frame_times(self, t_start: pd.Timestamp) -> list[pd.Timestamp]:
        t_start = self._sky_collect_ts_local(t_start)
        w = self.skyimg_window_size
        return [t_start + timedelta(minutes=i * self._skyimg_dt_min) for i in range(w)]

    def _utc_to_sky_local_filename_ts(self, ts_utc: pd.Timestamp) -> pd.Timestamp:
        """UTC instant → Asia/Shanghai wall time (second floored) for sky JPEG naming."""
        return (
            pd.Timestamp(ts_utc)
            .tz_convert("Asia/Shanghai")
            .replace(tzinfo=None)
            .replace(second=0, microsecond=0, nanosecond=0)
        )

    def _stack_sky_frames(self, frame_times_utc: list[pd.Timestamp]) -> torch.Tensor:
        """``frame_times_utc``: UTC-aware timestamps (same convention as ``forecast_timestamps_utc``)."""
        return torch.stack(
            [
                self._load_sky_tensor(self._sky_jpg_path(self._utc_to_sky_local_filename_ts(t)))
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
        """UTC naive timestamp floored to 10 min; derived from local collectTime (Asia/Shanghai)."""
        local_naive = self._sky_collect_ts_local(ts_raw)
        # ambiguous=True: older pandas rejects ambiguous="infer" on Timestamp.tz_localize;
        # Asia/Shanghai has no DST so wall times are not ambiguous.
        loc = local_naive.tz_localize("Asia/Shanghai", ambiguous=True)
        u = loc.tz_convert("UTC").tz_localize(None)
        return self._satimg_floor_utc_10min(u)

    def _satimg_npy_path(self, t_utc_naive: pd.Timestamp) -> Path:
        u = self._satimg_floor_utc_10min(t_utc_naive)
        stem = (
            f"NC_H09_{u.strftime('%Y%m%d')}_{u.strftime('%H%M')}_L2CLP010_FLDK.02401_02401"
        )
        return self._satimg_dir / f"{stem}.npy"

    def _dummy_satimg_tensor(self) -> torch.Tensor:
        h, w, c = self._satimg_npy_shape_hwc
        return torch.zeros((c, h, w), dtype=torch.float32)

    def _load_satimg_tensor(self, path: Path) -> torch.Tensor:
        try:
            if path.is_file():
                arr = np.load(path, allow_pickle=False)
                if arr.shape == self._satimg_npy_shape_hwc and arr.dtype == np.float32:
                    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        except Exception:
            pass
        return self._dummy_satimg_tensor()

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
        timestamps = self._to_utc_timestamps(list(sub_x["collectTime"]), naive_tz="Asia/Shanghai")
        time0_utc = timestamps[-1]

        inv_x = pd.to_numeric(sub_x[INVERTER_STATE_COL], errors="coerce").fillna(0).astype(np.int32).values
        pow_x = pd.to_numeric(sub_x["active_power"], errors="coerce").fillna(0).values.astype(np.float32) / 50.0
        pv_mask = torch.from_numpy((inv_x == VALID_STATE).astype(np.float32)).unsqueeze(0)
        pv = torch.from_numpy(pow_x.astype(np.float32)).unsqueeze(0)

        pv_solar_features = compute_solar_features(timestamps, self.latitude, self.longitude)
        pv_timefeats = solar_features_encoder(pv_solar_features)
        pv_dtimefeats = delta_time_encoder(timestamps, time0_utc)
        pv_timefeats = torch.cat([pv_timefeats, pv_dtimefeats.unsqueeze(1)], dim=1)

        forecast_timestamps_utc = [
            time0_utc + pd.Timedelta(minutes=self.pv_output_interval_min * (i + 1))
            for i in range(self.pv_output_len)
        ]
        forecast_solar_features = compute_solar_features(forecast_timestamps_utc, self.latitude, self.longitude)
        forecast_timefeats = solar_features_encoder(forecast_solar_features)
        forecast_dtimefeats = delta_time_encoder(forecast_timestamps_utc, time0_utc)
        forecast_timefeats = torch.cat([forecast_timefeats, forecast_dtimefeats.unsqueeze(1)], dim=1)

        # Interpolate NWP features on forecast_timestamps_utc → [T_out, 7] (ssrd + wind scalars)
        nwp_out = interpolate_nwp_features(self.nwp_solar_df, self.nwp_wind_df, forecast_timestamps_utc)
        if nwp_out is None:
            nwp_tensor = None
        else:
            nwp_tensor = torch.from_numpy(np.asarray(nwp_out, dtype=np.float32))

        inv_y = pd.to_numeric(sub_y[INVERTER_STATE_COL], errors="coerce").fillna(0).astype(np.int32).values
        pow_y = pd.to_numeric(sub_y["active_power"], errors="coerce").fillna(0).values.astype(np.float32)
        target_pv = torch.from_numpy((pow_y / 50.0).astype(np.float32))
        target_mask = torch.from_numpy((inv_y == VALID_STATE).astype(np.float32))

        t_x_end = sub_x["collectTime"].iloc[-1]
        sat_timestamps_utc = self._to_utc_timestamps(
            self._history_satimg_frame_utc_times(t_x_end), naive_tz="UTC"
        )
        skimg_timestamps_utc = self._to_utc_timestamps(
            self._history_sky_frame_times(t_x_end), naive_tz="Asia/Shanghai"
        )
        # timestamps, forecast_timestamps_utc, sat_timestamps_utc, skimg_timestamps_utc: list[pd.Timestamp] tz=UTC
        sat_solar_features = compute_solar_features(sat_timestamps_utc, self.latitude, self.longitude)
        sat_timefeats = solar_features_encoder(sat_solar_features)
        sat_dtimefeats = delta_time_encoder(sat_timestamps_utc, time0_utc)
        sat_timefeats = torch.cat([sat_timefeats, sat_dtimefeats.unsqueeze(1)], dim=1)
        sat_tensor = self._stack_satimg_frames(sat_timestamps_utc)

        skimg_solar_features = compute_solar_features(skimg_timestamps_utc, self.latitude, self.longitude)
        skimg_timefeats = solar_features_encoder(skimg_solar_features)
        skimg_dtimefeats = delta_time_encoder(skimg_timestamps_utc, time0_utc)
        skimg_timefeats = torch.cat([skimg_timefeats, skimg_dtimefeats.unsqueeze(1)], dim=1)
        skimg_tensor = self._stack_sky_frames(skimg_timestamps_utc)

        return {
            "dev_idx": dev_idx,
            "pv": pv,
            "pv_mask": pv_mask,
            "pv_timefeats": pv_timefeats,
            "forecast_timefeats": forecast_timefeats,
            "sat_tensor": sat_tensor,
            "sat_timefeats": sat_timefeats,
            "skimg_tensor": skimg_tensor,
            "skimg_timefeats": skimg_timefeats,
            "nwp_tensor": nwp_tensor,
            "target_pv": target_pv,
            "target_mask": target_mask,
        }


    def __getitem__(self, idx):
        if self.split == "train":
            sample_path = self.sample_files[idx]
            r_fixed: int | None = None
        else:
            nw = self._num_test_windows
            sample_path = self.sample_files[idx // nw]
            r_fixed = int(self._test_r_indices[idx % nw])

        devDn = sample_path.stem.replace("_", "=")
        try:
            dev_idx = torch.tensor(self.devDn_list.index(devDn), dtype=torch.long)
        except ValueError as e:
            raise ValueError(f"devDn {devDn!r} from {sample_path.name} not in pv_device list") from e

        df = load_csv(sample_path)
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
            assert self._test_last_x_time_ref is not None
            k = idx % self._num_test_windows
            T_ref = self._test_last_x_time_ref[k]
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
        "sat_tensor": _stack("sat_tensor"),
        "sat_timefeats": _stack("sat_timefeats"),
        "target_pv": _stack("target_pv"),
        "target_mask": _stack("target_mask"),
    }
    for key in ("skimg_tensor", "skimg_timefeats", "nwp_tensor"):
        vals = [s[key] for s in batch]
        if vals[0] is None:
            if not all(v is None for v in vals):
                raise ValueError(f"collate_batched: mixed None and tensor for {key!r}")
            out[key] = None
        else:
            out[key] = torch.stack(vals)
    return out

