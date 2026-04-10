"""
Training script for pv_forecasting_model.
Loads config from conf.yaml, builds a dataset (synthetic or real), and trains with MSE loss.
"""

import argparse
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
from pvlib import solarposition
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config_utils import get_resolved_paths
from models.models import pv_forecasting_model
from datasets.data_loader import (
    INVERTER_STATE_COL,
    VALID_STATE,
    list_csv_files,
    load_csv,
)
from datetime import timedelta

CONF_PATH = _PROJECT_ROOT / "config" / "conf.yaml"

# Every key must appear under ``training:`` in conf.yaml (values may be null where allowed).
TRAINING_HPARAM_KEYS = frozenset({
    "csv_interval_min",
    "pv_input_interval_min",
    "pv_output_interval_min",
    "pv_input_len",
    "pv_output_len",
    "test_anchor_stride_min",
    "skyimg_window_size",
    "skyimg_time_resolution_min",
    "skyimg_spatial_size",
    "staimg_window_size",
    "staimg_time_resolution_min",
    "staimg_npy_shape_hwc",
    "epochs",
    "lr",
    "batch_size",
    "save_every",
    "num_workers",
    "train_max_batches_per_epoch",
})


def load_config():
    with open(CONF_PATH) as f:
        conf = yaml.safe_load(f)
    return conf


def get_training_hparams_from_conf(conf: dict | None = None) -> dict:
    """Load ``conf['training']``; every :data:`TRAINING_HPARAM_KEYS` entry must be set in YAML."""
    if conf is None:
        conf = load_config()
    raw = conf.get("training")
    if not isinstance(raw, dict):
        raise ValueError("conf.yaml must define a non-empty 'training:' mapping")
    missing = sorted(TRAINING_HPARAM_KEYS - raw.keys())
    if missing:
        raise KeyError(
            "conf training section missing required key(s): "
            + ", ".join(missing)
            + " (see TRAINING_HPARAM_KEYS in train.py)"
        )
    out = {k: raw[k] for k in TRAINING_HPARAM_KEYS}
    if isinstance(out["lr"], str):
        out["lr"] = float(out["lr"])

    ss = out["skyimg_spatial_size"]
    if not isinstance(ss, int) or isinstance(ss, bool) or ss < 1:
        raise ValueError("training.skyimg_spatial_size must be a positive integer")
    out["skyimg_spatial_size"] = int(ss)

    shwc = out["staimg_npy_shape_hwc"]
    if not isinstance(shwc, (list, tuple)) or len(shwc) != 3:
        raise ValueError("training.staimg_npy_shape_hwc must be a length-3 sequence [H, W, C]")
    t = tuple(int(x) for x in shwc)
    if any(x < 1 for x in t):
        raise ValueError("training.staimg_npy_shape_hwc entries must be positive")
    out["staimg_npy_shape_hwc"] = t

    return out


def get_training_paths_from_conf(conf: dict | None = None) -> dict[str, str]:
    """
    Resolve PV CSV, sky image, and Himawari NPY dirs from ``conf.yaml`` ``paths``:
    ``pv_train_path`` / ``pv_test_path``, ``sky_image_train_path`` / ``sky_image_test_path``,
    and ``sat_*`` as direct children of ``data_dir`` (e.g. ``/data/data/asi_16613_train``).
    """
    if conf is None:
        conf = load_config()
    paths_cfg = conf.get("paths", {})
    resolved = get_resolved_paths(conf, _PROJECT_ROOT)
    data_dir = resolved.get("data_dir")
    if data_dir is None:
        raise ValueError("conf paths.data_dir is required")

    def _req(key: str) -> str:
        v = paths_cfg.get(key)
        if v is None or str(v).strip() == "":
            raise KeyError(f"conf paths.{key} is required")
        return str(v)

    pv_train = (data_dir / _req("pv_train_path")).resolve()
    pv_test = (data_dir / _req("pv_test_path")).resolve()
    sky_train = (data_dir / _req("sky_image_train_path")).resolve()
    sky_test = (data_dir / _req("sky_image_test_path")).resolve()
    sta_train = (data_dir / _req("sat_train_path")).resolve()
    sta_test = (data_dir / _req("sat_test_path")).resolve()

    return {
        "pv_train_dir": str(pv_train),
        "pv_test_dir": str(pv_test),
        "skyimg_train_dir": str(sky_train),
        "skyimg_test_dir": str(sky_test),
        "staimg_train_dir": str(sta_train),
        "staimg_test_dir": str(sta_test),
    }


def _load_training_defaults() -> tuple[dict[str, str], dict]:
    conf = load_config()
    return get_training_paths_from_conf(conf), get_training_hparams_from_conf(conf)


_TRAINING_PATH_DEFAULTS, _TRAINING_HPARAM_DEFAULTS = _load_training_defaults()


def utc_to_local_solar_time_pvlib(utc_times: pd.DatetimeIndex, longitude: float) -> pd.DatetimeIndex:
    """Convert UTC to local (apparent) solar time using longitude and pvlib equation of time."""
    if utc_times.tz is not None:
        utc_naive = utc_times.tz_convert("UTC").tz_localize(None)
    else:
        utc_naive = utc_times
    lmst_offset_hours = longitude / 15.0
    dayofyear = utc_naive.dayofyear
    eot_minutes = solarposition.equation_of_time_spencer71(dayofyear)
    local_solar = utc_naive + pd.Timedelta(hours=lmst_offset_hours) + pd.to_timedelta(eot_minutes, unit="m")
    return local_solar


def compute_solar_features(
    forecast_timestamps_utc, latitude: float, longitude: float
) -> list[dict]:
    """
    From UTC forecast times and site lat/lon, compute per timestep:
    - local_solar_time: apparent solar time at the site (naive datetime)
    - azimuth: sun azimuth (degrees)
    - sin_azimuth, cos_azimuth: sin and cos of azimuth (radians)
    - zenith: sun zenith angle (degrees)
    - day_of_year: 1-366
    - hour_of_day: hour in local solar time (0-24, decimal)
    Returns a list of dicts, one per timestep.
    """
    times_utc = pd.to_datetime(forecast_timestamps_utc)
    if times_utc.tz is None:
        times_utc = times_utc.tz_localize("UTC", ambiguous="infer")
    else:
        times_utc = times_utc.tz_convert("UTC")

    # Local solar time (naive)
    local_solar = utc_to_local_solar_time_pvlib(times_utc, longitude)

    # Sun position (pvlib expects localized times; uses UTC)
    solpos = solarposition.get_solarposition(times_utc, latitude, longitude)
    azimuth = solpos["azimuth"].values
    zenith = solpos["zenith"].values

    day_of_year = local_solar.dayofyear.values
    hour_of_day = (
        local_solar.hour.values
        + local_solar.minute.values / 60.0
        + local_solar.second.values / 3600.0
    )

    return [
        {
            "local_solar_time": local_solar[i].floor("us").to_pydatetime(),
            "azimuth": float(azimuth[i]),
            "zenith": float(zenith[i]),
            "day_of_year": int(day_of_year[i]),
            "hour_of_day": float(hour_of_day[i]),
        }
        for i in range(len(times_utc))
    ]


class PVDataset(Dataset):
    """
    Train: ``__len__`` = number of CSVs; ``__getitem__`` picks a **random** anchor where Y has at
    least one ``inverter_state == VALID_STATE``.

    Test (``split="test"``): ``__len__`` = ``len(sample_files) * num_test_windows``; anchors are
    fixed on a grid every ``test_anchor_stride_min`` minutes along the CSV row axis (no Y validity
    filter). All files with the same row count share the same ``num_test_windows``.

    Sky images (under ``data_dir`` / ``sky_image_{train,test}_path``, ``YYYYMMDDHHMMSS_12.jpg``): history sequence
    ends at the last X ``collectTime``; forecast sequence starts at the first Y ``collectTime``;
    step size is ``skyimg_time_resolution_min`` (independent of PV CSV spacing).
    Missing files → black ``(3, H, W)`` tensors resized to ``training.skyimg_spatial_size``.

    Himawari NPY (``staimg``): names ``NC_H09_YYYYMMDD_HHMM_L2CLP010_FLDK.02401_02401.npy`` with
    ``YYYYMMDD_HHMM`` in UTC; ``collectTime`` is interpreted as Asia/Shanghai local, converted to UTC,
    then floored to 10-minute boundaries for the key timestep. History/forecast windows step in UTC
    (sizes from ``conf.yaml`` ``training``). Arrays are float32 HWC per ``training.staimg_npy_shape_hwc``;
    missing files → zeros.
    """

    def __init__(
        self,
        pv_dir: str,
        skyimg_dir: str,
        staimg_dir: str,
        *,
        split: str = "train",
        csv_interval_min: int = _TRAINING_HPARAM_DEFAULTS["csv_interval_min"],
        pv_input_interval_min: int = _TRAINING_HPARAM_DEFAULTS["pv_input_interval_min"],
        pv_input_len: int = _TRAINING_HPARAM_DEFAULTS["pv_input_len"],
        pv_output_interval_min: int = _TRAINING_HPARAM_DEFAULTS["pv_output_interval_min"],
        pv_output_len: int = _TRAINING_HPARAM_DEFAULTS["pv_output_len"],
        test_anchor_stride_min: int = _TRAINING_HPARAM_DEFAULTS["test_anchor_stride_min"],
        skyimg_window_size: int = _TRAINING_HPARAM_DEFAULTS["skyimg_window_size"],
        skyimg_time_resolution_min: int = _TRAINING_HPARAM_DEFAULTS["skyimg_time_resolution_min"],
        skyimg_spatial_size: int = _TRAINING_HPARAM_DEFAULTS["skyimg_spatial_size"],
        staimg_window_size: int = _TRAINING_HPARAM_DEFAULTS["staimg_window_size"],
        staimg_time_resolution_min: int = _TRAINING_HPARAM_DEFAULTS["staimg_time_resolution_min"],
        staimg_npy_shape_hwc: tuple[int, int, int] = _TRAINING_HPARAM_DEFAULTS["staimg_npy_shape_hwc"],
    ):
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        self.split = split
        if skyimg_window_size < 1:
            raise ValueError("skyimg_window_size must be >= 1")
        self.skyimg_window_size = skyimg_window_size
        if staimg_window_size < 1:
            raise ValueError("staimg_window_size must be >= 1")
        self.staimg_window_size = staimg_window_size
        if csv_interval_min <= 0 or pv_input_interval_min % csv_interval_min:
            raise ValueError("pv_input_interval_min must be a positive multiple of csv_interval_min")
        if pv_output_interval_min % csv_interval_min:
            raise ValueError("pv_output_interval_min must be a positive multiple of csv_interval_min")
        self._sx = pv_input_interval_min // csv_interval_min
        self._sy = pv_output_interval_min // csv_interval_min
        self.pv_input_len = pv_input_len
        self.pv_output_len = pv_output_len
        self.pv_output_interval_min = pv_output_interval_min
        if skyimg_time_resolution_min <= 0:
            raise ValueError("skyimg_time_resolution_min must be positive")
        self._skyimg_dt_min = skyimg_time_resolution_min
        self._skyimg_dir = Path(skyimg_dir).resolve()
        if skyimg_spatial_size < 1:
            raise ValueError("skyimg_spatial_size must be >= 1")
        self._skyimg_spatial_size = int(skyimg_spatial_size)
        if len(staimg_npy_shape_hwc) != 3 or any(x < 1 for x in staimg_npy_shape_hwc):
            raise ValueError("staimg_npy_shape_hwc must be three positive ints (H, W, C)")
        self._staimg_npy_shape_hwc = tuple(int(x) for x in staimg_npy_shape_hwc)
        if staimg_time_resolution_min <= 0:
            raise ValueError("staimg_time_resolution_min must be positive")
        self._staimg_dt_min = staimg_time_resolution_min
        self._staimg_dir = Path(staimg_dir).resolve()
        if test_anchor_stride_min <= 0 or test_anchor_stride_min % csv_interval_min:
            raise ValueError(
                "test_anchor_stride_min must be a positive multiple of csv_interval_min "
                f"(got {test_anchor_stride_min}, csv_interval_min={csv_interval_min})"
            )
        self._test_anchor_stride_rows = test_anchor_stride_min // csv_interval_min

        CONF_PATH = _PROJECT_ROOT / "config" / "conf.yaml"
        with open(CONF_PATH) as f:
            conf = yaml.safe_load(f)
        paths = get_resolved_paths(conf, _PROJECT_ROOT)
        pv_device_path = paths["pv_device_path"]

        site = conf.get("site", {})
        self.latitude = site.get("latitude")
        self.longitude = site.get("longitude")

        pv_device_df = pd.read_excel(pv_device_path)
        self.devDn_list = pv_device_df["devDn"].dropna().unique().tolist()

        self.sample_files = list_csv_files(0, 625, data_dir=pv_dir)
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

        n_anchors = len(anchors)
        self._test_r_indices = np.arange(
            0, n_anchors, self._test_anchor_stride_rows, dtype=np.intp
        )
        self._num_test_windows = int(self._test_r_indices.size)
        if self.split == "test" and self._num_test_windows == 0:
            raise RuntimeError("split=test: stride leaves no anchor indices (increase data or reduce stride)")

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
                if inv[self._y_idx_per_anchor].any(axis=1).any():
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

    def _stack_sky_frames(self, frame_times: list[pd.Timestamp]) -> torch.Tensor:
        return torch.stack(
            [self._load_sky_tensor(self._sky_jpg_path(t)) for t in frame_times], dim=0
        )

    @staticmethod
    def _staimg_floor_utc_10min(ts_utc_naive: pd.Timestamp) -> pd.Timestamp:
        u = pd.Timestamp(ts_utc_naive)
        m = (int(u.minute) // 10) * 10
        return u.replace(minute=m, second=0, microsecond=0, nanosecond=0)

    def _staimg_utc_key_time(self, ts_raw) -> pd.Timestamp:
        """UTC naive timestamp floored to 10 min; derived from local collectTime (Asia/Shanghai)."""
        local_naive = self._sky_collect_ts_local(ts_raw)
        # ambiguous=True: older pandas rejects ambiguous="infer" on Timestamp.tz_localize;
        # Asia/Shanghai has no DST so wall times are not ambiguous.
        loc = local_naive.tz_localize("Asia/Shanghai", ambiguous=True)
        u = loc.tz_convert("UTC").tz_localize(None)
        return self._staimg_floor_utc_10min(u)

    def _staimg_npy_path(self, t_utc_naive: pd.Timestamp) -> Path:
        u = self._staimg_floor_utc_10min(t_utc_naive)
        stem = (
            f"NC_H09_{u.strftime('%Y%m%d')}_{u.strftime('%H%M')}_L2CLP010_FLDK.02401_02401"
        )
        return self._staimg_dir / f"{stem}.npy"

    def _dummy_staimg_tensor(self) -> torch.Tensor:
        h, w, c = self._staimg_npy_shape_hwc
        return torch.zeros((c, h, w), dtype=torch.float32)

    def _load_staimg_tensor(self, path: Path) -> torch.Tensor:
        try:
            if path.is_file():
                arr = np.load(path, allow_pickle=False)
                if arr.shape == self._staimg_npy_shape_hwc and arr.dtype == np.float32:
                    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        except Exception:
            pass
        return self._dummy_staimg_tensor()

    def _history_staimg_frame_utc_times(self, t_end_raw) -> list[pd.Timestamp]:
        t_end_utc = self._staimg_utc_key_time(t_end_raw)
        w = self.staimg_window_size
        return [
            self._staimg_floor_utc_10min(
                t_end_utc - timedelta(minutes=(w - 1 - i) * self._staimg_dt_min)
            )
            for i in range(w)
        ]

    def _forecast_staimg_frame_utc_times(self, t_start_raw) -> list[pd.Timestamp]:
        t0_utc = self._staimg_utc_key_time(t_start_raw)
        w = self.staimg_window_size
        return [
            self._staimg_floor_utc_10min(t0_utc + timedelta(minutes=i * self._staimg_dt_min))
            for i in range(w)
        ]

    def _stack_staimg_frames(self, frame_utc_times: list[pd.Timestamp]) -> torch.Tensor:
        return torch.stack([self._load_staimg_tensor(self._staimg_npy_path(t)) for t in frame_utc_times], dim=0)

    def _build_sample(self, df: pd.DataFrame, dev_idx: torch.Tensor, r: int) -> dict:
        x_idx = self._x_idx_per_anchor[r]
        y_idx_1d = self._y_idx_per_anchor[r]
        sub_x = df.iloc[x_idx]
        sub_y = df.iloc[y_idx_1d]

        ts_beijing = ZoneInfo("Asia/Shanghai")
        timestamps = []
        for t in sub_x["collectTime"]:
            ts = pd.Timestamp(t)
            if ts.tz is None:
                dt = ts.to_pydatetime().replace(tzinfo=ts_beijing)
            else:
                dt = ts.astimezone(ts_beijing).to_pydatetime()
            timestamps.append(dt)

        inv_x = pd.to_numeric(sub_x[INVERTER_STATE_COL], errors="coerce").fillna(0).astype(np.int32).values
        pow_x = pd.to_numeric(sub_x["active_power"], errors="coerce").fillna(0).values.astype(np.float32) / 50.0
        pv_mask = torch.from_numpy((inv_x == VALID_STATE).astype(np.float32)).unsqueeze(0)
        pv = torch.from_numpy(pow_x.astype(np.float32)).unsqueeze(0)

        pv_solar_features = compute_solar_features(timestamps, self.latitude, self.longitude)
        pv_solar_features_array = []
        for pv_solar_feature in pv_solar_features:
            azimuth_rad = np.deg2rad(pv_solar_feature["azimuth"])
            sin_azimuth = np.sin(azimuth_rad)
            cos_azimuth = np.cos(azimuth_rad)
            zenith_rad = np.deg2rad(pv_solar_feature["zenith"])
            cos_zenith = np.cos(zenith_rad)
            sin_zenith = np.sin(zenith_rad)
            cos_hod = np.cos(2 * np.pi * pv_solar_feature["hour_of_day"] / 24)
            sin_hod = np.sin(2 * np.pi * pv_solar_feature["hour_of_day"] / 24)
            pv_solar_features_array.append(
                np.array([sin_azimuth, cos_azimuth, sin_zenith, cos_zenith, sin_hod, cos_hod])
            )
        history_solar_features = torch.from_numpy(np.asarray(pv_solar_features_array, dtype=np.float32))

        time0 = timestamps[-1]
        forecast_timestamps_utc = [
            time0 + timedelta(minutes=self.pv_output_interval_min * (i + 1))
            for i in range(self.pv_output_len)
        ]
        solar_features = compute_solar_features(forecast_timestamps_utc, self.latitude, self.longitude)
        solar_features_array = []
        for solar_feature in solar_features:
            azimuth_rad = np.deg2rad(solar_feature["azimuth"])
            sin_azimuth = np.sin(azimuth_rad)
            cos_azimuth = np.cos(azimuth_rad)
            zenith_rad = np.deg2rad(solar_feature["zenith"])
            cos_zenith = np.cos(zenith_rad)
            sin_zenith = np.sin(zenith_rad)
            cos_dofy = np.cos(2 * np.pi * solar_feature["day_of_year"] / 366)
            sin_dofy = np.sin(2 * np.pi * solar_feature["day_of_year"] / 366)
            cos_hod = np.cos(2 * np.pi * solar_feature["hour_of_day"] / 24)
            sin_hod = np.sin(2 * np.pi * solar_feature["hour_of_day"] / 24)
            solar_features_array.append(
                np.array(
                    [sin_azimuth, cos_azimuth, sin_zenith, cos_zenith, sin_dofy, cos_dofy, sin_hod, cos_hod]
                )
            )
        forecast_solar_features = torch.from_numpy(np.asarray(solar_features_array, dtype=np.float32))

        inv_y = pd.to_numeric(sub_y[INVERTER_STATE_COL], errors="coerce").fillna(0).astype(np.int32).values
        pow_y = pd.to_numeric(sub_y["active_power"], errors="coerce").fillna(0).values.astype(np.float32)
        target_pv = torch.from_numpy((pow_y / 50.0).astype(np.float32))
        target_mask = torch.from_numpy((inv_y == VALID_STATE).astype(np.float32))

        t_x_end = sub_x["collectTime"].iloc[-1]
        t_y0 = sub_y["collectTime"].iloc[0]
        history_frame_times = self._history_sky_frame_times(t_x_end)
        forecast_frame_times = self._forecast_sky_frame_times(t_y0)
        history_skyimg = self._stack_sky_frames(history_frame_times)
        forecast_skyimg = self._stack_sky_frames(forecast_frame_times)
        history_sky_ts = [t.strftime("%Y%m%d%H%M%S") for t in history_frame_times]
        forecast_sky_ts = [t.strftime("%Y%m%d%H%M%S") for t in forecast_frame_times]

        history_staimg_utc = self._history_staimg_frame_utc_times(t_x_end)
        forecast_staimg_utc = self._forecast_staimg_frame_utc_times(t_y0)
        history_staimg = self._stack_staimg_frames(history_staimg_utc)
        forecast_staimg = self._stack_staimg_frames(forecast_staimg_utc)
        history_staimg_ts = [t.strftime("%Y%m%d%H%M") for t in history_staimg_utc]
        forecast_staimg_ts = [t.strftime("%Y%m%d%H%M") for t in forecast_staimg_utc]

        return {
            "dev_idx": dev_idx,
            "pv": pv,
            "pv_mask": pv_mask,
            "history_solar_features": history_solar_features,
            "forecast_solar_features": forecast_solar_features,
            "target_pv": target_pv,
            "target_mask": target_mask,
            "history_skyimg": history_skyimg,
            "history_staimg": history_staimg,
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
            valid_mask = inv[self._y_idx_per_anchor].any(axis=1)
            valid_rows = np.nonzero(valid_mask)[0]
            if valid_rows.size == 0:
                raise RuntimeError(
                    f"{sample_path.name}: no anchor with at least one Y row where "
                    f"{INVERTER_STATE_COL}=={VALID_STATE}"
                )
            r = int(np.random.choice(valid_rows))
        else:
            assert r_fixed is not None
            r = r_fixed

        return self._build_sample(df, dev_idx, r)


def collate_batched(batch):
    """Stack list of samples into one dict of tensors with batch dim B in front."""
    return {
        "dev_idx": torch.stack([s["dev_idx"] for s in batch]),
        "pv": torch.stack([s["pv"] for s in batch]),
        "pv_mask": torch.stack([s["pv_mask"] for s in batch]),
        "history_solar_features": torch.stack([s["history_solar_features"] for s in batch]),
        "forecast_solar_features": torch.stack([s["forecast_solar_features"] for s in batch]),
        "target_pv": torch.stack([s["target_pv"] for s in batch]),
        "target_mask": torch.stack([s["target_mask"] for s in batch]),
        "history_skyimg": torch.stack([s["history_skyimg"] for s in batch]),
        "history_staimg": torch.stack([s["history_staimg"] for s in batch]),
    }


def train_one_epoch(model, device, loader, criterion, optimizer, max_batches: int | None = None):
    model.train()
    total_loss = 0.0
    n = 0
    num_batches = len(loader)
    print("number of batches: ", num_batches)
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        B = batch["dev_idx"].size(0)
        device_id = batch["dev_idx"].to(device)            # [B]
        pv = batch["pv"].to(device)                        # [B, C, T_in]
        mask = batch["pv_mask"].to(device)                 # [B, C, T_in]
        history_solar_features = batch["history_solar_features"].to(device)
        forecast_solar_features = batch["forecast_solar_features"].to(device)
        target_pv = batch["target_pv"].to(device)
        target_mask = batch["target_mask"].to(device)

        optimizer.zero_grad()
        pv_pred = model(device_id, pv, mask, history_solar_features, forecast_solar_features)
        loss = criterion(pv_pred * target_mask, target_pv * target_mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += B
    print()
    return total_loss / max(n, 1)


def evaluate(model, device, loader, criterion):
    """Compute mean loss on a dataset (e.g. test set). No gradient."""
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            B = batch["dev_idx"].size(0)
            device_id = batch["dev_idx"].to(device)
            pv = batch["pv"].to(device)
            mask = batch["pv_mask"].to(device)
            history_solar_features = batch["history_solar_features"].to(device)
            forecast_solar_features = batch["forecast_solar_features"].to(device)
            target_pv = batch["target_pv"].to(device)
            target_mask = batch["target_mask"].to(device)
            pv_pred = model(device_id, pv, mask, history_solar_features, forecast_solar_features)
            loss = criterion(pv_pred * target_mask, target_pv * target_mask)
            total_loss += loss.item()
            n += B
    return total_loss / max(n, 1)


def main():
    h = _TRAINING_HPARAM_DEFAULTS
    parser = argparse.ArgumentParser(description="Train PV forecasting model")
    parser.add_argument("--epochs", type=int, default=h["epochs"])
    parser.add_argument("--lr", type=float, default=h["lr"])
    parser.add_argument("--batch_size", type=int, default=h["batch_size"])
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=h["save_every"])
    parser.add_argument(
        "--pv_train_dir",
        type=str,
        default=_TRAINING_PATH_DEFAULTS["pv_train_dir"],
        help=f"Training CSV directory (default from conf: {_TRAINING_PATH_DEFAULTS['pv_train_dir']!r}).",
    )
    parser.add_argument(
        "--pv_test_dir",
        type=str,
        default=_TRAINING_PATH_DEFAULTS["pv_test_dir"],
        help=f"Eval/test CSV directory (default from conf: {_TRAINING_PATH_DEFAULTS['pv_test_dir']!r}).",
    )
    parser.add_argument(
        "--skyimg_train_dir",
        type=str,
        default=_TRAINING_PATH_DEFAULTS["skyimg_train_dir"],
        help=f"Training skyimg directory (default from conf: {_TRAINING_PATH_DEFAULTS['skyimg_train_dir']!r}).",
    )
    parser.add_argument(
        "--skyimg_test_dir",
        type=str,
        default=_TRAINING_PATH_DEFAULTS["skyimg_test_dir"],
        help=f"Eval/test skyimg directory (default from conf: {_TRAINING_PATH_DEFAULTS['skyimg_test_dir']!r}).",
    )
    parser.add_argument(
        "--csv_interval_min",
        type=int,
        default=h["csv_interval_min"],
        help="CSV row spacing in minutes (must divide pv_input/output intervals).",
    )
    parser.add_argument(
        "--pv_input_interval_min",
        type=int,
        default=h["pv_input_interval_min"],
        help="Minutes between consecutive PV input (X) samples.",
    )
    parser.add_argument(
        "--pv_output_interval_min",
        type=int,
        default=h["pv_output_interval_min"],
        help="Minutes between consecutive PV target (Y) samples.",
    )
    parser.add_argument("--pv_input_len", type=int, default=h["pv_input_len"], help="Input sequence length (X).")
    parser.add_argument("--pv_output_len", type=int, default=h["pv_output_len"], help="Target sequence length (Y).")
    parser.add_argument(
        "--test_anchor_stride_min",
        type=int,
        default=h["test_anchor_stride_min"],
        help="For split=test: minutes between consecutive eval anchors (multiple of CSV row interval).",
    )
    parser.add_argument(
        "--skyimg_window_size",
        type=int,
        default=h["skyimg_window_size"],
        help="Number of sky images per history and per forecast sequence.",
    )
    parser.add_argument(
        "--skyimg_time_resolution_min",
        type=int,
        default=h["skyimg_time_resolution_min"],
        help="Minutes between consecutive sky frames (independent of PV input spacing).",
    )
    parser.add_argument(
        "--skyimg_spatial_size",
        type=int,
        default=h["skyimg_spatial_size"],
        help="Sky JPEG resize side length (square, pixels).",
    )
    parser.add_argument(
        "--staimg_train_dir",
        type=str,
        default=_TRAINING_PATH_DEFAULTS["staimg_train_dir"],
        help=f"Himawari NPY train dir (default from conf: {_TRAINING_PATH_DEFAULTS['staimg_train_dir']!r}).",
    )
    parser.add_argument(
        "--staimg_test_dir",
        type=str,
        default=_TRAINING_PATH_DEFAULTS["staimg_test_dir"],
        help=f"Himawari NPY test dir (default from conf: {_TRAINING_PATH_DEFAULTS['staimg_test_dir']!r}).",
    )
    parser.add_argument(
        "--staimg_window_size",
        type=int,
        default=h["staimg_window_size"],
        help="Number of Himawari NPY frames per history and per forecast sequence.",
    )
    parser.add_argument(
        "--staimg_time_resolution_min",
        type=int,
        default=h["staimg_time_resolution_min"],
        help="Minutes between consecutive staimg frames in UTC.",
    )
    parser.add_argument(
        "--staimg_npy_shape_hwc",
        type=int,
        nargs=3,
        default=list(h["staimg_npy_shape_hwc"]),
        metavar=("H", "W", "C"),
        help="Expected Himawari NPY array shape H W C (default from conf).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=h["num_workers"],
        help="DataLoader worker processes.",
    )
    parser.add_argument(
        "--train_max_batches_per_epoch",
        type=int,
        default=h["train_max_batches_per_epoch"],
        help="Stop each training epoch after this many batches (default from conf; null = no cap).",
    )
    args = parser.parse_args()
    staimg_hwc = tuple(args.staimg_npy_shape_hwc)

    conf = load_config()
    paths = get_resolved_paths(conf, _PROJECT_ROOT)
    pv_device_path = paths["pv_device_path"]
    if pv_device_path is None or not pv_device_path.is_file():
        raise FileNotFoundError(f"pv_device_path not found: {pv_device_path}")
    pv_device_df = pd.read_excel(pv_device_path)
    dev_dn_list = pv_device_df["devDn"].dropna().unique().tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = pv_forecasting_model(out_dim=64, dev_dn_list=dev_dn_list).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    train_dataset = PVDataset(
        pv_dir=args.pv_train_dir,
        skyimg_dir=args.skyimg_train_dir,
        staimg_dir=args.staimg_train_dir,
        split="train",
        csv_interval_min=args.csv_interval_min,
        pv_input_interval_min=args.pv_input_interval_min,
        pv_input_len=args.pv_input_len,
        pv_output_interval_min=args.pv_output_interval_min,
        pv_output_len=args.pv_output_len,
        test_anchor_stride_min=args.test_anchor_stride_min,
        skyimg_window_size=args.skyimg_window_size,
        skyimg_time_resolution_min=args.skyimg_time_resolution_min,
        skyimg_spatial_size=args.skyimg_spatial_size,
        staimg_window_size=args.staimg_window_size,
        staimg_time_resolution_min=args.staimg_time_resolution_min,
        staimg_npy_shape_hwc=staimg_hwc,
    )
    test_dataset = PVDataset(
        pv_dir=args.pv_test_dir,
        skyimg_dir=args.skyimg_test_dir,
        staimg_dir=args.staimg_test_dir,
        split="test",
        csv_interval_min=args.csv_interval_min,
        pv_input_interval_min=args.pv_input_interval_min,
        pv_input_len=args.pv_input_len,
        pv_output_interval_min=args.pv_output_interval_min,
        pv_output_len=args.pv_output_len,
        test_anchor_stride_min=args.test_anchor_stride_min,
        skyimg_window_size=args.skyimg_window_size,
        skyimg_time_resolution_min=args.skyimg_time_resolution_min,
        skyimg_spatial_size=args.skyimg_spatial_size,
        staimg_window_size=args.staimg_window_size,
        staimg_time_resolution_min=args.staimg_time_resolution_min,
        staimg_npy_shape_hwc=staimg_hwc,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batched,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batched,
        num_workers=args.num_workers,
    )

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else _PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    initial_test_loss = evaluate(model, device, test_loader, criterion)
    print(f"Initial test loss: {initial_test_loss:.6f}")

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(
            model,
            device,
            train_loader,
            criterion,
            optimizer,
            max_batches=args.train_max_batches_per_epoch,
        )
        test_loss = evaluate(model, device, test_loader, criterion)
        print(f"Epoch {epoch}/{args.epochs}  train_loss={avg_loss:.6f}  test_loss={test_loss:.6f}")

        if args.save_every and epoch % args.save_every == 0:
            path = checkpoint_dir / f"pv_forecast_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "dev_dn_list": dev_dn_list,
            }, path)
            print(f"  saved {path}")

    final_path = checkpoint_dir / "pv_forecast_final.pt"
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "dev_dn_list": dev_dn_list,
    }, final_path)
    print(f"Saved final checkpoint to {final_path}")


if __name__ == "__main__":
    main()
