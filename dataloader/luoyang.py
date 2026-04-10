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
from torch.utils.data import DataLoader, Dataset
from modules.solar_encoder import compute_solar_features, solar_features_encoder, delta_time_encoder

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config_utils import get_resolved_paths
from training.training_conf import (
    CONF_PATH,
    get_training_hparams_from_conf,
    get_training_paths_from_conf,
    load_config,
)
from training.data_loader import (
    INVERTER_STATE_COL,
    VALID_STATE,
    list_csv_files,
    load_csv,
)


def _load_training_defaults() -> tuple[dict[str, str], dict]:
    conf = load_config()
    return get_training_paths_from_conf(conf, _PROJECT_ROOT), get_training_hparams_from_conf(conf)


_TRAINING_PATH_DEFAULTS, _TRAINING_HPARAM_DEFAULTS = _load_training_defaults()


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

    Himawari NPY (``satimg``): names ``NC_H09_YYYYMMDD_HHMM_L2CLP010_FLDK.02401_02401.npy`` with
    ``YYYYMMDD_HHMM`` in UTC; ``collectTime`` is interpreted as Asia/Shanghai local, converted to UTC,
    then floored to 10-minute boundaries for the key timestep. History/forecast windows step in UTC
    (sizes from ``conf.yaml`` ``training``). Arrays are float32 HWC per ``training.satimg_npy_shape_hwc``;
    missing files → zeros.
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
        test_anchor_stride_min: int = _TRAINING_HPARAM_DEFAULTS["test_anchor_stride_min"],
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
            if i>5:
                break
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

    def _build_sample(self, df: pd.DataFrame, dev_idx: torch.Tensor, r: int) -> dict:
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
            "nwp_tensor": None,         # TODO: add nwp features
            "nwp_timefeats": None,      # TODO: add nwp features
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
    for key in ("skimg_tensor", "skimg_timefeats", "nwp_tensor", "nwp_timefeats"):
        vals = [s[key] for s in batch]
        if vals[0] is None:
            if not all(v is None for v in vals):
                raise ValueError(f"collate_batched: mixed None and tensor for {key!r}")
            out[key] = None
        else:
            out[key] = torch.stack(vals)
    return out


def loader_test(
    *,
    pv_train_dir: str = _TRAINING_PATH_DEFAULTS["pv_train_dir"],
    pv_test_dir: str = _TRAINING_PATH_DEFAULTS["pv_test_dir"],
    skyimg_train_dir: str = _TRAINING_PATH_DEFAULTS["skyimg_train_dir"],
    skyimg_test_dir: str = _TRAINING_PATH_DEFAULTS["skyimg_test_dir"],
    satimg_train_dir: str = _TRAINING_PATH_DEFAULTS["satimg_train_dir"],
    satimg_test_dir: str = _TRAINING_PATH_DEFAULTS["satimg_test_dir"],
    csv_interval_min: int = _TRAINING_HPARAM_DEFAULTS["csv_interval_min"],
    pv_input_interval_min: int = _TRAINING_HPARAM_DEFAULTS["pv_input_interval_min"],
    pv_input_len: int = _TRAINING_HPARAM_DEFAULTS["pv_input_len"],
    pv_output_interval_min: int = _TRAINING_HPARAM_DEFAULTS["pv_output_interval_min"],
    pv_output_len: int = _TRAINING_HPARAM_DEFAULTS["pv_output_len"],
    test_anchor_stride_min: int = _TRAINING_HPARAM_DEFAULTS["test_anchor_stride_min"],
    skyimg_window_size: int = _TRAINING_HPARAM_DEFAULTS["skyimg_window_size"],
    skyimg_time_resolution_min: int = _TRAINING_HPARAM_DEFAULTS["skyimg_time_resolution_min"],
    skyimg_spatial_size: int = _TRAINING_HPARAM_DEFAULTS["skyimg_spatial_size"],
    satimg_window_size: int = _TRAINING_HPARAM_DEFAULTS["satimg_window_size"],
    satimg_time_resolution_min: int = _TRAINING_HPARAM_DEFAULTS["satimg_time_resolution_min"],
    satimg_npy_shape_hwc: tuple[int, int, int] = _TRAINING_HPARAM_DEFAULTS["satimg_npy_shape_hwc"],
    batch_size: int = _TRAINING_HPARAM_DEFAULTS["loader_test_batch_size"],
    epochs: int = 1,
    max_batches: int | None = None,
    num_workers: int = _TRAINING_HPARAM_DEFAULTS["loader_test_num_workers"],
) -> dict:
    """
    :class:`PVDataset` / ``DataLoader`` for ``train_data_dir`` and, if it differs from
    ``test_data_dir`` after resolving paths, a second loader for the test folder.
    """
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be >= 1 when set")

    pv_train_dir = str(Path(pv_train_dir).resolve())
    pv_test_dir = str(Path(pv_test_dir).resolve())
    skyimg_train_dir = str(Path(skyimg_train_dir).resolve())
    skyimg_test_dir = str(Path(skyimg_test_dir).resolve())
    satimg_train_dir = str(Path(satimg_train_dir).resolve())
    satimg_test_dir = str(Path(satimg_test_dir).resolve())

    train_dataset = PVDataset(
        pv_dir=pv_train_dir,
        skyimg_dir=skyimg_train_dir,
        satimg_dir=satimg_train_dir,
        split="train",
        csv_interval_min=csv_interval_min,
        pv_input_interval_min=pv_input_interval_min,
        pv_input_len=pv_input_len,
        pv_output_interval_min=pv_output_interval_min,
        pv_output_len=pv_output_len,
        test_anchor_stride_min=test_anchor_stride_min,
        skyimg_window_size=skyimg_window_size,
        skyimg_time_resolution_min=skyimg_time_resolution_min,
        skyimg_spatial_size=skyimg_spatial_size,
        satimg_window_size=satimg_window_size,
        satimg_time_resolution_min=satimg_time_resolution_min,
        satimg_npy_shape_hwc=satimg_npy_shape_hwc,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batched,
        num_workers=num_workers,
    )

    test_dataset: PVDataset | None = None
    test_loader: DataLoader | None = None
    if pv_test_dir != pv_train_dir:
        test_dataset = PVDataset(
            pv_dir=pv_test_dir,
            skyimg_dir=skyimg_test_dir,
            satimg_dir=satimg_test_dir,
            split="test",
            csv_interval_min=csv_interval_min,
            pv_input_interval_min=pv_input_interval_min,
            pv_input_len=pv_input_len,
            pv_output_interval_min=pv_output_interval_min,
            pv_output_len=pv_output_len,
            test_anchor_stride_min=test_anchor_stride_min,
            skyimg_window_size=skyimg_window_size,
            skyimg_time_resolution_min=skyimg_time_resolution_min,
            skyimg_spatial_size=skyimg_spatial_size,
            satimg_window_size=satimg_window_size,
            satimg_time_resolution_min=satimg_time_resolution_min,
            satimg_npy_shape_hwc=satimg_npy_shape_hwc,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_batched,
            num_workers=num_workers,
        )

    print(f"loader_test: train_data_dir={pv_train_dir!r}")
    print(f"  train_dataset: {len(train_dataset)} files  batches/epoch={len(train_loader)}  epochs={epochs}")
    if max_batches is not None:
        print(f"  max_batches/epoch={max_batches}")
    if test_loader is not None:
        assert test_dataset is not None
        print(f"loader_test: test_data_dir={pv_test_dir!r}")
        print(f"  test_dataset:  {len(test_dataset)} files  batches/epoch={len(test_loader)}")

    def _run_split(name: str, loader: DataLoader) -> None:
        for epoch in range(epochs):
            for batch_idx, batch in enumerate(loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                print(
                    f"  [{name}] epoch {epoch + 1}/{epochs} batch {batch_idx}: ",
                    end="",
                )
                tensor_keys = [k for k in batch if torch.is_tensor(batch[k])]
                parts = [f"{k} {tuple(batch[k].shape)} {batch[k].dtype}" for k in tensor_keys]
                none_keys = [k for k in batch if batch[k] is None]
                if none_keys:
                    parts.append(f"(None: {', '.join(none_keys)})")
                print(", ".join(parts))

    _run_split("train", train_loader)
    if test_loader is not None:
        _run_split("test", test_loader)

    out: dict = {
        "train_dataset": train_dataset,
        "train_loader": train_loader,
    }
    if test_dataset is not None and test_loader is not None:
        out["test_dataset"] = test_dataset
        out["test_loader"] = test_loader
    return out
