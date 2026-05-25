"""
YLJ dataloader: Parquet PV matrices + optional Himawari Zarr (``dataloader/ylj_zarr.py``).

PV from fixed-length matrix columns (``observe_power`` / ``observe_power_future``).
Opt-in via ``--ylj_raw_parquet`` in training. Optional NWP: ``--ylj_parquet_nwp``.
Optional satellite Zarr (Luoyang window): ``--ylj_sat_zarr``.
``training.pv_value_column``: ``active_power`` (raw power / ``ylj_raw_parquet.pv_value_scale``) or
``clear_sky_ratio`` (power / GHI from ``ghi_csv``; train ``ghi_col_train``, test ``ghi_col_test``).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from modules.solar_encoder import compute_solar_features, delta_time_encoder, solar_features_encoder

# Keep aligned with ``dataloader.luoyang.VALID_STATE``; avoid importing luoyang (loads config at import).
VALID_STATE = 512

# Luoyang Zarr satellite window (``dataloader/luoyang_zarr.py``).
_SAT_FRAMES = 24
_SAT_MIN_BEFORE_ANCHOR = 245 + 30
_SAT_MIN_AFTER_ANCHOR = 30

_PARQUET_NWP_SSRD_COL = "ssrd_100_55_29_95_predict"
_PARQUET_NWP_T2M_COL = "t2m_100_6_29_9_predict"
_GHI_COL_DEFAULT_TRAIN = "GHI_real"
_GHI_COL_DEFAULT_TEST = "GHI_clear_sky"
_POWER_COL = "Power"
_TRAIN_MAX_RESAMPLE_TRIES = 3200


@dataclass(frozen=True)
class YljRawParquetMatrixConfig:
    """Parquet matrix layout; YAML via :func:`ylj_raw_parquet_matrix_config_from_conf`."""

    hist_len: int = 672
    fut_len: int = 192
    native_interval_min: int = 15
    naive_tz: str = "Asia/Shanghai"
    train_parquet: str = "ds_v322_2024.parquet"
    test_parquet: str = "ds_v322_1219_2025_1-12.parquet"
    pv_value_scale: float = 50.0
    ghi_csv: str = "processed_2024_2025.csv"
    ghi_col_train: str = "GHI_real"
    ghi_col_test: str = "GHI_clear_sky"


_DEFAULT_MATRIX_CONFIG = YljRawParquetMatrixConfig()


def ylj_raw_parquet_matrix_config_from_conf(conf: dict) -> YljRawParquetMatrixConfig:
    """Load ``conf['ylj_raw_parquet']``; missing keys use :data:`_DEFAULT_MATRIX_CONFIG`."""
    raw = conf.get("ylj_raw_parquet")
    if not isinstance(raw, dict):
        return _DEFAULT_MATRIX_CONFIG

    def _ig(key: str, default: int) -> int:
        if key not in raw:
            return default
        return int(raw[key])

    def _fg(key: str, default: float) -> float:
        if key not in raw:
            return default
        return float(raw[key])

    def _sg(key: str, default: str) -> str:
        if key not in raw:
            return default
        return str(raw[key]).strip()

    d = _DEFAULT_MATRIX_CONFIG
    cfg = YljRawParquetMatrixConfig(
        hist_len=_ig("hist_len", d.hist_len),
        fut_len=_ig("fut_len", d.fut_len),
        native_interval_min=_ig("native_interval_min", d.native_interval_min),
        naive_tz=_sg("naive_tz", d.naive_tz),
        train_parquet=_sg("train_parquet", d.train_parquet),
        test_parquet=_sg("test_parquet", d.test_parquet),
        pv_value_scale=_fg("pv_value_scale", d.pv_value_scale),
        ghi_csv=_sg("ghi_csv", d.ghi_csv),
        ghi_col_train=_sg("ghi_col_train", d.ghi_col_train),
        ghi_col_test=_sg("ghi_col_test", d.ghi_col_test),
    )
    if cfg.hist_len < 1 or cfg.fut_len < 1 or cfg.native_interval_min < 1:
        raise ValueError(f"ylj_raw_parquet: hist_len, fut_len, native_interval_min must be >= 1 (got {cfg})")
    if not cfg.naive_tz:
        raise ValueError("ylj_raw_parquet.naive_tz must be non-empty")
    if not cfg.ghi_csv:
        raise ValueError("ylj_raw_parquet.ghi_csv must be non-empty")
    return cfg


def load_satellite_from_zarr(
    sat_ds,
    time0_utc: pd.Timestamp,
    *,
    include_doy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Himawari Zarr slice aligned to Luoyang: ``[time0 - 275min, time0 - 30min]`` UTC,
    pad/trim to 24 frames, solar timefeats without DOY when ``include_doy=False``.

    Returns:
        ``sat_tensor`` ``[T=24, C=3, H, W]``, ``sat_timefeats`` ``[T=24, F]`` (F=7 for YLJ).
    """
    t0 = pd.Timestamp(time0_utc)
    if t0.tzinfo is None:
        t0 = t0.tz_localize("UTC")
    else:
        t0 = t0.tz_convert("UTC")

    sat_t0 = t0 - timedelta(minutes=_SAT_MIN_BEFORE_ANCHOR)
    sat_t1 = t0 - timedelta(minutes=_SAT_MIN_AFTER_ANCHOR)
    sat_t0 = pd.Timestamp(sat_t0).tz_convert("UTC").tz_localize(None)
    sat_t1 = pd.Timestamp(sat_t1).tz_convert("UTC").tz_localize(None)
    sat_data = sat_ds.sel(time_utc=slice(sat_t0, sat_t1))

    sat_solar_features = {
        "azimuth": sat_data["azimuth"].values,
        "zenith": sat_data["zenith"].values,
        "day_of_year": sat_data["day_of_year"].values,
        "hour_of_day": sat_data["hour_of_day"].values,
    }
    sat_timestamps_utc = sat_data["time_utc"].values
    time0_naive = t0.tz_convert("UTC").tz_localize(None)

    sat_timefeats = solar_features_encoder(sat_solar_features, include_doy=include_doy)
    sat_dtimefeats = delta_time_encoder(sat_timestamps_utc, time0_naive)
    sat_timefeats = torch.cat([sat_timefeats, sat_dtimefeats.unsqueeze(1)], dim=1)

    sat_tensor = torch.from_numpy(np.asarray(sat_data["images"].values, dtype=np.float32))
    if sat_tensor.ndim == 4 and sat_tensor.shape[-1] == 3:
        sat_tensor = sat_tensor.permute(0, 3, 1, 2).contiguous()

    if sat_tensor.shape[0] > _SAT_FRAMES:
        sat_tensor = sat_tensor[-_SAT_FRAMES:, :, :, :]
        sat_timefeats = sat_timefeats[-_SAT_FRAMES:, :]
    elif sat_tensor.shape[0] < _SAT_FRAMES:
        n_pad = _SAT_FRAMES - sat_tensor.shape[0]
        if sat_tensor.numel() == 0:
            h, w = 100, 100
            sat_tensor = torch.zeros(_SAT_FRAMES, 3, h, w, dtype=torch.float32)
            sat_timefeats = torch.zeros(_SAT_FRAMES, sat_timefeats.shape[1], dtype=torch.float32)
        else:
            z = torch.zeros(n_pad, *sat_tensor.shape[1:], dtype=sat_tensor.dtype)
            sat_tensor = torch.cat([z, sat_tensor], dim=0)
            zt = torch.zeros(n_pad, sat_timefeats.shape[1], dtype=sat_timefeats.dtype)
            sat_timefeats = torch.cat([zt, sat_timefeats], dim=0)

    return sat_tensor, sat_timefeats


def _utc_wall_naive(ts: pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC", ambiguous=True).tz_convert("UTC").tz_localize(None)
    return t.tz_convert("UTC").tz_localize(None)


def _ts_key_local(ts: pd.Timestamp) -> pd.Timestamp:
    """Naive local wall time for GHI CSV ``dtime`` lookup."""
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t.floor("us")


def _load_ghi_table(
    ghi_path: Path, *, ghi_col: str
) -> tuple[dict[pd.Timestamp, float], dict[pd.Timestamp, float]]:
    if not ghi_path.is_file():
        raise FileNotFoundError(f"GHI table not found: {ghi_path}")
    ghi_col = str(ghi_col).strip()
    if not ghi_col:
        raise ValueError("ghi_col must be non-empty")
    gdf = pd.read_csv(ghi_path, usecols=["dtime", ghi_col, _POWER_COL])
    gdf["dtime"] = pd.to_datetime(gdf["dtime"], errors="coerce")
    gdf = gdf.dropna(subset=["dtime"])
    ghi_map: dict[pd.Timestamp, float] = {}
    pw_map: dict[pd.Timestamp, float] = {}
    for t, g, p in zip(gdf["dtime"], gdf[ghi_col], gdf[_POWER_COL], strict=False):
        k = _ts_key_local(pd.Timestamp(t))
        if k not in ghi_map:
            ghi_map[k] = float(g)
            pw_map[k] = float(p)
    return ghi_map, pw_map


class YljRawParquetEmptyValDataset(Dataset):
    """Placeholder val split when training from YLJ raw Parquet (no val windows)."""

    def __init__(self) -> None:
        self.sample_files: list[Path] = []

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> dict:
        raise IndexError("YljRawParquetEmptyValDataset is empty")


class YljRawParquetDataset(Dataset):
    """
    One row = one sample. ``timestamp_win`` = last ``observe_power`` time (naive local).

    ``pv_value_column`` ``active_power``: raw power / ``ylj_raw_parquet.pv_value_scale``.
    ``clear_sky_ratio``: ``power / GHI`` at aligned local timestamps (column from matrix config); masks per step.
    """

    def __init__(
        self,
        raw_dir: str,
        *,
        split: str,
        pv_input_interval_min: int,
        pv_input_len: int,
        pv_output_interval_min: int,
        pv_output_len: int,
        latitude: float,
        longitude: float,
        dev_dn_index: int,
        matrix: YljRawParquetMatrixConfig | None = None,
        use_nwp: bool = False,
        use_sat_zarr: bool = False,
        sat_zarr_dir: str | Path | None = None,
        pv_value_column: str = "active_power",
        training_pv_value_scale: float = 1.0,
        clear_sky_ratio_max_valid: float = 1.5,
        include_export_metadata: bool = False,
    ) -> None:
        if split not in ("train", "test", "val"):
            raise ValueError("split must be 'train', 'val', or 'test'")
        if split == "val":
            raise ValueError("use YljRawParquetEmptyValDataset for val split")
        mx = matrix if matrix is not None else _DEFAULT_MATRIX_CONFIG
        self._hist_len = int(mx.hist_len)
        self._fut_len = int(mx.fut_len)
        self._native_min = int(mx.native_interval_min)
        self._naive_tz = str(mx.naive_tz)
        self._train_parquet = str(mx.train_parquet)
        self._test_parquet = str(mx.test_parquet)

        self.split = split
        self.pv_input_len = int(pv_input_len)
        self.pv_output_len = int(pv_output_len)
        self.pv_input_interval_min = int(pv_input_interval_min)
        self.pv_output_interval_min = int(pv_output_interval_min)
        self.latitude = float(latitude)
        self.longitude = float(longitude)
        self._use_nwp = bool(use_nwp)
        self._use_sat_zarr = bool(use_sat_zarr)
        self._sat_ds = None
        if self._use_sat_zarr:
            if sat_zarr_dir is None:
                raise ValueError("use_sat_zarr=True requires sat_zarr_dir")
            sat_root = Path(sat_zarr_dir).expanduser().resolve()
            if not sat_root.is_dir():
                raise FileNotFoundError(f"YLJ satellite Zarr directory not found: {sat_root}")
            import xarray as xr

            self._sat_ds = xr.open_zarr(sat_root)
            print(f"[YljRawParquetDataset] satellite Zarr: {sat_root}")
        self._include_export_metadata = bool(include_export_metadata)
        self.sample_files: list[Path] = []
        self.supports_single_horizon_test_only = False

        self._pv_value_col = str(pv_value_column).strip()
        self._ratio_mode = self._pv_value_col == "clear_sky_ratio"
        self._loss_on_power = self._ratio_mode
        self._ratio_max = float(clear_sky_ratio_max_valid)
        if self._ratio_mode:
            self._pv_value_scale = float(training_pv_value_scale)
            self._loss_power_scale = float(mx.pv_value_scale)
        else:
            self._pv_value_scale = float(mx.pv_value_scale)
            self._loss_power_scale = None

        nat = self._native_min
        in_stride = self.pv_input_interval_min // nat
        out_stride = self.pv_output_interval_min // nat
        if in_stride * nat != self.pv_input_interval_min or in_stride < 1:
            raise ValueError(f"pv_input_interval_min={self.pv_input_interval_min} must be a positive multiple of {nat}")
        if out_stride * nat != self.pv_output_interval_min or out_stride < 1:
            raise ValueError(
                f"pv_output_interval_min={self.pv_output_interval_min} must be a positive multiple of {nat}"
            )
        last_hist_ix = self._hist_len - 1
        first_hist_ix = last_hist_ix - (self.pv_input_len - 1) * in_stride
        if self.pv_input_len < 1 or first_hist_ix < 0:
            raise ValueError(
                f"pv_input_len={self.pv_input_len} with interval {self.pv_input_interval_min} min "
                f"does not fit in {self._hist_len} native steps"
            )
        if self.pv_output_len < 1:
            raise ValueError("pv_output_len must be >= 1")
        max_fut_ix = (self.pv_output_len - 1) * out_stride
        if max_fut_ix > self._fut_len - 1:
            raise ValueError(
                f"pv_output_len={self.pv_output_len} with interval {self.pv_output_interval_min} min "
                f"does not fit in {self._fut_len} future native steps"
            )

        self._in_stride = in_stride
        self._out_stride = out_stride

        root = Path(raw_dir).expanduser().resolve()
        fn = self._train_parquet if split == "train" else self._test_parquet
        path = root / fn
        if not path.is_file():
            raise FileNotFoundError(f"YLJ raw Parquet not found: {path}")

        read_cols = ["timestamp_win", "observe_power", "observe_power_future"]
        if self._use_nwp:
            read_cols.extend([_PARQUET_NWP_SSRD_COL, _PARQUET_NWP_T2M_COL])
        df = pd.read_parquet(path, columns=read_cols, engine="pyarrow")
        need = {"timestamp_win", "observe_power", "observe_power_future"}
        miss = need - set(df.columns)
        if miss:
            raise ValueError(f"{path.name}: missing column(s): {sorted(miss)}")
        if self._use_nwp:
            for c in (_PARQUET_NWP_SSRD_COL, _PARQUET_NWP_T2M_COL):
                if c not in df.columns:
                    raise ValueError(f"{path.name}: NWP enabled but missing column {c!r}")

        m = df["observe_power"].notna() & df["observe_power_future"].notna()
        df = df.loc[m].reset_index(drop=True)
        self._timestamp_win = pd.to_datetime(df["timestamp_win"], errors="coerce")
        if bool(self._timestamp_win.isna().any()):
            bad = int(self._timestamp_win.isna().sum())
            raise ValueError(f"{path.name}: timestamp_win has {bad} invalid values after list-null filter")

        self._observe_power = df["observe_power"].to_numpy()
        self._observe_power_future = df["observe_power_future"].to_numpy()
        if self._use_nwp:
            self._ssrd_future = df[_PARQUET_NWP_SSRD_COL].to_numpy()
            self._t2m_future = df[_PARQUET_NWP_T2M_COL].to_numpy()
            print(
                f"[YljRawParquetDataset] NWP from Parquet: {_PARQUET_NWP_SSRD_COL}, {_PARQUET_NWP_T2M_COL} "
                f"({path.name})"
            )
        else:
            self._ssrd_future = None
            self._t2m_future = None

        self._ghi_map: dict[pd.Timestamp, float] = {}
        self._power_map: dict[pd.Timestamp, float] = {}
        if self._ratio_mode:
            ghi_path = root / str(mx.ghi_csv)
            if split == "test":
                self._ghi_col = str(mx.ghi_col_test)
            else:
                self._ghi_col = str(mx.ghi_col_train)
            self._ghi_map, self._power_map = _load_ghi_table(ghi_path, ghi_col=self._ghi_col)
            print(
                f"[YljRawParquetDataset] clear_sky_ratio split={split} from {ghi_path.name} "
                f"(GHI={self._ghi_col}, scale={self._pv_value_scale}, max_valid={self._ratio_max})"
            )
        else:
            self._ghi_col = ""
            print(
                f"[YljRawParquetDataset] active_power mode (scale={self._pv_value_scale}) "
                f"({path.name})"
            )

        self._parquet_path = path
        self.dev_idx = torch.tensor(int(dev_dn_index), dtype=torch.long)

    def __len__(self) -> int:
        return int(len(self._timestamp_win))

    def _row_arrays(self, row: int) -> tuple[np.ndarray, np.ndarray, pd.Timestamp]:
        op = np.asarray(self._observe_power[row], dtype=np.float64).reshape(-1)
        of = np.asarray(self._observe_power_future[row], dtype=np.float64).reshape(-1)
        if op.shape != (self._hist_len,) or of.shape != (self._fut_len,):
            raise ValueError(
                f"{self._parquet_path.name} row {row}: expected observe_power len {self._hist_len} "
                f"and observe_power_future len {self._fut_len}, got {op.shape[0]} and {of.shape[0]}"
            )
        twin = pd.Timestamp(self._timestamp_win.iloc[row])
        return op, of, twin

    def _coerce_fut_vec(self, cell) -> np.ndarray:
        fl = self._fut_len
        if cell is None:
            return np.full((fl,), np.nan, dtype=np.float64)
        if np.isscalar(cell) and pd.isna(cell):
            return np.full((fl,), np.nan, dtype=np.float64)
        a = np.asarray(cell, dtype=np.float64).reshape(-1)
        if a.shape != (fl,):
            return np.full((fl,), np.nan, dtype=np.float64)
        return a

    def _t_win_utc(self, twin_naive: pd.Timestamp) -> pd.Timestamp:
        t = pd.Timestamp(twin_naive)
        if t.tzinfo is None:
            t = t.tz_localize(self._naive_tz, ambiguous=True)
        return t.tz_convert("UTC")

    def _ghi_at(self, t_local: pd.Timestamp) -> float | None:
        return self._ghi_map.get(_ts_key_local(t_local))

    def _power_at(self, t_local: pd.Timestamp) -> float | None:
        return self._power_map.get(_ts_key_local(t_local))

    def _ratio_and_mask(self, power: float, t_local: pd.Timestamp) -> tuple[float, float]:
        """Return (clear_sky_ratio, mask). Missing GHI -> (0, 0); night GHI=0 -> (0, 1)."""
        ghi = self._ghi_at(t_local)
        if ghi is None:
            return 0.0, 0.0
        g = float(ghi)
        if not np.isfinite(g) or g < 0.0:
            return 0.0, 0.0
        p = float(power)
        if not np.isfinite(p):
            return 0.0, 0.0
        if g == 0.0:
            return 0.0, 1.0
        r = p / g
        if not np.isfinite(r) or r > self._ratio_max + 1e-12:
            return 0.0, 0.0
        return float(r), 1.0

    def _build_sample(self, row: int) -> dict:
        op, of, twin_local = self._row_arrays(int(row))
        t_win_utc = self._t_win_utc(twin_local)
        t0u = _utc_wall_naive(t_win_utc)

        in_s = self._in_stride
        last_ix = self._hist_len - 1
        first_ix = last_ix - (self.pv_input_len - 1) * in_s
        hist_ix = np.arange(first_ix, last_ix + 1, in_s, dtype=np.int64)
        nat = self._native_min

        ts_x_local: list[pd.Timestamp] = []
        ts_x_utc: list[pd.Timestamp] = []
        for iix in hist_ix:
            minutes_before = int(last_ix - iix) * nat
            t_loc = pd.Timestamp(twin_local) - pd.Timedelta(minutes=minutes_before)
            ts_x_local.append(t_loc)
            ts_x_utc.append(t_win_utc - pd.Timedelta(minutes=minutes_before))

        out_s = self._out_stride
        fut_ix = np.arange(0, self.pv_output_len * out_s, out_s, dtype=np.int64)
        ts_y_local: list[pd.Timestamp] = []
        y_times_utc: list[pd.Timestamp] = []
        for jix in fut_ix:
            t_loc = pd.Timestamp(twin_local) + pd.Timedelta(minutes=nat * (int(jix) + 1))
            ts_y_local.append(t_loc)
            y_times_utc.append(t_win_utc + pd.Timedelta(minutes=nat * (int(jix) + 1)))

        sc = float(self._pv_value_scale)
        pv_vals = np.zeros(self.pv_input_len, dtype=np.float32)
        pv_m = np.zeros(self.pv_input_len, dtype=np.float32)
        pow_y = np.zeros(self.pv_output_len, dtype=np.float64)
        pow_y_raw = np.zeros(self.pv_output_len, dtype=np.float64)
        ghi_y = np.zeros(self.pv_output_len, dtype=np.float32)
        y_masks = np.zeros(self.pv_output_len, dtype=np.float32)

        if self._ratio_mode:
            for i, iix in enumerate(hist_ix):
                r, m = self._ratio_and_mask(float(op[iix]), ts_x_local[i])
                pv_vals[i] = np.float32(r / sc)
                pv_m[i] = m
            for i, jix in enumerate(fut_ix):
                p_raw = float(of[jix])
                r, m = self._ratio_and_mask(p_raw, ts_y_local[i])
                pow_y[i] = r
                pow_y_raw[i] = p_raw
                y_masks[i] = m
                ghi = self._ghi_at(ts_y_local[i])
                if ghi is not None and np.isfinite(ghi) and float(ghi) >= 0.0:
                    ghi_y[i] = np.float32(ghi)
        else:
            pw_x = op[hist_ix].astype(np.float32)
            pv_vals = pw_x / sc
            pv_m[:] = 1.0
            pow_y = of[fut_ix].astype(np.float64)
            y_masks[:] = 1.0

        xs_u = ts_x_utc
        pv = torch.from_numpy(pv_vals).unsqueeze(0)
        pv_tf = solar_features_encoder(
            compute_solar_features(xs_u, self.latitude, self.longitude),
            include_doy=False,
        )
        pv_dt = delta_time_encoder(xs_u, t0u)
        pv_timefeats = torch.cat([pv_tf, pv_dt.unsqueeze(1)], dim=1)

        y_u = y_times_utc
        fc_tf = solar_features_encoder(
            compute_solar_features(y_u, self.latitude, self.longitude),
            include_doy=False,
        )
        fc_dt = delta_time_encoder(y_u, t0u)
        forecast_timefeats = torch.cat([fc_tf, fc_dt.unsqueeze(1)], dim=1)

        sat_tensor: torch.Tensor | None = None
        sat_timefeats: torch.Tensor | None = None
        if self._use_sat_zarr and self._sat_ds is not None:
            sat_tensor, sat_timefeats = load_satellite_from_zarr(
                self._sat_ds, t_win_utc, include_doy=False
            )

        nwp_tensor: torch.Tensor | None = None
        if self._use_nwp:
            ssrd_full = self._coerce_fut_vec(self._ssrd_future[int(row)])
            t2m_full = self._coerce_fut_vec(self._t2m_future[int(row)])
            out_nwp = np.zeros((self.pv_output_len, 3), dtype=np.float32)
            for ti in range(self.pv_output_len):
                jix = int(fut_ix[ti])
                s = float(ssrd_full[jix])
                t = float(t2m_full[jix])
                if np.isfinite(s) and np.isfinite(t):
                    out_nwp[ti, 0] = np.float32(s)
                    out_nwp[ti, 1] = np.float32(t)
                    out_nwp[ti, 2] = 1.0
            nwp_tensor = torch.from_numpy(out_nwp)

        out: dict = {
            "dev_idx": self.dev_idx.clone(),
            "pv": pv,
            "pv_mask": torch.from_numpy(pv_m).unsqueeze(0),
            "pv_timefeats": pv_timefeats,
            "forecast_timefeats": forecast_timefeats,
            "target_pv": torch.tensor([p / sc for p in pow_y], dtype=torch.float32),
            "target_mask": torch.tensor(y_masks, dtype=torch.float32),
            "sat_tensor": sat_tensor,
            "sat_timefeats": sat_timefeats,
            "skimg_tensor": None,
            "skimg_timefeats": None,
            "nwp_tensor": nwp_tensor,
        }
        if self._ratio_mode:
            out["target_power"] = torch.tensor(pow_y_raw, dtype=torch.float32)
            out["y_ghi"] = torch.tensor(ghi_y, dtype=torch.float32)
        if self._include_export_metadata:
            t_utc = self._t_win_utc(twin_local)
            out["csv_collect_time_utc"] = t_utc.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S")
            if self._ratio_mode:
                out["target_timestamps_local"] = [
                    pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S") for t in ts_y_local
                ]
        return out

    def row_window_timestamps_utc(self, row: int) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
        """X/Y UTC timestamps for one row (debug CLI only; not in training samples)."""
        _, _, twin_local = self._row_arrays(int(row))
        t_win_utc = self._t_win_utc(twin_local)
        in_s = self._in_stride
        last_ix = self._hist_len - 1
        first_ix = last_ix - (self.pv_input_len - 1) * in_s
        hist_ix = np.arange(first_ix, last_ix + 1, in_s, dtype=np.int64)
        nat = self._native_min
        ts_x_utc = [
            t_win_utc - pd.Timedelta(minutes=int(last_ix - iix) * nat) for iix in hist_ix
        ]
        out_s = self._out_stride
        fut_ix = np.arange(0, self.pv_output_len * out_s, out_s, dtype=np.int64)
        y_times_utc = [
            t_win_utc + pd.Timedelta(minutes=nat * (int(jix) + 1)) for jix in fut_ix
        ]
        return ts_x_utc, y_times_utc

    def __getitem__(self, idx: int) -> dict:
        if self.split == "train" and self._ratio_mode:
            s_try = self._build_sample(int(idx))
            if float(s_try["target_mask"].sum().item()) > 0.0:
                return s_try
            n = len(self)
            rng = np.random.default_rng()
            for _ in range(_TRAIN_MAX_RESAMPLE_TRIES):
                row = int(rng.integers(0, n))
                s = self._build_sample(row)
                if float(s["target_mask"].sum().item()) > 0.0:
                    return s
            raise RuntimeError(
                f"YljRawParquetDataset: could not sample row with valid Y after "
                f"{_TRAIN_MAX_RESAMPLE_TRIES} tries (clear_sky_ratio mode)"
            )
        return self._build_sample(int(idx))


def collate_ylj_batched(batch: list[dict]) -> dict:
    """Stack YLJ Parquet samples; optional sat/sky/nwp may be None."""
    if not batch:
        raise ValueError("empty batch")

    out: dict = {
        "dev_idx": torch.stack([x["dev_idx"] for x in batch]),
        "pv": torch.stack([x["pv"] for x in batch]),
        "pv_mask": torch.stack([x["pv_mask"] for x in batch]),
        "pv_timefeats": torch.stack([x["pv_timefeats"] for x in batch]),
        "forecast_timefeats": torch.stack([x["forecast_timefeats"] for x in batch]),
        "target_pv": torch.stack([x["target_pv"] for x in batch]),
        "target_mask": torch.stack([x["target_mask"] for x in batch]),
    }
    if "csv_collect_time_utc" in batch[0]:
        out["csv_collect_time_utc"] = [x["csv_collect_time_utc"] for x in batch]
    if "target_timestamps_local" in batch[0]:
        out["target_timestamps_local"] = [x["target_timestamps_local"] for x in batch]
    for key in ("target_power", "y_ghi"):
        if key in batch[0]:
            out[key] = torch.stack([x[key] for x in batch])
    for key in ("sat_tensor", "sat_timefeats", "skimg_tensor", "skimg_timefeats", "nwp_tensor"):
        if batch[0][key] is None:
            if not all(x[key] is None for x in batch):
                raise ValueError(f"mixed None/non-None for {key}")
            out[key] = None
        else:
            out[key] = torch.stack([x[key] for x in batch])
    return out
