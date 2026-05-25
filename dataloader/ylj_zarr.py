"""
YLJ dataloader: Parquet PV matrices + optional Himawari Zarr (``dataloader/ylj_zarr.py``).

PV from fixed-length matrix columns (``observe_power`` / ``observe_power_future``).
Opt-in via ``--ylj_raw_parquet`` in training. Optional NWP: ``--ylj_parquet_nwp``.
Optional satellite Zarr (Luoyang window): ``--ylj_sat_zarr``.
PV targets: raw ``observe_power`` / ``training.pv_value_scale`` (kW). Model input uses ``kt`` from
``solar_features_csv`` (``train_ylj.py`` applies kt/20 and power reconstruction at train time).
Solar timefeats from ``solar_features_csv`` via ``extract_solar_features`` (missing times -> mask 0).
Also exposes ``kt``, ``kt_mask``, ``p_cs`` (history), ``p_mean``, ``target_p_cs`` (forecast), aligned with Luoyang Zarr.
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

from modules.solar_encoder import (
    delta_time_encoder,
    extract_solar_features,
    solar_features_encoder,
)

# Keep aligned with ``dataloader.luoyang.VALID_STATE``; avoid importing luoyang (loads config at import).
VALID_STATE = 512

# Luoyang Zarr satellite window (``dataloader/luoyang_zarr.py``).
_SAT_FRAMES = 24
_SAT_MIN_BEFORE_ANCHOR = 245 + 30
_SAT_MIN_AFTER_ANCHOR = 30

_PARQUET_NWP_SSRD_COL = "ssrd_100_55_29_95_predict"
_PARQUET_NWP_T2M_COL = "t2m_100_6_29_9_predict"
@dataclass(frozen=True)
class YljRawParquetMatrixConfig:
    """Parquet matrix layout; YAML via :func:`ylj_raw_parquet_matrix_config_from_conf`."""

    hist_len: int = 672
    fut_len: int = 192
    native_interval_min: int = 15
    naive_tz: str = "Asia/Shanghai"
    train_parquet: str = "ds_v322_2024.parquet"
    test_parquet: str = "ds_v322_1219_2025_1-12.parquet"
    solar_features_csv: str = "solar_features_ylj_2024_2025_15min.csv"


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
        solar_features_csv=_sg("solar_features_csv", d.solar_features_csv),
    )
    if cfg.hist_len < 1 or cfg.fut_len < 1 or cfg.native_interval_min < 1:
        raise ValueError(f"ylj_raw_parquet: hist_len, fut_len, native_interval_min must be >= 1 (got {cfg})")
    if not cfg.naive_tz:
        raise ValueError("ylj_raw_parquet.naive_tz must be non-empty")
    if not cfg.solar_features_csv:
        raise ValueError("ylj_raw_parquet.solar_features_csv must be non-empty")
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
    """Naive local wall time for solar CSV ``china_local_time`` lookup."""
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t.floor("us")


def _load_solar_table(solar_path: Path) -> dict[pd.Timestamp, dict[str, object]]:
    """Map naive China-local time -> row fields for :func:`extract_solar_features`."""
    if not solar_path.is_file():
        raise FileNotFoundError(f"solar features table not found: {solar_path}")
    usecols = [
        "china_local_time",
        "local_solar_time",
        "solar_azimuth",
        "solar_zenith",
        "day_of_year",
        "hour_of_day",
        "p_cs",
        "kt",
        "kt_mask",
        "p_mean",
    ]
    sdf = pd.read_csv(solar_path, usecols=usecols)
    sdf["china_local_time"] = pd.to_datetime(sdf["china_local_time"], errors="coerce")
    sdf = sdf.dropna(subset=["china_local_time"])
    solar_map: dict[pd.Timestamp, dict[str, object]] = {}
    for row in sdf.itertuples(index=False):
        k = _ts_key_local(pd.Timestamp(row.china_local_time))
        if k in solar_map:
            continue
        solar_map[k] = {
            "local_solar_time": str(row.local_solar_time),
            "solar_azimuth": float(row.solar_azimuth),
            "solar_zenith": float(row.solar_zenith),
            "day_of_year": int(row.day_of_year),
            "hour_of_day": float(row.hour_of_day),
            "p_cs": float(row.p_cs),
            "kt": float(row.kt),
            "kt_mask": float(row.kt_mask),
            "p_mean": float(row.p_mean),
        }
    return solar_map


def _solar_series_at_times(
    solar_map: dict[pd.Timestamp, dict[str, object]],
    ts_local: list[pd.Timestamp],
    field: str,
) -> np.ndarray:
    """Per-step values from solar CSV; missing timestamps -> 0."""
    out = np.zeros(len(ts_local), dtype=np.float32)
    for i, t in enumerate(ts_local):
        hit = solar_map.get(_ts_key_local(pd.Timestamp(t)))
        if hit is not None:
            out[i] = float(hit[field])
    return out


def _solar_features_for_local_times(
    solar_map: dict[pd.Timestamp, dict[str, object]],
    ts_local: list[pd.Timestamp],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Lookup precomputed solar rows by China-local time.

    Missing timestamps -> numeric zeros and ``solar_mask[i]=0`` (encoded solar dims zeroed later).
    """
    n = len(ts_local)
    mask = np.zeros(n, dtype=np.float32)
    records: list[dict[str, object]] = []
    for i, t in enumerate(ts_local):
        k = _ts_key_local(pd.Timestamp(t))
        hit = solar_map.get(k)
        if hit is None:
            records.append(
                {
                    "local_solar_time": "1970-01-01 00:00:00",
                    "solar_azimuth": 0.0,
                    "solar_zenith": 0.0,
                    "day_of_year": 0,
                    "hour_of_day": 0.0,
                }
            )
        else:
            mask[i] = 1.0
            records.append(hit)
    feats = extract_solar_features(pd.DataFrame(records))
    return feats, mask


def _build_solar_timefeats(
    solar_map: dict[pd.Timestamp, dict[str, object]],
    ts_local: list[pd.Timestamp],
    ts_utc: list[pd.Timestamp],
    t0_utc_naive: pd.Timestamp,
) -> torch.Tensor:
    """``[T, 7]`` = masked solar encoder (6) + ``delta_time_encoder`` (1)."""
    feats, solar_mask = _solar_features_for_local_times(solar_map, ts_local)
    enc = solar_features_encoder(feats, include_doy=False)
    m = torch.from_numpy(solar_mask).to(dtype=enc.dtype).unsqueeze(-1)
    enc = enc * m
    dt = delta_time_encoder(ts_utc, t0_utc_naive)
    return torch.cat([enc, dt.unsqueeze(1)], dim=1)


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

    ``pv`` / ``target_pv``: raw Parquet power divided by ``pv_value_scale`` (kW when scale=1).
    ``kt``, ``p_cs``, ``p_mean``, ``target_p_cs``: from ``solar_features_csv``.
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
        pv_value_scale: float = 1.0,
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

        self._pv_value_scale = float(pv_value_scale)
        if self._pv_value_scale <= 0:
            raise ValueError(f"pv_value_scale must be positive, got {self._pv_value_scale}")

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

        print(
            f"[YljRawParquetDataset] split={split} pv_value_scale={self._pv_value_scale} "
            f"({path.name})"
        )

        solar_path = root / str(mx.solar_features_csv)
        self._solar_map = _load_solar_table(solar_path)
        print(
            f"[YljRawParquetDataset] solar features CSV: {solar_path.name} "
            f"({len(self._solar_map)} timestamps)"
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
        pw_x = op[hist_ix].astype(np.float32)
        pv_vals = pw_x / sc
        pv_m = np.ones(self.pv_input_len, dtype=np.float32)
        pow_y = of[fut_ix].astype(np.float64)

        pv = torch.from_numpy(pv_vals).unsqueeze(0)
        pv_timefeats = _build_solar_timefeats(
            self._solar_map, ts_x_local, ts_x_utc, t0u
        )
        forecast_timefeats = _build_solar_timefeats(
            self._solar_map, ts_y_local, y_times_utc, t0u
        )

        kt_x = _solar_series_at_times(self._solar_map, ts_x_local, "kt")
        kt_mask_x = _solar_series_at_times(self._solar_map, ts_x_local, "kt_mask")
        p_cs_x = _solar_series_at_times(self._solar_map, ts_x_local, "p_cs")
        target_p_cs = _solar_series_at_times(self._solar_map, ts_y_local, "p_cs")
        win_hit = self._solar_map.get(_ts_key_local(pd.Timestamp(twin_local)))
        p_mean_val = float(win_hit["p_mean"]) if win_hit is not None else 0.0

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
            "kt": torch.from_numpy(kt_x).unsqueeze(0),
            "kt_mask": torch.from_numpy(kt_mask_x).unsqueeze(0),
            "p_cs": torch.from_numpy(p_cs_x).unsqueeze(0),
            "p_mean": torch.tensor(p_mean_val, dtype=torch.float32),
            "forecast_timefeats": forecast_timefeats,
            "target_pv": torch.tensor([p / sc for p in pow_y], dtype=torch.float32),
            "target_mask": torch.ones(self.pv_output_len, dtype=torch.float32),
            "target_p_cs": torch.from_numpy(target_p_cs.astype(np.float32)),
            "sat_tensor": sat_tensor,
            "sat_timefeats": sat_timefeats,
            "skimg_tensor": None,
            "skimg_timefeats": None,
            "nwp_tensor": nwp_tensor,
        }
        if self._include_export_metadata:
            t_utc = self._t_win_utc(twin_local)
            out["csv_collect_time_utc"] = t_utc.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S")
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
        "kt": torch.stack([x["kt"] for x in batch]),
        "kt_mask": torch.stack([x["kt_mask"] for x in batch]),
        "p_cs": torch.stack([x["p_cs"] for x in batch]),
        "p_mean": torch.stack([x["p_mean"] for x in batch]),
        "forecast_timefeats": torch.stack([x["forecast_timefeats"] for x in batch]),
        "target_pv": torch.stack([x["target_pv"] for x in batch]),
        "target_mask": torch.stack([x["target_mask"] for x in batch]),
        "target_p_cs": torch.stack([x["target_p_cs"] for x in batch]),
    }
    if "csv_collect_time_utc" in batch[0]:
        out["csv_collect_time_utc"] = [x["csv_collect_time_utc"] for x in batch]
    for key in ("sat_tensor", "sat_timefeats", "skimg_tensor", "skimg_timefeats", "nwp_tensor"):
        if batch[0][key] is None:
            if not all(x[key] is None for x in batch):
                raise ValueError(f"mixed None/non-None for {key}")
            out[key] = None
        else:
            out[key] = torch.stack([x[key] for x in batch])
    return out
