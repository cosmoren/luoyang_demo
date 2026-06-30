"""Standalone Luoyang 2026 PV dataloader (no luoyang_zarr inheritance/reuse)."""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from modules.solar_encoder import (
    compute_solar_features,
    delta_time_encoder,
    solar_features_encoder,
)

VALID_STATE = 512
INVERTER_STATE_COL = "inverter_state"
_NS_PER_HOUR = 3_600_000_000_000
_NS_PER_DAY = 86_400_000_000_000
_NWP_DEFAULT_TZ = "Asia/Shanghai"


def _dtime_bj_to_utc_datetime(values: pd.Series) -> pd.Series:
    """
    Parse dtime as Beijing wall-clock time and convert to UTC.

    This enforces the project convention for Luoyang NWP/NWP_history ``dtime``:
    it should be interpreted as local Beijing time before interpolation.
    """
    dt = pd.to_datetime(values, errors="coerce")
    if dt.dt.tz is None:
        dt_bj = dt.dt.tz_localize(
            _NWP_DEFAULT_TZ,
            ambiguous="NaT",
            nonexistent="shift_forward",
        )
    else:
        # Even if timezone information exists in raw dtime strings,
        # treat them as Beijing wall-clock by dropping tz and relocalizing.
        dt_bj = dt.dt.tz_localize(None).dt.tz_localize(
            _NWP_DEFAULT_TZ,
            ambiguous="NaT",
            nonexistent="shift_forward",
        )
    return dt_bj.dt.tz_convert("UTC")


def _nwp_to_utc_datetime(values: pd.Series) -> pd.Series:
    """
    Vectorized parse: interpret all input timestamps as Beijing time (Asia/Shanghai),
    then convert to UTC.
    """
    dt = pd.to_datetime(values, errors="coerce")
    if dt.dt.tz is None:
        dt_bj = dt.dt.tz_localize(
            _NWP_DEFAULT_TZ,
            ambiguous="NaT",
            nonexistent="shift_forward",
        )
    else:
        dt_bj = dt.dt.tz_convert(_NWP_DEFAULT_TZ)
    return dt_bj.dt.tz_convert("UTC")


def load_csv(csv_path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "collectTime" not in df.columns:
        raise KeyError(f"{csv_path}: missing collectTime")
    df["collectTime"] = pd.to_datetime(df["collectTime"], errors="coerce")
    if df["collectTime"].isna().any():
        raise ValueError(f"{csv_path}: collectTime has NaT")
    return df.sort_values("collectTime").reset_index(drop=True)


def list_csv_files(data_dir: Path | str) -> list[Path]:
    p = Path(data_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"PV dir not found: {p}")
    return sorted(p.glob("*.csv"))


def _sanitize_nwp_interp(nwp_interp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(nwp_interp, dtype=np.float64)
    bad = ~np.isfinite(x)
    x_clean = np.where(bad, 0.0, x).astype(np.float32, copy=False)
    row_bad = bad.any(axis=1).astype(np.float32).reshape(-1, 1)
    return x_clean, row_bad


def _precompute_nwp_series(
    df: pd.DataFrame | None, value_cols: tuple[str, ...]
) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    """
    Build a single merged per-column series on the UTC forecast_time axis.

    forecast_time is already UTC (produced by _normalize_nwp_frame), so all runs
    are pooled into one time series and interpolated directly on UTC. Duplicate
    forecast_time values (same timestamp coming from multiple runs) are collapsed
    by averaging, yielding a strictly increasing axis suitable for np.interp.
    """
    if df is None:
        return None
    if "forecast_time" not in df.columns:
        return {}
    ft_ns_all = _nwp_to_utc_datetime(df["forecast_time"]).astype("int64").to_numpy()
    n_total = ft_ns_all.shape[0]
    if n_total == 0:
        return {}

    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for col in value_cols:
        if col not in df.columns:
            series[col] = (np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64))
            continue
        y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        valid = ~np.isnan(y) & ~np.isnan(ft_ns_all.astype(np.float64))
        if not valid.any():
            series[col] = (np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64))
            continue
        ft_v = ft_ns_all[valid].astype(np.float64)
        y_v = y[valid]
        # Merge duplicate timestamps by averaging to keep a strictly increasing axis.
        uniq_ft, inv = np.unique(ft_v, return_inverse=True)
        sums = np.zeros(uniq_ft.shape[0], dtype=np.float64)
        counts = np.zeros(uniq_ft.shape[0], dtype=np.float64)
        np.add.at(sums, inv, y_v)
        np.add.at(counts, inv, 1.0)
        series[col] = (uniq_ft, sums / counts)
    return series


def _precompute_nwp_series_latest_start(
    df: pd.DataFrame | None, value_cols: tuple[str, ...]
) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    """
    Build per-column series on UTC ``forecast_time`` by taking latest ``start_time`` per timestamp.

    Unlike ``_precompute_nwp_series`` (which averages duplicate forecast_time points),
    this keeps exactly one row per forecast_time: the row with max start_time.
    """
    if df is None:
        return None
    if "forecast_time" not in df.columns:
        return {}
    if "start_time" not in df.columns:
        # Fallback for datasets without run initialization time.
        return _precompute_nwp_series(df, value_cols)

    ft_ns_all = _nwp_to_utc_datetime(df["forecast_time"]).astype("int64").to_numpy()
    st_ns_all = _nwp_to_utc_datetime(df["start_time"]).astype("int64").to_numpy()
    nat_i64 = np.iinfo(np.int64).min
    valid_time = (ft_ns_all != nat_i64) & (st_ns_all != nat_i64)
    if not valid_time.any():
        return {}

    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for col in value_cols:
        if col not in df.columns:
            series[col] = (np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64))
            continue
        y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        valid = valid_time & np.isfinite(y)
        if not valid.any():
            series[col] = (np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64))
            continue
        tmp = pd.DataFrame(
            {
                "forecast_time_ns": ft_ns_all[valid].astype(np.int64, copy=False),
                "start_time_ns": st_ns_all[valid].astype(np.int64, copy=False),
                "value": y[valid],
            }
        )
        tmp = tmp.sort_values(["forecast_time_ns", "start_time_ns"]).drop_duplicates(
            subset=["forecast_time_ns"], keep="last"
        )
        xp = tmp["forecast_time_ns"].to_numpy(dtype=np.float64, copy=False)
        fp = tmp["value"].to_numpy(dtype=np.float64, copy=False)
        series[col] = (xp, fp)
    return series


def _normalize_nwp_frame(df: pd.DataFrame, *, kind: str) -> pd.DataFrame:
    """
    Normalize NWP schema to expected columns:
    - required time cols: start_time, forecast_time
    - solar value col: ssrd
    - wind value cols: msl, t2m, u10, v10, u100, v100
    """
    out = df.copy()
    if "forecast_time" not in out.columns and "dtime" in out.columns:
        dt_raw = out["dtime"]
        # Luoyang NWP/NWP_history often stores Beijing absolute wall-clock timestamps in dtime.
        looks_like_abs_time = (
            dt_raw.astype(str).str.contains(r"[-/:]", regex=True, na=False).mean() > 0.5
        )
        if looks_like_abs_time or "start_time" not in out.columns:
            out["forecast_time"] = _dtime_bj_to_utc_datetime(dt_raw)
        else:
            dt_num = pd.to_numeric(dt_raw, errors="coerce")
            start_utc = _nwp_to_utc_datetime(out["start_time"])
            if dt_num.notna().any():
                out["forecast_time"] = start_utc + pd.to_timedelta(dt_num, unit="h")
            else:
                dt_td = pd.to_timedelta(dt_raw, errors="coerce")
                out["forecast_time"] = start_utc + dt_td

    if kind == "solar":
        if "ssrd" not in out.columns and "GHI_mean" in out.columns:
            out["ssrd"] = pd.to_numeric(out["GHI_mean"], errors="coerce")
        if "GHI_mean" not in out.columns and "ssrd" in out.columns:
            out["GHI_mean"] = pd.to_numeric(out["ssrd"], errors="coerce")
    
    return out


def _resolve_nwp_csv_paths(nwp_dir: Path) -> tuple[Path | None, Path | None]:
    solar_csv = nwp_dir / "solar.csv"
    wind_csv = nwp_dir / "wind.csv"
    if solar_csv.is_file() and wind_csv.is_file():
        return solar_csv, wind_csv

    solar_candidates = sorted(
        [p for p in nwp_dir.glob("*.csv") if "solar" in p.name.lower()]
    )
    wind_candidates = sorted(
        [p for p in nwp_dir.glob("*.csv") if "wind" in p.name.lower()]
    )
    return (solar_candidates[0] if solar_candidates else None, wind_candidates[0] if wind_candidates else None)


def _interp_nwp_series_col(
    series: dict[str, tuple[np.ndarray, np.ndarray]] | None,
    col: str,
    xq_ns_f: np.ndarray,
) -> np.ndarray:
    """Interpolate a single column directly on the merged UTC forecast_time axis."""
    n = xq_ns_f.shape[0]
    if not series:
        return np.full(n, np.nan, dtype=np.float64)
    pair = series.get(col)
    if pair is None or pair[0].size == 0:
        return np.full(n, np.nan, dtype=np.float64)
    xp_ns_f, fp_f = pair
    return np.interp(xq_ns_f, xp_ns_f, fp_f)


def _lookup_nwp_series_col_exact(
    series: dict[str, tuple[np.ndarray, np.ndarray]] | None,
    col: str,
    xq_ns_i64: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact timestamp lookup for one NWP column.

    Returns:
      values: float32, missing filled with 0.
      mask: float32, 1 where exact timestamp exists, else 0.
    """
    n = int(xq_ns_i64.shape[0])
    values = np.zeros(n, dtype=np.float32)
    mask = np.zeros(n, dtype=np.float32)
    if not series:
        return values, mask
    pair = series.get(col)
    if pair is None or pair[0].size == 0:
        return values, mask

    xp_ns_f, fp_f = pair
    xp_ns_i64 = xp_ns_f.astype(np.int64, copy=False)
    if xp_ns_i64.size == 0:
        return values, mask

    pos = np.searchsorted(xp_ns_i64, xq_ns_i64, side="left")
    in_bounds = pos < xp_ns_i64.size
    exact = np.zeros(n, dtype=bool)
    exact[in_bounds] = xp_ns_i64[pos[in_bounds]] == xq_ns_i64[in_bounds]
    if exact.any():
        values[exact] = fp_f[pos[exact]].astype(np.float32, copy=False)
        mask[exact] = 1.0
    return values, mask


def interpolate_nwp_features(
    nwp_solar_series: dict[str, tuple[np.ndarray, np.ndarray]] | None,
    nwp_wind_series: dict[str, tuple[np.ndarray, np.ndarray]] | None,
    forecast_timestamps_utc: list[pd.Timestamp],
    dhour: int = 12,
) -> np.ndarray | None:
    if not forecast_timestamps_utc:
        return None

    xq_ns_i64 = pd.DatetimeIndex(forecast_timestamps_utc).asi8.astype(np.int64, copy=False)
    _ = dhour  # kept for call-site compatibility
    ssrd_vals, ssrd_mask = _lookup_nwp_series_col_exact(nwp_solar_series, "ssrd", xq_ns_i64)
    t2m_vals, t2m_mask = _lookup_nwp_series_col_exact(nwp_wind_series, "t2m", xq_ns_i64)
    # Forecast nwp_tensor channel order:
    # [ssrd, ssrd_mask, t2m, t2m_mask]
    return np.column_stack([ssrd_vals, ssrd_mask, t2m_vals, t2m_mask]).astype(np.float32, copy=False)


def interpolate_nwp_history_features(
    nwp_hist_solar_series: dict[str, tuple[np.ndarray, np.ndarray]] | None,
    nwp_hist_wind_series: dict[str, tuple[np.ndarray, np.ndarray]] | None,
    history_timestamps_utc: list[pd.Timestamp],
) -> np.ndarray | None:
    """
    Interpolate NWP history channels on PV history timestamps.

    Output channel order:
      [GHI_mean, msl, t2m, u10, v10, u100, v100, hist_valid_mask]
    where hist_valid_mask = 1 means all feature values at that timestamp are valid,
    and 0 means at least one feature is missing/invalid.
    """
    if nwp_hist_solar_series is None or nwp_hist_wind_series is None:
        return None
    if not history_timestamps_utc:
        return None

    xq_ns_f = pd.DatetimeIndex(history_timestamps_utc).asi8.astype(np.float64)
    ghi_interp = _interp_nwp_series_col(nwp_hist_solar_series, "GHI_mean", xq_ns_f)
    wind_cols = ("msl", "t2m", "u10", "v10", "u100", "v100")
    wind_interp = [_interp_nwp_series_col(nwp_hist_wind_series, c, xq_ns_f) for c in wind_cols]
    nwp_hist = np.column_stack([ghi_interp] + wind_interp)
    nwp_hist_clean, nwp_hist_row_bad = _sanitize_nwp_interp(nwp_hist)
    nwp_hist_valid_mask = 1.0 - nwp_hist_row_bad
    return np.concatenate([nwp_hist_clean, nwp_hist_valid_mask], axis=1)


def interpolate_nwp_forecast_history_features(
    nwp_fcst_hist_solar_series: dict[str, tuple[np.ndarray, np.ndarray]] | None,
    nwp_fcst_hist_wind_series: dict[str, tuple[np.ndarray, np.ndarray]] | None,
    history_timestamps_utc: list[pd.Timestamp],
) -> np.ndarray | None:
    """
    Interpolate forecast-derived history channels on PV history timestamps.

    Output channel order matches nwp_history:
      [GHI_mean, msl, t2m, u10, v10, u100, v100, hist_valid_mask]
    """
    if nwp_fcst_hist_solar_series is None or nwp_fcst_hist_wind_series is None:
        return None
    if not history_timestamps_utc:
        return None

    xq_ns_f = pd.DatetimeIndex(history_timestamps_utc).asi8.astype(np.float64)
    ghi_interp = _interp_nwp_series_col(nwp_fcst_hist_solar_series, "GHI_mean", xq_ns_f)
    wind_cols = ("msl", "t2m", "u10", "v10", "u100", "v100")
    wind_interp = [_interp_nwp_series_col(nwp_fcst_hist_wind_series, c, xq_ns_f) for c in wind_cols]
    nwp_hist = np.column_stack([ghi_interp] + wind_interp)
    nwp_hist_clean, nwp_hist_row_bad = _sanitize_nwp_interp(nwp_hist)
    nwp_hist_valid_mask = 1.0 - nwp_hist_row_bad
    return np.concatenate([nwp_hist_clean, nwp_hist_valid_mask], axis=1)


class PVDataset(Dataset):
    """Standalone dataset using final_power and BJ-time split policy."""
    _GLOBAL_ZARR_MEM_CACHE: dict[str, dict[str, np.ndarray | torch.Tensor]] = {}
    _GLOBAL_CSV_DF_CACHE: dict[str, pd.DataFrame] = {}
    _GLOBAL_CSV_NP_CACHE: dict[str, dict[str, np.ndarray]] = {}
    _GLOBAL_NWP_DF_CACHE: dict[str, pd.DataFrame] = {}
    _GLOBAL_NWP_BLOCK_CACHE: dict[
        tuple[str, tuple[str, ...]],
        dict[str, tuple[np.ndarray, np.ndarray]],
    ] = {}

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
        pv_train_time_fraction: float,  # compatibility only
        test_anchor_stride_min: int,
        val_anchor_stride_min: int,
        test_collect_time_match_tolerance_min: int,
        skyimg_window_size: int,
        skyimg_time_resolution_min: int,
        skyimg_spatial_size: int,
        satimg_window_size: int,
        satimg_time_resolution_min: int,
        satimg_npy_shape_hwc: tuple[int, int, int],
        train_samples_per_csv: int = 1,
        kt_noise_std: float = 0.0,
        train_fraction: float = 0.85,
        val_fraction: float = 0.15,
        test_start_bj: str = "2026-05-11 00:00:00",
        max_files: int | None = None,
        sample_file_subset: list[Path | str] | None = None,
        enable_sat_sky_cache: bool = False,
    ) -> None:
        t_init = time.perf_counter()
        t_prev = t_init

        def _log_stage(stage: str) -> None:
            nonlocal t_prev
            t_now = time.perf_counter()
            step_s = t_now - t_prev
            total_s = t_now - t_init
            print(
                f"[PVDataset2026][{split}] {stage}: step={step_s:.3f}s total={total_s:.3f}s",
                flush=True,
            )
            t_prev = t_now

        if split not in ("train", "val", "test"):
            raise ValueError("split must be train|val|test")
        self.split = split
        self._config_path = Path(config_path).resolve()
        _ = pv_train_time_fraction  # kept only for call-site compatibility
        self._train_samples_per_csv = max(1, int(train_samples_per_csv))
        self._kt_noise_std = max(0.0, float(kt_noise_std))

        self.pv_input_len = int(pv_input_len)
        self.pv_output_len = int(pv_output_len)
        self.pv_output_interval_min = int(pv_output_interval_min)
        self.skyimg_window_size = int(skyimg_window_size)
        self.satimg_window_size = int(satimg_window_size)

        if csv_interval_min <= 0:
            raise ValueError("csv_interval_min must be positive")
        if pv_input_interval_min % csv_interval_min != 0:
            raise ValueError("pv_input_interval_min must be multiple of csv_interval_min")
        if pv_output_interval_min % csv_interval_min != 0:
            raise ValueError("pv_output_interval_min must be multiple of csv_interval_min")
        self._sx = pv_input_interval_min // csv_interval_min
        self._sy = pv_output_interval_min // csv_interval_min

        if test_anchor_stride_min <= 0 or test_anchor_stride_min % csv_interval_min != 0:
            raise ValueError("test_anchor_stride_min must be positive multiple of csv_interval_min")
        if val_anchor_stride_min <= 0 or val_anchor_stride_min % csv_interval_min != 0:
            raise ValueError("val_anchor_stride_min must be positive multiple of csv_interval_min")
        self._test_anchor_stride_rows = test_anchor_stride_min // csv_interval_min
        self._val_anchor_stride_rows = val_anchor_stride_min // csv_interval_min

        self._test_collect_time_match_tolerance_min = int(test_collect_time_match_tolerance_min)
        self._test_collect_tolerance_ns = int(self._test_collect_time_match_tolerance_min * 60 * 1e9)

        self._skyimg_dt_min = int(skyimg_time_resolution_min)
        self._skyimg_spatial_size = int(skyimg_spatial_size)
        self._satimg_dt_min = int(satimg_time_resolution_min)
        self._satimg_npy_shape_hwc = tuple(int(x) for x in satimg_npy_shape_hwc)
        if len(self._satimg_npy_shape_hwc) != 3:
            raise ValueError("satimg_npy_shape_hwc must be (H,W,C)")

        if train_fraction <= 0 or val_fraction <= 0:
            raise ValueError("train_fraction and val_fraction must be positive")
        if abs((train_fraction + val_fraction) - 1.0) > 1e-9:
            raise ValueError("train_fraction + val_fraction must equal 1.0")
        self._train_fraction = float(train_fraction)
        self._val_fraction = float(val_fraction)
        self._test_start_bj = pd.Timestamp(test_start_bj).tz_localize("Asia/Shanghai")

        pv_raw = str(pv_dir).strip() if pv_dir is not None else ""
        if not pv_raw:
            raise ValueError("pv_dir must point to a total-power CSV file path")
        pv_candidate = Path(pv_raw).expanduser().resolve()
        if pv_candidate.is_dir():
            raise ValueError(f"pv_dir must be a CSV file, got directory: {pv_candidate}")
        self._pv_total_csv = pv_candidate
        if not self._pv_total_csv.is_file():
            raise FileNotFoundError(f"Total PV csv not found: {self._pv_total_csv}")
        self._dev_dn_total = "NE=total"
        self._skyimg_dir = Path(skyimg_dir).resolve()
        self._satimg_dir = Path(satimg_dir).resolve()
        self.enable_sat_sky_cache = bool(enable_sat_sky_cache)
        self._sat_sky_cache_win_idx: int | None = None
        self._sat_sky_cache_bundle: dict[str, torch.Tensor | None] | None = None
        self._sat_sky_cache_loads = 0
        self._sat_sky_cache_hits = 0

        self.satimg_ds = None
        self.skyimg_ds = None
        self._satimg_mem: dict[str, np.ndarray | torch.Tensor] | None = None
        self._skyimg_mem: dict[str, np.ndarray | torch.Tensor] | None = None
        if self._satimg_dir.exists():
            try:
                sat_key = self._satimg_dir.resolve().as_posix()
                if sat_key in self._GLOBAL_ZARR_MEM_CACHE:
                    self._satimg_mem = self._GLOBAL_ZARR_MEM_CACHE[sat_key]
                else:
                    self.satimg_ds = xr.open_zarr(self._satimg_dir)
                    self._satimg_mem = self._zarr_ds_to_mem(self.satimg_ds)
                    self._GLOBAL_ZARR_MEM_CACHE[sat_key] = self._satimg_mem
                self.satimg_ds = None
            except Exception as e:
                print(f"[PVDataset2026] WARNING: open sat zarr failed: {e}")
        else:
            print(f"[PVDataset2026] WARNING: sat zarr dir not found: {self._satimg_dir}")
        if self._skyimg_dir.exists():
            try:
                sky_key = self._skyimg_dir.resolve().as_posix()
                if sky_key in self._GLOBAL_ZARR_MEM_CACHE:
                    self._skyimg_mem = self._GLOBAL_ZARR_MEM_CACHE[sky_key]
                else:
                    self.skyimg_ds = xr.open_zarr(self._skyimg_dir)
                    self._skyimg_mem = self._zarr_ds_to_mem(self.skyimg_ds)
                    self._GLOBAL_ZARR_MEM_CACHE[sky_key] = self._skyimg_mem
                self.skyimg_ds = None
            except Exception as e:
                print(f"[PVDataset2026] WARNING: open sky zarr failed: {e}")
        else:
            print(f"[PVDataset2026] WARNING: sky zarr dir not found: {self._skyimg_dir}")
        _log_stage("sat/sky source init")

        cfg = {}
        if self._config_path.is_file():
            cfg = yaml.safe_load(self._config_path.read_text(encoding="utf-8")) or {}
        paths_cfg = cfg.get("paths", {}) or {}

        self.nwp_solar_df = None
        self.nwp_wind_df = None
        solar_csv_key: str | None = None
        wind_csv_key: str | None = None
        nwp_path = paths_cfg.get("nwp_path")
        data_dir = paths_cfg.get("data_dir")
        if nwp_path and data_dir:
            nwp_dir = Path(data_dir) / str(nwp_path)
            if not nwp_dir.is_dir():
                print(f"[PVDataset2026] WARNING: NWP dir not found: {nwp_dir}")
            solar_csv, wind_csv = _resolve_nwp_csv_paths(nwp_dir)
            try:
                if solar_csv is not None and solar_csv.is_file():
                    solar_csv_key = solar_csv.resolve().as_posix()
                    if solar_csv_key in self._GLOBAL_NWP_DF_CACHE:
                        self.nwp_solar_df = self._GLOBAL_NWP_DF_CACHE[solar_csv_key]
                    else:
                        self.nwp_solar_df = _normalize_nwp_frame(pd.read_csv(solar_csv), kind="solar")
                        self._GLOBAL_NWP_DF_CACHE[solar_csv_key] = self.nwp_solar_df
                if wind_csv is not None and wind_csv.is_file():
                    wind_csv_key = wind_csv.resolve().as_posix()
                    if wind_csv_key in self._GLOBAL_NWP_DF_CACHE:
                        self.nwp_wind_df = self._GLOBAL_NWP_DF_CACHE[wind_csv_key]
                    else:
                        self.nwp_wind_df = _normalize_nwp_frame(pd.read_csv(wind_csv), kind="wind")
                        self._GLOBAL_NWP_DF_CACHE[wind_csv_key] = self.nwp_wind_df
            except Exception as e:
                print(f"[PVDataset2026] WARNING: read NWP CSV failed: {e}")
        self._nwp_solar_blocks = None
        if self.nwp_solar_df is not None and solar_csv_key is not None:
            solar_block_key = (solar_csv_key, ("ssrd",))
            if solar_block_key in self._GLOBAL_NWP_BLOCK_CACHE:
                self._nwp_solar_blocks = self._GLOBAL_NWP_BLOCK_CACHE[solar_block_key]
            else:
                self._nwp_solar_blocks = _precompute_nwp_series(self.nwp_solar_df, ("ssrd",))
                if self._nwp_solar_blocks is not None:
                    self._GLOBAL_NWP_BLOCK_CACHE[solar_block_key] = self._nwp_solar_blocks

        self._nwp_wind_blocks = None
        if self.nwp_wind_df is not None and wind_csv_key is not None:
            wind_block_key = (wind_csv_key, ("msl", "t2m", "u10", "v10", "u100", "v100"))
            if wind_block_key in self._GLOBAL_NWP_BLOCK_CACHE:
                self._nwp_wind_blocks = self._GLOBAL_NWP_BLOCK_CACHE[wind_block_key]
            else:
                self._nwp_wind_blocks = _precompute_nwp_series(
                    self.nwp_wind_df, ("msl", "t2m", "u10", "v10", "u100", "v100")
                )
                if self._nwp_wind_blocks is not None:
                    self._GLOBAL_NWP_BLOCK_CACHE[wind_block_key] = self._nwp_wind_blocks

        # Forecast-history NWP: aligned to PV history timestamps but sourced from NWP forecast CSVs.
        # For duplicate forecast_time values, keep latest start_time per timestamp.
        self._nwp_forecast_hist_solar_blocks = None
        if self.nwp_solar_df is not None and solar_csv_key is not None:
            fcst_hist_solar_block_key = (solar_csv_key, ("latest_start", "GHI_mean"))
            if fcst_hist_solar_block_key in self._GLOBAL_NWP_BLOCK_CACHE:
                self._nwp_forecast_hist_solar_blocks = self._GLOBAL_NWP_BLOCK_CACHE[fcst_hist_solar_block_key]
            else:
                self._nwp_forecast_hist_solar_blocks = _precompute_nwp_series_latest_start(
                    self.nwp_solar_df, ("GHI_mean",)
                )
                if self._nwp_forecast_hist_solar_blocks is not None:
                    self._GLOBAL_NWP_BLOCK_CACHE[fcst_hist_solar_block_key] = (
                        self._nwp_forecast_hist_solar_blocks
                    )

        self._nwp_forecast_hist_wind_blocks = None
        if self.nwp_wind_df is not None and wind_csv_key is not None:
            fcst_hist_wind_block_key = (
                wind_csv_key,
                ("latest_start", "msl", "t2m", "u10", "v10", "u100", "v100"),
            )
            if fcst_hist_wind_block_key in self._GLOBAL_NWP_BLOCK_CACHE:
                self._nwp_forecast_hist_wind_blocks = self._GLOBAL_NWP_BLOCK_CACHE[fcst_hist_wind_block_key]
            else:
                self._nwp_forecast_hist_wind_blocks = _precompute_nwp_series_latest_start(
                    self.nwp_wind_df, ("msl", "t2m", "u10", "v10", "u100", "v100")
                )
                if self._nwp_forecast_hist_wind_blocks is not None:
                    self._GLOBAL_NWP_BLOCK_CACHE[fcst_hist_wind_block_key] = (
                        self._nwp_forecast_hist_wind_blocks
                    )
        _log_stage("forecast NWP load+precompute")

        # NWP history: aligned with PV history timestamps (x_idx).
        self.nwp_hist_solar_df = None
        self.nwp_hist_wind_df = None
        self._nwp_hist_solar_blocks = None
        self._nwp_hist_wind_blocks = None
        hist_solar_csv_key: str | None = None
        hist_wind_csv_key: str | None = None
        nwp_hist_path = paths_cfg.get("nwp_history_path", "NWP_history")
        if data_dir and nwp_hist_path:
            nwp_hist_dir = Path(data_dir) / str(nwp_hist_path)
            if not nwp_hist_dir.is_dir():
                print(f"[PVDataset2026] WARNING: NWP_history dir not found: {nwp_hist_dir}")
            hist_solar_csv, hist_wind_csv = _resolve_nwp_csv_paths(nwp_hist_dir)
            try:
                if hist_solar_csv is not None and hist_solar_csv.is_file():
                    hist_solar_csv_key = hist_solar_csv.resolve().as_posix()
                    if hist_solar_csv_key in self._GLOBAL_NWP_DF_CACHE:
                        self.nwp_hist_solar_df = self._GLOBAL_NWP_DF_CACHE[hist_solar_csv_key]
                    else:
                        self.nwp_hist_solar_df = _normalize_nwp_frame(
                            pd.read_csv(hist_solar_csv), kind="solar"
                        )
                        self._GLOBAL_NWP_DF_CACHE[hist_solar_csv_key] = self.nwp_hist_solar_df
                if hist_wind_csv is not None and hist_wind_csv.is_file():
                    hist_wind_csv_key = hist_wind_csv.resolve().as_posix()
                    if hist_wind_csv_key in self._GLOBAL_NWP_DF_CACHE:
                        self.nwp_hist_wind_df = self._GLOBAL_NWP_DF_CACHE[hist_wind_csv_key]
                    else:
                        self.nwp_hist_wind_df = _normalize_nwp_frame(
                            pd.read_csv(hist_wind_csv), kind="wind"
                        )
                        self._GLOBAL_NWP_DF_CACHE[hist_wind_csv_key] = self.nwp_hist_wind_df
            except Exception as e:
                print(f"[PVDataset2026] WARNING: read NWP_history CSV failed: {e}")

        if self.nwp_hist_solar_df is not None and hist_solar_csv_key is not None:
            hist_solar_block_key = (hist_solar_csv_key, ("GHI_mean",))
            if hist_solar_block_key in self._GLOBAL_NWP_BLOCK_CACHE:
                self._nwp_hist_solar_blocks = self._GLOBAL_NWP_BLOCK_CACHE[hist_solar_block_key]
            else:
                self._nwp_hist_solar_blocks = _precompute_nwp_series(
                    self.nwp_hist_solar_df, ("GHI_mean",)
                )
                if self._nwp_hist_solar_blocks is not None:
                    self._GLOBAL_NWP_BLOCK_CACHE[hist_solar_block_key] = self._nwp_hist_solar_blocks

        if self.nwp_hist_wind_df is not None and hist_wind_csv_key is not None:
            hist_wind_block_key = (hist_wind_csv_key, ("msl", "t2m", "u10", "v10", "u100", "v100"))
            if hist_wind_block_key in self._GLOBAL_NWP_BLOCK_CACHE:
                self._nwp_hist_wind_blocks = self._GLOBAL_NWP_BLOCK_CACHE[hist_wind_block_key]
            else:
                self._nwp_hist_wind_blocks = _precompute_nwp_series(
                    self.nwp_hist_wind_df, ("msl", "t2m", "u10", "v10", "u100", "v100")
                )
                if self._nwp_hist_wind_blocks is not None:
                    self._GLOBAL_NWP_BLOCK_CACHE[hist_wind_block_key] = self._nwp_hist_wind_blocks
        _log_stage("history NWP load+precompute")

        # Total-station variant uses one fixed CSV (no per-inverter fanout).
        self.devDn_list = [self._dev_dn_total]
        self._dev_idx_map = {self._dev_dn_total: 0}
        self.sample_files = [self._pv_total_csv]

        self._csv_cache: dict[str, pd.DataFrame] = {}
        self._csv_np_cache: dict[str, dict[str, np.ndarray]] = {}
        for p in self.sample_files:
            k = p.resolve().as_posix()
            if k in self._GLOBAL_CSV_DF_CACHE:
                df = self._GLOBAL_CSV_DF_CACHE[k]
            else:
                df = load_csv(p)
                self._GLOBAL_CSV_DF_CACHE[k] = df
            self._csv_cache[k] = df

            if k in self._GLOBAL_CSV_NP_CACHE:
                np_data = self._GLOBAL_CSV_NP_CACHE[k]
            else:
                np_data = self._csv_df_to_numpy_cache(df)
                self._GLOBAL_CSV_NP_CACHE[k] = np_data
            self._csv_np_cache[k] = np_data
        _log_stage("PV csv load+numpy cache")
        ref_df = self._csv_cache[self.sample_files[0].resolve().as_posix()]
        self._init_anchor_tables(ref_df)
        _log_stage("anchor table init")
        self._build_split_masks(ref_df)
        _log_stage("split mask build")
        self._prefilter_files()
        _log_stage("file prefilter")
        self._skipped_sample_template: dict | None = None
        if self.split in ("val", "test"):
            self._window_skip_warned_files: set[str] = set()
            ref_key = self.sample_files[0].resolve().as_posix()
            ref_np_keep = self._csv_np_cache[ref_key]
            r0 = int(
                self._test_r_indices[0] if self.split == "test" else self._val_r_indices[0]
            )
            tpl = self._build_sample(ref_np_keep, torch.tensor(0, dtype=torch.long), r0)
            tpl["sample_valid"] = torch.tensor(0.0, dtype=torch.float32)
            tpl["target_mask"] = torch.zeros_like(tpl["target_mask"])
            tpl["pv_mask"] = torch.zeros_like(tpl["pv_mask"])
            self._skipped_sample_template = tpl
        _log_stage("skip-template build")

    def _init_anchor_tables(self, ref_df: pd.DataFrame) -> None:
        n = len(ref_df)
        lx, ly = self.pv_input_len, self.pv_output_len
        min_anchor = (lx - 1) * self._sx
        y_last_off = self._sy * ly
        max_anchor = n - 1 - y_last_off
        if min_anchor > max_anchor:
            raise RuntimeError(f"no valid anchors, n={n}, range=[{min_anchor},{max_anchor}]")
        anchors = np.arange(min_anchor, max_anchor + 1, dtype=np.intp)
        y_off = self._sy + np.arange(ly, dtype=np.intp) * self._sy
        x_tail = (-(lx - 1) * self._sx + np.arange(lx, dtype=np.intp) * self._sx).reshape(1, -1)
        self._anchors = anchors
        self._csv_row_count = n
        self._y_idx_per_anchor = anchors[:, None] + y_off[None, :]
        self._x_idx_per_anchor = anchors[:, None] + x_tail
        self._x_tail_1d = (-(lx - 1) * self._sx + np.arange(lx, dtype=np.intp) * self._sx).astype(np.intp, copy=False)
        self._y_off_1d = (self._sy + np.arange(ly, dtype=np.intp) * self._sy).astype(np.intp, copy=False)

    def _build_split_masks(self, ref_df: pd.DataFrame) -> None:
        collect_utc = pd.to_datetime(ref_df["collectTime"], utc=True)
        collect_bj = collect_utc.dt.tz_convert("Asia/Shanghai")
        first_target_row = self._y_idx_per_anchor[:, 0]
        max_row = self._y_idx_per_anchor[:, -1]
        first_target_time = collect_bj.iloc[first_target_row]
        max_time = collect_bj.iloc[max_row]

        # Test split should be defined by target timestamps (not history start),
        # otherwise long history windows can delay the first exported target time.
        self._test_anchor_mask = (first_target_time >= self._test_start_bj).to_numpy(dtype=bool)
        pre_mask = (max_time < self._test_start_bj).to_numpy(dtype=bool)
        pre_idx = np.nonzero(pre_mask)[0]
        if pre_idx.size == 0:
            raise RuntimeError("no pre-test anchors found")
        val_n = max(1, int(round(pre_idx.size * self._val_fraction)))
        if val_n >= pre_idx.size:
            val_n = pre_idx.size - 1
        val_idx = pre_idx[:val_n]
        train_idx = pre_idx[val_n:]

        self._train_anchor_mask = np.zeros_like(pre_mask, dtype=bool)
        self._val_anchor_mask = np.zeros_like(pre_mask, dtype=bool)
        self._train_anchor_mask[train_idx] = True
        self._val_anchor_mask[val_idx] = True

        self._val_r_indices = np.nonzero(self._val_anchor_mask)[0][:: self._val_anchor_stride_rows].astype(np.intp, copy=False)
        self._test_r_indices = np.nonzero(self._test_anchor_mask)[0][:: self._test_anchor_stride_rows].astype(np.intp, copy=False)
        self._num_val_windows = int(self._val_r_indices.size)
        self._num_test_windows = int(self._test_r_indices.size)
        ref_ct = pd.to_datetime(ref_df["collectTime"], errors="coerce")
        self._val_last_x_time_ref = [pd.Timestamp(ref_ct.iloc[int(self._anchors[r])]) for r in self._val_r_indices] if self._num_val_windows > 0 else None
        self._test_last_x_time_ref = [pd.Timestamp(ref_ct.iloc[int(self._anchors[r])]) for r in self._test_r_indices] if self._num_test_windows > 0 else None

    def _prefilter_files(self) -> None:
        keep_files: list[Path] = []
        keep_cache: dict[str, pd.DataFrame] = {}
        keep_np_cache: dict[str, dict[str, np.ndarray]] = {}
        valid_anchor_rows: dict[str, np.ndarray] = {}

        for i, p in enumerate(self.sample_files):
            print(f"Processing file {i + 1} of {len(self.sample_files)}: {p.name}")
            key = p.resolve().as_posix()
            df = self._csv_cache[key]
            np_data = self._csv_np_cache[key]
            if self.split == "train":
                # Build per-file train anchors so files with different row counts are still usable.
                n = len(df)
                min_anchor = (self.pv_input_len - 1) * self._sx
                max_anchor = n - 1 - (self._sy * self.pv_output_len)
                if min_anchor > max_anchor:
                    continue
                anchors = np.arange(min_anchor, max_anchor + 1, dtype=np.intp)
                y_off = self._sy + np.arange(self.pv_output_len, dtype=np.intp) * self._sy
                y_idx = anchors[:, None] + y_off[None, :]
                collect_bj = pd.to_datetime(np_data["collect_dt64"], utc=True).tz_convert("Asia/Shanghai")
                max_time = collect_bj[y_idx[:, -1]]
                pre_mask = (max_time < self._test_start_bj).astype(bool, copy=False)
                pre_idx = np.nonzero(pre_mask)[0]
                if pre_idx.size == 0:
                    continue
                val_n = max(1, int(round(pre_idx.size * self._val_fraction)))
                if val_n >= pre_idx.size:
                    val_n = pre_idx.size - 1
                train_idx = pre_idx[val_n:]
                if train_idx.size == 0:
                    continue
                inv_valid = np.asarray(np_data["inv_valid"], dtype=bool)
                has_valid_y = inv_valid[y_idx].any(axis=1)
                train_mask_local = np.zeros_like(has_valid_y, dtype=bool)
                train_mask_local[train_idx] = True
                valid_local_pos = np.nonzero(has_valid_y & train_mask_local)[0].astype(np.intp, copy=False)
                if valid_local_pos.size > 0:
                    # Store last-row index j (not reference-anchor position).
                    train_last_rows = anchors[valid_local_pos].astype(np.intp, copy=False)
                    keep_files.append(p)
                    keep_cache[key] = df
                    keep_np_cache[key] = np_data
                    valid_anchor_rows[key] = train_last_rows
            else:
                keep_files.append(p)
                keep_cache[key] = df
                keep_np_cache[key] = np_data

        self.sample_files = keep_files
        self._csv_cache = keep_cache
        self._csv_np_cache = keep_np_cache
        self._valid_anchor_rows = valid_anchor_rows
        if not self.sample_files:
            raise RuntimeError(f"No CSV files left after prefilter for split={self.split}")

    def __len__(self) -> int:
        if self.split == "train":
            return len(self.sample_files) * self._train_samples_per_csv
        if self.split == "val":
            return len(self.sample_files) * self._num_val_windows
        return len(self.sample_files) * self._num_test_windows

    def _row_index_for_collect_time_match(
        self,
        collect_ns: np.ndarray,
        target_ts: pd.Timestamp,
        csv_name: str,
    ) -> int:
        times_ns = np.asarray(collect_ns, dtype=np.int64)
        tgt = pd.Timestamp(target_ts)
        if tgt.tzinfo is not None:
            tgt = tgt.tz_convert("UTC").tz_localize(None)
        tgt_ns = np.datetime64(tgt.to_datetime64(), "ns").astype(np.int64)
        pos = int(np.searchsorted(times_ns, tgt_ns, side="left"))
        candidates = []
        if pos > 0:
            candidates.append(pos - 1)
        if pos < len(times_ns):
            candidates.append(pos)
        best = candidates[0]
        best_d = abs(int(times_ns[best] - tgt_ns))
        for j in candidates[1:]:
            d = abs(int(times_ns[j] - tgt_ns))
            if d < best_d:
                best, best_d = j, d
        if best_d > self._test_collect_tolerance_ns:
            raise ValueError(f"{csv_name}: no timestamp within tolerance for target {target_ts}")
        return int(best)

    def _anchor_row_in_bounds(self, n_rows: int, j: int) -> bool:
        """True when row j can host the full PV input/output window in a CSV of length n_rows."""
        j = int(j)
        n_rows = int(n_rows)
        min_x = j + int(self._x_tail_1d[0])
        max_y = j + int(self._y_off_1d[-1])
        return min_x >= 0 and max_y < n_rows

    def _time0_for_window(self, win_idx: int) -> pd.Timestamp:
        if self.split == "test":
            assert self._test_last_x_time_ref is not None
            t_ref = self._test_last_x_time_ref[int(win_idx)]
        elif self.split == "val":
            assert self._val_last_x_time_ref is not None
            t_ref = self._val_last_x_time_ref[int(win_idx)]
        else:
            raise ValueError("_time0_for_window requires split test|val")
        t0 = pd.Timestamp(t_ref)
        if t0.tzinfo is None:
            return t0.tz_localize("UTC")
        return t0.tz_convert("UTC")

    @staticmethod
    def _clone_sat_sky_bundle(
        bundle: dict[str, torch.Tensor | None],
    ) -> dict[str, torch.Tensor | None]:
        out: dict[str, torch.Tensor | None] = {}
        for key, val in bundle.items():
            out[key] = val.clone() if isinstance(val, torch.Tensor) else val
        return out

    @staticmethod
    def _zarr_ds_to_mem(ds: xr.Dataset) -> dict[str, np.ndarray | torch.Tensor]:
        time_vals = np.asarray(ds["time_utc"].values)
        time_dt64 = time_vals.astype("datetime64[ns]")
        time_ns = time_dt64.astype(np.int64)
        return {
            "time_dt64": time_dt64,
            "time_ns": time_ns,
            "images": torch.from_numpy(np.asarray(ds["images"].values, dtype=np.float32)),
            "azimuth": np.asarray(ds["azimuth"].values, dtype=np.float32),
            "zenith": np.asarray(ds["zenith"].values, dtype=np.float32),
            "day_of_year": np.asarray(ds["day_of_year"].values, dtype=np.int32),
            "hour_of_day": np.asarray(ds["hour_of_day"].values, dtype=np.float32),
        }

    @staticmethod
    def _csv_col_numeric(
        df: pd.DataFrame,
        col: str,
        *,
        default: float,
        dtype: np.dtype,
    ) -> np.ndarray:
        if col in df.columns:
            return (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(default)
                .to_numpy(dtype=dtype)
            )
        return np.full(len(df), default, dtype=dtype)

    @classmethod
    def _csv_df_to_numpy_cache(cls, df: pd.DataFrame) -> dict[str, np.ndarray]:
        collect_dt64 = pd.to_datetime(df["collectTime"], errors="coerce").to_numpy(dtype="datetime64[ns]")
        if np.isnat(collect_dt64).any():
            raise ValueError("collectTime has invalid values after normalization")
        collect_ns = collect_dt64.astype(np.int64)

        out: dict[str, np.ndarray] = {
            "collect_ns": collect_ns,
            "collect_dt64": collect_dt64,
            "final_power": cls._csv_col_numeric(df, "final_power", default=0.0, dtype=np.float32),
            "p_cs": cls._csv_col_numeric(df, "p_cs", default=1.0, dtype=np.float32),
            "p_mean_raw": cls._csv_col_numeric(df, "p_mean", default=np.nan, dtype=np.float32),
            "kt": cls._csv_col_numeric(df, "kt", default=0.0, dtype=np.float32),
            "kt_mask": cls._csv_col_numeric(df, "kt_mask", default=1.0, dtype=np.float32),
            "weather_ghi": cls._csv_col_numeric(df, "weather_ghi", default=0.0, dtype=np.float32),
            "theory_ghi": cls._csv_col_numeric(df, "theory_ghi", default=0.0, dtype=np.float32),
        }

        if INVERTER_STATE_COL in df.columns:
            inv = (
                pd.to_numeric(df[INVERTER_STATE_COL], errors="coerce")
                .fillna(0)
                .to_numpy(dtype=np.int32)
            )
        else:
            inv = np.full(len(df), VALID_STATE, dtype=np.int32)
        out["inv_valid"] = (inv == VALID_STATE)

        has_solar_cols = {"solar_azimuth", "solar_zenith", "day_of_year", "hour_of_day"}.issubset(
            set(df.columns)
        )
        out["has_solar_cols"] = np.asarray([1 if has_solar_cols else 0], dtype=np.int8)
        if has_solar_cols:
            out["solar_azimuth"] = cls._csv_col_numeric(df, "solar_azimuth", default=0.0, dtype=np.float32)
            out["solar_zenith"] = cls._csv_col_numeric(df, "solar_zenith", default=0.0, dtype=np.float32)
            out["day_of_year"] = cls._csv_col_numeric(df, "day_of_year", default=0.0, dtype=np.int32)
            out["hour_of_day"] = cls._csv_col_numeric(df, "hour_of_day", default=0.0, dtype=np.float32)
        return out

    def _load_sat_sky_modality(self, time0_utc: pd.Timestamp) -> dict[str, torch.Tensor | None]:
        sat_tensor = None
        sat_timefeats = None
        sat_valid = torch.tensor(0.0, dtype=torch.float32)
        if self._satimg_mem is not None:
            sat_t0 = pd.Timestamp(time0_utc - timedelta(minutes=(245 + 30))).tz_convert("UTC").tz_localize(None)
            sat_t1 = pd.Timestamp(time0_utc - timedelta(minutes=30)).tz_convert("UTC").tz_localize(None)
            sat_t0_ns = np.datetime64(sat_t0.to_datetime64(), "ns").astype(np.int64)
            sat_t1_ns = np.datetime64(sat_t1.to_datetime64(), "ns").astype(np.int64)
            sat_ns = self._satimg_mem["time_ns"]
            l = int(np.searchsorted(sat_ns, sat_t0_ns, side="left"))
            r = int(np.searchsorted(sat_ns, sat_t1_ns, side="right"))
            if r > l:
                sat_solar_features = {
                    "azimuth": self._satimg_mem["azimuth"][l:r],
                    "zenith": self._satimg_mem["zenith"][l:r],
                    "day_of_year": self._satimg_mem["day_of_year"][l:r],
                    "hour_of_day": self._satimg_mem["hour_of_day"][l:r],
                }
                sat_timefeats = solar_features_encoder(sat_solar_features)
                sat_dtime = delta_time_encoder(self._satimg_mem["time_dt64"][l:r], time0_utc)
                sat_timefeats = torch.cat([sat_timefeats, sat_dtime.unsqueeze(1)], dim=1)
                sat_tensor = self._satimg_mem["images"][l:r].clone()
                exp_t = self.satimg_window_size
                if sat_tensor.shape[0] > exp_t:
                    sat_tensor = sat_tensor[-exp_t:, ...]
                    sat_timefeats = sat_timefeats[-exp_t:, :]
                elif sat_tensor.shape[0] < exp_t:
                    pad_t = exp_t - sat_tensor.shape[0]
                    sat_tensor = torch.cat(
                        [torch.zeros(pad_t, *sat_tensor.shape[1:], dtype=sat_tensor.dtype), sat_tensor],
                        dim=0,
                    )
                    sat_timefeats = torch.cat(
                        [
                            torch.zeros(pad_t, sat_timefeats.shape[1], dtype=sat_timefeats.dtype),
                            sat_timefeats,
                        ],
                        dim=0,
                    )
                sat_valid = torch.tensor(1.0, dtype=torch.float32)

        sky_tensor = None
        sky_timefeats = None
        skimg_valid = torch.tensor(0.0, dtype=torch.float32)
        if self._skyimg_mem is not None:
            sky_t0 = pd.Timestamp(time0_utc - timedelta(minutes=30)).tz_convert("UTC").tz_localize(None)
            sky_t1 = pd.Timestamp(time0_utc).tz_convert("UTC").tz_localize(None)
            sky_t0_ns = np.datetime64(sky_t0.to_datetime64(), "ns").astype(np.int64)
            sky_t1_ns = np.datetime64(sky_t1.to_datetime64(), "ns").astype(np.int64)
            sky_ns = self._skyimg_mem["time_ns"]
            l = int(np.searchsorted(sky_ns, sky_t0_ns, side="left"))
            r = int(np.searchsorted(sky_ns, sky_t1_ns, side="right"))
            if r > l:
                sky_solar_features = {
                    "azimuth": self._skyimg_mem["azimuth"][l:r],
                    "zenith": self._skyimg_mem["zenith"][l:r],
                    "day_of_year": self._skyimg_mem["day_of_year"][l:r],
                    "hour_of_day": self._skyimg_mem["hour_of_day"][l:r],
                }
                sky_timefeats = solar_features_encoder(sky_solar_features)
                sky_dtime = delta_time_encoder(self._skyimg_mem["time_dt64"][l:r], time0_utc)
                sky_timefeats = torch.cat([sky_timefeats, sky_dtime.unsqueeze(1)], dim=1)
                sky_tensor = self._skyimg_mem["images"][l:r].clone()
                exp_t = self.skyimg_window_size
                if sky_tensor.shape[0] > exp_t:
                    sky_tensor = sky_tensor[-exp_t:, ...]
                    sky_timefeats = sky_timefeats[-exp_t:, :]
                elif sky_tensor.shape[0] < exp_t:
                    pad_t = exp_t - sky_tensor.shape[0]
                    sky_tensor = torch.cat(
                        [torch.zeros(pad_t, *sky_tensor.shape[1:], dtype=sky_tensor.dtype), sky_tensor],
                        dim=0,
                    )
                    sky_timefeats = torch.cat(
                        [
                            torch.zeros(pad_t, sky_timefeats.shape[1], dtype=sky_timefeats.dtype),
                            sky_timefeats,
                        ],
                        dim=0,
                    )
                skimg_valid = torch.tensor(1.0, dtype=torch.float32)

        return {
            "sat_tensor": sat_tensor,
            "sat_timefeats": sat_timefeats,
            "sat_valid": sat_valid,
            "skimg_tensor": sky_tensor,
            "skimg_timefeats": sky_timefeats,
            "skimg_valid": skimg_valid,
        }

    def load_sat_sky_for_window(self, win_idx: int) -> dict[str, torch.Tensor | None]:
        """Load sat/sky zarr once for a test/val window; reuse across all inverters."""
        if self.split not in ("test", "val"):
            raise ValueError("load_sat_sky_for_window requires split test|val")
        win_idx = int(win_idx)
        if self._sat_sky_cache_win_idx == win_idx and self._sat_sky_cache_bundle is not None:
            return self._sat_sky_cache_bundle
        time0_utc = self._time0_for_window(win_idx)
        self._sat_sky_cache_bundle = self._load_sat_sky_modality(time0_utc)
        self._sat_sky_cache_win_idx = win_idx
        self._sat_sky_cache_loads += 1
        return self._sat_sky_cache_bundle

    def prepare_sat_sky_window(self, win_idx: int) -> None:
        """Load sat/sky once for a test/val window (shared across inverters)."""
        if not self.enable_sat_sky_cache or self.split not in ("test", "val"):
            return
        self.load_sat_sky_for_window(win_idx)

    def sat_sky_cache_stats(self) -> tuple[int, int]:
        return self._sat_sky_cache_loads, self._sat_sky_cache_hits

    def _sat_sky_for_sample(
        self,
        time0_utc: pd.Timestamp,
        *,
        win_idx: int | None,
    ) -> dict[str, torch.Tensor | None]:
        if (
            self.enable_sat_sky_cache
            and win_idx is not None
            and self._sat_sky_cache_win_idx == int(win_idx)
            and self._sat_sky_cache_bundle is not None
        ):
            self._sat_sky_cache_hits += 1
            return self._clone_sat_sky_bundle(self._sat_sky_cache_bundle)
        return self._load_sat_sky_modality(time0_utc)

    def _try_build_sample_for_split_window(
        self,
        np_data: dict[str, np.ndarray],
        dev_idx: torch.Tensor,
        r_fixed: int,
        t_ref: pd.Timestamp,
        csv_name: str,
        *,
        win_idx: int | None = None,
        include_sat_sky: bool = True,
    ) -> dict | None:
        try:
            j = self._row_index_for_collect_time_match(np_data["collect_ns"], t_ref, csv_name)
        except ValueError:
            return None
        if not self._anchor_row_in_bounds(int(np_data["collect_ns"].shape[0]), j):
            return None
        try:
            return self._build_sample(
                np_data,
                dev_idx,
                r_fixed,
                anchor_last_row=j,
                win_idx=win_idx,
                include_sat_sky=include_sat_sky,
            )
        except IndexError:
            return None

    def build_pv_sample_for_window(self, file_idx: int, win_idx: int) -> dict:
        """Build PV/NWP/forecast fields only (no sat/sky zarr) for one inverter window."""
        if self.split == "test":
            nw = self._num_test_windows
            assert self._test_last_x_time_ref is not None
            t_ref = self._test_last_x_time_ref[int(win_idx)]
            r_fixed = int(self._test_r_indices[int(win_idx)])
        elif self.split == "val":
            nw = self._num_val_windows
            assert self._val_last_x_time_ref is not None
            t_ref = self._val_last_x_time_ref[int(win_idx)]
            r_fixed = int(self._val_r_indices[int(win_idx)])
        else:
            raise ValueError("build_pv_sample_for_window requires split test|val")

        file_idx = int(file_idx)
        win_idx = int(win_idx)
        p = self.sample_files[0]
        k = p.resolve().as_posix()
        np_data = self._csv_np_cache[k]
        dev_idx_i = torch.tensor(0, dtype=torch.long)

        sample = self._try_build_sample_for_split_window(
            np_data,
            dev_idx_i,
            r_fixed,
            t_ref,
            p.name,
            win_idx=win_idx,
            include_sat_sky=False,
        )
        if sample is not None:
            return sample
        return self._skipped_pv_sample(dev_idx_i)

    def _skipped_pv_sample(self, dev_idx: torch.Tensor) -> dict:
        assert self._skipped_sample_template is not None
        pv_keys = (
            "dev_idx",
            "pv",
            "pv_mask",
            "pv_timefeats",
            "kt",
            "kt_mask",
            "p_cs",
            "weather_ghi",
            "theory_ghi",
            "p_mean",
            "forecast_timefeats",
            "nwp_tensor",
            "nwp_history",
            "target_pv",
            "target_mask",
            "target_p_cs",
            "target_weather_ghi",
            "target_theory_ghi",
            "sample_valid",
        )
        tpl = self._skipped_sample_template
        out: dict = {}
        for key in pv_keys:
            val = tpl[key]
            out[key] = val.clone() if isinstance(val, torch.Tensor) else val
        out["dev_idx"] = dev_idx
        out["sample_valid"] = torch.tensor(0.0, dtype=torch.float32)
        return out

    def _skipped_sample(self, dev_idx: torch.Tensor) -> dict:
        assert self._skipped_sample_template is not None
        out = {
            k: (v.clone() if isinstance(v, torch.Tensor) else v)
            for k, v in self._skipped_sample_template.items()
        }
        out["dev_idx"] = dev_idx
        out["sample_valid"] = torch.tensor(0.0, dtype=torch.float32)
        return out

    def _build_sample(
        self,
        np_data: dict[str, np.ndarray],
        dev_idx: torch.Tensor,
        r: int,
        *,
        anchor_last_row: int | None = None,
        win_idx: int | None = None,
        include_sat_sky: bool = True,
    ) -> dict:
        if anchor_last_row is None:
            x_idx = self._x_idx_per_anchor[r]
            y_idx = self._y_idx_per_anchor[r]
        else:
            j = int(anchor_last_row)
            x_idx = j + self._x_tail_1d
            y_idx = j + self._y_off_1d
        collect_dt64 = np_data["collect_dt64"]
        timestamps = collect_dt64[x_idx]
        forecast_timestamps = collect_dt64[y_idx]
        time0_utc = pd.Timestamp(timestamps[-1]).tz_localize("UTC")

        pow_x = np_data["final_power"][x_idx]
        pv = torch.from_numpy(pow_x).unsqueeze(0)
        pv_mask = torch.from_numpy(np_data["inv_valid"][x_idx].astype(np.float32)).unsqueeze(0)

        mean_pow_x = float(np.mean(pow_x)) if len(pow_x) else 0.0
        p_cs_np = np_data["p_cs"][x_idx]
        p_mean_np = np_data["p_mean_raw"][x_idx].copy()
        if p_mean_np.size > 0:
            p_mean_np[np.isnan(p_mean_np)] = mean_pow_x
        else:
            p_mean_np = np.full(len(x_idx), max(mean_pow_x, 1e-6), dtype=np.float32)
        kt_np = np_data["kt"][x_idx]
        kt_mask_np = np_data["kt_mask"][x_idx]
        if self.split == "train" and self._kt_noise_std > 0.0:
            noise = np.random.normal(0.0, self._kt_noise_std, size=kt_np.shape).astype(np.float32)
            valid_kt = (kt_mask_np > 0.5).astype(np.float32, copy=False)
            kt_np = np.clip(kt_np + noise * valid_kt, 0.0, None)
        weather_ghi_np = np_data["weather_ghi"][x_idx]
        theory_ghi_np = np_data["theory_ghi"][x_idx]
        kt = torch.from_numpy(kt_np).unsqueeze(0)
        kt_mask = torch.from_numpy(kt_mask_np).unsqueeze(0)
        p_cs = torch.from_numpy(p_cs_np).unsqueeze(0)
        weather_ghi = torch.from_numpy(weather_ghi_np).unsqueeze(0)
        theory_ghi = torch.from_numpy(theory_ghi_np).unsqueeze(0)
        p_mean = torch.tensor(float(p_mean_np[-1]) if len(p_mean_np) else 0.0, dtype=torch.float32)

        if int(np_data["has_solar_cols"][0]) == 1:
            pv_solar_features = {
                "azimuth": np_data["solar_azimuth"][x_idx],
                "zenith": np_data["solar_zenith"][x_idx],
                "day_of_year": np_data["day_of_year"][x_idx],
                "hour_of_day": np_data["hour_of_day"][x_idx],
            }
        else:
            pv_solar_features = compute_solar_features(timestamps, latitude=34.69984, longitude=112.28440)
        pv_timefeats = solar_features_encoder(pv_solar_features)
        pv_dtimefeats = delta_time_encoder(timestamps, time0_utc)
        pv_timefeats = torch.cat([pv_timefeats, pv_dtimefeats.unsqueeze(1)], dim=1)

        if int(np_data["has_solar_cols"][0]) == 1:
            forecast_solar_features = {
                "azimuth": np_data["solar_azimuth"][y_idx],
                "zenith": np_data["solar_zenith"][y_idx],
                "day_of_year": np_data["day_of_year"][y_idx],
                "hour_of_day": np_data["hour_of_day"][y_idx],
            }
        else:
            forecast_solar_features = compute_solar_features(
                forecast_timestamps, latitude=34.69984, longitude=112.28440
            )
        forecast_timefeats = solar_features_encoder(forecast_solar_features)
        forecast_dtimefeats = delta_time_encoder(forecast_timestamps, time0_utc)
        forecast_timefeats = torch.cat([forecast_timefeats, forecast_dtimefeats.unsqueeze(1)], dim=1)

        nwp_out = interpolate_nwp_features(
            self._nwp_solar_blocks, self._nwp_wind_blocks, list(pd.DatetimeIndex(forecast_timestamps))
        )
        nwp_tensor = None if nwp_out is None else torch.from_numpy(np.asarray(nwp_out, dtype=np.float32))
        nwp_hist_out = interpolate_nwp_history_features(
            self._nwp_hist_solar_blocks, self._nwp_hist_wind_blocks, list(pd.DatetimeIndex(timestamps))
        )
        nwp_forecast_hist_out = interpolate_nwp_forecast_history_features(
            self._nwp_forecast_hist_solar_blocks,
            self._nwp_forecast_hist_wind_blocks,
            list(pd.DatetimeIndex(timestamps)),
        )
        # nwp_history must always be present:
        # when history CSV is missing/unavailable, fill zeros and set mask channel to 0.
        if nwp_hist_out is None:
            nwp_history_np = np.zeros((len(x_idx), 8), dtype=np.float32)
        else:
            nwp_history_np = np.asarray(nwp_hist_out, dtype=np.float32)
            if nwp_history_np.ndim != 2 or nwp_history_np.shape[1] != 8:
                fallback = np.zeros((len(x_idx), 8), dtype=np.float32)
                rows = min(fallback.shape[0], nwp_history_np.shape[0] if nwp_history_np.ndim == 2 else 0)
                cols = min(fallback.shape[1], nwp_history_np.shape[1] if nwp_history_np.ndim == 2 else 0)
                if rows > 0 and cols > 0:
                    fallback[:rows, :cols] = nwp_history_np[:rows, :cols]
                nwp_history_np = fallback
        nwp_history = torch.from_numpy(nwp_history_np)
        # nwp_forecast_history must always be present:
        # values are from NWP forecast files, aligned to PV history timestamps.
        if nwp_forecast_hist_out is None:
            nwp_forecast_history_np = np.zeros((len(x_idx), 8), dtype=np.float32)
        else:
            nwp_forecast_history_np = np.asarray(nwp_forecast_hist_out, dtype=np.float32)
            if nwp_forecast_history_np.ndim != 2 or nwp_forecast_history_np.shape[1] != 8:
                fallback = np.zeros((len(x_idx), 8), dtype=np.float32)
                rows = min(
                    fallback.shape[0],
                    nwp_forecast_history_np.shape[0] if nwp_forecast_history_np.ndim == 2 else 0,
                )
                cols = min(
                    fallback.shape[1],
                    nwp_forecast_history_np.shape[1] if nwp_forecast_history_np.ndim == 2 else 0,
                )
                if rows > 0 and cols > 0:
                    fallback[:rows, :cols] = nwp_forecast_history_np[:rows, :cols]
                nwp_forecast_history_np = fallback
        nwp_forecast_history = torch.from_numpy(nwp_forecast_history_np)

        pow_y = np_data["final_power"][y_idx]
        target_pv = torch.from_numpy(pow_y)
        target_mask = torch.from_numpy(np_data["inv_valid"][y_idx].astype(np.float32))
        target_p_cs_np = np_data["p_cs"][y_idx]
        target_p_cs = torch.from_numpy(target_p_cs_np)
        target_weather_ghi_np = np_data["weather_ghi"][y_idx]
        target_weather_ghi = torch.from_numpy(target_weather_ghi_np)
        target_theory_ghi_np = np_data["theory_ghi"][y_idx]
        target_theory_ghi = torch.from_numpy(target_theory_ghi_np)

        out: dict = {
            "dev_idx": dev_idx,
            "pv": pv,
            "pv_mask": pv_mask,
            "pv_timefeats": pv_timefeats,
            "kt": kt,
            "kt_mask": kt_mask,
            "p_cs": p_cs,
            "weather_ghi": weather_ghi,
            "theory_ghi": theory_ghi,
            "p_mean": p_mean,
            "forecast_timefeats": forecast_timefeats,
            "nwp_tensor": nwp_tensor,
            "nwp_history": nwp_history,
            "nwp_forecast_history": nwp_forecast_history,
            "target_pv": target_pv,
            "target_mask": target_mask,
            "target_p_cs": target_p_cs,
            "target_weather_ghi": target_weather_ghi,
            "target_theory_ghi": target_theory_ghi,
            "sample_valid": torch.tensor(1.0, dtype=torch.float32),
            "anchor_time_utc_ns": torch.tensor(int(time0_utc.value), dtype=torch.int64),
        }
        if not include_sat_sky:
            return out

        sat_sky = self._sat_sky_for_sample(time0_utc, win_idx=win_idx)
        out["sat_tensor"] = sat_sky["sat_tensor"]
        out["sat_timefeats"] = sat_sky["sat_timefeats"]
        out["sat_valid"] = sat_sky["sat_valid"]
        out["skimg_tensor"] = sat_sky["skimg_tensor"]
        out["skimg_timefeats"] = sat_sky["skimg_timefeats"]
        out["skimg_valid"] = sat_sky["skimg_valid"]
        return out

    def __getitem__(self, idx: int) -> dict:
        if self.split == "train":
            sample_path = self.sample_files[0]
            r_fixed = None
        elif self.split == "val":
            nw = self._num_val_windows
            sample_path = self.sample_files[0]
            r_fixed = int(self._val_r_indices[idx % nw])
        else:
            nw = self._num_test_windows
            sample_path = self.sample_files[0]
            r_fixed = int(self._test_r_indices[idx % nw])

        dev_idx = torch.tensor(0, dtype=torch.long)
        key = sample_path.resolve().as_posix()
        np_data = self._csv_np_cache[key]

        if self.split == "train":
            valid_last_rows = self._valid_anchor_rows[key]
            j = int(np.random.choice(valid_last_rows))
            return self._build_sample(np_data, dev_idx, 0, anchor_last_row=j)

        assert r_fixed is not None
        win_idx = idx % nw
        if self.split == "val":
            assert self._val_last_x_time_ref is not None
            t_ref = self._val_last_x_time_ref[win_idx]
        else:
            assert self._test_last_x_time_ref is not None
            t_ref = self._test_last_x_time_ref[win_idx]

        p = self.sample_files[0]
        k = p.resolve().as_posix()
        np_data_i = self._csv_np_cache[k]
        dev_idx_i = torch.tensor(0, dtype=torch.long)

        sample = self._try_build_sample_for_split_window(
            np_data_i, dev_idx_i, r_fixed, t_ref, p.name, win_idx=win_idx
        )
        if sample is not None:
            return sample

        if p.name not in self._window_skip_warned_files:
            self._window_skip_warned_files.add(p.name)
            print(
                f"[PVDataset2026] WARNING: {p.name} has windows skipped for split={self.split} "
                f"(time mismatch or CSV too short for full input/output window)",
                flush=True,
            )
        return self._skipped_sample(dev_idx_i)


def collate_batched(batch: list[dict]) -> dict:
    if not batch:
        raise ValueError("empty batch")

    def _stack(key: str) -> torch.Tensor:
        return torch.stack([s[key] for s in batch])

    def _collate_img_modality(
        tensor_key: str,
        time_key: str,
        valid_key: str,
        *,
        default_tensor_shape: tuple[int, int, int, int],
        default_time_shape: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tensors = [s[tensor_key] for s in batch]
        timefeats = [s[time_key] for s in batch]
        valid = torch.stack([s[valid_key].to(torch.float32) for s in batch], dim=0)

        sample_tensor = next((v for v in tensors if v is not None), None)
        sample_time = next((v for v in timefeats if v is not None), None)
        if sample_tensor is None:
            sample_tensor = torch.zeros(*default_tensor_shape, dtype=torch.float32)
        if sample_time is None:
            sample_time = torch.zeros(*default_time_shape, dtype=torch.float32)

        out_tensors: list[torch.Tensor] = []
        out_timefeats: list[torch.Tensor] = []
        for t, tf in zip(tensors, timefeats):
            out_tensors.append(torch.zeros_like(sample_tensor) if t is None else t)
            out_timefeats.append(torch.zeros_like(sample_time) if tf is None else tf)
        return torch.stack(out_tensors), torch.stack(out_timefeats), valid

    out = {
        "dev_idx": _stack("dev_idx"),
        "pv": _stack("pv"),
        "pv_mask": _stack("pv_mask"),
        "pv_timefeats": _stack("pv_timefeats"),
        "kt": _stack("kt"),
        "kt_mask": _stack("kt_mask"),
        "p_cs": _stack("p_cs"),
        "weather_ghi": _stack("weather_ghi"),
        "theory_ghi": _stack("theory_ghi"),
        "p_mean": _stack("p_mean"),
        "forecast_timefeats": _stack("forecast_timefeats"),
        "target_pv": _stack("target_pv"),
        "target_mask": _stack("target_mask"),
        "target_p_cs": _stack("target_p_cs"),
        "target_weather_ghi": _stack("target_weather_ghi"),
        "target_theory_ghi": _stack("target_theory_ghi"),
    }
    if "sample_valid" in batch[0]:
        out["sample_valid"] = _stack("sample_valid")
    if "anchor_time_utc_ns" in batch[0]:
        out["anchor_time_utc_ns"] = _stack("anchor_time_utc_ns")
    sat_tensor, sat_timefeats, sat_valid_mask = _collate_img_modality(
        "sat_tensor",
        "sat_timefeats",
        "sat_valid",
        default_tensor_shape=(24, 3, 100, 100),
        default_time_shape=(24, 9),
    )
    out["sat_tensor"] = sat_tensor
    out["sat_timefeats"] = sat_timefeats
    out["sat_valid_mask"] = sat_valid_mask

    skimg_tensor, skimg_timefeats, skimg_valid_mask = _collate_img_modality(
        "skimg_tensor",
        "skimg_timefeats",
        "skimg_valid",
        default_tensor_shape=(30, 3, 224, 224),
        default_time_shape=(30, 9),
    )
    out["skimg_tensor"] = skimg_tensor
    out["skimg_timefeats"] = skimg_timefeats
    out["skimg_valid_mask"] = skimg_valid_mask

    vals = [s["nwp_tensor"] for s in batch]
    out["nwp_tensor"] = None if any(v is None for v in vals) else torch.stack(vals)
    vals_hist = [s.get("nwp_history") for s in batch]
    sample_hist = next((v for v in vals_hist if isinstance(v, torch.Tensor)), None)
    if sample_hist is None:
        sample_hist = torch.zeros((batch[0]["pv"].shape[-1], 8), dtype=torch.float32)
    out["nwp_history"] = torch.stack(
        [v if isinstance(v, torch.Tensor) else torch.zeros_like(sample_hist) for v in vals_hist]
    )
    vals_fcst_hist = [s.get("nwp_forecast_history") for s in batch]
    sample_fcst_hist = next((v for v in vals_fcst_hist if isinstance(v, torch.Tensor)), None)
    if sample_fcst_hist is None:
        sample_fcst_hist = torch.zeros((batch[0]["pv"].shape[-1], 8), dtype=torch.float32)
    out["nwp_forecast_history"] = torch.stack(
        [v if isinstance(v, torch.Tensor) else torch.zeros_like(sample_fcst_hist) for v in vals_fcst_hist]
    )
    return out


def collate_with_shared_sat_sky(
    pv_samples: list[dict],
    sat_sky_bundle: dict[str, torch.Tensor | None],
) -> dict:
    """Stack PV-only samples and broadcast shared sat/sky imagery to batch dim."""
    if not pv_samples:
        raise ValueError("empty batch")

    def _stack(key: str) -> torch.Tensor:
        return torch.stack([s[key] for s in pv_samples])

    bsz = len(pv_samples)
    out: dict = {
        "dev_idx": _stack("dev_idx"),
        "pv": _stack("pv"),
        "pv_mask": _stack("pv_mask"),
        "pv_timefeats": _stack("pv_timefeats"),
        "kt": _stack("kt"),
        "kt_mask": _stack("kt_mask"),
        "p_cs": _stack("p_cs"),
        "weather_ghi": _stack("weather_ghi"),
        "theory_ghi": _stack("theory_ghi"),
        "p_mean": _stack("p_mean"),
        "forecast_timefeats": _stack("forecast_timefeats"),
        "target_pv": _stack("target_pv"),
        "target_mask": _stack("target_mask"),
        "target_p_cs": _stack("target_p_cs"),
        "target_weather_ghi": _stack("target_weather_ghi"),
        "target_theory_ghi": _stack("target_theory_ghi"),
    }
    if "sample_valid" in pv_samples[0]:
        out["sample_valid"] = _stack("sample_valid")
    if "anchor_time_utc_ns" in pv_samples[0]:
        out["anchor_time_utc_ns"] = _stack("anchor_time_utc_ns")

    vals = [s["nwp_tensor"] for s in pv_samples]
    out["nwp_tensor"] = None if any(v is None for v in vals) else torch.stack(vals)
    vals_hist = [s.get("nwp_history") for s in pv_samples]
    sample_hist = next((v for v in vals_hist if isinstance(v, torch.Tensor)), None)
    if sample_hist is None:
        sample_hist = torch.zeros((pv_samples[0]["pv"].shape[-1], 8), dtype=torch.float32)
    out["nwp_history"] = torch.stack(
        [v if isinstance(v, torch.Tensor) else torch.zeros_like(sample_hist) for v in vals_hist]
    )
    vals_fcst_hist = [s.get("nwp_forecast_history") for s in pv_samples]
    sample_fcst_hist = next((v for v in vals_fcst_hist if isinstance(v, torch.Tensor)), None)
    if sample_fcst_hist is None:
        sample_fcst_hist = torch.zeros((pv_samples[0]["pv"].shape[-1], 8), dtype=torch.float32)
    out["nwp_forecast_history"] = torch.stack(
        [v if isinstance(v, torch.Tensor) else torch.zeros_like(sample_fcst_hist) for v in vals_fcst_hist]
    )

    def _expand_img(
        tensor: torch.Tensor | None,
        timefeats: torch.Tensor | None,
        valid: torch.Tensor | None,
        *,
        default_tensor_shape: tuple[int, ...],
        default_time_shape: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if tensor is None:
            tensor = torch.zeros(*default_tensor_shape, dtype=torch.float32)
        if timefeats is None:
            timefeats = torch.zeros(*default_time_shape, dtype=torch.float32)
        valid_scalar = float(valid.item()) if valid is not None else 0.0
        sat_b = tensor.unsqueeze(0).expand(bsz, *tensor.shape).contiguous()
        tf_b = timefeats.unsqueeze(0).expand(bsz, *timefeats.shape).contiguous()
        valid_b = torch.full((bsz,), valid_scalar, dtype=torch.float32)
        return sat_b, tf_b, valid_b

    sat_tensor, sat_timefeats, sat_valid_mask = _expand_img(
        sat_sky_bundle.get("sat_tensor"),
        sat_sky_bundle.get("sat_timefeats"),
        sat_sky_bundle.get("sat_valid"),
        default_tensor_shape=(24, 3, 100, 100),
        default_time_shape=(24, 9),
    )
    out["sat_tensor"] = sat_tensor
    out["sat_timefeats"] = sat_timefeats
    out["sat_valid_mask"] = sat_valid_mask

    skimg_tensor, skimg_timefeats, skimg_valid_mask = _expand_img(
        sat_sky_bundle.get("skimg_tensor"),
        sat_sky_bundle.get("skimg_timefeats"),
        sat_sky_bundle.get("skimg_valid"),
        default_tensor_shape=(30, 3, 224, 224),
        default_time_shape=(30, 9),
    )
    out["skimg_tensor"] = skimg_tensor
    out["skimg_timefeats"] = skimg_timefeats
    out["skimg_valid_mask"] = skimg_valid_mask
    return out


def _build_dataset_from_cfg(config_path: Path, split: str, max_files: int | None = None) -> PVDataset:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    paths_cfg = cfg.get("paths", {})
    sampling = cfg.get("sampling", {})
    split_cfg = cfg.get("split_policy", {})

    data_dir = Path(paths_cfg["data_dir"]).expanduser()
    pv_total_raw = paths_cfg.get("pv_total_path")
    if not pv_total_raw:
        raise KeyError("dataset config paths.pv_total_path is required for luoyang_2026total_zarr")
    pv_total_csv = Path(str(pv_total_raw)).expanduser()
    if not pv_total_csv.is_absolute():
        pv_total_csv = data_dir / pv_total_csv
    sky_dir = data_dir / paths_cfg.get("sky_image_path", "luoyangASI_skimg_zarr")
    sat_dir = data_dir / paths_cfg.get("sat_path", "luoyang_sat_zarr")

    return PVDataset(
        config_path=config_path,
        pv_dir=str(pv_total_csv),
        skyimg_dir=str(sky_dir),
        satimg_dir=str(sat_dir),
        split=split,
        csv_interval_min=int(sampling.get("csv_interval_min", 5)),
        pv_input_interval_min=int(sampling.get("pv_input_interval_min", 5)),
        pv_input_len=int(sampling.get("pv_input_len", 576)),
        pv_output_interval_min=int(sampling.get("pv_output_interval_min", 15)),
        pv_output_len=int(sampling.get("pv_output_len", 192)),
        pv_train_time_fraction=float(sampling.get("pv_train_time_fraction", 0.7)),
        test_anchor_stride_min=int(sampling.get("test_anchor_stride_min", 120)),
        val_anchor_stride_min=int(sampling.get("val_anchor_stride_min", 120)),
        test_collect_time_match_tolerance_min=int(sampling.get("test_collect_time_match_tolerance_min", 0)),
        skyimg_window_size=int(sampling.get("skyimg_window_size", 30)),
        skyimg_time_resolution_min=int(sampling.get("skyimg_time_resolution_min", 1)),
        skyimg_spatial_size=int(sampling.get("skyimg_spatial_size", 224)),
        satimg_window_size=int(sampling.get("satimg_window_size", 24)),
        satimg_time_resolution_min=int(sampling.get("satimg_time_resolution_min", 10)),
        satimg_npy_shape_hwc=tuple(sampling.get("satimg_npy_shape_hwc", [100, 100, 3])),
        train_samples_per_csv=int(sampling.get("train_samples_per_csv", 100)),
        kt_noise_std=float(sampling.get("kt_noise_std", 0.01)),
        train_fraction=float(split_cfg.get("train_fraction", 0.85)),
        val_fraction=float(split_cfg.get("val_fraction", 0.15)),
        test_start_bj=str(split_cfg.get("test_start_bj", "2026-05-11 00:00:00")),
        max_files=max_files,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for luoyang_2026_zarr dataloader")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/datasets/conf_luoyang_2026.yaml"),
        help="Dataset config yaml path",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-files", type=int, default=8, help="Limit CSV count for fast smoke test")
    parser.add_argument(
        "--plot-batch-idx",
        type=int,
        default=3,
        help="Backward-compatible start offset: start sample ~= plot_batch_idx * batch_size.",
    )
    parser.add_argument(
        "--plot-start-sample",
        type=int,
        default=None,
        help="Absolute dataset sample index to start plotting from. Overrides --plot-batch-idx mapping if set.",
    )
    parser.add_argument(
        "--plot-num-anchors",
        type=int,
        default=12,
        help="Number of different anchors (samples) to plot per split.",
    )
    parser.add_argument(
        "--plot-anchor-step",
        type=int,
        default=1,
        help="Sample index stride between plotted anchors.",
    )
    parser.add_argument(
        "--plot-out-dir",
        type=Path,
        default=Path("tmp_2026_loader_plots"),
        help="Directory to save quick-check plots",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    print(f"[main] config: {config_path}")
    args.plot_out_dir.mkdir(parents=True, exist_ok=True)

    def _anchor_strings(sample: dict) -> tuple[str, str]:
        if "anchor_time_utc_ns" not in sample:
            return "unknown", "unknown"
        ts_utc = pd.Timestamp(int(sample["anchor_time_utc_ns"].item()), unit="ns", tz="UTC")
        ts_bj = ts_utc.tz_convert("Asia/Shanghai")
        return ts_utc.strftime("%Y-%m-%d %H:%M:%S UTC"), ts_bj.strftime("%Y-%m-%d %H:%M:%S BJ")

    def _anchor_tag(sample: dict) -> str:
        if "anchor_time_utc_ns" not in sample:
            return "unknown"
        ts_utc = pd.Timestamp(int(sample["anchor_time_utc_ns"].item()), unit="ns", tz="UTC")
        return ts_utc.strftime("%Y%m%dT%H%M%SZ")

    def _current_frame_rgb(
        seq: torch.Tensor | None,
        *,
        fallback_hw: tuple[int, int] = (128, 128),
    ) -> tuple[np.ndarray, str]:
        """
        Convert a [T, C, H, W] tensor sequence to displayable current-frame RGB [H, W, 3].
        Uses the last frame (closest to anchor time in current sampling policy).
        """
        h0, w0 = fallback_hw
        empty = np.zeros((h0, w0, 3), dtype=np.float32)
        if seq is None:
            return empty, "missing"
        if not isinstance(seq, torch.Tensor) or seq.ndim != 4 or seq.shape[0] == 0:
            return empty, "invalid"

        frame = seq[-1].detach().cpu().float().numpy()  # [C,H,W]
        if frame.ndim != 3:
            return empty, "invalid"

        if frame.shape[0] in (1, 3):
            img = np.transpose(frame, (1, 2, 0))
        else:
            # Fallback for unexpected channel order.
            img = frame
            if img.ndim == 2:
                img = img[..., None]

        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        elif img.shape[-1] > 3:
            img = img[..., :3]

        finite = np.isfinite(img)
        if not finite.any():
            return np.zeros_like(img, dtype=np.float32), "all-nan"
        vals = img[finite]
        lo = float(np.percentile(vals, 1.0))
        hi = float(np.percentile(vals, 99.0))
        if hi <= lo:
            out = np.zeros_like(img, dtype=np.float32)
        else:
            out = np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32, copy=False)
        return out, "ok"

    for split in ("train", "val", "test"):
        ds = _build_dataset_from_cfg(config_path, split=split, max_files=args.max_files)
        print(
            f"[{split}] files={len(ds.sample_files)} len={len(ds)} "
            f"train_mask={int(ds._train_anchor_mask.sum())} "
            f"val_mask={int(ds._val_anchor_mask.sum())} "
            f"test_mask={int(ds._test_anchor_mask.sum())}"
        )
        if len(ds) == 0:
            continue

        start_idx = (
            int(args.plot_start_sample)
            if args.plot_start_sample is not None
            else max(0, int(args.plot_batch_idx) * int(args.batch_size))
        )
        step = max(1, int(args.plot_anchor_step))
        max_to_plot = max(1, int(args.plot_num_anchors))

        plotted = 0
        for k in range(max_to_plot):
            idx = start_idx + k * step
            if idx >= len(ds):
                break
            sample = ds[idx]
            anchor_utc_str, anchor_bj_str = _anchor_strings(sample)
            anchor_tag = _anchor_tag(sample)
            print(f"[{split}] plotting idx={idx} anchor={anchor_utc_str} ({anchor_bj_str})")

            pv_np = sample["pv"][0].detach().cpu().numpy()
            p_cs_np = sample["p_cs"][0].detach().cpu().numpy()
            kt_np = sample["kt"][0].detach().cpu().numpy()
            kt_mask_np = sample["kt_mask"][0].detach().cpu().numpy()
            if sample.get("nwp_history") is None:
                nwp_hist_ghi_np = np.zeros_like(pv_np)
                ghi_hist_label = "nwp_history_ghi_mean (missing -> zeros)"
            else:
                # nwp_history channel order:
                # [GHI_mean, msl, t2m, u10, v10, u100, v100, hist_valid_mask]
                nwp_hist_ghi_np = sample["nwp_history"][:, 0].detach().cpu().numpy()
                ghi_hist_label = "nwp_history_ghi_mean"
            if sample.get("nwp_forecast_history") is None:
                nwp_fcst_hist_ghi_np = np.zeros_like(pv_np)
                ghi_fcst_hist_label = "nwp_forecast_history_ghi_mean (missing -> zeros)"
            else:
                # nwp_forecast_history channel order:
                # [GHI_mean, msl, t2m, u10, v10, u100, v100, hist_valid_mask]
                nwp_fcst_hist_ghi_np = sample["nwp_forecast_history"][:, 0].detach().cpu().numpy()
                ghi_fcst_hist_label = "nwp_forecast_history_ghi_mean"

            sat_img, sat_state = _current_frame_rgb(sample.get("sat_tensor"), fallback_hw=(100, 100))
            sky_img, sky_state = _current_frame_rgb(sample.get("skimg_tensor"), fallback_hw=(224, 224))

            x = np.arange(len(pv_np))
            fig = plt.figure(figsize=(18, 10))
            gs = fig.add_gridspec(4, 2, width_ratios=[3.2, 1.8], wspace=0.25, hspace=0.35)
            axes = [fig.add_subplot(gs[i, 0]) for i in range(4)]
            axes[0].plot(x, pv_np, lw=1.2, color="tab:blue")
            axes[0].set_ylabel("pv")
            axes[0].grid(True, alpha=0.25)
            axes[1].plot(x, p_cs_np, lw=1.2, color="tab:orange")
            axes[1].set_ylabel("p_cs")
            axes[1].grid(True, alpha=0.25)
            axes[2].plot(x, kt_np, lw=1.2, color="tab:green")
            axes[2].set_ylabel("kt")
            axes[2].grid(True, alpha=0.25)
            axes[3].plot(x, kt_mask_np, lw=1.2, color="tab:red")
            axes[3].set_ylabel("kt_mask")
            axes[3].set_xlabel("history step")
            axes[3].set_ylim(-0.05, 1.05)
            axes[3].grid(True, alpha=0.25)

            ax_sat = fig.add_subplot(gs[0:2, 1])
            ax_sky = fig.add_subplot(gs[2:4, 1])
            ax_sat.imshow(sat_img)
            ax_sky.imshow(sky_img)
            ax_sat.set_title(f"sat current frame (-1) [{sat_state}]")
            ax_sky.set_title(f"sky current frame (-1) [{sky_state}]")
            ax_sat.axis("off")
            ax_sky.axis("off")
            fig.suptitle(
                f"{split} idx={idx}: pv / p_cs / kt / kt_mask + current sat/sky | "
                f"anchor={anchor_utc_str} ({anchor_bj_str})"
            )
            fig.tight_layout()
            out_path = args.plot_out_dir / f"{split}_idx{idx:06d}_{anchor_tag}_signals.png"
            fig.savefig(out_path, dpi=160)
            plt.close(fig)

            # Overlay with scale adjustment:
            # - Left axis: pv (raw)
            # - Right axis: p_cs/kt and normalized NWP-history GHI_mean (0..1)
            ghi_min = float(np.nanmin(nwp_hist_ghi_np)) if nwp_hist_ghi_np.size else 0.0
            ghi_max = float(np.nanmax(nwp_hist_ghi_np)) if nwp_hist_ghi_np.size else 0.0
            if np.isfinite(ghi_min) and np.isfinite(ghi_max) and ghi_max > ghi_min:
                nwp_hist_ghi_norm = (nwp_hist_ghi_np - ghi_min) / (ghi_max - ghi_min)
            else:
                nwp_hist_ghi_norm = np.zeros_like(nwp_hist_ghi_np, dtype=np.float32)
            fcst_ghi_min = float(np.nanmin(nwp_fcst_hist_ghi_np)) if nwp_fcst_hist_ghi_np.size else 0.0
            fcst_ghi_max = float(np.nanmax(nwp_fcst_hist_ghi_np)) if nwp_fcst_hist_ghi_np.size else 0.0
            if np.isfinite(fcst_ghi_min) and np.isfinite(fcst_ghi_max) and fcst_ghi_max > fcst_ghi_min:
                nwp_fcst_hist_ghi_norm = (nwp_fcst_hist_ghi_np - fcst_ghi_min) / (fcst_ghi_max - fcst_ghi_min)
            else:
                nwp_fcst_hist_ghi_norm = np.zeros_like(nwp_fcst_hist_ghi_np, dtype=np.float32)

            fig_hist, ax_hist_l = plt.subplots(1, 1, figsize=(14, 4.5))
            ax_hist_l.plot(x, pv_np, lw=1.1, color="tab:blue", label="pv")
            ax_hist_l.set_xlabel("history step")
            ax_hist_l.set_ylabel("pv", color="tab:blue")
            ax_hist_l.tick_params(axis="y", labelcolor="tab:blue")
            ax_hist_l.grid(True, alpha=0.25)

            ax_hist_r = ax_hist_l.twinx()
            ax_hist_r.plot(x, p_cs_np, lw=1.1, color="tab:orange", label="p_cs")
            ax_hist_r.plot(x, kt_np, lw=1.1, color="tab:green", label="kt")
            ax_hist_r.plot(
                x,
                nwp_hist_ghi_norm,
                lw=1.1,
                color="tab:red",
                label=f"{ghi_hist_label}_norm",
            )
            ax_hist_r.plot(
                x,
                nwp_fcst_hist_ghi_norm,
                lw=1.1,
                color="tab:purple",
                label=f"{ghi_fcst_hist_label}_norm",
            )
            ax_hist_r.set_ylabel("p_cs / kt / ghi_norm", color="tab:orange")
            ax_hist_r.tick_params(axis="y", labelcolor="tab:orange")
            ax_hist_r.set_ylim(-0.05, 1.05)

            lines_l, labels_l = ax_hist_l.get_legend_handles_labels()
            lines_r, labels_r = ax_hist_r.get_legend_handles_labels()
            ax_hist_l.legend(lines_l + lines_r, labels_l + labels_r, loc="upper right")
            fig_hist.suptitle(
                f"{split} idx={idx}: pv / p_cs / kt / nwp_history_ghi_mean / nwp_forecast_history_ghi_mean | "
                f"anchor={anchor_utc_str} ({anchor_bj_str})"
            )
            fig_hist.tight_layout()
            out_path_hist = args.plot_out_dir / f"{split}_idx{idx:06d}_{anchor_tag}_history_ghi_overlay.png"
            fig_hist.savefig(out_path_hist, dpi=160)
            plt.close(fig_hist)

            # Direct history-vs-forecast-history feature check for loader validation:
            # - ssrd-like channel: index 0 (GHI_mean)
            # - t2m channel: index 2
            if sample.get("nwp_history") is None:
                hist_ssrd_like_np = np.zeros_like(pv_np)
                hist_t2m_np = np.zeros_like(pv_np)
            else:
                hist_ssrd_like_np = sample["nwp_history"][:, 0].detach().cpu().numpy()
                hist_t2m_np = sample["nwp_history"][:, 2].detach().cpu().numpy()
            if sample.get("nwp_forecast_history") is None:
                fcst_hist_ssrd_like_np = np.zeros_like(pv_np)
                fcst_hist_t2m_np = np.zeros_like(pv_np)
            else:
                fcst_hist_ssrd_like_np = sample["nwp_forecast_history"][:, 0].detach().cpu().numpy()
                fcst_hist_t2m_np = sample["nwp_forecast_history"][:, 2].detach().cpu().numpy()

            fig_cmp, axes_cmp = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
            axes_cmp[0].plot(x, hist_ssrd_like_np, lw=1.1, color="tab:blue", label="nwp_history_ssrd_like(ch0)")
            axes_cmp[0].plot(
                x,
                fcst_hist_ssrd_like_np,
                lw=1.1,
                color="tab:orange",
                label="nwp_forecast_history_ssrd_like(ch0)",
            )
            axes_cmp[0].set_ylabel("ssrd-like")
            axes_cmp[0].grid(True, alpha=0.25)
            axes_cmp[0].legend(loc="upper right")

            axes_cmp[1].plot(x, hist_t2m_np, lw=1.1, color="tab:green", label="nwp_history_t2m(ch2)")
            axes_cmp[1].plot(
                x,
                fcst_hist_t2m_np,
                lw=1.1,
                color="tab:red",
                label="nwp_forecast_history_t2m(ch2)",
            )
            axes_cmp[1].set_ylabel("t2m")
            axes_cmp[1].set_xlabel("history step")
            axes_cmp[1].grid(True, alpha=0.25)
            axes_cmp[1].legend(loc="upper right")

            fig_cmp.suptitle(
                f"{split} idx={idx}: nwp_history vs nwp_forecast_history (ssrd-like & t2m) | "
                f"anchor={anchor_utc_str} ({anchor_bj_str})"
            )
            fig_cmp.tight_layout()
            out_path_cmp = (
                args.plot_out_dir / f"{split}_idx{idx:06d}_{anchor_tag}_history_vs_forecast_history_ssrd_t2m.png"
            )
            fig_cmp.savefig(out_path_cmp, dpi=160)
            plt.close(fig_cmp)

            target_pv_np = sample["target_pv"].detach().cpu().numpy()
            if sample["nwp_tensor"] is None:
                ssrd_np = np.zeros_like(target_pv_np)
                ssrd_mask_np = np.zeros_like(target_pv_np)
                t2m_np = np.zeros_like(target_pv_np)
                t2m_mask_np = np.zeros_like(target_pv_np)
                ssrd_label = "ssrd (nwp missing -> zeros)"
                t2m_label = "t2m (nwp missing -> zeros)"
            else:
                # nwp feature order: [ssrd, ssrd_mask, t2m, t2m_mask]
                ssrd_np = sample["nwp_tensor"][:, 0].detach().cpu().numpy()
                ssrd_mask_np = sample["nwp_tensor"][:, 1].detach().cpu().numpy()
                t2m_np = sample["nwp_tensor"][:, 2].detach().cpu().numpy()
                t2m_mask_np = sample["nwp_tensor"][:, 3].detach().cpu().numpy()
                ssrd_label = "ssrd"
                t2m_label = "t2m"

            x_out = np.arange(len(target_pv_np))
            fig2, axes2 = plt.subplots(5, 1, figsize=(14, 11), sharex=True)
            axes2[0].plot(x_out, target_pv_np, lw=1.2, color="tab:blue")
            axes2[0].set_ylabel("target_pv")
            axes2[0].grid(True, alpha=0.25)
            axes2[1].plot(x_out, ssrd_np, lw=1.2, color="tab:green")
            axes2[1].set_ylabel(ssrd_label)
            axes2[1].grid(True, alpha=0.25)
            axes2[2].plot(x_out, ssrd_mask_np, lw=1.2, color="tab:purple")
            axes2[2].set_ylabel("ssrd_mask")
            axes2[2].set_ylim(-0.05, 1.05)
            axes2[2].grid(True, alpha=0.25)
            axes2[3].plot(x_out, t2m_np, lw=1.2, color="tab:red")
            axes2[3].set_ylabel(t2m_label)
            axes2[3].grid(True, alpha=0.25)
            axes2[4].plot(x_out, t2m_mask_np, lw=1.2, color="tab:brown")
            axes2[4].set_ylabel("t2m_mask")
            axes2[4].set_ylim(-0.05, 1.05)
            axes2[4].set_xlabel("forecast step")
            axes2[4].grid(True, alpha=0.25)
            fig2.suptitle(
                f"{split} idx={idx}: target_pv / ssrd / ssrd_mask / t2m / t2m_mask | "
                f"anchor={anchor_utc_str} ({anchor_bj_str})"
            )
            fig2.tight_layout()
            out_path2 = args.plot_out_dir / f"{split}_idx{idx:06d}_{anchor_tag}_targets_nwp_4ch.png"
            fig2.savefig(out_path2, dpi=160)
            plt.close(fig2)

            print(
                f"[{split}] plots saved: {out_path.name}, {out_path_hist.name}, {out_path_cmp.name}, "
                f"{out_path2.name}"
            )
            plotted += 1

        print(
            f"[{split}] plotted {plotted}/{max_to_plot} anchors "
            f"(start={start_idx}, step={step}, len={len(ds)})"
        )


if __name__ == "__main__":
    main()
