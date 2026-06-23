"""Standalone Luoyang 2026 PV dataloader (no luoyang_zarr inheritance/reuse)."""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
from typing import Any
import sys

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
    extract_solar_features,
    solar_features_encoder,
)

VALID_STATE = 512
INVERTER_STATE_COL = "inverter_state"
_NS_PER_HOUR = 3_600_000_000_000
_NS_PER_DAY = 86_400_000_000_000


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


def _precompute_nwp_blocks(
    df: pd.DataFrame | None, value_cols: tuple[str, ...]
) -> dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] | None:
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
        rows = order[bounds[i] : bounds[i + 1]]
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


def _normalize_nwp_frame(df: pd.DataFrame, *, kind: str) -> pd.DataFrame:
    """
    Normalize NWP schema to expected columns:
    - required time cols: start_time, forecast_time
    - solar value col: ssrd
    - wind value cols: msl, t2m, u10, v10, u100, v100
    """
    out = df.copy()
    if "forecast_time" not in out.columns and "dtime" in out.columns and "start_time" in out.columns:
        dt_raw = out["dtime"]
        dt_num = pd.to_numeric(dt_raw, errors="coerce")
        if dt_num.notna().any():
            out["forecast_time"] = pd.to_datetime(out["start_time"], errors="coerce", utc=True) + pd.to_timedelta(dt_num, unit="h")
        else:
            dt_td = pd.to_timedelta(dt_raw, errors="coerce")
            out["forecast_time"] = pd.to_datetime(out["start_time"], errors="coerce", utc=True) + dt_td

    if kind == "solar":
        if "ssrd" not in out.columns and "GHI_mean" in out.columns:
            out["ssrd"] = pd.to_numeric(out["GHI_mean"], errors="coerce")
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


def _interp_nwp_col(
    blocks: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] | None,
    start_ns: int,
    col: str,
    xq_ns_f: np.ndarray,
) -> np.ndarray:
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


def interpolate_nwp_features(
    nwp_solar_blocks: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] | None,
    nwp_wind_blocks: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] | None,
    forecast_timestamps_utc: list[pd.Timestamp],
    dhour: int = 12,
) -> np.ndarray | None:
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
    wind_cols = ("msl", "t2m", "u10", "v10", "u100", "v100")
    wind_interp = [
        _interp_nwp_col(nwp_wind_blocks, prev_noon_ns, c, xq_ns_f) for c in wind_cols
    ]
    nwp_interp = np.column_stack([ssrd_interp] + wind_interp)
    nwp_interp_clean, nwp_mask = _sanitize_nwp_interp(nwp_interp)
    return np.concatenate([nwp_interp_clean, nwp_mask], axis=1)


class PVDataset(Dataset):
    """Standalone dataset using final_power and BJ-time split policy."""

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
        train_fraction: float = 0.85,
        val_fraction: float = 0.15,
        test_start_bj: str = "2026-05-11 00:00:00",
        max_files: int | None = None,
        sample_file_subset: list[Path | str] | None = None,
        enable_sat_sky_cache: bool = False,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train|val|test")
        self.split = split
        self._config_path = Path(config_path).resolve()
        _ = pv_train_time_fraction  # kept only for call-site compatibility
        self._train_samples_per_csv = max(1, int(train_samples_per_csv))

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

        self._pv_dir = Path(pv_dir).resolve()
        self._skyimg_dir = Path(skyimg_dir).resolve()
        self._satimg_dir = Path(satimg_dir).resolve()
        self.enable_sat_sky_cache = bool(enable_sat_sky_cache)
        self._sat_sky_cache_win_idx: int | None = None
        self._sat_sky_cache_bundle: dict[str, torch.Tensor | None] | None = None
        self._sat_sky_cache_loads = 0
        self._sat_sky_cache_hits = 0

        self.satimg_ds = None
        self.skyimg_ds = None
        if self._satimg_dir.exists():
            try:
                self.satimg_ds = xr.open_zarr(self._satimg_dir)
            except Exception as e:
                print(f"[PVDataset2026] WARNING: open sat zarr failed: {e}")
        else:
            print(f"[PVDataset2026] WARNING: sat zarr dir not found: {self._satimg_dir}")
        if self._skyimg_dir.exists():
            try:
                self.skyimg_ds = xr.open_zarr(self._skyimg_dir)
            except Exception as e:
                print(f"[PVDataset2026] WARNING: open sky zarr failed: {e}")
        else:
            print(f"[PVDataset2026] WARNING: sky zarr dir not found: {self._skyimg_dir}")

        cfg = {}
        if self._config_path.is_file():
            cfg = yaml.safe_load(self._config_path.read_text(encoding="utf-8")) or {}
        paths_cfg = cfg.get("paths", {}) or {}

        self.nwp_solar_df = None
        self.nwp_wind_df = None
        nwp_path = paths_cfg.get("nwp_path")
        data_dir = paths_cfg.get("data_dir")
        if nwp_path and data_dir:
            nwp_dir = Path(data_dir) / str(nwp_path)
            solar_csv, wind_csv = _resolve_nwp_csv_paths(nwp_dir)
            try:
                if solar_csv is not None and solar_csv.is_file():
                    self.nwp_solar_df = _normalize_nwp_frame(pd.read_csv(solar_csv), kind="solar")
                if wind_csv is not None and wind_csv.is_file():
                    self.nwp_wind_df = _normalize_nwp_frame(pd.read_csv(wind_csv), kind="wind")
            except Exception as e:
                print(f"[PVDataset2026] WARNING: read NWP CSV failed: {e}")
        self._nwp_solar_blocks = _precompute_nwp_blocks(self.nwp_solar_df, ("ssrd",))
        self._nwp_wind_blocks = _precompute_nwp_blocks(
            self.nwp_wind_df, ("msl", "t2m", "u10", "v10", "u100", "v100")
        )

        all_files = list_csv_files(self._pv_dir)
        if not all_files:
            raise FileNotFoundError(f"No CSV files found in {self._pv_dir}")
        # Stable global mapping: same file name always gets same dev_idx across runs.
        self.devDn_list = [p.stem.replace("_", "=") for p in all_files]
        self._dev_idx_map = {d: i for i, d in enumerate(self.devDn_list)}

        self.sample_files = all_files
        if sample_file_subset is not None:
            path_by_resolved = {p.resolve(): p for p in all_files}
            subset: list[Path] = []
            for raw in sample_file_subset:
                key = Path(raw).resolve()
                if key not in path_by_resolved:
                    raise FileNotFoundError(f"sample_file_subset path not under pv_dir: {raw}")
                subset.append(path_by_resolved[key])
            self.sample_files = subset
        elif max_files is not None and int(max_files) > 0:
            self.sample_files = self.sample_files[: int(max_files)]
        if not self.sample_files:
            raise FileNotFoundError(f"No CSV files found in {self._pv_dir}")

        self._csv_cache: dict[str, pd.DataFrame] = {
            p.resolve().as_posix(): load_csv(p) for p in self.sample_files
        }
        ref_df = self._csv_cache[self.sample_files[0].resolve().as_posix()]
        self._init_anchor_tables(ref_df)
        self._build_split_masks(ref_df)
        self._prefilter_files()
        self._skipped_sample_template: dict | None = None
        if self.split in ("val", "test"):
            self._window_skip_warned_files: set[str] = set()
            ref_key = self.sample_files[0].resolve().as_posix()
            ref_df_keep = self._csv_cache[ref_key]
            r0 = int(
                self._test_r_indices[0] if self.split == "test" else self._val_r_indices[0]
            )
            tpl = self._build_sample(ref_df_keep, torch.tensor(0, dtype=torch.long), r0)
            tpl["sample_valid"] = torch.tensor(0.0, dtype=torch.float32)
            tpl["target_mask"] = torch.zeros_like(tpl["target_mask"])
            tpl["pv_mask"] = torch.zeros_like(tpl["pv_mask"])
            self._skipped_sample_template = tpl

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
        min_row = self._x_idx_per_anchor[:, 0]
        max_row = self._y_idx_per_anchor[:, -1]
        min_time = collect_bj.iloc[min_row]
        max_time = collect_bj.iloc[max_row]

        self._test_anchor_mask = (min_time >= self._test_start_bj).to_numpy(dtype=bool)
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
        valid_anchor_rows: dict[str, np.ndarray] = {}

        for i, p in enumerate(self.sample_files):
            print(f"Processing file {i + 1} of {len(self.sample_files)}: {p.name}")
            key = p.resolve().as_posix()
            df = self._csv_cache[key]
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
                collect_bj = pd.to_datetime(df["collectTime"], utc=True).dt.tz_convert("Asia/Shanghai")
                max_time = collect_bj.iloc[y_idx[:, -1]]
                pre_mask = (max_time < self._test_start_bj).to_numpy(dtype=bool)
                pre_idx = np.nonzero(pre_mask)[0]
                if pre_idx.size == 0:
                    continue
                val_n = max(1, int(round(pre_idx.size * self._val_fraction)))
                if val_n >= pre_idx.size:
                    val_n = pre_idx.size - 1
                train_idx = pre_idx[val_n:]
                if train_idx.size == 0:
                    continue
                if INVERTER_STATE_COL in df.columns:
                    inv_valid = (
                        pd.to_numeric(df[INVERTER_STATE_COL], errors="coerce")
                        .fillna(0)
                        .astype(int)
                        .values
                        == VALID_STATE
                    )
                else:
                    inv_valid = np.ones(n, dtype=bool)
                has_valid_y = inv_valid[y_idx].any(axis=1)
                train_mask_local = np.zeros_like(has_valid_y, dtype=bool)
                train_mask_local[train_idx] = True
                valid_local_pos = np.nonzero(has_valid_y & train_mask_local)[0].astype(np.intp, copy=False)
                if valid_local_pos.size > 0:
                    # Store last-row index j (not reference-anchor position).
                    train_last_rows = anchors[valid_local_pos].astype(np.intp, copy=False)
                    keep_files.append(p)
                    keep_cache[key] = df
                    valid_anchor_rows[key] = train_last_rows
            else:
                keep_files.append(p)
                keep_cache[key] = df

        self.sample_files = keep_files
        self._csv_cache = keep_cache
        self._valid_anchor_rows = valid_anchor_rows
        if not self.sample_files:
            raise RuntimeError(f"No CSV files left after prefilter for split={self.split}")

    def __len__(self) -> int:
        if self.split == "train":
            return len(self.sample_files) * self._train_samples_per_csv
        if self.split == "val":
            return len(self.sample_files) * self._num_val_windows
        return len(self.sample_files) * self._num_test_windows

    @staticmethod
    def _to_utc_timestamps(values: list[Any]) -> list[pd.Timestamp]:
        out: list[pd.Timestamp] = []
        for v in values:
            t = pd.Timestamp(v)
            if t.tzinfo is None:
                t = t.tz_localize("UTC")
            else:
                t = t.tz_convert("UTC")
            out.append(t)
        return out

    def _row_index_for_collect_time_match(self, collect_time: pd.Series, target_ts: pd.Timestamp, csv_name: str) -> int:
        ct = pd.to_datetime(collect_time, errors="coerce")
        if ct.isna().any():
            raise ValueError(f"{csv_name}: NaT in collectTime")
        times_ns = ct.to_numpy(dtype="datetime64[ns]").astype(np.int64)
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

    def _load_sat_sky_modality(self, time0_utc: pd.Timestamp) -> dict[str, torch.Tensor | None]:
        sat_tensor = None
        sat_timefeats = None
        sat_valid = torch.tensor(0.0, dtype=torch.float32)
        if self.satimg_ds is not None:
            sat_t0 = pd.Timestamp(time0_utc - timedelta(minutes=(245 + 30))).tz_convert("UTC").tz_localize(None)
            sat_t1 = pd.Timestamp(time0_utc - timedelta(minutes=30)).tz_convert("UTC").tz_localize(None)
            sat_data = self.satimg_ds.sel(time_utc=slice(sat_t0, sat_t1))
            if int(sat_data.sizes.get("time_utc", 0)) > 0:
                sat_solar_features = {
                    "azimuth": sat_data["azimuth"].values,
                    "zenith": sat_data["zenith"].values,
                    "day_of_year": sat_data["day_of_year"].values,
                    "hour_of_day": sat_data["hour_of_day"].values,
                }
                sat_timefeats = solar_features_encoder(sat_solar_features)
                sat_dtime = delta_time_encoder(sat_data["time_utc"].values, time0_utc)
                sat_timefeats = torch.cat([sat_timefeats, sat_dtime.unsqueeze(1)], dim=1)
                sat_tensor = torch.from_numpy(np.asarray(sat_data["images"].values, dtype=np.float32))
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
        if self.skyimg_ds is not None:
            sky_t0 = pd.Timestamp(time0_utc - timedelta(minutes=30)).tz_convert("UTC").tz_localize(None)
            sky_t1 = pd.Timestamp(time0_utc).tz_convert("UTC").tz_localize(None)
            sky_data = self.skyimg_ds.sel(time_utc=slice(sky_t0, sky_t1))
            if int(sky_data.sizes.get("time_utc", 0)) > 0:
                sky_solar_features = {
                    "azimuth": sky_data["azimuth"].values,
                    "zenith": sky_data["zenith"].values,
                    "day_of_year": sky_data["day_of_year"].values,
                    "hour_of_day": sky_data["hour_of_day"].values,
                }
                sky_timefeats = solar_features_encoder(sky_solar_features)
                sky_dtime = delta_time_encoder(sky_data["time_utc"].values, time0_utc)
                sky_timefeats = torch.cat([sky_timefeats, sky_dtime.unsqueeze(1)], dim=1)
                sky_tensor = torch.from_numpy(np.asarray(sky_data["images"].values, dtype=np.float32))
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
        df: pd.DataFrame,
        dev_idx: torch.Tensor,
        r_fixed: int,
        t_ref: pd.Timestamp,
        csv_name: str,
        *,
        win_idx: int | None = None,
        include_sat_sky: bool = True,
    ) -> dict | None:
        try:
            j = self._row_index_for_collect_time_match(df["collectTime"], t_ref, csv_name)
        except ValueError:
            return None
        if not self._anchor_row_in_bounds(len(df), j):
            return None
        try:
            return self._build_sample(
                df,
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
        p = self.sample_files[file_idx]
        k = p.resolve().as_posix()
        dfi = self._csv_cache[k]
        dev_dn_i = p.stem.replace("_", "=")
        dev_idx_i = torch.tensor(self._dev_idx_map[dev_dn_i], dtype=torch.long)

        sample = self._try_build_sample_for_split_window(
            dfi,
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
        df: pd.DataFrame,
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
        sub_x = df.iloc[x_idx]
        sub_y = df.iloc[y_idx]
        timestamps = self._to_utc_timestamps(list(sub_x["collectTime"]))
        time0_utc = timestamps[-1]

        pow_x = pd.to_numeric(sub_x["final_power"], errors="coerce").fillna(0).values.astype(np.float32)
        pv = torch.from_numpy(pow_x).unsqueeze(0)
        if INVERTER_STATE_COL in sub_x.columns:
            inv_x = pd.to_numeric(sub_x[INVERTER_STATE_COL], errors="coerce").fillna(0).astype(np.int32).values
            pv_mask = torch.from_numpy((inv_x == VALID_STATE).astype(np.float32)).unsqueeze(0)
        else:
            pv_mask = torch.ones(1, len(sub_x), dtype=torch.float32)

        mean_pow_x = float(np.mean(pow_x)) if len(pow_x) else 0.0
        if "p_cs" in sub_x.columns:
            p_cs_np = (
                pd.to_numeric(sub_x["p_cs"], errors="coerce")
                .fillna(1.0)
                .values.astype(np.float32)
            )
        else:
            p_cs_np = np.ones(len(sub_x), dtype=np.float32)

        if "p_mean" in sub_x.columns:
            p_mean_np = (
                pd.to_numeric(sub_x["p_mean"], errors="coerce")
                .fillna(mean_pow_x)
                .values.astype(np.float32)
            )
        else:
            p_mean_np = np.full(len(sub_x), max(mean_pow_x, 1e-6), dtype=np.float32)

        if "kt" in sub_x.columns:
            kt_np = (
                pd.to_numeric(sub_x["kt"], errors="coerce")
                .fillna(0.0)
                .values.astype(np.float32)
            )
        else:
            kt_np = np.zeros(len(sub_x), dtype=np.float32)

        if "kt_mask" in sub_x.columns:
            kt_mask_np = (
                pd.to_numeric(sub_x["kt_mask"], errors="coerce")
                .fillna(1.0)
                .values.astype(np.float32)
            )
        else:
            kt_mask_np = np.ones(len(sub_x), dtype=np.float32)
        if "weather_ghi" in sub_x.columns:
            weather_ghi_np = (
                pd.to_numeric(sub_x["weather_ghi"], errors="coerce")
                .fillna(0.0)
                .values.astype(np.float32)
            )
        else:
            weather_ghi_np = np.zeros(len(sub_x), dtype=np.float32)
        if "theory_ghi" in sub_x.columns:
            theory_ghi_np = (
                pd.to_numeric(sub_x["theory_ghi"], errors="coerce")
                .fillna(0.0)
                .values.astype(np.float32)
            )
        else:
            theory_ghi_np = np.zeros(len(sub_x), dtype=np.float32)
        kt = torch.from_numpy(kt_np).unsqueeze(0)
        kt_mask = torch.from_numpy(kt_mask_np).unsqueeze(0)
        p_cs = torch.from_numpy(p_cs_np).unsqueeze(0)
        weather_ghi = torch.from_numpy(weather_ghi_np).unsqueeze(0)
        theory_ghi = torch.from_numpy(theory_ghi_np).unsqueeze(0)
        p_mean = torch.tensor(float(p_mean_np[-1]) if len(p_mean_np) else 0.0, dtype=torch.float32)

        if {"solar_azimuth", "solar_zenith", "day_of_year", "hour_of_day"}.issubset(set(sub_x.columns)):
            pv_solar_features = extract_solar_features(sub_x)
        else:
            pv_solar_features = compute_solar_features(timestamps, latitude=34.69984, longitude=112.28440)
        pv_timefeats = solar_features_encoder(pv_solar_features)
        pv_dtimefeats = delta_time_encoder(timestamps, time0_utc)
        pv_timefeats = torch.cat([pv_timefeats, pv_dtimefeats.unsqueeze(1)], dim=1)

        forecast_timestamps_utc = self._to_utc_timestamps(list(sub_y["collectTime"]))
        if {"solar_azimuth", "solar_zenith", "day_of_year", "hour_of_day"}.issubset(set(sub_y.columns)):
            forecast_solar_features = extract_solar_features(sub_y)
        else:
            forecast_solar_features = compute_solar_features(forecast_timestamps_utc, latitude=34.69984, longitude=112.28440)
        forecast_timefeats = solar_features_encoder(forecast_solar_features)
        forecast_dtimefeats = delta_time_encoder(forecast_timestamps_utc, time0_utc)
        forecast_timefeats = torch.cat([forecast_timefeats, forecast_dtimefeats.unsqueeze(1)], dim=1)

        nwp_out = interpolate_nwp_features(self._nwp_solar_blocks, self._nwp_wind_blocks, forecast_timestamps_utc)
        nwp_tensor = None if nwp_out is None else torch.from_numpy(np.asarray(nwp_out, dtype=np.float32))

        pow_y = pd.to_numeric(sub_y["final_power"], errors="coerce").fillna(0).values.astype(np.float32)
        target_pv = torch.from_numpy(pow_y)
        if INVERTER_STATE_COL in sub_y.columns:
            inv_y = pd.to_numeric(sub_y[INVERTER_STATE_COL], errors="coerce").fillna(0).astype(np.int32).values
            target_mask = torch.from_numpy((inv_y == VALID_STATE).astype(np.float32))
        else:
            target_mask = torch.ones(len(sub_y), dtype=torch.float32)
        if "p_cs" in sub_y.columns:
            target_p_cs_np = (
                pd.to_numeric(sub_y["p_cs"], errors="coerce")
                .fillna(1.0)
                .values.astype(np.float32)
            )
        else:
            target_p_cs_np = np.ones(len(sub_y), dtype=np.float32)
        target_p_cs = torch.from_numpy(target_p_cs_np)
        if "weather_ghi" in sub_y.columns:
            target_weather_ghi_np = (
                pd.to_numeric(sub_y["weather_ghi"], errors="coerce")
                .fillna(0.0)
                .values.astype(np.float32)
            )
        else:
            target_weather_ghi_np = np.zeros(len(sub_y), dtype=np.float32)
        target_weather_ghi = torch.from_numpy(target_weather_ghi_np)
        if "theory_ghi" in sub_y.columns:
            target_theory_ghi_np = (
                pd.to_numeric(sub_y["theory_ghi"], errors="coerce")
                .fillna(0.0)
                .values.astype(np.float32)
            )
        else:
            target_theory_ghi_np = np.zeros(len(sub_y), dtype=np.float32)
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
            "target_pv": target_pv,
            "target_mask": target_mask,
            "target_p_cs": target_p_cs,
            "target_weather_ghi": target_weather_ghi,
            "target_theory_ghi": target_theory_ghi,
            "sample_valid": torch.tensor(1.0, dtype=torch.float32),
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
            sample_path = self.sample_files[idx % len(self.sample_files)]
            r_fixed = None
        elif self.split == "val":
            nw = self._num_val_windows
            sample_path = self.sample_files[idx // nw]
            r_fixed = int(self._val_r_indices[idx % nw])
        else:
            nw = self._num_test_windows
            sample_path = self.sample_files[idx // nw]
            r_fixed = int(self._test_r_indices[idx % nw])

        dev_dn = sample_path.stem.replace("_", "=")
        dev_idx = torch.tensor(self._dev_idx_map[dev_dn], dtype=torch.long)
        key = sample_path.resolve().as_posix()
        df = self._csv_cache[key]

        if self.split == "train":
            valid_last_rows = self._valid_anchor_rows[key]
            j = int(np.random.choice(valid_last_rows))
            return self._build_sample(df, dev_idx, 0, anchor_last_row=j)

        assert r_fixed is not None
        win_idx = idx % nw
        if self.split == "val":
            assert self._val_last_x_time_ref is not None
            t_ref = self._val_last_x_time_ref[win_idx]
        else:
            assert self._test_last_x_time_ref is not None
            t_ref = self._test_last_x_time_ref[win_idx]

        file_idx = idx // nw
        p = self.sample_files[file_idx]
        k = p.resolve().as_posix()
        dfi = self._csv_cache[k]
        dev_dn_i = p.stem.replace("_", "=")
        dev_idx_i = torch.tensor(self._dev_idx_map[dev_dn_i], dtype=torch.long)

        sample = self._try_build_sample_for_split_window(
            dfi, dev_idx_i, r_fixed, t_ref, p.name, win_idx=win_idx
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

    vals = [s["nwp_tensor"] for s in pv_samples]
    out["nwp_tensor"] = None if any(v is None for v in vals) else torch.stack(vals)

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
    pv_dir = data_dir / paths_cfg.get("pv_path", "pv")
    sky_dir = data_dir / paths_cfg.get("sky_image_path", "luoyangASI_skimg_zarr")
    sat_dir = data_dir / paths_cfg.get("sat_path", "luoyang_sat_zarr")

    return PVDataset(
        config_path=config_path,
        pv_dir=str(pv_dir),
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
        "--plot-out-dir",
        type=Path,
        default=Path("tmp_2026_loader_plots"),
        help="Directory to save quick-check plots",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    print(f"[main] config: {config_path}")
    args.plot_out_dir.mkdir(parents=True, exist_ok=True)

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
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batched)
        batch = next(iter(loader))
        print(
            f"[{split}] pv={tuple(batch['pv'].shape)} target_pv={tuple(batch['target_pv'].shape)} "
            f"sat={'None' if batch['sat_tensor'] is None else tuple(batch['sat_tensor'].shape)} "
            f"sky={'None' if batch['skimg_tensor'] is None else tuple(batch['skimg_tensor'].shape)} "
            f"nwp={'None' if batch['nwp_tensor'] is None else tuple(batch['nwp_tensor'].shape)}"
        )
        # Plot one sample from this batch for quick visual inspection.
        sample_idx = 0
        pv_np = batch["pv"][sample_idx, 0].detach().cpu().numpy()
        p_cs_np = batch["p_cs"][sample_idx, 0].detach().cpu().numpy()
        kt_np = batch["kt"][sample_idx, 0].detach().cpu().numpy()
        kt_mask_np = batch["kt_mask"][sample_idx, 0].detach().cpu().numpy()
        x = np.arange(len(pv_np))
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
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
        fig.suptitle(f"{split} batch sample0: pv / p_cs / kt / kt_mask")
        fig.tight_layout()
        out_path = args.plot_out_dir / f"{split}_batch_sample0_signals.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"[{split}] plot saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
