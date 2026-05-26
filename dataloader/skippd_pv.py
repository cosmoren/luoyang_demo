"""
SKIPP'd PV dataset: one CSV with time + PV power (kW) plus sky Zarr (no NWP, no satellite).

This module is a near-1:1 port of :class:`dataloader.folsom.FolsomIrradianceDataset`
(the kt-aware Folsom code on this branch). The only substantive substitutions are:

* CSV schema: hardcoded ``Date`` (time) + ``Huang_E4102_kW`` (PV power, kW) — SKIPP'd CSV
  has only those two columns, so Folsom's GHI/DNI/DHI header detection is replaced with
  fixed names.
* Site lat/lon: read from the dataset YAML's ``site.{latitude, longitude}`` (with a
  hardcoded Stanford rooftop fallback) — SKIPP'd has no ``<data_dir>/info.yaml``.
* PV normalization: ``pv_norm = pv_kW / _SKIPPD_PV_SCALE`` with ``_SKIPPD_PV_SCALE = 30``
  kW (analog of Folsom's ``_FOLSOM_GHI_SCALE = 1100`` W/m²).
* Clear-sky scale: ``p_cs = clearsky_ghi / _SKIPPD_CS_GHI_SCALE`` with
  ``_SKIPPD_CS_GHI_SCALE = 1000`` W/m². The PV scale (30 kW) and clear-sky GHI scale
  (1000 W/m²) MUST decouple because PV is in kW and GHI is in W/m² — the user's
  trainer-side "1100 → 30" substitution applies only to the PV normalization, not the
  clear-sky scale (see the head-of-file note in
  ``training/train_vit_test_skippd.py`` for the trainer-side math).
* No NWP merged CSV: ``nwp_tensor`` is emitted as zeros + invalid-mask ones, with the
  same ``[T_out, len(_FOLSOM_NWP_FEATURE_COLS) + 1]`` shape Folsom emits when its NWP
  file is missing. The SKIPP'd trainer never reads it (the trainer drops the
  ``--use-nwp`` / NWP-remap code paths Folsom uses).
* No satellite: ``sat_tensor`` / ``sat_timefeats`` are ``None`` (same as Folsom).
* Train anchor filter: Folsom keeps anchors whose Y window has any GHI > 10 W/m² row.
  SKIPP'd has no GHI; we keep anchors whose Y window has any ``kt_mask == 1`` row
  (i.e. ``p_cs > _SKIPPD_KT_DAYTIME_THRESHOLD``). Same daytime intent.

Everything else — Folsom-style 60/10/30 row split, random train anchors per epoch,
strided val/test anchors, Ineichen clear-sky model, sky-Zarr nearest-frame stacking
with the same ``_sky_anchor_max_lag`` / 90s tolerance, ``kt = pv_norm / (p_cs + eps) *
kt_mask``, ``kt_mask = (p_cs > 0.1)``, ``p_mean = 1.0`` (literal mirror of Folsom),
``dev_idx = 700``, time features via ``compute_solar_features`` / encoders — is a
direct mirror.

Sky-Zarr helpers (``_folsom_*``) are imported from :mod:`dataloader.folsom` to avoid
code duplication; the instance methods that wrap them (``_stack_sky_from_zarr``,
``_nominal_sky_frame_times``, ``_resize_sky_chw``, ``_tensor_from_zarr_image_tile``,
``_black_sky_tensor``, ``_sky_filename_ts``, ``_validate_sky_zarr_schema``) are copied
verbatim from Folsom so a side-by-side diff is trivial.

Returned sample-dict keys (one ``__getitem__`` -> one dict; the SKIPP'd trainer's
``collate_batched`` call from :mod:`dataloader.luoyang_zarr` stacks the tensor keys
with a leading batch dim B):

    dev_idx, pv, pv_mask, pv_timefeats, forecast_timefeats,
    kt, kt_mask, p_cs, p_mean, target_p_cs,
    target_pv, target_mask,
    sat_tensor=None, sat_timefeats=None,
    skimg_tensor, skimg_timefeats,
    nwp_tensor (zeros placeholder),
    input_timestamps_utc, forecast_timestamps_utc, skimg_timestamps.
"""

from __future__ import annotations

import argparse
import bisect  # noqa: F401  # imported for parity with dataloader/folsom.py
import os
import sys
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pvlib
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

try:
    import xarray as xr
except ImportError:  # pragma: no cover - paths.sky_format=zarr requires xarray
    xr = None  # type: ignore[assignment]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Sky-Zarr helpers reused from dataloader/folsom.py (module-level functions only;
# instance methods are copied verbatim below — same source comment Folsom uses).
from dataloader.folsom import (  # noqa: E402
    _FOLSOM_NWP_FEATURE_COLS,
    _folsom_parse_zarr_utc_naive,
    _folsom_sky_zarr_count_in_time_range,
    _folsom_sky_zarr_len_time_utc,
    _folsom_sky_zarr_time_dim_and_values,
)
from dataloader.luoyang_mem import list_csv_files  # noqa: E402
from modules.solar_encoder import (  # noqa: E402
    compute_solar_features,
    delta_time_encoder,
    solar_features_encoder,
)

# One open handle per Zarr path (train/val/test share the same store path).
# Mirrors ``_ZARR_SKY_DS_CACHE`` in :mod:`dataloader.folsom`.
_ZARR_SKY_DS_CACHE: dict[str, Any] = {}

# Default dataset YAML for SKIPP'd under ``config/datasets/`` (smoke CLI convenience).
_DEFAULT_SKIPPD_DATASET_CONFIG = _PROJECT_ROOT / "config" / "datasets" / "conf_skippd.yaml"

_DEFAULT_SKIPPD_TRAIN_EPOCH_LEN = 50_000

# Train mode: keep anchors whose Y window has at least one ``kt_mask == 1`` row
# (i.e. clearsky-daytime). Replaces Folsom's GHI > 10 W/m² filter.
_SKIPPD_TRAIN_DAYTIME_KT_MASK_THRESHOLD = 0.5

# PV normalization scale (analog of Folsom's _FOLSOM_GHI_SCALE = 1100 W/m²).
# Used to put pv_kW into the same normalized space as target_pv / pv_pred.
_SKIPPD_PV_SCALE = 30.0
# Clear-sky GHI normalization scale (Folsom uses 1100; SKIPP'd uses 1000 — a more
# standard peak-GHI reference, matching the user's hint "p_cs based on clear-sky
# GHI / 1000"). Decoupled from _SKIPPD_PV_SCALE because PV (kW) and GHI (W/m²)
# live in different unit spaces.
_SKIPPD_CS_GHI_SCALE = 1000.0
# Daytime guard (Luoyang/Folsom convention): kt_mask = (p_cs > 0.1).
_SKIPPD_KT_DAYTIME_THRESHOLD = 0.1
_SKIPPD_KT_EPS = 1e-6

# Hardcoded CSV columns for SKIPP'd (the user requested no auto-detection).
_SKIPPD_TIME_COL = "Date"
_SKIPPD_PV_COL = "Huang_E4102_kW"

# Fallback site coordinates (Stanford SOLAR / Huang Engineering rooftop) used when
# ``site.{latitude, longitude}`` is missing from the dataset YAML. Folsom reads
# coords from ``<data_dir>/info.yaml``; SKIPP'd has no info.yaml, so the dataset
# YAML's ``site:`` block is the source of truth.
_DEFAULT_SKIPPD_LAT = 37.4275
_DEFAULT_SKIPPD_LON = -122.1697


def _compute_skippd_p_cs(
    lat: float,
    lon: float,
    utc_index: pd.DatetimeIndex,
) -> np.ndarray:
    """
    Normalized clear-sky GHI (``clearsky_ghi / _SKIPPD_CS_GHI_SCALE``, clipped to
    ``[0, 1.2]``).

    Direct mirror of :func:`dataloader.folsom._compute_folsom_p_cs`; only the
    scaling constant differs (Folsom: 1100 W/m²; SKIPP'd: 1000 W/m²). NO POA
    transposition (Folsom is GHI-only and so is the SKIPP'd port; tilt/azimuth
    are explicitly NOT used here).
    """
    idx = utc_index
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")
    loc = pvlib.location.Location(float(lat), float(lon))
    cs = loc.get_clearsky(idx, model="ineichen")
    ghi_cs = np.asarray(cs["ghi"].values, dtype=np.float64)
    p_cs = np.clip(ghi_cs / _SKIPPD_CS_GHI_SCALE, 0.0, 1.2)
    return p_cs.astype(np.float32, copy=False)


def _skippd_progress(msg: str) -> None:
    """Progress to stderr so training stdout stays clean; SKIPPD_QUIET=1 disables.

    Mirrors :func:`dataloader.folsom._folsom_progress`.
    """
    if os.environ.get("SKIPPD_QUIET", "").strip().lower() in ("1", "true", "yes"):
        return
    print(f"[SKIPPd] {msg}", file=sys.stderr, flush=True)


def _load_skippd_pv_csv(path: Path) -> tuple[pd.DataFrame, str, str]:
    """
    Load the single SKIPP'd PV CSV into memory.

    Returns ``(df, time_col, pv_col)`` where ``df`` has columns ``[time_col, pv_col]``
    (hardcoded ``Date`` and ``Huang_E4102_kW``). Mirrors
    :func:`dataloader.folsom._load_folsom_irradiance_csv` but without GHI/DNI/DHI
    auto-detection.
    """
    p = path.resolve()
    _skippd_progress(f"loading PV CSV {p.name} into memory ...")
    raw = pd.read_csv(p, engine="c")
    missing = [c for c in (_SKIPPD_TIME_COL, _SKIPPD_PV_COL) if c not in raw.columns]
    if missing:
        raise KeyError(
            f"{p.name}: missing required SKIPP'd column(s): {missing} "
            f"(expected {_SKIPPD_TIME_COL!r} and {_SKIPPD_PV_COL!r})"
        )
    df = raw[[_SKIPPD_TIME_COL, _SKIPPD_PV_COL]].copy()
    df[_SKIPPD_TIME_COL] = pd.to_datetime(
        df[_SKIPPD_TIME_COL], format="%Y-%m-%dT%H:%M:%S", errors="coerce"
    )
    if bool(df[_SKIPPD_TIME_COL].isna().any()):
        raise ValueError(f"{p.name}: NaT in {_SKIPPD_TIME_COL!r} after parsing")
    df[_SKIPPD_PV_COL] = pd.to_numeric(df[_SKIPPD_PV_COL], errors="coerce")
    _skippd_progress(
        f"PV CSV ready: {len(df):,} rows in RAM "
        f"({_SKIPPD_TIME_COL!r}, {_SKIPPD_PV_COL!r})"
    )
    return df, _SKIPPD_TIME_COL, _SKIPPD_PV_COL


def _skippd_to_timestamps(values) -> list[pd.Timestamp]:
    """Parse values as pandas timestamps; mirrors ``_folsom_to_timestamps``."""
    return [pd.Timestamp(v) for v in values]


def load_skippd_conf(path: Path | str) -> dict:
    """Load a SKIPP'd dataset YAML. Mirrors :func:`dataloader.folsom.load_folsom_conf`."""
    if path is None:
        raise TypeError("load_skippd_conf(path) is required; no canonical default")
    p = Path(path)
    with p.open() as f:
        return yaml.safe_load(f) or {}


class SkippdPvDataset(Dataset):
    """
    Stanford SKIPP'd PV dataset (single rooftop PV CSV + sky Zarr; no NWP, no satellite).

    Constructor signature matches :class:`dataloader.folsom.FolsomIrradianceDataset` so
    the SKIPP'd trainer's ``_dataset_kwargs(...)`` builder is the Folsom builder with a
    different default YAML filename. The ``satimg_*`` kwargs are accepted for API
    parity but unused — ``sat_tensor`` and ``sat_timefeats`` in returned samples are
    ``None`` (same as Folsom).

    ``pv_dir`` must contain exactly one PV CSV with columns ``Date`` and
    ``Huang_E4102_kW``. Site coordinates come from the per-instance ``config_path``'s
    ``site.{latitude, longitude}`` (with a Stanford rooftop fallback) — SKIPP'd has no
    ``<paths.data_dir>/info.yaml``. ``paths.sky_format`` must be ``zarr``.

    Splits: rows are partitioned 60% train / 10% val / 30% test (same as Folsom).
    Train samples a random valid anchor per ``__getitem__`` (epoch length defaults to
    ``_DEFAULT_SKIPPD_TRAIN_EPOCH_LEN``; settable via ``self._train_epoch_len``);
    val/test use the respective ``*_anchor_stride_min`` strides.
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
            raise FileNotFoundError(
                f"SkippdPvDataset config_path not found: {self._config_path}"
            )
        if split not in ("train", "val", "test"):
            raise ValueError("split must be 'train', 'val', or 'test'")
        self.split = split

        if skyimg_window_size < 1:
            raise ValueError("skyimg_window_size must be >= 1")
        self.skyimg_window_size = int(skyimg_window_size)
        if satimg_window_size < 1:
            raise ValueError("satimg_window_size must be >= 1")
        self.satimg_window_size = int(satimg_window_size)

        if csv_interval_min <= 0 or pv_input_interval_min % csv_interval_min:
            raise ValueError("pv_input_interval_min must be a positive multiple of csv_interval_min")
        if pv_output_interval_min % csv_interval_min:
            raise ValueError("pv_output_interval_min must be a positive multiple of csv_interval_min")
        self._sx = pv_input_interval_min // csv_interval_min
        self._sy = pv_output_interval_min // csv_interval_min
        self.pv_input_len = int(pv_input_len)
        self.pv_output_len = int(pv_output_len)
        self.pv_output_interval_min = int(pv_output_interval_min)
        self._lx = self.pv_input_len
        self._ly = self.pv_output_len

        tf = float(pv_train_time_fraction)
        if not (0.0 < tf < 1.0):
            raise ValueError("pv_train_time_fraction must be strictly between 0 and 1")
        self._pv_train_time_fraction = tf

        if skyimg_time_resolution_min <= 0:
            raise ValueError("skyimg_time_resolution_min must be positive")
        self._skyimg_dt_min = int(skyimg_time_resolution_min)
        self._skyimg_dir = Path(skyimg_dir).resolve()
        if skyimg_spatial_size < 1:
            raise ValueError("skyimg_spatial_size must be >= 1")
        self._skyimg_spatial_size = int(skyimg_spatial_size)

        if len(satimg_npy_shape_hwc) != 3 or any(x < 1 for x in satimg_npy_shape_hwc):
            raise ValueError("satimg_npy_shape_hwc must be three positive ints (H, W, C)")
        self._satimg_npy_shape_hwc = tuple(int(x) for x in satimg_npy_shape_hwc)
        if satimg_time_resolution_min <= 0:
            raise ValueError("satimg_time_resolution_min must be positive")
        self._satimg_dt_min = int(satimg_time_resolution_min)
        self._satimg_dir = Path(satimg_dir).resolve() if str(satimg_dir) else Path(".")

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

        # Site: read from the dataset YAML's ``site:`` block. SKIPP'd has no
        # ``<data_dir>/info.yaml`` (Folsom reads coords from that file); fall back
        # to the Stanford rooftop defaults with a warning when site.lat/lon are
        # missing so the dataset can still construct.
        with self._config_path.open() as f:
            conf = yaml.safe_load(f) or {}
        paths_cfg = conf.get("paths") or {}
        site_cfg = conf.get("site") or {}
        lat = site_cfg.get("latitude")
        lon = site_cfg.get("longitude")
        if lat is None or lon is None:
            warnings.warn(
                f"{self._config_path}: site.latitude / site.longitude missing; "
                f"falling back to SKIPP'd defaults "
                f"(lat={_DEFAULT_SKIPPD_LAT}, lon={_DEFAULT_SKIPPD_LON}).",
                stacklevel=2,
            )
            lat = _DEFAULT_SKIPPD_LAT
            lon = _DEFAULT_SKIPPD_LON
        self.latitude = float(lat)
        self.longitude = float(lon)

        raw_sf = paths_cfg.get("sky_format", "zarr")
        sky_fmt = str(raw_sf).strip().lower()
        if sky_fmt != "zarr":
            raise ValueError(
                f"paths.sky_format must be 'zarr' for SKIPP'd (got {raw_sf!r}) in {self._config_path}"
            )
        self._sky_format = sky_fmt

        # API parity with PVDataset / FolsomIrradianceDataset.
        self.devDn_list = [0]

        # CSV: glob ``pv_dir`` for *.csv (PVDataset convention); SKIPP'd expects exactly one.
        self.sample_files = list_csv_files(data_dir=pv_dir)
        if not self.sample_files:
            raise FileNotFoundError(f"No CSV files in {pv_dir!r}")
        if len(self.sample_files) != 1:
            names = ", ".join(p.name for p in self.sample_files)
            raise RuntimeError(
                f"SKIPP'd dataset expects exactly one PV CSV under {pv_dir!r}, "
                f"found {len(self.sample_files)}: {names}"
            )
        self._csv_path = self.sample_files[0].resolve()
        _skippd_progress(f"dataset split={split!r}: preparing {self._csv_path.name} ...")

        # Sky: Zarr only (SKIPP'd has no JPG path). Direct mirror of Folsom's Zarr branch.
        self._sky_gap_threshold = pd.Timedelta(minutes=5)
        self._sky_anchor_max_lag = pd.Timedelta(minutes=5)
        if xr is None:
            raise ImportError(
                "paths.sky_format=zarr requires ``xarray`` (and a Zarr backend such as ``zarr``). "
                "Install them."
            )
        zp = self._skyimg_dir
        if not zp.exists():
            raise FileNotFoundError(f"sky Zarr path not found: {zp}")
        zkey = zp.resolve().as_posix()
        if zkey not in _ZARR_SKY_DS_CACHE:
            _ZARR_SKY_DS_CACHE[zkey] = xr.open_zarr(zp)
        self._skyimg_ds = _ZARR_SKY_DS_CACHE[zkey]
        self._validate_sky_zarr_schema(self._skyimg_ds)
        try:
            nt = _folsom_sky_zarr_len_time_utc(self._skyimg_ds)
        except Exception:
            nt = 0
        _skippd_progress(f"sky Zarr: {zp}  (time steps ≈ {nt:,})")

        # PV CSV: one in-memory table (Luoyang ``_csv_cache`` style).
        self._df, self._time_col, self._pv_col = _load_skippd_pv_csv(self._csv_path)
        self._n = int(len(self._df))
        if self._n < 1:
            raise RuntimeError(f"{self._csv_path.name}: expected at least one data row")

        # Precompute normalized clear-sky GHI per CSV row once. Direct mirror of
        # Folsom's per-row pvlib (ineichen) precomputation. No POA transposition.
        _skippd_progress("computing per-row clear-sky GHI via pvlib (ineichen) ...")
        _times_utc = pd.DatetimeIndex(
            pd.to_datetime(self._df[self._time_col].to_numpy(), utc=True)
        )
        self._p_cs_full = _compute_skippd_p_cs(self.latitude, self.longitude, _times_utc)
        _skippd_progress(
            f"p_cs ready: {len(self._p_cs_full):,} rows, max={float(self._p_cs_full.max()):.3f}"
        )

        # Anchor bookkeeping (Luoyang / Folsom convention: anchor = last input row index).
        n = self._n
        lx, ly = self._lx, self._ly
        sx, sy = self._sx, self._sy
        amin = (lx - 1) * sx
        y_last_off = sy * ly
        amax = n - 1 - y_last_off
        if n == 0 or amin > amax:
            raise RuntimeError(
                f"{self._csv_path.name}: no anchor fits bounds "
                f"(n={n}, need {amin}<=anchor<={amax}); check row count and window lengths"
            )
        self._anchors = np.arange(amin, amax + 1, dtype=np.intp)
        self._x_tail_1d = (-(lx - 1) * sx + np.arange(lx, dtype=np.intp) * sx).astype(np.intp, copy=False)
        self._y_off_1d = (sy + np.arange(ly, dtype=np.intp) * sy).astype(np.intp, copy=False)

        # Fixed 60% / 10% / 30% train/val/test split (matches Folsom / PVDataset).
        split_train_end = int(n * 0.6)
        split_val_end = int(n * 0.7)
        if not (0 < split_train_end < split_val_end < n):
            raise ValueError(
                f"fixed 60%/10%/30% row split invalid for n={n}: "
                f"split_train_end={split_train_end}, split_val_end={split_val_end}"
            )
        min_row = self._anchors - (lx - 1) * sx
        max_row = self._anchors + ly * sy
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

        train_positions = np.nonzero(self._train_anchor_mask)[0]
        self._train_anchor_positions = train_positions.astype(np.intp, copy=False)
        self._num_train_anchors = int(train_positions.size)

        val_positions = np.nonzero(self._val_anchor_mask)[0]
        self._val_r_indices = val_positions[::self._val_anchor_stride_rows].astype(np.intp, copy=False)
        self._num_val_windows = int(self._val_r_indices.size)
        if self.split == "val" and self._num_val_windows == 0:
            raise RuntimeError(
                "split=val: no val anchors after stride subsampling "
                "(reduce val_anchor_stride_min or widen the val segment)"
            )

        test_positions = np.nonzero(self._test_anchor_mask)[0]
        self._test_r_indices = test_positions[::self._test_anchor_stride_rows].astype(np.intp, copy=False)
        self._num_test_windows = int(self._test_r_indices.size)
        if self.split == "test" and self._num_test_windows == 0:
            raise RuntimeError(
                "split=test: no test anchors after time split and stride "
                "(reduce test_anchor_stride_min or widen the test segment)"
            )

        # Internal: train epoch length (random anchors per epoch). Builders may override.
        self._train_epoch_len = _DEFAULT_SKIPPD_TRAIN_EPOCH_LEN

        # Train anchor validity filter: keep only anchors whose Y window has at least
        # one daytime row (kt_mask == 1, i.e. p_cs > _SKIPPD_KT_DAYTIME_THRESHOLD).
        # Mirrors Folsom's GHI > 10 W/m² filter in intent (avoid all-night windows);
        # SKIPP'd uses kt_mask because the CSV has no GHI column to threshold on.
        if self.split == "train":
            self._train_anchor_valid_positions = self._compute_train_anchor_valid_positions()
        else:
            self._train_anchor_valid_positions = self._train_anchor_positions

    def _compute_train_anchor_valid_positions(self) -> np.ndarray:
        """
        Scan the per-row kt_mask once and return the subset of
        ``self._train_anchor_positions`` whose Y window has any ``kt_mask == 1`` row.

        Direct analog of :meth:`FolsomIrradianceDataset._compute_train_anchor_valid_positions`;
        SKIPP'd uses the clear-sky daytime mask instead of GHI > 10 W/m² because the
        SKIPP'd CSV has no GHI column.
        """
        _skippd_progress(
            f"train anchor filter: scanning kt_mask (p_cs > {_SKIPPD_KT_DAYTIME_THRESHOLD:g}) "
            "for daytime Y windows ..."
        )
        kt_mask_full = (self._p_cs_full > _SKIPPD_KT_DAYTIME_THRESHOLD).astype(np.float32, copy=False)

        train_anchor_rows = self._anchors[self._train_anchor_positions]  # [N_train]
        if train_anchor_rows.size == 0:
            return self._train_anchor_positions
        y_rows = train_anchor_rows[:, None] + self._y_off_1d[None, :]
        y_kt = kt_mask_full[y_rows]
        has_daytime = (y_kt > _SKIPPD_TRAIN_DAYTIME_KT_MASK_THRESHOLD).any(axis=1)
        kept = self._train_anchor_positions[has_daytime].astype(np.intp, copy=False)
        n_kept = int(kept.size)
        n_total = int(train_anchor_rows.size)
        _skippd_progress(
            f"train anchor filter: {n_kept:,} / {n_total:,} train anchors kept "
            f"(kt_mask==1 in Y window)"
        )
        if n_kept == 0:
            raise RuntimeError(
                "split=train: no anchor with any daytime (kt_mask==1) step in Y window; "
                "check clear-sky configuration"
            )
        return kept

    # ------------------------------------------------------------------ #
    # Sky-Zarr helpers — copied verbatim from dataloader/folsom.py        #
    # (module-level _folsom_* helpers are imported above; the instance   #
    # methods below mirror Folsom line-by-line, only the class name      #
    # differs).                                                          #
    # ------------------------------------------------------------------ #
    def _black_sky_tensor(self) -> torch.Tensor:
        """Return ``[3, s, s]`` float32 (zeros). Same convention as Folsom."""
        s = self._skyimg_spatial_size
        return torch.zeros((3, s, s), dtype=torch.float32)

    @staticmethod
    def _sky_filename_ts(ts_raw) -> pd.Timestamp:
        """Naive UTC timestamp for sky frame alignment; seconds floored to 0."""
        ts = pd.Timestamp(ts_raw)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return ts.replace(second=0, microsecond=0, nanosecond=0)

    def _validate_sky_zarr_schema(self, ds: Any) -> None:
        """Require ``images`` plus an alignable ``time_utc`` timeline (mirror Folsom)."""
        if "images" not in ds.data_vars:
            raise KeyError(
                "SKIPP'd sky Zarr must define data variable ``images`` "
                "(see config/datasets/conf_skippd.yaml)."
            )
        _folsom_sky_zarr_time_dim_and_values(ds, ds["images"])

    def _nominal_sky_frame_times(self, t_end_wall: Any) -> list[pd.Timestamp]:
        """Oldest→newest ``skyimg_window_size`` timestamps spaced by ``_skyimg_dt_min``."""
        t_end = self._sky_filename_ts(t_end_wall)
        w = self.skyimg_window_size
        dt = self._skyimg_dt_min
        return [t_end - timedelta(minutes=(w - 1 - i) * dt) for i in range(w)]

    def _resize_sky_chw(self, chw: torch.Tensor) -> torch.Tensor:
        """``[3,H,W]`` float32 → ``[3,s,s]`` bilinear."""
        s = self._skyimg_spatial_size
        if chw.shape[-2:] == (s, s):
            return chw
        x = chw.unsqueeze(0)
        y = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
        return y.squeeze(0)

    def _tensor_from_zarr_image_tile(self, tile: np.ndarray) -> torch.Tensor:
        """One Zarr timestep tile (HWC or CHW) → ``[3, s, s]`` float32 in ``[0, 1]``."""
        t = torch.from_numpy(np.asarray(tile, dtype=np.float32))
        if t.ndim != 3:
            raise ValueError(f"sky Zarr ``images`` tile must be 3D, got shape {tuple(tile.shape)}")
        if t.shape[-1] == 3 and t.shape[0] != 3:
            t = t.permute(2, 0, 1).contiguous()
        elif t.shape[0] != 3:
            raise ValueError(
                f"sky Zarr ``images`` must be HWC or CHW with 3 channels; got {tuple(t.shape)}"
            )
        mx = float(t.detach().max().item()) if t.numel() else 0.0
        if mx > 1.5:
            t = (t / 255.0).clamp(0.0, 1.0)
        else:
            t = t.clamp(0.0, 1.0)
        return self._resize_sky_chw(t)

    def _stack_sky_from_zarr(self, t_end_wall: Any) -> torch.Tensor:
        """
        Stack ``[W, 3, H, W]`` from Zarr using the same nominal UTC grid as Folsom's
        JPEG loader. Black frames when no Zarr row falls within the nominal window
        or the newest kept row is too far before the anchor.
        """
        w = self.skyimg_window_size
        nominal = self._nominal_sky_frame_times(t_end_wall)
        black = torch.stack([self._black_sky_tensor()] * w, dim=0)
        if self._skyimg_ds is None:
            return black
        ds = self._skyimg_ds
        img = ds["images"]
        time_dim, raw_t = _folsom_sky_zarr_time_dim_and_values(ds, img)
        z_times = _folsom_parse_zarr_utc_naive(raw_t)
        t_lo, t_hi = pd.Timestamp(nominal[0]), pd.Timestamp(nominal[-1])
        mask = (z_times >= t_lo) & (z_times <= t_hi)
        idx = np.nonzero(np.asarray(mask, dtype=bool))[0]
        if idx.size == 0:
            return black
        sub = img.isel({time_dim: idx})
        z_sub = z_times[idx]
        z_np = np.asarray(sub.values, dtype=np.float32)
        ta = int(sub.get_axis_num(time_dim))
        if ta != 0:
            z_np = np.moveaxis(z_np, ta, 0)
        if z_np.shape[0] != len(z_sub):
            raise RuntimeError(
                f"sky Zarr internal mismatch: time len {len(z_sub)} vs stacked dim 0 {z_np.shape[0]}"
            )
        t_end = pd.Timestamp(self._sky_filename_ts(nominal[-1]))
        newest = z_sub[-1]
        if t_end - newest > self._sky_anchor_max_lag:
            return black
        z_ns = z_sub.asi8.astype(np.int64)
        max_delta = int(pd.Timedelta(seconds=90).value)
        frames: list[torch.Tensor] = []
        for want in nominal:
            wn = self._sky_filename_ts(want)
            want_ns = int(wn.value)
            j = int(np.argmin(np.abs(z_ns - want_ns)))
            if abs(int(z_ns[j]) - want_ns) > max_delta:
                frames.append(self._black_sky_tensor())
            else:
                frames.append(self._tensor_from_zarr_image_tile(z_np[j]))
        return torch.stack(frames, dim=0)

    # ------------------------------------------------------------------ #
    # NWP: SKIPP'd has none. Emit zeros + invalid mask of Folsom's shape #
    # so the model's nwp_tensor[:, :, 0|2|-1] indexing doesn't crash and #
    # the trainer can drop --use-nwp / NWP-remap code paths entirely.    #
    # ------------------------------------------------------------------ #
    def _zero_nwp_tensor(self) -> torch.Tensor:
        """Zeros features + ones mask, ``[T_out, len(_FOLSOM_NWP_FEATURE_COLS) + 1]``.

        Mirrors :meth:`FolsomIrradianceDataset._interpolate_nwp` when its merged-NWP
        CSV is missing (zeros for features, ones for the per-step invalid mask).
        """
        t_out = self.pv_output_len
        c = len(_FOLSOM_NWP_FEATURE_COLS)
        zeros = np.zeros((t_out, c), dtype=np.float32)
        ones_mask = np.ones((t_out, 1), dtype=np.float32)
        return torch.from_numpy(np.concatenate([zeros, ones_mask], axis=1))

    # ------------------------------------------------------------------ #
    # Sample building                                                     #
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        if self.split == "train":
            return self._train_epoch_len
        if self.split == "val":
            return self._num_val_windows
        return self._num_test_windows

    def sky_inspect(self, anchor: int) -> dict:
        """Resolve the sky-Zarr slice covering ``anchor`` without building tensors."""
        x_idx = anchor + self._x_tail_1d
        sub_x = self._df.iloc[x_idx]
        t_x_end = sub_x[self._time_col].iloc[-1]
        nominal = self._nominal_sky_frame_times(t_x_end)
        t_last_n = pd.Timestamp(nominal[-1])
        n_z = 0
        if self._skyimg_ds is not None and len(nominal) > 0:
            try:
                n_z = _folsom_sky_zarr_count_in_time_range(
                    self._skyimg_ds, nominal[0], nominal[-1]
                )
            except Exception:
                n_z = 0
        return {
            "anchor_row": int(anchor),
            "last_input_time_utc_naive": str(t_last_n),
            "n_frames": len(nominal),
            "utc_times": [str(pd.Timestamp(t)) for t in nominal],
            "n_files_found": n_z,
            "skyimg_dir": self._skyimg_dir,
            "sky_format": "zarr",
            "zarr_slice_timesteps": n_z,
        }

    def _build_tensors(self, anchor: int) -> dict[str, Any]:
        x_idx = anchor + self._x_tail_1d
        y_idx = anchor + self._y_off_1d
        sub_x = self._df.iloc[x_idx]
        sub_y = self._df.iloc[y_idx]

        # Raw PV (kW) input + target.
        pow_x = sub_x[self._pv_col].to_numpy(dtype=np.float32, copy=False)
        pow_y = sub_y[self._pv_col].to_numpy(dtype=np.float32, copy=False)
        pow_x = np.nan_to_num(pow_x, nan=0.0, posinf=0.0, neginf=0.0)
        pow_y_raw = pow_y  # keep pre-sanitization for the target mask (isfinite)
        pow_y = np.nan_to_num(pow_y, nan=0.0, posinf=0.0, neginf=0.0)

        valid_in = np.isfinite(sub_x[self._pv_col].to_numpy())
        input_mask = torch.from_numpy(valid_in.astype(np.float32)).unsqueeze(0)
        valid_out = np.isfinite(pow_y_raw)
        target_mask = torch.from_numpy(valid_out.astype(np.float32))

        # Clear-sky / kt fields. Direct mirror of Folsom's recipe — the only
        # difference is the normalization constants (Folsom: /1100 for both PV
        # and clear-sky GHI; SKIPP'd: /30 for PV and /1000 for clear-sky GHI
        # because PV/GHI live in different unit spaces).
        p_cs_x = self._p_cs_full[x_idx]
        p_cs_y = self._p_cs_full[y_idx]
        kt_mask_np = (p_cs_x > _SKIPPD_KT_DAYTIME_THRESHOLD).astype(np.float32)
        pv_norm_x = pow_x / _SKIPPD_PV_SCALE
        kt_np = (pv_norm_x / (p_cs_x + _SKIPPD_KT_EPS)) * kt_mask_np
        kt = torch.from_numpy(kt_np.astype(np.float32)).unsqueeze(0)
        kt_mask = torch.from_numpy(kt_mask_np).unsqueeze(0)
        p_cs = torch.from_numpy(p_cs_x.astype(np.float32)).unsqueeze(0)
        target_p_cs = torch.from_numpy(p_cs_y.astype(np.float32))
        # p_mean is literally Folsom's constant 1.0 (NOT the SKIPP'd rated 30 kW).
        # Reason: Folsom's trainer reconstruction is
        #     pv_pred = kt_pred * target_p_cs * p_mean ≈ target_pv
        # which is an algebraic identity in the normalized space iff p_mean = 1.0.
        # SKIPP'd target_pv = pv_kW / 30 is also in normalized space, so p_mean
        # must stay 1.0 for the trainer's pv_pred ≈ target_pv reconstruction to
        # hold. The user's "use 30 for p_mean" suggestion in the task brief would
        # require target_pv to be in raw kW, which conflicts with the trainer-side
        # "1100 → 30" substitution (the trainer multiplies the normalized MAE by
        # the capacity = 30 kW; that only makes sense when target_pv is /30).
        p_mean = torch.tensor(1.0, dtype=torch.float32)

        x_times = sub_x[self._time_col]
        if bool(x_times.isna().any()):
            raise ValueError(f"NaT in {self._time_col!r} for input window")
        timestamps = _skippd_to_timestamps(x_times.tolist())
        time0 = timestamps[-1]
        forecast_timestamps = [
            time0 + pd.Timedelta(minutes=self.pv_output_interval_min * (i + 1))
            for i in range(self.pv_output_len)
        ]
        nwp_tensor = self._zero_nwp_tensor()

        pv_solar = compute_solar_features(timestamps, self.latitude, self.longitude)
        pv_tf = solar_features_encoder(pv_solar)
        pv_dtf = delta_time_encoder(timestamps, time0)
        pv_timefeats = torch.cat([pv_tf, pv_dtf.unsqueeze(1)], dim=1)

        forecast_solar = compute_solar_features(forecast_timestamps, self.latitude, self.longitude)
        f_tf = solar_features_encoder(forecast_solar)
        f_dtf = delta_time_encoder(forecast_timestamps, time0)
        forecast_timefeats = torch.cat([f_tf, f_dtf.unsqueeze(1)], dim=1)

        t_x_end = sub_x[self._time_col].iloc[-1]
        nominal = self._nominal_sky_frame_times(t_x_end)
        skimg_tensor = self._stack_sky_from_zarr(t_x_end)
        skimg_solar = compute_solar_features(nominal, self.latitude, self.longitude)
        skimg_tf = solar_features_encoder(skimg_solar)
        skimg_dtf = delta_time_encoder(nominal, time0)
        skimg_timefeats = torch.cat([skimg_tf, skimg_dtf.unsqueeze(1)], dim=1)
        skimg_timestamps = [t.strftime("%Y%m%d%H%M%S") for t in nominal]

        input_timestamps_utc = [str(pd.Timestamp(t)) for t in timestamps]
        forecast_timestamps_utc = [str(pd.Timestamp(t)) for t in forecast_timestamps]
        # ``dev_idx = 700`` literally mirrors Folsom (arbitrary slot inside the
        # model's ``nn.Embedding(1000)`` device-id table; SKIPP'd is single-sensor).
        dev_idx = torch.tensor(700, dtype=torch.long)

        # PV input / target in normalized space (analog of Folsom's ghi / 1100).
        pv_tensor = torch.from_numpy(pv_norm_x.astype(np.float32)).unsqueeze(0)
        target_pv_tensor = torch.from_numpy((pow_y / _SKIPPD_PV_SCALE).astype(np.float32))

        return {
            "dev_idx": dev_idx,
            "pv": pv_tensor,
            "pv_mask": input_mask,
            "pv_timefeats": pv_timefeats,
            "kt": kt,
            "kt_mask": kt_mask,
            "p_cs": p_cs,
            "p_mean": p_mean,
            "input_mask": input_mask,
            "forecast_timefeats": forecast_timefeats,
            "target_pv": target_pv_tensor,
            "target_mask": target_mask,
            "target_p_cs": target_p_cs,
            "sat_tensor": None,
            "sat_timefeats": None,
            "skimg_tensor": skimg_tensor,
            "skimg_timefeats": skimg_timefeats,
            "nwp_tensor": nwp_tensor,
            "input_timestamps_utc": input_timestamps_utc,
            "forecast_timestamps_utc": forecast_timestamps_utc,
            "skimg_timestamps": skimg_timestamps,
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.split == "train":
            r = int(np.random.choice(self._train_anchor_valid_positions))
        elif self.split == "val":
            r = int(self._val_r_indices[idx])
        else:
            r = int(self._test_r_indices[idx])
        anchor = int(self._anchors[r])
        return self._build_tensors(anchor)


def _resolve_skippd_dataset_paths(conf: dict, conf_path: Path) -> tuple[Path, Path, Path]:
    """Return ``(pv_dir, skyimg_dir, satimg_dir)`` from a SKIPP'd dataset YAML.

    Mirrors :func:`dataloader.folsom._resolve_folsom_dataset_paths`.
    """
    paths_cfg = conf.get("paths") or {}
    raw_dd = paths_cfg.get("data_dir")
    if raw_dd is None or str(raw_dd).strip() == "":
        raise KeyError(f"dataset config paths.data_dir is required (in {conf_path})")
    dd = Path(str(raw_dd))
    data_dir = dd.resolve() if dd.is_absolute() else (_PROJECT_ROOT / dd).resolve()

    def _req(key: str) -> Path:
        v = paths_cfg.get(key)
        if v is None or str(v).strip() == "":
            raise KeyError(f"dataset config paths.{key} is required (in {conf_path})")
        return (data_dir / Path(str(v))).resolve()

    return _req("pv_path"), _req("sky_image_path"), _req("sat_path")


def _skippd_kwargs_from_conf(conf: dict, cfg_path: Path) -> dict:
    """Shared kwargs builder for ``build_skippd_pv_datasets_from_conf`` and the smoke CLI.

    Returns the full ``SkippdPvDataset.__init__`` kwargs **without** ``split``; callers
    add ``split=...`` before instantiating. Mirrors Folsom's pattern.
    """
    sampling_cfg = conf.get("sampling") or {}
    if not sampling_cfg:
        raise KeyError(f"dataset config {cfg_path} is missing a non-empty 'sampling:' section")

    def _req_s(key: str):
        if key not in sampling_cfg:
            raise KeyError(f"dataset config sampling.{key} is required (in {cfg_path})")
        return sampling_cfg[key]

    pv_dir, skyimg_dir, satimg_dir = _resolve_skippd_dataset_paths(conf, cfg_path)
    shwc = _req_s("satimg_npy_shape_hwc")
    if not isinstance(shwc, (list, tuple)) or len(shwc) != 3:
        raise ValueError(f"sampling.satimg_npy_shape_hwc must be [H, W, C] (in {cfg_path})")
    return dict(
        config_path=str(cfg_path),
        pv_dir=str(pv_dir),
        skyimg_dir=str(skyimg_dir),
        satimg_dir=str(satimg_dir),
        csv_interval_min=int(_req_s("csv_interval_min")),
        pv_input_interval_min=int(_req_s("pv_input_interval_min")),
        pv_input_len=int(_req_s("pv_input_len")),
        pv_output_interval_min=int(_req_s("pv_output_interval_min")),
        pv_output_len=int(_req_s("pv_output_len")),
        pv_train_time_fraction=float(_req_s("pv_train_time_fraction")),
        test_anchor_stride_min=int(_req_s("test_anchor_stride_min")),
        val_anchor_stride_min=int(_req_s("val_anchor_stride_min")),
        test_collect_time_match_tolerance_min=int(_req_s("test_collect_time_match_tolerance_min")),
        skyimg_window_size=int(_req_s("skyimg_window_size")),
        skyimg_time_resolution_min=int(_req_s("skyimg_time_resolution_min")),
        skyimg_spatial_size=int(_req_s("skyimg_spatial_size")),
        satimg_window_size=int(_req_s("satimg_window_size")),
        satimg_time_resolution_min=int(_req_s("satimg_time_resolution_min")),
        satimg_npy_shape_hwc=tuple(int(x) for x in shwc),
    )


def build_skippd_pv_datasets_from_conf(
    conf: dict | None = None,
    *,
    conf_path: Path | str | None = None,
    train_epoch_len: int = 50_000,
    skyimg_window_size: int | None = None,
) -> tuple[SkippdPvDataset, SkippdPvDataset]:
    """
    Build train/test :class:`SkippdPvDataset` from a SKIPP'd dataset YAML.

    Mirrors :func:`dataloader.folsom.build_folsom_irradiance_datasets_from_conf`.
    Reads ``paths.{data_dir, pv_path, sky_image_path, sat_path, sky_format}`` and the
    ``sampling:`` section. Site lat/lon come from the YAML's ``site:`` block.
    """
    if conf_path is None:
        raise TypeError("build_skippd_pv_datasets_from_conf: conf_path is required")
    cfg_path = Path(conf_path)
    if conf is None:
        conf = load_skippd_conf(cfg_path)

    kwargs = _skippd_kwargs_from_conf(conf, cfg_path)
    if skyimg_window_size is not None:
        kwargs["skyimg_window_size"] = int(skyimg_window_size)
    train_ds = SkippdPvDataset(split="train", **kwargs)
    test_ds = SkippdPvDataset(split="test", **kwargs)
    train_ds._train_epoch_len = max(1, int(train_epoch_len))
    return train_ds, test_ds


__all__ = [
    "SkippdPvDataset",
    "build_skippd_pv_datasets_from_conf",
    "load_skippd_conf",
    "run_smoke_cli",
    "_SKIPPD_PV_SCALE",
    "_SKIPPD_CS_GHI_SCALE",
    "_SKIPPD_KT_DAYTIME_THRESHOLD",
    "_SKIPPD_KT_EPS",
]


# ---------------------------------------------------------------------------- #
# Smoke entry: instantiate the dataset and print one sample's shapes.          #
# ---------------------------------------------------------------------------- #
def run_smoke_cli(argv: list[str] | None = None) -> int:
    """
    Smoke: instantiate ``SkippdPvDataset`` from the YAML, pull one sample, print shapes.

    Mirrors the Folsom smoke in spirit but minimal — SKIPP'd has no per-anchor
    irradiance / NWP report to render.
    """
    parser = argparse.ArgumentParser(
        description="Smoke-test SkippdPvDataset (reads paths from YAML)"
    )
    parser.add_argument(
        "--conf",
        "--config",
        type=Path,
        default=_DEFAULT_SKIPPD_DATASET_CONFIG,
        dest="conf",
    )
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        with open(args.conf) as f:
            conf = yaml.safe_load(f) or {}
        kwargs = _skippd_kwargs_from_conf(conf, Path(args.conf))
        ds = SkippdPvDataset(split=args.split, **kwargs)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Failed to load SKIPP'd data from {args.conf.resolve()}:\n  {e}", file=sys.stderr)
        return 1

    print(f"[conf] {args.conf.resolve()}")
    print(f"  csv={ds._csv_path}  split={args.split}  len(ds)={len(ds)}")
    n_ds = len(ds)
    idx = max(0, min(int(args.index), max(0, n_ds - 1)))
    s = ds[idx]
    shapes: dict[str, str] = {}
    for k, v in s.items():
        if hasattr(v, "shape"):
            shapes[k] = f"tensor{tuple(v.shape)} dtype={v.dtype}"
        elif v is None:
            shapes[k] = "None"
        elif isinstance(v, list):
            shapes[k] = f"list[len={len(v)}]"
        else:
            shapes[k] = type(v).__name__
    for k in (
        "dev_idx",
        "pv",
        "pv_mask",
        "pv_timefeats",
        "forecast_timefeats",
        "kt",
        "kt_mask",
        "p_cs",
        "p_mean",
        "target_p_cs",
        "target_pv",
        "target_mask",
        "skimg_tensor",
        "skimg_timefeats",
        "nwp_tensor",
        "sat_tensor",
        "sat_timefeats",
        "input_timestamps_utc",
        "forecast_timestamps_utc",
        "skimg_timestamps",
    ):
        print(f"  {k}: {shapes.get(k, '<missing>')}")
    print(
        f"  pv (input) min={float(s['pv'].min()):.4f} "
        f"max={float(s['pv'].max()):.4f}  "
        f"target_pv min={float(s['target_pv'].min()):.4f} "
        f"max={float(s['target_pv'].max()):.4f}"
    )
    print(
        f"  kt_mask sum(input)={float(s['kt_mask'].sum()):.0f}/{s['kt_mask'].numel()}  "
        f"target_mask sum={float(s['target_mask'].sum()):.0f}/{s['target_mask'].numel()}"
    )

    # One collated batch via the same collate the trainer uses.
    from dataloader.luoyang_zarr import collate_batched
    bs = min(2, n_ds) if n_ds > 0 else 1
    loader = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate_batched, num_workers=0)
    batch = next(iter(loader))
    print("DATALOADER (first batch):")
    for k in ("pv", "kt", "kt_mask", "p_cs", "target_pv", "target_mask",
              "target_p_cs", "skimg_tensor", "skimg_timefeats", "nwp_tensor"):
        v = batch.get(k)
        if v is None:
            print(f"  {k}: None")
        else:
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
    print("smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_smoke_cli())
