"""
SKIPP'd PV dataset: one CSV with time + PV power (kW) plus sky Zarr (no NWP, no satellite).

Modeled on :class:`dataloader.folsom.FolsomIrradianceDataset` but specialized for
Stanford SKIPP'd:

* PV CSV (``Date, Huang_E4102_kW``, 1-min UTC cadence) is loaded once and cached.
* Site lat/lon comes from the dataset YAML ``site.{latitude,longitude}``; no info.yaml.
* Clear-sky POA ``p_cs`` is precomputed once on the full CSV time index via pvlib
  (``Location.get_clearsky`` -> ``get_total_irradiance``) and cached.
* Each sample emits ``kt = pv / (p_cs * p_mean + 1e-6)`` plus ``kt_mask = (p_cs > 0.1)``;
  nighttime negatives are masked out by ``kt_mask`` (daytime negatives are tiny and kept).
* Train anchor filter: keep anchors whose Y window has at least one daytime step
  (``kt_mask == 1``). Replaces Folsom's ``GHI > 10`` filter.
* Sky stack uses the same Zarr-reading helpers as Folsom (imported / copy-pasted; see
  ``# Sky-Zarr helpers reused from dataloader/folsom.py`` block below).
* No NWP and no satellite: the dataset emits zeros for ``nwp_tensor`` and ``None`` for
  ``sat_tensor`` / ``sat_timefeats``. The SKIPP'd trainer rewrites ``nwp_tensor`` just
  before the forward pass so the model's slot-0/2 normalizations don't see -1.0 / -28.815.

POA projection assumption (tunable in ``site:`` block of the dataset YAML):
    tilt_deg = 0.0 (horizontal)
    surface_azimuth_deg = 180.0 (south-facing reference)
These match a fixed-mount rooftop reference; SKIPP'd doesn't publish exact tilt/azimuth,
so adjust ``site.tilt_deg`` / ``site.surface_azimuth_deg`` later if better info shows up.

Batch contract — :meth:`SkippdPvDataset.__getitem__` returns the union of keys consumed
by :func:`training.train_vit_test_skippd._batch_to_device` AND stacked by
:func:`dataloader.luoyang_zarr.collate_batched`:

    dev_idx, pv, pv_mask, pv_timefeats, forecast_timefeats,
    kt, kt_mask, p_cs, p_mean, target_p_cs,
    target_pv, target_mask,
    skimg_tensor, skimg_timefeats,
    sat_tensor=None, sat_timefeats=None,
    nwp_tensor (zeros placeholder; trainer overwrites),
    input_timestamps_utc, forecast_timestamps_utc, skimg_timestamps.
"""

from __future__ import annotations

import argparse
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

# Sky-Zarr helpers reused from dataloader/folsom.py (module-level functions are imported
# directly; instance methods like _stack_sky_from_zarr / _nominal_sky_frame_times can't
# be cleanly imported, so they are copy-pasted below with explicit source references).
from dataloader.folsom import (  # noqa: E402
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

# Module-level constants ----------------------------------------------------

# System rated capacity (kW). Used both as the PV normalization scale for the
# model's input/target and as ``p_mean`` in ``kt = pv / (p_cs * p_mean + 1e-6)``.
# Close to the observed max (29.59 kW); bump to 31 if you re-calibrate.
_DEFAULT_SKIPPD_P_MEAN = 30.0

# Random anchor count per train epoch (matches Folsom).
_DEFAULT_SKIPPD_TRAIN_EPOCH_LEN = 50_000

# PV reporting scale (kW). Multiply normalized predictions / targets by this to get kW.
# Analogous to Folsom's 1100 W/m^2 GHI scale.
_SKIPPD_PV_KW_SCALE = 30.0

# Default YAML for the smoke CLI (the class itself takes config_path as required arg).
_DEFAULT_SKIPPD_DATASET_CONFIG = (
    _PROJECT_ROOT / "config" / "datasets" / "conf_skippd.yaml"
)

# Fallback site coordinates if the YAML's ``site:`` block omits them (warned).
_DEFAULT_SKIPPD_LAT = 37.4275
_DEFAULT_SKIPPD_LON = -122.1697

# Hardcoded CSV columns for SKIPP'd (Date in ISO 8601 UTC, PV power in kW).
_SKIPPD_TIME_COL = "Date"
_SKIPPD_PV_COL = "Huang_E4102_kW"

# Class-level caches (shared across train/val/test splits and DataLoader workers).
_SKIPPD_CSV_CACHE: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}
_SKIPPD_ZARR_DS_CACHE: dict[str, Any] = {}
_SKIPPD_CLEARSKY_CACHE: dict[tuple[float, float, str, float, float], np.ndarray] = {}


def _skippd_progress(msg: str) -> None:
    """Progress to stderr so training stdout stays clean; set SKIPPD_QUIET=1 to disable."""
    if os.environ.get("SKIPPD_QUIET", "").strip().lower() in ("1", "true", "yes"):
        return
    print(f"[SKIPPd] {msg}", file=sys.stderr, flush=True)


def _load_skippd_pv_csv(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load the SKIPP'd PV CSV into memory and cache it across dataset instances.

    Returns ``(df, time_ns_naive_utc, pv_kw)`` where ``df`` has the two original columns,
    ``time_ns_naive_utc`` is ``int64`` ns-since-epoch (UTC, naive) for fast lookups, and
    ``pv_kw`` is the float32 power column.
    """
    p = path.resolve()
    key = p.as_posix()
    cached = _SKIPPD_CSV_CACHE.get(key)
    if cached is not None:
        return cached
    _skippd_progress(f"loading PV CSV {p.name} into memory ...")
    raw = pd.read_csv(p, engine="c")
    missing = [c for c in (_SKIPPD_TIME_COL, _SKIPPD_PV_COL) if c not in raw.columns]
    if missing:
        raise KeyError(
            f"{p.name}: missing required SKIPP'd column(s): {missing} "
            f"(expected {_SKIPPD_TIME_COL!r} and {_SKIPPD_PV_COL!r})"
        )
    df = raw[[_SKIPPD_TIME_COL, _SKIPPD_PV_COL]].copy()
    # SKIPP'd PV times are ISO 8601 (e.g. "2017-01-01T08:00:00"); CSV is already in UTC.
    df[_SKIPPD_TIME_COL] = pd.to_datetime(
        df[_SKIPPD_TIME_COL], format="%Y-%m-%dT%H:%M:%S", errors="coerce"
    )
    if bool(df[_SKIPPD_TIME_COL].isna().any()):
        raise ValueError(
            f"{p.name}: NaT in {_SKIPPD_TIME_COL!r} after parsing (expected ISO 8601 UTC)"
        )
    df[_SKIPPD_PV_COL] = pd.to_numeric(df[_SKIPPD_PV_COL], errors="coerce").astype(
        np.float32
    )
    time_ns = pd.DatetimeIndex(df[_SKIPPD_TIME_COL]).asi8.astype(np.int64, copy=False)
    pv_kw = df[_SKIPPD_PV_COL].to_numpy(dtype=np.float32, copy=False)
    _skippd_progress(
        f"PV CSV ready: {len(df):,} rows in RAM "
        f"(time span {df[_SKIPPD_TIME_COL].iloc[0]} -> {df[_SKIPPD_TIME_COL].iloc[-1]})"
    )
    cached = (df, time_ns, pv_kw)
    _SKIPPD_CSV_CACHE[key] = cached
    return cached


def _compute_skippd_p_cs_on_csv_index(
    lat: float,
    lon: float,
    csv_path: Path,
    utc_naive_index: pd.DatetimeIndex,
    *,
    tilt_deg: float,
    surface_azimuth_deg: float,
) -> np.ndarray:
    """
    Normalized clear-sky POA on the full CSV time index (cached by (lat, lon, csv, tilt, az)).

    Uses pvlib's ``Location.get_clearsky()`` default (Ineichen) then projects onto a fixed
    plane via ``get_total_irradiance``. ``p_cs = (poa_global / 1000).clip(0, 1.2)``.
    Recipe mirrors ``SPMF_preprocessing/luoyang/aggregate_by_devdn_solarfeats.py:99-123``.
    """
    key = (
        round(float(lat), 5),
        round(float(lon), 5),
        csv_path.resolve().as_posix(),
        round(float(tilt_deg), 3),
        round(float(surface_azimuth_deg), 3),
    )
    cached = _SKIPPD_CLEARSKY_CACHE.get(key)
    if cached is not None:
        return cached
    n = len(utc_naive_index)
    _skippd_progress(
        f"computing clear-sky POA for ({lat:.4f}, {lon:.4f}) tilt={tilt_deg:g} "
        f"az={surface_azimuth_deg:g} over {n:,} CSV steps ..."
    )
    # pvlib requires tz-aware DatetimeIndex; SKIPP'd CSV is already UTC.
    utc_index = utc_naive_index.tz_localize("UTC")
    loc = pvlib.location.Location(float(lat), float(lon))
    cs = loc.get_clearsky(utc_index)  # Ineichen by default
    solpos = pvlib.solarposition.get_solarposition(utc_index, float(lat), float(lon))
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=float(tilt_deg),
        surface_azimuth=float(surface_azimuth_deg),
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=cs["dni"],
        ghi=cs["ghi"],
        dhi=cs["dhi"],
        albedo=0.2,
    )
    poa_clear = poa["poa_global"].clip(lower=0)
    p_cs = (poa_clear / 1000.0).clip(0, 1.2).to_numpy(dtype=np.float32, copy=False)
    _skippd_progress(
        f"clear-sky ready: max p_cs={float(p_cs.max()):.3f}, "
        f"daytime fraction (>0.1)={float((p_cs > 0.1).mean()):.3f}"
    )
    _SKIPPD_CLEARSKY_CACHE[key] = p_cs
    return p_cs


class SkippdPvDataset(Dataset):
    """
    Stanford SKIPP'd PV dataset: one PV CSV + sky Zarr, no NWP, no satellite.

    Same constructor shape as :class:`dataloader.folsom.FolsomIrradianceDataset` for
    drop-in compatibility with the trainer's ``_dataset_kwargs`` builder. The
    ``satimg_*`` arguments are accepted for API parity but unused (no satellite data).

    Reads ``site.{latitude,longitude,tilt_deg,surface_azimuth_deg,rated_capacity_kw}``
    plus ``paths.{data_dir, pv_path, sky_image_path, sky_format}`` and ``sampling.*``
    from the per-instance ``config_path``. No info.yaml needed.

    Splits: fixed 60% / 10% / 30% by row count (same as Folsom).
    Train: random anchor per ``__getitem__`` from precomputed valid-Y positions
    (epoch length :data:`_DEFAULT_SKIPPD_TRAIN_EPOCH_LEN`).
    Val/test: deterministic strided positions per the YAML strides.
    """

    def __init__(
        self,
        config_path: str | Path,
        pv_dir: str | None = None,
        skyimg_dir: str | None = None,
        satimg_dir: str | None = None,
        *,
        split: str,
        csv_interval_min: int | None = None,
        pv_input_interval_min: int | None = None,
        pv_input_len: int | None = None,
        pv_output_interval_min: int | None = None,
        pv_output_len: int | None = None,
        pv_train_time_fraction: float | None = None,
        test_anchor_stride_min: int | None = None,
        val_anchor_stride_min: int | None = None,
        test_collect_time_match_tolerance_min: int | None = None,
        skyimg_window_size: int | None = None,
        skyimg_time_resolution_min: int | None = None,
        skyimg_spatial_size: int | None = None,
        satimg_window_size: int | None = None,
        satimg_time_resolution_min: int | None = None,
        satimg_npy_shape_hwc: tuple[int, int, int] | None = None,
        sky_format: str = "zarr",
    ):
        """
        Trainer call sites (mirroring Folsom) pass every kwarg explicitly; the smoke /
        adhoc call sites can pass only ``config_path`` + ``split`` and let the constructor
        fall back to the YAML's ``paths`` / ``sampling`` blocks. The Folsom-shape kwarg
        signature is preserved so the trainer's ``_dataset_kwargs`` builder is unchanged.
        """
        self._config_path = Path(config_path).resolve()
        if not self._config_path.is_file():
            raise FileNotFoundError(
                f"SkippdPvDataset config_path not found: {self._config_path}"
            )
        if split not in ("train", "val", "test"):
            raise ValueError("split must be 'train', 'val', or 'test'")
        self.split = split

        # YAML fallback: load the per-instance config now so any None kwargs can be filled
        # in from the ``paths:`` / ``sampling:`` blocks. The trainer passes all kwargs
        # explicitly so this fallback is a no-op there.
        with self._config_path.open() as _fb_f:
            _fb_conf = yaml.safe_load(_fb_f) or {}
        _fb_paths = _fb_conf.get("paths") or {}
        _fb_sampling = _fb_conf.get("sampling") or {}
        _fb_data_dir_raw = _fb_paths.get("data_dir")
        if _fb_data_dir_raw is None or str(_fb_data_dir_raw).strip() == "":
            _fb_data_dir = None
        else:
            _fb_dd = Path(str(_fb_data_dir_raw))
            _fb_data_dir = _fb_dd.resolve() if _fb_dd.is_absolute() else (_PROJECT_ROOT / _fb_dd).resolve()

        def _path_fb(arg: str | None, key: str, default_subdir: str | None = None) -> str:
            if arg is not None and str(arg).strip() != "":
                return str(arg)
            v = _fb_paths.get(key)
            if v is None or str(v).strip() == "":
                if default_subdir is not None:
                    v = default_subdir
                else:
                    raise KeyError(
                        f"paths.{key} is required in {self._config_path} when {key!r} kwarg is omitted"
                    )
            v_p = Path(str(v))
            if v_p.is_absolute():
                return str(v_p.resolve())
            if _fb_data_dir is None:
                raise KeyError(
                    f"paths.data_dir is required in {self._config_path} to resolve "
                    f"relative paths.{key}={v!r}"
                )
            return str((_fb_data_dir / v_p).resolve())

        def _samp_fb(arg, key: str, *, required: bool = True):
            if arg is not None:
                return arg
            if key in _fb_sampling:
                return _fb_sampling[key]
            if required:
                raise KeyError(
                    f"sampling.{key} is required in {self._config_path} when {key!r} kwarg is omitted"
                )
            return None

        pv_dir = _path_fb(pv_dir, "pv_path")
        skyimg_dir = _path_fb(skyimg_dir, "sky_image_path")
        satimg_dir = _path_fb(satimg_dir, "sat_path", default_subdir="sat")

        csv_interval_min = int(_samp_fb(csv_interval_min, "csv_interval_min"))
        pv_input_interval_min = int(_samp_fb(pv_input_interval_min, "pv_input_interval_min"))
        pv_input_len = int(_samp_fb(pv_input_len, "pv_input_len"))
        pv_output_interval_min = int(_samp_fb(pv_output_interval_min, "pv_output_interval_min"))
        pv_output_len = int(_samp_fb(pv_output_len, "pv_output_len"))
        pv_train_time_fraction = float(_samp_fb(pv_train_time_fraction, "pv_train_time_fraction"))
        test_anchor_stride_min = int(_samp_fb(test_anchor_stride_min, "test_anchor_stride_min"))
        val_anchor_stride_min = int(_samp_fb(val_anchor_stride_min, "val_anchor_stride_min"))
        test_collect_time_match_tolerance_min = int(
            _samp_fb(test_collect_time_match_tolerance_min, "test_collect_time_match_tolerance_min")
        )
        skyimg_window_size = int(_samp_fb(skyimg_window_size, "skyimg_window_size"))
        skyimg_time_resolution_min = int(_samp_fb(skyimg_time_resolution_min, "skyimg_time_resolution_min"))
        skyimg_spatial_size = int(_samp_fb(skyimg_spatial_size, "skyimg_spatial_size"))
        satimg_window_size = int(_samp_fb(satimg_window_size, "satimg_window_size"))
        satimg_time_resolution_min = int(_samp_fb(satimg_time_resolution_min, "satimg_time_resolution_min"))
        if satimg_npy_shape_hwc is None:
            satimg_npy_shape_hwc_raw = _samp_fb(None, "satimg_npy_shape_hwc", required=False) or [100, 100, 3]
            satimg_npy_shape_hwc = tuple(int(x) for x in satimg_npy_shape_hwc_raw)

        if skyimg_window_size < 1:
            raise ValueError("skyimg_window_size must be >= 1")
        self.skyimg_window_size = int(skyimg_window_size)
        if satimg_window_size < 1:
            raise ValueError("satimg_window_size must be >= 1")
        self.satimg_window_size = int(satimg_window_size)

        if csv_interval_min <= 0 or pv_input_interval_min % csv_interval_min:
            raise ValueError(
                "pv_input_interval_min must be a positive multiple of csv_interval_min"
            )
        if pv_output_interval_min % csv_interval_min:
            raise ValueError(
                "pv_output_interval_min must be a positive multiple of csv_interval_min"
            )
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

        # Site config: read from per-instance YAML. SKIPP'd doesn't use info.yaml.
        with self._config_path.open() as f:
            conf = yaml.safe_load(f) or {}
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
        self._tilt_deg = float(site_cfg.get("tilt_deg", 0.0))
        self._surface_azimuth_deg = float(site_cfg.get("surface_azimuth_deg", 180.0))
        # rated_capacity_kw is currently informational on the dataset side; the
        # actual normalization / p_mean uses the module-level constant so trainers
        # and the dataset agree on a single source of truth. If you change one,
        # change the other (the YAML field is here for documentation / future use).
        self._rated_capacity_kw = float(
            site_cfg.get("rated_capacity_kw", _DEFAULT_SKIPPD_P_MEAN)
        )
        self.p_mean = _DEFAULT_SKIPPD_P_MEAN

        paths_cfg = conf.get("paths") or {}
        raw_sf = paths_cfg.get("sky_format", sky_format)
        sky_fmt = str(raw_sf).strip().lower()
        if sky_fmt not in ("zarr",):
            raise ValueError(
                f"paths.sky_format must be 'zarr' for SKIPP'd (got {raw_sf!r}) in {self._config_path}"
            )
        self._sky_format = sky_fmt

        # API parity with PVDataset / FolsomIrradianceDataset.
        self.devDn_list = [0]

        # PV CSV: glob ``pv_dir`` for *.csv (PVDataset convention); SKIPP'd expects exactly one.
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
        _skippd_progress(
            f"dataset split={split!r}: preparing {self._csv_path.name} ..."
        )

        # Sky Zarr: open once per process and cache the xr.Dataset handle.
        self._sky_gap_threshold = pd.Timedelta(minutes=5)
        self._sky_anchor_max_lag = pd.Timedelta(minutes=5)
        if xr is None:
            raise ImportError(
                "SkippdPvDataset requires ``xarray`` (and a Zarr backend such as ``zarr``)."
            )
        zp = self._skyimg_dir
        if not zp.exists():
            raise FileNotFoundError(f"sky Zarr path not found: {zp}")
        zkey = zp.resolve().as_posix()
        if zkey not in _SKIPPD_ZARR_DS_CACHE:
            _SKIPPD_ZARR_DS_CACHE[zkey] = xr.open_zarr(zp)
        self._skyimg_ds = _SKIPPD_ZARR_DS_CACHE[zkey]
        self._validate_sky_zarr_schema(self._skyimg_ds)
        try:
            nt = _folsom_sky_zarr_len_time_utc(self._skyimg_ds)
        except Exception:
            nt = 0
        _skippd_progress(f"sky Zarr: {zp}  (time steps ~ {nt:,})")

        # PV CSV: one in-memory table (Luoyang ``_csv_cache`` style).
        self._df, self._time_ns, self._pv_kw = _load_skippd_pv_csv(self._csv_path)
        self._time_col = _SKIPPD_TIME_COL
        self._pv_col = _SKIPPD_PV_COL
        self._n = int(len(self._df))
        if self._n < 1:
            raise RuntimeError(f"{self._csv_path.name}: expected at least one data row")

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
        self._x_tail_1d = (
            -(lx - 1) * sx + np.arange(lx, dtype=np.intp) * sx
        ).astype(np.intp, copy=False)
        self._y_off_1d = (sy + np.arange(ly, dtype=np.intp) * sy).astype(
            np.intp, copy=False
        )

        # Fixed 60% / 10% / 30% train/val/test split (same as Folsom / PVDataset).
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
        self._val_r_indices = val_positions[:: self._val_anchor_stride_rows].astype(
            np.intp, copy=False
        )
        self._num_val_windows = int(self._val_r_indices.size)
        if self.split == "val" and self._num_val_windows == 0:
            raise RuntimeError(
                "split=val: no val anchors after stride subsampling "
                "(reduce val_anchor_stride_min or widen the val segment)"
            )

        test_positions = np.nonzero(self._test_anchor_mask)[0]
        self._test_r_indices = test_positions[:: self._test_anchor_stride_rows].astype(
            np.intp, copy=False
        )
        self._num_test_windows = int(self._test_r_indices.size)
        if self.split == "test" and self._num_test_windows == 0:
            raise RuntimeError(
                "split=test: no test anchors after time split and stride "
                "(reduce test_anchor_stride_min or widen the test segment)"
            )

        # Random-anchor epoch length (Builders may override on the train instance).
        self._train_epoch_len = _DEFAULT_SKIPPD_TRAIN_EPOCH_LEN

        # Clear-sky p_cs on full CSV time index (cached). Built once per (lat, lon,
        # csv_path, tilt, azimuth) tuple so train/val/test share the same array.
        csv_time_index = pd.DatetimeIndex(self._df[self._time_col])
        self._p_cs_full = _compute_skippd_p_cs_on_csv_index(
            self.latitude,
            self.longitude,
            self._csv_path,
            csv_time_index,
            tilt_deg=self._tilt_deg,
            surface_azimuth_deg=self._surface_azimuth_deg,
        )
        if self._p_cs_full.shape[0] != self._n:
            raise RuntimeError(
                f"p_cs length {self._p_cs_full.shape[0]} != CSV rows {self._n}"
            )
        # Pre-derive the daytime mask (per CSV row) so anchor filtering is one lookup.
        self._kt_mask_full = (self._p_cs_full > 0.1).astype(np.float32, copy=False)

        # Train anchor validity filter: keep only anchors whose Y window has at least
        # one daytime step (``kt_mask == 1``). Replaces Folsom's GHI > 10 filter.
        if self.split == "train":
            self._train_anchor_valid_positions = (
                self._compute_train_anchor_valid_positions()
            )
        else:
            self._train_anchor_valid_positions = self._train_anchor_positions

    # ------------------------------------------------------------------ #
    # Train anchor filter                                                #
    # ------------------------------------------------------------------ #
    def _compute_train_anchor_valid_positions(self) -> np.ndarray:
        """
        Return the subset of ``_train_anchor_positions`` whose Y window has any
        ``kt_mask == 1`` step. Mirrors Folsom's GHI-daytime filter so random
        anchors don't land on all-night windows.
        """
        _skippd_progress(
            "train anchor filter: scanning kt_mask (p_cs > 0.1) for daytime Y windows ..."
        )
        train_anchor_rows = self._anchors[self._train_anchor_positions]
        if train_anchor_rows.size == 0:
            return self._train_anchor_positions
        y_rows = train_anchor_rows[:, None] + self._y_off_1d[None, :]
        y_kt_mask = self._kt_mask_full[y_rows]
        has_daytime = (y_kt_mask > 0.5).any(axis=1)
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
    # Sky-Zarr helpers - Copied from dataloader/folsom.py (instance methods   #
    # can't be cleanly imported; the module-level _folsom_* helpers are      #
    # imported above).                                                       #
    # ------------------------------------------------------------------ #
    def _black_sky_tensor(self) -> torch.Tensor:
        # Copied from dataloader/folsom.py:808-811
        s = self._skyimg_spatial_size
        return torch.zeros((3, s, s), dtype=torch.float32)

    @staticmethod
    def _sky_filename_ts(ts_raw) -> pd.Timestamp:
        # Copied from dataloader/folsom.py:813-819
        ts = pd.Timestamp(ts_raw)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return ts.replace(second=0, microsecond=0, nanosecond=0)

    def _validate_sky_zarr_schema(self, ds: Any) -> None:
        # Copied from dataloader/folsom.py:821-833
        if "images" not in ds.data_vars:
            raise KeyError(
                "SKIPP'd sky Zarr must define data variable ``images`` "
                "(see config/datasets/conf_skippd.yaml)."
            )
        _folsom_sky_zarr_time_dim_and_values(ds, ds["images"])

    def _nominal_sky_frame_times(self, t_end_wall: Any) -> list[pd.Timestamp]:
        # Copied from dataloader/folsom.py:835-840
        t_end = self._sky_filename_ts(t_end_wall)
        w = self.skyimg_window_size
        dt = self._skyimg_dt_min
        return [t_end - timedelta(minutes=(w - 1 - i) * dt) for i in range(w)]

    def _resize_sky_chw(self, chw: torch.Tensor) -> torch.Tensor:
        # Copied from dataloader/folsom.py:842-849
        s = self._skyimg_spatial_size
        if chw.shape[-2:] == (s, s):
            return chw
        x = chw.unsqueeze(0)
        y = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
        return y.squeeze(0)

    def _tensor_from_zarr_image_tile(self, tile: np.ndarray) -> torch.Tensor:
        # Copied from dataloader/folsom.py:851-867
        t = torch.from_numpy(np.asarray(tile, dtype=np.float32))
        if t.ndim != 3:
            raise ValueError(
                f"sky Zarr ``images`` tile must be 3D, got shape {tuple(tile.shape)}"
            )
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
        # Copied from dataloader/folsom.py:869-917
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
    # Sample building                                                     #
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        if self.split == "train":
            return self._train_epoch_len
        if self.split == "val":
            return self._num_val_windows
        return self._num_test_windows

    def _build_tensors(self, anchor: int) -> dict[str, Any]:
        x_idx = anchor + self._x_tail_1d
        y_idx = anchor + self._y_off_1d
        sub_x = self._df.iloc[x_idx]
        sub_y = self._df.iloc[y_idx]

        pow_x = sub_x[self._pv_col].to_numpy(dtype=np.float32, copy=False)
        pow_y = sub_y[self._pv_col].to_numpy(dtype=np.float32, copy=False)
        p_cs_x = self._p_cs_full[x_idx]
        p_cs_y = self._p_cs_full[y_idx]
        kt_mask_x = self._kt_mask_full[x_idx]
        kt_mask_y = self._kt_mask_full[y_idx]

        # kt = pv / (p_cs * p_mean + 1e-6) * kt_mask. pv is raw kW (NOT normalized).
        denom_x = p_cs_x * self.p_mean + 1e-6
        kt_x = np.nan_to_num(
            (pow_x / denom_x) * kt_mask_x, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32, copy=False)

        # Normalized PV (divide by capacity for input + target). Folsom-equivalent of /1100.
        pv_norm_x = (pow_x / _SKIPPD_PV_KW_SCALE).astype(np.float32, copy=False)
        pv_norm_y = (pow_y / _SKIPPD_PV_KW_SCALE).astype(np.float32, copy=False)
        pv_norm_x = np.nan_to_num(pv_norm_x, nan=0.0, posinf=0.0, neginf=0.0)
        pv_norm_y = np.nan_to_num(pv_norm_y, nan=0.0, posinf=0.0, neginf=0.0)

        # Target mask: isfinite(pv) AND kt_mask>0 (daytime). pv is finite (no NaN in CSV),
        # so this reduces to the daytime mask but isfinite() guards against future drift.
        finite_y = np.isfinite(pow_y)
        target_mask_np = (finite_y & (kt_mask_y > 0.5)).astype(np.float32)

        # Tensor shapes match dataloader.luoyang_zarr / collate_batched contract:
        #   pv, kt, kt_mask, p_cs: [1, T_in]; target_pv, target_mask, target_p_cs: [T_out].
        pv_tensor = torch.from_numpy(pv_norm_x).unsqueeze(0)
        kt_tensor = torch.from_numpy(kt_x).unsqueeze(0)
        kt_mask_tensor = torch.from_numpy(kt_mask_x).unsqueeze(0)
        p_cs_tensor = torch.from_numpy(p_cs_x.astype(np.float32, copy=False)).unsqueeze(0)
        p_mean_tensor = torch.tensor(float(self.p_mean), dtype=torch.float32)
        # pv_mask: spec says "all-ones for now". Shape mirrors Folsom's input_mask: [1, T_in].
        pv_mask_tensor = torch.ones((1, self.pv_input_len), dtype=torch.float32)

        target_pv_tensor = torch.from_numpy(pv_norm_y)
        target_mask_tensor = torch.from_numpy(target_mask_np)
        target_p_cs_tensor = torch.from_numpy(p_cs_y.astype(np.float32, copy=False))

        # Time features.
        x_times = sub_x[self._time_col]
        if bool(x_times.isna().any()):
            raise ValueError(f"NaT in {self._time_col!r} for input window")
        timestamps = [pd.Timestamp(v) for v in x_times.tolist()]
        time0 = timestamps[-1]
        forecast_timestamps = [
            time0 + pd.Timedelta(minutes=self.pv_output_interval_min * (i + 1))
            for i in range(self.pv_output_len)
        ]

        pv_solar = compute_solar_features(timestamps, self.latitude, self.longitude)
        pv_tf = solar_features_encoder(pv_solar)
        pv_dtf = delta_time_encoder(timestamps, time0)
        pv_timefeats = torch.cat([pv_tf, pv_dtf.unsqueeze(1)], dim=1)

        forecast_solar = compute_solar_features(
            forecast_timestamps, self.latitude, self.longitude
        )
        f_tf = solar_features_encoder(forecast_solar)
        f_dtf = delta_time_encoder(forecast_timestamps, time0)
        forecast_timefeats = torch.cat([f_tf, f_dtf.unsqueeze(1)], dim=1)

        # Sky stack: nominal UTC grid ending at anchor (oldest -> newest).
        t_x_end = x_times.iloc[-1]
        nominal = self._nominal_sky_frame_times(t_x_end)
        skimg_tensor = self._stack_sky_from_zarr(t_x_end)
        skimg_solar = compute_solar_features(nominal, self.latitude, self.longitude)
        skimg_tf = solar_features_encoder(skimg_solar)
        skimg_dtf = delta_time_encoder(nominal, time0)
        skimg_timefeats = torch.cat([skimg_tf, skimg_dtf.unsqueeze(1)], dim=1)
        skimg_timestamps = [t.strftime("%Y%m%d%H%M%S") for t in nominal]

        # NWP: SKIPP'd has none. Emit zeros of the shape the model indexes ([T_out, 3]);
        # the SKIPP'd trainer rewrites slots [:, :, 0] / [:, :, 2] before forward so the
        # model's (x/1000 - 0.5)*2 and (x - 288.15)/10 normalizations see sensible values.
        nwp_tensor = torch.zeros((self.pv_output_len, 3), dtype=torch.float32)

        input_timestamps_utc = [str(pd.Timestamp(t)) for t in timestamps]
        forecast_timestamps_utc = [str(pd.Timestamp(t)) for t in forecast_timestamps]
        dev_idx = torch.tensor(0, dtype=torch.long)

        return {
            "dev_idx": dev_idx,
            "pv": pv_tensor,
            "pv_mask": pv_mask_tensor,
            "pv_timefeats": pv_timefeats,
            "forecast_timefeats": forecast_timefeats,
            "kt": kt_tensor,
            "kt_mask": kt_mask_tensor,
            "p_cs": p_cs_tensor,
            "p_mean": p_mean_tensor,
            "target_p_cs": target_p_cs_tensor,
            "target_pv": target_pv_tensor,
            "target_mask": target_mask_tensor,
            "sat_tensor": None,
            "sat_timefeats": None,
            "skimg_tensor": skimg_tensor,
            "skimg_timefeats": skimg_timefeats,
            "nwp_tensor": nwp_tensor,
            "input_timestamps_utc": input_timestamps_utc,
            "forecast_timestamps_utc": forecast_timestamps_utc,
            "skimg_timestamps": skimg_timestamps,
        }

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

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.split == "train":
            r = int(np.random.choice(self._train_anchor_valid_positions))
        elif self.split == "val":
            r = int(self._val_r_indices[idx])
        else:
            r = int(self._test_r_indices[idx])
        anchor = int(self._anchors[r])
        return self._build_tensors(anchor)


__all__ = [
    "SkippdPvDataset",
    "_DEFAULT_SKIPPD_P_MEAN",
    "_DEFAULT_SKIPPD_TRAIN_EPOCH_LEN",
    "_SKIPPD_PV_KW_SCALE",
]


# ---------------------------------------------------------------------------- #
# Smoke entry: instantiate the dataset and print one sample's shapes.          #
# ---------------------------------------------------------------------------- #
def _resolve_skippd_dataset_paths(conf: dict, conf_path: Path) -> tuple[Path, Path, Path]:
    """Return ``(pv_dir, skyimg_dir, satimg_dir)`` from a SKIPP'd dataset YAML."""
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

    sat_v = paths_cfg.get("sat_path", "sat")
    sat_dir = (data_dir / Path(str(sat_v))).resolve()
    return _req("pv_path"), _req("sky_image_path"), sat_dir


def _build_skippd_kwargs(conf: dict, cfg_path: Path, split: str) -> dict:
    sampling_cfg = conf.get("sampling") or {}
    if not sampling_cfg:
        raise KeyError(f"dataset config {cfg_path} is missing a non-empty 'sampling:' section")
    pv_dir, skyimg_dir, satimg_dir = _resolve_skippd_dataset_paths(conf, cfg_path)
    shwc = sampling_cfg.get("satimg_npy_shape_hwc", [100, 100, 3])
    if not isinstance(shwc, (list, tuple)) or len(shwc) != 3:
        raise ValueError(f"sampling.satimg_npy_shape_hwc must be [H, W, C] (in {cfg_path})")

    def _req(key: str):
        if key not in sampling_cfg:
            raise KeyError(f"dataset config sampling.{key} is required (in {cfg_path})")
        return sampling_cfg[key]

    return dict(
        config_path=str(cfg_path),
        pv_dir=str(pv_dir),
        skyimg_dir=str(skyimg_dir),
        satimg_dir=str(satimg_dir),
        split=split,
        csv_interval_min=int(_req("csv_interval_min")),
        pv_input_interval_min=int(_req("pv_input_interval_min")),
        pv_input_len=int(_req("pv_input_len")),
        pv_output_interval_min=int(_req("pv_output_interval_min")),
        pv_output_len=int(_req("pv_output_len")),
        pv_train_time_fraction=float(_req("pv_train_time_fraction")),
        test_anchor_stride_min=int(_req("test_anchor_stride_min")),
        val_anchor_stride_min=int(_req("val_anchor_stride_min")),
        test_collect_time_match_tolerance_min=int(_req("test_collect_time_match_tolerance_min")),
        skyimg_window_size=int(_req("skyimg_window_size")),
        skyimg_time_resolution_min=int(_req("skyimg_time_resolution_min")),
        skyimg_spatial_size=int(_req("skyimg_spatial_size")),
        satimg_window_size=int(_req("satimg_window_size")),
        satimg_time_resolution_min=int(_req("satimg_time_resolution_min")),
        satimg_npy_shape_hwc=tuple(int(x) for x in shwc),
    )


def run_smoke_cli(argv: list[str] | None = None) -> int:
    """
    Smoke: instantiate ``SkippdPvDataset`` from the YAML, pull one sample, print shapes.

    Mirrors Folsom's smoke but kept minimal (no per-anchor irradiance / NWP report;
    SKIPP'd has neither).
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
        kwargs = _build_skippd_kwargs(conf, args.conf, args.split)
        ds = SkippdPvDataset(**kwargs)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Failed to load SKIPP'd data from {args.conf.resolve()}:\n  {e}", file=sys.stderr)
        return 1

    print(f"[conf] {args.conf.resolve()}")
    print(f"  csv={ds._csv_path}  split={args.split}  len(ds)={len(ds)}")
    s = ds[max(0, min(int(args.index), len(ds) - 1))]
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
    print("smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_smoke_cli())
