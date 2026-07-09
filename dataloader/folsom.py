"""
Folsom dataset: one CSV with time + GHI, DNI, DHI (and optional header aliases).

The irradiance CSV is loaded once in :class:`FolsomIrradianceDataset.__init__` (``self._df``,
Luoyang-style in-memory cache; ~80–150 MiB for the typical 1.5M-row / 4-column file).
Each sample uses ``iloc`` on that table (no per-sample ``read_csv`` + ``skiprows``).

:class:`FolsomIrradianceDataset` mirrors :class:`dataloader.luoyang_mem.PVDataset`'s constructor.

Horizon math matches Luoyang anchor conventions (anchor = last input row index).

Training usage (same two-step pattern as ``dataloader.luoyang``):

1. **Sample** — :meth:`FolsomIrradianceDataset.__getitem__` → :meth:`FolsomIrradianceDataset._build_tensors`
   returns one ``dict`` per index.
2. **Batch** — :func:`collate_folsom_irradiance` stacks a ``list`` of those dicts; every key in
   :data:`FOLSOM_BATCH_TENSOR_KEYS` gains a leading batch dimension ``B``.

**Batched tensor keys** (after ``collate_folsom_irradiance``; shapes use ``T_in`` = ``pv_input_len``,
``T_out`` = ``pv_output_len``, ``T_sky`` = ``skyimg_window_size``, ``C_nwp`` = NWP feature count + 1 mask channel):

- ``ghi``, ``dni``, ``dhi``: ``[B, T_in]``
- ``input_mask``: ``[B, 1, T_in]`` (valid **GHI** input timesteps; leading ``1`` matches Luoyang-style mask layout)
- ``irr_timefeats``: ``[B, T_in, 9]`` (solar + delta-time encoding on input window)
- ``forecast_timefeats``: ``[B, T_out, 9]`` (same on forecast timesteps)
- ``target_ghi``, ``target_dni``, ``target_dhi``: ``[B, T_out]``
- ``target_mask``: ``[B, T_out]`` (valid **GHI** forecast timesteps; DNI/DHI validity does not affect this mask)
- ``skimg_tensor``: ``[B, T_sky, C_sky, H, W]`` where ``C_sky`` = ``sky_in_channels``
  (default 3 = RGB only; optionally +ray_map +sun_mask +sky_mask) and ``H=W=skyimg_spatial_size``
- ``skimg_timefeats``: ``[B, T_sky, feat_dim]``
- ``nwp_tensor``: ``[B, T_out, C_nwp]`` (zeros + invalid mask if NWP file missing)

**Not stacked by collate** (debug / metadata; lists length ``B`` of per-sample lists):

- ``skimg_timestamps``, ``input_timestamps_utc``, ``forecast_timestamps_utc``

Optional keys ``skimg_tensor``, ``skimg_timefeats``, ``nwp_tensor``, ``sat_tensor``, ``sat_timefeats``
may be stacked as ``None`` if the dataset omits them — same guard pattern as
:func:`dataloader.luoyang_zarr.collate_batched`. The satellite branch is gated on
``sampling.use_satellite`` in the dataset YAML (overridable via the trainer's
``--use-satellite`` / ``--no-use-satellite`` CLI flags): when False, ``sat_tensor`` and
``sat_timefeats`` are returned as ``None`` (see ``_sat_enabled``); when True, the loader
resolves per-frame ``.npy`` shards under ``<data_dir>/<paths.sat_path>/YYYY/MM/`` and
emits real tensors. The model accepts both: ``models/models.py`` zero-paths the sat arm
when ``sat_tensor is None`` (or its max is 0).

Sky imagery: auto-detected from the resolved ``paths.sky_image_path`` folder — ``jpg`` when it holds
``YYYYMMDDHHMMSS.jpg`` files, ``zarr`` when it is a Zarr store (``.zarr`` suffix or Zarr metadata /
``images`` / ``time_utc`` subgroups). Set ``paths.sky_format`` explicitly to override. Naive CSV times
are read as **UTC**.
"""

import argparse
import bisect
import csv
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pvlib
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    import xarray as xr
except ImportError:  # pragma: no cover - optional until sky_format=zarr
    xr = None  # type: ignore[assignment]

# One open handle per Zarr path (train/val/test share the same store path in typical runs).
_ZARR_SKY_DS_CACHE: dict[str, Any] = {}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config_utils import get_resolved_paths
from dataloader.luoyang_mem import list_csv_files
from modules.solar_encoder import compute_solar_features, delta_time_encoder, solar_features_encoder
from SPMF_preprocessing.fisheye_calib import fisheye_sunmask
from SPMF_preprocessing.fisheye_calib.fisheye_sunmask import (
    DEFAULT_SUN_MASK_SIGMA_DEG,
    DEFAULT_SUN_MASK_SIGMA_PX,
)
from SPMF_preprocessing.fisheye_calib.fisheye_raymap import compute_ray_map
from SPMF_preprocessing.fisheye_calib.sky_disc_mask import (
    compute_sky_disc_mask,
    normalize_sky_disc_mask_mode,
)

# Sky-branch channel-selection abstraction (Folsom only, for now). The dataset YAML
# exposes 3 knobs under ``sampling:`` (``ray_map`` / ``sun_mask`` / ``sky_mask``); see
# ``sky_knobs_to_internal``, which translates them into the internal ``sky_channels``
# list. The resulting ``skimg_tensor`` concatenates each feature along the channel dim
# in canonical order (rgb, ray_map, sun_mask, sky_mask). Default (all knobs off) is
# byte-identical to the historical 3-channel behavior.
#
# Fisheye geometry conventions (training + inference share the same rules):
#   * Training RGB: raw, unflipped Folsom JPGs resized to ``skyimg_spatial_size`` (224).
#   * Calibration: (cx, cy, f, alpha0) fitted at native 1536 in horizontally flipped-u
#     space; see ``SPMF_preprocessing/fisheye_calib/`` and ``folsom_fisheye_fit.csv``.
#   * ray_map: ``fisheye_raymap.compute_ray_map`` — image-axis unit vectors; cx mirrored
#     to raw (``cx_raw = (native-1) - cx_fit``), ``(N-1)/(native-1)`` scaling; alpha0
#     does not enter.
#   * sun_mask: ``fisheye_sunmask.compute_sun_mask`` — project in flip space, mirror u to
#     raw via ``u_raw = (native-1) - u_flip``, Euclidean disc ``R = f_s * deg2rad(radius)``.
#   * sky_mask: ``sky_disc_mask.compute_sky_disc_mask`` — optional ``[T, 1, H, W]`` float32
#     0/1 keep-region channel appended when ``sky_mask`` knob != none; RGB is left unchanged.
_SKY_CHANNEL_RGB = "rgb"
_SKY_CHANNEL_RAY_MAP = "ray_map"
_SKY_CHANNEL_SUN_MASK = "sun_mask"
_SKY_CHANNEL_SKY_MASK = "sky_mask"
_SKY_CHANNEL_WIDTHS: dict[str, int] = {
    _SKY_CHANNEL_RGB: 3,
    _SKY_CHANNEL_RAY_MAP: 3,
    _SKY_CHANNEL_SUN_MASK: 1,
    _SKY_CHANNEL_SKY_MASK: 1,
}
_DEFAULT_SKY_CHANNELS: tuple[str, ...] = (_SKY_CHANNEL_RGB,)

# Default mask radius (degrees, converted to pixels as ``R = f_s * deg2rad(radius)``)
# for the ``sun_mask`` channel; overridable via ``sampling.sun_mask_radius_deg``
# in the dataset YAML. The sun's apparent radius is ~0.27°, but the bright glare
# halo on the Folsom fisheye saturates pixels out to ~18-19° from the sun center
# on overhead-noon clear-sky frames (empirically measured in
# playground/2026-06-15_sunmask-bigger/measure_halo_v2.py on 2014-05-11:
# r_max = 18.70° at 20:00Z noon, much smaller at low-sun hours).
# 30° gives ~60% safety margin so the mask reliably covers sun+halo at all sun
# positions; at 224x224 this is ~37 px Euclidean radius in image space.
# See ``_compute_sun_mask_for_frames``.
_DEFAULT_SUN_MASK_RADIUS_DEG: float = 30.0

def _normalize_sky_channels(raw: Any) -> tuple[str, ...]:
    """Validate + canonicalize a ``sky_channels`` config value to a tuple of names.

    ``None`` (or missing) → default ``("rgb",)``. Otherwise the value must be a
    non-empty sequence of unique known names from :data:`_SKY_CHANNEL_WIDTHS`.
    """
    if raw is None:
        return _DEFAULT_SKY_CHANNELS
    if isinstance(raw, str) or not hasattr(raw, "__iter__"):
        raise TypeError(
            f"sky_channels must be a list/tuple of feature names, got {type(raw).__name__}"
        )
    names = [str(x).strip() for x in raw]
    if len(names) == 0:
        raise ValueError("sky_channels must contain at least one feature name")
    seen: set[str] = set()
    for n in names:
        if n not in _SKY_CHANNEL_WIDTHS:
            raise ValueError(
                f"sky_channels: unknown feature {n!r}; valid names are "
                f"{sorted(_SKY_CHANNEL_WIDTHS)}"
            )
        if n in seen:
            raise ValueError(f"sky_channels: duplicate feature {n!r}")
        seen.add(n)
    return tuple(names)


def _sky_in_channels(channels: tuple[str, ...]) -> int:
    return int(sum(_SKY_CHANNEL_WIDTHS[c] for c in channels))


# Public sky-branch config surface (3 knobs, replacing the raw ``sky_channels`` /
# ``sky_disc_mask_mode`` / ``sun_mask_radius_deg`` / ``sky_disc_mask_radius_px`` keys):
#   * ``ray_map``  (bool)                      -> add the ray_map channel (3ch)
#   * ``sun_mask`` (none|sun_only|sun_halo)    -> add the sun_mask channel (1ch) + radius preset
#   * ``sky_mask`` (none|loose|tight|valid_disc) -> optional sky_mask channel (1ch) +
#     internal disc mode for mask computation (RGB left unmodified)
# ``sky_knobs_to_internal`` translates these into the constructor's internal
# ``sky_channels`` / ``sun_mask_radius_deg`` / ``sky_disc_mask_mode`` representation.
_SUN_MASK_MODES: tuple[str, ...] = (
    "none",
    "sun_only",
    "sun_halo",
    "gaussian_pixel",
    "gaussian_angular",
)
# Angular radius presets (degrees) for the sun_mask channel per ``sun_mask`` mode.
# ``sun_halo`` keeps the historical wide default; ``sun_only`` is a tight disc.
_SUN_MASK_RADIUS_DEG_PRESETS: dict[str, float] = {
    "sun_only": 10.0,
    "sun_halo": _DEFAULT_SUN_MASK_RADIUS_DEG,
}
_SUN_MASK_HARD_MODES: frozenset[str] = frozenset({"sun_only", "sun_halo"})
_SKY_MASK_MODES: tuple[str, ...] = ("none", "loose", "tight", "valid_disc")
_SKY_MASK_TO_DISC_MODE: dict[str, str] = {
    "none": "none",
    "loose": "manual_loose",
    "tight": "manual_tight",
    "valid_disc": "valid_disc",
}


def normalize_ray_map(raw: Any) -> bool:
    """Validate + canonicalize the ``ray_map`` knob to a bool (default ``False``)."""
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        v = raw.strip().lower()
        if v in ("true", "1", "yes", "on"):
            return True
        if v in ("false", "0", "no", "off"):
            return False
    raise ValueError(f"ray_map must be a boolean, got {raw!r}")


def normalize_sun_mask(raw: Any) -> str:
    """Validate + canonicalize the ``sun_mask`` knob (default ``'none'``)."""
    if raw is None:
        return "none"
    mode = str(raw).strip()
    if mode not in _SUN_MASK_MODES:
        raise ValueError(
            f"sun_mask must be one of {list(_SUN_MASK_MODES)}, got {raw!r}"
        )
    return mode


def normalize_sky_mask(raw: Any) -> str:
    """Validate + canonicalize the ``sky_mask`` knob (default ``'none'``)."""
    if raw is None:
        return "none"
    mode = str(raw).strip()
    if mode not in _SKY_MASK_MODES:
        raise ValueError(
            f"sky_mask must be one of {list(_SKY_MASK_MODES)}, got {raw!r}"
        )
    return mode


def resolve_sun_mask_sigmas(
    sampling_cfg: dict | None = None,
    *,
    sigma_px_override: float | None = None,
    sigma_deg_override: float | None = None,
) -> tuple[float, float]:
    """Read ``sun_mask_sigma_px`` / ``sun_mask_sigma_deg`` from sampling config."""
    cfg = sampling_cfg or {}
    sigma_px = (
        float(sigma_px_override)
        if sigma_px_override is not None
        else float(cfg.get("sun_mask_sigma_px", DEFAULT_SUN_MASK_SIGMA_PX))
    )
    sigma_deg = (
        float(sigma_deg_override)
        if sigma_deg_override is not None
        else float(cfg.get("sun_mask_sigma_deg", DEFAULT_SUN_MASK_SIGMA_DEG))
    )
    if not (sigma_px > 0.0):
        raise ValueError(f"sun_mask_sigma_px must be > 0 (got {sigma_px!r})")
    if not (sigma_deg > 0.0):
        raise ValueError(f"sun_mask_sigma_deg must be > 0 (got {sigma_deg!r})")
    return sigma_px, sigma_deg


def sky_knobs_to_internal(
    ray_map: Any,
    sun_mask: Any,
    sky_mask: Any,
) -> tuple[tuple[str, ...], float | None, str, str]:
    """Translate the 3 public sky knobs into the internal constructor representation.

    Returns ``(sky_channels, sun_mask_radius_deg, sky_disc_mask_mode, sun_mask_mode)``:
      * ``sky_channels`` = ``("rgb",)`` + ``("ray_map",)`` if ``ray_map`` +
        ``("sun_mask",)`` if ``sun_mask != 'none'`` + ``("sky_mask",)`` if
        ``sky_mask != 'none'`` (canonical order preserved).
      * ``sun_mask_radius_deg`` = preset for hard-disc modes (``None`` when
        ``sun_mask`` is ``'none'`` or a Gaussian mode — the channel is absent or
        the radius is unused).
      * ``sky_disc_mask_mode`` = internal disc mode for ``sky_mask`` channel computation
        (``'none'`` when the knob is off).
      * ``sun_mask_mode`` = canonical ``sun_mask`` knob value.
    """
    ray_map_on = normalize_ray_map(ray_map)
    sun_mask_mode = normalize_sun_mask(sun_mask)
    sky_mask_mode = normalize_sky_mask(sky_mask)
    channels: list[str] = [_SKY_CHANNEL_RGB]
    if ray_map_on:
        channels.append(_SKY_CHANNEL_RAY_MAP)
    if sun_mask_mode != "none":
        channels.append(_SKY_CHANNEL_SUN_MASK)
    if sky_mask_mode != "none":
        channels.append(_SKY_CHANNEL_SKY_MASK)
    sun_mask_radius_deg = _SUN_MASK_RADIUS_DEG_PRESETS.get(sun_mask_mode)
    sky_disc_mask_mode = _SKY_MASK_TO_DISC_MODE[sky_mask_mode]
    return tuple(channels), sun_mask_radius_deg, sky_disc_mask_mode, sun_mask_mode

# Default dataset YAML for Folsom under the new ``config/datasets/`` layout. Used only by the
# smoke CLI as a convenience default; ``FolsomIrradianceDataset`` itself takes ``config_path``
# as a required constructor argument and never falls back to a hardcoded path.
_DEFAULT_FOLSOM_DATASET_CONFIG = _PROJECT_ROOT / "config" / "datasets" / "conf_folsom.yaml"

# Keys collate stacks with batch dim B first (single source of truth for trainers).
FOLSOM_GHI_DNI_DHI_KEYS: tuple[str, ...] = (
    "ghi",
    "dni",
    "dhi",
    "input_mask",
    "irr_timefeats",
    "forecast_timefeats",
    "target_ghi",
    "target_dni",
    "target_dhi",
    "target_mask",
    "skimg_tensor",
    "skimg_timefeats",
    "nwp_tensor",
)
FOLSOM_BATCH_TENSOR_KEYS = FOLSOM_GHI_DNI_DHI_KEYS

_TIME_HEADER_CANDIDATES = frozenset(
    {"time", "timestamp", "datetime", "collecttime", "date_time", "dt", "local_time"}
)
_FOLSOM_NWP_TIME_COLS = ("reftime", "valtime")
_FOLSOM_NWP_FEATURE_COLS = (
    "dwsw",
    "cloud_cover",
    "precipitation",
    "pressure",
    "wind-u",
    "wind-v",
    "temperature",
    "rel_humidity",
)
_SKY_INDEX_CACHE: dict[str, tuple[list[pd.Timestamp], list[Path], list[int]]] = {}


_DEFAULT_FOLSOM_TRAIN_EPOCH_LEN = 100_000

# Train mode: a Y window is "valid" if any of its rows has finite GHI strictly above this
# threshold (W/m^2). Mirrors PVDataset's "any inverter_state == VALID_STATE" filter so we
# avoid sampling all-night windows where target_pv is uniformly 0.
_FOLSOM_TRAIN_GHI_DAYTIME_THRESHOLD = 10.0

# Folsom satellite (use_satellite=True): satellite history ends ``_FOLSOM_SAT_ANCHOR_OFFSET_MIN``
# minutes before the PV anchor time0 (to leave room for real-time delivery latency that
# GridSat-CONUS shards would have in a live forecast pipeline). Unused when
# ``sampling.use_satellite`` is false (sat_tensor / sat_timefeats are returned as None).
_FOLSOM_SAT_ANCHOR_OFFSET_MIN = 30

# GHI scaling factor used ONLY for ``p_cs = clearsky_ghi / _FOLSOM_GHI_SCALE`` (mirrors
# Luoyang's ``poa_global / 1000`` recipe in
# ``SPMF_preprocessing/luoyang/aggregate_by_devdn_solarfeats.py``). 1000 W/m^2 is the
# standard "1 sun" reference irradiance (STC), so ``p_cs`` is dimensionless and ~1.0 at
# a perfectly clear noon. The raw GHI signal (``pv`` / ``target_pv`` / numerator of
# ``kt``) is NOT divided by this constant; ``p_mean`` is held at 1.0 (see ``__init__``),
# so ``kt = ghi_raw / (p_cs + eps) * kt_mask`` is in W/m^2-ish units and the ViT input
# rescale is handled trainer-side (``kt / 4000`` and ``* 4000``).
_FOLSOM_GHI_SCALE = 1000.0
# Daytime guard: matches Luoyang's preprocessing ``kt_mask = (p_cs > 0.1)`` rule. Anchors
# where ``p_cs <= 0.1`` (nighttime / very low sun) get ``kt = 0`` via the mask product.
_FOLSOM_KT_DAYTIME_THRESHOLD = 0.1
_FOLSOM_KT_EPS = 1e-6
# Daytime gate for ``target_mask`` (folsom-kt overlay). Forecast-horizon rows are kept iff
# clear-sky GHI is at least this value in normalized ``p_cs`` units. The value below is
# 20.0 W/m^2 / _FOLSOM_GHI_SCALE = 20.0 / 1000.0 = 0.02, the exact p_cs equivalent of
# folsom-kt's ``GHI_CS_NIGHT_THRESHOLD = 20.0`` W/m^2. Looser than ``_FOLSOM_KT_DAYTIME_THRESHOLD``
# (0.1) on purpose: that constant gates the kt regression target, this one gates the loss
# mask, and kt-side semantics are the source of truth for the latter.
_FOLSOM_TARGET_MASK_NIGHT_THRESHOLD_P_CS = 0.02
# ViT input/output scaling for ``kt`` (W/m^2-ish; see ``forward_vit`` docstring in the
# trainer). Single source of truth so trainer / eval / inference stay in lockstep.
_FOLSOM_KT_INPUT_SCALE = 4000.0
# Huber loss delta in W/m^2 for PV-target residuals. Shared by trainer + eval Huber sites.
_FOLSOM_HUBER_DELTA = 700.0


def _compute_folsom_p_cs(
    lat: float,
    lon: float,
    utc_index: pd.DatetimeIndex,
) -> np.ndarray:
    """
    Normalized clear-sky GHI (``clearsky_ghi / _FOLSOM_GHI_SCALE``, clipped to ``[0, 1.2]``).

    Mirrors :func:`SPMF_preprocessing/luoyang/aggregate_by_devdn_solarfeats.compute_clearsky_power`
    but uses ``clearsky["ghi"]`` directly (Folsom's active signal is GHI, not PV power), so no
    POA transposition / tilt assumption is needed.
    """
    idx = utc_index
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")
    loc = pvlib.location.Location(float(lat), float(lon))
    cs = loc.get_clearsky(idx, model="ineichen")
    ghi_cs = np.asarray(cs["ghi"].values, dtype=np.float64)
    p_cs = np.clip(ghi_cs / _FOLSOM_GHI_SCALE, 0.0, 1.2)
    return p_cs.astype(np.float32, copy=False)


def _folsom_parse_zarr_utc_naive(raw: Any) -> pd.DatetimeIndex:
    """Parse Zarr time values to naive UTC :class:`pandas.DatetimeIndex` (Luoyang / Folsom convention)."""
    return pd.DatetimeIndex(
        pd.to_datetime(np.asarray(raw), utc=True)
    ).tz_convert("UTC").tz_localize(None)


def _folsom_sky_zarr_time_dim_and_values(ds: Any, img: Any) -> tuple[str, np.ndarray]:
    """
    Return ``(time_dim_name_on_images, time_values_raw)`` so ``len(values) == images.sizes[dim]``.

    Supports:

    * ``images`` with a ``time_utc`` dimension and coordinate (typical ``Dataset.to_zarr``), or
    * ``images`` with some time-like dimension whose length matches the 1D ``time_utc`` array
      (stores with separate Zarr groups ``time_utc/`` and ``images/``, e.g. ``sky_xr_120.zarr``).
    """
    if "images" not in ds.data_vars:
        raise KeyError("Folsom sky Zarr must define data variable ``images``.")
    if "time_utc" in img.dims:
        if "time_utc" in img.coords:
            return "time_utc", np.asarray(img.coords["time_utc"].values)
        if "time_utc" in ds.variables:
            t1 = np.asarray(ds["time_utc"].values)
            if int(t1.shape[0]) != int(img.sizes["time_utc"]):
                raise ValueError(
                    f"``time_utc`` length {int(t1.shape[0])} != images time axis "
                    f"{int(img.sizes['time_utc'])}"
                )
            return "time_utc", t1
        raise KeyError(
            "``images`` has dimension ``time_utc`` but no coordinate or ``time_utc`` variable was found."
        )
    if "time_utc" not in ds.variables:
        raise KeyError(
            "Folsom sky Zarr needs ``time_utc`` on the ``images`` dimension or as a 1D ``time_utc`` "
            "variable aligned with one dimension of ``images`` (see config/datasets/conf_folsom.yaml)."
        )
    tda = ds["time_utc"]
    if int(getattr(tda, "ndim", 0) or 0) != 1:
        raise ValueError(f"``time_utc`` must be 1D, got shape {getattr(tda, 'shape', None)}")
    n = int(tda.shape[0])
    tvals = np.asarray(tda.values)
    for dim_name, sz in img.sizes.items():
        if int(sz) == n:
            return str(dim_name), tvals
    raise KeyError(
        f"No ``images`` dimension has length {n} (len(time_utc)); image dims/sizes={dict(img.sizes)}"
    )


def _folsom_sky_zarr_len_time_utc(ds: Any) -> int:
    """Number of sky timesteps (for progress logging)."""
    img = ds["images"]
    _, raw = _folsom_sky_zarr_time_dim_and_values(ds, img)
    return int(np.asarray(raw).shape[0])


def _folsom_sky_zarr_count_in_time_range(ds: Any, t0: Any, t1: Any) -> int:
    """Count Zarr timesteps with naive UTC in ``[t0, t1]`` inclusive."""
    img = ds["images"]
    _, raw = _folsom_sky_zarr_time_dim_and_values(ds, img)
    zt = _folsom_parse_zarr_utc_naive(raw)
    a0, a1 = pd.Timestamp(t0), pd.Timestamp(t1)
    return int(np.count_nonzero((zt >= a0) & (zt <= a1)))


def _folsom_progress(msg: str) -> None:
    """Progress to stderr so training stdout stays clean; set FOLSOM_QUIET=1 to disable."""
    if os.environ.get("FOLSOM_QUIET", "").strip().lower() in ("1", "true", "yes"):
        return
    print(f"[Folsom] {msg}", file=sys.stderr, flush=True)


def _folsom_path_has_zarr_markers(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / ".zgroup").is_file() or (path / ".zarray").is_file():
        return True
    if (path / "zarr.json").is_file():
        return True
    return False


def _folsom_sky_path_looks_like_zarr(sky_path: Path) -> bool:
    sky_path = sky_path.resolve()
    if sky_path.name.lower().endswith(".zarr"):
        return True
    if _folsom_path_has_zarr_markers(sky_path):
        return True
    for sub in ("images", "time_utc"):
        if _folsom_path_has_zarr_markers(sky_path / sub):
            return True
    return False


def _folsom_sky_path_has_jpg(sky_path: Path) -> bool:
    if not sky_path.is_dir():
        return False
    for p in sky_path.iterdir():
        if p.is_file() and p.suffix.lower() == ".jpg":
            return True
    return False


def _detect_folsom_sky_format(
    sky_path: Path,
    *,
    config_path: Path | None = None,
) -> str:
    """
    Infer ``jpg`` vs ``zarr`` from the resolved sky folder when ``paths.sky_format`` is omitted.
    """
    sky_path = sky_path.resolve()
    cfg_hint = f" (config: {config_path})" if config_path else ""

    is_zarr = _folsom_sky_path_looks_like_zarr(sky_path)
    has_jpg = _folsom_sky_path_has_jpg(sky_path)

    if is_zarr and has_jpg:
        raise ValueError(
            f"Could not auto-detect sky format under {sky_path}{cfg_hint}: "
            "directory looks like both Zarr and JPEG. "
            "Set paths.sky_format explicitly to 'jpg' or 'zarr'."
        )
    if is_zarr:
        return "zarr"
    if has_jpg:
        return "jpg"

    if not sky_path.exists():
        raise FileNotFoundError(
            f"Sky image path not found: {sky_path}{cfg_hint}. "
            "Create the directory with JPEG or Zarr data, or set paths.sky_format explicitly."
        )
    if not sky_path.is_dir():
        raise ValueError(
            f"Sky image path is not a directory: {sky_path}{cfg_hint}. "
            "Expected a folder of YYYYMMDDHHMMSS.jpg files or a Zarr store; "
            "set paths.sky_format explicitly if using a non-standard layout."
        )
    raise ValueError(
        f"Could not auto-detect sky format under {sky_path}{cfg_hint}: "
        "no Zarr metadata (.zgroup/.zarray/zarr.json or images/time_utc subgroups) "
        "and no *.jpg files found. Populate the folder or set paths.sky_format to 'jpg' or 'zarr'."
    )


def _count_newlines(path: Path) -> int:
    """Count lines in file (including header) using buffered binary read."""
    n = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            n += chunk.count(b"\n")
    return n


def _read_header_line(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return f.readline().rstrip("\n\r")


def _load_folsom_irradiance_csv(path: Path) -> tuple[pd.DataFrame, str, list[str]]:
    """
    Load the single Folsom irradiance CSV into memory with parsed time + float irradiance columns.

    Returns ``(df, time_col, ghi_dni_dhi_cols)`` where ``df`` has columns
    ``[time_col, ghi, dni, dhi]`` (names auto-detected from the header).
    """
    p = path.resolve()
    _folsom_progress(f"loading irradiance CSV {p.name} into memory ...")
    raw = pd.read_csv(p, engine="c")
    time_col, _order, _all_cols = _pick_time_and_ghi_dni_dhi_columns(list(raw.columns))
    ghi_dni_dhi_cols = _order[1:]
    df = raw[[time_col, *ghi_dni_dhi_cols]].copy()
    df[time_col] = pd.to_datetime(df[time_col], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    if bool(df[time_col].isna().any()):
        raise ValueError(f"{p.name}: NaT in {time_col!r} after parsing")
    for c in ghi_dni_dhi_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    _folsom_progress(
        f"irradiance CSV ready: {len(df):,} rows in RAM "
        f"({time_col!r}, {', '.join(ghi_dni_dhi_cols)})"
    )
    return df, time_col, ghi_dni_dhi_cols


def _resolve_folsom_csv_path(conf: dict, project_root: Path | None = None) -> Path:
    root = project_root if project_root is not None else _PROJECT_ROOT
    paths = conf.get("paths") or {}
    if paths.get("data_dir") is None or not str(paths.get("data_dir", "")).strip():
        raise KeyError("conf paths.data_dir is required for Folsom")
    data_dir = Path(paths["data_dir"])
    if not data_dir.is_absolute():
        data_dir = (root / data_dir).resolve()
    else:
        data_dir = data_dir.resolve()

    rel = paths.get("folsom_irradiance_csv")
    if rel is None or not str(rel).strip():
        raise KeyError("conf paths.folsom_irradiance_csv is required for Folsom")
    rel_s = str(rel).strip()
    if rel_s in (".", ".."):
        raise ValueError(
            f"paths.folsom_irradiance_csv must name a CSV file (e.g. Folsom_irradiance.csv), not {rel_s!r}. "
            "Using '.' makes the path resolve to data_dir only (a folder), which then fails."
        )

    rel_p = Path(rel_s)
    if rel_p.is_absolute():
        p = rel_p.resolve()
    else:
        p = (data_dir / rel_p).resolve()

    if p.is_dir():
        raise FileNotFoundError(
            f"Folsom CSV path is a directory, not a file: {p}\n"
            f"  data_dir={data_dir}\n"
            f"  folsom_irradiance_csv={rel_s!r}\n"
            "If you meant a file inside data_dir, use a relative name like Folsom_irradiance.csv "
            "(not an absolute path to a folder, and not '.')."
        )
    if not p.is_file():
        raise FileNotFoundError(
            f"Folsom GHI/DNI/DHI CSV not found: {p}\n"
            f"  data_dir={data_dir}\n"
            f"  folsom_irradiance_csv={rel_s!r}\n"
            "Create or copy the file under data_dir (see paths.folsom_irradiance_csv)."
        )
    return p


def _resolve_folsom_nwp_csv_path(conf: dict, project_root: Path | None = None) -> Path:
    """Resolve the Folsom NWP merged CSV from config.

    When ``paths.folsom_nwp_merged_csv`` is set it is used verbatim (relative to
    ``data_dir`` unless absolute). When absent/empty, the ``paths.nwp_path`` folder
    is globbed: a file literally named ``nwp_merged_averaged.csv`` is preferred, else
    the sole ``*.csv`` in the folder is used. Raises ``FileNotFoundError`` when no CSV
    can be located (the caller treats this as "no NWP"); raises ``RuntimeError`` when
    the folder is ambiguous (several CSVs, none named ``nwp_merged_averaged.csv``).
    """
    root = project_root if project_root is not None else _PROJECT_ROOT
    paths = conf.get("paths") or {}
    if paths.get("data_dir") is None or not str(paths.get("data_dir", "")).strip():
        raise KeyError("conf paths.data_dir is required for Folsom")
    data_dir = Path(paths["data_dir"])
    if not data_dir.is_absolute():
        data_dir = (root / data_dir).resolve()
    else:
        data_dir = data_dir.resolve()

    rel = paths.get("folsom_nwp_merged_csv")
    if rel is not None and str(rel).strip():
        rel_p = Path(str(rel).strip())
        p = rel_p.resolve() if rel_p.is_absolute() else (data_dir / rel_p).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Folsom NWP merged CSV not found: {p}")
        return p

    nwp_rel = paths.get("nwp_path")
    if nwp_rel is None or not str(nwp_rel).strip():
        raise FileNotFoundError(
            "Folsom NWP not configured: set paths.nwp_path (folder) or paths.folsom_nwp_merged_csv (file)"
        )
    nwp_p = Path(str(nwp_rel).strip())
    nwp_dir = nwp_p.resolve() if nwp_p.is_absolute() else (data_dir / nwp_p).resolve()
    if not nwp_dir.is_dir():
        raise FileNotFoundError(f"Folsom NWP folder not found: {nwp_dir}")
    csvs = sorted(nwp_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No NWP CSV found in {nwp_dir}")
    preferred = [c for c in csvs if c.name == "nwp_merged_averaged.csv"]
    if preferred:
        return preferred[0].resolve()
    if len(csvs) == 1:
        return csvs[0].resolve()
    names = ", ".join(c.name for c in csvs)
    raise RuntimeError(
        f"Folsom NWP folder {nwp_dir} has multiple CSVs and none named 'nwp_merged_averaged.csv': {names}. "
        "Set paths.folsom_nwp_merged_csv to disambiguate."
    )


def _normalize_col(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _pick_time_and_ghi_dni_dhi_columns(header_cells: list[str]) -> tuple[str, list[str], list[str]]:
    """
    Map header to (time_col, [ghi_col, dni_col, dhi_col], all_cols_in_file_order).

    Raises if required columns cannot be identified.
    """
    raw = [h.strip() for h in header_cells]
    norm = [_normalize_col(h) for h in raw]
    lower_to_orig: dict[str, str] = {}
    for o, n in zip(raw, norm):
        lower_to_orig.setdefault(n, o)

    def pick_one(cands: set[str], label: str) -> str:
        for n in norm:
            if n in cands:
                return lower_to_orig[n]
        raise ValueError(f"Could not find {label} column in header {raw!r}")

    ghi = pick_one({"ghi", "global_horizontal_irradiance"}, "GHI")
    dni = pick_one({"dni", "direct_normal_irradiance"}, "DNI")
    dhi = pick_one({"dhi", "diffuse_horizontal_irradiance", "diffuse_irradiance"}, "DHI")

    time_col = None
    for o, n in zip(raw, norm):
        if n in _TIME_HEADER_CANDIDATES or "time" in n or "date" in n:
            time_col = o
            break
    if time_col is None:
        raise ValueError(f"Could not infer time column from header {raw!r}")

    order = [time_col, ghi, dni, dhi]
    return time_col, order, raw


def load_folsom_conf(path: Path | str) -> dict:
    """Load a Folsom dataset YAML (typically ``config/datasets/conf_folsom.yaml``).

    Caller must supply the path explicitly; this module never reads a hardcoded canonical
    config file. The expected schema mirrors ``config/datasets/conf_luoyang.yaml``:
    ``paths.{data_dir, pv_path, sky_image_path, sat_path, ...}`` plus a ``sampling:``
    section with the PVDataset-style window / stride / image-shape fields.
    """
    if path is None:
        raise TypeError("load_folsom_conf(path) is required; no canonical default")
    p = Path(path)
    with p.open() as f:
        return yaml.safe_load(f) or {}


def _folsom_to_timestamps(values) -> list[pd.Timestamp]:
    """Parse values as pandas timestamps without timezone conversion/localization."""
    return [pd.Timestamp(v) for v in values]


def _sanitize_nwp_interp(nwp_interp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Replace non-finite values in ``nwp_interp`` (shape ``[T, C]``) with 0 and emit a timestep mask.

    Returns ``(nwp_clean, nwp_mask)`` where ``nwp_mask`` has shape ``[T, 1]``, dtype float32:
    ``1.0`` if any feature in that timestep was non-finite (or out-of-range interpolation), else ``0.0``.
    """
    x = np.asarray(nwp_interp, dtype=np.float64)
    bad = ~np.isfinite(x)
    x_clean = np.where(bad, 0.0, x).astype(np.float32, copy=False)
    row_bad = bad.any(axis=1).astype(np.float32).reshape(-1, 1)
    return x_clean, row_bad


def _strict_interp_series(xp_ns: np.ndarray, fp: np.ndarray, xq_ns: np.ndarray) -> np.ndarray:
    """
    Interpolate ``fp`` over ``xp_ns`` onto ``xq_ns`` without extrapolation.

    Outside ``[xp_ns[0], xp_ns[-1]]`` values become ``NaN`` (strict mode).
    """
    if xp_ns.size == 0:
        return np.full(xq_ns.shape[0], np.nan, dtype=np.float64)
    if xp_ns.size == 1:
        out = np.full(xq_ns.shape[0], np.nan, dtype=np.float64)
        out[xq_ns == xp_ns[0]] = fp[0]
        return out
    out = np.interp(xq_ns, xp_ns, fp).astype(np.float64, copy=False)
    out[(xq_ns < xp_ns[0]) | (xq_ns > xp_ns[-1])] = np.nan
    return out


def _load_folsom_nwp_merged_csv(path: Path | str) -> pd.DataFrame:
    """
    Load and normalize Folsom NWP merged CSV.

    Required columns: ``reftime``, ``valtime`` + :data:`_FOLSOM_NWP_FEATURE_COLS`.
    Duplicate ``valtime`` rows keep the latest ``reftime``.
    """
    p = Path(path).resolve()
    _folsom_progress(f"loading NWP merged CSV {p.name} ...")
    df = pd.read_csv(p)
    missing = [c for c in (*_FOLSOM_NWP_TIME_COLS, *_FOLSOM_NWP_FEATURE_COLS) if c not in df.columns]
    if missing:
        raise KeyError(f"{p.name}: missing required NWP column(s): {missing}")

    out = pd.DataFrame(index=df.index)
    # Explicit format avoids the ``Could not infer format ... falling back to dateutil``
    # warning and the per-element slow path. CSV stores naive ``YYYY-MM-DD HH:MM:SS``.
    out["reftime"] = pd.to_datetime(df["reftime"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    out["valtime"] = pd.to_datetime(df["valtime"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    for c in _FOLSOM_NWP_FEATURE_COLS:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    out = out.dropna(subset=["reftime", "valtime"]).copy()
    out = out.sort_values(["valtime", "reftime"]).drop_duplicates(subset=["valtime"], keep="last")
    out = out.sort_values("valtime").reset_index(drop=True)
    if out.empty:
        raise ValueError(f"{p.name}: no valid NWP rows after datetime parsing and dedup")
    _folsom_progress(f"NWP ready: {len(out):,} rows in {p.name}")
    return out


class FolsomIrradianceDataset(Dataset):
    """
    Folsom irradiance dataset with the **same** ``__init__`` signature as
    :class:`dataloader.luoyang_mem.PVDataset` (``pv_dir``/``skyimg_dir``/``satimg_dir`` plus the
    Luoyang-style ``pv_*`` / ``skyimg_*`` / ``satimg_*`` keyword args). Folsom has no satellite
    data, so the ``satimg_*`` arguments are accepted for API parity but unused — ``sat_tensor``
    and ``sat_timefeats`` in returned samples are ``None``.

    ``pv_dir`` must contain exactly one irradiance CSV (time + GHI/DNI/DHI columns; column names
    auto-detected from the header). The optional NWP merged CSV is resolved from
    ``paths.folsom_nwp_merged_csv`` if set, else globbed from the ``paths.nwp_path`` folder, in
    the per-instance ``config_path``; site coordinates come
    from ``<paths.data_dir>/info.yaml`` (``site.latitude`` / ``site.longitude``), matching
    :class:`dataloader.luoyang_mem.PVDataset`.

    Splits: rows are partitioned chronologically by ``train_split`` / ``val_split`` /
    ``test_split`` (defaults ``0.66`` / ``0.18`` / ``0.16``). ``pv_train_time_fraction`` is
    kept for call-site parity but **not** used. Train
    samples a random valid anchor per ``__getitem__`` (epoch length defaults to
    ``_DEFAULT_FOLSOM_TRAIN_EPOCH_LEN``; settable via ``self._train_epoch_len``); val/test use the
    respective ``*_anchor_stride_min`` strides over their bands.

    ``skyimg_window_size`` is the count of sky frames ending at the last input timestep (anchor),
    spaced by ``skyimg_time_resolution_min`` (oldest first in ``skimg_tensor``). Sky format is
    auto-detected from ``skyimg_dir`` (Zarr store vs JPEG folder) unless ``paths.sky_format`` is set.
    All timestamps are UTC.
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
        train_split: float = 0.66,
        val_split: float = 0.18,
        test_split: float = 0.16,
        test_anchor_stride_min: int,
        val_anchor_stride_min: int,
        test_collect_time_match_tolerance_min: int,
        skyimg_window_size: int,
        skyimg_time_resolution_min: int,
        skyimg_spatial_size: int,
        satimg_window_size: int,
        satimg_time_resolution_min: int,
        satimg_npy_shape_hwc: tuple[int, int, int],
        use_satellite: bool = False,
        sky_channels: list[str] | tuple[str, ...] | None = None,
        sun_mask_mode: str | None = None,
        sun_mask_radius_deg: float | None = None,
        sun_mask_sigma_px: float | None = None,
        sun_mask_sigma_deg: float | None = None,
        sky_disc_mask_mode: str | None = None,
        sky_disc_mask_radius_px: float | None = None,
    ):
        self._config_path = Path(config_path).resolve()
        if not self._config_path.is_file():
            raise FileNotFoundError(
                f"FolsomIrradianceDataset config_path not found: {self._config_path}"
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

        self._train_split = float(train_split)
        self._val_split = float(val_split)
        self._test_split = float(test_split)
        for name, frac in (
            ("train_split", self._train_split),
            ("val_split", self._val_split),
            ("test_split", self._test_split),
        ):
            if not (0.0 < frac < 1.0):
                raise ValueError(f"{name} must be strictly between 0 and 1 (got {frac!r})")
        split_sum = self._train_split + self._val_split + self._test_split
        if abs(split_sum - 1.0) >= 1e-6:
            raise ValueError(
                f"train_split + val_split + test_split must sum to 1.0 (got {split_sum:.6f})"
            )

        if skyimg_time_resolution_min <= 0:
            raise ValueError("skyimg_time_resolution_min must be positive")
        self._skyimg_dt_min = int(skyimg_time_resolution_min)
        self._skyimg_dir = Path(skyimg_dir).resolve()
        if skyimg_spatial_size < 1:
            raise ValueError("skyimg_spatial_size must be >= 1")
        self._skyimg_spatial_size = int(skyimg_spatial_size)

        # Sky-branch channel selection. ``sky_channels`` is a YAML-driven list of
        # feature names; default ``("rgb",)`` keeps existing behavior (3-channel
        # ``skimg_tensor``). The calibrated ray map + its companion validity
        # mask are built lazily on first use and cached since
        # ``compute_ray_map`` is a pure function of ``(H, W, fit)``. ``sun_mask``
        # reuses the same fit cache plus per-frame solar geometry from
        # ``compute_solar_features``.
        self.sky_channels: tuple[str, ...] = _normalize_sky_channels(sky_channels)
        self.sky_in_channels: int = _sky_in_channels(self.sky_channels)
        self._ray_map_cache: torch.Tensor | None = None
        self._sky_valid_cache: torch.Tensor | None = None
        self._fisheye_fit: dict | None = None
        self.sun_mask_mode: str = normalize_sun_mask(sun_mask_mode)
        if self.sun_mask_mode in _SUN_MASK_HARD_MODES:
            radius = (
                _SUN_MASK_RADIUS_DEG_PRESETS[self.sun_mask_mode]
                if sun_mask_radius_deg is None
                else float(sun_mask_radius_deg)
            )
            if not (radius > 0.0):
                raise ValueError(
                    f"sun_mask_radius_deg must be > 0 (got {sun_mask_radius_deg!r})"
                )
            if radius >= 90.0:
                raise ValueError(
                    f"sun_mask_radius_deg must be < 90 (got {sun_mask_radius_deg!r}); "
                    "a half-sky disc is almost certainly a config mistake"
                )
            self.sun_mask_radius_deg: float = radius
        else:
            self.sun_mask_radius_deg = float(
                sun_mask_radius_deg or _DEFAULT_SUN_MASK_RADIUS_DEG
            )
        sigma_px, sigma_deg = resolve_sun_mask_sigmas(
            sigma_px_override=sun_mask_sigma_px,
            sigma_deg_override=sun_mask_sigma_deg,
        )
        self.sun_mask_sigma_px: float = sigma_px
        self.sun_mask_sigma_deg: float = sigma_deg

        self.sky_disc_mask_mode: str = normalize_sky_disc_mask_mode(sky_disc_mask_mode)
        if sky_disc_mask_radius_px is not None:
            r_px = float(sky_disc_mask_radius_px)
            if not (r_px > 0.0):
                raise ValueError(
                    f"sky_disc_mask_radius_px must be > 0 (got {sky_disc_mask_radius_px!r})"
                )
            self.sky_disc_mask_radius_px: float | None = r_px
        else:
            self.sky_disc_mask_radius_px = None

        # Sat config: when ``use_satellite=True`` the loader reads per-frame .npy shards
        # under ``satimg_dir/YYYY/MM/goes15_YYYYMMDD_HHMM.npy`` (GOES-15 GridSat-CONUS,
        # shape ``satimg_npy_shape_hwc`` HWC or its CHW permutation, float16 [0, 1]) and
        # emits real ``sat_tensor`` / ``sat_timefeats`` tensors. When False, sat fields are
        # returned as None and the model's ``if sat_tensor is None`` branch zero-paths the
        # satellite arm. The shape / cadence fields are still validated either way for
        # API parity with PVDataset.
        if len(satimg_npy_shape_hwc) != 3 or any(x < 1 for x in satimg_npy_shape_hwc):
            raise ValueError("satimg_npy_shape_hwc must be three positive ints (H, W, C)")
        self._satimg_npy_shape_hwc = tuple(int(x) for x in satimg_npy_shape_hwc)
        if satimg_time_resolution_min <= 0:
            raise ValueError("satimg_time_resolution_min must be positive")
        self._satimg_dt_min = int(satimg_time_resolution_min)
        self._satimg_dir = Path(satimg_dir).resolve() if str(satimg_dir) else Path(".")
        self._sat_enabled = bool(use_satellite)

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

        # Site + NWP from the per-instance dataset YAML (``self._config_path``):
        #   * lat/lon → ``<paths.data_dir>/info.yaml`` (site.latitude / site.longitude),
        #     matching :class:`dataloader.luoyang_mem.PVDataset`.
        #   * NWP merged CSV → ``paths.folsom_nwp_merged_csv`` if set, else globbed from
        #     the ``paths.nwp_path`` folder (relative to ``data_dir`` unless absolute);
        #     missing/unreadable falls back to None (zero NWP at runtime).
        with self._config_path.open() as f:
            conf = yaml.safe_load(f) or {}
        paths = get_resolved_paths(conf, _PROJECT_ROOT)
        paths_cfg = conf.get("paths") or {}
        data_dir = paths.get("data_dir")
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
        site = info.get("site") or {}
        lat = site.get("latitude")
        lon = site.get("longitude")
        if lat is None or lon is None:
            raise KeyError(
                f"{info_path} must define both site.latitude and site.longitude"
            )
        self.latitude = float(lat)
        self.longitude = float(lon)

        raw_sf = paths_cfg.get("sky_format")
        if raw_sf is not None and str(raw_sf).strip() != "":
            sky_fmt = str(raw_sf).strip().lower()
            if sky_fmt not in ("jpg", "zarr"):
                raise ValueError(
                    f"paths.sky_format must be 'jpg' or 'zarr' (got {raw_sf!r}) in {self._config_path}"
                )
            self._sky_format = sky_fmt
        else:
            self._sky_format = _detect_folsom_sky_format(
                self._skyimg_dir, config_path=self._config_path
            )
            _folsom_progress(
                f"sky format auto-detected: {self._sky_format!r} ({self._skyimg_dir})"
            )

        self._nwp_feature_cols = tuple(_FOLSOM_NWP_FEATURE_COLS)
        # Explicit ``paths.folsom_nwp_merged_csv`` (backward compatible) else glob
        # ``paths.nwp_path`` for the merged CSV. A missing folder / no CSV is tolerated
        # (falls back to zero NWP + invalid mask below); an ambiguous folder is not.
        self._nwp_merged_df = None
        try:
            nwp_csv = _resolve_folsom_nwp_csv_path(conf)
            self._nwp_merged_df = _load_folsom_nwp_merged_csv(nwp_csv)
        except (FileNotFoundError, KeyError):
            self._nwp_merged_df = None

        # API parity with PVDataset: trainer reads ``train_dataset.devDn_list`` to size the
        # device-id embedding. Folsom is a single-sensor station, so a length-1 list is fine
        # (paired with ``dev_idx=700`` returned by ``_build_tensors``).
        self.devDn_list = [0]

        # Irradiance CSV: explicit ``paths.folsom_irradiance_csv`` (preferred when multiple
        # CSVs share ``pv_path``), else glob ``pv_dir`` for exactly one *.csv.
        irr_rel = paths_cfg.get("folsom_irradiance_csv")
        if irr_rel is not None and str(irr_rel).strip() != "":
            self._csv_path = _resolve_folsom_csv_path(conf)
            self.sample_files = [self._csv_path]
        else:
            self.sample_files = list_csv_files(data_dir=pv_dir)
            if not self.sample_files:
                raise FileNotFoundError(f"No CSV files in {pv_dir!r}")
            if len(self.sample_files) != 1:
                names = ", ".join(p.name for p in self.sample_files)
                raise RuntimeError(
                    f"Folsom dataset expects exactly one irradiance CSV under {pv_dir!r}, "
                    f"found {len(self.sample_files)}: {names}"
                )
            self._csv_path = self.sample_files[0].resolve()
        _folsom_progress(f"dataset split={split!r}: preparing {self._csv_path.name} ...")

        # Sky: JPEG directory index or Zarr store (auto-detected or ``paths.sky_format`` override).
        self._sky_gap_threshold = pd.Timedelta(minutes=5)
        self._sky_anchor_max_lag = pd.Timedelta(minutes=5)
        if self._sky_format == "zarr":
            if xr is None:
                raise ImportError(
                    "paths.sky_format=zarr requires ``xarray`` (and a Zarr backend such as ``zarr``). "
                    "Install them or set paths.sky_format to 'jpg'."
                )
            zp = self._skyimg_dir
            if not zp.exists():
                raise FileNotFoundError(f"sky Zarr path not found: {zp}")
            zkey = zp.resolve().as_posix()
            if zkey not in _ZARR_SKY_DS_CACHE:
                _ZARR_SKY_DS_CACHE[zkey] = xr.open_zarr(zp)
            self._skyimg_ds = _ZARR_SKY_DS_CACHE[zkey]
            self._validate_sky_zarr_schema(self._skyimg_ds)
            self._sky_times, self._sky_paths, self._sky_times_ns = [], [], []
            try:
                nt = _folsom_sky_zarr_len_time_utc(self._skyimg_ds)
            except Exception:
                nt = 0
            _folsom_progress(f"sky Zarr: {zp}  (time steps ≈ {nt:,})")
        else:
            self._skyimg_ds = None
            cache_key = f"{self._skyimg_dir}|jpg"
            cached = _SKY_INDEX_CACHE.get(cache_key)
            if cached is None:
                sky_times, sky_paths = self._scan_sky_index()
                sky_times_ns = [int(t.value) for t in sky_times]
                _SKY_INDEX_CACHE[cache_key] = (sky_times, sky_paths, sky_times_ns)
                self._sky_times, self._sky_paths, self._sky_times_ns = sky_times, sky_paths, sky_times_ns
            else:
                self._sky_times, self._sky_paths, self._sky_times_ns = cached
                _folsom_progress(
                    f"sky index (cached): {len(self._sky_times):,} JPGs under {self._skyimg_dir}"
                )

        # Irradiance CSV: one in-memory table (Luoyang ``_csv_cache`` style).
        self._df, self._time_col, self._ghi_dni_dhi_cols = _load_folsom_irradiance_csv(self._csv_path)
        self._n = int(len(self._df))
        if self._n < 1:
            raise RuntimeError(f"{self._csv_path.name}: expected at least one data row")

        # ``p_mean`` is held at 1.0 for Folsom (single GHI sensor; the reconstruction
        # ``pv_pred = kt_pred * target_p_cs * p_mean`` therefore reduces to
        # ``pv_pred = kt_pred * target_p_cs``). Other normalization choices (raw-GHI mean,
        # capacity, daytime-only mean) are intentionally NOT used here yet -- this is the
        # first surgical step in a wider Folsom-vs-Luoyang alignment pass; the kt-input
        # rescale and loss-space changes are tracked separately.
        self._p_mean_scalar = 1.0

        # Precompute normalized clear-sky GHI per CSV row once (1.5M rows for Folsom is fast
        # in pvlib). ``_build_tensors`` slices into this array for both the input window and
        # the forecast window (both are integer CSV row offsets from the anchor), so no
        # per-sample pvlib call is needed. Mirrors the Luoyang offline preprocessing that
        # stores ``p_cs`` in the CSV (see ``SPMF_preprocessing/luoyang/...``).
        _folsom_progress("computing per-row clear-sky GHI via pvlib (ineichen) ...")
        _times_utc = pd.DatetimeIndex(
            pd.to_datetime(self._df[self._time_col].to_numpy(), utc=True)
        )
        self._p_cs_full = _compute_folsom_p_cs(self.latitude, self.longitude, _times_utc)
        _folsom_progress(
            f"p_cs ready: {len(self._p_cs_full):,} rows, max={float(self._p_cs_full.max()):.3f}"
        )

        # Anchor bookkeeping.
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

        # Chronological train/val/test split from config (train_split / val_split / test_split).
        split_train_end = int(n * self._train_split)
        split_val_end = int(n * (self._train_split + self._val_split))
        if not (0 < split_train_end < split_val_end < n):
            raise ValueError(
                f"row split invalid for n={n} "
                f"(train_split={self._train_split}, val_split={self._val_split}, "
                f"test_split={self._test_split}): "
                f"split_train_end={split_train_end}, split_val_end={split_val_end}"
            )
        min_row = self._anchors - (lx - 1) * sx
        max_row = self._anchors + ly * sy
        self._train_anchor_mask = max_row < split_train_end
        self._val_anchor_mask = (min_row >= split_train_end) & (max_row < split_val_end)
        self._test_anchor_mask = min_row >= split_val_end
        if self.split == "train" and not bool(self._train_anchor_mask.any()):
            raise RuntimeError(
                f"split=train: no anchor fits entirely in the first {split_train_end} rows "
                f"(train_split={self._train_split} of n={n}); shorten windows or check data length"
            )
        if self.split == "val" and not bool(self._val_anchor_mask.any()):
            raise RuntimeError(
                f"split=val: no anchor fits entirely in rows [{split_train_end}, {split_val_end}) "
                f"(val_split={self._val_split}); adjust window lengths or stride"
            )
        if self.split == "test" and not bool(self._test_anchor_mask.any()):
            raise RuntimeError(
                f"split=test: no anchor fits entirely from row {split_val_end} onward "
                f"(test_split={self._test_split}); adjust window lengths"
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
        self._train_epoch_len = _DEFAULT_FOLSOM_TRAIN_EPOCH_LEN

        # Train anchor validity filter: keep only anchors whose Y window has at least one
        # row with finite GHI > _FOLSOM_TRAIN_GHI_DAYTIME_THRESHOLD (avoid all-night windows).
        # Only computed for split=="train" — val/test use deterministic strided positions.
        if self.split == "train":
            self._train_anchor_valid_positions = self._compute_train_anchor_valid_positions()
        else:
            self._train_anchor_valid_positions = self._train_anchor_positions

    def _compute_train_anchor_valid_positions(self) -> np.ndarray:
        """
        Scan the GHI column once and return the subset of ``self._train_anchor_positions`` whose
        Y window has any finite ``GHI > _FOLSOM_TRAIN_GHI_DAYTIME_THRESHOLD``.

        Mirrors :class:`dataloader.luoyang_mem.PVDataset`'s train-time filter
        (``inverter_state == VALID_STATE`` on Y rows) so random anchors don't land on
        all-night windows where ``target_pv`` is uniformly zero.
        """
        ghi_col = self._ghi_dni_dhi_cols[0]
        _folsom_progress(
            f"train anchor filter: scanning {ghi_col} column for daytime Y windows ..."
        )
        ghi_full = self._df[ghi_col].to_numpy(dtype=np.float32, copy=False)
        ghi_full = np.where(np.isfinite(ghi_full), ghi_full, 0.0)

        train_anchor_rows = self._anchors[self._train_anchor_positions]  # [N_train]
        if train_anchor_rows.size == 0:
            return self._train_anchor_positions
        # [N_train, ly] absolute Y row indices.
        y_rows = train_anchor_rows[:, None] + self._y_off_1d[None, :]
        y_ghi = ghi_full[y_rows]
        has_daytime = (y_ghi > _FOLSOM_TRAIN_GHI_DAYTIME_THRESHOLD).any(axis=1)
        kept = self._train_anchor_positions[has_daytime].astype(np.intp, copy=False)
        n_kept = int(kept.size)
        n_total = int(train_anchor_rows.size)
        _folsom_progress(
            f"train anchor filter: {n_kept:,} / {n_total:,} train anchors kept "
            f"(GHI > {_FOLSOM_TRAIN_GHI_DAYTIME_THRESHOLD:g} W/m^2 in Y window)"
        )
        if n_kept == 0:
            raise RuntimeError(
                "split=train: no anchor with any Y row above the GHI daytime threshold "
                f"({_FOLSOM_TRAIN_GHI_DAYTIME_THRESHOLD} W/m^2); check data or lower threshold"
            )
        return kept

    def _scan_sky_index(self) -> tuple[list[pd.Timestamp], list[Path]]:
        """
        Build sorted sky index from existing JPG files in ``skyimg_dir``.

        Accepts stems in ``YYYYMMDDHHMMSS`` and keeps only parseable files.
        """
        times: list[pd.Timestamp] = []
        paths: list[Path] = []
        if not self._skyimg_dir.is_dir():
            _folsom_progress(f"sky image dir missing or not a directory: {self._skyimg_dir}")
            return times, paths
        _folsom_progress(f"scanning sky JPEG index in {self._skyimg_dir} ...")
        dir_entries = 0
        for p in self._skyimg_dir.iterdir():
            dir_entries += 1
            if dir_entries % 50_000 == 0:
                _folsom_progress(f"  ... {dir_entries:,} dir entries scanned, {len(times):,} valid JPGs so far")
            if not p.is_file() or p.suffix.lower() != ".jpg":
                continue
            stem = p.stem.strip()
            if len(stem) != 14 or not stem.isdigit():
                continue
            try:
                t = pd.to_datetime(stem, format="%Y%m%d%H%M%S", errors="raise")
            except Exception:
                continue
            times.append(pd.Timestamp(t))
            paths.append(p)
        if not times:
            _folsom_progress("sky index: no valid YYYYMMDDHHMMSS.jpg files found")
            return [], []
        order = np.argsort(np.asarray([t.value for t in times], dtype=np.int64), kind="mergesort")
        times = [times[int(i)] for i in order]
        paths = [paths[int(i)] for i in order]
        _folsom_progress(f"sky index ready: {len(times):,} sorted JPGs ({dir_entries:,} dir entries)")
        return times, paths

    def _black_sky_tensor(self) -> torch.Tensor:
        """Return ``[3, s, s]`` float32 (zeros). Matches PVDataset's float32-in-[0,1] convention."""
        s = self._skyimg_spatial_size
        return torch.zeros((3, s, s), dtype=torch.float32)

    @staticmethod
    def _sky_filename_ts(ts_raw) -> pd.Timestamp:
        """Naive UTC timestamp for sky frame alignment; seconds floored to 0 (Luoyang convention)."""
        ts = pd.Timestamp(ts_raw)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return ts.replace(second=0, microsecond=0, nanosecond=0)

    def _validate_sky_zarr_schema(self, ds: Any) -> None:
        """
        Require ``images`` plus an alignable ``time_utc`` timeline.

        Extra arrays (e.g. ``azimuth``, ``zenith``, ``day_of_year``) are ignored; ``skimg_timefeats``
        still comes from ``compute_solar_features`` on nominal UTC frame times.
        """
        if "images" not in ds.data_vars:
            raise KeyError(
                "Folsom sky Zarr must define data variable ``images`` "
                "(see config/datasets/conf_folsom.yaml)."
            )
        _folsom_sky_zarr_time_dim_and_values(ds, ds["images"])

    def _nominal_sky_frame_times(self, t_end_wall: Any) -> list[pd.Timestamp]:
        """Oldest→newest ``skyimg_window_size`` timestamps spaced by ``_skyimg_dt_min`` ending at anchor."""
        t_end = self._sky_filename_ts(t_end_wall)
        w = self.skyimg_window_size
        dt = self._skyimg_dt_min
        return [t_end - timedelta(minutes=(w - 1 - i) * dt) for i in range(w)]

    def _resize_sky_chw(self, chw: torch.Tensor) -> torch.Tensor:
        """``[3,H,W]`` float32 → ``[3,s,s]`` bilinear (matches JPEG resize target size)."""
        s = self._skyimg_spatial_size
        if chw.shape[-2:] == (s, s):
            return chw
        x = chw.unsqueeze(0)
        y = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
        return y.squeeze(0)

    def _tensor_from_zarr_image_tile(self, tile: np.ndarray) -> torch.Tensor:
        """One timestep tile: HWC or CHW → ``[3,s,s]`` float32 in ``[0, 1]``."""
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
        Stack ``[W, 3, H, W]`` from Zarr using the same nominal UTC grid as the JPEG loader.

        Selects rows whose ``time_utc`` falls in ``[nominal[0], nominal[-1]]`` (index-based, so it
        works for ``sky_xr_120.zarr``-style stores with a separate ``time_utc`` array), then
        nearest-neighbour per nominal step (90 s tolerance). If the newest kept row is too far
        before the anchor (``_sky_anchor_max_lag``), returns black frames (same spirit as JPEG).
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

    def _load_sky_tensor(self, path: Path) -> torch.Tensor:
        """Return ``[3, s, s]`` float32 in ``[0, 1]`` from a raw (unflipped) Folsom JPG."""
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
                    arr = np.asarray(im, dtype=np.uint8).copy()
                t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
                return t.to(torch.float32) / 255.0
        except Exception:
            pass
        return self._black_sky_tensor()

    def _history_sky_frame_records(self, t_end) -> tuple[list[pd.Timestamp], list[Path | None]]:
        """
        Resolve the last ``N=skyimg_window_size`` existing sky images at/before ``t_end``.

        - No exact-minute filename assumptions are made.
        - If fewer than ``N`` images exist, left-pad with ``None`` placeholders (black frames).
        - Day/night breaks are handled by a continuity rule: once a consecutive timestamp gap
          exceeds ``self._sky_gap_threshold``, all older frames are invalidated (set to black).
        - If the newest kept sky frame is more than ``self._sky_anchor_max_lag`` before ``t_end``,
          no sky data is considered valid for this anchor: all slots are ``None`` (black tensors).
        - Returned lists are oldest → newest and always length ``N``.
        """
        t_end = pd.Timestamp(t_end)
        w = self.skyimg_window_size
        if not self._sky_times:
            return [t_end] * w, [None] * w
        cutoff = bisect.bisect_right(self._sky_times_ns, int(t_end.value))
        start = max(0, cutoff - w)
        sel_times = self._sky_times[start:cutoff]
        sel_paths = self._sky_paths[start:cutoff]

        # Keep only the newest contiguous tail near anchor. Once a big gap is found, older frames are invalid.
        if len(sel_times) >= 2:
            keep_start = 0
            for j in range(len(sel_times) - 1, 0, -1):
                if (sel_times[j] - sel_times[j - 1]) > self._sky_gap_threshold:
                    keep_start = j
                    break
            if keep_start > 0:
                sel_times = sel_times[keep_start:]
                sel_paths = sel_paths[keep_start:]

        # Newest real sky must be close to anchor; otherwise images are unrelated (e.g. hours old).
        if sel_times:
            newest = pd.Timestamp(sel_times[-1])
            if t_end - newest > self._sky_anchor_max_lag:
                sel_times = []
                sel_paths = []

        pad = w - len(sel_times)
        if pad > 0:
            sel_times = [t_end] * pad + sel_times
            sel_paths = [None] * pad + sel_paths
        return sel_times, sel_paths

    def _stack_sky_frames(self, frame_paths: list[Path | None]) -> torch.Tensor:
        # Per-sample hot path (called from every __getitem__): keep silent. The one-time
        # ``sky index ready: N JPGs`` line emitted at construction is enough to confirm setup.
        frames: list[torch.Tensor] = [
            self._black_sky_tensor() if p is None else self._load_sky_tensor(p)
            for p in frame_paths
        ]
        return torch.stack(frames, dim=0)

    def _get_fisheye_fit(self) -> dict:
        """Lazy-load + cache the Folsom fisheye fit (``cx, cy, f, alpha0``).

        Shared by :meth:`_build_ray_and_valid_cache` (for the calibrated
        ``ray_map``) and :meth:`_compute_sun_mask_for_frames` (for the sun
        mask). The CSV is read at most once per dataset instance.
        """
        if self._fisheye_fit is None:
            self._fisheye_fit = fisheye_sunmask.load_fisheye_fit("folsom")
        return self._fisheye_fit

    def _build_ray_and_valid_cache(self) -> None:
        """Populate ``_ray_map_cache`` and ``_sky_valid_cache`` together (one ``compute_ray_map`` call).

        ``compute_ray_map`` is a pure function of ``(H, W, fit)`` so a single
        instance suffices for all samples; ``sun_mask`` reuses the same fit.
        """
        s = self._skyimg_spatial_size
        ray, valid = compute_ray_map(s, s, self._get_fisheye_fit())
        self._ray_map_cache = torch.from_numpy(np.ascontiguousarray(ray, dtype=np.float32))
        self._sky_valid_cache = torch.from_numpy(np.ascontiguousarray(valid, dtype=np.float32))

    def _get_ray_map(self) -> torch.Tensor:
        """Lazy ``[3, H, W]`` float32 image-axis ray map (``fisheye_raymap.compute_ray_map``)."""
        if self._ray_map_cache is None:
            self._build_ray_and_valid_cache()
        return self._ray_map_cache  # type: ignore[return-value]

    def _get_sky_valid(self) -> torch.Tensor:
        """Lazy ``[1, H, W]`` float32 fisheye validity mask (1.0 inside the image circle)."""
        if self._sky_valid_cache is None:
            self._build_ray_and_valid_cache()
        return self._sky_valid_cache  # type: ignore[return-value]

    def _compute_sun_mask_for_frames(
        self,
        frame_timestamps_utc: list[pd.Timestamp],
    ) -> torch.Tensor:
        """Per-frame ``[T, 1, H, W]`` float32 sun mask on the raw-image pixel grid.

        Thin wrapper around
        :func:`SPMF_preprocessing.fisheye_calib.fisheye_sunmask.compute_sun_mask`,
        which owns the lens math (fit at native 1536 in flipped-u space; project in
        flip space, mirror ``u`` to raw, Euclidean disc ``R = f_s * deg2rad(radius)``).
        This method only:

        1. Calls :func:`compute_solar_features` for the per-frame
           ``(azimuth, zenith)`` (meteorological convention: 0°=N, 90°=E,
           clockwise; zenith from up).
        2. Lazily loads the Folsom fit on first call.
        3. Delegates the actual mask construction to the calibration module.
        4. Wraps the result as a contiguous ``[T, 1, H, W]`` ``torch.float32``
           tensor with a channel dim.

        Frames with ``zen >= 90°`` (sun below horizon) return all zeros.
        """
        if not frame_timestamps_utc:
            raise ValueError("_compute_sun_mask_for_frames: frame_timestamps_utc is empty")
        t = int(len(frame_timestamps_utc))

        feats = compute_solar_features(frame_timestamps_utc, self.latitude, self.longitude)
        az_deg = np.asarray(feats["azimuth"], dtype=np.float64)
        ze_deg = np.asarray(feats["zenith"], dtype=np.float64)
        if az_deg.shape[0] != t or ze_deg.shape[0] != t:
            raise RuntimeError(
                f"compute_solar_features returned T={az_deg.shape[0]}/{ze_deg.shape[0]} "
                f"for {t} frames"
            )

        mask_np = fisheye_sunmask.compute_sun_mask(
            az_deg=az_deg,
            zen_deg=ze_deg,
            image_size=self._skyimg_spatial_size,
            radius_deg=self.sun_mask_radius_deg,
            fit=self._get_fisheye_fit(),
            mode=self.sun_mask_mode,
            sigma_px=self.sun_mask_sigma_px,
            sigma_deg=self.sun_mask_sigma_deg,
        )
        mask_t = torch.from_numpy(np.ascontiguousarray(mask_np, dtype=np.float32))
        return mask_t.unsqueeze(1).contiguous()  # [T, 1, H, W]

    def _compute_sky_disc_mask_for_frames(
        self,
        frame_timestamps: list[pd.Timestamp],
        t_dim: int,
        h_dim: int,
        w_dim: int,
    ) -> torch.Tensor:
        """Per-frame ``[T, 1, H, W]`` float32 sky-disc keep mask (0/1)."""
        if self.sky_disc_mask_mode == "none":
            raise ValueError(
                "_compute_sky_disc_mask_for_frames called but sky_disc_mask_mode is 'none'"
            )
        return compute_sky_disc_mask(
            t_dim,
            h_dim,
            w_dim,
            self.sky_disc_mask_mode,
            fit=self._get_fisheye_fit(),
            frame_timestamps=frame_timestamps,
            latitude=self.latitude,
            longitude=self.longitude,
            radius_px_override=self.sky_disc_mask_radius_px,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    def _build_sky_channels(
        self,
        rgb_frames: torch.Tensor,
        frame_timestamps: list[pd.Timestamp] | None = None,
    ) -> torch.Tensor:
        """Assemble ``[T, sky_in_channels, H, W]`` sky tensor per ``self.sky_channels``.

        ``rgb_frames`` is the existing ``[T, 3, H, W]`` float32 tensor produced by
        :meth:`_stack_sky_from_zarr` / :meth:`_stack_sky_frames`. Channel order in the
        output follows ``self.sky_channels`` exactly. Default config (``("rgb",)``)
        returns ``rgb_frames`` unchanged so behavior is byte-identical to today.
        """
        if self.sky_channels == _DEFAULT_SKY_CHANNELS:
            return rgb_frames
        if rgb_frames.ndim != 4 or rgb_frames.shape[1] != 3:
            raise ValueError(
                f"_build_sky_channels: expected rgb_frames [T, 3, H, W], got {tuple(rgb_frames.shape)}"
            )
        t_dim, _, h_dim, w_dim = rgb_frames.shape
        parts: list[torch.Tensor] = []
        for name in self.sky_channels:
            if name == _SKY_CHANNEL_RGB:
                parts.append(rgb_frames)
            elif name == _SKY_CHANNEL_RAY_MAP:
                ray = self._get_ray_map()
                if ray.shape[-2:] != (h_dim, w_dim):
                    raise RuntimeError(
                        f"ray map cache shape {tuple(ray.shape)} does not match "
                        f"sky frames spatial size {(h_dim, w_dim)}"
                    )
                parts.append(ray.unsqueeze(0).expand(t_dim, -1, -1, -1))
            elif name == _SKY_CHANNEL_SUN_MASK:
                if frame_timestamps is None:
                    raise ValueError(
                        "_build_sky_channels: 'sun_mask' requires frame_timestamps "
                        "(per-frame UTC pd.Timestamps for compute_solar_features)"
                    )
                sun_mask = self._compute_sun_mask_for_frames(list(frame_timestamps))
                if sun_mask.shape[0] != t_dim:
                    raise RuntimeError(
                        f"sun mask T={sun_mask.shape[0]} does not match rgb T={t_dim}"
                    )
                if tuple(sun_mask.shape[-2:]) != (h_dim, w_dim):
                    raise RuntimeError(
                        f"sun mask spatial size {tuple(sun_mask.shape[-2:])} does not match "
                        f"sky frame spatial size {(h_dim, w_dim)}"
                    )
                parts.append(sun_mask)
            elif name == _SKY_CHANNEL_SKY_MASK:
                if frame_timestamps is None:
                    raise ValueError(
                        "_build_sky_channels: 'sky_mask' requires frame_timestamps "
                        "(per-frame UTC pd.Timestamps for disc mask computation)"
                    )
                disc_mask = self._compute_sky_disc_mask_for_frames(
                    list(frame_timestamps), t_dim, h_dim, w_dim
                )
                # Black/padding frames from _black_sky_tensor() are all-zero RGB.
                valid = rgb_frames.abs().amax(dim=(1, 2, 3)) > 0
                disc_mask = disc_mask * valid.view(t_dim, 1, 1, 1).to(disc_mask.dtype)
                parts.append(disc_mask)
            else:
                raise ValueError(f"sky_channels: unknown feature {name!r}")
        out = torch.cat(parts, dim=1).contiguous()
        if out.dtype != torch.float32:
            out = out.to(torch.float32)
        return out

    def _interpolate_nwp(self, forecast_timestamps: list[pd.Timestamp]) -> torch.Tensor:
        """
        Interpolate merged NWP features to forecast timestamps (strict, no extrapolation).

        Output shape: ``[T_out, C+1]`` = features + one per-timestep invalid mask channel.
        If NWP table is missing, returns zeros for features and ``1`` mask everywhere.
        """
        t_out = len(forecast_timestamps)
        c = len(self._nwp_feature_cols)
        if self._nwp_merged_df is None:
            zeros = np.zeros((t_out, c), dtype=np.float32)
            ones_mask = np.ones((t_out, 1), dtype=np.float32)
            return torch.from_numpy(np.concatenate([zeros, ones_mask], axis=1))

        vt = pd.DatetimeIndex(self._nwp_merged_df["valtime"])
        xp_ns = vt.asi8.astype(np.float64)
        xq_ns = pd.DatetimeIndex(pd.to_datetime(forecast_timestamps)).asi8.astype(np.float64)
        cols = []
        for col in self._nwp_feature_cols:
            fp = self._nwp_merged_df[col].to_numpy(dtype=np.float64)
            cols.append(_strict_interp_series(xp_ns, fp, xq_ns))
        nwp_interp = np.column_stack(cols) if cols else np.empty((t_out, 0), dtype=np.float64)
        clean, bad_mask = _sanitize_nwp_interp(nwp_interp)
        return torch.from_numpy(np.concatenate([clean, bad_mask], axis=1))

    # ------------------------------------------------------------------
    # Folsom GOES-15 satellite frames (per-frame .npy shards under
    # ``satimg_dir/YYYY/MM/goes15_YYYYMMDD_HHMM.npy``; shape (3, 100, 100)
    # float16 in [0, 1]; structural NCEI gaps yield missing files which we
    # zero-fill below). Only invoked when ``self._sat_enabled`` is True;
    # the helpers themselves are pure functions of the constructor-set sat
    # config fields and cost nothing when unused.
    # ------------------------------------------------------------------
    def _satimg_anchor_end_utc(self, time0: Any) -> pd.Timestamp:
        """``time0`` shifted back by ``_FOLSOM_SAT_ANCHOR_OFFSET_MIN`` and floored to the
        ``_satimg_dt_min`` cadence (naive UTC). This is the **newest** frame time."""
        t = pd.Timestamp(time0)
        if t.tzinfo is not None:
            t = t.tz_convert("UTC").tz_localize(None)
        t = t - pd.Timedelta(minutes=_FOLSOM_SAT_ANCHOR_OFFSET_MIN)
        dt = int(self._satimg_dt_min)
        floored_minute = (int(t.minute) // dt) * dt
        return t.replace(
            minute=floored_minute, second=0, microsecond=0, nanosecond=0
        )

    def _satimg_history_frame_times(self, time0: Any) -> list[pd.Timestamp]:
        """``satimg_window_size`` naive-UTC timestamps at ``_satimg_dt_min`` spacing ending at
        :meth:`_satimg_anchor_end_utc` (``time0``). Returned oldest → newest."""
        t_end = self._satimg_anchor_end_utc(time0)
        w = int(self.satimg_window_size)
        dt = int(self._satimg_dt_min)
        return [t_end - pd.Timedelta(minutes=(w - 1 - i) * dt) for i in range(w)]

    def _satimg_npy_path(self, t_utc_naive: pd.Timestamp) -> Path:
        """``<satimg_dir>/YYYY/MM/goes15_YYYYMMDD_HHMM.npy`` for a naive-UTC timestamp."""
        u = pd.Timestamp(t_utc_naive)
        return (
            self._satimg_dir
            / f"{u.year:04d}"
            / f"{u.month:02d}"
            / f"goes15_{u.strftime('%Y%m%d_%H%M')}.npy"
        )

    def _dummy_satimg_tensor(self) -> torch.Tensor:
        """Zero ``[3, H, W]`` float32 (PVDataset-shaped) for missing/unreadable shards."""
        h, w, c = self._satimg_npy_shape_hwc
        return torch.zeros((c, h, w), dtype=torch.float32)

    def _load_satimg_tensor(self, path: Path) -> torch.Tensor:
        """Folsom GridSat .npy → ``[3, H, W]`` float32 in [0, 1].

        Files on disk are CHW (``(3, 100, 100)``) float16, already pre-normalized. Accepts
        both CHW and HWC (for forward compatibility) by matching against
        ``_satimg_npy_shape_hwc`` (HWC) and the CHW permutation thereof. Anything else (or
        unreadable / missing file) → zero tensor (mirrors the structural NCEI gap policy).
        """
        h, w, c = self._satimg_npy_shape_hwc
        chw_shape = (c, h, w)
        hwc_shape = (h, w, c)
        try:
            if path.is_file():
                arr = np.load(path, allow_pickle=False)
                if arr.shape == chw_shape:
                    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
                if arr.shape == hwc_shape:
                    arr = np.ascontiguousarray(arr, dtype=np.float32)
                    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        except Exception:
            pass
        return self._dummy_satimg_tensor()

    def _stack_satimg_frames(self, frame_times_utc: list[pd.Timestamp]) -> torch.Tensor:
        """Stack ``[W, 3, H, W]`` float32 sat history. Missing shards → zero frames."""
        return torch.stack(
            [self._load_satimg_tensor(self._satimg_npy_path(t)) for t in frame_times_utc],
            dim=0,
        )

    def __len__(self) -> int:
        if self.split == "train":
            return self._train_epoch_len
        if self.split == "val":
            return self._num_val_windows
        return self._num_test_windows

    def sky_inspect(self, anchor: int) -> dict:
        """
        Resolve last-N sky records for ``anchor`` without building full tensors.

        JPEG mode: per-frame paths. Zarr mode: nominal UTC grid and row count inside the Zarr slice.
        """
        x_idx = anchor + self._x_tail_1d
        sub_x = self._df.iloc[x_idx]
        t_x_end = sub_x[self._time_col].iloc[-1]
        if self._sky_format == "zarr":
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
                "paths": [],
                "n_files_found": n_z,
                "skyimg_dir": self._skyimg_dir,
                "sky_format": "zarr",
                "zarr_slice_timesteps": n_z,
            }
        sk_times, sk_paths = self._history_sky_frame_records(t_x_end)
        n_found = sum(1 for p in sk_paths if p is not None)
        t_last_n = pd.Timestamp(sk_times[-1])
        return {
            "anchor_row": int(anchor),
            "last_input_time_utc_naive": str(t_last_n),
            "n_frames": len(sk_times),
            "utc_times": [str(pd.Timestamp(t)) for t in sk_times],
            "paths": [p if p is not None else Path("<black>") for p in sk_paths],
            "n_files_found": n_found,
            "skyimg_dir": self._skyimg_dir,
            "sky_format": "jpg",
        }

    def _build_tensors(self, anchor: int) -> dict[str, Any]:
        x_idx = anchor + self._x_tail_1d
        y_idx = anchor + self._y_off_1d
        sub_x = self._df.iloc[x_idx]
        sub_y = self._df.iloc[y_idx]

        gx = sub_x[self._ghi_dni_dhi_cols[0]]
        dx = sub_x[self._ghi_dni_dhi_cols[1]]
        hx = sub_x[self._ghi_dni_dhi_cols[2]]
        x_stack = np.stack([gx.to_numpy(), dx.to_numpy(), hx.to_numpy()], axis=0).astype(np.float32)
        x_stack = np.nan_to_num(x_stack, nan=0.0, posinf=0.0, neginf=0.0)
        # Mask depends on GHI only; DNI/DHI validity is intentionally ignored (PV ViT trains on GHI).
        valid_in = np.isfinite(gx.to_numpy())
        input_mask = torch.from_numpy(valid_in.astype(np.float32)).unsqueeze(0)

        gy = sub_y[self._ghi_dni_dhi_cols[0]]
        dy = sub_y[self._ghi_dni_dhi_cols[1]]
        hy = sub_y[self._ghi_dni_dhi_cols[2]]
        y_raw = np.stack([gy.to_numpy(), dy.to_numpy(), hy.to_numpy()], axis=0).astype(np.float32)
        # Mask depends on GHI only (row 0 of y_raw); DNI/DHI validity is intentionally ignored.
        # folsom-kt overlay: AND finite-GHI mask with a daytime gate sourced from the
        # precomputed clear-sky ``_p_cs_full`` (normalized clearsky_ghi / _FOLSOM_GHI_SCALE).
        # See ``_FOLSOM_TARGET_MASK_NIGHT_THRESHOLD_P_CS`` for the kt-equivalent threshold.
        valid_out = np.isfinite(y_raw[0])
        daytime_out = self._p_cs_full[y_idx] >= _FOLSOM_TARGET_MASK_NIGHT_THRESHOLD_P_CS
        target_mask_np = (valid_out & daytime_out).astype(np.float32)
        target_mask = torch.from_numpy(target_mask_np)
        y_stack = np.nan_to_num(y_raw, nan=0.0, posinf=0.0, neginf=0.0)

        ghi, dni, dhi = x_stack[0], x_stack[1], x_stack[2]
        tg, td, th = y_stack[0], y_stack[1], y_stack[2]

        # Clear-sky / kt fields. ``p_cs`` is the normalized clear-sky GHI
        # ``clearsky_ghi / _FOLSOM_GHI_SCALE``; ``p_mean`` is held at 1.0 (see ``__init__``)
        # so ``kt = ghi_raw / (p_cs + eps) * kt_mask`` and the reconstruction
        # ``pv_pred = kt_pred * target_p_cs * p_mean`` reduces to
        # ``pv_pred = kt_pred * target_p_cs``. Note: ``ghi_raw`` is in W/m^2, so ``kt`` here
        # is unbounded (~up to a few thousand) until the kt-input rescale is added in a
        # follow-up change.
        p_cs_x = self._p_cs_full[x_idx]
        p_cs_y = self._p_cs_full[y_idx]
        kt_mask_np = (p_cs_x > _FOLSOM_KT_DAYTIME_THRESHOLD).astype(np.float32)
        kt_np = (ghi / (p_cs_x * self._p_mean_scalar + _FOLSOM_KT_EPS)) * kt_mask_np
        kt = torch.from_numpy(kt_np.astype(np.float32)).unsqueeze(0)
        kt_mask = torch.from_numpy(kt_mask_np).unsqueeze(0)
        p_cs = torch.from_numpy(p_cs_x.astype(np.float32)).unsqueeze(0)
        target_p_cs = torch.from_numpy(p_cs_y.astype(np.float32))
        p_mean = torch.tensor(self._p_mean_scalar, dtype=torch.float32)

        x_times = sub_x[self._time_col]
        if bool(x_times.isna().any()):
            raise ValueError(f"NaT in {self._time_col!r} for input window")
        timestamps = _folsom_to_timestamps(x_times.tolist())
        time0 = timestamps[-1]
        forecast_timestamps = [
            time0 + pd.Timedelta(minutes=self.pv_output_interval_min * (i + 1))
            for i in range(self.pv_output_len)
        ]
        nwp_tensor = self._interpolate_nwp(forecast_timestamps)

        irr_solar = compute_solar_features(timestamps, self.latitude, self.longitude)
        irr_tf = solar_features_encoder(irr_solar)
        irr_dtf = delta_time_encoder(timestamps, time0)
        irr_timefeats = torch.cat([irr_tf, irr_dtf.unsqueeze(1)], dim=1)

        forecast_solar = compute_solar_features(forecast_timestamps, self.latitude, self.longitude)
        f_tf = solar_features_encoder(forecast_solar)
        f_dtf = delta_time_encoder(forecast_timestamps, time0)
        forecast_timefeats = torch.cat([f_tf, f_dtf.unsqueeze(1)], dim=1)

        if self._sat_enabled:
            sat_times = self._satimg_history_frame_times(time0)
            sat_tensor = self._stack_satimg_frames(sat_times)
            sat_solar = compute_solar_features(sat_times, self.latitude, self.longitude)
            sat_tf = solar_features_encoder(sat_solar)
            sat_dtf = delta_time_encoder(sat_times, time0)
            sat_timefeats = torch.cat([sat_tf, sat_dtf.unsqueeze(1)], dim=1)
        else:
            sat_tensor = None
            sat_timefeats = None

        t_x_end = sub_x[self._time_col].iloc[-1]
        if self._sky_format == "zarr":
            nominal = self._nominal_sky_frame_times(t_x_end)
            rgb_frames = self._stack_sky_from_zarr(t_x_end)
            skimg_tensor = self._build_sky_channels(rgb_frames, nominal)
            skimg_solar_features = compute_solar_features(nominal, self.latitude, self.longitude)
            skimg_tf = solar_features_encoder(skimg_solar_features)
            skimg_dtf = delta_time_encoder(nominal, time0)
            skimg_timefeats = torch.cat([skimg_tf, skimg_dtf.unsqueeze(1)], dim=1)
            skimg_timestamps = [t.strftime("%Y%m%d%H%M%S") for t in nominal]
        else:
            skimg_timestamps, skimg_paths = self._history_sky_frame_records(t_x_end)
            skimg_solar_features = compute_solar_features(
                skimg_timestamps, self.latitude, self.longitude
            )
            skimg_tf = solar_features_encoder(skimg_solar_features)
            skimg_dtf = delta_time_encoder(skimg_timestamps, time0)
            skimg_timefeats = torch.cat([skimg_tf, skimg_dtf.unsqueeze(1)], dim=1)
            rgb_frames = self._stack_sky_frames(skimg_paths)
            skimg_tensor = self._build_sky_channels(rgb_frames, skimg_timestamps)
            skimg_timestamps = [
                (None if p is None else pd.Timestamp(t).strftime("%Y%m%d%H%M%S"))
                for t, p in zip(skimg_timestamps, skimg_paths)
            ]

        input_timestamps_utc = [str(pd.Timestamp(t)) for t in timestamps]
        forecast_timestamps_utc = [str(pd.Timestamp(t)) for t in forecast_timestamps]
        # Single-sensor station; index into ``self.devDn_list = [0]``.
        dev_idx = torch.tensor(700, dtype=torch.long)

        # Match PVDataset: pv is [1, T_in] (sensor/dev dim leading), target_pv is [T_out].
        # Luoyang-parity: pv / target_pv are the RAW signal (W/m^2 for Folsom GHI). The
        # normalization role is played by ``p_mean`` inside the ``kt`` denominator, not by
        # dividing the signal here. Reconstruction ``pv = kt * p_cs * p_mean`` then recovers
        # raw GHI in W/m^2, and the loss ``criterion(pv_pred, target_pv)`` is in raw W/m^2.
        pv_tensor = torch.from_numpy(ghi.astype(np.float32)).unsqueeze(0)
        target_pv_tensor = torch.from_numpy(tg.astype(np.float32))
        return {
            "dev_idx": dev_idx,
            "pv": pv_tensor,
            "pv_mask": input_mask,
            "pv_timefeats": irr_timefeats,
            "kt": kt,
            "kt_mask": kt_mask,
            "p_cs": p_cs,
            "p_mean": p_mean,
            "ghi": torch.from_numpy(ghi.astype(np.float32)),
            "dni": torch.from_numpy(dni.astype(np.float32)),
            "dhi": torch.from_numpy(dhi.astype(np.float32)),
            "input_mask": input_mask,
            "irr_timefeats": irr_timefeats,
            "forecast_timefeats": forecast_timefeats,
            "target_ghi": torch.from_numpy(tg.astype(np.float32)),
            "target_dni": torch.from_numpy(td.astype(np.float32)),
            "target_dhi": torch.from_numpy(th.astype(np.float32)),
            "target_pv": target_pv_tensor,
            "target_mask": target_mask,
            "target_p_cs": target_p_cs,
            "sat_tensor": sat_tensor,
            "sat_timefeats": sat_timefeats,
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


def collate_folsom_irradiance(batch: list[dict]) -> dict:
    """Stack list of samples into one dict with batch dim ``B`` first (Luoyang ``collate_batched`` style)."""
    if not batch:
        raise ValueError("empty batch")

    def _stack(key: str) -> torch.Tensor:
        return torch.stack([s[key] for s in batch])

    out: dict[str, Any] = {
        "ghi": _stack("ghi"),
        "dni": _stack("dni"),
        "dhi": _stack("dhi"),
        "input_mask": _stack("input_mask"),
        "irr_timefeats": _stack("irr_timefeats"),
        "forecast_timefeats": _stack("forecast_timefeats"),
        "target_ghi": _stack("target_ghi"),
        "target_dni": _stack("target_dni"),
        "target_dhi": _stack("target_dhi"),
        "target_mask": _stack("target_mask"),
    }
    for key in ("skimg_tensor", "skimg_timefeats", "nwp_tensor"):
        vals = [s[key] for s in batch]
        if vals[0] is None:
            if not all(v is None for v in vals):
                raise ValueError(f"collate_folsom_irradiance: mixed None and tensor for {key!r}")
            out[key] = None
        else:
            out[key] = torch.stack(vals)
    out["skimg_timestamps"] = [b["skimg_timestamps"] for b in batch]
    return out


def _resolve_folsom_dataset_paths(conf: dict, conf_path: Path) -> tuple[Path, Path, Path]:
    """Return ``(pv_dir, skyimg_dir, satimg_dir)`` from a Folsom dataset YAML (new schema)."""
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


def build_folsom_irradiance_datasets_from_conf(
    conf: dict | None = None,
    *,
    conf_path: Path | str | None = None,
    train_epoch_len: int = _DEFAULT_FOLSOM_TRAIN_EPOCH_LEN,
    skyimg_window_size: int | None = None,
) -> tuple[FolsomIrradianceDataset, FolsomIrradianceDataset]:
    """
    Build train/test :class:`FolsomIrradianceDataset` from a Folsom dataset YAML (new schema:
    ``config/datasets/conf_folsom.yaml``).

    Reads ``paths.{data_dir, pv_path, sky_image_path, sat_path}`` and the ``sampling:`` section
    (PVDataset-style window / stride / image-shape fields). Lat/lon comes from
    ``<paths.data_dir>/info.yaml`` and the optional NWP merged CSV from
    ``paths.folsom_nwp_merged_csv`` (or globbed from ``paths.nwp_path`` when unset) — both read
    inside the dataset constructor.

    A ``conf_path`` is required (it is also forwarded to the dataset as ``config_path``); pass
    ``conf`` if you've already loaded the YAML to avoid re-reading it. ``train_epoch_len`` is
    written onto the returned train dataset (the constructor mirrors PVDataset and does not
    take it). If ``skyimg_window_size`` is set, it overrides ``sampling.skyimg_window_size``.
    """
    if conf_path is None:
        raise TypeError("build_folsom_irradiance_datasets_from_conf: conf_path is required")
    cfg_path = Path(conf_path)
    if conf is None:
        conf = load_folsom_conf(cfg_path)

    sampling_cfg = conf.get("sampling") or {}
    if not sampling_cfg:
        raise KeyError(f"dataset config {cfg_path} is missing a non-empty 'sampling:' section")

    def _req_s(key: str):
        if key not in sampling_cfg:
            raise KeyError(f"dataset config sampling.{key} is required (in {cfg_path})")
        return sampling_cfg[key]

    pv_dir, skyimg_dir, satimg_dir = _resolve_folsom_dataset_paths(conf, cfg_path)

    sky_w = int(skyimg_window_size if skyimg_window_size is not None else _req_s("skyimg_window_size"))
    shwc = _req_s("satimg_npy_shape_hwc")
    if not isinstance(shwc, (list, tuple)) or len(shwc) != 3:
        raise ValueError(f"sampling.satimg_npy_shape_hwc must be [H, W, C] (in {cfg_path})")

    kwargs: dict[str, Any] = dict(
        config_path=str(cfg_path),
        pv_dir=str(pv_dir),
        skyimg_dir=str(skyimg_dir),
        satimg_dir=str(satimg_dir),
        csv_interval_min=int(_req_s("csv_interval_min")),
        pv_input_interval_min=int(_req_s("pv_input_interval_min")),
        pv_input_len=int(_req_s("pv_input_len")),
        pv_output_interval_min=int(_req_s("pv_output_interval_min")),
        pv_output_len=int(_req_s("pv_output_len")),
        pv_train_time_fraction=float(sampling_cfg.get("pv_train_time_fraction", 0.7)),
        train_split=float(sampling_cfg.get("train_split", 0.66)),
        val_split=float(sampling_cfg.get("val_split", 0.18)),
        test_split=float(sampling_cfg.get("test_split", 0.16)),
        test_anchor_stride_min=int(_req_s("test_anchor_stride_min")),
        val_anchor_stride_min=int(_req_s("val_anchor_stride_min")),
        test_collect_time_match_tolerance_min=int(sampling_cfg.get("test_collect_time_match_tolerance_min", 0)),
        skyimg_window_size=sky_w,
        skyimg_time_resolution_min=int(_req_s("skyimg_time_resolution_min")),
        skyimg_spatial_size=int(_req_s("skyimg_spatial_size")),
        satimg_window_size=int(_req_s("satimg_window_size")),
        satimg_time_resolution_min=int(_req_s("satimg_time_resolution_min")),
        satimg_npy_shape_hwc=tuple(int(x) for x in shwc),
        use_satellite=bool(sampling_cfg.get("use_satellite", False)),
    )
    sky_channels, sun_mask_radius_deg, sky_disc_mask_mode, sun_mask_mode = sky_knobs_to_internal(
        sampling_cfg.get("ray_map"),
        sampling_cfg.get("sun_mask"),
        sampling_cfg.get("sky_mask"),
    )
    sigma_px, sigma_deg = resolve_sun_mask_sigmas(sampling_cfg)
    kwargs.update(
        sky_channels=sky_channels,
        sun_mask_mode=sun_mask_mode,
        sun_mask_radius_deg=sun_mask_radius_deg,
        sun_mask_sigma_px=sigma_px,
        sun_mask_sigma_deg=sigma_deg,
        sky_disc_mask_mode=sky_disc_mask_mode,
        sky_disc_mask_radius_px=None,
    )
    train_ds = FolsomIrradianceDataset(split="train", **kwargs)
    test_ds = FolsomIrradianceDataset(split="test", **kwargs)
    train_ds._train_epoch_len = max(1, int(train_epoch_len))
    return train_ds, test_ds


__all__ = [
    "FolsomIrradianceDataset",
    "FOLSOM_GHI_DNI_DHI_KEYS",
    "FOLSOM_BATCH_TENSOR_KEYS",
    "collate_folsom_irradiance",
    "build_folsom_irradiance_datasets_from_conf",
    "load_folsom_conf",
    "run_smoke_cli",
    "sky_knobs_to_internal",
    "normalize_ray_map",
    "normalize_sun_mask",
    "normalize_sky_mask",
    "resolve_sun_mask_sigmas",
]


_SKY_SAMPLE_KEYS = frozenset({"skimg_tensor", "skimg_timefeats"})


def _smoke_section(title: str, lines: list[str]) -> None:
    bar = "=" * 72
    print(bar)
    print(title)
    print(bar)
    for line in lines:
        print(f"    {line}")


def _smoke_irradiance_lines(
    times: list[str],
    g: torch.Tensor,
    dn: torch.Tensor,
    dh: torch.Tensor,
) -> list[str]:
    n = int(g.shape[0])
    lines: list[str] = [f"size (per series, num steps): ({n},)"]
    if n == 0:
        lines.append("  (empty series)")
        return lines
    for i in range(min(2, n)):
        lines.append(
            f"  first[{i}]  t={times[i]}  ghi={float(g[i]):.3f}  dni={float(dn[i]):.3f}  dhi={float(dh[i]):.3f}"
        )
    if n > 2:
        for i in (n - 2, n - 1):
            lines.append(
                f"  last[{i}]  t={times[i]}  ghi={float(g[i]):.3f}  dni={float(dn[i]):.3f}  dhi={float(dh[i]):.3f}"
            )
    return lines


def _smoke_sky_stem_lines(sk_ts: list) -> list[str]:
    n = len(sk_ts)
    lines: list[str] = []
    if n == 0:
        lines.append("  (no sky frames in window)")
        return lines
    for i in range(min(2, n)):
        lines.append(f"  first[{i}]  stem={sk_ts[i]!r}")
    if n > 2:
        for i in (n - 2, n - 1):
            lines.append(f"  last[{i}]  stem={sk_ts[i]!r}")
    return lines


def _find_csv_data_row_index_for_time(
    df: pd.DataFrame,
    time_col: str,
    target: pd.Timestamp,
) -> int:
    """
    Return the first data row index whose ``time_col`` **exactly** matches ``target``.

    ``df`` is the in-memory irradiance table (``FolsomIrradianceDataset._df``).
    """
    tgt = pd.Timestamp(target)
    if tgt.tzinfo is not None:
        tgt = tgt.tz_convert("UTC").tz_localize(None)
    ts = df[time_col]
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    ok = ts == tgt
    if not bool(ok.any()):
        raise ValueError(
            f"No CSV row with {time_col!r} exactly equal to {tgt}. "
            "Use the same string as in the file (minute-aligned for 1-minute CSV)."
        )
    return int(np.flatnonzero(ok.to_numpy())[0])


def _validate_smoke_anchor_train(ds: FolsomIrradianceDataset, anchor: int) -> None:
    amin = int(ds._anchors[0])
    amax = int(ds._anchors[-1])
    if not (amin <= anchor <= amax):
        raise ValueError(
            f"anchor_row={anchor} is outside the valid anchor range [{amin}, {amax}] "
            f"(needs room for input length {ds.pv_input_len} and output length {ds.pv_output_len})."
        )
    r = anchor - amin
    if not bool(ds._train_anchor_mask[r]):
        raise ValueError(
            f"anchor_row={anchor} falls outside the train time band "
            f"(train_split={ds._train_split}, val_split={ds._val_split}, test_split={ds._test_split}). "
            "Smoke uses the train dataset only: pick an earlier calendar time."
        )


def _smoke_nwp_lines(times: list[str], nwp: torch.Tensor) -> list[str]:
    nwp_np = nwp.detach().cpu().numpy()
    t_n = int(nwp_np.shape[0])
    lines = [
        f"nwp_tensor shape: {tuple(nwp.shape)}  [T_out, features + per-step invalid mask]",
    ]
    if t_n == 0:
        lines.append("  (empty NWP tensor)")
        return lines
    for i in range(min(2, t_n)):
        lines.append(f"  first[{i}]  t={times[i]}  row={np.round(nwp_np[i], 4).tolist()}")
    if t_n > 2:
        for i in (t_n - 2, t_n - 1):
            lines.append(f"  last[{i}]  t={times[i]}  row={np.round(nwp_np[i], 4).tolist()}")
    return lines


def run_smoke_cli(argv: list[str] | None = None) -> int:
    """
    CLI entry: one deterministic train anchor + titled smoke report + one collated batch.

    Prints anchor/time windows, irradiance I/O (first/last steps), sky stats, NWP rows, then
    batched tensor shapes (train ``__getitem__`` uses random anchors, so batch rows differ).

    Loads ``--conf`` (default ``config/datasets/conf_folsom.yaml``): CSV, sky JPEG dir, and NWP
    paths come from that YAML; lat/lon comes from ``<paths.data_dir>/info.yaml``.

    Optional ``--last-input-time`` sets the anchor (last input CSV row time); default is first valid train anchor.
    """
    p = argparse.ArgumentParser(description="Smoke-test Folsom DataLoader (reads paths from YAML)")
    p.add_argument(
        "--conf",
        "--config",
        type=Path,
        default=_DEFAULT_FOLSOM_DATASET_CONFIG,
        dest="conf",
        help="Path to dataset YAML (paths.data_dir + pv/sky/sat, NWP).",
    )
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--train-epoch-len", type=int, default=8, help="Dataset __len__ for train split smoke.")
    p.add_argument(
        "--skyimg-window",
        type=int,
        default=None,
        metavar="N",
        help="Override training.skyimg_window_size (last N sky frames ending at last input row).",
    )
    p.add_argument(
        "--last-input-time",
        type=str,
        default=None,
        metavar="T",
        help="Anchor: time on the last input CSV row, e.g. 2014-01-04 07:59:00 (must be in train split).",
    )
    args = p.parse_args(argv)

    try:
        conf = load_folsom_conf(args.conf)
        train_ds, test_ds = build_folsom_irradiance_datasets_from_conf(
            conf,
            conf_path=args.conf,
            train_epoch_len=args.train_epoch_len,
            skyimg_window_size=args.skyimg_window,
        )
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Failed to load Folsom data from {args.conf.resolve()}:\n  {e}", file=sys.stderr)
        print(
            "Fix paths.data_dir, paths.pv_path (folder with the irradiance CSV), "
            "paths.sky_image_path, paths.sat_path (placeholder), paths.nwp_path (folder with the "
            "merged NWP CSV) in that YAML, and ensure <data_dir>/info.yaml provides site.latitude / "
            "site.longitude so files exist on disk.",
            file=sys.stderr,
        )
        return 1
    ds = train_ds
    print(f"[conf] {args.conf.resolve()}")
    print(f"  csv={ds._csv_path}  train_len={len(ds)}  test_len={len(test_ds)}")

    if args.last_input_time:
        anchor0 = _find_csv_data_row_index_for_time(
            ds._df,
            ds._time_col,
            pd.to_datetime(args.last_input_time),
        )
    else:
        r0 = int(ds._train_anchor_positions[0])
        anchor0 = int(ds._anchors[r0])

    try:
        _validate_smoke_anchor_train(ds, anchor0)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    sky = ds.sky_inspect(anchor0)
    sample = ds._build_tensors(anchor0)

    tin: list[str] = sample["input_timestamps_utc"]
    tout: list[str] = sample["forecast_timestamps_utc"]
    g, dn, dh = sample["ghi"], sample["dni"], sample["dhi"]
    tg, tdn, tdh = sample["target_ghi"], sample["target_dni"], sample["target_dhi"]
    sk = sample["skimg_tensor"]
    sk_ts: list = sample["skimg_timestamps"]
    nwp = sample["nwp_tensor"]

    im = sample["input_mask"]
    inv = float(im.sum())
    im_tot = int(im.numel())
    tm = sample["target_mask"]
    tnv = float(tm.sum())
    tm_tot = int(tm.numel())

    sky_mean = float(sk.float().mean())
    n_none = sum(1 for x in sk_ts if x is None)

    _smoke_section("ANCHOR DATE AND TIME", [
        f"anchor_row (last input CSV row index): {anchor0}",
        f"input window:      {tin[0]}  →  {tin[-1]}",
        f"forecast window:   {tout[0]}  →  {tout[-1]}",
        f"skyimg_dir={sky['skyimg_dir']}",
        (
            f"zarr: slice rows in [oldest,nominal-newest] time range ≈ {sky['n_files_found']} / "
            f"nominal_slots={sky['n_frames']}  (paths.sky_format=zarr)"
            if sky.get("sky_format") == "zarr"
            else f"jpeg_files_found (on disk vs window slots): {sky['n_files_found']}/{sky['n_frames']}"
        ),
    ])

    in_lines = _smoke_irradiance_lines(tin, g, dn, dh)
    in_lines.append(f"input_mask valid values: {inv:.0f} / {im_tot}")
    _smoke_section("IRRADIANCE INPUT (GHI, DNI, DHI)", in_lines)

    out_lines = _smoke_irradiance_lines(tout, tg, tdn, tdh)
    out_lines.append(f"target_mask valid values: {tnv:.0f} / {tm_tot}")
    _smoke_section("IRRADIANCE OUTPUT / TARGETS (GHI, DNI, DHI)", out_lines)

    sky_lines = [
        f"skimg_tensor shape: {tuple(sk.shape)}  (N, C, H, W)",
        f"mean pixel (≈0 if all black): {sky_mean:.6f}",
        f"skimg_timefeats shape: {tuple(sample['skimg_timefeats'].shape)}",
        f"sky JPEG stems (None = black / pad):  none_count={n_none}",
    ]
    sky_lines.extend(_smoke_sky_stem_lines(sk_ts))
    _smoke_section("SKY IMAGE", sky_lines)

    _smoke_section("NWP (interpolated at forecast times)", _smoke_nwp_lines(tout, nwp))

    bs = max(1, min(int(args.batch_size), len(ds)))
    np.random.seed(0)
    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        collate_fn=collate_folsom_irradiance,
        num_workers=0,
    )
    batch = next(iter(loader))
    batch_lines = [
        f"batch_size={bs}  (train split randomizes anchor each __getitem__; shapes only here)",
    ]
    for k in (
        "ghi",
        "irr_timefeats",
        "forecast_timefeats",
        "target_ghi",
        "skimg_tensor",
        "skimg_timefeats",
        "nwp_tensor",
    ):
        v = batch[k]
        batch_lines.append(f"{k}: shape={tuple(v.shape)} dtype={v.dtype}")
    _smoke_section("DATALOADER (first batch)", batch_lines)

    print("smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_smoke_cli())
