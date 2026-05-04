"""
Folsom dataset: one CSV with time + GHI, DNI, DHI (and optional header aliases).

Designed for multi-million-row files: never loads the full table into RAM.
Each sample reads one contiguous row block (length set by ``pv_input_len`` / ``pv_output_len`` —
:class:`FolsomIrradianceDataset` mirrors :class:`dataloader.luoyang_mem.PVDataset`'s constructor).

Horizon math matches Luoyang anchor conventions (anchor = last input row index).

Training usage (same two-step pattern as ``dataloader.luoyang``):

1. **Sample** — :meth:`FolsomIrradianceDataset.__getitem__` → :meth:`FolsomIrradianceDataset._build_tensors`
   returns one ``dict`` per index.
2. **Batch** — :func:`collate_folsom_irradiance` stacks a ``list`` of those dicts; every key in
   :data:`FOLSOM_BATCH_TENSOR_KEYS` gains a leading batch dimension ``B``.

**Batched tensor keys** (after ``collate_folsom_irradiance``; shapes use ``T_in`` = ``pv_input_len``,
``T_out`` = ``pv_output_len``, ``T_sky`` = ``skyimg_window_size``, ``C_nwp`` = NWP feature count + 1 mask channel):

- ``ghi``, ``dni``, ``dhi``: ``[B, T_in]``
- ``input_mask``: ``[B, 1, T_in]`` (valid input timesteps; leading ``1`` matches Luoyang-style mask layout)
- ``irr_timefeats``: ``[B, T_in, 9]`` (solar + delta-time encoding on input window)
- ``forecast_timefeats``: ``[B, T_out, 9]`` (same on forecast timesteps)
- ``target_ghi``, ``target_dni``, ``target_dhi``: ``[B, T_out]``
- ``target_mask``: ``[B, T_out]``
- ``skimg_tensor``: ``[B, T_sky, 3, H, W]`` with ``H=W=skyimg_spatial_size``
- ``skimg_timefeats``: ``[B, T_sky, feat_dim]``
- ``nwp_tensor``: ``[B, T_out, C_nwp]`` (zeros + invalid mask if NWP file missing)

**Not stacked by collate** (debug / metadata; lists length ``B`` of per-sample lists):

- ``skimg_timestamps``, ``input_timestamps_utc``, ``forecast_timestamps_utc``

Optional keys ``skimg_tensor``, ``skimg_timefeats``, ``nwp_tensor`` may be stacked as ``None`` if a future
sample path omits them — same guard pattern as :func:`dataloader.luoyang.collate_batched`.

JPEG stems ``YYYYMMDDHHMMSS.jpg`` use **UTC**; naive CSV times are read as **UTC**.
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
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config_utils import get_resolved_paths
from dataloader.luoyang_mem import list_csv_files
from modules.solar_encoder import compute_solar_features, delta_time_encoder, solar_features_encoder

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


_DEFAULT_FOLSOM_TRAIN_EPOCH_LEN = 50_000

# Train mode: a Y window is "valid" if any of its rows has finite GHI strictly above this
# threshold (W/m^2). Mirrors PVDataset's "any inverter_state == VALID_STATE" filter so we
# avoid sampling all-night windows where target_pv is uniformly 0.
_FOLSOM_TRAIN_GHI_DAYTIME_THRESHOLD = 10.0


def _folsom_progress(msg: str) -> None:
    """Progress to stderr so training stdout stays clean; set FOLSOM_QUIET=1 to disable."""
    if os.environ.get("FOLSOM_QUIET", "").strip().lower() in ("1", "true", "yes"):
        return
    print(f"[Folsom] {msg}", file=sys.stderr, flush=True)


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
    """Resolve ``paths.folsom_nwp_merged_csv`` from config."""
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
    if rel is None or not str(rel).strip():
        raise KeyError("conf paths.folsom_nwp_merged_csv is required for Folsom")
    rel_p = Path(str(rel).strip())
    p = rel_p.resolve() if rel_p.is_absolute() else (data_dir / rel_p).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Folsom NWP merged CSV not found: {p}")
    return p


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
    ``paths.folsom_nwp_merged_csv`` in the per-instance ``config_path``; site coordinates come
    from ``<paths.data_dir>/info.yaml`` (``site.latitude`` / ``site.longitude``), matching
    :class:`dataloader.luoyang_mem.PVDataset`.

    Splits: rows are partitioned in fixed proportions ``60% train / 10% val / 30% test`` (same as
    PVDataset). ``pv_train_time_fraction`` is kept for call-site parity but **not** used. Train
    samples a random valid anchor per ``__getitem__`` (epoch length defaults to
    ``_DEFAULT_FOLSOM_TRAIN_EPOCH_LEN``; settable via ``self._train_epoch_len``); val/test use the
    respective ``*_anchor_stride_min`` strides over their bands.

    ``skyimg_window_size`` is the count of sky JPEGs ending at the last input timestep (anchor),
    spaced by ``skyimg_time_resolution_min`` (oldest first in ``skimg_tensor``). All timestamps
    are UTC.
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

        if skyimg_time_resolution_min <= 0:
            raise ValueError("skyimg_time_resolution_min must be positive")
        self._skyimg_dt_min = int(skyimg_time_resolution_min)
        self._skyimg_dir = Path(skyimg_dir).resolve()
        if skyimg_spatial_size < 1:
            raise ValueError("skyimg_spatial_size must be >= 1")
        self._skyimg_spatial_size = int(skyimg_spatial_size)

        # Sat config: accepted for API parity with PVDataset; Folsom has no satellite data.
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

        # Site + NWP from the per-instance dataset YAML (``self._config_path``):
        #   * lat/lon → ``<paths.data_dir>/info.yaml`` (site.latitude / site.longitude),
        #     matching :class:`dataloader.luoyang_mem.PVDataset`.
        #   * NWP merged CSV → ``paths.folsom_nwp_merged_csv`` (relative to ``data_dir``
        #     unless absolute); missing/unreadable falls back to None (zero NWP at runtime).
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

        self._nwp_feature_cols = tuple(_FOLSOM_NWP_FEATURE_COLS)
        nwp_rel = paths_cfg.get("folsom_nwp_merged_csv")
        self._nwp_merged_df = None
        if nwp_rel is not None and str(nwp_rel).strip() != "":
            try:
                nwp_csv = _resolve_folsom_nwp_csv_path(conf)
                self._nwp_merged_df = _load_folsom_nwp_merged_csv(nwp_csv)
            except (FileNotFoundError, KeyError):
                self._nwp_merged_df = None

        # API parity with PVDataset: trainer reads ``train_dataset.devDn_list`` to size the
        # device-id embedding. Folsom is a single-sensor station, so a length-1 list is fine
        # (paired with ``dev_idx=0`` returned by ``_build_tensors``).
        self.devDn_list = [0]

        # CSV: glob ``pv_dir`` for *.csv (PVDataset convention); Folsom expects exactly one.
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

        # Sky index (cached across instances sharing skyimg_dir).
        cache_key = str(self._skyimg_dir)
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
        self._sky_gap_threshold = pd.Timedelta(minutes=5)
        self._sky_anchor_max_lag = pd.Timedelta(minutes=5)

        # CSV header & row count.
        header_line = _read_header_line(self._csv_path)
        reader = csv.reader([header_line])
        header_cells = next(reader)
        self._time_col, _order, self._file_columns = _pick_time_and_ghi_dni_dhi_columns(header_cells)
        self._ghi_dni_dhi_cols = _order[1:]
        _folsom_progress(
            f"counting lines in irradiance CSV {self._csv_path.name} (large files can take a bit) ..."
        )
        total_lines = _count_newlines(self._csv_path)
        _folsom_progress(f"irradiance CSV: {total_lines - 1:,} data rows (+ header)")
        if total_lines < 2:
            raise RuntimeError(f"{self._csv_path.name}: expected header + at least one data row")
        self._n = int(total_lines - 1)

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

        # Fixed 60% / 10% / 30% train/val/test split (matches PVDataset).
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
        self._train_epoch_len = _DEFAULT_FOLSOM_TRAIN_EPOCH_LEN
        # Contiguous data rows from first X row through last Y row (inclusive).
        self._block_nrows = int((lx - 1) * sx + ly * sy + 1)

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
        # Read just the GHI column (full file). Use str + to_numeric to tolerate non-numeric tokens.
        ghi_series = pd.read_csv(
            self._csv_path,
            usecols=[ghi_col],
            engine="c",
            memory_map=True,
            dtype={ghi_col: str},
        )[ghi_col]
        ghi_full = pd.to_numeric(ghi_series, errors="coerce").to_numpy(dtype=np.float32)
        if ghi_full.size != self._n:
            raise RuntimeError(
                f"{self._csv_path.name}: expected {self._n} GHI rows, got {ghi_full.size}"
            )
        # Replace NaN/inf with 0 so the threshold check excludes them.
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

    def _load_sky_tensor(self, path: Path) -> torch.Tensor:
        """Return ``[3, s, s]`` float32 in ``[0, 1]`` (matches PVDataset)."""
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

    def _read_block(self, first_data_row: int) -> pd.DataFrame:
        """Read ``self._block_nrows`` data rows starting at data-row index ``first_data_row``."""
        skiprows = int(first_data_row + 1)
        return pd.read_csv(
            self._csv_path,
            skiprows=skiprows,
            nrows=self._block_nrows,
            header=None,
            names=self._file_columns,
            dtype=str,
            engine="c",
            memory_map=True,
        )

    def __len__(self) -> int:
        if self.split == "train":
            return self._train_epoch_len
        if self.split == "val":
            return self._num_val_windows
        return self._num_test_windows

    def sky_inspect(self, anchor: int) -> dict:
        """
        Resolve last-N sky JPEG records for ``anchor`` without building full tensors.

        Returns keys: ``anchor_row``, ``last_input_time_utc_naive``, ``n_frames``,
        ``utc_times``, ``paths``, ``n_files_found``, ``skyimg_dir``.
        """
        first = int(anchor - (self._lx - 1) * self._sx)
        block = self._read_block(first)
        if len(block) < self._block_nrows:
            raise RuntimeError("short read in sky_inspect")
        x_rel = anchor + self._x_tail_1d - first
        sub_x = block.iloc[x_rel]
        t_x_end = sub_x[self._time_col].iloc[-1]
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
        }

    def _build_tensors(self, anchor: int) -> dict[str, Any]:
        first = int(anchor - (self._lx - 1) * self._sx)
        block = self._read_block(first)
        if len(block) < self._block_nrows:
            raise RuntimeError(
                f"{self._csv_path.name}: short read at anchor={anchor} "
                f"(got {len(block)} rows, need {self._block_nrows}); file truncated?"
            )

        # iloc into ``block``: absolute data row index minus ``first`` (block row 0 = ``first``).
        # ``_x_tail_1d`` / ``_y_off_1d`` are offsets from ``anchor``, not absolute indices.
        x_rel = anchor + self._x_tail_1d - first
        y_rel = anchor + self._y_off_1d - first
        sub_x = block.iloc[x_rel]
        sub_y = block.iloc[y_rel]

        for name in self._ghi_dni_dhi_cols:
            if name not in sub_x.columns:
                raise KeyError(f"missing column {name!r}")

        gx = pd.to_numeric(sub_x[self._ghi_dni_dhi_cols[0]], errors="coerce")
        dx = pd.to_numeric(sub_x[self._ghi_dni_dhi_cols[1]], errors="coerce")
        hx = pd.to_numeric(sub_x[self._ghi_dni_dhi_cols[2]], errors="coerce")
        x_stack = np.stack([gx.to_numpy(), dx.to_numpy(), hx.to_numpy()], axis=0).astype(np.float32)
        x_stack = np.nan_to_num(x_stack, nan=0.0, posinf=0.0, neginf=0.0)
        valid_in = np.isfinite(np.stack([gx.to_numpy(), dx.to_numpy(), hx.to_numpy()], axis=0)).all(axis=0)
        input_mask = torch.from_numpy(valid_in.astype(np.float32)).unsqueeze(0)

        gy = pd.to_numeric(sub_y[self._ghi_dni_dhi_cols[0]], errors="coerce")
        dy = pd.to_numeric(sub_y[self._ghi_dni_dhi_cols[1]], errors="coerce")
        hy = pd.to_numeric(sub_y[self._ghi_dni_dhi_cols[2]], errors="coerce")
        y_raw = np.stack([gy.to_numpy(), dy.to_numpy(), hy.to_numpy()], axis=0).astype(np.float32)
        valid_out = np.isfinite(y_raw).all(axis=0)
        target_mask = torch.from_numpy(valid_out.astype(np.float32))
        y_stack = np.nan_to_num(y_raw, nan=0.0, posinf=0.0, neginf=0.0)

        ghi, dni, dhi = x_stack[0], x_stack[1], x_stack[2]
        tg, td, th = y_stack[0], y_stack[1], y_stack[2]

        # Explicit format keeps __getitem__ on pandas's vectorized C path (instead of the
        # per-element ``dateutil`` fallback, which warns and is ~50-100x slower).
        x_times = pd.to_datetime(
            sub_x[self._time_col], format="%Y-%m-%d %H:%M:%S", errors="coerce"
        )
        if bool(x_times.isna().any()):
            raise ValueError(f"NaT in {self._time_col!r} for input window")
        timestamps = _folsom_to_timestamps(list(x_times))
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

        t_x_end = sub_x[self._time_col].iloc[-1]
        skimg_timestamps, skimg_paths = self._history_sky_frame_records(t_x_end)
        skimg_solar_features = compute_solar_features(
            skimg_timestamps, self.latitude, self.longitude
        )
        skimg_tf = solar_features_encoder(skimg_solar_features)
        skimg_dtf = delta_time_encoder(skimg_timestamps, time0)
        skimg_timefeats = torch.cat([skimg_tf, skimg_dtf.unsqueeze(1)], dim=1)
        skimg_tensor = self._stack_sky_frames(skimg_paths)
        skimg_timestamps = [
            (None if p is None else pd.Timestamp(t).strftime("%Y%m%d%H%M%S"))
            for t, p in zip(skimg_timestamps, skimg_paths)
        ]

        input_timestamps_utc = [str(pd.Timestamp(t)) for t in timestamps]
        forecast_timestamps_utc = [str(pd.Timestamp(t)) for t in forecast_timestamps]
        # Single-sensor station; index into ``self.devDn_list = [0]``.
        dev_idx = torch.tensor(700, dtype=torch.long)

        # Match PVDataset: pv is [1, T_in] (sensor/dev dim leading), target_pv is [T_out].
        pv_tensor = torch.from_numpy((ghi / 1100.0).astype(np.float32)).unsqueeze(0)
        target_pv_tensor = torch.from_numpy((tg / 1100.0).astype(np.float32))
        return {
            "dev_idx": dev_idx,
            "pv": pv_tensor,
            "pv_mask": input_mask,
            "pv_timefeats": irr_timefeats,
            "forecast_timefeats": forecast_timefeats,
            "target_pv": target_pv_tensor,
            "target_mask": target_mask,
            "sat_tensor": None,
            "sat_timefeats": None,
            "skimg_tensor": skimg_tensor,
            "skimg_timefeats": skimg_timefeats,
            "nwp_tensor": nwp_tensor,
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
    train_epoch_len: int = 50_000,
    skyimg_window_size: int | None = None,
) -> tuple[FolsomIrradianceDataset, FolsomIrradianceDataset]:
    """
    Build train/test :class:`FolsomIrradianceDataset` from a Folsom dataset YAML (new schema:
    ``config/datasets/conf_folsom.yaml``).

    Reads ``paths.{data_dir, pv_path, sky_image_path, sat_path}`` and the ``sampling:`` section
    (PVDataset-style window / stride / image-shape fields). Lat/lon comes from
    ``<paths.data_dir>/info.yaml`` and the optional NWP merged CSV from
    ``paths.folsom_nwp_merged_csv`` — both read inside the dataset constructor.

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
        pv_train_time_fraction=float(_req_s("pv_train_time_fraction")),
        test_anchor_stride_min=int(_req_s("test_anchor_stride_min")),
        val_anchor_stride_min=int(_req_s("val_anchor_stride_min")),
        test_collect_time_match_tolerance_min=int(_req_s("test_collect_time_match_tolerance_min")),
        skyimg_window_size=sky_w,
        skyimg_time_resolution_min=int(_req_s("skyimg_time_resolution_min")),
        skyimg_spatial_size=int(_req_s("skyimg_spatial_size")),
        satimg_window_size=int(_req_s("satimg_window_size")),
        satimg_time_resolution_min=int(_req_s("satimg_time_resolution_min")),
        satimg_npy_shape_hwc=tuple(int(x) for x in shwc),
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
    csv_path: Path,
    time_col: str,
    target: pd.Timestamp,
) -> int:
    """
    Scan irradiance CSV for the first data row whose ``time_col`` **exactly** matches ``target`` after parsing.

    Returns **0-based data row index** (row 0 = first line after the header), matching ``anchor_row`` in the dataset.
    """
    tgt = pd.Timestamp(target)
    if tgt.tzinfo is not None:
        tgt = tgt.tz_convert("UTC").tz_localize(None)
    offset = 0
    for chunk in pd.read_csv(
        csv_path,
        usecols=[time_col],
        chunksize=300_000,
        dtype={time_col: str},
        engine="c",
        memory_map=True,
    ):
        ts = pd.to_datetime(chunk[time_col], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        if getattr(ts.dt, "tz", None) is not None:
            ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
        ok = ts == tgt
        if ok.any():
            pos = int(np.flatnonzero(ok.to_numpy())[0])
            return offset + pos
        offset += len(chunk)
    raise ValueError(
        f"No CSV row with {time_col!r} exactly equal to {tgt} after parsing. "
        "Use the same string as in the file (minute-aligned for 1-minute CSV)."
    )


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
            f"anchor_row={anchor} falls outside the train time band (fixed 60% train / 10% val / 30% test). "
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
        type=Path,
        default=_DEFAULT_FOLSOM_DATASET_CONFIG,
        help="Path to YAML (paths.data_dir + pv/sky/sat, NWP).",
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
            "paths.sky_image_path, paths.sat_path (placeholder), paths.folsom_nwp_merged_csv "
            "in that YAML, and ensure <data_dir>/info.yaml provides site.latitude / "
            "site.longitude so files exist on disk.",
            file=sys.stderr,
        )
        return 1
    ds = train_ds
    print(f"[conf] {args.conf.resolve()}")
    print(f"  csv={ds._csv_path}  train_len={len(ds)}  test_len={len(test_ds)}")

    if args.last_input_time:
        anchor0 = _find_csv_data_row_index_for_time(
            ds._csv_path,
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
        f"jpeg_files_found (on disk vs window slots): {sky['n_files_found']}/{sky['n_frames']}",
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
