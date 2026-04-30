"""
Folsom dataset: one CSV with time + GHI, DNI, DHI (and optional header aliases).

Designed for multi-million-row files: never loads the full table into RAM.
Each sample reads one contiguous row block (length set by ``irr_input_len`` / ``irr_output_len``).

Horizon math matches Luoyang anchor conventions (anchor = last input row index).

Training usage (same two-step pattern as ``dataloader.luoyang``):

1. **Sample** — :meth:`FolsomIrradianceDataset.__getitem__` → :meth:`FolsomIrradianceDataset._build_tensors`
   returns one ``dict`` per index.
2. **Batch** — :func:`collate_folsom_irradiance` stacks a ``list`` of those dicts; every key in
   :data:`FOLSOM_BATCH_TENSOR_KEYS` gains a leading batch dimension ``B``.

**Batched tensor keys** (after ``collate_folsom_irradiance``; shapes use ``T_in`` = ``irr_input_len``,
``T_out`` = ``irr_output_len``, ``T_sky`` = ``skyimg_window_size``, ``C_nwp`` = NWP feature count + 1 mask channel):

- ``dev_idx``: ``[B]`` (constant 0 for single-station Folsom)
- ``pv``: ``[B, 1, T_in]`` (mapped from input GHI)
- ``pv_mask``: ``[B, 1, T_in]`` (finite/valid input GHI timesteps)
- ``pv_timefeats``: ``[B, T_in, 9]`` (solar + delta-time encoding on input window)
- ``forecast_timefeats``: ``[B, T_out, 9]`` (same on forecast timesteps)
- ``target_pv``: ``[B, T_out]`` (mapped from target GHI)
- ``target_mask``: ``[B, T_out]``
- ``sat_tensor``, ``sat_timefeats``: ``None`` (kept for Luoyang contract parity)
- ``skimg_tensor``: ``[B, T_sky, 3, H, W]`` with ``H=W=skyimg_spatial_size``
- ``skimg_timefeats``: ``[B, T_sky, feat_dim]``
- ``nwp_tensor``: ``[B, T_out, C_nwp]`` (zeros + invalid mask if NWP file missing)

Optional keys ``skimg_tensor``, ``skimg_timefeats``, ``nwp_tensor`` may be stacked as ``None`` if a future
sample path omits them — same guard pattern as :func:`dataloader.luoyang.collate_batched`.

JPEG stems ``YYYYMMDDHHMMSS.jpg`` use **UTC**; naive CSV times are read as **UTC**.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from modules.solar_encoder import compute_solar_features, delta_time_encoder, solar_features_encoder
from training.training_conf import get_training_paths_from_conf

# Keys collate stacks with batch dim B first (Luoyang-compatible contract).
FOLSOM_GHI_DNI_DHI_KEYS: tuple[str, ...] = (
    "dev_idx",
    "pv",
    "pv_mask",
    "pv_timefeats",
    "forecast_timefeats",
    "target_pv",
    "target_mask",
    "sat_tensor",
    "sat_timefeats",
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


def load_folsom_conf(path: Path | str | None = None) -> dict:
    """Load ``config/conf_folsom.yaml`` (or a custom path)."""
    p = Path(path) if path is not None else _PROJECT_ROOT / "config" / "conf_folsom.yaml"
    with p.open() as f:
        return yaml.safe_load(f)


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
    out["reftime"] = pd.to_datetime(df["reftime"], errors="coerce")
    out["valtime"] = pd.to_datetime(df["valtime"], errors="coerce")
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
    Single CSV with time + GHI, DNI, DHI columns (names detected from header).

    - **Train** (``split="train"``): ``__len__`` = ``train_epoch_len``; each item samples a
      random valid train anchor (same CSV row can repeat across epochs).
    - **Test** (``split="test"``): ``__len__`` = number of strided test anchors; deterministic.

    Parameters mirror ``conf_folsom.yaml`` / ``training`` plus ``site`` (lat/lon) and ``paths.skyimg_dir``
    from :func:`training.training_conf.get_training_paths_from_conf`. All times are **UTC**.

    ``skyimg_window_size`` is the count of sky frames **ending at the last input timestep** (anchor),
    spaced by ``skyimg_time_resolution_min`` (oldest first in ``skimg_tensor``).

    For the training batch contract (stacked keys, shapes), see :data:`FOLSOM_BATCH_TENSOR_KEYS` and
    :func:`collate_folsom_irradiance`.
    """

    def __init__(
        self,
        csv_path: Path | str,
        *,
        split: Literal["train", "test"],
        csv_interval_min: int,
        irr_input_interval_min: int,
        irr_input_len: int,
        irr_output_interval_min: int,
        irr_output_len: int,
        irr_train_time_fraction: float,
        test_anchor_stride_min: int,
        train_epoch_len: int,
        skyimg_dir: Path | str,
        skyimg_window_size: int,
        skyimg_time_resolution_min: int,
        skyimg_spatial_size: int,
        nwp_merged_csv_path: Path | str | None = None,
        latitude: float,
        longitude: float,
    ):
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        self.split = split
        self._csv_path = Path(csv_path).resolve()
        if not self._csv_path.is_file():
            raise FileNotFoundError(str(self._csv_path))
        _folsom_progress(f"dataset split={split!r}: preparing {self._csv_path.name} ...")

        self._skyimg_dir = Path(skyimg_dir).resolve()
        if skyimg_window_size < 1:
            raise ValueError("skyimg_window_size must be >= 1")
        self.skyimg_window_size = int(skyimg_window_size)
        self._skyimg_dt_min = int(skyimg_time_resolution_min)
        if self._skyimg_dt_min <= 0:
            raise ValueError("skyimg_time_resolution_min must be positive")
        ss = int(skyimg_spatial_size)
        if ss < 1:
            raise ValueError("skyimg_spatial_size must be >= 1")
        self._skyimg_spatial_size = ss
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
        # If consecutive sky frames are farther apart than this, treat older frames as invalid history.
        self._sky_gap_threshold = pd.Timedelta(minutes=5)
        # Newest sky frame must be within this lag of anchor (last input time); else whole window is black.
        self._sky_anchor_max_lag = pd.Timedelta(minutes=5)
        self._nwp_feature_cols = tuple(_FOLSOM_NWP_FEATURE_COLS)
        if nwp_merged_csv_path is None:
            self._nwp_merged_df = None
        else:
            self._nwp_merged_df = _load_folsom_nwp_merged_csv(nwp_merged_csv_path)
        self.latitude = float(latitude)
        self.longitude = float(longitude)
        self.irr_output_interval_min = int(irr_output_interval_min)

        if csv_interval_min <= 0 or irr_input_interval_min % csv_interval_min:
            raise ValueError("irr_input_interval_min must be a positive multiple of csv_interval_min")
        if irr_output_interval_min % csv_interval_min:
            raise ValueError("irr_output_interval_min must be a positive multiple of csv_interval_min")
        self._sx = irr_input_interval_min // csv_interval_min
        self._sy = irr_output_interval_min // csv_interval_min
        self.irr_input_len = int(irr_input_len)
        self.irr_output_len = int(irr_output_len)
        self._lx = self.irr_input_len
        self._ly = self.irr_output_len

        tf = float(irr_train_time_fraction)
        if not (0.0 < tf < 1.0):
            raise ValueError("irr_train_time_fraction must be strictly between 0 and 1")
        self._irr_train_time_fraction = tf

        if test_anchor_stride_min <= 0 or test_anchor_stride_min % csv_interval_min:
            raise ValueError(
                "test_anchor_stride_min must be a positive multiple of csv_interval_min "
                f"(got {test_anchor_stride_min}, csv_interval_min={csv_interval_min})"
            )
        self._test_anchor_stride_rows = test_anchor_stride_min // csv_interval_min

        header_line = _read_header_line(self._csv_path)
        reader = csv.reader([header_line])
        header_cells = next(reader)
        self._time_col, _order, self._file_columns = _pick_time_and_ghi_dni_dhi_columns(header_cells)
        self._ghi_dni_dhi_cols = _order[1:]  # file header names for ghi, dni, dhi in that order

        _folsom_progress(f"counting lines in irradiance CSV {self._csv_path.name} (large files can take a bit) ...")
        total_lines = _count_newlines(self._csv_path)
        _folsom_progress(f"irradiance CSV: {total_lines - 1:,} data rows (+ header)")
        if total_lines < 2:
            raise RuntimeError(f"{self._csv_path.name}: expected header + at least one data row")
        self._n = int(total_lines - 1)

        lx, ly = self._lx, self._ly
        sx, sy = self._sx, self._sy
        amin = (lx - 1) * sx
        y_last_off = sy * ly
        amax = self._n - 1 - y_last_off
        if self._n == 0 or amin > amax:
            raise RuntimeError(
                f"{self._csv_path.name}: no anchor fits bounds "
                f"(n={self._n}, need {amin}<=anchor<={amax}); check row count and window lengths"
            )
        self._anchors = np.arange(amin, amax + 1, dtype=np.intp)
        self._x_tail_1d = (-(lx - 1) * sx + np.arange(lx, dtype=np.intp) * sx).astype(np.intp, copy=False)
        self._y_off_1d = (sy + np.arange(ly, dtype=np.intp) * sy).astype(np.intp, copy=False)

        split_idx = int(self._n * self._irr_train_time_fraction)
        if split_idx <= 0 or split_idx >= self._n:
            raise ValueError(
                f"irr_train_time_fraction={self._irr_train_time_fraction} gives split_idx={split_idx} for n={self._n}"
            )
        min_row = self._anchors - (lx - 1) * sx
        max_row = self._anchors + ly * sy
        self._train_anchor_mask = max_row < split_idx
        self._test_anchor_mask = min_row >= split_idx
        if self.split == "train" and not bool(self._train_anchor_mask.any()):
            raise RuntimeError("split=train: no valid train anchors after time split")
        if self.split == "test" and not bool(self._test_anchor_mask.any()):
            raise RuntimeError("split=test: no valid test anchors after time split")

        train_positions = np.nonzero(self._train_anchor_mask)[0]
        self._train_anchor_positions = train_positions.astype(np.intp, copy=False)
        self._num_train_anchors = int(train_positions.size)

        test_positions = np.nonzero(self._test_anchor_mask)[0]
        stride = self._test_anchor_stride_rows
        self._test_r_indices = test_positions[::stride].astype(np.intp, copy=False)
        self._num_test_windows = int(self._test_r_indices.size)
        if self.split == "test" and self._num_test_windows == 0:
            raise RuntimeError("split=test: no test windows after stride; reduce test_anchor_stride_min")

        self._train_epoch_len = max(1, int(train_epoch_len))
        # Contiguous data rows from first X row through last Y row (inclusive).
        self._block_nrows = int((lx - 1) * sx + ly * sy + 1)

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
        s = self._skyimg_spatial_size
        return torch.zeros((3, s, s), dtype=torch.uint8)

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
                    arr = np.asarray(im, dtype=np.uint8).copy()
                return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
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
        frames: list[torch.Tensor] = []
        to_load = sum(1 for x in frame_paths if x is not None)
        s = self._skyimg_spatial_size
        if to_load > 0:
            _folsom_progress(
                f"loading sky window: {to_load} JPEG(s) from disk (resize to {s}x{s}) ..."
            )
        load_i = 0
        for slot, p in enumerate(frame_paths):
            if p is None:
                frames.append(self._black_sky_tensor())
            else:
                load_i += 1
                if load_i == 1 or load_i == to_load:
                    _folsom_progress(f"  sky JPEG {load_i}/{to_load} (slot {slot}) ...")
                elif to_load >= 10:
                    step = max(2, to_load // 5)
                    if load_i % step == 0:
                        _folsom_progress(f"  sky JPEG {load_i}/{to_load} (slot {slot}) ...")
                frames.append(self._load_sky_tensor(p))
        if to_load > 0:
            _folsom_progress("sky window stack done")
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

        # Luoyang-compatible model contract uses one signal (`pv`) + masks.
        # For Folsom we map that signal to GHI only.
        gx = pd.to_numeric(sub_x[self._ghi_dni_dhi_cols[0]], errors="coerce")
        ghi_in_raw = gx.to_numpy(dtype=np.float32)
        valid_in = np.isfinite(ghi_in_raw)
        ghi_in = np.nan_to_num(ghi_in_raw, nan=0.0, posinf=0.0, neginf=0.0)
        pv = torch.from_numpy(ghi_in).unsqueeze(0)
        pv_mask = torch.from_numpy(valid_in.astype(np.float32)).unsqueeze(0)

        gy = pd.to_numeric(sub_y[self._ghi_dni_dhi_cols[0]], errors="coerce")
        ghi_out_raw = gy.to_numpy(dtype=np.float32)
        valid_out = np.isfinite(ghi_out_raw)
        ghi_out = np.nan_to_num(ghi_out_raw, nan=0.0, posinf=0.0, neginf=0.0)
        target_pv = torch.from_numpy(ghi_out)
        target_mask = torch.from_numpy(valid_out.astype(np.float32))

        x_times = pd.to_datetime(sub_x[self._time_col], errors="coerce")
        if bool(x_times.isna().any()):
            raise ValueError(f"NaT in {self._time_col!r} for input window")
        timestamps = _folsom_to_timestamps(list(x_times))
        time0 = timestamps[-1]
        forecast_timestamps = [
            time0 + pd.Timedelta(minutes=self.irr_output_interval_min * (i + 1))
            for i in range(self.irr_output_len)
        ]
        nwp_tensor = self._interpolate_nwp(forecast_timestamps)

        irr_solar = compute_solar_features(timestamps, self.latitude, self.longitude)
        irr_tf = solar_features_encoder(irr_solar)
        irr_dtf = delta_time_encoder(timestamps, time0)
        pv_timefeats = torch.cat([irr_tf, irr_dtf.unsqueeze(1)], dim=1)

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
        return {
            "dev_idx": torch.tensor(0, dtype=torch.long),
            "pv": pv,
            "pv_mask": pv_mask,
            "pv_timefeats": pv_timefeats,
            "forecast_timefeats": forecast_timefeats,
            "sat_tensor": None,
            "sat_timefeats": None,
            "target_pv": target_pv,
            "target_mask": target_mask,
            "skimg_tensor": skimg_tensor,
            "skimg_timefeats": skimg_timefeats,
            "nwp_tensor": nwp_tensor,
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.split == "train":
            r = int(np.random.choice(self._train_anchor_positions))
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
        "dev_idx": _stack("dev_idx"),
        "pv": _stack("pv"),
        "pv_mask": _stack("pv_mask"),
        "pv_timefeats": _stack("pv_timefeats"),
        "forecast_timefeats": _stack("forecast_timefeats"),
        "nwp_tensor": _stack("nwp_tensor"),
        "target_pv": _stack("target_pv"),
        "target_mask": _stack("target_mask"),
    }
    for key in ("sat_tensor", "sat_timefeats", "skimg_tensor", "skimg_timefeats"):
        vals = [s[key] for s in batch]
        if vals[0] is None:
            if not all(v is None for v in vals):
                raise ValueError(f"collate_folsom_irradiance: mixed None and tensor for {key!r}")
            out[key] = None
        else:
            out[key] = torch.stack(vals)
    return out


def build_folsom_irradiance_datasets_from_conf(
    conf: dict | None = None,
    *,
    conf_path: Path | str | None = None,
    train_epoch_len: int = 50_000,
    skyimg_window_size: int | None = None,
) -> tuple[FolsomIrradianceDataset, FolsomIrradianceDataset]:
    """
    Build train/test :class:`FolsomIrradianceDataset` from ``conf_folsom.yaml`` (or passed ``conf``).

    If ``skyimg_window_size`` is set, it overrides ``training.skyimg_window_size`` (last N JPEGs at anchor).
    """
    if conf is None:
        conf = load_folsom_conf(conf_path)
    training_raw = conf.get("training") or {}
    required = (
        "csv_interval_min",
        "irr_input_interval_min",
        "irr_input_len",
        "irr_output_interval_min",
        "irr_output_len",
        "irr_train_time_fraction",
        "test_anchor_stride_min",
        "skyimg_window_size",
        "skyimg_time_resolution_min",
        "skyimg_spatial_size",
    )
    missing = [k for k in required if k not in training_raw]
    if missing:
        raise KeyError(f"Folsom training section missing required key(s): {missing}")
    h = {k: training_raw[k] for k in required}
    if skyimg_window_size is not None:
        h["skyimg_window_size"] = int(skyimg_window_size)
    csv_path = _resolve_folsom_csv_path(conf)
    nwp_csv_path = _resolve_folsom_nwp_csv_path(conf)
    paths = get_training_paths_from_conf(conf)
    site = conf.get("site") or {}
    lat = float(site["latitude"])
    lon = float(site["longitude"])
    kwargs = dict(
        csv_path=csv_path,
        csv_interval_min=h["csv_interval_min"],
        irr_input_interval_min=h["irr_input_interval_min"],
        irr_input_len=h["irr_input_len"],
        irr_output_interval_min=h["irr_output_interval_min"],
        irr_output_len=h["irr_output_len"],
        irr_train_time_fraction=h["irr_train_time_fraction"],
        test_anchor_stride_min=h["test_anchor_stride_min"],
        train_epoch_len=train_epoch_len,
        skyimg_dir=paths["skyimg_dir"],
        skyimg_window_size=h["skyimg_window_size"],
        skyimg_time_resolution_min=h["skyimg_time_resolution_min"],
        skyimg_spatial_size=h["skyimg_spatial_size"],
        nwp_merged_csv_path=nwp_csv_path,
        latitude=lat,
        longitude=lon,
    )
    return (
        FolsomIrradianceDataset(split="train", **kwargs),
        FolsomIrradianceDataset(split="test", **kwargs),
    )


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
) -> list[str]:
    n = int(g.shape[0])
    lines: list[str] = [f"size (per series, num steps): ({n},)"]
    if n == 0:
        lines.append("  (empty series)")
        return lines
    for i in range(min(2, n)):
        lines.append(
            f"  first[{i}]  t={times[i]}  ghi={float(g[i]):.3f}"
        )
    if n > 2:
        for i in (n - 2, n - 1):
            lines.append(
                f"  last[{i}]  t={times[i]}  ghi={float(g[i]):.3f}"
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
        ts = pd.to_datetime(chunk[time_col], errors="coerce")
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
            f"(needs room for input length {ds.irr_input_len} and output length {ds.irr_output_len})."
        )
    r = anchor - amin
    if not bool(ds._train_anchor_mask[r]):
        raise ValueError(
            f"anchor_row={anchor} falls in the **test** time split (irr_train_time_fraction). "
            "Smoke uses the train dataset only: pick an earlier calendar time, or increase the train fraction in YAML."
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

    Loads ``--conf`` (default ``config/conf_folsom.yaml``): CSV, sky JPEG dir, and NWP paths from disk.

    Optional ``--last-input-time`` sets the anchor (last input CSV row time); default is first valid train anchor.
    """
    p = argparse.ArgumentParser(description="Smoke-test Folsom DataLoader (reads paths from YAML)")
    p.add_argument(
        "--conf",
        type=Path,
        default=_PROJECT_ROOT / "config" / "conf_folsom.yaml",
        help="Path to folsom YAML (paths.data_dir + folsom_irradiance_csv, sky, NWP).",
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
            train_epoch_len=args.train_epoch_len,
            skyimg_window_size=args.skyimg_window,
        )
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Failed to load Folsom data from {args.conf.resolve()}:\n  {e}", file=sys.stderr)
        print(
            "Fix paths.data_dir, paths.folsom_irradiance_csv, paths.sky_image_path, "
            "and paths.folsom_nwp_merged_csv in that YAML so files exist on disk.",
            file=sys.stderr,
        )
        return 1
    ds = train_ds
    print(f"[conf] {args.conf.resolve()}")
    print(f"  csv={_resolve_folsom_csv_path(conf)}  train_len={len(ds)}  test_len={len(test_ds)}")

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

    t_end = sky["last_input_time_utc_naive"]
    sk_times, sk_paths = ds._history_sky_frame_records(t_end)
    tin: list[str] = [f"step-{i}" for i in range(int(sample["pv"].shape[-1]))]
    tout: list[str] = [f"step+{i + 1}" for i in range(int(sample["target_pv"].shape[0]))]
    g = sample["pv"][0]
    tg = sample["target_pv"]
    sk = sample["skimg_tensor"]
    sk_ts = [
        (None if p is None else pd.Timestamp(t).strftime("%Y%m%d%H%M%S"))
        for t, p in zip(sk_times, sk_paths)
    ]
    nwp = sample["nwp_tensor"]

    im = sample["pv_mask"]
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

    in_lines = _smoke_irradiance_lines(tin, g)
    in_lines.append(f"input_mask valid values: {inv:.0f} / {im_tot}")
    _smoke_section("IRRADIANCE INPUT (GHI)", in_lines)

    out_lines = _smoke_irradiance_lines(tout, tg)
    out_lines.append(f"target_mask valid values: {tnv:.0f} / {tm_tot}")
    _smoke_section("IRRADIANCE OUTPUT / TARGETS (GHI)", out_lines)

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
        "dev_idx",
        "pv",
        "pv_mask",
        "pv_timefeats",
        "forecast_timefeats",
        "target_pv",
        "sat_tensor",
        "sat_timefeats",
        "skimg_tensor",
        "skimg_timefeats",
        "nwp_tensor",
    ):
        v = batch[k]
        if v is None:
            batch_lines.append(f"{k}: None")
        else:
            batch_lines.append(f"{k}: shape={tuple(v.shape)} dtype={v.dtype}")
    _smoke_section("DATALOADER (first batch)", batch_lines)

    print("smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_smoke_cli())
