"""YLJ parquet dataset — output format matches ``luoyang_zarr.PVDataset``."""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml
from torch.utils.data import Dataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config_utils import get_resolved_paths
from dataloader.luoyang_zarr import _sanitize_nwp_interp
from modules.solar_encoder import delta_time_encoder, solar_features_encoder

# Re-export so training scripts can ``from dataloader.ylj import YLJDataset, collate_batched``.
__all__ = ["YLJDataset", "collate_batched"]

# Expected sequence lengths in augmented parquet (see ``SPMF_preprocessing/ylj/ylj_load.py``).
PV_INPUT_LEN = 576
PV_OUTPUT_LEN = 192

# Default forecast NWP columns in augmented YLJ parquet (grid nearest site ~29.94N, 100.62E).
_DEFAULT_NWP_SSRD_COL = "ssrd_100_6_29_9_predict"
_DEFAULT_NWP_T2M_COL = "t2m_100_6_29_9_predict"
# Stack order matches ``luoyang_zarr._NWP_INTERP_STACK_COLS``: ssrd, msl, t2m, u10, v10, u100, v100.
_NWP_STACK_COLS = ("ssrd", "msl", "t2m", "u10", "v10", "u100", "v100")
YLJ_DEV_DN = "YLJ"
_DEFAULT_SATIMG_WINDOW_SIZE = 24
# Match ``luoyang_zarr.PVDataset._build_sample`` sat history window: [t0-275min, t0-30min].
_SAT_HISTORY_START_MIN_BEFORE_T0 = 245 + 30
_SAT_HISTORY_END_MIN_BEFORE_T0 = 30


def _as_float_array(value, *, dtype=np.float32) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def _as_utc_index(value) -> pd.DatetimeIndex:
    if isinstance(value, pd.DatetimeIndex):
        idx = value
    else:
        idx = pd.DatetimeIndex(pd.to_datetime(list(value), utc=True))
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    return idx


def _solar_features_from_parquet_cell(cell) -> dict[str, np.ndarray]:
    """``solar_time_features`` / ``forecast_solar_time_features`` stored as dict-of-lists in parquet."""
    if not isinstance(cell, dict):
        raise TypeError(f"expected dict solar features cell, got {type(cell)!r}")

    def _get(name: str, *aliases: str) -> np.ndarray:
        for key in (name, *aliases):
            if key in cell:
                return _as_float_array(cell[key], dtype=np.int32 if name == "day_of_year" else np.float32)
        raise KeyError(f"missing {name!r} in solar features cell (aliases={aliases})")

    return {
        "azimuth": _get("azimuth", "solar_azimuth"),
        "zenith": _get("zenith", "solar_zenith"),
        "day_of_year": _get("day_of_year"),
        "hour_of_day": _get("hour_of_day"),
    }


def _encode_timefeats(
    solar_cell,
    timestamps: pd.DatetimeIndex,
    time0_utc: pd.Timestamp,
) -> torch.Tensor:
    solar = _solar_features_from_parquet_cell(solar_cell)
    feats = solar_features_encoder(solar)
    dtime = delta_time_encoder(timestamps, time0_utc)
    return torch.cat([feats, dtime.unsqueeze(1)], dim=1)


def _as_utc_timestamp(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def _build_sat_from_zarr(
    satimg_ds: xr.Dataset,
    time0_utc: pd.Timestamp,
    *,
    window_size: int = _DEFAULT_SATIMG_WINDOW_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load Himawari sat sequence from zarr, same logic as ``luoyang_zarr.PVDataset._build_sample``.
    Returns ``sat_tensor`` ``[T, C, H, W]`` and ``sat_timefeats`` ``[T, 9]``, padded/truncated to ``window_size``.
    """
    t0 = _as_utc_timestamp(time0_utc)
    sat_t0 = pd.Timestamp(t0 - timedelta(minutes=_SAT_HISTORY_START_MIN_BEFORE_T0)).tz_convert("UTC").tz_localize(None)
    sat_t1 = pd.Timestamp(t0 - timedelta(minutes=_SAT_HISTORY_END_MIN_BEFORE_T0)).tz_convert("UTC").tz_localize(None)

    sat_data = satimg_ds.sel(time_utc=slice(sat_t0, sat_t1))
    sat_solar_features = {
        "azimuth": sat_data["azimuth"].values,
        "zenith": sat_data["zenith"].values,
        "day_of_year": sat_data["day_of_year"].values,
        "hour_of_day": sat_data["hour_of_day"].values,
    }
    sat_timestamps_utc = sat_data["time_utc"].values

    sat_timefeats = solar_features_encoder(sat_solar_features)
    sat_dtimefeats = delta_time_encoder(sat_timestamps_utc, t0)
    sat_timefeats = torch.cat([sat_timefeats, sat_dtimefeats.unsqueeze(1)], dim=1)

    sat_tensor = torch.from_numpy(np.asarray(sat_data["images"].values, dtype=np.float32))

    if sat_tensor.shape[0] > window_size:
        sat_tensor = sat_tensor[-window_size:, :, :, :]
        sat_timefeats = sat_timefeats[-window_size:, :]
    elif sat_tensor.shape[0] < window_size:
        pad = window_size - sat_tensor.shape[0]
        sat_tensor = torch.cat(
            [torch.zeros(pad, *sat_tensor.shape[1:], dtype=sat_tensor.dtype), sat_tensor],
            dim=0,
        )
        sat_timefeats = torch.cat(
            [torch.zeros(pad, sat_timefeats.shape[1], dtype=sat_timefeats.dtype), sat_timefeats],
            dim=0,
        )

    return sat_tensor, sat_timefeats


def _build_nwp_tensor(
    row: pd.Series,
    *,
    ssrd_col: str,
    t2m_col: str,
    t_out: int = PV_OUTPUT_LEN,
) -> torch.Tensor:
    """
    Build ``[T_out, 8]`` NWP tensor like ``luoyang_zarr.interpolate_nwp_features``:
    7 forecast scalars (ssrd + msl + t2m + wind) plus a per-step validity mask column.
    """
    ssrd = _as_float_array(row[ssrd_col], dtype=np.float64)
    t2m_c = _as_float_array(row[t2m_col], dtype=np.float64)
    if ssrd.shape[0] != t_out or t2m_c.shape[0] != t_out:
        raise ValueError(
            f"NWP forecast length mismatch: {ssrd_col}={ssrd.shape[0]}, "
            f"{t2m_col}={t2m_c.shape[0]}, expected {t_out}"
        )

    # Luoyang NWP ``t2m`` is Kelvin; YLJ parquet stores Celsius.
    t2m_k = t2m_c + 273.15

    nwp_interp = np.zeros((t_out, len(_NWP_STACK_COLS)), dtype=np.float64)
    col_idx = {name: i for i, name in enumerate(_NWP_STACK_COLS)}
    nwp_interp[:, col_idx["ssrd"]] = ssrd
    nwp_interp[:, col_idx["t2m"]] = t2m_k

    nwp_clean, nwp_mask = _sanitize_nwp_interp(nwp_interp)
    nwp_wmask = np.concatenate([nwp_clean, nwp_mask], axis=1)
    return torch.from_numpy(nwp_wmask.astype(np.float32, copy=False))


def _is_valid_augmented_row(row: pd.Series) -> bool:
    """Skip rows where parquet augmentation failed (e.g. ``observe_power`` was None)."""
    required = (
        "interp_times",
        "interp_power",
        "forecast_times_utc",
        "solar_time_features",
        "forecast_solar_time_features",
        "kt",
        "kt_mask",
        "p_cs",
        "observe_power_future",
        "forecast_p_cs",
    )
    for key in required:
        val = row.get(key)
        if val is None:
            return False
        if isinstance(val, float) and np.isnan(val):
            return False
    return True


def _filter_valid_indices(df: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    kept = [int(i) for i in indices if _is_valid_augmented_row(df.iloc[int(i)])]
    dropped = int(indices.size) - len(kept)
    if dropped:
        print(f"[YLJDataset] dropped {dropped} rows with missing augmented features")
    return np.asarray(kept, dtype=np.intp)


def collate_batched(batch: list[dict]) -> dict:
    """Stack YLJ samples into one dict with batch dim ``B`` first."""
    if not batch:
        raise ValueError("empty batch")

    def _stack(key: str) -> torch.Tensor:
        return torch.stack([s[key] for s in batch])

    out: dict = {
        "dev_idx": _stack("dev_idx"),
        "t0": [s["t0"] for s in batch],
        "pv": _stack("pv"),
        "pv_mask": _stack("pv_mask"),
        "pv_timefeats": _stack("pv_timefeats"),
        "forecast_timefeats": _stack("forecast_timefeats"),
        "kt": _stack("kt"),
        "kt_mask": _stack("kt_mask"),
        "p_cs": _stack("p_cs"),
        "p_mean": _stack("p_mean"),
        "target_pv": _stack("target_pv"),
        "target_mask": _stack("target_mask"),
        "target_p_cs": _stack("target_p_cs"),
    }
    for key in ("nwp_tensor", "sat_tensor", "sat_timefeats", "skimg_tensor", "skimg_timefeats"):
        vals = [s[key] for s in batch]
        if vals[0] is None:
            if not all(v is None for v in vals):
                raise ValueError(f"collate_batched: mixed None and tensor for {key!r}")
            out[key] = None
        else:
            out[key] = torch.stack(vals)
    return out


class YLJDataset(Dataset):
    """
    Load pre-augmented YLJ parquet rows and emit the same sample dict as ``PVDataset``.

    Split policy:
    - ``train`` / ``val``: rows from ``paths.ylj_train_val_parquet``, partitioned by ``timestamp_win``.
    - ``test``: all rows from ``paths.ylj_test_parquet`` in time order (no randomness).

    Each parquet row is one training sample (no per-CSV anchor logic).

    Train: ``__len__`` = ``train_epoch_len``; each ``__getitem__`` draws one row uniformly at
    random from the full train index pool (with replacement), so an epoch can reach all train
    samples rather than only the first ``train_epoch_len`` rows in time order.
    """

    def __init__(
        self,
        config_path: str | Path,
        *,
        split: str = "train",
        train_time_fraction: float = 0.70,
        val_time_fraction: float = 0.15,
        train_epoch_len: int | None = None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train|val|test, got {split!r}")

        self._config_path = Path(config_path)
        self.split = split
        self._train_time_fraction = float(train_time_fraction)
        self._val_time_fraction = float(val_time_fraction)
        self._train_epoch_len = train_epoch_len

        with open(self._config_path) as f:
            conf = yaml.safe_load(f) or {}
        paths = get_resolved_paths(conf, _PROJECT_ROOT)
        paths_cfg = conf.get("paths", {}) or {}
        sampling_cfg = conf.get("sampling", {}) or {}
        self._nwp_ssrd_col = str(sampling_cfg.get("nwp_ssrd_col", _DEFAULT_NWP_SSRD_COL))
        self._nwp_t2m_col = str(sampling_cfg.get("nwp_t2m_col", _DEFAULT_NWP_T2M_COL))
        self._satimg_window_size = int(sampling_cfg.get("satimg_window_size", _DEFAULT_SATIMG_WINDOW_SIZE))

        data_dir = Path(paths.get("data_dir") or _PROJECT_ROOT)
        if not data_dir.is_absolute():
            data_dir = (_PROJECT_ROOT / data_dir).resolve()

        sat_path = paths_cfg.get("sat_path")
        self.satimg_ds: xr.Dataset | None = None
        if sat_path is not None and str(sat_path).strip():
            satimg_dir = Path(str(sat_path))
            if not satimg_dir.is_absolute():
                satimg_dir = (data_dir / satimg_dir).resolve()
            if not satimg_dir.is_dir():
                raise FileNotFoundError(f"sat zarr directory not found: {satimg_dir}")
            self.satimg_ds = xr.open_zarr(satimg_dir)
            print(f"[YLJDataset] opened sat zarr: {satimg_dir}")

        train_val_path = paths_cfg.get("ylj_train_val_parquet")
        test_path = paths_cfg.get("ylj_test_parquet")

        if train_val_path is None:
            raise KeyError(f"{self._config_path}: paths.ylj_train_val_parquet is required")
        train_val_path = Path(train_val_path)
        if not train_val_path.is_absolute():
            train_val_path = (data_dir / train_val_path).resolve()

        if split == "test":
            if test_path is None:
                raise KeyError(
                    f"{self._config_path}: paths.ylj_test_parquet is required for split='test'"
                )
            parquet_path = Path(test_path)
            if not parquet_path.is_absolute():
                parquet_path = (data_dir / parquet_path).resolve()
            if not parquet_path.is_file():
                raise FileNotFoundError(f"test parquet not found: {parquet_path}")
            self._df = pd.read_parquet(parquet_path)
            self._indices = _filter_valid_indices(
                self._df,
                self._sequential_indices(self._df["timestamp_win"]),
            )
        else:
            self._df = pd.read_parquet(train_val_path)
            self._indices = _filter_valid_indices(
                self._df,
                self._split_indices_by_time(
                    self._df["timestamp_win"],
                    split=split,
                    train_frac=self._train_time_fraction,
                    val_frac=self._val_time_fraction,
                ),
            )

        if len(self._indices) == 0:
            raise RuntimeError(f"YLJDataset split={split!r} has zero samples")

        # Model expects a devDn list even for single-site YLJ.
        self.devDn_list = [YLJ_DEV_DN]

    @staticmethod
    def _sequential_indices(timestamp_win: pd.Series) -> np.ndarray:
        """Deterministic row order: sort by ``timestamp_win``, then walk 0..N-1."""
        ts = pd.to_datetime(timestamp_win, errors="coerce")
        return np.argsort(ts.values).astype(np.intp, copy=False)

    @staticmethod
    def _split_indices_by_time(
        timestamp_win: pd.Series,
        *,
        split: str,
        train_frac: float,
        val_frac: float,
    ) -> np.ndarray:
        ts = pd.to_datetime(timestamp_win, errors="coerce")
        order = np.argsort(ts.values)
        n = len(order)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        if split == "train":
            return order[:n_train].astype(np.intp, copy=False)
        if split == "val":
            return order[n_train : n_train + n_val].astype(np.intp, copy=False)
        raise ValueError(split)

    def __len__(self) -> int:
        if self.split == "train" and self._train_epoch_len is not None:
            return int(self._train_epoch_len)
        return int(self._indices.size)

    def _row_to_sample(self, row: pd.Series) -> dict:
        """Map one augmented parquet row -> ``PVDataset``-compatible sample dict."""
        dev_idx = torch.tensor(650, dtype=torch.long)

        interp_times = _as_utc_index(row["interp_times"])
        forecast_times = _as_utc_index(row["forecast_times_utc"])
        time0_utc = interp_times[-1]

        interp_power = _as_float_array(row["interp_power"])
        kt = _as_float_array(row["kt"])
        kt_mask = _as_float_array(row["kt_mask"])
        p_cs = _as_float_array(row["p_cs"])
        p_mean_val = 1.0

        pv = torch.from_numpy(interp_power).unsqueeze(0)
        kt_t = torch.from_numpy(kt).unsqueeze(0)
        kt_mask_t = torch.from_numpy(kt_mask).unsqueeze(0)
        p_cs_t = torch.from_numpy(p_cs).unsqueeze(0)
        p_mean = torch.tensor(p_mean_val, dtype=torch.float32)

        # YLJ parquet has no inverter_state; treat all timesteps as valid for now.
        pv_mask = torch.ones(1, PV_INPUT_LEN, dtype=torch.float32)

        pv_timefeats = _encode_timefeats(row["solar_time_features"], interp_times, time0_utc)
        forecast_timefeats = _encode_timefeats(
            row["forecast_solar_time_features"], forecast_times, time0_utc
        )

        target_pv = torch.from_numpy(_as_float_array(row["observe_power_future"]))
        target_p_cs = torch.from_numpy(_as_float_array(row["forecast_p_cs"]))
        target_mask = torch.ones(PV_OUTPUT_LEN, dtype=torch.float32)

        nwp_tensor = _build_nwp_tensor(
            row,
            ssrd_col=self._nwp_ssrd_col,
            t2m_col=self._nwp_t2m_col,
        )
        if self.satimg_ds is not None:
            sat_tensor, sat_timefeats = _build_sat_from_zarr(
                self.satimg_ds,
                time0_utc,
                window_size=self._satimg_window_size,
            )
        else:
            sat_tensor = None
            sat_timefeats = None

        # Zero 5-min slots 0,1 / 3,4 / 6,7 / ...; keep every 3rd column (15-min) from parquet.
        kt_mask_t[:, 0::3] = 0.0
        kt_mask_t[:, 1::3] = 0.0

        pv_mask[:, 0::3] = 0.0
        pv_mask[:, 1::3] = 0.0

        kt_t[:, 0::3] = 0.0
        kt_t[:, 1::3] = 0.0

        pv[:, 0::3] = 0.0
        pv[:, 1::3] = 0.0

        return {
            "dev_idx": dev_idx,
            "t0": row["timestamp_win"],
            "pv": pv,
            "pv_mask": pv_mask,
            "pv_timefeats": pv_timefeats,
            "kt": kt_t/1000,
            "kt_mask": kt_mask_t,
            "p_cs": p_cs_t,
            "p_mean": p_mean,
            "forecast_timefeats": forecast_timefeats,
            "sat_tensor": sat_tensor,
            "sat_timefeats": sat_timefeats,
            "skimg_tensor": None,
            "skimg_timefeats": None,
            "nwp_tensor": nwp_tensor,
            "target_pv": target_pv,
            "target_mask": target_mask,
            "target_p_cs": target_p_cs,
        }

    def __getitem__(self, idx: int) -> dict:
        if self.split == "train":
            # idx only defines epoch length; row is sampled uniformly from all train indices.
            row_idx = int(self._indices[np.random.randint(len(self._indices))])
        else:
            if idx < 0 or idx >= len(self._indices):
                raise IndexError(f"index {idx} out of range for split={self.split!r} (size={len(self._indices)})")
            row_idx = int(self._indices[idx])
        row = self._df.iloc[row_idx]
        return self._row_to_sample(row)


def _smoke_test() -> None:
    """Quick manual check: ``python -m dataloader.ylj``."""
    cfg = _PROJECT_ROOT / "config/datasets/conf_ylj.yaml"
    if not cfg.is_file():
        print(f"skip smoke test: missing {cfg}")
        return
    ds = YLJDataset(cfg, split="train", train_epoch_len=4)
    sample = ds[0]
    print("len:", len(ds))
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(k, tuple(v.shape), v.dtype)
        else:
            print(k, v)


if __name__ == "__main__":
    _smoke_test()
