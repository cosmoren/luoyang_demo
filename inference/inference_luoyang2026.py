"""
Luoyang 2026 test-set inference (15m / 4h single-horizon).

For each inverter and each test anchor window, run the trained model and save:
  * per-inverter CSV: forecast time (t0+15m or t0+4h), PV prediction, PV truth
  * station-total CSV: same times with summed pred/truth across inverters
  * prints station-level total-power RMSE at the end

Window-major inference (zarr / cache) appends CSV rows after each window completes.

By default ``--stride_min 5`` rolls one anchor every 5 minutes (dense coverage).
Use ``--stride_min 1200`` to match the sparse test stride in conf_luoyang_2026.yaml.

Multi-GPU: shard inverters across GPUs (one process per GPU). Each GPU loads only
its CSV subset and writes inverter CSVs; rank 0 merges station totals.

Quick run examples:
python inference/inference_luoyang2026.py \
  --task 15m \
  --checkpoint checkpoint_2026_fixedhuber/pv_forecast_vit_best_task_15m_gpu0.pt \
  --dataset-config conf_luoyang_2026.yaml \
  --stride_min 5 \
  --batch_size 256 \
  --cache_dir inference_cache/luoyang2026_test_cache \
  --profile_every 100 \
  --output_dir inference_results/luoyang2026_15m_fullcache

python inference/inference_luoyang2026.py \
  --task 48h \
  --checkpoint checkpoint_2026_fixedhuber/pv_forecast_vit_best_task_48h_gpu7.pt \
  --dataset-config conf_luoyang_2026.yaml \
  --device cuda:0 \
  --output_dir inference_results/luoyang2026_48h_daily
"""

from __future__ import annotations

import argparse
import csv
import contextlib
import hashlib
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dataloader.luoyang_2026_zarr import (  # noqa: E402
    INVERTER_STATE_COL,
    PVDataset,
    VALID_STATE,
    collate_batched,
    collate_with_shared_sat_sky,
    interpolate_nwp_features,
    list_csv_files,
)
from models.models import pv_forecasting_model_vit_imgs  # noqa: E402
from modules.solar_encoder import (  # noqa: E402
    compute_solar_features,
    delta_time_encoder,
    solar_features_encoder,
)
from training.train_vit_luoyang2026 import (  # noqa: E402
    TASK_TO_INDEX,
    _batch_to_device,
    _dataset_kwargs,
    _resolve_data_dir,
    _resolve_named_config,
    forward_vit,
)

_DEFAULT_DATASET_CONFIG = "conf_luoyang_2026.yaml"
_DEFAULT_STRIDE_MIN = 5
_DATASETS_CONFIG_DIR = _PROJECT_ROOT / "config" / "datasets"
_TEST_RANGE_START_BJ = pd.Timestamp("2026-05-11", tz="Asia/Shanghai").date()
_TEST_RANGE_END_BJ = pd.Timestamp("2026-06-11", tz="Asia/Shanghai").date()


@dataclass(frozen=True)
class InferWorkerConfig:
    task: str
    checkpoint: str
    output_dir: str
    dataset_config: str
    stride_min: int
    batch_size: int
    num_workers: int
    max_inverters: int | None
    limit_batches: int | None
    horizon_idx: int
    cache_dir: str | None
    cache_full: bool
    profile_every: int


def _dataset_kwargs_for_infer(
    dataset_config_name: str,
    split: str,
    *,
    test_stride_min_override: int,
    sample_file_subset: list[Path] | None = None,
) -> dict:
    """Build PVDataset kwargs for inference, overriding test anchor stride."""
    kwargs = _dataset_kwargs(dataset_config_name, split, max_files=None)
    kwargs["test_anchor_stride_min"] = int(test_stride_min_override)
    if sample_file_subset is not None:
        kwargs["sample_file_subset"] = sample_file_subset
    kwargs["enable_sat_sky_cache"] = True
    return kwargs


def _horizon_idx(task: str) -> int:
    idx = TASK_TO_INDEX[task]
    if idx is None:
        raise NotImplementedError("48h inference is not implemented yet.")
    return int(idx)


def _forecast_times_for_windows(
    t0_refs: list[pd.Timestamp],
    *,
    horizon_idx: int,
    output_interval_min: int,
) -> tuple[list[str], list[str]]:
    """Return ISO UTC / Beijing strings for forecast target times (not anchor t0)."""
    offset_min = (horizon_idx + 1) * int(output_interval_min)
    utc_strs: list[str] = []
    bj_strs: list[str] = []
    for t0 in t0_refs:
        t0_utc = pd.Timestamp(t0)
        if t0_utc.tzinfo is None:
            t0_utc = t0_utc.tz_localize("UTC")
        else:
            t0_utc = t0_utc.tz_convert("UTC")
        forecast_utc = t0_utc + pd.Timedelta(minutes=offset_min)
        forecast_bj = forecast_utc.tz_convert("Asia/Shanghai")
        utc_strs.append(forecast_utc.isoformat())
        bj_strs.append(forecast_bj.isoformat())
    return utc_strs, bj_strs


def _to_utc_iso(ts: pd.Timestamp) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.isoformat()


def _cfg_sha256(cfg_path: Path) -> str:
    text = cfg_path.read_text(encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_cache_manifest(cache_dir: Path) -> dict:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"cache manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid cache manifest json: {manifest_path} ({e})") from e
    if not isinstance(manifest, dict):
        raise TypeError(f"cache manifest must be JSON object: {manifest_path}")
    return manifest


def _load_sat_sky_cache_window(
    cache_dir: Path,
    win_idx: int,
    *,
    expected_time_utc: str | None = None,
) -> dict[str, torch.Tensor | None]:
    pt_path = cache_dir / "windows" / f"win_{int(win_idx):05d}.pt"
    if not pt_path.is_file():
        raise FileNotFoundError(f"cache window file not found: {pt_path}")
    try:
        rec = torch.load(pt_path, map_location="cpu", weights_only=False)
    except TypeError:
        rec = torch.load(pt_path, map_location="cpu")
    if not isinstance(rec, dict):
        raise TypeError(f"invalid cache window content (expect dict): {pt_path}")
    required = (
        "sat_tensor",
        "sat_timefeats",
        "sat_valid_mask",
        "skimg_tensor",
        "skimg_timefeats",
        "skimg_valid_mask",
    )
    missing = [k for k in required if k not in rec]
    if missing:
        raise KeyError(f"{pt_path} missing required keys: {missing}")
    time_utc = rec.get("time_utc")
    if expected_time_utc is not None and time_utc is not None and str(time_utc) != str(expected_time_utc):
        raise ValueError(
            f"cache window time mismatch at win={win_idx}: cache={time_utc} expected={expected_time_utc}"
        )
    sat_valid_mask = rec["sat_valid_mask"]
    skimg_valid_mask = rec["skimg_valid_mask"]
    if not isinstance(sat_valid_mask, torch.Tensor) or sat_valid_mask.numel() == 0:
        raise ValueError(f"invalid sat_valid_mask in {pt_path}")
    if not isinstance(skimg_valid_mask, torch.Tensor) or skimg_valid_mask.numel() == 0:
        raise ValueError(f"invalid skimg_valid_mask in {pt_path}")
    return {
        "sat_tensor": rec["sat_tensor"],
        "sat_timefeats": rec["sat_timefeats"],
        "sat_valid": torch.tensor(float(sat_valid_mask.reshape(-1)[0].item()), dtype=torch.float32),
        "skimg_tensor": rec["skimg_tensor"],
        "skimg_timefeats": rec["skimg_timefeats"],
        "skimg_valid": torch.tensor(float(skimg_valid_mask.reshape(-1)[0].item()), dtype=torch.float32),
    }


def _load_pv_batch_cache(
    cache_dir: Path,
    *,
    win_idx: int,
    batch_idx: int,
    file_start: int,
    file_end: int,
) -> dict:
    pt_path = cache_dir / "pv_batches" / f"win_{int(win_idx):05d}" / f"batch_{int(batch_idx):03d}.pt"
    if not pt_path.is_file():
        raise FileNotFoundError(f"PV cache batch file not found: {pt_path}")
    try:
        rec = torch.load(pt_path, map_location="cpu", weights_only=False)
    except TypeError:
        rec = torch.load(pt_path, map_location="cpu")
    if not isinstance(rec, dict):
        raise TypeError(f"invalid PV cache content (expect dict): {pt_path}")
    required = (
        "win_idx",
        "file_start",
        "file_end",
        "dev_idx",
        "kt",
        "kt_mask",
        "pv_timefeats",
        "forecast_timefeats",
        "target_pv",
        "target_p_cs",
        "p_mean",
        "sample_valid",
    )
    missing = [k for k in required if k not in rec]
    if missing:
        raise KeyError(f"{pt_path} missing required keys: {missing}")
    if int(rec["win_idx"]) != int(win_idx):
        raise ValueError(f"{pt_path} win_idx mismatch: cache={rec['win_idx']} expected={win_idx}")
    if int(rec["file_start"]) != int(file_start) or int(rec["file_end"]) != int(file_end):
        raise ValueError(
            f"{pt_path} file range mismatch: cache=[{rec['file_start']},{rec['file_end']}) "
            f"expected=[{file_start},{file_end})"
        )
    return rec


def _merge_cached_pv_with_sat_sky(
    pv_batch: dict,
    sat_sky_bundle: dict[str, torch.Tensor | None],
) -> dict:
    bsz = int(pv_batch["dev_idx"].shape[0])
    out = {
        "dev_idx": pv_batch["dev_idx"],
        "kt": pv_batch["kt"],
        "kt_mask": pv_batch["kt_mask"],
        "pv_timefeats": pv_batch["pv_timefeats"],
        "forecast_timefeats": pv_batch["forecast_timefeats"],
        "target_pv": pv_batch["target_pv"],
        "target_p_cs": pv_batch["target_p_cs"],
        "p_mean": pv_batch["p_mean"],
        "sample_valid": pv_batch["sample_valid"],
        "nwp_tensor": pv_batch.get("nwp_tensor"),
    }

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
        t_b = tensor.unsqueeze(0).expand(bsz, *tensor.shape).contiguous()
        tf_b = timefeats.unsqueeze(0).expand(bsz, *timefeats.shape).contiguous()
        valid_b = torch.full((bsz,), valid_scalar, dtype=torch.float32)
        return t_b, tf_b, valid_b

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


def predict_pv_kW(model: nn.Module, d: dict) -> torch.Tensor:
    """Return PV power forecast in kW, shape [B, T_out]. Matches train_vit_luoyang2026."""
    kt_pred = forward_vit(model, d)
    return kt_pred * d["target_p_cs"] * d["p_mean"].unsqueeze(1)


def _parse_gpu_ids(num_gpus: int, gpu_ids_str: str | None) -> list[int]:
    if num_gpus < 1:
        raise ValueError(f"--num_gpus must be >= 1 (got {num_gpus})")
    if not torch.cuda.is_available():
        if num_gpus > 1:
            raise RuntimeError("CUDA is not available; cannot use --num_gpus > 1")
        return [0]

    n_visible = torch.cuda.device_count()
    if gpu_ids_str:
        ids = [int(x.strip()) for x in gpu_ids_str.split(",") if x.strip()]
        if not ids:
            raise ValueError("--gpu_ids must list at least one GPU id")
        if len(set(ids)) != len(ids):
            raise ValueError(f"--gpu_ids contains duplicates: {gpu_ids_str}")
        for gid in ids:
            if gid < 0 or gid >= n_visible:
                raise ValueError(
                    f"GPU id {gid} out of range for {n_visible} visible CUDA device(s)"
                )
        if num_gpus != len(ids):
            raise ValueError(
                f"--num_gpus={num_gpus} must match number of --gpu_ids ({len(ids)})"
            )
        return ids

    if num_gpus > n_visible:
        raise ValueError(
            f"--num_gpus={num_gpus} exceeds visible CUDA devices ({n_visible})"
        )
    return list(range(num_gpus))


def _list_infer_sample_files(
    dataset_config_name: str,
    *,
    max_inverters: int | None,
) -> list[Path]:
    cfg_path = _resolve_named_config(_DATASETS_CONFIG_DIR, dataset_config_name, "dataset-config")
    import yaml

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    paths_cfg = cfg.get("paths", {}) or {}
    data_dir = _resolve_data_dir(paths_cfg, cfg_path)
    pv_dir = data_dir / str(paths_cfg["pv_path"])
    files = list_csv_files(pv_dir)
    if max_inverters is not None:
        if max_inverters < 1:
            raise ValueError(f"--max_inverters must be >= 1 (got {max_inverters})")
        files = files[: int(max_inverters)]
    return files


def _shard_file_list(files: list[Path], rank: int, world_size: int) -> list[Path]:
    if world_size <= 1:
        return list(files)
    n = len(files)
    chunk = (n + world_size - 1) // world_size
    start = rank * chunk
    end = min(start + chunk, n)
    return files[start:end]


def _log_prefix(rank: int, world_size: int, gpu_id: int) -> str:
    if world_size <= 1:
        return "[inference_luoyang2026]"
    return f"[inference_luoyang2026 gpu={gpu_id} rank={rank + 1}/{world_size}]"


def _resolve_output_dir(output_dir: str | None, task: str) -> Path:
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    return (_PROJECT_ROOT / "inference_results" / f"luoyang2026_{task}").resolve()


def _load_model(
    checkpoint: Path,
    device: torch.device,
    dev_dn_list: list[str],
) -> nn.Module:
    try:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location=device)

    if ckpt.get("ema"):
        print("[inference_luoyang2026] checkpoint was saved with EMA weights", flush=True)

    model = pv_forecasting_model_vit_imgs(dev_dn_list=dev_dn_list).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(
            f"[inference_luoyang2026] WARNING: missing keys: {len(missing)} "
            f"(first 5: {missing[:5]})",
            flush=True,
        )
    if unexpected:
        print(
            f"[inference_luoyang2026] WARNING: unexpected keys: {len(unexpected)} "
            f"(first 5: {unexpected[:5]})",
            flush=True,
        )
    model.eval()
    return model


def _write_per_inverter_csvs(
    out_inv_dir: Path,
    *,
    inverter_names: list[str],
    n_files: int,
    nw: int,
    pred_buf: dict[int, np.ndarray],
    true_buf: dict[int, np.ndarray],
    filled: dict[int, np.ndarray],
    time_utc: list[str],
    time_bj: list[str],
) -> int:
    n_written = 0
    for file_idx in range(n_files):
        mask = filled[file_idx]
        if not mask.any():
            continue
        rows = {
            "time_utc": [time_utc[w] for w in range(nw) if mask[w]],
            "time_bj": [time_bj[w] for w in range(nw) if mask[w]],
            "pv_pred_kW": pred_buf[file_idx][mask].astype(np.float64),
            "pv_true_kW": true_buf[file_idx][mask].astype(np.float64),
        }
        safe_name = inverter_names[file_idx].replace("=", "_").replace("/", "_")
        out_path = out_inv_dir / f"{safe_name}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        n_written += 1
    return n_written


def _station_rmse(pred_total: np.ndarray, true_total: np.ndarray) -> float:
    diff = pred_total.astype(np.float64) - true_total.astype(np.float64)
    return float(np.sqrt(np.mean(diff**2)))


def _partial_station_arrays(
    *,
    n_files: int,
    nw: int,
    pred_buf: dict[int, np.ndarray],
    true_buf: dict[int, np.ndarray],
    filled: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_total = np.zeros(nw, dtype=np.float64)
    true_total = np.zeros(nw, dtype=np.float64)
    counts = np.zeros(nw, dtype=np.int32)
    for file_idx in range(n_files):
        mask = filled[file_idx]
        pred_total[mask] += pred_buf[file_idx][mask].astype(np.float64)
        true_total[mask] += true_buf[file_idx][mask].astype(np.float64)
        counts[mask] += 1
    return pred_total, true_total, counts


def _write_station_csv(
    out_dir: Path,
    *,
    task: str,
    nw: int,
    pred_total: np.ndarray,
    true_total: np.ndarray,
    counts: np.ndarray,
    time_utc: list[str],
    time_bj: list[str],
) -> tuple[Path, float, int]:
    valid_windows = counts > 0
    if not valid_windows.all():
        n_missing = int((~valid_windows).sum())
        print(
            f"[inference_luoyang2026] WARNING: {n_missing}/{nw} windows have no "
            f"predictions from any inverter (partial station sums).",
            flush=True,
        )

    pred_valid = pred_total[valid_windows]
    true_valid = true_total[valid_windows]
    n_valid = int(pred_valid.size)
    rmse = _station_rmse(pred_valid, true_valid) if n_valid > 0 else float("nan")

    station_path = out_dir / f"station_total_{task}.csv"
    pd.DataFrame(
        {
            "time_utc": [time_utc[w] for w in range(nw) if valid_windows[w]],
            "time_bj": [time_bj[w] for w in range(nw) if valid_windows[w]],
            "pv_pred_kW_total": pred_valid,
            "pv_true_kW_total": true_valid,
        }
    ).to_csv(station_path, index=False)
    return station_path, rmse, n_valid


_INVERTER_CSV_FIELDS = ("time_utc", "time_bj", "pv_pred_kW", "pv_true_kW")
_STATION_CSV_FIELDS = ("time_utc", "time_bj", "pv_pred_kW_total", "pv_true_kW_total")


class _IncrementalResultWriter:
    """Append per-inverter and (optionally) station CSV rows as windows complete."""

    def __init__(
        self,
        *,
        out_inv_dir: Path,
        out_dir: Path,
        task: str,
        inverter_names: list[str],
        time_utc: list[str],
        time_bj: list[str],
        write_station: bool,
    ) -> None:
        self._out_inv_dir = out_inv_dir
        self._out_dir = out_dir
        self._task = task
        self._inverter_names = inverter_names
        self._time_utc = time_utc
        self._time_bj = time_bj
        self._write_station = write_station
        self._inv_handles: dict[int, tuple[object, csv.DictWriter]] = {}
        self._inv_paths_written: set[int] = set()
        self._station_path: Path | None = None
        self._station_handle = None
        self._station_writer: csv.DictWriter | None = None
        self._station_sse = 0.0
        self._station_n = 0
        out_inv_dir.mkdir(parents=True, exist_ok=True)

    def _inverter_csv_path(self, file_idx: int) -> Path:
        safe_name = self._inverter_names[file_idx].replace("=", "_").replace("/", "_")
        return self._out_inv_dir / f"{safe_name}.csv"

    def _get_inverter_writer(self, file_idx: int) -> csv.DictWriter:
        if file_idx not in self._inv_handles:
            path = self._inverter_csv_path(file_idx)
            handle = path.open("w", newline="", encoding="utf-8")
            writer = csv.DictWriter(handle, fieldnames=_INVERTER_CSV_FIELDS)
            writer.writeheader()
            self._inv_handles[file_idx] = (handle, writer)
        return self._inv_handles[file_idx][1]

    def _ensure_station_writer(self) -> csv.DictWriter:
        if self._station_writer is None:
            self._station_path = self._out_dir / f"station_total_{self._task}.csv"
            self._station_handle = self._station_path.open("w", newline="", encoding="utf-8")
            self._station_writer = csv.DictWriter(self._station_handle, fieldnames=_STATION_CSV_FIELDS)
            self._station_writer.writeheader()
        return self._station_writer

    def flush_window(self, win_idx: int, window_preds: dict[int, tuple[float, float]]) -> None:
        if not window_preds:
            return
        for file_idx, (pred, true) in window_preds.items():
            writer = self._get_inverter_writer(file_idx)
            writer.writerow(
                {
                    "time_utc": self._time_utc[win_idx],
                    "time_bj": self._time_bj[win_idx],
                    "pv_pred_kW": float(pred),
                    "pv_true_kW": float(true),
                }
            )
            self._inv_paths_written.add(file_idx)

        for handle, _ in self._inv_handles.values():
            handle.flush()

        if self._write_station:
            pred_sum = float(sum(p for p, _ in window_preds.values()))
            true_sum = float(sum(t for _, t in window_preds.values()))
            station_writer = self._ensure_station_writer()
            station_writer.writerow(
                {
                    "time_utc": self._time_utc[win_idx],
                    "time_bj": self._time_bj[win_idx],
                    "pv_pred_kW_total": pred_sum,
                    "pv_true_kW_total": true_sum,
                }
            )
            assert self._station_handle is not None
            self._station_handle.flush()
            diff = pred_sum - true_sum
            self._station_sse += diff * diff
            self._station_n += 1

    def finish(self) -> tuple[int, float, int, Path | None]:
        for handle, _ in self._inv_handles.values():
            handle.close()
        self._inv_handles.clear()
        if self._station_handle is not None:
            self._station_handle.close()
            self._station_handle = None
        rmse = (
            float(np.sqrt(self._station_sse / self._station_n))
            if self._station_n > 0
            else float("nan")
        )
        return len(self._inv_paths_written), rmse, self._station_n, self._station_path


def _merge_station_shards(
    shard_dir: Path,
    *,
    out_dir: Path,
    task: str,
    nw: int,
    time_utc: list[str],
    time_bj: list[str],
) -> tuple[Path, float, int]:
    shard_paths = sorted(shard_dir.glob("shard_*.npz"))
    if not shard_paths:
        raise RuntimeError(f"No shard files found under {shard_dir}")

    pred_total = np.zeros(nw, dtype=np.float64)
    true_total = np.zeros(nw, dtype=np.float64)
    counts = np.zeros(nw, dtype=np.int32)
    for path in shard_paths:
        data = np.load(path)
        pred_total += data["pred_total"]
        true_total += data["true_total"]
        counts += data["counts"]

    return _write_station_csv(
        out_dir,
        task=task,
        nw=nw,
        pred_total=pred_total,
        true_total=true_total,
        counts=counts,
        time_utc=time_utc,
        time_bj=time_bj,
    )


def _run_inference_shard(
    *,
    worker_cfg: InferWorkerConfig,
    shard_files: list[Path],
    gpu_id: int,
    rank: int,
    world_size: int,
    out_inv_dir: Path,
    shard_dir: Path | None,
    device_override: str | None = None,
) -> tuple[int, int]:
    if not shard_files:
        tag = _log_prefix(rank, world_size, gpu_id)
        print(f"{tag} empty shard; nothing to do", flush=True)
        return 0, 0

    tag = _log_prefix(rank, world_size, gpu_id)
    if device_override is not None:
        device = torch.device(device_override)
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
    print(f"{tag} device={device} inverters={len(shard_files)}", flush=True)

    test_dataset = PVDataset(
        **_dataset_kwargs_for_infer(
            worker_cfg.dataset_config,
            "test",
            test_stride_min_override=worker_cfg.stride_min,
            sample_file_subset=shard_files,
        )
    )

    nw = test_dataset._num_test_windows
    n_files = len(test_dataset.sample_files)
    n_total = len(test_dataset)
    dt_min = int(test_dataset.pv_output_interval_min)
    print(
        f"{tag} windows_per_inverter={nw} total_samples={n_total} "
        f"batch_size={worker_cfg.batch_size}",
        flush=True,
    )

    t0_refs: list[pd.Timestamp] = list(test_dataset._test_last_x_time_ref or [])
    if len(t0_refs) != nw:
        raise RuntimeError(f"|_test_last_x_time_ref|={len(t0_refs)} != nw={nw}")
    time_utc, time_bj = _forecast_times_for_windows(
        t0_refs,
        horizon_idx=worker_cfg.horizon_idx,
        output_interval_min=dt_min,
    )
    inverter_names = [p.stem.replace("_", "=") for p in test_dataset.sample_files]

    ckpt_path = Path(worker_cfg.checkpoint).expanduser().resolve()
    dev_dn_list = test_dataset.devDn_list
    model = _load_model(ckpt_path, device, dev_dn_list)

    cache_dir = Path(worker_cfg.cache_dir).expanduser().resolve() if worker_cfg.cache_dir else None
    full_cache_mode = bool(cache_dir is not None and worker_cfg.cache_full)
    use_shared_sat_sky = world_size == 1 or cache_dir is not None

    result_writer: _IncrementalResultWriter | None = None
    shard_pred_total: np.ndarray | None = None
    shard_true_total: np.ndarray | None = None
    shard_counts: np.ndarray | None = None
    if use_shared_sat_sky:
        result_writer = _IncrementalResultWriter(
            out_inv_dir=out_inv_dir,
            out_dir=out_inv_dir.parent,
            task=worker_cfg.task,
            inverter_names=inverter_names,
            time_utc=time_utc,
            time_bj=time_bj,
            write_station=(world_size == 1 and shard_dir is None),
        )
        if world_size > 1:
            shard_pred_total = np.zeros(nw, dtype=np.float64)
            shard_true_total = np.zeros(nw, dtype=np.float64)
            shard_counts = np.zeros(nw, dtype=np.int32)
        pred_buf = None
        true_buf = None
        filled = None
    else:
        pred_buf = {i: np.full(nw, np.nan, dtype=np.float32) for i in range(n_files)}
        true_buf = {i: np.full(nw, np.nan, dtype=np.float32) for i in range(n_files)}
        filled = {i: np.zeros(nw, dtype=bool) for i in range(n_files)}

    if device.type == "cuda":
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    else:
        autocast_ctx = contextlib.nullcontext()

    n_batches_per_window = (n_files + worker_cfg.batch_size - 1) // worker_cfg.batch_size
    n_batches_total = nw * n_batches_per_window
    n_batches_loader = (n_total + worker_cfg.batch_size - 1) // worker_cfg.batch_size
    batch_idx_global = 0
    n_skipped = 0
    prof = {
        "modality_load_s": 0.0,
        "sample_build_s": 0.0,
        "collate_s": 0.0,
        "h2d_s": 0.0,
        "gpu_s": 0.0,
        "post_s": 0.0,
        "batch_total_s": 0.0,
        "windows": 0,
    }

    if full_cache_mode:
        if world_size > 1:
            raise ValueError("Full cache mode currently supports single-GPU inference only.")
        print(
            f"{tag} full-cache mode: windows={nw} batches_per_window={n_batches_per_window}",
            flush=True,
        )

    if use_shared_sat_sky:
        if full_cache_mode:
            mode = "full-cache"
        else:
            mode = "sat-sky-cache" if cache_dir is not None else "zarr"
        runtime = "single-GPU" if world_size == 1 else "multi-GPU"
        print(
            f"{tag} Running inference ({runtime} window-major: "
            f"1 {mode} load per window, {n_files} inverters)...",
            flush=True,
        )
    else:
        if worker_cfg.num_workers > 0:
            print(
                f"{tag} WARNING: num_workers>0 can hang on first batch (zarr + fork). "
                f"Prefer --num_workers 0.",
                flush=True,
            )
        print(f"{tag} Running inference (dataloader mode)...", flush=True)

    if not use_shared_sat_sky:
        test_loader = DataLoader(
            test_dataset,
            batch_size=worker_cfg.batch_size,
            shuffle=False,
            collate_fn=collate_batched,
            num_workers=worker_cfg.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(worker_cfg.num_workers > 0),
        )
        batch_iter = iter(test_loader)

    with torch.no_grad():
        if use_shared_sat_sky:
            for win_idx in range(nw):
                t_win0 = time.perf_counter()
                window_preds: dict[int, tuple[float, float]] = {}
                window_finished = False
                if cache_dir is not None:
                    sat_sky = _load_sat_sky_cache_window(
                        cache_dir,
                        win_idx,
                        expected_time_utc=_to_utc_iso(t0_refs[win_idx]),
                    )
                else:
                    sat_sky = test_dataset.load_sat_sky_for_window(win_idx)
                t_mod = time.perf_counter()
                prof["modality_load_s"] += float(t_mod - t_win0)
                prof["windows"] += 1

                for file_start in range(0, n_files, worker_cfg.batch_size):
                    if worker_cfg.limit_batches is not None and batch_idx_global >= worker_cfg.limit_batches:
                        break
                    t_batch0 = time.perf_counter()
                    file_end = min(file_start + worker_cfg.batch_size, n_files)
                    if full_cache_mode:
                        assert cache_dir is not None
                        t_sample0 = time.perf_counter()
                        batch_idx_local = file_start // worker_cfg.batch_size
                        pv_batch = _load_pv_batch_cache(
                            cache_dir,
                            win_idx=win_idx,
                            batch_idx=batch_idx_local,
                            file_start=file_start,
                            file_end=file_end,
                        )
                        t_sample1 = time.perf_counter()
                        prof["sample_build_s"] += float(t_sample1 - t_sample0)
                        batch = _merge_cached_pv_with_sat_sky(pv_batch, sat_sky)
                        t_data = time.perf_counter()
                        prof["collate_s"] += float(t_data - t_sample1)
                    else:
                        t_sample0 = time.perf_counter()
                        pv_samples = [
                            test_dataset.build_pv_sample_for_window(file_idx, win_idx)
                            for file_idx in range(file_start, file_end)
                        ]
                        t_sample1 = time.perf_counter()
                        prof["sample_build_s"] += float(t_sample1 - t_sample0)
                        batch = collate_with_shared_sat_sky(pv_samples, sat_sky)
                        t_data = time.perf_counter()
                        prof["collate_s"] += float(t_data - t_sample1)

                    if batch_idx_global == 0:
                        pv_label = "pv_cache_load+merge" if full_cache_mode else "pv_collate"
                        print(
                            f"{tag} first window modality_load={t_mod - t_win0:.2f}s "
                            f"{pv_label}={t_data - t_batch0:.2f}s "
                            f"(window {win_idx}, inverters {file_start}-{file_end - 1})",
                            flush=True,
                        )

                    t_h2d0 = time.perf_counter()
                    d = _batch_to_device(batch, device)
                    t_h2d = time.perf_counter()
                    prof["h2d_s"] += float(t_h2d - t_h2d0)

                    sample_valid = batch.get("sample_valid")
                    with autocast_ctx:
                        pv_pred = predict_pv_kW(model, d)
                    pv_pred = pv_pred.float()
                    t_gpu = time.perf_counter()
                    prof["gpu_s"] += float(t_gpu - t_h2d)
                    if batch_idx_global == 0:
                        print(f"{tag} first batch GPU forward in {t_gpu - t_data:.3f}s", flush=True)

                    t_post0 = time.perf_counter()
                    pred_np = pv_pred[:, worker_cfg.horizon_idx].detach().cpu().numpy()
                    true_np = d["target_pv"][:, worker_cfg.horizon_idx].detach().cpu().numpy()
                    bsz = pred_np.shape[0]
                    for i in range(bsz):
                        file_idx = file_start + i
                        if sample_valid is not None and float(sample_valid[i].item()) <= 0.0:
                            n_skipped += 1
                            continue
                        window_preds[file_idx] = (float(pred_np[i]), float(true_np[i]))
                    if file_end >= n_files:
                        window_finished = True
                    t_post = time.perf_counter()
                    prof["post_s"] += float(t_post - t_post0)

                    batch_idx_global += 1
                    t_now = time.perf_counter()
                    prof["batch_total_s"] += float(t_now - t_batch0)
                    if (batch_idx_global % 10 == 0) or batch_idx_global == 1:
                        print(
                            f"{tag} batch {batch_idx_global}/{n_batches_total} "
                            f"window={win_idx + 1}/{nw} batch_time={t_now - t_batch0:.1f}s",
                            flush=True,
                        )
                    if worker_cfg.profile_every > 0 and (
                        batch_idx_global % worker_cfg.profile_every == 0 or batch_idx_global == 1
                    ):
                        bt = max(float(prof["batch_total_s"]), 1e-9)
                        bcnt = max(int(batch_idx_global), 1)
                        wcnt = max(int(prof["windows"]), 1)
                        print(
                            f"{tag} profile@batch={batch_idx_global}: "
                            f"avg_batch={bt/bcnt:.3f}s, "
                            f"sample_build={prof['sample_build_s']/bt*100:.1f}%, "
                            f"collate={prof['collate_s']/bt*100:.1f}%, "
                            f"h2d={prof['h2d_s']/bt*100:.1f}%, "
                            f"gpu={prof['gpu_s']/bt*100:.1f}%, "
                            f"post={prof['post_s']/bt*100:.1f}%, "
                            f"modality_per_window={prof['modality_load_s']/wcnt:.3f}s",
                            flush=True,
                        )
                    if worker_cfg.limit_batches is not None and batch_idx_global >= worker_cfg.limit_batches:
                        break

                if window_finished and result_writer is not None:
                    result_writer.flush_window(win_idx, window_preds)
                    if shard_pred_total is not None and shard_true_total is not None and shard_counts is not None:
                        pred_sum = float(sum(p for p, _ in window_preds.values()))
                        true_sum = float(sum(t for _, t in window_preds.values()))
                        shard_pred_total[win_idx] = pred_sum
                        shard_true_total[win_idx] = true_sum
                        shard_counts[win_idx] = len(window_preds)
                if worker_cfg.limit_batches is not None and batch_idx_global >= worker_cfg.limit_batches:
                    break
        else:
            for batch_idx in range(n_batches_loader):
                if worker_cfg.limit_batches is not None and batch_idx >= worker_cfg.limit_batches:
                    break
                t_batch0 = time.perf_counter()
                batch = next(batch_iter)
                t_data = time.perf_counter()
                prof["sample_build_s"] += float(t_data - t_batch0)
                d = _batch_to_device(batch, device)
                t_h2d = time.perf_counter()
                prof["h2d_s"] += float(t_h2d - t_data)
                sample_valid = batch.get("sample_valid")
                with autocast_ctx:
                    pv_pred = predict_pv_kW(model, d)
                pv_pred = pv_pred.float()
                t_gpu = time.perf_counter()
                prof["gpu_s"] += float(t_gpu - t_h2d)

                pred_np = pv_pred[:, worker_cfg.horizon_idx].detach().cpu().numpy()
                true_np = d["target_pv"][:, worker_cfg.horizon_idx].detach().cpu().numpy()
                bsz = pred_np.shape[0]
                global_base = batch_idx * worker_cfg.batch_size
                for i in range(bsz):
                    idx = global_base + i
                    if sample_valid is not None and float(sample_valid[i].item()) <= 0.0:
                        n_skipped += 1
                        continue
                    file_idx = idx // nw
                    win_idx = idx % nw
                    if file_idx >= n_files:
                        continue
                    pred_buf[file_idx][win_idx] = pred_np[i]
                    true_buf[file_idx][win_idx] = true_np[i]
                    filled[file_idx][win_idx] = True
                batch_idx_global += 1
                t_now = time.perf_counter()
                prof["batch_total_s"] += float(t_now - t_batch0)
                if (batch_idx_global % 10 == 0) or batch_idx_global == 1:
                    print(
                        f"{tag} batch {batch_idx_global}/{n_batches_loader} "
                        f"batch_time={t_now - t_batch0:.1f}s",
                        flush=True,
                    )

    print(
        f"{tag} inference done; batches={batch_idx_global} skipped_windows={n_skipped}",
        flush=True,
    )
    if batch_idx_global > 0:
        bt = max(float(prof["batch_total_s"]), 1e-9)
        wcnt = max(int(prof["windows"]), 1)
        print(
            f"{tag} profile-final: avg_batch={bt/batch_idx_global:.3f}s "
            f"sample_build={prof['sample_build_s']/bt*100:.1f}% "
            f"collate={prof['collate_s']/bt*100:.1f}% "
            f"h2d={prof['h2d_s']/bt*100:.1f}% "
            f"gpu={prof['gpu_s']/bt*100:.1f}% "
            f"post={prof['post_s']/bt*100:.1f}% "
            f"modality_per_window={prof['modality_load_s']/wcnt:.3f}s",
            flush=True,
        )

    if use_shared_sat_sky:
        assert result_writer is not None
        n_inv_written, station_rmse, n_station_points, station_path = result_writer.finish()
        print(f"{tag} wrote {n_inv_written} inverter CSVs (incremental)", flush=True)
        if world_size > 1:
            assert shard_dir is not None
            assert shard_pred_total is not None and shard_true_total is not None and shard_counts is not None
            shard_path = shard_dir / f"shard_{rank:03d}.npz"
            np.savez(
                shard_path,
                pred_total=shard_pred_total,
                true_total=shard_true_total,
                counts=shard_counts,
            )
            print(f"{tag} wrote shard {shard_path.name}", flush=True)
        elif station_path is not None:
            print(
                f"{tag} station_total RMSE={station_rmse:.6f} kW "
                f"(task={worker_cfg.task}, n_points={n_station_points}) -> {station_path}",
                flush=True,
            )
    else:
        assert pred_buf is not None and true_buf is not None and filled is not None
        n_inv_written = _write_per_inverter_csvs(
            out_inv_dir,
            inverter_names=inverter_names,
            n_files=n_files,
            nw=nw,
            pred_buf=pred_buf,
            true_buf=true_buf,
            filled=filled,
            time_utc=time_utc,
            time_bj=time_bj,
        )
        print(f"{tag} wrote {n_inv_written} inverter CSVs", flush=True)

        if world_size > 1:
            assert shard_dir is not None
            pred_total, true_total, counts = _partial_station_arrays(
                n_files=n_files,
                nw=nw,
                pred_buf=pred_buf,
                true_buf=true_buf,
                filled=filled,
            )
            shard_path = shard_dir / f"shard_{rank:03d}.npz"
            np.savez(
                shard_path,
                pred_total=pred_total,
                true_total=true_total,
                counts=counts,
            )
            print(f"{tag} wrote shard {shard_path.name}", flush=True)
        elif shard_dir is None:
            pred_total, true_total, counts = _partial_station_arrays(
                n_files=n_files,
                nw=nw,
                pred_buf=pred_buf,
                true_buf=true_buf,
                filled=filled,
            )
            station_path, station_rmse, n_station_points = _write_station_csv(
                out_inv_dir.parent,
                task=worker_cfg.task,
                nw=nw,
                pred_total=pred_total,
                true_total=true_total,
                counts=counts,
                time_utc=time_utc,
                time_bj=time_bj,
            )
            print(
                f"{tag} station_total RMSE={station_rmse:.6f} kW "
                f"(task={worker_cfg.task}, n_points={n_station_points}) -> {station_path}",
                flush=True,
            )

    return batch_idx_global, n_skipped


def _worker_entry(
    rank: int,
    world_size: int,
    gpu_ids: list[int],
    worker_cfg: InferWorkerConfig,
    all_files: list[Path],
    out_inv_dir: str,
    shard_dir: str,
) -> None:
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_ids[rank])
    shard_files = _shard_file_list(all_files, rank, world_size)
    _run_inference_shard(
        worker_cfg=worker_cfg,
        shard_files=shard_files,
        gpu_id=gpu_ids[rank],
        rank=rank,
        world_size=world_size,
        out_inv_dir=Path(out_inv_dir),
        shard_dir=Path(shard_dir) if world_size > 1 else None,
    )


def _build_48h_daily_batch(
    ds: PVDataset,
    df: pd.DataFrame,
    *,
    row_idx: int,
    dev_idx: torch.Tensor,
    query_utc: list[pd.Timestamp],
) -> dict:
    j = int(row_idx)
    x_idx = j + ds._x_tail_1d
    if int(x_idx.min()) < 0 or int(x_idx.max()) >= len(df):
        raise ValueError(f"history out of bounds at row_idx={row_idx}")

    sub_x = df.iloc[x_idx]
    hist_ts_utc = ds._to_utc_timestamps(list(sub_x["collectTime"]))
    time0_utc = hist_ts_utc[-1]

    pow_x = pd.to_numeric(sub_x["final_power"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    if "kt" in sub_x.columns:
        kt_np = pd.to_numeric(sub_x["kt"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    else:
        kt_np = np.zeros_like(pow_x, dtype=np.float32)

    if "kt_mask" in sub_x.columns:
        kt_mask_np = pd.to_numeric(sub_x["kt_mask"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32)
    elif INVERTER_STATE_COL in sub_x.columns:
        inv_x = pd.to_numeric(sub_x[INVERTER_STATE_COL], errors="coerce").fillna(0).to_numpy(dtype=np.int32)
        kt_mask_np = (inv_x == VALID_STATE).astype(np.float32)
    else:
        kt_mask_np = np.ones_like(pow_x, dtype=np.float32)

    mean_pow_x = float(np.mean(pow_x)) if len(pow_x) else 0.0
    if "p_mean" in sub_x.columns:
        p_mean_np = pd.to_numeric(sub_x["p_mean"], errors="coerce").fillna(mean_pow_x).to_numpy(dtype=np.float32)
        p_mean = float(p_mean_np[-1]) if len(p_mean_np) else max(mean_pow_x, 1e-6)
    else:
        p_mean = max(mean_pow_x, 1e-6)

    pv_solar = compute_solar_features(hist_ts_utc, latitude=34.69984, longitude=112.28440)
    pv_timefeats = solar_features_encoder(pv_solar)
    pv_dtime = delta_time_encoder(hist_ts_utc, time0_utc)
    pv_timefeats = torch.cat([pv_timefeats, pv_dtime.unsqueeze(1)], dim=1).unsqueeze(0)

    query_solar = compute_solar_features(query_utc, latitude=34.69984, longitude=112.28440)
    forecast_timefeats = solar_features_encoder(query_solar)
    forecast_dtime = delta_time_encoder(query_utc, time0_utc)
    forecast_timefeats = torch.cat([forecast_timefeats, forecast_dtime.unsqueeze(1)], dim=1).unsqueeze(0)

    nwp_out = interpolate_nwp_features(ds._nwp_solar_blocks, ds._nwp_wind_blocks, query_utc)
    nwp_tensor = None if nwp_out is None else torch.from_numpy(np.asarray(nwp_out, dtype=np.float32)).unsqueeze(0)

    q_utc_index = pd.DatetimeIndex(query_utc)
    truth_series = df.set_index("collectTime_utc")["final_power"].reindex(q_utc_index)
    if truth_series.isna().any():
        raise ValueError("missing truth final_power for next-day 288 points")
    target_pv = torch.from_numpy(pd.to_numeric(truth_series, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)).unsqueeze(0)

    if "p_cs" in df.columns:
        pcs_series = df.set_index("collectTime_utc")["p_cs"].reindex(q_utc_index)
        if pcs_series.isna().any():
            target_p_cs = torch.ones(1, len(query_utc), dtype=torch.float32)
        else:
            target_p_cs = torch.from_numpy(pd.to_numeric(pcs_series, errors="coerce").fillna(1.0).to_numpy(dtype=np.float32)).unsqueeze(0)
    else:
        target_p_cs = torch.ones(1, len(query_utc), dtype=torch.float32)

    sat_bundle = ds._load_sat_sky_modality(time0_utc)
    sat_tensor_raw = sat_bundle.get("sat_tensor")
    sat_timefeats_raw = sat_bundle.get("sat_timefeats")
    sat_valid_raw = sat_bundle.get("sat_valid")
    skimg_tensor_raw = sat_bundle.get("skimg_tensor")
    skimg_timefeats_raw = sat_bundle.get("skimg_timefeats")
    skimg_valid_raw = sat_bundle.get("skimg_valid")

    sat_tensor = None if sat_tensor_raw is None else sat_tensor_raw.unsqueeze(0)
    sat_timefeats = None if sat_timefeats_raw is None else sat_timefeats_raw.unsqueeze(0)
    sat_valid_mask = (
        None
        if sat_valid_raw is None
        else torch.tensor([float(sat_valid_raw.item())], dtype=torch.float32)
    )
    skimg_tensor = None if skimg_tensor_raw is None else skimg_tensor_raw.unsqueeze(0)
    skimg_timefeats = None if skimg_timefeats_raw is None else skimg_timefeats_raw.unsqueeze(0)
    skimg_valid_mask = (
        None
        if skimg_valid_raw is None
        else torch.tensor([float(skimg_valid_raw.item())], dtype=torch.float32)
    )

    return {
        "dev_idx": dev_idx.unsqueeze(0),
        "kt": torch.from_numpy(kt_np).unsqueeze(0).unsqueeze(0),
        "kt_mask": torch.from_numpy(kt_mask_np).unsqueeze(0).unsqueeze(0),
        "pv_timefeats": pv_timefeats,
        "forecast_timefeats": forecast_timefeats,
        "target_pv": target_pv,
        "target_p_cs": target_p_cs,
        "p_mean": torch.tensor([p_mean], dtype=torch.float32),
        "sat_tensor": sat_tensor,
        "sat_timefeats": sat_timefeats,
        "sat_valid_mask": sat_valid_mask,
        "skimg_tensor": skimg_tensor,
        "skimg_timefeats": skimg_timefeats,
        "skimg_valid_mask": skimg_valid_mask,
        "nwp_tensor": nwp_tensor,
    }


def _run_inference_48h_daily(
    *,
    checkpoint: Path,
    dataset_config: str,
    out_dir: Path,
    device: torch.device,
    max_inverters: int | None,
) -> None:
    ds = PVDataset(**_dataset_kwargs_for_infer(dataset_config, "test", test_stride_min_override=_DEFAULT_STRIDE_MIN))
    all_files = list(ds.sample_files)
    if max_inverters is not None:
        all_files = all_files[: int(max_inverters)]
    if not all_files:
        raise RuntimeError("No inverter CSV files found for 48h daily inference")

    model = _load_model(checkpoint, device, ds.devDn_list)
    out_daily_dir = out_dir / "daily_48h" / "inverters"
    out_daily_dir.mkdir(parents=True, exist_ok=True)
    station_out_path = out_dir / "daily_48h" / "station_total_48h.csv"
    station_daily_sum: dict[str, dict[str, object]] = {}
    station_mismatch_days = 0

    tag = "[inference_luoyang2026 48h]"
    print(
        f"{tag} device={device} inverters={len(all_files)} "
        f"target_day_bj=[{_TEST_RANGE_START_BJ}..{_TEST_RANGE_END_BJ}]",
        flush=True,
    )
    done_files = 0
    total_days = 0
    total_skipped = 0

    with torch.no_grad():
        for file_idx, p in enumerate(all_files, start=1):
            k = p.resolve().as_posix()
            if k in ds._csv_cache:
                df = ds._csv_cache[k].copy()
            else:
                df = pd.read_csv(p)
            if "collectTime" not in df.columns:
                print(f"{tag} skip {p.name}: missing collectTime", flush=True)
                continue
            df["collectTime_utc"] = pd.to_datetime(df["collectTime"], errors="coerce", utc=True)
            df = df.dropna(subset=["collectTime_utc"]).sort_values("collectTime_utc").reset_index(drop=True)
            if df.empty:
                print(f"{tag} skip {p.name}: no valid timestamps", flush=True)
                continue

            ts_bj = df["collectTime_utc"].dt.tz_convert("Asia/Shanghai")
            anchor_mask = (ts_bj.dt.hour == 9) & (ts_bj.dt.minute == 0) & (ts_bj.dt.second == 0)
            anchor_rows = np.flatnonzero(anchor_mask.to_numpy(dtype=bool))
            if anchor_rows.size == 0:
                print(f"{tag} {p.name}: no 09:00 BJ anchors", flush=True)
                continue

            dev_dn = p.stem.replace("_", "=")
            dev_idx = torch.tensor(ds._dev_idx_map[dev_dn], dtype=torch.long)
            safe_inv = dev_dn.replace("=", "_").replace("/", "_")
            out_path = out_daily_dir / f"{safe_inv}.csv"
            day_to_anchor: dict[str, int] = {}
            for r in anchor_rows.tolist():
                day_key = str(ts_bj.iloc[int(r)].date())
                if day_key not in day_to_anchor:
                    day_to_anchor[day_key] = int(r)

            success_days = 0
            skipped_days = 0
            in_range_days = 0
            with open(out_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                for anchor_day, r in sorted(day_to_anchor.items()):
                    t_anchor_bj = ts_bj.iloc[int(r)]
                    target_day_start_bj = t_anchor_bj.normalize() + pd.Timedelta(days=1)
                    target_day_date = pd.Timestamp(target_day_start_bj).date()
                    if not (_TEST_RANGE_START_BJ <= target_day_date <= _TEST_RANGE_END_BJ):
                        continue
                    in_range_days += 1
                    query_bj = pd.date_range(start=target_day_start_bj, periods=288, freq="5min", tz="Asia/Shanghai")
                    query_utc_index = query_bj.tz_convert("UTC")
                    query_utc = [pd.Timestamp(t) for t in query_utc_index]

                    try:
                        batch = _build_48h_daily_batch(
                            ds,
                            df,
                            row_idx=int(r),
                            dev_idx=dev_idx,
                            query_utc=query_utc,
                        )
                    except Exception as e:
                        skipped_days += 1
                        print(f"{tag} {p.name} day={anchor_day} skipped: {e}", flush=True)
                        continue

                    d = _batch_to_device(batch, device)
                    pred_kW = predict_pv_kW(model, d).float()
                    pred_row = pred_kW[0].detach().cpu().numpy()
                    true_row = batch["target_pv"][0].detach().cpu().numpy()
                    if pred_row.shape[0] != 288 or true_row.shape[0] != 288:
                        skipped_days += 1
                        print(
                            f"{tag} {p.name} day={anchor_day} skipped: invalid lengths "
                            f"pred={pred_row.shape[0]} true={true_row.shape[0]}",
                            flush=True,
                        )
                        continue

                    utc_list = [pd.Timestamp(t).isoformat() for t in query_utc_index]
                    bj_list = [pd.Timestamp(t).isoformat() for t in query_bj]
                    time_pairs = [f"{u}|{b}" for u, b in zip(utc_list, bj_list)]
                    target_day = str(target_day_start_bj.date())

                    writer.writerow([f"time_utc_bj|{target_day}"] + time_pairs)
                    writer.writerow([f"pv_pred_kW|{target_day}"] + [f"{float(v):.6f}" for v in pred_row.tolist()])
                    writer.writerow([f"pv_true_kW|{target_day}"] + [f"{float(v):.6f}" for v in true_row.tolist()])

                    pred_np = pred_row.astype(np.float64, copy=False)
                    true_np = true_row.astype(np.float64, copy=False)
                    agg = station_daily_sum.get(target_day)
                    if agg is None:
                        station_daily_sum[target_day] = {
                            "time_pairs": list(time_pairs),
                            "pred_sum": pred_np.copy(),
                            "true_sum": true_np.copy(),
                            "count": 1,
                        }
                    else:
                        agg_time_pairs = agg["time_pairs"]
                        agg_pred = agg["pred_sum"]
                        agg_true = agg["true_sum"]
                        if (
                            not isinstance(agg_time_pairs, list)
                            or not isinstance(agg_pred, np.ndarray)
                            or not isinstance(agg_true, np.ndarray)
                            or len(agg_time_pairs) != len(time_pairs)
                        ):
                            station_mismatch_days += 1
                            print(
                                f"{tag} station aggregate mismatch day={target_day} file={p.name}: "
                                "inconsistent schema, skip this inverter/day for station sum",
                                flush=True,
                            )
                        else:
                            agg_pred += pred_np
                            agg_true += true_np
                            agg["count"] = int(agg.get("count", 0)) + 1

                    success_days += 1

            total_days += int(in_range_days)
            total_skipped += int(skipped_days)
            done_files += 1
            print(
                f"{tag} {file_idx}/{len(all_files)} {p.name}: anchors_all={len(day_to_anchor)} "
                f"in_range={in_range_days} "
                f"success={success_days} skipped={skipped_days}",
                flush=True,
            )

    station_days_written = 0
    with open(station_out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        for target_day in sorted(station_daily_sum.keys()):
            rec = station_daily_sum[target_day]
            time_pairs = rec["time_pairs"]
            pred_sum = rec["pred_sum"]
            true_sum = rec["true_sum"]
            if not isinstance(time_pairs, list) or not isinstance(pred_sum, np.ndarray) or not isinstance(true_sum, np.ndarray):
                continue
            writer.writerow([f"time_utc_bj|{target_day}"] + list(time_pairs))
            writer.writerow([f"pv_pred_kW|{target_day}"] + [f"{float(v):.6f}" for v in pred_sum.tolist()])
            writer.writerow([f"pv_true_kW|{target_day}"] + [f"{float(v):.6f}" for v in true_sum.tolist()])
            station_days_written += 1

    print(
        f"{tag} done: inverters={done_files}/{len(all_files)} total_anchor_days={total_days} "
        f"skipped_days={total_skipped} station_days={station_days_written} "
        f"station_mismatch_days={station_mismatch_days} output_dir={out_daily_dir} "
        f"station_file={station_out_path}",
        flush=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Luoyang 2026 test-set inference (15m / 4h single horizon)."
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=("15m", "4h", "48h"),
        help=(
            "Forecast mode: 15m(index 0), 4h(index 15), "
            "48h(daily mode: BJ 09:00 anchor -> next-day 288x5min output)."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a trained checkpoint (.pt) from train_vit_luoyang2026.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output root directory (default: inference_results/luoyang2026_<task>).",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=_DEFAULT_DATASET_CONFIG,
        help=f"Dataset config under config/datasets/ (default: {_DEFAULT_DATASET_CONFIG}).",
    )
    parser.add_argument(
        "--stride_min",
        type=int,
        default=_DEFAULT_STRIDE_MIN,
        help=(
            "Test-anchor stride in minutes (default: 5 = one point every 5 min). "
            "Must be a positive multiple of csv_interval_min. "
            "Use 1200 to match sparse training/val eval in conf_luoyang_2026.yaml."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help=(
            "DataLoader workers per GPU process (default: 0). Keep at 0: zarr + fork deadlocks."
        ),
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs for parallel inference (default: 1). Shards inverters across GPUs.",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Comma-separated physical GPU ids (e.g. 0,1,2,3). Default: 0..num_gpus-1.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Single-GPU only: cuda / cuda:N / cpu (ignored when --num_gpus > 1).",
    )
    parser.add_argument(
        "--max_inverters",
        type=int,
        default=None,
        help="Optional: process only the first N inverters (sorted by filename).",
    )
    parser.add_argument(
        "--limit_batches",
        type=int,
        default=None,
        help="Optional: stop after N batches per GPU shard (debug only).",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help=(
            "Optional cache directory built by preprocess_luoyang2026test.py. "
            "If only windows/ exists, use sat/sky cache. If pv_batches/ also exists, "
            "use full-cache mode and bypass runtime PV sample building."
        ),
    )
    parser.add_argument(
        "--profile_every",
        type=int,
        default=100,
        help="Print averaged pipeline time breakdown every N batches (<=0 disables).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.stride_min <= 0:
        raise ValueError(f"--stride_min must be positive (got {args.stride_min})")
    cache_dir: Path | None = None
    cache_manifest: dict | None = None
    cache_full = False
    if args.cache_dir is not None:
        cache_dir = Path(args.cache_dir).expanduser().resolve()
        windows_dir = cache_dir / "windows"
        if not windows_dir.is_dir():
            raise FileNotFoundError(f"cache windows dir not found: {windows_dir}")
        cache_manifest = _load_cache_manifest(cache_dir)
        cache_full = (cache_dir / "pv_batches").is_dir()

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    out_dir = _resolve_output_dir(args.output_dir, args.task)
    out_inv_dir = out_dir / "inverters"
    shard_dir = out_dir / "shards"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_inv_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "48h":
        gpu_ids = _parse_gpu_ids(args.num_gpus, args.gpu_ids)
        if len(gpu_ids) != 1:
            raise ValueError("48h daily mode currently supports single-GPU only (--num_gpus 1).")
        if args.cache_dir is not None:
            print(
                "[inference_luoyang2026] WARNING: --cache_dir is ignored in 48h daily mode.",
                flush=True,
            )
        device_override = args.device
        if device_override is None and torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu_ids[0]}")
        elif device_override is not None:
            device = torch.device(device_override)
        else:
            device = torch.device("cpu")
        print(
            f"[inference_luoyang2026] task=48h checkpoint={ckpt_path} output_dir={out_dir} "
            f"device={device} max_inverters={args.max_inverters}",
            flush=True,
        )
        _run_inference_48h_daily(
            checkpoint=ckpt_path,
            dataset_config=args.dataset_config,
            out_dir=out_dir,
            device=device,
            max_inverters=args.max_inverters,
        )
        return

    horizon_idx = _horizon_idx(args.task)
    gpu_ids = _parse_gpu_ids(args.num_gpus, args.gpu_ids)
    world_size = len(gpu_ids)

    all_files = _list_infer_sample_files(args.dataset_config, max_inverters=args.max_inverters)
    if not all_files:
        raise RuntimeError("No inverter CSV files found for inference")
    if cache_dir is not None:
        cfg_path = _resolve_named_config(_DATASETS_CONFIG_DIR, args.dataset_config, "dataset-config")
        cfg_sha = _cfg_sha256(cfg_path)
        assert cache_manifest is not None
        man_sha = str((cache_manifest.get("dataset_config") or {}).get("sha256", ""))
        if man_sha and man_sha != cfg_sha:
            raise ValueError(
                f"cache dataset-config hash mismatch: cache={man_sha[:12]} expected={cfg_sha[:12]} "
                f"(config={cfg_path})"
            )
        man_stride = cache_manifest.get("stride_min")
        if man_stride is not None and int(man_stride) != int(args.stride_min):
            raise ValueError(f"cache stride mismatch: cache={man_stride} expected={args.stride_min}")
        man_bs = cache_manifest.get("batch_size")
        if man_bs is not None and int(man_bs) != int(args.batch_size):
            raise ValueError(f"cache batch_size mismatch: cache={man_bs} expected={args.batch_size}")
        man_num_inv = cache_manifest.get("num_inverters")
        if man_num_inv is not None and int(man_num_inv) != int(len(all_files)):
            raise ValueError(
                f"cache num_inverters mismatch: cache={man_num_inv} expected={len(all_files)} "
                "(check --max_inverters and preprocessing args)"
            )
        if cache_full and world_size > 1:
            raise ValueError("Full cache mode currently supports single-GPU inference only.")

    worker_cfg = InferWorkerConfig(
        task=args.task,
        checkpoint=str(ckpt_path),
        output_dir=str(out_dir),
        dataset_config=args.dataset_config,
        stride_min=args.stride_min,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_inverters=args.max_inverters,
        limit_batches=args.limit_batches,
        horizon_idx=horizon_idx,
        cache_dir=None if cache_dir is None else str(cache_dir),
        cache_full=bool(cache_full),
        profile_every=int(args.profile_every),
    )

    print(
        f"[inference_luoyang2026] task={args.task} checkpoint={ckpt_path} "
        f"output_dir={out_dir} inverters={len(all_files)} num_gpus={world_size} "
        f"gpu_ids={gpu_ids} stride_min={args.stride_min} batch_size={args.batch_size} "
        f"cache_dir={cache_dir} cache_full={cache_full}",
        flush=True,
    )
    if args.stride_min == _DEFAULT_STRIDE_MIN and args.limit_batches is None:
        est_samples = len(all_files) * 9470  # approximate for stride=5
        print(
            f"[inference_luoyang2026] NOTE: dense stride_min=5 => ~{est_samples} samples total; "
            f"multi-GPU splits inverters (~{est_samples // world_size} samples/GPU).",
            flush=True,
        )

    if world_size == 1:
        device_override = args.device
        if device_override is None and torch.cuda.is_available():
            gpu_id = gpu_ids[0]
        elif device_override is not None and torch.device(device_override).type == "cuda":
            dev = torch.device(device_override)
            gpu_id = int(dev.index if dev.index is not None else 0)
        else:
            gpu_id = 0

        _run_inference_shard(
            worker_cfg=worker_cfg,
            shard_files=all_files,
            gpu_id=gpu_id,
            rank=0,
            world_size=1,
            out_inv_dir=out_inv_dir,
            shard_dir=None,
            device_override=device_override,
        )
        return

    if args.device is not None:
        print(
            "[inference_luoyang2026] WARNING: --device is ignored when --num_gpus > 1",
            flush=True,
        )

    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    mp.set_start_method("spawn", force=True)
    processes: list[mp.Process] = []
    for rank in range(world_size):
        p = mp.Process(
            target=_worker_entry,
            args=(
                rank,
                world_size,
                gpu_ids,
                worker_cfg,
                all_files,
                str(out_inv_dir),
                str(shard_dir),
            ),
        )
        p.start()
        processes.append(p)

    exit_codes = []
    for p in processes:
        p.join()
        exit_codes.append(p.exitcode)

    if any(code != 0 for code in exit_codes):
        raise RuntimeError(f"One or more GPU workers failed: exit codes={exit_codes}")

    probe_ds = PVDataset(
        **_dataset_kwargs_for_infer(
            args.dataset_config,
            "test",
            test_stride_min_override=args.stride_min,
            sample_file_subset=all_files[:1],
        )
    )
    nw = probe_ds._num_test_windows
    t0_refs = list(probe_ds._test_last_x_time_ref or [])
    time_utc, time_bj = _forecast_times_for_windows(
        t0_refs,
        horizon_idx=horizon_idx,
        output_interval_min=int(probe_ds.pv_output_interval_min),
    )

    station_path, station_rmse, n_station_points = _merge_station_shards(
        shard_dir,
        out_dir=out_dir,
        task=args.task,
        nw=nw,
        time_utc=time_utc,
        time_bj=time_bj,
    )
    print(
        f"[inference_luoyang2026] merged {world_size} GPU shards -> {station_path}",
        flush=True,
    )
    print(
        f"[inference_luoyang2026] station_total RMSE={station_rmse:.6f} kW "
        f"(task={args.task}, n_points={n_station_points})",
        flush=True,
    )


if __name__ == "__main__":
    main()
