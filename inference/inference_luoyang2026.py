"""
Luoyang 2026 test-set inference (15m / 4h single-horizon).

For each inverter and each test anchor window, run the trained model and save:
  * per-inverter CSV: forecast time (t0+15m or t0+4h), PV prediction, PV truth
  * station-total CSV: same times with summed pred/truth across inverters
  * prints station-level total-power RMSE at the end

By default ``--stride_min 5`` rolls one anchor every 5 minutes (dense coverage).
Use ``--stride_min 1200`` to match the sparse test stride in conf_luoyang_2026.yaml.

Multi-GPU: shard inverters across GPUs (one process per GPU). Each GPU loads only
its CSV subset and writes inverter CSVs; rank 0 merges station totals.

Quick run examples:
  python inference/inference_luoyang2026.py --task 15m --checkpoint checkpoint_2026_fixedhuber/pv_forecast_vit_best_task_15m_gpu0.pt
  CUDA_VISIBLE_DEVICES=0,1,2,3 python inference/inference_luoyang2026.py --task 15m --checkpoint ... --num_gpus 4
  python inference/inference_luoyang2026.py --task 4h --checkpoint checkpoint_2026_fixedhuber/pv_forecast_vit_best_task_4h_gpu6.pt
"""

from __future__ import annotations

import argparse
import contextlib
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

from dataloader.luoyang_2026_zarr import PVDataset, collate_batched, list_csv_files  # noqa: E402
from models.models import pv_forecasting_model_vit_imgs  # noqa: E402
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

    if worker_cfg.num_workers > 0:
        print(
            f"{tag} WARNING: num_workers>0 can hang on first batch (zarr + fork). "
            f"Prefer --num_workers 0.",
            flush=True,
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=worker_cfg.batch_size,
        shuffle=False,
        collate_fn=collate_batched,
        num_workers=worker_cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(worker_cfg.num_workers > 0),
    )

    pred_buf = {i: np.full(nw, np.nan, dtype=np.float32) for i in range(n_files)}
    true_buf = {i: np.full(nw, np.nan, dtype=np.float32) for i in range(n_files)}
    filled = {i: np.zeros(nw, dtype=bool) for i in range(n_files)}

    if device.type == "cuda":
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    else:
        autocast_ctx = contextlib.nullcontext()

    n_batches = (n_total + worker_cfg.batch_size - 1) // worker_cfg.batch_size
    global_idx = 0
    n_skipped = 0
    print(f"{tag} Running inference...", flush=True)
    batch_iter = iter(test_loader)
    with torch.no_grad():
        for batch_idx in range(n_batches):
            if worker_cfg.limit_batches is not None and batch_idx >= worker_cfg.limit_batches:
                break
            t_batch0 = time.perf_counter()
            batch = next(batch_iter)
            t_data = time.perf_counter()
            if batch_idx == 0:
                print(
                    f"{tag} first batch ready in {t_data - t_batch0:.1f}s "
                    f"(data loading; GPU forward follows)",
                    flush=True,
                )
            d = _batch_to_device(batch, device)
            sample_valid = batch.get("sample_valid")
            with autocast_ctx:
                pv_pred = predict_pv_kW(model, d)
            pv_pred = pv_pred.float()
            t_gpu = time.perf_counter()
            if batch_idx == 0:
                print(f"{tag} first batch GPU forward in {t_gpu - t_data:.3f}s", flush=True)

            pred_np = pv_pred[:, worker_cfg.horizon_idx].detach().cpu().numpy()
            true_np = d["target_pv"][:, worker_cfg.horizon_idx].detach().cpu().numpy()
            bsz = pred_np.shape[0]
            for i in range(bsz):
                idx = global_idx + i
                if sample_valid is not None and float(sample_valid[i].item()) <= 0.0:
                    n_skipped += 1
                    continue
                file_idx = idx // nw
                win_idx = idx % nw
                if file_idx >= n_files:
                    raise RuntimeError(f"file_idx {file_idx} out of range {n_files}")
                pred_buf[file_idx][win_idx] = pred_np[i]
                true_buf[file_idx][win_idx] = true_np[i]
                filled[file_idx][win_idx] = True

            global_idx += bsz
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                t_now = time.perf_counter()
                print(
                    f"{tag} batch {batch_idx + 1}/{n_batches} "
                    f"processed_samples={global_idx} batch_time={t_now - t_batch0:.1f}s",
                    flush=True,
                )

    print(
        f"{tag} inference done; processed={global_idx} skipped_windows={n_skipped}",
        flush=True,
    )

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

    return global_idx, n_skipped


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Luoyang 2026 test-set inference (15m / 4h single horizon)."
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=("15m", "4h", "48h"),
        help="Forecast horizon: 15m (index 0), 4h (index 15), 48h (not implemented).",
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.task == "48h":
        raise NotImplementedError("48h inference is not implemented yet.")
    if args.stride_min <= 0:
        raise ValueError(f"--stride_min must be positive (got {args.stride_min})")

    horizon_idx = _horizon_idx(args.task)
    gpu_ids = _parse_gpu_ids(args.num_gpus, args.gpu_ids)
    world_size = len(gpu_ids)

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    out_dir = _resolve_output_dir(args.output_dir, args.task)
    out_inv_dir = out_dir / "inverters"
    shard_dir = out_dir / "shards"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_inv_dir.mkdir(parents=True, exist_ok=True)

    all_files = _list_infer_sample_files(args.dataset_config, max_inverters=args.max_inverters)
    if not all_files:
        raise RuntimeError("No inverter CSV files found for inference")

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
    )

    print(
        f"[inference_luoyang2026] task={args.task} checkpoint={ckpt_path} "
        f"output_dir={out_dir} inverters={len(all_files)} num_gpus={world_size} "
        f"gpu_ids={gpu_ids} stride_min={args.stride_min} batch_size={args.batch_size}",
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
