"""Standalone trainer for Luoyang 2026 total-station tasks.

Quick run examples:
- 15m task:
  python training/train_vit_luoyang2026total.py --task 15m --dataset-config conf_luoyang_2026_15m.yaml --config conf_train.yaml
- 4h task:
  python training/train_vit_luoyang2026total.py --task 4h --dataset-config conf_luoyang_2026_4h.yaml --config conf_train.yaml
- 48h task:
  python training/train_vit_luoyang2026total.py --task 48h --dataset-config conf_luoyang_2026_48h.yaml --config conf_train.yaml
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import yaml
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"
_TRAIN_CONFIG_DIR = _CONFIG_DIR / "train"
_DATASETS_CONFIG_DIR = _CONFIG_DIR / "datasets"
_DEFAULT_TRAIN_CONF_NAME = "conf_train.yaml"
_DEFAULT_DATASET_CONF_NAME = "conf_luoyang_2026.yaml"
sys.path.insert(0, str(_PROJECT_ROOT))

from dataloader.luoyang_2026total_zarr import PVDataset, collate_batched
from models.models import pv_forecasting_model_vit_imgs

# For point tasks, keep index mapping only as legacy fallback when target has multiple steps.
# If dataloader already outputs a single future point (pv_output_len=1), we always supervise index 0.
TASK_TO_INDEX = {"15m": 0, "4h": 15, "48h": None}
OPTIONAL_MODALITY_KEYS = (
    "sat_tensor",
    "sat_timefeats",
    "sat_valid_mask",
    "skimg_tensor",
    "skimg_timefeats",
    "skimg_valid_mask",
    "nwp_tensor",
    "nwp_history",
    "nwp_forecast_history",
)


def _gpu_id_for_checkpoint() -> int:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return torch.cuda.current_device()
    s = raw.strip()
    if not s:
        return torch.cuda.current_device()
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return torch.cuda.current_device()
    return int(parts[0]) if parts[0].isdigit() else torch.cuda.current_device()


def _shape_of(x) -> str:
    if x is None:
        return "None"
    if isinstance(x, torch.Tensor):
        return str(tuple(x.shape))
    return f"non-tensor:{type(x).__name__}"


def _require_tensor(batch: dict, key: str) -> torch.Tensor:
    if key not in batch:
        raise KeyError(f"Batch is missing required key {key!r}")
    val = batch[key]
    if val is None:
        raise ValueError(
            f"Batch key {key!r} is None. '{key}' is required. "
            "For Luoyang 2026 training, kt/kt_mask/pv_timefeats/forecast_timefeats/targets are mandatory."
        )
    if not isinstance(val, torch.Tensor):
        raise TypeError(f"Batch key {key!r} must be torch.Tensor, got {type(val).__name__}")
    return val


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {
        "device_id": _require_tensor(batch, "dev_idx").to(device),
        "kt": _require_tensor(batch, "kt").to(device),
        "kt_mask": _require_tensor(batch, "kt_mask").to(device),
        "pv_timefeats": _require_tensor(batch, "pv_timefeats").to(device),
        "forecast_timefeats": _require_tensor(batch, "forecast_timefeats").to(device),
        "target_pv": _require_tensor(batch, "target_pv").to(device),
        "target_p_cs": _require_tensor(batch, "target_p_cs").to(device),
        "p_mean": _require_tensor(batch, "p_mean").to(device),
    }
    for key in OPTIONAL_MODALITY_KEYS:
        v = batch.get(key)
        if v is None:
            out[key] = None
        elif isinstance(v, torch.Tensor):
            out[key] = v.to(device)
        else:
            raise TypeError(f"Optional key {key!r} must be torch.Tensor or None, got {type(v).__name__}")
    return out


def forward_vit(model: nn.Module, d: dict) -> torch.Tensor:
    try:
        return model(
            d["device_id"],
            d["kt"],
            pv_mask=d["kt_mask"],
            pv_timefeats=d["pv_timefeats"],
            forecast_timefeats=d["forecast_timefeats"],
            sat_tensor=d["sat_tensor"],
            sat_timefeats=d["sat_timefeats"],
            skimg_tensor=d["skimg_tensor"],
            skimg_timefeats=d["skimg_timefeats"],
            sat_valid_mask=d["sat_valid_mask"],
            skimg_valid_mask=d["skimg_valid_mask"],
            nwp_tensor=d["nwp_tensor"],
            nwp_history=d["nwp_history"],
            nwp_forecast_history=d["nwp_forecast_history"],
        )
    except Exception as e:
        details = ", ".join(
            [
                f"kt={_shape_of(d.get('kt'))}",
                f"kt_mask={_shape_of(d.get('kt_mask'))}",
                f"pv_timefeats={_shape_of(d.get('pv_timefeats'))}",
                f"forecast_timefeats={_shape_of(d.get('forecast_timefeats'))}",
                f"sat_tensor={_shape_of(d.get('sat_tensor'))}",
                f"sat_timefeats={_shape_of(d.get('sat_timefeats'))}",
                f"sat_valid_mask={_shape_of(d.get('sat_valid_mask'))}",
                f"skimg_tensor={_shape_of(d.get('skimg_tensor'))}",
                f"skimg_timefeats={_shape_of(d.get('skimg_timefeats'))}",
                f"skimg_valid_mask={_shape_of(d.get('skimg_valid_mask'))}",
                f"nwp_tensor={_shape_of(d.get('nwp_tensor'))}",
                f"nwp_history={_shape_of(d.get('nwp_history'))}",
                f"nwp_forecast_history={_shape_of(d.get('nwp_forecast_history'))}",
            ]
        )
        raise RuntimeError(f"forward_vit failed with tensor shapes: {details}") from e


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            k: v.detach().clone().float()
            for k, v in model.state_dict().items()
            if v.is_floating_point()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)

    @contextlib.contextmanager
    def apply(self, model: nn.Module):
        msd = model.state_dict()
        backup = {k: msd[k].clone() for k in self.shadow}
        for k, v in self.shadow.items():
            msd[k].copy_(v.to(msd[k].dtype))
        try:
            yield
        finally:
            msd = model.state_dict()
            for k, v in backup.items():
                msd[k].copy_(v)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow

    def full_state_dict(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Complete state dict for checkpointing: model state with EMA weights overlaid.

        The shadow alone only holds floating-point entries, so saving it directly
        produces a state dict that fails strict ``load_state_dict`` (missing
        non-float buffers such as BatchNorm ``num_batches_tracked``).
        """
        sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for k, v in self.shadow.items():
            sd[k] = v.detach().clone().to(sd[k].dtype)
        return sd


@dataclass
class TaskMetrics:
    loss: float
    rmse: float
    mae: float


def _task_loss_and_vectors(
    pv_pred: torch.Tensor,
    target_pv: torch.Tensor,
    criterion: nn.Module,
    task: str,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, int | None]:
    idx = TASK_TO_INDEX[task]
    if idx is None:
        loss = criterion(pv_pred, target_pv)
        pred_np = pv_pred.detach().cpu().float().numpy().reshape(-1)
        tgt_np = target_pv.detach().cpu().float().numpy().reshape(-1)
        used_idx = None
    else:
        # For point tasks (15m/4h), supervise the point that dataloader actually outputs.
        # When target has a single step, this is always index 0 regardless of task name.
        t_out = int(target_pv.shape[1]) if target_pv.ndim >= 2 else 1
        if t_out <= 1:
            used_idx = 0
        else:
            used_idx = int(min(max(idx, 0), t_out - 1))
        if pv_pred.ndim == 2 and pv_pred.shape[1] > used_idx:
            pred_pt = pv_pred[:, used_idx]
        else:
            pred_pt = pv_pred.reshape(-1)
        tgt_pt = target_pv[:, used_idx]
        loss = criterion(pred_pt, tgt_pt)
        pred_np = pred_pt.detach().cpu().float().numpy()
        tgt_np = tgt_pt.detach().cpu().float().numpy()
    return loss, pred_np, tgt_np, used_idx


def _mae_rmse(pred: np.ndarray, tgt: np.ndarray) -> tuple[float, float]:
    diff = pred - tgt
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    return mae, rmse


def train_one_epoch_task(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    task: str,
    max_batches: int | None = None,
    ema: ModelEMA | None = None,
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    sat_valid_sum = 0.0
    skimg_valid_sum = 0.0
    total_batches = len(loader)
    total_iters = total_batches if max_batches is None else min(total_batches, max_batches)
    print("number of batches:", total_batches)
    t0 = time.time()
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        d = _batch_to_device(batch, device)
        bsz = int(d["device_id"].size(0))
        optimizer.zero_grad()
        kt_pred = forward_vit(model, d)

        pv_pred = kt_pred * d["target_p_cs"] * d["p_mean"].unsqueeze(1)
        loss, _, _, _ = _task_loss_and_vectors(pv_pred, d["target_pv"], criterion, task)
        if torch.isnan(loss) and not getattr(train_one_epoch_task, "_nan_reported", False):
            train_one_epoch_task._nan_reported = True
            print(
                f"\n[NaN-debug] batch={batch_idx} "
                f"kt_pred={torch.isnan(kt_pred).any().item()} "
                f"target_p_cs={torch.isnan(d['target_p_cs']).any().item()} "
                f"p_mean={torch.isnan(d['p_mean']).any().item()} "
                f"target_pv={torch.isnan(d['target_pv']).any().item()} "
                f"sky_valid={float(d['skimg_valid_mask'].sum().item()) if isinstance(d.get('skimg_valid_mask'), torch.Tensor) else 'NA'}"
            )
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model)
        total_loss += float(loss.item()) * bsz
        total_samples += bsz
        if isinstance(d.get("sat_valid_mask"), torch.Tensor):
            sat_valid_sum += float(d["sat_valid_mask"].float().sum().item())
        if isinstance(d.get("skimg_valid_mask"), torch.Tensor):
            skimg_valid_sum += float(d["skimg_valid_mask"].float().sum().item())
        done = batch_idx + 1
        elapsed = max(time.time() - t0, 1e-6)
        it_per_sec = done / elapsed
        eta_sec = max(total_iters - done, 0) / max(it_per_sec, 1e-6)
        sys.stdout.write(
            f"\r[train] iter {done}/{total_iters} "
            f"({100.0 * done / max(total_iters, 1):5.1f}%) "
            f"avg_loss={total_loss / max(total_samples, 1):.4f} "
            f"eta={eta_sec:6.1f}s"
        )
        sys.stdout.flush()
    if total_iters > 0:
        sys.stdout.write("\n")
        sys.stdout.flush()

    denom = max(total_samples, 1)
    avg_loss = total_loss / denom
    sat_avail_rate = sat_valid_sum / denom
    skimg_avail_rate = skimg_valid_sum / denom
    return avg_loss, sat_avail_rate, skimg_avail_rate


def evaluate_task(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    criterion: nn.Module,
    *,
    task: str,
    collect_records: bool = False,
    pv_output_interval_min: int | None = None,
) -> TaskMetrics | tuple[TaskMetrics, dict[str, np.ndarray]]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    pred_list: list[np.ndarray] = []
    tgt_list: list[np.ndarray] = []
    ts_list: list[np.ndarray] = []
    interval_ns = (
        int(pv_output_interval_min) * 60 * 1_000_000_000
        if pv_output_interval_min is not None
        else None
    )
    total_batches = len(loader)
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            d = _batch_to_device(batch, device)
            bsz = int(d["device_id"].size(0))
            kt_pred = forward_vit(model, d)
            pv_pred = kt_pred * d["target_p_cs"] * d["p_mean"].unsqueeze(1)
            loss, pred_np, tgt_np, used_idx = _task_loss_and_vectors(pv_pred, d["target_pv"], criterion, task)
            total_loss += float(loss.item()) * bsz
            total_samples += bsz
            done = batch_idx + 1
            elapsed = max(time.time() - t0, 1e-6)
            eta_sec = max(total_batches - done, 0) / max(done / elapsed, 1e-6)
            avg_loss = total_loss / max(total_samples, 1)
            sys.stdout.write(
                f"\r[eval]  iter {done}/{total_batches} "
                f"({100.0 * done / max(total_batches, 1):5.1f}%) "
                f"avg_loss={avg_loss:.4f} "
                f"eta={eta_sec:6.1f}s"
            )
            sys.stdout.flush()
            pred_list.append(pred_np)
            tgt_list.append(tgt_np)
            if collect_records:
                if interval_ns is None:
                    raise ValueError("collect_records=True requires pv_output_interval_min")
                anchor_raw = batch.get("anchor_time_utc_ns")
                if not isinstance(anchor_raw, torch.Tensor):
                    raise KeyError(
                        "Batch missing required 'anchor_time_utc_ns' for prediction timestamp export"
                    )
                anchor_ns = anchor_raw.detach().cpu().numpy().astype(np.int64, copy=False)  # [B]
                if used_idx is None:
                    t_out = int(d["target_pv"].shape[1])
                    offsets = (np.arange(t_out, dtype=np.int64) + 1) * np.int64(interval_ns)
                    ts_ns = anchor_ns[:, None] + offsets[None, :]
                    ts_list.append(ts_ns.reshape(-1).astype("datetime64[ns]"))
                else:
                    ts_ns = anchor_ns + np.int64(used_idx + 1) * np.int64(interval_ns)
                    ts_list.append(ts_ns.astype("datetime64[ns]"))

    pred = np.concatenate(pred_list, axis=0) if pred_list else np.array([], dtype=np.float32)
    tgt = np.concatenate(tgt_list, axis=0) if tgt_list else np.array([], dtype=np.float32)
    if pred.size == 0:
        mae = float("nan")
        rmse = float("nan")
    else:
        mae, rmse = _mae_rmse(pred, tgt)
    metrics = TaskMetrics(loss=total_loss / max(total_samples, 1), rmse=rmse, mae=mae)
    if not collect_records:
        return metrics
    ts = np.concatenate(ts_list, axis=0) if ts_list else np.array([], dtype="datetime64[ns]")
    return metrics, {"pred": pred, "tgt": tgt, "timestamp_utc": ts}


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    warmup_epochs: int,
    lr_min: float,
) -> LRScheduler:
    total_epochs = max(int(epochs), 1)
    warmup = max(int(warmup_epochs), 0)
    min_ratio = max(float(lr_min), 0.0)
    if warmup <= 0:
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=min_ratio)
    cosine_epochs = max(total_epochs - warmup, 1)
    warmup_sched = lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0 / max(warmup, 1),
        end_factor=1.0,
        total_iters=warmup,
    )
    cosine_sched = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=min_ratio,
    )
    return lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup],
    )


def _resolve_named_config(config_dir: Path, name_or_path: str, label: str) -> Path:
    p = Path(name_or_path).expanduser()
    if p.is_absolute():
        if not p.is_file():
            raise FileNotFoundError(f"{label} not found: {p}")
        return p.resolve()
    if p.parent != Path("."):
        cand = (_PROJECT_ROOT / p).resolve()
        if not cand.is_file():
            raise FileNotFoundError(f"{label} not found: {cand}")
        return cand
    stem = p.name
    if not stem.endswith(".yaml"):
        stem_yaml = stem + ".yaml"
    else:
        stem_yaml = stem
    cand = (config_dir / stem_yaml).resolve()
    if not cand.is_file():
        raise FileNotFoundError(f"{label} not found: {cand}")
    return cand


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _resolve_data_dir(paths_cfg: dict, cfg_path: Path) -> Path:
    raw = paths_cfg.get("data_dir")
    if raw is None or str(raw).strip() == "":
        raise KeyError(f"dataset config paths.data_dir is required (in {cfg_path})")
    p = Path(str(raw))
    return p.resolve() if p.is_absolute() else (_PROJECT_ROOT / p).resolve()


def _build_parser(h: dict, config_default: str, dataset_default: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ViT model on Luoyang 2026 total-station dataloader")
    parser.add_argument("--config", type=str, default=config_default)
    parser.add_argument("--dataset-config", type=str, default=dataset_default)
    parser.add_argument("--task", type=str, default="48h", choices=("15m", "4h", "48h"))
    parser.add_argument("--epochs", type=int, default=int(h["epochs"]))
    parser.add_argument("--lr", type=float, default=float(h["lr"]))
    parser.add_argument("--batch_size", type=int, default=int(h["batch_size"]))
    parser.add_argument("--save_every", type=int, default=int(h["save_every"]))
    parser.add_argument("--num_workers", type=int, default=int(h["num_workers"]))
    mb = h.get("train_max_batches_per_epoch")
    parser.add_argument(
        "--train_max_batches_per_epoch",
        type=int,
        default=None if mb is None else int(mb),
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=int(h.get("warmup_epochs", 0)))
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--huber-delta", type=float, default=float(h.get("huber_delta", 35.0)))
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--init-checkpoint", type=str, default=None)
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    # TabM freeze switch. Default comes from the training config (training.freeze_tabm);
    # CLI flags override the config value.
    parser.add_argument("--freeze-tabm", dest="freeze_tabm", action="store_true")
    parser.add_argument("--no-freeze-tabm", dest="freeze_tabm", action="store_false")
    parser.set_defaults(freeze_tabm=bool(h.get("freeze_tabm", False)))
    parser.add_argument("--use-ema", dest="use_ema", action="store_true")
    parser.add_argument("--no-ema", dest="use_ema", action="store_false")
    parser.set_defaults(use_ema=False)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--ema-warmup-epochs", type=int, default=5)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--nwp-dropout-prob", type=float, default=0.0)
    parser.add_argument("--nwp-history-dropout-prob", type=float, default=0.0)
    return parser


def _dataset_kwargs(dataset_config_name: str, split: str, max_files: int | None) -> dict:
    cfg_path = _resolve_named_config(_DATASETS_CONFIG_DIR, dataset_config_name, "dataset-config")
    cfg = _load_yaml(cfg_path)
    paths_cfg = cfg.get("paths", {}) or {}
    sampling_cfg = cfg.get("sampling", {}) or {}
    split_cfg = cfg.get("split_policy", {}) or {}
    data_dir = _resolve_data_dir(paths_cfg, cfg_path)

    def _req_path(key: str) -> str:
        v = paths_cfg.get(key)
        if v is None or str(v).strip() == "":
            raise KeyError(f"dataset config paths.{key} is required (in {cfg_path})")
        return str(v)

    def _req_sampling(key: str):
        if key not in sampling_cfg:
            raise KeyError(f"dataset config sampling.{key} is required (in {cfg_path})")
        return sampling_cfg[key]

    return dict(
        config_path=str(cfg_path),
        pv_dir=str((data_dir / _req_path("pv_total_path")).resolve()),
        skyimg_dir=str((data_dir / _req_path("sky_image_path")).resolve()),
        satimg_dir=str((data_dir / _req_path("sat_path")).resolve()),
        split=split,
        csv_interval_min=int(_req_sampling("csv_interval_min")),
        pv_input_interval_min=int(_req_sampling("pv_input_interval_min")),
        pv_input_len=int(_req_sampling("pv_input_len")),
        pv_output_interval_min=int(_req_sampling("pv_output_interval_min")),
        pv_output_len=int(_req_sampling("pv_output_len")),
        pv_train_time_fraction=float(sampling_cfg.get("pv_train_time_fraction", 0.7)),
        test_anchor_stride_min=int(_req_sampling("test_anchor_stride_min")),
        val_anchor_stride_min=int(_req_sampling("val_anchor_stride_min")),
        test_collect_time_match_tolerance_min=int(_req_sampling("test_collect_time_match_tolerance_min")),
        skyimg_window_size=int(_req_sampling("skyimg_window_size")),
        skyimg_time_resolution_min=int(_req_sampling("skyimg_time_resolution_min")),
        skyimg_spatial_size=int(sampling_cfg.get("skyimg_spatial_size", 224)),
        satimg_window_size=int(_req_sampling("satimg_window_size")),
        satimg_time_resolution_min=int(_req_sampling("satimg_time_resolution_min")),
        satimg_npy_shape_hwc=tuple(int(x) for x in sampling_cfg.get("satimg_npy_shape_hwc", [100, 100, 3])),
        train_samples_per_csv=int(sampling_cfg.get("train_samples_per_csv", 1)),
        train_fraction=float(split_cfg.get("train_fraction", 0.85)),
        val_fraction=float(split_cfg.get("val_fraction", 0.15)),
        test_start_bj=str(split_cfg.get("test_start_bj", "2026-05-11 00:00:00")),
        max_files=max_files,
    )


def main() -> None:
    print("[startup] Parsing CLI and loading configs...")
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=_DEFAULT_TRAIN_CONF_NAME)
    pre_parser.add_argument("--dataset-config", type=str, default=_DEFAULT_DATASET_CONF_NAME)
    pre_args, _ = pre_parser.parse_known_args()

    train_conf_path = _resolve_named_config(_TRAIN_CONFIG_DIR, pre_args.config, "train-config")
    train_conf = _load_yaml(train_conf_path)
    h = train_conf.get("training") or {}
    if not h:
        raise KeyError(f"training config {train_conf_path} is missing a 'training:' section")

    parser = _build_parser(h, config_default=pre_args.config, dataset_default=pre_args.dataset_config)
    args = parser.parse_args()
    print(
        f"[startup] task={args.task} config={args.config} dataset_config={args.dataset_config} "
        f"epochs={args.epochs} batch_size={args.batch_size} "
        f"nwp_dropout={args.nwp_dropout_prob} nwp_history_dropout={args.nwp_history_dropout_prob} "
        f"freeze_tabm={args.freeze_tabm} huber_delta={args.huber_delta}"
    )
    if args.init_checkpoint and args.resume_checkpoint:
        raise ValueError("Use only one of --init-checkpoint or --resume-checkpoint.")

    print("[startup] Building datasets (train/val/test)...")
    train_dataset = PVDataset(**_dataset_kwargs(args.dataset_config, "train", args.max_files))
    print(f"[startup] train dataset ready: files={len(train_dataset.sample_files)} samples={len(train_dataset)}")
    val_dataset = PVDataset(**_dataset_kwargs(args.dataset_config, "val", args.max_files))
    print(f"[startup] val dataset ready: files={len(val_dataset.sample_files)} samples={len(val_dataset)}")
    test_dataset = PVDataset(**_dataset_kwargs(args.dataset_config, "test", args.max_files))
    print(f"[startup] test dataset ready: files={len(test_dataset.sample_files)} samples={len(test_dataset)}")
    dev_dn_list = train_dataset.devDn_list

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[startup] Initializing model/optimizer on device={device}...")
    model = pv_forecasting_model_vit_imgs(
        dev_dn_list=dev_dn_list,
        nwp_dropout_prob=args.nwp_dropout_prob,
        nwp_history_dropout_prob=args.nwp_history_dropout_prob,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = _build_lr_scheduler(
        optimizer,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        lr_min=args.lr_min,
    )
    criterion = nn.HuberLoss(delta=args.huber_delta)
    ema: ModelEMA | None = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None
    print(
        f"EMA: {'enabled' if args.use_ema else 'disabled'}"
        + (f" (decay={args.ema_decay}, warmup={args.ema_warmup_epochs} epoch)" if args.use_ema else "")
    )

    start_epoch = 1
    best_val_rmse = float("inf")
    if args.resume_checkpoint:
        print(f"[startup] Loading resume checkpoint: {args.resume_checkpoint}")
        resume_path = Path(args.resume_checkpoint).expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_rmse = float(ckpt.get("val_rmse", ckpt.get("val_loss", best_val_rmse)))
        print(f"Resumed from {resume_path}, start_epoch={start_epoch}, best_val_rmse={best_val_rmse:.6f}")
    elif args.init_checkpoint:
        print(f"[startup] Loading init checkpoint: {args.init_checkpoint}")
        init_path = Path(args.init_checkpoint).expanduser().resolve()
        if not init_path.is_file():
            raise FileNotFoundError(f"init checkpoint not found: {init_path}")
        ckpt = torch.load(init_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Initialized model weights from {init_path}")

    if args.freeze_tabm:
        if not (args.resume_checkpoint or args.init_checkpoint):
            print(
                "[startup] WARNING: freeze_tabm=True without init/resume checkpoint; "
                "TabM will stay at random init and never train."
            )
        for param in model.pv_tabm_head.parameters():
            param.requires_grad = False
        n_frozen = sum(p.numel() for p in model.pv_tabm_head.parameters())
        print(f"[startup] TabM frozen: {n_frozen:,} params locked (requires_grad=False)")
    else:
        print("[startup] TabM trainable (freeze_tabm=False)")

    persistent = args.num_workers > 0
    print(
        f"[startup] Building DataLoaders (num_workers={args.num_workers}, "
        f"persistent_workers={persistent})..."
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batched,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batched,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batched,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=persistent,
    )

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else (_PROJECT_ROOT / "checkpoints_2026total")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"task_{args.task}_{('gpu' + str(_gpu_id_for_checkpoint())) if device.type == 'cuda' else 'cpu'}"
    best_ckpt_path = checkpoint_dir / f"pv_forecast_vit_best_{suffix}.pt"
    tb_log_dir = _PROJECT_ROOT / "runs_luoyang2026total" / suffix
    writer = SummaryWriter(log_dir=str(tb_log_dir))
    print(f"TensorBoard log dir: {tb_log_dir}")
    print("[startup] Initialization complete. Starting training loop...")

    for epoch in range(start_epoch, args.epochs + 1):
        cur_lr = optimizer.param_groups[0]["lr"]
        ema_active = ema is not None and epoch > args.ema_warmup_epochs
        if ema is not None and not ema_active:
            for k, v in model.state_dict().items():
                if k in ema.shadow:
                    ema.shadow[k].copy_(v.detach().float())

        train_loss, sat_avail_rate, skimg_avail_rate = train_one_epoch_task(
            model,
            device,
            train_loader,
            criterion,
            optimizer,
            task=args.task,
            max_batches=args.train_max_batches_per_epoch,
            ema=ema if ema_active else None,
        )
        if ema_active:
            with ema.apply(model):
                val_metrics = evaluate_task(model, device, val_loader, criterion, task=args.task)
        else:
            val_metrics = evaluate_task(model, device, val_loader, criterion, task=args.task)
        scheduler.step()

        print(
            f"Epoch {epoch}/{args.epochs} task={args.task} lr={cur_lr:.2e} "
            f"train_loss={train_loss:.6f} val_loss={val_metrics.loss:.6f} "
            f"val_rmse={val_metrics.rmse:.6f} val_mae={val_metrics.mae:.6f} "
            f"sat_avail={sat_avail_rate:.3f} skimg_avail={skimg_avail_rate:.3f}"
        )
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_metrics.loss, epoch)
        writer.add_scalar("metric/val_rmse", val_metrics.rmse, epoch)
        writer.add_scalar("metric/val_mae", val_metrics.mae, epoch)
        writer.add_scalar("availability/train_sat", sat_avail_rate, epoch)
        writer.add_scalar("availability/train_skimg", skimg_avail_rate, epoch)
        writer.add_scalar("lr", cur_lr, epoch)

        if args.save_every and epoch % args.save_every == 0:
            path = checkpoint_dir / f"pv_forecast_vit_epoch_{epoch}_{suffix}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "task": args.task,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": train_loss,
                    "dev_dn_list": dev_dn_list,
                },
                path,
            )
            print(f"  saved {path}")

        if val_metrics.rmse < best_val_rmse:
            best_val_rmse = val_metrics.rmse
            best_state = ema.full_state_dict(model) if ema_active else model.state_dict()
            torch.save(
                {
                    "epoch": epoch,
                    "task": args.task,
                    "model_state_dict": best_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": train_loss,
                    "val_loss": val_metrics.loss,
                    "val_rmse": val_metrics.rmse,
                    "dev_dn_list": dev_dn_list,
                    "ema": ema_active,
                },
                best_ckpt_path,
            )

    final_path = checkpoint_dir / f"pv_forecast_vit_final_{suffix}.pt"
    final_state = ema.full_state_dict(model) if ema is not None else model.state_dict()
    torch.save(
        {
            "epoch": args.epochs,
            "task": args.task,
            "model_state_dict": final_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "dev_dn_list": dev_dn_list,
            "ema": ema is not None,
        },
        final_path,
    )
    print(f"Saved final checkpoint to {final_path}")

    # Evaluate both the best-val and final checkpoints on the test set,
    # exporting one prediction CSV per checkpoint.
    for tag, ckpt_path in (("best", best_ckpt_path), ("final", final_path)):
        if not ckpt_path.is_file():
            print(f"No {ckpt_path.name} on disk; skip {tag} test evaluation.")
            continue
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_metrics, test_records = evaluate_task(
            model,
            device,
            test_loader,
            criterion,
            task=args.task,
            collect_records=True,
            pv_output_interval_min=test_dataset.pv_output_interval_min,
        )
        print(
            f"{tag.capitalize()} checkpoint ({ckpt_path.name}, epoch={ckpt.get('epoch', '?')}): "
            f"test_loss={test_metrics.loss:.6f}, test_rmse={test_metrics.rmse:.6f}, test_mae={test_metrics.mae:.6f}"
        )
        pred_path = checkpoint_dir / f"pv_forecast_vit_test_predictions_{suffix}_{tag}.csv"
        ts_utc = pd.to_datetime(test_records["timestamp_utc"], utc=True)
        ts_bj = ts_utc.tz_convert("Asia/Shanghai")
        range_end_bj = pd.Timestamp("2026-06-11 23:55:00", tz="Asia/Shanghai")
        range_start_bj_by_task = {
            "15m": pd.Timestamp("2026-05-11 00:15:00", tz="Asia/Shanghai"),
            "4h": pd.Timestamp("2026-05-11 04:00:00", tz="Asia/Shanghai"),
        }
        if args.task in range_start_bj_by_task:
            range_start_bj = range_start_bj_by_task[args.task]
            keep = (ts_bj >= range_start_bj) & (ts_bj <= range_end_bj)
        else:
            keep = np.ones(len(ts_utc), dtype=bool)
        pred_df = pd.DataFrame(
            {
                "timestamp_utc": ts_utc[keep].strftime("%Y-%m-%d %H:%M:%S"),
                "timestamp_bj": ts_bj[keep].strftime("%Y-%m-%d %H:%M:%S"),
                "pv_pred_kW": np.asarray(test_records["pred"])[keep],
                "pv_true_kW": np.asarray(test_records["tgt"])[keep],
                "task": args.task,
            }
        )
        pred_df.to_csv(pred_path, index=False)
        if len(pred_df) > 0:
            bj_min = str(pred_df["timestamp_bj"].iloc[0])
            bj_max = str(pred_df["timestamp_bj"].iloc[-1])
        else:
            bj_min = "N/A"
            bj_max = "N/A"
        print(
            f"Saved {tag} test predictions to {pred_path} "
            f"(rows={len(pred_df)}, task={args.task}, bj_range=[{bj_min} -> {bj_max}])"
        )
        writer.add_scalar(f"metric/test_rmse_{tag}", test_metrics.rmse, args.epochs)
        writer.add_scalar(f"metric/test_mae_{tag}", test_metrics.mae, args.epochs)
    writer.close()


if __name__ == "__main__":
    main()
