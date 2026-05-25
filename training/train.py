"""
Training script for pv_forecasting_model.
Loads config from conf.yaml, builds train/val/test splits, trains with MSE loss.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from training.training_conf import bootstrap_config_from_argv

bootstrap_config_from_argv()

from config_utils import get_resolved_paths
from dataloader.luoyang import PVDataset, collate_batched
from models.models import pv_forecasting_model_vit_nwp
import training.training_conf as training_conf_module
from training.training_conf import (
    get_training_hparams_from_conf,
    get_training_paths_from_conf,
    load_config,
)


def _pv_phys_scale_from_loader(loader: DataLoader) -> float:
    """Undo model-input normalization: raw CSV units = tensor * pv_value_scale (from config)."""
    return float(getattr(loader.dataset, "_pv_value_scale", 50.0))


def _batch_model_kwargs(batch: dict, device: torch.device) -> dict:
    """Optional modalities for ``pv_forecasting_model_vit_nwp`` (NWP, satellite)."""
    out: dict = {}
    nwp = batch.get("nwp_tensor")
    out["nwp_tensor"] = None if nwp is None else nwp.to(device)
    for key in ("sat_tensor", "sat_timefeats"):
        v = batch.get(key)
        out[key] = None if v is None else v.to(device)
    return out


def _masked_pv_loss_tensors(
    batch: dict,
    pv_pred: torch.Tensor,
    target_pv: torch.Tensor,
    target_mask: torch.Tensor,
    device: torch.device,
    *,
    loss_power_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (pred, target) for criterion; clear_sky_ratio uses power/scale = ratio * GHI / scale."""
    if "target_power" in batch and "y_ghi" in batch:
        ghi = batch["y_ghi"].to(device)
        tgt_power = batch["target_power"].to(device)
        m = target_mask
        pred_p = pv_pred * ghi
        if loss_power_scale is not None and loss_power_scale > 0:
            sc = float(loss_power_scale)
            pred_p = pred_p / sc
            tgt_power = tgt_power / sc
        return pred_p * m, tgt_power * m
    return pv_pred * target_mask, target_pv * target_mask


def _build_pv_dataset(args: argparse.Namespace, satimg_hwc: tuple[int, int, int], split: str) -> PVDataset:
    return PVDataset(
        pv_dir=args.pv_dir,
        skyimg_dir=args.skyimg_dir,
        satimg_dir=args.satimg_dir,
        split=split,
        csv_interval_min=args.csv_interval_min,
        pv_input_interval_min=args.pv_input_interval_min,
        pv_input_len=args.pv_input_len,
        pv_output_interval_min=args.pv_output_interval_min,
        pv_output_len=args.pv_output_len,
        pv_train_time_fraction=args.pv_train_time_fraction,
        test_anchor_stride_min=args.test_anchor_stride_min,
        val_anchor_stride_min=args.val_anchor_stride_min,
        test_collect_time_match_tolerance_min=args.test_collect_time_match_tolerance_min,
        skyimg_window_size=args.skyimg_window_size,
        skyimg_time_resolution_min=args.skyimg_time_resolution_min,
        skyimg_spatial_size=args.skyimg_spatial_size,
        satimg_window_size=args.satimg_window_size,
        satimg_time_resolution_min=args.satimg_time_resolution_min,
        satimg_npy_shape_hwc=satimg_hwc,
    )


def _apply_config_override(config_path: str | None) -> None:
    if not config_path:
        return
    p = Path(config_path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"config file not found: {p}")
    training_conf_module.CONF_PATH = p
    # Keep dataloader module-level config path in sync.
    import dataloader.luoyang as luoyang_module

    luoyang_module.CONF_PATH = p
    print(f"[train] using config: {p}")


def loader_test(args: argparse.Namespace, satimg_hwc: tuple[int, int, int]) -> None:
    """Build PVDataset + DataLoader and print dataset lengths and one batch tensor shapes."""
    train_ds = _build_pv_dataset(args, satimg_hwc, "train")
    if len(train_ds) == 0:
        print("[dataloader test] train dataset is empty; cannot fetch a batch.")
        return

    bs = min(max(1, args.batch_size), len(train_ds))
    loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        collate_fn=collate_batched,
        num_workers=0,
    )
    batch = next(iter(loader))
    summary_keys = list(batch.keys())
    print(f"[dataloader test] batch_size={bs} keys={list(batch.keys())}")
    for k in summary_keys:
        v = batch[k]
        if v is None:
            print(f"  {k}: None")
        elif isinstance(v, torch.Tensor):
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v).__name__}")

    B = int(batch["dev_idx"].size(0))
    for b in range(B):
        print(f"\n[dataloader test] sample index {b} (batch of {B})")
        print(f"  pv[b]: {batch['pv'][b]}")
        print(f"  pv_mask[b]: {batch['pv_mask'][b]}")
        print(f"  target_pv[b]: {batch['target_pv'][b]}")
        print(f"  target_mask[b]: {batch['target_mask'][b]}")


def train_one_epoch(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    max_batches: int | None = None,
    *,
    epoch: int | None = None,
    log_every: int = 50,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    num_batches = len(loader)
    if max_batches is not None:
        num_batches = min(num_batches, max_batches)
    ep = "" if epoch is None else f"epoch {epoch} "
    print(f"{ep}number of batches: {len(loader)}" + (f" (capped at {max_batches})" if max_batches is not None else ""))
    running_loss = 0.0
    running_batches = 0
    log_every = max(1, int(log_every))
    loss_power_scale = getattr(loader.dataset, "_loss_power_scale", None)
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        B = batch["dev_idx"].size(0)
        device_id = batch["dev_idx"].to(device)
        pv = batch["pv"].to(device)
        mask = batch["pv_mask"].to(device)
        pv_timefeats = batch["pv_timefeats"].to(device)
        forecast_timefeats = batch["forecast_timefeats"].to(device)
        target_pv = batch["target_pv"].to(device)
        target_mask = batch["target_mask"].to(device)
        model_kw = _batch_model_kwargs(batch, device)

        optimizer.zero_grad()
        pv_pred = model(
            device_id,
            pv,
            mask,
            pv_timefeats,
            forecast_timefeats,
            **model_kw,
        )
        pred_loss, tgt_loss = _masked_pv_loss_tensors(
            batch,
            pv_pred,
            target_pv,
            target_mask,
            device,
            loss_power_scale=loss_power_scale,
        )
        loss = criterion(pred_loss, tgt_loss)
        loss.backward()
        optimizer.step()
        loss_b = float(loss.item())
        total_loss += loss_b
        n += B
        running_loss += loss_b
        running_batches += 1
        done = batch_idx + 1
        if done == 1 or done % log_every == 0 or done == num_batches:
            avg_running = running_loss / max(running_batches, 1)
            print(
                f"  [train] {ep}batch {done}/{num_batches}  "
                f"loss_batch={loss_b:.6f}  loss_avg_running={avg_running:.6f}"
            )
    avg_loss = total_loss / max(n, 1)
    print(f"  [train] {ep}end  mean_loss_returned={avg_loss:.6f} (sum_batch_loss / n_samples={n})")
    return avg_loss


def evaluate(
    model: nn.Module, device: torch.device, loader: DataLoader, criterion: nn.Module
) -> tuple[float, float]:
    """Mean masked loss and aggregated masked RMSE (kW, station-sum logic aligned with train_vit.py)."""
    model.eval()
    total_loss = 0.0
    n = 0
    pred_dict: dict[int, list] = {}
    target_dict: dict[int, list] = {}
    target_mask_dict: dict[int, list] = {}
    loss_on_power = getattr(loader.dataset, "_loss_on_power", False)
    loss_power_scale = getattr(loader.dataset, "_loss_power_scale", None)
    with torch.no_grad():
        for batch in loader:
            B = batch["dev_idx"].size(0)
            device_id = batch["dev_idx"].to(device)
            pv = batch["pv"].to(device)
            mask = batch["pv_mask"].to(device)
            pv_timefeats = batch["pv_timefeats"].to(device)
            forecast_timefeats = batch["forecast_timefeats"].to(device)
            target_pv = batch["target_pv"].to(device)
            target_mask = batch["target_mask"].to(device)
            model_kw = _batch_model_kwargs(batch, device)
            pv_pred = model(
                device_id,
                pv,
                mask,
                pv_timefeats,
                forecast_timefeats,
                **model_kw,
            )
            pred_loss, tgt_loss = _masked_pv_loss_tensors(
                batch,
                pv_pred,
                target_pv,
                target_mask,
                device,
                loss_power_scale=loss_power_scale,
            )
            loss = criterion(pred_loss, tgt_loss)
            total_loss += loss.item()
            n += B
            for i in range(pv_pred.shape[0]):
                kk = int(device_id[i].item())
                if loss_on_power:
                    ghi_np = batch["y_ghi"][i].detach().cpu().float().numpy()
                    pred_np = (pv_pred[i].detach().cpu().float().numpy() * ghi_np).copy()
                    tgt_np = batch["target_power"][i].detach().cpu().float().numpy().copy()
                else:
                    pred_np = pv_pred[i].detach().cpu().float().numpy().copy()
                    tgt_np = target_pv[i].detach().cpu().float().numpy().copy()
                msk_np = target_mask[i].detach().cpu().float().numpy().copy()
                discrete_pred = pred_np.tolist()
                discrete_target = tgt_np.tolist()
                discrete_mask = msk_np.tolist()
                if kk not in pred_dict:
                    pred_dict[kk] = []
                    target_dict[kk] = []
                    target_mask_dict[kk] = []
                pred_dict[kk].append(discrete_pred)
                target_dict[kk].append(discrete_target)
                target_mask_dict[kk].append(discrete_mask)

        if n == 0:
            return 0.0, float("nan")

        total_pred = None
        total_target = None
        total_valid = None
        for kk in pred_dict.keys():
            cur_pred = np.asarray(pred_dict[kk], dtype=np.float64).reshape(-1)
            cur_target = np.asarray(target_dict[kk], dtype=np.float64).reshape(-1)
            cur_valid = np.asarray(target_mask_dict[kk], dtype=np.float64).reshape(-1)
            if total_pred is None:
                total_pred = cur_pred * cur_valid
            else:
                total_pred = total_pred + (cur_pred * cur_valid)

            if total_target is None:
                total_target = cur_target * cur_valid
            else:
                total_target = total_target + (cur_target * cur_valid)

            if total_valid is None:
                total_valid = cur_valid
            else:
                total_valid = total_valid + cur_valid

        valid_points = total_valid > 0
        if not np.any(valid_points):
            return total_loss / max(n, 1), float("nan")

        pv_scale = 1.0 if loss_on_power else _pv_phys_scale_from_loader(loader)
        err = (total_pred - total_target)[valid_points] * pv_scale
        mae = np.mean(np.abs(err))
        rmse = np.sqrt(np.mean(err**2))

        capacity = 54600
        print(
            f"t0 - t0+48h, 15min interval, 192 points. Capacity: {capacity}(KW) "
            f"MAE: {mae:.6f}, RMSE: {rmse:.6f}, ACC(MAE): {1.0 - mae / capacity:.6f}, "
            f"ACC(RMSE): {1.0 - rmse / capacity:.6f}"
        )

    return total_loss / max(n, 1), rmse


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    warmup_epochs: int,
    lr_min: float,
) -> LRScheduler:
    """Linear warmup (epoch-wise) then cosine decay to ``lr_min``."""
    warmup_epochs = max(0, int(warmup_epochs))
    epochs = max(1, int(epochs))
    if warmup_epochs == 0:
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)
    if warmup_epochs >= epochs:
        return lr_scheduler.LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0, total_iters=epochs
        )
    return lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            lr_scheduler.LinearLR(
                optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_epochs
            ),
            lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - warmup_epochs, eta_min=lr_min
            ),
        ],
        milestones=[warmup_epochs],
    )


def _should_validate(epoch: int, total_epochs: int, val_every: int) -> bool:
    """Run validation every ``val_every`` epochs and always on the last epoch."""
    if val_every < 1:
        return True
    return (epoch % val_every == 0) or (epoch == total_epochs)


def export_test_horizon_csvs(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    out_plus15_path: Path,
    out_plus4h_path: Path,
) -> tuple[int, int]:
    """Export +15min and +4h forecast-vs-GT rows for test samples."""
    model.eval()
    rows_plus15: list[dict] = []
    rows_plus4h: list[dict] = []
    total = len(loader.dataset)
    processed = 0
    bar_width = 30
    pv_scale = _pv_phys_scale_from_loader(loader)
    with torch.no_grad():
        for batch in loader:
            device_id = batch["dev_idx"].to(device)
            pv = batch["pv"].to(device)
            mask = batch["pv_mask"].to(device)
            pv_timefeats = batch["pv_timefeats"].to(device)
            forecast_timefeats = batch["forecast_timefeats"].to(device)
            target_pv = batch["target_pv"].to(device)
            model_kw = _batch_model_kwargs(batch, device)
            pv_pred = model(
                device_id,
                pv,
                mask,
                pv_timefeats,
                forecast_timefeats,
                **model_kw,
            )
            pred_np = (pv_pred.detach().cpu().float().numpy() * pv_scale)
            gt_np = (target_pv.detach().cpu().float().numpy() * pv_scale)
            ts_meta = batch["target_timestamps_utc"]
            state_meta = batch["target_inverter_state"]
            B = int(pred_np.shape[0])
            for i in range(B):
                ts_list = ts_meta[i]
                inv_state_list = state_meta[i]
                rows_plus15.append(
                    {
                        "timestamp": ts_list[0],
                        "inverter_state": int(inv_state_list[0]),
                        "gt_active_power": float(gt_np[i, 0]),
                        "forecasted_power": float(pred_np[i, 0]),
                    }
                )
                rows_plus4h.append(
                    {
                        "timestamp": ts_list[-1],
                        "inverter_state": int(inv_state_list[-1]),
                        "gt_active_power": float(gt_np[i, -1]),
                        "forecasted_power": float(pred_np[i, -1]),
                    }
                )
            processed += B
            pct = (100.0 * processed / max(total, 1))
            filled = int(bar_width * processed / max(total, 1))
            bar = "#" * filled + "-" * (bar_width - filled)
            print(
                f"\rPredicting test samples [{bar}] {processed}/{total} ({pct:5.1f}%)",
                end="",
                flush=True,
            )
    print()
    pd.DataFrame(rows_plus15).to_csv(out_plus15_path, index=False)
    pd.DataFrame(rows_plus4h).to_csv(out_plus4h_path, index=False)
    return len(rows_plus15), len(rows_plus4h)


def export_test_single_target_csv(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    out_path: Path,
) -> int:
    """Export deterministic single-target test rows with 3 columns."""
    model.eval()
    rows: list[dict] = []
    total = len(loader.dataset)
    processed = 0
    bar_width = 30
    pv_scale = _pv_phys_scale_from_loader(loader)
    with torch.no_grad():
        for batch in loader:
            device_id = batch["dev_idx"].to(device)
            pv = batch["pv"].to(device)
            mask = batch["pv_mask"].to(device)
            pv_timefeats = batch["pv_timefeats"].to(device)
            forecast_timefeats = batch["forecast_timefeats"].to(device)
            target_pv = batch["target_pv"].to(device)
            model_kw = _batch_model_kwargs(batch, device)
            pv_pred = model(
                device_id,
                pv,
                mask,
                pv_timefeats,
                forecast_timefeats,
                **model_kw,
            )
            pred_np = (pv_pred.detach().cpu().float().numpy() * pv_scale)
            gt_np = (target_pv.detach().cpu().float().numpy() * pv_scale)
            ts_meta = batch["target_timestamps_utc"]
            B = int(pred_np.shape[0])
            for i in range(B):
                rows.append(
                    {
                        "collectTime": ts_meta[i][0],
                        "gt_active_power": float(gt_np[i, 0]),
                        "forecasted_active_power": float(pred_np[i, 0]),
                    }
                )
            processed += B
            pct = (100.0 * processed / max(total, 1))
            filled = int(bar_width * processed / max(total, 1))
            bar = "#" * filled + "-" * (bar_width - filled)
            print(
                f"\rPredicting test samples [{bar}] {processed}/{total} ({pct:5.1f}%)",
                end="",
                flush=True,
            )
    print()
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return len(rows)


def _find_latest_epoch_ckpt(checkpoint_dir: Path) -> Path | None:
    """Find the latest ``pv_forecast_epoch_*.pt`` checkpoint by epoch number."""
    latest_path: Path | None = None
    latest_epoch = -1
    for path in checkpoint_dir.glob("pv_forecast_epoch_*.pt"):
        stem = path.stem
        prefix = "pv_forecast_epoch_"
        if not stem.startswith(prefix):
            continue
        suffix = stem[len(prefix) :]
        if not suffix.isdigit():
            continue
        epoch = int(suffix)
        if epoch > latest_epoch:
            latest_epoch = epoch
            latest_path = path
    return latest_path


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()
    _apply_config_override(pre_args.config)

    conf = load_config()
    h = get_training_hparams_from_conf(conf)
    path_defaults = get_training_paths_from_conf(conf)
    parser = argparse.ArgumentParser(description="Train PV forecasting model")
    parser.add_argument(
        "--config",
        type=str,
        default=pre_args.config,
        help="Path to config YAML (default: config/conf.yaml).",
    )
    parser.add_argument("--epochs", type=int, default=h["epochs"])
    parser.add_argument("--lr", type=float, default=h["lr"])
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay (default 0.01).",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=3,
        help="Linear LR warmup in epoch units before cosine decay (0 = no warmup).",
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=1e-6,
        help="Minimum learning rate for cosine tail (default 1e-6).",
    )
    parser.add_argument("--batch_size", type=int, default=h["batch_size"])
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument(
        "--val-every",
        type=int,
        default=500,
        help="Run validation (and consider updating best checkpoint) every N epochs; always validates on the last epoch.",
    )
    parser.add_argument("--save_every", type=int, default=h["save_every"])
    parser.add_argument(
        "--pv-dir",
        type=str,
        default=path_defaults["pv_dir"],
        help=f"PV CSV directory (default: {path_defaults['pv_dir']!r}).",
    )
    parser.add_argument(
        "--skyimg-dir",
        type=str,
        default=path_defaults["skyimg_dir"],
        help=f"Sky JPEG root (default: {path_defaults['skyimg_dir']!r}).",
    )
    parser.add_argument(
        "--nwp-dir",
        type=str,
        default=path_defaults.get("nwp_dir"),
        help="Optional NWP directory containing modality CSVs (e.g. solar/temp).",
    )
    parser.add_argument(
        "--csv_interval_min",
        type=int,
        default=h["csv_interval_min"],
        help="CSV row spacing in minutes (must divide pv_input/output intervals).",
    )
    parser.add_argument(
        "--pv_input_interval_min",
        type=int,
        default=h["pv_input_interval_min"],
        help="Minutes between consecutive PV input (X) samples.",
    )
    parser.add_argument(
        "--pv_output_interval_min",
        type=int,
        default=h["pv_output_interval_min"],
        help="Minutes between consecutive PV target (Y) samples.",
    )
    parser.add_argument(
        "--t_off_min",
        type=int,
        default=h["t_off_min"],
        help="Single-horizon offset in minutes from last X timestamp (used when pv_output_len=1).",
    )
    parser.add_argument("--pv_input_len", type=int, default=h["pv_input_len"], help="Input sequence length (X).")
    parser.add_argument("--pv_output_len", type=int, default=h["pv_output_len"], help="Target sequence length (Y).")
    parser.add_argument(
        "--pv_output_rand",
        action="store_true",
        default=h["pv_output_rand"],
        help="When pv_output_len>1: sample random sparse Y anchors instead of deterministic interval grid.",
    )
    parser.add_argument(
        "--pv_train_time_fraction",
        type=float,
        default=h["pv_train_time_fraction"],
        help="Per CSV: first int(n*fraction) rows are train segment, rest test; anchors must fit entirely in segment.",
    )
    parser.add_argument(
        "--test_anchor_stride_min",
        type=int,
        default=h["test_anchor_stride_min"],
        help="For split=test: minutes between consecutive eval anchors (multiple of CSV row interval).",
    )
    parser.add_argument(
        "--val_anchor_stride_min",
        type=int,
        default=h["val_anchor_stride_min"],
        help="For split=val: minutes between consecutive val anchors (multiple of CSV row interval).",
    )
    parser.add_argument(
        "--test_collect_time_match_tolerance_min",
        type=int,
        default=h["test_collect_time_match_tolerance_min"],
        help="For split=test: max minutes between ref last-X collectTime and matched row in each CSV (0=exact).",
    )
    parser.add_argument(
        "--skyimg_window_size",
        type=int,
        default=h["skyimg_window_size"],
        help="Number of sky images per history and forecast sequence.",
    )
    parser.add_argument(
        "--skyimg_time_resolution_min",
        type=int,
        default=h["skyimg_time_resolution_min"],
        help="Minutes between consecutive sky frames (independent of PV input spacing).",
    )
    parser.add_argument(
        "--skyimg_spatial_size",
        type=int,
        default=h["skyimg_spatial_size"],
        help="Sky JPEG resize side length (square, pixels).",
    )
    parser.add_argument(
        "--satimg-dir",
        type=str,
        default=path_defaults["satimg_dir"],
        help=f"Himawari NPY root (default: {path_defaults['satimg_dir']!r}).",
    )
    parser.add_argument(
        "--satimg_window_size",
        type=int,
        default=h["satimg_window_size"],
        help="Number of Himawari NPY frames per history and forecast sequence.",
    )
    parser.add_argument(
        "--satimg_time_resolution_min",
        type=int,
        default=h["satimg_time_resolution_min"],
        help="Minutes between consecutive satimg frames in UTC.",
    )
    parser.add_argument(
        "--satimg_npy_shape_hwc",
        type=int,
        nargs=3,
        default=list(h["satimg_npy_shape_hwc"]),
        metavar=("H", "W", "C"),
        help="Expected Himawari NPY array shape H W C (default from conf).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=h["num_workers"],
        help="DataLoader worker processes.",
    )
    parser.add_argument(
        "--pin-memory",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "DataLoader pin_memory mode: auto=enable on CUDA except test_only, "
            "on=always enable, off=disable (useful for unstable CUDA pin-memory threads)."
        ),
    )
    parser.add_argument(
        "--train_max_batches_per_epoch",
        type=int,
        default=h["train_max_batches_per_epoch"],
        help="Stop each training epoch after this many batches (default from conf; null = no cap).",
    )
    parser.add_argument(
        "--test_dataloader",
        action="store_true",
        help="Build PVDataset/DataLoader, print lengths and one batch shapes, then exit.",
    )
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="Skip training and run test evaluation only (loads an existing checkpoint when available).",
    )
    parser.add_argument(
        "--test_only_single_file",
        action="store_true",
        help="In test-only mode, evaluate using only the first CSV in pv_dir.",
    )
    parser.add_argument(
        "--test_only_use_val",
        action="store_true",
        help="In test-only mode, build the eval DataLoader from the validation split instead of test.",
    )
    parser.add_argument(
        "--test_only_plus15_csv",
        type=str,
        default=None,
        help="Optional output CSV path for +15min test-only results.",
    )
    parser.add_argument(
        "--test_only_plus4h_csv",
        type=str,
        default=None,
        help="Optional output CSV path for +4h test-only results.",
    )
    parser.add_argument(
        "--test_only_ckpt",
        type=str,
        default=None,
        help="Optional explicit checkpoint file path for test-only mode.",
    )
    parser.add_argument(
        "--test_only_y_offset_min",
        type=int,
        default=None,
        help="Internal: deterministic test-only mode uses X ending at (ts_y - offset minutes).",
    )
    parser.add_argument(
        "--test_only_all_last30_rows",
        action="store_true",
        help="Internal: deterministic test-only mode evaluates all rows in the last 30%.",
    )
    parser.add_argument(
        "--ylj_raw_parquet",
        action="store_true",
        help="YLJ + train_ylj: load PV from paths.ylj_raw_parquet_dir Parquet matrices instead of CSV.",
    )
    parser.add_argument(
        "--ylj_parquet_nwp",
        action="store_true",
        help="With --ylj_raw_parquet: attach NWP from fixed SSRD/T2m predict columns in the Parquet (no NWP files).",
    )
    parser.add_argument(
        "--ylj_sat_zarr",
        action="store_true",
        help="With --ylj_raw_parquet: load Himawari satellite from paths.sat_path Zarr (yalongjiang_zarr).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest checkpoint in checkpoint_dir.",
    )
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="Optional explicit checkpoint file path to resume training from.",
    )
    args = parser.parse_args()
    satimg_hwc = tuple(args.satimg_npy_shape_hwc)

    if args.test_dataloader:
        loader_test(args, satimg_hwc)
        return

    resolved_paths = get_resolved_paths(conf, _PROJECT_ROOT)
    pv_device_path = resolved_paths["pv_device_path"]
    if pv_device_path is None or not pv_device_path.is_file():
        raise FileNotFoundError(f"pv_device_path not found: {pv_device_path}")
    pv_device_df = pd.read_excel(pv_device_path)
    dev_dn_list = pv_device_df["devDn"].dropna().unique().tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = pv_forecasting_model_vit_nwp(dev_dn_list=dev_dn_list).to(device)
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
    # criterion = nn.MSELoss()
    criterion = nn.HuberLoss(delta=1.0)  # nn.MSELoss()

    eval_split = "val" if args.test_only and args.test_only_use_val else "test"
    if args.test_only and args.test_only_use_val:
        print("test_only_use_val=True -> eval DataLoader uses validation split.")
    test_dataset = _build_pv_dataset(args, satimg_hwc, eval_split)
    if args.test_only and args.test_only_single_file and len(test_dataset.sample_files) > 1:
        test_dataset.sample_files = test_dataset.sample_files[:1]
        print(
            f"test_only_single_file=True -> using only {test_dataset.sample_files[0].name} for {eval_split}."
        )

    nw_workers = args.num_workers
    if args.pin_memory == "on":
        use_pin_memory = True
    elif args.pin_memory == "off":
        use_pin_memory = False
    else:
        # In test-only mode, avoid pin-memory thread CUDA failures on unstable GPU runtime.
        use_pin_memory = torch.cuda.is_available() and (not args.test_only)
    dl_kw = dict(
        collate_fn=collate_batched,
        num_workers=nw_workers,
        pin_memory=use_pin_memory,
        persistent_workers=nw_workers > 0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **dl_kw,
    )

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else _PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    first_ckpt_path = checkpoint_dir / "pv_forecast_first.pt"
    best_ckpt_path = checkpoint_dir / "pv_forecast_best.pt"
    last_ckpt_path = checkpoint_dir / "pv_forecast_last.pt"

    if args.test_only:
        ckpt_path = Path(args.test_only_ckpt) if args.test_only_ckpt else None
        if ckpt_path is not None and not ckpt_path.is_file():
            raise FileNotFoundError(f"test_only_ckpt not found: {ckpt_path}")
        if ckpt_path is None:
            for path in (best_ckpt_path, last_ckpt_path, first_ckpt_path):
                if path.is_file():
                    ckpt_path = path
                    break
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded checkpoint for test-only mode: {ckpt_path} (epoch={ckpt.get('epoch', '?')})")
        else:
            print(
                "test_only=True but no checkpoint found among "
                f"{best_ckpt_path.name}, {last_ckpt_path.name}, {first_ckpt_path.name}; "
                "testing with current in-memory model weights."
            )
        plus15_csv = (
            Path(args.test_only_plus15_csv)
            if args.test_only_plus15_csv
            else (checkpoint_dir / "test_only_plus15min_results.csv")
        )
        plus4h_csv = (
            Path(args.test_only_plus4h_csv)
            if args.test_only_plus4h_csv
            else (checkpoint_dir / "test_only_plus4h_results.csv")
        )
        if args.test_only_single_file and getattr(test_dataset, "supports_single_horizon_test_only", False):
            args_15 = argparse.Namespace(**vars(args))
            args_15.test_only_y_offset_min = int(args.t_off_min)
            args_15.test_only_all_last30_rows = True
            args_15.pv_output_len = 1
            ds_15 = _build_pv_dataset(args_15, satimg_hwc, eval_split)
            ds_15.sample_files = ds_15.sample_files[:1]
            loader_15 = DataLoader(
                ds_15,
                batch_size=args.batch_size,
                shuffle=False,
                **dl_kw,
            )

            # Disabled for now: +4h test-only export path.
            # args_4h = argparse.Namespace(**vars(args))
            # args_4h.test_only_y_offset_min = 240
            # args_4h.test_only_all_last30_rows = True
            # args_4h.pv_output_len = 1
            # ds_4h = _build_pv_dataset(args_4h, satimg_hwc, "test")
            # ds_4h.sample_files = ds_4h.sample_files[:1]
            # loader_4h = DataLoader(
            #     ds_4h,
            #     batch_size=args.batch_size,
            #     shuffle=False,
            #     **dl_kw,
            # )

            n15 = export_test_single_target_csv(model, device, loader_15, plus15_csv)
            print(f"Saved +{int(args.t_off_min)}min rows ({n15}) to {plus15_csv}")
            # n4 = export_test_single_target_csv(model, device, loader_4h, plus4h_csv)
            # print(f"Saved +4h rows ({n4}) to {plus4h_csv}")
            return

        # Disabled for now: +4h test-only export path.
        # n15, n4 = export_test_horizon_csvs(model, device, test_loader, plus15_csv, plus4h_csv)
        n15 = export_test_single_target_csv(model, device, test_loader, plus15_csv)
        print(f"Saved +{int(args.t_off_min)}min rows ({n15}) to {plus15_csv}")
        # print(f"Saved +4h rows ({n4}) to {plus4h_csv}")
        test_loss, test_rmse = evaluate(model, device, test_loader, criterion)
        print(
            f"Test-only result (split={eval_split}): loss={test_loss:.6f}, RMSE={test_rmse:.6f}"
        )
        return

    start_epoch = 1
    rmse_min = 1e8
    if args.resume_ckpt and not args.resume:
        print("--resume_ckpt was provided without --resume; ignoring resume_ckpt.")
    if args.resume:
        resume_path = Path(args.resume_ckpt) if args.resume_ckpt else None
        if resume_path is not None and not resume_path.is_file():
            raise FileNotFoundError(f"resume_ckpt not found: {resume_path}")
        if resume_path is None:
            resume_path = _find_latest_epoch_ckpt(checkpoint_dir)
            if resume_path is None and last_ckpt_path.is_file():
                resume_path = last_ckpt_path
        if resume_path is None:
            print("resume=True but no checkpoint found; starting training from epoch 1.")
        else:
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            if "val_rmse" in ckpt:
                try:
                    rmse_min = float(ckpt["val_rmse"])
                except Exception:
                    rmse_min = 1e8
            print(
                f"Resumed training from checkpoint: {resume_path} "
                f"(loaded epoch={ckpt.get('epoch', '?')}, next_epoch={start_epoch})"
            )

    train_dataset = _build_pv_dataset(args, satimg_hwc, "train")
    val_dataset = _build_pv_dataset(args, satimg_hwc, "val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **dl_kw,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **dl_kw,
    )

    # initial_test_loss, _ = evaluate(model, device, test_loader, criterion)
    # print(f"Initial test loss: {initial_test_loss:.6f}")

    if start_epoch > args.epochs:
        print(
            f"Resume next_epoch={start_epoch} is greater than configured epochs={args.epochs}; "
            "nothing to train."
        )
        return

    for epoch in range(start_epoch, args.epochs + 1):
        cur_lr = optimizer.param_groups[0]["lr"]
        avg_loss = train_one_epoch(
            model,
            device,
            train_loader,
            criterion,
            optimizer,
            max_batches=args.train_max_batches_per_epoch,
            epoch=epoch,
            log_every=50,
        )
    
        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": avg_loss,
            "dev_dn_list": dev_dn_list,
        }

        if epoch == 1:
            torch.save(ckpt_payload, first_ckpt_path)
            print(f"  saved first -> {first_ckpt_path.name}")

        run_val = _should_validate(epoch, args.epochs, args.val_every)
        if run_val:
            val_loss, val_rmse = evaluate(model, device, val_loader, criterion)
            print(
                f"Epoch {epoch}/{args.epochs}  lr={cur_lr:.2e}  "
                f"train_loss={avg_loss:.6f}  val_loss={val_loss:.6f}  val_rmse={val_rmse:.6f}"
            )
            if val_rmse < rmse_min:
                rmse_min = val_rmse
                torch.save(
                    {
                        **ckpt_payload,
                        "val_loss": val_loss,
                        "val_rmse": val_rmse,
                    },
                    best_ckpt_path,
                )
                print(f"  saved best (val_rmse={val_rmse:.6f}) -> {best_ckpt_path.name}")
        else:
            print(f"Epoch {epoch}/{args.epochs}  lr={cur_lr:.2e}  train_loss={avg_loss:.6f}")

        scheduler.step()

        if args.save_every and epoch % args.save_every == 0:
            path = checkpoint_dir / f"pv_forecast_epoch_{epoch}.pt"
            torch.save(ckpt_payload, path)
            print(f"  saved {path}")

    torch.save(ckpt_payload, last_ckpt_path)
    print(f"Saved last -> {last_ckpt_path}")

    if best_ckpt_path.is_file():
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        test_loss_best, test_rmse_best = evaluate(model, device, test_loader, criterion)
        print(
            f"Test set with best val-RMSE checkpoint ({best_ckpt_path.name}, epoch={ckpt.get('epoch', '?')}): "
            f"loss={test_loss_best:.6f}, RMSE={test_rmse_best:.6f}"
        )
    else:
        print(f"No {best_ckpt_path.name} on disk; skip test evaluation with best checkpoint.")


if __name__ == "__main__":
    main()
