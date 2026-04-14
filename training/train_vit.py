"""
Training script for pv_forecasting_model_vit (sat + sky TimeSformer + PV TCN).
Uses the same Luoyang CSV / sky / sat DataLoader as train.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config_utils import get_resolved_paths
from dataloader.luoyang import PVDataset, collate_batched
from models.models import pv_forecasting_model_vit_nwp
from training.training_conf import (
    get_training_hparams_from_conf,
    get_training_paths_from_conf,
    load_config,
)


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move luoyang collate fields to device; optional tensors may be None."""
    out = {
        "device_id": batch["dev_idx"].to(device),
        "pv": batch["pv"].to(device),
        "pv_mask": batch["pv_mask"].to(device),
        "pv_timefeats": batch["pv_timefeats"].to(device),
        "forecast_timefeats": batch["forecast_timefeats"].to(device),
        "target_pv": batch["target_pv"].to(device),
        "target_mask": batch["target_mask"].to(device),
    }
    for key in ("sat_tensor", "sat_timefeats", "skimg_tensor", "skimg_timefeats", "nwp_tensor"):
        v = batch.get(key)
        out[key] = None if v is None else v.to(device)
    return out


def forward_vit(model: nn.Module, d: dict) -> torch.Tensor:
    return model(
        d["device_id"],
        d["pv"],
        pv_mask=d["pv_mask"],
        pv_timefeats=d["pv_timefeats"],                  # [B, T, C=9]
        forecast_timefeats=d["forecast_timefeats"],      # [B, T, C=9]
        sat_tensor=d["sat_tensor"],                      # [B, T=24, C=3, H=100, W=100]
        sat_timefeats=d["sat_timefeats"],                # [B, T=24, C=9]
        skimg_tensor=d["skimg_tensor"],                  # [B, T=30, C=3, H=224, W=224]
        skimg_timefeats=d["skimg_timefeats"],            # [B, T=30, C=9]
        nwp_tensor=d["nwp_tensor"],                     # [B, T_out, 7] or None
    )


def train_one_epoch(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    max_batches: int | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    print("number of batches: ", len(loader))
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        d = _batch_to_device(batch, device)
        B = d["device_id"].size(0)
        optimizer.zero_grad()
        pv_pred = forward_vit(model, d)
        # loss = criterion(pv_pred * d["target_mask"], d["target_pv"] * d["target_mask"])
        loss = criterion(pv_pred, d["target_pv"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += B
    print()
    return total_loss / max(n, 1)


def evaluate(model: nn.Module, device: torch.device, loader: DataLoader, criterion: nn.Module) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    pred_dict = {}
    target_dict = {}
    with torch.no_grad():
        for batch in loader:
            d = _batch_to_device(batch, device)
            B = d["device_id"].size(0)
            pv_pred = forward_vit(model, d)
            # loss = criterion(pv_pred * d["target_mask"], d["target_pv"] * d["target_mask"])
            loss = criterion(pv_pred, d["target_pv"])
            total_loss += loss.item()
            n += B
            # save these values and used them to compute the MAE and RMSE of the total station
            for i in range(pv_pred.shape[0]):
                kk = int(d["device_id"][i].item())
                if kk not in pred_dict:
                    pred_dict[kk] = []
                    target_dict[kk] = []
                discrete_pred = pv_pred[i].detach().cpu().float().tolist()
                pred_dict[kk].append(discrete_pred)
                discrete_target = d["target_pv"][i].detach().cpu().float().tolist()
                target_dict[kk].append(discrete_target)
        
        total_pred = None
        total_target = None
        for kk in pred_dict.keys():
            if total_pred is None:
                total_pred = np.asarray(pred_dict[kk]).reshape(-1)
            else:
                total_pred = total_pred + np.asarray(pred_dict[kk]).reshape(-1)
            
            if total_target is None:
                total_target = np.asarray(target_dict[kk]).reshape(-1)
            else:
                total_target = total_target + np.asarray(target_dict[kk]).reshape(-1)
        
        mae = np.mean(np.abs(total_pred*50 - total_target*50))
        rmse = np.sqrt(np.mean((total_pred*50 - total_target*50) ** 2))

        capacity = 54600
        print(f"t0 - t0+48h, 15min interval, 192 points. Capacity: {capacity}(KW)")
        print(f"MAE: {mae:.6f}, RMSE: {rmse:.6f}, ACC(MAE): {1.0 - mae/capacity:.6f}, ACC(RMSE): {1.0 - rmse/capacity:.6f}")
    
    return total_loss / max(n, 1)


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    warmup_epochs: int,
    lr_min: float,
) -> LRScheduler:
    """Linear warmup (epoch-wise) then cosine decay to ``lr_min``.

    Call ``scheduler.step()`` only **after** ``optimizer.step()`` in that epoch
    (e.g. at end of each epoch); epoch 1 trains at the initial ``lr`` before the first step.
    """
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


def _build_parser(path_defaults: dict, h: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PV forecasting model (ViT / TimeSformer)")
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
    parser.add_argument("--save_every", type=int, default=h["save_every"])
    parser.add_argument(
        "--pv-dir",
        type=str,
        default=path_defaults["pv_dir"],
        help=f"PV CSV directory (train/test split by time in each CSV; default: {path_defaults['pv_dir']!r}).",
    )
    parser.add_argument(
        "--skyimg-dir",
        type=str,
        default=path_defaults["skyimg_dir"],
        help=f"Sky JPEG root (default: {path_defaults['skyimg_dir']!r}).",
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
    parser.add_argument("--pv_input_len", type=int, default=h["pv_input_len"], help="Input sequence length (X).")
    parser.add_argument("--pv_output_len", type=int, default=h["pv_output_len"], help="Target sequence length (Y).")
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
        "--test_collect_time_match_tolerance_min",
        type=int,
        default=h["test_collect_time_match_tolerance_min"],
        help="For split=test: max minutes between ref last-X collectTime and matched row in each CSV (0=exact).",
    )
    parser.add_argument(
        "--skyimg_window_size",
        type=int,
        default=h["skyimg_window_size"],
        help="Number of sky images per history and per forecast sequence.",
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
        help="Number of Himawari NPY frames per history and per forecast sequence.",
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
        "--train_max_batches_per_epoch",
        type=int,
        default=h["train_max_batches_per_epoch"],
        help="Stop each training epoch after this many batches (default from conf; null = no cap).",
    )
    return parser


def _dataset_kwargs(args: argparse.Namespace, satimg_hwc: tuple[int, int, int], split: str) -> dict:
    return dict(
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
        test_collect_time_match_tolerance_min=args.test_collect_time_match_tolerance_min,
        skyimg_window_size=args.skyimg_window_size,
        skyimg_time_resolution_min=args.skyimg_time_resolution_min,
        skyimg_spatial_size=args.skyimg_spatial_size,
        satimg_window_size=args.satimg_window_size,
        satimg_time_resolution_min=args.satimg_time_resolution_min,
        satimg_npy_shape_hwc=satimg_hwc,
    )


def main() -> None:
    conf = load_config()
    h = get_training_hparams_from_conf(conf)
    path_defaults = get_training_paths_from_conf(conf)
    parser = _build_parser(path_defaults, h)
    args = parser.parse_args()
    satimg_hwc = tuple(args.satimg_npy_shape_hwc)

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
    criterion = nn.MSELoss()

    train_dataset = PVDataset(**_dataset_kwargs(args, satimg_hwc, "train"))
    test_dataset = PVDataset(**_dataset_kwargs(args, satimg_hwc, "test"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batched,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batched,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else _PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    initial_test_loss = evaluate(model, device, test_loader, criterion)
    print(f"Initial test loss: {initial_test_loss:.6f}")

    for epoch in range(1, args.epochs + 1):
        cur_lr = optimizer.param_groups[0]["lr"]
        avg_loss = train_one_epoch(
            model,
            device,
            train_loader,
            criterion,
            optimizer,
            max_batches=args.train_max_batches_per_epoch,
        )
        test_loss = evaluate(model, device, test_loader, criterion)
        print(
            f"Epoch {epoch}/{args.epochs}  lr={cur_lr:.2e}  "
            f"train_loss={avg_loss:.6f}  test_loss={test_loss:.6f}"
        )
        scheduler.step()

        if args.save_every and epoch % args.save_every == 0:
            path = checkpoint_dir / f"pv_forecast_vit_epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg_loss,
                    "dev_dn_list": dev_dn_list,
                },
                path,
            )
            print(f"  saved {path}")

    final_path = checkpoint_dir / "pv_forecast_vit_final.pt"
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "dev_dn_list": dev_dn_list,
        },
        final_path,
    )
    print(f"Saved final checkpoint to {final_path}")


if __name__ == "__main__":
    main()
