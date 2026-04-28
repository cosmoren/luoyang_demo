"""
Train Folsom irradiance forecaster (``irr_forecasting_model_vit_folsom``).
Luoyang PV trainer: ``training/train_vit.py``.

Primary batch-contract validation (no model/training): ``python -m dataloader.folsom``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dataloader.folsom import (
    FOLSOM_BATCH_TENSOR_KEYS,
    build_folsom_irradiance_datasets_from_conf,
    collate_folsom_irradiance,
)
from models.models import irr_forecasting_model_vit_folsom
from training.training_conf import (
    FOLSOM_CONF_PATH,
    get_training_hparams_from_conf,
    load_config_path,
)


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    warmup_epochs: int,
    lr_min: float,
) -> LRScheduler:
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


def _batch_to_device_folsom(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in FOLSOM_BATCH_TENSOR_KEYS:
        v = batch.get(key)
        out[key] = None if v is None else v.to(device)
    return out


def forward_folsom(model: nn.Module, d: dict[str, Any]) -> torch.Tensor:
    return model(
        d["ghi"],
        d["dni"],
        d["dhi"],
        input_mask=d["input_mask"],
        irr_timefeats=d["irr_timefeats"],
        forecast_timefeats=d["forecast_timefeats"],
        skimg_tensor=d.get("skimg_tensor"),
        skimg_timefeats=d.get("skimg_timefeats"),
        nwp_tensor=d.get("nwp_tensor"),
    )


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """pred/target ``[B, T_out, 3]``, mask ``[B, T_out]``."""
    m = mask.unsqueeze(-1).to(pred.dtype)
    se = (pred - target) ** 2 * m
    denom = m.sum().clamp(min=1.0)
    return se.sum() / denom


def train_one_epoch(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    max_batches: int | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        d = _batch_to_device_folsom(batch, device)
        optimizer.zero_grad()
        pred = forward_folsom(model, d)
        target = torch.stack([d["target_ghi"], d["target_dni"], d["target_dhi"]], dim=-1)
        loss = _masked_mse(pred, target, d["target_mask"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def _evaluate_one_pass(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
) -> tuple[float, float, float]:
    """Single dataloader pass: masked MSE, MAE, RMSE over all masked elements."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    sum_abs = 0.0
    sum_sq = 0.0
    n_elem = 0.0
    with torch.no_grad():
        for batch in loader:
            d = _batch_to_device_folsom(batch, device)
            pred = forward_folsom(model, d)
            target = torch.stack([d["target_ghi"], d["target_dni"], d["target_dhi"]], dim=-1)
            loss = _masked_mse(pred, target, d["target_mask"])
            total_loss += loss.item()
            n_batches += 1
            m = d["target_mask"].unsqueeze(-1)
            diff = pred - target
            sum_abs += (diff.abs() * m).sum().item()
            sum_sq += ((diff ** 2) * m).sum().item()
            n_elem += m.sum().item()
    mean_loss = total_loss / max(n_batches, 1)
    mae = sum_abs / max(n_elem, 1.0)
    rmse = (sum_sq / max(n_elem, 1.0)) ** 0.5
    return mean_loss, mae, rmse


def _build_parser(h: dict) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Folsom irradiance ViT-style model")
    p.add_argument(
        "--config",
        type=str,
        default=str(FOLSOM_CONF_PATH),
        help="YAML config (default: config/conf_folsom.yaml).",
    )
    p.add_argument("--epochs", type=int, default=int(h["epochs"]))
    p.add_argument("--lr", type=float, default=float(h["lr"]))
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--batch-size", type=int, default=int(h["batch_size"]))
    p.add_argument("--num-workers", type=int, default=int(h["num_workers"]))
    p.add_argument(
        "--train-epoch-len",
        type=int,
        default=50_000,
        help="Train Dataset __len__ (random anchors per epoch).",
    )
    mb = h.get("train_max_batches_per_epoch")
    p.add_argument(
        "--train-max-batches-per-epoch",
        type=int,
        default=-1 if mb is None else int(mb),
        help="Cap batches per epoch (-1 = no cap).",
    )
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--save-every", type=int, default=int(h["save_every"]))
    p.add_argument(
        "--debug-batch-only",
        action="store_true",
        help="Pull one train batch, run forward smoke test, then exit (no full training).",
    )
    return p


def _smoke_forward(model: nn.Module, device: torch.device, train_loader: DataLoader) -> None:
    batch = next(iter(train_loader))
    d = _batch_to_device_folsom(batch, device)
    model.eval()
    with torch.no_grad():
        pred = forward_folsom(model, d)
    tgt = torch.stack([d["target_ghi"], d["target_dni"], d["target_dhi"]], dim=-1)
    B, tout, c = pred.shape
    assert c == 3
    assert pred.shape == tgt.shape, (pred.shape, tgt.shape)
    assert torch.isfinite(pred).all(), "non-finite predictions"
    print(f"forward smoke OK: pred shape {tuple(pred.shape)}")


def main() -> None:
    default_conf = load_config_path(FOLSOM_CONF_PATH)
    h0 = get_training_hparams_from_conf(default_conf)
    parser = _build_parser(h0)
    args = parser.parse_args()
    conf = load_config_path(args.config)
    h = get_training_hparams_from_conf(conf)

    max_batches = args.train_max_batches_per_epoch
    if max_batches is not None and max_batches < 0:
        max_batches = None
    elif max_batches is not None:
        max_batches = int(max_batches)

    train_ds, test_ds = build_folsom_irradiance_datasets_from_conf(
        conf,
        train_epoch_len=int(args.train_epoch_len),
    )
    nw = int(args.num_workers)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        collate_fn=collate_folsom_irradiance,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=nw > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=collate_folsom_irradiance,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=nw > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = irr_forecasting_model_vit_folsom(
        skyimg_window_size=int(h["skyimg_window_size"]),
        skyimg_spatial_size=int(h["skyimg_spatial_size"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        betas=(0.9, 0.999),
    )
    scheduler = _build_lr_scheduler(
        optimizer,
        epochs=int(args.epochs),
        warmup_epochs=int(args.warmup_epochs),
        lr_min=float(args.lr_min),
    )

    if args.debug_batch_only:
        _smoke_forward(model, device, train_loader)
        return

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else _PROJECT_ROOT / "checkpoints_folsom"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    init_loss, init_mae, init_rmse = _evaluate_one_pass(model, device, test_loader)
    print(f"Initial test  masked_mse={init_loss:.6f}  MAE={init_mae:.6f}  RMSE={init_rmse:.6f}")

    for epoch in range(1, int(args.epochs) + 1):
        cur_lr = optimizer.param_groups[0]["lr"]
        avg_loss = train_one_epoch(
            model,
            device,
            train_loader,
            optimizer,
            max_batches=max_batches,
        )
        test_loss, test_mae, test_rmse = _evaluate_one_pass(model, device, test_loader)
        print(
            f"Epoch {epoch}/{args.epochs}  lr={cur_lr:.2e}  "
            f"train_mse={avg_loss:.6f}  test_mse={test_loss:.6f}  "
            f"test_MAE={test_mae:.6f}  test_RMSE={test_rmse:.6f}"
        )
        scheduler.step()

        se = int(h["save_every"])
        if se and epoch % se == 0:
            path = checkpoint_dir / f"folsom_irr_epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": avg_loss,
                    "config_path": str(Path(args.config).resolve()),
                },
                path,
            )
            print(f"  saved {path}")

    final_path = checkpoint_dir / "folsom_irr_final.pt"
    torch.save(
        {
            "epoch": int(args.epochs),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config_path": str(Path(args.config).resolve()),
        },
        final_path,
    )
    print(f"Saved final checkpoint to {final_path}")


if __name__ == "__main__":
    main()
