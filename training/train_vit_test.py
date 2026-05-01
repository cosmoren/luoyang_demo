"""
Training script for pv_forecasting_model_vit (sat + sky TimeSformer + PV TCN).
Uses the same Luoyang CSV / sky / sat DataLoader as train.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import yaml
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"
_TRAIN_CONFIG_DIR = _CONFIG_DIR / "train"
_DATASETS_CONFIG_DIR = _CONFIG_DIR / "datasets"
_DEFAULT_TRAIN_CONF_NAME = "conf_train.yaml"
sys.path.insert(0, str(_PROJECT_ROOT))

from dataloader.luoyang_mem import PVDataset, collate_batched
# from dataloader.folsom import FolsomIrradianceDataset
from models.models import pv_forecasting_model_vit_imgs


def _gpu_id_for_checkpoint() -> int:
    """
    Label for ``pv_forecast_vit_best_gpu*.pt`` (not necessarily PyTorch logical index).

    If ``CUDA_VISIBLE_DEVICES`` is set to a comma-separated list, use the first token
    when it is a non-negative integer (e.g. ``4`` or ``4,5`` -> 4). Otherwise fall back
    to ``torch.cuda.current_device()`` (logical id, often 0 when only ``cuda`` is used).
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return torch.cuda.current_device()
    s = raw.strip()
    if not s:
        return torch.cuda.current_device()
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return torch.cuda.current_device()
    first = parts[0]
    if first.isdigit():
        return int(first)
    return torch.cuda.current_device()


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
        nwp_tensor=d["nwp_tensor"],                      # [B, T_out, 7] or None
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
        loss = criterion( (pv_pred * d["target_mask"])[:,0:16], (d["target_pv"] * d["target_mask"])[:,0:16])
        # loss = criterion(pv_pred, d["target_pv"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += B
    print()
    return total_loss / max(n, 1)


def evaluate(
    model: nn.Module, device: torch.device, loader: DataLoader, criterion: nn.Module
) -> tuple[float, float, float]:
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
            loss = criterion( (pv_pred * d["target_mask"])[:,0:16], (d["target_pv"] * d["target_mask"])[:,0:16] )
            # loss = criterion(pv_pred, d["target_pv"])
            total_loss += loss.item()
            n += B
            # save these values and used them to compute the MAE and RMSE of the total station
            for i in range(pv_pred.shape[0]):
                kk = int(d["device_id"][i].item()) # the inverter ID
                # forecast_timefeats[:, 3] == cos_zenith (see solar_features_encoder column order)
                # RMSE/MAE aggregation: first 16 horizons only (match training loss window)
                cos_zenith = d["forecast_timefeats"][i][:16, 3].detach().cpu().float().numpy()
                night = cos_zenith < 0
                pred_np = pv_pred[i, :16].detach().cpu().float().numpy().copy()
                tgt_np = d["target_pv"][i, :16].detach().cpu().float().numpy().copy()
                pred_np[night] = 0.0
                # tgt_np[night] = 0.0
                discrete_pred = pred_np.tolist()
                discrete_target = tgt_np.tolist()
                if kk not in pred_dict:
                    pred_dict[kk] = []
                    target_dict[kk] = []
                pred_dict[kk].append(discrete_pred)
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
        print(f"RMSE/MAE on first 16 steps (15min), ~4h. Capacity: {capacity}(KW)")
        print(f"MAE: {mae:.6f}, RMSE: {rmse:.6f}, ACC(MAE): {1.0 - mae/capacity:.6f}, ACC(RMSE): {1.0 - rmse/capacity:.6f}")
    
    return total_loss / max(n, 1), rmse, mae


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


def _resolve_named_config(directory: Path, name: str, label: str) -> Path:
    cfg = Path(name)
    if cfg.name != name:
        raise ValueError(
            f"--{label} only accepts a bare filename under {directory.relative_to(_PROJECT_ROOT)}/, "
            f"e.g. --{label} {cfg.name}"
        )
    cfg_path = directory / cfg.name
    if not cfg_path.is_file():
        raise FileNotFoundError(f"{label} file not found: {cfg_path}")
    return cfg_path


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f)
    return data or {}


def _resolve_data_dir(paths_cfg: dict, cfg_path: Path) -> Path:
    raw = paths_cfg.get("data_dir")
    if raw is None or str(raw).strip() == "":
        raise KeyError(f"dataset config paths.data_dir is required (in {cfg_path})")
    p = Path(str(raw))
    return p.resolve() if p.is_absolute() else (_PROJECT_ROOT / p).resolve()


def _build_parser(h: dict, config_default: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PV forecasting model (ViT / TimeSformer)")
    parser.add_argument(
        "--config",
        type=str,
        default=config_default,
        help=(
            f"Training config filename under config/train/ "
            f"(default: {config_default!r})."
        ),
    )
    parser.add_argument("--epochs", type=int, default=int(h["epochs"]))
    parser.add_argument("--lr", type=float, default=float(h["lr"]))
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
    parser.add_argument("--batch_size", type=int, default=int(h["batch_size"]))
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=int(h["save_every"]))
    parser.add_argument(
        "--num_workers",
        type=int,
        default=int(h["num_workers"]),
        help="DataLoader worker processes.",
    )
    mb = h.get("train_max_batches_per_epoch")
    parser.add_argument(
        "--train_max_batches_per_epoch",
        type=int,
        default=None if mb is None else int(mb),
        help="Stop each training epoch after this many batches (default from conf; null = no cap).",
    )
    return parser


def _dataset_kwargs(dataset_config_name: str, split: str) -> dict:
    """
    Build PVDataset-compatible kwargs from ``config/datasets/<dataset_config_name>``.

    The dataset YAML is expected to contain:
      - ``paths.data_dir`` plus ``pv_path`` / ``sky_image_path`` / ``sat_path``
      - a ``sampling:`` section with all PVDataset window / stride / image-shape fields
    """
    cfg_path = _resolve_named_config(_DATASETS_CONFIG_DIR, dataset_config_name, "dataset-config")
    cfg = _load_yaml(cfg_path)
    paths_cfg = cfg.get("paths", {}) or {}
    sampling_cfg = cfg.get("sampling", {}) or {}
    if not sampling_cfg:
        raise KeyError(f"dataset config {cfg_path} is missing a non-empty 'sampling:' section")

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

    pv_dir = (data_dir / _req_path("pv_path")).resolve()
    skyimg_dir = (data_dir / _req_path("sky_image_path")).resolve()
    satimg_dir = (data_dir / _req_path("sat_path")).resolve()

    shwc = _req_sampling("satimg_npy_shape_hwc")
    if not isinstance(shwc, (list, tuple)) or len(shwc) != 3:
        raise ValueError(
            f"sampling.satimg_npy_shape_hwc must be [H, W, C] (in {cfg_path})"
        )

    return dict(
        config_path=str(cfg_path),
        pv_dir=str(pv_dir),
        skyimg_dir=str(skyimg_dir),
        satimg_dir=str(satimg_dir),
        split=split,
        csv_interval_min=int(_req_sampling("csv_interval_min")),
        pv_input_interval_min=int(_req_sampling("pv_input_interval_min")),
        pv_input_len=int(_req_sampling("pv_input_len")),
        pv_output_interval_min=int(_req_sampling("pv_output_interval_min")),
        pv_output_len=int(_req_sampling("pv_output_len")),
        pv_train_time_fraction=float(_req_sampling("pv_train_time_fraction")),
        test_anchor_stride_min=int(_req_sampling("test_anchor_stride_min")),
        val_anchor_stride_min=int(_req_sampling("val_anchor_stride_min")),
        test_collect_time_match_tolerance_min=int(_req_sampling("test_collect_time_match_tolerance_min")),
        skyimg_window_size=int(_req_sampling("skyimg_window_size")),
        skyimg_time_resolution_min=int(_req_sampling("skyimg_time_resolution_min")),
        skyimg_spatial_size=int(_req_sampling("skyimg_spatial_size")),
        satimg_window_size=int(_req_sampling("satimg_window_size")),
        satimg_time_resolution_min=int(_req_sampling("satimg_time_resolution_min")),
        satimg_npy_shape_hwc=tuple(int(x) for x in shwc),
    )


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=_DEFAULT_TRAIN_CONF_NAME)
    pre_args, _ = pre_parser.parse_known_args()

    train_conf_path = _resolve_named_config(_TRAIN_CONFIG_DIR, pre_args.config, "config")
    train_conf = _load_yaml(train_conf_path)
    h = train_conf.get("training") or {}
    if not h:
        raise KeyError(f"training config {train_conf_path} is missing a 'training:' section")

    parser = _build_parser(h, config_default=pre_args.config)
    args = parser.parse_args()

    # =========================================================================
    # Dataset selection: change DATASET_CLS to a different Dataset class and
    # DATASET_CONFIG to a YAML filename under config/datasets/ to train on a
    # different dataset (e.g. ``conf_folsom.yaml``). The chosen YAML supplies
    # ``paths.*`` and the ``sampling:`` section consumed by ``_dataset_kwargs``.
    # =========================================================================
    DATASET_CLS = PVDataset
    DATASET_CONFIG = "conf_luoyang.yaml" # config file for the luoyang dataset

    train_dataset = DATASET_CLS(**_dataset_kwargs(DATASET_CONFIG, "train"))
    val_dataset = DATASET_CLS(**_dataset_kwargs(DATASET_CONFIG, "val"))
    test_dataset = DATASET_CLS(**_dataset_kwargs(DATASET_CONFIG, "test"))

    # DATASET_CLS = FolsomIrradianceDataset
    # DATASET_CONFIG = "conf_folsom.yaml" # config file for the folsom dataset

    # train_dataset = DATASET_CLS(**_dataset_kwargs(DATASET_CONFIG, "train"))
    # val_dataset = DATASET_CLS(**_dataset_kwargs(DATASET_CONFIG, "val"))
    # test_dataset = DATASET_CLS(**_dataset_kwargs(DATASET_CONFIG, "test"))

    # =========================================================================

    dev_dn_list = train_dataset.devDn_list

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = pv_forecasting_model_vit_imgs(dev_dn_list=dev_dn_list).to(device)
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
    criterion = nn.HuberLoss(delta=1.0)  # nn.MSELoss()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batched,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
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
    if device.type == "cuda":
        _gpu_id = _gpu_id_for_checkpoint()
        _ckpt_suffix = f"gpu{_gpu_id}"
    else:
        _ckpt_suffix = "cpu"
    best_ckpt_path = checkpoint_dir / f"pv_forecast_vit_best_{_ckpt_suffix}.pt"

    initial_test_loss, _, _ = evaluate(model, device, test_loader, criterion)
    print(f"Initial test loss: {initial_test_loss:.6f}")

    rmse_min = 1e8
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
        val_loss, val_rmse, _ = evaluate(model, device, val_loader, criterion)
        print(
            f"Epoch {epoch}/{args.epochs}  lr={cur_lr:.2e}  "
            f"train_loss={avg_loss:.6f}  val_loss={val_loss:.6f}"
        )
        scheduler.step()

        if args.save_every and epoch % args.save_every == 0:
            path = checkpoint_dir / f"pv_forecast_vit_epoch_{epoch}_{_ckpt_suffix}.pt"
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
        
        if val_rmse < rmse_min:
            rmse_min = val_rmse
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg_loss,
                    "dev_dn_list": dev_dn_list,
                },
                best_ckpt_path,
            )

    final_path = checkpoint_dir / f"pv_forecast_vit_final_{_ckpt_suffix}.pt"
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

    if best_ckpt_path.is_file():
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_loss_best, test_rmse_best, test_mae_best = evaluate(model, device, test_loader, criterion)
        print(
            f"Test set with best val-RMSE checkpoint ({best_ckpt_path.name}, epoch={ckpt.get('epoch', '?')}): "
            f"loss={test_loss_best:.6f}, RMSE={test_rmse_best:.6f}, MAE={test_mae_best:.6f}"
        )
        metrics_log = checkpoint_dir / f"pv_forecast_4h_pv_sat_{_ckpt_suffix}.txt"
        with open(metrics_log, "a", encoding="utf-8") as mf:
            mf.write(
                f"{test_loss_best:.8f}\t{test_rmse_best:.8f}\t{test_mae_best:.8f}\n"
            )
        print(f"Appended best-test metrics to {metrics_log}")
    else:
        print(f"No {best_ckpt_path.name} on disk; skip test evaluation with best checkpoint.")




if __name__ == "__main__":
    main()
