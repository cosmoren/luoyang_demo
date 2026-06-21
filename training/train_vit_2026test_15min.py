"""
Train script for Luoyang-2026 date-split dataset (train/test only).

This mirrors ``training/train_vit_test.py`` but switches dataloader to
``dataloader/luoyang_2026_zarr.py`` and uses test as the evaluation split
throughout training (no validation split in 2026 dataset wrapper).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dataloader.luoyang_2026_zarr import PVDataset, collate_batched  # noqa: E402
from models.models import pv_forecasting_model_vit_imgs  # noqa: E402
from training.train_vit_test import (  # noqa: E402
    WEATHER_SCORE_THRESHOLD,
    EvalMetrics,
    ModelEMA,
    _batch_to_device,
    _build_lr_scheduler,
    _build_parser as _base_build_parser,
    _dataset_kwargs,
    _gpu_id_for_checkpoint,
    _load_yaml,
    _resolve_named_config,
    forward_vit,
)

_CONFIG_DIR = _PROJECT_ROOT / "config"
_TRAIN_CONFIG_DIR = _CONFIG_DIR / "train"
_DEFAULT_TRAIN_CONF_NAME = "conf_train.yaml"
_DEFAULT_DATASET_CONFIG = "conf_luoyang_2026.yaml"


def train_one_epoch_first_point(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    max_batches: int | None = None,
    ema: ModelEMA | None = None,
) -> float:
    """Train one epoch with first-horizon (t0+15m) supervision only."""
    model.train()
    total_loss = 0.0
    n = 0
    print("number of batches: ", len(loader))
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        d = _batch_to_device(batch, device)
        bsz = d["device_id"].size(0)
        optimizer.zero_grad()
        kt_pred = forward_vit(model, d) * 20.0
        pv_pred = kt_pred * d["target_p_cs"] * d["p_mean"].unsqueeze(1)
        loss = criterion(pv_pred[:, 0], d["target_pv"][:, 0])
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model)
        total_loss += loss.item()
        n += bsz
    print()
    return total_loss / max(n, 1)


def evaluate_first_point(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    criterion: nn.Module,
    *,
    weather_score_threshold: float | None = None,
) -> EvalMetrics:
    """Evaluate with first-horizon (t0+15m) loss/RMSE/MAE only."""
    model.eval()
    total_loss = 0.0
    n = 0
    pred_dict: dict[int, list[np.ndarray]] = {}
    target_dict: dict[int, list[np.ndarray]] = {}
    ws_dict: dict[int, list[np.ndarray]] = {}
    mask_dict: dict[int, list[np.ndarray]] = {}
    collect_weather = weather_score_threshold is not None

    with torch.no_grad():
        for batch in loader:
            d = _batch_to_device(batch, device)
            bsz = d["device_id"].size(0)
            kt_pred = forward_vit(model, d) * 20.0
            pv_pred = kt_pred * d["target_p_cs"] * d["p_mean"].unsqueeze(1)
            pred_15m = pv_pred[:, 0]
            tgt_15m = d["target_pv"][:, 0]
            loss = criterion(pred_15m, tgt_15m)
            total_loss += loss.item()
            n += bsz

            for i in range(bsz):
                kk = int(d["device_id"][i].item())
                pred_np = pred_15m[i].detach().cpu().float().numpy().copy()
                tgt_np = tgt_15m[i].detach().cpu().float().numpy().copy()
                if kk not in pred_dict:
                    pred_dict[kk] = []
                    target_dict[kk] = []
                    if collect_weather:
                        ws_dict[kk] = []
                        mask_dict[kk] = []
                pred_dict[kk].append(pred_np)
                target_dict[kk].append(tgt_np)
                if collect_weather:
                    ws_dict[kk].append(
                        d["target_weather_score"][i, 0].detach().cpu().float().numpy().copy()
                    )
                    mask_dict[kk].append(
                        d["target_mask"][i, 0].detach().cpu().float().numpy().copy()
                    )

    if not pred_dict:
        return EvalMetrics(
            loss=float("nan"),
            loss_15m=float("nan"),
            loss_4h=float("nan"),
            loss_48h=float("nan"),
            rmse_15m=float("nan"),
            mae_15m=float("nan"),
            rmse_4h=float("nan"),
            mae_4h=float("nan"),
            rmse_48h=float("nan"),
            mae_48h=float("nan"),
        )

    station_pred = None
    station_target = None
    station_ws = None
    station_mask = None
    for kk in pred_dict.keys():
        pred_k = np.asarray(pred_dict[kk])
        tgt_k = np.asarray(target_dict[kk])
        if station_pred is None:
            station_pred = pred_k
            station_target = tgt_k
            if collect_weather:
                station_ws = np.asarray(ws_dict[kk])
                station_mask = np.asarray(mask_dict[kk])
        else:
            station_pred = station_pred + pred_k
            station_target = station_target + tgt_k
            if collect_weather:
                assert station_mask is not None
                station_mask = np.clip(station_mask + np.asarray(mask_dict[kk]), 0.0, 1.0)

    assert station_pred is not None and station_target is not None
    diff = station_pred - station_target
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    rmse_ws = mae_ws = float("nan")
    n_valid_ws = 0
    if collect_weather and station_ws is not None and station_mask is not None:
        valid = (station_ws > float(weather_score_threshold)) & (station_mask > 0.5)
        n_valid_ws = int(valid.sum())
        if n_valid_ws > 0:
            d_ws = diff[valid]
            mae_ws = float(np.mean(np.abs(d_ws)))
            rmse_ws = float(np.sqrt(np.mean(d_ws ** 2)))

    capacity = 54600
    print("RMSE/MAE [t0+15m only]. Capacity: 54600(KW)")
    print(
        f"  MAE: {mae:.6f}, RMSE: {rmse:.6f}, "
        f"ACC(MAE): {1.0 - mae / capacity:.6f}, ACC(RMSE): {1.0 - rmse / capacity:.6f}"
    )

    if collect_weather:
        ws_thr = float(weather_score_threshold)
        print(f"RMSE/MAE [t0+15m only, weather_score>{ws_thr:g} & valid]. Capacity: 54600(KW)")
        if n_valid_ws == 0:
            print("  MAE: nan, RMSE: nan, n_valid=0")
        else:
            print(
                f"  MAE: {mae_ws:.6f}, RMSE: {rmse_ws:.6f}, "
                f"ACC(MAE): {1.0 - mae_ws / capacity:.6f}, "
                f"ACC(RMSE): {1.0 - rmse_ws / capacity:.6f}, n_valid={n_valid_ws}"
            )

    # Keep EvalMetrics shape-compatible with existing log/checkpoint code.
    denom = max(n, 1)
    avg_loss = total_loss / denom
    return EvalMetrics(
        loss=avg_loss,
        loss_15m=avg_loss,
        loss_4h=avg_loss,
        loss_48h=avg_loss,
        rmse_15m=rmse,
        mae_15m=mae,
        rmse_4h=rmse,
        mae_4h=mae,
        rmse_48h=rmse,
        mae_48h=mae,
        rmse_15m_ws=rmse_ws,
        mae_15m_ws=mae_ws,
        n_valid_15m_ws=n_valid_ws,
        rmse_4h_ws=rmse_ws,
        mae_4h_ws=mae_ws,
        n_valid_4h_ws=n_valid_ws,
        rmse_48h_ws=rmse_ws,
        mae_48h_ws=mae_ws,
        n_valid_48h_ws=n_valid_ws,
    )


def _build_parser(h: dict, config_default: str) -> argparse.ArgumentParser:
    parser = _base_build_parser(h, config_default)
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=_DEFAULT_DATASET_CONFIG,
        help=(
            "Dataset config filename under config/datasets/ "
            f"(default: {_DEFAULT_DATASET_CONFIG!r})."
        ),
    )
    parser.description = "Train PV model on Luoyang-2026 date-split dataset (train/test only)"
    return parser


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
    if args.init_checkpoint and args.resume_checkpoint:
        raise ValueError("Use only one of --init-checkpoint or --resume-checkpoint, not both.")

    seed = int(h.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_dataset = PVDataset(**_dataset_kwargs(args.dataset_config, "train"))
    test_dataset = PVDataset(**_dataset_kwargs(args.dataset_config, "test"))
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
    criterion = nn.HuberLoss(delta=1.0)
    ema: ModelEMA | None = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None
    print(
        f"EMA: {'enabled' if args.use_ema else 'disabled'}"
        + (f" (decay={args.ema_decay}, warmup={args.ema_warmup_epochs} epoch)" if args.use_ema else "")
    )

    start_epoch = 1
    best_eval_loss = float("inf")
    if args.resume_checkpoint:
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
        best_eval_loss = float(ckpt.get("eval_loss", ckpt.get("val_loss", best_eval_loss)))
        print(
            f"Resumed from {resume_path} "
            f"(epoch={ckpt.get('epoch', '?')}, start_epoch={start_epoch}, best_eval_loss={best_eval_loss:.6f})"
        )
    elif args.init_checkpoint:
        init_path = Path(args.init_checkpoint).expanduser().resolve()
        if not init_path.is_file():
            raise FileNotFoundError(f"init checkpoint not found: {init_path}")
        ckpt = torch.load(init_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Initialized model weights from {init_path}; starting finetune from epoch 1.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batched,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batched,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else _PROJECT_ROOT / "checkpoints_2026"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        ckpt_suffix = f"gpu{_gpu_id_for_checkpoint()}"
    else:
        ckpt_suffix = "cpu"
    best_ckpt_path = checkpoint_dir / f"pv2026_forecast_vit_best_{ckpt_suffix}.pt"

    tb_log_dir = _PROJECT_ROOT / "runs" / f"pv2026_{ckpt_suffix}"
    writer = SummaryWriter(log_dir=str(tb_log_dir))
    print(f"TensorBoard log dir: {tb_log_dir}")

    if start_epoch > args.epochs:
        print(
            f"start_epoch ({start_epoch}) > epochs ({args.epochs}); "
            "skip training loop and run best-checkpoint test evaluation."
        )

    for epoch in range(start_epoch, args.epochs + 1):
        cur_lr = optimizer.param_groups[0]["lr"]
        ema_active = ema is not None and epoch > args.ema_warmup_epochs
        if ema is not None and not ema_active:
            for k, v in model.state_dict().items():
                if k in ema.shadow:
                    ema.shadow[k].copy_(v.detach().float())

        avg_loss = train_one_epoch_first_point(
            model,
            device,
            train_loader,
            criterion,
            optimizer,
            max_batches=args.train_max_batches_per_epoch,
            ema=ema if ema_active else None,
        )
        if ema_active:
            assert ema is not None
            with ema.apply(model):
                eval_metrics = evaluate_first_point(
                    model,
                    device,
                    test_loader,
                    criterion,
                    weather_score_threshold=WEATHER_SCORE_THRESHOLD,
                )
        else:
            eval_metrics = evaluate_first_point(
                model,
                device,
                test_loader,
                criterion,
                weather_score_threshold=WEATHER_SCORE_THRESHOLD,
            )
        eval_loss = eval_metrics.loss

        print(
            f"Epoch {epoch}/{args.epochs}  lr={cur_lr:.2e}  "
            f"train_loss={avg_loss:.6f}  test_loss={eval_loss:.6f}"
        )
        writer.add_scalar("loss/train", avg_loss, epoch)
        writer.add_scalar("loss/test", eval_loss, epoch)
        writer.add_scalar("metric/test_rmse", eval_metrics.rmse, epoch)
        writer.add_scalar("metric/test_mae", eval_metrics.mae, epoch)
        writer.add_scalar("lr", cur_lr, epoch)
        scheduler.step()

        if args.save_every and epoch % args.save_every == 0:
            path = checkpoint_dir / f"pv2026_forecast_vit_epoch_{epoch}_{ckpt_suffix}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg_loss,
                    "eval_loss": eval_loss,
                    "dataset_config": args.dataset_config,
                    "dev_dn_list": dev_dn_list,
                },
                path,
            )
            print(f"  saved {path}")

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_state = ema.state_dict() if ema_active else model.state_dict()
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg_loss,
                    "eval_loss": eval_loss,
                    "dataset_config": args.dataset_config,
                    "dev_dn_list": dev_dn_list,
                    "ema": ema_active,
                },
                best_ckpt_path,
            )

    final_path = checkpoint_dir / f"pv2026_forecast_vit_final_{ckpt_suffix}.pt"
    final_state = ema.state_dict() if ema is not None else model.state_dict()
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": final_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "dataset_config": args.dataset_config,
            "dev_dn_list": dev_dn_list,
            "ema": ema is not None,
        },
        final_path,
    )
    print(f"Saved final checkpoint to {final_path}")

    if best_ckpt_path.is_file():
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_metrics = evaluate_first_point(
            model,
            device,
            test_loader,
            criterion,
            weather_score_threshold=WEATHER_SCORE_THRESHOLD,
        )
        print(
            f"Test set with best eval-loss checkpoint ({best_ckpt_path.name}, epoch={ckpt.get('epoch', '?')}): "
            f"loss={test_metrics.loss:.6f}, RMSE={test_metrics.rmse:.6f}, MAE={test_metrics.mae:.6f}"
        )
        metrics_log = checkpoint_dir / f"pv2026_forecast_{ckpt_suffix}.txt"
        header = EvalMetrics.log_header()
        with open(metrics_log, "a", encoding="utf-8") as mf:
            if (not metrics_log.exists()) or metrics_log.stat().st_size == 0:
                mf.write(header)
            else:
                first_line = metrics_log.read_text(encoding="utf-8").splitlines()[0]
                if first_line.strip() != header.strip():
                    mf.write("\n" + header)
            mf.write(test_metrics.to_log_line())
        print(f"Appended best-test metrics to {metrics_log}")
    else:
        print(f"No {best_ckpt_path.name} on disk; skip best-checkpoint test evaluation.")

    writer.close()


if __name__ == "__main__":
    main()

