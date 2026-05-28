"""
Training script for pv_forecasting_model_vit (sat + sky TimeSformer + PV TCN).
Uses the same Luoyang CSV / sky / sat DataLoader as train.py.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

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

from dataloader.ylj import YLJDataset, collate_batched
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
        "kt": batch["kt"].to(device),
        "kt_mask": batch["kt_mask"].to(device),
        "p_mean": batch["p_mean"].to(device),
        "target_pv": batch["target_pv"].to(device),
        "target_mask": batch["target_mask"].to(device),
        "target_p_cs": batch["target_p_cs"].to(device),
    }
    for key in ("sat_tensor", "sat_timefeats", "skimg_tensor", "skimg_timefeats", "nwp_tensor"):
        v = batch.get(key)
        out[key] = None if v is None else v.to(device)
    return out


def forward_vit(model: nn.Module, d: dict) -> torch.Tensor:
    return model(
        d["device_id"],
        d["kt"],
        pv_mask=d["kt_mask"],
        pv_timefeats=d["pv_timefeats"],                  # [B, T, C=9]
        forecast_timefeats=d["forecast_timefeats"],      # [B, T, C=9]
        sat_tensor=d["sat_tensor"],                      # [B, T=24, C=3, H=100, W=100]
        sat_timefeats=d["sat_timefeats"],                # [B, T=24, C=9]
        skimg_tensor=d["skimg_tensor"],                  # [B, T=30, C=3, H=224, W=224]
        skimg_timefeats=d["skimg_timefeats"],            # [B, T=30, C=9]
        nwp_tensor=d["nwp_tensor"],                      # [B, T_out, 7] or None
    )


class ModelEMA:
    """Exponential Moving Average of model weights.

    Maintains a shadow copy of floating-point parameters updated as:
        shadow = decay * shadow + (1 - decay) * param
    Use ``ema.apply(model)`` as a context manager to temporarily swap the
    model weights to EMA weights for evaluation, then restore originals.
    """

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
        """Temporarily load EMA weights into *model* for eval, then restore."""
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

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.detach().clone().float() for k, v in state_dict.items()}


def _checkpoint_payload(
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler,
    dev_dn_list: list,
    loss: float,
    best_val_loss: float | None = None,
    ema: ModelEMA | None = None,
    ema_active: bool = False,
    use_ema_weights: bool = False,
) -> dict:
    state = ema.state_dict() if use_ema_weights and ema is not None else model.state_dict()
    payload = {
        "epoch": epoch,
        "model_state_dict": state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
        "dev_dn_list": dev_dn_list,
        "ema": ema_active,
    }
    if best_val_loss is not None:
        payload["best_val_loss"] = best_val_loss
    if ema is not None:
        payload["ema_state_dict"] = ema.state_dict()
    return payload


def _load_resume_checkpoint(
    path: Path,
    device: torch.device,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler,
    ema: ModelEMA | None,
) -> tuple[int, float]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    else:
        print("[resume] checkpoint has no optimizer_state_dict; optimizer stays freshly initialized")
    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    else:
        print("[resume] checkpoint has no scheduler_state_dict; scheduler stays freshly initialized")
    if ema is not None:
        ema_sd = ckpt.get("ema_state_dict")
        if ema_sd:
            ema.load_state_dict(ema_sd)
        else:
            for k, v in model.state_dict().items():
                if k in ema.shadow:
                    ema.shadow[k].copy_(v.detach().float())
    last_epoch = int(ckpt.get("epoch", 0))
    start_epoch = last_epoch + 1
    loss_min = float(ckpt.get("best_val_loss", ckpt.get("loss", 1e8)))
    print(
        f"[resume] loaded {path}  last_epoch={last_epoch}  "
        f"next_epoch={start_epoch}  best_val_loss={loss_min:.6f}"
    )
    return start_epoch, loss_min


def train_one_epoch(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    max_batches: int | None = None,
    ema: ModelEMA | None = None,
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
        kt_pred = forward_vit(model, d) * 1000.0
        pv_pred = kt_pred * d["target_p_cs"] * d["p_mean"].unsqueeze(1)   # [B, T_out=192]

        loss_15m = criterion( pv_pred[:,0], d["target_pv"][:,0] )
        loss_4h = criterion( pv_pred[:,15], d["target_pv"][:,15] )
        loss_48h = criterion( pv_pred, d["target_pv"] )
        
        lambda_15m = 2.0
        lambda_4h = 2.0
        lambda_48h = 1.0
        loss = lambda_15m*loss_15m + lambda_4h*loss_4h + lambda_48h*loss_48h
        
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model)
        total_loss += loss.item()
        n += B
    print()
    return total_loss / max(n, 1)


_HORIZON_15M_IDX = 0
_HORIZON_4H_IDX = 15
_FORECAST_STEP_MINUTES = 15


def _forecast_timestamp(t0, step_idx: int) -> str:
    """Return ISO time for forecast step ``step_idx`` (step 0 = t0 + 15 min)."""
    anchor = datetime.fromisoformat(str(t0).replace("Z", "+00:00"))
    target = anchor + timedelta(minutes=(step_idx + 1) * _FORECAST_STEP_MINUTES)
    return target.strftime("%Y-%m-%dT%H:%M:%S")


def _is_daily_9am(t0) -> bool:
    """True when ``t0`` falls on 09:00:00 (same timezone as stored in parquet)."""
    ts = datetime.fromisoformat(str(t0).replace("Z", "+00:00"))
    return ts.hour == 9 and ts.minute == 0 and ts.second == 0


def _rmse_mae(pred: np.ndarray, tgt: np.ndarray) -> tuple[float, float]:
    err = pred.astype(np.float64) - tgt.astype(np.float64)
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return rmse, mae


def _open_pred_log_dir(save_dir: Path) -> dict[str, object]:
    """Open append-only prediction logs for 15m / 4h / 48h under ``save_dir``."""
    save_dir.mkdir(parents=True, exist_ok=True)
    return {
        "15m": open(save_dir / "pv_pred_15m.txt", "a", encoding="utf-8"),
        "4h": open(save_dir / "pv_pred_4h.txt", "a", encoding="utf-8"),
        "48h": open(save_dir / "pv_pred_48h.txt", "a", encoding="utf-8"),
    }


def _close_pred_logs(logs: dict[str, object] | None) -> None:
    if not logs:
        return
    for f in logs.values():
        f.close()


def evaluate(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    criterion: nn.Module,
    *,
    save_file: bool = False,
    save_path: Path | str | None = None,
) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    pred_dict = {}
    target_dict = {}
    pred_logs = _open_pred_log_dir(Path(save_path)) if save_file and save_path else None
    try:
        with torch.no_grad():
            for batch in loader:
                d = _batch_to_device(batch, device)
                B = d["device_id"].size(0)
                kt_pred = forward_vit(model, d) * 1000.0
                pv_pred = kt_pred * d["target_p_cs"] * d["p_mean"].unsqueeze(1) # [B, T_out=192]

                loss_15m = criterion( pv_pred[:,0], d["target_pv"][:,0] )
                loss_4h = criterion( pv_pred[:,16], d["target_pv"][:,16] )
                loss_48h = criterion( pv_pred, d["target_pv"] )
                
                lambda_15m = 2.0
                lambda_4h = 2.0
                lambda_48h = 1.0
                loss = lambda_15m*loss_15m + lambda_4h*loss_4h + lambda_48h*loss_48h

                total_loss += loss.item()
                n += B
                # save these values and used them to compute the MAE and RMSE of the total station
                for i in range(pv_pred.shape[0]):
                    kk = int(d["device_id"][i].item()) # the inverter ID
                    pred_np = pv_pred[i, :].detach().cpu().float().numpy().copy()
                    tgt_np = d["target_pv"][i, :].detach().cpu().float().numpy().copy()
                    
                    if kk not in pred_dict:
                        pred_dict[kk] = []
                        target_dict[kk] = []
                    pred_dict[kk].append(pred_np)
                    target_dict[kk].append(tgt_np)

                    if pred_logs is not None:
                        t0 = batch["t0"][i]
                        t_15m = _forecast_timestamp(t0, _HORIZON_15M_IDX)
                        pred_logs["15m"].write(
                            f"{t_15m}\t{pred_np[_HORIZON_15M_IDX]:.6f}\t{tgt_np[_HORIZON_15M_IDX]:.6f}\n"
                        )
                        t_4h = _forecast_timestamp(t0, _HORIZON_4H_IDX)
                        pred_logs["4h"].write(
                            f"{t_4h}\t{pred_np[_HORIZON_4H_IDX]:.6f}\t{tgt_np[_HORIZON_4H_IDX]:.6f}\n"
                        )
                        if _is_daily_9am(t0):
                            pred_seq = " ".join(f"{x:.6f}" for x in pred_np[59:155])
                            tgt_seq = " ".join(f"{x:.6f}" for x in tgt_np[59:155])
                            pred_logs["48h"].write(f"{t0}\t{pred_seq}\t{tgt_seq}\n")

            
            pred_rows: list[np.ndarray] = []
            tgt_rows: list[np.ndarray] = []
            for kk in pred_dict:
                pred_rows.extend(pred_dict[kk])
                tgt_rows.extend(target_dict[kk])
            pred_arr = np.asarray(pred_rows, dtype=np.float64)
            tgt_arr = np.asarray(tgt_rows, dtype=np.float64)

            rmse_15m, mae_15m = _rmse_mae(pred_arr[:, _HORIZON_15M_IDX], tgt_arr[:, _HORIZON_15M_IDX])
            rmse_4h, mae_4h = _rmse_mae(pred_arr[:, _HORIZON_4H_IDX], tgt_arr[:, _HORIZON_4H_IDX])
            rmse_48h, mae_48h = _rmse_mae(pred_arr.reshape(-1), tgt_arr.reshape(-1))

            capacity = 535.87 # MW
            print(f"Capacity: {capacity}(KW)")
            for label, rmse, mae in (
                ("15m", rmse_15m, mae_15m),
                ("4h", rmse_4h, mae_4h),
                ("48h", rmse_48h, mae_48h),
            ):
                print(
                    f"  horizon={label}: MAE={mae:.6f}, RMSE={rmse:.6f}, "
                    f"ACC(MAE)={1.0 - mae / capacity:.6f}, ACC(RMSE)={1.0 - rmse / capacity:.6f}"
                )
    finally:
        _close_pred_logs(pred_logs)
    
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
        default=5,
        help="Linear LR warmup in epoch units before cosine decay (0 = no warmup).",
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=1e-6,
        help="Minimum learning rate for cosine tail (default 1e-6).",
    )
    parser.add_argument(
        "--use-ema",
        dest="use_ema",
        action="store_true",
        help="Enable Exponential Moving Average of model weights.",
    )
    parser.add_argument(
        "--no-ema",
        dest="use_ema",
        action="store_false",
        help="Disable EMA (default).",
    )
    parser.set_defaults(use_ema=False)
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.99,
        help="EMA decay (default 0.99). Lower decay catches up faster.",
    )
    parser.add_argument(
        "--ema-warmup-epochs",
        type=int,
        default=5,
        help="Skip EMA updates and use raw model weights for the first N epochs "
             "so EMA doesn't lag while model is far from convergence.",
    )
    parser.add_argument("--batch_size", type=int, default=int(h["batch_size"]))
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Path to a .pt checkpoint to resume training "
            "(loads model, optimizer, scheduler, EMA; continues from epoch+1). "
            "Prefer pv_forecast_vit_epoch_* or pv_forecast_vit_final_* over best."
        ),
    )
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
    Build YLJDataset kwargs from ``config/datasets/<dataset_config_name>``.

    The dataset YAML is expected to contain:
      - ``paths.ylj_train_val_parquet`` (train/val)
      - ``paths.ylj_test_parquet`` (test, optional until ready)
      - ``sampling.train_time_fraction`` / ``val_time_fraction`` / ``train_epoch_len``
    """
    cfg_path = _resolve_named_config(_DATASETS_CONFIG_DIR, dataset_config_name, "dataset-config")
    cfg = _load_yaml(cfg_path)
    sampling_cfg = cfg.get("sampling", {}) or {}

    train_epoch_len = sampling_cfg.get("train_epoch_len")
    return dict(
        config_path=str(cfg_path),
        split=split,
        train_time_fraction=float(sampling_cfg.get("train_time_fraction", 0.70)),
        val_time_fraction=float(sampling_cfg.get("val_time_fraction", 0.15)),
        train_epoch_len=int(train_epoch_len) if split == "train" and train_epoch_len is not None else None,
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
    DATASET_CLS = YLJDataset
    DATASET_CONFIG = "conf_ylj.yaml"

    train_dataset = DATASET_CLS(**_dataset_kwargs(DATASET_CONFIG, "train"))
    val_dataset = DATASET_CLS(**_dataset_kwargs(DATASET_CONFIG, "val"))
    test_dataset = DATASET_CLS(**_dataset_kwargs(DATASET_CONFIG, "test"))

    # Folsom: use training/train_vit_test_folsom.py with config/datasets/conf_folsom.yaml

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
    ema: ModelEMA | None = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None
    print(f"EMA: {'enabled' if args.use_ema else 'disabled'}"
          + (f" (decay={args.ema_decay}, warmup={args.ema_warmup_epochs} epoch)" if args.use_ema else ""))

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

    tb_log_dir = _PROJECT_ROOT / "runs" / _ckpt_suffix
    writer = SummaryWriter(log_dir=str(tb_log_dir))
    print(f"TensorBoard log dir: {tb_log_dir}")

    start_epoch = 1
    loss_min = 1e8
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"--resume checkpoint not found: {resume_path}")
        start_epoch, loss_min = _load_resume_checkpoint(
            resume_path, device, model, optimizer, scheduler, ema
        )
        if start_epoch > args.epochs:
            raise ValueError(
                f"--resume next epoch ({start_epoch}) is past --epochs ({args.epochs}); "
                "increase --epochs for continued training."
            )

    # initial_test_loss = evaluate(model, device, test_loader, criterion)
    # print(f"Initial test loss: {initial_test_loss:.6f}")

    for epoch in range(start_epoch, args.epochs + 1):
        cur_lr = optimizer.param_groups[0]["lr"]
        ema_active = ema is not None and epoch > args.ema_warmup_epochs
        if ema is not None and not ema_active:
            for k, v in model.state_dict().items():
                if k in ema.shadow:
                    ema.shadow[k].copy_(v.detach().float())
        avg_loss = train_one_epoch(
            model,
            device,
            train_loader,
            criterion,
            optimizer,
            max_batches=args.train_max_batches_per_epoch,
            ema=ema if ema_active else None,
        )
        if ema_active:
            with ema.apply(model):
                val_loss = evaluate(model, device, val_loader, criterion)
        else:
            val_loss = evaluate(model, device, val_loader, criterion)
        print(
            f"Epoch {epoch}/{args.epochs}  lr={cur_lr:.2e}  "
            f"train_loss={avg_loss:.6f}  val_loss={val_loss:.6f}"
        )
        writer.add_scalar("loss/train", avg_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("lr", cur_lr, epoch)
        scheduler.step()

        if args.save_every and epoch % args.save_every == 0:
            path = checkpoint_dir / f"pv_forecast_vit_epoch_{epoch}_{_ckpt_suffix}.pt"
            torch.save(
                _checkpoint_payload(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    dev_dn_list=dev_dn_list,
                    loss=avg_loss,
                    best_val_loss=loss_min,
                    ema=ema,
                    ema_active=ema_active,
                ),
                path,
            )
            print(f"  saved {path}")
        
        if val_loss < loss_min:
            loss_min = val_loss
            torch.save(
                _checkpoint_payload(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    dev_dn_list=dev_dn_list,
                    loss=val_loss,
                    best_val_loss=loss_min,
                    ema=ema,
                    ema_active=ema_active,
                    use_ema_weights=ema_active,
                ),
                best_ckpt_path,
            )
            print(f"  saved best checkpoint (val_loss={val_loss:.6f}) -> {best_ckpt_path}")

    final_path = checkpoint_dir / f"pv_forecast_vit_final_{_ckpt_suffix}.pt"
    torch.save(
        _checkpoint_payload(
            epoch=args.epochs,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dev_dn_list=dev_dn_list,
            loss=val_loss,
            best_val_loss=loss_min,
            ema=ema,
            ema_active=ema is not None,
            use_ema_weights=ema is not None,
        ),
        final_path,
    )
    print(f"Saved final checkpoint to {final_path}")

    if best_ckpt_path.is_file():
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print("==================test==================")
        pred_log_dir = checkpoint_dir / f"test_pred_{_ckpt_suffix}"
        test_loss_best = evaluate(
            model,
            device,
            test_loader,
            criterion,
            save_file=True,
            save_path=pred_log_dir,
        )
        print(
            f"Test set with best val-loss checkpoint ({best_ckpt_path.name}, epoch={ckpt.get('epoch', '?')}): "
            f"loss={test_loss_best:.6f}"
        )
        writer.add_hparams(
            {
                "lr": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "warmup_epochs": args.warmup_epochs,
                "weight_decay": args.weight_decay,
            },
            {
                "hparam/test_loss": test_loss_best,
            },
        )
        metrics_log = checkpoint_dir / f"pv_forecast_4h_pv_sat_{_ckpt_suffix}.txt"
        with open(metrics_log, "a", encoding="utf-8") as mf:
            mf.write(f"{test_loss_best:.8f}\n")
        print(f"Appended best-test metrics to {metrics_log}")
        print(f"Saved test predictions under {pred_log_dir}/")
    else:
        print(f"No {best_ckpt_path.name} on disk; skip test evaluation with best checkpoint.")

    writer.close()




if __name__ == "__main__":
    main()
