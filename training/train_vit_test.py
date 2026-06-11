"""
Training script for pv_forecasting_model_vit (sat + sky TimeSformer + PV TCN).
Uses the same Luoyang CSV / sky / sat DataLoader as train.py.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from dataclasses import dataclass
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
WEATHER_SCORE_THRESHOLD = 0.5
sys.path.insert(0, str(_PROJECT_ROOT))

from dataloader.luoyang_zarr import PVDataset, collate_batched
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
        "target_weather_score": batch["target_weather_score"].to(device),
    }
    for key in ("sat_tensor", "sat_timefeats", "skimg_tensor", "skimg_timefeats", "nwp_tensor"):
        v = batch.get(key)
        out[key] = None if v is None else v.to(device)
    return out


@dataclass
class EvalMetrics:
    """Validation/test metrics for combined loss and per-horizon windows."""

    loss: float
    loss_15m: float
    loss_4h: float
    loss_48h: float
    rmse_15m: float
    mae_15m: float
    rmse_4h: float
    mae_4h: float
    rmse_48h: float
    mae_48h: float
    rmse_15m_ws: float = float("nan")
    mae_15m_ws: float = float("nan")
    n_valid_15m_ws: int = 0
    rmse_4h_ws: float = float("nan")
    mae_4h_ws: float = float("nan")
    n_valid_4h_ws: int = 0
    rmse_48h_ws: float = float("nan")
    mae_48h_ws: float = float("nan")
    n_valid_48h_ws: int = 0

    @property
    def rmse(self) -> float:
        return self.rmse_48h

    @property
    def mae(self) -> float:
        return self.mae_48h

    @staticmethod
    def _fmt_metric(v: float) -> str:
        return f"{v:.8f}" if np.isfinite(v) else "nan"

    @staticmethod
    def log_header() -> str:
        return (
            "loss_15m\trmse_15m\tmae_15m\t"
            "loss_4h\trmse_4h\tmae_4h\t"
            "loss_48h\trmse_48h\tmae_48h\t"
            "rmse_15m_ws\tmae_15m_ws\tn_valid_15m_ws\t"
            "rmse_4h_ws\tmae_4h_ws\tn_valid_4h_ws\t"
            "rmse_48h_ws\tmae_48h_ws\tn_valid_48h_ws\n"
        )

    def to_log_line(self) -> str:
        return (
            f"{self.loss_15m:.8f}\t{self.rmse_15m:.8f}\t{self.mae_15m:.8f}\t"
            f"{self.loss_4h:.8f}\t{self.rmse_4h:.8f}\t{self.mae_4h:.8f}\t"
            f"{self.loss_48h:.8f}\t{self.rmse_48h:.8f}\t{self.mae_48h:.8f}\t"
            f"{self._fmt_metric(self.rmse_15m_ws)}\t{self._fmt_metric(self.mae_15m_ws)}\t{self.n_valid_15m_ws}\t"
            f"{self._fmt_metric(self.rmse_4h_ws)}\t{self._fmt_metric(self.mae_4h_ws)}\t{self.n_valid_4h_ws}\t"
            f"{self._fmt_metric(self.rmse_48h_ws)}\t{self._fmt_metric(self.mae_48h_ws)}\t{self.n_valid_48h_ws}\n"
        )


def forward_vit(model: nn.Module, d: dict) -> torch.Tensor:
    return model(
        d["device_id"],
        d["kt"]/20.0,
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
        kt_pred = forward_vit(model, d) * 20.0
        pv_pred = kt_pred * d["target_p_cs"] * d["p_mean"].unsqueeze(1)

        loss_15m = criterion( pv_pred[:,0], d["target_pv"][:,0] )
        loss_4h = criterion( pv_pred[:,15], d["target_pv"][:,15] )
        loss_48h = criterion( pv_pred, d["target_pv"] )

        lambda_15m = 2.0
        lambda_4h = 2.0
        lambda_48h = 1.0
        loss = lambda_15m*loss_15m + lambda_4h*loss_4h + lambda_48h*loss_48h
        # loss = criterion(pv_pred, d["target_pv"])
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model)
        total_loss += loss.item()
        n += B
    print()
    return total_loss / max(n, 1)


def evaluate(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    criterion: nn.Module,
    *,
    weather_score_threshold: float | None = None,
) -> EvalMetrics:
    model.eval()
    total_loss = 0.0
    total_loss_15m = 0.0
    total_loss_4h = 0.0
    total_loss_48h = 0.0
    n = 0
    pred_dict: dict[int, list[np.ndarray]] = {}
    target_dict: dict[int, list[np.ndarray]] = {}
    ws_dict: dict[int, list[np.ndarray]] = {}
    mask_dict: dict[int, list[np.ndarray]] = {}
    collect_weather = weather_score_threshold is not None
    with torch.no_grad():
        for batch in loader:
            d = _batch_to_device(batch, device)
            B = d["device_id"].size(0)
            kt_pred = forward_vit(model, d) * 20.0
            pv_pred = kt_pred * d["target_p_cs"] * d["p_mean"].unsqueeze(1)
            loss_15m = criterion( pv_pred[:,0], d["target_pv"][:,0] )
            loss_4h = criterion( pv_pred[:,15], d["target_pv"][:,15] )
            loss_48h = criterion( pv_pred, d["target_pv"] )

            lambda_15m = 2.0
            lambda_4h = 2.0
            lambda_48h = 1.0
            loss = lambda_15m * loss_15m + lambda_4h * loss_4h + lambda_48h * loss_48h
            # loss = criterion(pv_pred, d["target_pv"])
            total_loss += loss.item()
            total_loss_15m += loss_15m.item()
            total_loss_4h += loss_4h.item()
            total_loss_48h += loss_48h.item()
            n += B
            # save these values and used them to compute the MAE and RMSE of the total station
            for i in range(pv_pred.shape[0]):
                kk = int(d["device_id"][i].item())  # the inverter ID
                pred_np = pv_pred[i, :].detach().cpu().float().numpy().copy()
                tgt_np = d["target_pv"][i, :].detach().cpu().float().numpy().copy()

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
                        d["target_weather_score"][i, :].detach().cpu().float().numpy().copy()
                    )
                    mask_dict[kk].append(
                        d["target_mask"][i, :].detach().cpu().float().numpy().copy()
                    )

        # Sum inverter outputs per sample -> station-level [N, T] arrays.
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
                    station_mask = np.clip(station_mask + np.asarray(mask_dict[kk]), 0.0, 1.0)

        if collect_weather and station_ws is None:
            print(
                "[evaluate] WARNING: weather_score metrics requested but no weather_score "
                "collected (empty loader?)"
            )

        def _mae_rmse(pred: np.ndarray, tgt: np.ndarray) -> tuple[float, float]:
            diff = pred - tgt
            return float(np.mean(np.abs(diff))), float(np.sqrt(np.mean(diff ** 2)))

        def _mae_rmse_masked(
            pred: np.ndarray, tgt: np.ndarray, valid: np.ndarray
        ) -> tuple[float, float, int]:
            valid = valid.astype(bool)
            n_valid = int(valid.sum())
            if n_valid == 0:
                return float("nan"), float("nan"), 0
            p = pred[valid]
            t = tgt[valid]
            diff = p - t
            return float(np.mean(np.abs(diff))), float(np.sqrt(np.mean(diff ** 2))), n_valid

        # Horizons aligned with training loss: idx 0 = t0+15m, idx 15 = t0+4h, all = 48h window.
        idx_15m, idx_4h = 0, 15
        mae_15m, rmse_15m = _mae_rmse(station_pred[:, idx_15m], station_target[:, idx_15m])
        mae_4h, rmse_4h = _mae_rmse(station_pred[:, idx_4h], station_target[:, idx_4h])
        mae_48h, rmse_48h = _mae_rmse(station_pred.reshape(-1), station_target.reshape(-1))

        rmse_15m_ws = mae_15m_ws = float("nan")
        rmse_4h_ws = mae_4h_ws = float("nan")
        rmse_48h_ws = mae_48h_ws = float("nan")
        n_valid_15m_ws = n_valid_4h_ws = n_valid_48h_ws = 0

        capacity = 54600
        for label, mae, rmse in (
            ("t0+15m", mae_15m, rmse_15m),
            ("t0+4h", mae_4h, rmse_4h),
            ("t0+48h", mae_48h, rmse_48h),
        ):
            print(f"RMSE/MAE [{label}]. Capacity: {capacity}(KW)")
            print(
                f"  MAE: {mae:.6f}, RMSE: {rmse:.6f}, "
                f"ACC(MAE): {1.0 - mae / capacity:.6f}, ACC(RMSE): {1.0 - rmse / capacity:.6f}"
            )

        if collect_weather and station_ws is not None:
            ws_thr = float(weather_score_threshold)
            valid_15m = (station_ws[:, idx_15m] > ws_thr) & (station_mask[:, idx_15m] > 0.5)
            valid_4h = (station_ws[:, idx_4h] > ws_thr) & (station_mask[:, idx_4h] > 0.5)
            valid_48h = (
                (station_ws.reshape(-1) > ws_thr) & (station_mask.reshape(-1) > 0.5)
            )
            ws_label_suffix = f"weather_score>{ws_thr:g} & valid"
            mae_15m_ws, rmse_15m_ws, n_valid_15m_ws = _mae_rmse_masked(
                station_pred[:, idx_15m], station_target[:, idx_15m], valid_15m
            )
            mae_4h_ws, rmse_4h_ws, n_valid_4h_ws = _mae_rmse_masked(
                station_pred[:, idx_4h], station_target[:, idx_4h], valid_4h
            )
            mae_48h_ws, rmse_48h_ws, n_valid_48h_ws = _mae_rmse_masked(
                station_pred.reshape(-1), station_target.reshape(-1), valid_48h
            )
            for label, mae, rmse, n_valid in (
                ("t0+15m", mae_15m_ws, rmse_15m_ws, n_valid_15m_ws),
                ("t0+4h", mae_4h_ws, rmse_4h_ws, n_valid_4h_ws),
                ("t0+48h", mae_48h_ws, rmse_48h_ws, n_valid_48h_ws),
            ):
                print(f"RMSE/MAE [{label}, {ws_label_suffix}]. Capacity: {capacity}(KW)")
                if n_valid == 0:
                    print("  MAE: nan, RMSE: nan, n_valid=0")
                else:
                    print(
                        f"  MAE: {mae:.6f}, RMSE: {rmse:.6f}, "
                        f"ACC(MAE): {1.0 - mae / capacity:.6f}, "
                        f"ACC(RMSE): {1.0 - rmse / capacity:.6f}, n_valid={n_valid}"
                    )

        mae, rmse = mae_48h, rmse_48h

    denom = max(n, 1)
    return EvalMetrics(
        loss=total_loss / denom,
        loss_15m=total_loss_15m / denom,
        loss_4h=total_loss_4h / denom,
        loss_48h=total_loss_48h / denom,
        rmse_15m=rmse_15m,
        mae_15m=mae_15m,
        rmse_4h=rmse_4h,
        mae_4h=mae_4h,
        rmse_48h=rmse_48h,
        mae_48h=mae_48h,
        rmse_15m_ws=rmse_15m_ws,
        mae_15m_ws=mae_15m_ws,
        n_valid_15m_ws=n_valid_15m_ws,
        rmse_4h_ws=rmse_4h_ws,
        mae_4h_ws=mae_4h_ws,
        n_valid_4h_ws=n_valid_4h_ws,
        rmse_48h_ws=rmse_48h_ws,
        mae_48h_ws=mae_48h_ws,
        n_valid_48h_ws=n_valid_48h_ws,
    )


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
        train_samples_per_csv=int(sampling_cfg.get("train_samples_per_csv", 1)),
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
    DATASET_CONFIG = "conf_luoyang_shm.yaml"  # sat/sky read from /dev/shm (RAM); see scripts/vit_test.sh

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

    # initial_test_loss, _, _ = evaluate(model, device, test_loader, criterion)
    # print(f"Initial test loss: {initial_test_loss:.6f}")

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
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
                val_metrics = evaluate(model, device, val_loader, criterion)
        else:
            val_metrics = evaluate(model, device, val_loader, criterion)
        val_loss = val_metrics.loss
        print(
            f"Epoch {epoch}/{args.epochs}  lr={cur_lr:.2e}  "
            f"train_loss={avg_loss:.6f}  val_loss={val_loss:.6f}"
        )
        writer.add_scalar("loss/train", avg_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("metric/val_rmse", val_metrics.rmse, epoch)
        writer.add_scalar("metric/val_mae", val_metrics.mae, epoch)
        writer.add_scalar("lr", cur_lr, epoch)
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
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = ema.state_dict() if ema_active else model.state_dict()
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg_loss,
                    "val_loss": val_loss,
                    "dev_dn_list": dev_dn_list,
                    "ema": ema_active,
                },
                best_ckpt_path,
            )

    final_path = checkpoint_dir / f"pv_forecast_vit_final_{_ckpt_suffix}.pt"
    final_state = ema.state_dict() if ema is not None else model.state_dict()
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": final_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "dev_dn_list": dev_dn_list,
            "ema": ema is not None,
        },
        final_path,
    )
    print(f"Saved final checkpoint to {final_path}")

    if best_ckpt_path.is_file():
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_metrics = evaluate(
            model,
            device,
            test_loader,
            criterion,
            weather_score_threshold=WEATHER_SCORE_THRESHOLD,
        )
        print(
            f"Test set with best val-loss checkpoint ({best_ckpt_path.name}, epoch={ckpt.get('epoch', '?')}): "
            f"loss={test_metrics.loss:.6f}, RMSE={test_metrics.rmse:.6f}, MAE={test_metrics.mae:.6f}"
        )
        writer.add_scalar("metric/test_rmse", test_metrics.rmse, args.epochs)
        writer.add_scalar("metric/test_mae", test_metrics.mae, args.epochs)
        writer.add_hparams(
            {
                "lr": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "warmup_epochs": args.warmup_epochs,
                "weight_decay": args.weight_decay,
            },
            {
                "hparam/test_rmse": test_metrics.rmse,
                "hparam/test_mae": test_metrics.mae,
            },
        )
        metrics_log = checkpoint_dir / f"pv_forecast_pv_sat_{_ckpt_suffix}.txt"
        header = EvalMetrics.log_header()
        with open(metrics_log, "a", encoding="utf-8") as mf:
            if not metrics_log.exists() or metrics_log.stat().st_size == 0:
                mf.write(header)
            else:
                first_line = metrics_log.read_text(encoding="utf-8").splitlines()[0]
                if first_line.strip() != header.strip():
                    mf.write("\n" + header)
            mf.write(test_metrics.to_log_line())
        print(f"Appended best-test metrics to {metrics_log}")
    else:
        print(f"No {best_ckpt_path.name} on disk; skip test evaluation with best checkpoint.")

    writer.close()




if __name__ == "__main__":
    main()
