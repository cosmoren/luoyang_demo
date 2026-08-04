"""
Folsom trainer for ``pv_forecasting_model_vit_dinov2`` (parallel to ``train_vit_test_folsom.py``).

PV/GHI + sky by default: satellite off; NWP remapped Folsom→Luoyang slots on by default
(``--use-nwp`` / ``--no-use-nwp``; ``--use-satellite`` opt-in). The DINOv2 head currently
predicts a single horizon (``T_out=1``); Folsom targets may be longer (e.g. 16). Train/eval
slice targets/masks to the model output length (first-horizon loss only).

Uses ``dataloader.folsom.FolsomIrradianceDataset`` + ``collate_batched``. Configs:
``config/train/conf_train.yaml``, ``config/datasets/conf_folsom.yaml``.

Local smoke (1 logical GPU, tiny run):

  python training/train_vit_test_folsom_dinov2.py --epochs 1 --train_max_batches_per_epoch 3 \
    --eval_max_batches 2 --num_workers 0 --batch_size 1

  # NWP remapped on by default; zero sky for ablation:
  python training/train_vit_test_folsom_dinov2.py --zero-sky

  # NWP ablation (zero nwp_tensor):
  python training/train_vit_test_folsom_dinov2.py --no-use-nwp
"""
from __future__ import annotations

import argparse
import contextlib
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
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
_DEFAULT_FOLSOM_DATASET_CONFIG = "conf_folsom.yaml"
sys.path.insert(0, str(_PROJECT_ROOT))

from dataloader.folsom import (  # noqa: E402
    _DEFAULT_FOLSOM_TRAIN_EPOCH_LEN,
    _FOLSOM_HUBER_DELTA,
    _FOLSOM_KT_INPUT_SCALE,
    _FOLSOM_NWP_FEATURE_COLS,
    FolsomIrradianceDataset,
    normalize_ray_map,
    normalize_sky_mask,
    normalize_sun_mask,
    resolve_sun_mask_sigmas,
    sky_knobs_to_internal,
)
from dataloader.luoyang_zarr import collate_batched  # noqa: E402
from models.models import pv_forecasting_model_vit_dinov2  # noqa: E402

# Folsom merged-NWP feature indices (``_interpolate_nwp`` order; mask is the trailing channel).
_FOLSOM_NWP_DWSW_INDEX = _FOLSOM_NWP_FEATURE_COLS.index("dwsw")
_FOLSOM_NWP_PRESSURE_INDEX = _FOLSOM_NWP_FEATURE_COLS.index("pressure")
_FOLSOM_NWP_TEMPERATURE_INDEX = _FOLSOM_NWP_FEATURE_COLS.index("temperature")
_FOLSOM_NWP_WIND_U_INDEX = _FOLSOM_NWP_FEATURE_COLS.index("wind-u")
# Luoyang-style slots hardcoded by ``pv_forecasting_model_vit_dinov2`` forward:
# ``[ssrd, msl, t2m, u10, ...]`` (+ trailing invalid mask).
_VIT_DINOV2_NWP_SSRD_SLOT = 0
_VIT_DINOV2_NWP_MSL_SLOT = 1
_VIT_DINOV2_NWP_T2M_SLOT = 2
_VIT_DINOV2_NWP_U10_SLOT = 3
# Optional override for train/eval loss+metrics horizon (first N forecast steps).
# ``None`` = use ``sampling.pv_output_len`` from the dataset config (default). Set an int
# to score loss on fewer steps while keeping full model ``T_out``.
# DINOv2 head is single-step; default loss/metrics to first forecast horizon only.
_LOSS_METRIC_HORIZON: int | None = 1


def remap_nwp_tensor_for_pv_vit_dinov2(nwp_tensor: torch.Tensor) -> torch.Tensor:
    """
    Copy Folsom merged-NWP columns into the Luoyang slot layout ``vit_dinov2`` hardcodes.

    Folsom ``_interpolate_nwp`` stacks ``_FOLSOM_NWP_FEATURE_COLS`` then appends an invalid
    mask. The model reads Luoyang slots ``0=ssrd``, ``1=msl``, ``2=t2m``, ``3=u10`` and applies
    its own Luoyang scaling — this helper only reorders/copies values from the original
    unmodified tensor (no pressure rescale; unused Folsom feature cols 4–7 and the trailing
    mask stay as-is).
    """
    if nwp_tensor.ndim != 3:
        raise ValueError(f"nwp_tensor expected [B, T, C], got shape {tuple(nwp_tensor.shape)}")
    n_feat = len(_FOLSOM_NWP_FEATURE_COLS)
    if nwp_tensor.shape[-1] != n_feat + 1:
        raise ValueError(
            f"nwp_tensor last dim expected {n_feat + 1} (features + mask), got {nwp_tensor.shape[-1]}"
        )
    out = nwp_tensor.clone()
    out[:, :, _VIT_DINOV2_NWP_SSRD_SLOT] = nwp_tensor[:, :, _FOLSOM_NWP_DWSW_INDEX]
    out[:, :, _VIT_DINOV2_NWP_MSL_SLOT] = nwp_tensor[:, :, _FOLSOM_NWP_PRESSURE_INDEX]
    out[:, :, _VIT_DINOV2_NWP_T2M_SLOT] = nwp_tensor[:, :, _FOLSOM_NWP_TEMPERATURE_INDEX]
    out[:, :, _VIT_DINOV2_NWP_U10_SLOT] = nwp_tensor[:, :, _FOLSOM_NWP_WIND_U_INDEX]
    return out


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
    first = parts[0]
    if first.isdigit():
        return int(first)
    return torch.cuda.current_device()


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    # Diagnostic memory run: keep H2D sync (non_blocking=False) so transfers are not
    # overlapped with host work / pinned-buffer staging.
    out = {
        "device_id": batch["dev_idx"].to(device, non_blocking=False),
        "pv": batch["pv"].to(device, non_blocking=False),
        "pv_mask": batch["pv_mask"].to(device, non_blocking=False),
        "pv_timefeats": batch["pv_timefeats"].to(device, non_blocking=False),
        "forecast_timefeats": batch["forecast_timefeats"].to(device, non_blocking=False),
        "kt": batch["kt"].to(device, non_blocking=False),
        "kt_mask": batch["kt_mask"].to(device, non_blocking=False),
        "p_mean": batch["p_mean"].to(device, non_blocking=False),
        "target_pv": batch["target_pv"].to(device, non_blocking=False),
        "target_mask": batch["target_mask"].to(device, non_blocking=False),
        "target_p_cs": batch["target_p_cs"].to(device, non_blocking=False),
    }
    for key in ("sat_tensor", "sat_timefeats", "skimg_tensor", "skimg_timefeats", "nwp_tensor"):
        v = batch.get(key)
        out[key] = None if v is None else v.to(device, non_blocking=False)
    return out


def _prepare_nwp_for_vit(d: dict, *, use_nwp: bool) -> dict:
    """
    Prepare ``d['nwp_tensor']`` for the ViT, in place.

      * ``use_nwp=True``  -> remap Folsom feature order into Luoyang slots for ``vit_dinov2``,
        then pass through (model scaling unchanged).
      * ``use_nwp=False`` -> overwrite with ``zeros_like`` (NWP-ablation baseline).

    Keep original shape/dtype/device; ``None`` would crash the forward pass.
    """
    nwp = d.get("nwp_tensor")
    if nwp is None:
        return d
    if not use_nwp:
        d["nwp_tensor"] = torch.zeros_like(nwp)
    else:
        d["nwp_tensor"] = remap_nwp_tensor_for_pv_vit_dinov2(nwp)
    return d


def _prepare_sky_for_vit(d: dict, *, zero_sky: bool) -> dict:
    """
    Optionally zero sky tensors after ``_batch_to_device`` so the ViT sees no sky signal while the
    dataloader still loads real Zarr/JPEG (avoids bogus paths). Matches the model branch for
    ``skimg_tensor.max() == 0`` (see ``pv_forecasting_model_vit_imgs``).
    """
    if not zero_sky:
        return d
    for key in ("skimg_tensor", "skimg_timefeats"):
        t = d.get(key)
        if t is not None:
            d[key] = torch.zeros_like(t)
    return d


def _seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: distinct-but-deterministic per-worker RNGs.

    ``dataloader.folsom`` calls ``np.random.choice`` per ``__getitem__``, so without this each
    worker would share whatever numpy/random state it forked with.
    """
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def forward_vit(model: nn.Module, d: dict) -> torch.Tensor:
    """Mirrors ``training/train_vit_test.py::forward_vit``: the ViT is fed normalized
    ``kt`` (clear-sky index / 4000.0) and the daytime ``kt_mask``; the caller scales the
    output back to ``kt`` and multiplies by ``target_p_cs * p_mean`` to recover ``pv``.
    Folsom's divisor is 4000 (vs Luoyang's 20) because Folsom kt is in W/m^2-ish units
    (numerator is raw GHI ~1000 W/m^2, denominator is dimensionless ``p_cs``) so empirical
    kt p99 ~= 1434 / max ~= 2630; ``/4000`` lands the ViT input at p99 ~= 0.36 and max ~=
    0.66, matching Luoyang's headroom (Luoyang p99/20 = 0.38, max/20 = 0.60)."""
    return model(
        d["device_id"],
        d["kt"] / _FOLSOM_KT_INPUT_SCALE,
        pv_mask=d["kt_mask"],
        pv_timefeats=d["pv_timefeats"],
        forecast_timefeats=d["forecast_timefeats"],
        sat_tensor=d["sat_tensor"],
        sat_timefeats=d["sat_timefeats"],
        skimg_tensor=d["skimg_tensor"],
        skimg_timefeats=d["skimg_timefeats"],
        nwp_tensor=d["nwp_tensor"],
    )


class ModelEMA:
    """Exponential Moving Average of model weights (same idea as ``training/train_vit_test.py``)."""

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


def train_one_epoch(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    max_batches: int | None = None,
    ema: ModelEMA | None = None,
    *,
    loss_metric_horizon: int,
    use_nwp: bool = True,
    zero_sky: bool = False,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    print("number of batches: ", len(loader))
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        d = _batch_to_device(batch, device)
        _prepare_nwp_for_vit(d, use_nwp=use_nwp)
        _prepare_sky_for_vit(d, zero_sky=zero_sky)
        B = d["device_id"].size(0)
        optimizer.zero_grad()
        kt_pred = forward_vit(model, d) * _FOLSOM_KT_INPUT_SCALE
        t_out = int(kt_pred.shape[1])
        target_len = int(d["target_pv"].shape[1])
        # DINOv2 is single-horizon (T_out=1) while Folsom targets may be longer (e.g. 16).
        assert target_len >= t_out, (kt_pred.shape, d["target_pv"].shape)
        pcs = d["target_p_cs"][:, :t_out]
        pv_pred = kt_pred * pcs * d["p_mean"].unsqueeze(1)
        h = min(int(loss_metric_horizon), t_out)
        m = d["target_mask"][:, :h]
        loss = criterion(
            (pv_pred[:, :h] * m),
            (d["target_pv"][:, :h] * m),
        )
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model)
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def evaluate(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    criterion: nn.Module,
    *,
    loss_metric_horizon: int,
    max_batches: int | None = None,
    use_nwp: bool = True,
    zero_sky: bool = False,
) -> tuple[float, float, float]:
    """Returns mean Huber loss (first ``loss_metric_horizon`` steps, masked like train), RMSE and
    MAE in **normalized** GHI space over the same slice (``target_mask``; predictions at night
    cos-zenith < 0 are zeroed before residuals, matching ``train_vit_test.py``).

    ``loss_metric_horizon`` defaults to 1 for this DINOv2 entrypoint (model ``T_out=1``);
    targets longer than the prediction are sliced to the first ``t_out`` steps.

    If ``max_batches`` is set, only the first N batches are used (smoke / faster dev runs; metrics
    are not a full pass over the split).
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    sum_abs = 0.0
    sum_sq = 0.0
    n_elem = 0.0
    horizon = int(loss_metric_horizon)
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            d = _batch_to_device(batch, device)
            _prepare_nwp_for_vit(d, use_nwp=use_nwp)
            _prepare_sky_for_vit(d, zero_sky=zero_sky)
            kt_pred = forward_vit(model, d) * _FOLSOM_KT_INPUT_SCALE
            t_out = int(kt_pred.shape[1])
            target_len = int(d["target_pv"].shape[1])
            assert target_len >= t_out, (kt_pred.shape, d["target_pv"].shape)
            pcs = d["target_p_cs"][:, :t_out]
            pv_pred = kt_pred * pcs * d["p_mean"].unsqueeze(1)
            h = min(horizon, t_out)
            m = d["target_mask"][:, :h]
            tgt = d["target_pv"][:, :h]
            loss = criterion((pv_pred[:, :h] * m), (tgt * m))
            total_loss += loss.item()
            n_batches += 1
            # Night mask on predictions (``forecast_timefeats[:, :, 3]`` == cos zenith), like ``train_vit_test``.
            pred_h = pv_pred[:, :h].clone()
            night = d["forecast_timefeats"][:, :h, 3] < 0
            pred_h[night] = 0.0
            diff = pred_h - tgt
            sum_abs += (diff.abs() * m).sum().item()
            sum_sq += ((diff ** 2) * m).sum().item()
            n_elem += m.sum().item()

    mean_loss = total_loss / max(n_batches, 1)
    # Post-alignment (commit 518dca9) target_pv is raw W/m^2, so per-element residual
    # means are already in W/m^2 -- no denormalization needed.
    mae_wm2 = sum_abs / max(n_elem, 1.0)
    rmse_wm2 = (sum_sq / max(n_elem, 1.0)) ** 0.5
    print(
        f"First-{horizon}-step metrics (masked GHI; pred zeroed at night): "
        f"MAE={mae_wm2:.4f} W/m²  RMSE={rmse_wm2:.4f} W/m²"
    )
    return mean_loss, rmse_wm2, mae_wm2


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


def _folsom_pv_dataset_config_path(base: Path) -> Path:
    """Return the dataset YAML path for ``FolsomIrradianceDataset`` (sky format auto-detected in loader)."""
    return base.resolve()


def _resolve_data_dir(paths_cfg: dict, cfg_path: Path) -> Path:
    raw = paths_cfg.get("data_dir")
    if raw is None or str(raw).strip() == "":
        raise KeyError(f"dataset config paths.data_dir is required (in {cfg_path})")
    p = Path(str(raw))
    return p.resolve() if p.is_absolute() else (_PROJECT_ROOT / p).resolve()


def _build_parser(h: dict, config_default: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train pv_forecasting_model_vit_dinov2 on Folsom (GHI as PV target)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=config_default,
        help=f"Training config filename under config/train/ (default: {config_default!r}).",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=_DEFAULT_FOLSOM_DATASET_CONFIG,
        help=f"Dataset YAML filename under config/datasets/ (default: {_DEFAULT_FOLSOM_DATASET_CONFIG!r}).",
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
        default=None,
        help=(
            "Linear LR warmup in epoch units before cosine decay (0 = no warmup). "
            "Default: floor(10%% of --epochs)."
        ),
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
        help="Skip EMA updates for the first N epochs (default 5).",
    )
    parser.add_argument("--batch_size", type=int, default=int(h["batch_size"]))
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for python/numpy/torch + dataloader workers (default 0).",
    )
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=int(h["save_every"]))
    parser.add_argument("--num_workers", type=int, default=int(h["num_workers"]))
    mb = h.get("train_max_batches_per_epoch")
    parser.add_argument(
        "--train_max_batches_per_epoch",
        type=int,
        default=None if mb is None else int(mb),
        help="Cap batches per epoch (default from train YAML; null = no cap).",
    )
    parser.add_argument(
        "--eval_max_batches",
        type=int,
        default=None,
        metavar="N",
        help="If set, cap val/test ``evaluate()`` to the first N batches each call (default: full loader).",
    )
    parser.add_argument(
        "--train_epoch_len",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Random anchor draws per epoch from the valid train pool. "
            "Precedence: this flag > sampling.train_epoch_len in the dataset YAML > "
            "dataloader default (_DEFAULT_FOLSOM_TRAIN_EPOCH_LEN). With replacement; "
            "more draws -> better anchor coverage at the cost of per-epoch wall time."
        ),
    )
    parser.add_argument(
        "--use-nwp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Feed remapped Folsom merged-NWP (Folsom columns → Luoyang vit_dinov2 slots) to the "
            "model. Default ON (like Luoyang always feeding NWP). Pass --no-use-nwp to zero "
            "nwp_tensor for an NWP-ablation baseline."
        ),
    )
    parser.add_argument(
        "--zero-sky",
        action="store_true",
        help=(
            "After each batch is on device, replace sky image tensors (and sky time features) with "
            "zeros so the ViT uses the empty-sky branch while the dataset still loads real Zarr/JPEG. "
            "Use for PV+NWP vs PV+NWP+sky comparisons (NWP is on by default)."
        ),
    )
    parser.add_argument(
        "--use-satellite",
        dest="use_satellite",
        action="store_true",
        help=(
            "Enable the Folsom GOES-15 satellite branch (loads per-frame NPY shards from "
            "<data_dir>/<paths.sat_path>/YYYY/MM/goes15_*.npy and feeds sat_tensor / "
            "sat_timefeats into the model). Overrides ``sampling.use_satellite`` in the "
            "dataset YAML when set."
        ),
    )
    parser.add_argument(
        "--no-use-satellite",
        dest="use_satellite",
        action="store_false",
        help=(
            "Force the satellite branch off (sat_tensor / sat_timefeats = None; model uses its "
            "zero-sat embedding). Overrides ``sampling.use_satellite`` in the dataset YAML."
        ),
    )
    parser.set_defaults(use_satellite=None)
    parser.add_argument(
        "--ray-map",
        dest="ray_map",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Add the fixed fisheye ray_map sky channels (3ch). Use --no-ray-map to force "
            "off. Precedence: this flag > sampling.ray_map in the dataset YAML > false."
        ),
    )
    parser.add_argument(
        "--sun-mask",
        dest="sun_mask",
        type=str,
        default=None,
        choices=[
            "none",
            "sun_only",
            "sun_halo",
            "gaussian_pixel",
            "gaussian_angular",
        ],
        metavar="MODE",
        help=(
            "Sun_mask channel (1ch): 'none' omits it; 'sun_only'/'sun_halo' hard discs "
            "(10/30 deg); 'gaussian_pixel'/'gaussian_angular' soft Gaussians (sigma from "
            "sampling.sun_mask_sigma_px / sun_mask_sigma_deg, defaults 15 px / 10 deg). "
            "Precedence: this flag > sampling.sun_mask in the dataset YAML > none."
        ),
    )
    parser.add_argument(
        "--sun-mask-sigma-px",
        dest="sun_mask_sigma_px",
        type=float,
        default=None,
        help=(
            "Override sampling.sun_mask_sigma_px for gaussian_pixel (default 15 when unset)."
        ),
    )
    parser.add_argument(
        "--sun-mask-sigma-deg",
        dest="sun_mask_sigma_deg",
        type=float,
        default=None,
        help=(
            "Override sampling.sun_mask_sigma_deg for gaussian_angular (default 10 when unset)."
        ),
    )
    parser.add_argument(
        "--sky-mask",
        dest="sky_mask",
        type=str,
        default=None,
        choices=["none", "loose", "tight", "valid_disc"],
        metavar="MODE",
        help=(
            "Append optional sky_mask channel (1ch float 0/1 keep-region disc). 'none' "
            "omits it; 'loose'/'tight' use hand-drawn masks; 'valid_disc' the optical-center "
            "disc. RGB is left unmodified. Precedence: this flag > sampling.sky_mask in "
            "YAML > none."
        ),
    )
    parser.add_argument(
        "--tb-log-dir",
        type=str,
        default=None,
        help=(
            "Explicit TensorBoard log directory for this run. Precedence: "
            "(1) this flag if set; (2) else derived from --checkpoint_dir as runs/<basename>; "
            "(3) else legacy default runs/folsom_pv_gpu{N}. "
            "Set this (or a distinct --checkpoint_dir) when launching parallel runs to avoid "
            "SummaryWriter event-file collisions."
        ),
    )
    return parser


def _dataset_kwargs(
    dataset_config_name: str,
    split: str,
    use_satellite_override: bool | None = None,
    sky_channels_override: tuple[str, ...] | None = None,
    sun_mask_mode_override: str | None = None,
    sun_mask_radius_deg_override: float | None = None,
    sun_mask_sigma_px_override: float | None = None,
    sun_mask_sigma_deg_override: float | None = None,
    sky_disc_mask_mode_override: str | None = None,
    sky_disc_mask_radius_px_override: float | None = None,
) -> dict:
    base_cfg_path = _resolve_named_config(_DATASETS_CONFIG_DIR, dataset_config_name, "dataset-config")
    cfg_path = _folsom_pv_dataset_config_path(base_cfg_path)
    cfg = _load_yaml(cfg_path)
    paths_cfg = cfg.get("paths", {}) or {}
    sampling_cfg = cfg.get("sampling", {}) or {}
    if not sampling_cfg:
        raise KeyError(
            f"dataset config {base_cfg_path} is missing a non-empty 'sampling:' section"
        )

    data_dir = _resolve_data_dir(paths_cfg, base_cfg_path)

    def _req_path(key: str) -> str:
        v = paths_cfg.get(key)
        if v is None or str(v).strip() == "":
            raise KeyError(f"dataset config paths.{key} is required (in {base_cfg_path})")
        return str(v)

    def _req_sampling(key: str):
        if key not in sampling_cfg:
            raise KeyError(f"dataset config sampling.{key} is required (in {base_cfg_path})")
        return sampling_cfg[key]

    pv_dir = (data_dir / _req_path("pv_path")).resolve()
    skyimg_dir = (data_dir / _req_path("sky_image_path")).resolve()
    satimg_dir = (data_dir / _req_path("sat_path")).resolve()

    shwc = _req_sampling("satimg_npy_shape_hwc")
    if not isinstance(shwc, (list, tuple)) or len(shwc) != 3:
        raise ValueError(f"sampling.satimg_npy_shape_hwc must be [H, W, C] (in {base_cfg_path})")

    if use_satellite_override is None:
        use_satellite = bool(sampling_cfg.get("use_satellite", False))
    else:
        use_satellite = bool(use_satellite_override)

    if sky_channels_override is not None:
        sky_channels = list(sky_channels_override)
        sun_mask_mode = normalize_sun_mask(sun_mask_mode_override)
        sun_mask_radius_deg = sun_mask_radius_deg_override
        sky_disc_mask_mode = sky_disc_mask_mode_override
        sky_disc_mask_radius_px = sky_disc_mask_radius_px_override
    else:
        _sc, _smr, _sdm, _smm = sky_knobs_to_internal(
            sampling_cfg.get("ray_map"),
            sampling_cfg.get("sun_mask"),
            sampling_cfg.get("sky_mask"),
        )
        sky_channels = list(_sc)
        sun_mask_mode = _smm
        sun_mask_radius_deg = _smr
        sky_disc_mask_mode = _sdm
        sky_disc_mask_radius_px = None

    sigma_px, sigma_deg = resolve_sun_mask_sigmas(
        sampling_cfg,
        sigma_px_override=sun_mask_sigma_px_override,
        sigma_deg_override=sun_mask_sigma_deg_override,
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
        pv_train_time_fraction=float(sampling_cfg.get("pv_train_time_fraction", 0.7)),
        test_anchor_stride_min=int(_req_sampling("test_anchor_stride_min")),
        val_anchor_stride_min=int(_req_sampling("val_anchor_stride_min")),
        test_collect_time_match_tolerance_min=int(sampling_cfg.get("test_collect_time_match_tolerance_min", 0)),
        train_split=float(sampling_cfg.get("train_split", 0.66)),
        val_split=float(sampling_cfg.get("val_split", 0.18)),
        test_split=float(sampling_cfg.get("test_split", 0.16)),
        skyimg_window_size=int(_req_sampling("skyimg_window_size")),
        skyimg_time_resolution_min=int(_req_sampling("skyimg_time_resolution_min")),
        skyimg_spatial_size=int(_req_sampling("skyimg_spatial_size")),
        satimg_window_size=int(_req_sampling("satimg_window_size")),
        satimg_time_resolution_min=int(_req_sampling("satimg_time_resolution_min")),
        satimg_npy_shape_hwc=tuple(int(x) for x in shwc),
        use_satellite=use_satellite,
        sky_channels=sky_channels,
        sun_mask_mode=sun_mask_mode,
        sun_mask_radius_deg=sun_mask_radius_deg,
        sun_mask_sigma_px=sigma_px,
        sun_mask_sigma_deg=sigma_deg,
        sky_disc_mask_mode=sky_disc_mask_mode,
        sky_disc_mask_radius_px=sky_disc_mask_radius_px,
    )


def _resolve_ray_map(dataset_config_name: str, cli_value: bool | None) -> bool:
    """Resolve the ``ray_map`` knob: CLI ``--ray-map/--no-ray-map`` > YAML > ``False``."""
    if cli_value is not None:
        return bool(cli_value)
    base_cfg_path = _resolve_named_config(_DATASETS_CONFIG_DIR, dataset_config_name, "dataset-config")
    cfg_path = _folsom_pv_dataset_config_path(base_cfg_path)
    cfg = _load_yaml(cfg_path)
    return normalize_ray_map((cfg.get("sampling", {}) or {}).get("ray_map"))


def _resolve_sun_mask(dataset_config_name: str, cli_value: str | None) -> str:
    """Resolve the ``sun_mask`` knob: CLI ``--sun-mask`` > YAML > ``'none'``."""
    if cli_value is not None:
        return normalize_sun_mask(cli_value)
    base_cfg_path = _resolve_named_config(_DATASETS_CONFIG_DIR, dataset_config_name, "dataset-config")
    cfg_path = _folsom_pv_dataset_config_path(base_cfg_path)
    cfg = _load_yaml(cfg_path)
    return normalize_sun_mask((cfg.get("sampling", {}) or {}).get("sun_mask"))


def _resolve_sky_mask(dataset_config_name: str, cli_value: str | None) -> str:
    """Resolve the ``sky_mask`` knob: CLI ``--sky-mask`` > YAML > ``'none'``."""
    if cli_value is not None:
        return normalize_sky_mask(cli_value)
    base_cfg_path = _resolve_named_config(_DATASETS_CONFIG_DIR, dataset_config_name, "dataset-config")
    cfg_path = _folsom_pv_dataset_config_path(base_cfg_path)
    cfg = _load_yaml(cfg_path)
    return normalize_sky_mask((cfg.get("sampling", {}) or {}).get("sky_mask"))


def _sky_knob_overrides(
    dataset_config_name: str,
    cli_ray_map: bool | None,
    cli_sun_mask: str | None,
    cli_sky_mask: str | None,
    cli_sun_mask_sigma_px: float | None = None,
    cli_sun_mask_sigma_deg: float | None = None,
) -> dict:
    """Resolve the 3 sky knobs (CLI > YAML > default) and translate to dataset overrides.

    Returns the ``*_override`` kwargs consumed by :func:`_dataset_kwargs`, keeping the
    dataset construction path stable.
    """
    ray_map = _resolve_ray_map(dataset_config_name, cli_ray_map)
    sun_mask = _resolve_sun_mask(dataset_config_name, cli_sun_mask)
    sky_mask = _resolve_sky_mask(dataset_config_name, cli_sky_mask)
    sky_channels, sun_mask_radius_deg, sky_disc_mask_mode, sun_mask_mode = sky_knobs_to_internal(
        ray_map, sun_mask, sky_mask
    )
    return dict(
        sky_channels_override=sky_channels,
        sun_mask_mode_override=sun_mask_mode,
        sun_mask_radius_deg_override=sun_mask_radius_deg,
        sun_mask_sigma_px_override=cli_sun_mask_sigma_px,
        sun_mask_sigma_deg_override=cli_sun_mask_sigma_deg,
        sky_disc_mask_mode_override=sky_disc_mask_mode,
        sky_disc_mask_radius_px_override=None,
    )


def _resolve_use_satellite(dataset_config_name: str, cli_value: bool | None) -> bool:
    """Resolve ``use_satellite``: CLI flag > YAML ``sampling.use_satellite`` > False."""
    if cli_value is not None:
        return bool(cli_value)
    base_cfg_path = _resolve_named_config(_DATASETS_CONFIG_DIR, dataset_config_name, "dataset-config")
    cfg_path = _folsom_pv_dataset_config_path(base_cfg_path)
    cfg = _load_yaml(cfg_path)
    return bool((cfg.get("sampling", {}) or {}).get("use_satellite", False))


def _resolve_train_epoch_len(dataset_config_name: str, cli_value: int | None) -> int | None:
    """Pick ``train_epoch_len`` precedence: CLI flag > YAML ``sampling.train_epoch_len`` > None.

    ``None`` means "leave the dataset's own default" (``_DEFAULT_FOLSOM_TRAIN_EPOCH_LEN``).
    The dataset constructor does not accept this kwarg; the trainer applies the result
    by writing ``train_dataset._train_epoch_len`` after construction.
    """
    if cli_value is not None:
        return int(cli_value)
    base_cfg_path = _resolve_named_config(_DATASETS_CONFIG_DIR, dataset_config_name, "dataset-config")
    cfg_path = _folsom_pv_dataset_config_path(base_cfg_path)
    cfg = _load_yaml(cfg_path)
    yaml_value = (cfg.get("sampling", {}) or {}).get("train_epoch_len")
    if yaml_value is None:
        return None
    return int(yaml_value)


_TRAINING_PARAM_CLI_FLAGS: dict[str, tuple[str, ...]] = {
    "epochs": ("--epochs",),
    "lr": ("--lr",),
    "batch_size": ("--batch_size",),
    "save_every": ("--save_every",),
    "num_workers": ("--num_workers",),
    "train_max_batches_per_epoch": ("--train_max_batches_per_epoch",),
    "train_epoch_len": ("--train_epoch_len",),
}


def _argv_has_cli_flag(argv: list[str], *flags: str) -> bool:
    for arg in argv:
        for flag in flags:
            if arg == flag or arg.startswith(flag + "="):
                return True
    return False


def _fmt_training_value(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _training_yaml_param_source(
    key: str,
    *,
    argv: list[str],
    train_conf_name: str,
    dataset_cfg_name: str,
    ds_training_override: dict,
) -> str:
    cli_flags = _TRAINING_PARAM_CLI_FLAGS.get(key, ())
    if cli_flags and _argv_has_cli_flag(argv, *cli_flags):
        return "CLI"
    if key in ds_training_override and ds_training_override[key] is not None:
        return dataset_cfg_name
    return train_conf_name


def _resolved_train_epoch_len(
    *,
    argv: list[str],
    dataset_cfg_name: str,
    dataset_cfg_raw: dict,
    cli_value: int | None,
) -> tuple[int, str]:
    if _argv_has_cli_flag(argv, "--train_epoch_len"):
        return max(1, int(cli_value)), "CLI"
    yaml_value = (dataset_cfg_raw.get("sampling") or {}).get("train_epoch_len")
    if yaml_value is not None:
        return max(1, int(yaml_value)), dataset_cfg_name
    return _DEFAULT_FOLSOM_TRAIN_EPOCH_LEN, "dataloader default"


def _print_resolved_training_block(
    *,
    args: argparse.Namespace,
    argv: list[str],
    train_conf_name: str,
    dataset_cfg_name: str,
    ds_training_override: dict,
) -> None:
    param_keys = (
        "epochs",
        "lr",
        "batch_size",
        "save_every",
        "num_workers",
        "train_max_batches_per_epoch",
    )
    arg_lookup = {
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "save_every": args.save_every,
        "num_workers": args.num_workers,
        "train_max_batches_per_epoch": args.train_max_batches_per_epoch,
    }
    rows: list[tuple[str, str]] = []
    for key in param_keys:
        rows.append((key, _fmt_training_value(arg_lookup[key])))

    headers = ("parameter", "value")
    cols = list(zip(headers, *rows)) if rows else [(h,) for h in headers]
    widths = [max(len(str(cell)) for cell in col) for col in cols]

    def _border() -> str:
        return "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def _row(cells: tuple[str, str]) -> str:
        return "| " + " | ".join(
            str(cell).ljust(widths[i]) for i, cell in enumerate(cells)
        ) + " |"

    table_width = len(_border())
    print()
    print("TRAINING".center(table_width))
    print(_border())
    print(_row(headers))
    print(_border())
    for row in rows:
        print(_row(row))
    print(_border())


def main() -> None:
    os.environ["FOLSOM_QUIET"] = "1"

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=_DEFAULT_TRAIN_CONF_NAME)
    pre_parser.add_argument("--dataset-config", type=str, default=_DEFAULT_FOLSOM_DATASET_CONFIG)
    pre_args, _ = pre_parser.parse_known_args()

    train_conf_path = _resolve_named_config(_TRAIN_CONFIG_DIR, pre_args.config, "config")
    train_conf = _load_yaml(train_conf_path)
    h = dict(train_conf.get("training") or {})
    if not h:
        raise KeyError(f"training config {train_conf_path} is missing a 'training:' section")

    # Dataset YAML may carry a ``training:`` override block (Folsom uses this for
    # epochs=40 etc, so dataset-specific knobs live alongside dataset paths/sampling
    # without forking the shared conf_train.yaml). Override only keys explicitly set
    # to a non-None value; missing keys inherit from the shared base.
    dataset_cfg_path = _resolve_named_config(_DATASETS_CONFIG_DIR, pre_args.dataset_config, "dataset-config")
    dataset_cfg_raw = _load_yaml(_folsom_pv_dataset_config_path(dataset_cfg_path))
    _ds_training_override = dataset_cfg_raw.get("training") or {}
    for _k, _v in _ds_training_override.items():
        if _v is not None:
            h[_k] = _v

    parser = _build_parser(h, config_default=pre_args.config)
    args = parser.parse_args()
    if args.warmup_epochs is None:
        args.warmup_epochs = math.ceil(args.epochs * 0.1)

    _train_epoch_len, _train_epoch_len_src = _resolved_train_epoch_len(
        argv=sys.argv,
        dataset_cfg_name=pre_args.dataset_config,
        dataset_cfg_raw=dataset_cfg_raw,
        cli_value=args.train_epoch_len,
    )
    _print_resolved_training_block(
        args=args,
        argv=sys.argv,
        train_conf_name=pre_args.config,
        dataset_cfg_name=pre_args.dataset_config,
        ds_training_override=_ds_training_override,
    )

    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Soft seeding: keep cudnn autotuner on (benchmark=True) and skip the deterministic
    # algo selection so we don't pay the perf hit. Multi-seed A/Bs still see real variance
    # since the python/numpy/torch RNGs above pin sample order, init, and worker draws.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    dataset_cfg = args.dataset_config
    use_satellite = _resolve_use_satellite(dataset_cfg, args.use_satellite)
    _ds_kw = dict(
        use_satellite_override=use_satellite,
        **_sky_knob_overrides(
            dataset_cfg,
            args.ray_map,
            args.sun_mask,
            args.sky_mask,
            args.sun_mask_sigma_px,
            args.sun_mask_sigma_deg,
        ),
    )
    train_dataset = FolsomIrradianceDataset(
        **_dataset_kwargs(dataset_cfg, "train", **_ds_kw)
    )
    val_dataset = FolsomIrradianceDataset(
        **_dataset_kwargs(dataset_cfg, "val", **_ds_kw)
    )
    test_dataset = FolsomIrradianceDataset(
        **_dataset_kwargs(dataset_cfg, "test", **_ds_kw)
    )
    _epoch_len_override = _resolve_train_epoch_len(dataset_cfg, args.train_epoch_len)
    if _epoch_len_override is not None:
        train_dataset._train_epoch_len = max(1, int(_epoch_len_override))
    else:
        train_dataset._train_epoch_len = _train_epoch_len
    dev_dn_list = train_dataset.devDn_list

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not hasattr(train_dataset, "sky_in_channels"):
        raise AttributeError(
            "Folsom dataset missing sky_in_channels; cannot construct vit_dinov2 sky paths"
        )
    sky_in_channels = int(train_dataset.sky_in_channels)
    sky_channels = tuple(getattr(train_dataset, "sky_channels", ()))
    use_sun_mask = "sun_mask" in sky_channels
    print(
        f"model_type=vit_dinov2 sky_in_channels={sky_in_channels} "
        f"sky_channels={sky_channels!r} use_sun_mask={use_sun_mask}"
    )
    model = pv_forecasting_model_vit_dinov2(
        dev_dn_list=dev_dn_list,
        sky_in_channels=sky_in_channels,
        use_sun_mask=use_sun_mask,
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
    criterion = nn.HuberLoss(delta=_FOLSOM_HUBER_DELTA)
    ema: ModelEMA | None = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None

    nw = int(args.num_workers)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batched,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=(nw > 0),
        worker_init_fn=_seed_worker,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batched,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=(nw > 0),
        worker_init_fn=_seed_worker,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batched,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=(nw > 0),
        worker_init_fn=_seed_worker,
    )

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else _PROJECT_ROOT / "checkpoints_folsom_dinov2"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        _gpu_id = _gpu_id_for_checkpoint()
        _ckpt_suffix = f"gpu{_gpu_id}"
    else:
        _ckpt_suffix = "cpu"
    best_ckpt_path = checkpoint_dir / f"folsom_pv_forecast_dinov2_best_{_ckpt_suffix}.pt"

    if getattr(args, "tb_log_dir", None):
        tb_log_dir = Path(args.tb_log_dir)
        if not tb_log_dir.is_absolute():
            tb_log_dir = _PROJECT_ROOT / tb_log_dir
    elif args.checkpoint_dir:
        tb_log_dir = _PROJECT_ROOT / "runs" / Path(args.checkpoint_dir).name
    else:
        tb_log_dir = _PROJECT_ROOT / "runs" / f"folsom_dinov2_{_ckpt_suffix}"
    writer = SummaryWriter(log_dir=str(tb_log_dir))

    max_batches = args.train_max_batches_per_epoch
    if max_batches is not None and max_batches < 0:
        max_batches = None

    eval_cap = args.eval_max_batches
    use_nwp = bool(args.use_nwp)
    zero_sky = bool(args.zero_sky)
    # DINOv2 predicts one step; default loss horizon is 1 (see ``_LOSS_METRIC_HORIZON``).
    # Predictions are always sliced against the first ``model T_out`` target steps.
    loss_metric_horizon = (
        int(_LOSS_METRIC_HORIZON)
        if _LOSS_METRIC_HORIZON is not None
        else 1
    )
    initial_test_loss, _, _ = evaluate(
        model,
        device,
        test_loader,
        criterion,
        loss_metric_horizon=loss_metric_horizon,
        max_batches=eval_cap,
        use_nwp=use_nwp,
        zero_sky=zero_sky,
    )
    print(f"Initial test loss: {initial_test_loss:.6f}")

    rmse_min = 1e8
    saved_best_val_ckpt = False
    warmup_epochs = max(0, int(args.warmup_epochs))
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
            max_batches,
            ema=ema if ema_active else None,
            loss_metric_horizon=loss_metric_horizon,
            use_nwp=use_nwp,
            zero_sky=zero_sky,
        )
        if ema_active:
            assert ema is not None
            with ema.apply(model):
                val_loss, val_rmse, val_mae = evaluate(
                    model,
                    device,
                    val_loader,
                    criterion,
                    loss_metric_horizon=loss_metric_horizon,
                    max_batches=eval_cap,
                    use_nwp=use_nwp,
                    zero_sky=zero_sky,
                )
        else:
            val_loss, val_rmse, val_mae = evaluate(
                model,
                device,
                val_loader,
                criterion,
                loss_metric_horizon=loss_metric_horizon,
                max_batches=eval_cap,
                use_nwp=use_nwp,
                zero_sky=zero_sky,
            )
        print(
            f"Epoch {epoch}/{args.epochs}  lr={cur_lr:.2e}  "
            f"train_loss={avg_loss:.6f}  val_loss={val_loss:.6f}  val_RMSE={val_rmse:.4f} W/m²"
        )
        writer.add_scalar("loss/train", avg_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("metric/val_rmse", val_rmse, epoch)
        writer.add_scalar("metric/val_mae", val_mae, epoch)
        writer.add_scalar("lr", cur_lr, epoch)
        scheduler.step()

        if args.save_every and epoch % args.save_every == 0:
            path = checkpoint_dir / f"folsom_pv_forecast_dinov2_epoch_{epoch}_{_ckpt_suffix}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg_loss,
                    "dev_dn_list": dev_dn_list,
                    "dataset_config": dataset_cfg,
                    "ema": ema_active,
                    "zero_sky": zero_sky,
                    "use_nwp": use_nwp,
                    "model_type": "vit_dinov2",
                },
                path,
            )
            print(f"  saved {path}")

        # Best-val selection: epochs 1..warmup_epochs are LR warmup only (1-indexed loop).
        if epoch > warmup_epochs and val_rmse < rmse_min:
            rmse_min = val_rmse
            saved_best_val_ckpt = True
            best_state = ema.state_dict() if ema_active else model.state_dict()
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg_loss,
                    "dev_dn_list": dev_dn_list,
                    "dataset_config": dataset_cfg,
                    "ema": ema_active,
                    "zero_sky": zero_sky,
                    "use_nwp": use_nwp,
                    "model_type": "vit_dinov2",
                },
                best_ckpt_path,
            )

    final_path = checkpoint_dir / f"folsom_pv_forecast_dinov2_final_{_ckpt_suffix}.pt"
    final_state = ema.state_dict() if ema is not None else model.state_dict()
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": final_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "dev_dn_list": dev_dn_list,
            "dataset_config": dataset_cfg,
            "ema": ema is not None,
            "zero_sky": zero_sky,
            "use_nwp": use_nwp,
            "model_type": "vit_dinov2",
        },
        final_path,
    )
    print(f"Saved final checkpoint to {final_path}")

    metrics_log = checkpoint_dir / f"folsom_pv_forecast_dinov2_metrics_{_ckpt_suffix}.txt"
    hparam_metrics: dict[str, float] = {}

    if final_path.is_file():
        ckpt_final = torch.load(final_path, map_location=device)
        model.load_state_dict(ckpt_final["model_state_dict"])
        test_loss_final, test_rmse_final, test_mae_final = evaluate(
            model,
            device,
            test_loader,
            criterion,
            loss_metric_horizon=loss_metric_horizon,
            max_batches=eval_cap,
            use_nwp=use_nwp,
            zero_sky=zero_sky,
        )
        print(
            f"Test set with last-epoch checkpoint ({final_path.name}, epoch={ckpt_final.get('epoch', '?')}): "
            f"loss={test_loss_final:.6f}, RMSE={test_rmse_final:.4f} W/m², MAE={test_mae_final:.4f} W/m²"
        )
        writer.add_scalar("metric/test_rmse_last_epoch", test_rmse_final, args.epochs)
        writer.add_scalar("metric/test_mae_last_epoch", test_mae_final, args.epochs)
        hparam_metrics["hparam/test_rmse_last_epoch"] = test_rmse_final
        hparam_metrics["hparam/test_mae_last_epoch"] = test_mae_final
        with open(metrics_log, "a", encoding="utf-8") as mf:
            mf.write(
                f"last_epoch\t{test_loss_final:.8f}\t{test_rmse_final:.8f}\t{test_mae_final:.8f}\n"
            )
    else:
        print(f"No {final_path.name} on disk; skip test evaluation with last-epoch checkpoint.")

    if saved_best_val_ckpt and best_ckpt_path.is_file():
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_loss_best, test_rmse_best, test_mae_best = evaluate(
            model,
            device,
            test_loader,
            criterion,
            loss_metric_horizon=loss_metric_horizon,
            max_batches=eval_cap,
            use_nwp=use_nwp,
            zero_sky=zero_sky,
        )
        print(
            f"Test set with best val-RMSE checkpoint ({best_ckpt_path.name}, epoch={ckpt.get('epoch', '?')}): "
            f"loss={test_loss_best:.6f}, RMSE={test_rmse_best:.4f} W/m², MAE={test_mae_best:.4f} W/m²"
        )
        writer.add_scalar("metric/test_rmse", test_rmse_best, args.epochs)
        writer.add_scalar("metric/test_mae", test_mae_best, args.epochs)
        writer.add_scalar("metric/test_rmse_best_val", test_rmse_best, args.epochs)
        writer.add_scalar("metric/test_mae_best_val", test_mae_best, args.epochs)
        hparam_metrics["hparam/test_rmse"] = test_rmse_best
        hparam_metrics["hparam/test_mae"] = test_mae_best
        with open(metrics_log, "a", encoding="utf-8") as mf:
            mf.write(
                f"best_val\t{test_loss_best:.8f}\t{test_rmse_best:.8f}\t{test_mae_best:.8f}\n"
            )
        print(f"Appended test metrics to {metrics_log}")
    else:
        print(
            f"No post-warmup best val-RMSE checkpoint saved"
            f" (warmup_epochs={warmup_epochs}); skip test evaluation with best checkpoint."
        )

    if hparam_metrics:
        writer.add_hparams(
            {
                "lr": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "warmup_epochs": args.warmup_epochs,
                "weight_decay": args.weight_decay,
                "seed": int(args.seed),
                "use_nwp": int(use_nwp),
                "zero_sky": int(zero_sky),
                "dataset_config": dataset_cfg,
                "eval_max_batches": -1 if eval_cap is None else int(eval_cap),
                "use_ema": int(args.use_ema),
            },
            hparam_metrics,
        )

    writer.close()


if __name__ == "__main__":
    main()
