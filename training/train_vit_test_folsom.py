"""
Canonical **Folsom** trainer for ``pv_forecasting_model_vit_imgs`` (long-lived entrypoint).

Uses ``dataloader.folsom.FolsomIrradianceDataset`` (zarr/JPEG skies, merged NWP, Luoyang-shaped
``collate_batched`` batches). If ``paths.sky_format`` is omitted in the dataset YAML, this script
injects **zarr** (see ``_folsom_pv_dataset_config_path``).

Compared to ``training/train_vit_test.py`` (Luoyang), this file adds Folsom semantics (NWP remap /
zero baseline, ``--eval_max_batches``, GHI-scale metrics, optional ``--zero-sky``) while keeping
TensorBoard logging and optional EMA (same pattern as ``train_vit_test.py``). The 4-modality
satellite branch (formerly the ``train_vit_test_folsom_2.py`` sidecar) is unified in: feeding
``sat_tensor`` to the model is toggled by ``sampling.use_satellite`` in the dataset YAML and the
``--use-satellite`` / ``--no-use-satellite`` CLI overrides (default off; CLI > YAML > False).

Local smoke (1 logical GPU, tiny run):

  python training/train_vit_test_folsom.py --epochs 1 --train_max_batches_per_epoch 3 \\
    --eval_max_batches 20 --num_workers 0 --batch_size 1

  # Manager-style PV+NWP (real NWP, sky tensors zeroed after load; dataloader still reads Zarr):
  python training/train_vit_test_folsom.py --use-nwp --zero-sky  # add your usual epoch/batch flags

  # 4-modality run (PV + sky + NWP + GOES-15 sat); overrides the dataset YAML's use_satellite key:
  python training/train_vit_test_folsom.py --use-satellite --use-nwp  # add your usual epoch/batch flags

Training hyperparameters: ``config/train/conf_train.yaml`` (``--config``). Dataset paths:
``config/datasets/conf_folsom.yaml`` (``--dataset-config``).

Sky-branch extras (default rgb-only) via CLI flags; precedence CLI > YAML > loader default::

  python training/train_vit_test_folsom.py --sun-mask --sun-mask-radius-deg 20
  python training/train_vit_test_folsom.py --ray-map
  python training/train_vit_test_folsom.py --ray-map --sun-mask --sun-mask-radius-deg 20
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import copy
import os
import random
import shutil
import sys
import tempfile
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
# When the dataset YAML omits ``paths.sky_format``, ``dataloader.folsom`` would default to jpg;
# this trainer injects ``zarr`` instead (JPEG users must set ``paths.sky_format: jpg``).
_DEFAULT_SKY_FORMAT_FOR_PV_TRAINER = "zarr"
_FOLSOM_PV_TEMP_CFG_DIRS: list[Path] = []
sys.path.insert(0, str(_PROJECT_ROOT))

from dataloader.folsom import (  # noqa: E402
    _FOLSOM_HUBER_DELTA,
    _FOLSOM_KT_INPUT_SCALE,
    _FOLSOM_NWP_FEATURE_COLS,
    FolsomIrradianceDataset,
)
from dataloader.luoyang_zarr import collate_batched  # noqa: E402
from models.models import (  # noqa: E402
    NWP_FEATURE_NORMALIZERS,
    pv_forecasting_model_vit_imgs,
)

# Column indices in ``nwp_tensor`` **before** remap: 8 features from ``_FOLSOM_NWP_FEATURE_COLS`` + 1 trailing mask.
_FOLSOM_NWP_TEMPERATURE_INDEX = _FOLSOM_NWP_FEATURE_COLS.index("temperature")
# ``pv_forecasting_model_vit_imgs`` reads ``nwp_tensor[:, :, 0]`` as shortwave-like and ``[:, :, 2]`` as Kelvin temp.
_VIT_IMGS_NWP_TEMPERATURE_SLOT = 2
# Folsom forecast horizon: first 16 output steps (~4 h at 15 min; Folsom uses 15 min output).
# Capped at 16 (~4 h) to mirror Luoyang; the dataset's ``pv_output_len`` is set to 16 so the
# model output, loss, and masked RMSE/MAE all cover the same 4 h forecast window.
_LOSS_METRIC_HORIZON = 16

# Special token in ``--nwp-features`` that toggles the per-step invalid-mask channel
# (``nwp_tensor[:, :, -1]``); not a real NWP feature so kept out of the features list.
_NWP_INVALID_MASK_TOKEN = "invalid_mask"
# Presets resolved by ``_parse_nwp_features``.
_NWP_FEATURE_PRESETS: dict[str, tuple[tuple[str, ...], bool]] = {
    "minimal": (("dwsw", "temperature"), False),
    "all": (tuple(_FOLSOM_NWP_FEATURE_COLS), True),
}


def remap_nwp_tensor_for_pv_vit_imgs(nwp_tensor: torch.Tensor) -> torch.Tensor:
    """
    Reorder Folsom merged-NWP features for ``pv_forecasting_model_vit_imgs``.

    Folsom ``_interpolate_nwp`` stacks columns in ``_FOLSOM_NWP_FEATURE_COLS`` order, then appends
    an invalid mask. The PV ViT used to assume channel 0 ≈ surface shortwave (W/m²) and channel 2
    ≈ air temperature (K), matching Luoyang's ``ssrd`` / ``t2m`` positions. Here ``dwsw`` is
    already at index 0; ``temperature`` is at index 6 and used to be copied into index 2.

    NOTE: this helper is no longer used by ``pv_forecasting_model_vit_imgs`` (which now reads
    raw ``_FOLSOM_NWP_FEATURE_COLS`` indices directly via the per-feature dispatch dict; see
    ``models.models.NWP_FEATURE_NORMALIZERS`` and the model's ``nwp_features`` constructor
    argument). Kept here as a no-touch reference for any older variant that might still want the
    Luoyang-style channel layout.
    """
    if nwp_tensor.ndim != 3:
        raise ValueError(f"nwp_tensor expected [B, T, C], got shape {tuple(nwp_tensor.shape)}")
    n_feat = len(_FOLSOM_NWP_FEATURE_COLS)
    if nwp_tensor.shape[-1] != n_feat + 1:
        raise ValueError(
            f"nwp_tensor last dim expected {n_feat + 1} (features + mask), got {nwp_tensor.shape[-1]}"
        )
    out = nwp_tensor.clone()
    out[:, :, _VIT_IMGS_NWP_TEMPERATURE_SLOT] = nwp_tensor[:, :, _FOLSOM_NWP_TEMPERATURE_INDEX]
    # Channel 0 is already ``dwsw`` (first column of ``_FOLSOM_NWP_FEATURE_COLS``).
    return out


def _parse_nwp_features(spec: str) -> tuple[list[str], bool]:
    """Resolve ``--nwp-features`` into ``(features, use_invalid_mask)``.

    Accepts a preset name (see ``_NWP_FEATURE_PRESETS``) or a comma-separated list of
    canonical feature names. The special token ``invalid_mask`` (or ``+invalid_mask``)
    toggles the per-step invalid mask channel instead of adding a feature.

    Raises ``ValueError`` with a clear message on unknown / duplicate names.
    """
    spec = (spec or "").strip()
    if not spec:
        raise ValueError("--nwp-features may not be empty")
    if spec in _NWP_FEATURE_PRESETS:
        feats, use_mask = _NWP_FEATURE_PRESETS[spec]
        return list(feats), bool(use_mask)

    features: list[str] = []
    use_invalid_mask = False
    valid = set(NWP_FEATURE_NORMALIZERS)
    seen: set[str] = set()
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        if token in (_NWP_INVALID_MASK_TOKEN, "+" + _NWP_INVALID_MASK_TOKEN):
            use_invalid_mask = True
            continue
        if token not in valid:
            raise ValueError(
                f"--nwp-features: unknown feature {token!r}. Valid features: "
                f"{sorted(valid)} (presets: {sorted(_NWP_FEATURE_PRESETS)}; "
                f"add {_NWP_INVALID_MASK_TOKEN!r} to include the per-step invalid mask)."
            )
        if token in seen:
            raise ValueError(f"--nwp-features: duplicate feature {token!r}")
        seen.add(token)
        features.append(token)

    if not features:
        raise ValueError(
            "--nwp-features must select at least one feature "
            f"(got spec {spec!r}; valid features: {sorted(valid)})"
        )
    return features, use_invalid_mask


def _format_nwp_features_for_log(features: list[str], use_invalid_mask: bool) -> str:
    """Compact, deterministic string for logs / TensorBoard hparams."""
    parts = list(features)
    if use_invalid_mask:
        parts.append(_NWP_INVALID_MASK_TOKEN)
    return ",".join(parts) if parts else "<none>"


# Default ``nwp_features`` / ``nwp_use_invalid_mask`` for checkpoints that pre-date the
# per-feature selector (i.e. saved with the hardcoded ``(ssrd, t2m)`` query). Matches
# ``models.models._DEFAULT_VIT_IMGS_NWP_FEATURES`` and the ``minimal`` preset.
_LEGACY_CKPT_NWP_FEATURES: tuple[str, ...] = ("dwsw", "temperature")
_LEGACY_CKPT_NWP_USE_INVALID_MASK: bool = False


def resolve_nwp_features_from_ckpt(ckpt: dict) -> tuple[list[str], bool]:
    """Read ``nwp_features`` / ``nwp_use_invalid_mask`` from a loaded checkpoint dict.

    Falls back to the pre-refactor defaults (``["dwsw", "temperature"]`` / ``False``) when
    the keys are absent so older checkpoints can be reloaded by the same code path.
    Also validates that any feature name listed in the checkpoint is currently known to
    the dispatch dict; otherwise the model factory would explode further downstream.
    """
    raw_feats = ckpt.get("nwp_features")
    if raw_feats is None:
        features = list(_LEGACY_CKPT_NWP_FEATURES)
    else:
        features = [str(n) for n in raw_feats]
        unknown = [n for n in features if n not in NWP_FEATURE_NORMALIZERS]
        if unknown:
            raise ValueError(
                f"checkpoint nwp_features contains unknown name(s) {unknown}; "
                f"valid features are {sorted(NWP_FEATURE_NORMALIZERS)}"
            )

    raw_mask = ckpt.get("nwp_use_invalid_mask")
    if raw_mask is None:
        use_invalid_mask = _LEGACY_CKPT_NWP_USE_INVALID_MASK
    else:
        use_invalid_mask = bool(raw_mask)

    return features, use_invalid_mask


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


def _prepare_nwp_for_vit(d: dict, *, use_nwp: bool) -> dict:
    """
    Prepare ``d['nwp_tensor']`` for the ViT, in place.

    Two parallel paths share the same downstream call signature so the model code is unchanged:
      * ``use_nwp=True``  -> pass the raw merged-NWP tensor through unchanged (post-refactor:
        ``pv_forecasting_model_vit_imgs`` reads ``_FOLSOM_NWP_FEATURE_COLS`` indices itself via
        the per-feature dispatch dict; channel remap is no longer needed).
      * ``use_nwp=False`` -> overwrite with ``zeros_like`` (blacked-out / NWP-ablation baseline).

    The model always indexes ``nwp_tensor`` columns, so we keep the original
    shape/dtype/device and only swap the values; ``None`` would crash the forward pass.
    """
    nwp = d.get("nwp_tensor")
    if nwp is None:
        return d
    if not use_nwp:
        d["nwp_tensor"] = torch.zeros_like(nwp)
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
    use_nwp: bool = False,
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
        pv_pred = kt_pred * d["target_p_cs"] * d["p_mean"].unsqueeze(1)
        t_out = int(pv_pred.shape[1])
        assert d["target_pv"].shape[1] == t_out, (pv_pred.shape, d["target_pv"].shape)
        h = min(_LOSS_METRIC_HORIZON, t_out)
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
    print()
    return total_loss / max(n, 1)


def evaluate(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    criterion: nn.Module,
    *,
    max_batches: int | None = None,
    use_nwp: bool = False,
    zero_sky: bool = False,
) -> tuple[float, float, float]:
    """Returns mean Huber loss (first ``_LOSS_METRIC_HORIZON`` steps, masked like train), RMSE and
    MAE in **normalized** GHI space over the same slice (``target_mask``; predictions at night
    cos-zenith < 0 are zeroed before residuals, matching ``train_vit_test.py``).

    If ``max_batches`` is set, only the first N batches are used (smoke / faster dev runs; metrics
    are not a full pass over the split).
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    sum_abs = 0.0
    sum_sq = 0.0
    n_elem = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            d = _batch_to_device(batch, device)
            _prepare_nwp_for_vit(d, use_nwp=use_nwp)
            _prepare_sky_for_vit(d, zero_sky=zero_sky)
            kt_pred = forward_vit(model, d) * _FOLSOM_KT_INPUT_SCALE
            pv_pred = kt_pred * d["target_p_cs"] * d["p_mean"].unsqueeze(1)
            t_out = int(pv_pred.shape[1])
            h = min(_LOSS_METRIC_HORIZON, t_out)
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
        f"First-{_LOSS_METRIC_HORIZON}-step metrics (masked GHI; pred zeroed at night): "
        f"MAE={mae_wm2:.4f} W/m²  RMSE={rmse_wm2:.4f} W/m²  "
        f"(~4 h horizon at 15 min)"
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


def _cleanup_folsom_pv_temp_cfg_dirs() -> None:
    for d in _FOLSOM_PV_TEMP_CFG_DIRS:
        shutil.rmtree(d, ignore_errors=True)


atexit.register(_cleanup_folsom_pv_temp_cfg_dirs)


def _folsom_pv_dataset_config_path(base: Path) -> Path:
    """
    YAML path for ``FolsomIrradianceDataset``: ``base`` as-is, or a temp copy with
    ``paths.sky_format`` set to ``_DEFAULT_SKY_FORMAT_FOR_PV_TRAINER`` when missing/blank.
    """
    cfg = _load_yaml(base)
    raw = (cfg.get("paths") or {}).get("sky_format")
    if raw is not None and str(raw).strip() != "":
        return base
    cfg2 = copy.deepcopy(cfg)
    cfg2.setdefault("paths", {})["sky_format"] = _DEFAULT_SKY_FORMAT_FOR_PV_TRAINER
    tmp = Path(tempfile.mkdtemp(prefix="folsom_pv_ds_cfg_"))
    _FOLSOM_PV_TEMP_CFG_DIRS.append(tmp)
    out = tmp / "dataset.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg2, f, sort_keys=False, allow_unicode=True)
    return out.resolve()


def _resolve_data_dir(paths_cfg: dict, cfg_path: Path) -> Path:
    raw = paths_cfg.get("data_dir")
    if raw is None or str(raw).strip() == "":
        raise KeyError(f"dataset config paths.data_dir is required (in {cfg_path})")
    p = Path(str(raw))
    return p.resolve() if p.is_absolute() else (_PROJECT_ROOT / p).resolve()


def _build_parser(h: dict, config_default: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train pv_forecasting_model_vit_imgs on Folsom (GHI as PV target)"
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
        action="store_true",
        help=(
            "Feed the real Folsom merged-NWP tensor to the ViT. Default is OFF: the NWP "
            "tensor is replaced with zeros (blacked-out baseline). The per-feature selection "
            "is controlled by --nwp-features below; --use-nwp toggles whether the data is real "
            "or zeroed-out at input."
        ),
    )
    parser.add_argument(
        "--nwp-features",
        type=str,
        default="minimal",
        help=(
            "Which NWP feature channels to feed into the forecast-query MLP of "
            "pv_forecasting_model_vit_imgs (also fixes the model's query_mlp input dim). "
            "Comma-separated list from "
            f"{sorted(NWP_FEATURE_NORMALIZERS)} (canonical order: "
            f"{list(_FOLSOM_NWP_FEATURE_COLS)}); add the special token "
            f"'{_NWP_INVALID_MASK_TOKEN}' (or '+{_NWP_INVALID_MASK_TOKEN}') to also append the "
            "per-step invalid mask channel. Presets: "
            f"{sorted(_NWP_FEATURE_PRESETS)}. Default 'minimal' = "
            f"{list(_NWP_FEATURE_PRESETS['minimal'][0])} (mask off). When --use-nwp is off, this "
            "flag still picks the architecture but the input data is zeroed (existing "
            "_prepare_nwp_for_vit behaviour)."
        ),
    )
    parser.add_argument(
        "--zero-sky",
        action="store_true",
        help=(
            "After each batch is on device, replace sky image tensors (and sky time features) with "
            "zeros so the ViT uses the empty-sky branch while the dataset still loads real Zarr/JPEG. "
            "Use with --use-nwp for PV+NWP vs PV+NWP+sky comparisons."
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
        action="store_true",
        help=(
            "Add the fixed fisheye ray_map sky channels (3ch). When this flag or "
            "--sun-mask is set, overrides sampling.sky_channels in the dataset YAML "
            "(rgb is always included)."
        ),
    )
    parser.add_argument(
        "--sun-mask",
        dest="sun_mask",
        action="store_true",
        help=(
            "Add the per-frame sun_mask sky channel (1ch). When this flag or "
            "--ray-map is set, overrides sampling.sky_channels in the dataset YAML."
        ),
    )
    parser.set_defaults(ray_map=None, sun_mask=None)
    parser.add_argument(
        "--sun-mask-radius-deg",
        type=float,
        default=None,
        metavar="DEG",
        help=(
            "Angular radius of the sun_mask disc in degrees. Precedence: this flag > "
            "sampling.sun_mask_radius_deg in the dataset YAML > dataloader default."
        ),
    )
    parser.add_argument(
        "--sky-disc-mask",
        type=str,
        default=None,
        choices=[
            "none",
            "valid_disc",
            "tight_disc",
            "sun_halo",
            "sun_only",
            "manual_loose",
            "manual_tight",
        ],
        metavar="MODE",
        help=(
            "Sky-disc gating mode: keep pixels inside a Euclidean disc, zero RGB outside. "
            "Precedence: this flag > sampling.sky_disc_mask_mode in the dataset YAML > "
            "'none' (no gating). Does not affect ray_map or sun_mask channels."
        ),
    )
    parser.add_argument(
        "--sky-disc-mask-radius-px",
        type=float,
        default=None,
        metavar="PX",
        help=(
            "Override the disc radius in pixels at 224×224 for the active --sky-disc-mask "
            "mode. Precedence: this flag > sampling.sky_disc_mask_radius_px in YAML > "
            "mode-specific default."
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
    sun_mask_radius_deg_override: float | None = None,
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
        use_satellite=use_satellite,
        sky_channels=(
            list(sky_channels_override)
            if sky_channels_override is not None
            else sampling_cfg.get("sky_channels")
        ),
        sun_mask_radius_deg=(
            sun_mask_radius_deg_override
            if sun_mask_radius_deg_override is not None
            else sampling_cfg.get("sun_mask_radius_deg")
        ),
        sky_disc_mask_mode=(
            sky_disc_mask_mode_override
            if sky_disc_mask_mode_override is not None
            else sampling_cfg.get("sky_disc_mask_mode")
        ),
        sky_disc_mask_radius_px=(
            sky_disc_mask_radius_px_override
            if sky_disc_mask_radius_px_override is not None
            else sampling_cfg.get("sky_disc_mask_radius_px")
        ),
    )


def _resolve_sky_channels(
    dataset_config_name: str,
    cli_ray_map: bool | None,
    cli_sun_mask: bool | None,
) -> tuple[str, ...] | None:
    """Resolve sky channel list: CLI flags > YAML ``sampling.sky_channels`` > loader default.

    Returns ``None`` when no CLI override was requested (delegate to YAML / default).
    When either ``--ray-map`` or ``--sun-mask`` is passed, builds ``rgb`` + optional extras
    in canonical order (rgb, ray_map, sun_mask).
    """
    if cli_ray_map is None and cli_sun_mask is None:
        return None
    channels = ["rgb"]
    if cli_ray_map:
        channels.append("ray_map")
    if cli_sun_mask:
        channels.append("sun_mask")
    return tuple(channels)


def _resolve_sun_mask_radius_deg(
    dataset_config_name: str,
    cli_value: float | None,
) -> float | None:
    """Resolve ``sun_mask_radius_deg``: CLI flag > YAML > ``None`` (loader default)."""
    if cli_value is not None:
        return float(cli_value)
    base_cfg_path = _resolve_named_config(_DATASETS_CONFIG_DIR, dataset_config_name, "dataset-config")
    cfg_path = _folsom_pv_dataset_config_path(base_cfg_path)
    cfg = _load_yaml(cfg_path)
    yaml_value = (cfg.get("sampling", {}) or {}).get("sun_mask_radius_deg")
    if yaml_value is None:
        return None
    return float(yaml_value)


def _resolve_sky_disc_mask_mode(
    dataset_config_name: str,
    cli_value: str | None,
) -> str | None:
    """Resolve ``sky_disc_mask_mode``: CLI flag > YAML > ``None`` (loader default ``none``)."""
    if cli_value is not None:
        return str(cli_value)
    base_cfg_path = _resolve_named_config(_DATASETS_CONFIG_DIR, dataset_config_name, "dataset-config")
    cfg_path = _folsom_pv_dataset_config_path(base_cfg_path)
    cfg = _load_yaml(cfg_path)
    yaml_value = (cfg.get("sampling", {}) or {}).get("sky_disc_mask_mode")
    if yaml_value is None:
        return None
    return str(yaml_value)


def _resolve_sky_disc_mask_radius_px(
    dataset_config_name: str,
    cli_value: float | None,
) -> float | None:
    """Resolve ``sky_disc_mask_radius_px``: CLI flag > YAML > ``None`` (mode default)."""
    if cli_value is not None:
        return float(cli_value)
    base_cfg_path = _resolve_named_config(_DATASETS_CONFIG_DIR, dataset_config_name, "dataset-config")
    cfg_path = _folsom_pv_dataset_config_path(base_cfg_path)
    cfg = _load_yaml(cfg_path)
    yaml_value = (cfg.get("sampling", {}) or {}).get("sky_disc_mask_radius_px")
    if yaml_value is None:
        return None
    return float(yaml_value)


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


def main() -> None:
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
    _use_sat_src = (
        "CLI flag" if args.use_satellite is not None else f"YAML ({dataset_cfg})"
    )
    print(f"use_satellite: {use_satellite} (source: {_use_sat_src})")
    sky_channels_override = _resolve_sky_channels(dataset_cfg, args.ray_map, args.sun_mask)
    sun_mask_radius_deg_override = _resolve_sun_mask_radius_deg(
        dataset_cfg, args.sun_mask_radius_deg
    )
    sky_disc_mask_mode_override = _resolve_sky_disc_mask_mode(
        dataset_cfg, args.sky_disc_mask
    )
    sky_disc_mask_radius_px_override = _resolve_sky_disc_mask_radius_px(
        dataset_cfg, args.sky_disc_mask_radius_px
    )
    _ds_kw = dict(
        use_satellite_override=use_satellite,
        sky_channels_override=sky_channels_override,
        sun_mask_radius_deg_override=sun_mask_radius_deg_override,
        sky_disc_mask_mode_override=sky_disc_mask_mode_override,
        sky_disc_mask_radius_px_override=sky_disc_mask_radius_px_override,
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
    print(
        f"train_epoch_len: {train_dataset._train_epoch_len:,} "
        f"(valid train anchors: {len(train_dataset._train_anchor_valid_positions):,})"
    )

    dev_dn_list = train_dataset.devDn_list

    nwp_features, nwp_use_invalid_mask = _parse_nwp_features(args.nwp_features)
    nwp_features_str = _format_nwp_features_for_log(nwp_features, nwp_use_invalid_mask)
    print(
        f"NWP features (resolved from --nwp-features={args.nwp_features!r}): "
        f"{nwp_features_str}  (use_invalid_mask={nwp_use_invalid_mask})"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ``sky_in_channels`` is the dataset-side source of truth for sky-branch input
    # width (rgb=3, +ray_map=+3, +sun_mask=+1). Configured via ``--ray-map`` /
    # ``--sun-mask`` CLI flags or ``sampling.sky_channels`` in the dataset YAML.
    sky_in_channels = int(getattr(train_dataset, "sky_in_channels", 3))
    sky_channels_resolved = tuple(getattr(train_dataset, "sky_channels", ("rgb",)))
    if args.ray_map is not None or args.sun_mask is not None:
        _sky_src = "CLI flags"
    else:
        _sky_src = f"YAML ({dataset_cfg})"
    print(
        f"Sky channels (source: {_sky_src}): {list(sky_channels_resolved)} "
        f"-> sky_in_channels={sky_in_channels}"
    )
    if "sun_mask" in sky_channels_resolved:
        if args.sun_mask_radius_deg is not None:
            _radius_src = "CLI flag"
        else:
            _yaml_radius = (dataset_cfg_raw.get("sampling") or {}).get("sun_mask_radius_deg")
            _radius_src = (
                f"YAML ({dataset_cfg})" if _yaml_radius is not None else "dataloader default"
            )
        print(
            f"sun_mask_radius_deg: {train_dataset.sun_mask_radius_deg} "
            f"(source: {_radius_src})"
        )
    if args.sky_disc_mask is not None:
        _disc_mask_src = "CLI flag"
    else:
        _yaml_disc_mask = (dataset_cfg_raw.get("sampling") or {}).get("sky_disc_mask_mode")
        _disc_mask_src = (
            f"YAML ({dataset_cfg})" if _yaml_disc_mask is not None else "dataloader default"
        )
    print(
        f"sky_disc_mask_mode: {train_dataset.sky_disc_mask_mode} "
        f"(source: {_disc_mask_src})"
    )
    if train_dataset.sky_disc_mask_mode != "none":
        if args.sky_disc_mask_radius_px is not None:
            _disc_radius_src = "CLI flag"
        else:
            _yaml_disc_radius = (dataset_cfg_raw.get("sampling") or {}).get(
                "sky_disc_mask_radius_px"
            )
            _disc_radius_src = (
                f"YAML ({dataset_cfg})"
                if _yaml_disc_radius is not None
                else "mode default at 224²"
            )
        _radius_disp = (
            train_dataset.sky_disc_mask_radius_px
            if train_dataset.sky_disc_mask_radius_px is not None
            else "mode default"
        )
        print(
            f"sky_disc_mask_radius_px: {_radius_disp} (source: {_disc_radius_src})"
        )
    model = pv_forecasting_model_vit_imgs(
        dev_dn_list=dev_dn_list,
        nwp_features=nwp_features,
        use_invalid_mask=nwp_use_invalid_mask,
        sky_in_channels=sky_in_channels,
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
    # delta is sized to Folsom's W/m^2 residual scale (Luoyang uses delta=1 kW
    # ~= 3% of 33 kW peak; Folsom analog is 3% of 1000 W/m^2 peak ~= 33 W/m^2,
    # rounded to 30). Keeps the Huber MSE region active for "good" predictions
    # and the MAE region for outliers, matching Luoyang's effective behavior.
    criterion = nn.HuberLoss(delta=_FOLSOM_HUBER_DELTA)
    ema: ModelEMA | None = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None
    print(
        f"EMA: {'enabled' if args.use_ema else 'disabled'}"
        + (f" (decay={args.ema_decay}, warmup={args.ema_warmup_epochs} epoch)" if args.use_ema else "")
    )
    print(f"Seed: {seed} (soft cudnn: benchmark=True, deterministic=False)")

    nw = int(args.num_workers)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batched,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=nw > 0,
        worker_init_fn=_seed_worker,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batched,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=nw > 0,
        worker_init_fn=_seed_worker,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batched,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=nw > 0,
        worker_init_fn=_seed_worker,
    )

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else _PROJECT_ROOT / "checkpoints_folsom_pv"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        _gpu_id = _gpu_id_for_checkpoint()
        _ckpt_suffix = f"gpu{_gpu_id}"
    else:
        _ckpt_suffix = "cpu"
    best_ckpt_path = checkpoint_dir / f"folsom_pv_forecast_vit_best_{_ckpt_suffix}.pt"

    if getattr(args, "tb_log_dir", None):
        tb_log_dir = Path(args.tb_log_dir)
        if not tb_log_dir.is_absolute():
            tb_log_dir = _PROJECT_ROOT / tb_log_dir
    elif args.checkpoint_dir:
        tb_log_dir = _PROJECT_ROOT / "runs" / Path(args.checkpoint_dir).name
    else:
        tb_log_dir = _PROJECT_ROOT / "runs" / f"folsom_pv_{_ckpt_suffix}"
    writer = SummaryWriter(log_dir=str(tb_log_dir))
    print(f"TensorBoard log dir: {tb_log_dir}")
    # Persist the resolved feature selection in TB so the run is self-describing in the UI.
    writer.add_text(
        "nwp/features",
        f"--nwp-features={args.nwp_features!r} -> resolved={nwp_features_str} "
        f"(use_invalid_mask={nwp_use_invalid_mask})",
    )

    max_batches = args.train_max_batches_per_epoch
    if max_batches is not None and max_batches < 0:
        max_batches = None

    eval_cap = args.eval_max_batches
    use_nwp = bool(args.use_nwp)
    zero_sky = bool(args.zero_sky)
    print(f"NWP input: {'REAL (raw _FOLSOM_NWP_FEATURE_COLS)' if use_nwp else 'ZEROED-OUT (baseline)'}")
    print(f"NWP features (model arch): {nwp_features_str}  use_invalid_mask={nwp_use_invalid_mask}")
    print(f"Sky images: {'ZEROED (--zero-sky; PV+NWP-style ablation)' if zero_sky else 'REAL from dataset'}")
    initial_test_loss, _, _ = evaluate(
        model,
        device,
        test_loader,
        criterion,
        max_batches=eval_cap,
        use_nwp=use_nwp,
        zero_sky=zero_sky,
    )
    print(f"Initial test loss: {initial_test_loss:.6f}")

    rmse_min = 1e8
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
            path = checkpoint_dir / f"folsom_pv_forecast_vit_epoch_{epoch}_{_ckpt_suffix}.pt"
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
                    "nwp_features": list(nwp_features),
                    "nwp_use_invalid_mask": bool(nwp_use_invalid_mask),
                },
                path,
            )
            print(f"  saved {path}")

        if val_rmse < rmse_min:
            rmse_min = val_rmse
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
                    "nwp_features": list(nwp_features),
                    "nwp_use_invalid_mask": bool(nwp_use_invalid_mask),
                },
                best_ckpt_path,
            )

    final_path = checkpoint_dir / f"folsom_pv_forecast_vit_final_{_ckpt_suffix}.pt"
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
            "nwp_features": list(nwp_features),
            "nwp_use_invalid_mask": bool(nwp_use_invalid_mask),
        },
        final_path,
    )
    print(f"Saved final checkpoint to {final_path}")

    if best_ckpt_path.is_file():
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_loss_best, test_rmse_best, test_mae_best = evaluate(
            model,
            device,
            test_loader,
            criterion,
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
                "nwp_features": nwp_features_str,
                "nwp_use_invalid_mask": int(nwp_use_invalid_mask),
                "dataset_config": dataset_cfg,
                "eval_max_batches": -1 if eval_cap is None else int(eval_cap),
                "use_ema": int(args.use_ema),
            },
            {
                "hparam/test_rmse": test_rmse_best,
                "hparam/test_mae": test_mae_best,
            },
        )
        metrics_log = checkpoint_dir / f"folsom_pv_forecast_metrics_{_ckpt_suffix}.txt"
        with open(metrics_log, "a", encoding="utf-8") as mf:
            mf.write(
                f"{test_loss_best:.8f}\t{test_rmse_best:.8f}\t{test_mae_best:.8f}\n"
            )
        print(f"Appended best-test metrics to {metrics_log}")
    else:
        print(f"No {best_ckpt_path.name} on disk; skip test evaluation with best checkpoint.")

    writer.close()


if __name__ == "__main__":
    main()
