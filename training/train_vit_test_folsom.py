"""
Train ``pv_forecasting_model_vit_imgs`` on Folsom (irradiance CSV + sky Zarr or JPEG + merged NWP).

If the dataset YAML omits ``paths.sky_format``, this entrypoint injects **``zarr``** (see
``_folsom_pv_dataset_config_path``). For JPEG skies set ``paths.sky_format: jpg`` explicitly.

Mirrors ``training/train_vit_test.py`` (loop, scheduler, checkpoints) but:
  - ``FolsomIrradianceDataset`` + ``config/datasets/conf_folsom.yaml`` (Luoyang-shaped batches).
  - NWP **defaults to a blacked-out (all-zero) tensor** so the ViT trains as an NWP-free baseline.
    Pass ``--use-nwp`` to feed the real Folsom merged NWP; in that case the tensor is remapped so
    ``pv_forecasting_model_vit_imgs`` slot 0 = ``dwsw``, slot 2 = ``temperature`` (K); see
    ``remap_nwp_tensor_for_pv_vit_imgs`` (must match ``dataloader.folsom2._FOLSOM_NWP_FEATURE_COLS``).
  - Train/eval Huber loss on the **first 16 forecast steps** only (same window as ``train_vit_test.py``).
    ``evaluate`` MAE/RMSE use the same slice, ``target_mask`` weighting, and a **night mask** on
    predictions where ``forecast_timefeats[:, :16, 3]`` (cos zenith) is negative (pred set to 0).
  - Prints an ACC line with capacity **1100** (GHI W/m² scale matching normalization), not Luoyang kW.

Run (example, from repo root, with conda env that has PyTorch):

  python training/train_vit_test_folsom.py --epochs 1 --train_max_batches_per_epoch 3 --eval_max_batches 20 --num_workers 0 --batch_size 1

Training hyperparameters default from ``config/train/conf_train.yaml`` (``--config conf_train.yaml``).
Dataset paths from ``config/datasets/conf_folsom.yaml`` (override with ``--dataset-config other.yaml``).
"""

from __future__ import annotations

import argparse
import atexit
import copy
import os
import shutil
import sys
import tempfile
from pathlib import Path

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
_DEFAULT_FOLSOM_DATASET_CONFIG = "conf_folsom.yaml"
# When the dataset YAML omits ``paths.sky_format``, ``dataloader.folsom2`` would default to jpg;
# this trainer injects ``zarr`` instead (JPEG users must set ``paths.sky_format: jpg``).
_DEFAULT_SKY_FORMAT_FOR_PV_TRAINER = "zarr"
_FOLSOM_PV_TEMP_CFG_DIRS: list[Path] = []
sys.path.insert(0, str(_PROJECT_ROOT))

from dataloader.folsom2 import FolsomIrradianceDataset, _FOLSOM_NWP_FEATURE_COLS  # noqa: E402
from dataloader.luoyang_zarr import collate_batched  # noqa: E402
from models.models import pv_forecasting_model_vit_imgs  # noqa: E402

# Column indices in ``nwp_tensor`` **before** remap: 8 features from ``_FOLSOM_NWP_FEATURE_COLS`` + 1 trailing mask.
_FOLSOM_NWP_TEMPERATURE_INDEX = _FOLSOM_NWP_FEATURE_COLS.index("temperature")
# ``pv_forecasting_model_vit_imgs`` reads ``nwp_tensor[:, :, 0]`` as shortwave-like and ``[:, :, 2]`` as Kelvin temp.
_VIT_IMGS_NWP_TEMPERATURE_SLOT = 2
# Match ``training/train_vit_test.py``: first 16 output steps (~4 h at 15 min; Folsom uses 15 min output).
_LOSS_METRIC_HORIZON = 16


def remap_nwp_tensor_for_pv_vit_imgs(nwp_tensor: torch.Tensor) -> torch.Tensor:
    """
    Reorder Folsom merged-NWP features for ``pv_forecasting_model_vit_imgs``.

    Folsom ``_interpolate_nwp`` stacks columns in ``_FOLSOM_NWP_FEATURE_COLS`` order, then appends
    an invalid mask. The PV ViT assumes channel 0 ≈ surface shortwave (W/m²) and channel 2 ≈ air
    temperature (K), matching Luoyang's ``ssrd`` / ``t2m`` positions. Here ``dwsw`` is already at
    index 0; ``temperature`` is at index 6 and must be copied into index 2.
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
        "target_pv": batch["target_pv"].to(device),
        "target_mask": batch["target_mask"].to(device),
    }
    for key in ("sat_tensor", "sat_timefeats", "skimg_tensor", "skimg_timefeats", "nwp_tensor"):
        v = batch.get(key)
        out[key] = None if v is None else v.to(device)
    return out


def _prepare_nwp_for_vit(d: dict, *, use_nwp: bool) -> dict:
    """
    Prepare ``d['nwp_tensor']`` for the ViT, in place.

    Two parallel paths share the same downstream call signature so the model code is unchanged:
      * ``use_nwp=True``  -> remap channels (real NWP path; see ``remap_nwp_tensor_for_pv_vit_imgs``).
      * ``use_nwp=False`` -> overwrite with ``zeros_like`` (blacked-out / NWP-ablation baseline).

    The model indexes ``nwp_tensor[:, :, 0|1|2|-1]`` unconditionally, so we keep the original
    shape/dtype/device and only swap the values; ``None`` would crash the forward pass.
    """
    nwp = d.get("nwp_tensor")
    if nwp is None:
        return d
    if use_nwp:
        d["nwp_tensor"] = remap_nwp_tensor_for_pv_vit_imgs(nwp)
    else:
        d["nwp_tensor"] = torch.zeros_like(nwp)
    return d


def forward_vit(model: nn.Module, d: dict) -> torch.Tensor:
    return model(
        d["device_id"],
        d["pv"],
        pv_mask=d["pv_mask"],
        pv_timefeats=d["pv_timefeats"],
        forecast_timefeats=d["forecast_timefeats"],
        sat_tensor=d["sat_tensor"],
        sat_timefeats=d["sat_timefeats"],
        skimg_tensor=d["skimg_tensor"],
        skimg_timefeats=d["skimg_timefeats"],
        nwp_tensor=d["nwp_tensor"],
    )


def train_one_epoch(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    max_batches: int | None = None,
    *,
    use_nwp: bool = False,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        d = _batch_to_device(batch, device)
        _prepare_nwp_for_vit(d, use_nwp=use_nwp)
        B = d["device_id"].size(0)
        optimizer.zero_grad()
        pv_pred = forward_vit(model, d)
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
        total_loss += loss.item()
        n += B
    return total_loss / max(n, 1)


def evaluate(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    criterion: nn.Module,
    *,
    max_batches: int | None = None,
    use_nwp: bool = False,
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
            pv_pred = forward_vit(model, d)
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
    mae_norm = sum_abs / max(n_elem, 1.0)
    rmse_norm = (sum_sq / max(n_elem, 1.0)) ** 0.5
    # Dataset stores GHI / 1100; convert error to ~W/m² for readability.
    ghi_scale = 1100.0
    mae_wm2 = mae_norm * ghi_scale
    rmse_wm2 = rmse_norm * ghi_scale
    capacity = ghi_scale
    print(
        f"First-{_LOSS_METRIC_HORIZON}-step metrics (masked GHI; pred zeroed at night): "
        f"MAE(norm)={mae_norm:.6f}  RMSE(norm)={rmse_norm:.6f}  "
        f"MAE≈{mae_wm2:.2f} W/m²  RMSE≈{rmse_wm2:.2f} W/m²"
    )
    print(
        f"RMSE/MAE on first {_LOSS_METRIC_HORIZON} forecast steps (~4 h at 15 min). "
        f"Capacity: {capacity:.0f} (W/m² GHI scale)"
    )
    print(
        f"MAE: {mae_wm2:.6f}, RMSE: {rmse_wm2:.6f}, "
        f"ACC(MAE): {1.0 - mae_wm2 / capacity:.6f}, ACC(RMSE): {1.0 - rmse_wm2 / capacity:.6f}"
    )
    return mean_loss, rmse_norm, mae_norm


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
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=int(h["batch_size"]))
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
        "--use-nwp",
        action="store_true",
        help=(
            "Feed the real Folsom merged-NWP tensor to the ViT (channels remapped). "
            "Default is OFF: the NWP tensor is replaced with zeros (blacked-out baseline)."
        ),
    )
    return parser


def _dataset_kwargs(dataset_config_name: str, split: str) -> dict:
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

    dataset_cfg = args.dataset_config
    train_dataset = FolsomIrradianceDataset(**_dataset_kwargs(dataset_cfg, "train"))
    val_dataset = FolsomIrradianceDataset(**_dataset_kwargs(dataset_cfg, "val"))
    test_dataset = FolsomIrradianceDataset(**_dataset_kwargs(dataset_cfg, "test"))

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
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batched,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=nw > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batched,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=nw > 0,
    )

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else _PROJECT_ROOT / "checkpoints_folsom_pv"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        _gpu_id = _gpu_id_for_checkpoint()
        _ckpt_suffix = f"gpu{_gpu_id}"
    else:
        _ckpt_suffix = "cpu"
    best_ckpt_path = checkpoint_dir / f"folsom_pv_forecast_vit_best_{_ckpt_suffix}.pt"

    max_batches = args.train_max_batches_per_epoch
    if max_batches is not None and max_batches < 0:
        max_batches = None

    eval_cap = args.eval_max_batches
    use_nwp = bool(args.use_nwp)
    print(f"NWP input: {'REAL (remapped)' if use_nwp else 'ZEROED-OUT (baseline)'}")
    initial_test_loss, _, _ = evaluate(
        model, device, test_loader, criterion, max_batches=eval_cap, use_nwp=use_nwp
    )
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
            max_batches=max_batches,
            use_nwp=use_nwp,
        )
        val_loss, val_rmse, _ = evaluate(
            model, device, val_loader, criterion, max_batches=eval_cap, use_nwp=use_nwp
        )
        print(
            f"Epoch {epoch}/{args.epochs}  lr={cur_lr:.2e}  "
            f"train_loss={avg_loss:.6f}  val_loss={val_loss:.6f}  val_RMSE(norm)={val_rmse:.6f}"
        )
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
                    "dataset_config": dataset_cfg,
                },
                best_ckpt_path,
            )

    final_path = checkpoint_dir / f"folsom_pv_forecast_vit_final_{_ckpt_suffix}.pt"
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "dev_dn_list": dev_dn_list,
            "dataset_config": dataset_cfg,
        },
        final_path,
    )
    print(f"Saved final checkpoint to {final_path}")

    if best_ckpt_path.is_file():
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_loss_best, test_rmse_best, test_mae_best = evaluate(
            model, device, test_loader, criterion, max_batches=eval_cap, use_nwp=use_nwp
        )
        print(
            f"Test set with best val-RMSE checkpoint ({best_ckpt_path.name}, epoch={ckpt.get('epoch', '?')}): "
            f"loss={test_loss_best:.6f}, RMSE(norm)={test_rmse_best:.6f}, MAE(norm)={test_mae_best:.6f}"
        )
        metrics_log = checkpoint_dir / f"folsom_pv_forecast_metrics_{_ckpt_suffix}.txt"
        with open(metrics_log, "a", encoding="utf-8") as mf:
            mf.write(
                f"{test_loss_best:.8f}\t{test_rmse_best:.8f}\t{test_mae_best:.8f}\n"
            )
        print(f"Appended best-test metrics to {metrics_log}")
    else:
        print(f"No {best_ckpt_path.name} on disk; skip test evaluation with best checkpoint.")


if __name__ == "__main__":
    main()
