"""
Mixed Luoyang + Folsom trainer for ``pv_forecasting_model_vit_imgs``.

Design goals:
- Reuse the training structure of ``train_vit_test_folsom.py``.
- Train from the mixed dataloader ``dataloader/mixed_luoyang_folsom.py``.
- Balance Luoyang power and Folsom GHI via source-wise robust scaling (p95) and
  equalized per-source loss aggregation.
- Keep NWP optional; in mixed mode Folsom NWP is forced off (None/zero path),
  Luoyang format is the reference.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import yaml
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"
_TRAIN_CONFIG_DIR = _CONFIG_DIR / "train"
_DATASETS_CONFIG_DIR = _CONFIG_DIR / "datasets"
_DEFAULT_TRAIN_CONF_NAME = "conf_train_mixed.yaml"
_DEFAULT_LUOYANG_DATASET_CONFIG = "conf_luoyang_shm.yaml"
_DEFAULT_FOLSOM_DATASET_CONFIG = "conf_folsom.yaml"
_DEFAULT_COMMON_OUTPUT_LEN = 16
_P95_SAMPLES_CAP = 8192
_P95_EPS = 1e-6
_NWP_FOLSOM_MASK_VALUE = 1.0

sys.path.insert(0, str(_PROJECT_ROOT))

from config_utils import get_resolved_paths  # noqa: E402
from dataloader.folsom import (  # noqa: E402
    _FOLSOM_HUBER_DELTA,
    _FOLSOM_KT_INPUT_SCALE,
    FolsomIrradianceDataset,
)
import dataloader.luoyang_zarr as luoyang_zarr_module  # noqa: E402
from dataloader.luoyang_zarr import PVDataset, collate_batched  # noqa: E402
from dataloader.mixed_luoyang_folsom import (  # noqa: E402
    MixedLuoyangFolsomDataset,
    collate_mixed_luoyang_folsom,
)
from models.models import pv_forecasting_model_vit_imgs  # noqa: E402


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
    p = argparse.ArgumentParser(
        description="Train pv_forecasting_model_vit_imgs with mixed Luoyang + Folsom datasets"
    )
    p.add_argument("--config", type=str, default=config_default)
    p.add_argument("--luoyang-dataset-config", type=str, default=_DEFAULT_LUOYANG_DATASET_CONFIG)
    p.add_argument("--folsom-dataset-config", type=str, default=_DEFAULT_FOLSOM_DATASET_CONFIG)

    p.add_argument("--epochs", type=int, default=int(h["epochs"]))
    p.add_argument("--lr", type=float, default=float(h["lr"]))
    p.add_argument("--weight-decay", type=float, default=float(h.get("weight_decay", 0.01)))
    p.add_argument("--warmup-epochs", type=int, default=int(h.get("warmup_epochs", 5)))
    p.add_argument("--lr-min", type=float, default=float(h.get("lr_min", 1e-6)))
    p.add_argument("--batch_size", type=int, default=int(h["batch_size"]))
    p.add_argument("--num_workers", type=int, default=int(h["num_workers"]))
    p.add_argument("--seed", type=int, default=int(h.get("seed", 0)))

    p.add_argument("--use-ema", dest="use_ema", action="store_true")
    p.add_argument("--no-ema", dest="use_ema", action="store_false")
    p.set_defaults(use_ema=bool(h.get("use_ema", False)))
    p.add_argument("--ema-decay", type=float, default=float(h.get("ema_decay", 0.99)))
    p.add_argument("--ema-warmup-epochs", type=int, default=int(h.get("ema_warmup_epochs", 5)))

    p.add_argument("--save_every", type=int, default=int(h.get("save_every", 5)))
    p.add_argument("--checkpoint_dir", type=str, default=h.get("checkpoint_dir"))
    p.add_argument("--tb-log-dir", type=str, default=h.get("tb_log_dir"))
    p.add_argument(
        "--train_max_batches_per_epoch",
        type=int,
        default=h.get("train_max_batches_per_epoch"),
    )
    p.add_argument("--eval_max_batches", type=int, default=h.get("eval_max_batches"))

    p.add_argument("--luoyang-prob", type=float, default=float(h.get("luoyang_prob", 0.5)))
    p.add_argument("--folsom-prob", type=float, default=float(h.get("folsom_prob", 0.5)))
    p.add_argument("--mixed-train-epoch-len", type=int, default=int(h.get("mixed_train_epoch_len", 100000)))
    p.add_argument("--common-output-len", type=int, default=int(h.get("common_output_len", _DEFAULT_COMMON_OUTPUT_LEN)))
    p.add_argument("--p95-samples-cap", type=int, default=int(h.get("p95_samples_cap", _P95_SAMPLES_CAP)))
    p.add_argument("--luoyang-max-inverters", type=int, default=h.get("luoyang_max_inverters"))

    p.add_argument("--use-nwp", action="store_true", default=bool(h.get("use_nwp", False)))
    p.add_argument("--zero-sky", action="store_true", default=bool(h.get("zero_sky", False)))
    return p


def _dataset_kwargs_luoyang(dataset_config_name: str, split: str, *, common_output_len: int) -> dict:
    cfg_path = _resolve_named_config(_DATASETS_CONFIG_DIR, dataset_config_name, "luoyang-dataset-config")
    cfg = _load_yaml(cfg_path)
    paths_cfg = cfg.get("paths", {}) or {}
    sampling_cfg = copy.deepcopy(cfg.get("sampling", {}) or {})
    if not sampling_cfg:
        raise KeyError(f"dataset config {cfg_path} is missing a non-empty 'sampling:' section")
    data_dir = _resolve_data_dir(paths_cfg, cfg_path)

    sampling_cfg["pv_output_len"] = int(common_output_len)

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
        raise ValueError(f"sampling.satimg_npy_shape_hwc must be [H, W, C] (in {cfg_path})")
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


def _dataset_kwargs_folsom(dataset_config_name: str, split: str, *, common_output_len: int) -> dict:
    cfg_path = _resolve_named_config(_DATASETS_CONFIG_DIR, dataset_config_name, "folsom-dataset-config")
    cfg = _load_yaml(cfg_path)
    paths_cfg = cfg.get("paths", {}) or {}
    sampling_cfg = copy.deepcopy(cfg.get("sampling", {}) or {})
    if not sampling_cfg:
        raise KeyError(f"dataset config {cfg_path} is missing a non-empty 'sampling:' section")
    data_dir = _resolve_data_dir(paths_cfg, cfg_path)

    sampling_cfg["pv_output_len"] = int(common_output_len)

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
        raise ValueError(f"sampling.satimg_npy_shape_hwc must be [H, W, C] (in {cfg_path})")
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
        use_satellite=bool(sampling_cfg.get("use_satellite", False)),
    )


class _FolsomNoNwpDataset(Dataset):
    """Adapter: force ``nwp_tensor=None`` for every Folsom sample."""

    def __init__(self, inner: Dataset):
        self.inner = inner

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = dict(self.inner[idx])
        s["nwp_tensor"] = None
        return s


@contextlib.contextmanager
def _temporary_luoyang_csv_limit(max_inverters: int | None):
    """Temporarily cap luoyang CSV enumeration before PVDataset __init__ loads files."""
    if max_inverters is None:
        yield
        return
    n_limit = int(max_inverters)
    if n_limit < 1:
        print(f"[mixed] luoyang_max_inverters={n_limit} < 1, disable limit")
        yield
        return

    original = luoyang_zarr_module.list_csv_files

    def _limited_list_csv_files(*args, **kwargs):
        out = original(*args, **kwargs)
        if len(out) > n_limit:
            return out[:n_limit]
        return out

    luoyang_zarr_module.list_csv_files = _limited_list_csv_files
    try:
        yield
    finally:
        luoyang_zarr_module.list_csv_files = original


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {
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
        "source_dataset": list(batch.get("source_dataset", [])),
    }
    for key in ("sat_tensor", "sat_timefeats", "skimg_tensor", "skimg_timefeats", "nwp_tensor"):
        v = batch.get(key)
        out[key] = None if v is None else v.to(device)
    return out


def _mask_for_source(source_list: list[str], source_name: str, device: torch.device) -> torch.Tensor:
    return torch.tensor([s == source_name for s in source_list], dtype=torch.bool, device=device)


def _prepare_nwp_for_mixed(
    d: dict[str, Any],
    *,
    use_nwp: bool,
) -> None:
    """NWP optional path with Luoyang-reference format.

    - ``use_nwp=False``: force zeros_like.
    - ``use_nwp=True``:
      - keep Luoyang NWP rows as-is;
      - force Folsom rows to "none-like" zero + invalid-mask=1.
    """
    nwp = d.get("nwp_tensor")
    if nwp is None:
        return
    if not use_nwp:
        d["nwp_tensor"] = torch.zeros_like(nwp)
        return
    source_list = d.get("source_dataset") or []
    if not source_list:
        return
    out = nwp.clone()
    mask_f = _mask_for_source(source_list, "folsom", out.device)
    if bool(mask_f.any()):
        out[mask_f] = 0.0
        out[mask_f, :, -1] = _NWP_FOLSOM_MASK_VALUE
    d["nwp_tensor"] = out


def _prepare_sky_for_vit(d: dict[str, Any], *, zero_sky: bool) -> None:
    if not zero_sky:
        return
    for key in ("skimg_tensor", "skimg_timefeats"):
        t = d.get(key)
        if t is not None:
            d[key] = torch.zeros_like(t)


def forward_vit(model: nn.Module, d: dict[str, Any]) -> torch.Tensor:
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


def _estimate_source_p95(
    ds: Dataset,
    source_name: str,
    *,
    samples_cap: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(int(seed))
    vals: list[np.ndarray] = []

    # Fast path: read directly from dataset in-memory tables, avoid __getitem__ heavy decode.
    inner = getattr(ds, "inner", ds)
    source_l = str(source_name).strip().lower()
    if source_l == "folsom":
        df = getattr(inner, "_df", None)
        if df is not None:
            ghi_col = None
            cols = list(df.columns)
            for cand in ("ghi", "GHI", "target_ghi"):
                if cand in cols:
                    ghi_col = cand
                    break
            if ghi_col is not None:
                arr = pd.to_numeric(df[ghi_col], errors="coerce").to_numpy(dtype=np.float32)
                arr = np.abs(arr[np.isfinite(arr)])
                if arr.size:
                    n_take = min(max(1, int(samples_cap)), int(arr.size))
                    if n_take < arr.size:
                        arr = arr[rng.choice(arr.size, size=n_take, replace=False)]
                    vals.append(arr)
    elif source_l == "luoyang":
        cache = getattr(inner, "_csv_cache", None)
        if isinstance(cache, dict) and cache:
            all_vals: list[np.ndarray] = []
            for df in cache.values():
                if "active_power" not in df.columns:
                    continue
                arr = pd.to_numeric(df["active_power"], errors="coerce").to_numpy(dtype=np.float32)
                arr = np.abs(arr[np.isfinite(arr)])
                if arr.size:
                    all_vals.append(arr)
            if all_vals:
                cat = np.concatenate(all_vals)
                n_take = min(max(1, int(samples_cap)), int(cat.size))
                if n_take < cat.size:
                    cat = cat[rng.choice(cat.size, size=n_take, replace=False)]
                vals.append(cat)

    # Fallback path (kept for unknown dataset wrappers).
    if not vals:
        n = int(len(ds))
        if n < 1:
            return 1.0
        n_take = min(max(1, int(samples_cap)), n)
        idxs = rng.choice(n, size=n_take, replace=False) if n_take < n else np.arange(n)
        for idx in idxs:
            s = ds[int(idx)]
            tgt = s.get("target_pv")
            m = s.get("target_mask")
            if not torch.is_tensor(tgt):
                continue
            t = tgt.detach().cpu().numpy().astype(np.float32, copy=False)
            if torch.is_tensor(m):
                mm = m.detach().cpu().numpy().astype(np.float32, copy=False) > 0.5
                t = t[mm]
            if t.size:
                vals.append(np.abs(t.reshape(-1)))
    if not vals:
        print(f"[mixed] WARNING: unable to estimate p95 for source={source_name}; fallback 1.0")
        return 1.0
    cat = np.concatenate(vals)
    p95 = float(np.percentile(cat, 95.0))
    if not np.isfinite(p95) or p95 <= _P95_EPS:
        print(f"[mixed] WARNING: invalid p95 for source={source_name} ({p95}); fallback 1.0")
        return 1.0
    return p95


def _balanced_source_loss(
    *,
    pv_pred: torch.Tensor,
    target_pv: torch.Tensor,
    target_mask: torch.Tensor,
    source_list: list[str],
    criterion: nn.Module,
    scale_l: float,
    scale_f: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = pv_pred.device
    losses: list[torch.Tensor] = []
    stats: dict[str, float] = {}
    for src, scale in (("luoyang", scale_l), ("folsom", scale_f)):
        msrc = _mask_for_source(source_list, src, device)
        n_src = int(msrc.sum().item())
        stats[f"n_{src}"] = float(n_src)
        if n_src == 0:
            continue
        p = pv_pred[msrc] / scale
        t = target_pv[msrc] / scale
        m = target_mask[msrc]
        l_src = criterion(p * m, t * m)
        losses.append(l_src)
        stats[f"loss_{src}"] = float(l_src.detach().item())
    if not losses:
        loss = criterion(pv_pred * target_mask, target_pv * target_mask)
        stats["loss_fallback"] = float(loss.detach().item())
        return loss, stats
    loss = torch.stack(losses).mean()
    stats["loss_balanced"] = float(loss.detach().item())
    return loss, stats


def _masked_rmse_mae(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    diff = pred - tgt
    denom = float(mask.sum().item())
    if denom <= 0:
        return float("nan"), float("nan")
    mae = float((diff.abs() * mask).sum().item() / denom)
    rmse = float((((diff ** 2) * mask).sum().item() / denom) ** 0.5)
    return rmse, mae


def train_one_epoch(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    scale_l: float,
    scale_f: float,
    use_nwp: bool,
    zero_sky: bool,
    max_batches: int | None = None,
    ema: ModelEMA | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    print("number of batches: ", len(loader))
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        d = _batch_to_device(batch, device)
        _prepare_nwp_for_mixed(d, use_nwp=use_nwp)
        _prepare_sky_for_vit(d, zero_sky=zero_sky)
        optimizer.zero_grad()
        kt_pred = forward_vit(model, d) * _FOLSOM_KT_INPUT_SCALE
        pv_pred = kt_pred * d["target_p_cs"] * d["p_mean"].unsqueeze(1)
        t_out = int(pv_pred.shape[1])
        h = min(t_out, int(d["target_pv"].shape[1]))
        pred_h = pv_pred[:, :h]
        tgt_h = d["target_pv"][:, :h]
        m_h = d["target_mask"][:, :h]
        loss, _ = _balanced_source_loss(
            pv_pred=pred_h,
            target_pv=tgt_h,
            target_mask=m_h,
            source_list=d["source_dataset"],
            criterion=criterion,
            scale_l=scale_l,
            scale_f=scale_f,
        )
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model)
        total_loss += float(loss.item())
        n += 1
    print()
    return total_loss / max(n, 1)


def evaluate_single_source(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    criterion: nn.Module,
    *,
    source_name: str,
    source_scale: float,
    use_nwp: bool,
    zero_sky: bool,
    max_batches: int | None = None,
) -> tuple[float, float, float]:
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
            _prepare_nwp_for_mixed(d, use_nwp=use_nwp)
            _prepare_sky_for_vit(d, zero_sky=zero_sky)
            kt_pred = forward_vit(model, d) * _FOLSOM_KT_INPUT_SCALE
            pv_pred = kt_pred * d["target_p_cs"] * d["p_mean"].unsqueeze(1)
            h = min(int(pv_pred.shape[1]), int(d["target_pv"].shape[1]))
            pred_h = pv_pred[:, :h]
            tgt_h = d["target_pv"][:, :h]
            m_h = d["target_mask"][:, :h]
            loss = criterion((pred_h / source_scale) * m_h, (tgt_h / source_scale) * m_h)
            total_loss += float(loss.item())
            n_batches += 1
            diff = pred_h - tgt_h
            sum_abs += float((diff.abs() * m_h).sum().item())
            sum_sq += float((((diff ** 2) * m_h).sum().item()))
            n_elem += float(m_h.sum().item())
    mean_loss = total_loss / max(n_batches, 1)
    mae = sum_abs / max(n_elem, 1.0)
    rmse = (sum_sq / max(n_elem, 1.0)) ** 0.5
    print(
        f"[{source_name}] eval: loss_scaled={mean_loss:.6f}, "
        f"RMSE_raw={rmse:.4f}, MAE_raw={mae:.4f}"
    )
    return mean_loss, rmse, mae


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


def _seed_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=_DEFAULT_TRAIN_CONF_NAME)
    pre_args, _ = pre_parser.parse_known_args()

    train_conf_path = _resolve_named_config(_TRAIN_CONFIG_DIR, pre_args.config, "config")
    train_conf = _load_yaml(train_conf_path)
    h = dict(train_conf.get("training") or {})
    if not h:
        raise KeyError(f"training config {train_conf_path} is missing a 'training:' section")
    parser = _build_parser(h, config_default=pre_args.config)
    args = parser.parse_args()

    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    print("[mixed] NWP mismatch note:")
    print(
        "[mixed] - Luoyang NWP is historically [ssrd, msl, t2m, u10, v10, u100, v100, mask]."
    )
    print(
        "[mixed] - Folsom/model feature semantics are canonical names "
        "[dwsw, cloud_cover, precipitation, pressure, wind-u, wind-v, temperature, rel_humidity, mask]."
    )
    print(
        "[mixed] - Mixed trainer policy: Luoyang format as reference; Folsom NWP forced OFF (nwp_tensor=None)."
    )

    common_output_len = int(args.common_output_len)
    with _temporary_luoyang_csv_limit(args.luoyang_max_inverters):
        luo_train = PVDataset(**_dataset_kwargs_luoyang(args.luoyang_dataset_config, "train", common_output_len=common_output_len))
        luo_val = PVDataset(**_dataset_kwargs_luoyang(args.luoyang_dataset_config, "val", common_output_len=common_output_len))
        luo_test = PVDataset(**_dataset_kwargs_luoyang(args.luoyang_dataset_config, "test", common_output_len=common_output_len))
    print(
        f"[mixed] Luoyang loaded inverter CSVs: train={len(getattr(luo_train, 'sample_files', []))}, "
        f"val={len(getattr(luo_val, 'sample_files', []))}, "
        f"test={len(getattr(luo_test, 'sample_files', []))}"
    )

    fol_train_inner = FolsomIrradianceDataset(**_dataset_kwargs_folsom(args.folsom_dataset_config, "train", common_output_len=common_output_len))
    fol_val_inner = FolsomIrradianceDataset(**_dataset_kwargs_folsom(args.folsom_dataset_config, "val", common_output_len=common_output_len))
    fol_test_inner = FolsomIrradianceDataset(**_dataset_kwargs_folsom(args.folsom_dataset_config, "test", common_output_len=common_output_len))
    fol_train = _FolsomNoNwpDataset(fol_train_inner)
    fol_val = _FolsomNoNwpDataset(fol_val_inner)
    fol_test = _FolsomNoNwpDataset(fol_test_inner)

    luo_prob = float(args.luoyang_prob)
    fol_prob = float(args.folsom_prob)
    mixed_train = MixedLuoyangFolsomDataset(
        luoyang_dataset=luo_train,
        folsom_dataset=fol_train,
        probs=(luo_prob, fol_prob),
        epoch_len=int(args.mixed_train_epoch_len),
        sample_with_replacement=True,
        deterministic_by_index=False,
        seed=seed,
    )
    mixed_val = MixedLuoyangFolsomDataset(
        luoyang_dataset=luo_val,
        folsom_dataset=fol_val,
        probs=(luo_prob, fol_prob),
        epoch_len=min(len(luo_val), len(fol_val)),
        sample_with_replacement=False,
        deterministic_by_index=True,
        seed=seed + 1,
    )

    p95_l = _estimate_source_p95(luo_train, "luoyang", samples_cap=int(args.p95_samples_cap), seed=seed + 11)
    p95_f = _estimate_source_p95(fol_train, "folsom", samples_cap=int(args.p95_samples_cap), seed=seed + 17)
    print(f"[mixed] robust target scales (p95): luoyang={p95_l:.6f}, folsom={p95_f:.6f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = pv_forecasting_model_vit_imgs(
        dev_dn_list=getattr(luo_train, "devDn_list", [0]),
        nwp_features=["dwsw", "temperature"],
        use_invalid_mask=False,
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
    criterion = nn.HuberLoss(delta=_FOLSOM_HUBER_DELTA)
    ema: ModelEMA | None = ModelEMA(model, decay=float(args.ema_decay)) if bool(args.use_ema) else None
    print(
        f"EMA: {'enabled' if args.use_ema else 'disabled'}"
        + (f" (decay={args.ema_decay}, warmup={args.ema_warmup_epochs} epoch)" if args.use_ema else "")
    )
    print(f"Seed: {seed} (soft cudnn: benchmark=True, deterministic=False)")

    nw = int(args.num_workers)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        mixed_train,
        batch_size=int(args.batch_size),
        shuffle=True,
        collate_fn=collate_mixed_luoyang_folsom,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=nw > 0,
        worker_init_fn=_seed_worker,
    )
    val_loader = DataLoader(
        mixed_val,
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=collate_mixed_luoyang_folsom,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=nw > 0,
        worker_init_fn=_seed_worker,
    )
    luo_test_loader = DataLoader(
        luo_test,
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=collate_batched,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=nw > 0,
        worker_init_fn=_seed_worker,
    )
    fol_test_loader = DataLoader(
        fol_test,
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=collate_batched,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=nw > 0,
        worker_init_fn=_seed_worker,
    )

    checkpoint_dir = (
        Path(args.checkpoint_dir) if args.checkpoint_dir else _PROJECT_ROOT / "checkpoints_mixed"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        ckpt_suffix = f"gpu{_gpu_id_for_checkpoint()}"
    else:
        ckpt_suffix = "cpu"
    best_ckpt = checkpoint_dir / f"mixed_pv_forecast_vit_best_{ckpt_suffix}.pt"
    final_ckpt = checkpoint_dir / f"mixed_pv_forecast_vit_final_{ckpt_suffix}.pt"

    if args.tb_log_dir:
        tb_log_dir = Path(args.tb_log_dir)
        if not tb_log_dir.is_absolute():
            tb_log_dir = _PROJECT_ROOT / tb_log_dir
    else:
        tb_log_dir = _PROJECT_ROOT / "runs" / f"mixed_pv_{ckpt_suffix}"
    writer = SummaryWriter(log_dir=str(tb_log_dir))
    print(f"TensorBoard log dir: {tb_log_dir}")
    writer.add_text("mixed/source_probs", f"luoyang={luo_prob}, folsom={fol_prob}")
    writer.add_text("mixed/scales_p95", f"luoyang={p95_l:.6f}, folsom={p95_f:.6f}")

    use_nwp = bool(args.use_nwp)
    zero_sky = bool(args.zero_sky)
    print(f"[mixed] NWP input: {'ENABLED (Luoyang ref; Folsom forced off)' if use_nwp else 'DISABLED (zeroed)'}")
    print(f"[mixed] Sky images: {'ZEROED (--zero-sky)' if zero_sky else 'REAL from dataset'}")

    eval_cap = args.eval_max_batches
    val_loss0, val_rmse0, val_mae0 = evaluate_single_source(
        model,
        device,
        val_loader,
        criterion,
        source_name="mixed_val",
        source_scale=1.0,
        use_nwp=use_nwp,
        zero_sky=zero_sky,
        max_batches=eval_cap,
    )
    print(
        f"Initial mixed-val (raw metrics over mixed stream): "
        f"loss={val_loss0:.6f}, RMSE={val_rmse0:.4f}, MAE={val_mae0:.4f}"
    )

    best_val = float("inf")
    max_batches = args.train_max_batches_per_epoch
    for epoch in range(1, int(args.epochs) + 1):
        cur_lr = optimizer.param_groups[0]["lr"]
        ema_active = ema is not None and epoch > int(args.ema_warmup_epochs)
        if ema is not None and not ema_active:
            for k, v in model.state_dict().items():
                if k in ema.shadow:
                    ema.shadow[k].copy_(v.detach().float())

        train_loss = train_one_epoch(
            model,
            device,
            train_loader,
            criterion,
            optimizer,
            scale_l=p95_l,
            scale_f=p95_f,
            use_nwp=use_nwp,
            zero_sky=zero_sky,
            max_batches=max_batches,
            ema=ema if ema_active else None,
        )

        if ema_active:
            assert ema is not None
            with ema.apply(model):
                val_loss, val_rmse, val_mae = evaluate_single_source(
                    model,
                    device,
                    val_loader,
                    criterion,
                    source_name="mixed_val",
                    source_scale=1.0,
                    use_nwp=use_nwp,
                    zero_sky=zero_sky,
                    max_batches=eval_cap,
                )
        else:
            val_loss, val_rmse, val_mae = evaluate_single_source(
                model,
                device,
                val_loader,
                criterion,
                source_name="mixed_val",
                source_scale=1.0,
                use_nwp=use_nwp,
                zero_sky=zero_sky,
                max_batches=eval_cap,
            )

        print(
            f"Epoch {epoch}/{args.epochs}  lr={cur_lr:.2e}  "
            f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  val_rmse={val_rmse:.4f}"
        )
        writer.add_scalar("loss/train_balanced", train_loss, epoch)
        writer.add_scalar("loss/val_mixed", val_loss, epoch)
        writer.add_scalar("metric/val_mixed_rmse", val_rmse, epoch)
        writer.add_scalar("metric/val_mixed_mae", val_mae, epoch)
        writer.add_scalar("lr", cur_lr, epoch)
        scheduler.step()

        if args.save_every and epoch % int(args.save_every) == 0:
            path = checkpoint_dir / f"mixed_pv_forecast_vit_epoch_{epoch}_{ckpt_suffix}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": train_loss,
                    "ema": ema_active,
                    "use_nwp": use_nwp,
                    "zero_sky": zero_sky,
                    "luoyang_dataset_config": args.luoyang_dataset_config,
                    "folsom_dataset_config": args.folsom_dataset_config,
                    "luoyang_prob": luo_prob,
                    "folsom_prob": fol_prob,
                    "scale_p95_luoyang": p95_l,
                    "scale_p95_folsom": p95_f,
                    "common_output_len": common_output_len,
                },
                path,
            )
            print(f"  saved {path}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = ema.state_dict() if ema_active else model.state_dict()
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": train_loss,
                    "val_loss": val_loss,
                    "ema": ema_active,
                    "use_nwp": use_nwp,
                    "zero_sky": zero_sky,
                    "luoyang_dataset_config": args.luoyang_dataset_config,
                    "folsom_dataset_config": args.folsom_dataset_config,
                    "luoyang_prob": luo_prob,
                    "folsom_prob": fol_prob,
                    "scale_p95_luoyang": p95_l,
                    "scale_p95_folsom": p95_f,
                    "common_output_len": common_output_len,
                },
                best_ckpt,
            )

    final_state = ema.state_dict() if ema is not None else model.state_dict()
    torch.save(
        {
            "epoch": int(args.epochs),
            "model_state_dict": final_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "ema": ema is not None,
            "use_nwp": use_nwp,
            "zero_sky": zero_sky,
            "luoyang_dataset_config": args.luoyang_dataset_config,
            "folsom_dataset_config": args.folsom_dataset_config,
            "luoyang_prob": luo_prob,
            "folsom_prob": fol_prob,
            "scale_p95_luoyang": p95_l,
            "scale_p95_folsom": p95_f,
            "common_output_len": common_output_len,
        },
        final_ckpt,
    )
    print(f"Saved final checkpoint to {final_ckpt}")

    if best_ckpt.is_file():
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        luo_loss, luo_rmse, luo_mae = evaluate_single_source(
            model,
            device,
            luo_test_loader,
            criterion,
            source_name="luoyang_test",
            source_scale=p95_l,
            use_nwp=use_nwp,
            zero_sky=zero_sky,
            max_batches=eval_cap,
        )
        fol_loss, fol_rmse, fol_mae = evaluate_single_source(
            model,
            device,
            fol_test_loader,
            criterion,
            source_name="folsom_test",
            source_scale=p95_f,
            use_nwp=use_nwp,
            zero_sky=zero_sky,
            max_batches=eval_cap,
        )
        print(
            f"Best checkpoint test metrics ({best_ckpt.name}, epoch={ckpt.get('epoch', '?')}):\n"
            f"  Luoyang  loss={luo_loss:.6f}  RMSE={luo_rmse:.4f}  MAE={luo_mae:.4f}\n"
            f"  Folsom   loss={fol_loss:.6f}  RMSE={fol_rmse:.4f}  MAE={fol_mae:.4f}"
        )
        writer.add_scalar("metric/test_luoyang_rmse", luo_rmse, int(args.epochs))
        writer.add_scalar("metric/test_luoyang_mae", luo_mae, int(args.epochs))
        writer.add_scalar("metric/test_folsom_rmse", fol_rmse, int(args.epochs))
        writer.add_scalar("metric/test_folsom_mae", fol_mae, int(args.epochs))
    else:
        print(f"No {best_ckpt.name} on disk; skip best-checkpoint test evaluation.")

    writer.close()


if __name__ == "__main__":
    main()
