"""
Training script for pv_forecasting_model_vit (sat + sky TimeSformer + PV TCN).
Uses the same Luoyang CSV / sky / sat DataLoader as train.py.

Folsom irradiance trainer: ``training/train_vit_folsom.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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
from dataloader.luoyang import PVDataset, collate_batched, load_csv, INVERTER_STATE_COL, VALID_STATE, what_to_fetch, data_fetched
from models.models import pv_forecasting_model_vit_nwp
from training.training_conf import (
    get_training_hparams_from_conf,
    get_training_paths_from_conf,
    load_config,
)

# Debug `what_to_fetch`/`data_fetched` by printing one sample and optionally exiting before training.
# If random selection is off, set CSV path plus the exact `collectTime` for the window's last PV input row.
_RUN_DEBUG_FETCH_INSPECT = False
_DEBUG_FETCH_SPLIT = "train"  # "train" or "test"
_DEBUG_FETCH_USE_RANDOM_FILE_AND_ROW = False
_DEBUG_FETCH_CSV_PATH: Path | str = "/home/kyber/projects/digital energy/dataset/processed/luoayng_data_626/NE_333621155.csv"
_DEBUG_FETCH_LAST_INPUT_COLLECT_TIME: str = "2025-01-04 22:00:00"
# After fetch-debug (when not exiting early): pull one train batch and assert layout + optional forward smoke test.
_RUN_DEBUG_BATCH_INSPECT = True


def batch_evaluation(
    batch: dict[str, Any],
    *,
    num_devices: int,
    pv_input_len: int,
    pv_output_len: int,
    satimg_window_size: int,
    skyimg_window_size: int,
    skyimg_spatial_size: int,
    satimg_npy_shape_hwc: tuple[int, int, int],
    model: nn.Module,
    device: torch.device,
) -> None:
    """Run a descriptive, multi-check batch validation and raise on the first failure."""
    title = "batch inspect"
    print(f"{title}: start train batch validation")
    passed = 0
    total = 8

    def _run_check(name: str, why: str, fn) -> None:
        nonlocal passed
        try:
            detail = fn()
            print(f"{title}: {name}: PASS - {why}. {detail}")
            passed += 1
        except Exception as e:
            print(f"{title}: {name}: FAIL - {why}. {e}")
            print(f"{title}: summary FAIL ({passed}/{total} checks passed)")
            raise

    expected_keys = (
        "dev_idx",
        "pv",
        "pv_mask",
        "pv_timefeats",
        "forecast_timefeats",
        "sat_tensor",
        "sat_timefeats",
        "skimg_tensor",
        "skimg_timefeats",
        "nwp_tensor",
        "target_pv",
        "target_mask",
    )
    h_w_c = satimg_npy_shape_hwc
    exp_c, exp_h, exp_w = h_w_c[2], h_w_c[0], h_w_c[1]

    def _check_keys() -> str:
        missing = [k for k in expected_keys if k not in batch]
        if missing:
            raise AssertionError(f"missing required keys: {missing}")
        extra = sorted(set(batch.keys()) - set(expected_keys))
        return f"found all {len(expected_keys)} required keys; extra keys={extra if extra else 'none'}"

    _run_check(
        "1/8 Keys",
        "model and trainer need a stable batch contract",
        _check_keys,
    )

    dev_idx = batch["dev_idx"]
    B = int(dev_idx.shape[0])

    def _check_batch_and_device_ids() -> str:
        if dev_idx.dtype != torch.long:
            raise AssertionError(f"dev_idx dtype should be torch.int64, got {dev_idx.dtype}")
        if B < 1:
            raise AssertionError(f"batch size must be >=1, got {B}")
        if (dev_idx < 0).any() or (dev_idx >= num_devices).any():
            raise AssertionError(
                f"dev_idx out of range [0, {num_devices}); min={int(dev_idx.min())} max={int(dev_idx.max())}"
            )
        return (
            f"B={B}, dev_idx dtype={dev_idx.dtype}, "
            f"dev_idx range={int(dev_idx.min())}..{int(dev_idx.max())} in [0,{num_devices})"
        )

    _run_check(
        "2/8 Batch size and dev ids",
        "all modalities must align on the same sample axis and device ids must be valid",
        _check_batch_and_device_ids,
    )

    pv, pm = batch["pv"], batch["pv_mask"]
    pvf, ftf = batch["pv_timefeats"], batch["forecast_timefeats"]
    tpv, tm = batch["target_pv"], batch["target_mask"]

    def _check_pv_target_time_shapes() -> str:
        if pv.shape[0] != B or pm.shape[0] != B:
            raise AssertionError(f"pv/pv_mask leading dim mismatch for B={B}: pv={tuple(pv.shape)} pm={tuple(pm.shape)}")
        if pv.shape != pm.shape:
            raise AssertionError(f"pv and pv_mask must match exactly: pv={tuple(pv.shape)} pm={tuple(pm.shape)}")
        if int(pv.shape[-1]) != pv_input_len:
            raise AssertionError(f"pv T_in mismatch: expected {pv_input_len}, got {tuple(pv.shape)}")
        if tpv.shape[0] != B or tm.shape[0] != B:
            raise AssertionError(f"target leading dim mismatch for B={B}: target={tuple(tpv.shape)} mask={tuple(tm.shape)}")
        if tpv.shape != tm.shape:
            raise AssertionError(f"target_pv and target_mask must match exactly: {tuple(tpv.shape)} vs {tuple(tm.shape)}")
        if int(tpv.shape[-1]) != pv_output_len:
            raise AssertionError(f"target T_out mismatch: expected {pv_output_len}, got {tuple(tpv.shape)}")
        return f"pv={tuple(pv.shape)}, pv_mask={tuple(pm.shape)}, target_pv={tuple(tpv.shape)}, target_mask={tuple(tm.shape)}"

    _run_check(
        "3/8 PV and target shapes",
        "input and output horizons must match configuration",
        _check_pv_target_time_shapes,
    )

    def _check_time_feature_alignment() -> str:
        if pvf.shape[0] != B or ftf.shape[0] != B:
            raise AssertionError(f"timefeats leading dim mismatch for B={B}: pvf={tuple(pvf.shape)} ftf={tuple(ftf.shape)}")
        if int(pvf.shape[1]) != pv_input_len:
            raise AssertionError(f"pv_timefeats T_in mismatch: expected {pv_input_len}, got {tuple(pvf.shape)}")
        if int(ftf.shape[1]) != pv_output_len:
            raise AssertionError(f"forecast_timefeats T_out mismatch: expected {pv_output_len}, got {tuple(ftf.shape)}")
        if int(pvf.shape[2]) != int(ftf.shape[2]):
            raise AssertionError(f"timefeat channel mismatch: pv={pvf.shape[2]} forecast={ftf.shape[2]}")
        return f"pv_timefeats={tuple(pvf.shape)}, forecast_timefeats={tuple(ftf.shape)}, C_tf={int(pvf.shape[2])}"

    _run_check(
        "4/8 Time features",
        "history and forecast temporal features must align with their horizons",
        _check_time_feature_alignment,
    )

    st, stf = batch["sat_tensor"], batch["sat_timefeats"]
    sk, skf = batch["skimg_tensor"], batch["skimg_timefeats"]
    nwp = batch["nwp_tensor"]

    def _check_modal_shapes() -> str:
        if st.shape[0] != B or stf.shape[0] != B:
            raise AssertionError(f"sat leading dim mismatch for B={B}: sat={tuple(st.shape)} sat_tf={tuple(stf.shape)}")
        if int(st.shape[1]) != satimg_window_size or int(stf.shape[1]) != satimg_window_size:
            raise AssertionError(
                f"sat window mismatch: expected T={satimg_window_size}, got sat={tuple(st.shape)} sat_tf={tuple(stf.shape)}"
            )
        if tuple(st.shape[2:5]) != (exp_c, exp_h, exp_w):
            raise AssertionError(
                f"sat CHW mismatch: expected {(exp_c, exp_h, exp_w)}, got {tuple(st.shape[2:5])}"
            )
        if sk is None or skf is None:
            raise AssertionError("skimg tensors are None but model path expects sky inputs")
        if sk.shape[0] != B or skf.shape[0] != B:
            raise AssertionError(f"sky leading dim mismatch for B={B}: sky={tuple(sk.shape)} sky_tf={tuple(skf.shape)}")
        if int(sk.shape[1]) != skyimg_window_size or int(skf.shape[1]) != skyimg_window_size:
            raise AssertionError(
                f"sky window mismatch: expected T={skyimg_window_size}, got sky={tuple(sk.shape)} sky_tf={tuple(skf.shape)}"
            )
        if int(sk.shape[3]) != skyimg_spatial_size or int(sk.shape[4]) != skyimg_spatial_size:
            raise AssertionError(
                f"sky H/W mismatch: expected {skyimg_spatial_size}, got sky={tuple(sk.shape)}"
            )
        if nwp is None:
            raise AssertionError("nwp_tensor is None but model forward uses NWP columns")
        if nwp.shape[0] != B or int(nwp.shape[1]) != pv_output_len or int(nwp.shape[2]) < 2:
            raise AssertionError(
                f"nwp shape mismatch: expected (B={B}, T_out={pv_output_len}, C>=2), got {tuple(nwp.shape)}"
            )
        return f"sat={tuple(st.shape)}, sky={tuple(sk.shape)}, nwp={tuple(nwp.shape)}"

    _run_check(
        "5/8 Modal tensor shapes",
        "satellite, sky, and NWP branches must be geometrically consistent",
        _check_modal_shapes,
    )

    float_tensors = (
        ("pv", pv),
        ("pv_mask", pm),
        ("pv_timefeats", pvf),
        ("forecast_timefeats", ftf),
        ("sat_tensor", st),
        ("sat_timefeats", stf),
        ("skimg_tensor", sk),
        ("skimg_timefeats", skf),
        ("nwp_tensor", nwp),
        ("target_pv", tpv),
        ("target_mask", tm),
    )

    def _check_dtypes() -> str:
        parts: list[str] = []
        for name, t in float_tensors:
            if not t.is_floating_point():
                raise AssertionError(f"{name} must be floating type, got {t.dtype}")
            parts.append(f"{name}={t.dtype}")
        return ", ".join(parts)

    _run_check(
        "6/8 Dtypes",
        "model math expects float inputs (with integer dev indices only)",
        _check_dtypes,
    )

    def _check_finite_values() -> str:
        checked = []
        for name, t in float_tensors:
            if not torch.isfinite(t).all():
                n_bad = int((~torch.isfinite(t)).sum().item())
                raise AssertionError(f"{name} has non-finite values; bad_count={n_bad}, shape={tuple(t.shape)}")
            checked.append(name)
        return f"all finite: {', '.join(checked)}"

    _run_check(
        "7/8 Finite values",
        "NaN/Inf in a batch usually breaks loss/gradients quickly",
        _check_finite_values,
    )

    def _check_forward_smoke() -> str:
        model.eval()
        with torch.no_grad():
            d = _batch_to_device(batch, device)
            out = forward_vit(model, d)
        if out.shape != (B, pv_output_len):
            raise AssertionError(f"output shape mismatch: expected ({B}, {pv_output_len}), got {tuple(out.shape)}")
        if not torch.isfinite(out).all():
            raise AssertionError("output contains NaN/Inf")
        return f"device={device!r}, output_shape={tuple(out.shape)}, output_dtype={out.dtype}"

    _run_check(
        "8/8 Forward smoke test",
        "end-to-end collation-to-model path must execute cleanly once before training",
        _check_forward_smoke,
    )

    print(f"{title}: summary PASS ({passed}/{total} checks passed)")


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

    # Debug: inspect one sample before training loop
    if _RUN_DEBUG_FETCH_INSPECT:
        debug_split = str(_DEBUG_FETCH_SPLIT).strip().lower()
        if debug_split not in {"train", "test"}:
            raise ValueError(f"_DEBUG_FETCH_SPLIT must be 'train' or 'test', got {debug_split!r}")
        debug_dataset = train_dataset if debug_split == "train" else test_dataset
        split_mask = debug_dataset._train_anchor_mask if debug_split == "train" else debug_dataset._test_anchor_mask

        if _DEBUG_FETCH_USE_RANDOM_FILE_AND_ROW:
            rng = np.random.default_rng()
            fi = int(rng.integers(0, len(debug_dataset.sample_files)))
            sample_path = debug_dataset.sample_files[fi]
            df = load_csv(sample_path)
            inv = (
                pd.to_numeric(df[INVERTER_STATE_COL], errors="coerce").fillna(0).astype(int).values
                == VALID_STATE
            )
            valid_mask = inv[debug_dataset._y_idx_per_anchor].any(axis=1) & split_mask
            valid_rows = np.nonzero(valid_mask)[0]
            r = int(rng.choice(valid_rows))
            what_to_fetch(debug_dataset, df, r)
            data_fetched(debug_dataset, sample_path, df, r)
        else:
            if not str(_DEBUG_FETCH_CSV_PATH).strip():
                raise ValueError(
                    "_DEBUG_FETCH_CSV_PATH must be set when _DEBUG_FETCH_USE_RANDOM_FILE_AND_ROW is False"
                )
            want = Path(_DEBUG_FETCH_CSV_PATH).expanduser()
            resolved_files = {p.resolve(): p for p in debug_dataset.sample_files}
            key = want.resolve()
            if key not in resolved_files:
                names = {p.name: p for p in debug_dataset.sample_files}
                if want.name in names:
                    sample_path = names[want.name]
                else:
                    raise ValueError(
                        f"_DEBUG_FETCH_CSV_PATH={want!r} is not in {debug_split}_dataset.sample_files "
                        f"(try a full path or exact CSV basename from pv_dir)"
                    )
            else:
                sample_path = resolved_files[key]
            df = load_csv(sample_path)
            inv = (
                pd.to_numeric(df[INVERTER_STATE_COL], errors="coerce").fillna(0).astype(int).values
                == VALID_STATE
            )
            valid_mask = inv[debug_dataset._y_idx_per_anchor].any(axis=1) & split_mask
            valid_rows = np.nonzero(valid_mask)[0]
            ct = pd.to_datetime(df["collectTime"], errors="coerce")
            desired = pd.Timestamp(_DEBUG_FETCH_LAST_INPUT_COLLECT_TIME)
            matches = np.flatnonzero((ct == desired).to_numpy())
            if matches.size != 1:
                raise ValueError(
                    f"expected exactly one row with collectTime={desired!r}, found {matches.size} in {sample_path.name}"
                )
            j = int(matches[0])
            last_x_for_valid = debug_dataset._x_idx_per_anchor[valid_rows, -1]
            valid_last_set = set(last_x_for_valid.tolist())
            if j not in valid_last_set:
                order = np.argsort(np.abs(last_x_for_valid.astype(np.int64) - j))
                hint_lines = []
                for k in order[:10]:
                    jj = int(last_x_for_valid[int(k)])
                    hint_lines.append(f"  row={jj}  collectTime={ct.iloc[jj]}")
                j_best = int(last_x_for_valid[int(order[0])])
                hint = "\n".join(hint_lines)
                raise ValueError(
                    f"collectTime row index j={j} is not the last PV input row of any valid {debug_split} anchor "
                    f"for {sample_path.name} ({debug_split} split + inverter mask). "
                    "This split only uses windows whose last input row is one of these anchor end rows.\n"
                    "Nearest valid last-input rows (copy a collectTime into _DEBUG_FETCH_LAST_INPUT_COLLECT_TIME):\n"
                    f"{hint}\n"
                    f"Closest valid row index: {j_best} (Δ {abs(j_best - j)} rows from your j={j})."
                )
            what_to_fetch(debug_dataset, df, 0, anchor_last_row=j)
            data_fetched(debug_dataset, sample_path, df, 0, anchor_last_row=j)
        return

    if _RUN_DEBUG_BATCH_INSPECT:
        batch0 = next(iter(train_loader))
        batch_evaluation(
            batch0,
            num_devices=len(dev_dn_list),
            pv_input_len=args.pv_input_len,
            pv_output_len=args.pv_output_len,
            satimg_window_size=args.satimg_window_size,
            skyimg_window_size=args.skyimg_window_size,
            skyimg_spatial_size=args.skyimg_spatial_size,
            satimg_npy_shape_hwc=satimg_hwc,
            model=model,
            device=device,
        )

    return

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
