"""
YLJ matrix Parquet training entrypoint.

Uses ``dataloader/ylj_zarr.py`` and the shared loop in ``training/train.py``.
Requires ``--config config/datasets/conf_ylj.yaml`` and ``--ylj_raw_parquet``.

Training uses kt from the solar features CSV (Luoyang yr pattern): model input ``kt/20``,
loss on reconstructed kW ``kt_pred * target_p_cs * p_mean``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.training_conf import bootstrap_config_from_argv

bootstrap_config_from_argv()

import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from config_utils import get_resolved_paths
from dataloader.ylj_zarr import (
    YljRawParquetDataset,
    YljRawParquetEmptyValDataset,
    _utc_wall_naive,
    collate_ylj_batched,
    ylj_raw_parquet_matrix_config_from_conf,
)
import training.train as base_train

# Luoyang yr ``train_vit_test.py``: model input kt/KT_INPUT_SCALE, output * scale, power = kt * p_cs * p_mean.
KT_INPUT_SCALE = 20.0


def _ylj_require_kt_batch(batch: dict) -> None:
    for key in ("kt", "kt_mask", "p_mean", "target_p_cs"):
        if key not in batch:
            raise RuntimeError(
                f"YLJ kt training requires batch['{key}']; set ylj_raw_parquet.solar_features_csv "
                "and rebuild solar_features CSV."
            )


def _ylj_forward_kt(
    model: nn.Module,
    batch: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run model on kt/KT_INPUT_SCALE; return (kt_pred, pv_pred_kw)."""
    _ylj_require_kt_batch(batch)
    device_id = batch["dev_idx"].to(device)
    kt_in = batch["kt"].to(device) / KT_INPUT_SCALE
    kt_mask = batch["kt_mask"].to(device)
    pv_timefeats = batch["pv_timefeats"].to(device)
    forecast_timefeats = batch["forecast_timefeats"].to(device)
    model_kw = base_train._batch_model_kwargs(batch, device)
    kt_raw = model(
        device_id,
        kt_in,
        kt_mask,
        pv_timefeats,
        forecast_timefeats,
        **model_kw,
    )
    kt_pred = kt_raw * KT_INPUT_SCALE
    p_mean = batch["p_mean"].to(device)
    target_p_cs = batch["target_p_cs"].to(device)
    pv_pred = kt_pred * target_p_cs * p_mean.unsqueeze(1)
    return kt_pred, pv_pred


def _ylj_train_one_epoch(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    max_batches: int | None = None,
    *,
    epoch: int | None = None,
    log_every: int = 50,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    num_batches = len(loader)
    if max_batches is not None:
        num_batches = min(num_batches, max_batches)
    ep = "" if epoch is None else f"epoch {epoch} "
    print(f"{ep}number of batches: {len(loader)}" + (f" (capped at {max_batches})" if max_batches is not None else ""))
    print(f"[train_ylj] kt training: input kt/{KT_INPUT_SCALE}, loss on reconstructed kW power")
    running_loss = 0.0
    running_batches = 0
    log_every = max(1, int(log_every))
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        B = batch["dev_idx"].size(0)
        optimizer.zero_grad()
        _, pv_pred = _ylj_forward_kt(model, batch, device)
        target_pv = batch["target_pv"].to(device)
        loss = criterion(pv_pred, target_pv)
        loss.backward()
        optimizer.step()
        loss_b = float(loss.item())
        total_loss += loss_b
        n += B
        running_loss += loss_b
        running_batches += 1
        done = batch_idx + 1
        if done == 1 or done % log_every == 0 or done == num_batches:
            avg_running = running_loss / max(running_batches, 1)
            print(
                f"  [train] {ep}batch {done}/{num_batches}  "
                f"loss_batch={loss_b:.6f}  loss_avg_running={avg_running:.6f}"
            )
    avg_loss = total_loss / max(n, 1)
    print(f"  [train] {ep}end  mean_loss_returned={avg_loss:.6f} (sum_batch_loss / n_samples={n})")
    return avg_loss


def _is_ylj_mode() -> bool:
    conf = base_train.load_config()
    site = conf.get("site", {})
    return str(site.get("name", "")).strip().lower() == "ylj"


def _use_ylj_raw_parquet(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "ylj_raw_parquet", False))


def _use_ylj_parquet_nwp(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "ylj_parquet_nwp", False))


def _use_ylj_sat_zarr(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "ylj_sat_zarr", False))


def _build_pv_dataset(
    args: argparse.Namespace, satimg_hwc: tuple[int, int, int], split: str
) -> Dataset:
    del satimg_hwc  # Zarr satellite uses fixed Luoyang window; NPY shape unused.
    if _use_ylj_parquet_nwp(args) and not _use_ylj_raw_parquet(args):
        raise ValueError("--ylj_parquet_nwp requires --ylj_raw_parquet")
    if _use_ylj_sat_zarr(args) and not _use_ylj_raw_parquet(args):
        raise ValueError("--ylj_sat_zarr requires --ylj_raw_parquet")
    if not _use_ylj_raw_parquet(args):
        raise ValueError("train_ylj.py requires --ylj_raw_parquet (matrix Parquet backend only)")
    if not _is_ylj_mode():
        raise ValueError("--ylj_raw_parquet requires site.name: ylj in the active config")
    conf = base_train.load_config()
    site = conf.get("site", {})
    lat = float(site["latitude"])
    lon = float(site["longitude"])
    paths = get_resolved_paths(conf, _PROJECT_ROOT)
    path_defaults = base_train.get_training_paths_from_conf(conf)
    raw_root = paths.get("ylj_raw_parquet_dir")
    if raw_root is None or not raw_root.is_dir():
        raise FileNotFoundError(
            "When using --ylj_raw_parquet, conf paths.ylj_raw_parquet_dir must be set to an existing directory "
            f"(got {raw_root!r})"
        )
    pv_dev = paths.get("pv_device_path")
    if pv_dev is None or not pv_dev.is_file():
        raise FileNotFoundError(f"pv_device_path not found: {pv_dev}")
    dev_dn_list = pd.read_excel(pv_dev)["devDn"].dropna().unique().tolist()
    stem = "NE=ylj"
    try:
        dev_i = int(dev_dn_list.index(stem))
    except ValueError:
        dev_i = 0
    if split == "val":
        if bool(getattr(args, "test_only", False)) and bool(getattr(args, "test_only_use_val", False)):
            raise ValueError(
                "YLJ raw Parquet mode has no validation split; do not combine --ylj_raw_parquet with "
                "--test_only_use_val."
            )
        return YljRawParquetEmptyValDataset()
    hp = base_train.get_training_hparams_from_conf(conf)
    mx_cfg = ylj_raw_parquet_matrix_config_from_conf(conf)
    include_export = bool(getattr(args, "test_only", False)) and split == "test"
    sat_zarr_dir = None
    if _use_ylj_sat_zarr(args):
        sat_rel = str(conf.get("paths", {}).get("sat_path", "yalongjiang_zarr")).strip()
        if not sat_rel:
            raise ValueError("paths.sat_path must be set for --ylj_sat_zarr")
        # Zarr lives next to Parquet under ``ylj_raw_parquet_dir`` (e.g. ``.../ylj_dataset_raw/yalongjiang_zarr``).
        sat_candidates = [
            Path(raw_root) / sat_rel,
            Path(path_defaults["satimg_dir"]),
        ]
        for cand in sat_candidates:
            if cand.is_dir():
                sat_zarr_dir = str(cand.resolve())
                break
        if sat_zarr_dir is None:
            raise FileNotFoundError(
                f"--ylj_sat_zarr: satellite Zarr not found; tried {[str(p) for p in sat_candidates]}"
            )
    return YljRawParquetDataset(
        str(raw_root),
        split=split,
        pv_input_interval_min=int(args.pv_input_interval_min),
        pv_input_len=int(args.pv_input_len),
        pv_output_interval_min=int(args.pv_output_interval_min),
        pv_output_len=int(args.pv_output_len),
        latitude=lat,
        longitude=lon,
        dev_dn_index=dev_i,
        matrix=mx_cfg,
        use_nwp=_use_ylj_parquet_nwp(args),
        use_sat_zarr=_use_ylj_sat_zarr(args),
        sat_zarr_dir=sat_zarr_dir,
        pv_value_scale=float(hp["pv_value_scale"]),
        include_export_metadata=include_export,
    )


def _export_parquet_test_sequence_pairs_csv(
    model: torch.nn.Module,
    device: torch.device,
    loader: DataLoader,
    out_path: Path,
) -> int:
    """YLJ Parquet test export: ``collectTime`` = anchor UTC; ``gt_pred_pairs`` = ``[[gt_kw, pred_kw, kt_pred], ...]``."""
    model.eval()
    ds = loader.dataset
    if not isinstance(ds, YljRawParquetDataset):
        raise TypeError(f"expected YljRawParquetDataset, got {type(ds).__name__}")
    pv_scale = float(ds._pv_value_scale)
    total = len(ds)
    rows: list[dict] = []
    processed = 0
    bar_width = 30
    with torch.no_grad():
        for batch in loader:
            if "csv_collect_time_utc" not in batch:
                raise RuntimeError(
                    "Parquet test export requires csv_collect_time_utc on each batch; "
                    "ensure collate_ylj_batched is used and YljRawParquetDataset is current."
                )
            kt_pred_t, pv_pred_t = _ylj_forward_kt(model, batch, device)
            kt_np = kt_pred_t.detach().cpu().float().numpy()
            pred_np = pv_pred_t.detach().cpu().float().numpy()
            gt_np = batch["target_pv"].detach().cpu().float().numpy() * pv_scale
            collect_list = batch["csv_collect_time_utc"]
            B = int(pred_np.shape[0])
            for i in range(B):
                pairs = [
                    [float(gt_np[i, k]), float(pred_np[i, k]), float(kt_np[i, k])]
                    for k in range(pred_np.shape[1])
                ]
                rows.append(
                    {
                        "collectTime": str(collect_list[i]),
                        "gt_pred_pairs": json.dumps(pairs, ensure_ascii=False),
                    }
                )
            processed += B
            pct = (100.0 * processed / max(total, 1))
            filled = int(bar_width * processed / max(total, 1))
            bar = "#" * filled + "-" * (bar_width - filled)
            print(
                f"\rPredicting test samples [{bar}] {processed}/{total} ({pct:5.1f}%)",
                end="",
                flush=True,
            )
    print()
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return len(rows)


def _export_test_sequence_pairs_csv(
    model: torch.nn.Module,
    device: torch.device,
    loader: DataLoader,
    out_path: Path,
) -> int:
    """YLJ test export: one row per anchor; each horizon step is ``[gt_kw, pred_kw, kt_pred]``."""
    ds = loader.dataset
    if isinstance(ds, YljRawParquetDataset):
        return _export_parquet_test_sequence_pairs_csv(model, device, loader, out_path)
    raise TypeError(
        f"train_ylj test export requires YljRawParquetDataset, got {type(ds).__name__}"
    )


def _run_test_x_timestamps() -> bool:
    """Print the last 10 exact X timestamps from one training batch, then exit."""
    parser = argparse.ArgumentParser(add_help=False)
    conf = base_train.load_config()
    h = base_train.get_training_hparams_from_conf(conf)
    path_defaults = base_train.get_training_paths_from_conf(conf)
    parser.add_argument("--test_x_timestamps", action="store_true")
    parser.add_argument("--batch_size", type=int, default=h["batch_size"])
    parser.add_argument("--pv-dir", type=str, default=path_defaults["pv_dir"])
    parser.add_argument("--skyimg-dir", type=str, default=path_defaults["skyimg_dir"])
    parser.add_argument("--satimg-dir", type=str, default=path_defaults["satimg_dir"])
    parser.add_argument("--nwp-dir", type=str, default=path_defaults.get("nwp_dir"))
    parser.add_argument("--csv_interval_min", type=int, default=h["csv_interval_min"])
    parser.add_argument("--pv_input_interval_min", type=int, default=h["pv_input_interval_min"])
    parser.add_argument("--pv_input_len", type=int, default=h["pv_input_len"])
    parser.add_argument("--pv_output_len", type=int, default=h["pv_output_len"])
    parser.add_argument("--pv_output_interval_min", type=int, default=h["pv_output_interval_min"])
    parser.add_argument("--t_off_min", type=int, default=h["t_off_min"])
    parser.add_argument("--pv_output_rand", action="store_true", default=h["pv_output_rand"])
    parser.add_argument("--pv_train_time_fraction", type=float, default=h["pv_train_time_fraction"])
    parser.add_argument("--test_anchor_stride_min", type=int, default=h["test_anchor_stride_min"])
    parser.add_argument("--val_anchor_stride_min", type=int, default=h["val_anchor_stride_min"])
    parser.add_argument(
        "--test_collect_time_match_tolerance_min",
        type=int,
        default=h["test_collect_time_match_tolerance_min"],
    )
    parser.add_argument("--skyimg_window_size", type=int, default=h["skyimg_window_size"])
    parser.add_argument("--skyimg_time_resolution_min", type=int, default=h["skyimg_time_resolution_min"])
    parser.add_argument("--skyimg_spatial_size", type=int, default=h["skyimg_spatial_size"])
    parser.add_argument("--satimg_window_size", type=int, default=h["satimg_window_size"])
    parser.add_argument("--satimg_time_resolution_min", type=int, default=h["satimg_time_resolution_min"])
    parser.add_argument(
        "--satimg_npy_shape_hwc",
        type=int,
        nargs=3,
        default=list(h["satimg_npy_shape_hwc"]),
        metavar=("H", "W", "C"),
    )
    args, _ = parser.parse_known_args()
    if not args.test_x_timestamps:
        return False
    setattr(args, "ylj_raw_parquet", True)

    satimg_hwc = tuple(args.satimg_npy_shape_hwc)
    ds = _build_pv_dataset(args, satimg_hwc, "train")
    if len(ds) == 0:
        print("[x-ts test] train dataset is empty.")
        return True

    bs = min(max(1, args.batch_size), len(ds))
    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        collate_fn=collate_ylj_batched,
        num_workers=0,
    )
    batch = next(iter(loader))
    row = 0
    ts_x, ts_y = ds.row_window_timestamps_utc(row)
    x_ts = [_utc_wall_naive(t).strftime("%Y%m%d_%H%M%S") for t in ts_x]
    y_ts = [_utc_wall_naive(t).strftime("%Y-%m-%d %H:%M:%S") for t in ts_y]
    pv_scale = float(ds._pv_value_scale)
    x_p = (batch["pv"][0, 0].detach().cpu().float().numpy() * pv_scale).tolist()
    y_p = (batch["target_pv"][0].detach().cpu().float().numpy() * pv_scale).tolist()
    print("[x-ts test] last 10 X timestamps (YYYYMMDD_HHMMSS):")
    print(x_ts[-10:])
    print("[x-ts test] last 10 X timestamp/power pairs:")
    print(list(zip(x_ts[-10:], x_p[-10:])))
    print("[x-ts test] all Y timestamps:")
    print(y_ts)
    print("[x-ts test] all Y timestamp/power pairs:")
    print(list(zip(y_ts, y_p)))
    return True


def _run_loader_test_simple() -> bool:
    """Print X[-2], X[-1], Y[0] timestamps/power/masks, then exit."""
    parser = argparse.ArgumentParser(add_help=False)
    conf = base_train.load_config()
    h = base_train.get_training_hparams_from_conf(conf)
    path_defaults = base_train.get_training_paths_from_conf(conf)
    parser.add_argument("--loader_test_simple", action="store_true")
    parser.add_argument("--loader_test_split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--loader_test_index", type=int, default=0)
    parser.add_argument("--loader_test_max_tries", type=int, default=64)
    parser.add_argument("--loader_test_target_hhmm", type=str, default="14:30")
    parser.add_argument("--loader_test_target_tol_min", type=int, default=30)
    parser.add_argument("--pv-dir", type=str, default=path_defaults["pv_dir"])
    parser.add_argument("--skyimg-dir", type=str, default=path_defaults["skyimg_dir"])
    parser.add_argument("--satimg-dir", type=str, default=path_defaults["satimg_dir"])
    parser.add_argument("--nwp-dir", type=str, default=path_defaults.get("nwp_dir"))
    parser.add_argument("--csv_interval_min", type=int, default=h["csv_interval_min"])
    parser.add_argument("--pv_input_interval_min", type=int, default=h["pv_input_interval_min"])
    parser.add_argument("--pv_input_len", type=int, default=h["pv_input_len"])
    parser.add_argument("--pv_output_len", type=int, default=h["pv_output_len"])
    parser.add_argument("--pv_output_interval_min", type=int, default=h["pv_output_interval_min"])
    parser.add_argument("--t_off_min", type=int, default=h["t_off_min"])
    parser.add_argument("--pv_output_rand", action="store_true", default=h["pv_output_rand"])
    parser.add_argument("--pv_train_time_fraction", type=float, default=h["pv_train_time_fraction"])
    parser.add_argument("--test_anchor_stride_min", type=int, default=h["test_anchor_stride_min"])
    parser.add_argument("--val_anchor_stride_min", type=int, default=h["val_anchor_stride_min"])
    parser.add_argument(
        "--test_collect_time_match_tolerance_min",
        type=int,
        default=h["test_collect_time_match_tolerance_min"],
    )
    parser.add_argument("--skyimg_window_size", type=int, default=h["skyimg_window_size"])
    parser.add_argument("--skyimg_time_resolution_min", type=int, default=h["skyimg_time_resolution_min"])
    parser.add_argument("--skyimg_spatial_size", type=int, default=h["skyimg_spatial_size"])
    parser.add_argument("--satimg_window_size", type=int, default=h["satimg_window_size"])
    parser.add_argument("--satimg_time_resolution_min", type=int, default=h["satimg_time_resolution_min"])
    parser.add_argument(
        "--satimg_npy_shape_hwc",
        type=int,
        nargs=3,
        default=list(h["satimg_npy_shape_hwc"]),
        metavar=("H", "W", "C"),
    )
    args, _ = parser.parse_known_args()
    if not args.loader_test_simple:
        return False
    setattr(args, "ylj_raw_parquet", True)

    satimg_hwc = tuple(args.satimg_npy_shape_hwc)
    ds = _build_pv_dataset(args, satimg_hwc, args.loader_test_split)
    if len(ds) == 0:
        print("[loader-test] dataset is empty.")
        return True
    idx = max(0, min(int(args.loader_test_index), len(ds) - 1))

    sample = None
    tries = max(1, int(args.loader_test_max_tries))
    hhmm = str(args.loader_test_target_hhmm).strip()
    try:
        target_hour, target_minute = [int(x) for x in hhmm.split(":", 1)]
    except Exception as exc:  # pragma: no cover - CLI parse guard
        raise ValueError(f"invalid --loader_test_target_hhmm={hhmm!r}, expected HH:MM") from exc
    if not (0 <= target_hour <= 23 and 0 <= target_minute <= 59):
        raise ValueError(f"invalid --loader_test_target_hhmm={hhmm!r}, hour/min out of range")
    target_tod_min = target_hour * 60 + target_minute
    tol_min = max(0, int(args.loader_test_target_tol_min))

    for try_i in range(tries):
        row_i = (idx + try_i) % len(ds)
        s = ds[row_i]
        y_mask0_try = float(s["target_mask"][0].detach().cpu().item())
        if y_mask0_try <= 0.0:
            continue
        _, y_times_utc = ds.row_window_timestamps_utc(row_i)
        y0_try = pd.Timestamp(y_times_utc[0])
        t_last_try = y0_try - pd.Timedelta(minutes=int(args.t_off_min))
        tod_min = int(t_last_try.hour) * 60 + int(t_last_try.minute)
        if abs(tod_min - target_tod_min) <= tol_min:
            sample = s
            break
    if sample is None:
        print(
            f"[loader-test] no sample found after {tries} tries for X end around "
            f"{target_hour:02d}:{target_minute:02d} +/- {tol_min}min with valid Y."
        )
        return True

    _, y_times_utc = ds.row_window_timestamps_utc(idx)
    y0 = pd.Timestamp(y_times_utc[0])
    t_last = y0 - pd.Timedelta(minutes=int(args.t_off_min))
    x_last = pd.Timestamp(t_last)
    x_prev = pd.Timestamp(t_last - pd.Timedelta(minutes=int(args.pv_input_interval_min)))
    x_mask = sample["pv_mask"][0, -2:].detach().cpu().float().tolist()
    y_mask0 = float(sample["target_mask"][0].detach().cpu().item())
    pv_scale = float(ds._pv_value_scale)
    x_vals = (sample["pv"][0, -2:].detach().cpu().float().numpy() * pv_scale).tolist()
    y_val0 = float(sample["target_pv"][0].detach().cpu().item() * pv_scale)
    x_all_vals = (sample["pv"][0].detach().cpu().float().numpy() * pv_scale).tolist()
    x_all_mask = sample["pv_mask"][0].detach().cpu().float().tolist()
    x_all_ts = [
        pd.Timestamp(t_last - pd.Timedelta(minutes=int(args.pv_input_interval_min) * (len(x_all_vals) - 1 - i)))
        for i in range(len(x_all_vals))
    ]

    fmt = "%Y-%m-%d_%H-%M-%S"
    print("X (all):")
    for i, (tx, pv, mk) in enumerate(zip(x_all_ts, x_all_vals, x_all_mask)):
        print(f"  X[{i:03d}]: {tx.strftime(fmt)} | pv={float(pv)} | mask={float(mk)}")
    print(f"X[-2]: {x_prev.strftime(fmt)} | pv={float(x_vals[0])} | mask={x_mask[0]}")
    print(f"X[-1]: {x_last.strftime(fmt)} | pv={float(x_vals[1])} | mask={x_mask[1]}")
    print(f"Y[0] : {y0.strftime(fmt)} | pv={y_val0} | mask={y_mask0}")
    return True


def _run_debug_test_only_timestamps() -> bool:
    """
    Debug helper for deterministic test-only mode.

    Prints X (timestamp, active_power) and single-point Y (timestamp, active_power)
    for user-provided target timestamps, using the configured single-horizon offset:
    ``t_last_x = ts_y - t_off_min``.
    """
    parser = argparse.ArgumentParser(add_help=False)
    conf = base_train.load_config()
    h = base_train.get_training_hparams_from_conf(conf)
    path_defaults = base_train.get_training_paths_from_conf(conf)
    parser.add_argument("--debug_test_timestamps", nargs="+", default=None)
    parser.add_argument("--pv-dir", type=str, default=path_defaults["pv_dir"])
    parser.add_argument("--skyimg-dir", type=str, default=path_defaults["skyimg_dir"])
    parser.add_argument("--satimg-dir", type=str, default=path_defaults["satimg_dir"])
    parser.add_argument("--nwp-dir", type=str, default=path_defaults.get("nwp_dir"))
    parser.add_argument("--csv_interval_min", type=int, default=h["csv_interval_min"])
    parser.add_argument("--pv_input_interval_min", type=int, default=h["pv_input_interval_min"])
    parser.add_argument("--pv_input_len", type=int, default=h["pv_input_len"])
    parser.add_argument("--pv_output_len", type=int, default=h["pv_output_len"])
    parser.add_argument("--pv_output_interval_min", type=int, default=h["pv_output_interval_min"])
    parser.add_argument("--t_off_min", type=int, default=h["t_off_min"])
    parser.add_argument("--pv_output_rand", action="store_true", default=h["pv_output_rand"])
    parser.add_argument("--pv_train_time_fraction", type=float, default=h["pv_train_time_fraction"])
    parser.add_argument("--test_anchor_stride_min", type=int, default=h["test_anchor_stride_min"])
    parser.add_argument("--val_anchor_stride_min", type=int, default=h["val_anchor_stride_min"])
    parser.add_argument(
        "--test_collect_time_match_tolerance_min",
        type=int,
        default=h["test_collect_time_match_tolerance_min"],
    )
    parser.add_argument("--skyimg_window_size", type=int, default=h["skyimg_window_size"])
    parser.add_argument("--skyimg_time_resolution_min", type=int, default=h["skyimg_time_resolution_min"])
    parser.add_argument("--skyimg_spatial_size", type=int, default=h["skyimg_spatial_size"])
    parser.add_argument("--satimg_window_size", type=int, default=h["satimg_window_size"])
    parser.add_argument("--satimg_time_resolution_min", type=int, default=h["satimg_time_resolution_min"])
    parser.add_argument(
        "--satimg_npy_shape_hwc",
        type=int,
        nargs=3,
        default=list(h["satimg_npy_shape_hwc"]),
        metavar=("H", "W", "C"),
    )
    args, _ = parser.parse_known_args()
    if not args.debug_test_timestamps:
        return False

    satimg_hwc = tuple(args.satimg_npy_shape_hwc)
    # Force deterministic single-horizon mode over all rows in the last 30%.
    setattr(args, "test_only_y_offset_min", int(args.t_off_min))
    setattr(args, "test_only_all_last30_rows", True)
    args.pv_output_len = 1
    # Avoid preloading every station CSV; debug only needs one file (first in sorted list).
    setattr(args, "skippd_max_pv_files_to_load", 1)
    print("[debug-test-ts] fast path: loading only the first CSV under pv_dir.")
    ds = _build_pv_dataset(args, satimg_hwc, "test")
    if len(ds) == 0:
        print("[debug-test-ts] test dataset is empty.")
        return True

    path = ds.sample_files[0]
    if path not in ds._df_cache or path not in ds._time_index_cache:
        raise RuntimeError(f"missing cached data for {path}")
    df = ds._df_cache[path]
    m = ds._time_index_cache[path]
    ct = pd.to_datetime(df["collectTime"], errors="coerce")
    print(f"[debug-test-ts] using pv file: {path}")

    stem = path.stem.replace("_", "=")
    try:
        dev_idx = torch.tensor(ds.devDn_list.index(stem), dtype=torch.long)
    except ValueError:
        dev_idx = torch.tensor(0, dtype=torch.long)

    for raw_ts in args.debug_test_timestamps:
        ts_y = pd.Timestamp(raw_ts)
        print(f"\n===== debug ts_y={ts_y} (+{int(args.t_off_min)}min case) =====")
        if not bool((ct == ts_y).any()):
            print("NOT FOUND in CSV collectTime; skipping.")
            continue

        ts_x_last = ts_y - pd.Timedelta(minutes=int(args.t_off_min))
        direct_pw, direct_inv = ds._get_x_with_interp(path, df, m, ts_x_last)
        print(
            "[debug-test-ts] direct _get_x_with_interp at "
            f"{ts_x_last.strftime('%Y-%m-%d %H:%M:%S')}: "
            f"active_power={float(direct_pw)}, inverter_state={int(direct_inv)}"
        )

        sample = ds._build_test_only_single_y_sample(path, dev_idx, ts_y)

        t_last = ts_x_last
        t0 = t_last - ds._pv_step * (ds.pv_input_len - 1)
        neighbor_max_dist_ns = int(pd.Timedelta(minutes=1) / pd.Timedelta("1ns"))
        ts_x, pw_x, _, _ = ds._build_x(path, df, m, t0, neighbor_max_dist_ns)

        print("X (timestamp, active_power):")
        for t, p in zip(ts_x, pw_x.tolist()):
            print(f"{pd.Timestamp(t).strftime('%Y-%m-%d %H:%M:%S')}, {float(p)}")

        y_ts = sample["target_timestamps_utc"][0]
        y_pw = float(sample["target_pv"][0].item() * float(ds._pv_value_scale))
        print("Y (timestamp, active_power):")
        print(f"{y_ts}, {y_pw}")
    return True


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre.add_argument("--test_only", action="store_true")
    pre_args, _ = pre.parse_known_args()
    base_train._apply_config_override(pre_args.config)

    if _run_loader_test_simple():
        return
    if _run_test_x_timestamps():
        return
    # Reuse the original training logic while switching only the data backend.
    base_train._build_pv_dataset = _build_pv_dataset
    base_train.collate_batched = collate_ylj_batched
    if _is_ylj_mode() and pre_args.test_only:
        # YLJ test-only: dense all-row evaluation from pv_test with sequence-pair CSV export.
        base_train.export_test_single_target_csv = _export_test_sequence_pairs_csv
        base_train.export_test_horizon_csvs = lambda model, device, loader, p15, p4: (
            _export_test_sequence_pairs_csv(model, device, loader, p15),
            0,
        )
        # Force all-row anchors for test split when running through base_train.
        _orig_parse_args = argparse.ArgumentParser.parse_args

        def _parse_args_with_ylj(self, *args, **kwargs):
            ns = _orig_parse_args(self, *args, **kwargs)
            setattr(ns, "test_only_all_rows", True)
            return ns

        argparse.ArgumentParser.parse_args = _parse_args_with_ylj
    if _is_ylj_mode() and (not pre_args.test_only):
        print(
            "[train_ylj] ylj mode active: full data is used for training; "
            "validation/testing evaluation is disabled."
        )
        base_train._should_validate = lambda epoch, total_epochs, val_every: False

        def _eval_skip(model, device, loader, criterion):
            print("[train_ylj] ylj mode: skip evaluate() on val/test.")
            return float("nan"), float("nan")

        base_train.evaluate = _eval_skip
        base_train.train_one_epoch = _ylj_train_one_epoch
    base_train.main()


if __name__ == "__main__":
    main()
