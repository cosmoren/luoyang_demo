"""
Rolling inference over every test-set timestamp.

For each (inverter, t0) anchor in the test split, run the trained model and save
the full 192-step (15-min cadence, covers t0+15min ... t0+48h) forecast plus
targets / mask / cos(zenith) into ``<output_dir>/<inverter_devDn>.npz``
(one NPZ per inverter).

Step k (0-indexed) corresponds to ``t0 + (k+1) * 15 min``:
  * step 0  = t0 + 15 min
  * step 15 = t0 + 4 h
  * step 191 = t0 + 48 h

NPZ contents per inverter (all arrays aligned along the window axis, length nw).
All entries are plain numpy arrays — no pickle needed; ``np.load(path)`` works
without ``allow_pickle=True``:
  * t0_utc            : (nw,)     '<U32'  ISO UTC string for each rolling anchor
  * pred_kW           : (nw, 192) float32  forecast PV power in kW
  * target_kW         : (nw, 192) float32  ground-truth PV power in kW
  * target_mask       : (nw, 192) uint8    1 = valid, 0 = inverter_state != 512
  * cos_zenith        : (nw, 192) float32  cos(solar zenith) at each horizon
  * forecast_dt_min   : ()        int32    forecast step in minutes (=15)
  * pv_output_len     : ()        int32    number of horizons (=192)
  * device_id         : ()        int32    PVDataset.devDn_list index
  * devDn             : ()        '<U64'   inverter devDn (e.g. ``X.YY=A1``)
  * stride_min        : ()        int32    test-anchor stride used
  * pv_scale_kW       : ()        float32  multiplied into preds/targets (=50.0)

The test anchor stride can be tightened from the default of
``pv_output_interval_min`` (15 min) all the way down to ``csv_interval_min``
(5 min) via ``--stride_min`` (default 5).

Example:

  python inference/infer_testset.py \
      --checkpoint /mnt/nfs/slurm/home/yuan/workspace/checkpoints_4h/pv_forecast_vit_best_gpu0.pt \
      --output_dir inference_results/test_rolling \
      --dataset_config conf_luoyang.yaml \
      --stride_min 5
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"
_DATASETS_CONFIG_DIR = _CONFIG_DIR / "datasets"
sys.path.insert(0, str(_PROJECT_ROOT))

from dataloader.luoyang_zarr import PVDataset, collate_batched
from models.models import pv_forecasting_model_vit_imgs


# Re-scale factor: dataset divides active_power by 50 when normalising,
# so we multiply back here to express predictions in kW.
PV_SCALE_KW = 50.0

# forecast_timefeats columns: [sin_az, cos_az, sin_ze, cos_ze, sin_doy, cos_doy, sin_hod, cos_hod, delta_t]
COS_ZENITH_COL = 3


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f)
    return data or {}


def _resolve_dataset_cfg_path(name: str) -> Path:
    p = Path(name)
    if p.name != name:
        raise ValueError(
            f"--dataset_config only accepts a bare filename under "
            f"{_DATASETS_CONFIG_DIR.relative_to(_PROJECT_ROOT)}/ (got {name!r})"
        )
    full = _DATASETS_CONFIG_DIR / p.name
    if not full.is_file():
        raise FileNotFoundError(f"dataset config not found: {full}")
    return full


def _resolve_data_dir(paths_cfg: dict, cfg_path: Path) -> Path:
    raw = paths_cfg.get("data_dir")
    if raw is None or str(raw).strip() == "":
        raise KeyError(f"dataset config paths.data_dir is required (in {cfg_path})")
    p = Path(str(raw))
    return p.resolve() if p.is_absolute() else (_PROJECT_ROOT / p).resolve()


def _dataset_kwargs(
    dataset_config_name: str,
    split: str,
    *,
    stride_min_override: int | None = None,
) -> dict:
    """Mirror ``training/train_vit_test.py::_dataset_kwargs`` but allow stride override."""
    cfg_path = _resolve_dataset_cfg_path(dataset_config_name)
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

    def _req(key: str):
        if key not in sampling_cfg:
            raise KeyError(f"dataset config sampling.{key} is required (in {cfg_path})")
        return sampling_cfg[key]

    pv_dir = (data_dir / _req_path("pv_path")).resolve()
    skyimg_dir = (data_dir / _req_path("sky_image_path")).resolve()
    satimg_dir = (data_dir / _req_path("sat_path")).resolve()

    shwc = _req("satimg_npy_shape_hwc")
    if not isinstance(shwc, (list, tuple)) or len(shwc) != 3:
        raise ValueError(
            f"sampling.satimg_npy_shape_hwc must be [H, W, C] (in {cfg_path})"
        )

    test_stride = int(_req("test_anchor_stride_min"))
    if stride_min_override is not None:
        test_stride = int(stride_min_override)

    return dict(
        config_path=str(cfg_path),
        pv_dir=str(pv_dir),
        skyimg_dir=str(skyimg_dir),
        satimg_dir=str(satimg_dir),
        split=split,
        csv_interval_min=int(_req("csv_interval_min")),
        pv_input_interval_min=int(_req("pv_input_interval_min")),
        pv_input_len=int(_req("pv_input_len")),
        pv_output_interval_min=int(_req("pv_output_interval_min")),
        pv_output_len=int(_req("pv_output_len")),
        pv_train_time_fraction=float(_req("pv_train_time_fraction")),
        test_anchor_stride_min=test_stride,
        val_anchor_stride_min=int(_req("val_anchor_stride_min")),
        test_collect_time_match_tolerance_min=int(_req("test_collect_time_match_tolerance_min")),
        skyimg_window_size=int(_req("skyimg_window_size")),
        skyimg_time_resolution_min=int(_req("skyimg_time_resolution_min")),
        skyimg_spatial_size=int(_req("skyimg_spatial_size")),
        satimg_window_size=int(_req("satimg_window_size")),
        satimg_time_resolution_min=int(_req("satimg_time_resolution_min")),
        satimg_npy_shape_hwc=tuple(int(x) for x in shwc),
        train_samples_per_csv=int(sampling_cfg.get("train_samples_per_csv", 1)),
    )


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {
        "device_id": batch["dev_idx"].to(device, non_blocking=True),
        "pv": batch["pv"].to(device, non_blocking=True),
        "pv_mask": batch["pv_mask"].to(device, non_blocking=True),
        "pv_timefeats": batch["pv_timefeats"].to(device, non_blocking=True),
        "forecast_timefeats": batch["forecast_timefeats"].to(device, non_blocking=True),
        "target_pv": batch["target_pv"].to(device, non_blocking=True),
        "target_mask": batch["target_mask"].to(device, non_blocking=True),
    }
    for key in ("sat_tensor", "sat_timefeats", "skimg_tensor", "skimg_timefeats", "nwp_tensor"):
        v = batch.get(key)
        out[key] = None if v is None else v.to(device, non_blocking=True)
    return out


def _forward(model: torch.nn.Module, d: dict) -> torch.Tensor:
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rolling inference over the PV test set; save per-inverter "
                    "full 192-step forecasts to NPZ."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a trained checkpoint (.pt) for pv_forecasting_model_vit_imgs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(_PROJECT_ROOT / "inference_results" / "test_rolling"),
        help="Directory to write per-inverter NPZ files into.",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="conf_luoyang.yaml",
        help="Bare YAML filename under config/datasets/ (default: conf_luoyang.yaml).",
    )
    parser.add_argument(
        "--stride_min",
        type=int,
        default=5,
        help="Test-anchor stride in minutes (default 5 = one t0 every CSV row). "
             "Must be a positive multiple of csv_interval_min from the dataset config.",
    )
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / cuda:N / cpu (default: cuda if available else cpu).")
    parser.add_argument("--mask_night", action="store_true", default=True,
                        help="Zero predictions where forecast cos(zenith) < 0 (sun below horizon).")
    parser.add_argument("--no_mask_night", dest="mask_night", action="store_false",
                        help="Disable night masking; save raw model output.")
    parser.add_argument("--limit_batches", type=int, default=None,
                        help="Optional: stop after N batches (debug only).")
    parser.add_argument("--max_inverters", type=int, default=None,
                        help="Optional: process only the first N inverters (sorted by "
                             "filename, same order PVDataset uses). Useful for quickly "
                             "checking output format before running on all 626 inverters.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[infer_testset] device={device}")

    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    print(f"[infer_testset] checkpoint={ckpt_path}")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[infer_testset] output_dir={out_dir}")

    test_dataset = PVDataset(
        **_dataset_kwargs(
            args.dataset_config, "test", stride_min_override=args.stride_min
        )
    )
    if args.max_inverters is not None:
        if args.max_inverters < 1:
            raise ValueError(f"--max_inverters must be >= 1 (got {args.max_inverters})")
        n_before = len(test_dataset.sample_files)
        test_dataset.sample_files = test_dataset.sample_files[: args.max_inverters]
        print(
            f"[infer_testset] --max_inverters={args.max_inverters}: "
            f"processing {len(test_dataset.sample_files)}/{n_before} inverters"
        )
    nw = test_dataset._num_test_windows
    n_files = len(test_dataset.sample_files)
    n_total = len(test_dataset)
    T_out = int(test_dataset.pv_output_len)
    dt_min = int(test_dataset.pv_output_interval_min)
    print(
        f"[infer_testset] inverters={n_files}  windows_per_inverter={nw}  "
        f"total_samples={n_total}  stride_min={args.stride_min}  "
        f"horizons={T_out}  step_min={dt_min}"
    )

    t0_list_per_window: list[pd.Timestamp] = list(test_dataset._test_last_x_time_ref)
    if len(t0_list_per_window) != nw:
        raise RuntimeError(
            f"sanity: |_test_last_x_time_ref|={len(t0_list_per_window)} != nw={nw}"
        )
    t0_strs_per_window = np.asarray(
        [pd.Timestamp(t).isoformat() for t in t0_list_per_window], dtype="<U32"
    )
    inverter_names: list[str] = [
        p.stem.replace("_", "=") for p in test_dataset.sample_files
    ]

    model = pv_forecasting_model_vit_imgs(dev_dn_list=test_dataset.devDn_list).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[infer_testset] WARNING: missing keys in state_dict: {len(missing)} "
              f"(first 5: {missing[:5]})")
    if unexpected:
        print(f"[infer_testset] WARNING: unexpected keys in state_dict: {len(unexpected)} "
              f"(first 5: {unexpected[:5]})")
    model.eval()

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batched,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    # Pre-allocate per-inverter arrays so we never carry the whole dataset in RAM as Python objects.
    preds_buf: dict[int, np.ndarray] = {
        i: np.zeros((nw, T_out), dtype=np.float32) for i in range(n_files)
    }
    targets_buf: dict[int, np.ndarray] = {
        i: np.zeros((nw, T_out), dtype=np.float32) for i in range(n_files)
    }
    mask_buf: dict[int, np.ndarray] = {
        i: np.zeros((nw, T_out), dtype=np.uint8) for i in range(n_files)
    }
    cz_buf: dict[int, np.ndarray] = {
        i: np.zeros((nw, T_out), dtype=np.float32) for i in range(n_files)
    }
    device_id_seen: dict[int, int] = {}
    filled: dict[int, np.ndarray] = {
        i: np.zeros((nw,), dtype=bool) for i in range(n_files)
    }

    if device.type == "cuda":
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    else:
        autocast_ctx = contextlib.nullcontext()

    global_idx = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if args.limit_batches is not None and batch_idx >= args.limit_batches:
                break
            d = _batch_to_device(batch, device)
            with autocast_ctx:
                pv_pred = _forward(model, d)
            pv_pred = pv_pred.float()  # [B, T_out]

            pred_np = pv_pred.detach().cpu().numpy()              # [B, T_out]
            tgt_np = d["target_pv"].detach().cpu().numpy()        # [B, T_out]
            mask_np = d["target_mask"].detach().cpu().numpy()     # [B, T_out]
            cz_np = d["forecast_timefeats"][:, :, COS_ZENITH_COL].detach().cpu().numpy()  # [B, T_out]
            dev_ids_np = d["device_id"].detach().cpu().numpy()    # [B]

            if args.mask_night:
                night = cz_np < 0
                pred_np = np.where(night, 0.0, pred_np)

            B = pred_np.shape[0]
            for i in range(B):
                idx = global_idx + i
                file_idx = idx // nw
                win_idx = idx % nw
                if file_idx >= n_files:
                    raise RuntimeError(f"file_idx {file_idx} out of range {n_files}")
                preds_buf[file_idx][win_idx] = pred_np[i] * PV_SCALE_KW
                targets_buf[file_idx][win_idx] = tgt_np[i] * PV_SCALE_KW
                mask_buf[file_idx][win_idx] = mask_np[i].astype(np.uint8)
                cz_buf[file_idx][win_idx] = cz_np[i]
                filled[file_idx][win_idx] = True
                device_id_seen.setdefault(file_idx, int(dev_ids_np[i]))

            global_idx += B
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                print(
                    f"[infer_testset] batch {batch_idx + 1}/"
                    f"{(n_total + args.batch_size - 1) // args.batch_size}  "
                    f"processed_samples={global_idx}"
                )

    print(f"[infer_testset] inference done; total processed = {global_idx}")

    n_written = 0
    summary_rows: list[dict] = []
    for file_idx in range(n_files):
        mask_filled = filled[file_idx]
        if not mask_filled.any():
            continue
        if not mask_filled.all() and args.limit_batches is None:
            print(
                f"[infer_testset] WARNING: inverter {inverter_names[file_idx]!r} "
                f"only had {int(mask_filled.sum())}/{nw} windows filled"
            )

        kept = np.nonzero(mask_filled)[0]
        preds = preds_buf[file_idx][kept]
        targets = targets_buf[file_idx][kept]
        target_mask = mask_buf[file_idx][kept]
        cos_zenith = cz_buf[file_idx][kept]
        t0_strs = t0_strs_per_window[kept]

        inverter_name = inverter_names[file_idx]
        safe_name = inverter_name.replace("=", "_").replace("/", "_")
        out_path = out_dir / f"{safe_name}.npz"
        np.savez_compressed(
            out_path,
            t0_utc=t0_strs,
            pred_kW=preds,
            target_kW=targets,
            target_mask=target_mask,
            cos_zenith=cos_zenith,
            forecast_dt_min=np.int32(dt_min),
            pv_output_len=np.int32(T_out),
            device_id=np.int32(device_id_seen.get(file_idx, -1)),
            devDn=np.asarray(inverter_name, dtype="<U64"),
            stride_min=np.int32(args.stride_min),
            pv_scale_kW=np.float32(PV_SCALE_KW),
        )
        n_written += 1

        # Quick metrics on the two horizons the user originally cared about.
        for label, k in (("15min", 0), ("4h", 15)):
            m = target_mask[:, k].astype(bool)
            if m.any():
                err = preds[m, k] - targets[m, k]
                rmse = float(np.sqrt(np.mean(err * err)))
                mae = float(np.mean(np.abs(err)))
            else:
                rmse = float("nan")
                mae = float("nan")
            summary_rows.append({
                "inverter": inverter_name,
                "horizon": label,
                "n_valid": int(m.sum()),
                "rmse_kW": rmse,
                "mae_kW": mae,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"[infer_testset] wrote {n_written} per-inverter NPZs to {out_dir}")
    print(f"[infer_testset] summary saved to {summary_path}")
    if not summary_df.empty:
        for h in ["15min", "4h"]:
            sub = summary_df[summary_df["horizon"] == h]
            if not sub.empty:
                print(
                    f"  horizon={h}: "
                    f"mean RMSE={sub['rmse_kW'].mean():.3f} kW  "
                    f"mean MAE={sub['mae_kW'].mean():.3f} kW  "
                    f"(over {len(sub)} inverters)"
                )


if __name__ == "__main__":
    main()
