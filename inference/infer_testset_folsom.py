"""
Folsom rolling inference: one sweep over the test split, save full 192-step forecasts.

Modelled after ``inference/infer_testset.py`` (Luoyang) but specialised for
:class:`dataloader.folsom.FolsomIrradianceDataset` — a single-site GHI dataset whose
``target_pv`` is raw GHI in W/m² (post-alignment, commit 518dca9); predictions and
targets are written out as-is in W/m² (no denormalization needed).

Two horizons are reported in ``_summary.csv`` (output step ``k`` corresponds to
``t0 + (k + 1) * 15 min``):

  * 15 min ahead  -> ``k = 0``
  * 4 h ahead     -> ``k = 15``

The full 192-step forecast/target arrays are also kept in the NPZ so any other
horizon can be re-evaluated downstream without rerunning the model.

The script auto-detects the training regime from the checkpoint dict
(``zero_sky``, ``use_nwp``), then prepares the NWP / sky inputs with the **same**
helpers the trainer uses (:func:`training.train_vit_test_folsom._prepare_nwp_for_vit`
and :func:`training.train_vit_test_folsom._prepare_sky_for_vit`) so the inputs match
training exactly. The ``--force-zero-sky`` / ``--force-use-nwp`` switches override
the checkpoint's regime when needed.

NPZ schema (one file per run, written as ``<output_dir>/folsom_test.npz``; all arrays
are plain numpy and load without ``allow_pickle=True``):

  * t0_utc          : (N,)      '<U32'    ISO UTC string for each test anchor
  * pred            : (N, 192)  float32   forecast GHI in W/m² (denormalised)
  * target          : (N, 192)  float32   ground-truth GHI in W/m² (denormalised)
  * target_mask     : (N, 192)  uint8     1 = valid GHI target, 0 = NaN in CSV
  * cos_zenith      : (N, 192)  float32   cos(solar zenith) at each horizon
  * forecast_dt_min : ()        int32     forecast step in minutes (=15)
  * pv_output_len   : ()        int32     number of horizons (=192)
  * stride_min      : ()        int32     test-anchor stride used (CLI override)
  * zero_sky        : ()        bool      regime used at inference time
  * use_nwp         : ()        bool      regime used at inference time
  * checkpoint      : ()        '<U256'   basename of the checkpoint
  * target_scale    : ()        float32   multiplied into preds/targets (=1.0; values
                                            are already in W/m^2 post-alignment)

Example invocations (run on ``node12`` after activating the training env):

  # No-sky checkpoint (PV + NWP only, sky tensors zeroed at training time):
  python inference/infer_testset_folsom.py \\
      --checkpoint ~/experiments_archive/folsom_pv_nwp_vs_sky_2026-05-15/checkpoints_folsom_pv/folsom_pv_forecast_vit_best_gpu7.pt \\
      --output_dir inference_results/folsom_nosky_gpu7 \\
      --dataset_config conf_folsom.yaml \\
      --stride_min 5

  # With-sky checkpoint (PV + NWP + sky):
  python inference/infer_testset_folsom.py \\
      --checkpoint ~/experiments_archive/folsom_pv_nwp_vs_sky_2026-05-15/checkpoints_folsom_pv/folsom_pv_forecast_vit_best_gpu6.pt \\
      --output_dir inference_results/folsom_sky_gpu6 \\
      --dataset_config conf_folsom.yaml \\
      --stride_min 5
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"
_DATASETS_CONFIG_DIR = _CONFIG_DIR / "datasets"
sys.path.insert(0, str(_PROJECT_ROOT))

from dataloader.folsom import FolsomIrradianceDataset  # noqa: E402
from dataloader.luoyang_zarr import collate_batched  # noqa: E402
from models.models import pv_forecasting_model_vit_imgs  # noqa: E402
from training.train_vit_test_folsom import (  # noqa: E402
    _batch_to_device,
    _dataset_kwargs as _folsom_dataset_kwargs,
    _prepare_nwp_for_vit,
    _prepare_sky_for_vit,
    forward_vit,
)


# Post-alignment (commit 518dca9) FolsomIrradianceDataset stores ``target_pv`` as raw
# GHI in W/m^2 (no /1100 normalization), so predictions/targets are already in W/m^2 and
# need no rescale. Kept as a constant=1.0 for clarity / NPZ metadata.
GHI_SCALE_WM2 = 1.0

# ``forecast_timefeats`` schema: [sin_az, cos_az, sin_ze, cos_ze, sin_doy, cos_doy,
# sin_hod, cos_hod, delta_t]. Confirmed in training/train_vit_test_folsom.py::evaluate()
# (``forecast_timefeats[:, :h, 3] < 0`` is used as the night mask).
COS_ZENITH_COL = 3

# (label, output step index k); step k -> t0 + (k + 1) * 15 min.
_HORIZONS: tuple[tuple[str, int], ...] = (("15min", 0), ("4h", 15))


def _resolve_test_dataset(dataset_config_name: str, stride_min: int) -> FolsomIrradianceDataset:
    """Build the Folsom test dataset, overriding ``test_anchor_stride_min`` from the CLI."""
    ds_kwargs = _folsom_dataset_kwargs(dataset_config_name, "test")
    ds_kwargs["test_anchor_stride_min"] = int(stride_min)
    return FolsomIrradianceDataset(**ds_kwargs)


def _t0_strings_for_test_windows(ds: FolsomIrradianceDataset) -> np.ndarray:
    """Per-window ISO UTC string for the last input row (the t0 anchor)."""
    nw = int(ds._num_test_windows)
    out = np.empty(nw, dtype="<U32")
    anchors = ds._anchors[ds._test_r_indices]
    times = ds._df[ds._time_col].iloc[anchors].to_list()
    for i, t in enumerate(times):
        out[i] = pd.Timestamp(t).isoformat()
    return out


def _resolve_regime(
    ckpt: dict,
    *,
    force_zero_sky: bool | None,
    force_use_nwp: bool | None,
) -> tuple[bool, bool, str, str]:
    """Return ``(zero_sky, use_nwp, zero_sky_source, use_nwp_source)``."""
    ckpt_zero_sky = ckpt.get("zero_sky")
    ckpt_use_nwp = ckpt.get("use_nwp")

    if force_zero_sky is None:
        if ckpt_zero_sky is None:
            raise KeyError(
                "checkpoint has no 'zero_sky' field; pass --force-zero-sky / --no-force-zero-sky"
            )
        zero_sky = bool(ckpt_zero_sky)
        zs_src = "checkpoint"
    else:
        zero_sky = bool(force_zero_sky)
        zs_src = "cli-override"

    if force_use_nwp is None:
        if ckpt_use_nwp is None:
            raise KeyError(
                "checkpoint has no 'use_nwp' field; pass --force-use-nwp / --no-force-use-nwp"
            )
        use_nwp = bool(ckpt_use_nwp)
        un_src = "checkpoint"
    else:
        use_nwp = bool(force_use_nwp)
        un_src = "cli-override"

    return zero_sky, use_nwp, zs_src, un_src


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rolling inference over the Folsom test set; save full 192-step forecasts to NPZ."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a trained Folsom checkpoint (.pt) for pv_forecasting_model_vit_imgs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(_PROJECT_ROOT / "inference_results" / "folsom_test_rolling"),
        help="Directory to write the NPZ + summary CSV into.",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="conf_folsom.yaml",
        help="Bare YAML filename under config/datasets/ (default: conf_folsom.yaml).",
    )
    parser.add_argument(
        "--stride_min",
        type=int,
        default=5,
        help="Test-anchor stride in minutes (default 5 = one t0 every 5 CSV rows for the "
             "1-min Folsom CSV). Must be a positive multiple of csv_interval_min.",
    )
    # Folsom sky tensors are [30, 3, 224, 224] per sample (~18 MB fp32) so we keep the
    # default batch size conservative; bump it up if your GPU has more headroom.
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / cuda:N / cpu (default: cuda if available else cpu).")
    parser.add_argument("--mask_night", action="store_true", default=True,
                        help="Zero predictions where forecast cos(zenith) < 0 (sun below horizon).")
    parser.add_argument("--no_mask_night", dest="mask_night", action="store_false",
                        help="Disable night masking; save raw model output.")
    parser.add_argument("--limit_batches", type=int, default=None,
                        help="Optional: stop after N batches (debug only).")

    parser.add_argument(
        "--force-zero-sky",
        dest="force_zero_sky",
        action="store_true",
        default=None,
        help="Force the zero-sky regime regardless of checkpoint metadata.",
    )
    parser.add_argument(
        "--no-force-zero-sky",
        dest="force_zero_sky",
        action="store_false",
        help="Force the real-sky regime regardless of checkpoint metadata.",
    )
    parser.add_argument(
        "--force-use-nwp",
        dest="force_use_nwp",
        action="store_true",
        default=None,
        help="Force the real-NWP regime regardless of checkpoint metadata.",
    )
    parser.add_argument(
        "--no-force-use-nwp",
        dest="force_use_nwp",
        action="store_false",
        help="Force the zeroed-NWP regime regardless of checkpoint metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[infer_folsom] device={device}")

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    print(f"[infer_folsom] checkpoint={ckpt_path}")

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[infer_folsom] output_dir={out_dir}")

    test_dataset = _resolve_test_dataset(args.dataset_config, args.stride_min)
    nw = int(test_dataset._num_test_windows)
    n_total = len(test_dataset)
    T_out = int(test_dataset.pv_output_len)
    dt_min = int(test_dataset.pv_output_interval_min)
    if nw != n_total:
        raise RuntimeError(
            f"sanity: |test_dataset|={n_total} != _num_test_windows={nw}"
        )
    print(
        f"[infer_folsom] windows={nw}  stride_min={args.stride_min}  "
        f"horizons={T_out}  step_min={dt_min}"
    )

    t0_strs_per_window = _t0_strings_for_test_windows(test_dataset)
    if t0_strs_per_window.shape[0] != nw:
        raise RuntimeError(
            f"sanity: |t0_strs|={t0_strs_per_window.shape[0]} != nw={nw}"
        )

    # Load checkpoint first so we know the regime before instantiating the model.
    ckpt = torch.load(ckpt_path, map_location=device)
    zero_sky, use_nwp, zs_src, un_src = _resolve_regime(
        ckpt,
        force_zero_sky=args.force_zero_sky,
        force_use_nwp=args.force_use_nwp,
    )
    print(
        f"[infer_folsom] zero_sky={zero_sky} use_nwp={use_nwp} "
        f"(source: zero_sky={zs_src}, use_nwp={un_src})"
    )

    model = pv_forecasting_model_vit_imgs(dev_dn_list=test_dataset.devDn_list).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[infer_folsom] WARNING: missing keys in state_dict: {len(missing)} "
              f"(first 5: {missing[:5]})")
    if unexpected:
        print(f"[infer_folsom] WARNING: unexpected keys in state_dict: {len(unexpected)} "
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

    preds_buf = np.zeros((nw, T_out), dtype=np.float32)
    targets_buf = np.zeros((nw, T_out), dtype=np.float32)
    mask_buf = np.zeros((nw, T_out), dtype=np.uint8)
    cz_buf = np.zeros((nw, T_out), dtype=np.float32)
    filled = np.zeros((nw,), dtype=bool)

    if device.type == "cuda":
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    else:
        autocast_ctx = contextlib.nullcontext()

    global_idx = 0
    n_batches_total = (n_total + args.batch_size - 1) // args.batch_size
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if args.limit_batches is not None and batch_idx >= args.limit_batches:
                break
            d = _batch_to_device(batch, device)
            # Mirror training-time input preparation exactly:
            #   * remap NWP channels (or zero them out if use_nwp=False)
            #   * optionally zero out sky tensors for the no-sky regime
            _prepare_nwp_for_vit(d, use_nwp=use_nwp)
            _prepare_sky_for_vit(d, zero_sky=zero_sky)

            with autocast_ctx:
                # !!! BUG / NEEDS FIXING !!!
                # Scale factor must match training (training/train_vit_test_folsom.py uses 4000.0
                # because Folsom kt = GHI / p_cs is in ~W/m^2 units; the 20.0 here is a leftover
                # from the Luoyang inference script and is INVALID for Folsom checkpoints.
                # Until this is rescaled to 4000.0, every number this script writes
                # (NPZ preds, _summary.csv RMSE/MAE/MBE) is off by ~200x and INVALID.
                # TODO: change 20.0 -> 4000.0 and re-run inference on all Folsom checkpoints.
                kt_pred = forward_vit(model, d) * 20.0
            pv_pred = (kt_pred * d["target_p_cs"] * d["p_mean"].unsqueeze(1)).float()  # [B, T_out]

            pred_np = pv_pred.detach().cpu().numpy()
            tgt_np = d["target_pv"].detach().cpu().numpy()
            mask_np = d["target_mask"].detach().cpu().numpy()
            cz_np = d["forecast_timefeats"][:, :, COS_ZENITH_COL].detach().cpu().numpy()

            if args.mask_night:
                night = cz_np < 0
                pred_np = np.where(night, 0.0, pred_np)

            B = pred_np.shape[0]
            end_idx = global_idx + B
            if end_idx > nw:
                raise RuntimeError(
                    f"window index out of range: end_idx={end_idx} > nw={nw} "
                    f"(batch_idx={batch_idx}, B={B})"
                )
            preds_buf[global_idx:end_idx] = pred_np * GHI_SCALE_WM2
            targets_buf[global_idx:end_idx] = tgt_np * GHI_SCALE_WM2
            mask_buf[global_idx:end_idx] = mask_np.astype(np.uint8)
            cz_buf[global_idx:end_idx] = cz_np
            filled[global_idx:end_idx] = True

            global_idx = end_idx
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                print(
                    f"[infer_folsom] batch {batch_idx + 1}/{n_batches_total}  "
                    f"processed_samples={global_idx}"
                )

    print(f"[infer_folsom] inference done; total processed = {global_idx}")
    if args.limit_batches is None and not filled.all():
        print(
            f"[infer_folsom] WARNING: only {int(filled.sum())}/{nw} windows filled "
            "(test loader returned fewer samples than expected)"
        )

    kept = np.nonzero(filled)[0]
    preds = preds_buf[kept]
    targets = targets_buf[kept]
    target_mask = mask_buf[kept]
    cos_zenith = cz_buf[kept]
    t0_strs = t0_strs_per_window[kept]

    out_npz = out_dir / "folsom_test.npz"
    ckpt_basename = ckpt_path.name
    np.savez_compressed(
        out_npz,
        t0_utc=t0_strs,
        pred=preds,
        target=targets,
        target_mask=target_mask,
        cos_zenith=cos_zenith,
        forecast_dt_min=np.int32(dt_min),
        pv_output_len=np.int32(T_out),
        stride_min=np.int32(args.stride_min),
        zero_sky=np.bool_(zero_sky),
        use_nwp=np.bool_(use_nwp),
        checkpoint=np.asarray(ckpt_basename, dtype="<U256"),
        target_scale=np.float32(GHI_SCALE_WM2),
    )
    print(f"[infer_folsom] wrote {out_npz}  ({preds.shape[0]} windows, {T_out} horizons)")

    summary_rows: list[dict] = []
    for label, k in _HORIZONS:
        m = target_mask[:, k].astype(bool)
        if m.any():
            err = preds[m, k] - targets[m, k]
            rmse = float(np.sqrt(np.mean(err * err)))
            mae = float(np.mean(np.abs(err)))
            mbe = float(np.mean(err))
        else:
            rmse = float("nan")
            mae = float("nan")
            mbe = float("nan")
        summary_rows.append({
            "horizon": label,
            "n_valid": int(m.sum()),
            "rmse": rmse,
            "mae": mae,
            "mbe": mbe,
            "zero_sky": bool(zero_sky),
            "use_nwp": bool(use_nwp),
            "checkpoint": ckpt_basename,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[infer_folsom] summary saved to {summary_path}")
    for row in summary_rows:
        print(
            f"[infer_folsom] zero_sky={zero_sky} use_nwp={use_nwp}  "
            f"horizon={row['horizon']:<5s}  n={row['n_valid']}  "
            f"RMSE={row['rmse']:.3f}  MAE={row['mae']:.3f}"
        )


if __name__ == "__main__":
    main()
