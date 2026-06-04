"""
Save per-window predictions / targets / clear-sky GHI for a single Folsom PV ViT
checkpoint so downstream analyses (clear-sky-index regime slicing, error
calibration, etc.) can be done without re-running inference.

This is a strict superset of what ``scripts/eval_per_horizon.py`` does: same
test-loader semantics, same night-zero rule on predictions, same Huber loss.
The only differences are:

  * a custom collate that also stacks per-sample ``input_timestamps_utc`` and
    ``forecast_timestamps_utc`` lists (the default ``collate_batched`` drops
    them);
  * we accumulate per-window arrays (``pred``, ``target``, ``mask``, ``p_cs``,
    ``input_ghi``, ``input_p_cs``, timestamps) and dump them as a single NPZ.

After writing, we recompute the aggregate RMSE / MAE from the saved arrays and
sanity-check against the per-run metrics file next to the checkpoint. If the
mismatch is above tolerance and ``--strict-sanity`` is on (default), the script
exits non-zero -- exactly the same contract as ``eval_per_horizon.py``.

Clear-sky GHI source (reused, not recomputed):
  * ``batch["p_cs"]`` and ``batch["target_p_cs"]`` come from
    ``dataloader.folsom._compute_folsom_p_cs`` (pvlib's ``ineichen`` clear-sky
    model with lat/lon from ``<data_dir>/info.yaml``; pvlib ships the Linke
    turbidity table locally so this is an offline call).
  * Both are normalised by ``_FOLSOM_GHI_SCALE = 1000`` and clipped to
    ``[0, 1.2]``. To recover clear-sky GHI in W/m^2 multiply by 1000.

The checkpoint loader is identical to ``eval_per_horizon.py`` and accepts the
saver format used by ``training/train_vit_test_folsom.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dataloader.folsom import (  # noqa: E402
    _FOLSOM_GHI_SCALE,
    _FOLSOM_HUBER_DELTA,
    _FOLSOM_KT_INPUT_SCALE,
    FolsomIrradianceDataset,
)
from dataloader.luoyang_zarr import collate_batched  # noqa: E402
from models.models import pv_forecasting_model_vit_imgs  # noqa: E402
from training.train_vit_test_folsom import (  # noqa: E402
    _LOSS_METRIC_HORIZON,
    _batch_to_device,
    _dataset_kwargs,
    _format_nwp_features_for_log,
    _prepare_nwp_for_vit,
    _prepare_sky_for_vit,
    forward_vit,
    resolve_nwp_features_from_ckpt,
)

_OUTPUT_INTERVAL_MIN = 15

# Same tolerances as scripts/eval_per_horizon.py.
_SANITY_ABS_TOL = 5e-2  # W/m^2
_SANITY_REL_TOL = 5e-4  # 0.05%


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Save per-window predictions / targets / clear-sky GHI for a Folsom "
            "PV ViT checkpoint, with the same eval semantics as "
            "scripts/eval_per_horizon.py."
        )
    )
    p.add_argument("--ckpt", required=True, type=Path,
                   help="Path to a Folsom PV ViT checkpoint (.pt).")
    p.add_argument("--zero-sky", action="store_true",
                   help="Zero sky tensors (required for ghi_only_* checkpoints; "
                        "do not pass for ghi_sky_* checkpoints).")
    p.add_argument("--use-nwp", action="store_true",
                   help="Feed remapped NWP (default off; archived runs were trained without).")
    p.add_argument("--dataset-config", type=str, default="conf_folsom.yaml",
                   help="Dataset YAML filename under config/datasets/.")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Eval batch size. 64 matches the existing per-horizon eval.")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers. Default 0 -- test set is ~300 windows "
                        "so multiprocessing overhead is not worth it.")
    p.add_argument("--output", required=True, type=Path,
                   help="Where to write the per-window NPZ.")
    p.add_argument("--sanity-output", type=Path, default=None,
                   help="Optional JSON path for the aggregate sanity-check result. "
                        "Defaults to <output>.sanity.json.")
    p.add_argument("--max_batches", type=int, default=None,
                   help="Optional cap on test batches (smoke test only).")
    p.add_argument("--strict-sanity", dest="strict_sanity", action="store_true",
                   help="(default) Abort with non-zero exit if aggregate RMSE/MAE "
                        "do not match the per-run metrics file.")
    p.add_argument("--no-strict-sanity", dest="strict_sanity", action="store_false",
                   help="Allow mismatch but still write outputs.")
    p.set_defaults(strict_sanity=True)
    return p.parse_args()


def _load_checkpoint(ckpt_path: Path, device: torch.device) -> dict:
    obj = torch.load(ckpt_path, map_location=device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj
    if isinstance(obj, dict):
        return {"model_state_dict": obj, "dev_dn_list": None, "ema": None}
    raise ValueError(
        f"Unrecognized checkpoint object at {ckpt_path}: type={type(obj).__name__}."
    )


def _collate_with_timestamps(batch: list[dict]) -> dict:
    """Wrap ``collate_batched`` to also keep per-sample timestamp lists."""
    out = collate_batched(batch)
    out["input_timestamps_utc"] = [s["input_timestamps_utc"] for s in batch]
    out["forecast_timestamps_utc"] = [s["forecast_timestamps_utc"] for s in batch]
    return out


def _build_test_loader(dataset_config: str, batch_size: int, num_workers: int) -> DataLoader:
    test_dataset = FolsomIrradianceDataset(**_dataset_kwargs(dataset_config, "test"))
    pin = torch.cuda.is_available()
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_with_timestamps,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )


def _read_sanity_metrics(ckpt_path: Path) -> tuple[float, float, float] | None:
    """Mirror of ``scripts/eval_per_horizon._read_sanity_metrics``."""
    name = ckpt_path.name
    suffix: str | None = None
    for prefix in (
        "folsom_pv_forecast_vit_best_",
        "folsom_pv_forecast_vit_final_",
    ):
        if name.startswith(prefix) and name.endswith(".pt"):
            suffix = name[len(prefix):-len(".pt")]
            break
    if suffix is None and name.startswith("folsom_pv_forecast_vit_epoch_") and name.endswith(".pt"):
        stem = name[len("folsom_pv_forecast_vit_epoch_"):-len(".pt")]
        _, _, suffix = stem.partition("_")
    if not suffix:
        return None
    metrics_path = ckpt_path.parent / f"folsom_pv_forecast_metrics_{suffix}.txt"
    if not metrics_path.is_file():
        return None
    with open(metrics_path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        return None
    parts = lines[-1].split("\t")
    if len(parts) != 3:
        return None
    return float(parts[0]), float(parts[1]), float(parts[2])


def _run_inference(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    *,
    use_nwp: bool,
    zero_sky: bool,
    max_batches: int | None,
    h_steps: int,
) -> dict:
    """Run inference once and return per-window arrays + the same aggregate scalars
    the trainer's ``evaluate`` would report."""
    model.eval()

    pred_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    mask_chunks: list[np.ndarray] = []
    target_p_cs_chunks: list[np.ndarray] = []
    cos_zenith_chunks: list[np.ndarray] = []

    # Input window (used for input-CSI binning). Shapes are dataset-config-dependent
    # (pv_input_len = 576 for conf_folsom.yaml), so we figure out T_in from the first
    # batch and reject any later batch that disagrees -- catches a silent config drift.
    input_ghi_chunks: list[np.ndarray] = []
    input_p_cs_chunks: list[np.ndarray] = []

    input_ts_chunks: list[list[list[str]]] = []
    forecast_ts_chunks: list[list[list[str]]] = []

    huber_per_elem = nn.HuberLoss(delta=_FOLSOM_HUBER_DELTA, reduction="none")
    sum_abs = 0.0
    sum_sq = 0.0
    n_elem = 0.0
    total_loss_sum = 0.0
    n_batches = 0
    t_in_expected: int | None = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            input_ts = batch.pop("input_timestamps_utc")
            forecast_ts = batch.pop("forecast_timestamps_utc")

            d = _batch_to_device(batch, device)
            _prepare_nwp_for_vit(d, use_nwp=use_nwp)
            _prepare_sky_for_vit(d, zero_sky=zero_sky)

            kt_pred = forward_vit(model, d) * _FOLSOM_KT_INPUT_SCALE
            pv_pred = kt_pred * d["target_p_cs"] * d["p_mean"].unsqueeze(1)
            t_out = int(pv_pred.shape[1])
            h = min(h_steps, t_out)
            assert h == h_steps, (
                f"Model output length {t_out} < requested horizon {h_steps}; "
                "dataset config and model must match."
            )

            tgt = d["target_pv"][:, :h]
            m = d["target_mask"][:, :h]

            # ``evaluate``-equivalent aggregate loss for the sanity check.
            criterion_loss = (huber_per_elem(pv_pred[:, :h] * m, tgt * m).mean()).item()
            total_loss_sum += float(criterion_loss)

            # Night-zero the predictions exactly like ``evaluate`` before residuals.
            pred_h = pv_pred[:, :h].clone()
            cos_zen_h = d["forecast_timefeats"][:, :h, 3]
            night = cos_zen_h < 0
            pred_h[night] = 0.0
            diff = pred_h - tgt
            sum_abs += float((diff.abs() * m).sum().item())
            sum_sq += float(((diff ** 2) * m).sum().item())
            n_elem += float(m.sum().item())
            n_batches += 1

            pred_chunks.append(pred_h.detach().to("cpu", torch.float32).numpy())
            target_chunks.append(tgt.detach().to("cpu", torch.float32).numpy())
            mask_chunks.append(m.detach().to("cpu", torch.float32).numpy())
            target_p_cs_chunks.append(
                d["target_p_cs"][:, :h].detach().to("cpu", torch.float32).numpy()
            )
            cos_zenith_chunks.append(cos_zen_h.detach().to("cpu", torch.float32).numpy())

            # Input window: ``pv`` is [B, 1, T_in] -- squeeze the sensor dim. ``p_cs``
            # ditto. These are the dataset-level GHI / normalised clear-sky for the
            # *input* window we will use for input-CSI binning.
            pv_in = d["pv"].squeeze(1).detach().to("cpu", torch.float32).numpy()
            p_cs_in = batch["p_cs"].squeeze(1).detach().to("cpu", torch.float32).numpy()
            if t_in_expected is None:
                t_in_expected = int(pv_in.shape[1])
            else:
                if int(pv_in.shape[1]) != t_in_expected:
                    raise RuntimeError(
                        f"Input window length drift: batch {batch_idx} has T_in="
                        f"{int(pv_in.shape[1])} but expected {t_in_expected}"
                    )
            input_ghi_chunks.append(pv_in)
            input_p_cs_chunks.append(p_cs_in)

            input_ts_chunks.append(input_ts)
            forecast_ts_chunks.append(forecast_ts)

    if n_batches == 0:
        raise RuntimeError("No batches produced -- empty test loader?")

    pred = np.concatenate(pred_chunks, axis=0)
    target = np.concatenate(target_chunks, axis=0)
    mask = np.concatenate(mask_chunks, axis=0)
    target_p_cs = np.concatenate(target_p_cs_chunks, axis=0)
    cos_zenith = np.concatenate(cos_zenith_chunks, axis=0)
    input_ghi = np.concatenate(input_ghi_chunks, axis=0)
    input_p_cs = np.concatenate(input_p_cs_chunks, axis=0)
    flat_input_ts = [tlist for chunk in input_ts_chunks for tlist in chunk]
    flat_forecast_ts = [tlist for chunk in forecast_ts_chunks for tlist in chunk]

    agg_n = max(n_elem, 1.0)
    agg_rmse = math.sqrt(sum_sq / agg_n)
    agg_mae = sum_abs / agg_n
    mean_loss = total_loss_sum / max(n_batches, 1)

    # The "anchor timestamp" is the last input timestamp (the model's t0). The
    # forecast_timestamps are t0 + i * 15 min for i=1..16.
    anchor_ts = np.array([ts[-1] for ts in flat_input_ts])

    return {
        "pred": pred,
        "target": target,
        "mask": mask,
        "target_p_cs": target_p_cs,
        "cos_zenith": cos_zenith,
        "input_ghi": input_ghi,
        "input_p_cs": input_p_cs,
        "anchor_ts": anchor_ts,
        "forecast_ts": np.array(flat_forecast_ts),
        "input_ts": np.array(flat_input_ts),
        "agg_rmse": agg_rmse,
        "agg_mae": agg_mae,
        "agg_n_valid": int(n_elem),
        "agg_loss": mean_loss,
        "n_batches": n_batches,
    }


def main() -> int:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sanity_path = args.sanity_output or args.output.with_suffix(args.output.suffix + ".sanity.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval-save] device={device} ckpt={args.ckpt}")
    print(f"[eval-save] zero_sky={args.zero_sky} use_nwp={args.use_nwp} dataset={args.dataset_config}")

    ckpt = _load_checkpoint(args.ckpt, device)
    dev_dn_list = ckpt.get("dev_dn_list")
    ckpt_epoch = ckpt.get("epoch")
    ckpt_zero_sky = ckpt.get("zero_sky")
    ckpt_use_nwp = ckpt.get("use_nwp")
    if dev_dn_list is None:
        raise RuntimeError(
            f"Checkpoint {args.ckpt} does not contain 'dev_dn_list'; model factory needs it."
        )

    nwp_features, nwp_use_invalid_mask = resolve_nwp_features_from_ckpt(ckpt)
    print(
        f"[eval-save] nwp_features={_format_nwp_features_for_log(nwp_features, nwp_use_invalid_mask)} "
        f"(source: {'checkpoint' if 'nwp_features' in ckpt else 'legacy default'})"
    )
    model = pv_forecasting_model_vit_imgs(
        dev_dn_list=dev_dn_list,
        nwp_features=nwp_features,
        use_invalid_mask=nwp_use_invalid_mask,
    ).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing or unexpected:
        print(f"[eval-save][warn] load_state_dict mismatch: "
              f"missing={len(missing)} unexpected={len(unexpected)}")

    if ckpt_zero_sky is not None and bool(ckpt_zero_sky) != bool(args.zero_sky):
        print(f"[eval-save][warn] --zero-sky={args.zero_sky} but ckpt zero_sky={ckpt_zero_sky}.")
    if ckpt_use_nwp is not None and bool(ckpt_use_nwp) != bool(args.use_nwp):
        print(f"[eval-save][warn] --use-nwp={args.use_nwp} but ckpt use_nwp={ckpt_use_nwp}.")

    loader = _build_test_loader(args.dataset_config, args.batch_size, args.num_workers)
    print(f"[eval-save] test batches: {len(loader)} (batch_size={args.batch_size})")

    results = _run_inference(
        model,
        device,
        loader,
        use_nwp=bool(args.use_nwp),
        zero_sky=bool(args.zero_sky),
        max_batches=args.max_batches,
        h_steps=_LOSS_METRIC_HORIZON,
    )

    expected = _read_sanity_metrics(args.ckpt)
    sanity: dict = {
        "expected": None,
        "computed": {"loss": results["agg_loss"], "rmse": results["agg_rmse"], "mae": results["agg_mae"]},
        "rmse_abs_err": None, "mae_abs_err": None, "loss_abs_err": None,
        "passed": None, "tol_abs": _SANITY_ABS_TOL, "tol_rel": _SANITY_REL_TOL,
    }
    if expected is None:
        print("[eval-save][warn] no folsom_pv_forecast_metrics_*.txt next to ckpt; "
              "skipping aggregate sanity check.")
    else:
        exp_loss, exp_rmse, exp_mae = expected
        rmse_err = abs(results["agg_rmse"] - exp_rmse)
        mae_err = abs(results["agg_mae"] - exp_mae)
        loss_err = abs(results["agg_loss"] - exp_loss)
        rmse_ok = rmse_err <= max(_SANITY_ABS_TOL, _SANITY_REL_TOL * exp_rmse)
        mae_ok = mae_err <= max(_SANITY_ABS_TOL, _SANITY_REL_TOL * exp_mae)
        loss_ok = loss_err <= max(1.0, _SANITY_REL_TOL * 10 * exp_loss)
        passed = bool(rmse_ok and mae_ok)
        sanity.update(
            expected={"loss": exp_loss, "rmse": exp_rmse, "mae": exp_mae},
            rmse_abs_err=rmse_err, mae_abs_err=mae_err, loss_abs_err=loss_err,
            loss_ok=loss_ok, rmse_ok=rmse_ok, mae_ok=mae_ok, passed=passed,
        )
        status = "PASS" if passed else "FAIL"
        print(
            f"[eval-save][sanity {status}] expected RMSE={exp_rmse:.6f} MAE={exp_mae:.6f} "
            f"loss={exp_loss:.6f}  |  computed RMSE={results['agg_rmse']:.6f} "
            f"MAE={results['agg_mae']:.6f} loss={results['agg_loss']:.6f}  |  "
            f"err RMSE={rmse_err:.3e} MAE={mae_err:.3e} loss={loss_err:.3e}"
        )

    payload_meta = {
        "ckpt": str(args.ckpt),
        "ckpt_epoch": ckpt_epoch,
        "ckpt_recorded_zero_sky": ckpt_zero_sky,
        "ckpt_recorded_use_nwp": ckpt_use_nwp,
        "eval_zero_sky": bool(args.zero_sky),
        "eval_use_nwp": bool(args.use_nwp),
        "dataset_config": args.dataset_config,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_batches": args.max_batches,
        "output_interval_min": _OUTPUT_INTERVAL_MIN,
        "horizon_steps": _LOSS_METRIC_HORIZON,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "folsom_ghi_scale": float(_FOLSOM_GHI_SCALE),
        "aggregate": {
            "rmse": results["agg_rmse"],
            "mae": results["agg_mae"],
            "loss": results["agg_loss"],
            "n_valid": results["agg_n_valid"],
            "n_batches": results["n_batches"],
        },
        "sanity_check": sanity,
        "n_windows": int(results["pred"].shape[0]),
        "input_window_len": int(results["input_ghi"].shape[1]),
        "arrays": {
            "pred": "[N, 16] W/m^2 -- night-zeroed PV/GHI prediction (model output)",
            "target": "[N, 16] W/m^2 -- target_pv == target GHI for Folsom",
            "mask": "[N, 16] -- target_mask, 0/1",
            "target_p_cs": "[N, 16] -- pvlib ineichen clearsky GHI / 1000, clipped [0, 1.2]",
            "cos_zenith": "[N, 16] -- cos(solar zenith) for each forecast step",
            "input_ghi": "[N, T_in] W/m^2 -- input-window raw GHI (pv)",
            "input_p_cs": "[N, T_in] -- input-window normalised clearsky GHI / 1000",
            "anchor_ts": "[N] -- ISO UTC string of t0 (last input timestamp)",
            "forecast_ts": "[N, 16] -- ISO UTC strings of forecast steps",
            "input_ts": "[N, T_in] -- ISO UTC strings of input window",
        },
    }

    if not (sanity["passed"] or sanity["passed"] is None) and args.strict_sanity:
        sanity_path.write_text(json.dumps({**payload_meta, "abort_reason": "sanity_failed"}, indent=2))
        print(f"[eval-save] wrote sanity report {sanity_path} (aborting before NPZ).")
        print("[eval-save] aborting due to sanity-check mismatch (use --no-strict-sanity to override).")
        return 2

    np.savez_compressed(
        args.output,
        pred=results["pred"].astype(np.float32),
        target=results["target"].astype(np.float32),
        mask=results["mask"].astype(np.float32),
        target_p_cs=results["target_p_cs"].astype(np.float32),
        cos_zenith=results["cos_zenith"].astype(np.float32),
        input_ghi=results["input_ghi"].astype(np.float32),
        input_p_cs=results["input_p_cs"].astype(np.float32),
        anchor_ts=results["anchor_ts"],
        forecast_ts=results["forecast_ts"],
        input_ts=results["input_ts"],
        meta_json=np.array(json.dumps(payload_meta), dtype=object),
    )
    print(f"[eval-save] wrote {args.output}  (N={results['pred'].shape[0]} windows)")

    sanity_path.write_text(json.dumps(payload_meta, indent=2))
    print(f"[eval-save] wrote sanity report {sanity_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
