"""
Per-output-step (per-horizon) evaluation of a single Folsom PV ViT checkpoint.

Replays exactly the test-set evaluation that the training script does at the end of a run
(see ``training/train_vit_test_folsom.py::evaluate`` and the post-training test block around
lines 847-888), but instead of collapsing the first 16 forecast steps into one scalar
RMSE/MAE, we accumulate residuals per output-step so the user can plot RMSE vs horizon.

Reuses the training script's helpers directly to avoid drift: ``_dataset_kwargs``,
``_batch_to_device``, ``_prepare_nwp_for_vit``, ``_prepare_sky_for_vit``, ``forward_vit``,
and ``_LOSS_METRIC_HORIZON``. We also re-derive the night-mask and W/m^2 conversion
identically.

Sanity check: after running, we collapse the per-step accumulators back to a single
RMSE/MAE over the first ``_LOSS_METRIC_HORIZON`` steps and compare against the matching
``folsom_pv_forecast_metrics_gpu{N}.txt`` next to the checkpoint. If the numbers don't
match within tolerance we abort with a non-zero exit code so an outer driver can stop
instead of writing bad per-horizon numbers.

The checkpoint format follows the training script's saver:
    {"model_state_dict": ..., "dev_dn_list": [...], "ema": bool,
     "zero_sky": bool, "use_nwp": bool, "dataset_config": str, ...}
``model_state_dict`` already holds the right weights for the test-eval (the trainer
picks EMA shadow vs raw weights at *save* time; see lines 810-827 of the training
script), so we just load it straight in. No EMA toggling is needed here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dataloader.folsom import FolsomIrradianceDataset  # noqa: E402
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

# Folsom output cadence is 15 min/step (see ``config/datasets/conf_folsom.yaml``:
# ``sampling.pv_output_interval_min: 15``). Step 1 = 15 min ahead, step 16 = 240 min = 4 h.
_OUTPUT_INTERVAL_MIN = 15

# Float tolerance for the aggregate sanity check vs the per-run metrics file.
# The metrics file is written with ``"{:.8f}"`` formatting (lines 884-886 of the
# training script) and we use float64 accumulators here, so an absolute tolerance
# in the 1e-3 W/m^2 range is comfortable; we also include a small relative slack
# in case CUDA non-determinism shifts the last few digits when we replay forward.
_SANITY_ABS_TOL = 5e-2  # W/m^2
_SANITY_REL_TOL = 5e-4  # 0.05%


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Per-output-step RMSE / MAE / Huber-loss for a single Folsom PV ViT checkpoint, "
            "matching the test-eval semantics of training/train_vit_test_folsom.py."
        )
    )
    p.add_argument("--ckpt", required=True, type=Path,
                   help="Path to a Folsom PV ViT checkpoint (.pt). Expects the saver "
                        "format used by train_vit_test_folsom.py.")
    p.add_argument("--zero-sky", action="store_true",
                   help="Replace sky tensors with zeros after each batch is on device "
                        "(mirror --zero-sky from training). REQUIRED for ghi_only_* "
                        "checkpoints; do NOT pass for ghi_sky_* checkpoints.")
    p.add_argument("--use-nwp", action="store_true",
                   help="Feed real (remapped) NWP. Default OFF -- NWP tensor is zeroed, "
                        "matching the archived runs which did not pass --use-nwp.")
    p.add_argument("--dataset-config", type=str, default="conf_folsom.yaml",
                   help="Dataset YAML filename under config/datasets/. Default conf_folsom.yaml.")
    p.add_argument("--batch_size", type=int, default=2,
                   help="Eval batch size. Default 2 to match the trainer's per-GPU default.")
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader workers. Default 4.")
    p.add_argument("--output", required=True, type=Path,
                   help="Where to write the per-horizon JSON report.")
    p.add_argument("--max_batches", type=int, default=None,
                   help="Optional cap on test batches (for smoke testing only; the "
                        "sanity check will fail if you cap and the metrics file was "
                        "written on the full split).")
    p.add_argument("--strict-sanity", dest="strict_sanity", action="store_true",
                   help="(default) Abort with non-zero exit if the aggregate RMSE/MAE "
                        "do not match the per-run metrics file within tolerance.")
    p.add_argument("--no-strict-sanity", dest="strict_sanity", action="store_false",
                   help="Allow mismatch but still record the result in the JSON.")
    p.set_defaults(strict_sanity=True)
    return p.parse_args()


def _load_checkpoint(ckpt_path: Path, device: torch.device) -> dict:
    obj = torch.load(ckpt_path, map_location=device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj
    # Raw state-dict fallback. Wrap in the same shape the rest of the script expects.
    if isinstance(obj, dict):
        return {"model_state_dict": obj, "dev_dn_list": None, "ema": None}
    raise ValueError(
        f"Unrecognized checkpoint object at {ckpt_path}: type={type(obj).__name__}. "
        "Expected dict with 'model_state_dict' or a raw state-dict."
    )


def _build_test_loader(dataset_config: str, batch_size: int, num_workers: int) -> DataLoader:
    test_dataset = FolsomIrradianceDataset(**_dataset_kwargs(dataset_config, "test"))
    pin = torch.cuda.is_available()
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batched,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )


def _read_sanity_metrics(ckpt_path: Path) -> tuple[float, float, float] | None:
    """Read the matching ``folsom_pv_forecast_metrics_gpu{N}.txt`` next to the checkpoint.

    Returns ``(test_loss, rmse, mae)`` for the LAST line of the metrics file
    (the trainer appends one line per run; in the archive there's exactly one).
    Returns None if no metrics file is found.
    """
    # Checkpoint filename pattern (training script lines 706, 829):
    #   folsom_pv_forecast_vit_best_gpu{N}.pt  /  ..._final_gpu{N}.pt  /  ..._epoch_{E}_gpu{N}.pt
    # The metrics file shares the gpu suffix:
    #   folsom_pv_forecast_metrics_gpu{N}.txt   (line 883)
    name = ckpt_path.name
    # Try to peel off the well-known prefixes to recover the suffix.
    suffix = None
    for prefix in (
        "folsom_pv_forecast_vit_best_",
        "folsom_pv_forecast_vit_final_",
    ):
        if name.startswith(prefix) and name.endswith(".pt"):
            suffix = name[len(prefix):-len(".pt")]
            break
    if suffix is None and name.startswith("folsom_pv_forecast_vit_epoch_") and name.endswith(".pt"):
        # epoch checkpoints embed the epoch number, e.g. epoch_10_gpu0
        stem = name[len("folsom_pv_forecast_vit_epoch_"):-len(".pt")]
        # Drop leading "<epoch>_" -> suffix is everything after the first underscore.
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


def _per_step_eval(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    criterion: nn.Module,
    *,
    use_nwp: bool,
    zero_sky: bool,
    max_batches: int | None,
    h_steps: int,
) -> dict:
    """Replay ``evaluate(...)`` but accumulate per-output-step.

    Returns a dict with per-step arrays (length ``h_steps``) for ``sum_abs``,
    ``sum_sq``, ``n_elem``, plus ``per_step_loss`` (mean Huber per batch then
    averaged over batches, same reduction shape as the training criterion),
    plus running stats for prediction / target means.
    """
    model.eval()
    sum_abs = torch.zeros(h_steps, dtype=torch.float64)
    sum_sq = torch.zeros(h_steps, dtype=torch.float64)
    sum_pred = torch.zeros(h_steps, dtype=torch.float64)
    sum_tgt = torch.zeros(h_steps, dtype=torch.float64)
    n_elem = torch.zeros(h_steps, dtype=torch.float64)
    per_step_loss_sum = torch.zeros(h_steps, dtype=torch.float64)
    total_loss_sum = 0.0
    n_batches = 0

    huber_per_elem = nn.HuberLoss(delta=30.0, reduction="none")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            d = _batch_to_device(batch, device)
            _prepare_nwp_for_vit(d, use_nwp=use_nwp)
            _prepare_sky_for_vit(d, zero_sky=zero_sky)

            kt_pred = forward_vit(model, d) * 4000.0
            pv_pred = kt_pred * d["target_p_cs"] * d["p_mean"].unsqueeze(1)
            t_out = int(pv_pred.shape[1])
            h = min(h_steps, t_out)
            assert h == h_steps, (
                f"Model output length {t_out} < requested horizon {h_steps}; "
                "dataset config and model would need to match."
            )

            m = d["target_mask"][:, :h]
            tgt = d["target_pv"][:, :h]

            # Aggregate-loss line: mirror training/evaluate() exactly so the sanity
            # check below can recover the same scalar that landed in the metrics file.
            loss_agg = criterion((pv_pred[:, :h] * m), (tgt * m))
            total_loss_sum += float(loss_agg.item())

            # Per-step Huber (mean over batch dim only, like the criterion does
            # internally except we keep the step axis). We use the masked tensors
            # so this is comparable to the aggregate loss (same zero-out trick).
            per_elem = huber_per_elem(pv_pred[:, :h] * m, tgt * m)  # [B, h]
            per_step_loss_sum += per_elem.mean(dim=0).detach().to("cpu", torch.float64)

            # RMSE / MAE accumulators -- night-zero predictions before residuals,
            # exactly like the training-time evaluate() does.
            pred_h = pv_pred[:, :h].clone()
            night = d["forecast_timefeats"][:, :h, 3] < 0
            pred_h[night] = 0.0
            diff = pred_h - tgt

            abs_masked = (diff.abs() * m).sum(dim=0).detach().to("cpu", torch.float64)
            sq_masked = ((diff ** 2) * m).sum(dim=0).detach().to("cpu", torch.float64)
            pred_masked = (pred_h * m).sum(dim=0).detach().to("cpu", torch.float64)
            tgt_masked = (tgt * m).sum(dim=0).detach().to("cpu", torch.float64)
            mask_sum = m.sum(dim=0).detach().to("cpu", torch.float64)

            sum_abs += abs_masked
            sum_sq += sq_masked
            sum_pred += pred_masked
            sum_tgt += tgt_masked
            n_elem += mask_sum
            n_batches += 1

    mean_loss = total_loss_sum / max(n_batches, 1)
    per_step_loss = (per_step_loss_sum / max(n_batches, 1)).tolist()

    rmse_per_step = [(sum_sq[s].item() / max(n_elem[s].item(), 1.0)) ** 0.5 for s in range(h_steps)]
    mae_per_step = [sum_abs[s].item() / max(n_elem[s].item(), 1.0) for s in range(h_steps)]
    pred_mean_per_step = [sum_pred[s].item() / max(n_elem[s].item(), 1.0) for s in range(h_steps)]
    tgt_mean_per_step = [sum_tgt[s].item() / max(n_elem[s].item(), 1.0) for s in range(h_steps)]
    n_per_step = [int(n_elem[s].item()) for s in range(h_steps)]

    # Aggregate over the same first-h_steps window: should reproduce evaluate()'s scalar.
    agg_n = float(n_elem.sum().item())
    agg_rmse = math.sqrt(float(sum_sq.sum().item()) / max(agg_n, 1.0))
    agg_mae = float(sum_abs.sum().item()) / max(agg_n, 1.0)

    return {
        "per_step_rmse": rmse_per_step,
        "per_step_mae": mae_per_step,
        "per_step_loss": per_step_loss,
        "per_step_n_valid": n_per_step,
        "per_step_pred_mean": pred_mean_per_step,
        "per_step_target_mean": tgt_mean_per_step,
        "aggregate_rmse": agg_rmse,
        "aggregate_mae": agg_mae,
        "aggregate_n_valid": int(agg_n),
        "aggregate_loss": mean_loss,
        "n_batches": n_batches,
    }


def main() -> int:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device} ckpt={args.ckpt}")
    print(f"[eval] zero_sky={args.zero_sky} use_nwp={args.use_nwp} dataset={args.dataset_config}")

    ckpt = _load_checkpoint(args.ckpt, device)
    dev_dn_list = ckpt.get("dev_dn_list")
    ckpt_epoch = ckpt.get("epoch")
    if dev_dn_list is None:
        raise RuntimeError(
            f"Checkpoint {args.ckpt} does not contain 'dev_dn_list'; the model factory needs it. "
            "Pass a checkpoint saved by training/train_vit_test_folsom.py."
        )

    nwp_features, nwp_use_invalid_mask = resolve_nwp_features_from_ckpt(ckpt)
    print(
        f"[eval] nwp_features={_format_nwp_features_for_log(nwp_features, nwp_use_invalid_mask)} "
        f"(source: {'checkpoint' if 'nwp_features' in ckpt else 'legacy default'})"
    )
    model = pv_forecasting_model_vit_imgs(
        dev_dn_list=dev_dn_list,
        nwp_features=nwp_features,
        use_invalid_mask=nwp_use_invalid_mask,
    ).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing or unexpected:
        # Helpful diagnostic if the saver format ever drifts.
        print(f"[eval][warn] load_state_dict mismatch: "
              f"missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"[eval][warn] first missing: {missing[:3]}")
        if unexpected:
            print(f"[eval][warn] first unexpected: {unexpected[:3]}")

    # Sanity: surface trainer-recorded flags so we notice if the user pointed --zero-sky
    # at the wrong arm of the experiment.
    ckpt_zero_sky = ckpt.get("zero_sky")
    ckpt_use_nwp = ckpt.get("use_nwp")
    if ckpt_zero_sky is not None and bool(ckpt_zero_sky) != bool(args.zero_sky):
        print(f"[eval][warn] --zero-sky={args.zero_sky} but checkpoint was trained with "
              f"zero_sky={ckpt_zero_sky}. This is intentional only if you are running a "
              "cross-arm evaluation.")
    if ckpt_use_nwp is not None and bool(ckpt_use_nwp) != bool(args.use_nwp):
        print(f"[eval][warn] --use-nwp={args.use_nwp} but checkpoint was trained with "
              f"use_nwp={ckpt_use_nwp}.")

    criterion = nn.HuberLoss(delta=30.0)  # Match trainer (line 662 of train_vit_test_folsom.py).

    loader = _build_test_loader(args.dataset_config, args.batch_size, args.num_workers)
    print(f"[eval] test batches: {len(loader)} (batch_size={args.batch_size})")

    results = _per_step_eval(
        model,
        device,
        loader,
        criterion,
        use_nwp=bool(args.use_nwp),
        zero_sky=bool(args.zero_sky),
        max_batches=args.max_batches,
        h_steps=_LOSS_METRIC_HORIZON,
    )

    # ---- Sanity check vs the per-run metrics file ----
    expected = _read_sanity_metrics(args.ckpt)
    sanity = {"expected_metrics_file": None, "expected": None, "computed": None,
              "rmse_abs_err": None, "mae_abs_err": None, "loss_abs_err": None,
              "passed": None, "tol_abs": _SANITY_ABS_TOL, "tol_rel": _SANITY_REL_TOL}
    if expected is None:
        print("[eval][warn] No matching folsom_pv_forecast_metrics_gpu*.txt file next to "
              "the checkpoint; skipping aggregate sanity check.")
        sanity["passed"] = None
    else:
        exp_loss, exp_rmse, exp_mae = expected
        rmse_err = abs(results["aggregate_rmse"] - exp_rmse)
        mae_err = abs(results["aggregate_mae"] - exp_mae)
        loss_err = abs(results["aggregate_loss"] - exp_loss)
        rmse_ok = rmse_err <= max(_SANITY_ABS_TOL, _SANITY_REL_TOL * exp_rmse)
        mae_ok = mae_err <= max(_SANITY_ABS_TOL, _SANITY_REL_TOL * exp_mae)
        # Loss is noisier (huber + mean over batches with possibly different last-batch
        # size). Use a wider tolerance and only warn -- the RMSE/MAE checks are the
        # authoritative sanity check.
        loss_ok = loss_err <= max(1.0, _SANITY_REL_TOL * 10 * exp_loss)
        passed = bool(rmse_ok and mae_ok)
        sanity.update(
            expected_metrics_file=str(args.ckpt.parent / f"folsom_pv_forecast_metrics_gpu*.txt"),
            expected={"loss": exp_loss, "rmse": exp_rmse, "mae": exp_mae},
            computed={"loss": results["aggregate_loss"],
                      "rmse": results["aggregate_rmse"],
                      "mae": results["aggregate_mae"]},
            rmse_abs_err=rmse_err, mae_abs_err=mae_err, loss_abs_err=loss_err,
            loss_ok=loss_ok, rmse_ok=rmse_ok, mae_ok=mae_ok,
            passed=passed,
        )
        status = "PASS" if passed else "FAIL"
        print(
            f"[eval][sanity {status}] expected RMSE={exp_rmse:.6f} MAE={exp_mae:.6f} loss={exp_loss:.6f}  |  "
            f"computed RMSE={results['aggregate_rmse']:.6f} MAE={results['aggregate_mae']:.6f} "
            f"loss={results['aggregate_loss']:.6f}  |  "
            f"err RMSE={rmse_err:.3e} MAE={mae_err:.3e} loss={loss_err:.3e}"
        )
        if not passed and args.strict_sanity:
            # Still write the JSON so the operator can inspect, then exit non-zero.
            payload = _build_payload(args, ckpt_epoch, ckpt_zero_sky, ckpt_use_nwp,
                                     results, sanity)
            args.output.write_text(json.dumps(payload, indent=2))
            print(f"[eval] wrote {args.output} (sanity FAILED)")
            print("[eval] aborting due to sanity-check mismatch (use --no-strict-sanity to override).")
            return 2

    payload = _build_payload(args, ckpt_epoch, ckpt_zero_sky, ckpt_use_nwp, results, sanity)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"[eval] wrote {args.output}")
    if sanity["passed"]:
        print(f"[eval][sanity OK] aggregate RMSE/MAE match metrics file within tolerance.")
    return 0


def _build_payload(args, ckpt_epoch, ckpt_zero_sky, ckpt_use_nwp, results, sanity) -> dict:
    per_step = []
    for s in range(_LOSS_METRIC_HORIZON):
        per_step.append({
            "step": s + 1,
            "horizon_min": (s + 1) * _OUTPUT_INTERVAL_MIN,
            "rmse": results["per_step_rmse"][s],
            "mae": results["per_step_mae"][s],
            "loss": results["per_step_loss"][s],
            "n_valid": results["per_step_n_valid"][s],
            "pred_mean": results["per_step_pred_mean"][s],
            "target_mean": results["per_step_target_mean"][s],
        })
    return {
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
        "per_step": per_step,
        "aggregate": {
            "rmse": results["aggregate_rmse"],
            "mae": results["aggregate_mae"],
            "loss": results["aggregate_loss"],
            "n_valid": results["aggregate_n_valid"],
            "n_batches": results["n_batches"],
        },
        "sanity_check": sanity,
    }


if __name__ == "__main__":
    sys.exit(main())
