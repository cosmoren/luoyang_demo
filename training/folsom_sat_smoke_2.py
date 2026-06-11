"""Forward+backward smoke test for the Folsom **satellite branch** (``_2`` sidecars).

What it checks
--------------
1. Builds the train + val :class:`dataloader.folsom_2.FolsomIrradianceDataset` using
   ``config/datasets/conf_folsom_2.yaml`` via the same ``_dataset_kwargs`` helper that
   the ``training/train_vit_test_folsom_2.py`` trainer uses.
2. Pulls one batch through ``collate_folsom_irradiance``, asserting:
     * ``sat_tensor`` shape ``[B, 24, 3, 100, 100]`` float32
     * ``sat_timefeats`` shape ``[B, 24, 9]`` float32
     * the batch is not entirely missing -- at least one sample's ``sat_tensor.max() > 0``
       (else the model short-circuits to a zero sat path and sat-branch params get
       no gradient, defeating the point of the smoke test).
3. Runs one forward + backward pass on ``pv_forecasting_model_vit_imgs``.
4. Asserts: no NaN/Inf in loss or gradients, and sat-branch parameters
   (``sat_patch_embed`` / ``sat_alt_attn`` / ``sat_two_stage_compressor`` etc.) have
   non-zero gradient norms.

Run::

    micromamba run -n luoyang python training/folsom_sat_smoke_2.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dataloader.folsom_2 import (  # noqa: E402
    FolsomIrradianceDataset,
    collate_folsom_irradiance,
)
from models.models import pv_forecasting_model_vit_imgs  # noqa: E402
from training.train_vit_test_folsom_2 import (  # noqa: E402
    _DEFAULT_FOLSOM_DATASET_CONFIG,
    _batch_to_device,
    _dataset_kwargs,
    _prepare_nwp_for_vit,
    _prepare_sky_for_vit,
    forward_vit,
)

# Substrings used to flag a parameter as belonging to the "satellite branch" of
# ``pv_forecasting_model_vit_imgs`` (see ``models/models.py``: the sat path uses
# ``sat_patch_embed`` -> ``sat_alt_attn`` -> ``sat_two_stage_compressor`` plus a
# learnable ``sat_mod_embed`` and a shared ``time_mlp`` applied to sat timefeats).
_SAT_PARAM_TOKENS = (
    "sat_patch_embed",
    "sat_alt_attn",
    "sat_two_stage_compressor",
    "sat_mod_embed",
)


def _print_batch_shapes(batch: dict) -> None:
    keys = (
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
    print("[smoke] batch tensor shapes:")
    for k in keys:
        v = batch.get(k)
        if v is None:
            print(f"    {k:>20s}: None")
        elif isinstance(v, torch.Tensor):
            print(f"    {k:>20s}: shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"    {k:>20s}: {type(v).__name__}")


def _assert_sat_tensor_ok(batch: dict, expected_T: int, expected_C: int, expected_HW: int) -> None:
    sat = batch.get("sat_tensor")
    sat_tf = batch.get("sat_timefeats")
    if sat is None or sat_tf is None:
        raise AssertionError(
            "sat_tensor / sat_timefeats is None in the collated batch -- the _2 dataloader "
            "is not returning satellite data."
        )
    B = sat.shape[0]
    if sat.shape != (B, expected_T, expected_C, expected_HW, expected_HW):
        raise AssertionError(
            f"sat_tensor shape {tuple(sat.shape)} != "
            f"[B={B}, T={expected_T}, C={expected_C}, H=W={expected_HW}]"
        )
    if sat_tf.shape != (B, expected_T, 9):
        raise AssertionError(
            f"sat_timefeats shape {tuple(sat_tf.shape)} != [B={B}, T={expected_T}, 9]"
        )
    if sat.dtype != torch.float32:
        raise AssertionError(f"sat_tensor dtype {sat.dtype} != torch.float32")
    if not torch.isfinite(sat).all():
        raise AssertionError("sat_tensor contains non-finite values")
    if not torch.isfinite(sat_tf).all():
        raise AssertionError("sat_timefeats contains non-finite values")
    per_sample_max = sat.flatten(1).amax(dim=1)
    nonzero = int((per_sample_max > 0).sum().item())
    print(
        f"[smoke] sat_tensor: B={B}  per-sample max>0 in {nonzero}/{B} samples; "
        f"global max={float(sat.max()):.4f}  global mean={float(sat.mean()):.6f}"
    )
    if float(sat.max()) == 0.0:
        raise AssertionError(
            "All sat_tensor frames are zero -- the model would short-circuit the sat branch "
            "and we could not check non-zero gradients on sat-branch params. Try a different "
            "epoch / random seed (or check that /work/folsom_dataset/sat_goes_gridsat/ has data)."
        )


def _gradient_diagnostics(model: nn.Module) -> tuple[float, dict[str, float]]:
    total_sq = 0.0
    n_with_grad = 0
    sat_grad_norms: dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        if not torch.isfinite(g).all():
            raise AssertionError(f"non-finite gradient in parameter {name!r}")
        gnorm = float(g.norm().item())
        total_sq += gnorm * gnorm
        n_with_grad += 1
        for tok in _SAT_PARAM_TOKENS:
            if tok in name:
                sat_grad_norms[name] = gnorm
                break
    if n_with_grad == 0:
        raise AssertionError("no parameter received a gradient -- something is very wrong")
    total_norm = math.sqrt(total_sq)
    return total_norm, sat_grad_norms


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Folsom satellite branch smoke test")
    p.add_argument(
        "--dataset-config",
        type=str,
        default=_DEFAULT_FOLSOM_DATASET_CONFIG,
        help=f"Dataset YAML filename under config/datasets/ (default: {_DEFAULT_FOLSOM_DATASET_CONFIG!r}).",
    )
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override torch device (default: cuda if available, else cpu).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device={device}")

    kwargs = _dataset_kwargs(args.dataset_config, "val")
    print(f"[smoke] dataset config: {kwargs['config_path']}")
    print(f"[smoke] satimg_dir:     {kwargs['satimg_dir']}")
    print(
        "[smoke] satimg sampling: "
        f"window_size={kwargs['satimg_window_size']}  "
        f"dt={kwargs['satimg_time_resolution_min']} min  "
        f"shape_hwc={kwargs['satimg_npy_shape_hwc']}"
    )

    ds = FolsomIrradianceDataset(**kwargs)
    print(f"[smoke] val dataset built; len={len(ds)}")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_folsom_irradiance,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    batch = next(iter(loader))
    _print_batch_shapes(batch)

    H, W, C = kwargs["satimg_npy_shape_hwc"]
    if H != W:
        raise AssertionError(f"smoke assumes square sat frames; got H={H}, W={W}")
    _assert_sat_tensor_ok(
        batch,
        expected_T=int(kwargs["satimg_window_size"]),
        expected_C=int(C),
        expected_HW=int(H),
    )

    dev_dn_list = ds.devDn_list
    model = pv_forecasting_model_vit_imgs(dev_dn_list=dev_dn_list).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.HuberLoss(delta=1.0)

    d = _batch_to_device(batch, device)
    _prepare_nwp_for_vit(d, use_nwp=False)
    _prepare_sky_for_vit(d, zero_sky=False)

    optimizer.zero_grad(set_to_none=True)
    pv_pred = forward_vit(model, d)
    if pv_pred.dim() != 2:
        raise AssertionError(f"pv_pred expected shape [B, T_out], got {tuple(pv_pred.shape)}")
    t_out = int(pv_pred.shape[1])
    h = min(16, t_out)
    m = d["target_mask"][:, :h]
    loss = criterion((pv_pred[:, :h] * m), (d["target_pv"][:, :h] * m))
    print(f"[smoke] forward OK; pv_pred shape={tuple(pv_pred.shape)}  loss={float(loss):.6f}")
    if not torch.isfinite(loss).item():
        raise AssertionError(f"loss is non-finite: {float(loss)}")

    loss.backward()
    total_norm, sat_grads = _gradient_diagnostics(model)
    print(f"[smoke] backward OK; total grad-norm={total_norm:.4f}")

    if not sat_grads:
        raise AssertionError(
            "no sat-branch parameter (tokens "
            f"{_SAT_PARAM_TOKENS}) received a gradient -- model is not actually using the "
            "satellite data."
        )
    n_sat_nonzero = sum(1 for v in sat_grads.values() if v > 0)
    n_sat_total = len(sat_grads)
    sat_norm_sq = sum(v * v for v in sat_grads.values())
    print(
        f"[smoke] sat-branch params with non-zero grad: {n_sat_nonzero}/{n_sat_total}  "
        f"(L2-norm over sat-branch grads = {math.sqrt(sat_norm_sq):.4f})"
    )
    top_sat = sorted(sat_grads.items(), key=lambda kv: kv[1], reverse=True)[:6]
    for name, g in top_sat:
        print(f"    sat-grad[{name}] = {g:.6g}")
    if n_sat_nonzero == 0:
        raise AssertionError(
            "all sat-branch parameter gradients are exactly zero -- model is short-circuiting "
            "the satellite path. Check `sat_tensor.max() == 0` heuristic in pv_forecasting_model_vit_imgs."
        )

    print("[smoke] SUCCESS: forward+backward clean, sat-branch params received non-zero gradients.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
