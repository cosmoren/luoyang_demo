#!/usr/bin/env python3
"""Implant Folsom weights into a trained Luoyang base checkpoint.

Writes a NEW resume-ready file; never overwrites the host checkpoint.

--mode sky (default): copy Folsom sky encoder into Luoyang host.
--mode full: copy sky encoder + sat encoder + time_mlp + concat head.

Examples (from repo root):
  TORCH_HOME=/home/kyber/.cache/torch \\
    python ft/luo-sky-implant.py \\
    --mode sky \\
    --host /path/to/luoyang_host_best.pt \\
    --donor /path/to/folsom_sky_donor_best.pt \\
    --out  ft/luoyang_host_folsom_sky_implant.pt

  TORCH_HOME=/home/kyber/.cache/torch \\
    python ft/luo-sky-implant.py \\
    --mode full \\
    --host /path/to/luoyang_host_best.pt \\
    --donor /path/to/folsom_donor_best.pt \\
    --out  ft/luoyang_host_folsom_full_implant.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

_FT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _FT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("TORCH_HOME", "/home/kyber/.cache/torch")

from models.models import pv_forecasting_model_vit_dinov2  # noqa: E402

# Sky branch prefixes to copy from Folsom (shape-matched only).
_SKY_COPY_PREFIXES = (
    "sky_patch_embed.",
    "sky_alt_attn.",
    "sky_two_stage_compressor.",
    "sky_mod_embed",
)
# Folsom-only sky extras: copied if present and shape-matched; else skipped.
_SKY_EXTRA_PREFIXES = (
    "sky_sun_patch_embed.",
    "sky_sun_alpha",
)
_SAT_COPY_PREFIXES = (
    "sat_patch_embed.",
    "sat_alt_attn.",
    "sat_two_stage_compressor.",
    "sat_mod_embed",
)
_SHARED_COPY_PREFIXES = (
    "time_mlp.",
    "time_mlp",
)
_CONCAT_COPY_PREFIXES = (
    "cross_attention_pv.",
    "pv_feats_head.",
    "fc.",
)

_IMPLANT_LABEL = {
    "sky": "folsom_sky_into_luoyang_base",
    "full": "folsom_full_sky_sat_concat_into_luoyang_base",
}


def _prefixes_for_mode(mode: str) -> tuple[str, ...]:
    prefixes = _SKY_COPY_PREFIXES + _SKY_EXTRA_PREFIXES
    if mode == "full":
        prefixes = prefixes + _SAT_COPY_PREFIXES + _SHARED_COPY_PREFIXES + _CONCAT_COPY_PREFIXES
    return prefixes


def _is_copyable_key(key: str, prefixes: tuple[str, ...]) -> bool:
    return any(key == p or key.startswith(p) for p in prefixes)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Implant Folsom weights into a Luoyang base checkpoint."
    )
    p.add_argument(
        "--mode",
        choices=("sky", "full"),
        default="sky",
        help="sky: copy sky encoder only (default). full: sky + sat + time_mlp + concat head.",
    )
    p.add_argument(
        "--host",
        type=Path,
        required=True,
        help="Luoyang host best checkpoint (.pt). Never overwritten.",
    )
    p.add_argument(
        "--donor",
        type=Path,
        required=True,
        help="Folsom donor checkpoint (.pt).",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="New output checkpoint path (must differ from --host).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    host_path = args.host.resolve()
    donor_path = args.donor.resolve()
    out_path = args.out.resolve()

    if not host_path.is_file():
        raise FileNotFoundError(f"host checkpoint not found: {host_path}")
    if not donor_path.is_file():
        raise FileNotFoundError(f"donor checkpoint not found: {donor_path}")
    if out_path == host_path:
        raise RuntimeError("refusing to write over host checkpoint path")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    host_mtime_before = host_path.stat().st_mtime_ns
    host_size_before = host_path.stat().st_size

    print(f"[implant] loading Luoyang host: {host_path}", flush=True)
    host_ckpt = torch.load(host_path, map_location="cpu", weights_only=False)
    host_sd = host_ckpt["model_state_dict"]
    # Deep-copy host tensors so later asserts compare against the pristine host.
    host_sd_ref = {k: v.detach().cpu().clone() for k, v in host_sd.items()}
    out_sd = {k: v.detach().cpu().clone() for k, v in host_sd.items()}

    print(f"[implant] loading Folsom donor: {donor_path}", flush=True)
    donor_ckpt = torch.load(donor_path, map_location="cpu", weights_only=False)
    donor_sd = donor_ckpt["model_state_dict"]

    mode = args.mode
    copy_prefixes = _prefixes_for_mode(mode)
    print(f"[implant] mode={mode} prefixes={list(copy_prefixes)}", flush=True)

    n_copied = 0
    n_skipped_missing = 0
    n_skipped_shape = 0
    missing_skips: list[str] = []
    shape_skips: list[str] = []
    copied_keys: list[str] = []

    for key, tensor in donor_sd.items():
        if not _is_copyable_key(key, copy_prefixes):
            continue
        if key not in out_sd:
            n_skipped_missing += 1
            missing_skips.append(key)
            continue
        if tuple(out_sd[key].shape) != tuple(tensor.shape):
            n_skipped_shape += 1
            shape_skips.append(
                f"{key}: donor={tuple(tensor.shape)} host={tuple(out_sd[key].shape)}"
            )
            continue
        out_sd[key] = tensor.detach().cpu().clone()
        copied_keys.append(key)
        n_copied += 1

    out = {
        "model_state_dict": out_sd,
        "epoch": int(host_ckpt.get("epoch", 0)),
        "host_path": str(host_path),
        "donor_path": str(donor_path),
        "n_copied": n_copied,
        "n_skipped_missing": n_skipped_missing,
        "n_skipped_shape": n_skipped_shape,
        "missing_skips": missing_skips,
        "shape_skips": shape_skips,
        "copied_keys": copied_keys,
        "model": "pv_forecasting_model_vit_dinov2",
        "sky_in_channels": 3,
        "mode": mode,
        "implant": _IMPLANT_LABEL[mode],
    }
    torch.save(out, out_path)
    size_mb = out_path.stat().st_size / (1024 * 1024)

    print(
        f"[implant] copied={n_copied} skipped_missing={n_skipped_missing} "
        f"skipped_shape={n_skipped_shape}",
        flush=True,
    )
    if missing_skips:
        print("[implant] missing-in-host skips:", flush=True)
        for line in missing_skips:
            print(f"  {line}", flush=True)
    if shape_skips:
        print("[implant] shape skips:", flush=True)
        for line in shape_skips:
            print(f"  {line}", flush=True)
    print(f"[implant] wrote {out_path} ({size_mb:.1f} MiB)", flush=True)

    # HARD: host file must be unchanged on disk.
    host_stat_after = host_path.stat()
    if host_stat_after.st_mtime_ns != host_mtime_before or host_stat_after.st_size != host_size_before:
        raise RuntimeError(f"host checkpoint was modified on disk: {host_path}")

    print("[smoke] verifying output checkpoint ...", flush=True)
    loaded = torch.load(out_path, map_location="cpu", weights_only=False)
    verify = pv_forecasting_model_vit_dinov2(sky_in_channels=3)
    verify.load_state_dict(loaded["model_state_dict"], strict=True)
    verify_sd = verify.state_dict()

    for key in copied_keys:
        if not torch.equal(verify_sd[key].cpu(), donor_sd[key].cpu()):
            raise AssertionError(f"copied tensor mismatch after reload: {key}")

    copied_set = set(copied_keys)
    for key, host_t in host_sd_ref.items():
        if key in copied_set:
            continue
        if key not in verify_sd:
            raise AssertionError(f"non-copied host key missing after reload: {key}")
        if not torch.equal(verify_sd[key].cpu(), host_t.cpu()):
            raise AssertionError(f"non-copied host tensor changed after implant: {key}")

    print(
        f"[smoke] ok: strict load; {len(copied_keys)} copied tensors match donor; "
        f"{len(host_sd_ref) - len(copied_keys)} non-copied tensors match host; "
        f"host file unchanged",
        flush=True,
    )


if __name__ == "__main__":
    main()
