"""
Lightweight Folsom test-set evaluation that reproduces the end-of-training test pass in
``training/train_vit_test_folsom.py`` (Huber loss + first-16-step masked RMSE/MAE in W/m²).

Loads a saved checkpoint, rebuilds the test ``DataLoader`` with the same dataset kwargs as the
trainer (YAML ``test_anchor_stride_min`` by default — 1500 in ``conf_folsom.yaml``), and calls
the trainer's ``evaluate()`` helper unchanged.

Example invocations (run after activating the training env; metrics only, no NPZ export):

  # Best val-RMSE checkpoint (matches trainer's final "Test set with best val-RMSE checkpoint" line):
  python inference/eval_folsom_checkpoint.py \\
      --checkpoint checkpoints_folsom_pv/folsom_pv_forecast_vit_best_gpu0.pt

  # Last-epoch checkpoint:
  python inference/eval_folsom_checkpoint.py \\
      --checkpoint checkpoints_folsom_pv/folsom_pv_forecast_vit_final_gpu0.pt

  # Optional JSON summary:
  python inference/eval_folsom_checkpoint.py \\
      --checkpoint checkpoints_folsom_pv/folsom_pv_forecast_vit_best_gpu0.pt \\
      --output-dir inference_results/eval_best_gpu0

  # Sky-branch knobs must match training (CLI > YAML):
  python inference/eval_folsom_checkpoint.py \\
      --checkpoint .../folsom_pv_forecast_vit_best_gpu0.pt \\
      --ray-map --sun-mask sun_halo --sky-mask valid_disc
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"
_TRAIN_CONFIG_DIR = _CONFIG_DIR / "train"
_DATASETS_CONFIG_DIR = _CONFIG_DIR / "datasets"
_DEFAULT_TRAIN_CONF_NAME = "conf_train.yaml"
_DEFAULT_FOLSOM_DATASET_CONFIG = "conf_folsom.yaml"
sys.path.insert(0, str(_PROJECT_ROOT))

from dataloader.folsom import (  # noqa: E402
    _FOLSOM_HUBER_DELTA,
    FolsomIrradianceDataset,
    collate_folsom_vit_batch,
)
from models.models import pv_forecasting_model_vit_imgs  # noqa: E402
from training.train_vit_test_folsom import (  # noqa: E402
    _DEFAULT_FOLSOM_DATASET_CONFIG as _TRAINER_DEFAULT_DS,
    _dataset_kwargs,
    _folsom_pv_dataset_config_path,
    _format_nwp_features_for_log,
    _load_yaml,
    _parse_nwp_features,
    _resolve_named_config,
    _resolve_use_satellite,
    _seed_worker,
    _sky_knob_overrides,
    evaluate,
    resolve_nwp_features_from_ckpt,
)


def _load_training_defaults(
    train_config_name: str,
    dataset_config_name: str,
) -> dict:
    """Merge shared train YAML + dataset ``training:`` overrides (same as the trainer)."""
    train_conf_path = _resolve_named_config(_TRAIN_CONFIG_DIR, train_config_name, "config")
    train_conf = _load_yaml(train_conf_path)
    h = dict(train_conf.get("training") or {})
    if not h:
        raise KeyError(f"training config {train_conf_path} is missing a 'training:' section")

    dataset_cfg_path = _resolve_named_config(
        _DATASETS_CONFIG_DIR, dataset_config_name, "dataset-config"
    )
    dataset_cfg_raw = _load_yaml(_folsom_pv_dataset_config_path(dataset_cfg_path))
    for k, v in (dataset_cfg_raw.get("training") or {}).items():
        if v is not None:
            h[k] = v
    return h


def _resolve_regime(
    ckpt: dict,
    *,
    cli_use_nwp: bool | None,
    cli_zero_sky: bool | None,
) -> tuple[bool, bool, str, str]:
    """Return ``(use_nwp, zero_sky, use_nwp_source, zero_sky_source)``."""
    if cli_use_nwp is None:
        if "use_nwp" not in ckpt:
            raise KeyError(
                "checkpoint has no 'use_nwp' field; pass --use-nwp or --no-use-nwp"
            )
        use_nwp = bool(ckpt["use_nwp"])
        un_src = "checkpoint"
    else:
        use_nwp = bool(cli_use_nwp)
        un_src = "cli-override"

    if cli_zero_sky is None:
        if "zero_sky" not in ckpt:
            raise KeyError(
                "checkpoint has no 'zero_sky' field; pass --zero-sky or --no-zero-sky"
            )
        zero_sky = bool(ckpt["zero_sky"])
        zs_src = "checkpoint"
    else:
        zero_sky = bool(cli_zero_sky)
        zs_src = "cli-override"

    return use_nwp, zero_sky, un_src, zs_src


def _build_parser(h: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a Folsom pv_forecasting_model_vit_imgs checkpoint on the test split "
            "(same metrics as train_vit_test_folsom end-of-run evaluate())."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a trained checkpoint (.pt), e.g. folsom_pv_forecast_vit_best_gpuN.pt.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=_DEFAULT_TRAIN_CONF_NAME,
        help=f"Training config filename under config/train/ (batch_size / num_workers defaults; "
        f"default: {_DEFAULT_TRAIN_CONF_NAME!r}).",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=_DEFAULT_FOLSOM_DATASET_CONFIG,
        help=f"Dataset YAML filename under config/datasets/ (default: {_TRAINER_DEFAULT_DS!r}).",
    )
    parser.add_argument(
        "--test-stride-min",
        type=int,
        default=None,
        metavar="MIN",
        help=(
            "Override sampling.test_anchor_stride_min from the dataset YAML. "
            "Default: use YAML value (1500 in conf_folsom.yaml), matching training."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(h["batch_size"]),
        help=f"Test DataLoader batch size (default from train config: {int(h['batch_size'])}).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(h["num_workers"]),
        help=f"DataLoader worker count (default from train config: {int(h['num_workers'])}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda / cuda:N / cpu (default: cuda if available else cpu).",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        metavar="N",
        help="Cap evaluate() to the first N batches (trainer --eval_max_batches).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory for a small JSON summary (metrics + flags used).",
    )
    parser.add_argument(
        "--use-nwp",
        dest="use_nwp",
        action="store_true",
        default=None,
        help="Force real NWP input (override checkpoint use_nwp).",
    )
    parser.add_argument(
        "--no-use-nwp",
        dest="use_nwp",
        action="store_false",
        help="Force zeroed NWP input (override checkpoint use_nwp).",
    )
    parser.add_argument(
        "--zero-sky",
        dest="zero_sky",
        action="store_true",
        default=None,
        help="Force zeroed sky tensors (override checkpoint zero_sky).",
    )
    parser.add_argument(
        "--no-zero-sky",
        dest="zero_sky",
        action="store_false",
        help="Force real sky tensors (override checkpoint zero_sky).",
    )
    parser.add_argument(
        "--nwp-features",
        type=str,
        default=None,
        help=(
            "Override NWP feature selection for model construction. "
            "Default: read nwp_features / nwp_use_invalid_mask from the checkpoint."
        ),
    )
    parser.add_argument(
        "--use-satellite",
        dest="use_satellite",
        action="store_true",
        help="Enable GOES-15 satellite branch (overrides YAML sampling.use_satellite).",
    )
    parser.add_argument(
        "--no-use-satellite",
        dest="use_satellite",
        action="store_false",
        help="Disable satellite branch (overrides YAML sampling.use_satellite).",
    )
    parser.set_defaults(use_satellite=None)
    parser.add_argument(
        "--ray-map",
        dest="ray_map",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Add fisheye ray_map sky channels (CLI > YAML sampling.ray_map > false).",
    )
    parser.add_argument(
        "--sun-mask",
        dest="sun_mask",
        type=str,
        default=None,
        choices=["none", "sun_only", "sun_halo"],
        metavar="MODE",
        help="Sun_mask channel: none|sun_only|sun_halo (CLI > YAML sampling.sun_mask > none).",
    )
    parser.add_argument(
        "--sky-mask",
        dest="sky_mask",
        type=str,
        default=None,
        choices=["none", "loose", "tight", "valid_disc"],
        metavar="MODE",
        help="RGB sky-disc gating: none|loose|tight|valid_disc (CLI > YAML sampling.sky_mask > none).",
    )
    return parser


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=_DEFAULT_TRAIN_CONF_NAME)
    pre_parser.add_argument("--dataset-config", type=str, default=_DEFAULT_FOLSOM_DATASET_CONFIG)
    pre_args, _ = pre_parser.parse_known_args()

    h = _load_training_defaults(pre_args.config, pre_args.dataset_config)
    parser = _build_parser(h)
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[eval_folsom] device={device}")

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    print(f"[eval_folsom] checkpoint={ckpt_path}")

    dataset_cfg = args.dataset_config
    use_satellite = _resolve_use_satellite(dataset_cfg, args.use_satellite)
    _ds_kw = dict(
        use_satellite_override=use_satellite,
        **_sky_knob_overrides(dataset_cfg, args.ray_map, args.sun_mask, args.sky_mask),
    )

    ds_kwargs = _dataset_kwargs(dataset_cfg, "test", **_ds_kw)
    if args.test_stride_min is not None:
        ds_kwargs["test_anchor_stride_min"] = int(args.test_stride_min)
    test_stride_min = int(ds_kwargs["test_anchor_stride_min"])

    test_dataset = FolsomIrradianceDataset(**ds_kwargs)
    n_windows = len(test_dataset)
    print(
        f"[eval_folsom] test windows={n_windows:,}  "
        f"test_anchor_stride_min={test_stride_min}  "
        f"pv_output_len={test_dataset.pv_output_len}"
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    use_nwp, zero_sky, un_src, zs_src = _resolve_regime(
        ckpt,
        cli_use_nwp=args.use_nwp,
        cli_zero_sky=args.zero_sky,
    )
    print(
        f"[eval_folsom] use_nwp={use_nwp} (source: {un_src})  "
        f"zero_sky={zero_sky} (source: {zs_src})"
    )
    print(
        f"[eval_folsom] NWP input: "
        f"{'REAL (raw _FOLSOM_NWP_FEATURE_COLS)' if use_nwp else 'ZEROED-OUT (baseline)'}"
    )
    print(
        f"[eval_folsom] Sky images: "
        f"{'ZEROED (--zero-sky; PV+NWP-style ablation)' if zero_sky else 'REAL from dataset'}"
    )

    if args.nwp_features is not None:
        nwp_features, nwp_use_invalid_mask = _parse_nwp_features(args.nwp_features)
        nwp_src = f"CLI (--nwp-features={args.nwp_features!r})"
    else:
        nwp_features, nwp_use_invalid_mask = resolve_nwp_features_from_ckpt(ckpt)
        nwp_src = "checkpoint" if "nwp_features" in ckpt else "legacy default"
    nwp_features_str = _format_nwp_features_for_log(nwp_features, nwp_use_invalid_mask)
    print(
        f"[eval_folsom] nwp_features={nwp_features_str}  "
        f"use_invalid_mask={nwp_use_invalid_mask} (source: {nwp_src})"
    )

    sky_in_channels = int(test_dataset.sky_in_channels)
    sky_channels_resolved = tuple(getattr(test_dataset, "sky_channels", ("rgb",)))
    print(
        f"[eval_folsom] sky_channels={list(sky_channels_resolved)}  "
        f"sky_in_channels={sky_in_channels}"
    )
    print(f"[eval_folsom] use_satellite={use_satellite}")

    model = pv_forecasting_model_vit_imgs(
        dev_dn_list=test_dataset.devDn_list,
        nwp_features=nwp_features,
        use_invalid_mask=nwp_use_invalid_mask,
        sky_in_channels=sky_in_channels,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    criterion = nn.HuberLoss(delta=_FOLSOM_HUBER_DELTA)

    nw = int(args.num_workers)
    pin = device.type == "cuda"
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=collate_folsom_vit_batch,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=nw > 0,
        worker_init_fn=_seed_worker,
    )

    eval_cap = args.max_batches
    test_loss, test_rmse, test_mae = evaluate(
        model,
        device,
        test_loader,
        criterion,
        max_batches=eval_cap,
        use_nwp=use_nwp,
        zero_sky=zero_sky,
    )
    epoch = ckpt.get("epoch", "?")
    ckpt_loss = ckpt.get("loss")
    print(
        f"Test set ({ckpt_path.name}, epoch={epoch}): "
        f"loss={test_loss:.6f}, RMSE={test_rmse:.4f} W/m², MAE={test_mae:.4f} W/m²"
    )

    if args.output_dir is not None:
        out_dir = Path(args.output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "checkpoint": str(ckpt_path),
            "checkpoint_basename": ckpt_path.name,
            "epoch": epoch,
            "checkpoint_loss": None if ckpt_loss is None else float(ckpt_loss),
            "test_loss": float(test_loss),
            "test_rmse_wm2": float(test_rmse),
            "test_mae_wm2": float(test_mae),
            "test_windows": int(n_windows),
            "test_anchor_stride_min": test_stride_min,
            "pv_output_len": int(test_dataset.pv_output_len),
            "batch_size": int(args.batch_size),
            "max_batches": eval_cap,
            "use_nwp": bool(use_nwp),
            "zero_sky": bool(zero_sky),
            "use_satellite": bool(use_satellite),
            "sky_channels": list(sky_channels_resolved),
            "sky_in_channels": sky_in_channels,
            "nwp_features": list(nwp_features),
            "nwp_use_invalid_mask": bool(nwp_use_invalid_mask),
            "dataset_config": dataset_cfg,
        }
        summary_path = out_dir / "eval_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
        print(f"[eval_folsom] wrote {summary_path}")


if __name__ == "__main__":
    main()
