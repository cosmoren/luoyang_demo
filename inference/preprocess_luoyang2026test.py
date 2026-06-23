"""
Precompute Luoyang 2026 full test cache (sat/sky + PV batches) into one folder.

Output layout:
  - <cache_dir>/manifest.json
  - <cache_dir>/windows/win_<idx>.pt              (window-level sat/sky cache)
  - <cache_dir>/pv_batches/win_<idx>/batch_<k>.pt (PV batch cache)

Example:
  python inference/preprocess_luoyang2026test.py \
    --dataset-config conf_luoyang_2026.yaml \
    --stride_min 5 \
    --batch_size 256 \
    --output_dir inference_cache/luoyang2026_test_full
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dataloader.luoyang_2026_zarr import PVDataset  # noqa: E402
from training.train_vit_luoyang2026 import _dataset_kwargs, _resolve_named_config  # noqa: E402

_DEFAULT_DATASET_CONFIG = "conf_luoyang_2026.yaml"
_DEFAULT_STRIDE_MIN = 5
_DATASETS_CONFIG_DIR = _PROJECT_ROOT / "config" / "datasets"
_WINDOW_REQUIRED_KEYS = (
    "sat_tensor",
    "sat_timefeats",
    "sat_valid_mask",
    "skimg_tensor",
    "skimg_timefeats",
    "skimg_valid_mask",
)
_PV_BATCH_REQUIRED_KEYS = (
    "dev_idx",
    "kt",
    "kt_mask",
    "pv_timefeats",
    "forecast_timefeats",
    "target_pv",
    "target_p_cs",
    "p_mean",
    "sample_valid",
)


def _dataset_kwargs_for_preprocess(
    dataset_config_name: str,
    *,
    test_stride_min_override: int,
    max_inverters: int | None,
) -> dict[str, Any]:
    kwargs = _dataset_kwargs(dataset_config_name, split="test", max_files=max_inverters)
    kwargs["test_anchor_stride_min"] = int(test_stride_min_override)
    kwargs["enable_sat_sky_cache"] = True
    return kwargs


def _cfg_sha256(cfg_path: Path) -> str:
    text = cfg_path.read_text(encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _as_timestamp_str(ts: Any) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.isoformat()


def _materialize_window_record(
    ds: PVDataset,
    win_idx: int,
    *,
    sat_default_shape: tuple[int, ...],
    skimg_default_shape: tuple[int, ...],
) -> dict[str, Any]:
    bundle = ds.load_sat_sky_for_window(win_idx)

    sat_tensor = bundle.get("sat_tensor")
    sat_timefeats = bundle.get("sat_timefeats")
    sat_valid_scalar = bundle.get("sat_valid")
    if sat_tensor is None:
        sat_tensor = torch.zeros(*sat_default_shape, dtype=torch.float32)
    if sat_timefeats is None:
        sat_timefeats = torch.zeros(sat_default_shape[0], 9, dtype=torch.float32)
    sat_valid_mask = torch.tensor(
        [float(sat_valid_scalar.item()) if sat_valid_scalar is not None else 0.0],
        dtype=torch.float32,
    )

    skimg_tensor = bundle.get("skimg_tensor")
    skimg_timefeats = bundle.get("skimg_timefeats")
    skimg_valid_scalar = bundle.get("skimg_valid")
    if skimg_tensor is None:
        skimg_tensor = torch.zeros(*skimg_default_shape, dtype=torch.float32)
    if skimg_timefeats is None:
        skimg_timefeats = torch.zeros(skimg_default_shape[0], 9, dtype=torch.float32)
    skimg_valid_mask = torch.tensor(
        [float(skimg_valid_scalar.item()) if skimg_valid_scalar is not None else 0.0],
        dtype=torch.float32,
    )

    if sat_tensor.dim() != 4:
        raise ValueError(f"win={win_idx} sat_tensor must be 4D [T,C,H,W], got {tuple(sat_tensor.shape)}")
    if sat_timefeats.dim() != 2 or sat_timefeats.shape[1] != 9:
        raise ValueError(
            f"win={win_idx} sat_timefeats must be [T,9], got {tuple(sat_timefeats.shape)}"
        )
    if skimg_tensor.dim() != 4:
        raise ValueError(
            f"win={win_idx} skimg_tensor must be 4D [T,C,H,W], got {tuple(skimg_tensor.shape)}"
        )
    if skimg_timefeats.dim() != 2 or skimg_timefeats.shape[1] != 9:
        raise ValueError(
            f"win={win_idx} skimg_timefeats must be [T,9], got {tuple(skimg_timefeats.shape)}"
        )
    if tuple(sat_tensor.shape) != sat_default_shape:
        raise ValueError(
            f"win={win_idx} sat_tensor shape mismatch: expected={sat_default_shape}, got={tuple(sat_tensor.shape)}"
        )
    if tuple(skimg_tensor.shape) != skimg_default_shape:
        raise ValueError(
            f"win={win_idx} skimg_tensor shape mismatch: expected={skimg_default_shape}, got={tuple(skimg_tensor.shape)}"
        )
    if sat_timefeats.shape[0] != sat_default_shape[0]:
        raise ValueError(
            f"win={win_idx} sat_timefeats time mismatch: expected T={sat_default_shape[0]}, got={sat_timefeats.shape[0]}"
        )
    if skimg_timefeats.shape[0] != skimg_default_shape[0]:
        raise ValueError(
            f"win={win_idx} skimg_timefeats time mismatch: expected T={skimg_default_shape[0]}, got={skimg_timefeats.shape[0]}"
        )

    assert sat_valid_mask.shape == (1,)
    assert skimg_valid_mask.shape == (1,)

    assert ds._test_last_x_time_ref is not None
    time_utc = _as_timestamp_str(ds._test_last_x_time_ref[win_idx])
    return {
        "win_idx": int(win_idx),
        "time_utc": time_utc,
        "sat_tensor": sat_tensor.to(dtype=torch.float32).contiguous(),
        "sat_timefeats": sat_timefeats.to(dtype=torch.float32).contiguous(),
        "sat_valid_mask": sat_valid_mask,
        "skimg_tensor": skimg_tensor.to(dtype=torch.float32).contiguous(),
        "skimg_timefeats": skimg_timefeats.to(dtype=torch.float32).contiguous(),
        "skimg_valid_mask": skimg_valid_mask,
    }


def _build_pv_batch_record(
    ds: PVDataset,
    *,
    win_idx: int,
    file_start: int,
    file_end: int,
    inverter_names: list[str],
) -> dict[str, Any]:
    pv_samples = [
        ds.build_pv_sample_for_window(file_idx, win_idx)
        for file_idx in range(file_start, file_end)
    ]
    if not pv_samples:
        raise ValueError(f"empty PV batch at win={win_idx} [{file_start},{file_end})")

    def _stack(key: str) -> torch.Tensor:
        return torch.stack([s[key] for s in pv_samples])

    nwp_vals = [s.get("nwp_tensor") for s in pv_samples]
    nwp_tensor = None
    if all(isinstance(v, torch.Tensor) for v in nwp_vals):
        nwp_tensor = torch.stack([v for v in nwp_vals if isinstance(v, torch.Tensor)])

    sample_valid = (
        _stack("sample_valid")
        if "sample_valid" in pv_samples[0]
        else torch.ones(len(pv_samples), dtype=torch.float32)
    )
    out = {
        "win_idx": int(win_idx),
        "file_start": int(file_start),
        "file_end": int(file_end),
        "inverter_names": list(inverter_names[file_start:file_end]),
        "dev_idx": _stack("dev_idx"),
        "kt": _stack("kt"),
        "kt_mask": _stack("kt_mask"),
        "pv_timefeats": _stack("pv_timefeats"),
        "forecast_timefeats": _stack("forecast_timefeats"),
        "target_pv": _stack("target_pv"),
        "target_p_cs": _stack("target_p_cs"),
        "p_mean": _stack("p_mean"),
        "sample_valid": sample_valid,
        "nwp_tensor": nwp_tensor,
    }
    for key in _PV_BATCH_REQUIRED_KEYS:
        if key not in out:
            raise KeyError(f"PV batch missing required key {key!r}")
    bsz = file_end - file_start
    if int(out["dev_idx"].shape[0]) != bsz:
        raise ValueError(
            f"PV batch size mismatch at win={win_idx} [{file_start},{file_end}): "
            f"dev_idx batch={out['dev_idx'].shape[0]} expected={bsz}"
        )
    return out


def _write_manifest(
    out_dir: Path,
    *,
    args: argparse.Namespace,
    cfg_path: Path,
    cfg_sha256: str,
    ds: PVDataset,
    n_files: int,
    n_batches_per_window: int,
    pv_batch_size: int,
    total_windows: int,
    processed_windows: int,
    processed_pv_batches: int,
    sat_valid_sum: float,
    skimg_valid_sum: float,
    first_time_utc: str,
    last_time_utc: str,
) -> None:
    manifest = {
        "version": 2,
        "created_unix": int(time.time()),
        "cache_mode": "full",
        "split": "test",
        "dataset_config": {
            "arg": args.dataset_config,
            "resolved_path": str(cfg_path),
            "sha256": cfg_sha256,
        },
        "stride_min": int(args.stride_min),
        "batch_size": int(pv_batch_size),
        "max_inverters": None if args.max_inverters is None else int(args.max_inverters),
        "num_inverters": int(n_files),
        "num_batches_per_window": int(n_batches_per_window),
        "num_test_windows_total": int(total_windows),
        "num_test_windows_processed": int(processed_windows),
        "num_pv_batches_processed": int(processed_pv_batches),
        "has_pv_batches": True,
        "window_range": {
            "start_win_idx": int(args.start_win_idx),
            "max_windows": None if args.max_windows is None else int(args.max_windows),
            "first_time_utc": first_time_utc,
            "last_time_utc": last_time_utc,
        },
        "required_keys": {
            "windows": list(_WINDOW_REQUIRED_KEYS),
            "pv_batch": list(_PV_BATCH_REQUIRED_KEYS),
        },
        "paths": {
            "windows_dir": "windows",
            "pv_batches_dir": "pv_batches",
        },
        "shapes": {
            "sat_tensor": [int(ds.satimg_window_size), 3, 100, 100],
            "sat_timefeats": [int(ds.satimg_window_size), 9],
            "sat_valid_mask": [1],
            "skimg_tensor": [int(ds.skyimg_window_size), 3, 224, 224],
            "skimg_timefeats": [int(ds.skyimg_window_size), 9],
            "skimg_valid_mask": [1],
        },
        "availability": {
            "sat_valid_rate": float(sat_valid_sum / max(processed_windows, 1)),
            "skimg_valid_rate": float(skimg_valid_sum / max(processed_windows, 1)),
            "sat_valid_count": float(sat_valid_sum),
            "skimg_valid_count": float(skimg_valid_sum),
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[preprocess] wrote manifest: {manifest_path}", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess Luoyang 2026 full test cache (sat/sky + PV batches)")
    parser.add_argument("--dataset-config", type=str, default=_DEFAULT_DATASET_CONFIG)
    parser.add_argument("--stride_min", type=int, default=_DEFAULT_STRIDE_MIN)
    parser.add_argument("--batch_size", type=int, default=256, help="PV cache batch size for inference.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_inverters", type=int, default=None, help="Optional: limit loaded CSV files.")
    parser.add_argument("--start_win_idx", type=int, default=0, help="Start window index for partial runs.")
    parser.add_argument("--max_windows", type=int, default=None, help="Optional: number of windows to preprocess.")
    parser.add_argument("--log_every", type=int, default=50, help="Progress print frequency.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing win_*.pt files.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.stride_min <= 0:
        raise ValueError(f"--stride_min must be > 0 (got {args.stride_min})")
    if args.max_inverters is not None and args.max_inverters <= 0:
        raise ValueError(f"--max_inverters must be > 0 (got {args.max_inverters})")
    if args.batch_size <= 0:
        raise ValueError(f"--batch_size must be > 0 (got {args.batch_size})")
    if args.start_win_idx < 0:
        raise ValueError(f"--start_win_idx must be >= 0 (got {args.start_win_idx})")
    if args.max_windows is not None and args.max_windows <= 0:
        raise ValueError(f"--max_windows must be > 0 (got {args.max_windows})")
    if args.log_every <= 0:
        raise ValueError(f"--log_every must be > 0 (got {args.log_every})")

    out_dir = Path(args.output_dir).expanduser().resolve()
    windows_dir = out_dir / "windows"
    pv_batches_root = out_dir / "pv_batches"
    windows_dir.mkdir(parents=True, exist_ok=True)
    pv_batches_root.mkdir(parents=True, exist_ok=True)

    cfg_path = _resolve_named_config(_DATASETS_CONFIG_DIR, args.dataset_config, "dataset-config")
    cfg_sha256 = _cfg_sha256(cfg_path)
    print(
        f"[preprocess] dataset_config={args.dataset_config} resolved={cfg_path} sha256={cfg_sha256[:12]}...",
        flush=True,
    )
    print(
        f"[preprocess] stride_min={args.stride_min} output_dir={out_dir} "
        f"max_inverters={args.max_inverters} batch_size={args.batch_size}",
        flush=True,
    )

    ds = PVDataset(
        **_dataset_kwargs_for_preprocess(
            args.dataset_config,
            test_stride_min_override=args.stride_min,
            max_inverters=args.max_inverters,
        )
    )
    total_windows = int(ds._num_test_windows)
    print(
        f"[preprocess] dataset ready: files={len(ds.sample_files)} total_test_windows={total_windows}",
        flush=True,
    )
    if total_windows <= 0:
        raise ValueError("No test windows found. Check dataset config and split policy.")
    if args.start_win_idx >= total_windows:
        raise ValueError(
            f"--start_win_idx ({args.start_win_idx}) out of range for total windows={total_windows}"
        )

    start_idx = int(args.start_win_idx)
    end_idx = total_windows if args.max_windows is None else min(total_windows, start_idx + int(args.max_windows))
    if end_idx <= start_idx:
        raise ValueError(f"empty window range: start={start_idx}, end={end_idx}")

    sat_default_shape = (int(ds.satimg_window_size), 3, 100, 100)
    skimg_default_shape = (int(ds.skyimg_window_size), 3, 224, 224)
    n_files = int(len(ds.sample_files))
    n_batches_per_window = int(math.ceil(n_files / int(args.batch_size)))
    inverter_names = [p.stem.replace("_", "=") for p in ds.sample_files]

    sat_valid_sum = 0.0
    skimg_valid_sum = 0.0
    pv_batches_processed = 0
    processed = 0
    first_time_utc = ""
    last_time_utc = ""
    t0 = time.time()

    for win_idx in range(start_idx, end_idx):
        record = _materialize_window_record(
            ds,
            win_idx,
            sat_default_shape=sat_default_shape,
            skimg_default_shape=skimg_default_shape,
        )
        missing = [k for k in _WINDOW_REQUIRED_KEYS if k not in record]
        if missing:
            raise KeyError(f"win={win_idx} missing required keys: {missing}")

        pt_path = windows_dir / f"win_{win_idx:05d}.pt"
        if pt_path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {pt_path} (use --overwrite)")
        torch.save(record, pt_path)

        win_pv_dir = pv_batches_root / f"win_{win_idx:05d}"
        win_pv_dir.mkdir(parents=True, exist_ok=True)
        for file_start in range(0, n_files, int(args.batch_size)):
            file_end = min(file_start + int(args.batch_size), n_files)
            batch_idx = file_start // int(args.batch_size)
            pv_batch = _build_pv_batch_record(
                ds,
                win_idx=win_idx,
                file_start=file_start,
                file_end=file_end,
                inverter_names=inverter_names,
            )
            pv_path = win_pv_dir / f"batch_{batch_idx:03d}.pt"
            if pv_path.exists() and not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite existing file: {pv_path} (use --overwrite)")
            torch.save(pv_batch, pv_path)
            pv_batches_processed += 1

        sat_valid_sum += float(record["sat_valid_mask"].mean().item())
        skimg_valid_sum += float(record["skimg_valid_mask"].mean().item())
        processed += 1
        if processed == 1:
            first_time_utc = str(record["time_utc"])
        last_time_utc = str(record["time_utc"])

        if processed % int(args.log_every) == 0 or win_idx == end_idx - 1:
            elapsed = max(time.time() - t0, 1e-6)
            speed = processed / elapsed
            print(
                f"[preprocess] progress {processed}/{end_idx - start_idx} "
                f"({speed:.2f} win/s) sat_rate={sat_valid_sum/processed:.4f} "
                f"skimg_rate={skimg_valid_sum/processed:.4f} "
                f"pv_batches={pv_batches_processed}",
                flush=True,
            )

    _write_manifest(
        out_dir,
        args=args,
        cfg_path=cfg_path,
        cfg_sha256=cfg_sha256,
        ds=ds,
        n_files=n_files,
        n_batches_per_window=n_batches_per_window,
        pv_batch_size=int(args.batch_size),
        total_windows=total_windows,
        processed_windows=processed,
        processed_pv_batches=pv_batches_processed,
        sat_valid_sum=sat_valid_sum,
        skimg_valid_sum=skimg_valid_sum,
        first_time_utc=first_time_utc,
        last_time_utc=last_time_utc,
    )

    print(
        f"[preprocess] done: processed={processed} windows, "
        f"sat_valid_rate={sat_valid_sum/max(processed,1):.4f}, "
        f"skimg_valid_rate={skimg_valid_sum/max(processed,1):.4f}, "
        f"pv_batches={pv_batches_processed}",
        flush=True,
    )


if __name__ == "__main__":
    main()
