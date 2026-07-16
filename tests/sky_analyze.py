"""
Load the full test set (chronological order) and extract aligned arrays of
kt, pv power, and sky images for analysis.

Usage
-----
    python tests/sky_analyze.py

Output (numpy arrays, all length = number of test samples)
-----------------------------------------------------------
    times_utc_ns : np.ndarray [N]         int64  UTC timestamp in nanoseconds
    kt_arr       : np.ndarray [N]         float32  kt at anchor time t0 (last history step)
    pv_arr       : np.ndarray [N]         float32  PV power (W) at anchor time t0
    sky_arr      : np.ndarray [N, H, W, C] float32  last sky-image frame (zeros if unavailable)
    sky_valid    : np.ndarray [N]         float32  1.0 if sky image available, else 0.0
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dataloader.luoyang_2026total_zarr import PVDataset, collate_batched

# ── Dataset config (sky enabled) ──────────────────────────────────────────────
_DATASET_CONFIG = "conf_luoyang_2026_15m_sky.yaml"
_DATASETS_DIR   = _PROJECT_ROOT / "config" / "datasets"

# ── Build the test PVDataset ───────────────────────────────────────────────────
def _build_test_dataset() -> PVDataset:
    import yaml

    cfg_path = _DATASETS_DIR / _DATASET_CONFIG
    cfg = yaml.safe_load(cfg_path.read_text())
    paths_cfg    = cfg.get("paths", {})
    sampling_cfg = cfg.get("sampling", {})
    split_cfg    = cfg.get("split_policy", {})

    data_dir = Path(paths_cfg["data_dir"])

    return PVDataset(
        config_path=str(cfg_path),
        pv_dir=str(data_dir / paths_cfg["pv_total_path"]),
        skyimg_dir=str(data_dir / paths_cfg["sky_image_path"]),
        satimg_dir=str(data_dir / paths_cfg["sat_path"]),
        split="test",
        csv_interval_min=int(sampling_cfg["csv_interval_min"]),
        pv_input_interval_min=int(sampling_cfg["pv_input_interval_min"]),
        pv_input_len=int(sampling_cfg["pv_input_len"]),
        pv_output_interval_min=int(sampling_cfg["pv_output_interval_min"]),
        pv_output_len=int(sampling_cfg["pv_output_len"]),
        pv_train_time_fraction=float(sampling_cfg.get("pv_train_time_fraction", 0.7)),
        test_anchor_stride_min=int(sampling_cfg["test_anchor_stride_min"]),
        val_anchor_stride_min=int(sampling_cfg["val_anchor_stride_min"]),
        test_collect_time_match_tolerance_min=int(sampling_cfg["test_collect_time_match_tolerance_min"]),
        skyimg_window_size=int(sampling_cfg["skyimg_window_size"]),
        skyimg_time_resolution_min=int(sampling_cfg["skyimg_time_resolution_min"]),
        skyimg_spatial_size=int(sampling_cfg.get("skyimg_spatial_size", 224)),
        satimg_window_size=int(sampling_cfg["satimg_window_size"]),
        satimg_time_resolution_min=int(sampling_cfg["satimg_time_resolution_min"]),
        satimg_npy_shape_hwc=tuple(int(x) for x in sampling_cfg.get("satimg_npy_shape_hwc", [100, 100, 3])),
        test_start_bj=split_cfg.get("test_start_bj", "2026-05-11 00:00:00"),
        train_fraction=float(split_cfg.get("train_fraction", 0.85)),
        val_fraction=float(split_cfg.get("val_fraction", 0.15)),
    )


def load_test_arrays():
    """
    Returns
    -------
    times_utc_ns : np.ndarray [N] int64
    kt_arr       : np.ndarray [N] float32   – kt at anchor time t0
    pv_arr       : np.ndarray [N] float32   – PV power (W) at anchor time t0
    sky_arr      : np.ndarray [N, H, W, 3] float32  – last sky-image frame (RGB), zeros if unavailable
    sky_valid    : np.ndarray [N] float32   – 1.0 if sky image present, else 0.0
    """
    print("Building test dataset ...")
    ds = _build_test_dataset()
    N = len(ds)
    print(f"  test samples: {N}")

    # Determine sky image spatial size from the dataset
    sky_h = sky_w = ds._skyimg_spatial_size if hasattr(ds, "_skyimg_spatial_size") else 224

    times_list  = []
    kt_list     = []
    pv_list     = []
    sky_list    = []
    valid_list  = []

    for i in range(N):
        if i % 500 == 0:
            print(f"  [{i}/{N}] ...")
        d = ds[i]

        # Anchor time
        times_list.append(int(d["anchor_time_utc_ns"].item()))

        # kt at anchor time t0: last step of history kt
        kt_list.append(float(d["kt"][0, -1].item()))

        # PV power at anchor time t0: last step of history pv
        pv_list.append(float(d["pv"][0, -1].item()))

        # Sky image: last frame in the window, shape [C, H, W] → [H, W, C]
        skimg = d.get("skimg_tensor")
        valid = float(d.get("skimg_valid", torch.tensor(0.0)).item())
        valid_list.append(valid)
        if skimg is not None and valid > 0.5:
            last_frame = skimg[-1].permute(1, 2, 0).numpy()  # [H, W, C]
            # keep only first 3 channels (RGB), drop asi_mask channel if present
            sky_list.append(last_frame[:, :, :3].astype(np.float32))
        else:
            sky_list.append(np.zeros((sky_h, sky_w, 3), dtype=np.float32))

    # Sort chronologically by anchor time
    order = np.argsort(times_list)

    times_utc_ns = np.array(times_list,  dtype=np.int64  )[order]
    kt_arr       = np.array(kt_list,     dtype=np.float32)[order]
    pv_arr       = np.array(pv_list,     dtype=np.float32)[order]
    sky_arr      = np.stack(sky_list,    axis=0           )[order]   # [N, H, W, 3]
    sky_valid    = np.array(valid_list,  dtype=np.float32)[order]

    print(f"Done. sky_valid ratio: {sky_valid.mean():.3f}")
    return times_utc_ns, kt_arr, pv_arr, sky_arr, sky_valid


if __name__ == "__main__":
    times_utc_ns, kt_arr, pv_arr, sky_arr, sky_valid = load_test_arrays()
    print(f"times shape : {times_utc_ns.shape}")
    print(f"kt    shape : {kt_arr.shape},  range [{kt_arr.min():.3f}, {kt_arr.max():.3f}]")
    print(f"pv    shape : {pv_arr.shape},  range [{pv_arr.min():.1f}, {pv_arr.max():.1f}] W")
    print(f"sky   shape : {sky_arr.shape}")
    print(f"valid shape : {sky_valid.shape},  mean={sky_valid.mean():.3f}")
