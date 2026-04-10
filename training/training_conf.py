"""Load ``config/conf.yaml`` training section and resolved PV / sky / sat paths for CLI defaults."""

from __future__ import annotations

from pathlib import Path

import yaml

from config_utils import get_resolved_paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONF_PATH = PROJECT_ROOT / "config" / "conf.yaml"

TRAINING_HPARAM_KEYS = frozenset({
    "csv_interval_min",
    "pv_input_interval_min",
    "pv_output_interval_min",
    "pv_input_len",
    "pv_output_len",
    "test_anchor_stride_min",
    "skyimg_window_size",
    "skyimg_time_resolution_min",
    "skyimg_spatial_size",
    "satimg_window_size",
    "satimg_time_resolution_min",
    "satimg_npy_shape_hwc",
    "epochs",
    "lr",
    "batch_size",
    "save_every",
    "num_workers",
    "train_max_batches_per_epoch",
    "loader_test_batch_size",
    "loader_test_num_workers",
})


def load_config() -> dict:
    with open(CONF_PATH) as f:
        return yaml.safe_load(f)


def get_training_hparams_from_conf(conf: dict | None = None) -> dict:
    """Load ``conf['training']``; every :data:`TRAINING_HPARAM_KEYS` entry must be set in YAML."""
    if conf is None:
        conf = load_config()
    raw = conf.get("training")
    if not isinstance(raw, dict):
        raise ValueError("conf.yaml must define a non-empty 'training:' mapping")
    missing = sorted(TRAINING_HPARAM_KEYS - raw.keys())
    if missing:
        raise KeyError(
            "conf training section missing required key(s): "
            + ", ".join(missing)
            + " (see TRAINING_HPARAM_KEYS in training/training_conf.py)"
        )
    out = {k: raw[k] for k in TRAINING_HPARAM_KEYS}
    if isinstance(out["lr"], str):
        out["lr"] = float(out["lr"])

    ss = out["skyimg_spatial_size"]
    if not isinstance(ss, int) or isinstance(ss, bool) or ss < 1:
        raise ValueError("training.skyimg_spatial_size must be a positive integer")
    out["skyimg_spatial_size"] = int(ss)

    shwc = out["satimg_npy_shape_hwc"]
    if not isinstance(shwc, (list, tuple)) or len(shwc) != 3:
        raise ValueError("training.satimg_npy_shape_hwc must be a length-3 sequence [H, W, C]")
    t = tuple(int(x) for x in shwc)
    if any(x < 1 for x in t):
        raise ValueError("training.satimg_npy_shape_hwc entries must be positive")
    out["satimg_npy_shape_hwc"] = t

    return out


def get_training_paths_from_conf(conf: dict | None = None, project_root: Path | None = None) -> dict[str, str]:
    """
    Resolve PV CSV, sky image, and Himawari NPY dirs from ``conf.yaml`` ``paths``:
    ``pv_train_path`` / ``pv_test_path``, ``sky_image_train_path`` / ``sky_image_test_path``,
    and ``sat_*`` as direct children of ``data_dir``.
    """
    if conf is None:
        conf = load_config()
    root = project_root if project_root is not None else PROJECT_ROOT
    paths_cfg = conf.get("paths", {})
    resolved = get_resolved_paths(conf, root)
    data_dir = resolved.get("data_dir")
    if data_dir is None:
        raise ValueError("conf paths.data_dir is required")

    def _req(key: str) -> str:
        v = paths_cfg.get(key)
        if v is None or str(v).strip() == "":
            raise KeyError(f"conf paths.{key} is required")
        return str(v)

    pv_train = (data_dir / _req("pv_train_path")).resolve()
    pv_test = (data_dir / _req("pv_test_path")).resolve()
    sky_train = (data_dir / _req("sky_image_train_path")).resolve()
    sky_test = (data_dir / _req("sky_image_test_path")).resolve()
    sat_train = (data_dir / _req("sat_train_path")).resolve()
    sat_test = (data_dir / _req("sat_test_path")).resolve()

    return {
        "pv_train_dir": str(pv_train),
        "pv_test_dir": str(pv_test),
        "skyimg_train_dir": str(sky_train),
        "skyimg_test_dir": str(sky_test),
        "satimg_train_dir": str(sat_train),
        "satimg_test_dir": str(sat_test),
    }
