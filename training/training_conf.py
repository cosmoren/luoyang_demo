"""Load ``config/conf.yaml`` training section and resolved data paths for CLI defaults."""

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
    "pv_train_time_fraction",
    "test_anchor_stride_min",
    "test_collect_time_match_tolerance_min",
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
})

# Folsom (irradiance + sky + NWP): no satellite branch — training YAML omits ``satimg_*`` keys.
FOLSOM_TRAINING_HPARAM_KEYS = TRAINING_HPARAM_KEYS - frozenset(
    {"satimg_window_size", "satimg_time_resolution_min", "satimg_npy_shape_hwc"}
)


def _dataset_profile_is_folsom(conf: dict) -> bool:
    return str(conf.get("dataset_profile", "")).strip().lower() == "folsom"


def load_config() -> dict:
    with open(CONF_PATH) as f:
        return yaml.safe_load(f)


def get_training_hparams_from_conf(conf: dict | None = None) -> dict:
    """Load ``conf['training']``; required keys depend on ``dataset_profile`` (Folsom omits ``satimg_*``)."""
    if conf is None:
        conf = load_config()
    raw = conf.get("training")
    if not isinstance(raw, dict):
        raise ValueError("conf.yaml must define a non-empty 'training:' mapping")
    is_folsom = _dataset_profile_is_folsom(conf)
    keys = FOLSOM_TRAINING_HPARAM_KEYS if is_folsom else TRAINING_HPARAM_KEYS
    missing = sorted(keys - raw.keys())
    if missing:
        ref = "FOLSOM_TRAINING_HPARAM_KEYS" if is_folsom else "TRAINING_HPARAM_KEYS"
        raise KeyError(
            "conf training section missing required key(s): "
            + ", ".join(missing)
            + f" (see {ref} in training/training_conf.py)"
        )
    out = {k: raw[k] for k in keys}
    if isinstance(out["lr"], str):
        out["lr"] = float(out["lr"])

    ss = out["skyimg_spatial_size"]
    if not isinstance(ss, int) or isinstance(ss, bool) or ss < 1:
        raise ValueError("training.skyimg_spatial_size must be a positive integer")
    out["skyimg_spatial_size"] = int(ss)

    if not is_folsom:
        shwc = out["satimg_npy_shape_hwc"]
        if not isinstance(shwc, (list, tuple)) or len(shwc) != 3:
            raise ValueError("training.satimg_npy_shape_hwc must be a length-3 sequence [H, W, C]")
        t = tuple(int(x) for x in shwc)
        if any(x < 1 for x in t):
            raise ValueError("training.satimg_npy_shape_hwc entries must be positive")
        out["satimg_npy_shape_hwc"] = t

    tf = out["pv_train_time_fraction"]
    if isinstance(tf, str):
        tf = float(tf)
    if not isinstance(tf, (int, float)) or isinstance(tf, bool):
        raise ValueError("training.pv_train_time_fraction must be a number in (0, 1)")
    tf = float(tf)
    if not (0.0 < tf < 1.0):
        raise ValueError("training.pv_train_time_fraction must be strictly between 0 and 1")
    out["pv_train_time_fraction"] = tf

    tol = out["test_collect_time_match_tolerance_min"]
    if isinstance(tol, str):
        tol = int(float(tol))
    if not isinstance(tol, int) or isinstance(tol, bool) or tol < 0:
        raise ValueError("training.test_collect_time_match_tolerance_min must be a non-negative int (minutes)")
    out["test_collect_time_match_tolerance_min"] = int(tol)

    return out


def get_training_paths_from_conf(conf: dict | None = None, project_root: Path | None = None) -> dict[str, str]:
    """
    Resolve data roots under ``paths.data_dir``.

    - Default (Luoyang): ``pv_path``, ``sky_image_path``, ``sat_path`` relative to ``data_dir``;
      returns ``pv_dir``, ``skyimg_dir``, ``satimg_dir``.
    - Folsom (``dataset_profile: folsom``): ``folsom_irradiance_csv`` + ``sky_image_path`` only;
      returns ``pv_dir`` and ``skyimg_dir`` (no ``satimg_dir`` — no satellite data).
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

    if _dataset_profile_is_folsom(conf):
        irr_rel = paths_cfg.get("folsom_irradiance_csv")
        if irr_rel is None or not str(irr_rel).strip():
            raise KeyError("dataset_profile=folsom requires paths.folsom_irradiance_csv")
        irr_path = (data_dir / str(irr_rel).strip()).resolve()
        if not irr_path.is_file():
            raise FileNotFoundError(f"Folsom irradiance CSV not found: {irr_path}")
        pv_dir = irr_path.parent
        sky_dir = (data_dir / _req("sky_image_path")).resolve()
        return {
            "pv_dir": str(pv_dir),
            "skyimg_dir": str(sky_dir),
        }

    pv_dir = (data_dir / _req("pv_path")).resolve()
    sky_dir = (data_dir / _req("sky_image_path")).resolve()
    sat_dir = (data_dir / _req("sat_path")).resolve()

    return {
        "pv_dir": str(pv_dir),
        "skyimg_dir": str(sky_dir),
        "satimg_dir": str(sat_dir),
    }
