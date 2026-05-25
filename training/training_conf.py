"""Load config YAML training section and resolved data paths for CLI defaults."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from config_utils import get_resolved_paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONF_PATH = PROJECT_ROOT / "config" / "datasets" / "conf_luoyang.yaml"
FOLSOM_CONF_PATH = PROJECT_ROOT / "config" / "datasets" / "conf_folsom.yaml"

YLJ_TRAINING_EXTRA_KEYS = frozenset({
    "t_off_min",
    "pv_output_rand",
    "pv_value_column",
    "pv_value_scale",
    "clear_sky_ratio_max_valid",
})

TRAINING_HPARAM_KEYS = frozenset({
    "csv_interval_min",
    "pv_input_interval_min",
    "pv_output_interval_min",
    "pv_input_len",
    "pv_output_len",
    "pv_train_time_fraction",
    "test_anchor_stride_min",
    "val_anchor_stride_min",
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

# Legacy: Luoyang-style keys minus satimg (still uses ``pv_*`` — do not use for ``conf_folsom.yaml``).
FOLSOM_TRAINING_HPARAM_KEYS = TRAINING_HPARAM_KEYS - frozenset(
    {"satimg_window_size", "satimg_time_resolution_min", "satimg_npy_shape_hwc"}
)

# Folsom irradiance YAML (``irr_*`` horizons; no ``pv_*`` / no ``satimg_*``).
FOLSOM_IRR_TRAINING_HPARAM_KEYS = frozenset({
    "csv_interval_min",
    "irr_input_interval_min",
    "irr_output_interval_min",
    "irr_input_len",
    "irr_output_len",
    "irr_train_time_fraction",
    "test_anchor_stride_min",
    "test_collect_time_match_tolerance_min",
    "skyimg_window_size",
    "skyimg_time_resolution_min",
    "skyimg_spatial_size",
    "epochs",
    "lr",
    "batch_size",
    "save_every",
    "num_workers",
    "train_max_batches_per_epoch",
})


def _dataset_profile_is_folsom(conf: dict) -> bool:
    return str(conf.get("dataset_profile", "")).strip().lower() == "folsom"


def _site_is_ylj(conf: dict) -> bool:
    site = conf.get("site", {})
    return str(site.get("name", "")).strip().lower() == "ylj"


def set_config_path(path: str | Path) -> None:
    global CONF_PATH
    CONF_PATH = Path(path).expanduser().resolve()


def bootstrap_config_from_argv(argv: list[str] | None = None) -> None:
    """Apply ``--config PATH`` before other modules call :func:`load_config`."""
    args = sys.argv if argv is None else argv
    for i, arg in enumerate(args):
        if arg == "--config" and i + 1 < len(args):
            set_config_path(args[i + 1])
            print(f"[config] using config: {CONF_PATH}")
            return
        if arg.startswith("--config="):
            set_config_path(arg.split("=", 1)[1])
            print(f"[config] using config: {CONF_PATH}")
            return


def load_config(config_path: str | Path | None = None) -> dict:
    path = Path(config_path) if config_path is not None else CONF_PATH
    with open(path) as f:
        return yaml.safe_load(f)


def load_config_path(path: Path | str | None = None) -> dict:
    """Load a YAML config file (default: main Luoyang ``conf.yaml``)."""
    p = Path(path) if path is not None else CONF_PATH
    with open(p) as f:
        return yaml.safe_load(f)


def get_training_hparams_from_conf(conf: dict | None = None) -> dict:
    """Load ``conf['training']`` (or ``conf['sampling']`` for Luoyang); keys depend on site/profile."""
    if conf is None:
        conf = load_config()
    raw = conf.get("training")
    if not isinstance(raw, dict):
        raw = conf.get("sampling")
    if not isinstance(raw, dict):
        raise ValueError("conf must define a non-empty 'training:' or 'sampling:' mapping")
    is_folsom = _dataset_profile_is_folsom(conf)
    keys = FOLSOM_IRR_TRAINING_HPARAM_KEYS if is_folsom else TRAINING_HPARAM_KEYS
    if _site_is_ylj(conf):
        keys = keys | YLJ_TRAINING_EXTRA_KEYS
    missing = sorted(keys - raw.keys())
    if missing:
        ref = "FOLSOM_IRR_TRAINING_HPARAM_KEYS" if is_folsom else "TRAINING_HPARAM_KEYS"
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

    if is_folsom:
        tf = out["irr_train_time_fraction"]
        if isinstance(tf, str):
            tf = float(tf)
        if not isinstance(tf, (int, float)) or isinstance(tf, bool):
            raise ValueError("training.irr_train_time_fraction must be a number in (0, 1)")
        tf = float(tf)
        if not (0.0 < tf < 1.0):
            raise ValueError("training.irr_train_time_fraction must be strictly between 0 and 1")
        out["irr_train_time_fraction"] = tf
    else:
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

    if _site_is_ylj(conf):
        off = out["t_off_min"]
        if isinstance(off, str):
            off = int(float(off))
        if not isinstance(off, int) or isinstance(off, bool) or off < 0:
            raise ValueError("training.t_off_min must be a non-negative int (minutes)")
        out["t_off_min"] = int(off)
        out["pv_output_rand"] = bool(out["pv_output_rand"])
        col = str(out["pv_value_column"]).strip()
        if col not in ("active_power", "clear_sky_ratio"):
            raise ValueError("training.pv_value_column must be 'active_power' or 'clear_sky_ratio'")
        out["pv_value_column"] = col
        pvs = out["pv_value_scale"]
        if isinstance(pvs, str):
            pvs = float(pvs)
        if not isinstance(pvs, (int, float)) or isinstance(pvs, bool) or pvs <= 0:
            raise ValueError("training.pv_value_scale must be a positive number")
        out["pv_value_scale"] = float(pvs)
        csr = out["clear_sky_ratio_max_valid"]
        if isinstance(csr, str):
            csr = float(csr)
        if not isinstance(csr, (int, float)) or isinstance(csr, bool) or csr <= 0:
            raise ValueError("training.clear_sky_ratio_max_valid must be a positive number")
        out["clear_sky_ratio_max_valid"] = float(csr)

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
    out: dict[str, str] = {
        "pv_dir": str(pv_dir),
        "skyimg_dir": str(sky_dir),
        "satimg_dir": str(sat_dir),
    }
    nwp_rel = paths_cfg.get("nwp_path")
    if nwp_rel is not None and str(nwp_rel).strip():
        out["nwp_dir"] = str((data_dir / str(nwp_rel).strip()).resolve())
    raw_parquet = resolved.get("ylj_raw_parquet_dir")
    if raw_parquet is not None:
        out["ylj_raw_parquet_dir"] = str(raw_parquet)
    return out
