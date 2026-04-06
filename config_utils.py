"""
Shared config and path resolution for luoyang_demo.
Resolves path entries in conf.yaml against data_dir or project_root to avoid repetition.
"""

from pathlib import Path
from typing import Optional

# Keys under paths that are relative to data_dir when value is relative
DATA_DIR_KEYS = (
    "sat_download",
    "nwp_download",
    "nwp_newest",
    "pv_download",
    "skyimg_download",
    "skyimg_pred",
)

# All other path keys are relative to project_root when value is relative


def get_resolved_paths(conf: dict, project_root: Path) -> dict[str, Optional[Path]]:
    """
    Resolve all path entries in conf["paths"].
    - For keys in DATA_DIR_KEYS: relative values are resolved as data_dir / value.
    - For other path keys: relative values are resolved as project_root / value.
    - Absolute values are returned as-is (as Path). None values are passed through.
    """
    paths = conf.get("paths", {})
    raw_data_dir = paths.get("data_dir")
    if raw_data_dir is None:
        data_dir: Optional[Path] = None
    else:
        data_dir = Path(raw_data_dir)
        if not data_dir.is_absolute():
            data_dir = (project_root / data_dir).resolve()

    result: dict[str, Optional[Path]] = {}
    for key, value in paths.items():
        if value is None:
            result[key] = None
            continue
        p = Path(value)
        if p.is_absolute():
            result[key] = p
        elif key in DATA_DIR_KEYS and data_dir is not None:
            result[key] = (data_dir / p).resolve()
        else:
            result[key] = (project_root / p).resolve()
    return result


def get_infer_online_data_paths(conf: dict, project_root: Path) -> dict[str, Path]:
    """
    Paths for inference/infer_online.py: NC (sat), sky current/pred, NWP solar/wind.

    - NC: data_dir / sat_download (paths.sat_download)
    - Sky asi: skyimg_download / sky_image_asi_16613_subpath; pred: skyimg_pred / sky_image_asi_16613_subpath
    - Sky wylc: skyimg_download / sky_image_wylc_subpath; pred: skyimg_pred / same (optional sky_image_wylc_pred_subpath under data_dir)
    - NWP: data_dir / nwp_download / nwp.solar_subdir | wind_subdir
    """
    resolved = get_resolved_paths(conf, project_root)
    paths_cfg = conf.get("paths", {})
    nwp_cfg = conf.get("nwp", {})

    sat = resolved.get("sat_download")
    if sat is None:
        raise ValueError("conf paths.sat_download is required (NC processed directory)")

    sky_base = resolved.get("skyimg_download")
    sky_pred_base = resolved.get("skyimg_pred")
    sub = paths_cfg.get("sky_image_asi_16613_subpath", "")
    if not str(sub).strip():
        raise ValueError(
            "conf paths.sky_image_asi_16613_subpath is required "
            "(e.g. asi16/asi_16613/20260331), relative under skyimg_download / skyimg_pred"
        )
    sub_rel = Path(sub)
    if sky_base is None or sky_pred_base is None:
        raise ValueError("conf paths.skyimg_download and skyimg_pred are required")
    sky_image = (sky_base / sub_rel).resolve()
    sky_image_pred = (sky_pred_base / sub_rel).resolve()

    data_dir = resolved.get("data_dir")
    if data_dir is None:
        raise ValueError("conf paths.data_dir is required for optional sky_image_wylc_pred_subpath")

    wylc_sub_raw = paths_cfg.get("sky_image_wylc_subpath", "")
    if not str(wylc_sub_raw).strip():
        raise ValueError(
            "conf paths.sky_image_wylc_subpath is required "
            "(relative under skyimg_download / skyimg_pred, e.g. wylc)"
        )
    wylc_rel = Path(wylc_sub_raw)
    img_leaf = Path(paths_cfg.get("skyimg_download", "skyimg")).name
    if wylc_rel.parts and wylc_rel.parts[0] == img_leaf:
        wylc_rel = Path(*wylc_rel.parts[1:])
    if not wylc_rel.parts:
        raise ValueError(
            "conf paths.sky_image_wylc_subpath must not be only the skyimg_download folder name; "
            "use a subpath such as wylc"
        )
    sky_image_wylc = (sky_base / wylc_rel).resolve()

    wylc_pred_raw = paths_cfg.get("sky_image_wylc_pred_subpath", "")
    if str(wylc_pred_raw).strip():
        sky_image_wylc_pred = (data_dir / Path(wylc_pred_raw)).resolve()
    else:
        sky_image_wylc_pred = (sky_pred_base / wylc_rel).resolve()

    nwp_base = resolved.get("nwp_download")
    if nwp_base is None:
        raise ValueError("conf paths.nwp_download is required")
    solar_sub = nwp_cfg.get("solar_subdir", "solar")
    wind_sub = nwp_cfg.get("wind_subdir", "wind")
    nwp_solar = (nwp_base / solar_sub).resolve()
    nwp_wind = (nwp_base / wind_sub).resolve()

    return {
        "nc_processed": sat,
        "sky_image": sky_image,
        "sky_image_pred": sky_image_pred,
        "sky_image_wylc": sky_image_wylc,
        "sky_image_wylc_pred": sky_image_wylc_pred,
        "nwp_solar": nwp_solar,
        "nwp_wind": nwp_wind,
    }
