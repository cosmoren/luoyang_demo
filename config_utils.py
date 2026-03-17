"""
Shared config and path resolution for luoyang_demo.
Resolves path entries in conf.yaml against data_dir or project_root to avoid repetition.
"""

from pathlib import Path
from typing import Optional

# Keys under paths that are relative to data_dir when value is relative
DATA_DIR_KEYS = ("sat_download", "nwp_download", "nwp_newest", "pv_download", "skyimg_download")

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
