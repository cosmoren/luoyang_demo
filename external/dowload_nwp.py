# download_nwp.py is used to download the weather forecast results from the internet and store them in the local directory
# It need to be put in the crontab to run periodically
# The remote files are updated at 18:30 (UTC+8) every day

import yaml
import shutil
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen, Request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONF_PATH = PROJECT_ROOT / "config" / "conf.yaml"


def load_conf():
    with open(CONF_PATH) as f:
        return yaml.safe_load(f)


def resolve_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (PROJECT_ROOT / p).resolve()


def download_file(url: str, out_path: Path, user_agent: str) -> None:
    req = Request(url, headers={"User-Agent": user_agent})
    with urlopen(req) as resp:
        out_path.write_bytes(resp.read())
    print(f"Downloaded: {out_path.name}")


def list_nwp_files(root: Path, solar_subdir: str, wind_subdir: str) -> list[Path]:
    """Return relative paths of all NWP files under root (solar/ and wind/)."""
    out = []
    for subdir in (solar_subdir, wind_subdir):
        d = root / subdir
        if d.is_dir():
            for f in d.iterdir():
                if f.is_file():
                    out.append(f.relative_to(root))
    return sorted(out)


def newest_is_empty_or_different(
    download_dir: Path,
    newest_dir: Path,
    solar_subdir: str,
    wind_subdir: str,
) -> bool:
    """True if newest has no NWP files or any file content differs from download."""
    download_files = list_nwp_files(download_dir, solar_subdir, wind_subdir)
    if not download_files:
        return True
    newest_files = list_nwp_files(newest_dir, solar_subdir, wind_subdir)
    if not newest_files:
        return True
    for rel in download_files:
        down_path = download_dir / rel
        new_path = newest_dir / rel
        if not new_path.exists():
            return True
        if down_path.read_bytes() != new_path.read_bytes():
            return True
    return False


def copy_download_to_newest(
    download_dir: Path,
    newest_dir: Path,
    solar_subdir: str,
    wind_subdir: str,
) -> None:
    """Overwrite files in newest with files from download (same relative paths)."""
    update_time = datetime.now(timezone.utc)
    updated_files: list[tuple[str, datetime]] = []

    for subdir in (solar_subdir, wind_subdir):
        src_d = download_dir / subdir
        dst_d = newest_dir / subdir
        if not src_d.is_dir():
            continue
        dst_d.mkdir(parents=True, exist_ok=True)
        for f in src_d.iterdir():
            if f.is_file():
                dst_file = dst_d / f.name
                shutil.copy2(f, dst_file)
                mtime = datetime.fromtimestamp(dst_file.stat().st_mtime, tz=timezone.utc)
                updated_files.append((f"{subdir}/{f.name}", mtime))
                print(f"Updated: {subdir}/{f.name}")

    # Write update time log in nwp_newest (times in UTC with timezone)
    log_path = newest_dir / "update_time.txt"
    with open(log_path, "w") as out:
        out.write(f"Last update: {update_time.isoformat()}\n\n")
        out.write("Files in nwp_newest:\n")
        for rel_path, mtime in sorted(updated_files):
            out.write(f"  {rel_path}  {mtime.isoformat()}\n")
    print(f"Wrote update log: {log_path}")



def main():
    conf = load_conf()
    paths = conf["paths"]
    nwp_conf = conf["nwp"]
    download_conf = conf.get("download", {})

    download_dir = resolve_path(paths["nwp_download"])
    newest_dir = resolve_path(paths["nwp_newest"])
    nwp_config_path = resolve_path(paths["nwp_config"])
    user_agent = download_conf.get("user_agent", "Mozilla/5.0")

    solar_subdir = nwp_conf.get("solar_subdir", "solar")
    wind_subdir = nwp_conf.get("wind_subdir", "wind")

    with open(nwp_config_path) as f:
        nwp = yaml.safe_load(f)

    # 1. Download to nwp_download
    for kind, subdir in (("solar", solar_subdir), ("wind", wind_subdir)):
        out_dir = download_dir / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        for url in nwp[kind]:
            filename = url.rstrip("/").split("/")[-1]
            out_path = out_dir / filename
            download_file(url, out_path, user_agent)

    # 2. Compare with nwp_newest; replace if empty or content differs
    if newest_is_empty_or_different(
        download_dir, newest_dir, solar_subdir, wind_subdir
    ):
        copy_download_to_newest(
            download_dir, newest_dir, solar_subdir, wind_subdir
        )
        print(f"Done. Updated {newest_dir} with downloaded files.")
    else:
        print("The remote files are not updated.")


if __name__ == "__main__":
    main()
