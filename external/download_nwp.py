# download_nwp.py is used to download the weather forecast results from the internet and store them in the local directory
# Run once: python download_nwp.py
# Daily at UTC time: python download_nwp.py --schedule  (default 16:00 UTC)

# Or: python download_nwp.py --schedule --utc-time 10:45

# The remote files are updated at 16:00 (UTC+0) every day

import argparse
import re
import time
import yaml
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import urlopen, Request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))
from config_utils import get_resolved_paths

CONF_PATH = PROJECT_ROOT / "config" / "conf.yaml"

_UTC_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})$")


def parse_utc_time(s: str) -> tuple[int, int]:
    """Parse 'H:MM' or 'HH:MM' into (hour, minute) in 0..23 / 0..59."""
    m = _UTC_TIME_RE.match(s.strip())
    if not m:
        raise ValueError("expected HH:MM (UTC), e.g. 10:45")
    h, mi = int(m.group(1)), int(m.group(2))
    if not (0 <= h <= 23 and 0 <= mi <= 59):
        raise ValueError("hour must be 0-23, minute 0-59")
    return h, mi


def next_utc_run_after(hour: int, minute: int, now: datetime | None = None) -> tuple[datetime, float]:
    """Next occurrence of hour:minute UTC strictly after `now` (default: current UTC)."""
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target, (target - now).total_seconds()


def seconds_until_next_utc_run(hour: int, minute: int) -> float:
    _, sec = next_utc_run_after(hour, minute)
    return sec


def run_daily_utc_main(hour: int, minute: int) -> None:
    """Block forever; call main() every day at hour:minute UTC."""
    next_at, sec = next_utc_run_after(hour, minute)
    print(
        f"Scheduler: main() daily at {hour:02d}:{minute:02d} UTC "
        f"(next: {next_at.strftime('%Y-%m-%d %H:%M:%S')} UTC, in {sec:.0f}s)"
    )
    while True:
        time.sleep(seconds_until_next_utc_run(hour, minute))
        try:
            main()
        except Exception as e:
            print(f"Scheduled run failed: {e}")
        next_at, sec = next_utc_run_after(hour, minute)
        print(f"Next run: {next_at.strftime('%Y-%m-%d %H:%M:%S')} UTC (in {sec:.0f}s)")


def load_conf():
    with open(CONF_PATH) as f:
        return yaml.safe_load(f)


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
    paths = get_resolved_paths(conf, PROJECT_ROOT)
    nwp_conf = conf["nwp"]
    download_conf = conf.get("download", {})

    download_dir = paths["nwp_download"]
    newest_dir = paths["nwp_newest"]
    nwp_config_path = paths["nwp_config"]
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
    parser = argparse.ArgumentParser(description="Download NWP data; optional daily UTC schedule.")
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Run main() every day at --utc-time (UTC); default 16:00 UTC if omitted",
    )
    parser.add_argument(
        "--utc-time",
        metavar="HH:MM",
        default=None,
        help="Daily run time in UTC (default with --schedule: 16:00), e.g. 10:45",
    )
    args = parser.parse_args()
    if args.schedule:
        utc_str = args.utc_time or "16:00"
        try:
            h, m = parse_utc_time(utc_str)
        except ValueError as e:
            parser.error(str(e))
        run_daily_utc_main(h, m)
    else:
        if args.utc_time:
            parser.error("--utc-time is only valid with --schedule")
        main()
