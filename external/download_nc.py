#! /usr/bin/env python
import ftplib
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone

import yaml

# ================= CONFIG =================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config_utils import get_resolved_paths

CONF_PATH = PROJECT_ROOT / "config" / "conf.yaml"
PASSWORDS_PATH = PROJECT_ROOT / "config" / "passwords.yaml"

BASE_DIR = "/pub/himawari/L2/CLP/010"

CHECK_INTERVAL = 600  # 10 minutes
MAX_FILES = 24
HOURS_BACK_INIT = 6   # how far back to scan on first run
# =========================================


def load_conf():
    with open(CONF_PATH) as f:
        return yaml.safe_load(f)


def load_ftp_credentials():
    """Read FTP host, user, password from config/passwords.yaml (jaxa_ftp section)."""
    if not PASSWORDS_PATH.exists():
        raise SystemExit(f"Missing {PASSWORDS_PATH}. Add jaxa_ftp.host, jaxa_ftp.user, jaxa_ftp.password.")
    with open(PASSWORDS_PATH) as f:
        data = yaml.safe_load(f) or {}
    ftp = data.get("jaxa_ftp") or {}
    host = (ftp.get("host") or "").strip()
    user = (ftp.get("user") or "").strip()
    password = (ftp.get("password") or "").strip()
    if not host or not user or not password:
        raise SystemExit("Set jaxa_ftp.host, jaxa_ftp.user, jaxa_ftp.password in config/passwords.yaml")
    return host, user, password


def connect_ftp(host, user, password):
    ftp = ftplib.FTP(host)
    ftp.login(user, password)
    return ftp


def parse_time_from_name(fname):
    """
    NC_H09_YYYYMMDD_HHMM_L2CLP010_FLDK.02401_02401.nc
    """
    try:
        parts = fname.split("_")
        dt_str = parts[2] + parts[3]  # YYYYMMDDHHMM
        return datetime.strptime(dt_str, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def build_remote_dir(dt):
    """
    /YYYYMM/DD/HH
    """
    yyyymm = dt.strftime("%Y%m")
    dd = dt.strftime("%d")
    hh = dt.strftime("%H")
    return f"{BASE_DIR}/{yyyymm}/{dd}/{hh}"


def list_remote_files(ftp, hours_back):
    """
    List FLDK nc files from the last N UTC hours
    """
    files = []
    now = datetime.now(timezone.utc)

    for h in range(hours_back):
        dt = now - timedelta(hours=h)
        remote_dir = build_remote_dir(dt)

        try:
            ftp.cwd(remote_dir)
            names = ftp.nlst()

            for name in names:
                if name.endswith(".nc") and "FLDK" in name:
                    t = parse_time_from_name(name)
                    if t:
                        files.append((t, name, remote_dir))

        except ftplib.error_perm:
            # directory may not exist yet
            continue

    return files


def download_file(ftp, remote_dir, filename, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, filename)

    ftp.cwd(remote_dir)
    with open(local_path, "wb") as f:
        ftp.retrbinary(f"RETR {filename}", f.write)

    print(f"Downloaded: {filename}")


def local_files(local_dir):
    files = []
    if not os.path.exists(local_dir):
        return files

    for f in os.listdir(local_dir):
        if f.endswith(".nc") and "FLDK" in f:
            t = parse_time_from_name(f)
            if t:
                files.append((t, f))
    return files


def delete_oldest_local(local_dir):
    files = local_files(local_dir)
    if len(files) <= MAX_FILES:
        return

    files.sort()  # oldest first
    while len(files) > MAX_FILES:
        _, fname = files.pop(0)
        os.remove(os.path.join(local_dir, fname))
        print(f"Deleted oldest: {fname}")


def initial_download(local_dir, ftp_host, ftp_user, ftp_pass):
    ftp = connect_ftp(ftp_host, ftp_user, ftp_pass)
    files = list_remote_files(ftp, HOURS_BACK_INIT)

    files.sort(reverse=True)  # newest first
    for t, name, rdir in files[:MAX_FILES]:
        download_file(ftp, rdir, name, local_dir)

    ftp.quit()


def check_for_updates(local_dir, ftp_host, ftp_user, ftp_pass):
    ftp = connect_ftp(ftp_host, ftp_user, ftp_pass)

    # Look back far enough to catch multiple new files
    remote_files = list_remote_files(ftp, hours_back=4)
    remote_files.sort()  # oldest → newest

    local = local_files(local_dir)
    local_times = {t for t, _ in local}

    new_files = [
        (t, name, rdir)
        for t, name, rdir in remote_files
        if t not in local_times
    ]

    if not new_files:
        print("No new files.")
        ftp.quit()
        return

    for t, name, rdir in new_files:
        download_file(ftp, rdir, name, local_dir)

    delete_oldest_local(local_dir)
    ftp.quit()


def main():
    conf = load_conf()
    paths = get_resolved_paths(conf, PROJECT_ROOT)
    local_dir = paths["sat_download"]
    ftp_host, ftp_user, ftp_pass = load_ftp_credentials()
    os.makedirs(local_dir, exist_ok=True)

    if len(local_files(local_dir)) < MAX_FILES:
        print("Initial download...")
        initial_download(local_dir, ftp_host, ftp_user, ftp_pass)

    while True:
        try:
            check_for_updates(local_dir, ftp_host, ftp_user, ftp_pass)
        except Exception as e:
            print(f"Error: {e}")

        time.sleep(CHECK_INTERVAL)
        print(f"Sleeping for {CHECK_INTERVAL} seconds")


if __name__ == "__main__":
    main()