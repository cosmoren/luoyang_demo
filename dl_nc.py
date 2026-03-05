#! /usr/bin/env python
import ftplib
import os
import time
from datetime import datetime, timedelta, timezone

# ================= CONFIG =================
FTP_HOST = "ftp.ptree.jaxa.jp"
FTP_USER = "weize.zhang_h-partners.com"
FTP_PASS = "SP+wari8"

BASE_DIR = "/pub/himawari/L2/CLP/010"
LOCAL_DIR = "./nc_files"

CHECK_INTERVAL = 600  # 10 minutes
MAX_FILES = 24
HOURS_BACK_INIT = 6   # how far back to scan on first run
# =========================================


def connect_ftp():
    ftp = ftplib.FTP(FTP_HOST)
    ftp.login(FTP_USER, FTP_PASS)
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


def download_file(ftp, remote_dir, filename):
    os.makedirs(LOCAL_DIR, exist_ok=True)
    local_path = os.path.join(LOCAL_DIR, filename)

    ftp.cwd(remote_dir)
    with open(local_path, "wb") as f:
        ftp.retrbinary(f"RETR {filename}", f.write)

    print(f"Downloaded: {filename}")


def local_files():
    files = []
    if not os.path.exists(LOCAL_DIR):
        return files

    for f in os.listdir(LOCAL_DIR):
        if f.endswith(".nc") and "FLDK" in f:
            t = parse_time_from_name(f)
            if t:
                files.append((t, f))
    return files


def delete_oldest_local():
    files = local_files()
    if len(files) <= MAX_FILES:
        return

    files.sort()  # oldest first
    while len(files) > MAX_FILES:
        _, fname = files.pop(0)
        os.remove(os.path.join(LOCAL_DIR, fname))
        print(f"Deleted oldest: {fname}")


def initial_download():
    ftp = connect_ftp()
    files = list_remote_files(ftp, HOURS_BACK_INIT)

    files.sort(reverse=True)  # newest first
    for t, name, rdir in files[:MAX_FILES]:
        download_file(ftp, rdir, name)

    ftp.quit()


def check_for_updates():
    ftp = connect_ftp()

    # Look back far enough to catch multiple new files
    remote_files = list_remote_files(ftp, hours_back=4)
    remote_files.sort()  # oldest → newest

    local = local_files()
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
        download_file(ftp, rdir, name)

    delete_oldest_local()
    ftp.quit()


def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)

    if len(local_files()) < MAX_FILES:
        print("Initial download...")
        initial_download()

    while True:
        try:
            check_for_updates()
        except Exception as e:
            print(f"Error: {e}")

        time.sleep(CHECK_INTERVAL)
        print (f"Sleeping for {CHECK_INTERVAL} seconds")


if __name__ == "__main__":
    main()