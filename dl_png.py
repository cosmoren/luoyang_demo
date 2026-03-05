#! /usr/bin/env python
import ftplib
import os
import time
from datetime import datetime, timedelta, timezone

# ================= CONFIG =================
FTP_HOST = "ftp.ptree.jaxa.jp"
FTP_USER = "weize.zhang_h-partners.com"
FTP_PASS = "SP+wari8"

BASE_DIR = "/jma/hsd"
DOWNLOAD_DIR = "./satellite"
CHECK_INTERVAL = 600  # 10 minutes
STATE_FILE = "last_downloaded.txt"
# =========================================


# How many hours back to check (recommended: 1)
HOURS_BACK = 1
# =========================================


def load_last_downloaded():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return f.read().strip()
    return None


def save_last_downloaded(filename):
    with open(STATE_FILE, "w") as f:
        f.write(filename)


def connect_ftp():
    ftp = ftplib.FTP(FTP_HOST)
    ftp.login(FTP_USER, FTP_PASS)
    return ftp


def build_utc_paths(hours_back=0):
    """
    Build UTC folder path like /jma/hsd/YYYYMM/DD/HH
    """
    t = datetime.now(timezone.utc) - timedelta(hours=hours_back)

    yyyy = t.strftime("%Y%m")
    dd = t.strftime("%d")
    hh = t.strftime("%H")

    return f"{BASE_DIR}/{yyyy}/{dd}/{hh}"


def get_latest_fldk_file(ftp):
    candidates = []

    for h in range(HOURS_BACK + 1):
        remote_dir = build_utc_paths(h)

        try:
            ftp.cwd(remote_dir)
            files = ftp.nlst()

            fldk_files = [
                f for f in files
                if "FLDK" in f and f.lower().endswith(".png")
            ]

            for f in fldk_files:
                candidates.append((f, remote_dir))

        except ftplib.error_perm:
            # Folder may not exist yet
            continue

    if not candidates:
        return None, None

    # Sort by filename timestamp (newest first)
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0]


def download_file(ftp, remote_dir, filename):
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    local_path = os.path.join(DOWNLOAD_DIR, filename)

    ftp.cwd(remote_dir)
    print ("Downloading file: {filename} to {local_path}")
    with open(local_path, "wb") as f:
        ftp.retrbinary(f"RETR {filename}", f.write)

    print(f"Downloaded: {filename}")

def delete_existing_png():
    if not os.path.exists(DOWNLOAD_DIR):
        return

    for f in os.listdir(DOWNLOAD_DIR):
        if f.lower().endswith(".png") and "FLDK" in f:
            path = os.path.join(DOWNLOAD_DIR, f)
            os.remove(path)
            print(f"Deleted old file: {f}")

def main():
    last_downloaded = load_last_downloaded()

    while True:
        try:
            ftp = connect_ftp()
            latest_file, remote_dir = get_latest_fldk_file(ftp)

            if latest_file:
                if latest_file != last_downloaded:
                    delete_existing_png()
                    download_file(ftp, remote_dir, latest_file)
                    save_last_downloaded(latest_file)
                    last_downloaded = latest_file
                else:
                    print("Latest file already downloaded.")
            else:
                print("No FLDK file found.")

            ftp.quit()

        except Exception as e:
            print(f"Error: {e}")
        print (f"Sleeping for {CHECK_INTERVAL} seconds...")
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()