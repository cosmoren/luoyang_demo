import os
import time
import shutil

from PIL import Image

# Source directories
SRC_ASI = "/home/hw_xql_antispoof_skyimager/skyimg/asi16/asi_16613/"
SRC_WYLC = "/home/hw_xql_antispoof_skyimager/skyimg/wylc/"

# Destination directories
DST_ASI = "/home/pvforecaster/workspace/data/skyimg/asi_16613"
DST_WYLC = "/home/pvforecaster/workspace/data/skyimg/wylc"

MAX_FILES = 30
IMG_SIZE = (224, 224)


def resize_jpg_to_dst(src_path, dst_path, size=IMG_SIZE):
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS
    with Image.open(src_path) as img:
        img = img.convert("RGB")
        img = img.resize(size, resample)
        img.save(dst_path, "JPEG", quality=95)


def get_latest_date_folder(base_dir):
    folders = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    if not folders:
        return None

    folders.sort(reverse=True)
    return os.path.join(base_dir, folders[0])


def get_latest_timestamp_asi(files):
    timestamps = set()
    for f in files:
        if f.endswith("_11.jpg") or f.endswith("_12.jpg"):
            ts = f.split("_")[0]
            timestamps.add(ts)

    if not timestamps:
        return None

    return sorted(timestamps, reverse=True)[0]


def get_latest_file_by_prefix(folder, prefix, ext):
    candidates = []

    for f in os.listdir(folder):
        if f.startswith(prefix) and f.endswith(ext):
            candidates.append(f)

    if not candidates:
        return None

    candidates.sort(reverse=True)  # timestamp in filename
    return os.path.join(folder, candidates[0])


def copy_latest_asi(src_base, dst_dir):
    folder = get_latest_date_folder(src_base)
    if not folder:
        return

    files = os.listdir(folder)
    latest_ts = get_latest_timestamp_asi(files)

    if not latest_ts:
        return

    for suffix in ["_11.jpg", "_12.jpg"]:
        filename = latest_ts + suffix
        src_path = os.path.join(folder, filename)
        dst_path = os.path.join(dst_dir, filename)

        if os.path.exists(src_path) and not os.path.exists(dst_path):
            resize_jpg_to_dst(src_path, dst_path)
            print(f"Saved ASI (224x224): {filename}")


def copy_latest_wylc(src_base, dst_dir):
    folder = get_latest_date_folder(src_base)
    if not folder:
        return

    # Latest JPG
    latest_jpg = get_latest_file_by_prefix(folder, "ori", ".jpg")
    if latest_jpg:
        filename = os.path.basename(latest_jpg)
        dst_path = os.path.join(dst_dir, filename)

        if not os.path.exists(dst_path):
            resize_jpg_to_dst(latest_jpg, dst_path)
            print(f"Saved WYLC JPG (224x224): {filename}")

    # Latest CSV
    latest_csv = get_latest_file_by_prefix(folder, "output", ".csv")
    if latest_csv:
        filename = os.path.basename(latest_csv)
        dst_path = os.path.join(dst_dir, filename)

        if not os.path.exists(dst_path):
            shutil.copy2(latest_csv, dst_path)
            print(f"Copied WYLC CSV: {filename}")


def cleanup_files(folder, extension, basename_suffix=None):
    """Keep only latest MAX_FILES per group (extension; optional basename suffix)."""
    files = []
    ext = extension.lower()
    suff = basename_suffix.lower() if basename_suffix else None
    for f in os.listdir(folder):
        if not f.lower().endswith(ext):
            continue
        if suff is not None and not f.lower().endswith(suff):
            continue
        files.append(os.path.join(folder, f))

    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    for f in files[MAX_FILES:]:
        os.remove(f)
        tag = basename_suffix or extension
        print(f"Deleted old {tag}: {os.path.basename(f)}")


def main_loop():
    while True:
        try:
            copy_latest_asi(SRC_ASI, DST_ASI)
            copy_latest_wylc(SRC_WYLC, DST_WYLC)

            # ASI: keep 30 x _11.jpg and 30 x _12.jpg (60 total)
            cleanup_files(DST_ASI, ".jpg", "_11.jpg")
            cleanup_files(DST_ASI, ".jpg", "_12.jpg")

            cleanup_files(DST_WYLC, ".jpg")
            cleanup_files(DST_WYLC, ".csv")

        except Exception as e:
            print(f"Error: {e}")

        time.sleep(60)


if __name__ == "__main__":
    main_loop()
