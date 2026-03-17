"""
Capture images from USB camera at integer minutes and save to skyimg_download dir.
Run continuously (e.g. in background or cron every minute); captures once per minute at :00 seconds.
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

try:
    import cv2
except ImportError:
    cv2 = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config_utils import get_resolved_paths

CONF_PATH = PROJECT_ROOT / "config" / "conf.yaml"


def load_conf():
    with open(CONF_PATH) as f:
        return yaml.safe_load(f)


def open_camera(camera_id=0):
    """Open USB camera. Returns cv2.VideoCapture or None on failure."""
    if cv2 is None:
        raise ImportError("opencv-python is required: pip install opencv-python")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        return None
    return cap


def capture_at_integer_minutes(save_dir: Path, camera_id: int = 0):
    """
    Run loop: at each integer minute (seconds == 0), capture one frame from USB camera
    and save to save_dir with filename YYYYMMDD_HHMMSS.jpg.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = open_camera(camera_id)
    if cap is None:
        raise RuntimeError(f"Cannot open USB camera (id={camera_id}). Check connection and permissions.")
    # Prefer only the latest frame in buffer so each read() is newest (not all drivers support this)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def crop_square_h_keep(frame):
        """Crop to square: height unchanged, width = height (center crop or pad)."""
        h, w = frame.shape[:2]
        if w >= h:
            x = (w - h) // 2
            return frame[0:h, x : x + h]
        pad = (h - w) // 2
        return cv2.copyMakeBorder(frame, 0, 0, pad, h - w - pad, cv2.BORDER_REPLICATE)

    last_capture_minute = None
    try:
        while True:
            # Always read and discard so buffer stays at latest frame (every second we drain one)
            cap.grab()
            now = datetime.now(timezone.utc)
            minute_key = (now.year, now.month, now.day, now.hour, now.minute)
            if now.second == 0 and minute_key != last_capture_minute:
                # Retrieve the frame we just grabbed (or grab+retrieve for freshest)
                ret, frame = cap.retrieve()
                if not ret:
                    ret, frame = cap.read()
                if not ret:
                    print("Warning: failed to read frame from camera", flush=True)
                else:
                    frame = crop_square_h_keep(frame)
                    frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                    name = now.strftime("%Y%m%d_%H%M%S") + ".jpg"
                    path = save_dir / name
                    if cv2.imwrite(str(path), frame):
                        print(f"Saved: {path}", flush=True)
                    else:
                        print(f"Failed to write: {path}", flush=True)
                last_capture_minute = minute_key
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopped by user", flush=True)
    finally:
        cap.release()


def main():
    conf = load_conf()
    paths = get_resolved_paths(conf, PROJECT_ROOT)
    save_dir = paths.get("skyimg_download")
    if not save_dir:
        raise SystemExit("conf.yaml paths.skyimg_download is not set")
    save_dir = Path(save_dir)
    camera_id = int(conf.get("camera", {}).get("camera_id", 1))
    print(f"Sky images will be saved to: {save_dir}", flush=True)
    print("Capturing at integer minutes (UTC). Press Ctrl+C to stop.", flush=True)
    capture_at_integer_minutes(save_dir, camera_id=camera_id)


if __name__ == "__main__":
    main()
