"""
Offline inference simulator GUI.
Updates every 5 seconds. Current time is simulated (not computer time).
"""

import random
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from xml.parsers.expat import model

# Add project root so models.models can be imported
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from models.models import model_4h, model_48h
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from pathlib import Path
import xgboost as xgb

# Optional: load images for sky panels (PNG/JPG)
try:
    from PIL import Image, ImageTk
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# Dummy directory for sky images (contains many images named by timestamp)
# Example filenames: 2024-01-15_14-30.png, 20240115143000.png
SKY_IMAGE_DIR = Path("../sky_images")

# Filename patterns to parse timestamp (without extension); try in order
_SKY_IMAGE_TIME_FMTS = [
    "%Y-%m-%d_%H-%M",      # 2024-01-15_14-30
    "%Y-%m-%d_%H-%M-%S",  # 2024-01-15_14-30-00
    "%Y%m%d_%H%M%S",      # 20240115143000
    "%Y%m%d_%H%M",        # 202401151430
]


def _parse_sky_image_stem(stem: str) -> datetime | None:
    """Parse sky image filename stem; supports YYYYMMDDHHMMSS_11 / _12 and legacy patterns."""
    m = re.match(r"^(\d{14})_(11|12)$", stem)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
        except ValueError:
            return None
    for fmt in _SKY_IMAGE_TIME_FMTS:
        try:
            return datetime.strptime(stem, fmt)
        except ValueError:
            continue
    return None


def _find_sky_image_for_time(dir_path: Path, t: datetime) -> Path | None:
    """Find the sky image file that best matches the given timestamp.
    Looks for exact or nearest match in dir_path. Returns None if dir missing or no match.
    For YYYYMMDDHHMMSS_11 / _12, timestamps coincide—pick one at random among ties."""
    if not dir_path.is_dir():
        return None
    candidates: list[Path] = []
    best_delta: float | None = None
    t_ts = t.timestamp()
    for f in dir_path.iterdir():
        if not f.is_file() or f.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        parsed = _parse_sky_image_stem(f.stem)
        if parsed is None:
            continue
        delta = abs(parsed.timestamp() - t_ts)
        if best_delta is None or delta < best_delta:
            best_delta = delta
            candidates = [f]
        elif delta == best_delta:
            candidates.append(f)
    if not candidates:
        return None
    return random.choice(candidates)


def _load_sky_image(path: Path, max_size: tuple[int, int] = (200, 150)):
    """Load image from path for display; return PhotoImage or None if not found/invalid."""
    if not path.exists() or not _PIL_AVAILABLE:
        return None
    try:
        img = Image.open(path).convert("RGB")
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        img.thumbnail(max_size, resample)
        return ImageTk.PhotoImage(img)
    except Exception:
        return None


def _preload_sky_images(dir_path: Path, max_size: tuple[int, int] = (64, 64)):
    """
    Load all valid sky images from dir_path into memory once.
    Returns (times, images) where times is a sorted list of datetimes
    and images is a list of PhotoImage objects aligned with times.
    """
    if not dir_path.is_dir() or not _PIL_AVAILABLE:
        return [], []

    by_time: defaultdict[datetime, list[Path]] = defaultdict(list)
    for f in sorted(dir_path.iterdir()):
        if not f.is_file() or f.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        parsed_time = _parse_sky_image_stem(f.stem)
        if parsed_time is None:
            continue
        by_time[parsed_time].append(f)

    items = []
    for parsed_time in sorted(by_time.keys()):
        f = random.choice(by_time[parsed_time])
        ph = _load_sky_image(f, max_size=max_size)
        if ph is None:
            continue
        items.append((parsed_time, ph))

    items.sort(key=lambda x: x[0])
    times = [t for t, _ in items]
    images = [img for _, img in items]
    return times, images


def _get_sky_image_for_time(times, images, t: datetime):
    """
    Given a sorted list of times and aligned images, find the image
    with timestamp closest to t. Returns None if no images loaded.
    """
    if not times:
        return None

    # Binary search for insertion point
    lo, hi = 0, len(times)
    while lo < hi:
        mid = (lo + hi) // 2
        if times[mid] < t:
            lo = mid + 1
        else:
            hi = mid

    if lo == 0:
        return images[0]
    if lo == len(times):
        return images[-1]

    before_t = times[lo - 1]
    after_t = times[lo]
    if (t - before_t) <= (after_t - t):
        return images[lo - 1]
    return images[lo]
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from demo_luoyang import LuoyangDataLoader
# Simulator: advance simulated time by this amount each 5-sec GUI tick
SIM_TIME_STEP = timedelta(minutes=5)
GUI_UPDATE_MS = 100 # 1000
LOCAL_TZ = timezone(timedelta(hours=8))  # UTC+8 for x-axis display

def setup_axis_time_range(
    ax, t_start, t_end, t0, tick_interval_minutes: int | None = None, tick_interval_hours: int | None = None
):
    """Set x-axis time range and format (local time UTC+8)."""
    ax.set_xlim(t_start, t_end)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M", tz=LOCAL_TZ))
    if tick_interval_minutes is not None:
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=tick_interval_minutes))
    else:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=tick_interval_hours or 1))
    ax.set_xlabel("Time (Local UTC+8)")
    ax.grid(True, alpha=0.3)
    ax.axvline(t0, color="gray", linestyle="--", alpha=0.7, label="Now")

def load_historical_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load Open-Meteo historical/forecast CSV (15-min interval)."""
    if csv_path is None:
        csv_path = Path(__file__).resolve().parent.parent / "datasets" / "open-meteo-34.75N112.25E222m.csv"
    csv_path = Path(csv_path)
    # Skip metadata rows (first 3); row 4 has column headers
    df = pd.read_csv(csv_path)
    return df




def create_input_window(df: pd.DataFrame, count: int, num: int = 16) -> np.ndarray:
    INPUT_WINDOW_COLUMNS = [
        "temperature_2m (°C)",
        "precipitation (mm)",
        "relative_humidity_2m (%)",
        "shortwave_radiation (W/m²)",
        "shortwave_radiation_instant (W/m²)",
        "wind_speed_10m (km/h)",
        "wind_direction_10m (°)",
    ]
    """Select the past num elements ending at count (exclusive); return selected columns as numpy array."""
    start = max(0, count - num)
    df_window = df.iloc[start:count][INPUT_WINDOW_COLUMNS]
    return df_window.to_numpy(dtype=np.float32)


def create_infer_offline_gui(
    initial_time: datetime | None = None,
    time_step: timedelta = SIM_TIME_STEP,
    processed_data_path: str | Path | None = None,
):
    """Create offline simulator GUI with simulated time (not computer time)."""
    historical_data = load_historical_data(processed_data_path)
    
    ### added by weize
    loader = LuoyangDataLoader(
        
        feature_cols=[
            "total_active_kw",
            "mean_inner_temp",
            "status_ok"
        ],
        add_time_encoding=True,
        df=historical_data,
        solar_forecast_path="/data/luoyang_demo/datasets/112.285_34.700_UTC0_model_solar_v5.csv",
        wind_forecast_path="/data/luoyang_demo/datasets/112.285_34.700_UTC0_model_wind_v5.csv",
        forecast_feature_config={
            "solar": [
                "ssrd"
            ],
            "wind": [
                "t2m",
                #"msl",
                #"u10",
                #"v10",
                #"u100",
                #"v100",
                #"wind_speed_10m"
            ]
        },
    )

    # X, y, t = loader.make_ultra_short_dataset(history_len=32) # ultra short
    X_4h, y_4h, curr_gt_4h, t_4h = loader.make_short_dataset(history_len=64, horizon_hours=4) # short
    # X, y, t = loader.make_long_dataset(history_len=96, anchor_hour=9) # long
    X_48h, y_48h, curr_gt_48h, t_48h = loader.make_sequence_dataset(history_len=96, horizon_steps=192) # windowed-long
    model_4h_path = (
        "../datasets/luoyang_agg_short_xgb.json"
        #"D:\workspace\luoyang_demo_v2\models\luoyang_agg_short_xgb.json"
    )
    model_48h_path = (
        "../datasets/luoyang_agg_windowed-long_xgb.json"
    )

    # ---------------------------------------------------------
    # 展平 X
    # ---------------------------------------------------------

    N_4h = X_4h.shape[0]
    X_4h_flat = X_4h.reshape(N_4h, -1)
    N_48h = X_48h.shape[0]
    X_48h_flat = X_48h.reshape(N_48h, -1)

    # ---------------------------------------------------------
    # 构造监督样本
    # ---------------------------------------------------------

    # -------------------------------
    # ultra short / short
    # -------------------------------
    y_4h_flat = y_4h
    t_4h_flat = pd.to_datetime(t_4h)

    # -------------------------------
    # is_sequence
    # 每天一个样本 -> n_seq个点
    # 展开成 N*seq 个样本
    # -------------------------------

    y_list = []
    t_list = []
    X_list = []

    for i in range(N_48h):
        base_time = pd.to_datetime(t_48h[i])
        start_time = base_time + pd.Timedelta(minutes=loader.freq_minutes)
        times = pd.date_range(
            start_time,
            periods=y_48h.shape[1],
            freq=f"{loader.freq_minutes}min"
        )

        for k in range(y_48h.shape[1]):
            X_list.append(X_48h_flat[i])
            y_list.append(y_48h[i, k])
            t_list.append(times[k])

    X_48h_flat = np.asarray(X_list, dtype=np.float32)
    y_48h_flat = np.asarray(y_list, dtype=np.float32)
    t_48h_flat = pd.to_datetime(t_list)

    power_4h_gt = curr_gt_4h # [N]
    power_48h_gt = curr_gt_48h # [N]

    model_4h = xgb.XGBRegressor()   # 分类任务 XGBClassifier
    model_4h.load_model(model_4h_path)
    model_48h = xgb.XGBRegressor()   # 分类任务 XGBClassifier
    model_48h.load_model(model_48h_path)
    ###

    root = tk.Tk()
    root.title("Inference Offline (Simulator)")

    # Scale UI for different screens (DPI / resolution)
    try:
        dpi_scale = root.tk.call("tk", "scaling")
    except Exception:
        dpi_scale = 1.0
    scale = max(0.8, min(2.0, float(dpi_scale)))
    try:
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{min(900, int(w * 0.7))}x{min(650, int(h * 0.7))}")
        root.minsize(500, 400)
    except Exception:
        root.geometry("900x650")

    font_time = ("", max(10, int(14 * scale)))
    font_btn = ("", max(16, int(20 * scale)))
    pad_main = max(8, int(12 * scale))
    pad_btn = max(10, int(16 * scale))

    # Simulated time from historical_data; count indexes the current timestep
    count = 24*4
    n_rows = len(historical_data)

    # Top bar: simulated time display (read-only) - UTC, local, count
    top_frame = tk.Frame(root, padx=pad_main, pady=pad_main // 2)
    top_frame.pack(fill=tk.X)
    def get_sim_time():
        idx = min(count, n_rows - 1)
        return pd.Timestamp(historical_data["time"].iloc[idx]).to_pydatetime()

    sim_time = get_sim_time()
    tk.Label(top_frame, text="Simulated time (UTC):", font=font_time).pack(side=tk.LEFT)
    utc_var = tk.StringVar(value=sim_time.strftime("%Y-%m-%d %H:%M:%S"))
    tk.Entry(top_frame, width=28, font=font_time, state="readonly", textvariable=utc_var).pack(
        side=tk.LEFT, padx=(5, pad_main)
    )
    tk.Label(top_frame, text="Simulated time (UTC+8):", font=font_time).pack(side=tk.LEFT, padx=(0, 5))
    local_var = tk.StringVar(value=(sim_time + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S"))
    tk.Entry(top_frame, width=28, font=font_time, state="readonly", textvariable=local_var).pack(
        side=tk.LEFT, padx=(0, pad_main)
    )

    # Plots: upper row = sky images (left) + 4h plot (right); lower row = 48h plot
    fig_dpi = max(80, min(150, int(100 * scale)))
    fig_top = Figure(figsize=(6, 2.5), dpi=fig_dpi)
    ax_top = fig_top.add_subplot(1, 1, 1)
    fig_top.tight_layout()
    fig_top.autofmt_xdate()
    fig_bot = Figure(figsize=(8, 3), dpi=fig_dpi)
    ax_bot = fig_bot.add_subplot(1, 1, 1)
    fig_bot.tight_layout()
    fig_bot.autofmt_xdate()

    # Left panel: two sky image slots (upper row); updated by simulated time
    img_size = (200, 150)
    sky_photo_refs: list = []  # keep refs so images are not garbage-collected
    sky_times, sky_images = _preload_sky_images(SKY_IMAGE_DIR, max_size=img_size)

    def make_sky_slot(parent: tk.Frame, label_text: str, rows: int = 1, cols: int = 1):
        """Create a slot with a grid of rows x cols image labels (row-major order)."""
        frame = tk.Frame(parent, relief=tk.GROOVE, borderwidth=1, padx=4, pady=4)
        tk.Label(frame, text=label_text, font=font_time).pack()
        grid_frame = tk.Frame(frame)
        grid_frame.pack(pady=4)
        content_labels = []
        for r in range(rows):
            row_f = tk.Frame(grid_frame)
            row_f.pack()
            for c in range(cols):
                lbl = tk.Label(row_f, text="", font=font_time, fg="gray")
                lbl.pack(side=tk.LEFT, padx=2)
                content_labels.append(lbl)
        frame.content_labels = content_labels
        return frame

    def update_sky_images(t0: datetime):
        """Set current (8 images, 2x4) and predicted (8 images, 2x4) sky images using preloaded data."""
        nonlocal sky_photo_refs
        sky_photo_refs.clear()
        interval_min = 15

        # Current sky: most recent 8 images, 2 rows x 4 cols (rightmost = current time t0)
        # Times: t0-7*15min .. t0-15min, t0 (row-major: row0 then row1)
        current_times = [
            t0 - timedelta(minutes=k * interval_min) for k in range(7, -1, -1)
        ]
        current_photos = []
        for t in current_times:
            ph = _get_sky_image_for_time(sky_times, sky_images, t)
            if ph is not None:
                sky_photo_refs.append(ph)
                current_photos.append(ph)
        for i, lbl in enumerate(slot_current.content_labels):
            if i < len(current_photos):
                lbl.config(image=current_photos[i], text="")
            else:
                lbl.config(image="", text="")

        # Predicted sky: next 8 images starting after current (t0+15min .. t0+8*15min), 2 rows x 4 cols
        pred_times = [
            t0 + timedelta(minutes=k * interval_min) for k in range(1, 9)
        ]
        pred_photos = []
        for t in pred_times:
            ph = _get_sky_image_for_time(sky_times, sky_images, t)
            if ph is not None:
                sky_photo_refs.append(ph)
                pred_photos.append(ph)
        for i, lbl in enumerate(slot_predicted.content_labels):
            if i < len(pred_photos):
                lbl.config(image=pred_photos[i], text="")
            else:
                lbl.config(image="", text="")

        # Lower row panels: same images as upper row
        for i, lbl in enumerate(slot_current_bot.content_labels):
            if i < len(current_photos):
                lbl.config(image=current_photos[i], text="")
            else:
                lbl.config(image="", text="")
        for i, lbl in enumerate(slot_predicted_bot.content_labels):
            if i < len(pred_photos):
                lbl.config(image=pred_photos[i], text="")
            else:
                lbl.config(image="", text="")

    content = tk.Frame(root, padx=pad_main, pady=pad_main)
    content.pack(expand=True, fill=tk.BOTH)
    content.grid_columnconfigure(0, weight=0, minsize=1)   # left: image panels
    content.grid_columnconfigure(1, weight=1)             # right: plots
    content.grid_rowconfigure(0, weight=1)                # upper row (4h)
    content.grid_rowconfigure(1, weight=0, minsize=4)    # separator bar
    content.grid_rowconfigure(2, weight=1)                # lower row (48h)

    # Upper row: sky image panels (left) + 4h plot (right), aligned height
    top_row = tk.Frame(content)
    top_row.grid(row=0, column=0, columnspan=2, sticky="nsew")
    top_row.grid_columnconfigure(0, weight=0, minsize=1)
    top_row.grid_columnconfigure(1, weight=1)
    top_row.grid_rowconfigure(0, weight=1)
    left_panel = tk.Frame(top_row)
    left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, pad_main))
    left_panel.grid_rowconfigure(0, weight=0)
    left_panel.grid_rowconfigure(1, weight=0)
    slot_current = make_sky_slot(left_panel, "Current sky images", rows=2, cols=4)
    slot_current.grid(row=0, column=0, pady=(0, pad_main // 2), sticky="nw")
    slot_predicted = make_sky_slot(left_panel, "Predicted sky images", rows=2, cols=4)
    slot_predicted.grid(row=1, column=0, sticky="nw")
    canvas_top = FigureCanvasTkAgg(fig_top, master=top_row)
    canvas_top.get_tk_widget().grid(row=0, column=1, sticky="nsew")

    # Separator bar between upper and lower visualization
    sep = tk.Frame(content, height=4, bg="gray75", relief=tk.FLAT)
    sep.grid(row=1, column=0, columnspan=2, sticky="ew", pady=pad_main // 2)

    # Lower row: sky image panels (left) + 48h plot (right), aligned height
    bottom_row = tk.Frame(content)
    bottom_row.grid(row=2, column=0, columnspan=2, sticky="nsew")
    bottom_row.grid_columnconfigure(0, weight=0, minsize=1)
    bottom_row.grid_columnconfigure(1, weight=1)
    bottom_row.grid_rowconfigure(0, weight=1)
    left_panel_bot = tk.Frame(bottom_row)
    left_panel_bot.grid(row=0, column=0, sticky="nsew", padx=(0, pad_main))
    left_panel_bot.grid_rowconfigure(0, weight=0)
    left_panel_bot.grid_rowconfigure(1, weight=0)
    slot_current_bot = make_sky_slot(left_panel_bot, "Current sky images", rows=2, cols=4)
    slot_current_bot.grid(row=0, column=0, pady=(0, pad_main // 2), sticky="nw")
    slot_predicted_bot = make_sky_slot(left_panel_bot, "Predicted sky images", rows=2, cols=4)
    slot_predicted_bot.grid(row=1, column=0, sticky="nw")
    canvas_bot = FigureCanvasTkAgg(fig_bot, master=bottom_row)
    canvas_bot.get_tk_widget().grid(row=0, column=1, sticky="nsew")

    # Start/Stop button at bottom
    running = [False]
    after_id: list[object] = [None]

    def toggle_start_stop():
        if running[0]:
            running[0] = False
            if after_id[0] is not None:
                root.after_cancel(after_id[0])
                after_id[0] = None
            start_btn.config(text="Start", bg="#4CAF50")
        else:
            running[0] = True
            start_btn.config(text="Stop", bg="#E53935")
            root.after(0, lambda: tick(historical_data, X_4h_flat, y_4h_flat, power_4h_gt, \
                X_48h_flat, y_48h_flat, power_48h_gt))

    btn_frame = tk.Frame(root, pady=pad_btn)
    btn_frame.pack(fill=tk.X)
    start_btn = tk.Button(
        btn_frame,
        text="Start",
        font=font_btn,
        padx=pad_btn * 2,
        pady=pad_btn,
        command=toggle_start_stop,
        bg="#4CAF50",
        fg="white",
        relief=tk.RAISED,
        cursor="hand2",
    )
    start_btn.pack(pady=pad_btn // 2)

    # Trajectory of pred_4h: list of (t0+4h, pred_4h) as time advances (points move left)
    pred_4h_trajectory: list[tuple[datetime, float]] = []

    gt_4h_trajectory: list[tuple[datetime, float]] = []
    # Bottom window: pred_48h (x=t0+48h), gt_48h (x=t0)
    pred_48h_trajectory: list[tuple[datetime, float]] = []
    gt_48h_trajectory: list[tuple[datetime, float]] = []

    def tick(data: pd.DataFrame, X_4h=None, y_4h=None, power_4h_gt=None,
        X_48h=None, y_48h=None, power_48h_gt=None):
        nonlocal count
        idx = min(count, len(data) - 1)
        t0 = pd.Timestamp(data["time"].iloc[idx]).to_pydatetime()

        update_sky_images(t0)

        # Ground truth at t0 (x = t0)
        gt_4h = power_4h_gt[count:count+1]/1e3 #input_window_4h[0, 3]
        gt_4h_trajectory.append((t0, gt_4h))

        gt_48h = 0
        gt_48h_trajectory.append((t0, gt_48h))

        # Run the prediction model
        #pred_4h = model_4h(input_window_4h)
        pred_4h = model_4h.predict(X_4h_flat[count:count+1])/1e3
        t_pred_4h = t0 + timedelta(hours=4)
        pred_4h_trajectory.append((t_pred_4h, pred_4h))

        pred_48h = 0
        t_pred_48h = t0 + timedelta(hours=48)
        pred_48h_trajectory.append((t_pred_48h, pred_48h))

        utc_var.set(t0.strftime("%Y-%m-%d %H:%M:%S"))
        local_var.set((t0 + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S"))

        t_top_start = t0 - timedelta(hours=2)
        t_top_end = t0 + timedelta(hours=4.5)
        t_bot_start = t0 - timedelta(hours=24)
        t_bot_end = t0 + timedelta(hours=50)

        ax_top.clear()
        ax_bot.clear()
        setup_axis_time_range(ax_top, t_top_start, t_top_end, t0, tick_interval_minutes=30)
        ax_top.set_ylim(-1, 10)
        ax_top.axvline(t0 + timedelta(hours=4), color="red", linestyle="--", alpha=0.8, label="Now+4hours")
        ax_top.set_ylabel("Output (MW)")
        ax_top.set_title("Intra-day (t0 - 2 h to t0 + 4.5 h)")
        # pred_4h trajectory: x = t0+4h for each past prediction (moves left as t0 advances)
        while pred_4h_trajectory and pred_4h_trajectory[0][0] < t_top_start - timedelta(minutes=30):
            pred_4h_trajectory.pop(0)
        if pred_4h_trajectory:
            t_vals = [p[0] for p in pred_4h_trajectory]
            y_vals = [p[1] for p in pred_4h_trajectory]
            ax_top.plot(t_vals, y_vals, color="C1", marker="o", ms=4, lw=1.5, label="pred_4h")
        # gt_4h trajectory: x = t0 for each step (moves left as t0 advances)
        while gt_4h_trajectory and gt_4h_trajectory[0][0] < t_top_start - timedelta(minutes=30):
            gt_4h_trajectory.pop(0)
        if gt_4h_trajectory:
            t_vals = [p[0] for p in gt_4h_trajectory]
            y_vals = [p[1] for p in gt_4h_trajectory]
            ax_top.plot(t_vals, y_vals, color="C0", marker="s", ms=4, lw=1.5, label="gt_4h")
        ax_top.legend(loc="upper right")
        setup_axis_time_range(ax_bot, t_bot_start, t_bot_end, t0, tick_interval_hours=6)
        ax_bot.set_ylim(-1, 10)
        ax_bot.axvline(t0 + timedelta(hours=48), color="red", linestyle="--", alpha=0.8, label="Now+48hours")
        ax_bot.set_ylabel("Output (MW)")
        # ax_bot.set_title("Day-ahead (t0 − 24 h to t0 + 50 h)")
        # pred_48h trajectory: x = t0+48h (moves left as t0 advances)
        while pred_48h_trajectory and pred_48h_trajectory[0][0] < t_bot_start - timedelta(hours=2):
            pred_48h_trajectory.pop(0)
        if pred_48h_trajectory:
            t_vals = [p[0] for p in pred_48h_trajectory]
            y_vals = [p[1] for p in pred_48h_trajectory]
            ax_bot.plot(t_vals, y_vals, color="C1", marker="o", ms=4, lw=1.5, label="pred_48h")
        # gt_48h trajectory: x = t0 (moves left as t0 advances)
        while gt_48h_trajectory and gt_48h_trajectory[0][0] < t_bot_start - timedelta(hours=2):
            gt_48h_trajectory.pop(0)
        if gt_48h_trajectory:
            t_vals = [p[0] for p in gt_48h_trajectory]
            y_vals = [p[1] for p in gt_48h_trajectory]
            ax_bot.plot(t_vals, y_vals, color="C0", marker="s", ms=4, lw=1.5, label="gt_48h")
        ax_bot.legend(loc="upper right")

        canvas_top.draw()
        canvas_bot.draw()
        count += 1
        if running[0]:
            after_id[0] = root.after(GUI_UPDATE_MS, lambda: tick(historical_data, X_4h_flat, y_4h_flat, power_4h_gt, \
                X_48h_flat, y_48h_flat, power_48h_gt))

    # Initial draw
    t0 = get_sim_time()
    update_sky_images(t0)
    t_top_start = t0 - timedelta(hours=2)
    t_top_end = t0 + timedelta(hours=4.5)
    t_bot_start = t0 - timedelta(hours=24)
    t_bot_end = t0 + timedelta(hours=50)
    setup_axis_time_range(ax_top, t_top_start, t_top_end, t0, tick_interval_minutes=30)
    ax_top.set_ylim(-1, 10)
    ax_top.axvline(t0 + timedelta(hours=4), color="red", linestyle="--", alpha=0.8, label="Now+4hours")
    ax_top.set_ylabel("Output (MW)")
    ax_top.set_title("Intra-day (t0 - 4.5 h to t0 + 4.5 h)")
    ax_top.legend(loc="upper right")
    setup_axis_time_range(ax_bot, t_bot_start, t_bot_end, t0, tick_interval_hours=6)
    ax_bot.set_ylim(-1, 10)
    ax_bot.axvline(t0 + timedelta(hours=48), color="red", linestyle="--", alpha=0.8, label="Now+48hours")
    ax_bot.set_ylabel("Output (MW)")
    # ax_bot.set_title("Day-ahead (t0 − 24 h to t0 + 50 h)")
    ax_bot.legend(loc="upper right")
    canvas_top.draw()
    canvas_bot.draw()

    root.mainloop()


if __name__ == "__main__":
    # Path relative to script location (works from /luoyang_demo or /luoyang_demo/inference)
    # "/net/storage-1/home/w84179850/canadianlab/weize/data/luoyang_agg.csv" # for weize's linux
    # "..\datasets\luoyang_agg.csv" # for windows
    create_infer_offline_gui(processed_data_path=Path("../datasets/luoyang_agg.csv"))

