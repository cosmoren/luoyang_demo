"""
Offline inference simulator GUI.
Updates every 5 seconds. Current time is simulated (not computer time).
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

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
    df = pd.read_csv(csv_path, skiprows=3)
    df["time"] = pd.to_datetime(df["time"])
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
    openmeteo_path: str | Path | None = None,
):
    """Create offline simulator GUI with simulated time (not computer time)."""
    historical_data = load_historical_data(openmeteo_path)
    
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

    # Plots (figure scales with display)
    fig_dpi = max(80, min(150, int(100 * scale)))
    fig = Figure(figsize=(8, 6), dpi=fig_dpi)
    ax_top = fig.add_subplot(2, 1, 1)
    ax_bot = fig.add_subplot(2, 1, 2)
    fig.tight_layout()
    fig.autofmt_xdate()

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH, padx=(pad_main * 2, pad_main), pady=pad_main)

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
            root.after(0, lambda: tick(historical_data))

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

    def tick(data: pd.DataFrame):
        nonlocal count
        idx = min(count, len(data) - 1)
        t0 = pd.Timestamp(data["time"].iloc[idx]).to_pydatetime()
        input_window_4h = create_input_window(data, count, num=16)
        input_window_48h = create_input_window(data, count, num=48 * 4)

        # Ground truth at t0 (x = t0)
        gt_4h = input_window_4h[0, 3]
        gt_4h_trajectory.append((t0, gt_4h))

        gt_48h = float(input_window_48h[0, 3])
        gt_48h_trajectory.append((t0, gt_48h))

        # Run the prediction model
        pred_4h = model_4h(input_window_4h)
        t_pred_4h = t0 + timedelta(hours=4)
        pred_4h_trajectory.append((t_pred_4h, pred_4h))

        pred_48h = model_48h(input_window_48h)
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
        ax_top.set_ylim(-20, 600)
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
        ax_bot.set_ylim(-20, 600)
        ax_bot.axvline(t0 + timedelta(hours=48), color="red", linestyle="--", alpha=0.8, label="Now+48hours")
        ax_bot.set_ylabel("Output (MW)")
        ax_bot.set_title("Day-ahead (t0 − 24 h to t0 + 50 h)")
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

        canvas.draw()
        count += 1
        if running[0]:
            after_id[0] = root.after(GUI_UPDATE_MS, lambda: tick(historical_data))

    # Initial draw
    t0 = get_sim_time()
    t_top_start = t0 - timedelta(hours=2)
    t_top_end = t0 + timedelta(hours=4.5)
    t_bot_start = t0 - timedelta(hours=24)
    t_bot_end = t0 + timedelta(hours=50)
    setup_axis_time_range(ax_top, t_top_start, t_top_end, t0, tick_interval_minutes=30)
    ax_top.set_ylim(-20, 600)
    ax_top.axvline(t0 + timedelta(hours=4), color="red", linestyle="--", alpha=0.8, label="Now+4hours")
    ax_top.set_ylabel("Output (MW)")
    ax_top.set_title("Intra-day (t0 - 4.5 h to t0 + 4.5 h)")
    ax_top.legend(loc="upper right")
    setup_axis_time_range(ax_bot, t_bot_start, t_bot_end, t0, tick_interval_hours=6)
    ax_bot.set_ylim(-20, 600)
    ax_bot.axvline(t0 + timedelta(hours=48), color="red", linestyle="--", alpha=0.8, label="Now+48hours")
    ax_bot.set_ylabel("Output (MW)")
    ax_bot.set_title("Day-ahead (t0 − 24 h to t0 + 50 h)")
    ax_bot.legend(loc="upper right")
    canvas.draw()

    root.mainloop()


if __name__ == "__main__":
    # Path relative to script location (works from /luoyang_demo or /luoyang_demo/inference)
    create_infer_offline_gui(openmeteo_path="../datasets/open-meteo-34.75N112.25E222m.csv")
