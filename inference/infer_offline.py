"""
Offline inference simulator GUI.
Updates every 5 seconds. Current time is simulated (not computer time).
"""

import sys
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

        canvas.draw()
        count += 1
        if running[0]:
            after_id[0] = root.after(GUI_UPDATE_MS, lambda: tick(historical_data, X_4h_flat, y_4h_flat, power_4h_gt, \
                X_48h_flat, y_48h_flat, power_48h_gt))

    # Initial draw
    t0 = get_sim_time()
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
    canvas.draw()

    root.mainloop()


if __name__ == "__main__":
    # Path relative to script location (works from /luoyang_demo or /luoyang_demo/inference)
    # "/net/storage-1/home/w84179850/canadianlab/weize/data/luoyang_agg.csv" # for weize's linux
    # "..\datasets\luoyang_agg.csv" # for windows
    create_infer_offline_gui(processed_data_path=Path("../datasets/luoyang_agg.csv"))

