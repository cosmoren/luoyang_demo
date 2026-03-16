"""
Online inference GUI with two time-series plot windows.
Top: t0 - 15 min to t0 + 4.5 hours.
Bottom: t0 - 2 hours to t0 + 48 hours.
"""

from datetime import datetime, timedelta, timezone

import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk


def setup_axis_time_range(
    ax, t_start, t_end, t0, tick_interval_minutes: int | None = None, tick_interval_hours: int | None = None
):
    """Set x-axis time range and format."""
    ax.set_xlim(t_start, t_end)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M", tz=timezone.utc))
    if tick_interval_minutes is not None:
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=tick_interval_minutes))
    else:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=tick_interval_hours or 1))
    ax.set_xlabel("Time (UTC)")
    ax.grid(True, alpha=0.3)
    ax.axvline(t0, color="gray", linestyle="--", alpha=0.7, label="Now")


def create_infer_online_gui():
    """Create GUI with two stacked time-series plot windows."""
    root = tk.Tk()
    root.title("Inference Online")
    root.geometry("900x600")

    t0 = datetime.now(timezone.utc)

    # Top bar: current time display (read-only) - UTC and local
    top_frame = tk.Frame(root, padx=10, pady=5)
    top_frame.pack(fill=tk.X)
    tk.Label(top_frame, text="Current time (UTC):", font=("", 10)).pack(side=tk.LEFT)
    utc_var = tk.StringVar(value=t0.strftime("%Y-%m-%d %H:%M:%S"))
    tk.Entry(top_frame, width=24, font=("", 11), state="readonly", textvariable=utc_var).pack(
        side=tk.LEFT, padx=(5, 15)
    )
    tk.Label(top_frame, text="Current time (Local):", font=("", 10)).pack(side=tk.LEFT, padx=(0, 5))
    local_var = tk.StringVar(value=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"))
    tk.Entry(top_frame, width=28, font=("", 11), state="readonly", textvariable=local_var).pack(
        side=tk.LEFT, padx=(0, 15)
    )
    site_label = f"Site: {site.get('name', '—')}  lat={latitude}  lon={longitude}"
    tk.Label(top_frame, text=site_label, font=("", 9)).pack(side=tk.LEFT)

    def update_time():
        utc_var.set(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
        local_var.set(datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"))
        root.after(1000, update_time)

    root.after(1000, update_time)

    # Top: t0 - 15 min to t0 + 4.5 hours
    t_top_start = t0 - timedelta(minutes=15)
    t_top_end = t0 + timedelta(hours=4.5)

    # Bottom: t0 - 2 hours to t0 + 48 hours
    t_bot_start = t0 - timedelta(hours=2)
    t_bot_end = t0 + timedelta(hours=48)

    fig = Figure(figsize=(8, 6), dpi=100)
    ax_top = fig.add_subplot(2, 1, 1)
    ax_bot = fig.add_subplot(2, 1, 2)

    setup_axis_time_range(ax_top, t_top_start, t_top_end, t0, tick_interval_minutes=15)
    ax_top.set_ylabel("Value")
    ax_top.set_title("Short-term (t0 - 15 min to t0 + 4.5 h)")

    setup_axis_time_range(ax_bot, t_bot_start, t_bot_end, t0, tick_interval_hours=6)
    ax_bot.set_ylabel("Value")
    ax_bot.set_title("Long-term (t0 − 2 h to t0 + 48 h)")

    fig.tight_layout()
    fig.autofmt_xdate()

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    root.mainloop()


if __name__ == "__main__":
    create_infer_online_gui()
