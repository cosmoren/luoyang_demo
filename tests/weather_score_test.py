"""Plot station-level weather_score from a per-devDn aggregated CSV."""

import os
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path(os.path.expanduser("~/datasets/kkk"))
CSV_NAME = "NE_333620909.csv"
OUT_PATH = Path(__file__).resolve().parent / "weather_score_NE_333620909.png"

# One week window (UTC collectTime)
WINDOW_START = "2025-03-01 00:00:00" # "2025-07-01 00:00:00"
WINDOW_END = "2025-03-20 00:00:00" # "2025-07-07 00:00:00"


def main() -> None:
    csv_path = DATA_DIR / CSV_NAME
    if not csv_path.is_file():
        # fallback: luoyang_data_626 from earlier batch run
        alt = Path(os.path.expanduser("~/datasets/luoyang_data_626")) / CSV_NAME
        if alt.is_file():
            csv_path = alt
        else:
            raise FileNotFoundError(f"CSV not found: {csv_path} or {alt}")

    df = pd.read_csv(csv_path, parse_dates=["collectTime"])
    sub = df[(df["collectTime"] >= WINDOW_START) & (df["collectTime"] < WINDOW_END)].copy()
    if sub.empty:
        raise ValueError(f"No rows in [{WINDOW_START}, {WINDOW_END})")

    t = sub["collectTime"]
    ws = sub["weather_score"].astype(float)
    pv = sub["active_power"].astype(float)
    p_cs = sub["p_cs"].astype(float)
    daytime = p_cs > 0.1

    fig, ax1 = plt.subplots(figsize=(14, 4))
    ax1.plot(t, ws, color="#2563eb", linewidth=0.8, label="weather_score")
    ax1.fill_between(t, 0, ws, color="#2563eb", alpha=0.12)
    ax1.scatter(
        t[~daytime], ws[~daytime], s=4, c="#94a3b8", alpha=0.5, label="night (p_cs≤0.1)"
    )
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_ylabel("weather_score (high=noisy/cloudy)", color="#2563eb")
    ax1.tick_params(axis="y", labelcolor="#2563eb")

    ax2 = ax1.twinx()
    ax2.plot(t, pv, color="#ea580c", linewidth=0.8, alpha=0.85, label="active_power (PV)")
    ax2.set_ylabel("active_power (kW)", color="#ea580c")
    ax2.tick_params(axis="y", labelcolor="#ea580c")

    ax1.set_xlabel("collectTime (UTC)")
    ax1.set_title(
        f"{csv_path.name} — weather_score & PV ({WINDOW_START[:10]} ~ {WINDOW_END[:10]})"
    )
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)

    ws_day = ws[daytime]
    pv_day = pv[daytime]
    print(f"CSV: {csv_path}")
    print(f"Saved: {OUT_PATH}")
    print(
        f"Daytime weather_score: min={ws_day.min():.3f}, max={ws_day.max():.3f}, "
        f"mean={ws_day.mean():.3f}, n={len(ws_day)}"
    )
    print(
        f"Daytime active_power: min={pv_day.min():.1f}, max={pv_day.max():.1f}, "
        f"mean={pv_day.mean():.1f} kW"
    )


if __name__ == "__main__":
    main()
