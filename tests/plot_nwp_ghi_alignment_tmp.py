"""
Temporary plotting script for aligning and comparing NWP GHI_mean series.

It compares:
1) /work/datasets/luoyang_2026_SPMF/NWP/112.285_34.700_UTC0_solar_forecast.csv
2) /work/datasets/luoyang_2026_SPMF/NWP_history/112.285_34.700_UTC0_solar_obs.csv

By default:
- For forecast, pick the latest start_time for each dtime.
- Align with history by exact dtime match.
- Plot the first 10 days from aligned data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_FORECAST_CSV = Path(
    "/work/datasets/luoyang_2026_SPMF/NWP/112.285_34.700_UTC0_solar_forecast.csv"
)
DEFAULT_HISTORY_CSV = Path(
    "/work/datasets/luoyang_2026_SPMF/NWP_history/112.285_34.700_UTC0_solar_obs.csv"
)
DEFAULT_OUT = Path("tests/tmp_ghi_forecast_vs_history_first10days.png")


def _load_forecast(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["start_time", "dtime", "GHI_mean"])
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["dtime"] = pd.to_datetime(df["dtime"], errors="coerce")
    df["GHI_mean"] = pd.to_numeric(df["GHI_mean"], errors="coerce")
    df = df.dropna(subset=["start_time", "dtime", "GHI_mean"]).copy()
    # Keep one forecast value per target timestamp: latest initialization time.
    df = df.sort_values(["dtime", "start_time"]).drop_duplicates(subset=["dtime"], keep="last")
    return df[["dtime", "GHI_mean"]].rename(columns={"GHI_mean": "GHI_mean_forecast"})


def _load_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["dtime", "GHI_mean"])
    df["dtime"] = pd.to_datetime(df["dtime"], errors="coerce")
    df["GHI_mean"] = pd.to_numeric(df["GHI_mean"], errors="coerce")
    df = df.dropna(subset=["dtime", "GHI_mean"]).copy()
    return df[["dtime", "GHI_mean"]].rename(columns={"GHI_mean": "GHI_mean_history"})


def build_aligned_frame(forecast_csv: Path, history_csv: Path) -> pd.DataFrame:
    fcst = _load_forecast(forecast_csv)
    hist = _load_history(history_csv)
    merged = pd.merge(fcst, hist, on="dtime", how="inner").sort_values("dtime")
    return merged


def plot_first_days(df: pd.DataFrame, days: int, out_path: Path) -> None:
    if df.empty:
        raise ValueError("Aligned dataframe is empty, cannot plot.")

    t0 = df["dtime"].iloc[0]
    tend = t0 + pd.Timedelta(days=days)
    cut = df[df["dtime"] < tend].copy()
    if cut.empty:
        raise ValueError("No rows left for requested time window.")

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax2 = ax1.twinx()

    l1 = ax1.plot(
        cut["dtime"], cut["GHI_mean_forecast"], color="tab:blue", linewidth=1.0, label="forecast GHI_mean"
    )
    l2 = ax2.plot(
        cut["dtime"], cut["GHI_mean_history"], color="tab:orange", linewidth=1.0, label="history GHI_mean"
    )

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Forecast GHI_mean", color="tab:blue")
    ax2.set_ylabel("History GHI_mean", color="tab:orange")
    ax1.tick_params(axis="y", colors="tab:blue")
    ax2.tick_params(axis="y", colors="tab:orange")
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"Aligned GHI_mean (first {days} days)")

    lines = l1 + l2
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper right")

    fig.autofmt_xdate()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot aligned forecast/history GHI_mean for first N days.")
    parser.add_argument("--forecast-csv", type=Path, default=DEFAULT_FORECAST_CSV)
    parser.add_argument("--history-csv", type=Path, default=DEFAULT_HISTORY_CSV)
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    aligned = build_aligned_frame(args.forecast_csv, args.history_csv)
    plot_first_days(aligned, args.days, args.out)
    print(
        f"Saved plot to: {args.out}\n"
        f"aligned_rows={len(aligned)} "
        f"time_range=[{aligned['dtime'].iloc[0]} -> {aligned['dtime'].iloc[-1]}]"
    )


if __name__ == "__main__":
    main()
