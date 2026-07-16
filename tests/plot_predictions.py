"""
Plot pv_pred_kW vs pv_true_kW from one or two prediction CSV files.

Usage examples
--------------
# Single CSV, time-series:
python tests/plot_predictions.py --csv checkpoints_15m_baseline/pv_forecast_vit_test_predictions_task_15m_gpu0_best.csv

# Two CSVs compared on the same plot:
python tests/plot_predictions.py \
    --csv  checkpoints_15m_baseline/pv_forecast_vit_test_predictions_task_15m_gpu0_best.csv \
    --csv2 checkpoints_new/pv_forecast_vit_test_predictions_task_15m_gpu0_best.csv \
    --label1 baseline --label2 new_model

# Filter by date range (Beijing time):
python tests/plot_predictions.py --start 2026-05-15 --end 2026-05-20

# Daily subplots:
python tests/plot_predictions.py --mode daily --start 2026-05-15 --end 2026-05-20

# Scatter:
python tests/plot_predictions.py --mode scatter

# Daily RMSE bar chart for two CSVs:
python tests/plot_predictions.py --mode daily_rmse --csv file1.csv --csv2 file2.csv
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CSV = _PROJECT_ROOT / "checkpoints_15m_baseline" / "pv_forecast_vit_test_predictions_task_15m_gpu0_best.csv"


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp_bj"])
    df = df.rename(columns={"timestamp_bj": "time"})
    df = df.sort_values("time").reset_index(drop=True)
    df = df[~((df["pv_pred_kW"] == 0) & (df["pv_true_kW"] == 0))].copy()
    return df


def apply_time_filter(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start:
        df = df[df["time"] >= pd.Timestamp(start)]
    if end:
        df = df[df["time"] <= pd.Timestamp(end)]
    return df


def rmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2))


# ── single-CSV plot helpers ────────────────────────────────────────────────────

def _plot_one(df: pd.DataFrame, label: str, title: str, ax: plt.Axes) -> None:
    r = rmse(df["pv_pred_kW"].values, df["pv_true_kW"].values)
    ax.plot(df["time"], df["pv_true_kW"], label=f"true ({label})", linewidth=0.8, alpha=0.85)
    ax.plot(df["time"], df["pv_pred_kW"], label=f"pred ({label})  RMSE={r:.1f}", linewidth=0.8, alpha=0.85)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("kW")
    ax.legend(fontsize=7, loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)


# ── two-CSV plot helpers ───────────────────────────────────────────────────────

def _plot_two(df1: pd.DataFrame, label1: str,
              df2: pd.DataFrame, label2: str,
              title: str, ax: plt.Axes) -> None:
    r1 = rmse(df1["pv_pred_kW"].values, df1["pv_true_kW"].values)
    r2 = rmse(df2["pv_pred_kW"].values, df2["pv_true_kW"].values)
    # True curves — share the ground truth so draw only the first if they're the same
    ax.plot(df1["time"], df1["pv_true_kW"], color="gray", linewidth=0.7, alpha=0.6, label="true")
    ax.plot(df1["time"], df1["pv_pred_kW"], linewidth=0.9, alpha=0.85, label=f"pred {label1}  RMSE={r1:.1f}")
    ax.plot(df2["time"], df2["pv_pred_kW"], linewidth=0.9, alpha=0.85, label=f"pred {label2}  RMSE={r2:.1f}")
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("kW")
    ax.legend(fontsize=7, loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)


# ── mode: timeseries ──────────────────────────────────────────────────────────

def mode_timeseries(df1, label1, df2, label2, save):
    fig, ax = plt.subplots(figsize=(16, 4))
    if df2 is None:
        _plot_one(df1, label1, "All test data", ax)
    else:
        _plot_two(df1, label1, df2, label2, "All test data", ax)
    fig.tight_layout()
    _show_or_save(fig, save)


# ── mode: daily ───────────────────────────────────────────────────────────────

def mode_daily(df1, label1, df2, label2, save):
    df1["date"] = df1["time"].dt.date
    days = sorted(df1["date"].unique())
    if df2 is not None:
        df2["date"] = df2["time"].dt.date

    n = len(days)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4 * nrows), squeeze=False)

    for i, day in enumerate(days):
        ax = axes[i // ncols][i % ncols]
        sub1 = df1[df1["date"] == day]
        if df2 is None:
            _plot_one(sub1, label1, str(day), ax)
        else:
            sub2 = df2[df2["date"] == day]
            _plot_two(sub1, label1, sub2, label2, str(day), ax)

    for j in range(i + 1, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle("Daily PV forecast vs truth", fontsize=12)
    fig.tight_layout()
    _show_or_save(fig, save)


# ── mode: scatter ─────────────────────────────────────────────────────────────

def mode_scatter(df1, label1, df2, label2, save):
    n = 1 if df2 is None else 2
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), squeeze=False)

    for idx, (df, lbl) in enumerate([(df1, label1)] + ([(df2, label2)] if df2 is not None else [])):
        ax = axes[0][idx]
        ax.scatter(df["pv_true_kW"], df["pv_pred_kW"], s=2, alpha=0.3)
        lim = max(df["pv_true_kW"].max(), df["pv_pred_kW"].max()) * 1.05
        ax.plot([0, lim], [0, lim], "r--", linewidth=1, label="y=x")
        r = rmse(df["pv_pred_kW"].values, df["pv_true_kW"].values)
        ax.set_xlabel("pv_true_kW")
        ax.set_ylabel("pv_pred_kW")
        ax.set_title(f"{lbl}  RMSE={r:.1f} kW")
        ax.legend(fontsize=8)

    fig.tight_layout()
    _show_or_save(fig, save)


# ── mode: daily RMSE bars ────────────────────────────────────────────────────

def mode_daily_rmse(df1, label1, df2, label2, save):
    if df2 is None:
        raise ValueError("--mode daily_rmse requires --csv2")

    def _daily_values(df: pd.DataFrame) -> pd.Series:
        dates = df["time"].dt.date
        squared_error = (df["pv_pred_kW"] - df["pv_true_kW"]) ** 2
        return squared_error.groupby(dates).mean().pow(0.5)

    daily1 = _daily_values(df1)
    daily2 = _daily_values(df2)
    days = sorted(set(daily1.index) | set(daily2.index))
    if not days:
        raise ValueError("No data remains after applying the time filter")

    values1 = daily1.reindex(days).to_numpy()
    values2 = daily2.reindex(days).to_numpy()
    relative_change = np.divide(
        values2 - values1,
        values1,
        out=np.full_like(values1, np.nan, dtype=float),
        where=values1 != 0,
    ) * 100.0
    x = np.arange(len(days))
    width = 0.4

    fig_width = max(12, len(days) * 0.55)
    fig, (ax_rmse, ax_change) = plt.subplots(
        2,
        1,
        figsize=(fig_width, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    ax_rmse.bar(x - width / 2, values1, width, label=label1)
    ax_rmse.bar(x + width / 2, values2, width, label=label2)
    ax_rmse.set_ylabel("Daily RMSE (kW)")
    ax_rmse.set_title("Daily PV forecast RMSE")
    ax_rmse.legend()
    ax_rmse.grid(axis="y", linestyle="--", alpha=0.3)

    colors = np.where(relative_change >= 0, "tab:red", "tab:green")
    ax_change.bar(x, relative_change, width=0.7, color=colors)
    ax_change.axhline(0, color="black", linewidth=0.8)
    ax_change.set_xticks(x)
    ax_change.set_xticklabels(
        [pd.Timestamp(day).strftime("%m-%d") for day in days],
        rotation=45,
        ha="right",
    )
    ax_change.set_xlabel("Date (Beijing time)")
    ax_change.set_ylabel(f"{label2} vs {label1} (%)")
    ax_change.set_title("Daily RMSE relative change (positive = higher, negative = lower)")
    ax_change.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    _show_or_save(fig, save)


# ── utils ─────────────────────────────────────────────────────────────────────

def _show_or_save(fig: plt.Figure, save: str | None) -> None:
    if save:
        out = Path(save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    else:
        plt.show()


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else _PROJECT_ROOT / p


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PV prediction CSVs")
    parser.add_argument("--csv",  type=str, default=str(_DEFAULT_CSV), help="First CSV file")
    parser.add_argument("--csv2", type=str, default=None,              help="Second CSV file (optional, for comparison)")
    parser.add_argument("--label1", type=str, default="csv1",  help="Legend label for first CSV")
    parser.add_argument("--label2", type=str, default="csv2",  help="Legend label for second CSV")
    parser.add_argument(
        "--mode",
        choices=["timeseries", "daily", "scatter", "daily_rmse"],
        default="timeseries",
    )
    parser.add_argument("--start", type=str, default=None, help="Start datetime, Beijing time (e.g. 2026-05-15)")
    parser.add_argument("--end",   type=str, default=None, help="End datetime, Beijing time (e.g. 2026-05-20)")
    parser.add_argument("--save",  type=str, default=None, help="Save figure to file instead of showing")
    args = parser.parse_args()

    csv1_path = _resolve(args.csv)
    if not csv1_path.exists():
        print(f"CSV not found: {csv1_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {csv1_path} ...")
    df1 = load_csv(csv1_path)
    df1 = apply_time_filter(df1, args.start, args.end)
    print(f"  {args.label1}: {len(df1)} rows")

    df2 = None
    if args.csv2:
        csv2_path = _resolve(args.csv2)
        if not csv2_path.exists():
            print(f"CSV2 not found: {csv2_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Loading {csv2_path} ...")
        df2 = load_csv(csv2_path)
        df2 = apply_time_filter(df2, args.start, args.end)
        print(f"  {args.label2}: {len(df2)} rows")

    if args.mode == "timeseries":
        mode_timeseries(df1, args.label1, df2, args.label2, args.save)
    elif args.mode == "daily":
        mode_daily(df1, args.label1, df2, args.label2, args.save)
    elif args.mode == "scatter":
        mode_scatter(df1, args.label1, df2, args.label2, args.save)
    elif args.mode == "daily_rmse":
        mode_daily_rmse(df1, args.label1, df2, args.label2, args.save)


if __name__ == "__main__":
    main()
