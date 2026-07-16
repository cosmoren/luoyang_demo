"""Plot pv_total and multiple inverter power series over the same time range.

Usage examples
--------------
# Auto-load first 4 inverters from pv/ (one subplot each):
python tests/plot_pv_first10days.py --num-inverters 4 --save out.png

# Explicit inverter list (one row per CSV):
python tests/plot_pv_first10days.py \\
    --inverter-csv NE_333858376 NE_336801378 NE_335885957 \\
    --days 20 --save out.png
"""

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

_DEFAULT_TOTAL = Path("/shared_work/yuan/datasets/luoyang_2026_SPMF/pv_total.csv")
_DEFAULT_PV_DIR = Path("/shared_work/yuan/datasets/luoyang_2026_SPMF/pv")
_DAYS = 20
# 5-min sampling → 288 points/day
_POINTS_PER_DAY = 288


def _load_power_series(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=["dtime", "final_power"])
    df["time_bj"] = pd.to_datetime(df["dtime"])
    return df.sort_values("time_bj").reset_index(drop=True)


def _resolve_inverter_csv(path_arg: str, pv_dir: Path) -> Path:
    p = Path(path_arg)
    if p.is_file():
        return p
    candidate = pv_dir / path_arg
    if candidate.is_file():
        return candidate
    if not path_arg.endswith(".csv"):
        candidate = pv_dir / f"{path_arg}.csv"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Inverter CSV not found: {path_arg} (also tried under {pv_dir})")


def _slice_by_time(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    mask = (df["time_bj"] >= start) & (df["time_bj"] <= end)
    return df.loc[mask].copy()


def _expand_inverter_args(raw: list[str] | None) -> list[str]:
    """Flatten args; allow comma-separated names in one token."""
    if not raw:
        return []
    out: list[str] = []
    for item in raw:
        for part in item.split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def _discover_inverter_csvs(pv_dir: Path, num: int) -> list[str]:
    paths = sorted(pv_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No inverter CSV files under {pv_dir}")
    if num > len(paths):
        raise ValueError(f"--num-inverters={num} but only {len(paths)} CSV files in {pv_dir}")
    return [p.name for p in paths[:num]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot pv_total and multiple inverter CSVs over the same Beijing-time window",
    )
    parser.add_argument("--csv", type=Path, default=_DEFAULT_TOTAL, help="pv_total.csv path")
    parser.add_argument(
        "--inverter-csv",
        type=str,
        nargs="*",
        default=None,
        help=(
            "One or more inverter CSVs (filename, stem, full path, or comma-separated). "
            "Each gets its own subplot row below pv_total."
        ),
    )
    parser.add_argument(
        "--num-inverters",
        type=int,
        default=4,
        help="When --inverter-csv is omitted, load this many CSVs from pv/ (sorted by name). Default: 4",
    )
    parser.add_argument("--pv-dir", type=Path, default=_DEFAULT_PV_DIR, help="Directory of per-inverter CSVs")
    parser.add_argument("--days", type=int, default=_DAYS)
    parser.add_argument("--save", type=str, default=None, help="Output PNG path")
    args = parser.parse_args()

    total = _load_power_series(args.csv)
    n = args.days * _POINTS_PER_DAY
    total_sub = total.iloc[:n].copy()
    if total_sub.empty:
        raise ValueError("No rows in pv_total for the requested window")

    start = total_sub["time_bj"].iloc[0]
    end = total_sub["time_bj"].iloc[-1]

    inverter_args = _expand_inverter_args(args.inverter_csv)
    if not inverter_args:
        inverter_args = _discover_inverter_csvs(args.pv_dir, args.num_inverters)
        print(f"Auto-selected {len(inverter_args)} inverter CSV(s) from {args.pv_dir}:")
        for name in inverter_args:
            print(f"  {name}")

    inverter_series: list[tuple[str, pd.DataFrame]] = []
    for inverter_arg in inverter_args:
        inv_path = _resolve_inverter_csv(inverter_arg, args.pv_dir)
        inverter_sub = _slice_by_time(_load_power_series(inv_path), start, end)
        if inverter_sub.empty:
            raise ValueError(
                f"{inv_path} has no data in total-power window {start} ~ {end}"
            )
        inverter_series.append((inv_path.stem, inverter_sub))

    nrows = 1 + len(inverter_series)
    fig, axes_grid = plt.subplots(
        nrows,
        1,
        figsize=(16, 3 * nrows + 1),
        sharex=True,
        squeeze=False,
        gridspec_kw={"hspace": 0.12},
    )
    axes = axes_grid[:, 0]
    ax_total = axes[0]

    ax_total.plot(
        total_sub["time_bj"],
        total_sub["final_power"],
        linewidth=0.7,
        color="C0",
        label="pv_total",
    )
    ax_total.set_ylabel("Power (kW)")
    ax_total.set_title(
        f"PV ground truth — first {args.days} days ({start:%Y-%m-%d} ~ {end:%Y-%m-%d %H:%M}, Beijing)"
    )
    ax_total.legend(loc="upper right")
    ax_total.grid(True, alpha=0.3)

    for idx, (inv_label, inverter_sub) in enumerate(inverter_series, start=1):
        ax_inv = axes[idx]
        ax_inv.plot(
            inverter_sub["time_bj"],
            inverter_sub["final_power"],
            linewidth=0.7,
            color=f"C{idx % 10}",
            label=inv_label,
        )
        ax_inv.set_ylabel("Power (kW)")
        ax_inv.set_title(
            f"Inverter {inv_label}  ({len(inverter_sub)} points in same window)"
        )
        ax_inv.legend(loc="upper right")
        ax_inv.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (Beijing)")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
