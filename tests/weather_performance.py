"""Inspect / plot inference NPZ files (pred, target, weather_score)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_NPZ_DIR = (
    Path(__file__).resolve().parents[1]
    / "inference_results"
    / "test_rolling_stride15_ws"
)
DEFAULT_SAMPLE = "NE_333620909.npz"
# step k valid time = t0 + (k+1)*forecast_dt_min; k=15 → t0+4h at 15-min cadence
DEFAULT_HORIZON_IDX = 15

# Written by ``inference/infer_testset.py`` via ``np.savez_compressed``.
NPZ_KEYS = (
    "t0_utc",
    "pred_kW",
    "target_kW",
    "target_mask",
    "weather_score",
    "cos_zenith",
    "forecast_dt_min",
    "pv_output_len",
    "device_id",
    "devDn",
    "stride_min",
)


def describe_npz(path: Path) -> None:
    data = np.load(path, allow_pickle=False)
    print(f"\n{'=' * 72}")
    print(f"FILE: {path.name}")
    print(f"keys ({len(data.files)}): {list(data.files)}")

    missing = [k for k in NPZ_KEYS if k not in data.files]
    extra = [k for k in data.files if k not in NPZ_KEYS]
    if missing:
        print(f"  missing expected keys: {missing}")
    if extra:
        print(f"  extra keys: {extra}")

    for key in data.files:
        arr = data[key]
        if arr.ndim == 0:
            print(f"  {key:16s} scalar  dtype={arr.dtype}  value={arr.item()!r}")
            continue

        line = f"  {key:16s} shape={arr.shape}  dtype={arr.dtype}"
        if key == "t0_utc":
            print(f"{line}  first={arr[0]!r}  last={arr[-1]!r}")
            continue
        if arr.dtype == np.uint8 or np.issubdtype(arr.dtype, np.integer):
            print(f"{line}  min={int(arr.min())}  max={int(arr.max())}  mean={float(arr.mean()):.4f}")
            continue
        print(
            f"{line}  min={float(np.nanmin(arr)):.4g}  max={float(np.nanmax(arr)):.4g}  "
            f"mean={float(np.nanmean(arr)):.4g}"
        )

    n_win, t_out = data["pred_kW"].shape
    mask = data["target_mask"].astype(bool)
    err = data["pred_kW"] - data["target_kW"]
    for label, idx in (("15min", 0), ("4h", 15), ("48h", 191)):
        m = mask[:, idx]
        if m.any():
            e = err[m, idx]
            rmse = float(np.sqrt(np.mean(e * e)))
            mae = float(np.mean(np.abs(e)))
            print(f"  horizon {label:>4s} (idx={idx:3d}): n_valid={int(m.sum())}  rmse={rmse:.3f} kW  mae={mae:.3f} kW")
        else:
            print(f"  horizon {label:>4s} (idx={idx:3d}): n_valid=0")


def _horizon_label(horizon_idx: int, dt_min: int) -> str:
    lead_min = (horizon_idx + 1) * dt_min
    if lead_min % 60 == 0:
        return f"{lead_min // 60}h"
    return f"{lead_min}min"


def plot_first_days(
    path: Path,
    *,
    days: float = 5.0,
    horizon_idx: int = DEFAULT_HORIZON_IDX,
    out_path: Path | None = None,
) -> Path:
    """
    Plot ``pred_kW``, ``target_kW``, ``weather_score`` for the first ``days`` of data.

    Default ``horizon_idx=15`` → valid time = t0 + 4 h (15-min forecast steps).
    """
    data = np.load(path, allow_pickle=False)
    if "weather_score" not in data.files:
        raise KeyError(f"{path.name}: missing 'weather_score' (re-run infer_testset with ws CSV)")

    t0 = pd.to_datetime(data["t0_utc"])
    dt_min = int(data["forecast_dt_min"].item())
    valid_time = t0 + pd.to_timedelta((horizon_idx + 1) * dt_min, unit="min")

    t_start = t0.min()
    t_end = t_start + pd.Timedelta(days=days)
    sel = (t0 >= t_start) & (t0 < t_end)
    if not sel.any():
        raise ValueError(f"No windows in first {days} days of {path.name}")

    t = valid_time[sel]
    pred = data["pred_kW"][sel, horizon_idx]
    target = data["target_kW"][sel, horizon_idx]
    ws = data["weather_score"][sel, horizon_idx]
    mask = data["target_mask"][sel, horizon_idx].astype(bool)
    daytime = data["cos_zenith"][sel, horizon_idx] >= 0

    horizon_lbl = _horizon_label(horizon_idx, dt_min)
    if out_path is None:
        stem = path.stem
        out_path = (
            Path(__file__).resolve().parent
            / f"weather_perf_{stem}_first{int(days)}d_{horizon_lbl}.png"
        )

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, gridspec_kw={"hspace": 0.08})

    ax0 = axes[0]
    ax0.plot(t, pred, color="#16a34a", linewidth=0.9, label="pred_kW", alpha=0.9)
    ax0.plot(t, target, color="#dc2626", linewidth=0.9, label="target_kW", alpha=0.85)
    ax0.set_ylabel("kW")
    ax0.set_title(
        f"{path.stem} — first {days:g} days (t0+{horizon_lbl}, stride={int(data['stride_min'].item())} min)"
    )
    ax0.legend(loc="upper right")
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1]
    ax1.plot(t, ws, color="#2563eb", linewidth=0.9, label="weather_score")
    ax1.fill_between(t, 0, ws, color="#2563eb", alpha=0.12)
    ax1.scatter(t[~daytime], ws[~daytime], s=6, c="#94a3b8", alpha=0.5, label="night (cos_zenith<0)")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_ylabel("weather_score")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[2]
    err = pred - target
    ax2.plot(t, err, color="#7c3aed", linewidth=0.8, label="pred − target")
    ax2.axhline(0, color="#64748b", linewidth=0.6, linestyle="--")
    ax2.scatter(t[~mask], err[~mask], s=6, c="#cbd5e1", alpha=0.4, label="invalid (mask=0)")
    ax2.set_ylabel("kW error")
    ax2.set_xlabel("valid time (UTC)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {out_path}")
    print(f"  windows: {int(sel.sum())}  range: {t.min()} ~ {t.max()}")
    print(
        f"  daytime weather_score: min={ws[daytime].min():.3f}  max={ws[daytime].max():.3f}  "
        f"mean={ws[daytime].mean():.3f}"
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npz-dir",
        type=Path,
        default=DEFAULT_NPZ_DIR,
        help=f"Directory containing per-inverter .npz files (default: {DEFAULT_NPZ_DIR})",
    )
    parser.add_argument(
        "--sample",
        type=str,
        default=None,
        help="Inspect only this file (basename or full path). Default: first npz + summary.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Print full details for every .npz file (verbose).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot pred_kW / target_kW / weather_score for the first N days.",
    )
    parser.add_argument(
        "--days",
        type=float,
        default=5.0,
        help="Days to plot when using --plot (default: 5).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path for --plot.",
    )
    parser.add_argument(
        "--horizon-idx",
        type=int,
        default=DEFAULT_HORIZON_IDX,
        help="Forecast step index (default: 15 → t0+4h at 15-min steps).",
    )
    args = parser.parse_args()

    npz_dir = args.npz_dir.expanduser().resolve()
    if not npz_dir.is_dir():
        raise FileNotFoundError(f"NPZ directory not found: {npz_dir}")

    npz_files = sorted(npz_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files under {npz_dir}")

    summary_csv = npz_dir / "_summary.csv"
    print(f"NPZ directory: {npz_dir}")
    print(f"NPZ count: {len(npz_files)}")
    if summary_csv.is_file():
        print(f"Summary CSV: {summary_csv} ({summary_csv.stat().st_size} bytes)")

    sample_name = args.sample or (DEFAULT_SAMPLE if args.plot else None)
    if sample_name:
        sample_path = Path(sample_name)
        if not sample_path.is_file():
            sample_path = npz_dir / sample_name
            if not sample_path.suffix:
                sample_path = sample_path.with_suffix(".npz")
        if not sample_path.is_file():
            raise FileNotFoundError(f"Sample NPZ not found: {sample_name}")
        if args.plot:
            plot_first_days(
                sample_path,
                days=args.days,
                horizon_idx=args.horizon_idx,
                out_path=args.out,
            )
        else:
            describe_npz(sample_path)
        return

    if args.plot:
        plot_first_days(
            npz_files[0],
            days=args.days,
            horizon_idx=args.horizon_idx,
            out_path=args.out,
        )
        return

    describe_npz(npz_files[0])

    if args.all:
        for path in npz_files[1:]:
            describe_npz(path)
    else:
        print(f"\n(showing 1/{len(npz_files)} files; use --sample NAME or --all for more)")


if __name__ == "__main__":
    main()
