from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib
import yaml

LUOYANG_LATITUDE = 34.69984
LUOYANG_LONGITUDE = 112.28440
LUOYANG_SURFACE_AZIMUTH = 180.0
LUOYANG_SURFACE_TILT = 0.0


def load_total_csv(
    config_path: Path = Path("config/datasets/conf_luoyang_2026.yaml"),
) -> pd.DataFrame:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    paths_cfg = cfg.get("paths", {}) or {}

    data_dir = Path(paths_cfg.get("data_dir", "")).expanduser()
    pv_total = Path(paths_cfg.get("pv_total_path", "")).expanduser()
    if not pv_total.is_absolute():
        pv_total = data_dir / pv_total
    pv_total = pv_total.resolve()

    if not pv_total.is_file():
        raise FileNotFoundError(f"total csv not found: {pv_total}")

    df = pd.read_csv(pv_total)
    return df


def compute_normalized_clearsky_power(
    timestamps,
    *,
    latitude: float,
    longitude: float,
    surface_azimuth: float,
    surface_tilt: float,
    tz_for_naive: str = "Asia/Shanghai",
    clip_min: float = 0.0,
    clip_max: float = 1.2,
) -> np.ndarray:
    """
    Compute normalized clear-sky power (p_cs-like) for each timestamp.

    Steps:
    1) pvlib Ineichen clear-sky irradiance (GHI/DNI/DHI).
    2) POA global irradiance on the panel plane (given tilt/azimuth).
    3) Normalize by 1000 W/m^2 and clip to [clip_min, clip_max].
    """
    ts = pd.to_datetime(timestamps, errors="coerce")
    if isinstance(ts, pd.Series):
        ts_index = pd.DatetimeIndex(ts)
    else:
        ts_index = pd.DatetimeIndex(ts)

    if ts_index.tz is None:
        ts_aware = ts_index.tz_localize(tz_for_naive, ambiguous="NaT", nonexistent="shift_forward")
    else:
        ts_aware = ts_index
    ts_utc = ts_aware.tz_convert("UTC")

    valid = ~ts_utc.isna()
    out = np.full(len(ts_utc), np.nan, dtype=np.float32)
    if not valid.any():
        return out

    ts_valid = ts_utc[valid]
    solpos = pvlib.solarposition.get_solarposition(ts_valid, latitude, longitude)
    site = pvlib.location.Location(latitude, longitude)
    clearsky = site.get_clearsky(ts_valid, model="ineichen")
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=float(surface_tilt),
        surface_azimuth=float(surface_azimuth),
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=clearsky["dni"],
        ghi=clearsky["ghi"],
        dhi=clearsky["dhi"],
    )
    p_cs = (poa["poa_global"].clip(lower=0) / 1000.0).clip(lower=float(clip_min), upper=float(clip_max))
    out[np.asarray(valid)] = p_cs.to_numpy(dtype=np.float32, copy=False)
    return out


def add_normalized_clearsky_power_column(
    df: pd.DataFrame,
    *,
    latitude: float,
    longitude: float,
    surface_azimuth: float,
    surface_tilt: float,
    time_col: str = "dtime",
    out_col: str = "p_cs_norm",
    tz_for_naive: str = "Asia/Shanghai",
) -> pd.DataFrame:
    """
    Return a copy of df with normalized clear-sky power column added.
    """
    if time_col not in df.columns and "collectTime" not in df.columns:
        raise KeyError(f"missing required column: {time_col} (or fallback collectTime)")
    t_src = df[time_col] if time_col in df.columns else df["collectTime"]
    out = df.copy()
    out[out_col] = compute_normalized_clearsky_power(
        t_src,
        latitude=latitude,
        longitude=longitude,
        surface_azimuth=surface_azimuth,
        surface_tilt=surface_tilt,
        tz_for_naive=tz_for_naive,
    )
    return out


def compute_daily_smoothness(
    df: pd.DataFrame,
    *,
    time_col: str = "dtime",
    value_col: str = "final_power",
) -> pd.DataFrame:
    """
    Compute one smoothness score per day from final_power.
    Score is in (0, 1], larger means smoother:
      smoothness = 1 / (1 + mean(|diff(x)|) / (mean(|x|) + eps))
    """
    if value_col not in df.columns:
        raise KeyError(f"missing required column: {value_col}")
    if time_col in df.columns:
        # CSV dtime is Beijing local time.
        t = pd.to_datetime(df[time_col], errors="coerce")
    elif "collectTime" in df.columns:
        # Fallback: collectTime is UTC, convert to Beijing for day split.
        t = pd.to_datetime(df["collectTime"], errors="coerce", utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    else:
        raise KeyError(f"missing required column: {time_col} (or fallback collectTime)")
    x = pd.to_numeric(df[value_col], errors="coerce")
    ok = t.notna() & x.notna()
    tmp = pd.DataFrame({"timestamp": t[ok], "final_power": x[ok].astype(float)})
    tmp["date"] = tmp["timestamp"].dt.date.astype(str)

    rows: list[dict[str, float | int | str]] = []
    eps = 1e-6
    for date_str, grp in tmp.groupby("date", sort=True):
        y = grp["final_power"].to_numpy(dtype=np.float64)
        n = int(y.size)
        if n < 2:
            rows.append({"date": date_str, "n_points": n, "smoothness": np.nan})
            continue
        mad_diff = float(np.mean(np.abs(np.diff(y))))
        mean_abs = float(np.mean(np.abs(y)))
        smoothness = 1.0 / (1.0 + mad_diff / (mean_abs + eps))
        rows.append({"date": date_str, "n_points": n, "smoothness": smoothness})

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return out


def plot_final_power_with_mask(
    df: pd.DataFrame,
    mask_df: pd.DataFrame,
    *,
    clearsky_power: np.ndarray,
    out_path: Path,
    time_col: str = "dtime",
    value_col: str = "final_power",
    plot_days: int = 30,
    plot_start_date: str | None = None,
    plot_end_date: str | None = None,
) -> None:
    if time_col in df.columns:
        t = pd.to_datetime(df[time_col], errors="coerce")
    elif "collectTime" in df.columns:
        t = pd.to_datetime(df["collectTime"], errors="coerce", utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    else:
        raise KeyError(f"missing required column: {time_col} (or fallback collectTime)")
    x = pd.to_numeric(df[value_col], errors="coerce")
    mask = pd.to_numeric(mask_df["smooth_mask"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    p_cs = np.asarray(clearsky_power, dtype=np.float32)
    if p_cs.shape[0] != len(df):
        raise ValueError(f"clearsky_power length mismatch: got {p_cs.shape[0]} vs len(df)={len(df)}")

    ok = t.notna() & x.notna() & np.isfinite(mask) & np.isfinite(p_cs)
    data = pd.DataFrame(
        {
            "timestamp": t[ok],
            "final_power": x[ok].astype(float),
            "smooth_mask": mask[ok],
            "p_cs_norm": p_cs[ok],
        }
    ).sort_values("timestamp")
    if data.empty:
        raise ValueError("No valid points to plot after filtering timestamp/final_power/mask.")
    if plot_start_date is not None and plot_end_date is not None:
        start_ts = pd.Timestamp(plot_start_date)
        end_ts = pd.Timestamp(plot_end_date) + pd.Timedelta(days=1)
        data = data[(data["timestamp"] >= start_ts) & (data["timestamp"] < end_ts)]
        title_suffix = f"{start_ts.date()} to {pd.Timestamp(plot_end_date).date()}"
    else:
        n_days = max(1, int(plot_days))
        t0 = data["timestamp"].iloc[0]
        t1 = t0 + pd.Timedelta(days=n_days)
        data = data[data["timestamp"] < t1]
        title_suffix = f"first {n_days} days"
    if data.empty:
        raise ValueError("No data points in selected plotting window.")

    fig, ax1 = plt.subplots(1, 1, figsize=(16, 5))
    ax1.plot(data["timestamp"], data["final_power"], lw=1.0, color="tab:blue", label="final_power")
    ax1.set_ylabel("final_power", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(data["timestamp"], data["smooth_mask"], lw=1.0, color="tab:red", alpha=0.9, label="smooth_mask")
    ax2.plot(data["timestamp"], data["p_cs_norm"], lw=1.0, color="tab:green", alpha=0.9, label="p_cs_norm")
    ax2.set_ylabel("smooth_mask / p_cs_norm", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(-0.1, 1.3)

    ax1.set_xlabel("timestamp")
    ax1.set_title(f"final_power, smooth_mask, p_cs_norm ({title_suffix})")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def build_smooth_mask(
    df: pd.DataFrame,
    smooth_df: pd.DataFrame,
    *,
    threshold: float,
    time_col: str = "dtime",
    value_col: str = "final_power",
) -> pd.DataFrame:
    """
    Build a per-row binary mask aligned with final_power length.
    mask=1 when the row's day smoothness > threshold, else 0.
    """
    if value_col not in df.columns:
        raise KeyError(f"missing required column: {value_col}")
    if time_col in df.columns:
        t = pd.to_datetime(df[time_col], errors="coerce")
    elif "collectTime" in df.columns:
        t = pd.to_datetime(df["collectTime"], errors="coerce", utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    else:
        raise KeyError(f"missing required column: {time_col} (or fallback collectTime)")
    x = pd.to_numeric(df[value_col], errors="coerce")

    smooth_map = (
        smooth_df.assign(smoothness=pd.to_numeric(smooth_df["smoothness"], errors="coerce"))
        .set_index(smooth_df["date"].astype(str))["smoothness"]
    )
    date_str = pd.Series(t.dt.date.astype(str), index=df.index)
    day_smooth = date_str.map(smooth_map)
    valid = t.notna() & x.notna()
    mask = ((day_smooth > float(threshold)) & valid).astype(np.int8)
    return pd.DataFrame({"smooth_mask": mask}, index=df.index)


def build_4h_pv_estimation_csv(
    df: pd.DataFrame,
    *,
    clearsky_power: np.ndarray,
    start_date: str,
    end_date: str,
    horizon_hours: float = 4.0,
    time_col: str = "dtime",
    value_col: str = "final_power",
) -> pd.DataFrame:
    """
    Build 4h-ahead PV estimation table using:
      pv_pred(t+h) = final_power(t) * p_cs(t+h) / p_cs(t)
    Output columns:
      timestamp, final_power_true, pv_pred
    where timestamp is the target time (t+h).
    """
    if time_col in df.columns:
        ts = pd.to_datetime(df[time_col], errors="coerce")
    elif "collectTime" in df.columns:
        ts = pd.to_datetime(df["collectTime"], errors="coerce", utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    else:
        raise KeyError(f"missing required column: {time_col} (or fallback collectTime)")

    x = pd.to_numeric(df[value_col], errors="coerce")
    p_cs = np.asarray(clearsky_power, dtype=np.float32)
    if p_cs.shape[0] != len(df):
        raise ValueError(f"clearsky_power length mismatch: got {p_cs.shape[0]} vs len(df)={len(df)}")

    ok = ts.notna() & x.notna() & np.isfinite(p_cs)
    base = pd.DataFrame(
        {
            "timestamp": ts[ok],
            "final_power": x[ok].astype(float),
            "p_cs": p_cs[ok].astype(float),
        }
    ).sort_values("timestamp")
    if base.empty:
        return pd.DataFrame(columns=["timestamp", "final_power_true", "pv_pred"])

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    base = base[(base["timestamp"] >= start_ts) & (base["timestamp"] < end_ts)]
    if base.empty:
        return pd.DataFrame(columns=["timestamp", "final_power_true", "pv_pred"])

    h = pd.to_timedelta(float(horizon_hours), unit="h")
    src = base.rename(columns={"timestamp": "src_timestamp", "final_power": "final_power_src", "p_cs": "p_cs1"}).copy()
    src["timestamp"] = src["src_timestamp"] + h  # target timestamp

    full_target = pd.DataFrame(
        {
            "timestamp": ts[ok].astype("datetime64[ns]"),
            "final_power_true": x[ok].astype(float).to_numpy(),
            "p_cs2": p_cs[ok].astype(float),
        }
    ).sort_values("timestamp")

    merged = src.merge(full_target, on="timestamp", how="inner")
    fp_src = merged["final_power_src"].to_numpy(dtype=np.float64)
    # Temporary baseline requested: set pcs2/pcs1 = 1, so pv_pred(t+h) = final_power(t).
    pred = fp_src.copy()

    out = pd.DataFrame(
        {
            "timestamp": merged["timestamp"],
            "final_power_true": merged["final_power_true"].astype(float).to_numpy(),
            "pv_pred": pred.astype(np.float64),
        }
    ).sort_values("timestamp")
    return out


def summarize_rmse_mae_by_month_and_overall(est_df: pd.DataFrame) -> None:
    """
    Print monthly mean (computed from daily RMSE/MAE) and final overall mean of months.
    """
    if est_df.empty:
        print("METRICS: estimation table is empty, skip RMSE/MAE summary.")
        return

    df = est_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["final_power_true"] = pd.to_numeric(df["final_power_true"], errors="coerce")
    df["pv_pred"] = pd.to_numeric(df["pv_pred"], errors="coerce")
    df = df.dropna(subset=["timestamp", "final_power_true", "pv_pred"])
    if df.empty:
        print("METRICS: no valid rows after dropping NaNs, skip RMSE/MAE summary.")
        return

    df["date"] = df["timestamp"].dt.date

    def _rmse(y: np.ndarray, p: np.ndarray) -> float:
        diff = y - p
        return float(np.sqrt(np.mean(diff * diff)))

    def _mae(y: np.ndarray, p: np.ndarray) -> float:
        return float(np.mean(np.abs(y - p)))

    daily_rows: list[dict[str, float | int | str]] = []
    for d, g in df.groupby("date", sort=True):
        y = g["final_power_true"].to_numpy(dtype=np.float64)
        p = g["pv_pred"].to_numpy(dtype=np.float64)
        daily_rows.append(
            {
                "date": str(d),
                "month": pd.Timestamp(d).strftime("%Y-%m"),
                "n": int(len(g)),
                "rmse": _rmse(y, p),
                "mae": _mae(y, p),
            }
        )
    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        print("METRICS: no daily groups formed, skip RMSE/MAE summary.")
        return

    monthly = (
        daily.groupby("month", sort=True)
        .agg(days=("date", "count"), rmse=("rmse", "mean"), mae=("mae", "mean"))
        .reset_index()
    )
    if monthly.empty:
        print("METRICS: no monthly groups formed, skip RMSE/MAE summary.")
        return

    final_rmse = float(monthly["rmse"].mean())
    final_mae = float(monthly["mae"].mean())

    print("MONTHLY_MEAN_DAILY_METRICS")
    for _, r in monthly.iterrows():
        print(
            f"{r['month']} days={int(r['days'])} rmse={float(r['rmse']):.6f} "
            f"mae={float(r['mae']):.6f}"
        )
    print("FINAL_MEAN_OF_MONTHS")
    print(f"rmse={final_rmse:.6f} mae={final_mae:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot final_power, smooth mask, and normalized clear-sky power")
    parser.add_argument("--config", type=Path, default=Path("config/datasets/conf_luoyang_2026.yaml"))
    parser.add_argument(
        "--smooth-threshold",
        type=float,
        default=0.97,
        help="Rows whose day smoothness is > threshold get mask=1, else 0.",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=Path("tests/final_power_with_mask.png"),
        help="Output image path with final_power and smooth_mask overlaid.",
    )
    parser.add_argument(
        "--plot-days",
        type=int,
        default=30,
        help="Only plot the first N days to avoid over-dense curves.",
    )
    parser.add_argument(
        "--plot-start-date",
        type=str,
        default=None,
        help="Plot window start date (YYYY-MM-DD). Must be used with --plot-end-date.",
    )
    parser.add_argument(
        "--plot-end-date",
        type=str,
        default=None,
        help="Plot window end date (YYYY-MM-DD, inclusive). Must be used with --plot-start-date.",
    )
    parser.add_argument(
        "--forecast-start-date",
        type=str,
        default="2026-05-11",
        help="4h estimation source-window start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--forecast-end-date",
        type=str,
        default="2026-06-11",
        help="4h estimation source-window end date (YYYY-MM-DD, inclusive).",
    )
    parser.add_argument(
        "--forecast-hours",
        type=float,
        default=0.25,
        help="Forecast horizon in hours.",
    )
    parser.add_argument(
        "--forecast-out-csv",
        type=Path,
        default=Path("tests/pv_4h_estimation_20260511_20260611.csv"),
        help="Output CSV with timestamp/final_power_true/pv_pred.",
    )
    parser.add_argument("--latitude", type=float, default=LUOYANG_LATITUDE, help="Site latitude.")
    parser.add_argument("--longitude", type=float, default=LUOYANG_LONGITUDE, help="Site longitude.")
    parser.add_argument(
        "--surface-azimuth",
        type=float,
        default=LUOYANG_SURFACE_AZIMUTH,
        help="Panel azimuth in degrees (180=south).",
    )
    parser.add_argument(
        "--surface-tilt",
        type=float,
        default=LUOYANG_SURFACE_TILT,
        help="Panel tilt in degrees.",
    )
    args = parser.parse_args()

    df_total = load_total_csv(args.config)
    smooth_df = compute_daily_smoothness(df_total, value_col="final_power", time_col="dtime")
    mask_df = build_smooth_mask(
        df_total,
        smooth_df,
        threshold=args.smooth_threshold,
        value_col="final_power",
        time_col="dtime",
    )
    t_src = df_total["dtime"] if "dtime" in df_total.columns else df_total["collectTime"]
    p_cs_norm = compute_normalized_clearsky_power(
        t_src,
        latitude=args.latitude,
        longitude=args.longitude,
        surface_azimuth=args.surface_azimuth,
        surface_tilt=args.surface_tilt,
    )
    plot_final_power_with_mask(
        df_total,
        mask_df,
        clearsky_power=p_cs_norm,
        out_path=args.plot_out,
        value_col="final_power",
        time_col="dtime",
        plot_days=args.plot_days,
        plot_start_date=args.plot_start_date,
        plot_end_date=args.plot_end_date,
    )
    est_df = build_4h_pv_estimation_csv(
        df_total,
        clearsky_power=p_cs_norm,
        start_date=args.forecast_start_date,
        end_date=args.forecast_end_date,
        horizon_hours=args.forecast_hours,
        time_col="dtime",
        value_col="final_power",
    )
    args.forecast_out_csv.parent.mkdir(parents=True, exist_ok=True)
    est_df.to_csv(args.forecast_out_csv, index=False)
    print(
        f"saved overlay plot: {args.plot_out.resolve()} "
        f"(threshold={args.smooth_threshold}, plot_days={args.plot_days}, "
        f"plot_start_date={args.plot_start_date}, plot_end_date={args.plot_end_date}, "
        f"lat={args.latitude}, lon={args.longitude}, az={args.surface_azimuth}, tilt={args.surface_tilt}, "
        f"ones={int(mask_df['smooth_mask'].sum())}, n={len(mask_df)})"
    )
    print(
        f"saved 4h estimation csv: {args.forecast_out_csv.resolve()} "
        f"(start={args.forecast_start_date}, end={args.forecast_end_date}, "
        f"h={args.forecast_hours}, rows={len(est_df)})"
    )
    summarize_rmse_mae_by_month_and_overall(est_df)
