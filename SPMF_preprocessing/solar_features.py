import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pvlib import solarposition
from pvlib.location import Location

# Yalongjiang site (``config/datasets/conf_ylj.yaml``)
DEFAULT_LATITUDE = 29.9254
DEFAULT_LONGITUDE = 100.5703
DEFAULT_TZ = "Asia/Shanghai"
DEFAULT_GRID_START = "2017-01-01 00:00:00"
DEFAULT_GRID_END = "2024-12-31 23:45:00"
DEFAULT_FREQ_MIN = 15
DEFAULT_GRID_OUT = (
    "/data/training_data/ylj_dataset_raw/solar_features_ylj_2017_2024_15min.csv"
)


def utc_to_local_solar_time_pvlib(utc_times: pd.DatetimeIndex, longitude: float) -> pd.DatetimeIndex:
    """Convert UTC to local (apparent) solar time using longitude and pvlib equation of time."""
    if utc_times.tz is not None:
        utc_naive = utc_times.tz_convert("UTC").tz_localize(None)
    else:
        utc_naive = utc_times
    lmst_offset_hours = longitude / 15.0
    dayofyear = utc_naive.dayofyear
    eot_minutes = solarposition.equation_of_time_spencer71(dayofyear)
    local_solar = utc_naive + pd.Timedelta(hours=lmst_offset_hours) + pd.to_timedelta(eot_minutes, unit="m")
    return local_solar


def build_china_local_time_grid(
    start: str,
    end: str,
    *,
    freq_min: int = 15,
    tz: str = DEFAULT_TZ,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """15-minute grid in China local time; returns (local_aware, utc_aware)."""
    if freq_min < 1:
        raise ValueError("freq_min must be >= 1")
    local = pd.date_range(
        start,
        end,
        freq=f"{int(freq_min)}min",
        tz=tz,
        inclusive="both",
    )
    return local, local.tz_convert("UTC")


def compute_solar_features(
    forecast_timestamps_utc, latitude: float, longitude: float
) -> list[dict]:
    """
    From UTC forecast times and site lat/lon, compute per timestep:
    - local_solar_time: apparent solar time at the site (naive datetime)
    - azimuth: sun azimuth (degrees)
    - zenith: sun zenith angle (degrees)
    - day_of_year: 1-366
    - hour_of_day: hour in local solar time (0-24, decimal)
    - ghi, dni, dhi: clear-sky irradiance (Ineichen)
    Returns a list of dicts, one per timestep.
    """
    ts = pd.to_datetime(forecast_timestamps_utc)
    if isinstance(ts, pd.Series):
        times_utc = pd.DatetimeIndex(ts)
    elif isinstance(ts, pd.DatetimeIndex):
        times_utc = ts
    else:
        times_utc = pd.DatetimeIndex(ts)
    if times_utc.tz is None:
        times_utc = times_utc.tz_localize("UTC", ambiguous="infer")
    else:
        times_utc = times_utc.tz_convert("UTC")

    local_solar = utc_to_local_solar_time_pvlib(times_utc, longitude)

    solpos = solarposition.get_solarposition(times_utc, latitude, longitude)
    azimuth = solpos["azimuth"].values
    zenith = solpos["apparent_zenith"].values
    clearsky = Location(latitude, longitude, tz="UTC").get_clearsky(times_utc, model="ineichen")
    ghi = clearsky["ghi"].values
    dni = clearsky["dni"].values
    dhi = clearsky["dhi"].values

    day_of_year = local_solar.dayofyear.values
    hour_of_day = (
        local_solar.hour.values
        + local_solar.minute.values / 60.0
        + local_solar.second.values / 3600.0
    )

    return [
        {
            "utc_time": times_utc[i].to_pydatetime(),
            "local_solar_time": local_solar[i].floor("us").to_pydatetime(),
            "ghi": float(ghi[i]),
            "dni": float(dni[i]),
            "dhi": float(dhi[i]),
            "azimuth": float(azimuth[i]),
            "zenith": float(zenith[i]),
            "day_of_year": int(day_of_year[i]),
            "hour_of_day": float(hour_of_day[i]),
        }
        for i in range(len(times_utc))
    ]


def solar_features_to_dataframe(
    feats: list[dict],
    china_local: pd.DatetimeIndex,
    *,
    tz: str = DEFAULT_TZ,
) -> pd.DataFrame:
    """Attach China wall-clock column (naive local) for alignment with YLJ Parquet."""
    if len(feats) != len(china_local):
        raise ValueError(f"feats len {len(feats)} != china_local len {len(china_local)}")
    out = pd.DataFrame(feats)
    local_naive = china_local
    if local_naive.tz is not None:
        local_naive = local_naive.tz_convert(tz).tz_localize(None)
    out.insert(0, "china_local_time", local_naive)
    return out


def compute_importance_factor(
    ghi: np.ndarray,
    active_power: np.ndarray,
    inverter_state: np.ndarray,
    *,
    tau: float = 0.03,
    inverter_ok: int = 512,
) -> np.ndarray:
    """
    Per-row weight exp(-|Δ(ghi/1000) - Δ(AP/550)| / τ); first timestep Δ=0 on both.
    Rows with inverter_state != inverter_ok or non-finite ghi/active_power → 0.
    """
    g = np.asarray(ghi, dtype=np.float64) / 1000.0
    ap = np.asarray(active_power, dtype=np.float64) / 550.0
    state = np.asarray(inverter_state)
    ok = (state == inverter_ok) & np.isfinite(g) & np.isfinite(ap)

    n = g.shape[0]
    d_g = np.zeros(n, dtype=np.float64)
    d_ap = np.zeros(n, dtype=np.float64)
    if n > 1:
        d_g[1:] = np.diff(g)
        d_ap[1:] = np.diff(ap)

    w = np.exp(-np.abs(d_g - d_ap) / float(tau))
    out = np.zeros(n, dtype=np.float64)
    out[ok] = w[ok]
    return out


def run_grid_mode(args: argparse.Namespace) -> None:
    local, utc = build_china_local_time_grid(
        args.start,
        args.end,
        freq_min=int(args.freq_min),
        tz=str(args.tz),
    )
    n = len(local)
    print(
        f"[grid] {n} steps, {args.freq_min} min, tz={args.tz}, "
        f"lat={args.latitude} lon={args.longitude}"
    )
    print(f"[grid] china local: {local[0]} .. {local[-1]}")

    feats = compute_solar_features(utc, latitude=float(args.latitude), longitude=float(args.longitude))
    out_df = solar_features_to_dataframe(feats, local, tz=str(args.tz))

    out_path = Path(args.out or DEFAULT_GRID_OUT).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(out_df)} rows to {out_path}")


def run_csv_mode(args: argparse.Namespace) -> None:
    in_path = Path(args.csv).expanduser().resolve()
    if not in_path.is_file():
        raise FileNotFoundError(f"input csv not found: {in_path}")

    df = pd.read_csv(in_path)
    if "collectTime" not in df.columns:
        raise KeyError(f"{in_path}: missing required column 'collectTime'")

    ts = pd.to_datetime(df["collectTime"], errors="coerce")
    if ts.isna().any():
        bad = int(ts.isna().sum())
        raise ValueError(f"{in_path}: collectTime contains {bad} invalid timestamp(s)")

    feats = compute_solar_features(ts, latitude=float(args.latitude), longitude=float(args.longitude))
    out_df = pd.DataFrame(feats)
    for col in ("inverter_state", "active_power"):
        if col not in df.columns:
            raise KeyError(f"{in_path}: missing required column {col!r}")
    out_df["inverter_state"] = df["inverter_state"].to_numpy()
    out_df["active_power"] = df["active_power"].to_numpy()
    out_df["importance_factor"] = compute_importance_factor(
        out_df["ghi"].to_numpy(),
        out_df["active_power"].to_numpy(),
        out_df["inverter_state"].to_numpy(),
        tau=float(args.importance_tau),
    )

    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out is not None
        else in_path.with_name(f"{in_path.stem}_solar_features.csv")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(out_df)} rows to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute solar features for a site. Default: 15-min China-local grid "
            "(YLJ 2017–2024). Use --csv for PV CSV + importance_factor."
        )
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="If set, read collectTime from this CSV (UTC) and add importance_factor.",
    )
    parser.add_argument("--latitude", type=float, default=DEFAULT_LATITUDE)
    parser.add_argument("--longitude", type=float, default=DEFAULT_LONGITUDE)
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=f"Output CSV (grid default: {DEFAULT_GRID_OUT}).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=DEFAULT_GRID_START,
        help=f"Grid mode: China-local start (default {DEFAULT_GRID_START}).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=DEFAULT_GRID_END,
        help=f"Grid mode: China-local end inclusive (default {DEFAULT_GRID_END}).",
    )
    parser.add_argument(
        "--freq-min",
        type=int,
        default=DEFAULT_FREQ_MIN,
        help="Grid mode: step size in minutes (default 15).",
    )
    parser.add_argument(
        "--tz",
        type=str,
        default=DEFAULT_TZ,
        help=f"Grid mode: civil timezone (default {DEFAULT_TZ}).",
    )
    parser.add_argument(
        "--importance-tau",
        type=float,
        default=0.03,
        help="CSV mode only: τ for importance_factor (default 0.03).",
    )
    args = parser.parse_args()

    if args.csv:
        run_csv_mode(args)
    else:
        run_grid_mode(args)


if __name__ == "__main__":
    main()
