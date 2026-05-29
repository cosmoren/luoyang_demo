#!/usr/bin/env python3
"""
YLJ solar / kt features from matrix Parquet windows.

CSV rows:
- 2024 train: 7-day history before the first ``timestamp_win``, then one row per anchor
  (``china_local_time`` = ``timestamp_win`` = last ``observe_power``).
- 2025 test: anchors only (late 2024 lookback already covered by 2024 rows).

``p_mean`` = mean of all 2024 powers in that table (prefix steps + every window-end);
same scalar applied to 2025. ``p_cs`` / ``kt`` / ``kt_mask`` match ``aggregate_by_devdn_solarfeats``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SPMF_DIR = Path(__file__).resolve().parent
if str(_SPMF_DIR) not in sys.path:
    sys.path.insert(0, str(_SPMF_DIR))

from aggregate_by_devdn_solarfeats import compute_solar_arrays  # noqa: E402

# Yalongjiang (``config/datasets/conf_ylj.yaml``)
DEFAULT_LATITUDE = 29.9254
DEFAULT_LONGITUDE = 100.5703
DEFAULT_TZ = "Asia/Shanghai"
DEFAULT_RAW_DIR = "/data/training_data/ylj_dataset_raw"
DEFAULT_TRAIN_PARQUET = "synthetic_real.parquet"
DEFAULT_TEST_PARQUET = "ds_v322_1219_2025_1-12.parquet"
DEFAULT_HIST_LEN = 672
DEFAULT_NATIVE_INTERVAL_MIN = 15
DEFAULT_P_MEAN_YEAR = 2024
DEFAULT_OUT = (
    "/data/training_data/ylj_dataset_raw/solar_features_ylj_2017_2025_15min.csv"
)

OUTPUT_COLS = [
    "china_local_time",
    "solar_zenith",
    "solar_azimuth",
    "local_solar_time",
    "day_of_year",
    "hour_of_day",
    "p_cs",
    "kt",
    "kt_mask",
    "p_mean",
]


def _to_naive_china_local(times: pd.Series, tz: str) -> pd.Series:
    t = pd.to_datetime(times, errors="coerce")
    if t.isna().any():
        raise ValueError("invalid timestamp(s)")
    if t.dt.tz is None:
        return t.dt.tz_localize(tz, ambiguous=True).dt.tz_localize(None)
    return t.dt.tz_convert(tz).dt.tz_localize(None)


def _observe_row(cell, hist_len: int) -> np.ndarray:
    arr = np.asarray(cell, dtype=np.float64).reshape(-1)
    if arr.shape != (hist_len,):
        raise ValueError(
            f"observe_power length {arr.shape[0]} != expected hist_len={hist_len}"
        )
    return arr


def window_end_power(series: pd.Series, hist_len: int) -> np.ndarray:
    return np.array([_observe_row(c, hist_len)[-1] for c in series], dtype=np.float64)


def hist_times_before_win(
    win_local: pd.Timestamp, *, hist_len: int, native_min: int
) -> pd.DatetimeIndex:
    """``hist_len - 1`` China-local times strictly before ``win_local`` (15-min grid)."""
    offsets_min = (hist_len - 1 - np.arange(hist_len - 1, dtype=np.int64)) * int(native_min)
    base = pd.Timestamp(win_local)
    return pd.DatetimeIndex(
        [base - pd.Timedelta(minutes=int(m)) for m in offsets_min]
    )


def earliest_window_prefix(
    path: Path,
    *,
    hist_len: int,
    native_min: int,
    tz: str,
) -> pd.DataFrame:
    """
    Seven-day lookback on the row with earliest ``timestamp_win`` (steps before that anchor).
    """
    df = pd.read_parquet(
        path, columns=["timestamp_win", "observe_power"], engine="pyarrow"
    )
    m = df["observe_power"].notna()
    df = df.loc[m].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"{path.name}: no valid observe_power rows")

    wins = _to_naive_china_local(df["timestamp_win"], tz)
    idx = int(wins.idxmin())
    win_local = pd.Timestamp(wins.iloc[idx])
    op = _observe_row(df["observe_power"].iloc[idx], hist_len)

    times = hist_times_before_win(win_local, hist_len=hist_len, native_min=native_min)
    return pd.DataFrame(
        {
            "china_local_time": times,
            "power_end": op[: hist_len - 1],
        }
    )


def anchor_frame(path: Path, *, hist_len: int, tz: str) -> pd.DataFrame:
    """One row per sample: ``china_local_time`` = ``timestamp_win``."""
    df = pd.read_parquet(
        path, columns=["timestamp_win", "observe_power"], engine="pyarrow"
    )
    m = df["observe_power"].notna()
    df = df.loc[m].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"{path.name}: no valid observe_power rows")

    return pd.DataFrame(
        {
            "china_local_time": _to_naive_china_local(df["timestamp_win"], tz),
            "power_end": window_end_power(df["observe_power"], hist_len),
        }
    )


def compute_p_mean_2024(prefix: pd.DataFrame, anchors_2024: pd.DataFrame) -> float:
    """Mean of prefix + all 2024 window-end powers (includes 7 days before first anchor)."""
    parts = [prefix["power_end"].to_numpy(), anchors_2024["power_end"].to_numpy()]
    all_pw = np.concatenate(parts)
    p_mean = float(np.mean(all_pw))
    print(
        f"[p_mean] 2024 prefix n={len(prefix)} anchors n={len(anchors_2024)} "
        f"total n={len(all_pw)} p_mean={p_mean:.6f}"
    )
    return p_mean


def dedupe_power_rows(rows: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """One row per ``china_local_time``; keep first (train prefix before train anchors)."""
    rows = rows.sort_values("china_local_time").reset_index(drop=True)
    grouped = rows.groupby("china_local_time", sort=True)
    n_conflict = int(
        grouped["power_end"].apply(lambda s: s.nunique(dropna=False) > 1).sum()
    )
    if n_conflict > 0:
        print(f"[warn] {n_conflict} time(s) with conflicting power; using first")

    out = grouped.agg(power_end=("power_end", "first")).reset_index()
    return out, n_conflict


def attach_solar_and_kt(
    df: pd.DataFrame,
    *,
    latitude: float,
    longitude: float,
    tz: str,
    p_mean: float,
) -> pd.DataFrame:
    local = pd.to_datetime(df["china_local_time"])
    if local.dt.tz is None:
        local_tz = local.dt.tz_localize(tz, ambiguous=True)
    else:
        local_tz = local.dt.tz_convert(tz)
    utc = pd.DatetimeIndex(local_tz.dt.tz_convert("UTC"))

    import aggregate_by_devdn_solarfeats as agg_mod

    agg_mod._solar_cache.clear()
    arrays = compute_solar_arrays(utc, latitude, longitude)

    p_cs = np.asarray(arrays["p_cs"], dtype=np.float64)
    kt_mask = (p_cs > 0.1).astype(np.float64)
    power = df["power_end"].to_numpy(dtype=np.float64)
    kt = power / (p_cs * float(p_mean) + 1e-6) * kt_mask

    out = pd.DataFrame(
        {
            "china_local_time": local_tz.dt.tz_localize(None),
            "solar_zenith": np.round(arrays["zenith"].astype(np.float64), 4),
            "solar_azimuth": np.round(arrays["azimuth"].astype(np.float64), 4),
            "local_solar_time": arrays["lst_strs"],
            "day_of_year": arrays["day_of_year"].astype(int),
            "hour_of_day": np.round(arrays["hour_of_day"].astype(np.float64), 4),
            "p_cs": np.round(p_cs, 6),
            "kt": np.round(kt, 6),
            "kt_mask": kt_mask.astype(int),
            "p_mean": float(p_mean),
        }
    )
    return out[OUTPUT_COLS]


def run_parquet_mode(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw_dir).expanduser().resolve()
    train_path = raw_dir / str(args.train_parquet)
    test_path = raw_dir / str(args.test_parquet)
    for p in (train_path, test_path):
        if not p.is_file():
            raise FileNotFoundError(f"Parquet not found: {p}")

    hist_len = int(args.hist_len)
    native_min = int(args.native_interval_min)
    tz = str(args.tz)
    lat = float(args.latitude)
    lon = float(args.longitude)

    prefix_2024 = earliest_window_prefix(
        train_path, hist_len=hist_len, native_min=native_min, tz=tz
    )
    anchors_2024 = anchor_frame(train_path, hist_len=hist_len, tz=tz)
    anchors_2025 = anchor_frame(test_path, hist_len=hist_len, tz=tz)

    print(
        f"[parquet] 2024 prefix={len(prefix_2024)} anchors={len(anchors_2024)}; "
        f"2025 anchors={len(anchors_2025)}"
    )
    if len(prefix_2024):
        print(
            f"[parquet] prefix range: {prefix_2024['china_local_time'].min()} .. "
            f"{prefix_2024['china_local_time'].max()}"
        )
        print(
            f"[parquet] first anchor: {anchors_2024['china_local_time'].min()}"
        )

    p_mean = compute_p_mean_2024(prefix_2024, anchors_2024)

    combined = pd.concat(
        [prefix_2024, anchors_2024, anchors_2025],
        ignore_index=True,
    )
    deduped, n_conflict = dedupe_power_rows(combined)
    print(f"[parquet] unique rows: {len(deduped)}")

    years = pd.to_datetime(deduped["china_local_time"]).dt.year.value_counts().sort_index()
    print(f"[parquet] calendar year counts: {years.to_dict()}")
    if n_conflict:
        print(f"[parquet] power conflicts: {n_conflict}")

    out_df = attach_solar_and_kt(
        deduped, latitude=lat, longitude=lon, tz=tz, p_mean=p_mean
    )

    out_path = Path(args.out or DEFAULT_OUT).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(out_df)} rows to {out_path}")
    print(f"Columns: {list(out_df.columns)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "YLJ solar+kt CSV: 2024 includes 7d before first timestamp_win + anchors; "
            "2025 anchors only; p_mean from all 2024 powers."
        )
    )
    parser.add_argument("--raw-dir", type=str, default=DEFAULT_RAW_DIR)
    parser.add_argument("--train-parquet", type=str, default=DEFAULT_TRAIN_PARQUET)
    parser.add_argument("--test-parquet", type=str, default=DEFAULT_TEST_PARQUET)
    parser.add_argument("--out", type=str, default=None, help=f"Default: {DEFAULT_OUT}")
    parser.add_argument("--latitude", type=float, default=DEFAULT_LATITUDE)
    parser.add_argument("--longitude", type=float, default=DEFAULT_LONGITUDE)
    parser.add_argument("--hist-len", type=int, default=DEFAULT_HIST_LEN)
    parser.add_argument("--native-interval-min", type=int, default=DEFAULT_NATIVE_INTERVAL_MIN)
    parser.add_argument("--tz", type=str, default=DEFAULT_TZ)
    args = parser.parse_args()
    run_parquet_mode(args)


if __name__ == "__main__":
    main()
