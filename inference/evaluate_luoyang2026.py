"""Evaluate Luoyang2026 inference CSV by Beijing-day RMSE/MAE.

Example:
  python inference/evaluate_luoyang2026.py \
    --input_csv inference_results/luoyang2026_15min_1hourstride/station_total_15m.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_END_DATE_BJ = "2026-06-11"


def _resolve_columns(df: pd.DataFrame) -> tuple[str, str]:
    pred_candidates = ("pv_pred_kW_total", "pv_pred_kW", "pred", "y_pred")
    true_candidates = ("pv_true_kW_total", "pv_true_kW", "true", "y_true")
    pred_col = next((c for c in pred_candidates if c in df.columns), None)
    true_col = next((c for c in true_candidates if c in df.columns), None)
    if pred_col is None or true_col is None:
        raise KeyError(
            "Cannot find prediction/ground-truth columns. "
            f"Expected one of pred={pred_candidates}, true={true_candidates}. "
            f"Got columns={list(df.columns)}"
        )
    return pred_col, true_col


def _load_with_bj_time(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {path}")

    if "time_bj" in df.columns:
        t_bj = pd.to_datetime(df["time_bj"], errors="coerce")
        if t_bj.isna().any():
            raise ValueError(f"Found invalid time_bj values in {path}")
        if t_bj.dt.tz is None:
            t_bj = t_bj.dt.tz_localize("Asia/Shanghai")
        else:
            t_bj = t_bj.dt.tz_convert("Asia/Shanghai")
        df["time_bj_parsed"] = t_bj
    elif "time_utc" in df.columns:
        t_utc = pd.to_datetime(df["time_utc"], errors="coerce", utc=True)
        if t_utc.isna().any():
            raise ValueError(f"Found invalid time_utc values in {path}")
        df["time_bj_parsed"] = t_utc.dt.tz_convert("Asia/Shanghai")
    else:
        raise KeyError(f"{path} must contain either time_bj or time_utc column")

    pred_col, true_col = _resolve_columns(df)
    df[pred_col] = pd.to_numeric(df[pred_col], errors="coerce")
    df[true_col] = pd.to_numeric(df[true_col], errors="coerce")
    valid = df[pred_col].notna() & df[true_col].notna()
    df = df.loc[valid].copy()
    if df.empty:
        raise ValueError(f"No valid pred/true rows after numeric filtering: {path}")
    df["date_bj"] = df["time_bj_parsed"].dt.strftime("%Y-%m-%d")
    return df


def _rmse_mae(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    diff = pred - true
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    return rmse, mae


def evaluate_daily(path: Path) -> None:
    df = _load_with_bj_time(path)
    cutoff_end = pd.Timestamp(_END_DATE_BJ).tz_localize("Asia/Shanghai") + pd.Timedelta(days=1) - pd.Timedelta(
        microseconds=1
    )
    df = df.loc[df["time_bj_parsed"] <= cutoff_end].copy()
    if df.empty:
        raise RuntimeError(f"No rows on or before {_END_DATE_BJ} (Beijing time) in {path}")
    pred_col, true_col = _resolve_columns(df)

    daily_stats: list[tuple[str, int, float, float]] = []
    for date_bj, g in df.groupby("date_bj", sort=True):
        pred = g[pred_col].to_numpy(dtype=np.float64)
        true = g[true_col].to_numpy(dtype=np.float64)
        rmse, mae = _rmse_mae(pred, true)
        daily_stats.append((date_bj, len(g), rmse, mae))

    if not daily_stats:
        raise RuntimeError("No daily groups found for evaluation")

    avg_rmse = float(np.mean([x[2] for x in daily_stats]))
    avg_mae = float(np.mean([x[3] for x in daily_stats]))

    print(f"[evaluate_luoyang2026] file: {path}")
    print(f"[evaluate_luoyang2026] date_filter_bj: <= {_END_DATE_BJ}")
    print(f"[evaluate_luoyang2026] rows_used: {len(df)}  days: {len(daily_stats)}")
    print("-" * 78)
    print(f"{'date_bj':<12} {'n_points':>8} {'RMSE':>16} {'MAE':>16}")
    print("-" * 78)
    for date_bj, n_points, rmse, mae in daily_stats:
        print(f"{date_bj:<12} {n_points:>8d} {rmse:>16.6f} {mae:>16.6f}")
    print("-" * 78)
    print(f"{'daily-avg':<12} {'-':>8} {avg_rmse:>16.6f} {avg_mae:>16.6f}")
    print("-" * 78)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Luoyang2026 inference CSV by Beijing-day RMSE/MAE."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Input station/inverter CSV (supports station_total_*.csv same format).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    path = Path(args.input_csv).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"input_csv not found: {path}")
    evaluate_daily(path)


if __name__ == "__main__":
    main()
