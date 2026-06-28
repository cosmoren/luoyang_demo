from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_PRED_CSV = Path(
    "checkpoints_2026total_4h_satema/gpu7/pv_forecast_vit_test_predictions_task_4h_gpu7.csv"
)
CAP_KW = 48629.73


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = y_true - y_pred
    return float(np.sqrt(np.mean(diff * diff)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_daily_metrics(df: pd.DataFrame, cap_kw: float) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for day, g in df.groupby("date", sort=True):
        y_true = g["pv_true_kW"].to_numpy(dtype=np.float64)
        y_pred = g["pv_pred_kW"].to_numpy(dtype=np.float64)
        rmse = _rmse(y_true, y_pred)
        mae = _mae(y_true, y_pred)
        accuracy = 1.0 - rmse / float(cap_kw)
        rows.append(
            {
                "date": str(day),
                "month": pd.Timestamp(day).strftime("%Y-%m"),
                "n_points": int(len(g)),
                "rmse": rmse,
                "mae": mae,
                "accuracy": accuracy,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate prediction CSV by daily RMSE/MAE, monthly mean (May/June), "
            "and final mean across the two months."
        )
    )
    parser.add_argument(
        "--pred-csv",
        type=Path,
        default=DEFAULT_PRED_CSV,
        help="Prediction CSV path with columns: timestamp_utc, pv_pred_kW, pv_true_kW.",
    )
    parser.add_argument(
        "--cap-kw",
        type=float,
        default=CAP_KW,
        help="Plant capacity in kW for daily accuracy = 1 - rmse/cap.",
    )
    args = parser.parse_args()

    pred_csv = args.pred_csv.expanduser().resolve()
    if not pred_csv.is_file():
        raise FileNotFoundError(f"prediction csv not found: {pred_csv}")

    df = pd.read_csv(pred_csv)
    required = {"timestamp_utc", "pv_pred_kW", "pv_true_kW"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    df["pv_pred_kW"] = pd.to_numeric(df["pv_pred_kW"], errors="coerce")
    df["pv_true_kW"] = pd.to_numeric(df["pv_true_kW"], errors="coerce")
    df = df.dropna(subset=["timestamp_utc", "pv_pred_kW", "pv_true_kW"]).copy()
    if df.empty:
        raise ValueError("no valid rows after parsing timestamp_utc/pv_pred_kW/pv_true_kW")

    # Convert UTC to Beijing time for day/month grouping.
    ts_bj = df["timestamp_utc"].dt.tz_convert("Asia/Shanghai")
    df["date"] = ts_bj.dt.date
    df["month_num"] = ts_bj.dt.month

    # Restrict to May + June only (as requested).
    df = df[df["month_num"].isin([5, 6])].copy()
    if df.empty:
        raise ValueError("no rows in May/June after filtering")

    daily = compute_daily_metrics(df, cap_kw=args.cap_kw)
    monthly = (
        daily.groupby("month", sort=True)
        .agg(
            days=("date", "count"),
            rmse=("rmse", "mean"),
            mae=("mae", "mean"),
            accuracy=("accuracy", "mean"),
        )
        .reset_index()
    )

    # Final mean across the monthly metrics (equal month weighting).
    final_rmse = float(monthly["rmse"].mean())
    final_mae = float(monthly["mae"].mean())
    final_accuracy = float(monthly["accuracy"].mean())
    overall_rmse = _rmse(
        df["pv_true_kW"].to_numpy(dtype=np.float64),
        df["pv_pred_kW"].to_numpy(dtype=np.float64),
    )

    print(f"prediction_csv: {pred_csv}")
    print(f"valid_rows_may_june: {len(df)}")
    print(f"cap_kw: {args.cap_kw}")
    print("MONTHLY_MEAN_DAILY_METRICS")
    for _, row in monthly.iterrows():
        print(
            f"{row['month']} days={int(row['days'])} "
            f"rmse={float(row['rmse']):.6f} mae={float(row['mae']):.6f} "
            f"accuracy={float(row['accuracy']):.6f}"
        )
    print("FINAL_MEAN_OF_MONTHS")
    print(f"rmse={final_rmse:.6f} mae={final_mae:.6f} accuracy={final_accuracy:.6f}")
    print("OVERALL_ALL_POINTS")
    print(f"rmse={overall_rmse:.6f}")


if __name__ == "__main__":
    main()
