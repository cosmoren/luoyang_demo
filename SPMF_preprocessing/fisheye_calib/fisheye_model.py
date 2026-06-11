# u = cx + f * zenith * sin(azimuth - alpha0)
# v = cy - f * zenith * cos(azimuth - alpha0)
# r = f * zenith
# 10 sunny day detections are provided to fit this model
'''
python SPMF_preprocessing/fisheye_calib/fisheye_model.py \
  --input-dir /path/to/skyimg_labeled/asi16/asi_16613 \
  --output-csv /path/to/fisheye_fit_result.csv \
  --latitude 34.7 \
  --longitude 112.3 \
  --timezone Asia/Shanghai \
  --day-start 09:00 \
  --day-end 16:00
'''

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
from scipy.optimize import least_squares

IMAGE_SIZE = 224


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a simple equidistant fisheye model from labeled sky-image CSVs.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="skyimg_labeled/asi16/asi_16613/",
        help="Root directory; CSVs are read from ``<input-dir>/*/*.csv``.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="fisheye_fit_result.csv",
        help="Path to write fitted parameters (cx, cy, f, alpha0).",
    )
    parser.add_argument("--latitude", type=float, default=34.7, help="Site latitude (deg).")
    parser.add_argument("--longitude", type=float, default=112.3, help="Site longitude (deg).")
    parser.add_argument(
        "--timezone",
        type=str,
        default="Asia/Shanghai",
        help="Local timezone for daytime filtering (collectTime in CSV is UTC).",
    )
    parser.add_argument(
        "--day-start",
        type=str,
        default="09:00",
        help="Local daytime window start, HH:MM (inclusive).",
    )
    parser.add_argument(
        "--day-end",
        type=str,
        default="16:00",
        help="Local daytime window end, HH:MM (inclusive).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=IMAGE_SIZE,
        help="Image width/height in pixels (used for u flip and bounds check).",
    )
    return parser.parse_args()


def predict_simple(params, az, zen):
    cx, cy, f, alpha0 = params

    r = f * zen

    u = cx + r * np.sin(az - alpha0)
    v = cy - r * np.cos(az - alpha0)

    return u, v


def residuals_simple(params, az, zen, u_obs, v_obs):
    u_pred, v_pred = predict_simple(params, az, zen)

    return np.concatenate([
        u_pred - u_obs,
        v_pred - v_obs,
    ])


def main() -> None:
    args = parse_args()

    csv_root = Path(args.input_dir).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    latitude = float(args.latitude)
    longitude = float(args.longitude)
    local_timezone = str(args.timezone)
    day_start = str(args.day_start)
    day_end = str(args.day_end)
    image_size = int(args.image_size)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"input_dir   : {csv_root}")
    print(f"output_csv  : {output_csv}")
    print(f"lat/lon     : {latitude}, {longitude}")
    print(f"timezone    : {local_timezone}")
    print(f"day window  : {day_start} ~ {day_end} (local)")

    # =========================
    # Load all CSV files
    # =========================
    csv_files = sorted(glob.glob(str(csv_root / "*" / "*.csv")))

    if len(csv_files) == 0:
        raise ValueError(f"No CSV files found under: {csv_root}/*/*.csv")

    dfs = []

    for path in csv_files:
        temp = pd.read_csv(path)

        required_cols = ["collectTime", "u", "v"]
        missing = [c for c in required_cols if c not in temp.columns]

        if missing:
            print(f"Skip {path}: missing columns {missing}")
            continue

        temp["date_folder"] = os.path.basename(os.path.dirname(path))
        temp["source_csv"] = os.path.basename(path)
        temp["source_path"] = path

        dfs.append(temp)

        print(f"Loaded: {path}, rows={len(temp)}")

    if len(dfs) == 0:
        raise ValueError("No valid CSV files loaded.")

    df = pd.concat(dfs, ignore_index=True)

    print(f"\nTotal loaded rows: {len(df)}")

    # =========================
    # Parse UTC timestamp
    # =========================
    df["collectTime"] = pd.to_datetime(df["collectTime"], utc=True, errors="coerce")

    df = df.dropna(subset=["collectTime"]).copy()

    # =========================
    # Filter by local daytime
    # =========================
    df["local_time"] = df["collectTime"].dt.tz_convert(local_timezone)

    before_time_filter = len(df)

    df = df[
        (df["local_time"].dt.strftime("%H:%M") >= day_start) &
        (df["local_time"].dt.strftime("%H:%M") <= day_end)
    ].copy()

    print(f"Rows after local daytime filtering: {len(df)} / {before_time_filter}")

    # =========================
    # Calculate solar position
    # =========================
    solpos = pvlib.solarposition.get_solarposition(
        time=df["collectTime"],
        latitude=latitude,
        longitude=longitude,
    )

    df["azimuth"] = solpos["azimuth"].values
    df["zenith"] = solpos["zenith"].values

    # =========================
    # Clean samples
    # =========================
    df["u"] = pd.to_numeric(df["u"], errors="coerce")
    df["v"] = pd.to_numeric(df["v"], errors="coerce")

    # horizontally flip the image
    df["u"] = (image_size - 1) - df["u"]

    df = df.dropna(subset=["azimuth", "zenith", "u", "v"]).copy()

    df = df[
        (df["u"] >= 0) &
        (df["u"] < image_size) &
        (df["v"] >= 0) &
        (df["v"] < image_size)
    ].copy()

    if len(df) < 10:
        raise ValueError(f"Too few valid samples after filtering: {len(df)}")

    print(f"Valid samples after filtering: {len(df)}")

    # =========================
    # Prepare arrays
    # =========================
    az = np.deg2rad(df["azimuth"].to_numpy(dtype=float))
    zen = np.deg2rad(df["zenith"].to_numpy(dtype=float))

    u_obs = df["u"].to_numpy(dtype=float)
    v_obs = df["v"].to_numpy(dtype=float)

    # =========================
    # Initial guess
    # =========================
    cx0 = image_size / 2
    cy0 = image_size / 2
    f0 = image_size / np.pi
    alpha0_0 = 0.0

    x0 = np.array([cx0, cy0, f0, alpha0_0], dtype=float)

    # =========================
    # Fit simple model
    # =========================
    res = least_squares(
        residuals_simple,
        x0,
        args=(az, zen, u_obs, v_obs),
        loss="huber",
        f_scale=5.0,
    )

    cx, cy, f, alpha0 = res.x

    # normalize alpha0 to [0, 360)
    alpha0_deg = np.rad2deg(alpha0) % 360

    # =========================
    # Evaluate
    # =========================
    u_pred, v_pred = predict_simple(res.x, az, zen)

    err = np.sqrt((u_pred - u_obs) ** 2 + (v_pred - v_obs) ** 2)

    print("\n===== GLOBAL SIMPLE EQUIDISTANT FISHEYE FIT =====")
    print(f"cx       = {cx:.4f}")
    print(f"cy       = {cy:.4f}")
    print(f"f        = {f:.4f} px/rad")
    print(f"alpha0   = {alpha0:.6f} rad")
    print(f"alpha0   = {alpha0_deg:.4f} deg")

    print("\n===== ERROR =====")
    print(f"num samples   = {len(df)}")
    print(f"mean error    = {err.mean():.4f} px")
    print(f"median error  = {np.median(err):.4f} px")
    print(f"max error     = {err.max():.4f} px")

    # =========================
    # Per-day diagnostic
    # =========================
    day_stats = (
        df.assign(pixel_error=err)
        .groupby("date_folder")["pixel_error"]
        .agg(["count", "mean", "median", "max"])
        .reset_index()
    )

    print("\n===== PER-DAY ERROR =====")
    print(day_stats.to_string(index=False))

    # =========================
    # Save result
    # =========================
    params_df = pd.DataFrame([{
        "cx": cx,
        "cy": cy,
        "f": f,
        "alpha0": alpha0,
    }])

    params_df.to_csv(output_csv, index=False)

    print(f"\nSaved result to: {output_csv}")


if __name__ == "__main__":
    main()
