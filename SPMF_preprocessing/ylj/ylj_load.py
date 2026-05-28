import numpy as np
import pandas as pd
from datetime import timedelta, timezone
import pvlib

UTC = timezone.utc
UTC_PLUS_8 = timezone(timedelta(hours=8))
FEATURE_KEYS = [
    "interp_times",
    "interp_power",
    "p_cs",
    "solar_time_features",
    "kt",
    "kt_mask",
    "forecast_times_utc",
    "forecast_solar_time_features",
    "forecast_p_cs",
    "forecast_kt",
    "forecast_kt_mask",
]

def compute_clearsky_power(lat: float, lon: float, utc_index: pd.DatetimeIndex) -> pd.Series:
    """Normalized clear-sky POA (``poa_global / 1000``), clipped to [0, 1.2]."""
    tilt = abs(lat)  # unknown tilt: rough latitude-tilt assumption
    azimuth = 180 if lat >= 0 else 0  # north: south-facing; south: north-facing
    solpos = pvlib.solarposition.get_solarposition(utc_index, lat, lon)

    loc = pvlib.location.Location(lat, lon)
    # 1. Clear-sky GHI/DNI/DHI
    cs = loc.get_clearsky(utc_index, model="ineichen")

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=cs["dni"],
        ghi=cs["ghi"],
        dhi=cs["dhi"],
        albedo=0.2,
    )

    poa_clear = poa["poa_global"].clip(lower=0)
    p_cs = (poa_clear / 1000).clip(0, 1.5)

    return p_cs


def compute_solar_time_features(
    interp_times: pd.DatetimeIndex, lat: float, lon: float
) -> pd.DataFrame:
    utc_index = pd.DatetimeIndex(interp_times)
    if utc_index.tz is None:
        utc_index = utc_index.tz_localize(UTC)
    else:
        utc_index = utc_index.tz_convert(UTC)

    sp = pvlib.solarposition.get_solarposition(utc_index, lat, lon)
    eot_min = sp["equation_of_time"].to_numpy()

    offset = pd.to_timedelta(4.0 * float(lon) + eot_min, unit="min")
    utc_naive = utc_index.tz_convert(UTC).tz_localize(None)
    lst_index = utc_naive + offset

    return pd.DataFrame(
        {
            "interp_times": utc_index,
            "solar_zenith": sp["apparent_zenith"].to_numpy(),
            "solar_azimuth": sp["azimuth"].to_numpy(),
            "local_solar_time": lst_index.strftime("%Y-%m-%d %H:%M:%S"),
            "day_of_year": lst_index.dayofyear.to_numpy(),
            "hour_of_day": (
                lst_index.hour
                + lst_index.minute / 60.0
                + lst_index.second / 3600.0
            ).to_numpy(),
        }
    )


def interpolate_observe_power_last_48h(row: pd.Series) -> tuple[pd.DatetimeIndex, np.ndarray]:
    observe_power = np.asarray(row["observe_power"], dtype=np.float64)
    if observe_power.ndim == 0:
        raise ValueError("observe_power is not a sequence")
    end_time = pd.Timestamp(row["timestamp_win"])
    if end_time.tzinfo is None:
        end_time = end_time.tz_localize(UTC_PLUS_8)
    else:
        end_time = end_time.tz_convert(UTC_PLUS_8)
    end_time = end_time.tz_convert(UTC)

    source_times = pd.date_range(end=end_time, periods=len(observe_power), freq="15min")
    target_times = pd.date_range(end=end_time, periods=576, freq="5min")

    # Keep just enough source history to cover the 5-minute target grid.
    keep = source_times >= (target_times[0] - pd.Timedelta(minutes=10))
    source_times = source_times[keep]
    observe_power = observe_power[keep]

    source_ns = source_times.view("int64")
    target_ns = target_times.view("int64")
    interp_power = np.interp(target_ns, source_ns, observe_power)
    return target_times, interp_power


def _to_parquet_value(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Index, pd.DatetimeIndex)):
        return [_to_parquet_value(v) for v in value.tolist()]
    if isinstance(value, pd.Series):
        return [_to_parquet_value(v) for v in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return {
            col: [_to_parquet_value(v) for v in value[col].tolist()]
            for col in value.columns
        }
    if isinstance(value, dict):
        return {k: _to_parquet_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_parquet_value(v) for v in value]
    return value


def compute_row_feature_scaffold(row: pd.Series, lat: float, lon: float) -> dict:
    # historical data
    interp_times, interp_power = interpolate_observe_power_last_48h(row)
    p_cs = np.asarray(compute_clearsky_power(lat, lon, interp_times), dtype=np.float32)
    solar_time_features = compute_solar_time_features(interp_times, lat, lon)
    kt_mask = (p_cs > 0.10).astype(np.float32)
    kt = (interp_power.astype(np.float32) / (p_cs + 1e-6)) * kt_mask

    # forecast data
    forecast_times = pd.date_range(
        start=row["timestamp_win"] + pd.Timedelta(minutes=15),
        periods=192,
        freq="15min",
    )
    forecast_times = forecast_times.tz_localize(UTC_PLUS_8)
    forecast_times_utc = forecast_times.tz_convert(UTC)
    forecast_solar_time_features = compute_solar_time_features(forecast_times_utc, lat, lon)
    forecast_p_cs = np.asarray(compute_clearsky_power(lat, lon, forecast_times_utc), dtype=np.float32)
    forecast_kt_mask = (forecast_p_cs > 0.10).astype(np.float32)
    forecast_kt = (
        np.asarray(row["observe_power_future"], dtype=np.float32) / (forecast_p_cs + 1e-6)
    ) * forecast_kt_mask

    return {
        "interp_times": interp_times,
        "interp_power": interp_power.astype(np.float32),
        "p_cs": p_cs,
        "solar_time_features": solar_time_features,
        "kt": kt.astype(np.float32),
        "kt_mask": kt_mask,
        "forecast_times_utc": forecast_times_utc,
        "forecast_solar_time_features": forecast_solar_time_features,
        "forecast_p_cs": forecast_p_cs,
        "forecast_kt": forecast_kt.astype(np.float32),
        "forecast_kt_mask": forecast_kt_mask,
    }


def build_augmented_row(row: pd.Series, lat: float, lon: float) -> dict:
    out = {key: _to_parquet_value(value) for key, value in row.items()}

    try:
        features = compute_row_feature_scaffold(row, lat, lon)
        for key in FEATURE_KEYS:
            out[key] = _to_parquet_value(features[key])
    except (TypeError, ValueError):
        # Keep the original row even if this sample cannot be converted.
        for key in FEATURE_KEYS:
            out[key] = None

    return out


def write_augmented_parquet_example(
    input_path: str, output_path: str, lat: float, lon: float
) -> None:
    df = pd.read_parquet(input_path)
    out_rows = []
    total = len(df)

    for i in range(total):
        if i % 100 == 0:
            print(f"processing row {i}/{total}")
        out_rows.append(build_augmented_row(df.iloc[i], lat, lon))

    out_df = pd.DataFrame(out_rows, columns=list(df.columns) + FEATURE_KEYS)
    out_df.to_parquet(output_path, index=False)
    print(f"wrote: {output_path}")
    print(f"shape: {out_df.shape}")

def main():
    lat = 29.94
    lon = 100.62
    input_path = "/work/datasets/ylj_raw/ds_v322_1219_2025_1-12.parquet"
    output_path = "./ds_v322_2025_with_kt.parquet"
    write_augmented_parquet_example(input_path, output_path, lat, lon)

if __name__ == "__main__":
    main()
