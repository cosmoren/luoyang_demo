import numpy as np
import pandas as pd
from pvlib import solarposition
import torch


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


def extract_solar_features(df) -> dict[str, np.ndarray]:
    """Return per-timestep solar features as a dict of ``np.ndarray``s (length T)."""
    local_solar_parsed = pd.to_datetime(
        df["local_solar_time"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    local_solar_parsed_us = pd.to_datetime(
        df["local_solar_time"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce"
    )
    local_solar = local_solar_parsed.fillna(local_solar_parsed_us).values

    azimuth = pd.to_numeric(df["solar_azimuth"], errors="coerce").fillna(0).values.astype(np.float32)
    zenith = pd.to_numeric(df["solar_zenith"], errors="coerce").fillna(0).values.astype(np.float32)
    day_of_year = pd.to_numeric(df["day_of_year"], errors="coerce").fillna(0).values.astype(np.int32)
    hour_of_day = pd.to_numeric(df["hour_of_day"], errors="coerce").fillna(0).values.astype(np.float32)

    return {
        "local_solar_time": local_solar,
        "azimuth": azimuth,
        "zenith": zenith,
        "day_of_year": day_of_year,
        "hour_of_day": hour_of_day,
    }



def compute_solar_features(
    forecast_timestamps_utc, latitude: float, longitude: float
) -> dict[str, np.ndarray]:
    """
    From UTC forecast times and site lat/lon, compute per timestep:
    - local_solar_time: apparent solar time at the site (naive datetime64[ns])
    - azimuth: sun azimuth (degrees)
    - zenith: sun zenith angle (degrees)
    - day_of_year: 1-366
    - hour_of_day: hour in local solar time (0-24, decimal)
    Returns a dict of length-T ``np.ndarray``s (one entry per field).
    """
    times_utc = pd.to_datetime(forecast_timestamps_utc)
    if times_utc.tz is None:
        times_utc = times_utc.tz_localize("UTC", ambiguous="infer")
    else:
        times_utc = times_utc.tz_convert("UTC")

    local_solar = utc_to_local_solar_time_pvlib(times_utc, longitude)

    solpos = solarposition.get_solarposition(times_utc, latitude, longitude)
    azimuth = solpos["azimuth"].values.astype(np.float32)
    zenith = solpos["zenith"].values.astype(np.float32)

    day_of_year = local_solar.dayofyear.values.astype(np.int32)
    hour_of_day = (
        local_solar.hour.values
        + local_solar.minute.values / 60.0
        + local_solar.second.values / 3600.0
    ).astype(np.float32)

    return {
        "local_solar_time": local_solar.values,
        "azimuth": azimuth,
        "zenith": zenith,
        "day_of_year": day_of_year,
        "hour_of_day": hour_of_day,
    }

def solar_features_encoder(solar_features: dict[str, np.ndarray]) -> torch.Tensor:
    """Vectorized encoder. ``solar_features`` is a dict of length-T arrays:
    ``azimuth`` / ``zenith`` (deg), ``hour_of_day`` (0-24), ``day_of_year`` (1-366).
    Returns a ``[T, 8]`` float32 tensor with columns
    ``[sin_az, cos_az, sin_ze, cos_ze, sin_doy, cos_doy, sin_hod, cos_hod]``.
    """
    azimuth = np.asarray(solar_features["azimuth"], dtype=np.float32)
    zenith = np.asarray(solar_features["zenith"], dtype=np.float32)
    hour_of_day = np.asarray(solar_features["hour_of_day"], dtype=np.float32)
    day_of_year = np.asarray(solar_features["day_of_year"], dtype=np.float32)

    azimuth_rad = np.deg2rad(azimuth)
    zenith_rad = np.deg2rad(zenith)
    hod_rad = (2.0 * np.pi / 24.0) * hour_of_day
    doy_rad = (2.0 * np.pi / 366.0) * day_of_year

    feats = np.stack(
        [
            np.sin(azimuth_rad),
            np.cos(azimuth_rad),
            np.sin(zenith_rad),
            np.cos(zenith_rad),
            np.sin(doy_rad),
            np.cos(doy_rad),
            np.sin(hod_rad),
            np.cos(hod_rad),
        ],
        axis=-1,
    ).astype(np.float32, copy=False)
    return torch.from_numpy(feats)

def _as_utc_naive_pd(ts) -> pd.Timestamp:
    """Normalize to UTC wall-clock as tz-naive so deltas are well-defined across mixed inputs."""
    t = pd.Timestamp(ts)
    if t.tz is not None:
        return t.tz_convert("UTC").tz_localize(None)
    return t


def delta_time_encoder(timestamps, t0):
    """Hours relative to t0 for each timestamp (negative = before t0)."""
    t0_u = _as_utc_naive_pd(t0)
    delta_time = np.array(
        [(_as_utc_naive_pd(ts) - t0_u).total_seconds() / 3600.0 for ts in timestamps],
        dtype=np.float32,
    )
    delta_time_tensor = torch.from_numpy(delta_time / 48)
    return delta_time_tensor