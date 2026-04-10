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


def compute_solar_features(
    forecast_timestamps_utc, latitude: float, longitude: float
) -> list[dict]:
    """
    From UTC forecast times and site lat/lon, compute per timestep:
    - local_solar_time: apparent solar time at the site (naive datetime)
    - azimuth: sun azimuth (degrees)
    - sin_azimuth, cos_azimuth: sin and cos of azimuth (radians)
    - zenith: sun zenith angle (degrees)
    - day_of_year: 1-366
    - hour_of_day: hour in local solar time (0-24, decimal)
    Returns a list of dicts, one per timestep.
    """
    times_utc = pd.to_datetime(forecast_timestamps_utc)
    if times_utc.tz is None:
        times_utc = times_utc.tz_localize("UTC", ambiguous="infer")
    else:
        times_utc = times_utc.tz_convert("UTC")

    # Local solar time (naive)
    local_solar = utc_to_local_solar_time_pvlib(times_utc, longitude)

    # Sun position (pvlib expects localized times; uses UTC)
    solpos = solarposition.get_solarposition(times_utc, latitude, longitude)
    azimuth = solpos["azimuth"].values
    zenith = solpos["zenith"].values

    day_of_year = local_solar.dayofyear.values
    hour_of_day = (
        local_solar.hour.values
        + local_solar.minute.values / 60.0
        + local_solar.second.values / 3600.0
    )

    return [
        {
            "local_solar_time": local_solar[i].floor("us").to_pydatetime(),
            "azimuth": float(azimuth[i]),
            "zenith": float(zenith[i]),
            "day_of_year": int(day_of_year[i]),
            "hour_of_day": float(hour_of_day[i]),
        }
        for i in range(len(times_utc))
    ]

def solar_features_encoder(solar_features):
    solar_features_array = []
    for solar_feature in solar_features:
        azimuth_rad = np.deg2rad(solar_feature['azimuth'])
        sin_azimuth = np.sin(azimuth_rad)
        cos_azimuth = np.cos(azimuth_rad)
        zenith_rad = np.deg2rad(solar_feature['zenith'])
        sin_zenith = np.sin(zenith_rad)
        cos_zenith = np.cos(zenith_rad)
        cos_hod = np.cos(2*np.pi*solar_feature['hour_of_day']/24)
        sin_hod = np.sin(2*np.pi*solar_feature['hour_of_day']/24)
        cos_dofy = np.cos(2*np.pi*solar_feature['day_of_year']/366)
        sin_dofy = np.sin(2*np.pi*solar_feature['day_of_year']/366)
        solar_features_array.append(np.array([sin_azimuth, cos_azimuth, sin_zenith, cos_zenith, sin_dofy, cos_dofy, sin_hod, cos_hod]))
    solar_features_array = np.asarray(solar_features_array).astype(np.float32)
    solar_features_tensor = torch.from_numpy(solar_features_array)
    return solar_features_tensor

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