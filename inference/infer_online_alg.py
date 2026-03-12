import sys
import yaml
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from pvlib import solarposition

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from preprocessing.himawari_infer import create_himawari_tensor
from preprocessing.pv_infer import create_pv_dataframe
from models.models import pv_forecasting_model
import torch

CONF_PATH = _PROJECT_ROOT / "config" / "conf.yaml"


def _resolve_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (_PROJECT_ROOT / p).resolve()


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
            "local_solar_time": local_solar[i].to_pydatetime(),
            "azimuth": float(azimuth[i]),
            "zenith": float(zenith[i]),
            "day_of_year": int(day_of_year[i]),
            "hour_of_day": float(hour_of_day[i]),
        }
        for i in range(len(times_utc))
    ]


def infer_online_main():
    with open(CONF_PATH) as f:
        conf = yaml.safe_load(f)
    pv_path = _resolve_path(conf.get("paths", {}).get("pv_download", "data/pv"))
    sat_path = _resolve_path(conf.get("paths", {}).get("sat_download", "data/himawari"))

    site = conf.get("site", {})
    latitude = site.get("latitude")
    longitude = site.get("longitude")
    

    # Read the PV history data
    timestamps, paths, pv_dict = create_pv_dataframe(path=pv_path)
    timestamps = [t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t for t in timestamps]
    pv_solar_features = compute_solar_features(timestamps, latitude, longitude)
    pv_solar_features_array = []
    for pv_solar_feature in pv_solar_features:
        azimuth_rad = np.deg2rad(pv_solar_feature['azimuth'])
        sin_azimuth = np.sin(azimuth_rad)
        cos_azimuth = np.cos(azimuth_rad)
        zenith_rad = np.deg2rad(pv_solar_feature['zenith'])
        cos_zenith = np.cos(zenith_rad)
        sin_zenith = np.sin(zenith_rad)
        cos_hod = np.cos(2*np.pi*pv_solar_feature['hour_of_day']/24)
        sin_hod = np.sin(2*np.pi*pv_solar_feature['hour_of_day']/24)
        pv_solar_features_array.append(np.array([sin_azimuth, cos_azimuth, sin_zenith, cos_zenith, sin_hod, cos_hod]))
    pv_solar_features_array = np.asarray(pv_solar_features_array).astype(np.float32)
    pv_solar_features_tensor = torch.from_numpy(pv_solar_features_array)

    # the timestamp of the latest PV data (ensure UTC timezone)
    time0 = timestamps[-1]

    # Read satellite images
    latest_sat_timestamp, sat_tensor_np = create_himawari_tensor(sat_path, latitude, longitude, time_span = 10, frame_num = 12, size_km = 500, resample_size = 100)

    # Next 48 hours at 15-minute intervals (192 points), UTC
    forecast_timestamps_utc = [time0 + timedelta(minutes=15 * (i + 1)) for i in range(48 * 4)]

    # Compute solar features
    solar_features = compute_solar_features(forecast_timestamps_utc, latitude, longitude)

    solar_features_array = []
    for solar_feature in solar_features:
        azimuth_rad = np.deg2rad(solar_feature['azimuth'])
        sin_azimuth = np.sin(azimuth_rad)
        cos_azimuth = np.cos(azimuth_rad)
        zenith_rad = np.deg2rad(solar_feature['zenith'])
        cos_zenith = np.cos(zenith_rad)
        sin_zenith = np.sin(zenith_rad)
        cos_dofy = np.cos(2*np.pi*solar_feature['day_of_year']/366)
        sin_dofy = np.sin(2*np.pi*solar_feature['day_of_year']/366)
        cos_hod = np.cos(2*np.pi*solar_feature['hour_of_day']/24)
        sin_hod = np.sin(2*np.pi*solar_feature['hour_of_day']/24)
        solar_features_array.append(np.array([sin_azimuth, cos_azimuth, sin_zenith, cos_zenith, sin_dofy, cos_dofy, sin_hod, cos_hod]))
   
    solar_features_array  = np.asarray(solar_features_array).astype(np.float32)  # [192, 8]
    batch = solar_features_array.shape[0]
    solar_features_tensor = torch.from_numpy(solar_features_array)

    model = pv_forecasting_model(out_dim=64)

    # Expand once: (12, 6) -> (B, 6, 12) for all devices
    pv_ztime = pv_solar_features_tensor.unsqueeze(0).repeat(batch, 1, 1).permute(0, 2, 1)[:,:,-1]  # [B, 6, 12]

    for key, value in pv_dict.items():
        mask = (value['inverter_state']==512).astype(int)
        pv = (value['active_power']/50.0).astype(np.float32)
        pv = torch.from_numpy(pv)
        mask = torch.from_numpy(mask)
        pv = pv.unsqueeze(0).repeat(batch, 1).unsqueeze(1)  # [B, 1, 12]
        pv_mask = mask.unsqueeze(0).repeat(batch, 1)        # [B, 12]
        out = model(torch.tensor([1]), pv, pv_mask, pv_ztime, solar_features_tensor)
        



        

if __name__ == "__main__":
    infer_online_main()