import sys
import yaml
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from pvlib import solarposition

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from config_utils import get_resolved_paths
from preprocessing.himawari_infer import create_himawari_tensor
from preprocessing.pv_infer import create_pv_dataframe
from models.models import pv_forecasting_model
import torch

CONF_PATH = _PROJECT_ROOT / "config" / "conf.yaml"


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

class infer_online_alg:    # ckpt = torch.load(checkpoint_path, map_location=self.device)
    def __init__(self, checkpoint_path: str):
        with open(CONF_PATH) as f:
            conf = yaml.safe_load(f)
        paths = get_resolved_paths(conf, _PROJECT_ROOT)
        self.pv_path = paths["pv_download"]
        self.sat_path = paths["sat_download"]
        pv_device_path = paths["pv_device_path"]
        site = conf.get("site", {})
        self.latitude = site.get("latitude")
        self.longitude = site.get("longitude")

        pv_device_df = pd.read_excel(pv_device_path)
        self.devDn_list = pv_device_df["devDn"].dropna().unique().tolist()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = pv_forecasting_model(out_dim=64, dev_dn_list=self.devDn_list).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.model.eval()
    
    def inference(self):
        # Read the PV history data
        timestamps, paths, pv_dict = create_pv_dataframe(path=self.pv_path, num=576)
        timestamps = [t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t for t in timestamps]
        pv_solar_features = compute_solar_features(timestamps, self.latitude, self.longitude)
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
        history_solar_features = torch.from_numpy(pv_solar_features_array).to(self.device)
        history_solar_features = history_solar_features.unsqueeze(0)  # [B,C,T]=[1,6,576]
            

        # the timestamp of the latest PV data (ensure UTC timezone)
        time0 = timestamps[-1]

        # Read satellite images
        latest_sat_timestamp, sat_tensor_np = create_himawari_tensor(self.sat_path, self.latitude, self.longitude, time_span = 10, frame_num = 12, size_km = 500, resample_size = 100)

        # Next 48 hours at 15-minute intervals (192 points), UTC
        forecast_timestamps_utc = [time0 + timedelta(minutes=15 * (i + 1)) for i in range(48 * 4)]

        # Compute solar features
        solar_features = compute_solar_features(forecast_timestamps_utc, self.latitude, self.longitude)

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
        forecast_solar_features = torch.from_numpy(solar_features_array).to(self.device)
        forecast_solar_features = forecast_solar_features.unsqueeze(0)  # [B,C,T]=[1,8,192]

        pv_pred_dict = {}
        for key, value in pv_dict.items():
            pv_pred_dict[key] = {}
            dev_idx = torch.tensor(self.devDn_list.index(key), dtype=torch.long, device=self.device)
            dev_idx = dev_idx.unsqueeze(0)
            
            pv_mask = torch.from_numpy((value['inverter_state']==512).astype(np.float32)).to(self.device)
            pv = torch.from_numpy((value['active_power']/50.0).astype(np.float32)).to(self.device)
            pv_ori = torch.from_numpy((value['active_power']).astype(np.float32)).to(self.device)
            
            pv = pv.unsqueeze(0).unsqueeze(1)
            pv_ori = pv_ori.unsqueeze(0).unsqueeze(1)
            pv_mask = pv_mask.unsqueeze(0).unsqueeze(1)

            out = self.model(dev_idx, pv, pv_mask, history_solar_features, forecast_solar_features)

            zenith_mask = (forecast_solar_features[:,:,3]>0.02)
            pv_pred = out*50*zenith_mask

            pv_pred_dict[key]['pv_pred'] = pv_pred.detach().cpu().numpy()
            pv_pred_dict[key]['forecast_timestamps_utc'] = forecast_timestamps_utc
            pv_pred_dict[key]['timestamps'] = timestamps
            pv_pred_dict[key]['pv'] = pv.detach().cpu().numpy().squeeze(axis=1)
            pv_pred_dict[key]['pv_ori'] = pv_ori.detach().cpu().numpy().squeeze(axis=1)
        return pv_pred_dict


if __name__ == "__main__":
    inference_online = infer_online_alg(checkpoint_path='./checkpoints/pv_forecast_epoch_10.pt')
    pv_pred_dict = inference_online.inference()
    print (pv_pred_dict['NE=333858485']['pv_pred'].shape)
    print (len(pv_pred_dict['NE=333858485']['timestamps']))
    print (len(pv_pred_dict['NE=333858485']['forecast_timestamps_utc']))
    print (pv_pred_dict['NE=333858485']['pv'].shape)