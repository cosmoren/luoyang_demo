"""
Training script for pv_forecasting_model.
Loads config from conf.yaml, builds a dataset (synthetic or real), and trains with MSE loss.
"""

import argparse
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
import hashlib
import requests
import re
from urllib.parse import urlparse
from pvlib import solarposition

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config_utils import get_resolved_paths
from models.models import pv_forecasting_model_vit
from preprocessing.himawari_infer import parse_time_from_nc_name
from training.luoyang_data_loader import build_train_test_splits
from datetime import datetime, timedelta, timezone
import os

CONF_PATH = _PROJECT_ROOT / "config" / "conf.yaml"
FORECAST_STEPS = 192  # 48h at 15-min intervals
HISTORY_LEN = 12
SOLAR_FEATURE_DIM = 8
PV_ZTIME_DIM = 6

def load_config():
    with open(CONF_PATH) as f:
        conf = yaml.safe_load(f)
    return conf


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

def delta_time_encoder(timestamps, t0):
    """Hours relative to t0 for each timestamp (negative = before t0)."""
    delta_time = np.array([(ts - t0).total_seconds() / 3600.0 for ts in timestamps],
        dtype=np.float32)
    delta_time_tensor = torch.from_numpy(delta_time / 48)
    return delta_time_tensor

def sat_image_tensor(sat_image_path, time0, stepsize=10, timespan=48):
    if time0.tzinfo is None:
        time0_utc = time0.replace(tzinfo=timezone.utc)
    else:
        time0_utc = time0.astimezone(timezone.utc)
    # Round down to the nearest stepsize-minute boundary (e.g. 10:15 -> 10:10 when stepsize=10)
    time0_utc = time0_utc.replace(minute=(time0_utc.minute // stepsize) * stepsize, second=0, microsecond=0)
    total_steps = int(timespan * 60 // stepsize)
    filenames = [
        (time0_utc - timedelta(minutes=i * stepsize)).strftime("%Y%m%d_%H%M")
        for i in range(total_steps, -1, -1)
    ]
    timestamps = [time0_utc - timedelta(minutes=i * stepsize) for i in range(total_steps, -1, -1)]
    imgs = []
    for filename in filenames:
        if os.path.exists(os.path.join(sat_image_path, filename + ".npy")):
            img = np.load(os.path.join(sat_image_path, filename + ".npy"))
        else:
            img = np.zeros((100, 100, 3), dtype=np.float32)
        imgs.append(img.astype(np.float32))
    img_stack = np.stack(imgs, axis=0).transpose(0,3,1,2)  # (N, 3, 100, 100)
    imgs_tensor = torch.from_numpy(img_stack).float()

    return timestamps, imgs_tensor

def parse_latlon_from_path(path):
    name = str(path)
    # match 112.250_34.650
    m = re.search(r"(\d+\.\d+)_(\d+\.\d+)_UTC", name)
    if m:
        lon = float(m.group(1))
        lat = float(m.group(2))
        return lat, lon
    return None, None

# download + cache remote NWP CSV
def load_csv_with_cache(path, cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path_str = str(path)
    # ---------- 本地文件 ----------
    if not path_str.startswith("http"):
        return pd.read_csv(path, parse_dates=["start_time", "forecast_time"])
    # 用 short hash + original name 作为文件名（避免重复下载）
    parsed = urlparse(path_str)
    base_name = os.path.basename(parsed.path)
    short_hash = hashlib.md5(path_str.encode()).hexdigest()[:8]
    fname = f"{short_hash}_{base_name}"
    cache_path = cache_dir / fname
    # 如果没缓存就下载
    if not cache_path.exists():
        print(f"[NWP] downloading {path_str}")
        r = requests.get(path_str, timeout=60)
        r.raise_for_status()
        with open(cache_path, "wb") as f:
            f.write(r.content)
    return pd.read_csv(cache_path, parse_dates=["start_time", "forecast_time"])

def build_nwp_index_multi(csv_paths, cache_dir, target_lat, target_lon):
    issue_map = {}
    lat0, lon0 = target_lat, target_lon
    for path in csv_paths:
        df = load_csv_with_cache(path, cache_dir)
        df["_source"] = str(path)
        lat, lon = parse_latlon_from_path(path)
        for issue_time, group in df.groupby("start_time"):
            group = group.sort_values("forecast_time")
            dist = (lat - lat0)**2 + (lon - lon0)**2
            if issue_time not in issue_map:
                issue_map[issue_time] = (group, dist)
            else:
                _, best_dist = issue_map[issue_time]
                if dist < best_dist:
                    issue_map[issue_time] = (group, dist)
    issue_map = {k: v[0] for k, v in issue_map.items()}
    issue_times = sorted(issue_map.keys())
    return issue_map, issue_times

def find_latest_time_stochastic(times, t0, *, 
                                prob_second=0.5, to_datetime=True, remove_tz=False,
                                max_delay=None, # e.g. pd.Timedelta(hours=2)
                                ):
    if len(times) == 0:
        return None
    # --- normalize t0 ---
    if to_datetime:
        t0 = pd.Timestamp(t0)
    if remove_tz and getattr(t0, "tzinfo", None):
        t0 = t0.tz_localize(None)
    # --- normalize times ---
    if to_datetime:
        times = [pd.Timestamp(t) for t in times]
    if remove_tz:
        times = [
            t.tz_localize(None) if getattr(t, "tzinfo", None) else t
            for t in times
        ]
    times = np.array(times)
    # --- search ---
    idx = np.searchsorted(times, t0, side="right") - 1
    if idx < 0:
        return None
    # --- candidate indices ---
    candidates = [idx]
    if idx - 1 >= 0:
        candidates.append(idx - 1)
    # --- stochastic selection ---
    if len(candidates) == 2:
        if np.random.rand() < prob_second:
            chosen_idx = candidates[1]  # second latest
        else:
            chosen_idx = candidates[0]  # latest
    else:
        chosen_idx = candidates[0]
    chosen_time = times[chosen_idx]
    # --- max delay control ---
    if max_delay is not None:
        if (t0 - chosen_time) > max_delay:
            return None
    return chosen_time

def build_nwp_tensor(
    t0,
    fcst_times,
    solar_issue_map,
    solar_issue_times,
    wind_issue_map,
    wind_issue_times,
):
    # ========= 统一时间 =========
    t0 = pd.Timestamp(t0)
    if t0.tzinfo is not None:
        t0 = t0.tz_localize(None)
    t0_utc = pd.Timestamp(t0).tz_localize("Asia/Shanghai").tz_convert("UTC").tz_localize(None) # in UTC0
    # ========= forecast timeline =========
    target_times = pd.DatetimeIndex(fcst_times).tz_convert("UTC").tz_localize(None) # in UTC0
    target_times = pd.to_datetime(target_times)
    horizon_steps = len(target_times)
    issue_time = None
    # ========= SOLAR =========
    solar_block = np.zeros((horizon_steps, 1), dtype=np.float32)  # ssrd
    if solar_issue_map is not None:
        issue_time = find_latest_time_stochastic(solar_issue_times, t0_utc, remove_tz=True) # if solar_issue time is UTC0, use t0_utc
        if issue_time is not None:
            df = solar_issue_map[issue_time].copy()
            df["forecast_time"] = pd.to_datetime(df["forecast_time"])
            df["forecast_time"] = df["forecast_time"].dt.tz_localize(None)
            cols = []
            if "ssrd" in df.columns:
                cols.append("ssrd")
            df = df.set_index("forecast_time")[cols]
            full_index = df.index.union(target_times)
            df = df.reindex(full_index).sort_index()
            df = df.interpolate(method="time")
            max_gap = pd.Timedelta("1h")
            prev_valid = df.index.to_series().where(df.notna().any(axis=1)).ffill()
            next_valid = df.index.to_series().where(df.notna().any(axis=1)).bfill()
            gap = next_valid - prev_valid
            df = df.where(gap <= max_gap)
            df = df.loc[target_times]
            df = df.fillna(0.0)
            for i, name in enumerate(["ssrd"]):
                if name in df.columns:
                    solar_block[:, i] = df[name].values.astype(np.float32)
                    # normalize
                    if name == "ssrd":
                        solar_block[:, i] /= 300.0
    # ========= WIND =========
    wind_block = np.zeros((horizon_steps, 5), dtype=np.float32)  # t2m, u10, v10, u100, v100
    if wind_issue_map is not None:
        issue_time = find_latest_time_stochastic(wind_issue_times, t0_utc, remove_tz=True)
        if issue_time is not None:
            df = wind_issue_map[issue_time].copy()
            df["forecast_time"] = pd.to_datetime(df["forecast_time"])
            df["forecast_time"] = df["forecast_time"].dt.tz_localize(None)
            cols = []
            for c in ["t2m", "u10", "v10", "u100", "v100"]:
                if c in df.columns:
                    cols.append(c)
            df = df.set_index("forecast_time")[cols]
            full_index = df.index.union(target_times)
            df = df.reindex(full_index).sort_index()
            df = df.interpolate(method="time")
            max_gap = pd.Timedelta("1h")
            prev_valid = df.index.to_series().where(df.notna().any(axis=1)).ffill()
            next_valid = df.index.to_series().where(df.notna().any(axis=1)).bfill()
            gap = next_valid - prev_valid
            df = df.where(gap <= max_gap)
            df = df.loc[target_times]
            df = df.fillna(0.0)
            for i, name in enumerate(["t2m", "u10", "v10", "u100", "v100"]):
                if name in df.columns:
                    wind_block[:, i] = df[name].values.astype(np.float32)
                    # normalize
                    if name == "t2m":
                        wind_block[:, i] = (wind_block[:, i] - 273.15) / 40.0
                    if name in ["u10", "v10", "u100", "v100"]:
                        wind_block[:, i] /= 10.0
    # ========= issue time diff =====
    issue_diff_hours = (target_times - issue_time).total_seconds() / 3600
    issue_diff_hours = np.clip(issue_diff_hours / 48, 0.0, 1.0) # normalize over 48 hours
    issue_diff_block = np.expand_dims(np.array(issue_diff_hours), 1)
    # ========= CONCAT =========
    X_fcst = np.concatenate([solar_block, wind_block, issue_diff_block], axis=1)  # [T, 7]
    X_fcst = np.transpose(X_fcst, (1, 0)) # [7, T]
    return torch.from_numpy(X_fcst).float()

class PVDataset(Dataset):
    """
    Synthetic dataset with correct shapes for pv_forecasting_model.
    One sample = one device, 12-step history, 192-step forecast target.
    Replace with a real dataset that loads from pv_download + targets when available.
    """

    def __init__(self, data_dir: str, split: str = "train"):
        assert split in ("train", "test"), "split must be 'train' or 'test'"
        self.split = split
        CONF_PATH = _PROJECT_ROOT / "config" / "conf.yaml"
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

        self.samples = build_train_test_splits(
            data_dir=data_dir,
            max_train_per_file=1000,
            max_test_per_file=1000,
            split=split,
        )

        # NWP index
        with open(_PROJECT_ROOT/"config"/"NWP.yaml") as f:
            nwp_conf = yaml.safe_load(f)
        solar_paths = [_PROJECT_ROOT/"datasets/112.285_34.700_UTC0_model_solar_v5.csv", *nwp_conf["solar"]]
        wind_paths = [_PROJECT_ROOT/"datasets/112.285_34.700_UTC0_model_wind_v5.csv", *nwp_conf["wind"]]
        self.solar_issue_mapping, self.solar_issue_times = build_nwp_index_multi(
            solar_paths,
            cache_dir=_PROJECT_ROOT/"datasets",
            target_lat=self.latitude,
            target_lon=self.longitude,
        )
        self.wind_issue_mapping, self.wind_issue_times = build_nwp_index_multi(
            wind_paths,
            cache_dir=_PROJECT_ROOT/"datasets",
            target_lat=self.latitude,
            target_lon=self.longitude,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pred_time_stamp = sample['Y'][:, 0]
        devDn = sample['station_id']    # get the devDn
        dev_idx = torch.tensor(self.devDn_list.index(devDn), dtype=torch.long)
        timestamps = sample['X'][:,0]
        tz_beijing = ZoneInfo("Asia/Shanghai")  # Beijing/Shanghai time (UTC+8)
        timestamps = [pd.Timestamp(t).to_pydatetime().replace(tzinfo=tz_beijing) if pd.Timestamp(t).tz is None else pd.Timestamp(t).astimezone(tz_beijing).to_pydatetime() for t in timestamps]
        pv_mask = torch.from_numpy( (sample['X'][:,1]==512).astype(np.float32) ).unsqueeze(0)
        pv = torch.from_numpy( (sample['X'][:,2]/50).astype(np.float32) ).unsqueeze(0)
        # Compute azimut, zenith, hour of day and day of year
        pv_solar_features = compute_solar_features(timestamps, self.latitude, self.longitude)
        pv_timefeats = solar_features_encoder(pv_solar_features)
        pv_dtimefeats = delta_time_encoder(timestamps, timestamps[-1])
        pv_timefeats = torch.cat([pv_timefeats, pv_dtimefeats.unsqueeze(1)], dim=1)

        # the timestamp of the latest PV data (ensure UTC timezone)
        time0 = timestamps[-1]

        # Next 48 hours at 15-minute intervals (192 points), UTC
        forecast_timestamps_utc = [time0 + timedelta(minutes=15 * (i + 1)) for i in range(48 * 4)]
        #forecast_timestamps_utc = [time0 + timedelta(minutes=15), time0 + timedelta(hours=4)] # test only
        # Compute forecast time features
        forecast_solar_features = compute_solar_features(forecast_timestamps_utc, self.latitude, self.longitude)
        forecast_timefeats = solar_features_encoder(forecast_solar_features)
        forecast_dtimefeats = delta_time_encoder(forecast_timestamps_utc, time0)
        forecast_timefeats = torch.cat([forecast_timefeats, forecast_dtimefeats.unsqueeze(1)], dim=1)

        # load nwp data:
        nwp_timestamps_utc = [time0 + timedelta(minutes=15 * (i + 1)) for i in range(48 * 4)] # can be different from forecast_timestamps_utc
        nwp_solar_features = compute_solar_features(nwp_timestamps_utc, self.latitude, self.longitude)
        nwp_timefeats = solar_features_encoder(nwp_solar_features)
        nwp_dtimefeats = delta_time_encoder(nwp_timestamps_utc, time0)
        nwp_timefeats = torch.cat([nwp_timefeats, nwp_dtimefeats.unsqueeze(1)], dim=1)
        nwp_tensor = build_nwp_tensor(time0, nwp_timestamps_utc, self.solar_issue_mapping, self.solar_issue_times, self.wind_issue_mapping, self.wind_issue_times)

        # load satellite images: NC times in UTC; window [t0-48h, t0] in UTC
        sat_image_path = "/mnt/nfs/Ai4Energy/Datasets/luoyang/luoyang_crop"
        #sat_image_path = "/home/weize/ai4energy_crop/" # test only
        sat_timestamps_utc, sat_tensor = sat_image_tensor(sat_image_path, time0, stepsize=10, timespan=5)
        sat_solar_features = compute_solar_features(sat_timestamps_utc, self.latitude, self.longitude)
        sat_timefeats = solar_features_encoder(sat_solar_features)
        sat_dtimefeats = delta_time_encoder(sat_timestamps_utc, time0)
        sat_timefeats = torch.cat([sat_timefeats, sat_dtimefeats.unsqueeze(1)], dim=1)

        # target
        target_pv = torch.from_numpy( (sample['Y'][:,2]/50).astype(np.float32) )
        target_mask = torch.from_numpy( (sample['Y'][:,1]==512).astype(np.float32) )
        #target_mask[:] = 0 # test only
        #target_mask[15] = 1 # test only 0 or 15
        return {
            "pred_time_stamp": pred_time_stamp, # [192]
            "dev_idx": dev_idx,
            "pv": pv,
            "pv_mask": pv_mask,
            "pv_timefeats": pv_timefeats,
            "forecast_timefeats": forecast_timefeats,
            "sat_tensor": sat_tensor,
            "sat_timefeats": sat_timefeats,
            "skimg_tensor": None,
            "skimg_timefeats": None,
            "nwp_tensor": nwp_tensor, # [7, 192]
            "nwp_timefeats": nwp_timefeats, # [192, 8]
            "target_pv": target_pv,
            "target_mask": target_mask,
        }


def collate_single(batch):
    """Collate so we get a list of samples; we run model once per sample (batch_size 1 or loop)."""
    return batch


def collate_batched(batch):
    """Stack list of samples into one dict of tensors with batch dim B in front."""
    pv_tf = torch.stack([s["pv_timefeats"] for s in batch])
    fc_tf = torch.stack([s["forecast_timefeats"] for s in batch])
    return {
        "pred_time_stamp": np.stack([pd.to_datetime(s["pred_time_stamp"]) for s in batch]),
        "dev_idx": torch.stack([s["dev_idx"] for s in batch]),
        "pv": torch.stack([s["pv"] for s in batch]),
        "pv_mask": torch.stack([s["pv_mask"] for s in batch]),
        "pv_timefeats": pv_tf,
        "forecast_timefeats": fc_tf,
        "history_solar_features": pv_tf,
        "forecast_solar_features": fc_tf,
        "sat_tensor": torch.stack([s["sat_tensor"] for s in batch]),
        "sat_timefeats": torch.stack([s["sat_timefeats"] for s in batch]),
        "skimg_tensor": None,
        "skimg_timefeats": None,
        "nwp_tensor": torch.stack([s["nwp_tensor"] for s in batch]),
        "nwp_timefeats": torch.stack([s["nwp_timefeats"] for s in batch]),
        "target_pv": torch.stack([s["target_pv"] for s in batch]),
        "target_mask": torch.stack([s["target_mask"] for s in batch]),
    }


def train_one_epoch(model, device, loader, criterion, optimizer, scaler):
    model.train()
    total_loss = 0.0
    n = 0
    num_batches = len(loader)
    print('number of batches: ', num_batches)
    for batch_idx, batch in enumerate(loader):
        if batch_idx > 3000:
            break
        B = batch["dev_idx"].size(0)
        device_id = batch["dev_idx"].to(device)
        pv = batch["pv"].to(device)
        mask = batch["pv_mask"].to(device)
        pv_timefeats = batch["pv_timefeats"].to(device)
        forecast_timefeats = batch["forecast_timefeats"].to(device)
        sat_tensor = batch["sat_tensor"].to(device) if batch["sat_tensor"] is not None else None
        sat_timefeats = batch["sat_timefeats"].to(device) if batch["sat_timefeats"] is not None else None
        skimg_tensor = batch["skimg_tensor"].to(device) if batch["skimg_tensor"] is not None else None
        skimg_timefeats = batch["skimg_timefeats"].to(device) if batch["skimg_timefeats"] is not None else None
        nwp_tensor = batch["nwp_tensor"].to(device) if batch["nwp_tensor"] is not None else None
        nwp_timefeats = batch["nwp_timefeats"].to(device) if batch["nwp_timefeats"] is not None else None
        target_pv = batch["target_pv"].to(device)
        target_mask = batch["target_mask"].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            pv_pred = model(
                device_id,
                pv,
                pv_mask=mask,
                pv_timefeats=pv_timefeats,
                forecast_timefeats=forecast_timefeats,
                sat_tensor=sat_tensor,
                sat_timefeats=sat_timefeats,
                skimg_tensor=skimg_tensor,
                skimg_timefeats=skimg_timefeats,
                nwp_tensor=nwp_tensor,
                nwp_timefeats=nwp_timefeats,
            )
            loss = criterion(pv_pred * target_mask, target_pv * target_mask)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        #loss.backward()
        #optimizer.step()
        total_loss += loss.item()
        n += B
    print()
    return total_loss / max(n, 1)

def calc_metrics_from_inv_statistics(inv_statistics):
    total_target_pv_profile = {}
    total_pv_pred_profile = {}
    valid_sample_mask = None
    for v in inv_statistics.values():
        for i in range(v['count']):
            start_time_stamp = v['pred_time_stamp'][i][0]
            key = pd.Timestamp(start_time_stamp).to_datetime64()
            valid_sample_mask = v['mask'][i].cpu().numpy() if not isinstance(valid_sample_mask, np.ndarray) else valid_sample_mask
            if key not in total_pv_pred_profile:
                total_pv_pred_profile[key] = (v['pv_pred'][i] * v['mask'][i]).cpu().numpy()
            else:
                total_pv_pred_profile[key] += (v['pv_pred'][i] * v['mask'][i]).cpu().numpy()
            if key not in total_target_pv_profile:
                total_target_pv_profile[key] = (v['target_pv'][i] * v['mask'][i]).cpu().numpy()
            else:
                total_target_pv_profile[key] += (v['target_pv'][i] * v['mask'][i]).cpu().numpy()
    err_list = []
    pv_scal_factor = 50 # from Yuan Ren, PVs were normalized over 50kw
    for key in total_pv_pred_profile:
        valid = valid_sample_mask > 0
        err_for_1_time_stamp = (total_pv_pred_profile[key] - total_target_pv_profile[key])[valid]
        for i in range(err_for_1_time_stamp.shape[0]):
            err_list.append(err_for_1_time_stamp[i])
    err_list = np.array(err_list)
    mae = np.mean(np.abs(err_list)) * pv_scal_factor
    rmse = np.sqrt(np.mean(err_list ** 2)) * pv_scal_factor

    return mae, rmse

def evaluate(model, device, loader, criterion):
    """Compute mean loss on a dataset (e.g. test set). No gradient."""
    model.eval()
    total_loss = 0.0
    n = 0
    inv_statistics = {}
    with torch.no_grad():
        for batch in loader:
            B = batch["dev_idx"].size(0)
            device_id = batch["dev_idx"].to(device)
            pred_time_stamp = batch["pred_time_stamp"]
            pv = batch["pv"].to(device)
            mask = batch["pv_mask"].to(device)
            pv_timefeats = batch["pv_timefeats"].to(device)
            forecast_timefeats = batch["forecast_timefeats"].to(device)
            sat_tensor = batch["sat_tensor"].to(device) if batch["sat_tensor"] is not None else None
            sat_timefeats = batch["sat_timefeats"].to(device) if batch["sat_timefeats"] is not None else None
            skimg_tensor = batch["skimg_tensor"].to(device) if batch["skimg_tensor"] is not None else None
            skimg_timefeats = batch["skimg_timefeats"].to(device) if batch["skimg_timefeats"] is not None else None
            nwp_tensor = batch["nwp_tensor"].to(device) if batch["nwp_tensor"] is not None else None
            nwp_timefeats = batch["nwp_timefeats"].to(device) if batch["nwp_timefeats"] is not None else None
            target_pv = batch["target_pv"].to(device)
            target_mask = batch["target_mask"].to(device)
            pv_pred = model(
                device_id,
                pv,
                pv_mask=mask,
                pv_timefeats=pv_timefeats,
                forecast_timefeats=forecast_timefeats,
                sat_tensor=sat_tensor,
                sat_timefeats=sat_timefeats,
                skimg_tensor=skimg_tensor,
                skimg_timefeats=skimg_timefeats,
                nwp_tensor=nwp_tensor,
                nwp_timefeats=nwp_timefeats,
            )
            for i in range(B):
                idx = str(device_id[i].item())
                if idx not in inv_statistics:
                    inv_statistics[idx] = {'count': 1,
                    'pred_time_stamp': [pred_time_stamp[i]],
                    'target_pv': [target_pv[i]], 'pv_pred': [pv_pred[i]], 'mask': [target_mask[i]],
                    'masked_pv_err': [(target_pv[i] - pv_pred[i]) * target_mask[i]]}
                else:
                    inv_statistics[idx]['count'] += 1
                    inv_statistics[idx]['pred_time_stamp'].append(pred_time_stamp[i])
                    inv_statistics[idx]['target_pv'].append(target_pv[i])
                    inv_statistics[idx]['pv_pred'].append(pv_pred[i])
                    inv_statistics[idx]['mask'].append(target_mask[i])
                    inv_statistics[idx]['masked_pv_err'].append((target_pv[i] - pv_pred[i]) * target_mask[i])
            loss = criterion(pv_pred * target_mask, target_pv * target_mask)
            total_loss += loss.item()
            n += B
            #if n % (B * 10) == 0:
            #    print("n ", n , " B*(len loader) ", B*len(loader))
    mae, rmse = calc_metrics_from_inv_statistics(inv_statistics)
    print("mae ", mae, " rmse ", rmse, " max_pv_output ", 54.6*1000)
    return total_loss / max(n, 1)


def main(eval_only = False):
    parser = argparse.ArgumentParser(description="Train PV forecasting model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=5)
    args = parser.parse_args()

    conf = load_config()
    paths = get_resolved_paths(conf, _PROJECT_ROOT)
    pv_device_path = paths["pv_device_path"]
    if pv_device_path is None or not pv_device_path.is_file():
        raise FileNotFoundError(f"pv_device_path not found: {pv_device_path}")
    pv_device_df = pd.read_excel(pv_device_path)
    dev_dn_list = pv_device_df["devDn"].dropna().unique().tolist()
    num_devices = len(dev_dn_list)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not eval_only:
        model = pv_forecasting_model_vit(dev_dn_list=dev_dn_list).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    data_dir = "/mnt/nfs/Ai4Energy/Datasets/luoyang_data_626/"
    #data_dir = "/home/weize/remote_test_data/" # test only
    #data_dir = "/home/weize/remote_luoyang_data_626/" # test only
    if not eval_only:
        train_dataset = PVDataset(data_dir=data_dir, split="train")
    test_dataset = PVDataset(data_dir=data_dir, split="test")

    if not eval_only:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_batched,
            num_workers=24, # test only, default 24, could be smaller for memory issue
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batched,
        num_workers=24, # test only, default 24, could be smaller for memory issue
    )
    if eval_only:
        ckpt_path = _PROJECT_ROOT / "checkpoints_192/pv_forecast_epoch_10.pt"
        ckpt = torch.load(ckpt_path, map_location=device)
        model = pv_forecasting_model_vit(dev_dn_list=ckpt["dev_dn_list"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        eval_loss = evaluate(model, device, test_loader, criterion)
        print(f"Eval loss: {eval_loss:.6f}")
        return

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else _PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    initial_test_loss = evaluate(model, device, test_loader, criterion)
    print(f"Initial test loss: {initial_test_loss:.6f}")

    scaler = torch.amp.GradScaler('cuda') # mixed precision
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(model, device, train_loader, criterion, optimizer, scaler)
        test_loss = evaluate(model, device, test_loader, criterion)
        print(f"Epoch {epoch}/{args.epochs}  train_loss={avg_loss:.6f}  test_loss={test_loss:.6f}")

        if args.save_every and epoch % args.save_every == 0:
            path = checkpoint_dir / f"pv_forecast_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "dev_dn_list": dev_dn_list,
            }, path)
            print(f"  saved {path}")

    final_path = checkpoint_dir / "pv_forecast_final.pt"
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "dev_dn_list": dev_dn_list,
    }, final_path)
    print(f"Saved final checkpoint to {final_path}")


if __name__ == "__main__":
    main(eval_only = False)
