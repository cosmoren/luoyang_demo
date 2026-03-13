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
from pvlib import solarposition

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config_utils import get_resolved_paths
from models.models import pv_forecasting_model
from training.luoyang_data_loader import build_train_test_splits
from datetime import datetime, timedelta, timezone

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


class PVDataset(Dataset):
    """
    Synthetic dataset with correct shapes for pv_forecasting_model.
    One sample = one device, 12-step history, 192-step forecast target.
    Replace with a real dataset that loads from pv_download + targets when available.
    """

    def __init__(self, data_dir: str):

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
        
        self.train_list = build_train_test_splits(
            data_dir=data_dir,
            max_train_per_file = 2500,
            max_test_per_file = 3200,
            split="train",
        )

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):
        sample = self.train_list[idx]
        devDn = sample['station_id']    # get the devDn
        dev_idx = self.devDn_list.index(devDn)
        timestamps = sample['X'][:,0]
        tz_beijing = ZoneInfo("Asia/Shanghai")  # Beijing/Shanghai time (UTC+8)
        timestamps = [pd.Timestamp(t).to_pydatetime().replace(tzinfo=tz_beijing) if pd.Timestamp(t).tz is None else pd.Timestamp(t).astimezone(tz_beijing).to_pydatetime() for t in timestamps]
        pv_mask = (sample['X'][:,1]==512).astype(np.float32)
        pv = (sample['X'][:,2]/50).astype(np.float32)
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
        pv_solar_features_tensor = torch.from_numpy(pv_solar_features_array)

        # the timestamp of the latest PV data (ensure UTC timezone)
        time0 = timestamps[-1]

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
        batch = solar_features_array.shape[0]
        solar_features_tensor = torch.from_numpy(solar_features_array)

        # Expand once: (12, 6) -> (B, 6, 12) for all devices
        pv_ztime = pv_solar_features_tensor.unsqueeze(0).repeat(batch, 1, 1).permute(0, 2, 1)[:,:,-1]  # [B, 6, 12]

        # target
        target_pv = (sample['Y'][:,2]/50).astype(np.float32)
        target_mask = (sample['Y'][:,1]==512).astype(np.float32)

        return {
            "dev_idx": dev_idx,
            "pv": pv,
            "pv_mask": pv_mask,
            "pv_ztime": pv_ztime,
            "solar_features": solar_features_tensor,
            "target_pv": target_pv,
            "target_mask": target_mask,
        }


def collate_single(batch):
    """Collate so we get a list of samples; we run model once per sample (batch_size 1 or loop)."""
    return batch


def train_one_epoch(model, device, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    n = 0
    for batch_list in loader:
        optimizer.zero_grad()
        batch_loss = 0.0
        for sample in batch_list:
            device_id = torch.tensor([sample["dev_idx"]], dtype=torch.long, device=device)
            # Model expects x (B, 1, 12) with B=192 (same history repeated for each forecast step)
            pv = torch.from_numpy(sample["pv"]).to(device)
            pv = pv.unsqueeze(0).repeat(FORECAST_STEPS, 1, 1 )  # (192, 1, 12)
            mask = torch.from_numpy(sample["pv_mask"]).to(device)
            mask = mask.unsqueeze(0).repeat(FORECAST_STEPS, 1)  # (192, 12)
            pv_ztime = sample["pv_ztime"].to(device)
            solar = sample["solar_features"].to(device)
            target_pv = torch.from_numpy(sample["target_pv"]).to(device)
            target_mask = torch.from_numpy(sample["target_mask"]).to(device)

            pv_pred = model(device_id, pv, mask, pv_ztime, solar)
            pv_pred = pv_pred.squeeze()

            loss = criterion(pv_pred*target_mask, target_pv*target_mask)
            batch_loss += loss.item()
            n += 1
            loss.backward()
        total_loss += batch_loss
        optimizer.step()
    return total_loss / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description="Train PV forecasting model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=1)
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
    model = pv_forecasting_model(out_dim=64, dev_dn_list=dev_dn_list).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    dataset = PVDataset(data_dir = "/home/cosmo/workspace/luoyang_data_626")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_single,
        num_workers=0,
    )

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else _PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(model, device, loader, criterion, optimizer)
        print(f"Epoch {epoch}/{args.epochs}  loss={avg_loss:.6f}")

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
    main()
