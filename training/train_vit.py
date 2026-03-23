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
from preprocessing.himawari_infer import parse_time_from_nc_name
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
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
        # Compute forecast time features
        forecast_solar_features = compute_solar_features(forecast_timestamps_utc, self.latitude, self.longitude)
        forecast_timefeats = solar_features_encoder(forecast_solar_features)
        forecast_dtimefeats = delta_time_encoder(forecast_timestamps_utc, time0)
        forecast_timefeats = torch.cat([forecast_timefeats, forecast_dtimefeats.unsqueeze(1)], dim=1)

        # load satellite images: NC times in UTC; window [t0-48h, t0] in UTC
        sat_image_path = "/mnt/nfs/Ai4Energy/Datasets/luoyang/luoyang_crop"
        print('time0:', time0)

        # target
        target_pv = torch.from_numpy( (sample['Y'][:,2]/50).astype(np.float32) )
        target_mask = torch.from_numpy( (sample['Y'][:,1]==512).astype(np.float32) )

        return {
            "dev_idx": dev_idx,
            "pv": pv,
            "pv_mask": pv_mask,
            "history_solar_features": pv_timefeats,
            "delta_time_hours": pv_dtimefeats,
            "forecast_solar_features": forecast_timefeats,
            "sat_filenames": sat_filenames,
            "target_pv": target_pv,
            "target_mask": target_mask,
        }


def collate_single(batch):
    """Collate so we get a list of samples; we run model once per sample (batch_size 1 or loop)."""
    return batch


def collate_batched(batch):
    """Stack list of samples into one dict of tensors with batch dim B in front."""
    return {
        "dev_idx": torch.stack([s["dev_idx"] for s in batch]),
        "pv": torch.stack([s["pv"] for s in batch]),
        "pv_mask": torch.stack([s["pv_mask"] for s in batch]),
        "history_solar_features": torch.stack([s["history_solar_features"] for s in batch]),
        "forecast_solar_features": torch.stack([s["forecast_solar_features"] for s in batch]),
        "target_pv": torch.stack([s["target_pv"] for s in batch]),
        "target_mask": torch.stack([s["target_mask"] for s in batch]),
        "delta_time_hours": torch.stack([s["delta_time_hours"] for s in batch]),
        "sat_filenames": [s["sat_filenames"] for s in batch],
    }


def train_one_epoch(model, device, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    n = 0
    num_batches = len(loader)
    print('number of batches: ', num_batches)
    for batch_idx, batch in enumerate(loader):
        if batch_idx > 3000:
            break
        B = batch["dev_idx"].size(0)
        device_id = batch["dev_idx"].to(device)            # [B]
        pv = batch["pv"].to(device)                        # [B, C, T_in]
        mask = batch["pv_mask"].to(device)                 # [B, C, T_in]
        history_solar_features = batch["history_solar_features"].to(device)
        forecast_solar_features = batch["forecast_solar_features"].to(device)
        target_pv = batch["target_pv"].to(device)
        target_mask = batch["target_mask"].to(device)

        optimizer.zero_grad()
        pv_pred = model(device_id, pv, mask, history_solar_features, forecast_solar_features)
        loss = criterion(pv_pred * target_mask, target_pv * target_mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += B
    print()
    return total_loss / max(n, 1)


def evaluate(model, device, loader, criterion):
    """Compute mean loss on a dataset (e.g. test set). No gradient."""
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            B = batch["dev_idx"].size(0)
            device_id = batch["dev_idx"].to(device)
            pv = batch["pv"].to(device)
            mask = batch["pv_mask"].to(device)
            history_solar_features = batch["history_solar_features"].to(device)
            forecast_solar_features = batch["forecast_solar_features"].to(device)
            target_pv = batch["target_pv"].to(device)
            target_mask = batch["target_mask"].to(device)
            pv_pred = model(device_id, pv, mask, history_solar_features, forecast_solar_features)
            loss = criterion(pv_pred * target_mask, target_pv * target_mask)
            total_loss += loss.item()
            n += B
    return total_loss / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description="Train PV forecasting model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
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

    data_dir = "/mnt/nfs/Ai4Energy/Datasets/luoyang_data_626/"
    train_dataset = PVDataset(data_dir=data_dir, split="train")
    test_dataset = PVDataset(data_dir=data_dir, split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batched,
        num_workers=24,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batched,
        num_workers=24,
    )

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else _PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    initial_test_loss = evaluate(model, device, test_loader, criterion)
    print(f"Initial test loss: {initial_test_loss:.6f}")

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(model, device, train_loader, criterion, optimizer)
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
    main()
