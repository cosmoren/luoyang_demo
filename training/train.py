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
from training.data_loader import (
    INVERTER_STATE_COL,
    VALID_STATE,
    build_training_set,
    load_csv,
)
from datetime import datetime, timedelta, timezone

CONF_PATH = _PROJECT_ROOT / "config" / "conf.yaml"
DEFAULT_TRAIN_DATA_DIR = "/data/luoyang_data_626_train"
DEFAULT_TEST_DATA_DIR = "/data/luoyang_data_626_test"
DEFAULT_TEST_ANCHOR_STRIDE_MIN = 120    # e.g. 1200 min / 5 min CSV row = stride 240 anchors
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
    Train: ``__len__`` = number of CSVs; ``__getitem__`` picks a **random** anchor where Y has at
    least one ``inverter_state == VALID_STATE``.

    Test (``split="test"``): ``__len__`` = ``len(sample_files) * num_test_windows``; anchors are
    fixed on a grid every ``test_anchor_stride_min`` minutes along the CSV row axis (no Y validity
    filter). All files with the same row count share the same ``num_test_windows``.
    """

    def __init__(
        self,
        data_dir: str,
        *,
        split: str = "train",
        csv_interval_min: int = 5,
        pv_input_interval_min: int = 5,
        pv_input_len: int = 576,
        pv_output_interval_min: int = 15,
        pv_output_len: int = 192,
        test_anchor_stride_min: int = DEFAULT_TEST_ANCHOR_STRIDE_MIN,
    ):
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        self.split = split
        if csv_interval_min <= 0 or pv_input_interval_min % csv_interval_min:
            raise ValueError("pv_input_interval_min must be a positive multiple of csv_interval_min")
        if pv_output_interval_min % csv_interval_min:
            raise ValueError("pv_output_interval_min must be a positive multiple of csv_interval_min")
        self._sx = pv_input_interval_min // csv_interval_min
        self._sy = pv_output_interval_min // csv_interval_min
        self.pv_input_len = pv_input_len
        self.pv_output_len = pv_output_len
        self.pv_output_interval_min = pv_output_interval_min
        if test_anchor_stride_min <= 0 or test_anchor_stride_min % csv_interval_min:
            raise ValueError(
                "test_anchor_stride_min must be a positive multiple of csv_interval_min "
                f"(got {test_anchor_stride_min}, csv_interval_min={csv_interval_min})"
            )
        self._test_anchor_stride_rows = test_anchor_stride_min // csv_interval_min

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

        self.sample_files = build_training_set(0, 625, data_dir=data_dir)
        if not self.sample_files:
            raise FileNotFoundError(f"No CSV files in {data_dir!r}")

        ref_df = load_csv(self.sample_files[0])
        n = len(ref_df)
        lx, ly = self.pv_input_len, self.pv_output_len
        sx, sy = self._sx, self._sy
        amin = (lx - 1) * sx
        y_last_off = sy * ly
        amax = n - 1 - y_last_off
        if n == 0 or amin > amax:
            raise RuntimeError(
                f"reference CSV {self.sample_files[0].name}: no anchor fits bounds "
                f"(n={n}, need {amin}<=anchor<={amax})"
            )
        anchors = np.arange(amin, amax + 1, dtype=np.intp)
        y_off = sy + np.arange(ly, dtype=np.intp) * sy
        x_tail = (-(lx - 1) * sx + np.arange(lx, dtype=np.intp) * sx).reshape(1, -1)
        self._csv_row_count = n
        self._anchors = anchors
        self._y_off = y_off
        self._y_idx_per_anchor = anchors[:, None] + y_off[None, :]
        self._x_idx_per_anchor = anchors[:, None] + x_tail

        n_anchors = len(anchors)
        self._test_r_indices = np.arange(
            0, n_anchors, self._test_anchor_stride_rows, dtype=np.intp
        )
        self._num_test_windows = int(self._test_r_indices.size)
        if self.split == "test" and self._num_test_windows == 0:
            raise RuntimeError("split=test: stride leaves no anchor indices (increase data or reduce stride)")

        valid_files: list[Path] = []
        for i, p in enumerate(self.sample_files):
            print(f"Processing file {i+1} of {len(self.sample_files)}: {p.name}")
            df = ref_df if i == 0 else load_csv(p)
            if len(df) != n:
                continue
            if INVERTER_STATE_COL not in df.columns:
                continue
            if self.split == "train":
                inv = (
                    pd.to_numeric(df[INVERTER_STATE_COL], errors="coerce").fillna(0).astype(int).values
                    == VALID_STATE
                )
                if inv[self._y_idx_per_anchor].any(axis=1).any():
                    valid_files.append(p)
            else:
                valid_files.append(p)

        self.sample_files = valid_files
        _tw = (
            f", test_windows_per_file={self._num_test_windows}" if self.split == "test" else ""
        )
        print(f"Valid files: {len(self.sample_files)} ({self.split}{_tw})")
        if not self.sample_files:
            raise RuntimeError(
                f"No CSV files left after prefilter for split={self.split!r} in data_dir={data_dir!r}"
                + (
                    f" (train requires at least one Y row with {INVERTER_STATE_COL}=={VALID_STATE})"
                    if self.split == "train"
                    else ""
                )
            )

    def __len__(self):
        if self.split == "train":
            return len(self.sample_files)
        return len(self.sample_files) * self._num_test_windows

    def _build_sample(self, df: pd.DataFrame, dev_idx: torch.Tensor, r: int) -> dict:
        x_idx = self._x_idx_per_anchor[r]
        y_idx_1d = self._y_idx_per_anchor[r]
        sub_x = df.iloc[x_idx]
        sub_y = df.iloc[y_idx_1d]

        ts_beijing = ZoneInfo("Asia/Shanghai")
        timestamps = []
        for t in sub_x["collectTime"]:
            ts = pd.Timestamp(t)
            if ts.tz is None:
                dt = ts.to_pydatetime().replace(tzinfo=ts_beijing)
            else:
                dt = ts.astimezone(ts_beijing).to_pydatetime()
            timestamps.append(dt)

        inv_x = pd.to_numeric(sub_x[INVERTER_STATE_COL], errors="coerce").fillna(0).astype(np.int32).values
        pow_x = pd.to_numeric(sub_x["active_power"], errors="coerce").fillna(0).values.astype(np.float32) / 50.0
        pv_mask = torch.from_numpy((inv_x == VALID_STATE).astype(np.float32)).unsqueeze(0)
        pv = torch.from_numpy(pow_x.astype(np.float32)).unsqueeze(0)

        pv_solar_features = compute_solar_features(timestamps, self.latitude, self.longitude)
        pv_solar_features_array = []
        for pv_solar_feature in pv_solar_features:
            azimuth_rad = np.deg2rad(pv_solar_feature["azimuth"])
            sin_azimuth = np.sin(azimuth_rad)
            cos_azimuth = np.cos(azimuth_rad)
            zenith_rad = np.deg2rad(pv_solar_feature["zenith"])
            cos_zenith = np.cos(zenith_rad)
            sin_zenith = np.sin(zenith_rad)
            cos_hod = np.cos(2 * np.pi * pv_solar_feature["hour_of_day"] / 24)
            sin_hod = np.sin(2 * np.pi * pv_solar_feature["hour_of_day"] / 24)
            pv_solar_features_array.append(
                np.array([sin_azimuth, cos_azimuth, sin_zenith, cos_zenith, sin_hod, cos_hod])
            )
        history_solar_features = torch.from_numpy(np.asarray(pv_solar_features_array, dtype=np.float32))

        time0 = timestamps[-1]
        forecast_timestamps_utc = [
            time0 + timedelta(minutes=self.pv_output_interval_min * (i + 1))
            for i in range(self.pv_output_len)
        ]
        solar_features = compute_solar_features(forecast_timestamps_utc, self.latitude, self.longitude)
        solar_features_array = []
        for solar_feature in solar_features:
            azimuth_rad = np.deg2rad(solar_feature["azimuth"])
            sin_azimuth = np.sin(azimuth_rad)
            cos_azimuth = np.cos(azimuth_rad)
            zenith_rad = np.deg2rad(solar_feature["zenith"])
            cos_zenith = np.cos(zenith_rad)
            sin_zenith = np.sin(zenith_rad)
            cos_dofy = np.cos(2 * np.pi * solar_feature["day_of_year"] / 366)
            sin_dofy = np.sin(2 * np.pi * solar_feature["day_of_year"] / 366)
            cos_hod = np.cos(2 * np.pi * solar_feature["hour_of_day"] / 24)
            sin_hod = np.sin(2 * np.pi * solar_feature["hour_of_day"] / 24)
            solar_features_array.append(
                np.array(
                    [sin_azimuth, cos_azimuth, sin_zenith, cos_zenith, sin_dofy, cos_dofy, sin_hod, cos_hod]
                )
            )
        forecast_solar_features = torch.from_numpy(np.asarray(solar_features_array, dtype=np.float32))

        inv_y = pd.to_numeric(sub_y[INVERTER_STATE_COL], errors="coerce").fillna(0).astype(np.int32).values
        pow_y = pd.to_numeric(sub_y["active_power"], errors="coerce").fillna(0).values.astype(np.float32)
        target_pv = torch.from_numpy((pow_y / 50.0).astype(np.float32))
        target_mask = torch.from_numpy((inv_y == VALID_STATE).astype(np.float32))

        return {
            "dev_idx": dev_idx,
            "pv": pv,
            "pv_mask": pv_mask,
            "history_solar_features": history_solar_features,
            "forecast_solar_features": forecast_solar_features,
            "target_pv": target_pv,
            "target_mask": target_mask,
        }

    def __getitem__(self, idx):
        if self.split == "train":
            sample_path = self.sample_files[idx]
            r_fixed: int | None = None
        else:
            nw = self._num_test_windows
            sample_path = self.sample_files[idx // nw]
            r_fixed = int(self._test_r_indices[idx % nw])

        devDn = sample_path.stem.replace("_", "=")
        try:
            dev_idx = torch.tensor(self.devDn_list.index(devDn), dtype=torch.long)
        except ValueError as e:
            raise ValueError(f"devDn {devDn!r} from {sample_path.name} not in pv_device list") from e

        df = load_csv(sample_path)
        if INVERTER_STATE_COL not in df.columns:
            raise ValueError(f"missing {INVERTER_STATE_COL} in {sample_path}")

        n = len(df)
        if n != self._csv_row_count:
            raise ValueError(
                f"{sample_path.name}: expected {self._csv_row_count} rows (same as reference CSV), got {n}"
            )

        if self.split == "train":
            inv = (
                pd.to_numeric(df[INVERTER_STATE_COL], errors="coerce").fillna(0).astype(int).values
                == VALID_STATE
            )
            valid_mask = inv[self._y_idx_per_anchor].any(axis=1)
            valid_rows = np.nonzero(valid_mask)[0]
            if valid_rows.size == 0:
                raise RuntimeError(
                    f"{sample_path.name}: no anchor with at least one Y row where "
                    f"{INVERTER_STATE_COL}=={VALID_STATE}"
                )
            r = int(np.random.choice(valid_rows))
        else:
            assert r_fixed is not None
            r = r_fixed

        return self._build_sample(df, dev_idx, r)


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
    }


def loader_test(
    *,
    train_data_dir: str = DEFAULT_TRAIN_DATA_DIR,
    test_data_dir: str = DEFAULT_TEST_DATA_DIR,
    test_anchor_stride_min: int = DEFAULT_TEST_ANCHOR_STRIDE_MIN,
    batch_size: int = 4,
    epochs: int = 1,
    max_batches: int | None = None,
    num_workers: int = 0,
) -> dict:
    """
    :class:`PVDataset` / ``DataLoader`` for ``train_data_dir`` and, if it differs from
    ``test_data_dir`` after resolving paths, a second loader for the test folder.
    """
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be >= 1 when set")

    train_dir = str(Path(train_data_dir).resolve())
    test_dir = str(Path(test_data_dir).resolve())

    train_dataset = PVDataset(data_dir=train_dir, split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batched,
        num_workers=num_workers,
    )

    test_dataset: PVDataset | None = None
    test_loader: DataLoader | None = None
    if test_dir != train_dir:
        test_dataset = PVDataset(
            data_dir=test_dir,
            split="test",
            test_anchor_stride_min=test_anchor_stride_min,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_batched,
            num_workers=num_workers,
        )

    print(f"loader_test: train_data_dir={train_dir!r}")
    print(f"  train_dataset: {len(train_dataset)} files  batches/epoch={len(train_loader)}  epochs={epochs}")
    if max_batches is not None:
        print(f"  max_batches/epoch={max_batches}")
    if test_loader is not None:
        assert test_dataset is not None
        print(f"loader_test: test_data_dir={test_dir!r}")
        print(f"  test_dataset:  {len(test_dataset)} files  batches/epoch={len(test_loader)}")

    def _run_split(name: str, loader: DataLoader) -> None:
        for epoch in range(epochs):
            for batch_idx, batch in enumerate(loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                print(
                    f"  [{name}] epoch {epoch + 1}/{epochs} batch {batch_idx}: ",
                    end="",
                )
                parts = [f"{k} {tuple(batch[k].shape)} {batch[k].dtype}" for k in batch]
                print(", ".join(parts))

    _run_split("train", train_loader)
    if test_loader is not None:
        _run_split("test", test_loader)

    out: dict = {
        "train_dataset": train_dataset,
        "train_loader": train_loader,
    }
    if test_dataset is not None and test_loader is not None:
        out["test_dataset"] = test_dataset
        out["test_loader"] = test_loader
    return out


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
    parser.add_argument(
        "--loader-test",
        action="store_true",
        help="Only build train/test loaders and print a few batches; skip training.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=DEFAULT_TRAIN_DATA_DIR,
        help=f"Training CSV directory (default: {DEFAULT_TRAIN_DATA_DIR!r}).",
    )
    parser.add_argument(
        "--test_data_dir",
        type=str,
        default=DEFAULT_TEST_DATA_DIR,
        help=f"Eval/test CSV directory (default: {DEFAULT_TEST_DATA_DIR!r}).",
    )
    parser.add_argument(
        "--loader-test-epochs",
        type=int,
        default=1,
        help="With --loader-test: number of full DataLoader passes (epochs).",
    )
    parser.add_argument(
        "--loader-test-max-batches",
        type=int,
        default=None,
        help="With --loader-test: max batches per epoch (default: full epoch).",
    )
    parser.add_argument(
        "--test_anchor_stride_min",
        type=int,
        default=DEFAULT_TEST_ANCHOR_STRIDE_MIN,
        help="For split=test: minutes between consecutive eval anchors (multiple of CSV row interval, default 120).",
    )
    args = parser.parse_args()

    if args.loader_test:
        loader_test(
            train_data_dir=args.train_data_dir,
            test_data_dir=args.test_data_dir,
            test_anchor_stride_min=args.test_anchor_stride_min,
            batch_size=args.batch_size,
            epochs=args.loader_test_epochs,
            max_batches=args.loader_test_max_batches,
        )
        return

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

    train_dir = args.train_data_dir
    test_dir = args.test_data_dir
    train_dataset = PVDataset(data_dir=train_dir, split="train")
    test_dataset = PVDataset(
        data_dir=test_dir,
        split="test",
        test_anchor_stride_min=args.test_anchor_stride_min,
    )

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
