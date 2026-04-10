"""
Training script for pv_forecasting_model.
Loads config from conf.yaml, builds a dataset (synthetic or real), and trains with MSE loss.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config_utils import get_resolved_paths
from dataloader.luoyang import PVDataset, collate_batched
from models.models import pv_forecasting_model
from training.pv_loader_test import run_pv_loader_test
from training.training_conf import (
    get_training_hparams_from_conf,
    get_training_paths_from_conf,
    load_config,
)


def train_one_epoch(model, device, loader, criterion, optimizer, max_batches: int | None = None):
    model.train()
    total_loss = 0.0
    n = 0
    num_batches = len(loader)
    print("number of batches: ", num_batches)
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        B = batch["dev_idx"].size(0)
        device_id = batch["dev_idx"].to(device)            # [B]
        pv = batch["pv"].to(device)                        # [B, C, T_in]
        mask = batch["pv_mask"].to(device)                 # [B, C, T_in]
        pv_timefeats = batch["pv_timefeats"].to(device)   # [B, T_in, C_tf]
        forecast_timefeats = batch["forecast_timefeats"].to(device)  # [B, T_out, C_tf]
        target_pv = batch["target_pv"].to(device)
        target_mask = batch["target_mask"].to(device)

        optimizer.zero_grad()
        pv_pred = model(device_id, pv, mask, pv_timefeats, forecast_timefeats)
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
            pv_timefeats = batch["pv_timefeats"].to(device)
            forecast_timefeats = batch["forecast_timefeats"].to(device)
            target_pv = batch["target_pv"].to(device)
            target_mask = batch["target_mask"].to(device)
            pv_pred = model(device_id, pv, mask, pv_timefeats, forecast_timefeats)
            loss = criterion(pv_pred * target_mask, target_pv * target_mask)
            total_loss += loss.item()
            n += B
    return total_loss / max(n, 1)


def main():
    conf = load_config()
    h = get_training_hparams_from_conf(conf)
    path_defaults = get_training_paths_from_conf(conf)
    parser = argparse.ArgumentParser(description="Train PV forecasting model")
    parser.add_argument("--epochs", type=int, default=h["epochs"])
    parser.add_argument("--lr", type=float, default=h["lr"])
    parser.add_argument("--batch_size", type=int, default=h["batch_size"])
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=h["save_every"])
    parser.add_argument(
        "--loader-test",
        action="store_true",
        help="Only build train/test loaders and print a few batches; skip training.",
    )
    parser.add_argument(
        "--pv_train_dir",
        type=str,
        default=path_defaults["pv_train_dir"],
        help=f"Training CSV directory (default from conf: {path_defaults['pv_train_dir']!r}).",
    )
    parser.add_argument(
        "--pv_test_dir",
        type=str,
        default=path_defaults["pv_test_dir"],
        help=f"Eval/test CSV directory (default from conf: {path_defaults['pv_test_dir']!r}).",
    )
    parser.add_argument(
        "--skyimg_train_dir",
        type=str,
        default=path_defaults["skyimg_train_dir"],
        help=f"Training skyimg directory (default from conf: {path_defaults['skyimg_train_dir']!r}).",
    )
    parser.add_argument(
        "--skyimg_test_dir",
        type=str,
        default=path_defaults["skyimg_test_dir"],
        help=f"Eval/test skyimg directory (default from conf: {path_defaults['skyimg_test_dir']!r}).",
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
        "--csv_interval_min",
        type=int,
        default=h["csv_interval_min"],
        help="CSV row spacing in minutes (must divide pv_input/output intervals).",
    )
    parser.add_argument(
        "--pv_input_interval_min",
        type=int,
        default=h["pv_input_interval_min"],
        help="Minutes between consecutive PV input (X) samples.",
    )
    parser.add_argument(
        "--pv_output_interval_min",
        type=int,
        default=h["pv_output_interval_min"],
        help="Minutes between consecutive PV target (Y) samples.",
    )
    parser.add_argument("--pv_input_len", type=int, default=h["pv_input_len"], help="Input sequence length (X).")
    parser.add_argument("--pv_output_len", type=int, default=h["pv_output_len"], help="Target sequence length (Y).")
    parser.add_argument(
        "--test_anchor_stride_min",
        type=int,
        default=h["test_anchor_stride_min"],
        help="For split=test: minutes between consecutive eval anchors (multiple of CSV row interval).",
    )
    parser.add_argument(
        "--skyimg_window_size",
        type=int,
        default=h["skyimg_window_size"],
        help="Number of sky images per history and per forecast sequence.",
    )
    parser.add_argument(
        "--skyimg_time_resolution_min",
        type=int,
        default=h["skyimg_time_resolution_min"],
        help="Minutes between consecutive sky frames (independent of PV input spacing).",
    )
    parser.add_argument(
        "--skyimg_spatial_size",
        type=int,
        default=h["skyimg_spatial_size"],
        help="Sky JPEG resize side length (square, pixels).",
    )
    parser.add_argument(
        "--satimg_train_dir",
        type=str,
        default=path_defaults["satimg_train_dir"],
        help=f"Himawari NPY train dir (default from conf: {path_defaults['satimg_train_dir']!r}).",
    )
    parser.add_argument(
        "--satimg_test_dir",
        type=str,
        default=path_defaults["satimg_test_dir"],
        help=f"Himawari NPY test dir (default from conf: {path_defaults['satimg_test_dir']!r}).",
    )
    parser.add_argument(
        "--satimg_window_size",
        type=int,
        default=h["satimg_window_size"],
        help="Number of Himawari NPY frames per history and per forecast sequence.",
    )
    parser.add_argument(
        "--satimg_time_resolution_min",
        type=int,
        default=h["satimg_time_resolution_min"],
        help="Minutes between consecutive satimg frames in UTC.",
    )
    parser.add_argument(
        "--satimg_npy_shape_hwc",
        type=int,
        nargs=3,
        default=list(h["satimg_npy_shape_hwc"]),
        metavar=("H", "W", "C"),
        help="Expected Himawari NPY array shape H W C (default from conf).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=h["num_workers"],
        help="DataLoader worker processes.",
    )
    parser.add_argument(
        "--train_max_batches_per_epoch",
        type=int,
        default=h["train_max_batches_per_epoch"],
        help="Stop each training epoch after this many batches (default from conf; null = no cap).",
    )
    parser.add_argument(
        "--loader_test_batch_size",
        type=int,
        default=h["loader_test_batch_size"],
        help="With --loader-test: batch size (default from conf).",
    )
    parser.add_argument(
        "--loader_test_num_workers",
        type=int,
        default=h["loader_test_num_workers"],
        help="With --loader-test: DataLoader num_workers (default from conf).",
    )
    args = parser.parse_args()
    satimg_hwc = tuple(args.satimg_npy_shape_hwc)

    if args.loader_test:
        run_pv_loader_test(
            pv_train_dir=args.pv_train_dir,
            pv_test_dir=args.pv_test_dir,
            skyimg_train_dir=args.skyimg_train_dir,
            skyimg_test_dir=args.skyimg_test_dir,
            satimg_train_dir=args.satimg_train_dir,
            satimg_test_dir=args.satimg_test_dir,
            csv_interval_min=args.csv_interval_min,
            pv_input_interval_min=args.pv_input_interval_min,
            pv_input_len=args.pv_input_len,
            pv_output_interval_min=args.pv_output_interval_min,
            pv_output_len=args.pv_output_len,
            test_anchor_stride_min=args.test_anchor_stride_min,
            skyimg_window_size=args.skyimg_window_size,
            skyimg_time_resolution_min=args.skyimg_time_resolution_min,
            skyimg_spatial_size=args.skyimg_spatial_size,
            satimg_window_size=args.satimg_window_size,
            satimg_time_resolution_min=args.satimg_time_resolution_min,
            satimg_npy_shape_hwc=satimg_hwc,
            batch_size=args.loader_test_batch_size,
            epochs=args.loader_test_epochs,
            max_batches=args.loader_test_max_batches,
            num_workers=args.loader_test_num_workers,
        )
        return

    resolved_paths = get_resolved_paths(conf, _PROJECT_ROOT)
    pv_device_path = resolved_paths["pv_device_path"]
    if pv_device_path is None or not pv_device_path.is_file():
        raise FileNotFoundError(f"pv_device_path not found: {pv_device_path}")
    pv_device_df = pd.read_excel(pv_device_path)
    dev_dn_list = pv_device_df["devDn"].dropna().unique().tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = pv_forecasting_model(out_dim=64, dev_dn_list=dev_dn_list).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    train_dataset = PVDataset(
        pv_dir=args.pv_train_dir,
        skyimg_dir=args.skyimg_train_dir,
        satimg_dir=args.satimg_train_dir,
        split="train",
        csv_interval_min=args.csv_interval_min,
        pv_input_interval_min=args.pv_input_interval_min,
        pv_input_len=args.pv_input_len,
        pv_output_interval_min=args.pv_output_interval_min,
        pv_output_len=args.pv_output_len,
        test_anchor_stride_min=args.test_anchor_stride_min,
        skyimg_window_size=args.skyimg_window_size,
        skyimg_time_resolution_min=args.skyimg_time_resolution_min,
        skyimg_spatial_size=args.skyimg_spatial_size,
        satimg_window_size=args.satimg_window_size,
        satimg_time_resolution_min=args.satimg_time_resolution_min,
        satimg_npy_shape_hwc=satimg_hwc,
    )
    test_dataset = PVDataset(
        pv_dir=args.pv_test_dir,
        skyimg_dir=args.skyimg_test_dir,
        satimg_dir=args.satimg_test_dir,
        split="test",
        csv_interval_min=args.csv_interval_min,
        pv_input_interval_min=args.pv_input_interval_min,
        pv_input_len=args.pv_input_len,
        pv_output_interval_min=args.pv_output_interval_min,
        pv_output_len=args.pv_output_len,
        test_anchor_stride_min=args.test_anchor_stride_min,
        skyimg_window_size=args.skyimg_window_size,
        skyimg_time_resolution_min=args.skyimg_time_resolution_min,
        skyimg_spatial_size=args.skyimg_spatial_size,
        satimg_window_size=args.satimg_window_size,
        satimg_time_resolution_min=args.satimg_time_resolution_min,
        satimg_npy_shape_hwc=satimg_hwc,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batched,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batched,
        num_workers=args.num_workers,
    )

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else _PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    initial_test_loss = evaluate(model, device, test_loader, criterion)
    print(f"Initial test loss: {initial_test_loss:.6f}")

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(
            model,
            device,
            train_loader,
            criterion,
            optimizer,
            max_batches=args.train_max_batches_per_epoch,
        )
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
