"""
YLJ rolling daily finetune + operational forecast.

Each predict date ``D`` (China calendar):
  1. Fine-tune a fixed base checkpoint on Parquet rows with
     ``timestamp_win`` in ``[t_end - lookback_days + 15min, t_end]`` (China local),
     where ``t_end = D + finetune_end_time`` (default ``D 00:00``).
  2. Mask finetune loss for future targets at or after ``D 01:00 UTC`` (leakage guard).
  3. Predict at anchor ``D 09:00`` China (= ``D 01:00`` UTC); export 192-step sequence.

Opt-in entrypoint; normal ``train_ylj.py`` behavior is unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.training_conf import bootstrap_config_from_argv

bootstrap_config_from_argv()

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config_utils import get_resolved_paths
from dataloader.ylj_zarr import (
    YljRawParquetMatrixConfig,
    YljRawParquetDataset,
    collate_ylj_batched,
    rolling_finetune_window_bounds,
    ylj_raw_parquet_matrix_config_from_conf,
)
from models.models import pv_forecasting_model_vit_nwp
import training.train as base_train
import training.train_ylj as train_ylj

_DEFAULT_ROLLING = {
    "lookback_days": 7,
    "finetune_end_time": "00:00",
    "predict_issue_time": "09:00",
    "finetune_epochs": 2,
    "finetune_lr": 1.0e-4,
    "predict_start": "2025-01-01",
    "predict_end": "2025-12-30",
    "base_ckpt": (
        "/data/YANG/YLJ_stats/checkpoints_ylj_48h_48h_kt_raw_parquet_nwp_sat_run9/"
        "pv_forecast_epoch_3.pt"
    ),
    "train_parquet": "only_2024_real.parquet",
    "test_parquet": "ds_v322_1219_2025_1-12.parquet",
    "solar_features_csv": "solar_features_ylj_2024_2025_15min.csv",
    "pv_output_len": 192,
    "output_csv": "rolling_finetune_predictions.csv",
    "ckpt_dir": "rolling_finetune_ckpts",
}


@dataclass(frozen=True)
class RollingFinetuneConfig:
    lookback_days: int
    finetune_end_time: str
    predict_issue_time: str
    finetune_epochs: int
    finetune_lr: float
    predict_start: str
    predict_end: str
    train_parquet: str
    test_parquet: str
    solar_features_csv: str
    pv_output_len: int
    output_csv: str
    ckpt_dir: str
    base_ckpt: str


def rolling_finetune_config_from_conf(conf: dict) -> RollingFinetuneConfig:
    raw = conf.get("ylj_rolling_finetune")
    d = _DEFAULT_ROLLING if not isinstance(raw, dict) else {**_DEFAULT_ROLLING, **raw}

    def _ig(key: str) -> int:
        return int(d[key])

    def _fg(key: str) -> float:
        return float(d[key])

    def _sg(key: str) -> str:
        return str(d[key]).strip()

    return RollingFinetuneConfig(
        lookback_days=_ig("lookback_days"),
        finetune_end_time=_sg("finetune_end_time"),
        predict_issue_time=_sg("predict_issue_time"),
        finetune_epochs=_ig("finetune_epochs"),
        finetune_lr=_fg("finetune_lr"),
        predict_start=_sg("predict_start"),
        predict_end=_sg("predict_end"),
        train_parquet=_sg("train_parquet"),
        test_parquet=_sg("test_parquet"),
        solar_features_csv=_sg("solar_features_csv"),
        pv_output_len=_ig("pv_output_len"),
        output_csv=_sg("output_csv"),
        ckpt_dir=_sg("ckpt_dir"),
        base_ckpt=_sg("base_ckpt"),
    )


def _rolling_matrix_config(conf: dict, roll: RollingFinetuneConfig) -> YljRawParquetMatrixConfig:
    mx = ylj_raw_parquet_matrix_config_from_conf(conf)
    return YljRawParquetMatrixConfig(
        hist_len=mx.hist_len,
        fut_len=mx.fut_len,
        native_interval_min=mx.native_interval_min,
        naive_tz=mx.naive_tz,
        train_parquet=roll.train_parquet,
        test_parquet=roll.test_parquet,
        solar_features_csv=roll.solar_features_csv,
    )


def _resolve_sat_zarr_dir(conf: dict, raw_root: Path) -> str | None:
    sat_rel = str(conf.get("paths", {}).get("sat_path", "yalongjiang_zarr")).strip()
    if not sat_rel:
        return None
    path_defaults = base_train.get_training_paths_from_conf(conf)
    for cand in (raw_root / sat_rel, Path(path_defaults["satimg_dir"])):
        if cand.is_dir():
            return str(cand.resolve())
    return None


def _build_rolling_dataset(
    *,
    raw_root: Path,
    conf: dict,
    roll: RollingFinetuneConfig,
    mx: YljRawParquetMatrixConfig,
    lat: float,
    lon: float,
    dev_i: int,
    pv_value_scale: float,
    pv_input_len: int,
    pv_input_interval_min: int,
    pv_output_interval_min: int,
    use_nwp: bool,
    use_sat: bool,
    sat_zarr_dir: str | None,
    timestamp_start: pd.Timestamp | None,
    timestamp_end: pd.Timestamp | None,
    leakage_cutoff_utc: pd.Timestamp | None,
    include_export_metadata: bool,
    export_collect_time_utc: str | None,
) -> YljRawParquetDataset:
    return YljRawParquetDataset(
        str(raw_root),
        split="train",
        pv_input_interval_min=int(pv_input_interval_min),
        pv_input_len=int(pv_input_len),
        pv_output_interval_min=int(pv_output_interval_min),
        pv_output_len=int(roll.pv_output_len),
        latitude=float(lat),
        longitude=float(lon),
        dev_dn_index=int(dev_i),
        matrix=mx,
        use_nwp=bool(use_nwp),
        use_sat_zarr=bool(use_sat),
        sat_zarr_dir=sat_zarr_dir,
        pv_value_scale=float(pv_value_scale),
        include_export_metadata=bool(include_export_metadata),
        parquet_paths=[roll.train_parquet, roll.test_parquet],
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end,
        leakage_cutoff_utc=leakage_cutoff_utc,
        export_collect_time_utc=export_collect_time_utc,
    )


def _load_model_from_checkpoint(
    ckpt_path: Path,
    dev_dn_list: list,
    device: torch.device,
) -> pv_forecasting_model_vit_nwp:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = pv_forecasting_model_vit_nwp(
        dev_dn_list=dev_dn_list,
        n_parquet_hist_channels=int(ckpt.get("n_parquet_hist_channels", 0)),
        use_multi_kernel_tcn=bool(ckpt.get("tcn_multi_kernel", False)),
        cross_attn_layers=int(ckpt.get("cross_attn_layers", 1)),
        use_nwp_residual=bool(ckpt.get("nwp_residual", False)),
        use_hist_compression=bool(ckpt.get("hist_compression", False)),
        hist_compression_cond_forecast=bool(
            ckpt.get("hist_compression_cond_forecast", False)
        ),
        use_split_pv_sat_attn=bool(ckpt.get("split_pv_sat_attn", False)),
        use_last_k_head=bool(ckpt.get("last_k_head", False)),
        last_k_head_k=int(ckpt.get("last_k_head_k", 8)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def _finetune_one_day(
    *,
    model: pv_forecasting_model_vit_nwp,
    device: torch.device,
    loader: DataLoader,
    lr: float,
    epochs: int,
    predict_date: pd.Timestamp,
) -> None:
    if len(loader.dataset) == 0:
        raise ValueError(f"{predict_date.date()}: finetune dataset is empty")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=0.01)
    criterion = nn.HuberLoss(delta=1.0)
    date_s = predict_date.strftime("%Y-%m-%d")
    print(f"[rolling] finetune {date_s}: rows={len(loader.dataset)} epochs={epochs} lr={lr:g}")
    for ep in range(1, int(epochs) + 1):
        avg = train_ylj._ylj_train_one_epoch(
            model,
            device,
            loader,
            criterion,
            optimizer,
            epoch=ep,
            log_every=50,
        )
        print(f"[rolling] finetune {date_s} epoch {ep}/{epochs} mean_loss={avg:.6f}")


def _predict_one_row(
    *,
    model: pv_forecasting_model_vit_nwp,
    device: torch.device,
    ds: YljRawParquetDataset,
    row: int,
) -> dict:
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_ylj_batched,
        num_workers=0,
    )
    batch = next(iter(loader))
    with torch.no_grad():
        kt_pred_t, pv_pred_t = train_ylj._ylj_forward_kt(model, batch, device)
    pv_scale = float(ds._pv_value_scale)
    pairs = [
        [
            float(batch["target_pv"][0, k].item() * pv_scale),
            float(pv_pred_t[0, k].item()),
            float(kt_pred_t[0, k].item()),
        ]
        for k in range(int(pv_pred_t.shape[1]))
    ]
    return {
        "collectTime": str(batch["csv_collect_time_utc"][0]),
        "gt_pred_pairs": json.dumps(pairs, ensure_ascii=False),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="YLJ rolling daily finetune + forecast")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--ylj_raw_parquet", action="store_true", default=True)
    parser.add_argument("--ylj_parquet_nwp", action="store_true")
    parser.add_argument("--ylj_sat_zarr", action="store_true")
    parser.add_argument(
        "--rolling_base_ckpt",
        type=str,
        default=None,
        help=(
            "Pretrained base checkpoint loaded fresh for each day before finetune "
            f"(default: ylj_rolling_finetune.base_ckpt or {_DEFAULT_ROLLING['base_ckpt']})."
        ),
    )
    parser.add_argument("--rolling_ckpt_dir", type=str, default=None)
    parser.add_argument("--rolling_output_csv", type=str, default=None)
    parser.add_argument("--rolling_lookback_days", type=int, default=None)
    parser.add_argument(
        "--rolling_finetune_end_time",
        type=str,
        default=None,
        help="Finetune window end on day D, naive China HH:MM (default: ylj_rolling_finetune.finetune_end_time or 00:00).",
    )
    parser.add_argument(
        "--rolling_predict_issue_time",
        type=str,
        default=None,
        help="Operational forecast issue time on day D, naive China HH:MM (default: 09:00).",
    )
    parser.add_argument("--rolling_finetune_epochs", type=int, default=None)
    parser.add_argument("--rolling_finetune_lr", type=float, default=None)
    parser.add_argument("--rolling_predict_start", type=str, default=None)
    parser.add_argument("--rolling_predict_end", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()

    base_train._apply_config_override(args.config)
    conf = base_train.load_config()
    hp = base_train.get_training_hparams_from_conf(conf)
    roll = rolling_finetune_config_from_conf(conf)
    if args.rolling_lookback_days is not None:
        roll = RollingFinetuneConfig(**{**roll.__dict__, "lookback_days": int(args.rolling_lookback_days)})
    if args.rolling_finetune_end_time is not None:
        roll = RollingFinetuneConfig(
            **{**roll.__dict__, "finetune_end_time": str(args.rolling_finetune_end_time).strip()}
        )
    if args.rolling_predict_issue_time is not None:
        roll = RollingFinetuneConfig(
            **{**roll.__dict__, "predict_issue_time": str(args.rolling_predict_issue_time).strip()}
        )
    if args.rolling_finetune_epochs is not None:
        roll = RollingFinetuneConfig(**{**roll.__dict__, "finetune_epochs": int(args.rolling_finetune_epochs)})
    if args.rolling_finetune_lr is not None:
        roll = RollingFinetuneConfig(**{**roll.__dict__, "finetune_lr": float(args.rolling_finetune_lr)})
    if args.rolling_predict_start is not None:
        roll = RollingFinetuneConfig(**{**roll.__dict__, "predict_start": str(args.rolling_predict_start)})
    if args.rolling_predict_end is not None:
        roll = RollingFinetuneConfig(**{**roll.__dict__, "predict_end": str(args.rolling_predict_end)})

    paths = get_resolved_paths(conf, _PROJECT_ROOT)
    raw_root = paths.get("ylj_raw_parquet_dir")
    if raw_root is None or not raw_root.is_dir():
        raise FileNotFoundError(f"paths.ylj_raw_parquet_dir not found: {raw_root!r}")

    base_ckpt_path = args.rolling_base_ckpt or roll.base_ckpt
    base_ckpt = Path(base_ckpt_path).expanduser().resolve()
    if not base_ckpt.is_file():
        raise FileNotFoundError(f"rolling_base_ckpt not found: {base_ckpt}")

    ckpt_dir = Path(args.rolling_ckpt_dir or roll.ckpt_dir).expanduser()
    if not ckpt_dir.is_absolute():
        ckpt_dir = _PROJECT_ROOT / ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    out_csv = Path(args.rolling_output_csv or roll.output_csv).expanduser()
    if not out_csv.is_absolute():
        out_csv = _PROJECT_ROOT / out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    site = conf.get("site", {})
    lat = float(site["latitude"])
    lon = float(site["longitude"])
    mx = _rolling_matrix_config(conf, roll)

    pv_dev = paths.get("pv_device_path")
    if pv_dev is None or not pv_dev.is_file():
        raise FileNotFoundError(f"pv_device_path not found: {pv_dev}")
    dev_dn_list = pd.read_excel(pv_dev)["devDn"].dropna().unique().tolist()
    try:
        dev_i = int(dev_dn_list.index("NE=ylj"))
    except ValueError:
        dev_i = 0

    use_nwp = bool(args.ylj_parquet_nwp)
    use_sat = bool(args.ylj_sat_zarr)
    sat_zarr_dir = _resolve_sat_zarr_dir(conf, raw_root) if use_sat else None
    if use_sat and sat_zarr_dir is None:
        raise FileNotFoundError("--ylj_sat_zarr enabled but satellite Zarr was not found")

    batch_size = int(args.batch_size if args.batch_size is not None else hp["batch_size"])
    num_workers = int(args.num_workers if args.num_workers is not None else hp["num_workers"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predict_dates = pd.date_range(
        pd.Timestamp(roll.predict_start),
        pd.Timestamp(roll.predict_end),
        freq="D",
    )
    rows_out: list[dict] = []
    print(
        f"[rolling] dates={roll.predict_start}..{roll.predict_end} "
        f"lookback_days={roll.lookback_days} finetune_end_time={roll.finetune_end_time} "
        f"issue_time={roll.predict_issue_time} base_ckpt={base_ckpt.name}"
    )

    for predict_date in predict_dates:
        bounds = rolling_finetune_window_bounds(
            predict_date,
            lookback_days=roll.lookback_days,
            finetune_end_time=roll.finetune_end_time,
            predict_issue_time=roll.predict_issue_time,
            native_interval_min=mx.native_interval_min,
            naive_tz=mx.naive_tz,
        )
        t_start = bounds["t_start_local"]
        t_end = bounds["t_end_local"]
        anchor_local = bounds["predict_anchor_local"]
        leakage_cutoff = bounds["leakage_cutoff_utc"]
        export_collect = bounds["export_collect_time_utc"]
        date_s = predict_date.strftime("%Y-%m-%d")

        finetune_ds = _build_rolling_dataset(
            raw_root=raw_root,
            conf=conf,
            roll=roll,
            mx=mx,
            lat=lat,
            lon=lon,
            dev_i=dev_i,
            pv_value_scale=float(hp["pv_value_scale"]),
            pv_input_len=int(hp["pv_input_len"]),
            pv_input_interval_min=int(hp["pv_input_interval_min"]),
            pv_output_interval_min=int(hp["pv_output_interval_min"]),
            use_nwp=use_nwp,
            use_sat=use_sat,
            sat_zarr_dir=sat_zarr_dir,
            timestamp_start=t_start,
            timestamp_end=t_end,
            leakage_cutoff_utc=leakage_cutoff,
            include_export_metadata=False,
            export_collect_time_utc=None,
        )
        if len(finetune_ds) == 0:
            print(f"[rolling] skip {date_s}: no finetune rows in [{t_start}, {t_end}]")
            continue

        model = _load_model_from_checkpoint(base_ckpt, dev_dn_list, device)
        finetune_loader = DataLoader(
            finetune_ds,
            batch_size=min(batch_size, len(finetune_ds)),
            shuffle=True,
            collate_fn=collate_ylj_batched,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )
        _finetune_one_day(
            model=model,
            device=device,
            loader=finetune_loader,
            lr=roll.finetune_lr,
            epochs=roll.finetune_epochs,
            predict_date=pd.Timestamp(predict_date),
        )

        day_ckpt = ckpt_dir / f"finetune_{date_s}.pt"
        torch.save(
            {
                "epoch": roll.finetune_epochs,
                "predict_date": date_s,
                "model_state_dict": model.state_dict(),
                "finetune_rows": len(finetune_ds),
                "lookback_days": roll.lookback_days,
            },
            day_ckpt,
        )
        print(f"[rolling] saved {day_ckpt}")

        infer_ds = _build_rolling_dataset(
            raw_root=raw_root,
            conf=conf,
            roll=roll,
            mx=mx,
            lat=lat,
            lon=lon,
            dev_i=dev_i,
            pv_value_scale=float(hp["pv_value_scale"]),
            pv_input_len=int(hp["pv_input_len"]),
            pv_input_interval_min=int(hp["pv_input_interval_min"]),
            pv_output_interval_min=int(hp["pv_output_interval_min"]),
            use_nwp=use_nwp,
            use_sat=use_sat,
            sat_zarr_dir=sat_zarr_dir,
            timestamp_start=anchor_local,
            timestamp_end=anchor_local,
            leakage_cutoff_utc=None,
            include_export_metadata=True,
            export_collect_time_utc=str(export_collect),
        )
        row = infer_ds.find_row_index(anchor_local)
        if row is None:
            print(f"[rolling] skip predict {date_s}: anchor {anchor_local} not found in Parquet")
            continue

        model.eval()
        row_out = _predict_one_row(model=model, device=device, ds=infer_ds, row=row)
        rows_out.append(row_out)
        print(f"[rolling] predicted {date_s} collectTime={row_out['collectTime']}")

        pd.DataFrame(rows_out).to_csv(out_csv, index=False)

    print(f"[rolling] wrote {len(rows_out)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
