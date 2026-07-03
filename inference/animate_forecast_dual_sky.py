"""
Animate a rolling 15-min next-step nowcast with dual sky panels (Folsom, CA).

Layout (one frame):

    +------------------------+------------------------+
    |  Sky at T (issue time) |  Sky at T+15 (target)  |
    +------------------------+------------------------+
    |              GHI nowcast (full width)            |
    +--------------------------------------------------+

For each display time T on a 15-min local grid:

    anchor t0 = T - 15 min
    run the trained ViT once with pv_output_len = 1
    take output step 0  ==> the +15 min step  ==> prediction at T

Top-left shows the fisheye sky when the nowcast is issued (T).
Top-right shows the fisheye sky at the predicted time (T + 15 min).
Bottom panel plots ground truth (black) plus one-or-more model predictions
(red / blue / green) vs hour-of-day (local). X-axis ticks are placed at every
integer hour within the visible daylight window so times are easy to read.

Multi-model overlay: pass ``--checkpoint-extra`` and/or ``--checkpoint-third``
to overlay additional prediction lines (same colors as animate_forecast.py).

Usage::

    # Single model:
    micromamba run -n luoyang python inference/animate_forecast_dual_sky.py \
        --date 2014-12-01 \
        --out-dir "playground/2026-07-03_sky improve show/2026-07-03_imgs/2026-07-03_test_dual_sky" \
        --run-label 2026-07-03_test_dual_sky

    # Sky vs no-sky (same checkpoints as sample-5-test-split):
    micromamba run -n luoyang python inference/animate_forecast_dual_sky.py \
        --date 2014-12-01 \
        --out-dir "playground/2026-07-03_sky improve show/2026-07-03_imgs" \
        --run-label sample-5-test-split-dual \
        --checkpoint "playground/2026-07-03_sky improve show/2026-07-03_folsom250k_sky_rgb_s1/folsom_pv_forecast_vit_final_gpu1.pt" \
        --checkpoint-extra "playground/2026-07-03_sky improve show/2026-07-03_folsom250k_ghi_only_s1/folsom_pv_forecast_vit_final_gpu0.pt" \
        --label-main "GHI + full sky" \
        --label-extra "GHI only (zero sky)"

Outputs (under ``<out-dir>/<run_label>/``):

    <date>.gif                  the animation itself
    <date>_predictions.npz      per-frame (t, gt, pred[, pred_extra[, pred_third]]) arrays
    <date>_meta.json            checkpoint / run / config metadata
    <date>_frames/              per-frame PNGs (only with --keep-frames)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Reuse inference / IO helpers from the original animator (do not modify that file).
from inference.animate_forecast import (  # noqa: E402
    _DEFAULT_ARCHIVE_NAME,
    _DEFAULT_CHECKPOINT,
    _DEFAULT_DATASET_CONFIG,
    _DEFAULT_DATE,
    _DEFAULT_RUN_LABEL,
    _DEFAULT_RUN_NAME,
    _DEFAULT_SKY_ZARR,
    _EXTRA_COLOR,
    _FORCE_PV_OUTPUT_LEN,
    _FPS,
    _LOCAL_TZ,
    _MAIN_COLOR,
    _SKY_TIME_MATCH_TOLERANCE_S,
    _THIRD_COLOR,
    _assert_valid_sky_source,
    _build_folsom_dataset_for_inference,
    _find_display_grid_utc,
    _infer_sky_in_channels_from_state,
    _local_day_window,
    _read_sky_image_for_time,
    _resolve_anchor_row_for_t0,
    _run_model_on_anchors,
    _save_gif,
    _utc_naive_to_local_hour,
)

# Dual-sky layout: two sky panels on top, full-width plot below.
_FIG_W_INCHES = 10.0
_FIG_H_INCHES = 7.0
_FRAME_DPI = 110


def _integer_hour_ticks(xlim: tuple[float, float]) -> tuple[list[float], list[str]]:
    """Build x-axis ticks at every integer hour within ``xlim``.

    Returns tick positions (float hours) and labels (``"HH"`` strings). When the
    visible window spans many hours every hour is shown; matplotlib may rotate
    labels if they overlap on narrow windows.
    """
    lo = int(np.floor(xlim[0]))
    hi = int(np.ceil(xlim[1]))
    ticks = [float(h) for h in range(lo, hi + 1)]
    labels = [f"{h:02d}" for h in range(lo, hi + 1)]
    return ticks, labels


def _render_one_frame_dual_sky(
    *,
    t_utc: pd.Timestamp,
    hours_local: np.ndarray,
    gt_kw: np.ndarray,
    preds: list[tuple[np.ndarray, str, str]],
    cursor_idx: int,
    sky_img_t: np.ndarray | None,
    sky_img_t_plus_15: np.ndarray | None,
    date_local_str: str,
    ymin: float,
    ymax: float,
    xlim: tuple[float, float],
    title_plot: str,
    gt_label: str = "Ground truth",
) -> np.ndarray:
    """Render one dual-sky frame and return as ``[H, W, 3]`` uint8."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(_FIG_W_INCHES, _FIG_H_INCHES), dpi=_FRAME_DPI)
    fig.patch.set_facecolor("white")
    gs = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.0, 1.1],
        hspace=0.28,
        wspace=0.12,
    )
    ax_sky_t = fig.add_subplot(gs[0, 0])
    ax_sky_tp = fig.add_subplot(gs[0, 1])
    ax_plot = fig.add_subplot(gs[1, :])

    t_local = pd.Timestamp(t_utc).tz_localize("UTC").tz_convert(_LOCAL_TZ)
    t_plus_15 = pd.Timestamp(t_utc) + pd.Timedelta(minutes=15)
    t_plus_local = t_plus_15.tz_localize("UTC").tz_convert(_LOCAL_TZ)

    for ax, img, title in (
        (ax_sky_t, sky_img_t, f"Sky at T  {t_local.strftime('%H:%M')} local"),
        (ax_sky_tp, sky_img_t_plus_15, f"Sky at T+15  {t_plus_local.strftime('%H:%M')} local"),
    ):
        ax.set_facecolor("white")
        if img is None:
            ax.text(
                0.5,
                0.5,
                "(no sky image)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#555",
            )
        else:
            ax.imshow(img, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"Folsom  {date_local_str}  {title}", fontsize=10)

    ax_plot.set_facecolor("white")
    if cursor_idx + 1 > 0:
        sl = slice(0, cursor_idx + 1)
        ax_plot.plot(
            hours_local[sl],
            gt_kw[sl],
            color="black",
            linewidth=1.6,
            label=gt_label,
        )
        for values, color, label in preds:
            ax_plot.plot(
                hours_local[sl],
                values[sl],
                color=color,
                linewidth=1.6,
                label=label,
            )
        for values, color, _label in preds:
            ax_plot.scatter(
                [hours_local[cursor_idx]],
                [values[cursor_idx]],
                color=color,
                s=22,
                zorder=5,
            )
        ax_plot.scatter(
            [hours_local[cursor_idx]],
            [gt_kw[cursor_idx]],
            color="black",
            s=14,
            zorder=4,
        )

    ax_plot.set_xlim(xlim)
    ax_plot.set_ylim(ymin, ymax)
    ax_plot.set_xlabel("Hour of day (local)")
    ax_plot.set_ylabel("GHI (W/m²)")
    ax_plot.set_title(title_plot, fontsize=11)
    hour_ticks, hour_labels = _integer_hour_ticks(xlim)
    ax_plot.set_xticks(hour_ticks)
    ax_plot.set_xticklabels(hour_labels, fontsize=8)
    ax_plot.grid(True, color="#dddddd", linewidth=0.6, axis="both")
    ax_plot.legend(loc="upper left", frameon=False, fontsize=9)
    for s in ("top", "right"):
        ax_plot.spines[s].set_visible(False)

    fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.08, hspace=0.32, wspace=0.10)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    arr = rgba[..., :3].copy()
    plt.close(fig)
    return arr


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--date", type=str, default=_DEFAULT_DATE)
    p.add_argument("--checkpoint", type=str, default=str(_DEFAULT_CHECKPOINT))
    p.add_argument("--archive-name", type=str, default=_DEFAULT_ARCHIVE_NAME)
    p.add_argument("--run-name", type=str, default=_DEFAULT_RUN_NAME)
    p.add_argument("--checkpoint-extra", type=str, default=None)
    p.add_argument("--archive-name-extra", type=str, default=None)
    p.add_argument("--run-name-extra", type=str, default=None)
    p.add_argument(
        "--label-main",
        type=str,
        default=None,
        help="Legend label for main prediction (default: 'GHI + full sky' if extra set).",
    )
    p.add_argument("--label-extra", type=str, default="GHI only (zero sky)")
    p.add_argument("--checkpoint-third", type=str, default=None)
    p.add_argument("--archive-name-third", type=str, default=None)
    p.add_argument("--run-name-third", type=str, default=None)
    p.add_argument("--label-third", type=str, default="Third model")
    p.add_argument(
        "--ray-map-third",
        dest="ray_map_third",
        action="store_true",
        default=None,
    )
    p.add_argument(
        "--sun-mask-third",
        dest="sun_mask_third",
        action="store_true",
        default=None,
    )
    p.add_argument(
        "--sky-disc-mask-third",
        type=str,
        default=None,
        choices=[
            "none",
            "valid_disc",
            "tight_disc",
            "sun_halo",
            "sun_only",
            "manual_loose",
            "manual_tight",
        ],
    )
    p.add_argument("--label-gt", type=str, default="GT")
    p.add_argument("--stride-min", type=int, default=15)
    p.add_argument("--legacy-pre-518dca9", action="store_true")
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(_PROJECT_ROOT / "playground" / "animations"),
    )
    p.add_argument(
        "--run-label",
        type=str,
        default=None,
        help="Subfolder under --out-dir (default: dual_sky_<date>).",
    )
    p.add_argument("--force", action="store_true")
    p.add_argument("--keep-frames", action="store_true")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--fps", type=float, default=float(_FPS))
    p.add_argument("--sky-zarr", type=str, default=str(_DEFAULT_SKY_ZARR))
    p.add_argument("--dataset-config", type=str, default=_DEFAULT_DATASET_CONFIG)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    run_label = args.run_label or f"dual_sky_{args.date}"
    out_root = Path(args.out_dir).expanduser().resolve()
    out_dir = out_root / run_label
    out_dir.mkdir(parents=True, exist_ok=True)

    has_extra = bool(args.checkpoint_extra)
    has_third = bool(args.checkpoint_third)
    has_multi = has_extra or has_third
    label_main = args.label_main or ("GHI + full sky" if has_multi else "Point pred.")
    label_extra = args.label_extra
    label_third = args.label_third
    label_gt = args.label_gt

    stride_min = int(args.stride_min)
    stride_suffix = "" if stride_min == 15 else f"_stride{stride_min}"
    stem = f"{args.date}{stride_suffix}"
    npz_path = out_dir / f"{stem}_predictions.npz"
    meta_path = out_dir / f"{stem}_meta.json"
    gif_path = out_dir / f"{stem}.gif"
    frames_dir = out_dir / f"{stem}_frames"

    window = _local_day_window(args.date)
    print(
        f"[dual_sky] date_local={args.date}  utc_window=[{window.utc_start}, {window.utc_end})"
    )

    import xarray as xr

    sky_zarr_path = Path(args.sky_zarr).expanduser().resolve()
    _assert_valid_sky_source(sky_zarr_path, role="Sky display")
    if not sky_zarr_path.is_dir():
        raise FileNotFoundError(f"sky zarr not found: {sky_zarr_path}")
    print(f"[dual_sky] sky_zarr (display + model): {sky_zarr_path}")
    sky_ds = xr.open_zarr(str(sky_zarr_path), consolidated=False)
    sky_times_raw = sky_ds["time_utc"].values

    grid_utc, first_sky, last_sky = _find_display_grid_utc(
        window, np.asarray(sky_times_raw), stride_min=stride_min
    )
    print(
        f"[dual_sky] sky availability: first={first_sky}  last={last_sky}  "
        f"frames={len(grid_utc)} ({stride_min}-min stride)"
    )

    pred_extra_kw: np.ndarray | None = None
    pred_third_kw: np.ndarray | None = None

    if npz_path.is_file() and not args.force:
        print(f"[dual_sky] using cached predictions: {npz_path}")
        cached = np.load(npz_path, allow_pickle=False)
        keys = set(cached.files)
        t_utc_arr = cached["t_utc"]
        gt_kw = cached["gt_kw"]
        if "pred_sky_kw" in keys and "pred_nosky_kw" in keys:
            pred_kw = cached["pred_sky_kw"]
            pred_extra_kw = cached["pred_nosky_kw"]
        else:
            pred_kw = cached["pred_kw"]
        if "pred_third_kw" in keys:
            pred_third_kw = cached["pred_third_kw"]
        with meta_path.open() as f:
            meta = json.load(f)
        grid_utc = pd.DatetimeIndex(pd.to_datetime(t_utc_arr))
        if has_extra and pred_extra_kw is None:
            raise RuntimeError(
                f"--checkpoint-extra set but cached NPZ lacks pred_nosky_kw; use --force."
            )
        if has_third and pred_third_kw is None:
            raise RuntimeError(
                f"--checkpoint-third set but cached NPZ lacks pred_third_kw; use --force."
            )
    else:
        import torch

        if args.no_cuda:
            device = torch.device("cpu")
        elif args.device:
            device = torch.device(args.device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[dual_sky] device={device}")

        ds, model_sky_meta = _build_folsom_dataset_for_inference(
            _FORCE_PV_OUTPUT_LEN,
            sky_zarr=sky_zarr_path,
            dataset_config=str(args.dataset_config),
        )
        if int(ds.pv_output_len) != _FORCE_PV_OUTPUT_LEN:
            raise RuntimeError(
                f"dataset pv_output_len override failed: got {ds.pv_output_len}, "
                f"expected {_FORCE_PV_OUTPUT_LEN}"
            )

        anchors_rows: list[int] = []
        for t_display in grid_utc:
            t0 = pd.Timestamp(t_display) - pd.Timedelta(minutes=15)
            anchors_rows.append(_resolve_anchor_row_for_t0(ds, t0))
        print(f"[dual_sky] resolved {len(anchors_rows)} anchors")

        print(f"[dual_sky] forwarding main checkpoint: {args.checkpoint}")
        pred_kw, gt_kw, cz, run_meta = _run_model_on_anchors(
            ds,
            anchors_rows,
            checkpoint_path=Path(args.checkpoint).expanduser().resolve(),
            device=device,
            batch_size=int(args.batch_size),
            legacy_pre_518dca9=bool(args.legacy_pre_518dca9),
        )

        run_meta_extra: dict | None = None
        if has_extra:
            print(f"[dual_sky] forwarding extra checkpoint: {args.checkpoint_extra}")
            pred_extra_kw, gt_kw_extra, cz_extra, run_meta_extra = _run_model_on_anchors(
                ds,
                anchors_rows,
                checkpoint_path=Path(args.checkpoint_extra).expanduser().resolve(),
                device=device,
                batch_size=int(args.batch_size),
                legacy_pre_518dca9=bool(args.legacy_pre_518dca9),
            )
            if not np.allclose(gt_kw, gt_kw_extra, equal_nan=True):
                raise RuntimeError("GT mismatch between main and extra forwards.")
            if not np.allclose(cz, cz_extra, equal_nan=True):
                raise RuntimeError("cos_zenith mismatch between main and extra forwards.")

        run_meta_third: dict | None = None
        if has_third:
            ckpt_third_path = Path(args.checkpoint_third).expanduser().resolve()
            ckpt_third_peek = torch.load(ckpt_third_path, map_location="cpu")
            need_sky_ch = _infer_sky_in_channels_from_state(
                ckpt_third_peek.get("model_state_dict", ckpt_third_peek)
            )
            ds_third = ds
            if (
                int(ds.sky_in_channels) != need_sky_ch
                or args.ray_map_third is not None
                or args.sun_mask_third is not None
                or args.sky_disc_mask_third is not None
            ):
                ray_map_third = args.ray_map_third
                sun_mask_third = args.sun_mask_third
                if ray_map_third is None and sun_mask_third is None:
                    if need_sky_ch == 4:
                        sun_mask_third = True
                    elif need_sky_ch == 6:
                        ray_map_third = True
                    elif need_sky_ch == 7:
                        ray_map_third = True
                        sun_mask_third = True
                ds_third, _ = _build_folsom_dataset_for_inference(
                    _FORCE_PV_OUTPUT_LEN,
                    sky_zarr=sky_zarr_path,
                    dataset_config=str(args.dataset_config),
                    ray_map=ray_map_third,
                    sun_mask=sun_mask_third,
                    sky_disc_mask=args.sky_disc_mask_third,
                )
                if int(ds_third.sky_in_channels) != need_sky_ch:
                    raise RuntimeError(
                        f"Third checkpoint expects sky_in_channels={need_sky_ch} but "
                        f"built dataset has {ds_third.sky_in_channels}."
                    )
            print(f"[dual_sky] forwarding third checkpoint: {ckpt_third_path}")
            pred_third_kw, gt_kw_third, cz_third, run_meta_third = _run_model_on_anchors(
                ds_third,
                anchors_rows,
                checkpoint_path=ckpt_third_path,
                device=device,
                batch_size=int(args.batch_size),
                legacy_pre_518dca9=bool(args.legacy_pre_518dca9),
            )
            if not np.allclose(gt_kw, gt_kw_third, equal_nan=True):
                raise RuntimeError("GT mismatch between main and third forwards.")
            if not np.allclose(cz, cz_third, equal_nan=True):
                raise RuntimeError("cos_zenith mismatch between main and third forwards.")

        t_utc_arr = np.asarray(
            [pd.Timestamp(t).to_datetime64() for t in grid_utc], dtype="datetime64[s]"
        )
        t_local_arr = np.asarray(
            [
                pd.Timestamp(t)
                .tz_localize("UTC")
                .tz_convert(_LOCAL_TZ)
                .tz_localize(None)
                .to_datetime64()
                for t in grid_utc
            ],
            dtype="datetime64[s]",
        )
        if has_extra:
            npz_payload: dict[str, Any] = dict(
                t_utc=t_utc_arr,
                t_local=t_local_arr,
                gt_kw=gt_kw,
                pred_sky_kw=pred_kw,
                pred_nosky_kw=pred_extra_kw,
                cos_zenith=cz,
            )
        else:
            npz_payload = dict(
                t_utc=t_utc_arr,
                t_local=t_local_arr,
                gt_kw=gt_kw,
                pred_kw=pred_kw,
                cos_zenith=cz,
            )
        if has_third:
            npz_payload["pred_third_kw"] = pred_third_kw
        np.savez_compressed(npz_path, **npz_payload)

        meta = dict(
            date_local=args.date,
            run_label=run_label,
            layout="dual_sky",
            dataset_config=str(args.dataset_config),
            horizon_step_used=0,
            pv_output_len_at_inference=int(_FORCE_PV_OUTPUT_LEN),
            n_frames=int(len(grid_utc)),
            fps=float(args.fps),
            stride_min=int(stride_min),
            sky_zarr=str(sky_zarr_path),
            **model_sky_meta,
            sky_time_tolerance_s=int(_SKY_TIME_MATCH_TOLERANCE_S),
            local_tz="America/Los_Angeles",
            two_model_overlay=bool(has_extra),
            three_model_overlay=bool(has_third),
            legacy_pre_518dca9=bool(args.legacy_pre_518dca9),
            x_axis_ticks="integer_hours_every_hour",
        )
        if has_extra:
            meta["main"] = dict(
                label=label_main,
                color=_MAIN_COLOR,
                archive_name=args.archive_name,
                run_name=args.run_name,
                checkpoint=run_meta["checkpoint"],
                zero_sky=bool(run_meta["zero_sky"]),
                use_nwp=bool(run_meta["use_nwp"]),
                nwp_features=run_meta["nwp_features"],
                nwp_use_invalid_mask=bool(run_meta["nwp_use_invalid_mask"]),
                missing_keys_count=int(run_meta["missing_keys_count"]),
                unexpected_keys_count=int(run_meta["unexpected_keys_count"]),
                missing_keys_first5=list(run_meta["missing_keys_first5"]),
                unexpected_keys_first5=list(run_meta["unexpected_keys_first5"]),
            )
            assert run_meta_extra is not None
            meta["extra"] = dict(
                label=label_extra,
                color=_EXTRA_COLOR,
                archive_name=args.archive_name_extra,
                run_name=args.run_name_extra,
                checkpoint=run_meta_extra["checkpoint"],
                zero_sky=bool(run_meta_extra["zero_sky"]),
                use_nwp=bool(run_meta_extra["use_nwp"]),
                nwp_features=run_meta_extra["nwp_features"],
                nwp_use_invalid_mask=bool(run_meta_extra["nwp_use_invalid_mask"]),
                missing_keys_count=int(run_meta_extra["missing_keys_count"]),
                unexpected_keys_count=int(run_meta_extra["unexpected_keys_count"]),
                missing_keys_first5=list(run_meta_extra["missing_keys_first5"]),
                unexpected_keys_first5=list(run_meta_extra["unexpected_keys_first5"]),
            )
        if has_third:
            assert run_meta_third is not None
            meta["third"] = dict(
                label=label_third,
                color=_THIRD_COLOR,
                archive_name=args.archive_name_third,
                run_name=args.run_name_third,
                checkpoint=run_meta_third["checkpoint"],
                zero_sky=bool(run_meta_third["zero_sky"]),
                use_nwp=bool(run_meta_third["use_nwp"]),
                nwp_features=run_meta_third["nwp_features"],
                nwp_use_invalid_mask=bool(run_meta_third["nwp_use_invalid_mask"]),
                missing_keys_count=int(run_meta_third["missing_keys_count"]),
                unexpected_keys_count=int(run_meta_third["unexpected_keys_count"]),
                missing_keys_first5=list(run_meta_third["missing_keys_first5"]),
                unexpected_keys_first5=list(run_meta_third["unexpected_keys_first5"]),
            )
        if not has_extra:
            meta.update(
                archive_name=args.archive_name,
                run_name=args.run_name,
                checkpoint=run_meta["checkpoint"],
                zero_sky=bool(run_meta["zero_sky"]),
                use_nwp=bool(run_meta["use_nwp"]),
                nwp_features=run_meta["nwp_features"],
                nwp_use_invalid_mask=bool(run_meta["nwp_use_invalid_mask"]),
                missing_keys_count=int(run_meta["missing_keys_count"]),
                unexpected_keys_count=int(run_meta["unexpected_keys_count"]),
                missing_keys_first5=list(run_meta["missing_keys_first5"]),
                unexpected_keys_first5=list(run_meta["unexpected_keys_first5"]),
            )
        with meta_path.open("w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        print(f"[dual_sky] cached predictions -> {npz_path}")

    hours_local = np.asarray(
        [_utc_naive_to_local_hour(pd.Timestamp(t), _LOCAL_TZ) for t in grid_utc],
        dtype=np.float64,
    )

    arrays_for_ylim = [gt_kw, pred_kw]
    if pred_extra_kw is not None:
        arrays_for_ylim.append(pred_extra_kw)
    if pred_third_kw is not None:
        arrays_for_ylim.append(pred_third_kw)
    finite_vals = np.concatenate(arrays_for_ylim)
    ymax_data = float(np.nanmax(finite_vals)) if finite_vals.size else 1.0
    if not np.isfinite(ymax_data) or ymax_data <= 0:
        ymax_data = 1.0
    ymax = max(50.0, 1.10 * ymax_data)
    ymin = -0.04 * ymax
    xlim_pad = 0.4
    xlim = (
        float(np.floor(hours_local[0] - xlim_pad)),
        float(np.ceil(hours_local[-1] + xlim_pad)),
    )

    title_plot = f"Our approach (1-step ViT nowcast) — {args.date}"

    preds_for_render: list[tuple[np.ndarray, str, str]] = [
        (pred_kw, _MAIN_COLOR, label_main),
    ]
    if pred_extra_kw is not None:
        preds_for_render.append((pred_extra_kw, _EXTRA_COLOR, label_extra))
    if pred_third_kw is not None:
        preds_for_render.append((pred_third_kw, _THIRD_COLOR, label_third))

    if args.keep_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []
    for i, t in enumerate(grid_utc):
        t_ts = pd.Timestamp(t)
        sky_t = _read_sky_image_for_time(sky_ds, t_ts)
        sky_tp = _read_sky_image_for_time(sky_ds, t_ts + pd.Timedelta(minutes=15))
        frame = _render_one_frame_dual_sky(
            t_utc=t_ts,
            hours_local=hours_local,
            gt_kw=gt_kw,
            preds=preds_for_render,
            cursor_idx=i,
            sky_img_t=sky_t,
            sky_img_t_plus_15=sky_tp,
            date_local_str=args.date,
            ymin=ymin,
            ymax=ymax,
            xlim=xlim,
            title_plot=title_plot,
            gt_label=label_gt,
        )
        frames.append(frame)
        if args.keep_frames:
            from PIL import Image

            Image.fromarray(frame).save(frames_dir / f"frame_{i:03d}.png")
        if (i + 1) % 10 == 0 or i == 0 or i == len(grid_utc) - 1:
            print(f"[dual_sky] rendered frame {i + 1}/{len(grid_utc)}")

    _save_gif(frames, gif_path, fps=float(args.fps))
    gif_size_mb = gif_path.stat().st_size / (1024 * 1024)

    print("\n[dual_sky] === SUMMARY ===")
    print(f"  date_local      : {args.date}")
    print(f"  run_label       : {run_label}")
    print(f"  layout          : dual_sky (T | T+15 top, GHI bottom)")
    print(f"  x_axis_ticks    : every integer hour in [{xlim[0]:.0f}, {xlim[1]:.0f}]")
    print(f"  frames          : {len(frames)} @ {args.fps:g} fps")
    print(f"  gif             : {gif_path}  ({gif_size_mb:.2f} MB)")
    print(f"  npz             : {npz_path}")
    print(f"  peak_gt_kw      : {float(np.nanmax(gt_kw)):.2f}")
    print(f"  peak_pred_main  : {float(np.nanmax(pred_kw)):.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
