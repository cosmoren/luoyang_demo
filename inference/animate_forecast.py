"""
Animate a rolling 15-min next-step nowcast for one local day at Folsom, CA.

Produces a two-panel GIF:

    LEFT  : fisheye sky-camera image at the current display time T
    RIGHT : ground truth (black) + one-or-two point predictions vs
            hour-of-day (local)

For each display time T on a 15-min local grid:

    anchor t0 = T - 15 min
    run the trained ViT once with pv_output_len = 16 (4 h horizon)
    take output step 0  ==> the +15 min step  ==> prediction at T

This is the "one model forward = one displayed point" protocol described in
the project brief. Inference and rendering are kept separate so visual tweaks
do not require re-running the model: predictions are cached in a NPZ next to
the GIF (see ``--force`` to recompute).

Multi-model overlay (v2+ mode):

    Pass ``--checkpoint-extra <path>`` to overlay a second model's prediction
    on the same panel as a blue line. Optionally pass ``--checkpoint-third
    <path>`` for a third (green) line. This is used to visually compare sky /
    no-sky / other arms. Each checkpoint can carry different ``zero_sky`` flags;
    each forward uses its own checkpoint's flag. The dataset is constructed once
    and reused for all forwards.

Usage::

    # v1 (single model, same as before):
    micromamba run -n luoyang python inference/animate_forecast.py \
        --date 2014-01-15

    # v2 (sky vs no-sky overlay):
    micromamba run -n luoyang python inference/animate_forecast.py \
        --run-label v2_sky_vs_nosky \
        --date 2014-01-15 \
        --checkpoint /path/to/sky_best.pt \
        --checkpoint-extra /path/to/nosky_best.pt

    # v3 (three-model overlay):
    micromamba run -n luoyang python inference/animate_forecast.py \
        --run-label v3_three_way \
        --date 2014-01-15 \
        --checkpoint /path/to/sky_best.pt \
        --checkpoint-extra /path/to/nosky_best.pt \
        --checkpoint-third /path/to/other_best.pt

Outputs (under ``playground/animations/<run_label>/``):

    <date>.gif                  the animation itself
    <date>_predictions.npz      per-frame (t, gt, pred[, pred_extra[, pred_third]]) arrays
                                v1 schema: gt_kw, pred_kw
                                v2 schema: gt_kw, pred_sky_kw, pred_nosky_kw
                                v3 adds: pred_third_kw (when --checkpoint-third set)
    <date>_meta.json            checkpoint / run / config metadata
    <date>_frames/              per-frame PNGs (only with --keep-frames)
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_DEFAULT_DATE = "2014-01-15"
_DEFAULT_CHECKPOINT = (
    Path.home()
    / "experiments_archive"
    / "lr2e5_4h_2seeds_sky_nwp_2026-06-05"
    / "checkpoints_folsom_pv"
    / "ghi_sky_nwp_s0"
    / "folsom_pv_forecast_vit_best_gpu2.pt"
)
_DEFAULT_ARCHIVE_NAME = "lr2e5_4h_2seeds_sky_nwp_2026-06-05"
_DEFAULT_RUN_NAME = "ghi_sky_nwp_s0"
# Variant tag used as a subfolder under --out-dir. Change the label whenever you
# change model / horizon / anchor stride / rendering choices so old artifacts
# survive. RUNS.md alongside --out-dir documents what each label means.
_DEFAULT_RUN_LABEL = "v1_vit_sky_nwp_s0"
_DEFAULT_DATASET_CONFIG = "conf_folsom.yaml"
# Canonical Folsom full-resolution sky store (same imagery as ``processed/full/sky`` JPGs).
_DEFAULT_SKY_ZARR = (
    _PROJECT_ROOT.parent / "folsom_ds" / "processed" / "full" / "sky_xr_120.zarr"
).resolve()
_INVALID_SKY_PATH_MARKERS = ("sample_250k",)

# Folsom is California; January is PST = UTC-8, summer is PDT = UTC-7. zoneinfo
# does the right thing automatically for any date in [1970, 2037].
_LOCAL_TZ = ZoneInfo("America/Los_Angeles")

# Override the dataset YAML's pv_output_len so model output, forecast_timefeats,
# and target arrays match the checkpoint horizon (t0+15 models use 1 step).
_FORCE_PV_OUTPUT_LEN = 1

# Animation tuning. The brief asks for fps=5 and frame size ~1024x512 or smaller.
_FPS = 5
_FRAME_DPI = 110
_FIG_W_INCHES = 9.2
_FIG_H_INCHES = 4.6

# Sky lookup tolerance: when picking the sky frame to display at time T we
# accept the nearest Zarr time_utc within +/- this many seconds. Folsom sky is
# ~1 min cadence, so 120 s is generous.
_SKY_TIME_MATCH_TOLERANCE_S = 120


@dataclass(frozen=True)
class DayWindow:
    """Local-day display window with the UTC range we actually filter on."""

    date_local: datetime  # midnight local, naive (zone implied)
    utc_start: pd.Timestamp  # naive UTC
    utc_end: pd.Timestamp  # naive UTC, exclusive
    tz: ZoneInfo


def _local_day_window(date_str: str) -> DayWindow:
    """Build the [utc_start, utc_end) span that covers ``date_str`` in Pacific time."""
    y, m, d = (int(x) for x in date_str.split("-"))
    start_local = datetime(y, m, d, 0, 0, 0, tzinfo=_LOCAL_TZ)
    end_local = start_local + timedelta(days=1)
    utc_start = pd.Timestamp(start_local.astimezone(ZoneInfo("UTC"))).tz_convert("UTC").tz_localize(None)
    utc_end = pd.Timestamp(end_local.astimezone(ZoneInfo("UTC"))).tz_convert("UTC").tz_localize(None)
    return DayWindow(
        date_local=datetime(y, m, d),
        utc_start=utc_start,
        utc_end=utc_end,
        tz=_LOCAL_TZ,
    )


def _utc_naive_to_local_hour(t_utc: pd.Timestamp, tz: ZoneInfo) -> float:
    """Convert naive UTC timestamp to hour-of-day in the requested local zone."""
    t = pd.Timestamp(t_utc)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    t_local = t.tz_convert(tz)
    return t_local.hour + t_local.minute / 60.0 + t_local.second / 3600.0


def _find_display_grid_utc(
    window: DayWindow,
    sky_times_utc: np.ndarray,
    stride_min: int = 15,
) -> tuple[pd.DatetimeIndex, pd.Timestamp, pd.Timestamp]:
    """Build the ``stride_min``-spaced UTC display grid spanned by available sky imagery.

    ``stride_min`` controls the cadence of display times T (one model forward per
    T). Native CSV cadence is 1 min, so any integer-minute stride >= 1 is
    supported. The anchor lag (``t0 = T - 15 min``) is independent of stride and
    is set by the model's output_interval (step 0 = +15 min from anchor).
    """
    if int(stride_min) < 1:
        raise ValueError(f"stride_min must be >= 1 minute, got {stride_min}")
    if sky_times_utc.size == 0:
        raise RuntimeError(
            f"No sky imagery found in Zarr for {window.date_local:%Y-%m-%d} (local) "
            f"= UTC [{window.utc_start}, {window.utc_end})"
        )
    sky_idx = pd.DatetimeIndex(pd.to_datetime(sky_times_utc))
    in_window = (sky_idx >= window.utc_start) & (sky_idx < window.utc_end)
    sky_in = sky_idx[in_window]
    if len(sky_in) == 0:
        raise RuntimeError(
            f"No sky imagery in window for {window.date_local:%Y-%m-%d} (local) "
            f"= UTC [{window.utc_start}, {window.utc_end})"
        )
    first_sky = sky_in[0]
    last_sky = sky_in[-1]

    # Round to a clean ``stride_min``-min grid: first display T = ceil(first_sky);
    # last display T = floor(last_sky). pandas ``.ceil``/``.floor`` already
    # epoch-align to the requested resolution.
    freq = f"{int(stride_min)}min"
    t_first = pd.Timestamp(first_sky).ceil(freq)
    t_last = pd.Timestamp(last_sky).floor(freq)
    if t_last < t_first:
        raise RuntimeError(
            f"Sky imagery for {window.date_local:%Y-%m-%d} spans less than one "
            f"{int(stride_min)}-min boundary (first_sky={first_sky}, last_sky={last_sky})"
        )
    grid = pd.date_range(t_first, t_last, freq=freq)
    return grid, pd.Timestamp(first_sky), pd.Timestamp(last_sky)


# --------------------------------------------------------------------------- #
# Inference path
# --------------------------------------------------------------------------- #


def _assert_valid_sky_source(path: Path, *, role: str) -> None:
    """Reject sample_250k sky paths; playground GIFs must use full-resolution sky."""
    s = path.as_posix()
    for marker in _INVALID_SKY_PATH_MARKERS:
        if marker in s:
            raise ValueError(
                f"{role} sky path must not use {marker!r} (got {path}). "
                f"Use {_DEFAULT_SKY_ZARR} or folsom_ds/processed/full/sky JPGs."
            )


def _infer_sky_in_channels_from_state(state: dict) -> int:
    """Read sky-branch input width from a saved ``model_state_dict``."""
    key = "sky_patch_embed.patch_embed.weight"
    if key in state:
        return int(state[key].shape[1])
    return 3


def _build_folsom_dataset_for_inference(
    pv_output_len: int,
    *,
    sky_zarr: Path,
    dataset_config: str = _DEFAULT_DATASET_CONFIG,
    ray_map: bool | None = None,
    sun_mask: bool | None = None,
    sky_disc_mask: str | None = None,
    sky_disc_mask_radius_px: float | None = None,
) -> tuple[Any, dict[str, str]]:
    """Construct a FolsomIrradianceDataset with the given pv_output_len override.

    Uses the trainer's standard dataset kwargs and forces ``split="train"`` because
    early-2014 (well within the train-time band) is where the sample dates live.
    We never call ``__getitem__``; we drive ``_build_tensors(anchor)`` directly.

    ``sky_zarr`` is applied to **model inference** (``skyimg_dir`` + ``paths.sky_format:
    zarr``). It must match the left-panel ``--sky-zarr`` store so display and
    forward pass see the same full-resolution imagery (not ``sample_250k/sky``).
    """
    import yaml

    from training.train_vit_test_folsom import (
        _dataset_kwargs,
        _load_yaml,
        _resolve_sky_channels,
        _resolve_sky_disc_mask_mode,
        _resolve_sky_disc_mask_radius_px,
    )

    sky_zarr = Path(sky_zarr).expanduser().resolve()
    _assert_valid_sky_source(sky_zarr, role="Model inference")
    if not sky_zarr.is_dir():
        raise FileNotFoundError(f"model sky zarr not found: {sky_zarr}")

    sky_channels_override = _resolve_sky_channels(dataset_config, ray_map, sun_mask)
    sky_disc_mask_mode_override = _resolve_sky_disc_mask_mode(
        dataset_config, sky_disc_mask
    )
    sky_disc_mask_radius_px_override = _resolve_sky_disc_mask_radius_px(
        dataset_config, sky_disc_mask_radius_px
    )
    ds_kwargs = _dataset_kwargs(
        dataset_config,
        "train",
        sky_channels_override=sky_channels_override,
        sky_disc_mask_mode_override=sky_disc_mask_mode_override,
        sky_disc_mask_radius_px_override=sky_disc_mask_radius_px_override,
    )
    ds_kwargs["pv_output_len"] = int(pv_output_len)
    ds_kwargs["skyimg_dir"] = str(sky_zarr)

    base_cfg_path = Path(ds_kwargs["config_path"])
    cfg = _load_yaml(base_cfg_path)
    cfg2 = copy.deepcopy(cfg)
    cfg2.setdefault("paths", {})["sky_format"] = "zarr"
    tmp = Path(tempfile.mkdtemp(prefix="animate_sky_cfg_"))
    patched_cfg = tmp / "dataset.yaml"
    with patched_cfg.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg2, f, sort_keys=False, allow_unicode=True)
    ds_kwargs["config_path"] = str(patched_cfg.resolve())

    from dataloader.folsom import FolsomIrradianceDataset

    ds = FolsomIrradianceDataset(**ds_kwargs)
    sky_meta = dict(
        model_sky_zarr=str(sky_zarr),
        model_sky_format="zarr",
    )
    return ds, sky_meta


def _resolve_anchor_row_for_t0(ds, t0_utc: pd.Timestamp) -> int:
    """Return CSV row index whose ``time_col`` exactly matches ``t0_utc`` (naive UTC)."""
    t = pd.Timestamp(t0_utc)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    times = ds._df[ds._time_col]
    ts = times.dt.tz_convert("UTC").dt.tz_localize(None) if getattr(times.dt, "tz", None) is not None else times
    ok = ts == t
    if not bool(ok.any()):
        raise ValueError(
            f"No CSV row at {t!s} (naive UTC); first={ts.iloc[0]} last={ts.iloc[-1]}"
        )
    return int(np.flatnonzero(ok.to_numpy())[0])


def _run_model_on_anchors(
    ds,
    anchors: list[int],
    *,
    checkpoint_path: Path,
    device,
    batch_size: int = 8,
    legacy_pre_518dca9: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Forward the model once per anchor; return (pred_kw, gt_kw, cos_zenith, meta).

    Predictions/targets are *raw GHI* in W/m^2 (the Folsom training target post-
    commit 518dca9; see :mod:`inference.infer_testset_folsom`). The output arrays
    line up with ``anchors`` (one entry per anchor; we only ever take step 0).

    ``legacy_pre_518dca9`` switch: pre-518dca9 (May-26 era) checkpoints were
    trained with a different normalization recipe -- ``_FOLSOM_GHI_SCALE = 1100``
    in the dataloader, ``target_pv = ghi/1100`` (normalized), ``p_mean = 1``,
    ``kt = (ghi/1100) / (ghi_cs/1100) = ghi/ghi_cs ~ [0, 1.5]``, and trainer-side
    ``kt_input_scale = 20.0``. HEAD's dataloader instead emits raw-W/m^2 ``kt =
    1000 * ghi/ghi_cs`` and ``target_pv = ghi`` (W/m^2), with trainer scale
    ``_FOLSOM_KT_INPUT_SCALE = 4000.0``. To run a pre-518dca9 checkpoint
    correctly with HEAD's dataloader, we (a) pre-scale ``d["kt"]`` so the
    in-function ``/ _FOLSOM_KT_INPUT_SCALE`` yields ``kt_legacy/20``, (b)
    multiply the output by ``20`` to recover ``kt_legacy``, and (c) multiply
    the reconstructed ``pv_pred`` by an extra ``1000`` to land in W/m^2
    (HEAD's ``target_p_cs = ghi_cs/1000`` so ``kt_legacy * target_p_cs * 1000
    = (ghi/ghi_cs) * (ghi_cs/1000) * 1000 = ghi``). Without this switch the
    May-26 model sees inputs ~5x larger than training distribution and outputs
    saturate at ~30% of the correct magnitude.
    """
    import torch
    from torch.utils.data import DataLoader

    from dataloader.folsom import _FOLSOM_KT_INPUT_SCALE
    from dataloader.luoyang_zarr import collate_batched
    from models.models import pv_forecasting_model_vit_imgs
    from training.train_vit_test_folsom import (
        _batch_to_device,
        _prepare_nwp_for_vit,
        _prepare_sky_for_vit,
        forward_vit,
        resolve_nwp_features_from_ckpt,
    )

    # Legacy (pre-518dca9) constants: the May-26 trainer fed the model
    # ``kt_legacy / 20.0`` where ``kt_legacy = ghi/ghi_cs ~ [0, 1.5]``. Today's
    # dataloader emits ``kt = 1000 * ghi/ghi_cs``. Pre-scaling factor below
    # arranges that ``forward_vit``'s in-function ``/ _FOLSOM_KT_INPUT_SCALE``
    # yields ``kt_legacy / 20`` exactly. Reconstruction multiplier is the
    # extra 1000x needed to land in W/m^2 (since HEAD's ``target_p_cs`` lives
    # at the 1000-scale, not the 1100-scale).
    _LEGACY_KT_INPUT_SCALE = 20.0
    _LEGACY_KT_DATALOADER_SCALE_RATIO = 1000.0  # HEAD kt = 1000 * legacy kt
    _LEGACY_KT_PRESCALE = (
        _FOLSOM_KT_INPUT_SCALE / (_LEGACY_KT_DATALOADER_SCALE_RATIO * _LEGACY_KT_INPUT_SCALE)
    )  # = 4000 / (1000 * 20) = 0.2
    _LEGACY_PV_RECON_EXTRA_SCALE = _LEGACY_KT_DATALOADER_SCALE_RATIO  # 1000.0
    _LEGACY_KT_RECOVERY_MUL = _LEGACY_KT_INPUT_SCALE  # 20.0

    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_zero_sky = bool(ckpt.get("zero_sky", False))
    ckpt_use_nwp = bool(ckpt.get("use_nwp", True))
    nwp_features, nwp_use_invalid_mask = resolve_nwp_features_from_ckpt(ckpt)
    state = ckpt.get("model_state_dict", ckpt)
    sky_in_channels = _infer_sky_in_channels_from_state(state)
    if int(getattr(ds, "sky_in_channels", 3)) != sky_in_channels:
        raise RuntimeError(
            f"Dataset sky_in_channels={getattr(ds, 'sky_in_channels', 3)} but "
            f"checkpoint {checkpoint_path.name} expects {sky_in_channels}. "
            f"Rebuild the dataset with matching --ray-map / --sun-mask / "
            f"--sky-disc-mask flags for this checkpoint."
        )

    model = pv_forecasting_model_vit_imgs(
        dev_dn_list=ds.devDn_list,
        nwp_features=nwp_features,
        use_invalid_mask=nwp_use_invalid_mask,
        sky_in_channels=sky_in_channels,
    ).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing_list = list(missing)
    unexpected_list = list(unexpected)
    if missing_list:
        print(
            f"[animate] WARNING missing keys: {len(missing_list)} "
            f"(first 5: {missing_list[:5]}) for {checkpoint_path}"
        )
    if unexpected_list:
        print(
            f"[animate] WARNING unexpected keys: {len(unexpected_list)} "
            f"(first 5: {unexpected_list[:5]}) for {checkpoint_path}"
        )
    model.eval()

    class _AnchorView(torch.utils.data.Dataset):
        def __init__(self, ds, anchors):
            self.ds = ds
            self.anchors = list(anchors)

        def __len__(self):
            return len(self.anchors)

        def __getitem__(self, i):
            return self.ds._build_tensors(int(self.anchors[i]))

    loader = DataLoader(
        _AnchorView(ds, anchors),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batched,
        num_workers=0,
    )

    preds_step0 = np.zeros(len(anchors), dtype=np.float32)
    gts_step0 = np.zeros(len(anchors), dtype=np.float32)
    cz_step0 = np.zeros(len(anchors), dtype=np.float32)

    cursor = 0
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else __import__("contextlib").nullcontext()
    )
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            d = _batch_to_device(batch, device)
            _prepare_nwp_for_vit(d, use_nwp=ckpt_use_nwp)
            _prepare_sky_for_vit(d, zero_sky=ckpt_zero_sky)
            if legacy_pre_518dca9:
                # Pre-scale d["kt"] so forward_vit's in-function division by
                # _FOLSOM_KT_INPUT_SCALE (= 4000) yields kt_legacy/20.
                d["kt"] = d["kt"] * _LEGACY_KT_PRESCALE
            with autocast_ctx:
                output = forward_vit(model, d)
            if legacy_pre_518dca9:
                kt_pred = output * _LEGACY_KT_RECOVERY_MUL
                pv_pred = (
                    kt_pred
                    * d["target_p_cs"]
                    * d["p_mean"].unsqueeze(1)
                    * _LEGACY_PV_RECON_EXTRA_SCALE
                ).float()
            else:
                kt_pred = output * _FOLSOM_KT_INPUT_SCALE
                pv_pred = (kt_pred * d["target_p_cs"] * d["p_mean"].unsqueeze(1)).float()
            B = pv_pred.shape[0]
            cz = d["forecast_timefeats"][:, 0, 3].detach().cpu().numpy()
            night = cz < 0
            p0 = pv_pred[:, 0].detach().cpu().numpy()
            p0 = np.where(night, 0.0, p0)
            preds_step0[cursor : cursor + B] = p0
            gts_step0[cursor : cursor + B] = d["target_pv"][:, 0].detach().cpu().numpy()
            cz_step0[cursor : cursor + B] = cz
            cursor += B
            if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
                print(f"[animate] batch {batch_idx + 1}  cursor={cursor}/{len(anchors)}")

    meta = dict(
        checkpoint=str(checkpoint_path),
        zero_sky=ckpt_zero_sky,
        use_nwp=ckpt_use_nwp,
        nwp_features=list(nwp_features),
        nwp_use_invalid_mask=bool(nwp_use_invalid_mask),
        missing_keys_count=len(missing_list),
        unexpected_keys_count=len(unexpected_list),
        missing_keys_first5=missing_list[:5],
        unexpected_keys_first5=unexpected_list[:5],
        legacy_pre_518dca9=bool(legacy_pre_518dca9),
    )
    return preds_step0, gts_step0, cz_step0, meta


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _read_sky_image_for_time(
    sky_zarr_ds,
    t_utc: pd.Timestamp,
    *,
    tolerance_s: int = _SKY_TIME_MATCH_TOLERANCE_S,
) -> np.ndarray | None:
    """Return ``[H, W, 3]`` uint8 image nearest to ``t_utc`` within tolerance, else None."""
    raw = sky_zarr_ds["time_utc"].values
    sky_times = pd.DatetimeIndex(pd.to_datetime(raw))
    want = pd.Timestamp(t_utc)
    if want.tzinfo is not None:
        want = want.tz_convert("UTC").tz_localize(None)
    if sky_times.tz is not None:
        sky_times = sky_times.tz_convert("UTC").tz_localize(None)
    diffs = (sky_times - want).asi8
    i = int(np.argmin(np.abs(diffs)))
    if abs(int(diffs[i])) > tolerance_s * 1_000_000_000:
        return None
    img = np.asarray(sky_zarr_ds["images"].isel(time_utc=i).values)
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        m = float(np.nanmax(img))
        if m <= 1.5:
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _render_one_frame(
    *,
    t_utc: pd.Timestamp,
    hours_local: np.ndarray,
    gt_kw: np.ndarray,
    preds: list[tuple[np.ndarray, str, str]],
    cursor_idx: int,
    sky_img: np.ndarray | None,
    date_local_str: str,
    ymin: float,
    ymax: float,
    xlim: tuple[float, float],
    title_right: str,
    gt_label: str = "Ground truth",
) -> np.ndarray:
    """Render one (left=sky, right=plot) frame and return as ``[H, W, 3]`` uint8.

    ``preds`` is a list of ``(values, hex_color, legend_label)`` tuples; one
    line is plotted per entry. v1 mode passes a single tuple (red); multi-model
    mode passes up to three (red main, blue extra, green third).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(_FIG_W_INCHES, _FIG_H_INCHES), dpi=_FRAME_DPI, gridspec_kw={"width_ratios": [1.0, 1.4]}
    )
    fig.patch.set_facecolor("white")

    ax_l.set_facecolor("white")
    if sky_img is None:
        ax_l.text(
            0.5,
            0.5,
            "(no sky image)",
            ha="center",
            va="center",
            transform=ax_l.transAxes,
            color="#555",
        )
    else:
        ax_l.imshow(sky_img, interpolation="nearest")
    ax_l.set_xticks([])
    ax_l.set_yticks([])
    for s in ax_l.spines.values():
        s.set_visible(False)
    t_local = pd.Timestamp(t_utc).tz_localize("UTC").tz_convert(_LOCAL_TZ)
    ax_l.set_title(
        f"Folsom sky  {date_local_str}  {t_local.strftime('%H:%M')} local",
        fontsize=11,
    )

    ax_r.set_facecolor("white")
    if cursor_idx + 1 > 0:
        sl = slice(0, cursor_idx + 1)
        ax_r.plot(
            hours_local[sl],
            gt_kw[sl],
            color="black",
            linewidth=1.6,
            label=gt_label,
        )
        for values, color, label in preds:
            ax_r.plot(
                hours_local[sl],
                values[sl],
                color=color,
                linewidth=1.6,
                label=label,
            )
        for values, color, _label in preds:
            ax_r.scatter(
                [hours_local[cursor_idx]],
                [values[cursor_idx]],
                color=color,
                s=22,
                zorder=5,
            )
        ax_r.scatter(
            [hours_local[cursor_idx]],
            [gt_kw[cursor_idx]],
            color="black",
            s=14,
            zorder=4,
        )

    ax_r.set_xlim(xlim)
    ax_r.set_ylim(ymin, ymax)
    ax_r.set_xlabel("Hour of day (local)")
    ax_r.set_ylabel("GHI (W/m²)")
    ax_r.set_title(title_right, fontsize=11)
    ax_r.grid(True, color="#dddddd", linewidth=0.6)
    ax_r.legend(loc="upper left", frameon=False, fontsize=9)
    for s in ("top", "right"):
        ax_r.spines[s].set_visible(False)

    fig.tight_layout(pad=1.0)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    arr = rgba[..., :3].copy()
    plt.close(fig)
    return arr


def _save_gif(frames: list[np.ndarray], out_path: Path, *, fps: float = _FPS) -> None:
    """Save a list of ``[H, W, 3]`` uint8 frames as an animated GIF (loop forever)."""
    from PIL import Image

    if not frames:
        raise RuntimeError("no frames to save")
    pil_frames = [Image.fromarray(f, mode="RGB").convert("P", palette=Image.ADAPTIVE) for f in frames]
    duration_ms = int(round(1000 / max(1, fps)))
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument(
        "--date",
        type=str,
        default=_DEFAULT_DATE,
        help=f"Local-day to animate, format YYYY-MM-DD (default: {_DEFAULT_DATE}).",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(_DEFAULT_CHECKPOINT),
        help=f"Trained ViT checkpoint (default: {_DEFAULT_CHECKPOINT}).",
    )
    p.add_argument(
        "--archive-name",
        type=str,
        default=_DEFAULT_ARCHIVE_NAME,
        help="Archive folder name (metadata only).",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default=_DEFAULT_RUN_NAME,
        help="Run name inside the archive (metadata only).",
    )
    p.add_argument(
        "--checkpoint-extra",
        type=str,
        default=None,
        help=(
            "Optional second checkpoint. When set, a third (blue) line is "
            "overlaid on the right panel using this model's prediction. The "
            "second model's own ``zero_sky`` flag is used for its sky-input "
            "preparation (so a sky-arm and a no-sky-arm can be compared "
            "head-to-head)."
        ),
    )
    p.add_argument(
        "--archive-name-extra",
        type=str,
        default=None,
        help="Archive folder name for the second checkpoint (metadata only).",
    )
    p.add_argument(
        "--run-name-extra",
        type=str,
        default=None,
        help="Run name for the second checkpoint (metadata only).",
    )
    p.add_argument(
        "--label-main",
        type=str,
        default=None,
        help=(
            "Legend label for the main prediction line. Defaults to "
            "'GHI + full sky' when --checkpoint-extra is set, else 'Point pred.'."
        ),
    )
    p.add_argument(
        "--label-extra",
        type=str,
        default="GHI only (zero sky)",
        help=(
            "Legend label for the second (extra) prediction line "
            "(default: 'GHI only (zero sky)')."
        ),
    )
    p.add_argument(
        "--checkpoint-third",
        type=str,
        default=None,
        help=(
            "Optional third checkpoint. When set, a fourth (green) line is "
            "overlaid on the right panel. The third model's own ``zero_sky`` "
            "flag is used for its sky-input preparation."
        ),
    )
    p.add_argument(
        "--archive-name-third",
        type=str,
        default=None,
        help="Archive folder name for the third checkpoint (metadata only).",
    )
    p.add_argument(
        "--run-name-third",
        type=str,
        default=None,
        help="Run name for the third checkpoint (metadata only).",
    )
    p.add_argument(
        "--label-third",
        type=str,
        default="Third model",
        help=(
            "Legend label for the third prediction line "
            "(default: 'Third model')."
        ),
    )
    p.add_argument(
        "--ray-map-third",
        dest="ray_map_third",
        action="store_true",
        default=None,
        help="Build the third-model dataset with fisheye ray_map sky channels.",
    )
    p.add_argument(
        "--sun-mask-third",
        dest="sun_mask_third",
        action="store_true",
        default=None,
        help="Build the third-model dataset with per-frame sun_mask sky channel.",
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
        help="Sky-disc gating mode for the third-model dataset (matches training).",
    )
    p.add_argument(
        "--label-gt",
        type=str,
        default="GT",
        help="Legend label for the ground-truth line (default: 'GT').",
    )
    p.add_argument(
        "--stride-min",
        type=int,
        default=15,
        help=(
            "Display-grid cadence in minutes (default: 15). One model forward "
            "per display step; the anchor lag is fixed at 15 min by the model "
            "output_interval and is independent of this stride. Native CSV "
            "cadence is 1 min, so any integer >= 1 works. When != 15, output "
            "filenames carry a `_stride{N}` suffix to avoid colliding with "
            "existing 15-min outputs."
        ),
    )
    p.add_argument(
        "--legacy-pre-518dca9",
        action="store_true",
        help=(
            "Use legacy (pre-518dca9, May-26 era) Folsom kt/p_cs/p_mean "
            "normalization for inference. Required for checkpoints trained "
            "before commit 518dca9 (e.g. archive "
            "folsom_kt_sky_vs_nosky_40ep_4runs_2026-05-26 @ commit 24b9772). "
            "Without this flag the May-26 model sees inputs ~5x larger than "
            "training distribution and underpredicts by ~3x."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(_PROJECT_ROOT / "playground" / "animations"),
        help="Output base directory (default: playground/animations).",
    )
    p.add_argument(
        "--run-label",
        type=str,
        default=None,
        help=(
            "Subfolder under --out-dir (default: same as --date). Pass an "
            "explicit label when you want multiple variants for one date to "
            "coexist without overwriting."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run inference even if a cached <date>_predictions.npz exists.",
    )
    p.add_argument(
        "--keep-frames",
        action="store_true",
        help="Also save per-frame PNGs to <out-dir>/<date>_frames/ for debugging.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size (default 8; one day is ~40 forwards).",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override torch device (e.g. cuda:0). Default uses CUDA_VISIBLE_DEVICES.",
    )
    p.add_argument(
        "--no-cuda",
        action="store_true",
        help="Force CPU inference (slow; for debugging only).",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=float(_FPS),
        help=(
            f"GIF playback frames-per-second (default: {_FPS}). Lower = "
            "slower / easier to follow. Float allowed (e.g. 3.33 for 50%% "
            "slower than the default of 5)."
        ),
    )
    p.add_argument(
        "--sky-zarr",
        type=str,
        default=str(_DEFAULT_SKY_ZARR),
        help=(
            "Folsom sky xarray Zarr used for BOTH the left-panel fisheye frames "
            "AND model inference (must expose time_utc + images). "
            f"Default: {_DEFAULT_SKY_ZARR}."
        ),
    )
    p.add_argument(
        "--dataset-config",
        type=str,
        default=_DEFAULT_DATASET_CONFIG,
        help=(
            "Dataset YAML under config/datasets/ (default: conf_folsom.yaml). "
            "Use conf_folsom_full.yaml for dates outside sample_250k (e.g. 2015)."
        ),
    )
    return p.parse_args()


_MAIN_COLOR = "#cc1f1f"
_EXTRA_COLOR = "#1f4ec8"
_THIRD_COLOR = "#1a8f4a"


def main() -> int:
    args = _parse_args()
    run_label = args.run_label or args.date
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
    # Suffix is empty when stride==15 to preserve historical filenames; non-15
    # strides append "_stride{N}" between the date stem and the extension/tag.
    stride_suffix = "" if stride_min == 15 else f"_stride{stride_min}"
    stem = f"{args.date}{stride_suffix}"
    npz_path = out_dir / f"{stem}_predictions.npz"
    meta_path = out_dir / f"{stem}_meta.json"
    gif_path = out_dir / f"{stem}.gif"
    frames_dir = out_dir / f"{stem}_frames"

    window = _local_day_window(args.date)
    print(
        f"[animate] date_local={args.date}  utc_window=[{window.utc_start}, {window.utc_end})"
    )

    import xarray as xr

    sky_zarr_path = Path(args.sky_zarr).expanduser().resolve()
    _assert_valid_sky_source(sky_zarr_path, role="Left-panel display")
    if not sky_zarr_path.is_dir():
        raise FileNotFoundError(
            f"sky zarr not found: {sky_zarr_path}  "
            f"(pass --sky-zarr to a local xarray Zarr with time_utc + images)"
        )
    print(f"[animate] sky_zarr (display + model): {sky_zarr_path}")
    sky_ds = xr.open_zarr(str(sky_zarr_path), consolidated=False)
    sky_times_raw = sky_ds["time_utc"].values

    grid_utc, first_sky, last_sky = _find_display_grid_utc(
        window, np.asarray(sky_times_raw), stride_min=stride_min
    )
    print(
        f"[animate] sky availability in window: first={first_sky}  last={last_sky}  "
        f"frames={len(grid_utc)} ({stride_min}-min stride)"
    )

    pred_extra_kw: np.ndarray | None = None
    pred_third_kw: np.ndarray | None = None

    if npz_path.is_file() and not args.force:
        print(f"[animate] using cached predictions: {npz_path}")
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
        # Rebuild display grid from cache so a stale --date doesn't lie.
        grid_utc = pd.DatetimeIndex(pd.to_datetime(t_utc_arr))
        if has_extra and pred_extra_kw is None:
            raise RuntimeError(
                f"--checkpoint-extra was passed but cached NPZ {npz_path} only "
                f"holds a single prediction. Re-run with --force to recompute."
            )
        if has_third and pred_third_kw is None:
            raise RuntimeError(
                f"--checkpoint-third was passed but cached NPZ {npz_path} lacks "
                f"pred_third_kw. Re-run with --force to recompute."
            )
    else:
        import torch

        if args.no_cuda:
            device = torch.device("cpu")
        elif args.device:
            device = torch.device(args.device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[animate] device={device}")

        ds, model_sky_meta = _build_folsom_dataset_for_inference(
            _FORCE_PV_OUTPUT_LEN,
            sky_zarr=sky_zarr_path,
            dataset_config=str(args.dataset_config),
        )
        print(
            f"[animate] model skyimg_dir={model_sky_meta['model_sky_zarr']}  "
            f"format={model_sky_meta['model_sky_format']}"
        )
        if int(ds.pv_output_len) != _FORCE_PV_OUTPUT_LEN:
            raise RuntimeError(
                f"dataset pv_output_len override failed: got {ds.pv_output_len}, "
                f"expected {_FORCE_PV_OUTPUT_LEN}"
            )

        anchors_t0: list[pd.Timestamp] = []
        anchors_rows: list[int] = []
        for t_display in grid_utc:
            t0 = pd.Timestamp(t_display) - pd.Timedelta(minutes=15)
            row = _resolve_anchor_row_for_t0(ds, t0)
            anchors_t0.append(t0)
            anchors_rows.append(row)
        print(
            f"[animate] resolved {len(anchors_rows)} anchors (first row {anchors_rows[0]}, "
            f"last row {anchors_rows[-1]})"
        )

        print(
            f"[animate] forwarding main checkpoint: {args.checkpoint} "
            f"(legacy_pre_518dca9={bool(args.legacy_pre_518dca9)})"
        )
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
            print(
                f"[animate] forwarding extra checkpoint: {args.checkpoint_extra} "
                f"(legacy_pre_518dca9={bool(args.legacy_pre_518dca9)})"
            )
            pred_extra_kw, gt_kw_extra, cz_extra, run_meta_extra = _run_model_on_anchors(
                ds,
                anchors_rows,
                checkpoint_path=Path(args.checkpoint_extra).expanduser().resolve(),
                device=device,
                batch_size=int(args.batch_size),
                legacy_pre_518dca9=bool(args.legacy_pre_518dca9),
            )
            # Sanity: GT and cos_zenith are model-independent so they must match.
            if not np.allclose(gt_kw, gt_kw_extra, equal_nan=True):
                raise RuntimeError(
                    "GT mismatch between main and extra forwards (should be identical)."
                )
            if not np.allclose(cz, cz_extra, equal_nan=True):
                raise RuntimeError(
                    "cos_zenith mismatch between main and extra forwards (should be identical)."
                )

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
                        f"built dataset has {ds_third.sky_in_channels}. Pass matching "
                        f"--ray-map-third / --sun-mask-third / --sky-disc-mask-third."
                    )
                print(
                    f"[animate] third-model dataset sky_channels="
                    f"{list(getattr(ds_third, 'sky_channels', ('rgb',)))}  "
                    f"sky_in_channels={ds_third.sky_in_channels}"
                )
            print(
                f"[animate] forwarding third checkpoint: {ckpt_third_path} "
                f"(legacy_pre_518dca9={bool(args.legacy_pre_518dca9)})"
            )
            pred_third_kw, gt_kw_third, cz_third, run_meta_third = _run_model_on_anchors(
                ds_third,
                anchors_rows,
                checkpoint_path=ckpt_third_path,
                device=device,
                batch_size=int(args.batch_size),
                legacy_pre_518dca9=bool(args.legacy_pre_518dca9),
            )
            if not np.allclose(gt_kw, gt_kw_third, equal_nan=True):
                raise RuntimeError(
                    "GT mismatch between main and third forwards (should be identical)."
                )
            if not np.allclose(cz, cz_third, equal_nan=True):
                raise RuntimeError(
                    "cos_zenith mismatch between main and third forwards (should be identical)."
                )

        t_utc_arr = np.asarray([pd.Timestamp(t).to_datetime64() for t in grid_utc], dtype="datetime64[s]")
        t_local_arr = np.asarray(
            [
                pd.Timestamp(t).tz_localize("UTC").tz_convert(_LOCAL_TZ).tz_localize(None).to_datetime64()
                for t in grid_utc
            ],
            dtype="datetime64[s]",
        )
        if has_extra:
            npz_payload = dict(
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
        print(f"[animate] cached predictions -> {npz_path}")
        print(f"[animate] meta -> {meta_path}")

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

    title_right = f"Our approach (1-step ViT nowcast) — {args.date}"

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
        img = _read_sky_image_for_time(sky_ds, pd.Timestamp(t))
        frame = _render_one_frame(
            t_utc=pd.Timestamp(t),
            hours_local=hours_local,
            gt_kw=gt_kw,
            preds=preds_for_render,
            cursor_idx=i,
            sky_img=img,
            date_local_str=args.date,
            ymin=ymin,
            ymax=ymax,
            xlim=xlim,
            title_right=title_right,
            gt_label=label_gt,
        )
        frames.append(frame)
        if args.keep_frames:
            from PIL import Image

            Image.fromarray(frame).save(frames_dir / f"frame_{i:03d}.png")
        if (i + 1) % 10 == 0 or i == 0 or i == len(grid_utc) - 1:
            print(f"[animate] rendered frame {i + 1}/{len(grid_utc)}")

    _save_gif(frames, gif_path, fps=float(args.fps))
    gif_size_mb = gif_path.stat().st_size / (1024 * 1024)

    print("\n[animate] === SUMMARY ===")
    print(f"  date_local      : {args.date}")
    print(f"  run_label       : {run_label}")
    print(f"  two_model       : {has_extra}")
    print(f"  three_model     : {has_third}")
    if has_extra:
        m = meta.get("main", {})
        e = meta.get("extra", {})
        print(f"  main.archive    : {m.get('archive_name')}")
        print(f"  main.run        : {m.get('run_name')}")
        print(f"  main.checkpoint : {m.get('checkpoint')}")
        print(
            f"  main.zero_sky   : {m.get('zero_sky')}  "
            f"missing={m.get('missing_keys_count')}  unexpected={m.get('unexpected_keys_count')}"
        )
        print(f"  extra.archive   : {e.get('archive_name')}")
        print(f"  extra.run       : {e.get('run_name')}")
        print(f"  extra.checkpoint: {e.get('checkpoint')}")
        print(
            f"  extra.zero_sky  : {e.get('zero_sky')}  "
            f"missing={e.get('missing_keys_count')}  unexpected={e.get('unexpected_keys_count')}"
        )
    elif has_third:
        print(f"  archive_name    : {meta.get('archive_name')}")
        print(f"  run_name        : {meta.get('run_name')}")
        print(f"  checkpoint      : {meta.get('checkpoint')}")
        print(f"  zero_sky        : {meta.get('zero_sky')}")
    else:
        print(f"  archive_name    : {meta.get('archive_name')}")
        print(f"  run_name        : {meta.get('run_name')}")
        print(f"  checkpoint      : {meta.get('checkpoint')}")
        print(f"  zero_sky        : {meta.get('zero_sky')}")
    if has_third:
        t = meta.get("third", {})
        print(f"  third.archive   : {t.get('archive_name')}")
        print(f"  third.run       : {t.get('run_name')}")
        print(f"  third.checkpoint: {t.get('checkpoint')}")
        print(
            f"  third.zero_sky  : {t.get('zero_sky')}  "
            f"missing={t.get('missing_keys_count')}  unexpected={t.get('unexpected_keys_count')}"
        )
    print(f"  horizon step    : {meta.get('horizon_step_used')} (= +15 min from anchor)")
    print(f"  pv_output_len   : {meta.get('pv_output_len_at_inference')}")
    print(f"  frames          : {len(frames)} @ {args.fps:g} fps")
    print(f"  gif             : {gif_path}  ({gif_size_mb:.2f} MB)")
    print(f"  npz             : {npz_path}")
    print(f"  peak_gt_kw      : {float(np.nanmax(gt_kw)):.2f}")
    print(f"  peak_pred_main  : {float(np.nanmax(pred_kw)):.2f}")
    if pred_extra_kw is not None:
        print(f"  peak_pred_extra : {float(np.nanmax(pred_extra_kw)):.2f}")
    if pred_third_kw is not None:
        print(f"  peak_pred_third : {float(np.nanmax(pred_third_kw)):.2f}")
    if args.keep_frames:
        print(f"  frames_dir      : {frames_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
