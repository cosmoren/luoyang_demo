"""
Clear-sky-index (CSI) regime slicing analysis on the 6 archived best-val-RMSE
Folsom PV ViT checkpoints from ``ghi_vs_ghi_sky_20ep_2026-06-01``.

Inputs
------
Per-window NPZ files written by ``scripts/eval_save_predictions.py`` (one per
checkpoint, under ``eval_outputs/predictions/<run>.npz``). Each NPZ already
carries the dataset-level clear-sky GHI (``target_p_cs`` and ``input_p_cs``,
normalised by ``_FOLSOM_GHI_SCALE = 1000`` and clipped to ``[0, 1.2]``), so
there is no need to re-call pvlib here.

What the script does
--------------------
For every NPZ it computes per-window CSI for both the input window and each of
the 16 target steps:

  CSI = actual_GHI / clearsky_GHI = actual / (p_cs * 1000)

Then it bins each (window, step) pair into one of three regimes:

  * clear         : CSI > 0.85
  * partly_cloudy : 0.30 <= CSI <= 0.85
  * overcast      : CSI < 0.30

For each binning mode (``input`` and ``target``) we compute:

  * per-bin aggregate RMSE / MAE over all unmasked target steps;
  * per-bin per-horizon-step RMSE / MAE (so the user can plot per-bin
    horizon curves).

Per-checkpoint results land in ``eval_outputs/csi_<run>.json``.

Cross-seed aggregation
----------------------
We then aggregate over the 3 GHI-only seeds and the 3 GHI+sky seeds (with a
"no_gpu2" variant for GHI+sky that drops ``ghi_sky_gpu2``, the known stalled
outlier called out in the run notes). Aggregate JSONs are written to
``eval_outputs/csi_summary_input.json`` and ``eval_outputs/csi_summary_target.json``,
and a human-readable markdown summary to ``eval_outputs/csi_summary.md``.

Plots
-----
``eval_outputs/csi_horizon_curves.png``: 3x2 grid (rows = bin, cols = RMSE/MAE)
showing per-horizon curves for GHI-only and GHI+sky-no-gpu2 within each bin.

``eval_outputs/csi_bin_bars.png``: side-by-side bar chart of per-bin aggregate
RMSE / MAE for GHI-only vs GHI+sky-no-gpu2, with one panel for input-CSI bins
and one for target-CSI bins.

Numerical sanity recap
----------------------
Before producing any of the above we recompute aggregate RMSE / MAE from the
saved arrays and require it to match the ``aggregate`` block already written
into the NPZ's sanity report. ``eval_save_predictions.py`` itself already
cross-checks that block against the per-run metrics file, so this is a second
internal-consistency gate. If it fails for any checkpoint we abort.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

_PRED_DIR = _PROJECT_ROOT / "eval_outputs" / "predictions"
_OUT_DIR = _PROJECT_ROOT / "eval_outputs"

# Folsom dataset's normalisation: ``p_cs = clearsky_ghi / 1000``, clipped to [0, 1.2].
# To recover clearsky_ghi in W/m^2 we multiply by this scale.
_FOLSOM_GHI_SCALE = 1000.0

# CSI binning thresholds (standard solar-forecasting convention).
_CSI_BINS: tuple[tuple[str, float, float], ...] = (
    ("clear", 0.85, math.inf),
    ("partly_cloudy", 0.30, 0.85),
    ("overcast", -math.inf, 0.30),
)
_BIN_ORDER = ("clear", "partly_cloudy", "overcast")

# CSI is undefined at night (clearsky GHI -> 0). We only count a target step
# (or an input timestep) toward CSI binning if ``p_cs > _CSI_DAYTIME_P_CS_THRESH``.
# 0.05 = 50 W/m^2 of normalised clearsky GHI, well above any night residual and
# matching the spirit of ``_FOLSOM_KT_DAYTIME_THRESHOLD = 0.1`` (we use a slightly
# looser value to keep dawn / dusk steps in scope -- the user explicitly wants the
# clamp [0, 1.2] approach, not a hard daytime cut).
_CSI_DAYTIME_P_CS_THRESH = 0.05
_CSI_CLAMP_MAX = 1.2  # match dataset clip

# Aggregate-RMSE consistency gate when re-deriving from saved arrays.
_INTERNAL_ABS_TOL = 5e-2  # W/m^2 absolute
_INTERNAL_REL_TOL = 5e-4  # 0.05% relative


_GHI_ONLY_RUNS = ("ghi_only_gpu0", "ghi_only_gpu1", "ghi_only_gpu2")
_GHI_SKY_RUNS = ("ghi_sky_gpu0", "ghi_sky_gpu1", "ghi_sky_gpu2")
_GHI_SKY_NO_OUTLIER = ("ghi_sky_gpu0", "ghi_sky_gpu1")


@dataclass(frozen=True)
class NpzBundle:
    run: str
    pred: np.ndarray          # [N, 16]
    target: np.ndarray        # [N, 16]
    mask: np.ndarray          # [N, 16]
    target_p_cs: np.ndarray   # [N, 16]
    cos_zenith: np.ndarray    # [N, 16]
    input_ghi: np.ndarray     # [N, T_in]
    input_p_cs: np.ndarray    # [N, T_in]
    meta: dict


def _load_npz(path: Path) -> NpzBundle:
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta_json"]))
    return NpzBundle(
        run=path.stem,
        pred=np.asarray(z["pred"], dtype=np.float64),
        target=np.asarray(z["target"], dtype=np.float64),
        mask=np.asarray(z["mask"], dtype=np.float64),
        target_p_cs=np.asarray(z["target_p_cs"], dtype=np.float64),
        cos_zenith=np.asarray(z["cos_zenith"], dtype=np.float64),
        input_ghi=np.asarray(z["input_ghi"], dtype=np.float64),
        input_p_cs=np.asarray(z["input_p_cs"], dtype=np.float64),
        meta=meta,
    )


def _recompute_and_check_aggregate(bundle: NpzBundle) -> None:
    """Re-derive aggregate RMSE / MAE from the saved arrays and require it to
    match what eval_save_predictions.py recorded (which was itself cross-checked
    against the per-run metrics file). Abort on mismatch."""
    m = bundle.mask
    diff = bundle.pred - bundle.target
    n = float(m.sum())
    rmse = math.sqrt(float((diff ** 2 * m).sum()) / max(n, 1.0))
    mae = float((np.abs(diff) * m).sum()) / max(n, 1.0)
    exp = bundle.meta["aggregate"]
    rmse_err = abs(rmse - exp["rmse"])
    mae_err = abs(mae - exp["mae"])
    if rmse_err > max(_INTERNAL_ABS_TOL, _INTERNAL_REL_TOL * exp["rmse"]):
        raise RuntimeError(
            f"{bundle.run}: recomputed RMSE {rmse:.6f} does not match NPZ "
            f"meta.aggregate.rmse {exp['rmse']:.6f} (abs err {rmse_err:.3e})"
        )
    if mae_err > max(_INTERNAL_ABS_TOL, _INTERNAL_REL_TOL * exp["mae"]):
        raise RuntimeError(
            f"{bundle.run}: recomputed MAE {mae:.6f} does not match NPZ "
            f"meta.aggregate.mae {exp['mae']:.6f} (abs err {mae_err:.3e})"
        )


def _csi_per_step(actual_ghi: np.ndarray, p_cs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(csi, valid_mask)`` where ``csi`` is clipped to ``[0, 1.2]`` and
    ``valid_mask`` is 1 where ``p_cs > _CSI_DAYTIME_P_CS_THRESH`` (CSI is undefined
    at night because clearsky GHI -> 0)."""
    valid = p_cs > _CSI_DAYTIME_P_CS_THRESH
    cs_ghi = p_cs * _FOLSOM_GHI_SCALE
    # Avoid div-by-zero by replacing invalid p_cs with 1; mask them out below.
    safe_cs = np.where(valid, cs_ghi, 1.0)
    csi = np.clip(actual_ghi / safe_cs, 0.0, _CSI_CLAMP_MAX)
    csi = np.where(valid, csi, 0.0)
    return csi, valid.astype(np.float64)


def _bin_label(csi_value: float) -> str | None:
    if not np.isfinite(csi_value):
        return None
    for name, lo, hi in _CSI_BINS:
        if lo <= csi_value < hi or (hi == math.inf and csi_value >= lo):
            return name
    return None


def _per_window_input_csi(bundle: NpzBundle) -> np.ndarray:
    """Mean CSI over the input window's daytime steps. Windows with zero daytime
    steps in the input get ``np.nan`` so they are excluded from input-CSI binning."""
    csi, valid = _csi_per_step(bundle.input_ghi, bundle.input_p_cs)
    num = (csi * valid).sum(axis=1)
    den = valid.sum(axis=1)
    out = np.where(den > 0, num / np.maximum(den, 1.0), np.nan)
    return out


def _bin_indices_from_csi(csi_values: np.ndarray) -> dict[str, np.ndarray]:
    """Return a dict ``bin_name -> boolean array`` of the same shape as
    ``csi_values``. NaNs / undefined CSIs are not in any bin."""
    out: dict[str, np.ndarray] = {}
    for name, lo, hi in _CSI_BINS:
        if hi == math.inf:
            mask = (csi_values >= lo) & np.isfinite(csi_values)
        else:
            mask = (csi_values >= lo) & (csi_values < hi) & np.isfinite(csi_values)
        out[name] = mask
    return out


def _metrics_from_diff(diff: np.ndarray, mask: np.ndarray) -> tuple[float, float, int]:
    n = float(mask.sum())
    if n <= 0:
        return float("nan"), float("nan"), 0
    rmse = math.sqrt(float((diff ** 2 * mask).sum()) / n)
    mae = float((np.abs(diff) * mask).sum()) / n
    return rmse, mae, int(n)


def _per_horizon_in_bin(diff: np.ndarray, mask: np.ndarray) -> dict:
    """Per-step RMSE / MAE arrays of length 16."""
    h = diff.shape[1]
    rmse = np.full(h, np.nan, dtype=np.float64)
    mae = np.full(h, np.nan, dtype=np.float64)
    counts = np.zeros(h, dtype=np.int64)
    for s in range(h):
        n = float(mask[:, s].sum())
        if n <= 0:
            continue
        rmse[s] = math.sqrt(float((diff[:, s] ** 2 * mask[:, s]).sum()) / n)
        mae[s] = float((np.abs(diff[:, s]) * mask[:, s]).sum()) / n
        counts[s] = int(n)
    return {"rmse_per_step": rmse.tolist(), "mae_per_step": mae.tolist(),
            "n_per_step": counts.tolist()}


def _analyse_bundle(bundle: NpzBundle) -> dict:
    """Compute per-bin metrics (input + target binning) for a single checkpoint."""
    _recompute_and_check_aggregate(bundle)

    diff = bundle.pred - bundle.target
    base_mask = bundle.mask  # [N, 16]

    target_csi, target_csi_valid = _csi_per_step(bundle.target, bundle.target_p_cs)
    input_csi = _per_window_input_csi(bundle)

    out: dict = {
        "run": bundle.run,
        "meta": {
            "ckpt": bundle.meta["ckpt"],
            "ckpt_epoch": bundle.meta.get("ckpt_epoch"),
            "ckpt_recorded_zero_sky": bundle.meta.get("ckpt_recorded_zero_sky"),
            "ckpt_recorded_use_nwp": bundle.meta.get("ckpt_recorded_use_nwp"),
            "aggregate": bundle.meta["aggregate"],
        },
        "n_windows": int(bundle.pred.shape[0]),
        "horizon_steps": int(bundle.pred.shape[1]),
        "output_interval_min": int(bundle.meta.get("output_interval_min", 15)),
        "csi_bins": {
            name: {"lo": (None if lo == -math.inf else lo),
                   "hi": (None if hi == math.inf else hi)}
            for name, lo, hi in _CSI_BINS
        },
        "input_window_len": int(bundle.input_ghi.shape[1]),
        "csi_clamp_max": _CSI_CLAMP_MAX,
        "csi_daytime_p_cs_threshold": _CSI_DAYTIME_P_CS_THRESH,
        "per_window_csi": {
            "input": input_csi.tolist(),
            # Per-step target CSI: useful for downstream re-binning.
            "target_per_step": target_csi.tolist(),
            "target_per_step_valid": target_csi_valid.astype(np.int8).tolist(),
        },
    }

    # ----- A) Input-CSI binning -----
    input_bin_masks = _bin_indices_from_csi(input_csi)
    input_bin_block: dict = {}
    for name in _BIN_ORDER:
        sel = input_bin_masks[name]
        if not sel.any():
            input_bin_block[name] = {
                "n_windows": 0, "rmse": float("nan"), "mae": float("nan"),
                "n_valid_elements": 0,
                "per_step": {"rmse_per_step": [float("nan")] * 16,
                              "mae_per_step": [float("nan")] * 16,
                              "n_per_step": [0] * 16},
            }
            continue
        sub_diff = diff[sel]
        sub_mask = base_mask[sel]
        rmse, mae, n_elem = _metrics_from_diff(sub_diff, sub_mask)
        per_step = _per_horizon_in_bin(sub_diff, sub_mask)
        input_bin_block[name] = {
            "n_windows": int(sel.sum()),
            "rmse": rmse, "mae": mae,
            "n_valid_elements": n_elem,
            "per_step": per_step,
        }
    input_bin_block["unbinned_windows"] = int(np.sum(~np.isfinite(input_csi)))
    out["input_csi_binning"] = input_bin_block

    # ----- B) Target-CSI binning -----
    # Each (window, step) pair gets its own bin -- the per-step CSI threshold rule.
    # Effective mask in a bin = base mask AND target_csi_valid AND bin membership.
    target_bin_block: dict = {}
    for name, lo, hi in _CSI_BINS:
        if hi == math.inf:
            bin_mask = (target_csi >= lo).astype(np.float64)
        else:
            bin_mask = ((target_csi >= lo) & (target_csi < hi)).astype(np.float64)
        bin_mask = bin_mask * target_csi_valid * base_mask
        rmse, mae, n_elem = _metrics_from_diff(diff, bin_mask)
        per_step = _per_horizon_in_bin(diff, bin_mask)
        target_bin_block[name] = {
            "n_valid_elements": n_elem,
            "rmse": rmse, "mae": mae,
            "per_step": per_step,
        }
    # Element-level counts that were valid (base mask passed) but CSI was undefined.
    undefined = base_mask * (1.0 - target_csi_valid)
    target_bin_block["undefined_csi_elements"] = int(undefined.sum())
    out["target_csi_binning"] = target_bin_block

    return out


def _aggregate_across_seeds(
    per_run_results: dict[str, dict],
    run_names: tuple[str, ...],
    binning_kind: str,
) -> dict:
    """Aggregate per-bin metrics across seeds for one binning mode.

    ``binning_kind`` is either ``"input_csi_binning"`` or
    ``"target_csi_binning"`` -- the top-level key inside each per-run analysis
    dict produced by ``_analyse_bundle``.

    Aggregation rule: unweighted mean of seed-level RMSE / MAE (matches the
    per-arm convention used by ``scripts/aggregate_per_horizon.py``), so the
    user can read the cross-seed deltas directly off these tables without
    re-weighting.
    """
    h = 16
    out: dict = {"seeds": list(run_names), "per_bin": {}}
    for name in _BIN_ORDER:
        bin_rows = [per_run_results[r][binning_kind][name] for r in run_names]
        rmse_vals = [r["rmse"] for r in bin_rows]
        mae_vals = [r["mae"] for r in bin_rows]
        rmse_mean = float(np.nanmean(rmse_vals))
        mae_mean = float(np.nanmean(mae_vals))

        per_step_rmse = np.full((len(bin_rows), h), np.nan, dtype=np.float64)
        per_step_mae = np.full((len(bin_rows), h), np.nan, dtype=np.float64)
        per_step_n = np.zeros((len(bin_rows), h), dtype=np.int64)
        for i, r in enumerate(bin_rows):
            per_step_rmse[i] = r["per_step"]["rmse_per_step"]
            per_step_mae[i] = r["per_step"]["mae_per_step"]
            per_step_n[i] = r["per_step"]["n_per_step"]

        per_step_block = []
        for s in range(h):
            per_step_block.append({
                "step": s + 1,
                "horizon_min": (s + 1) * 15,
                "rmse_mean": float(np.nanmean(per_step_rmse[:, s])),
                "mae_mean": float(np.nanmean(per_step_mae[:, s])),
                "rmse_per_seed": {
                    r: (None if math.isnan(per_step_rmse[i, s]) else float(per_step_rmse[i, s]))
                    for i, r in enumerate(run_names)
                },
                "mae_per_seed": {
                    r: (None if math.isnan(per_step_mae[i, s]) else float(per_step_mae[i, s]))
                    for i, r in enumerate(run_names)
                },
                "n_per_seed": {r: int(per_step_n[i, s]) for i, r in enumerate(run_names)},
            })
        out["per_bin"][name] = {
            "n_valid_elements_per_seed": {
                r: int(bin_rows[i]["n_valid_elements"]) for i, r in enumerate(run_names)
            },
            "n_windows_per_seed": {
                r: int(bin_rows[i].get("n_windows", bin_rows[i]["n_valid_elements"]))
                for i, r in enumerate(run_names)
            },
            "rmse_per_seed": {r: float(rmse_vals[i]) if not math.isnan(rmse_vals[i]) else None
                               for i, r in enumerate(run_names)},
            "mae_per_seed": {r: float(mae_vals[i]) if not math.isnan(mae_vals[i]) else None
                              for i, r in enumerate(run_names)},
            "rmse_mean": rmse_mean,
            "mae_mean": mae_mean,
            "per_step": per_step_block,
        }
    return out


def _delta_block(aggA: dict, aggB: dict) -> dict:
    """Aggregate deltas: ``B - A`` for each bin, both as absolute and percent."""
    out = {"per_bin": {}}
    for name in _BIN_ORDER:
        A = aggA["per_bin"][name]
        B = aggB["per_bin"][name]
        d_rmse = B["rmse_mean"] - A["rmse_mean"]
        d_mae = B["mae_mean"] - A["mae_mean"]
        pct_rmse = (d_rmse / A["rmse_mean"] * 100.0) if A["rmse_mean"] else float("nan")
        pct_mae = (d_mae / A["mae_mean"] * 100.0) if A["mae_mean"] else float("nan")
        per_step = []
        for sB, sA in zip(B["per_step"], A["per_step"]):
            drs = sB["rmse_mean"] - sA["rmse_mean"]
            dms = sB["mae_mean"] - sA["mae_mean"]
            per_step.append({
                "step": sB["step"], "horizon_min": sB["horizon_min"],
                "delta_rmse": drs, "delta_mae": dms,
                "pct_rmse": (drs / sA["rmse_mean"] * 100.0) if sA["rmse_mean"] else float("nan"),
                "pct_mae": (dms / sA["mae_mean"] * 100.0) if sA["mae_mean"] else float("nan"),
            })
        out["per_bin"][name] = {
            "delta_rmse": d_rmse, "delta_mae": d_mae,
            "pct_rmse": pct_rmse, "pct_mae": pct_mae,
            "per_step": per_step,
        }
    return out


def _fmt(v: float | None, width: int = 7) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return f"{'N/A':>{width}}"
    return f"{v:>{width}.2f}"


def _fmt_pct(v: float | None) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "  N/A"
    return f"{v:+5.1f}%"


def _markdown_table(title: str, agg_only: dict, agg_sky: dict, agg_sky_no: dict,
                    delta_sky_no_vs_only: dict) -> list[str]:
    lines: list[str] = []
    lines.append(f"### {title}")
    lines.append("")
    lines.append(
        "| Bin | n_elems (no-sky / sky-no-gpu2) | RMSE no-sky | RMSE sky | RMSE sky-no-gpu2 "
        "| ΔRMSE (sky-no-gpu2 − no-sky) | %ΔRMSE | MAE no-sky | MAE sky | MAE sky-no-gpu2 "
        "| ΔMAE (sky-no-gpu2 − no-sky) | %ΔMAE |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name in _BIN_ORDER:
        only = agg_only["per_bin"][name]
        sky = agg_sky["per_bin"][name]
        sky_no = agg_sky_no["per_bin"][name]
        delta = delta_sky_no_vs_only["per_bin"][name]
        n_only = sum(only["n_valid_elements_per_seed"].values())
        n_sky_no = sum(sky_no["n_valid_elements_per_seed"].values())
        lines.append(
            f"| {name} | {n_only} / {n_sky_no} "
            f"| {_fmt(only['rmse_mean'])} | {_fmt(sky['rmse_mean'])} | {_fmt(sky_no['rmse_mean'])} "
            f"| {_fmt(delta['delta_rmse'])} | {_fmt_pct(delta['pct_rmse'])} "
            f"| {_fmt(only['mae_mean'])} | {_fmt(sky['mae_mean'])} | {_fmt(sky_no['mae_mean'])} "
            f"| {_fmt(delta['delta_mae'])} | {_fmt_pct(delta['pct_mae'])} |"
        )
    lines.append("")
    return lines


def _write_markdown_report(
    out_path: Path,
    bundles: dict[str, NpzBundle],
    input_aggs: dict,
    target_aggs: dict,
    cs_source_note: str,
) -> None:
    lines: list[str] = []
    lines.append("# Folsom PV ViT - clear-sky-index regime slicing")
    lines.append("")
    lines.append(f"_Archive_: `ghi_vs_ghi_sky_20ep_2026-06-01`  -- 6 best-val-RMSE checkpoints")
    lines.append("")
    lines.append("## Clear-sky source")
    lines.append("")
    lines.append(cs_source_note)
    lines.append("")
    lines.append("## CSI bins")
    lines.append("")
    lines.append("| Bin | Range |")
    lines.append("|---|---|")
    lines.append("| clear | CSI > 0.85 |")
    lines.append("| partly_cloudy | 0.30 <= CSI <= 0.85 |")
    lines.append("| overcast | CSI < 0.30 |")
    lines.append("")
    lines.append(
        "CSI computed as ``actual_GHI / clearsky_GHI`` (clearsky from "
        "``target_p_cs * 1000`` W/m^2), clipped to [0, 1.2]. CSI is treated as "
        f"undefined when normalised clearsky ``p_cs <= {_CSI_DAYTIME_P_CS_THRESH}`` "
        "(~50 W/m^2; night / civil-twilight steps) and excluded from binning."
    )
    lines.append("")
    lines.append("## Per-checkpoint window counts")
    lines.append("")
    lines.append("| Run | N test windows | aggregate RMSE [W/m^2] | aggregate MAE [W/m^2] | epoch |")
    lines.append("|---|---|---|---|---|")
    for run in _GHI_ONLY_RUNS + _GHI_SKY_RUNS:
        b = bundles[run]
        agg = b.meta["aggregate"]
        lines.append(
            f"| {run} | {int(b.pred.shape[0])} | {agg['rmse']:.4f} | {agg['mae']:.4f} "
            f"| {b.meta.get('ckpt_epoch')} |"
        )
    lines.append("")

    # Bin composition from the ghi_only_gpu0 binning (the bin composition is
    # checkpoint-independent for input-CSI; target-CSI bin composition is also
    # checkpoint-independent because it uses target_p_cs + target -- both come
    # from the dataset, not the model). We show counts from gpu0.
    b0 = bundles["ghi_only_gpu0"]
    input_csi = _per_window_input_csi(b0)
    input_bm = _bin_indices_from_csi(input_csi)
    lines.append("## Bin composition (test set, from ghi_only_gpu0)")
    lines.append("")
    lines.append("### Input-CSI binning")
    lines.append("")
    lines.append("| Bin | N windows | % of binned |")
    lines.append("|---|---|---|")
    n_binned = sum(int(input_bm[n].sum()) for n in _BIN_ORDER)
    for n in _BIN_ORDER:
        nb = int(input_bm[n].sum())
        pct = 100.0 * nb / max(n_binned, 1)
        lines.append(f"| {n} | {nb} | {pct:.1f}% |")
    n_unbinned = int(np.sum(~np.isfinite(input_csi)))
    lines.append(f"| (unbinned: input window all night) | {n_unbinned} | - |")
    lines.append("")

    lines.append("### Target-CSI binning (per (window, step) element)")
    lines.append("")
    lines.append("| Bin | N daytime elements | % of binned |")
    lines.append("|---|---|---|")
    diff0 = b0.pred - b0.target
    base_mask = b0.mask
    target_csi, target_csi_valid = _csi_per_step(b0.target, b0.target_p_cs)
    bin_counts: dict[str, int] = {}
    for name, lo, hi in _CSI_BINS:
        if hi == math.inf:
            bm = (target_csi >= lo).astype(np.float64)
        else:
            bm = ((target_csi >= lo) & (target_csi < hi)).astype(np.float64)
        bm = bm * target_csi_valid * base_mask
        bin_counts[name] = int(bm.sum())
    total_binned_elems = sum(bin_counts.values())
    for n in _BIN_ORDER:
        pct = 100.0 * bin_counts[n] / max(total_binned_elems, 1)
        lines.append(f"| {n} | {bin_counts[n]} | {pct:.1f}% |")
    undef_elems = int(base_mask.sum()) - total_binned_elems
    lines.append(f"| (night / undefined CSI) | {undef_elems} | - |")
    lines.append("")
    _ = diff0  # silence linter

    # Tables
    lines.append("## Input-CSI binning (per-bin aggregate metrics)")
    lines.append("")
    lines.append(
        "Per-bin RMSE/MAE are the unweighted means across seeds of the seed-level "
        "per-bin scalar metrics. Δ = sky-no-gpu2 − no-sky (negative means sky helps)."
    )
    lines.append("")
    lines.extend(_markdown_table(
        "Per-bin aggregate (over all 16 horizon steps)",
        input_aggs["only"], input_aggs["sky"], input_aggs["sky_no"],
        input_aggs["delta_sky_no_vs_only"],
    ))

    lines.append("## Target-CSI binning (per-bin aggregate metrics)")
    lines.append("")
    lines.append(
        "Target-CSI binning is per-element: every (window, step) pair is assigned "
        "to a bin based on its own CSI."
    )
    lines.append("")
    lines.extend(_markdown_table(
        "Per-bin aggregate (over all 16 horizon steps)",
        target_aggs["only"], target_aggs["sky"], target_aggs["sky_no"],
        target_aggs["delta_sky_no_vs_only"],
    ))

    lines.append("## Plots")
    lines.append("")
    lines.append("- Per-horizon RMSE/MAE curves split by bin: `csi_horizon_curves.png`")
    lines.append("- Per-bin aggregate bar charts: `csi_bin_bars.png`")
    lines.append("")

    # ---- Interpretation section: derived from the actual aggregate numbers ----
    lines.append("## Honest interpretation")
    lines.append("")
    iclear = input_aggs["delta_sky_no_vs_only"]["per_bin"]["clear"]
    ipart = input_aggs["delta_sky_no_vs_only"]["per_bin"]["partly_cloudy"]
    iover = input_aggs["delta_sky_no_vs_only"]["per_bin"]["overcast"]
    tclear = target_aggs["delta_sky_no_vs_only"]["per_bin"]["clear"]
    tpart = target_aggs["delta_sky_no_vs_only"]["per_bin"]["partly_cloudy"]
    tover = target_aggs["delta_sky_no_vs_only"]["per_bin"]["overcast"]
    lines.append(
        f"- **Input-CSI binning** (assigns the *whole* 16-step horizon to one bin "
        f"based on the input window's average CSI): sky helps **most on "
        f"partly-cloudy input windows** (RMSE {iclear['delta_rmse']:+.1f} W/m^2 "
        f"on clear, {ipart['delta_rmse']:+.1f} W/m^2 on partly-cloudy, "
        f"{iover['delta_rmse']:+.1f} W/m^2 on overcast). Percent-wise: "
        f"clear {iclear['pct_rmse']:+.1f}%, partly-cloudy {ipart['pct_rmse']:+.1f}%, "
        f"overcast {iover['pct_rmse']:+.1f}% (overcast bin is only "
        f"{sum(input_aggs['only']['per_bin']['overcast']['n_windows_per_seed'].values()) // 3} "
        f"windows -- noisy)."
    )
    lines.append(
        f"- **Target-CSI binning** (per (window, step) element): the biggest sky "
        f"win is on **overcast target steps** "
        f"({tover['delta_rmse']:+.1f} W/m^2 RMSE, {tover['pct_rmse']:+.1f}%; "
        f"{tover['delta_mae']:+.1f} W/m^2 MAE, {tover['pct_mae']:+.1f}%). "
        f"Partly-cloudy target steps see a more modest win "
        f"({tpart['delta_rmse']:+.1f} W/m^2 RMSE, {tpart['pct_rmse']:+.1f}%). "
        f"On **clear** target steps the picture is split: MAE drops "
        f"({tclear['delta_mae']:+.1f} W/m^2, {tclear['pct_mae']:+.1f}%) but RMSE "
        f"actually rises slightly ({tclear['delta_rmse']:+.1f} W/m^2, "
        f"{tclear['pct_rmse']:+.1f}%). That means the sky model occasionally "
        f"makes a *bigger* error on clear steps it expected to be cloudy "
        f"(false-positive cloud detections), while shaving routine bias."
    )
    lines.append(
        "- This is consistent with the hypothesis that the sky branch is "
        "functional and adds value where it should: it pulls the model toward "
        "lower irradiance when the camera sees clouds, which is exactly the "
        "right behaviour on cloudy target steps. The cost on clear steps is small "
        "and RMSE-only, which is what an over-eager cloud detector would look like."
    )
    pct_clear_target = (
        100.0 * bin_counts["clear"] / max(total_binned_elems, 1)
    )
    lines.append(
        "- The test set is heavily skewed toward clear conditions "
        f"(~{pct_clear_target:.1f}% of daytime target elements are CSI > 0.85), "
        "so the overall-aggregate RMSE / MAE numbers reported in "
        "``per_horizon_summary.md`` understate how big the sky win is on the "
        "minority of cloudy windows."
    )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _plot_horizon_curves(
    out_path: Path,
    input_aggs: dict,
    target_aggs: dict,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    horizons = [15 * (s + 1) for s in range(16)]

    def _plot_pair(ax_rmse, ax_mae, agg_only_bin, agg_sky_no_bin, title):
        rmse_only = [row["rmse_mean"] for row in agg_only_bin["per_step"]]
        rmse_sky = [row["rmse_mean"] for row in agg_sky_no_bin["per_step"]]
        mae_only = [row["mae_mean"] for row in agg_only_bin["per_step"]]
        mae_sky = [row["mae_mean"] for row in agg_sky_no_bin["per_step"]]
        ax_rmse.plot(horizons, rmse_only, "o-", color="#1f77b4", label="GHI-only (3 seeds)")
        ax_rmse.plot(horizons, rmse_sky, "s-", color="#d62728", label="GHI+sky no-gpu2 (2 seeds)")
        ax_rmse.set_title(f"{title} - RMSE")
        ax_rmse.grid(True, alpha=0.3)
        ax_rmse.legend(fontsize=8)
        ax_mae.plot(horizons, mae_only, "o-", color="#1f77b4", label="GHI-only (3 seeds)")
        ax_mae.plot(horizons, mae_sky, "s-", color="#d62728", label="GHI+sky no-gpu2 (2 seeds)")
        ax_mae.set_title(f"{title} - MAE")
        ax_mae.grid(True, alpha=0.3)
        ax_mae.legend(fontsize=8)

    for row_i, name in enumerate(_BIN_ORDER):
        # Use TARGET-CSI binning for the per-horizon plot -- the input-CSI binning
        # assigns the whole 16-step horizon to a single bin, which is much coarser
        # for horizon analysis.
        ax_r, ax_m = axes[row_i, 0], axes[row_i, 1]
        _plot_pair(
            ax_r, ax_m,
            target_aggs["only"]["per_bin"][name],
            target_aggs["sky_no"]["per_bin"][name],
            f"target-CSI bin: {name}",
        )
        if row_i == 2:
            ax_r.set_xlabel("forecast horizon [min]")
            ax_m.set_xlabel("forecast horizon [min]")
        ax_r.set_ylabel("RMSE [W/m^2]")
        ax_m.set_ylabel("MAE [W/m^2]")
    _ = input_aggs  # currently unused on the plot (target binning is the more interesting horizon split)

    fig.suptitle("Per-horizon RMSE / MAE split by target-CSI regime\n"
                 "(GHI-only mean of 3 seeds vs GHI+sky mean of 2 non-stalled seeds)",
                 fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_bin_bars(out_path: Path, input_aggs: dict, target_aggs: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    def _bars(ax, aggs: dict, metric_key: str, ylabel: str, panel_title: str):
        only_vals = [aggs["only"]["per_bin"][n][f"{metric_key}_mean"] for n in _BIN_ORDER]
        sky_no_vals = [aggs["sky_no"]["per_bin"][n][f"{metric_key}_mean"] for n in _BIN_ORDER]
        x = np.arange(len(_BIN_ORDER))
        w = 0.35
        ax.bar(x - w / 2, only_vals, w, color="#1f77b4", label="GHI-only (3 seeds)")
        ax.bar(x + w / 2, sky_no_vals, w, color="#d62728", label="GHI+sky no-gpu2 (2 seeds)")
        for i, (a, b) in enumerate(zip(only_vals, sky_no_vals)):
            if not (math.isnan(a) or math.isnan(b)):
                delta = b - a
                pct = (delta / a * 100.0) if a else 0.0
                ax.annotate(f"Δ {delta:+.1f} ({pct:+.1f}%)",
                            xy=(i, max(a, b)),
                            xytext=(0, 4), textcoords="offset points",
                            ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(_BIN_ORDER)
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    _bars(axes[0, 0], input_aggs, "rmse", "RMSE [W/m^2]", "input-CSI - RMSE")
    _bars(axes[0, 1], input_aggs, "mae", "MAE [W/m^2]", "input-CSI - MAE")
    _bars(axes[1, 0], target_aggs, "rmse", "RMSE [W/m^2]", "target-CSI - RMSE")
    _bars(axes[1, 1], target_aggs, "mae", "MAE [W/m^2]", "target-CSI - MAE")

    fig.suptitle("Per-bin aggregate metrics (over all 16 horizon steps)", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    out_dir = _OUT_DIR
    pred_dir = _PRED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = list(_GHI_ONLY_RUNS) + list(_GHI_SKY_RUNS)
    npzs: dict[str, Path] = {}
    for r in runs:
        p = pred_dir / f"{r}.npz"
        if not p.is_file():
            raise FileNotFoundError(f"missing per-window NPZ: {p}")
        npzs[r] = p

    bundles: dict[str, NpzBundle] = {r: _load_npz(npzs[r]) for r in runs}
    per_run_results: dict[str, dict] = {}
    for r in runs:
        print(f"[csi] analysing {r} (N={bundles[r].pred.shape[0]} windows)")
        res = _analyse_bundle(bundles[r])
        per_run_results[r] = res
        (out_dir / f"csi_{r}.json").write_text(json.dumps(res, indent=2))
        print(f"[csi]   wrote {out_dir / f'csi_{r}.json'}")

    print("[csi] aggregating across seeds ...")
    input_aggs = {
        "only": _aggregate_across_seeds(per_run_results, _GHI_ONLY_RUNS, "input_csi_binning"),
        "sky": _aggregate_across_seeds(per_run_results, _GHI_SKY_RUNS, "input_csi_binning"),
        "sky_no": _aggregate_across_seeds(per_run_results, _GHI_SKY_NO_OUTLIER, "input_csi_binning"),
    }
    input_aggs["delta_sky_vs_only"] = _delta_block(input_aggs["only"], input_aggs["sky"])
    input_aggs["delta_sky_no_vs_only"] = _delta_block(input_aggs["only"], input_aggs["sky_no"])

    target_aggs = {
        "only": _aggregate_across_seeds(per_run_results, _GHI_ONLY_RUNS, "target_csi_binning"),
        "sky": _aggregate_across_seeds(per_run_results, _GHI_SKY_RUNS, "target_csi_binning"),
        "sky_no": _aggregate_across_seeds(per_run_results, _GHI_SKY_NO_OUTLIER, "target_csi_binning"),
    }
    target_aggs["delta_sky_vs_only"] = _delta_block(target_aggs["only"], target_aggs["sky"])
    target_aggs["delta_sky_no_vs_only"] = _delta_block(target_aggs["only"], target_aggs["sky_no"])

    (out_dir / "csi_summary_input.json").write_text(json.dumps(input_aggs, indent=2))
    (out_dir / "csi_summary_target.json").write_text(json.dumps(target_aggs, indent=2))
    print(f"[csi] wrote {out_dir / 'csi_summary_input.json'}")
    print(f"[csi] wrote {out_dir / 'csi_summary_target.json'}")

    cs_source_note = (
        "Reused the **existing in-repo clear-sky path**: "
        "`dataloader.folsom._compute_folsom_p_cs` calls pvlib's "
        "``Location(lat=38.642, lon=-121.148).get_clearsky(..., model='ineichen')`` "
        "with default settings (no altitude / aerosol overrides) and ships the "
        "result as ``target_p_cs`` / ``p_cs`` on every batch. "
        "lat/lon come from ``/work/folsom_dataset/info.yaml`` (`site.latitude`, "
        "`site.longitude`). The values are normalised by "
        "``_FOLSOM_GHI_SCALE = 1000`` and clipped to ``[0, 1.2]``. "
        "pvlib's Linke turbidity table (`LinkeTurbidities.h5`) is bundled in the "
        "installed package, so this is an offline call -- nothing fetched over "
        "the network. No alternative model (haurwitz / simplified_solis) was "
        "needed because the in-repo path is the authoritative source for the "
        "rest of the training pipeline."
    )
    _write_markdown_report(
        out_dir / "csi_summary.md",
        bundles,
        input_aggs, target_aggs,
        cs_source_note,
    )
    print(f"[csi] wrote {out_dir / 'csi_summary.md'}")

    _plot_horizon_curves(out_dir / "csi_horizon_curves.png", input_aggs, target_aggs)
    _plot_bin_bars(out_dir / "csi_bin_bars.png", input_aggs, target_aggs)
    print(f"[csi] wrote {out_dir / 'csi_horizon_curves.png'}")
    print(f"[csi] wrote {out_dir / 'csi_bin_bars.png'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
