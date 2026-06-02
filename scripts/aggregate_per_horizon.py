"""
Aggregate the 6 per-horizon JSON reports produced by ``scripts/eval_per_horizon.py``
into a summary JSON + markdown table + RMSE/MAE-vs-horizon plot.

Arms in the archived ``ghi_vs_ghi_sky_20ep_2026-06-01`` experiment:
  - GHI-only:     ghi_only_gpu0, ghi_only_gpu1, ghi_only_gpu2  (--zero-sky)
  - GHI+sky:      ghi_sky_gpu0,  ghi_sky_gpu1,  ghi_sky_gpu2   (no flag)
  - GHI+sky\\gpu2: drop the gpu2 seed because it plateaued at epoch 7/20
    (see RUN_NOTES.txt "ghi_sky_gpu2 stalled" comment).

Outputs:
  - per_horizon_summary.json     -- structured data, ready for downstream consumers.
  - per_horizon_summary.md       -- human-readable table (no n_valid column per spec).
  - per_horizon_curve.png        -- two-subplot RMSE/MAE vs horizon, with ±range bands.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

_EVAL_DIR = Path("/home/erfan/projects/luoyang_demo/eval_outputs")
_GHI_ONLY_SEEDS = ["ghi_only_gpu0", "ghi_only_gpu1", "ghi_only_gpu2"]
_GHI_SKY_SEEDS = ["ghi_sky_gpu0", "ghi_sky_gpu1", "ghi_sky_gpu2"]
_GHI_SKY_NO_OUTLIER = ["ghi_sky_gpu0", "ghi_sky_gpu1"]  # drop gpu2 plateau-at-ep7 seed


def _load_seed(name: str) -> dict:
    p = _EVAL_DIR / f"per_horizon_{name}.json"
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _stack_metric(seed_data: dict[str, dict], seeds: list[str], metric: str) -> list[list[float]]:
    """Return shape [n_steps][n_seeds] for the requested metric ('rmse' or 'mae')."""
    n_steps = len(seed_data[seeds[0]]["per_step"])
    out = [[seed_data[s]["per_step"][step][metric] for s in seeds] for step in range(n_steps)]
    return out


def _arm_summary(seed_data: dict[str, dict], seeds: list[str]) -> dict:
    rmse_per_step = _stack_metric(seed_data, seeds, "rmse")
    mae_per_step = _stack_metric(seed_data, seeds, "mae")
    out = {"seeds": list(seeds), "per_step": []}
    for step, (rmses, maes) in enumerate(zip(rmse_per_step, mae_per_step)):
        out["per_step"].append({
            "step": step + 1,
            "horizon_min": seed_data[seeds[0]]["per_step"][step]["horizon_min"],
            "rmse_mean": _mean(rmses), "rmse_min": min(rmses), "rmse_max": max(rmses),
            "rmse_per_seed": dict(zip(seeds, rmses)),
            "mae_mean": _mean(maes), "mae_min": min(maes), "mae_max": max(maes),
            "mae_per_seed": dict(zip(seeds, maes)),
        })
    # Aggregate-over-horizon scalars per arm (mean of per-seed aggregate, also recorded
    # for cross-check against the existing metrics file numbers).
    out["aggregate_per_seed"] = {
        s: {"rmse": seed_data[s]["aggregate"]["rmse"],
            "mae": seed_data[s]["aggregate"]["mae"],
            "loss": seed_data[s]["aggregate"]["loss"]}
        for s in seeds
    }
    out["aggregate_mean"] = {
        "rmse": _mean([seed_data[s]["aggregate"]["rmse"] for s in seeds]),
        "mae": _mean([seed_data[s]["aggregate"]["mae"] for s in seeds]),
        "loss": _mean([seed_data[s]["aggregate"]["loss"] for s in seeds]),
    }
    return out


def _build_summary() -> dict:
    seed_data = {s: _load_seed(s) for s in (_GHI_ONLY_SEEDS + _GHI_SKY_SEEDS)}
    # Sanity: confirm every seed passed its sanity check; we don't want to silently
    # aggregate bad numbers.
    for s, d in seed_data.items():
        ok = d.get("sanity_check", {}).get("passed")
        if not ok:
            raise RuntimeError(
                f"seed {s} did not pass per-checkpoint sanity check (sanity_check.passed={ok}). "
                "Re-run scripts/eval_per_horizon.py before aggregating."
            )

    ghi_only = _arm_summary(seed_data, _GHI_ONLY_SEEDS)
    ghi_sky = _arm_summary(seed_data, _GHI_SKY_SEEDS)
    ghi_sky_no_outlier = _arm_summary(seed_data, _GHI_SKY_NO_OUTLIER)

    # Per-step deltas: sky - no_sky (negative = sky better).
    deltas_all = []
    deltas_no_outlier = []
    for i in range(len(ghi_only["per_step"])):
        only = ghi_only["per_step"][i]
        sky = ghi_sky["per_step"][i]
        sky_no = ghi_sky_no_outlier["per_step"][i]
        deltas_all.append({
            "step": only["step"], "horizon_min": only["horizon_min"],
            "delta_rmse": sky["rmse_mean"] - only["rmse_mean"],
            "delta_mae": sky["mae_mean"] - only["mae_mean"],
            "pct_rmse": 100.0 * (sky["rmse_mean"] - only["rmse_mean"]) / only["rmse_mean"],
            "pct_mae": 100.0 * (sky["mae_mean"] - only["mae_mean"]) / only["mae_mean"],
        })
        deltas_no_outlier.append({
            "step": only["step"], "horizon_min": only["horizon_min"],
            "delta_rmse": sky_no["rmse_mean"] - only["rmse_mean"],
            "delta_mae": sky_no["mae_mean"] - only["mae_mean"],
            "pct_rmse": 100.0 * (sky_no["rmse_mean"] - only["rmse_mean"]) / only["rmse_mean"],
            "pct_mae": 100.0 * (sky_no["mae_mean"] - only["mae_mean"]) / only["mae_mean"],
        })

    return {
        "arms": {
            "ghi_only": ghi_only,
            "ghi_sky": ghi_sky,
            "ghi_sky_no_outlier": ghi_sky_no_outlier,
        },
        "deltas": {
            "sky_vs_only": deltas_all,
            "sky_no_outlier_vs_only": deltas_no_outlier,
        },
        "notes": (
            "Per-arm mean is the unweighted mean across seeds of the per-step "
            "RMSE / MAE produced by scripts/eval_per_horizon.py. Deltas are "
            "sky - no_sky (negative = sky helps). 'sky_no_outlier' drops "
            "ghi_sky_gpu2 because it plateaued at epoch 7 of 20 (RUN_NOTES.txt)."
        ),
    }


def _fmt_md_table(summary: dict) -> str:
    only = summary["arms"]["ghi_only"]["per_step"]
    sky = summary["arms"]["ghi_sky"]["per_step"]
    sky_no = summary["arms"]["ghi_sky_no_outlier"]["per_step"]
    lines = []
    lines.append("# Per-horizon RMSE / MAE — Folsom GHI vs GHI+sky")
    lines.append("")
    lines.append(
        "Test split, batch_size=64, 5 batches. Per-step metrics replay "
        "`training/train_vit_test_folsom.py::evaluate` semantics (target_mask + "
        "night-pred-zero) but accumulate per output step. All values W/m^2. "
        "`Δ` columns = `GHI+sky mean − GHI-only mean` (negative ⇒ sky helps). "
        "All 6 checkpoints reproduced the metrics-file aggregate to <2e-6 W/m^2."
    )
    lines.append("")
    lines.append("## Per-output-step RMSE")
    lines.append("")
    lines.append("| Step | Horizon (min) | GHI-only mean | GHI+sky mean | GHI+sky (no gpu2) | Δ sky | Δ sky (no gpu2) |")
    lines.append("|-----:|--------------:|--------------:|-------------:|------------------:|------:|----------------:|")
    for i in range(len(only)):
        o, s, sn = only[i], sky[i], sky_no[i]
        d = s["rmse_mean"] - o["rmse_mean"]
        dn = sn["rmse_mean"] - o["rmse_mean"]
        lines.append(
            f"| {o['step']:>4} | {o['horizon_min']:>13} | "
            f"{o['rmse_mean']:>13.2f} | {s['rmse_mean']:>12.2f} | "
            f"{sn['rmse_mean']:>17.2f} | {d:>+6.2f} | {dn:>+15.2f} |"
        )
    lines.append("")
    lines.append("## Per-output-step MAE")
    lines.append("")
    lines.append("| Step | Horizon (min) | GHI-only mean | GHI+sky mean | GHI+sky (no gpu2) | Δ sky | Δ sky (no gpu2) |")
    lines.append("|-----:|--------------:|--------------:|-------------:|------------------:|------:|----------------:|")
    for i in range(len(only)):
        o, s, sn = only[i], sky[i], sky_no[i]
        d = s["mae_mean"] - o["mae_mean"]
        dn = sn["mae_mean"] - o["mae_mean"]
        lines.append(
            f"| {o['step']:>4} | {o['horizon_min']:>13} | "
            f"{o['mae_mean']:>13.2f} | {s['mae_mean']:>12.2f} | "
            f"{sn['mae_mean']:>17.2f} | {d:>+6.2f} | {dn:>+15.2f} |"
        )
    lines.append("")

    # Aggregate-over-horizon row for cross-check.
    agg_only = summary["arms"]["ghi_only"]["aggregate_mean"]
    agg_sky = summary["arms"]["ghi_sky"]["aggregate_mean"]
    agg_skyno = summary["arms"]["ghi_sky_no_outlier"]["aggregate_mean"]
    lines.append("## Aggregate over steps 1..16 (sanity vs RUN_NOTES.txt headline)")
    lines.append("")
    lines.append("| Arm | RMSE (mean) | MAE (mean) |")
    lines.append("|-----|------------:|-----------:|")
    lines.append(f"| GHI-only           | {agg_only['rmse']:>11.2f} | {agg_only['mae']:>10.2f} |")
    lines.append(f"| GHI+sky            | {agg_sky['rmse']:>11.2f} | {agg_sky['mae']:>10.2f} |")
    lines.append(f"| GHI+sky (no gpu2)  | {agg_skyno['rmse']:>11.2f} | {agg_skyno['mae']:>10.2f} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def _plot(summary: dict, out_path: Path) -> None:
    only = summary["arms"]["ghi_only"]["per_step"]
    sky = summary["arms"]["ghi_sky"]["per_step"]
    sky_no = summary["arms"]["ghi_sky_no_outlier"]["per_step"]
    horizons = [p["horizon_min"] for p in only]

    fig, (ax_r, ax_m) = plt.subplots(1, 2, figsize=(13, 5), sharex=True)

    def _draw(ax, metric: str, ylabel: str, title: str):
        only_mean = [p[f"{metric}_mean"] for p in only]
        only_lo = [p[f"{metric}_min"] for p in only]
        only_hi = [p[f"{metric}_max"] for p in only]
        sky_mean = [p[f"{metric}_mean"] for p in sky]
        sky_lo = [p[f"{metric}_min"] for p in sky]
        sky_hi = [p[f"{metric}_max"] for p in sky]
        skyno_mean = [p[f"{metric}_mean"] for p in sky_no]
        skyno_lo = [p[f"{metric}_min"] for p in sky_no]
        skyno_hi = [p[f"{metric}_max"] for p in sky_no]

        ax.fill_between(horizons, only_lo, only_hi, color="tab:gray", alpha=0.18,
                        label="GHI-only range (3 seeds)")
        ax.plot(horizons, only_mean, color="tab:gray", marker="o", lw=2,
                label="GHI-only mean (n=3)")

        ax.fill_between(horizons, sky_lo, sky_hi, color="tab:blue", alpha=0.18,
                        label="GHI+sky range (3 seeds)")
        ax.plot(horizons, sky_mean, color="tab:blue", marker="s", lw=2,
                label="GHI+sky mean (n=3)")

        ax.fill_between(horizons, skyno_lo, skyno_hi, color="tab:orange", alpha=0.18,
                        label="GHI+sky range (no gpu2)")
        ax.plot(horizons, skyno_mean, color="tab:orange", marker="^", lw=2,
                label="GHI+sky mean, no gpu2 (n=2)")

        ax.set_xlabel("Forecast horizon (min)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(horizons[::2])  # 15, 45, 75, ..., to avoid overlap

    _draw(ax_r, "rmse", "RMSE (W/m²)", "RMSE vs horizon (Folsom GHI test)")
    _draw(ax_m, "mae", "MAE (W/m²)", "MAE vs horizon (Folsom GHI test)")
    ax_r.legend(loc="upper left", fontsize=8)
    ax_m.legend(loc="upper left", fontsize=8)
    fig.suptitle("Per-horizon error: GHI-only vs GHI+sky (Folsom, 20ep, 3 seeds)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summary = _build_summary()
    (_EVAL_DIR / "per_horizon_summary.json").write_text(json.dumps(summary, indent=2))
    (_EVAL_DIR / "per_horizon_summary.md").write_text(_fmt_md_table(summary))
    _plot(summary, _EVAL_DIR / "per_horizon_curve.png")
    print(f"wrote {_EVAL_DIR / 'per_horizon_summary.json'}")
    print(f"wrote {_EVAL_DIR / 'per_horizon_summary.md'}")
    print(f"wrote {_EVAL_DIR / 'per_horizon_curve.png'}")


if __name__ == "__main__":
    main()
