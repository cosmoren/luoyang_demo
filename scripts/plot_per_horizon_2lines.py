"""Per-horizon RMSE/MAE plot, 2-line variant.

Sibling of ``scripts/aggregate_per_horizon.py`` that drops the "GHI+sky no
gpu2" curve and keeps only the two headline lines (GHI-only mean and GHI+sky
mean, both n=3). The shaded ±range band for GHI+sky is therefore recomputed
across all 3 sky seeds (rmse_min/max from the summary already reflect that).

Reads the pre-aggregated summary written by ``aggregate_per_horizon.py`` so
this script does not depend on the raw per-seed JSONs.

Re-run:
    ~/micromamba/envs/luoyang/bin/python \\
        ~/projects/luoyang_demo/scripts/plot_per_horizon_2lines.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

_SUMMARY_JSON = Path(
    "~/experiments_archive/ghi_vs_ghi_sky_20ep_2026-06-01/eval_outputs/"
    "per_horizon_summary.json"
).expanduser()
_OUT_PNG = Path(
    "~/projects/luoyang_demo/report_2026-06-03/plots/per_horizon_curve_2lines.png"
).expanduser()


def _draw(ax, per_step_only, per_step_sky, horizons, metric: str,
          ylabel: str, title: str) -> None:
    only_mean = [p[f"{metric}_mean"] for p in per_step_only]
    only_lo = [p[f"{metric}_min"] for p in per_step_only]
    only_hi = [p[f"{metric}_max"] for p in per_step_only]
    sky_mean = [p[f"{metric}_mean"] for p in per_step_sky]
    sky_lo = [p[f"{metric}_min"] for p in per_step_sky]
    sky_hi = [p[f"{metric}_max"] for p in per_step_sky]

    ax.fill_between(horizons, only_lo, only_hi, color="tab:gray", alpha=0.18,
                    label="GHI-only range (3 seeds)")
    ax.plot(horizons, only_mean, color="tab:gray", marker="o", lw=2,
            label="GHI-only mean (n=3)")

    ax.fill_between(horizons, sky_lo, sky_hi, color="tab:blue", alpha=0.18,
                    label="GHI+sky range (3 seeds)")
    ax.plot(horizons, sky_mean, color="tab:blue", marker="s", lw=2,
            label="GHI+sky mean (n=3)")

    ax.set_xlabel("Forecast horizon (min)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(horizons[::2])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    with open(_SUMMARY_JSON, encoding="utf-8") as f:
        summary = json.load(f)

    only = summary["arms"]["ghi_only"]["per_step"]
    sky = summary["arms"]["ghi_sky"]["per_step"]
    horizons = [p["horizon_min"] for p in only]

    fig, (ax_r, ax_m) = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)
    _draw(ax_r, only, sky, horizons, "rmse",
          "RMSE (W/m²)", "RMSE vs horizon (Folsom GHI test)")
    _draw(ax_m, only, sky, horizons, "mae",
          "MAE (W/m²)", "MAE vs horizon (Folsom GHI test)")
    ax_r.legend(loc="upper left", fontsize=7)
    ax_m.legend(loc="upper left", fontsize=7)
    fig.suptitle(
        "Per-horizon error: GHI-only vs GHI+sky (Folsom, 20ep, 3 seeds)",
        y=1.02, fontsize=11,
    )
    fig.tight_layout()

    _OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUT_PNG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {_OUT_PNG}")


if __name__ == "__main__":
    main()
