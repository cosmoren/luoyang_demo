"""Generate val-loss plots for Folsom base (May-26) and Folsom fixed (Jun-01).

Both saved into ~/projects/luoyang_demo/report_2026-06-03/plots/.

Re-run:
    ~/micromamba/envs/luoyang/bin/python ~/projects/luoyang_demo/scripts/build_val_loss_plots.py
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

OUT_DIR = pathlib.Path("~/projects/luoyang_demo/report_2026-06-03/plots").expanduser()
OUT_DIR.mkdir(parents=True, exist_ok=True)

ARCHIVES = {
    "base": {
        "title": "Folsom base -- val loss per epoch (May 26, 2026, 40 ep)",
        "subtitle": "folsom_kt_sky_vs_nosky_40ep_4runs_2026-05-26  |  4 runs (2 sky + 2 no-sky)",
        "out_path": OUT_DIR / "val_loss_folsom_base.png",
        "x_max": 40,
        "runs": {
            "sky_r1":   "/home/erfan/experiments_archive/folsom_kt_sky_vs_nosky_40ep_4runs_2026-05-26/runs/folsom_kt_sky_40ep_r1",
            "sky_r2":   "/home/erfan/experiments_archive/folsom_kt_sky_vs_nosky_40ep_4runs_2026-05-26/runs/folsom_kt_sky_40ep_r2",
            "nosky_r1": "/home/erfan/experiments_archive/folsom_kt_sky_vs_nosky_40ep_4runs_2026-05-26/runs/folsom_kt_nosky_40ep_r1",
            "nosky_r2": "/home/erfan/experiments_archive/folsom_kt_sky_vs_nosky_40ep_4runs_2026-05-26/runs/folsom_kt_nosky_40ep_r2",
        },
    },
    "fixed": {
        "title": "Folsom fixed -- val loss per epoch (Jun 1, 2026, 20 ep, commit 518dca9)",
        "subtitle": "folsom_kt4000_d30_20ep_2026-06-01  |  8 runs (4 sky + 4 no-sky)",
        "out_path": OUT_DIR / "val_loss_folsom_fixed.png",
        "x_max": 20,
        "runs": {
            "sky_r1": "/home/erfan/experiments_archive/folsom_kt4000_d30_20ep_2026-06-01/training_logs/folsom_20260529_214331/ghi_nwp_sky/run1/tb",
            "sky_r2": "/home/erfan/experiments_archive/folsom_kt4000_d30_20ep_2026-06-01/training_logs/folsom_20260529_214331/ghi_nwp_sky/run2/tb",
            "sky_r3": "/home/erfan/experiments_archive/folsom_kt4000_d30_20ep_2026-06-01/training_logs/folsom_20260529_214331/ghi_nwp_sky/run3/tb",
            "sky_r4": "/home/erfan/experiments_archive/folsom_kt4000_d30_20ep_2026-06-01/training_logs/folsom_20260529_214331/ghi_nwp_sky/run4/tb",
            "nosky_r1": "/home/erfan/experiments_archive/folsom_kt4000_d30_20ep_2026-06-01/training_logs/folsom_20260529_214331/ghi_nwp/run1/tb",
            "nosky_r2": "/home/erfan/experiments_archive/folsom_kt4000_d30_20ep_2026-06-01/training_logs/folsom_20260529_214331/ghi_nwp/run2/tb",
            "nosky_r3": "/home/erfan/experiments_archive/folsom_kt4000_d30_20ep_2026-06-01/training_logs/folsom_20260529_214331/ghi_nwp/run3/tb",
            "nosky_r4": "/home/erfan/experiments_archive/folsom_kt4000_d30_20ep_2026-06-01/training_logs/folsom_20260529_214331/ghi_nwp/run4/tb",
        },
    },
}

C_SKY = "#1d4ed8"
C_NOSKY = "#d97706"


def read_val_loss(tb_dir: str) -> tuple[list[int], list[float]]:
    ea = event_accumulator.EventAccumulator(tb_dir, size_guidance={"scalars": 0})
    ea.Reload()
    pts = ea.Scalars("loss/val")
    return [p.step + 1 for p in pts], [p.value for p in pts]


def plot_archive(spec: dict) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=160)

    sky_series, nosky_series = [], []
    for name, path in spec["runs"].items():
        steps, vals = read_val_loss(path)
        if name.startswith("sky"):
            sky_series.append((name, steps, vals))
        else:
            nosky_series.append((name, steps, vals))

    for i, (_, steps, vals) in enumerate(sky_series):
        label = "GHI + NWP + sky" if i == 0 else None
        ax.plot(steps, vals, color=C_SKY, alpha=0.55, lw=1.4, label=label)
    for i, (_, steps, vals) in enumerate(nosky_series):
        label = "GHI + NWP, no sky" if i == 0 else None
        ax.plot(steps, vals, color=C_NOSKY, alpha=0.55, lw=1.4, label=label)

    ax.set_xlim(1, spec["x_max"])
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation loss (Huber)")
    ax.set_title(spec["title"], fontsize=12, loc="left", pad=14)
    ax.text(
        0.0, 1.005, spec["subtitle"],
        transform=ax.transAxes, fontsize=9, color="#6b7280",
        ha="left", va="bottom",
    )
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(spec["out_path"], dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {spec['out_path']}")


for spec in ARCHIVES.values():
    plot_archive(spec)
