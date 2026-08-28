#!/usr/bin/env python3
"""Combine hand-picked runs from several experiment folders into B/W tables.

Unlike exp_results.py (which sweeps a whole runs/ tree), this script renders only
the rows listed in TABLES below: modalities | case | RMSE | MAE, sorted by RMSE
ascending. Metrics are the best-checkpoint values, averaged over seed_* dirs.

Editing TABLES is the intended way to add or change a comparison.

CLI
---
  python scripts/compare_results.py [--table NAME] [--force]
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_ABLATION_16H = (
    "/home/kyber/projects/digital_energy/experiment_files/old/old_local_runs/"
    "folsom sky channel ablation 16 horizons/"
    "2026-07-21_folsom-channel-ablation-16horizon"
)
_ANGULAR_16H = (
    "/home/kyber/projects/digital_energy/experiment_files/old/old_local_runs/"
    "folsom sky channel ablation 16 horizons/"
    "2026-08-13_folsom-guassian angular-16horizon"
)
_LUO_ADAPT = (
    "/home/kyber/projects/digital_energy/experiment_files/runs/"
    "fol luo adapt and fix/2026-08-14_luo-adapt"
)
_FOL2LUO = (
    "/home/kyber/projects/digital_energy/experiment_files/runs/"
    "fol luo adapt and fix/2026-08-17_fol-2-luo"
)
_IMGS = Path("/home/kyber/projects/digital_energy/report/week_19/imgs")

# Each table: name, out path, and one entry per row. "root" is the experiment
# parent folder, "run" the case folder inside it (found at any depth).
TABLES: list[dict] = [
    {
        "name": "sky_extra_channel",
        "out": _IMGS / "sky_extra_channel.png",
        "rows": [
            {
                "root": _ABLATION_16H,
                "run": "gaussian_pixel",
                "modalities": "ghi + sky",
                "case": "2d pixel gaussian",
            },
            {
                "root": _ANGULAR_16H,
                "run": "gaussian_angular",
                "modalities": "ghi + sky",
                "case": "2d angular gaussian",
            },
            {
                "root": _ABLATION_16H,
                "run": "sky_rgb",
                "modalities": "ghi + sky",
                "case": "-",
            },
        ],
    },
    {
        "name": "luoyang_base",
        "out": _IMGS / "luoyang_base.png",
        "rows": [
            {
                "root": _LUO_ADAPT,
                "run": "pv-nwp",
                "modalities": "pv + nwp",
                "case": "-",
            },
            {
                "root": _LUO_ADAPT,
                "run": "pv-only",
                "modalities": "pv",
                "case": "-",
            },
            {
                "root": _LUO_ADAPT,
                "run": "pv-sky",
                "modalities": "pv + sky",
                "case": "-",
            },
            {
                "root": _LUO_ADAPT,
                "run": "pv-sky-nwp",
                "modalities": "pv + sky + nwp",
                "case": "-",
            },
        ],
    },
    {
        "name": "luoyang_ft",
        "out": _IMGS / "luoyang_ft.png",
        "rows": [
            {
                "root": f"{_FOL2LUO}/(pv+nwp)(pv+nwp+sky)/(best)(best)",
                "run": "ft",
                "modalities": "pv + nwp + sky",
                "case": "folsom sky implant + ft",
            },
            {
                "root": f"{_FOL2LUO}/(pv)(pv+sky)/(best)(best)",
                "run": "ft",
                "modalities": "pv + sky",
                "case": "folsom sky implant + ft",
            },
        ],
    },
]

TEXT = "#000000"
BORDER = "#000000"
CELL_BG = "#FFFFFF"
HEADER_BG = "#FFFFFF"

_SEED_RE = re.compile(r"^seed_")


def _seed_dirs(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(
        [p for p in path.iterdir() if p.is_dir() and _SEED_RE.match(p.name)],
        key=lambda p: p.name,
    )


def find_run_dir(root: Path, run: str) -> Path:
    """Locate the case folder named `run` under `root` that holds seed_* dirs."""
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"root missing: {root}")

    candidates = [root / run] if (root / run).is_dir() else []
    candidates += [p for p in sorted(root.rglob(run)) if p.is_dir()]

    for cand in candidates:
        if _seed_dirs(cand):
            return cand
    raise FileNotFoundError(f"no seed_* dirs for run '{run}' under {root}")


def parse_best_metrics(seed_dir: Path) -> tuple[float | None, float | None]:
    """(rmse, mae) of the best_val row in the metrics file, or (None, None)."""
    matches = sorted(seed_dir.glob("*metrics_gpu*.txt"))
    if not matches:
        return (None, None)
    with open(matches[0]) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4 and parts[0] == "best_val":
                return (float(parts[2]), float(parts[3]))
    return (None, None)


def fmt_metric(values: list[float | None]) -> str:
    xs = [v for v in values if v is not None]
    if not xs:
        return "-"
    if len(xs) == 1:
        return f"{xs[0]:.2f}"
    return f"{statistics.mean(xs):.2f} ± {statistics.stdev(xs):.2f}"


def aggregate_run(run_dir: Path) -> dict:
    seed_dirs = _seed_dirs(run_dir)
    rmses: list[float | None] = []
    maes: list[float | None] = []
    notes: list[str] = []
    for sd in seed_dirs:
        rmse, mae = parse_best_metrics(sd)
        if rmse is None and mae is None:
            notes.append(f"{sd.name}: missing metrics")
        rmses.append(rmse)
        maes.append(mae)

    present = [v for v in rmses if v is not None]
    return {
        "n_seeds": len(seed_dirs),
        "sort_rmse": statistics.mean(present) if present else float("inf"),
        "rmse": fmt_metric(rmses),
        "mae": fmt_metric(maes),
        "notes": notes,
    }


def render_png(rows: list[dict], out_path: Path) -> None:
    headers = ["modalities", "case", "RMSE", "MAE"]
    keys = ["modalities", "case", "rmse", "mae"]
    col_w = [1.90, 2.35, 1.50, 1.50]

    row_h = 0.42
    header_h = 0.46
    table_w = sum(col_w)
    table_h = header_h + len(rows) * row_h

    fig, ax = plt.subplots(figsize=(table_w + 0.4, table_h + 0.4), dpi=160)
    ax.set_xlim(0, table_w)
    ax.set_ylim(0, table_h)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    def draw_cell(x0, y0, w, h, facecolor, text, weight="normal", fontsize=9):
        ax.add_patch(
            Rectangle(
                (x0, y0), w, h, facecolor=facecolor, edgecolor=BORDER, linewidth=0.8
            )
        )
        ax.text(
            x0 + w / 2,
            y0 + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=TEXT,
            fontweight=weight,
            fontfamily="DejaVu Sans",
            clip_on=True,
        )

    y_header = table_h - header_h
    x = 0.0
    for label, w in zip(headers, col_w):
        draw_cell(x, y_header, w, header_h, HEADER_BG, label, weight="bold", fontsize=10)
        x += w

    for r_i, row in enumerate(rows):
        y = y_header - (r_i + 1) * row_h
        x = 0.0
        for key, w in zip(keys, col_w):
            draw_cell(x, y, w, row_h, CELL_BG, row[key])
            x += w

    fig.tight_layout(pad=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_table(specs: list[dict[str, str]], out_png: Path) -> dict:
    out_png = out_png.expanduser().resolve()
    rows = []
    all_notes: list[str] = []
    total_seeds = 0

    for spec in specs:
        run_dir = find_run_dir(Path(spec["root"]), spec["run"])
        agg = aggregate_run(run_dir)
        total_seeds += agg["n_seeds"]
        rows.append(
            {
                "modalities": spec["modalities"],
                "case": spec["case"],
                "rmse": agg["rmse"],
                "mae": agg["mae"],
                "sort_rmse": agg["sort_rmse"],
                "run": spec["run"],
            }
        )
        all_notes += [f"{spec['run']}/{n}" for n in agg["notes"]]

    rows.sort(key=lambda r: (r["sort_rmse"], r["case"]))
    render_png(rows, out_png)
    return {
        "rows": len(rows),
        "seeds": total_seeds,
        "out": str(out_png),
        "row_order": [(r["modalities"], r["case"], r["rmse"]) for r in rows],
        "notes": all_notes,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    names = [t["name"] for t in TABLES]
    p = argparse.ArgumentParser(
        description="Render black-and-white comparison tables of hand-picked runs."
    )
    p.add_argument(
        "--table",
        choices=names,
        default=None,
        help="Render only this named table (default: all)",
    )
    p.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Allow overwriting an existing output PNG",
    )
    return p.parse_args(argv)


def _print_summary(name: str, summary: dict) -> None:
    print(f"TABLE={name}")
    print(f"ROWS={summary['rows']}")
    print(f"SEEDS={summary['seeds']}")
    print(f"OUT={summary['out']}")
    print("SORT=best-checkpoint RMSE ascending (missing at bottom)")
    print(
        "ROW_ORDER="
        + ", ".join(f"{mod}/{case}({rmse})" for mod, case, rmse in summary["row_order"])
    )
    if summary["notes"]:
        print("NOTES:")
        for n in summary["notes"]:
            print(f"  {n}")
    else:
        print("NOTES: none (all seeds had best-checkpoint metrics)")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    selected = [t for t in TABLES if args.table is None or t["name"] == args.table]
    rc = 0
    for table in selected:
        out = Path(table["out"])
        if out.exists() and not args.force:
            print(f"refusing to overwrite existing {out} (pass --force)", file=sys.stderr)
            rc = 1
            continue
        try:
            summary = build_table(table["rows"], out)
        except FileNotFoundError as e:
            print(str(e), file=sys.stderr)
            rc = 1
            continue
        _print_summary(table["name"], summary)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
