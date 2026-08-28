#!/usr/bin/env python3
"""Aggregate experiment metrics + TB training times into a hierarchical PNG table.

CLI
---
  python scripts/exp_results.py [INPUT_ROOT] [--out OUT_PNG]

Defaults:
  INPUT_ROOT  runs/  (relative to cwd)
  --out       <INPUT_ROOT>/results.png   (next to the study/parent)

Slash-command /exp-results still writes playground/<PARENT>.png by passing
--out explicitly; this script default suits later shell wiring.
"""

from __future__ import annotations

import argparse
import glob
import re
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

# Exactly 3 group colors (all subcols share the group fill).
# Cost: soft coral/rose — clearly distinct from performance blue.
GROUP_COLORS = {
    "performance": "#D6EAF8",
    "consistency": "#FCF3CF",
    "cost": "#F5D0C5",
}
GROUP_HEADER_COLORS = {
    "performance": "#AED6F1",
    "consistency": "#F9E79F",
    "cost": "#F0B27A",
}
TEXT = "#1C2833"
BORDER = "#AEB6BF"
CASE_BG = "#F8F9F9"
CASE_HEADER_BG = "#EAECEE"

BEST_EPOCH_RE = re.compile(
    r"best val-RMSE checkpoint\s*\([^)]*,\s*epoch\s*=\s*(\d+)\s*\)",
    re.IGNORECASE,
)


def _seed_dirs(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(
        [p for p in path.iterdir() if p.is_dir() and p.name.startswith("seed_")],
        key=lambda p: p.name,
    )


def find_cases(root: Path) -> list[tuple[str, Path]]:
    """Detect case / study / multi-study root and return (case_id, case_dir)."""
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"INPUT_ROOT missing: {root}")

    # Case root — immediate children are seed_*
    if _seed_dirs(root):
        return [(root.name, root)]

    # Study root — children contain seed_*
    study_cases: list[tuple[str, Path]] = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and _seed_dirs(child):
            study_cases.append((child.name, child))
    if study_cases:
        return study_cases

    # Multi-study root — study/case/seed_*
    multi: list[tuple[str, Path]] = []
    for study in sorted(root.iterdir()):
        if not study.is_dir():
            continue
        for case in sorted(study.iterdir()):
            if case.is_dir() and _seed_dirs(case):
                multi.append((f"{study.name}/{case.name}", case))
    return multi


def parse_metrics(seed_dir: Path) -> dict[str, tuple[float | None, float | None]]:
    """Return {tag: (rmse, mae)} for best_val / last_epoch."""
    matches = sorted(seed_dir.glob("*metrics_gpu*.txt"))
    out: dict[str, tuple[float | None, float | None]] = {
        "best_val": (None, None),
        "last_epoch": (None, None),
    }
    if not matches:
        return out
    path = matches[0]
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            tag, _loss, rmse_s, mae_s = parts[0], parts[1], parts[2], parts[3]
            if tag in out:
                out[tag] = (float(rmse_s), float(mae_s))
    return out


def parse_best_epoch(seed_dir: Path) -> int | None:
    """Epoch index of best checkpoint from train.out best-val line."""
    train_out = seed_dir / "train.out"
    if not train_out.is_file():
        return None
    text = train_out.read_text(errors="replace")
    matches = BEST_EPOCH_RE.findall(text)
    if not matches:
        return None
    return int(matches[-1])


def training_time_sec(seed_dir: Path) -> float | None:
    event_files = glob.glob(str(seed_dir / "**" / "events.out.tfevents.*"), recursive=True)
    if not event_files:
        return None
    t_min = None
    t_max = None
    for ef in event_files:
        try:
            loader = EventFileLoader(ef)
            for event in loader.Load():
                wt = event.wall_time
                if wt is None or wt == 0:
                    continue
                if t_min is None or wt < t_min:
                    t_min = wt
                if t_max is None or wt > t_max:
                    t_max = wt
        except Exception:
            continue
    if t_min is None or t_max is None:
        return None
    return float(t_max - t_min)


def fmt_metric(values: list[float | None]) -> str:
    xs = [v for v in values if v is not None]
    if not xs:
        return "-"
    if len(xs) == 1:
        return f"{xs[0]:.2f}"
    mean = statistics.mean(xs)
    std = statistics.stdev(xs)  # ddof=1
    return f"{mean:.2f} ± {std:.2f}"


def fmt_epoch(values: list[int | None]) -> str:
    """Comma-separated actual epoch numbers in seed order (no mean±std)."""
    xs = [str(v) for v in values if v is not None]
    if not xs:
        return "-"
    return ", ".join(xs)


def sec_to_hm(sec: float) -> str:
    """Round total seconds to nearest minute, format HH:MM."""
    total_min = int(round(sec / 60.0))
    h = total_min // 60
    m = total_min % 60
    return f"{h:02d}:{m:02d}"


def fmt_time(values: list[float | None]) -> str:
    xs = [v for v in values if v is not None]
    if not xs:
        return "-"
    if len(xs) == 1:
        return sec_to_hm(xs[0])
    mean = statistics.mean(xs)
    std = statistics.stdev(xs)
    return f"{sec_to_hm(mean)} ± {sec_to_hm(std)}"


def aggregate_case(case_dir: Path) -> dict:
    seed_dirs = _seed_dirs(case_dir)
    best_rmse, best_mae, last_rmse, last_mae = [], [], [], []
    best_epochs, times = [], []
    notes = []
    for sd in seed_dirs:
        m = parse_metrics(sd)
        if m["best_val"] == (None, None) and m["last_epoch"] == (None, None):
            notes.append(f"{sd.name}: missing metrics")
        br, ba = m["best_val"]
        lr, la = m["last_epoch"]
        best_rmse.append(br)
        best_mae.append(ba)
        last_rmse.append(lr)
        last_mae.append(la)
        be = parse_best_epoch(sd)
        best_epochs.append(be)
        if be is None:
            notes.append(f"{sd.name}: missing best epoch")
        t = training_time_sec(sd)
        times.append(t)
        if t is None:
            notes.append(f"{sd.name}: missing TB time")

    # Sort key: best-checkpoint RMSE mean (or single value); missing → +inf
    sort_vals = [v for v in best_rmse if v is not None]
    sort_rmse = statistics.mean(sort_vals) if sort_vals else float("inf")

    return {
        "n_seeds": len(seed_dirs),
        "sort_rmse": sort_rmse,
        "perf_rmse": fmt_metric(best_rmse),
        "perf_mae": fmt_metric(best_mae),
        "stab_rmse": fmt_metric(last_rmse),
        "stab_mae": fmt_metric(last_mae),
        "best_epoch": fmt_epoch(best_epochs),
        "training_time": fmt_time(times),
        "notes": notes,
    }


def render_png(rows: list[tuple[str, dict]], out_path: Path) -> None:
    col_keys = [
        "perf_rmse",
        "perf_mae",
        "stab_rmse",
        "stab_mae",
        "best_epoch",
        "training_time",
    ]
    # (primary, secondary_or_None) — secondary is smaller parenthetical
    sub_labels = [
        ("RMSE", "(best chkp)"),
        ("MAE", "(best chkp)"),
        ("RMSE", "(last chkp)"),
        ("MAE", "(last chkp)"),
        ("best epoch", None),
        ("training time", None),
    ]
    col_groups = [
        "performance",
        "performance",
        "consistency",
        "consistency",
        "consistency",
        "cost",
    ]
    n_rows = len(rows)
    n_data_cols = len(col_keys)

    case_w = 1.86
    data_w = 1.50
    row_h = 0.42
    header_h = 0.52
    group_h = 0.38

    fig_w = case_w + n_data_cols * data_w + 0.4
    fig_h = group_h + header_h + n_rows * row_h + 0.5

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=160)
    ax.set_xlim(0, case_w + n_data_cols * data_w)
    ax.set_ylim(0, group_h + header_h + n_rows * row_h)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    total_h = group_h + header_h + n_rows * row_h

    def draw_cell(x0, y0, w, h, facecolor, text, fontsize=9, weight="normal"):
        ax.add_patch(
            Rectangle(
                (x0, y0),
                w,
                h,
                facecolor=facecolor,
                edgecolor=BORDER,
                linewidth=0.8,
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

    def draw_two_line_header(x0, y0, w, h, facecolor, primary, secondary):
        ax.add_patch(
            Rectangle(
                (x0, y0),
                w,
                h,
                facecolor=facecolor,
                edgecolor=BORDER,
                linewidth=0.8,
            )
        )
        cx = x0 + w / 2
        cy = y0 + h / 2
        ax.text(
            cx,
            cy + 0.08,
            primary,
            ha="center",
            va="center",
            fontsize=9,
            color=TEXT,
            fontweight="bold",
            fontfamily="DejaVu Sans",
            clip_on=True,
        )
        ax.text(
            cx,
            cy - 0.10,
            secondary,
            ha="center",
            va="center",
            fontsize=6.5,
            color=TEXT,
            fontweight="normal",
            fontfamily="DejaVu Sans",
            clip_on=True,
        )

    # Case header spans both header rows
    y_group = total_h - group_h
    y_sub = y_group - header_h
    draw_cell(
        0,
        y_sub,
        case_w,
        group_h + header_h,
        CASE_HEADER_BG,
        "case",
        fontsize=10,
        weight="bold",
    )

    # Group headers
    draw_cell(
        case_w,
        y_group,
        2 * data_w,
        group_h,
        GROUP_HEADER_COLORS["performance"],
        "performance",
        fontsize=10,
        weight="bold",
    )
    draw_cell(
        case_w + 2 * data_w,
        y_group,
        3 * data_w,
        group_h,
        GROUP_HEADER_COLORS["consistency"],
        "consistency",
        fontsize=10,
        weight="bold",
    )
    draw_cell(
        case_w + 5 * data_w,
        y_group,
        data_w,
        group_h,
        GROUP_HEADER_COLORS["cost"],
        "cost",
        fontsize=10,
        weight="bold",
    )

    # Subheader row
    for i, ((primary, secondary), grp) in enumerate(zip(sub_labels, col_groups)):
        x0 = case_w + i * data_w
        if secondary is not None:
            draw_two_line_header(
                x0, y_sub, data_w, header_h, GROUP_COLORS[grp], primary, secondary
            )
        else:
            draw_cell(
                x0,
                y_sub,
                data_w,
                header_h,
                GROUP_COLORS[grp],
                primary,
                fontsize=8.5,
                weight="bold",
            )

    # Data rows
    for r_i, (case_id, agg) in enumerate(rows):
        y = y_sub - (r_i + 1) * row_h
        draw_cell(0, y, case_w, row_h, CASE_BG, case_id, fontsize=9)
        for i, (key, grp) in enumerate(zip(col_keys, col_groups)):
            draw_cell(
                case_w + i * data_w,
                y,
                data_w,
                row_h,
                GROUP_COLORS[grp],
                agg[key],
                fontsize=8.5,
            )

    fig.tight_layout(pad=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_table(input_root: Path, out_png: Path) -> dict:
    """Aggregate cases under input_root and write PNG. Returns summary dict."""
    input_root = input_root.resolve()
    out_png = out_png.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"INPUT_ROOT missing: {input_root}")

    cases = find_cases(input_root)
    rows = []
    all_notes = []
    total_seeds = 0
    for case_id, case_dir in cases:
        agg = aggregate_case(case_dir)
        total_seeds += agg["n_seeds"]
        rows.append((case_id, agg))
        for n in agg["notes"]:
            all_notes.append(f"{case_id}/{n}")

    # Ascending best-checkpoint RMSE (missing → bottom)
    rows.sort(key=lambda r: (r[1]["sort_rmse"], r[0]))

    render_png(rows, out_png)
    return {
        "cases": len(cases),
        "seeds": total_seeds,
        "out": str(out_png),
        "row_order": [(c, a["perf_rmse"]) for c, a in rows],
        "notes": all_notes,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a hierarchical experiment-results PNG from a runs/study/case "
            "folder tree (metrics + TensorBoard train times + best epoch)."
        )
    )
    p.add_argument(
        "input_root",
        nargs="?",
        default="runs",
        type=Path,
        help="Root of runs/study/case hierarchy (default: runs)",
    )
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: <input_root>/results.png)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_root = args.input_root
    out_png = args.out if args.out is not None else input_root / "results.png"

    try:
        summary = build_table(input_root, out_png)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1

    print(f"CASES={summary['cases']}")
    print(f"SEEDS={summary['seeds']}")
    print(f"OUT={summary['out']}")
    print("SORT=best-checkpoint RMSE ascending (missing at bottom)")
    print("BEST_EPOCH_SOURCE=train.out best val-RMSE checkpoint line")
    print(
        "ROW_ORDER="
        + ", ".join(f"{c}({rmse})" for c, rmse in summary["row_order"])
    )
    if summary["notes"]:
        print("NOTES:")
        for n in summary["notes"]:
            print(f"  {n}")
    else:
        print("NOTES: none (all seeds had metrics + best epoch + TB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
