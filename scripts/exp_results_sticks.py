#!/usr/bin/env python3
"""Horizontal mean±std stick plot of best-checkpoint RMSE across cases.

CLI
---
  python scripts/exp_results_sticks.py INPUT_ROOT [INPUT_ROOT ...] [--out OUT_PNG]

Defaults:
  --out   <INPUT_ROOT>/results_sticks.png when exactly one root is given
          (required when multiple roots are given)

Each positional path is an explicit study/case folder to include. Cases from
all roots are merged into one plot; when more than one root is given (or a
case id would collide), rows are labeled study_name/case_name.
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def _load_exp_results():
    """Load sibling exp_results.py as a module (scripts/ is not a package)."""
    path = Path(__file__).resolve().parent / "exp_results.py"
    spec = importlib.util.spec_from_file_location("exp_results", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_er = _load_exp_results()
find_cases = _er.find_cases
parse_metrics = _er.parse_metrics
_seed_dirs = _er._seed_dirs


def aggregate_best_rmse(case_dir: Path) -> dict:
    """Mean / std of best-checkpoint RMSE across seeds (ddof=1)."""
    values: list[float | None] = []
    notes: list[str] = []
    for sd in _seed_dirs(case_dir):
        rmse, _mae = parse_metrics(sd)["best_val"]
        values.append(rmse)
        if rmse is None:
            notes.append(f"{sd.name}: missing metrics")

    xs = [v for v in values if v is not None]
    if not xs:
        mean = None
        std = None
        sort_rmse = float("inf")
        label = "-"
    elif len(xs) == 1:
        mean = xs[0]
        std = None
        sort_rmse = mean
        label = f"{mean:.2f}"
    else:
        mean = statistics.mean(xs)
        std = statistics.stdev(xs)
        sort_rmse = mean
        label = f"{mean:.2f} ± {std:.2f}"

    return {
        "n_seeds": len(values),
        "n_valid": len(xs),
        "mean": mean,
        "std": std,
        "sort_rmse": sort_rmse,
        "label": label,
        "notes": notes,
    }


def _qualify_case_id(case_id: str, root: Path, multi_root: bool) -> str:
    """Prefix with study folder name when needed for disambiguation."""
    if "/" in case_id:
        return case_id
    if multi_root:
        return f"{root.name}/{case_id}"
    return case_id


def collect_rows(roots: list[Path]) -> tuple[list[tuple[str, dict]], list[str]]:
    multi_root = len(roots) > 1
    # First pass: gather raw ids to detect collisions even for a single root
    # that is itself a multi-study tree (ids already study/case).
    staged: list[tuple[str, Path, Path]] = []  # (raw_id, case_dir, root)
    for root in roots:
        for case_id, case_dir in find_cases(root):
            staged.append((case_id, case_dir, root))

    raw_ids = [cid for cid, _, _ in staged]
    # Collisions among raw ids → always qualify with root name when multi_root
    # or when the same bare case name appears more than once.
    need_qualify = multi_root or len(raw_ids) != len(set(raw_ids))

    rows: list[tuple[str, dict]] = []
    all_notes: list[str] = []
    seen: set[str] = set()
    for case_id, case_dir, root in staged:
        display_id = _qualify_case_id(case_id, root, need_qualify)
        if display_id in seen:
            display_id = f"{root.name}/{case_id}"
        seen.add(display_id)
        agg = aggregate_best_rmse(case_dir)
        rows.append((display_id, agg))
        for n in agg["notes"]:
            all_notes.append(f"{display_id}/{n}")

    rows.sort(key=lambda r: (r[1]["sort_rmse"], r[0]))
    return rows, all_notes


def render_sticks(rows: list[tuple[str, dict]], out_path: Path) -> None:
    n = len(rows)
    if n == 0:
        raise ValueError("no cases found")

    # rows already sorted best→worst; matplotlib y increases upward, so best
    # lands at the bottom of the axis and worst at the top.
    y_positions = list(range(n))
    means = [agg["mean"] for _, agg in rows]
    stds = [agg["std"] if agg["std"] is not None else 0.0 for _, agg in rows]
    labels = [agg["label"] for _, agg in rows]
    names = [cid for cid, _ in rows]
    has_err = [agg["std"] is not None for _, agg in rows]

    fig_h = max(2.5, 0.45 * n + 1.2)
    fig, ax = plt.subplots(figsize=(8.5, fig_h), dpi=160)

    for y, mean, std, err_ok in zip(y_positions, means, stds, has_err):
        if mean is None:
            continue
        ax.errorbar(
            mean,
            y,
            xerr=(std if err_ok else None),
            fmt="o",
            color="#1C2833",
            ecolor="#5D6D7E",
            elinewidth=1.6,
            capsize=4,
            markersize=6,
            zorder=3,
        )

    x_valid = [m for m in means if m is not None]
    if x_valid:
        x_max = max(
            (m + (s if e else 0.0))
            for m, s, e in zip(means, stds, has_err)
            if m is not None
        )
        x_min = min(
            (m - (s if e else 0.0))
            for m, s, e in zip(means, stds, has_err)
            if m is not None
        )
        pad = max(0.5, 0.04 * (x_max - x_min + 1e-9))
    else:
        x_min, x_max, pad = 0.0, 0.0, 1.0

    for y, mean, lab in zip(y_positions, means, labels):
        if mean is None:
            ax.text(0.0, y, lab, va="center", ha="left", fontsize=8, color="#7F8C8D")
            continue
        ax.text(
            x_max + pad,
            y,
            lab,
            va="center",
            ha="left",
            fontsize=8.5,
            color="#1C2833",
            fontfamily="DejaVu Sans",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(names, fontsize=9, fontfamily="DejaVu Sans")
    ax.set_xlabel("best-checkpoint RMSE", fontsize=10)
    ax.set_title("Best-checkpoint RMSE (mean ± std across seeds)", fontsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.45)
    ax.set_axisbelow(True)
    if x_valid:
        ax.set_xlim(x_min - pad, x_max + pad * 6)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a horizontal mean±std stick plot of best-checkpoint RMSE "
            "from one or more explicit study/case folders."
        )
    )
    p.add_argument(
        "input_roots",
        nargs="+",
        type=Path,
        help="One or more study/case roots to include (explicit paths only)",
    )
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help=(
            "Output PNG path (default: <only_root>/results_sticks.png; "
            "required when multiple roots are given)"
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    roots = [r.resolve() for r in args.input_roots]
    for root in roots:
        if not root.is_dir():
            print(f"INPUT_ROOT missing: {root}", file=sys.stderr)
            return 1

    if args.out is None:
        if len(roots) != 1:
            print(
                "--out is required when multiple input roots are given",
                file=sys.stderr,
            )
            return 1
        out_png = roots[0] / "results_sticks.png"
    else:
        out_png = args.out.resolve()

    try:
        rows, notes = collect_rows(roots)
        if not rows:
            print("no cases found under given roots", file=sys.stderr)
            return 1
        render_sticks(rows, out_png)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    total_seeds = sum(a["n_seeds"] for _, a in rows)
    print(f"CASES={len(rows)}")
    print(f"SEEDS={total_seeds}")
    print(f"OUT={out_png}")
    print("SORT=best-checkpoint RMSE ascending (missing at bottom)")
    print(
        "ROW_ORDER="
        + ", ".join(f"{c}({a['label']})" for c, a in rows)
    )
    if notes:
        print("NOTES:")
        for n in notes:
            print(f"  {n}")
    else:
        print("NOTES: none (all seeds had best-checkpoint metrics)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
