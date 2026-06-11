"""
Build the Folsom GridSat-CONUS goes15 smoke-test diagnostic report.

Reads the JSONL manifest from ``gridsat_to_shards.py`` and produces:
  * a Markdown report with summary tables + embedded plots
  * PNG figures (per-day count, alignment histogram, value-range drift, sample frames)

The manifest is the single source of truth. We re-load saved .npy shards only for the visual samples
and the all-month per-channel stats sanity check (cheap: ~3000 files * 60 KB = ~180 MB).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

STATUS_OK = "ok"
STATUSES = ["ok", "listing_miss", "download_error", "parse_error", "empty", "worker_crash"]


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _parse_iso(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--shard-root", type=Path, default=Path("/work/folsom_dataset/sat_goes_gridsat"))
    p.add_argument("--report", type=Path, required=True, help="Output Markdown path")
    p.add_argument("--figs-dir", type=Path, required=True, help="Output figures dir (PNGs)")
    p.add_argument("--label", type=str, default="2014-01")
    p.add_argument("--seed", type=int, default=20140115)
    args = p.parse_args()

    args.figs_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    records = _load_manifest(args.manifest)
    if not records:
        raise SystemExit(f"empty manifest: {args.manifest}")

    n_total = len(records)
    status_counts = Counter(r["status"] for r in records)
    n_ok = status_counts.get(STATUS_OK, 0)
    n_missing = n_total - n_ok

    # ----- per-day status counts -----
    by_day: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        d = r["anchor_iso"][:10]
        by_day[d][r["status"]] += 1
    days_sorted = sorted(by_day.keys())

    fig, ax = plt.subplots(figsize=(11, 4))
    bottom = np.zeros(len(days_sorted))
    colors = {
        "ok": "#2c7fb8",
        "listing_miss": "#fd8d3c",
        "download_error": "#cb181d",
        "parse_error": "#9e9ac8",
        "empty": "#74c476",
        "worker_crash": "#525252",
    }
    for st in STATUSES:
        vals = np.array([by_day[d].get(st, 0) for d in days_sorted], dtype=float)
        if vals.sum() == 0:
            continue
        ax.bar(days_sorted, vals, bottom=bottom, color=colors.get(st, "#888"), label=st)
        bottom += vals
    ax.axhline(96, color="k", lw=0.6, ls="--", alpha=0.5)
    ax.set_ylabel("count of 15-min anchors")
    ax.set_title(f"Per-day shard outcomes ({args.label}); dashed line = 96 = full day")
    ax.set_xticks(range(len(days_sorted)))
    ax.set_xticklabels(days_sorted, rotation=70, fontsize=7)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    per_day_png = args.figs_dir / "per_day_counts.png"
    fig.savefig(per_day_png, dpi=120)
    plt.close(fig)

    # ----- alignment histogram (offset_min for status==ok) -----
    offsets = [r["offset_min"] for r in records if r.get("offset_min") is not None]
    if offsets:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(offsets, bins=np.linspace(-7.5, 7.5, 31), color="#2c7fb8", edgecolor="black")
        ax.set_xlabel("file_time - anchor_time (min)")
        ax.set_ylabel("count")
        ax.set_title(f"Alignment offset distribution ({args.label})  n={len(offsets)}")
        fig.tight_layout()
        align_png = args.figs_dir / "alignment_hist.png"
        fig.savefig(align_png, dpi=120)
        plt.close(fig)
    else:
        align_png = None

    # ----- per-channel stats from manifest -----
    ch_norm_keys = ["ch0_norm", "ch1_norm", "ch2_norm"]
    ch_raw_keys = ["ch0_raw", "ch1_raw", "ch2_raw"]
    clip_keys = [
        ("ch0_below_lo", "ch0_above_hi"),
        ("ch1_below_lo", "ch1_above_hi"),
        ("ch2_below_lo", "ch2_above_hi"),
    ]

    # accumulate per-channel raw stats across the month (min/max are reductions)
    per_ch_raw_min = [math.inf] * 3
    per_ch_raw_max = [-math.inf] * 3
    per_ch_norm_min = [math.inf] * 3
    per_ch_norm_max = [-math.inf] * 3
    per_ch_mean_sum = [0.0] * 3  # weighted by 1 frame
    per_ch_median_list: list[list[float]] = [[], [], []]
    per_ch_clip_lo = [0] * 3
    per_ch_clip_hi = [0] * 3
    n_with_stats = 0
    n_pixels_total = 0
    pixels_per_frame = 100 * 100

    # per-day per-channel medians for drift plot
    per_day_med: dict[str, list[list[float]]] = defaultdict(lambda: [[], [], []])

    for r in records:
        st = r.get("stats") or {}
        if not st or "ch0_raw" not in st:
            continue
        n_with_stats += 1
        n_pixels_total += pixels_per_frame
        d = r["anchor_iso"][:10]
        for i in range(3):
            raw = st[ch_raw_keys[i]]
            norm = st[ch_norm_keys[i]]
            if not (math.isnan(raw["min"]) or math.isnan(raw["max"])):
                per_ch_raw_min[i] = min(per_ch_raw_min[i], raw["min"])
                per_ch_raw_max[i] = max(per_ch_raw_max[i], raw["max"])
            if not (math.isnan(norm["min"]) or math.isnan(norm["max"])):
                per_ch_norm_min[i] = min(per_ch_norm_min[i], norm["min"])
                per_ch_norm_max[i] = max(per_ch_norm_max[i], norm["max"])
            if not math.isnan(raw["mean"]):
                per_ch_mean_sum[i] += raw["mean"]
                per_day_med[d][i].append(raw["median"])
                per_ch_median_list[i].append(raw["median"])
            cc = st.get("clip_counts") or {}
            per_ch_clip_lo[i] += int(cc.get(clip_keys[i][0], 0))
            per_ch_clip_hi[i] += int(cc.get(clip_keys[i][1], 0))

    # all-zero / all-fill detection (raw min==max==fill or std==0 -> we proxy via raw min==max)
    n_flat = 0
    flat_examples: list[str] = []
    for r in records:
        st = r.get("stats") or {}
        if not st or "ch0_raw" not in st:
            continue
        flat_per_ch = []
        for k in ch_raw_keys:
            raw = st[k]
            flat_per_ch.append(
                (not math.isnan(raw["min"]))
                and (not math.isnan(raw["max"]))
                and (raw["min"] == raw["max"])
            )
        if all(flat_per_ch):
            n_flat += 1
            if len(flat_examples) < 8:
                flat_examples.append(r["anchor_iso"])

    # per-day per-channel median drift plot
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    ch_titles = [
        "ch0 raw VIS reflectance (ch1)  daily median",
        "ch1 raw IR window 10.7um BT [K] (ch4)  daily median",
        "ch2 raw water vapor 6.5um BT [K] (ch3)  daily median",
    ]
    for i, ax in enumerate(axes):
        meds = [float(np.median(per_day_med[d][i])) if per_day_med[d][i] else float("nan") for d in days_sorted]
        ax.plot(days_sorted, meds, marker="o", lw=1.2, color="#2c7fb8")
        ax.set_title(ch_titles[i])
        ax.grid(alpha=0.3)
    axes[-1].set_xticks(range(len(days_sorted)))
    axes[-1].set_xticklabels(days_sorted, rotation=70, fontsize=7)
    fig.tight_layout()
    drift_png = args.figs_dir / "per_day_median_drift.png"
    fig.savefig(drift_png, dpi=120)
    plt.close(fig)

    # ----- visual samples -----
    sample_anchors = []
    sample_anchors.append("2014-01-15T18:00:00Z")
    sample_anchors.append("2014-01-15T06:00:00Z")
    rng = np.random.default_rng(args.seed)
    ok_recs = [r for r in records if r["status"] == STATUS_OK and r.get("out_path")]
    daytime = [r for r in ok_recs if 16 <= int(r["anchor_iso"][11:13]) <= 23]  # 16-23 UTC = daytime CA
    if len(daytime) >= 2:
        idx = rng.choice(len(daytime), size=2, replace=False)
        sample_anchors.append(daytime[int(idx[0])]["anchor_iso"])
        sample_anchors.append(daytime[int(idx[1])]["anchor_iso"])

    rec_by_anchor = {r["anchor_iso"]: r for r in records}

    sample_pngs: list[tuple[str, Path | None, str]] = []
    ch_names = ["ch0 VIS (ch1) [0,1]", "ch1 IR 10.7um (ch4) -> [0,1]", "ch2 WV 6.5um (ch3) -> [0,1]"]
    for anchor in sample_anchors:
        rec = rec_by_anchor.get(anchor)
        out_path: Path | None = None
        note = ""
        if rec is None:
            sample_pngs.append((anchor, None, "anchor not in manifest"))
            continue
        if rec["status"] != STATUS_OK or not rec.get("out_path"):
            sample_pngs.append((anchor, None, f"status={rec['status']}"))
            continue
        npy_path = Path(rec["out_path"])
        if not npy_path.is_file():
            sample_pngs.append((anchor, None, "shard missing on disk"))
            continue
        arr = np.load(npy_path).astype(np.float32)
        fig, axs = plt.subplots(1, 3, figsize=(11, 3.5))
        for i, ax in enumerate(axs):
            im = ax.imshow(arr[i], cmap="gray" if i == 0 else "viridis", vmin=0, vmax=1, origin="lower")
            ax.set_title(f"{ch_names[i]}\nmin={arr[i].min():.3f} max={arr[i].max():.3f}")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"{anchor}  shape={arr.shape} dtype={np.load(npy_path).dtype}", y=1.02)
        fig.tight_layout()
        safe = anchor.replace(":", "").replace("-", "")
        sample_png = args.figs_dir / f"sample_{safe}.png"
        fig.savefig(sample_png, dpi=120, bbox_inches="tight")
        plt.close(fig)
        sample_pngs.append((anchor, sample_png, note))

    # ----- crop-shape sanity (first OK shard from disk) -----
    crop_check: dict[str, Any] = {}
    for r in ok_recs:
        op = Path(r["out_path"])
        if op.is_file():
            arr = np.load(op)
            crop_check = {
                "shard_path": str(op),
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "ch0_min": float(arr[0].min()),
                "ch0_max": float(arr[0].max()),
                "ch1_min": float(arr[1].min()),
                "ch1_max": float(arr[1].max()),
                "ch2_min": float(arr[2].min()),
                "ch2_max": float(arr[2].max()),
            }
            break

    # ----- write Markdown -----
    lines: list[str] = []
    lines.append(f"# Folsom GridSat-CONUS goes15 smoke test - {args.label}")
    lines.append("")
    lines.append(f"- Manifest: `{args.manifest}`")
    lines.append(f"- Shards under: `{args.shard_root}` (mirrored layout `<root>/YYYY/MM/goes15_YYYYMMDD_HHMM.npy`)")
    lines.append(f"- Anchors: 31 days x 96 anchors/day = {n_total} expected; {n_ok} produced ({n_ok / n_total * 100:.1f}%).")
    lines.append("")

    lines.append("## 1. Missing-frame rate")
    lines.append("")
    lines.append("Bucket counts across the month:")
    lines.append("")
    lines.append("| status | count | share |")
    lines.append("|---|---:|---:|")
    for st in STATUSES:
        c = status_counts.get(st, 0)
        lines.append(f"| {st} | {c} | {c / n_total * 100:.2f}% |")
    lines.append("")
    lines.append(f"Total non-ok (i.e. missing or degraded): **{n_missing}** out of {n_total} ({n_missing / n_total * 100:.2f}%).")
    lines.append("")
    lines.append(f"![per-day]({per_day_png.name})")
    lines.append("")
    lines.append("**Notes**: `listing_miss` means no goes15 file in the NCEI directory for that month was within the 7-min tolerance of the anchor. `empty` means the file was readable but ch1/3/4 had zero variance across the 100x100 crop. `download_error` / `parse_error` are transient or schema problems.")
    lines.append("")

    lines.append("## 2. Timestamp alignment")
    lines.append("")
    if align_png is not None:
        offs = np.array(offsets)
        lines.append(f"- n_matched = {len(offsets)}; mean offset = {offs.mean():+.3f} min; median = {np.median(offs):+.3f} min; |max| = {np.max(np.abs(offs)):.2f} min.")
        lines.append(f"- ![align]({align_png.name})")
    else:
        lines.append("- (no matched offsets in manifest)")
    lines.append("")

    lines.append("## 3. Crop shape sanity")
    lines.append("")
    if crop_check:
        lines.append("Loaded the first OK shard from disk:")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(crop_check, indent=2))
        lines.append("```")
    else:
        lines.append("(no OK shards on disk)")
    lines.append("")

    lines.append("## 4. Value ranges (raw, pre-clip; per channel across the whole month)")
    lines.append("")
    lines.append("| ch | role | clip range | observed raw min | observed raw max | overall mean (frame-mean avg) |")
    lines.append("|---|---|---|---:|---:|---:|")
    roles = ["ch1 VIS reflectance", "ch4 IR 10.7um BT [K]", "ch3 WV 6.5um BT [K]"]
    clips = ["[0, 1.2]", "[180, 330]", "[190, 270]"]
    for i in range(3):
        avg_mean = per_ch_mean_sum[i] / max(1, n_with_stats)
        lines.append(
            f"| {i} | {roles[i]} | {clips[i]} | {per_ch_raw_min[i]:.3f} | {per_ch_raw_max[i]:.3f} | {avg_mean:.3f} |"
        )
    lines.append("")
    lines.append("Per-channel clip rate (fraction of pixels touched by clip across the month):")
    lines.append("")
    lines.append("| ch | below clip lo | above clip hi |")
    lines.append("|---|---:|---:|")
    for i in range(3):
        lo_pct = per_ch_clip_lo[i] / max(1, n_pixels_total) * 100
        hi_pct = per_ch_clip_hi[i] / max(1, n_pixels_total) * 100
        lines.append(f"| {i} | {lo_pct:.4f}% | {hi_pct:.4f}% |")
    lines.append("")
    lines.append("Normalized [0,1] sanity (post-clip, per channel across month):")
    lines.append("")
    lines.append("| ch | norm min | norm max |")
    lines.append("|---|---:|---:|")
    for i in range(3):
        lines.append(f"| {i} | {per_ch_norm_min[i]:.4f} | {per_ch_norm_max[i]:.4f} |")
    lines.append("")
    lines.append(f"Frames flagged as flat (every channel constant across crop): **{n_flat}**.")
    if flat_examples:
        lines.append("")
        lines.append("Examples: " + ", ".join(f"`{a}`" for a in flat_examples))
    lines.append("")
    lines.append(f"![drift]({drift_png.name})")
    lines.append("")

    lines.append("## 5. Visual samples")
    lines.append("")
    for anchor, png, note in sample_pngs:
        if png is None:
            lines.append(f"- `{anchor}` -- skipped: {note}")
        else:
            lines.append(f"- `{anchor}`")
            lines.append("")
            lines.append(f"![{anchor}]({png.name})")
            lines.append("")

    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append(f"micromamba run -n luoyang python SPMF_preprocessing/folsom/gridsat_to_shards.py \\")
    lines.append(f"    --start 2014-01-01 --end 2014-01-31 \\")
    lines.append(f"    --out-root {args.shard_root} \\")
    lines.append("    --workers 8")
    lines.append(f"micromamba run -n luoyang python SPMF_preprocessing/folsom/report_smoke.py \\")
    lines.append(f"    --manifest {args.manifest} --shard-root {args.shard_root} \\")
    lines.append(f"    --report {args.report} --figs-dir {args.figs_dir}")
    lines.append("```")
    lines.append("")

    args.report.write_text("\n".join(lines))
    print(f"wrote {args.report} and {len(sample_pngs)} sample figs to {args.figs_dir}")


if __name__ == "__main__":
    main()
