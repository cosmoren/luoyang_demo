#!/usr/bin/env python3
"""
Resize sky-camera images (``.jpg`` / ``.jpeg`` / ``.png``) and normalize filenames from timestamps.

Source layouts (under --src, default /data/skyimg):

  A) One extra group level + device folder (e.g. ASI):
       <src>/asi/asi_16613/YYYYMMDD/*.{jpg,jpeg,png}

  B) Date folders directly under the group (e.g. WYLC):
       <src>/wylc/YYYYMMDD/*.{jpg,jpeg,png}

Input filenames encode time in **Asia/Shanghai (UTC+8)** wall clock (14-digit
``YYYYMMDDHHMMSS`` and related patterns).

Output filenames under ``skyimg_train`` use **UTC** wall clock by default:
``YYYYMMDDHHMMSS`` is the same instant expressed in UTC (``--filename-tz utc``,
the default). Use ``--filename-tz asia`` only if you want the output digits to
stay in UTC+8 (not recommended for UTC-aligned training data).

Output mirrors the same relative layout under --dst (default /data/skyimg_train),
with no date subfolders — only resized JPEG outputs:

       <dst>/asi/asi_16613/*.jpg
       <dst>/wylc/*.jpg

ASI-style names ``YYYYMMDDHHMMSS_11.*`` / ``_12.*``: only ``_12`` inputs are kept;
``_11`` files are ignored. Output stem is ``YYYYMMDDHHMMSS`` only (no ``_11``/``_12``),
unless you set ``--suffix``. PNG inputs are converted to RGB and saved as JPEG.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    from PIL import Image

    _RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS
except ImportError as e:
    raise SystemExit("pip install Pillow") from e

SH = ZoneInfo("Asia/Shanghai")
UTC = ZoneInfo("UTC")

RE_ASI = re.compile(r"^(\d{14})_(11|12)$", re.IGNORECASE)
RE_14 = re.compile(r"^(\d{14})$")
RE_D8 = re.compile(r"^\d{8}$")

INPUT_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})


def is_asi_camera_11_stem(stem: str) -> bool:
    """True for ASI second-camera slot _11 (skip these files)."""
    m = RE_ASI.match(stem)
    return m is not None and m.group(2).lower() == "11"


def parse_stem(stem: str, ddir: str | None) -> datetime | None:
    """Return naive datetime = Asia/Shanghai wall time from filename (not UTC)."""
    m = RE_ASI.match(stem)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    m = RE_14.match(stem)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    m = re.match(r"^ori(\d{14})$", stem, re.IGNORECASE)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    if ddir and RE_D8.match(ddir) and re.fullmatch(r"\d{6}", stem):
        return datetime.strptime(ddir + stem, "%Y%m%d%H%M%S")
    g = re.search(r"(\d{14})", stem)
    if g:
        try:
            return datetime.strptime(g.group(1), "%Y%m%d%H%M%S")
        except ValueError:
            return None
    return None


def to_stem(dt: datetime, tz: str, suf: str) -> str:
    """``dt`` is naive Asia/Shanghai; for tz=='utc' write UTC wall clock in stem."""
    loc = dt.replace(tzinfo=SH)
    w = loc.astimezone(UTC).replace(tzinfo=None) if tz == "utc" else dt
    return w.strftime("%Y%m%d%H%M%S") + suf


def image_list(site_root: Path) -> list[tuple[Path, str | None]]:
    """Paths under site_root; date_folder = first path part if YYYYMMDD."""
    out: list[tuple[Path, str | None]] = []
    for p in site_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in INPUT_SUFFIXES:
            continue
        rel = p.relative_to(site_root)
        parts = rel.parts
        dd = parts[0] if len(parts) >= 2 and RE_D8.match(parts[0]) else None
        out.append((p, dd))
    return out


def _child_dirs(d: Path) -> list[Path]:
    return sorted(x for x in d.iterdir() if x.is_dir())


def _all_children_are_date_dirs(d: Path) -> bool:
    subs = _child_dirs(d)
    return bool(subs) and all(RE_D8.match(x.name) for x in subs)


def _has_image_under(root: Path) -> bool:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in INPUT_SUFFIXES:
            return True
    return False


def iter_site_jobs(src: Path) -> list[tuple[Path, tuple[str, ...]]]:
    """
    Each job: (site_root, dst_rel_parts).
    dst_rel_parts is e.g. ('asi', 'asi_16613') or ('wylc',) or ('train',) for split folders.
    Top-level folders that only hold images (e.g. train/, test/) become separate jobs under dst.
    If the tree matches no pattern but contains images, one job uses ``(src, (basename,))``.
    """
    jobs: list[tuple[Path, tuple[str, ...]]] = []
    groups = sorted(p for p in src.iterdir() if p.is_dir())
    for group in groups:
        if _all_children_are_date_dirs(group):
            jobs.append((group, (group.name,)))
            continue
        added_leaf = False
        for leaf in _child_dirs(group):
            if RE_D8.match(leaf.name):
                continue
            jobs.append((leaf, (group.name, leaf.name)))
            added_leaf = True
        if not added_leaf and _has_image_under(group):
            jobs.append((group, (group.name,)))
    if not jobs and _has_image_under(src):
        label = src.resolve().name or "root"
        jobs.append((src, (label,)))
    return jobs


def site_matches(dst_rel: tuple[str, ...], filters: list[str] | None) -> bool:
    if not filters:
        return True
    rel = "/".join(dst_rel)
    for raw in filters:
        fn = raw.strip().strip("/")
        if not fn:
            continue
        if fn == rel or rel.endswith("/" + fn) or (dst_rel and dst_rel[-1] == fn):
            return True
        if "/" in fn and rel == fn:
            return True
    return False


def run_site(
    site_root: Path,
    dst_dir: Path,
    size: int,
    ftz: str,
    suf: str,
    dry: bool,
) -> tuple[int, int, int]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    ok, sk, n_skip_11 = 0, 0, 0
    for path, dd in image_list(site_root):
        if is_asi_camera_11_stem(path.stem):
            n_skip_11 += 1
            continue
        dt = parse_stem(path.stem, dd)
        if not dt:
            print(f"  skip: {path}")
            sk += 1
            continue
        ns = to_stem(dt, ftz, suf)
        on = ns + ".jpg"
        b, n = ns, 1
        while on in seen:
            n += 1
            ns = f"{b}_{n}"
            on = ns + ".jpg"
        seen.add(on)
        out = dst_dir / on
        if dry:
            print(f"  would {out} <- {path}")
            ok += 1
            continue
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                im = im.resize((size, size), _RESAMPLE)
                im.save(out, "JPEG", quality=95)
            ok += 1
        except Exception as e:
            print(f"  err {path}: {e}")
            sk += 1
    if n_skip_11:
        print(f"  ignored {n_skip_11} ASI *_11 inputs")
    return ok, sk, n_skip_11


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--src",
        type=Path,
        default=Path("/data/skyimg"),
        help="Root with group folders (asi, wylc, …).",
    )
    ap.add_argument(
        "--dst",
        type=Path,
        default=Path("/data/skyimg_train"),
        help="Output root; mirrors <group>/<leaf>/ under src.",
    )
    ap.add_argument("--size", type=int, default=224, help="Square side length after resize.")
    ap.add_argument(
        "--filename-tz",
        choices=("utc", "asia"),
        default="utc",
        help=(
            "Output stem timezone (default utc): "
            "utc = convert UTC+8 filename time to UTC; "
            "asia = keep UTC+8 digits in output name."
        ),
    )
    ap.add_argument(
        "--suffix",
        default="",
        help='Optional stem suffix before .jpg (default: none → YYYYMMDDHHMMSS.jpg only).',
    )
    ap.add_argument(
        "--sites",
        nargs="*",
        default=None,
        metavar="FILTER",
        help="Optional: only jobs whose output path matches (e.g. asi_16613 or asi/asi_16613).",
    )
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    src = a.src.expanduser().resolve()
    dst = a.dst.expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"bad src: {src}")

    jobs = iter_site_jobs(src)
    jobs = [(root, rel) for root, rel in jobs if site_matches(rel, a.sites)]
    if not jobs:
        raise SystemExit("no site jobs (check --src and --sites)")

    print(
        f"src={src}\ndst={dst}\nsize={a.size} "
        f"filename_tz={a.filename_tz} suffix={a.suffix!r}\n"
    )
    total_ok = total_skip = total_11 = 0
    for site_root, rel_parts in jobs:
        label = "/".join(rel_parts)
        dst_dir = dst.joinpath(*rel_parts)
        print(f"Job {label} ({site_root}) -> {dst_dir}")
        o, k, n11 = run_site(
            site_root,
            dst_dir,
            a.size,
            a.filename_tz,
            a.suffix,
            a.dry_run,
        )
        print(f"  ok {o} skip {k}")
        total_ok += o
        total_skip += k
        total_11 += n11

    print(f"Total ok {total_ok} skip {total_skip}  (ignored ASI *_11: {total_11})")


if __name__ == "__main__":
    main()
