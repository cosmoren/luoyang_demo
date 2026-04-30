#!/usr/bin/env python3
"""
Resize sky-camera images (``.jpg`` / ``.jpeg`` / ``.png``) and rename by timestamp:
**Pacific local time in filenames → UTC** stems.

Uses one IANA zone (default ``America/Los_Angeles``). PST vs PDT is **not** read from the name;
the zone database applies the correct offset for each instant.

Same directory layout as ``rename_utc.py`` (see that script). Input patterns match
``YYYYMMDDHHMMSS`` / ASI ``*_12`` / ``ori…`` / date-folder + ``HHMMSS``, etc.

Output: ``YYYYMMDDHHMMSS.jpg`` in **UTC** wall clock (+ optional ``--suffix``), resized to ``--size``
(inputs are converted to RGB and saved as JPEG).

If ``--src`` has subfolders like ``train/`` and ``test/`` with images only inside (no ASI-style
nesting), each becomes its own job: ``<dst>/train/``, ``<dst>/test/``, etc.

If ``--src`` is a single flat folder of images, one job goes to ``<dst>/<basename(src)>/``.
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

UTC = ZoneInfo("UTC")

RE_ASI = re.compile(r"^(\d{14})_(11|12)$", re.IGNORECASE)
RE_14 = re.compile(r"^(\d{14})$")
RE_D8 = re.compile(r"^\d{8}$")

INPUT_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})


def is_asi_camera_11_stem(stem: str) -> bool:
    m = RE_ASI.match(stem)
    return m is not None and m.group(2).lower() == "11"


def parse_stem(stem: str, ddir: str | None) -> datetime | None:
    """Naive datetime = Pacific **local civil** time from filename (not UTC)."""
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


def pacific_naive_to_utc_stem(dt_naive: datetime, tz: ZoneInfo, suf: str) -> str:
    """Interpret naive ``dt`` as local time in ``tz`` (DST from zone rules); return UTC stem + suf."""
    dt_local = dt_naive.replace(tzinfo=tz)
    dt_utc = dt_local.astimezone(UTC).replace(tzinfo=None)
    return dt_utc.strftime("%Y%m%d%H%M%S") + suf


def image_list(site_root: Path) -> list[tuple[Path, str | None]]:
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
        # e.g. train/ or test/ with images at this level only (no device leaf dirs)
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
    tz: ZoneInfo,
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
        try:
            ns = pacific_naive_to_utc_stem(dt, tz, suf)
        except Exception as e:
            print(f"  skip (time zone / DST): {path} ({e})")
            sk += 1
            continue
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
        default=Path("/data/skippd/skippd_images/"),
        help="Root with group folders (same layout as rename_utc.py).",
    )
    ap.add_argument(
        "--dst",
        type=Path,
        default=Path("/data/skippd/skippd_images_utc/"),
        help="Output root; mirrors <group>/<leaf>/ under src.",
    )
    ap.add_argument("--size", type=int, default=224, help="Square side length after resize.")
    ap.add_argument(
        "--zone",
        type=str,
        default="America/Los_Angeles",
        help="IANA tz for filename times (PST/PDT from rules). E.g. America/Vancouver.",
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
    try:
        la_tz = ZoneInfo(a.zone)
    except Exception as e:
        raise SystemExit(f"bad --zone {a.zone!r}: {e}") from e

    src = a.src.expanduser().resolve()
    dst = a.dst.expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"bad src: {src}")

    jobs = iter_site_jobs(src)
    jobs = [(root, rel) for root, rel in jobs if site_matches(rel, a.sites)]
    if not jobs:
        raise SystemExit("no site jobs (check --src and --sites)")

    print(f"src={src}\ndst={dst}\nsize={a.size} zone={a.zone!r} suffix={a.suffix!r}\n")

    total_ok = total_skip = total_11 = 0
    for site_root, rel_parts in jobs:
        label = "/".join(rel_parts)
        dst_dir = dst.joinpath(*rel_parts)
        print(f"Job {label} ({site_root}) -> {dst_dir}")
        o, k, n11 = run_site(site_root, dst_dir, a.size, la_tz, a.suffix, a.dry_run)
        print(f"  ok {o} skip {k}")
        total_ok += o
        total_skip += k
        total_11 += n11

    print(f"Total ok {total_ok} skip {total_skip}  (ignored ASI *_11: {total_11})")


if __name__ == "__main__":
    main()
