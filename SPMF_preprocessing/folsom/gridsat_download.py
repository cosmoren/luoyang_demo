"""
Build the URL list for NCEI GridSat-CONUS goes15 imagery aligned to a fixed 15-min UTC grid.

The native NCEI cadence is mixed (5/15/30 min). For each anchor in {:00, :15, :30, :45} of every UTC
hour we keep at most one file: the one whose stamped time is closest to the anchor and within
``tolerance_min`` (default 7). Anchors with no candidate within tolerance are reported as missing.

This module is import-safe and also runs as a CLI for inspection:

    python gridsat_download.py --start 2014-01-01 --end 2014-01-31 [--out anchors.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

NCEI_ROOT = "https://www.ncei.noaa.gov/data/gridsat-goes/access/conus"
SAT = "goes15"
DEFAULT_TOLERANCE_MIN = 7

_FNAME_RE = re.compile(
    r"GridSat-CONUS\.goes15\.(?P<y>\d{4})\.(?P<m>\d{2})\.(?P<d>\d{2})\.(?P<hhmm>\d{4})\.v01\.nc"
)


@dataclass
class AnchorMatch:
    """One target 15-min anchor and (optionally) the closest available file."""

    anchor_iso: str  # UTC, e.g. "2014-01-15T18:00:00Z"
    file_iso: str | None  # UTC of the actual file picked, or None
    url: str | None
    offset_min: float | None  # signed: file_dt - anchor_dt, in minutes

    @property
    def is_match(self) -> bool:
        return self.url is not None


def _list_month(year: int, month: int, session: requests.Session, timeout: float = 60.0) -> dict[datetime, str]:
    """Return {file_dt_utc: full_url} for every goes15 file in the given month directory."""
    url = f"{NCEI_ROOT}/{year:04d}/{month:02d}/"
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    out: dict[datetime, str] = {}
    seen = set()
    for m in _FNAME_RE.finditer(r.text):
        name = m.group(0)
        if name in seen:
            continue
        seen.add(name)
        y = int(m.group("y"))
        mo = int(m.group("m"))
        d = int(m.group("d"))
        hhmm = m.group("hhmm")
        hh = int(hhmm[:2])
        mm = int(hhmm[2:])
        dt = datetime(y, mo, d, hh, mm, tzinfo=timezone.utc)
        out[dt] = f"{url}{name}"
    return out


def _iter_anchors(start: datetime, end_exclusive: datetime) -> list[datetime]:
    """List 15-min anchors [start, end_exclusive)."""
    cur = start
    step = timedelta(minutes=15)
    res = []
    while cur < end_exclusive:
        res.append(cur)
        cur += step
    return res


def build_anchor_matches(
    start_utc: datetime,
    end_utc_exclusive: datetime,
    tolerance_min: float = DEFAULT_TOLERANCE_MIN,
    session: requests.Session | None = None,
) -> list[AnchorMatch]:
    """For each 15-min UTC anchor in [start, end), find the closest goes15 file within tolerance.

    Both bounds must be timezone-aware UTC. Listings are fetched per (year, month) and cached in
    memory for the duration of the call.
    """
    if start_utc.tzinfo is None or end_utc_exclusive.tzinfo is None:
        raise ValueError("start_utc and end_utc_exclusive must be timezone-aware (UTC)")

    sess = session or requests.Session()
    months_seen: dict[tuple[int, int], dict[datetime, str]] = {}

    def _listing(year: int, month: int) -> dict[datetime, str]:
        key = (year, month)
        if key not in months_seen:
            months_seen[key] = _list_month(year, month, sess)
        return months_seen[key]

    anchors = _iter_anchors(start_utc, end_utc_exclusive)
    out: list[AnchorMatch] = []
    tol = timedelta(minutes=tolerance_min)

    for a in anchors:
        # candidates can come from anchor's month or, near boundaries, neighboring month
        cand_months = {(a.year, a.month)}
        if a.day == 1 and a.hour == 0 and a.minute < 30:
            prev = (a - timedelta(days=1))
            cand_months.add((prev.year, prev.month))
        if a == _iter_anchors(start_utc, end_utc_exclusive)[-1]:
            nxt = a + timedelta(days=1)
            cand_months.add((nxt.year, nxt.month))

        best: tuple[datetime, str, timedelta] | None = None
        for ym in cand_months:
            listing = _listing(*ym)
            # narrow to anchor day/hour neighborhood
            lo = a - tol
            hi = a + tol
            for ft, url in listing.items():
                if lo <= ft <= hi:
                    delta = ft - a
                    if best is None or abs(delta) < abs(best[2]):
                        best = (ft, url, delta)

        if best is None:
            out.append(
                AnchorMatch(
                    anchor_iso=a.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    file_iso=None,
                    url=None,
                    offset_min=None,
                )
            )
        else:
            ft, url, delta = best
            out.append(
                AnchorMatch(
                    anchor_iso=a.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    file_iso=ft.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    url=url,
                    offset_min=delta.total_seconds() / 60.0,
                )
            )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="List NCEI GridSat-CONUS goes15 URLs aligned to 15-min anchors")
    p.add_argument("--start", required=True, help="UTC start date, YYYY-MM-DD (inclusive)")
    p.add_argument("--end", required=True, help="UTC end date, YYYY-MM-DD (inclusive)")
    p.add_argument("--tolerance-min", type=float, default=DEFAULT_TOLERANCE_MIN)
    p.add_argument("--out", type=Path, default=None, help="Optional JSON output for the anchor table")
    args = p.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_excl = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)

    matches = build_anchor_matches(start, end_excl, tolerance_min=args.tolerance_min)
    n_match = sum(1 for m in matches if m.is_match)
    n_total = len(matches)
    print(f"anchors: {n_total}  matched: {n_match}  missing: {n_total - n_match}", file=sys.stderr)

    if args.out:
        args.out.write_text(json.dumps([asdict(m) for m in matches], indent=2))
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        for m in matches:
            print(json.dumps(asdict(m)))


if __name__ == "__main__":
    main()
