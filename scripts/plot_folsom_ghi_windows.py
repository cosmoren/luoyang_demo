"""Plot 12 Folsom GHI windows (one per season-year) paired with clear-sky-index (CSI) panels.

The Folsom dataset spans 2014-01-02 -> 2016-12-31 (3 years). For each (season, year)
bucket -- 4 seasons x 3 years = 12 buckets -- one valid 10-day window start is sampled
uniformly with ``np.random.default_rng(42)`` and the full 10-day window is required to
stay inside the dataset.

Each window is written to its own PNG under ``outputs/folsom_windows/`` with two
side-by-side axes (1 row x 2 cols): GHI on the left, CSI on the right. Filenames are
``window_<NN>_<YYYY-MM-DD>_<season>.png`` with NN running 01..12 in chronological order.

Season convention: months are bucketed as Winter={Dec, Jan, Feb}, Spring={Mar, Apr, May},
Summer={Jun, Jul, Aug}, Fall={Sep, Oct, Nov}. For year-tagging we use the simplest
calendar-year mapping: "Winter <year>" = Jan, Feb, AND Dec of that same year (i.e.
December is bucketed with the same year, not the next one). With the 2014-2016 dataset
this gives every (season, year) bucket plenty of valid 10-day starts.

CSI uses pvlib's Ineichen clear-sky model with the same
``Location(lat, lon).get_clearsky(..., model="ineichen")`` convention as
``dataloader.folsom._compute_folsom_p_cs``: lat/lon come from ``<data_dir>/info.yaml``,
and naive CSV timestamps are localized to UTC before pvlib.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_CONF_PATH = PROJECT_ROOT / "config" / "datasets" / "conf_folsom.yaml"
OUT_DIR = PROJECT_ROOT / "outputs" / "folsom_windows"

WINDOW_DAYS = 10
SEED = 42

GHI_REF = 1000.0           # W/m^2 reference for the dashed horizontal line on the GHI panel
CSI_CLAMP_MAX = 1.5        # raised from 1.2 so cloud-enhancement spikes are visible (still bounded)
CSI_DAYTIME_P_CS_THRESH = 0.05   # clearsky_ghi/1000 > 0.05  ->  ~50 W/m^2 daytime gate
FOLSOM_GHI_SCALE = 1000.0  # mirrors dataloader.folsom._FOLSOM_GHI_SCALE

# Winter <year> = {Jan, Feb, Dec} of that same calendar year. (Simplest mapping;
# December is bucketed with the year it occurs in, not the following winter.)
SEASON_MONTHS = {
    "Winter": {12, 1, 2},
    "Spring": {3, 4, 5},
    "Summer": {6, 7, 8},
    "Fall":   {9, 10, 11},
}
SEASON_ORDER = ("Winter", "Spring", "Summer", "Fall")


def resolve_folsom_site() -> tuple[float, float, Path, Path]:
    """Read Folsom CSV path + site lat/lon following the project's convention.

    Resolution path mirrors ``FolsomIrradianceDataset.__init__``:
      * ``config/datasets/conf_folsom.yaml`` -> ``paths.data_dir``
      * ``paths.data_dir / paths.folsom_irradiance_csv`` (or ``paths.data_dir / paths.pv_path``
        glob fallback for the single Folsom irradiance CSV)
      * ``<data_dir>/info.yaml`` -> ``site.latitude``, ``site.longitude``

    Returns ``(lat, lon, csv_path, info_yaml_path)``.
    """
    with DATASET_CONF_PATH.open() as f:
        conf = yaml.safe_load(f) or {}
    paths_cfg = conf.get("paths") or {}
    data_dir_raw = paths_cfg.get("data_dir")
    if not data_dir_raw:
        raise KeyError(f"paths.data_dir missing in {DATASET_CONF_PATH}")
    data_dir = Path(data_dir_raw)
    if not data_dir.is_absolute():
        data_dir = (PROJECT_ROOT / data_dir).resolve()

    info_path = data_dir / "info.yaml"
    if not info_path.is_file():
        raise FileNotFoundError(f"site info file not found: {info_path}")
    with info_path.open() as f:
        info = yaml.safe_load(f) or {}
    site = info.get("site") or {}
    lat = site.get("latitude")
    lon = site.get("longitude")
    if lat is None or lon is None:
        raise KeyError(f"{info_path} must define both site.latitude and site.longitude")

    csv_rel = paths_cfg.get("folsom_irradiance_csv")
    if csv_rel and str(csv_rel).strip():
        rel_p = Path(str(csv_rel).strip())
        csv_path = rel_p.resolve() if rel_p.is_absolute() else (data_dir / rel_p).resolve()
    else:
        pv_rel = paths_cfg.get("pv_path") or "irradiance"
        pv_dir = (data_dir / pv_rel).resolve()
        cands = sorted(pv_dir.glob("*.csv"))
        if not cands:
            raise FileNotFoundError(f"no CSV found under {pv_dir}")
        if len(cands) != 1:
            raise RuntimeError(f"expected exactly one CSV under {pv_dir}, got {len(cands)}")
        csv_path = cands[0]
    if not csv_path.is_file():
        raise FileNotFoundError(f"Folsom irradiance CSV not found: {csv_path}")

    return float(lat), float(lon), csv_path, info_path


def compute_clearsky_ghi(lat: float, lon: float, idx: pd.DatetimeIndex) -> np.ndarray:
    """Ineichen clearsky GHI in W/m^2 for the given timestamps.

    Mirrors ``dataloader.folsom._compute_folsom_p_cs``: ``Location(lat, lon)`` (no
    altitude override -- the dataloader doesn't pass one either), tz-naive timestamps
    are localized to UTC before ``get_clearsky``.
    """
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")
    loc = pvlib.location.Location(float(lat), float(lon))
    cs = loc.get_clearsky(idx, model="ineichen")
    return np.asarray(cs["ghi"].values, dtype=np.float64)


def csi_from_ghi(ghi: np.ndarray, clearsky_ghi: np.ndarray) -> np.ndarray:
    """CSI = actual / clearsky, gated daytime-only (clearsky/1000 > 0.05) and clipped [0, CSI_CLAMP_MAX].

    Returns NaN where the daytime gate fails so matplotlib draws gaps instead of spikes.
    """
    ghi = np.asarray(ghi, dtype=np.float64)
    cs = np.asarray(clearsky_ghi, dtype=np.float64)
    daytime = (cs / FOLSOM_GHI_SCALE) > CSI_DAYTIME_P_CS_THRESH
    safe_cs = np.where(daytime & np.isfinite(cs), cs, np.nan)
    csi = ghi / safe_cs
    csi = np.where(daytime & np.isfinite(csi), csi, np.nan)
    return np.clip(csi, 0.0, CSI_CLAMP_MAX)


def pick_windows(
    candidate_starts: pd.DatetimeIndex,
    years: list[int],
    rng: np.random.Generator,
) -> tuple[list[pd.Timestamp], list[str], list[int]]:
    """Sample one valid 10-day window start per (season, year) bucket.

    Iteration order is (year asc, season in SEASON_ORDER) so the rng draws are
    deterministic for a given seed. A bucket is valid iff at least one candidate
    start has month in ``SEASON_MONTHS[season]`` AND year == bucket year.
    """
    chosen: list[pd.Timestamp] = []
    chosen_seasons: list[str] = []
    chosen_years: list[int] = []

    for year in years:
        for season in SEASON_ORDER:
            months = SEASON_MONTHS[season]
            cand = [d for d in candidate_starts if d.year == year and d.month in months]
            if not cand:
                raise RuntimeError(
                    f"no candidate {WINDOW_DAYS}-day window starts in bucket "
                    f"({season}, {year}); months={sorted(months)}"
                )
            pick = cand[int(rng.integers(len(cand)))]
            chosen.append(pick)
            chosen_seasons.append(season)
            chosen_years.append(year)

    return chosen, chosen_seasons, chosen_years


def main() -> None:
    lat, lon, csv_path, info_path = resolve_folsom_site()
    print(f"site source: {info_path}")
    print(f"  latitude={lat}  longitude={lon}  (no altitude in info.yaml; "
          f"pvlib.Location uses altitude=0, matching dataloader.folsom convention)")
    print(f"csv: {csv_path}")

    df = pd.read_csv(csv_path, usecols=["timeStamp", "ghi"])
    df["timeStamp"] = pd.to_datetime(df["timeStamp"], format="%Y-%m-%d %H:%M:%S")
    df = df.sort_values("timeStamp").reset_index(drop=True)

    diffs = df["timeStamp"].diff().dropna()
    median_dt = diffs.median()
    span_days = (df["timeStamp"].iloc[-1] - df["timeStamp"].iloc[0]).total_seconds() / 86400.0
    print(f"rows={len(df):,}  median_dt={median_dt}  span={span_days:.1f} days")
    print(f"range: {df['timeStamp'].iloc[0]} -> {df['timeStamp'].iloc[-1]}")
    if median_dt != pd.Timedelta(minutes=1):
        print(f"WARNING: median dt is {median_dt}, expected 1 min")
    if not (2 * 365 < span_days < 4 * 365):
        print(f"WARNING: span {span_days:.1f} days is not roughly 3 years")

    last_day = df["timeStamp"].iloc[-1].normalize()
    first_day = df["timeStamp"].iloc[0].normalize()
    last_valid_start = last_day - pd.Timedelta(days=WINDOW_DAYS - 1)
    candidate_starts = pd.date_range(first_day, last_valid_start, freq="D")

    years = sorted({int(ts.year) for ts in candidate_starts})
    expected_n = len(years) * len(SEASON_ORDER)
    print(f"\nyears in dataset: {years}  (expecting {expected_n} = "
          f"{len(years)} years x {len(SEASON_ORDER)} seasons windows)")

    rng = np.random.default_rng(SEED)
    chosen, chosen_seasons, chosen_years = pick_windows(candidate_starts, years, rng)

    order = sorted(range(len(chosen)), key=lambda k: chosen[k])
    chosen = [chosen[k] for k in order]
    chosen_seasons = [chosen_seasons[k] for k in order]
    chosen_years = [chosen_years[k] for k in order]

    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            gap = abs((chosen[i] - chosen[j]).days)
            assert gap >= WINDOW_DAYS, (
                f"window starts {chosen[i].date()} and {chosen[j].date()} are only "
                f"{gap} days apart (< WINDOW_DAYS={WINDOW_DAYS}); season-year buckets "
                f"should naturally separate them, so this indicates a bug"
            )

    print(f"\nchosen start dates (chronological, {len(chosen)} total):")
    for s, season, yr in zip(chosen, chosen_seasons, chosen_years):
        print(f"  {s.date()}  ({season} {yr})")

    df = df.set_index("timeStamp")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total_pts = 0
    total_over = 0
    max_ghi = -np.inf
    per_window_stats: list[tuple[str, str, int, float, float, float, float, int]] = []
    all_daytime_csi: list[np.ndarray] = []
    written_paths: list[Path] = []
    n_windows_with_enhancement = 0  # windows with at least one daytime CSI > 1.0
    global_max_csi = -np.inf         # post-clip-raise (so bounded by CSI_CLAMP_MAX)

    for i, (start, season, yr) in enumerate(zip(chosen, chosen_seasons, chosen_years), start=1):
        end = start + pd.Timedelta(days=WINDOW_DAYS)
        chunk = df.loc[start:end - pd.Timedelta(minutes=1), "ghi"]
        n_chunk = len(chunk)
        total_pts += n_chunk
        n_over = int((chunk > GHI_REF).sum())
        total_over += n_over
        win_max_ghi = float(chunk.max()) if n_chunk else float("nan")
        if n_chunk:
            max_ghi = max(max_ghi, win_max_ghi)
        win_pct_over = 100.0 * n_over / max(n_chunk, 1)

        ghi_vals = chunk.to_numpy(dtype=np.float64)
        ts_idx = pd.DatetimeIndex(chunk.index)
        cs_ghi = compute_clearsky_ghi(lat, lon, ts_idx)
        csi = csi_from_ghi(ghi_vals, cs_ghi)
        csi_plot = np.where(np.isfinite(csi), csi, 0.0)
        finite = np.isfinite(csi)
        if finite.any():
            csi_mean = float(np.nanmean(csi))
            csi_std = float(np.nanstd(csi))
            n_day = int(finite.sum())
            all_daytime_csi.append(csi[finite])
            win_max_csi = float(np.nanmax(csi))
            if win_max_csi > 1.0:
                n_windows_with_enhancement += 1
            global_max_csi = max(global_max_csi, win_max_csi)
        else:
            csi_mean = float("nan")
            csi_std = float("nan")
            n_day = 0

        per_window_stats.append(
            (str(start.date()), f"{season} {yr}", n_chunk,
             win_max_ghi, win_pct_over, csi_mean, csi_std, n_day)
        )

        fig, (ax_ghi, ax_csi) = plt.subplots(
            1, 2, figsize=(14, 4), constrained_layout=True
        )

        ax_ghi.plot(chunk.index, ghi_vals, linewidth=0.7, color="tab:blue")
        ax_ghi.axhline(GHI_REF, linestyle="--", color="tab:red", linewidth=0.9,
                       label=f"{GHI_REF:.0f} W/m²")
        ax_ghi.set_title(f"{start.date()} ({season} {yr}) - GHI")
        ax_ghi.set_ylabel("GHI [W/m²]")
        ax_ghi.set_ylim(bottom=-20)
        ax_ghi.grid(alpha=0.3)
        ax_ghi.legend(loc="upper right", fontsize=8)
        ax_ghi.set_xlim(start, end)
        ax_ghi.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax_ghi.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax_ghi.tick_params(axis="x", labelsize=8)

        ax_csi.plot(chunk.index, csi_plot, linewidth=0.9, color="tab:orange")
        ax_csi.axhline(1.0, linestyle="--", color="gray", linewidth=0.9)
        ax_csi.set_title(f"{start.date()} ({season} {yr}) - CSI")
        ax_csi.set_ylabel("CSI")
        ax_csi.set_ylim(0.0, 1.5)
        ax_csi.set_yticks([0.0, 0.5, 1.0, 1.5])
        ax_csi.grid(alpha=0.3)
        ax_csi.set_xlim(start, end)
        ax_csi.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax_csi.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax_csi.tick_params(axis="x", labelsize=8)

        fname = f"window_{i:02d}_{start.date()}_{season}.png"
        out_path = OUT_DIR / fname
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        written_paths.append(out_path)

    pct_over_global = 100.0 * total_over / max(total_pts, 1)
    n_windows = len(chosen)
    print(
        f"\npoints per window (approx): {total_pts // max(n_windows, 1):,}  "
        f"(total across {n_windows} windows: {total_pts:,})"
    )
    print(f"% points exceeding {GHI_REF:.0f} W/m^2 (combined): {pct_over_global:.3f}%")
    print(f"max GHI seen across windows: {max_ghi:.2f} W/m^2")

    print("\nper-window stats (chronological):")
    header = (f"  {'start':<12} {'bucket':<13} {'n_pts':>8} {'max_ghi':>10} "
              f"{'%>1000':>8} {'csi_mean':>10} {'csi_std':>10} {'n_day':>10}")
    print(header)
    for date, label, n_pts, mx, pct, m, s, n_day in per_window_stats:
        mx_s = "    nan  " if not np.isfinite(mx) else f"{mx:10.2f}"
        m_s = "  nan  " if not np.isfinite(m) else f"{m:10.4f}"
        s_s = "  nan  " if not np.isfinite(s) else f"{s:10.4f}"
        print(f"  {date:<12} {label:<13} {n_pts:>8,} {mx_s} {pct:>7.3f}% {m_s} {s_s} {n_day:>10,}")

    if all_daytime_csi:
        flat = np.concatenate(all_daytime_csi)
        g_mean = float(flat.mean())
        g_std = float(flat.std())
        print(
            f"\nglobal daytime-only CSI across all {n_windows} windows: "
            f"mean={g_mean:.4f}  std={g_std:.4f}  n={int(flat.size):,}"
        )
        gmx_s = "nan" if not np.isfinite(global_max_csi) else f"{global_max_csi:.4f}"
        print(
            f"cloud-enhancement summary (post-clip-raise, CSI_CLAMP_MAX={CSI_CLAMP_MAX}): "
            f"{n_windows_with_enhancement}/{n_windows} windows with at least one CSI > 1.0; "
            f"global max CSI = {gmx_s}"
        )
    else:
        print("\nno daytime CSI samples across any window (unexpected)")

    print(f"\nwrote {len(written_paths)} PNGs to: {OUT_DIR}")
    for p in written_paths:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
