# Satellite Imagery Sourcing Playbook (Folsom → SKIPP'D)

A short handoff covering how we found GOES-15 imagery for the Folsom PV dataset, the lessons that generalize to any site, and the concrete recommendation for adding satellite to SKIPP'D.

## TL;DR

For 2014–2016 Folsom we ended up on **NCEI GridSat-CONUS goes15** (public HTTPS, no auth, 5–15 min cadence, already calibrated and reprojected). The key lesson: do **not** confuse "no L2 cloud product" with "no satellite imagery" — modern ViTs are happy with raw brightness temperatures. For SKIPP'D (Stanford rooftop, 2017+), the cleanest path is the same archive — **NCEI GridSat-CONUS goes16/17/18** — so we can reuse Folsom's pipeline almost unchanged.

---

## 1. The Folsom story (what actually happened)

### The false starts

1. **Wrong satellite era.** We started by reaching for **GOES-18** on NOAA's NODD bucket (`s3://noaa-goes18/`) because that's how every recent GOES tutorial begins. Folsom PV covers **2014-01-01 → 2016-12-31**, and GOES-18 only went operational in early 2023. Off by a decade.

2. **Right satellite, wrong archive.** **GOES-15** was the operational GOES-West from 2011–2018 — perfect for the Folsom window. It uses the older GVAR Imager (not the ABI on GOES-16/17/18/19), but functionally fine for a ViT. We then assumed there must be an `s3://noaa-goes15/` mirror. There isn't. NOAA's NODD program only carries the **GOES-R series (16–19)**. Anything older was never folded into NODD.

3. **Found an L2 product, almost gave up.** First research pass landed on **SatCORPS Edition-4 GOES-15 L2 cloud optical depth** on NASA Langley ASDC. That product is hourly, daytime-only, 8 km, and behind an Earthdata Login. The first recommendation was to *abandon the satellite modality for Folsom* — it looked strictly worse than nothing.

### The pivot

4. **The pushback that mattered.** The team lead asked, roughly:
   > *"Are we conflating 'no sub-hourly cloud products' with 'no sub-hourly satellite imagery'? Could we use raw imagery directly as a ViT input?"*

   That reframed the search from "find a derived L2 cloud product" to "find any usable imagery at the right cadence." Different keyword, different archive.

5. **The breakthrough.** Second research pass surfaced **NCEI GridSat-CONUS goes15** at `https://www.ncei.noaa.gov/data/gridsat-goes/access/conus/`. It checked every box:

   | Property | Value |
   |---|---|
   | Access | Public HTTPS, anonymous (no Earthdata, no AWS account) |
   | Cadence | 5–15 min (mix of rapid-scan and routine schedule) |
   | Calibration | Brightness temperatures in K, VIS reflectance dimensionless |
   | Projection | Equal-angle lat/lon at 0.04° (~4 km), 650×1500 covering 24–50°N, 125–65°W |
   | Channels populated | VIS (ch1, 0.65 µm), SWIR (ch2, 3.9 µm), WV (ch3, 6.5 µm), IR window (ch4, 10.7 µm), CO2 IR (ch6, 13.3 µm) + 2 derived spatial-variability vars |
   | Coverage | Every day of 2014–2016 |
   | Format | NetCDF-4 / CF-1.6, ~4 MB/file |

6. **Verified live.** Pulled sample files for 2014-01-15 and 2015-06-15, opened with `netCDF4`, sliced a 100×100 window around Folsom (38.642°N, 121.148°W), confirmed all five real channels populate and the schema is identical across the 3-year window. **Five minutes of sanity checks saved days of misdirection.**

7. **What we built (preprocessing only).** A stream-processing pipeline: download → crop 100×100 around Folsom → channel-select `(ch1 VIS, ch4 IR window, ch3 WV)` → normalize each channel to `[0, 1]` → save as float16 NPY shard → delete the raw `.nc`. Eight parallel download workers, ~25 frames/s. The full 3-year pull finished in ~75 min, ~94k shards, ~3.7 GB on disk. Per-frame manifest as JSONL with status, per-channel stats, and clip rates.

---

## 2. General playbook (any site, any date range)

1. **Pin the operational satellite for that exact window.** Don't guess. Quick reference:

   | Era | GOES-East | GOES-West |
   |---|---|---|
   | ~2010–2018 | GOES-13 (GVAR) | GOES-15 (GVAR) |
   | late 2017–2022 | GOES-16 (ABI) | GOES-17 (ABI, IR degraded) |
   | 2023+ | GOES-16 → GOES-19 (ABI) | GOES-18 (ABI) |

2. **Decide what the model actually needs.** A modern vision encoder learns fine from **raw brightness temperatures + visible reflectance**. You almost never need a derived L2 cloud product just to get a ViT to work. Asking for "cloud optical depth" can route you to a hourly/daytime-only/auth-walled product when the underlying imagery is freely available at 5–15 min.

3. **Pick the archive by era, not by habit.**

   | If your window is… | Use… |
   |---|---|
   | Pre-2017 GOES (old GVAR satellites) | **NCEI GridSat-CONUS** (HTTPS, anonymous) |
   | 2017+ GOES-R series, raw radiances or L2 | **NOAA NODD** S3 (`s3://noaa-goes16/17/18/19/`, anonymous) |
   | 2017+ GOES-R, same uniform CF-compliant format as old era | **NCEI GridSat-CONUS** also continues into the GOES-R era — best for pipeline reuse |
   | Anything else and nothing else works | **NASA Earthdata Login** archives as a last resort |

4. **Pull one file. Open it. Eyeball it.** Before writing any pipeline:
   - `curl` one timestamp.
   - Open it (`netCDF4.Dataset`, `xarray.open_dataset`).
   - Print dimensions, variable names, units, fill values.
   - Slice a small window around your site and check the value range matches what the docs say.

   This is a 5-minute check. Skipping it is how you spend a week building against the wrong product.

5. **Stream-process, don't hoard.** Raw GOES files are small individually but add up fast. Download → crop → normalize → write a small shard → **delete the raw file**. Keep a JSONL manifest with status and per-channel stats; that is what saves you when something looks off in training.

6. **Smoke-test one month before the full pull.** Pull a single month end-to-end, generate a diagnostic report (missing-frame rate, value ranges, clip rates, a few visual samples), then commit to the full multi-year pull. We did this for Folsom (`smoke_2014_01.md` → `full_2014_2016.md`) and it caught a couple of edge cases cheaply.

### Lessons, restated explicitly

- **Archive layout ≠ data availability.** NODD only hosts GOES-R era. The old GOES-N/M satellites live elsewhere (NCEI GridSat).
- **L2 product ≠ raw imagery.** Searching for derived cloud products will miss perfectly good raw archives.
- **Verify with one sample file.** Cheap, and it catches schema/coverage surprises immediately.
- **Public-anonymous beats authenticated.** `curl` + no creds is fundamentally different from Earthdata Login at production scale.
- **Rule of thumb: pre-2017 GOES → NCEI GridSat. Post-2017 GOES → NODD S3 (or NCEI GridSat if you want one code path).**

---

## 3. Applied to SKIPP'D

### Site and dataset (verified)

| Item | Value | Source |
|---|---|---|
| Site | Stanford rooftop sky-imager + PV | `SPMF_preprocessing/skippd/process_skippd.py` |
| Latitude | **37.4275°N** | same file (`LATITUDE_DEVICE`) |
| Longitude | **−122.1697°W** | same file (`LONGITUDE_DEVICE`) |
| Source | HuggingFace `solarbench/SKIPPD` | `utils/skippd_process.py` |
| Existing modalities | 1-min sky-camera images + PV power | `densify_skippd_1min` (1-min UTC grid) |
| Date range | Inherited from the HF release; published SKIPP'D spans roughly **March 2017 → October 2019** | Not pinned in the repo — verify against the actual HF download before the pull |

> **Flag:** the exact start/end UTC instants are not pinned in the repo config; they are whatever the loaded HF split happens to contain. Confirm with one line of code (`min/max` over the `time` column) before launching a multi-year pull.

### Recommendation

Use **NCEI GridSat-CONUS** for SKIPP'D as well. Same archive family as Folsom, same NetCDF-4 schema, same 0.04° lat/lon grid, same `netCDF4` open + crop code path. Only the satellite tag and the crop center change.

| Decision | Value |
|---|---|
| Archive | NCEI GridSat-CONUS |
| Satellite tag | `goes16` for 2017-12-18 → 2018-12 (when GOES-16 was GOES-East during the GOES-17 transition), then `goes17` for the rest of the SKIPP'D window. For maximum simplicity, the SKIPP'D window can be covered end-to-end by **`goes16` (GOES-East)** — it sees CA at a worse viewing angle but is the most consistent single tag across 2017–2019. |
| Crop center | (37.4275, −122.1697) |
| Channel mapping | Same as Folsom: `(ch1 VIS, ch4 IR window, ch3 WV)` → float16, normalized to `[0, 1]` |
| URL pattern | `https://www.ncei.noaa.gov/data/gridsat-goes/access/conus/YYYY/MM/GridSat-CONUS.goesXX.YYYY.MM.DD.HHMM.v01.nc` |
| Anchor cadence | 15-min UTC anchors, 7-min tolerance (same as Folsom) |
| Expected volume | ~96 anchors/day × ~950 days ≈ ~91k frames at ~40 KB/shard ≈ **~3.5–4 GB on disk** (same order as Folsom) |

### Pipeline reuse

The Folsom scripts can be copied with two-line edits — change `SAT = "goes15"` to `"goes16"` (or `"goes17"`) and update `FOLSOM_LAT` / `FOLSOM_LON` to the Stanford coordinates. Everything else (anchor matching, parallel download, crop, normalize, manifest) is identical.

### Alternative options (if you want them later)

- **`ABI-L1b-RadC` on NODD S3** (`s3://noaa-goes16/ABI-L1b-RadC/`) — raw radiances, every 5 min, native ABI resolution. Best fidelity, but you handle reprojection yourself.
- **`ABI-L2-CODC` on NODD S3** — derived cloud optical depth if you ever want an L2 channel. Optional, not required for a ViT.

For first cut and consistency with Folsom: **stay with NCEI GridSat-CONUS.**

---

## 4. References

- **NCEI GridSat-CONUS overview** — https://www.ncei.noaa.gov/products/gridded-geostationary-conus
- **NCEI GridSat-CONUS HTTPS root** — https://www.ncei.noaa.gov/data/gridsat-goes/access/conus/
- **NOAA Open Data Dissemination (NODD) registry** — https://registry.opendata.aws/noaa-goes/
- **Folsom preprocessing scripts (in this repo):**
  - `SPMF_preprocessing/folsom/gridsat_download.py` — anchor matcher + URL builder
  - `SPMF_preprocessing/folsom/gridsat_to_shards.py` — stream pipeline: download → crop → normalize → NPY shard
  - `SPMF_preprocessing/folsom/report_smoke.py` — diagnostic report generator
- **Folsom diagnostic reports (in this repo):**
  - `SPMF_preprocessing/folsom/smoke_2014_01.md` — 1-month smoke test
  - `SPMF_preprocessing/folsom/full_2014_2016.md` — full 3-year pull report
