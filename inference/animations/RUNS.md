## Animated forecast variants

Each subfolder holds one rendering variant. To render a new variant:

```
python inference/animate_forecast.py --run-label <label> --date YYYY-MM-DD
```

Re-running the same `(--run-label, --date)` pair re-uses the cached
`<date>_predictions.npz` and only re-renders the GIF. Pass `--force` to
recompute predictions.

Each `<date>_meta.json` records the exact checkpoint, archive, NWP feature
list, horizon, etc. used for that GIF, so the table below is just a
human-readable index.

| label                | model            | sky | NWP                         | horizon | display step | days rendered          |
|----------------------|------------------|-----|-----------------------------|---------|--------------|------------------------|
| `v1_vit_sky_nwp_s0`  | ViT (`ghi_sky_nwp_s0` from archive `lr2e5_4h_2seeds_sky_nwp_2026-06-05`) | on  | `dwsw`, `temperature` (no invalid mask) | 4 h / 16-step (15 min stride) | step 0 (+15 min) | 2014-01-15, 2014-06-15, 2014-10-17, 2014-10-18 |

Add a row whenever you mint a new label.
