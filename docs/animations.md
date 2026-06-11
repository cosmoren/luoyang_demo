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
| `v2_sky_vs_nosky`    | ViT (kt + sky-encoder mirror, branch `folsom-test-new` @ commit `24b9772`); two checkpoints overlaid as 3-line plot. See per-arm details below. | sky-arm: on; no-sky-arm: `--zero-sky` | `dwsw`, `temperature` (no invalid mask) | 4 h / 16-step (15 min stride) | step 0 (+15 min) | 2014-10-17, 2014-10-18 (also 2014-01-15 for sanity validation; not part of the deliverable) |
| `v2_sky_vs_nosky` (stride 5) | Same checkpoints as the 15-min row above; only difference is the display-grid cadence is 5 min (one model forward per 5-min display T). Anchor lag stays at 15 min. Filenames carry a `_stride5` suffix so they sit alongside the 15-min outputs. | same | same | 4 h / 16-step (5-min display stride; 15-min anchor lag unchanged) | step 0 (+15 min) | 2014-10-17, 2014-10-18 |
| `v2_sky_vs_nosky` (stride 1) | Same checkpoints / labels / flags as the stride-5 row; only `--stride-min` differs (1 instead of 5). Display cadence is 1 min, native CSV cadence; anchor lag stays at 15 min. Filenames carry a `_stride1` suffix. | same | same | 4 h / 16-step (1-min display stride; 15-min anchor lag unchanged) | step 0 (+15 min) | 2014-10-17, 2014-10-18 |

Add a row whenever you mint a new label.

### `v2_sky_vs_nosky` details

Right panel overlays GT (black) + two model predictions on the same axis to
visually demonstrate the lift from sky-camera input. Both checkpoints come
from the same archive (same training launch, same hyperparameters), and
differ only in whether the sky branch is fed real imagery or
`--zero-sky` zeros.

- **Archive:** `folsom_kt_sky_vs_nosky_40ep_4runs_2026-05-26` (branch
  `folsom-test-new` @ commit `24b9772` "Adapt folsom and sky-image branch
  to use kt"). 40 epochs, 4 H100 80GB GPUs in parallel, default
  hyperparameters from `conf_train.yaml`. Both arms used `--use-nwp`
  (`dwsw` + `temperature`).
- **Sky-arm (red, `#cc1f1f`):** `folsom_kt_sky_40ep_r1`,
  `folsom_pv_forecast_vit_best_gpu4.pt` (best of two seeds for the sky
  arm). Test RMSE 69.30 W/m^2, test MAE 25.41 W/m^2 over the first
  16 forecast steps (~4 h horizon).
- **No-sky-arm (blue, `#1f4ec8`):** `folsom_kt_nosky_40ep_r1`,
  `folsom_pv_forecast_vit_best_gpu6.pt` (best of two `--zero-sky` seeds).
  Test RMSE 71.22 W/m^2, test MAE 26.87 W/m^2 (same horizon/protocol).
- **Legend labels (exact):** `Ground truth`, `GHI + NWP + sky`,
  `GHI + NWP (no sky)` for the original 15-min outputs;
  `GT`, `GHI + NWP + Sky`, `GHI + NWP` for the new 5-min-stride outputs
  (the 5-min renders ship with the updated label scheme requested
  alongside the stride change).
- **Inference flag:** `--legacy-pre-518dca9` is **required** when
  forwarding any checkpoint from this archive. Commit `24b9772` predates
  `518dca9` ("Folsom: align kt/p_cs/p_mean/loss with Luoyang recipe",
  2026-05-29), which changed the dataloader's kt/p_cs/target_pv
  normalization (`_FOLSOM_GHI_SCALE: 1100 -> 1000`, raw-W/m^2 target,
  `kt_input_scale: 20.0 -> 4000.0`). Without the flag the May-26 model
  sees inputs ~5x larger than its training distribution and underpredicts
  by ~3x (peak ~150 vs GT 521 on 2014-01-15). With the flag, predictions
  land where they should (peak ~549 vs GT 521 on 2014-01-15; Pearson
  r=0.9989 vs the v1 sky-arm prediction on the same day).
- **Strict-load:** 0 missing / 0 unexpected keys for both checkpoints
  against the current model definition (cleaner than the archive's
  RUN_NOTES anticipated; the dropped `sky_patch_embed.timefeats_mlp.*`
  layer concern was not realized in practice).

Re-render command (15-min stride, default):

```
CUDA_VISIBLE_DEVICES=<free-gpu> micromamba run -n luoyang \
  python inference/animate_forecast.py \
    --date <YYYY-MM-DD> --run-label v2_sky_vs_nosky \
    --legacy-pre-518dca9 \
    --checkpoint        ~/experiments_archive/folsom_kt_sky_vs_nosky_40ep_4runs_2026-05-26/checkpoints_folsom_pv/folsom_kt_sky_40ep_r1/folsom_pv_forecast_vit_best_gpu4.pt \
    --archive-name      folsom_kt_sky_vs_nosky_40ep_4runs_2026-05-26 \
    --run-name          folsom_kt_sky_40ep_r1 \
    --checkpoint-extra  ~/experiments_archive/folsom_kt_sky_vs_nosky_40ep_4runs_2026-05-26/checkpoints_folsom_pv/folsom_kt_nosky_40ep_r1/folsom_pv_forecast_vit_best_gpu6.pt \
    --archive-name-extra folsom_kt_sky_vs_nosky_40ep_4runs_2026-05-26 \
    --run-name-extra    folsom_kt_nosky_40ep_r1
```

### `v2_sky_vs_nosky` 5-min stride renders (2026-06-09)

Identical model setup to the 15-min row above; the only differences are the
display-grid cadence (5 min instead of 15 min, via `--stride-min 5`) and the
updated legend labels (`GT` / `GHI + NWP + Sky` / `GHI + NWP`). The anchor lag
is fixed at 15 min by the model's `pv_output_interval_min`, so changing the
display stride only changes how densely we sample T's; each frame is still
"one model forward = +15 min from anchor". Native CSV cadence is 1 min, so
any integer minute stride >= 1 is supported.

| date       | stride | fps | n_frames | gif size |
|------------|--------|-----|----------|----------|
| 2014-10-17 | 5 min  | 5   | 131      | 11.92 MB |
| 2014-10-18 | 5 min  | 5   | 131      | 10.14 MB |

Output filenames carry a `_stride5` suffix so they sit alongside the existing
15-min outputs without overwriting:

- `2014-10-17_stride5.gif` / `_stride5_predictions.npz` / `_stride5_meta.json`
- `2014-10-18_stride5.gif` / `_stride5_predictions.npz` / `_stride5_meta.json`

When `--stride-min == 15` (the default), the suffix is empty and filenames
remain `<date>.gif` / `<date>_predictions.npz` / `<date>_meta.json` exactly as
before, preserving historical artifacts.

Re-render command (5-min stride; add `--stride-min 5` and use the new label
scheme; everything else is identical):

```
CUDA_VISIBLE_DEVICES=<free-gpu> micromamba run -n luoyang \
  python inference/animate_forecast.py \
    --date <YYYY-MM-DD> --run-label v2_sky_vs_nosky \
    --stride-min 5 \
    --legacy-pre-518dca9 \
    --checkpoint        ~/experiments_archive/folsom_kt_sky_vs_nosky_40ep_4runs_2026-05-26/checkpoints_folsom_pv/folsom_kt_sky_40ep_r1/folsom_pv_forecast_vit_best_gpu4.pt \
    --archive-name      folsom_kt_sky_vs_nosky_40ep_4runs_2026-05-26 \
    --run-name          folsom_kt_sky_40ep_r1 \
    --checkpoint-extra  ~/experiments_archive/folsom_kt_sky_vs_nosky_40ep_4runs_2026-05-26/checkpoints_folsom_pv/folsom_kt_nosky_40ep_r1/folsom_pv_forecast_vit_best_gpu6.pt \
    --archive-name-extra folsom_kt_sky_vs_nosky_40ep_4runs_2026-05-26 \
    --run-name-extra    folsom_kt_nosky_40ep_r1
```

### `v2_sky_vs_nosky` 1-min stride renders (2026-06-09)

Same model setup, same legend labels, and same checkpoints as the 5-min stride
row above. The only change is `--stride-min 5` -> `--stride-min 1`, which steps
the display grid at the native CSV cadence (1 min). Anchor lag remains 15 min
(set by `pv_output_interval_min`), so each frame is still "one model forward
= +15 min from anchor"; we just sample T's ~5x more densely than the stride-5
row and ~15x more densely than the original 15-min row.

| date       | stride | fps | n_frames | gif size | render time |
|------------|--------|-----|----------|----------|-------------|
| 2014-10-17 | 1 min  | 5   | 658      | 60.73 MB | 245.6 s     |
| 2014-10-18 | 1 min  | 5   | 656      | 50.59 MB | 225.5 s     |

Output filenames carry a `_stride1` suffix so they sit alongside the existing
15-min and `_stride5` outputs without overwriting:

- `2014-10-17_stride1.gif` / `_stride1_predictions.npz` / `_stride1_meta.json`
- `2014-10-18_stride1.gif` / `_stride1_predictions.npz` / `_stride1_meta.json`

Re-render command (1-min stride; only `--stride-min` differs from the 5-min
recipe):

```
CUDA_VISIBLE_DEVICES=<free-gpu> micromamba run -n luoyang \
  python inference/animate_forecast.py \
    --date <YYYY-MM-DD> --run-label v2_sky_vs_nosky \
    --stride-min 1 \
    --legacy-pre-518dca9 \
    --checkpoint        ~/experiments_archive/folsom_kt_sky_vs_nosky_40ep_4runs_2026-05-26/checkpoints_folsom_pv/folsom_kt_sky_40ep_r1/folsom_pv_forecast_vit_best_gpu4.pt \
    --archive-name      folsom_kt_sky_vs_nosky_40ep_4runs_2026-05-26 \
    --run-name          folsom_kt_sky_40ep_r1 \
    --checkpoint-extra  ~/experiments_archive/folsom_kt_sky_vs_nosky_40ep_4runs_2026-05-26/checkpoints_folsom_pv/folsom_kt_nosky_40ep_r1/folsom_pv_forecast_vit_best_gpu6.pt \
    --archive-name-extra folsom_kt_sky_vs_nosky_40ep_4runs_2026-05-26 \
    --run-name-extra    folsom_kt_nosky_40ep_r1
```
