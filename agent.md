# Agent Guide (Short)

## Scope
- Main pipeline: Luoyang 2026 inverter-level forecasting.
- Key files:
  - `dataloader/luoyang_2026_zarr.py`
  - `training/train_vit_luoyang2026.py`
  - `inference/inference_luoyang2026.py`
  - `inference/evaluate_luoyang2026.py`
  - `models/models.py` (`pv_forecasting_model_vit_imgs`)

## Environment
- Use: `micromamba run -n SimVP python ...`
- GPU: `CUDA_VISIBLE_DEVICES=<id>`

## Configs
- Train: `config/train/conf_train.yaml`
- Dataset: `config/datasets/conf_luoyang_2026.yaml`

## Critical Conventions
- PV source field: `final_power`.
- Required training tensors: `kt`, `kt_mask`, `pv_timefeats`, `forecast_timefeats`, `target_pv`, `target_p_cs`, `p_mean`.
- Optional modalities: sat/sky/NWP.
- `weather_score` is removed.
- `weather_ghi` / `theory_ghi` exist in loader outputs.

## Sat/Sky Handling (Important)
- Full-mask mode is enabled for Luoyang 2026:
  - batch keys include `sat_valid_mask`, `skimg_valid_mask`.
  - missing sat/sky samples are zero-filled at collate.
  - model uses valid masks to zero invalid sample contributions before fusion.

## Training
- Entry: `training/train_vit_luoyang2026.py`
- Tasks:
  - `--task 15m` (idx 0)
  - `--task 4h` (idx 15)
  - `--task 48h` (full horizon)
- Checkpoint options:
  - `--resume-checkpoint`: full resume
  - `--init-checkpoint`: load weights only

## Inference / Eval
- Inference writes per-inverter CSV incrementally; single-GPU station CSV also streams.
- Eval script prints Beijing-day RMSE/MAE and daily-average RMSE/MAE.

## Known Pitfalls
- Test split boundary is full-window based; first output time can be later than `test_start_bj`.
- Some inverter CSVs may not cover early timestamps; loader has window-skip protection.
- `num_workers=0` is best for debugging loader exceptions.

## Quick Commands
```bash
# Train 15m
CUDA_VISIBLE_DEVICES=5 micromamba run -n SimVP python training/train_vit_luoyang2026.py --task 15m --dataset-config conf_luoyang_2026.yaml --config conf_train.yaml

# Inference 4h
CUDA_VISIBLE_DEVICES=4 micromamba run -n SimVP python inference/inference_luoyang2026.py --task 4h --checkpoint checkpoint_2026_fixedhuber/pv_forecast_vit_best_task_4h_gpu6.pt

# Daily eval
micromamba run -n SimVP python inference/evaluate_luoyang2026.py --input_csv inference_results/luoyang2026_15min_1hourstride/station_total_15m.csv
```
