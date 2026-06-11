#!/usr/bin/env bash
# One-shot driver: save per-window predictions for all 6 archived best-val checkpoints
# from ghi_vs_ghi_sky_20ep_2026-06-01. Output lands in eval_outputs/predictions/.
# Read-only against the archive.

set -euo pipefail

ARCHIVE=/home/erfan/experiments_archive/ghi_vs_ghi_sky_20ep_2026-06-01/checkpoints_folsom_pv
OUTDIR=/home/erfan/projects/luoyang_demo/eval_outputs/predictions
PY=/home/erfan/micromamba/envs/luoyang/bin/python
SCRIPT=/home/erfan/projects/luoyang_demo/scripts/eval_save_predictions.py

mkdir -p "$OUTDIR"

run_one() {
  local run_name="$1"
  local gpu_n="$2"
  local extra="$3"  # "" or "--zero-sky"
  local ckpt="$ARCHIVE/$run_name/folsom_pv_forecast_vit_best_gpu${gpu_n}.pt"
  local npz="$OUTDIR/${run_name}.npz"
  echo "===================================================================="
  echo "[$run_name] starting  ckpt=$ckpt"
  echo "===================================================================="
  CUDA_VISIBLE_DEVICES=0 FOLSOM_QUIET=1 "$PY" "$SCRIPT" \
    --ckpt "$ckpt" \
    --output "$npz" \
    $extra
}

run_one ghi_only_gpu0 0 "--zero-sky"
run_one ghi_only_gpu1 1 "--zero-sky"
run_one ghi_only_gpu2 2 "--zero-sky"
run_one ghi_sky_gpu1  1 ""
run_one ghi_sky_gpu2  2 ""
# ghi_sky_gpu0 was already produced by the smoke test; skip unless missing.
if [[ ! -f "$OUTDIR/ghi_sky_gpu0.npz" ]]; then
  run_one ghi_sky_gpu0 0 ""
fi

echo
echo "All 6 NPZs written under $OUTDIR/"
ls -lh "$OUTDIR"/*.npz
