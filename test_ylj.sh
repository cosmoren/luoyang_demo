#!/usr/bin/env bash
# Evaluate a fixed epoch across run0..run19 for a checkpoint experiment basename.
#
# Usage: bash test_ylj.sh <checkpoint_basename> <epoch> <cuda_visible_devices> [run_start] [run_end]
#
# Example:
#   bash test_ylj.sh checkpoints_ylj_48h_48h_kt_raw_parquet_no_nwp 5 1
#   bash test_ylj.sh checkpoints_ylj_48h_48h_kt_raw_parquet_no_nwp 5 1 0 4
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <checkpoint_basename> <epoch> <cuda_visible_devices> [run_start] [run_end]"
  exit 1
fi

BASENAME="$1"
EPOCH="$2"
CUDA_DEV="$3"
RUN_START="${4:-0}"
RUN_END="${5:-19}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="$CUDA_DEV"

# Config must match the corresponding training script (test_*_192.sh / test_*.sh).
if [[ "$BASENAME" == *"48h_48h"* ]]; then
  if [[ "$BASENAME" == *"nwp_syn"* ]]; then
    CFG="$ROOT/config/datasets/conf_ylj_syn_192.yaml"
  else
    CFG="$ROOT/config/datasets/conf_ylj_192.yaml"
  fi
else
  if [[ "$BASENAME" == *"nwp_syn"* ]]; then
    CFG="$ROOT/config/datasets/conf_ylj_syn.yaml"
  else
    CFG="$ROOT/config/datasets/conf_ylj.yaml"
  fi
fi

EXTRA=(--ylj_raw_parquet)
if [[ "$BASENAME" == *"no_nwp"* ]]; then
  :
elif [[ "$BASENAME" == *"nwp"* ]]; then
  EXTRA+=(--ylj_parquet_nwp)
fi
if [[ "$BASENAME" == *"nwp_sat"* ]]; then
  EXTRA+=(--ylj_sat_zarr)
fi

for run in $(seq "$RUN_START" "$RUN_END"); do
  CKPT_DIR="$ROOT/${BASENAME}_run${run}"
  CKPT="$CKPT_DIR/pv_forecast_epoch_${EPOCH}.pt"
  OUT_DIR="$CKPT_DIR/test_csvs_epoch_${EPOCH}_seqpairs"
  OUT_CSV="$OUT_DIR/test_run${run}_epoch_${EPOCH}_seqpairs.csv"

  if [[ ! -f "$CKPT" ]]; then
    echo "[skip] missing checkpoint: $CKPT"
    continue
  fi

  mkdir -p "$OUT_DIR"
  echo "[run ] run=$run epoch=$EPOCH device=$CUDA_DEV"
  python "$ROOT/training/train_ylj.py" \
    --config "$CFG" \
    "${EXTRA[@]}" \
    --test_only \
    --checkpoint_dir "$CKPT_DIR" \
    --test_only_ckpt "$CKPT" \
    --test_only_plus15_csv "$OUT_CSV"

  echo "[done] saved: $OUT_CSV"
done

echo "Finished. CSVs under each ${BASENAME}_run*/test_csvs_epoch_${EPOCH}_seqpairs/"
