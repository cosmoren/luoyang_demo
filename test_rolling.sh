#!/usr/bin/env bash
# Rolling daily finetune across run0..run19.
#
# Usage: bash test_rolling.sh [cuda_device] [lookback_days] [finetune_end_time] [run_start] [run_end]
#
# finetune_end_time: naive China HH:MM on day D (default 00:00). Window slides later on D
# without changing lookback_days span, e.g. 03:00 or 07:00 (must be <= 09:00 issue time).
#
# Examples:
#   bash test_rolling.sh 0 7
#   bash test_rolling.sh 3 10 07:00
#   bash test_rolling.sh 3 10 07:00 0 4
set -euo pipefail

CUDA_DEV="${1:-0}"
LOOKBACK="${2:-7}"
FINETUNE_END="${3:-00:00}"
RUN_START="${4:-0}"
RUN_END="${5:-19}"

if [ "$FINETUNE_END" = "00:00" ]; then
  BASE="checkpoints_ylj_48h_48h_kt_raw_parquet_nwp_sat_rolling_back_${LOOKBACK}_days"
else
  END_TAG="${FINETUNE_END//:/_}"
  BASE="checkpoints_ylj_48h_48h_kt_raw_parquet_nwp_sat_rolling_back_${LOOKBACK}_days_end_${END_TAG}"
fi

for run in $(seq "$RUN_START" "$RUN_END"); do
  CKPT_DIR="${BASE}_run${run}"
  CUDA_VISIBLE_DEVICES="$CUDA_DEV" python training/train_ylj_rolling.py \
    --config config/datasets/conf_ylj.yaml \
    --ylj_raw_parquet \
    --ylj_parquet_nwp \
    --ylj_sat_zarr \
    --rolling_ckpt_dir "$CKPT_DIR" \
    --rolling_output_csv "${CKPT_DIR}/rolling_finetune_predictions.csv" \
    --rolling_lookback_days "$LOOKBACK" \
    --rolling_finetune_end_time "$FINETUNE_END"
done
