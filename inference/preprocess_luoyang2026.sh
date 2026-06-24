#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OUT="${ROOT_DIR}/inference_cache/luoyang2026_test_full"
CFG="conf_luoyang_2026.yaml"
STRIDE=5
BS=256
N=8

TOTAL_WINDOWS=9470

if [[ "${TOTAL_WINDOWS}" -le 0 ]]; then
  echo "No test windows found, abort."
  exit 1
fi

CHUNK=$(( (TOTAL_WINDOWS + N - 1) / N ))
for i in $(seq 0 $((N-1))); do
  START=$(( i * CHUNK ))
  if [[ "${START}" -ge "${TOTAL_WINDOWS}" ]]; then
    continue
  fi
  REM=$(( TOTAL_WINDOWS - START ))
  MAXW=${CHUNK}
  if [[ "${REM}" -lt "${CHUNK}" ]]; then
    MAXW=${REM}
  fi

  nohup python "${ROOT_DIR}/inference/preprocess_luoyang2026test.py" \
      --dataset-config ${CFG} \
      --stride_min ${STRIDE} \
      --batch_size ${BS} \
      --output_dir ${OUT}_p${i} \
      --start_win_idx ${START} \
      --max_windows ${MAXW} \
      --log_every 20 \
  > cache_p${i}.log 2>&1 &
done
echo "Launched ${N} shard preprocess jobs."
echo "ROOT_DIR=${ROOT_DIR}"
echo "TOTAL_WINDOWS=${TOTAL_WINDOWS}, CHUNK=${CHUNK}"
echo "Check logs: cache_p0.log ... cache_p7.log"