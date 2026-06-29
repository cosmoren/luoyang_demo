#!/usr/bin/env bash
# Launch one training process per GPU for train_vit_luoyang2026total.py.
#
# Notes:
# - This trainer is single-process (non-DDP). "Multi-GPU" here means
#   multiple independent runs in parallel, one per GPU.
# - Override variables as needed, e.g.:
#   GPUS="0 1" TASK=4h DATASET_CONFIG=conf_luoyang_2026_4h.yaml bash scripts/train_vit_luoyang2026total_multigpu.sh
#
# Extra args example:
#   EXTRA_ARGS="--epochs 30 --batch_size 64 --num_workers 8"

set -euo pipefail

GPUS=(${GPUS:-0 1 2 3})
PYTHON_BIN=${PYTHON_BIN:-python}
CONFIG=${CONFIG:-conf_train.yaml}
DATASET_CONFIG=${DATASET_CONFIG:-conf_luoyang_2026_4h.yaml}
TASK=${TASK:-4h}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-checkpoints_pvnwp_128bs}
EXTRA_ARGS=${EXTRA_ARGS:-}

if [ "${#GPUS[@]}" -eq 0 ]; then
  echo "No GPUs configured. Set GPUS, e.g. GPUS=\"0 1\"." >&2
  exit 1
fi

mkdir -p "${CHECKPOINT_ROOT}"

read -r -a EXTRA_ARR <<< "${EXTRA_ARGS}"

echo "Launching ${#GPUS[@]} parallel runs"
echo "  task=${TASK}"
echo "  config=${CONFIG}"
echo "  dataset_config=${DATASET_CONFIG}"
echo "  checkpoint_root=${CHECKPOINT_ROOT}"

pids=()
for g in "${GPUS[@]}"; do
  run_ckpt_dir="${CHECKPOINT_ROOT}/gpu${g}"
  mkdir -p "${run_ckpt_dir}"
  log_file="${run_ckpt_dir}/train.log"

  echo "[gpu${g}] start -> ${log_file}"
  (
    CUDA_VISIBLE_DEVICES="${g}" \
    "${PYTHON_BIN}" training/train_vit_luoyang2026total.py \
      --task "${TASK}" \
      --config "${CONFIG}" \
      --dataset-config "${DATASET_CONFIG}" \
      --checkpoint_dir "${run_ckpt_dir}" \
      "${EXTRA_ARR[@]}" \
      --use-ema
  ) > "${log_file}" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [ "${status}" -ne 0 ]; then
  echo "One or more runs failed. Check per-GPU logs under ${CHECKPOINT_ROOT}/gpu*/train.log" >&2
else
  echo "All runs finished successfully."
fi
exit "${status}"
