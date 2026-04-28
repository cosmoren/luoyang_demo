#!/usr/bin/env bash
# Run train_vit_test.py TOTAL times across multiple GPUs in parallel.
# Each GPU gets its own checkpoint / metrics file (gpu{N} suffix), see _gpu_id_for_checkpoint
# in training/train_vit_test.py.
#
# Tunables: override via env, e.g. `GPUS="0 1" TOTAL=10 bash scripts/vit_test.sh`.

set -euo pipefail

GPUS=(${GPUS:-0 1 2 3})
TOTAL=${TOTAL:-20}
CONFIG=${CONFIG:-conf_train.yaml}

N=${#GPUS[@]}
if [ "$N" -eq 0 ]; then
    echo "No GPUs configured" >&2
    exit 1
fi

pids=()
for idx in "${!GPUS[@]}"; do
    g=${GPUS[$idx]}
    count=$(( TOTAL / N + (idx < TOTAL % N ? 1 : 0) ))
    if [ "$count" -le 0 ]; then
        continue
    fi
    (
        for i in $(seq 1 "$count"); do
            echo "[gpu${g}] run ${i}/${count}"
            CUDA_VISIBLE_DEVICES=${g} python training/train_vit_test.py --config "${CONFIG}"
        done
    ) &
    pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
exit "$status"
