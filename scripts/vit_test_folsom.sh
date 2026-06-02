#!/usr/bin/env bash
# Run train_vit_test_folsom.py multiple times across two GPUs in parallel.
# Each GPU runs its own modality combo, repeated REPEATS times sequentially:
#   GPU 0: GHI + NWP only         (--use-nwp --zero-sky)
#   GPU 1: GHI + NWP + sky        (--use-nwp)
#
# Default: REPEATS=4 per GPU -> 8 total trainings.
#
# Per-run isolation:
#   - --checkpoint_dir is set to LOG_DIR/<arm>/run<i>/ so checkpoints +
#     metrics text don't collide.
#   - TensorBoard log dir is hardcoded inside the trainer to runs/folsom_pv_gpu{N},
#     so we move it to LOG_DIR/<arm>/run<i>/tb/ after each run completes.
#
# Tunables (env override):
#   GPU_NWP=0          (GPU index for the GHI+NWP arm)
#   GPU_NWP_SKY=1      (GPU index for the GHI+NWP+sky arm)
#   REPEATS=4          (sequential runs per GPU)
#   CONFIG=conf_train.yaml          (training YAML in config/train/)
#   DATASET_CONFIG=conf_folsom.yaml (dataset YAML in config/datasets/)
#   LOG_DIR=training_logs/folsom_$(date +%Y%m%d_%H%M%S)

set -euo pipefail

GPU_NWP=${GPU_NWP:-0}
GPU_NWP_SKY=${GPU_NWP_SKY:-1}
REPEATS=${REPEATS:-4}
CONFIG=${CONFIG:-conf_train.yaml}
DATASET_CONFIG=${DATASET_CONFIG:-conf_folsom.yaml}
LOG_DIR=${LOG_DIR:-training_logs/folsom_$(date +%Y%m%d_%H%M%S)}
PROJECT_ROOT=$(pwd)
# Default to the luoyang micromamba env interpreter; override via `PYTHON=...`.
PYTHON=${PYTHON:-/home/erfan/micromamba/envs/luoyang/bin/python}

mkdir -p "${LOG_DIR}"
echo "[folsom] logs        -> ${LOG_DIR}"
echo "[folsom] GPU ${GPU_NWP}: ${REPEATS}x  GHI + NWP   (--use-nwp --zero-sky)"
echo "[folsom] GPU ${GPU_NWP_SKY}: ${REPEATS}x  GHI + NWP + sky   (--use-nwp)"

run_arm() {
    local gpu="$1"           # GPU index
    local arm_name="$2"      # short tag (folder name)
    local extra_flags="$3"   # extra trainer flags as a string

    local arm_dir="${LOG_DIR}/${arm_name}"
    mkdir -p "${arm_dir}"

    for i in $(seq 1 "${REPEATS}"); do
        local run_dir="${arm_dir}/run${i}"
        local ckpt_dir="${run_dir}/checkpoints"
        local stdout_log="${run_dir}/stdout.log"
        mkdir -p "${ckpt_dir}"

        echo "[gpu${gpu}] run ${i}/${REPEATS} (${arm_name})  -> ${run_dir}"
        # SMOKE-RUN ONLY -- do NOT enable for production training.
        # Production epoch length is controlled by sampling.train_epoch_len in
        # config/datasets/conf_folsom.yaml (default 100000 -> 1563 batches @ bs=64).
        # To smoke-test the pipeline, uncomment the line below (caps each epoch
        # at ~20 batches and runs only 1 epoch):
        #     --train_max_batches_per_epoch 20 --epochs 1 \
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES=${gpu} "${PYTHON}" training/train_vit_test_folsom.py \
            --config "${CONFIG}" \
            --dataset-config "${DATASET_CONFIG}" \
            --checkpoint_dir "${ckpt_dir}" \
            ${extra_flags} \
            > "${stdout_log}" 2>&1 \
            || { echo "[gpu${gpu}] run ${i}/${REPEATS} (${arm_name}) FAILED"; exit 1; }

        # Trainer's TB dir is hardcoded to runs/folsom_pv_gpu{N} where N is parsed
        # from the FIRST entry of CUDA_VISIBLE_DEVICES (see _gpu_id_for_checkpoint),
        # so each arm has its own gpuN folder but sequential same-arm runs would
        # collide -- move it out before the next run starts.
        local tb_src="${PROJECT_ROOT}/runs/folsom_pv_gpu${gpu}"
        if [ -d "${tb_src}" ]; then
            mv "${tb_src}" "${run_dir}/tb"
        fi
    done
}

run_arm "${GPU_NWP}"     "ghi_nwp"     "--use-nwp --zero-sky" &
PID_NWP=$!

run_arm "${GPU_NWP_SKY}" "ghi_nwp_sky" "--use-nwp" &
PID_NWP_SKY=$!

status=0
if ! wait "${PID_NWP}";     then status=1; echo "[folsom] arm 'ghi_nwp' failed";     fi
if ! wait "${PID_NWP_SKY}"; then status=1; echo "[folsom] arm 'ghi_nwp_sky' failed"; fi

echo "[folsom] all arms done (status=${status}). logs in ${LOG_DIR}"
exit "${status}"
