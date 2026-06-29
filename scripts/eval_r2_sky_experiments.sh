#!/usr/bin/env bash
# Evaluate all 12 archived R2 sky training runs × best + final checkpoints (24 evals).
#
# Parallel layout mirrors scripts/run_r2_sky_experiments.sh: one config group per GPU (0–3),
# seeds 1–3 sequential within each worker (best then final per seed).
#
# Archive:  ~/experiments_archive/sunmask_ray_skymask_20ep_2026-06-29/runs/<prefix>_s<seed>/
# Results:  ~/experiment_results/sunmask_ray_skymask_20ep_eval/inference/<prefix>_s<seed>/{best,final}/
#
# Usage (from anywhere):
#   ~/projects/luoyang_demo-sky-feats/scripts/eval_r2_sky_experiments.sh

set -euo pipefail

PROJECT_ROOT="${HOME}/projects/luoyang_demo-sky-feats"
PY="/home/erfan/micromamba/envs/luoyang/bin/python3.10"
EVAL_PY="${PROJECT_ROOT}/inference/eval_folsom_checkpoint.py"

ARCHIVE_ROOT="${HOME}/experiments_archive/sunmask_ray_skymask_20ep_2026-06-29/runs"
RESULTS_ROOT="${HOME}/experiment_results/sunmask_ray_skymask_20ep_eval/inference"
LAUNCHER_LOG="${RESULTS_ROOT}/eval_launcher.log"

cd "${PROJECT_ROOT}"
mkdir -p "${RESULTS_ROOT}"
exec > >(tee -a "${LAUNCHER_LOG}") 2>&1

eval_one() {
  local gpu=$1
  local prefix=$2
  local seed=$3
  local ckpt_kind=$4
  shift 4
  local -a sky_args=( "$@" )

  local ckpt_file="folsom_pv_forecast_vit_${ckpt_kind}_gpu${gpu}.pt"
  local ckpt_path="${ARCHIVE_ROOT}/${prefix}_s${seed}/${ckpt_file}"
  local out_dir="${RESULTS_ROOT}/${prefix}_s${seed}/${ckpt_kind}"

  mkdir -p "${out_dir}" 

  echo "  [GPU ${gpu}] ${prefix}_s${seed}/${ckpt_kind} -> ${out_dir}/"

  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" "${EVAL_PY}" \
    --checkpoint "${ckpt_path}" \
    --dataset-config conf_folsom.yaml \
    --no-use-satellite \
    --output-dir "${out_dir}" \
    "${sky_args[@]}" \
    > "${out_dir}/eval.log" 2>&1
}

run_seed_pair() {
  local gpu=$1
  local prefix=$2
  local seed=$3
  shift 3
  local -a sky_args=( "$@" )

  echo "[GPU ${gpu}] ${prefix}_s${seed}: best checkpoint ..."
  eval_one "${gpu}" "${prefix}" "${seed}" best "${sky_args[@]}"
  echo "[GPU ${gpu}] ${prefix}_s${seed}: final checkpoint ..."
  eval_one "${gpu}" "${prefix}" "${seed}" final "${sky_args[@]}"
  echo "[GPU ${gpu}] ${prefix}_s${seed}: done (best + final)."
}

worker_gpu0() {
  local gpu=0
  local prefix=r2_valid_disc_no_sun
  local -a sky_args=( --sky-disc-mask valid_disc )

  echo "================================================================================"
  echo "GPU ${gpu} worker: ${prefix} (seeds 1–3)"
  echo "================================================================================"

  for seed in 1 2 3; do
    run_seed_pair "${gpu}" "${prefix}" "${seed}" "${sky_args[@]}"
  done
}

worker_gpu1() {
  local gpu=1
  local prefix=r2_manual_tight_ray_no_sun
  local -a sky_args=( --sky-disc-mask manual_tight --ray-map )

  echo "================================================================================"
  echo "GPU ${gpu} worker: ${prefix} (seeds 1–3)"
  echo "================================================================================"

  for seed in 1 2 3; do
    run_seed_pair "${gpu}" "${prefix}" "${seed}" "${sky_args[@]}"
  done
}

worker_gpu2() {
  local gpu=2
  local prefix=r2_valid_disc_ray_halo25
  local -a sky_args=(
    --sky-disc-mask valid_disc
    --ray-map
    --sun-mask
    --sun-mask-radius-deg 25
  )

  echo "================================================================================"
  echo "GPU ${gpu} worker: ${prefix} (seeds 1–3)"
  echo "================================================================================"

  for seed in 1 2 3; do
    run_seed_pair "${gpu}" "${prefix}" "${seed}" "${sky_args[@]}"
  done
}

worker_gpu3() {
  local gpu=3
  local prefix=r2_manual_tight_ray_halo25
  local -a sky_args=(
    --sky-disc-mask manual_tight
    --ray-map
    --sun-mask
    --sun-mask-radius-deg 25
  )

  echo "================================================================================"
  echo "GPU ${gpu} worker: ${prefix} (seeds 1–3)"
  echo "================================================================================"

  for seed in 1 2 3; do
    run_seed_pair "${gpu}" "${prefix}" "${seed}" "${sky_args[@]}"
  done
}

echo "================================================================================"
echo "R2 sky checkpoint eval: launching 4 GPU workers (0–3), 24 evals total"
echo "Archive: ${ARCHIVE_ROOT}"
echo "Results: ${RESULTS_ROOT}"
echo "Launcher log: ${LAUNCHER_LOG}"
echo "================================================================================"

overall_status=0
pids=()

worker_gpu0 &
pids+=( "$!" )
worker_gpu1 &
pids+=( "$!" )
worker_gpu2 &
pids+=( "$!" )
worker_gpu3 &
pids+=( "$!" )

echo "Waiting for worker PIDs: ${pids[*]} ..."

for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    overall_status=1
  fi
done

echo ""
echo "================================================================================"
echo "R2 sky checkpoint eval finished (exit status ${overall_status})."
echo "Output directories (${#ALL_OUTPUT_DIRS[@]} total; expect 24):"
echo "================================================================================"

# Workers append in parallel; sort for stable listing.
mapfile -t sorted_dirs < <(printf '%s\n' "${ALL_OUTPUT_DIRS[@]}" | sort)
for d in "${sorted_dirs[@]}"; do
  echo "  ${d}"
done

exit "${overall_status}"
