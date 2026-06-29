#!/usr/bin/env bash
# R2 Folsom sky-feature experiment matrix (12 runs total).
#
# Four sky-input configs, one GPU each (0–3), × three RNG seeds (1, 2, 3).
# NWP is left at the training default (zeroed inputs; no --use-nwp). Satellite off.
#
# | GPU | --sky-disc-mask | --ray-map | --sun-mask (25°) | run prefix                    |
# |-----|-----------------|-----------|------------------|-------------------------------|
# | 0   | valid_disc      | no        | no               | r2_valid_disc_no_sun          |
# | 1   | manual_tight    | yes       | no               | r2_manual_tight_ray_no_sun    |
# | 2   | valid_disc      | yes       | yes              | r2_valid_disc_ray_halo25      |
# | 3   | manual_tight    | yes       | yes              | r2_manual_tight_ray_halo25    |
#
# Execution: for each seed, launch all four jobs in parallel (nohup), wait for all four
# before starting the next seed round.
#
# Usage (from anywhere):
#   ~/projects/luoyang_demo-sky-feats/scripts/run_r2_sky_experiments.sh

set -euo pipefail

PROJECT_ROOT="${HOME}/projects/luoyang_demo-sky-feats"
PY="/home/erfan/micromamba/envs/luoyang/bin/python3.10"

cd "${PROJECT_ROOT}"

declare -a ALL_RUN_DIRS=()

run_one() {
  local gpu=$1
  local prefix=$2
  local seed=$3
  shift 3
  local -a extra_args=( "$@" )

  local run_dir="${PROJECT_ROOT}/runs/${prefix}_s${seed}"
  mkdir -p "${run_dir}"
  ALL_RUN_DIRS+=( "${run_dir}" )

  echo "  [GPU ${gpu}] ${prefix}_s${seed} -> ${run_dir}/train.log"

  CUDA_VISIBLE_DEVICES="${gpu}" nohup "${PY}" "${PROJECT_ROOT}/training/train_vit_test_folsom.py" \
    --dataset-config conf_folsom.yaml \
    --epochs 20 \
    --warmup-epochs 4 \
    --no-use-satellite \
    --seed "${seed}" \
    --checkpoint_dir "${run_dir}" \
    "${extra_args[@]}" \
    > "${run_dir}/train.log" 2>&1 &
}

overall_status=0

for seed in 1 2 3; do
  echo "================================================================================"
  echo "Seed round ${seed}: launching 4 parallel runs on GPUs 0–3 ..."
  echo "================================================================================"

  pids=()

  run_one 0 r2_valid_disc_no_sun "${seed}" \
    --sky-disc-mask valid_disc
  pids+=( "$!" )

  run_one 1 r2_manual_tight_ray_no_sun "${seed}" \
    --sky-disc-mask manual_tight \
    --ray-map
  pids+=( "$!" )

  run_one 2 r2_valid_disc_ray_halo25 "${seed}" \
    --sky-disc-mask valid_disc \
    --ray-map \
    --sun-mask \
    --sun-mask-radius-deg 25
  pids+=( "$!" )

  run_one 3 r2_manual_tight_ray_halo25 "${seed}" \
    --sky-disc-mask manual_tight \
    --ray-map \
    --sun-mask \
    --sun-mask-radius-deg 25
  pids+=( "$!" )

  echo "Seed round ${seed}: waiting for PIDs ${pids[*]} ..."

  round_status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      round_status=1
    fi
  done

  if [[ "${round_status}" -ne 0 ]]; then
    echo "Seed round ${seed}: one or more runs FAILED (see train.log under each run dir)." >&2
    overall_status=1
  else
    echo "Seed round ${seed}: all 4 runs completed successfully."
  fi
done

echo ""
echo "================================================================================"
echo "R2 sky experiments finished (exit status ${overall_status})."
echo "Run directories (${#ALL_RUN_DIRS[@]} total):"
echo "================================================================================"
for d in "${ALL_RUN_DIRS[@]}"; do
  echo "  ${d}"
done

exit "${overall_status}"
