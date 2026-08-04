#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# 1) Knobs
# =============================================================================
RUN_ROOT="/home/kyber/projects/digital_energy/experiment_files/runs/2026-07-31_fol-tabm-nwp-2dg"
REPEAT=5
SEED_START=1
GPUS=(0 1)
FREE_MEM_MIB=2048
POLL_SEC=30
CONDA_ENV="luoyang"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Default trainer if an experiment line omits the script field (2-field form).
DEFAULT_TRAIN_SCRIPT="training/train_vit_test_folsom.py"

# =============================================================================
# 2) Experiments — one per line, either:
#      name|flags                         (uses DEFAULT_TRAIN_SCRIPT)
#      name|train_script|flags            (vit_imgs and dinov2 in one file)
#    Queue order: seed-outer (all exps @ SEED_START, then next seed, ...)
# =============================================================================
EXPERIMENTS=(
  # Weekend matrix: NWP on (trainer default), sun off vs gaussian_pixel
  "vit_imgs_nwp|training/train_vit_test_folsom.py|--use-nwp --no-ray-map --sun-mask none --sky-mask none"
  "vit_imgs_nwp_sun|training/train_vit_test_folsom.py|--use-nwp --no-ray-map --sun-mask gaussian_pixel --sky-mask none"
  "dinov2_nwp|training/train_vit_test_folsom_dinov2.py|--use-nwp --no-ray-map --sun-mask none --sky-mask none"
  "dinov2_nwp_sun|training/train_vit_test_folsom_dinov2.py|--use-nwp --no-ray-map --sun-mask gaussian_pixel --sky-mask none"
)

# =============================================================================
# 3) Launcher (usually leave alone)
# =============================================================================
_conda_activate() {
  if ! command -v conda >/dev/null 2>&1; then
    for candidate in \
      "${HOME}/miniconda3/etc/profile.d/conda.sh" \
      "${HOME}/anaconda3/etc/profile.d/conda.sh" \
      "/opt/conda/etc/profile.d/conda.sh"; do
      if [[ -f "${candidate}" ]]; then
        # shellcheck source=/dev/null
        source "${candidate}"
        break
      fi
    done
  fi
  # shellcheck disable=SC1091
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  conda activate "${CONDA_ENV}"
}

_gpu_used_mib() {
  local gpu_id=$1
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu_id}" \
    | tr -d ' '
}

_gpu_has_train_proc() {
  local gpu_id=$1
  local pids
  pids="$(nvidia-smi -i "${gpu_id}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | tr -d ' ' | grep -v '^$' || true)"
  [[ -z "${pids}" ]] && return 1
  local pid
  for pid in ${pids}; do
    if [[ -r "/proc/${pid}/cmdline" ]]; then
      local cmd
      cmd="$(tr '\0' ' ' <"/proc/${pid}/cmdline")"
      if [[ "${cmd}" == *python* ]] || [[ "${cmd}" == *train* ]]; then
        return 0
      fi
    fi
  done
  return 1
}

_gpu_is_free() {
  local gpu_id=$1
  local used
  used="$(_gpu_used_mib "${gpu_id}")"
  if (( used >= FREE_MEM_MIB )); then
    return 1
  fi
  if _gpu_has_train_proc "${gpu_id}"; then
    return 1
  fi
  return 0
}

# gpu_id -> pid (empty = free slot we own)
declare -A GPU_PID=()
for g in "${GPUS[@]}"; do
  GPU_PID["$g"]=""
done

_reap_finished() {
  local g pid
  for g in "${GPUS[@]}"; do
    pid="${GPU_PID[$g]:-}"
    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}" || true
      GPU_PID["$g"]=""
    fi
  done
}

_wait_for_free_gpu() {
  while true; do
    _reap_finished
    local g
    for g in "${GPUS[@]}"; do
      if [[ -z "${GPU_PID[$g]:-}" ]] && _gpu_is_free "${g}"; then
        echo "${g}"
        return 0
      fi
    done
    sleep "${POLL_SEC}"
  done
}

_conda_activate

JOB_PIDS=()
for ((i = 0; i < REPEAT; i++)); do
  seed=$((SEED_START + i))
  for entry in "${EXPERIMENTS[@]}"; do
    name="${entry%%|*}"
    rest="${entry#*|}"
    # 3-field: name|train_script|flags  |  2-field: name|flags
    if [[ "${rest}" == *"|"* ]]; then
      train_script="${rest%%|*}"
      flags="${rest#*|}"
    else
      train_script="${DEFAULT_TRAIN_SCRIPT}"
      flags="${rest}"
    fi
    run_dir="${RUN_ROOT}/${name}/seed_${seed}"
    mkdir -p "${run_dir}"

    gpu="$(_wait_for_free_gpu)"

    # shellcheck disable=SC2206
    extra=( ${flags} )
    cmd=(
      env "CUDA_VISIBLE_DEVICES=${gpu}"
      python "${train_script}"
      --seed "${seed}"
      --tb-log-dir "${run_dir}"
      --checkpoint_dir "${run_dir}"
      "${extra[@]}"
    )

    echo "[start] gpu=${gpu} name=${name} seed=${seed} dir=${run_dir}"
    "${cmd[@]}" >"${run_dir}/train.out" 2>&1 &
    pid=$!
    GPU_PID["$gpu"]="${pid}"
    JOB_PIDS+=("${pid}")
  done
done

echo "[wait] ${#JOB_PIDS[@]} jobs ..."
fail=0
for pid in "${JOB_PIDS[@]}"; do
  if ! wait "${pid}"; then
    fail=1
  fi
done
echo "[done]"

_seed_tag=""
for ((i = 0; i < REPEAT; i++)); do
  _seed_tag+="$((SEED_START + i))"
done
_results_png="${RUN_ROOT}/results_s${_seed_tag}.png"
if [[ -e "${_results_png}" ]]; then
  _n=2
  while [[ -e "${RUN_ROOT}/results_s${_seed_tag}_${_n}.png" ]]; do
    _n=$((_n + 1))
  done
  _results_png="${RUN_ROOT}/results_s${_seed_tag}_${_n}.png"
fi
echo "[results] generating ${_results_png} ..."
if ! python scripts/exp_results.py "${RUN_ROOT}" --out "${_results_png}"; then
  echo "[warn] results PNG generation failed for ${_results_png} (continuing)" >&2
fi

_used_sh="${RUN_ROOT}/local_run_used.sh"
if [[ -e "${_used_sh}" ]]; then
  _n=2
  while [[ -e "${RUN_ROOT}/local_run_used_${_n}.sh" ]]; do
    _n=$((_n + 1))
  done
  _used_sh="${RUN_ROOT}/local_run_used_${_n}.sh"
fi
echo "[snapshot] copying launcher -> ${_used_sh}"
cp "${BASH_SOURCE[0]}" "${_used_sh}"

exit "${fail}"
