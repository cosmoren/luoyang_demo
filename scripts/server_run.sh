#!/usr/bin/env bash
#SBATCH --job-name=server_run
#SBATCH --partition=gb10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --exclude=trt-gb10-1
#SBATCH --output=/shared_work/erfan/logs/server_run_%j.log
#
# Connect to the Slurm server (from local machine):
#   ssh erfan@207.35.188.227 -p 2221
#
# Simple 1-GPU Folsom train. Layout matches local_run.sh:
#   RUN_ROOT/<CASE>/seed_<SEED>/   (checkpoints + TB + train.out)
#
# Per-run: EXP_NAME / CODE_DIR / CASE / SEED, train flags, --time if needed.
# job-name / --output can stay fixed (%j keeps Slurm logs unique).
# Code always runs from SHARED_ROOT (not submit dir). Conda is activated in-script.
#
# Submit from any cwd on the server:
#   mkdir -p /shared_work/erfan/logs
#   sbatch /shared_work/erfan/codebases/fol-tabm-SR1/scripts/server_run.sh

set -euo pipefail

# =============================================================================
# Knobs (edit these per run)
# =============================================================================
EXP_NAME="2026-08-04_fol-dino-sunmask-alpha-fix"
CODE_DIR="fol-tabm-SR1"
CASE="dinov2_nwp_sun"
SEED=0

# =============================================================================
# Setup (usually leave alone)
# =============================================================================
RUN_ROOT="/shared_work/erfan/experiment_files/runs/${EXP_NAME}"
SHARED_ROOT="/shared_work/erfan/codebases/${CODE_DIR}"
DATA_DIR="/shared_work/erfan/datasets/folsom_ds"
CONDA_INIT="/shared_work/erfan/env/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV="luoyang_cuda"

_conda_activate() {
  if [[ ! -f "${CONDA_INIT}" ]]; then
    echo "ERROR: Conda setup not found: ${CONDA_INIT}" >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "${CONDA_INIT}"
  conda activate "${CONDA_ENV}"
}

# Always use SHARED_ROOT so the job runs the checkout you set in knobs,
# even if you sbatch from somewhere else in a rush.
REPO_ROOT="${SHARED_ROOT}"
cd "${REPO_ROOT}"
_conda_activate

mkdir -p /shared_work/erfan/logs
mkdir -p "${SHARED_ROOT}/.cache"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SHARED_ROOT}/.cache/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SHARED_ROOT}/.cache}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

RUN_DIR="${RUN_ROOT}/${CASE}/seed_${SEED}"
mkdir -p "${RUN_DIR}"

echo "Repo: ${REPO_ROOT}"
echo "Case: ${CASE}"
echo "seed=${SEED}"
echo "data_dir=${DATA_DIR}"
echo "run_dir=${RUN_DIR}"
echo "conda_env=${CONDA_ENV}"
echo "python: $(command -v python)"
echo "Start: $(date -Is)"

python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    print("ERROR: CUDA not available; refusing CPU fallback", file=sys.stderr)
    sys.exit(1)
print(f"CUDA OK: {torch.cuda.get_device_name(0)}")
PY

# Same dir for ckpt + TB as local_run; also mirror train.out into the seed folder.
python training/train_vit_test_folsom_dinov2.py \
  --seed "${SEED}" \
  --data-dir "${DATA_DIR}" \
  --checkpoint_dir "${RUN_DIR}" \
  --tb-log-dir "${RUN_DIR}" \
  --no-ray-map \
  --sun-mask gaussian_pixel \
  --sky-mask none \
  2>&1 | tee "${RUN_DIR}/train.out"

echo "Done: $(date -Is)"
