#!/usr/bin/env bash
#SBATCH --job-name=luo-pv-only
#SBATCH --partition=gb10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --exclude=trt-gb10-1
#SBATCH --output=/shared_work/erfan/logs/luo-pv-only_%j.log
#
# Connect to the Slurm server (from local machine):
#   ssh erfan@207.35.188.227 -p 2221
#
# Luoyang 15m DINOv2 train (PV only: sky hard-zeroed, no NWP). Layout:
#   RUN_ROOT/<CASE>/seed_<SEED>/   (checkpoints + train.out)
#
# Per-run: EXP_NAME / CODE_DIR / CASE / SEED, train flags, --time if needed.
# SEED is path-only (Luoyang trainer has no --seed); bump 1→5 across 5 submits.
# job-name / --output keep distinct prefixes (%j keeps Slurm logs unique).
# Code always runs from SHARED_ROOT (not submit dir). Conda is activated in-script.
# Data paths come from the dataset yaml only (no --data-dir CLI).
#
# Submit from any cwd on the server:
#   mkdir -p /shared_work/erfan/logs
#   sbatch /shared_work/erfan/codebases/fol-luo-SR1/scripts/luo-adapt-runs/pv-only.sh

set -euo pipefail

# =============================================================================
# Knobs (edit these per run)
# =============================================================================
EXP_NAME="fol luo adapt and fix/2026-08-14_luo-adapt"
CODE_DIR="fol-luo-SR1"
CASE="pv-only"
SEED=1

# =============================================================================
# Setup (usually leave alone)
# =============================================================================
RUN_ROOT="/shared_work/erfan/experiment_files/runs/${EXP_NAME}"
SHARED_ROOT="/shared_work/erfan/codebases/${CODE_DIR}"
CONDA_INIT="/shared_work/erfan/env/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV="luoyang_cuda"

_conda_activate() {
  if [[ ! -f "${CONDA_INIT}" ]]; then
    echo "ERROR: Conda setup not found: ${CONDA_INIT}" >&2
    exit 1
  fi
  # shellcheck source=/dev/null
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

# SEED is for RUN_DIR naming only; not passed to the trainer.
RUN_DIR="${RUN_ROOT}/${CASE}/seed_${SEED}"
mkdir -p "${RUN_DIR}"

echo "Repo: ${REPO_ROOT}"
echo "Case: ${CASE}"
echo "seed=${SEED} (path-only)"
echo "dataset_config=conf_luoyang_2026_15m.yaml"
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

# Checkpoints + train.out live under RUN_DIR (no TB CLI on this trainer).
python training/train_vit_luoyang2026total.py \
  --task 15m \
  --dataset-config conf_luoyang_2026_15m.yaml \
  --model pv_forecasting_model_vit_dinov2 \
  --sky-sat-load-mode lazy \
  --no-use-satellite \
  --zero-sky \
  --no-use-nwp \
  --checkpoint_dir "${RUN_DIR}" \
  2>&1 | tee "${RUN_DIR}/train.out"

echo "Done: $(date -Is)"
