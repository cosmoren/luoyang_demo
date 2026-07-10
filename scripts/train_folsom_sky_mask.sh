#!/usr/bin/env bash
#SBATCH --job-name=folsom_sky_mask
#SBATCH --partition=gb10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=23:00:00
#SBATCH --output=/shared_work/erfan/codebase/logs/train_folsom_sky_mask_%j.log
#
# Folsom sky/sun mask ablation: two sequential training cases.
#
# Before sbatch: conda activate your CUDA-capable env (same pattern as Yang).
# Then from the uploaded codebase root:
#   sbatch scripts/train_folsom_sky_mask.sh
#   sbatch --export=ALL,CASE=1 scripts/train_folsom_sky_mask.sh
#   sbatch --export=ALL,CASE=2 scripts/train_folsom_sky_mask.sh
#
# Usage (interactive / local):
#   CASE=all bash scripts/train_folsom_sky_mask.sh
#   CASE=1   bash scripts/train_folsom_sky_mask.sh
#   CASE=2   bash scripts/train_folsom_sky_mask.sh
#
# CASE: all (default) | 1 | 2
#   1 = GHI + SKI (RGB), no sun/sky masks
#   2 = GHI + SKI + sun_mask (gaussian_pixel) + sky_mask (tight)

set -euo pipefail

# Slurm copies the batch script; BASH_SOURCE may not point at the writable codebase.
# Prefer submit dir, else the known server tree.
REPO_ROOT="${SLURM_SUBMIT_DIR:-/shared_work/erfan/codebase}"
cd "${REPO_ROOT}"

# Prefer shared logs/cache under the server codebase tree (avoids home-dir writes on compute nodes).
mkdir -p /shared_work/erfan/codebase/logs
mkdir -p /shared_work/erfan/codebase/.cache
export MPLCONFIGDIR="${MPLCONFIGDIR:-/shared_work/erfan/codebase/.cache/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/shared_work/erfan/codebase/.cache}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

CASE="${CASE:-all}"
COMMON=(
  --epochs 10
  --warmup-epochs 2
  --batch_size 16
  --num_workers 4
  --no-ray-map
)

run_case1() {
  echo "===== Case 1: GHI + SKI (RGB), sun_mask=none, sky_mask=none ====="
  python training/train_vit_test_folsom.py \
    "${COMMON[@]}" \
    --sun-mask none \
    --sky-mask none
}

run_case2() {
  echo "===== Case 2: GHI + SKI + sun_mask=gaussian_pixel + sky_mask=tight ====="
  python training/train_vit_test_folsom.py \
    "${COMMON[@]}" \
    --sun-mask gaussian_pixel \
    --sky-mask tight
}

echo "Repo: ${REPO_ROOT}"
echo "CASE=${CASE}"
echo "python: $(command -v python)"
echo "Start: $(date -Is)"

case "${CASE}" in
  all)
    run_case1
    run_case2
    ;;
  1)
    run_case1
    ;;
  2)
    run_case2
    ;;
  *)
    echo "Unknown CASE='${CASE}'. Use all|1|2." >&2
    exit 1
    ;;
esac

echo "Done: $(date -Is)"
