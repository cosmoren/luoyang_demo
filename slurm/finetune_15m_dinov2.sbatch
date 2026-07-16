#!/usr/bin/env bash
#SBATCH --job-name=luoyang_2026_total_15m
#SBATCH --partition=energy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/shared_work/yuan/workspace/luoyang_demo/slurm/logs/train_%x_%j_15m.log
set -eo pipefail

# ── 任务选择：15m / 4h / 48h ─────────────────────────────────────────────────
TASK=${TASK:-15m}

# ── 路径 ─────────────────────────────────────────────────────────────────────
YUAN_PY=/shared_work/yuan/micromamba/envs/SimVP/bin/python
PROJECT_ROOT=/shared_work/yuan/workspace/luoyang_demo

# ── 缓存目录（计算节点无 /home 写权限）────────────────────────────────────────
export MPLCONFIGDIR=/shared_work/yuan/.config/matplotlib
export TRITON_CACHE_DIR=/shared_work/yuan/.triton/cache
# DINOv2 via torch.hub: default is ~/.cache/torch/hub → PermissionError on /home/yuan
export TORCH_HOME=/shared_work/yuan/.cache/torch
export PYTHONUNBUFFERED=1
mkdir -p "$MPLCONFIGDIR" "$TRITON_CACHE_DIR" "$TORCH_HOME/hub"

# ── 训练稳态参数（可在 sbatch 前通过环境变量覆盖）───────────────────────────────
NUM_WORKERS=${NUM_WORKERS:-0}

# ── 环境信息 ─────────────────────────────────────────────────────────────────
echo "=== job $SLURM_JOB_ID  host $(hostname)  task=$TASK ==="
echo "=== start $(date) ==="
"$YUAN_PY" -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"

# ── 根据 TASK 选配置文件 ──────────────────────────────────────────────────────
case "$TASK" in
  15m)  DATASET_CONF=conf_luoyang_2026_15m_sky.yaml ;;
  4h)   DATASET_CONF=conf_luoyang_2026_4h.yaml  ;;
  48h)  DATASET_CONF=conf_luoyang_2026_48h.yaml ;;
  *)    echo "Unknown TASK=$TASK, use 15m / 4h / 48h"; exit 1 ;;
esac

# ── 训练 ─────────────────────────────────────────────────────────────────────
cd "$PROJECT_ROOT"
"$YUAN_PY" -u training/train_vit_luoyang2026total.py \
    --task "$TASK" \
    --dataset-config "$DATASET_CONF" \
    --config conf_train.yaml \
    --num_workers "$NUM_WORKERS" \
    --huber-delta 2000 \
    --checkpoint_dir /shared_work/yuan/workspace/luoyang_demo/checkpoints_15m_dinov2_scratch1 \
    --model pv_forecasting_model_vit_dinov2

"$YUAN_PY" -u training/train_vit_luoyang2026total.py \
    --task "$TASK" \
    --dataset-config "$DATASET_CONF" \
    --config conf_train.yaml \
    --num_workers "$NUM_WORKERS" \
    --huber-delta 2000 \
    --checkpoint_dir /shared_work/yuan/workspace/luoyang_demo/checkpoints_15m_dinov2_scratch2 \
    --model pv_forecasting_model_vit_dinov2

"$YUAN_PY" -u training/train_vit_luoyang2026total.py \
    --task "$TASK" \
    --dataset-config "$DATASET_CONF" \
    --config conf_train.yaml \
    --num_workers "$NUM_WORKERS" \
    --huber-delta 2000 \
    --checkpoint_dir /shared_work/yuan/workspace/luoyang_demo/checkpoints_15m_dinov2_scratch3 \
    --model pv_forecasting_model_vit_dinov2

"$YUAN_PY" -u training/train_vit_luoyang2026total.py \
    --task "$TASK" \
    --dataset-config "$DATASET_CONF" \
    --config conf_train.yaml \
    --num_workers "$NUM_WORKERS" \
    --huber-delta 2000 \
    --checkpoint_dir /shared_work/yuan/workspace/luoyang_demo/checkpoints_15m_dinov2_scratch4 \
    --model pv_forecasting_model_vit_dinov2

"$YUAN_PY" -u training/train_vit_luoyang2026total.py \
    --task "$TASK" \
    --dataset-config "$DATASET_CONF" \
    --config conf_train.yaml \
    --num_workers "$NUM_WORKERS" \
    --huber-delta 2000 \
    --checkpoint_dir /shared_work/yuan/workspace/luoyang_demo/checkpoints_15m_dinov2_scratch5 \
    --model pv_forecasting_model_vit_dinov2

echo "=== done $(date) ==="
