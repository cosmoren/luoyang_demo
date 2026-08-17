#!/usr/bin/env bash
#SBATCH --job-name=luoyang_2026_total_4h
#SBATCH --partition=energy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/shared_work/yuan/workspace/luoyang_demo/slurm/logs/train_%x_%j_4h.log
set -eo pipefail

# ── 任务选择：15m / 4h / 48h ─────────────────────────────────────────────────
TASK=${TASK:-4h}

# ── 路径 ─────────────────────────────────────────────────────────────────────
YUAN_PY=/shared_work/yuan/micromamba/envs/SimVP/bin/python
PROJECT_ROOT=/shared_work/yuan/workspace/luoyang_demo

# ── 缓存目录（计算节点无 /home 写权限）────────────────────────────────────────
export MPLCONFIGDIR=/shared_work/yuan/.config/matplotlib
export TRITON_CACHE_DIR=/shared_work/yuan/.triton/cache
export HF_HOME=/shared_work/yuan/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/shared_work/yuan/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/shared_work/yuan/.cache/huggingface/transformers
export TORCH_HOME=/shared_work/yuan/.cache/torch
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME"

# ── 训练稳态参数（可在 sbatch 前通过环境变量覆盖）───────────────────────────────
NUM_WORKERS=${NUM_WORKERS:-0}
VJEPA_HF_REPO=${VJEPA_HF_REPO:-/shared_work/yuan/workspace/ckpt/vjepa2/hf_sat_e6}

# ── 环境信息 ─────────────────────────────────────────────────────────────────
echo "=== job $SLURM_JOB_ID  host $(hostname)  task=$TASK ==="
echo "=== start $(date) ==="
"$YUAN_PY" -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"

# ── 根据 TASK 选配置文件 ──────────────────────────────────────────────────────
case "$TASK" in
  15m)  DATASET_CONF=conf_luoyang_2026_15m_sky.yaml ;;
  4h)   DATASET_CONF=conf_luoyang_2026_4h_sat.yaml  ;;
  48h)  DATASET_CONF=conf_luoyang_2026_48h.yaml ;;
  *)    echo "Unknown TASK=$TASK, use 15m / 4h / 48h"; exit 1 ;;
esac

# ── 训练 ─────────────────────────────────────────────────────────────────────
cd "$PROJECT_ROOT"

"$YUAN_PY" -u training/train_vit_luoyang2026total.py \
    --task "$TASK" \
    --dataset-config "$DATASET_CONF" \
    --config conf_train_finetune.yaml \
    --num_workers "$NUM_WORKERS" \
    --huber-delta 2000 \
    --model pv_forecasting_model_vit_vjepa \
    --vjepa-hf-repo "$VJEPA_HF_REPO" \
    --checkpoint_dir /shared_work/yuan/workspace/luoyang_demo/checkpoints_4h_vjepa_e6_tabm \
    --init-checkpoint /shared_work/yuan/workspace/luoyang_demo/checkpoints_4h_baseline/pv_forecast_vit_best_task_4h_gpu0.pt \
    --freeze-tabm

echo "=== done $(date) ==="
