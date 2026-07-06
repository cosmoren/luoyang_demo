#!/usr/bin/env bash
#SBATCH --job-name=yuan_probe
#SBATCH --partition=gb10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=10
#SBATCH --output=/shared_work/yuan/workspace/luoyang_demo/slurm/logs/probe_%j.log
set -eo pipefail

export MPLCONFIGDIR=/shared_work/yuan/.config/matplotlib
export TRITON_CACHE_DIR=/shared_work/yuan/.triton/cache

YUAN_PY=/shared_work/yuan/micromamba/envs/SimVP/bin/python
PROJECT_ROOT=/shared_work/yuan/workspace/luoyang_demo

echo "=== host $(hostname); python $($YUAN_PY --version) ==="
echo "=== GPU check ==="
"$YUAN_PY" -c "
import torch
print('torch version :', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device name   :', torch.cuda.get_device_name(0))
    print('cuda capability:', torch.cuda.get_device_capability(0))
"

echo "=== key package check ==="
"$YUAN_PY" -c "
import importlib, sys
pkgs = ['numpy', 'pandas', 'yaml', 'zarr', 'torch', 'tensorboard']
for p in pkgs:
    try:
        m = importlib.import_module(p)
        print(f'  {p}: OK  ({getattr(m, \"__version__\", \"?\")})')
    except ImportError as e:
        print(f'  {p}: MISSING ({e})', file=sys.stderr)
"

echo "=== project import check ==="
cd "$PROJECT_ROOT"
"$YUAN_PY" -c "
import sys; sys.path.insert(0, '.')
from dataloader.luoyang_2026total_zarr import PVDataset
from models.models import pv_forecasting_model_vit_imgs
print('  project imports: OK')
"

echo "=== done ==="
