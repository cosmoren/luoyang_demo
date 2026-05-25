#!/usr/bin/env bash
# Train YLJ Parquet + NWP for a few epochs (smoke / resume test).
set -euo pipefail

ROOT="/data/luoyang_demo_0521"
CFG="$ROOT/config/datasets/conf_ylj.yaml"
CKPT_DIR="${1:-ylj_48h_48h_nwp_k_raw_parquet}"

cd "$ROOT"
python training/train_ylj.py \
  --config "$CFG" \
  --ylj_raw_parquet \
  --ylj_parquet_nwp \
  --checkpoint_dir "$CKPT_DIR" \
  --epochs "${2:-2}" \
  --batch_size "${3:-8}"
