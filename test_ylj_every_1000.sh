#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/luoyang_demo_0521"
CFG="$ROOT/config/datasets/conf_ylj.yaml"
CKPT_DIR="$ROOT/checkpoints_ylj_48h_4h_nwp_kt_raw_parquet_synthetic_real_smoothed_ssrd"
OUT_DIR="$CKPT_DIR/test_csvs_every10_seqpairs"

mkdir -p "$OUT_DIR"

for epoch in $(seq 2 1 10); do
  CKPT="$CKPT_DIR/pv_forecast_epoch_${epoch}.pt"
  if [[ ! -f "$CKPT" ]]; then
    echo "[skip] missing checkpoint: $CKPT"
    continue
  fi

  OUT_CSV="$OUT_DIR/test_epoch_${epoch}_seqpairs.csv"

  echo "[run ] epoch=$epoch"
  python "$ROOT/training/train_ylj.py" \
    --config "$CFG" \
    --ylj_raw_parquet \
    --ylj_parquet_nwp \
    --test_only \
    --checkpoint_dir "$CKPT_DIR" \
    --test_only_ckpt "$CKPT" \
    --test_only_plus15_csv "$OUT_CSV"

  echo "[done] saved: $OUT_CSV"
done

echo "All done. Outputs in: $OUT_DIR"
