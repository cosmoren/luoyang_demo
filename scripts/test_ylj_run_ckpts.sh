#!/usr/bin/env bash
# Test (test_only) every checkpoint in one run folder (run1, run2, ...).
# Exports one seqpairs CSV per checkpoint. Modality flags MUST match training.
# Prints a metric summary and highlights the best checkpoint by MAE / RMSE.
#
# Usage:
#   bash scripts/test_ylj_run_ckpts.sh run1
#   RUN=run5 BASE_DIR=/path/to/checkpoints_ylj_48h_4h_pv_ghi bash scripts/test_ylj_run_ckpts.sh
#
# Tunables (override via env):
#   RUN       run subfolder name, e.g. run1 (required; or pass as $1)
#   BASE_DIR  checkpoint root holding run* subdirs (default checkpoints_ylj_48h_4h_<tag>)
#   USE_ALL / USE_GHI / ... (see _ylj_parquet_hist_flags.inc.sh)
#   USE_NWP   1 -> add --ylj_parquet_nwp            (default 0; must match training)
#   USE_SAT   1 -> add --ylj_sat_zarr               (default 0; must match training)
#   USE_TCN_MULTI_KERNEL  1 -> --tcn_multi_kernel  (default 0; must match training)
#   USE_CROSS_ATTN_2LAYER 1 -> --cross_attn_2layer  (default 0; must match training)
#   USE_NWP_RESIDUAL  1 -> --nwp_residual           (default 0; must match training)
#   USE_HIST_COMPRESSION  1 -> --hist_compression     (default 0; must match training)
#   USE_HIST_COMPRESSION_COND_FORECAST 1 -> --hist_compression_cond_forecast (needs USE_HIST_COMPRESSION=1)
#   USE_SPLIT_PV_SAT_ATTN 1 -> --split_pv_sat_attn       (default 0; must match training)
#   USE_LAST_K_HEAD  1 -> --last_k_head                  (default 0; must match training)
#   LAST_K_HEAD_K    steps for last-K head               (default 8)
#   CKPT_GLOB glob for checkpoints in the run dir (default 'pv_forecast_epoch_*.pt')
#   EPOCH_START  skip checkpoints with epoch < this (default 11)
#   OUT_PREFIX prefix for output CSVs (default test_seqpairs_)
#   PYTHON    python interpreter            (default python)

set -euo pipefail

ROOT="/work/yang/luoyang_demo_0521"
CFG="$ROOT/config/datasets/conf_ylj.yaml"

RUN=${RUN:-${1:-}}
USE_NWP=${USE_NWP:-1}
USE_SAT=${USE_SAT:-1}
USE_TCN_MULTI_KERNEL=${USE_TCN_MULTI_KERNEL:-1}
USE_CROSS_ATTN_2LAYER=${USE_CROSS_ATTN_2LAYER:-0}
USE_NWP_RESIDUAL=${USE_NWP_RESIDUAL:-0}
USE_HIST_COMPRESSION=${USE_HIST_COMPRESSION:-1}
USE_HIST_COMPRESSION_COND_FORECAST=${USE_HIST_COMPRESSION_COND_FORECAST:-0}
USE_SPLIT_PV_SAT_ATTN=${USE_SPLIT_PV_SAT_ATTN:-0}
USE_LAST_K_HEAD=${USE_LAST_K_HEAD:-0}
LAST_K_HEAD_K=${LAST_K_HEAD_K:-8}
CKPT_GLOB=${CKPT_GLOB:-pv_forecast_epoch_*.pt}
EPOCH_START=${EPOCH_START:-11}
OUT_PREFIX=${OUT_PREFIX:-test_seqpairs_}
PYTHON=${PYTHON:-python}

_ckpt_epoch_num() {
  local base
  base=$(basename "$1" .pt)
  if [[ "$base" =~ epoch_([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo ""
  fi
}

if [[ -z "$RUN" ]]; then
  echo "[test_ylj_run_ckpts] ERROR: specify RUN (e.g. run1) as \$1 or RUN=run1" >&2
  exit 1
fi

# Allow "1" -> "run1"
if [[ "$RUN" =~ ^[0-9]+$ ]]; then
  RUN="run${RUN}"
fi

TAG="pv"
# shellcheck source=scripts/_ylj_parquet_hist_flags.inc.sh
source "$ROOT/scripts/_ylj_parquet_hist_flags.inc.sh"
ylj_build_parquet_hist_flags

NWP_FLAG=""
[ "$USE_NWP" = "1" ] && { NWP_FLAG="--ylj_parquet_nwp"; TAG="${TAG}_nwp"; }
SAT_FLAG=""
[ "$USE_SAT" = "1" ] && { SAT_FLAG="--ylj_sat_zarr"; TAG="${TAG}_sat"; }
TCN_MULTI_KERNEL_FLAG=""
[ "$USE_TCN_MULTI_KERNEL" = "1" ] && { TCN_MULTI_KERNEL_FLAG="--tcn_multi_kernel"; TAG="${TAG}_mk"; }
CROSS_ATTN_2LAYER_FLAG=""
[ "$USE_CROSS_ATTN_2LAYER" = "1" ] && { CROSS_ATTN_2LAYER_FLAG="--cross_attn_2layer"; TAG="${TAG}_ca2"; }
NWP_RESIDUAL_FLAG=""
[ "$USE_NWP_RESIDUAL" = "1" ] && { NWP_RESIDUAL_FLAG="--nwp_residual"; TAG="${TAG}_nwpres"; }
HIST_COMPRESSION_FLAG=""
[ "$USE_HIST_COMPRESSION" = "1" ] && { HIST_COMPRESSION_FLAG="--hist_compression"; TAG="${TAG}_hcmp"; }
HIST_COMPRESSION_COND_FORECAST_FLAG=""
if [ "$USE_HIST_COMPRESSION_COND_FORECAST" = "1" ]; then
  if [ "$USE_HIST_COMPRESSION" != "1" ]; then
    echo "[test_ylj_run_ckpts] ERROR: USE_HIST_COMPRESSION_COND_FORECAST=1 requires USE_HIST_COMPRESSION=1" >&2
    exit 1
  fi
  HIST_COMPRESSION_COND_FORECAST_FLAG="--hist_compression_cond_forecast"
  TAG="${TAG}_fc"
fi
SPLIT_PV_SAT_ATTN_FLAG=""
[ "$USE_SPLIT_PV_SAT_ATTN" = "1" ] && { SPLIT_PV_SAT_ATTN_FLAG="--split_pv_sat_attn"; TAG="${TAG}_splitattn"; }
LAST_K_HEAD_FLAG=""
if [ "$USE_LAST_K_HEAD" = "1" ]; then
  LAST_K_HEAD_FLAG="--last_k_head --last_k_head_k $LAST_K_HEAD_K"
  TAG="${TAG}_lk${LAST_K_HEAD_K}"
fi

BASE_DIR=${BASE_DIR:-"$ROOT/checkpoints_ylj_48h_4h_${TAG}"}
CKPT_DIR="$BASE_DIR/$RUN"

if [[ ! -d "$CKPT_DIR" ]]; then
  echo "[test_ylj_run_ckpts] ERROR: run folder not found: $CKPT_DIR" >&2
  exit 1
fi

cd "$ROOT"

mapfile -t _ALL_CKPTS < <(find "$CKPT_DIR" -maxdepth 1 -type f -name "$CKPT_GLOB" | sort -V)
CKPTS=()
for _c in "${_ALL_CKPTS[@]}"; do
  _ep=$(_ckpt_epoch_num "$_c")
  if [[ -n "$_ep" && "$_ep" -ge "$EPOCH_START" ]]; then
    CKPTS+=("$_c")
  fi
done
if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "[test_ylj_run_ckpts] ERROR: no checkpoints matching $CKPT_GLOB with epoch>=$EPOCH_START in $CKPT_DIR" >&2
  exit 1
fi

METRICS_FILE=$(mktemp)
trap 'rm -f "$METRICS_FILE"' EXIT

_parse_ylj_eval_metrics() {
  local log_file=$1
  local eval_line
  eval_line=$(grep '\[ylj eval\]' "$log_file" | tail -1 || true)
  if [[ -z "$eval_line" ]]; then
    echo "nan nan"
    return
  fi
  awk '
    /\[ylj eval\]/ {
      mae = "nan"; rmse = "nan"
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^MAE=/)  { mae  = substr($i, 5) }
        if ($i ~ /^RMSE=/) { rmse = substr($i, 6) }
      }
      print mae, rmse
    }
  ' <<< "$eval_line"
}

echo "[test_ylj_run_ckpts] tag=$TAG use_all=${USE_ALL:-0} nwp=$USE_NWP sat=$USE_SAT tcn_multi_kernel=$USE_TCN_MULTI_KERNEL cross_attn_2layer=$USE_CROSS_ATTN_2LAYER nwp_residual=$USE_NWP_RESIDUAL hist_compression=$USE_HIST_COMPRESSION hist_compression_cond_forecast=$USE_HIST_COMPRESSION_COND_FORECAST split_pv_sat_attn=$USE_SPLIT_PV_SAT_ATTN last_k_head=$USE_LAST_K_HEAD last_k_head_k=$LAST_K_HEAD_K"
echo "[test_ylj_run_ckpts] parquet hist flags: ${PARQUET_HIST_FLAGS:-<none>}"
echo "[test_ylj_run_ckpts] run=$RUN  dir=$CKPT_DIR  ckpts=${#CKPTS[@]} (epoch>=$EPOCH_START, found ${#_ALL_CKPTS[@]} total)"

tested=0
for CKPT in "${CKPTS[@]}"; do
  ckpt_base=$(basename "$CKPT" .pt)
  out_suffix="${ckpt_base#pv_forecast_}"
  OUT_CSV="$CKPT_DIR/${OUT_PREFIX}${out_suffix}.csv"

  echo "=================================================================="
  echo "[$RUN] testing $(basename "$CKPT") -> $(basename "$OUT_CSV")"

  LOG=$(mktemp)
  set +e
  # shellcheck disable=SC2086
  $PYTHON training/train_ylj.py \
    $TCN_MULTI_KERNEL_FLAG \
    $CROSS_ATTN_2LAYER_FLAG \
    $NWP_RESIDUAL_FLAG \
    $HIST_COMPRESSION_FLAG \
    $HIST_COMPRESSION_COND_FORECAST_FLAG \
    $SPLIT_PV_SAT_ATTN_FLAG \
    $LAST_K_HEAD_FLAG \
    --config "$CFG" \
    --ylj_raw_parquet $PARQUET_HIST_FLAGS $NWP_FLAG $SAT_FLAG \
    --test_only \
    --checkpoint_dir "$CKPT_DIR" \
    --test_only_ckpt "$CKPT" \
    --test_only_plus15_csv "$OUT_CSV" 2>&1 | tee "$LOG"
  train_status=${PIPESTATUS[0]}
  set -e

  if [[ "$train_status" -ne 0 ]]; then
    rm -f "$LOG"
    echo "[test_ylj_run_ckpts] ERROR: train_ylj.py failed for $(basename "$CKPT") (exit $train_status)" >&2
    exit "$train_status"
  fi

  read -r mae rmse <<< "$(_parse_ylj_eval_metrics "$LOG")"
  rm -f "$LOG"
  echo "$ckpt_base $mae $rmse" >> "$METRICS_FILE"

  echo "[$RUN] done -> $OUT_CSV  (MAE=$mae  RMSE=$rmse)"
  tested=$((tested + 1))
done

echo "=================================================================="
echo "[test_ylj_run_ckpts] metric summary for $RUN"
printf "%-28s %12s %12s\n" "checkpoint" "MAE" "RMSE"
while read -r ckpt mae rmse; do
  printf "%-28s %12s %12s\n" "$ckpt" "$mae" "$rmse"
done < "$METRICS_FILE"

awk '
  NF >= 3 && $2 != "nan" && $3 != "nan" {
    if (!have_mae || $2 + 0 < best_mae + 0) { best_mae = $2 + 0; best_mae_ckpt = $1 }
    if (!have_rmse || $3 + 0 < best_rmse + 0) { best_rmse = $3 + 0; best_rmse_ckpt = $1 }
    have_mae = 1; have_rmse = 1
  }
  END {
    if (!have_mae) {
      print "[test_ylj_run_ckpts] WARNING: could not parse MAE/RMSE from test output"
      exit 0
    }
    print "------------------------------------------------------------------"
    printf "[test_ylj_run_ckpts] best MAE  -> %s  (MAE=%.6f)\n", best_mae_ckpt, best_mae
    printf "[test_ylj_run_ckpts] best RMSE -> %s  (RMSE=%.6f)\n", best_rmse_ckpt, best_rmse
  }
' "$METRICS_FILE"

echo "=================================================================="
echo "[test_ylj_run_ckpts] done. tested=$tested ckpts in $CKPT_DIR"
