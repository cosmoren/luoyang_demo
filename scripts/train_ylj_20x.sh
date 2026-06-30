#!/usr/bin/env bash
# Train the YLJ kt experiment NRUNS times. Training only; testing is a separate script.
#
# Runs are sequential and rely on existing training nondeterminism (random init
# + shuffle=True), so each run differs. No --seed flag is required.
#
# Usage: bash scripts/train_ylj_20x.sh [cuda_device]
#
# Tunables (override via env), e.g.:
#   bash scripts/train_ylj_20x.sh 0
#   NRUNS=20 USE_GHI=1 EPOCHS=50 BATCH=64 bash scripts/train_ylj_20x.sh 1
#   RESUME=1 BASE_DIR=checkpoints_ylj_48h_4h_pv_ghi EPOCHS=50 bash scripts/train_ylj_20x.sh 0
#
#   (arg)    CUDA device id                    (default 0)
#   NRUNS    number of repeats (fresh train)   (default 10)
#   RUN_START first run id for fresh train      (default 1; e.g. RUN_START=6 NRUNS=10 -> run6..run15)
#   RESUME   1 -> resume every run* in BASE_DIR from latest pv_forecast_epoch_*.pt
#   USE_ALL  1 -> --ylj_parquet_use_all         (default 0; all 14 hist channels)
#   USE_GHI / USE_GHI_SOLARGIS / USE_TEMP_SOLARGIS (default 0 each)
#   USE_KT_RAMP / USE_GHI_RAMP / USE_GHI_ROLL_MEAN / USE_GHI_ROLL_STD
#   USE_OM_CLOUD_PCT / USE_OM_CLOUD_PCT_LOW_MID (default 0 each)
#   USE_WS_SOLARGIS / USE_WD_SOLARGIS / USE_PREC_SOLARGIS
#   USE_PWAT_SOLARGIS / USE_SDWE_SOLARGIS (default 0 each)
#   USE_NWP  1 -> add --ylj_parquet_nwp         (default 0)
#   USE_SAT  1 -> add --ylj_sat_zarr            (default 0)
#   USE_TCN_MULTI_KERNEL  1 -> --tcn_multi_kernel  (default 0; kernels [3,7,11,15,31])
#   USE_CROSS_ATTN_2LAYER 1 -> --cross_attn_2layer  (default 0)
#   USE_NWP_RESIDUAL  1 -> --nwp_residual           (default 0; concat NWP skip into head)
#   USE_HIST_COMPRESSION  1 -> --hist_compression     (default 0; PV history 192->48)
#   USE_HIST_COMPRESSION_COND_FORECAST 1 -> --hist_compression_cond_forecast (needs USE_HIST_COMPRESSION=1)
#   USE_SPLIT_PV_SAT_ATTN 1 -> --split_pv_sat_attn       (default 0; separate PV/sat cross-attn)
#   USE_LAST_K_HEAD  1 -> --last_k_head                  (default 0; last-K kt+parquet skip into head)
#   LAST_K_HEAD_K    steps for last-K head               (default 8; tag suffix _lk8)
#   EPOCHS   training epochs per run           (default 20)
#   BATCH    batch size                        (default 64)
#   BASE_DIR output root                       (default checkpoints_ylj_48h_4h_<tag>)
#   PYTHON   python interpreter                (default python)

set -euo pipefail

CUDA_DEV="${1:-0}"

ROOT="/work/yang/luoyang_demo_0521"
CFG="$ROOT/config/datasets/conf_ylj.yaml"

NRUNS=${NRUNS:-10}
RUN_START=${RUN_START:-1}
RESUME=${RESUME:-0}
USE_NWP=${USE_NWP:-1}
USE_SAT=${USE_SAT:-1}
USE_TCN_MULTI_KERNEL=${USE_TCN_MULTI_KERNEL:-1}
USE_CROSS_ATTN_2LAYER=${USE_CROSS_ATTN_2LAYER:-0}
USE_NWP_RESIDUAL=${USE_NWP_RESIDUAL:-0}
USE_HIST_COMPRESSION=${USE_HIST_COMPRESSION:-0}
USE_HIST_COMPRESSION_COND_FORECAST=${USE_HIST_COMPRESSION_COND_FORECAST:-0}
USE_SPLIT_PV_SAT_ATTN=${USE_SPLIT_PV_SAT_ATTN:-1}
USE_LAST_K_HEAD=${USE_LAST_K_HEAD:-1}
LAST_K_HEAD_K=${LAST_K_HEAD_K:-8}
EPOCHS=${EPOCHS:-20}
BATCH=${BATCH:-64}
PYTHON=${PYTHON:-python}

_latest_epoch_ckpt() {
  local dir=$1
  local latest="" ep=-1 f base n
  shopt -s nullglob
  for f in "$dir"/pv_forecast_epoch_*.pt; do
    base=$(basename "$f" .pt)
    [[ "$base" =~ ^pv_forecast_epoch_([0-9]+)$ ]] || continue
    n="${BASH_REMATCH[1]}"
    if (( n > ep )); then
      ep=$n
      latest=$f
    fi
  done
  shopt -u nullglob
  echo "$latest"
}

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
    echo "[train_ylj_20x] ERROR: USE_HIST_COMPRESSION_COND_FORECAST=1 requires USE_HIST_COMPRESSION=1" >&2
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

if [[ ! "$RUN_START" =~ ^[0-9]+$ ]] || (( RUN_START < 1 )); then
  echo "[train_ylj_20x] ERROR: RUN_START must be a positive integer, got: $RUN_START" >&2
  exit 1
fi

if [[ "$RESUME" = "1" ]]; then
  if [[ ! -d "$BASE_DIR" ]]; then
    echo "[train_ylj_20x] ERROR: RESUME=1 but BASE_DIR not found: $BASE_DIR" >&2
    exit 1
  fi
else
  mkdir -p "$BASE_DIR"
fi

cd "$ROOT"

RUN_DIRS=()
if [[ "$RESUME" = "1" ]]; then
  while IFS= read -r d; do
    RUN_DIRS+=("$d")
  done < <(find "$BASE_DIR" -maxdepth 1 -type d -name 'run*' | sort -V)
  if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
    echo "[train_ylj_20x] ERROR: RESUME=1 but no run* folders under $BASE_DIR" >&2
    exit 1
  fi
else
  run_end=$((RUN_START + NRUNS - 1))
  for i in $(seq "$RUN_START" "$run_end"); do
    RUN_DIRS+=("$BASE_DIR/run$i")
  done
fi

RESUME_FLAG=()
[[ "$RESUME" = "1" ]] && RESUME_FLAG=(--resume)

echo "[train_ylj_20x] cuda=$CUDA_DEV tag=$TAG resume=$RESUME runs=${#RUN_DIRS[@]} run_start=$RUN_START epochs=$EPOCHS batch=$BATCH"
echo "[train_ylj_20x] use_all=${USE_ALL:-0} nwp=$USE_NWP sat=$USE_SAT tcn_multi_kernel=$USE_TCN_MULTI_KERNEL cross_attn_2layer=$USE_CROSS_ATTN_2LAYER nwp_residual=$USE_NWP_RESIDUAL hist_compression=$USE_HIST_COMPRESSION hist_compression_cond_forecast=$USE_HIST_COMPRESSION_COND_FORECAST split_pv_sat_attn=$USE_SPLIT_PV_SAT_ATTN last_k_head=$USE_LAST_K_HEAD last_k_head_k=$LAST_K_HEAD_K"
echo "[train_ylj_20x] parquet hist flags: ${PARQUET_HIST_FLAGS:-<none>}"
echo "[train_ylj_20x] checkpoints -> $BASE_DIR"

run_idx=0
for CKPT_DIR in "${RUN_DIRS[@]}"; do
  run_idx=$((run_idx + 1))
  mkdir -p "$CKPT_DIR"
  run_name=$(basename "$CKPT_DIR")

  resume_note=""
  if [[ "$RESUME" = "1" ]]; then
    latest=$(_latest_epoch_ckpt "$CKPT_DIR")
    if [[ -n "$latest" ]]; then
      resume_note=" (resume from $(basename "$latest"))"
    else
      resume_note=" (no epoch ckpt; train_ylj.py may start at epoch 1 or use last.pt)"
    fi
  fi

  echo "=================================================================="
  echo "[${run_name} $run_idx/${#RUN_DIRS[@]}] training -> $CKPT_DIR${resume_note}"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$CUDA_DEV" $PYTHON training/train_ylj.py \
    $TCN_MULTI_KERNEL_FLAG \
    $CROSS_ATTN_2LAYER_FLAG \
    $NWP_RESIDUAL_FLAG \
    $HIST_COMPRESSION_FLAG \
    $HIST_COMPRESSION_COND_FORECAST_FLAG \
    $SPLIT_PV_SAT_ATTN_FLAG \
    $LAST_K_HEAD_FLAG \
    --config "$CFG" \
    --ylj_raw_parquet $PARQUET_HIST_FLAGS $NWP_FLAG $SAT_FLAG \
    --checkpoint_dir "$CKPT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH" \
    "${RESUME_FLAG[@]}"

  echo "[${run_name} $run_idx/${#RUN_DIRS[@]}] done -> $CKPT_DIR"
done

echo "=================================================================="
echo "[train_ylj_20x] all ${#RUN_DIRS[@]} run(s) done. Checkpoints under: $BASE_DIR"
