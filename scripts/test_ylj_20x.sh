#!/usr/bin/env bash
# Test (test_only) runs in a checkpoint folder produced by train_ylj_20x.sh.
# By default tests every run* subdir; pass run names to test a subset only.
#
# Usage:
#   bash scripts/test_ylj_20x.sh                    # all run* folders
#   bash scripts/test_ylj_20x.sh run1 run3          # only run1 and run3
#   bash scripts/test_ylj_20x.sh 1 6                # run1 through run6 (two plain numbers)
#   bash scripts/test_ylj_20x.sh 1-6                # same range, one arg
#   bash scripts/test_ylj_20x.sh run1 run3          # run1 and run3 only (not a range)
#   RUNS="1-6" bash scripts/test_ylj_20x.sh         # same via env
#   RUNS="run2 run4" bash scripts/test_ylj_20x.sh
#   EPOCH=15 bash scripts/test_ylj_20x.sh 1 6        # epoch 15, runs 1..6
#   bash scripts/test_ylj_20x.sh --epoch 15 1 6      # same via flag
#   EPOCH=latest bash scripts/test_ylj_20x.sh 1 6    # latest epoch per run
#
# Tunables (override via env), e.g.:
#   BASE_DIR=/data/luoyang_demo_0521/checkpoints_ylj_48h_4h_pv_ghi bash scripts/test_ylj_20x.sh
#
#   BASE_DIR  checkpoint folder holding run* subdirs (default checkpoints_ylj_48h_4h_<tag>)
#   RUNS      run list or range, e.g. "1-6", "run1 run3" (default: all run*)
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
#   EPOCH     optional: N -> pv_forecast_epoch_N.pt; "latest" -> highest epoch per run
#             (default unset -> pv_forecast_last.pt, same as before)
#   CKPT_NAME checkpoint file per run       (default pv_forecast_last.pt when EPOCH unset)
#   OUT_NAME  output CSV name per run       (default test_seqpairs.csv; epoch suffix if EPOCH set)
#   PYTHON    python interpreter            (default python)

set -euo pipefail

ROOT="/work/yang/luoyang_demo_0521"
CFG="$ROOT/config/datasets/conf_ylj.yaml"

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
PYTHON=${PYTHON:-python}

# Parse --epoch N / --epoch=N before run-folder args.
RUN_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --epoch)
      if [[ $# -lt 2 ]]; then
        echo "[test_ylj_20x] ERROR: --epoch requires a number" >&2
        exit 1
      fi
      EPOCH="$2"
      shift 2
      ;;
    --epoch=*)
      EPOCH="${1#*=}"
      shift
      ;;
    *)
      RUN_ARGS+=("$1")
      shift
      ;;
  esac
done
set -- "${RUN_ARGS[@]}"

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

_resolve_ckpt_for_run() {
  local dir=$1
  if [[ -n "${EPOCH:-}" ]]; then
    if [[ "$EPOCH" == "latest" ]]; then
      _latest_epoch_ckpt "$dir"
      return
    fi
    if [[ ! "$EPOCH" =~ ^[0-9]+$ ]]; then
      echo "[test_ylj_20x] ERROR: EPOCH must be a non-negative integer or 'latest', got: $EPOCH" >&2
      exit 1
    fi
    echo "$dir/pv_forecast_epoch_${EPOCH}.pt"
    return
  fi
  echo "$dir/${CKPT_NAME:-pv_forecast_last.pt}"
}

_resolve_out_csv_for_run() {
  local ckpt_dir=$1 ckpt_path=$2
  if [[ -n "${OUT_NAME:-}" ]]; then
    echo "$ckpt_dir/$OUT_NAME"
    return
  fi
  if [[ -n "${EPOCH:-}" ]]; then
    echo "$ckpt_dir/$(_default_out_name_for_ckpt "$ckpt_path")"
    return
  fi
  echo "$ckpt_dir/test_seqpairs.csv"
}

_default_out_name_for_ckpt() {
  local ckpt_path=$1
  local base
  base=$(basename "$ckpt_path" .pt)
  if [[ "$base" =~ ^pv_forecast_epoch_([0-9]+)$ ]]; then
    echo "test_seqpairs_epoch_${BASH_REMATCH[1]}.csv"
  else
    echo "test_seqpairs.csv"
  fi
}

_run_index() {
  local r=$1
  if [[ "$r" =~ ^run([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}"
  elif [[ "$r" =~ ^[0-9]+$ ]]; then
    echo "$r"
  else
    return 1
  fi
}

_append_run_range() {
  local lo=$1 hi=$2
  local -n _dirs=$3
  if (( lo > hi )); then
    local t=$lo
    lo=$hi
    hi=$t
  fi
  local i
  for ((i = lo; i <= hi; i++)); do
    _dirs+=("$BASE_DIR/run$i")
  done
}

_append_run_spec() {
  local arg=$1
  local -n _dirs=$2
  if [[ "$arg" =~ ^([0-9]+)-([0-9]+)$ ]]; then
    _append_run_range "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" _dirs
  elif [[ "$arg" =~ ^run([0-9]+)-run([0-9]+)$ ]]; then
    _append_run_range "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" _dirs
  else
    _dirs+=("$BASE_DIR/$(_normalize_run_name "$arg")")
  fi
}

_expand_run_args() {
  local -n _out=$1
  shift
  local args=("$@")
  # Two plain integers -> inclusive range (e.g. "1 6" -> run1..run6).
  if [[ ${#args[@]} -eq 2 && "${args[0]}" =~ ^[0-9]+$ && "${args[1]}" =~ ^[0-9]+$ ]]; then
    _append_run_range "${args[0]}" "${args[1]}" _out
    return
  fi
  local arg
  for arg in "${args[@]}"; do
    _append_run_spec "$arg" _out
  done
}

_normalize_run_name() {
  local r=$1
  if [[ "$r" =~ ^[0-9]+$ ]]; then
    echo "run${r}"
  else
    echo "$r"
  fi
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
    echo "[test_ylj_20x] ERROR: USE_HIST_COMPRESSION_COND_FORECAST=1 requires USE_HIST_COMPRESSION=1" >&2
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

if [[ ! -d "$BASE_DIR" ]]; then
  echo "[test_ylj_20x] ERROR: BASE_DIR not found: $BASE_DIR" >&2
  exit 1
fi

cd "$ROOT"

RUN_DIRS=()
if [[ $# -gt 0 ]]; then
  _expand_run_args RUN_DIRS "$@"
elif [[ -n "${RUNS:-}" ]]; then
  # shellcheck disable=SC2206
  _expand_run_args RUN_DIRS ${RUNS}
else
  mapfile -t RUN_DIRS < <(find "$BASE_DIR" -maxdepth 1 -type d -name 'run*' | sort -V)
fi

if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
  echo "[test_ylj_20x] ERROR: no run folders to test under $BASE_DIR" >&2
  exit 1
fi

if [[ -n "${CKPT_NAME:-}" ]]; then
  _ckpt_mode="ckpt=$CKPT_NAME"
elif [[ -n "${EPOCH:-}" ]]; then
  _ckpt_mode="epoch=$EPOCH"
else
  _ckpt_mode="epoch=latest"
fi
echo "[test_ylj_20x] tag=$TAG $_ckpt_mode use_all=${USE_ALL:-0} nwp=$USE_NWP sat=$USE_SAT tcn_multi_kernel=$USE_TCN_MULTI_KERNEL cross_attn_2layer=$USE_CROSS_ATTN_2LAYER nwp_residual=$USE_NWP_RESIDUAL hist_compression=$USE_HIST_COMPRESSION hist_compression_cond_forecast=$USE_HIST_COMPRESSION_COND_FORECAST split_pv_sat_attn=$USE_SPLIT_PV_SAT_ATTN last_k_head=$USE_LAST_K_HEAD last_k_head_k=$LAST_K_HEAD_K  base=$BASE_DIR"
echo "[test_ylj_20x] parquet hist flags: ${PARQUET_HIST_FLAGS:-<none>}"
echo "[test_ylj_20x] testing ${#RUN_DIRS[@]} run folder(s)"

tested=0
skipped=0
for CKPT_DIR in "${RUN_DIRS[@]}"; do
  name=$(basename "$CKPT_DIR")
  if [[ ! -d "$CKPT_DIR" ]]; then
    echo "[skip] $name: folder not found under $BASE_DIR"
    skipped=$((skipped + 1))
    continue
  fi

  CKPT=$(_resolve_ckpt_for_run "$CKPT_DIR")
  if [[ -z "$CKPT" ]]; then
    echo "[skip] $name: no pv_forecast_epoch_*.pt found"
    skipped=$((skipped + 1))
    continue
  fi
  if [[ ! -f "$CKPT" ]]; then
    echo "[skip] $name: missing $(basename "$CKPT")"
    skipped=$((skipped + 1))
    continue
  fi
  if [[ -n "${OUT_NAME:-}" ]]; then
    OUT_CSV="$CKPT_DIR/$OUT_NAME"
  else
    OUT_CSV="$CKPT_DIR/$(_default_out_name_for_ckpt "$CKPT")"
  fi

  echo "=================================================================="
  echo "[$name] testing $CKPT -> $OUT_CSV"
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
    --test_only_plus15_csv "$OUT_CSV"

  echo "[$name] done -> $OUT_CSV"
  tested=$((tested + 1))
done

echo "=================================================================="
echo "[test_ylj_20x] done. tested=$tested skipped=$skipped  base=$BASE_DIR"
