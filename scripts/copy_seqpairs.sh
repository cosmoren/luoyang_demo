#!/usr/bin/env bash
# Copy only test_seqpairs.csv files from SRC to DST, preserving the directory
# structure relative to SRC (empty dirs are pruned).
#
# Usage:
#   bash scripts/copy_seqpairs.sh SRC_DIR DST_DIR
# Example:
#   bash scripts/copy_seqpairs.sh \
#     /data/luoyang_demo_0521/checkpoints_ylj_48h_4h_pv_ghi \
#     /data/results/ylj_pv_ghi
#
# Override the filename to copy via FILE_NAME (default test_seqpairs.csv).

set -euo pipefail

SRC=${1:?"usage: copy_seqpairs.sh SRC_DIR DST_DIR"}
DST=${2:?"usage: copy_seqpairs.sh SRC_DIR DST_DIR"}
FILE_NAME=${FILE_NAME:-test_seqpairs.csv}

if [[ ! -d "$SRC" ]]; then
  echo "[copy_seqpairs] ERROR: SRC not found: $SRC" >&2
  exit 1
fi

mkdir -p "$DST"

# Resolve absolute DST so it stays valid after cd into SRC.
DST_ABS=$(cd "$DST" && pwd)

# Copy each match preserving its path relative to SRC.
count=0
cd "$SRC"
while IFS= read -r -d '' f; do
  rel=${f#./}
  mkdir -p "$DST_ABS/$(dirname "$rel")"
  cp -p "$f" "$DST_ABS/$rel"
  count=$((count + 1))
done < <(find . -type f -name "$FILE_NAME" -print0)
echo "[copy_seqpairs] copied $count '$FILE_NAME' file(s)"
echo "[copy_seqpairs] from: $SRC"
echo "[copy_seqpairs]   to: $DST"
