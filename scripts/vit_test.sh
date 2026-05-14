#!/usr/bin/env bash
# Run train_vit_test.py TOTAL times across multiple GPUs in parallel.
# Each GPU gets its own checkpoint / metrics file (gpu{N} suffix).
#
# Before launching training, satellite and sky zarr stores are rsynced from
# /work to /dev/shm (RAM-backed tmpfs) so all parallel runs share one fast
# in-memory copy with no NFS contention.  Only the first invocation does the
# rsync; subsequent calls (or parallel script launches) skip it via a done-flag
# protected by flock.
#
# Tunables: override via env, e.g. `GPUS="0 1" TOTAL=10 bash scripts/vit_test.sh`
# To force a fresh rsync (e.g. after dataset update): rm /dev/shm/luoyang_SPMF/.rsync_done

set -euo pipefail

GPUS=(${GPUS:-0 1 2 3})
TOTAL=${TOTAL:-20}
CONFIG=${CONFIG:-conf_train.yaml}

# ── /dev/shm setup ────────────────────────────────────────────────────────────
SRC="/work/datasets/luoyang_SPMF"
DST="/dev/shm/luoyang_SPMF"
LOCK="/tmp/luoyang_shm_rsync.lock"
DONE="${DST}/.rsync_done"

echo "[shm] Checking /dev/shm cache ..."
(
    flock -x 9
    if [ ! -f "${DONE}" ]; then
        echo "[shm] Copying hot data to /dev/shm (~1 GB, one-time) ..."
        mkdir -p "${DST}"
        rsync -a --info=progress2 "${SRC}/sat_himawari_zarr1" "${DST}/"
        rsync -a --info=progress2 "${SRC}/skimg_zarr"         "${DST}/"
        rsync -a                  "${SRC}/NWP"                "${DST}/"
        rsync -a                  "${SRC}/info.yaml"          "${DST}/"
        touch "${DONE}"
        echo "[shm] Done. $(du -sh ${DST} | cut -f1) in /dev/shm."
    else
        echo "[shm] Already cached (${DONE} exists), skipping rsync."
    fi
) 9>"${LOCK}"
# ──────────────────────────────────────────────────────────────────────────────

N=${#GPUS[@]}
if [ "$N" -eq 0 ]; then
    echo "No GPUs configured" >&2
    exit 1
fi

pids=()
for idx in "${!GPUS[@]}"; do
    g=${GPUS[$idx]}
    count=$(( TOTAL / N + (idx < TOTAL % N ? 1 : 0) ))
    if [ "$count" -le 0 ]; then
        continue
    fi
    (
        for i in $(seq 1 "$count"); do
            echo "[gpu${g}] run ${i}/${count}"
            CUDA_VISIBLE_DEVICES=${g} python training/train_vit_test.py --config "${CONFIG}"
        done
    ) &
    pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
exit "$status"
