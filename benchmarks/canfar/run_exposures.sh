#!/bin/bash
# Launch one headless CANFAR session per MegaCam observation id.
#
# Contract (in the scripts this ships):
#   * process_exposure.py downloads the exposure to /scratch and streams the
#     reviewable artifacts to the shared mount as it proceeds;
#   * verify_exposures.py re-derives every number from the mask pixels in a
#     *separate* session, and writes results/ns-review/streak_summary.tsv.
#
# This launcher exists because two of the facts that decide whether a run
# finishes are environment facts, both measured on 2026-10-14:
#
#   * WM_THREAD_PIN=1.  pod_run.py drops the image's OMP/MKL defaults (they
#     perturbed the parallel-efficiency campaign), and the libraries then fall
#     back to the *node's* CPU count -- 192 per worker on these pods -- so 16
#     HDUs ask for up to 16x192 threads on the 16 cores the pod was granted.
#     Measured on 1851960, 16 workers, same exposure: per-HDU wall 2.6-4.0 ks
#     before, 0.8-1.0 ks pinned.  Crowded fields also need the cost bound in
#     weightmask.yml (max_candidates/max_rejected) or the rejection cascade in
#     _detect_trails_sparse_ransac is unbounded.
#   * WM_TIMEOUT is a *processing* budget, measured from when WM_COMMAND starts,
#     and it is not a deadline the pod survives: pod_run kills the child and the
#     partial mask keeps only its unrenamed .wm-tmp-* name.  Size it from the
#     field, not from a previous exposure (per-HDU cost varies by 5x across
#     fields); 3 h cut runs that the bound now finishes in ~20 minutes.
#
# Usage:
#   bash benchmarks/canfar/run_exposures.sh 1729091 1774908 1851368 1851960 1853217 1853221 1853389
#
# Env (defaults suit the staged dirty tree on mlao/cfhtcast):
#   WM_TREE        staged tree to run (default cand-ce16410c5cae-dirty)
#   WM_TREE_ORIGIN origin of that tree, recorded in the run record
#   WM_TIMEOUT     processing budget in seconds (default 10800)
#   WM_WAIT        local ship deadline in seconds (default 14400)
#   WM_CPU/WM_RAM  pod size (defaults 16/128, the shape verified for 40 HDUs)
#   WM_KEEP_RECORD name for the interrupted attempt's record (default run.attempt2.json;
#                  `cp -n` never overwrites it, so later attempts leave it in place)
set -u

if [ "$#" -eq 0 ]; then
    echo "usage: run_exposures.sh OBSID [OBSID ...]" >&2
    exit 2
fi

cd "$(dirname "$0")/../.." || exit 1
ROOT="$(pwd)"
RES="${WM_RESULTS:-/arc/projects/mlao/cfhtcast/weightmask-perf/results}"
: "${WM_TREE:=/arc/projects/mlao/cfhtcast/weightmask-perf/repos/cand-ce16410c5cae-dirty}"
: "${WM_TREE_ORIGIN:=/arc/projects/mlao/cfhtcast/weightmask-perf/repos/cand-ce16410c5cae}"
: "${WM_TIMEOUT:=10800}"
: "${WM_WAIT:=14400}"
: "${WM_CPU:=16}"
: "${WM_RAM:=128}"
: "${WM_KEEP_RECORD:=run.attempt2.json}"

for OBSID in "$@"; do
    case "$OBSID" in
        [0-9]*) ;;
        *) echo "not an observation id: $OBSID" >&2; exit 2 ;;
    esac
done

for OBSID in "$@"; do
    OUT="$RES/ns-$OBSID"
    # Keep the interrupted attempt's record, drop its unrenamed partial mask, and
    # record this attempt's start: the driver overwrites run.json in place.
    CMD="cp -n $OUT/$OBSID.run.json $OUT/$OBSID.$WM_KEEP_RECORD; rm -f $OUT/.$OBSID.mask.fits.fz.wm-tmp-*;"
    CMD="$CMD {py} benchmarks/canfar/process_exposure.py $OBSID --out $OUT --workers $WM_CPU"
    CANFAR_BIN="${CANFAR_BIN:-canfar}" \
        nohup pixi run python benchmarks/canfar/pod_ship.py benchmarks/canfar/pod_run.py \
        --name "wm-ns-$OBSID" --cpu "$WM_CPU" --ram "$WM_RAM" --wait "$WM_WAIT" \
        --env "WM_TREE=$WM_TREE" \
        --env "WM_TREE_ORIGIN=$WM_TREE_ORIGIN" \
        --env "WM_COPY_TREE=1" \
        --env "WM_THREAD_PIN=1" \
        --env "WM_COMMAND=$CMD" \
        --env "WM_RESULTS_TAG=ns-$OBSID" \
        --env "WM_TIMEOUT=$WM_TIMEOUT" \
        > "/tmp/wm-exposure-$OBSID.log" 2>&1 &
    echo "launched $OBSID pid=$!"
done
