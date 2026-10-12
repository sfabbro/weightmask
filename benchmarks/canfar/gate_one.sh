#!/bin/bash
RUN_ONE_VERSION="wm-gate-v1"
export RUN_ONE_VERSION
# CANFAR headless runner for one weightmask release gate.
#
# Invoked INSIDE the container with a plain argv (no shell metachars at the
# outer `canfar create` layer):
#   /bin/bash <bootstrap>/benchmarks/canfar/gate_one.sh <TARGET>
#
# Outer-layer constraints (probes 2026-09-09, re-verified 2026-10-12): the
# create-layer CMD+args are split on spaces server-side and `$` expands, so
# every `canfar create` uses bare tokens only and reaches a FILE on the
# project mount. Full shell works here, inside the script.
#
# Headless jobs run on isolated resources, so unlike the historical Mac-side
# campaign there is no "no concurrent benchmark load" condition to respect:
# gate jobs for different targets may run at the same time.
#
# Env:
#   MANIFEST_SHA    repo commit to measure (required; the code under test)
#   PROJECT_MOUNT   default /arc/projects/mlao/cfhtcast
#   REPO_URL        default https://github.com/sfabbro/weightmask.git
#   KEEP_DATA       1 keeps the staged fusion frames on the mount (default 1)
#   EVIDENCE_MAX_BYTES  per-file cap for base64 retrieval (default 131072)
set -euo pipefail

TARGET="${1:?usage: gate_one.sh <lint|test|synthetic|megacam|acs|science-gate>}"
MANIFEST_SHA="${MANIFEST_SHA:?set MANIFEST_SHA to the commit under test}"
REPO_URL="${REPO_URL:-https://github.com/sfabbro/weightmask.git}"
PROJECT_MOUNT="${PROJECT_MOUNT:-/arc/projects/mlao/cfhtcast}"
EVIDENCE_MAX_BYTES="${EVIDENCE_MAX_BYTES:-131072}"
WORK_ROOT="$PROJECT_MOUNT/weightmask-perf"
DATA_DIR="$WORK_ROOT/data/megacam/perf"
ARCHIVE="$WORK_ROOT/gates/$TARGET-${MANIFEST_SHA:0:8}"

case "$TARGET" in
    lint) TASK="lint"; NEEDS_DATA=0 ;;
    test) TASK="test"; NEEDS_DATA=0 ;;
    synthetic) TASK="benchmark-synthetic"; NEEDS_DATA=0 ;;
    megacam) TASK="benchmark-megacam"; NEEDS_DATA=1 ;;
    # ACS fixtures are not redistributable and have no published CADC pattern
    # wired here, so the suite reports missing_data rather than a false pass.
    acs) TASK="benchmark-acs"; NEEDS_DATA=0 ;;
    science-gate) TASK="science-gate"; NEEDS_DATA=1 ;;
    *) echo "unknown target: $TARGET" >&2; exit 2 ;;
esac

if [ -d /scratch ]; then SCR_BASE=/scratch; else SCR_BASE=/tmp; fi
JOB_DIR="$SCR_BASE/wm-gate-$TARGET-$$"
REPO_DIR="$JOB_DIR/repo"
mkdir -p "$JOB_DIR" "$ARCHIVE"
# Job-local pixi cache. A shared cache deadlocks across containers via stale
# locks (2026-09-09), and images may export a read-only one.
export PIXI_CACHE_DIR="$JOB_DIR/pixi-cache"
mkdir -p "$PIXI_CACHE_DIR"
export PYTHONUNBUFFERED=1
# Never inherit image defaults for thread pinning; the manifest-owned settings
# are what a measurement is supposed to reflect.
unset OMP_NUM_THREADS MKL_NUM_THREADS

echo "== gate_one $TARGET =="
echo "code_sha=$MANIFEST_SHA scratch=$SCR_BASE mount=$PROJECT_MOUNT archive=$ARCHIVE"
echo "nproc=$(nproc) mem_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo 2>/dev/null || echo '?')"

sha256_of() { # portable across coreutils and BSD userlands
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

git clone --quiet "$REPO_URL" "$REPO_DIR"
git -C "$REPO_DIR" checkout --quiet "$MANIFEST_SHA"
CHECKOUT_SHA="$(git -C "$REPO_DIR" rev-parse HEAD)"
if [ "$CHECKOUT_SHA" != "$MANIFEST_SHA" ]; then
    echo "checkout SHA mismatch: ref=$MANIFEST_SHA actual=$CHECKOUT_SHA" >&2
    exit 1
fi
echo "checkout_sha=$CHECKOUT_SHA"
cd "$REPO_DIR"

# Stage the MegaCam perf frames the way the historical campaign did:
# published CADC products, so the CANFAR copy is verifiable against the local
# fixture set by SHA-256 rather than by trusting the transfer.
stage_data() {
    local pub="https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHT"
    local ids="1013719p 1013720p 1013721p 1013942p 1013943p 1013944p 1013948p 1013949p 1013950p 1013951p flat_08Bm01_r"
    mkdir -p "$DATA_DIR"
    local fid dest
    for fid in $ids; do
        dest="$DATA_DIR/$fid.fits.fz"
        if [ ! -s "$dest" ]; then
            echo "fetch: $fid"
            curl -fsSL "$pub/$fid.fits.fz" -o "$dest.part"
            mv "$dest.part" "$dest"
        fi
    done
    # Link, do not copy: the mount and /scratch are different filesystems, so a
    # hard link is not available and a copy would be 3.8 GB per job.
    mkdir -p "benchmark_data/megacam/perf"
    for fid in $ids; do
        ln -sf "$DATA_DIR/$fid.fits.fz" "benchmark_data/megacam/perf/$fid.fits.fz"
    done
}

if [ "$NEEDS_DATA" = "1" ]; then
    stage_data
fi

pick_pixi() {
    local v
    if command -v pixi >/dev/null 2>&1 && v="$(pixi --version 2>/dev/null)" \
        && [ "$(printf '%s\n' "$v" | grep -oE '[0-9]+\.[0-9]+' | head -n 1 | awk -F. '{print $1 * 1000 + $2}')" -ge 44 ]; then
        command -v pixi
    else
        echo "$JOB_DIR/pixi-home/bin/pixi"
    fi
}
PIXI="$(pick_pixi)"
if [ ! -x "$PIXI" ]; then
    export PIXI_HOME="$JOB_DIR/pixi-home"
    curl -fsSL https://pixi.sh/install.sh -o "$JOB_DIR/pixi-install.sh"
    bash "$JOB_DIR/pixi-install.sh"
fi
if [ ! -x "$PIXI" ]; then
    PIXI="$(command -v pixi || echo /opt/conda/bin/pixi)"
fi
echo "pixi=$PIXI version=$("$PIXI" --version)"
"$PIXI" install

T0=$SECONDS
STATUS=0
"$PIXI" run "$TASK" || STATUS=$?
WALL=$((SECONDS - T0))
echo "== target $TARGET exit=$STATUS wall=${WALL}s =="

# Archive the text evidence the release machinery reads. Binary products stay
# on scratch: they are large and the mount is a shared allocation.
mkdir -p "$ARCHIVE"
if [ -d test_outputs ]; then
    while IFS= read -r rel; do
        mkdir -p "$ARCHIVE/$(dirname "$rel")"
        cp "$rel" "$ARCHIVE/$rel"
    done < <(find test_outputs -type f \( -name '*.json' -o -name '*.md' -o -name '*.tsv' -o -name '*.txt' \) \
        -not -path '*/*.npz*' -size -"$EVIDENCE_MAX_BYTES"c)
fi
{
    echo "target=$TARGET"
    echo "task=$TASK"
    echo "exit_code=$STATUS"
    echo "wall_s=$WALL"
    echo "commit_sha=$CHECKOUT_SHA"
    echo "runner_version=$RUN_ONE_VERSION"
    echo "mount=$ARCHIVE"
    if [ "$NEEDS_DATA" = "1" ]; then
        for f in "$DATA_DIR"/*.fits.fz; do
            printf 'fixture_sha256 %s %s\n' "$(basename "$f")" "$(sha256_of "$f")"
        done
    fi
} > "$ARCHIVE/receipt.env"

echo "WM-RECEIPT-BEGIN"
cat "$ARCHIVE/receipt.env"
echo "WM-RECEIPT-END"

# Small text evidence is lifted back through the job log: the CANFAR log
# endpoint is the only channel from a headless pod to the client, so payloads
# are capped and everything is still kept on the mount.
for f in $(find "$ARCHIVE" -type f \( -name '*.json' -o -name '*.md' -o -name '*.tsv' -o -name '*.txt' \) | sort); do
    rel="${f#"$ARCHIVE"/}"
    bytes=$(wc -c < "$f")
    echo "WM-EVIDENCE-BEGIN $rel $bytes"
    base64 -w0 "$f" || base64 "$f"
    echo
    echo "WM-EVIDENCE-END $rel"
done

echo "gate_one done: $TARGET exit=$STATUS"
exit "$STATUS"
