#!/bin/bash
# Submit, wait for, and collect weightmask release-gate jobs on CANFAR.
#
# Headless jobs land on isolated resources, so every target can run at once;
# there is no "quiet machine" precondition like the historical Mac campaign.
# The measured code is pinned with MANIFEST_SHA and the runner scripts live on
# a separate branch, exactly as the perf campaign separated RUNNER_REF from
# code.pinned_sha.
#
# Usage:
#   benchmarks/canfar/submit_gate.sh setup                 # clone the runner to the mount (once)
#   benchmarks/canfar/submit_gate.sh run  lint test        # submit, wait, collect
#   benchmarks/canfar/submit_gate.sh all                   # every target
#   benchmarks/canfar/submit_gate.sh submit synthetic      # submit only (no wait)
#   benchmarks/canfar/submit_gate.sh wait   <id>...        # poll
#   benchmarks/canfar/submit_gate.sh collect <id>...       # lift receipts + evidence into test_outputs/canfar/
#
# Env:
#   MANIFEST_SHA   code under test (default: git rev-parse HEAD)
#   RUNNER_REF     branch holding the runner scripts (default: this branch)
#   PROJECT_MOUNT  default /arc/projects/mlao/cfhtcast
#   IMAGE          default images.canfar.net/astroai/base:latest
#   CPU / MEM      default 8 / 32
set -euo pipefail
RUN_ONE_VERSION="wm-gate-v1"

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
PROJECT_MOUNT="${PROJECT_MOUNT:-/arc/projects/mlao/cfhtcast}"
WORK_ROOT="$PROJECT_MOUNT/weightmask-perf"
BOOTSTRAP="${BOOTSTRAP:-$WORK_ROOT/repos/gate-harness}"
IMAGE="${IMAGE:-images.canfar.net/astroai/base:latest}"
CPU="${CPU:-8}"
MEM="${MEM:-32}"
REPO_URL="${REPO_URL:-https://github.com/sfabbro/weightmask.git}"
MANIFEST_SHA="${MANIFEST_SHA:-$(git -C "$REPO_ROOT" rev-parse HEAD)}"
RUNNER_REF="${RUNNER_REF:-$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)}"
LOCAL_RUNNER_SHA256="$(shasum -a 256 "$HERE/gate_one.sh" | awk '{print $1}')"
ALL_TARGETS="lint test synthetic science-gate"
COLLECT_DIR="${COLLECT_DIR:-$REPO_ROOT/test_outputs/canfar}"

grab_id() { grep -oE 'ID: [a-z0-9]+' | head -n 1 | awk '{print $2}'; }

registry_env_args() {
    local cfgkey val
    for cfgkey in username secret url; do
        val="$(canfar config get "registry.$cfgkey" 2>/dev/null || true)"
        case "$val" in ""|"null") ;; *) printf 'CANFAR_REGISTRY__%s=%s\n' "$(echo "$cfgkey" | tr '[:lower:]' '[:upper:]')" "$val";; esac
    done
}

do_setup() {
    echo "== setup: runner $RUNNER_REF -> $BOOTSTRAP"
    local out id
    if out="$(canfar create headless "$IMAGE" --name wm-gate-bootstrap --cpu 1 --memory 4 -- git clone "$REPO_URL" "$BOOTSTRAP" 2>/dev/null)"; then
        echo "$out"; wait_for "$(echo "$out" | grab_id)"
    else
        echo "bootstrap exists; reusing"
    fi
    out="$(canfar create headless "$IMAGE" --name wm-gate-fetch --cpu 1 --memory 4 -- git -C "$BOOTSTRAP" fetch origin)"
    echo "$out"; wait_for "$(echo "$out" | grab_id)"
    out="$(canfar create headless "$IMAGE" --name wm-gate-pin --cpu 1 --memory 4 -- git -C "$BOOTSTRAP" checkout -B runner "origin/$RUNNER_REF")"
    echo "$out"; wait_for "$(echo "$out" | grab_id)"
}

wait_for() {
    [ -n "${1:-}" ] || { echo "wait_for: empty session id" >&2; return 1; }
    local id="$1" st=""
    for _ in $(seq 1 240); do
        st="$(canfar info "$id" 2>/dev/null | grep -E 'Status' | head -n 1 || true)"
        case "$st" in
            *Completed*) echo "  $id: Completed"; return 0;;
            *Failed*|*Error*) echo "  $id: $st" >&2; return 1;;
        esac
        sleep 30
    done
    echo "TIMEOUT waiting for $id" >&2; return 1
}

submit_job() {
    local target="$1" out
    local env_args=(--env "MANIFEST_SHA=$MANIFEST_SHA" --env "PROJECT_MOUNT=$PROJECT_MOUNT" --env "RUNNER_VERSION=$RUN_ONE_VERSION")
    while IFS= read -r kv; do env_args+=(--env "$kv"); done < <(registry_env_args)
    echo "== submit wm-gate-$target (code=$MANIFEST_SHA) ==" >&2
    out="$(canfar create headless "$IMAGE" --name "wm-gate-$target" --cpu "$CPU" --memory "$MEM" \
        "${env_args[@]}" -- /bin/bash "$BOOTSTRAP/benchmarks/canfar/gate_one.sh" "$target")"
    echo "$out" >&2
    echo "$out" | grab_id
}

collect_one() {
    local id="$1" dest="$2" attempt
    mkdir -p "$dest"
    for attempt in 1 2 3 4 5; do
        # The log endpoint returns a transient HTTP 500 often enough that a
        # single fetch cannot be trusted.
        if ! canfar logs "$id" > "$dest/$id.log" 2>/dev/null; then sleep 20; continue; fi
        if grep -q "WM-RECEIPT-END" "$dest/$id.log"; then break; fi
        sleep 20
    done
    sed -n '/WM-RECEIPT-BEGIN/,/WM-RECEIPT-END/p' "$dest/$id.log" | sed '1d;$d' > "$dest/$id.receipt.env"
    python3 - "$dest/$id.log" "$dest" <<'PY'
import base64, os, re, sys
log_path, dest = sys.argv[1:3]
text = open(log_path, errors="replace").read()
pattern = re.compile(r"WM-EVIDENCE-BEGIN (\S+) (\d+)\n(.*?)\nWM-EVIDENCE-END", re.S)
written = []
for rel, size, payload in pattern.findall(text):
    payload = re.sub(r"\s+", "", payload)
    try:
        blob = base64.b64decode(payload)
    except Exception:
        continue
    out = os.path.join(dest, "evidence", rel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "wb").write(blob)
    written.append(f"{rel} ({len(blob)}B)")
print(f"  {os.path.basename(dest)}: {len(written)} evidence file(s)")
for item in written:
    print(f"    {item}")
PY
    echo "  receipt: $dest/$id.receipt.env"
    grep -E "^(target|exit_code|wall_s|commit_sha|fixture_sha256)" "$dest/$id.receipt.env" | sed 's/^/    /' || true
}

collect_ids() {
    local id target
    for id in "$@"; do
        target="$(canfar info "$id" 2>/dev/null | grep -oE 'wm-gate-[a-z-]+' | head -n 1 | sed 's/wm-gate-//' || true)"
        collect_one "$id" "$COLLECT_DIR/${target:-$id}-${MANIFEST_SHA:0:8}"
    done
}

run_targets() {
    local ids=() t id
    for t in "$@"; do ids+=("$(submit_job "$t")"); done
    echo "submitted: ${ids[*]}"
    for id in "${ids[@]}"; do wait_for "$id" || true; done
    collect_ids "${ids[@]}"
}

case "${1:-}" in
    setup) do_setup;;
    submit) shift; for t in "${@:-$ALL_TARGETS}"; do submit_job "$t"; done;;
    wait) shift; for id in "$@"; do wait_for "$id" || true; done;;
    collect) shift; collect_ids "$@";;
    run) shift; do_setup; run_targets "$@";;
    all) do_setup; run_targets $ALL_TARGETS;;
    *) echo "usage: $0 {setup|submit <T>|wait <id>|collect <id>|run <T>|all}" >&2
       echo "targets: $ALL_TARGETS"; echo "runner sha256: $LOCAL_RUNNER_SHA256"; exit 1;;
esac
