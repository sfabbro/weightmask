#!/bin/bash
# Submit the MegaCam CANFAR speed campaign from Mac.
#
# Outer-layer rule (probes 2026-09-09): headless CMD+args are split on spaces
# server-side and `$` triggers group expansion. Every create below therefore
# uses bare tokens only: `git ...` for setup, `/bin/bash <script> <exp> <tag>`
# for measurements (torchsky precedent). No shell metachars, no `$` in args.
#
# Usage:
#   benchmarks/canfar/submit_all.sh setup            # bootstrap clone+checkout (once)
#   benchmarks/canfar/submit_all.sh e0               # control first (waits)
#   benchmarks/canfar/submit_all.sh main             # E1-E7 (waits)
#   benchmarks/canfar/submit_all.sh e8               # winners last (waits)
#   benchmarks/canfar/submit_all.sh all              # setup+e0+main (E8 manual; needs winners)
#   benchmarks/canfar/submit_all.sh submit E4        # one group, no wait
# Append --no-wait to skip polling.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
MANIFEST="$HERE/../canfar_experiments/manifest.json"
PROJECT_MOUNT="${PROJECT_MOUNT:-/arc/projects/mlao/cfhtcast}"
WORK_ROOT="$PROJECT_MOUNT/weightmask-perf"
BOOTSTRAP="$WORK_ROOT/repos/bootstrap"
WAIT=1
case " $* " in *" --no-wait "*) WAIT=0;; esac

manifest_py() { python3 -c "import json,sys; m=json.load(open('$MANIFEST')); $1"; }
SHA="$(manifest_py "print(m['code']['pinned_sha'])")"
IMAGE="$(manifest_py "print(m['defaults']['image'])")"
CPU="$(manifest_py "print(m['defaults'].get('cpu',8))")"
MEM="$(manifest_py "print(m['defaults'].get('memory_gb',32))")"
REPO="https://github.com/astroai/weightmask.git"

grab_id() { grep -oE 'ID: [a-z0-9]+' | head -n 1 | awk '{print $2}'; }

wait_for() { # session_id
    [ -n "${1:-}" ] || { echo "wait_for: empty session id"; return 1; }
    local id="$1" st=""
    for _ in $(seq 1 120); do
        st="$(canfar info "$id" 2>/dev/null | grep -E 'Status' | head -n 1 || true)"
        echo "  $id: $st"
        case "$st" in *Completed*) return 0;; *Failed*|*Error*) echo "JOB FAILED: $id"; return 1;; esac
        sleep 60
    done
    echo "TIMEOUT waiting for $id"; return 1
}

do_setup() {
    echo "== setup: clone to $BOOTSTRAP @ $SHA =="
    local out id
    out="$(canfar create headless "$IMAGE" --name wm-bootstrap --cpu 1 --memory 4 -- git clone "$REPO" "$BOOTSTRAP")"
    echo "$out"
    id="$(echo "$out" | grab_id)"
    # clone fails if the dir already exists; fall through to checkout either way
    wait_for "$id" || true
    out="$(canfar create headless "$IMAGE" --name wm-fetch --cpu 1 --memory 4 -- git -C "$BOOTSTRAP" fetch origin)"
    echo "$out"
    wait_for "$(echo "$out" | grab_id)"
    out="$(canfar create headless "$IMAGE" --name wm-checkout --cpu 1 --memory 4 -- git -C "$BOOTSTRAP" checkout "$SHA")"
    echo "$out"
    wait_for "$(echo "$out" | grab_id)"
    out="$(canfar create headless "$IMAGE" --name wm-verify --cpu 1 --memory 4 -- git -C "$BOOTSTRAP" rev-parse HEAD)"
    echo "$out"
    wait_for "$(echo "$out" | grab_id)"
    echo "setup done"
}

registry_env_args() {
    # Harbor pull credentials for astroai images: `canfar config set
    # registry.username/secret` once, then every headless create carries them
    # (canfar-lab convention). Skipped silently when unconfigured.
    local cfgkey val
    for cfgkey in username secret url; do
        val="$(canfar config get "registry.$cfgkey" 2>/dev/null || true)"
        case "$val" in ""|"null") ;; *) printf 'CANFAR_REGISTRY__%s=%s\n' "$(echo "$cfgkey" | tr '[:lower:]' '[:upper:]')" "$val";; esac
    done
}

submit_job() { # exp_id job_tag -> echoes session id
    local exp="$1" tag="$2" name out
    name="wm-$(echo "$exp-$tag" | tr 'A-Z' 'a-z')"
    local keep=0
    case "$exp" in E0|E8) keep=1;; esac
    local checkout
    checkout="$(manifest_py "print(next(x for x in m['groups'] if x['exp_id']=='$exp').get('checkout') or '')")"
    case "$checkout" in ''|'TBD'*|'None') checkout="$SHA";; esac
    local env_args=(--env "MANIFEST_SHA=$SHA" --env "CHECKOUT_REF=$checkout" --env "KEEP_ALL=$keep" --env "PROJECT_MOUNT=$PROJECT_MOUNT")
    while IFS= read -r kv; do env_args+=(--env "$kv"); done < <(python3 -c "
import json
m = json.load(open('$MANIFEST'))
g = next(x for x in m['groups'] if x['exp_id'] == '$exp')
for k, v in g.get('env', {}).items(): print(k + '=' + str(v))
")
    while IFS= read -r kv; do env_args+=(--env "$kv"); done < <(registry_env_args)
    echo "== submit $name (keep=$keep) ==" >&2
    out="$(canfar create headless "$IMAGE" --name "$name" --cpu "$CPU" --memory "$MEM" \
        "${env_args[@]}" -- /bin/bash "$BOOTSTRAP/benchmarks/canfar/run_one.sh" "$exp" "$tag")"
    echo "$out" >&2
    echo "$out" | grab_id
}

submit_group() { # exp_id: one headless job per manifest job entry, echoes last id
    local exp="$1" tags last=""
    tags="$(manifest_py "print(' '.join(j['tag'] for j in next(x for x in m['groups'] if x['exp_id']=='$exp')['jobs']))")"
    for t in $tags; do last="$(submit_job "$exp" "$t")"; done
    echo "$last"
}

phase_e0() { local id; id="$(submit_group E0)"; wait_for "$id"; }
phase_main() { local e id; for e in E1 E2 E3 E4 E5 E6 E7; do id="$(submit_group "$e")"; wait_for "$id" || return 1; done; }
phase_e8() { local id; id="$(submit_group E8)"; wait_for "$id"; }

case "${1:-}" in
    setup) do_setup;;
    e0) phase_e0;;
    main) phase_main;;
    e8) phase_e8;;
    submit) submit_group "${2:?group id}";;
    all) do_setup; phase_e0; phase_main; echo "E8 left manual (needs winners).";;
    *) echo "usage: $0 {setup|e0|main|e8|submit <G>|all} [--no-wait]"; exit 1;;
esac
