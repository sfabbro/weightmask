#!/bin/bash
# CANFAR headless measurement wrapper for one experiment job.
#
# Invoked INSIDE the container with a plain argv (no shell metachars at the
# outer `canfar create` layer):
#   /bin/bash <bootstrap>/benchmarks/canfar/run_one.sh <EXP_ID> <JOB_TAG>
#
# Proven outer-layer constraints (2026-09-09 probes, recorded in manifest):
# - outer CMD+args are split on spaces server-side and `$` expands. This
#   script is a FILE on the project mount, so full shell works here.
# - `canfar logs` captures stdout only; resource numbers come from harness
#   rusage (exact: the backend is threaded) via the parallel-pass sidecar.
#
# Env (all but MANIFEST_SHA have defaults):
#   MANIFEST_SHA   repo commit to run (required; from manifest.json code.pinned_sha)
#   PROJECT_MOUNT  default /arc/projects/mlao/cfhtcast
#   REPO_URL       default https://github.com/astroai/weightmask.git
#   PIXI_CACHE_DIR default $WORK_ROOT/pixi-cache (shared across jobs)
#   KEEP_ALL       1 keeps weights+masks on the mount (E0 control, E8 winners);
#                  0 keeps metrics + harness JSON/report only (default)
set -euo pipefail

EXP_ID="${1:?usage: run_one.sh <EXP_ID> <JOB_TAG>}"
JOB_TAG="${2:?usage: run_one.sh <EXP_ID> <JOB_TAG>}"
BOOTSTRAP="$(cd "$(dirname "$0")/../.." && pwd)"
MANIFEST_SHA="${MANIFEST_SHA:?set MANIFEST_SHA to the manifest pinned_sha}"
CHECKOUT_REF="${CHECKOUT_REF:-$MANIFEST_SHA}"
REPO_URL="${REPO_URL:-https://github.com/astroai/weightmask.git}"
PROJECT_MOUNT="${PROJECT_MOUNT:-/arc/projects/mlao/cfhtcast}"
WORK_ROOT="$PROJECT_MOUNT/weightmask-perf"
PIXI_CACHE_DIR="${PIXI_CACHE_DIR:-$WORK_ROOT/pixi-cache}"
KEEP_ALL="${KEEP_ALL:-0}"
RESULTS_DIR="$WORK_ROOT/results/$EXP_ID-$JOB_TAG"

if [ -d /scratch ]; then SCR_BASE=/scratch; else SCR_BASE=/tmp; fi
JOB_DIR="$SCR_BASE/wm-$EXP_ID-$JOB_TAG-$$"
REPO_DIR="$JOB_DIR/repo"
DATA_NOTE=""

echo "== run_one $EXP_ID/$JOB_TAG =="
echo "bootstrap=$BOOTSTRAP sha=$MANIFEST_SHA checkout=$CHECKOUT_REF scratch=$SCR_BASE keep_all=$KEEP_ALL"
mkdir -p "$JOB_DIR" "$RESULTS_DIR"
# Job-local pixi cache (forced: images may export a read-only shared one;
# a shared cache also deadlocks across containers via stale locks).
export PIXI_CACHE_DIR="$JOB_DIR/pixi-cache"
mkdir -p "$PIXI_CACHE_DIR"
export PYTHONUNBUFFERED=1
# Ignore image defaults; explicit manifest settings are restored below.
unset OMP_NUM_THREADS MKL_NUM_THREADS
MANIFEST="$BOOTSTRAP/benchmarks/canfar_experiments/manifest.json"
GROUP_JSON="$JOB_DIR/group.json"
# Prefer a modern system pixi (needs >= 0.44 for `[workspace]`); otherwise
# install a job-local one. The skaha fallback path predates workspace.
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

python3 - "$MANIFEST" "$EXP_ID" "$JOB_TAG" "$GROUP_JSON" <<'EOF'
import json, sys
manifest_path, exp_id, job_tag, out = sys.argv[1:5]
m = json.load(open(manifest_path))
g = next(x for x in m["groups"] if x["exp_id"] == exp_id)
job = next(j for j in g["jobs"] if j["tag"] == job_tag)
if not g["safe_ids"] or "TBD" in g["safe_ids"]:
    raise ValueError("experiment exposure IDs are not resolved; fill winner inputs before submitting")
if len(g.get("flat_safe_ids", [])) > 1:
    raise ValueError("multiple flats require an explicit per-exposure mapping; this wrapper accepts one flat")
json.dump({"group": g, "job": job, "image": m["defaults"]["image"]}, open(out, "w"), indent=2)
print("group:", exp_id, "exposures:", g["safe_ids"], "workers:", job["workers"])
EOF

while IFS= read -r kv; do export "$kv"; done < <(python3 - "$GROUP_JSON" <<'EOF'
import json, sys
record = json.load(open(sys.argv[1]))
env = dict(record["group"].get("env", {}))
env.update(record["job"].get("env", {}))
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    if key in env:
        print(key + "=" + str(env[key]))
EOF
)

git clone "$REPO_URL" "$REPO_DIR"
git -C "$REPO_DIR" checkout "$CHECKOUT_REF"
git -C "$REPO_DIR" rev-parse HEAD
cd "$REPO_DIR"
if [ ! -x "$PIXI" ]; then
    export PIXI_HOME="$JOB_DIR/pixi-home"
    curl -fsSL https://pixi.sh/install.sh -o "$JOB_DIR/pixi-install.sh"
    bash "$JOB_DIR/pixi-install.sh"
fi
if [ ! -x "$PIXI" ]; then
    PIXI="$(command -v pixi || echo /opt/conda/bin/pixi)"
fi
"$PIXI" --version
"$PIXI" install

python3 - "$GROUP_JSON" "$JOB_DIR" <<'EOF'
import json, os, subprocess, sys, urllib.request
group_path, job_dir = sys.argv[1:3]
grp = json.load(open(group_path))["group"]
data_dir = os.path.join(job_dir, "data")
os.makedirs(data_dir, exist_ok=True)

def fetch(url, out):
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print("exists:", out)
        return
    print("fetch:", url)
    partial = out + ".part"
    try:
        urllib.request.urlretrieve(url, partial)
        os.replace(partial, out)
    finally:
        if os.path.exists(partial):
            os.remove(partial)

PUB = "https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHT"
recs = []
for pid, safe in zip(grp["publisherIDs"], grp["safe_ids"]):
    if safe == "TBD":
        continue
    dest = os.path.join(data_dir, safe + ".fits.fz")
    fetch(f"{PUB}/{safe}.fits.fz", dest)
    recs.append({"publisherID": pid, "safe_id": safe, "pending_file": dest})

flats = {}
for fsafe in grp.get("flat_safe_ids", []):
    dest = os.path.join(data_dir, "flat_" + fsafe + ".fits.fz")
    got = False
    for variant in (fsafe, fsafe.rsplit(".", 1)[0] + ".01", fsafe.rsplit(".", 1)[0] + ".00"):
        try:
            fetch(f"{PUB}/{variant}.fits.fz", dest)
            got = True
            flats[fsafe] = dest
            break
        except Exception as e:
            print("flat variant miss:", variant, str(e)[:120])
    if not got:
        raise RuntimeError("no usable flat downloaded for " + fsafe)
json.dump({"records": recs, "flats": flats}, open(os.path.join(job_dir, "staged.json"), "w"), indent=2)
EOF

"$PIXI" run python - "$JOB_DIR" <<'EOF'
import fitsio, json, os, sys
job_dir = sys.argv[1]
staged = json.load(open(os.path.join(job_dir, "staged.json")))
out = []
for r in staged["records"]:
    fp = r.pop("pending_file")
    repo_rel = os.path.join("benchmark_data", "megacam", "perf", os.path.basename(fp))
    repo_abs = os.path.join(os.getcwd(), repo_rel)
    os.makedirs(os.path.dirname(repo_abs), exist_ok=True)
    if os.path.abspath(fp) != os.path.abspath(repo_abs):
        if os.path.exists(repo_abs):
            os.remove(repo_abs)
        try:
            os.link(fp, repo_abs)
        except OSError:
            import shutil
            shutil.copy(fp, repo_abs)
    h = fitsio.FITS(repo_abs, "r")
    exptime = None
    for i in (1, 0):
        try:
            exptime = float(h[i].read_header().get("EXPTIME"))
            break
        except Exception:
            continue
    h.close()
    out.append({"publisherID": r["publisherID"], "safe_id": r["safe_id"],
                "file": repo_rel, "exptime": exptime,
                "size_bytes": os.path.getsize(repo_abs)})
flat0 = next(iter(staged["flats"].values()), None)
flat_rel = None
if flat0:
    flat_rel = os.path.join("benchmark_data", "megacam", "perf", os.path.basename(flat0))
    flat_abs = os.path.join(os.getcwd(), flat_rel)
    os.makedirs(os.path.dirname(flat_abs), exist_ok=True)
    if os.path.abspath(flat0) != os.path.abspath(flat_abs):
        if os.path.exists(flat_abs):
            os.remove(flat_abs)
        try:
            os.link(flat0, flat_abs)
        except OSError:
            import shutil
            shutil.copy(flat0, flat_abs)
os.makedirs("test_outputs/perf", exist_ok=True)
json.dump(out, open("test_outputs/perf/exposures.json", "w"), indent=2)
json.dump({"flat_rel": flat_rel}, open(os.path.join(job_dir, "flat.json"), "w"))
print("staged exposures:", [r["safe_id"] for r in out], "flat:", flat_rel)
EOF

ARGS=(--tag "$EXP_ID-$JOB_TAG" --masks --resume)
ARGS+=(--workers "$(python3 -c "import json;print(json.load(open('$GROUP_JSON'))['job']['workers'])")")
ARGS+=(--exposure-ids "$(python3 -c "import json;print(','.join(json.load(open('$GROUP_JSON'))['group']['safe_ids']))")")
while IFS= read -r kv; do ARGS+=(--config-set "$kv"); done < <(python3 -c "
import json
g = json.load(open('$GROUP_JSON'))['group']
def flat(prefix, d):
    for k, v in d.items():
        if isinstance(v, dict): yield from flat(prefix + k + '.', v)
        else: yield prefix + k + '=' + json.dumps(v)
for kv in flat('', g.get('config_set', {})): print(kv)
")
FLAT_REL="$(python3 -c "import json;print(json.load(open('$JOB_DIR/flat.json'))['flat_rel'] or '')")"
if [ -n "$FLAT_REL" ]; then ARGS+=(--flat "$FLAT_REL"); fi

echo "harness args: ${ARGS[*]}"
JOB_T0=$SECONDS
"$PIXI" run python benchmarks/perf_megacam.py "${ARGS[@]}"
"$PIXI" run python - "$JOB_DIR" "$RESULTS_DIR" "$EXP_ID" "$JOB_TAG" "$KEEP_ALL" <<'EOF'
import glob, json, os, sys
job_dir, res_dir, exp_id, job_tag, keep_all = sys.argv[1:6]
keep_all = keep_all == "1"
os.makedirs(res_dir, exist_ok=True)

tag = f"{exp_id}-{job_tag}"
repo = os.getcwd()
perf_dir = os.path.join(repo, "test_outputs", "perf")
# Prefer the parallel-pass sidecar (exact w8 wall/peak/CPU); fall back to the
# sequential report for workers=1 jobs. rusage is exact: backend is threaded.
workers = json.load(open(os.path.join(job_dir, "group.json")))["job"]["workers"]
sidecar_path = os.path.join(perf_dir, f"megacam_pass_{tag}.w{workers}.json")
if int(workers) != 1 and os.path.exists(sidecar_path):
    side = json.load(open(sidecar_path))
    wall = float(side["extra_pass_wall_s"])
    max_rss = float(side["peak_rss_kb"])
    cpu_s = float(side["cpu_s"])
    mpix = float(side["mpix_total"])
    nhdus = int(side["nhdus"])
    per_stage = {k: {"total_s": float(v.get("total_s", 0.0)),
                     "mean_per_hdu_s": float(v.get("mean_per_hdu_s", 0.0)),
                     "share": float(v.get("share", 0.0))} for k, v in side.get("stages", {}).items()}
    source = os.path.basename(sidecar_path)
    cpu_basis = side.get("cpu_basis", "legacy-cumulative")
else:
    perf = json.load(open(os.path.join(perf_dir, f"megacam_perf_{tag}.json")))
    stages = perf.get("stages", {})
    nhdus = sum(e.get("nhdus_processed", 0) for e in perf.get("per_exposure", []))
    mpix = float(perf.get("mpix_total", 0.0))
    wall = float(sum(float(e.get("wall_s", 0.0)) for e in perf.get("per_exposure", [])))
    max_rss = float(max([float(e.get("peak_rss_kb", 0.0)) for e in perf.get("per_exposure", [])] or [0.0]))
    cpu_s = float(sum(float(e.get("cpu_s", 0.0)) for e in perf.get("per_exposure", [])))
    per_stage = {k: {"total_s": float(v.get("total_s", 0.0)),
                     "mean_per_hdu_s": float(v.get("mean_per_hdu_s", 0.0)),
                     "share": float(v.get("share", 0.0))} for k, v in stages.items()}
    source = f"megacam_perf_{tag}.json"
    cpu_basis = perf.get("cpu_basis", "legacy-cumulative")
cpu_percent = (cpu_s / wall * 100.0) if wall else None

masks = sorted(glob.glob(os.path.join(repo, "test_outputs", "perf", "*.mask.fits")))
ctrl_dir = os.path.join(os.path.dirname(res_dir), "E0-w8")
diff = {"mode": None}
checksums = {}
try:
    import hashlib
    import numpy as np
    from astropy.io import fits as _fits

    def read_mask(path):
        with _fits.open(path) as h:
            arrays = [(i, np.array(hdu.data)) for i, hdu in enumerate(h)
                      if getattr(hdu, "data", None) is not None and hdu.data.ndim == 2]
        if not arrays:
            raise ValueError(f"no mask image HDUs in {path}")
        return arrays

    safe_ids = json.load(open(os.path.join(job_dir, "group.json")))["group"]["safe_ids"]
    same = {os.path.basename(p).split(".")[0]: p for p in masks
            if os.path.basename(p).split(".")[0] in safe_ids and f".w{workers}." in os.path.basename(p)}
    ctrl = sorted(glob.glob(os.path.join(ctrl_dir, "*.mask.fits")))
    ctrl_same = {os.path.basename(p).split(".")[0]: p for p in ctrl if ".w8." in os.path.basename(p)}
    fractions, comparisons = {}, {}
    for exposure, path in same.items():
        arrays = read_mask(path)
        fractions[exposure] = sum(np.count_nonzero(a) for _, a in arrays) / sum(a.size for _, a in arrays)
        digest = hashlib.sha256()
        for index, array in arrays:
            digest.update(str((index, array.shape, array.dtype.str)).encode())
            digest.update(np.ascontiguousarray(array).tobytes())
        checksums[exposure] = digest.hexdigest()
        if exposure not in ctrl_same:
            continue
        control = read_mask(ctrl_same[exposure])
        if [(i, a.shape) for i, a in arrays] != [(i, a.shape) for i, a in control]:
            raise ValueError(f"mask HDU layout mismatch for {exposure}")
        counts = dict(kept=0, lost=0, gained=0, union=0, identical=True)
        for (_, a1), (_, a0) in zip(arrays, control):
            b1, b0 = a1.astype(bool), a0.astype(bool)
            for key, pixels in (("kept", b1 & b0), ("lost", b0 & ~b1),
                                ("gained", b1 & ~b0), ("union", b1 | b0)):
                counts[key] += int(np.count_nonzero(pixels))
            counts["identical"] &= bool(np.array_equal(a1, a0))
        comparisons[exposure] = counts
    if comparisons:
        diff = {"mode": "pixel", "per_exposure": comparisons,
                **{key: sum(c[key] for c in comparisons.values()) for key in ("kept", "lost", "gained", "union")},
                "unmatched": sorted(set(same) - set(comparisons)),
                "identical": len(comparisons) == len(same) and all(c["identical"] for c in comparisons.values())}
    else:
        diff = {"mode": "fraction-self", "exp_fracs": fractions}
except Exception as e:
    diff = {"mode": "error", "note": str(e)[:200]}
    checksums = {}

metrics = {"exp_id": exp_id, "job_tag": job_tag, "wall_s": wall,
           "max_rss_kb": max_rss, "cpu_percent": cpu_percent,
           "workers": int(workers),
           "parallel_efficiency": (cpu_percent / (100.0 * int(workers))) if cpu_percent is not None else None,
           "mpix": mpix, "mpix_s": (mpix / wall) if wall else None,
           "nhdus": nhdus, "per_stage": per_stage, "mask_diff": diff,
           "mask_checksums": checksums, "mask_checksum_scope": "all-image-hdus-v1",
           "source": source, "cpu_basis": cpu_basis,
           "input_key": {"safe_ids": sorted(json.load(open(os.path.join(job_dir, "group.json")))["group"]["safe_ids"]),
                         "flat_safe_ids": sorted(json.load(open(os.path.join(job_dir, "group.json")))["group"].get("flat_safe_ids", [])),
                         "workers": int(workers), "nhdus": nhdus},
           "harness_report": f"megacam_perf_{tag}.json"}
json.dump(metrics, open(os.path.join(res_dir, "metrics.json"), "w"), indent=2)

import shutil
for name in (f"megacam_perf_{tag}.json", f"megacam_perf_{tag}.md", f"cprofile_hdu_{tag}.txt",
             f"checkpoint_{tag}.json", source):
    src = os.path.join(repo, "test_outputs", "perf", name)
    if os.path.exists(src):
        shutil.copy(src, res_dir)
if keep_all:
    for p in masks + sorted(glob.glob(os.path.join(repo, "test_outputs", "perf", "*.weight.fits"))):
        shutil.copy(p, res_dir)
print("metrics:", json.dumps({k: metrics[k] for k in ("wall_s", "max_rss_kb", "cpu_percent", "parallel_efficiency", "mpix_s")}))
print("mask_diff:", json.dumps(diff)[:300])
print("results:", res_dir, "keep_all:", keep_all)
EOF

echo "SUMMARY $EXP_ID/$JOB_TAG wall=$(python3 -c "import json;print(json.load(open('$RESULTS_DIR/metrics.json'))['wall_s'])")s rss=$(python3 -c "import json;print(json.load(open('$RESULTS_DIR/metrics.json'))['max_rss_kb'])")KB job_wall_s=$((SECONDS - JOB_T0))"
echo "results in $RESULTS_DIR"
