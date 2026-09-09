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
mkdir -p "$JOB_DIR" "$RESULTS_DIR" "$PIXI_CACHE_DIR"
export PIXI_CACHE_DIR

MANIFEST="$BOOTSTRAP/benchmarks/canfar_experiments/manifest.json"
GROUP_JSON="$JOB_DIR/group.json"
# In-image pixi is too old for `[workspace]` (E0 probe 2026-09-09); install a
# job-local pixi (fast, ~10 MB) and use it. Falls back to system pixi.
PIXI="$JOB_DIR/pixi-home/bin/pixi"

python3 - "$MANIFEST" "$EXP_ID" "$JOB_TAG" "$GROUP_JSON" <<'EOF'
import json, sys
manifest_path, exp_id, job_tag, out = sys.argv[1:5]
m = json.load(open(manifest_path))
g = next(x for x in m["groups"] if x["exp_id"] == exp_id)
job = next(j for j in g["jobs"] if j["tag"] == job_tag)
json.dump({"group": g, "job": job, "image": m["defaults"]["image"]}, open(out, "w"), indent=2)
print("group:", exp_id, "exposures:", g["safe_ids"], "workers:", job["workers"])
EOF

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
    urllib.request.urlretrieve(url, out)

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
    if not got and os.path.exists(dest):
        flats[fsafe] = dest
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
cpu_percent = (cpu_s / wall * 100.0) if wall else None

masks = sorted(glob.glob(os.path.join(repo, "test_outputs", "perf", "*.mask.fits")))
ctrl_dir = os.path.join(os.path.dirname(res_dir), "E0-w8")
diff = {"mode": None}
try:
    import numpy as np
    from astropy.io import fits as _fits

    def frac(path):
        with _fits.open(path) as h:
            d = None
            for hdu in h:
                if getattr(hdu, "data", None) is not None and hdu.data.size > 1:
                    d = np.asarray(hdu.data)
                    break
        return (None if d is None else float(np.count_nonzero(d)) / d.size,
                None if d is None else d)

    same = [p for p in masks if os.path.basename(p).startswith(tuple(
        json.load(open(os.path.join(job_dir, "group.json")))["group"]["safe_ids"]))]
    ctrl = sorted(glob.glob(os.path.join(ctrl_dir, "*.mask.fits")))
    ctrl_same = [p for p in ctrl if os.path.basename(p).split(".")[0] in
                 json.load(open(os.path.join(job_dir, "group.json")))["group"]["safe_ids"]]
    if same and ctrl_same and os.path.abspath(same[0]) != os.path.abspath(ctrl_same[0]):
        f1, d1 = frac(same[0])
        f0, d0 = frac(ctrl_same[0])
        if d1 is not None and d0 is not None and d1.shape == d0.shape:
            b1, b0 = d1.astype(bool), d0.astype(bool)
            union = int(np.count_nonzero(b1 | b0))
            diff = {"mode": "pixel", "control": os.path.basename(ctrl_same[0]),
                    "kept": int(np.count_nonzero(b1 & b0)), "lost": int(np.count_nonzero(b0 & ~b1)),
                    "gained": int(np.count_nonzero(b1 & ~b0)), "union": union,
                    "identical": bool(np.array_equal(b1, b0))}
        else:
            diff = {"mode": "fraction", "control_frac": f0, "exp_frac": f1}
    else:
        diff = {"mode": "fraction-self", "exp_fracs": {os.path.basename(p): frac(p)[0] for p in same}}
except Exception as e:
    diff = {"mode": "error", "note": str(e)[:200]}

import hashlib
checksums = {}
for p in masks:
    try:
        h = hashlib.sha1()
        with _fits.open(p) as fh:
            for hdu in fh:
                if getattr(hdu, "data", None) is not None and hdu.data.size > 1:
                    h.update(np.ascontiguousarray(hdu.data).tobytes())
                    break
        checksums[os.path.basename(p)] = h.hexdigest()[:16]
    except Exception as e:
        checksums[os.path.basename(p)] = "error:" + str(e)[:60]

metrics = {"exp_id": exp_id, "job_tag": job_tag, "wall_s": wall,
           "max_rss_kb": max_rss, "cpu_percent": cpu_percent,
           "parallel_efficiency": (cpu_percent / 800.0) if cpu_percent else None,
           "mpix": mpix, "mpix_s": (mpix / wall) if wall else None,
           "nhdus": nhdus, "per_stage": per_stage, "mask_diff": diff,
           "mask_checksums": checksums, "source": source,
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
