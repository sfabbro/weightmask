"""In-pod payload: prepare a staged tree, then run whatever command you name.

Shipped by ``benchmarks/canfar/pod_ship.py``; see that module for how a script
reaches a pod at all. This file is the only thing that runs inside the
container, and it is parameterised through environment variables so one runner
covers the science gate, the pytest suite, and the benchmarks:

    WM_TREE           staged tree to work in (required)
    WM_COMMAND        shell command to run in that tree; the literal placeholder
                      `{py}` is replaced by the interpreter this runner selected
                      (required). A literal `$` cannot be used: the platform runs
                      env values through a regex expansion and rejects them with
                      `400 Illegal group reference`.
    WM_STAGE_INPUTS   1 to place the fixture exposures under
                      benchmark_data/megacam/perf before running
    WM_RESULTS_TAG    copy test_outputs/harness outputs to
                      <results root>/<tag>
    WM_TIMEOUT        seconds allowed for WM_COMMAND (default 7200)
    WM_SKIP_ENV       1 to trust the image interpreter without probing
    WM_COPY_TREE      1 to copy the staged tree into scratch and work there, so
                      concurrent jobs cannot write over each other's outputs
    WM_PIXI_TRIES     attempts at `pixi install` (default 12). Each attempt is
                      cheap when DNS fails early and the package cache on
                      scratch keeps whatever it did fetch, so attempts compound.

Two constraints shape this file:

* The cluster's NodeLocal DNS cache fails intermittently (measured 2026-10-12),
  so every network step is retried and downloads are resumable.
* A pod is a fresh container with an empty scratch directory, so the environment
  is built per job. The image interpreter is preferred when it already
  satisfies the imports, which avoids a multi-minute pixi solve entirely.

Fixture inputs are looked for on the project mount before any download is
attempted: the September campaign left data under ``cfhtcast-data``, and
reading it from the mount is both faster and immune to the DNS fault.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

MOUNT_ROOTS = (
    "/arc/projects/mlao/cfhtcast/cfhtcast-data",
    "/arc/projects/mlao/cfhtcast",
)
RESULTS_ROOT = "/arc/projects/mlao/cfhtcast/weightmask-perf/results"
CADC_PUB = "https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHT"

# local name in benchmark_data/megacam/perf -> CADC published file name
NEEDED = {
    "1013719p.fits.fz": "1013719p.fits.fz",
    "1013720p.fits.fz": "1013720p.fits.fz",
    "1013721p.fits.fz": "1013721p.fits.fz",
    "flat_08Bm01_r.fits.fz": "08Bm01.flat.r.36.02.fits.fz",
}
# Packages the gate's subprocesses import; `sep` and `astroscrappy` are the ones
# an astro base image is most likely to be missing.
REQUIRED_IMPORTS = (
    "numpy",
    "scipy",
    "astropy",
    "fitsio",
    "sep",
    "skimage",
    "astroscrappy",
    "yaml",
)


def sh(command, cwd=None, timeout=None, env=None):
    """Run a shell command, echo it, return (rc, combined output)."""
    print(f"+ {command}", flush=True)
    started = time.time()
    try:
        proc = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        print(f"  !! timed out after {timeout}s", flush=True)
        return 124, "timeout"
    text = (proc.stdout or "") + (proc.stderr or "")
    if text:
        print(text, end="" if text.endswith("\n") else "\n", flush=True)
    print(f"  rc={proc.returncode} in {time.time() - started:.1f}s", flush=True)
    return proc.returncode, text


def check_imports(python):
    """Return the required imports the interpreter cannot satisfy."""
    code = (
        "import importlib,sys\n"
        "missing=[]\n"
        "for name in sys.argv[1:]:\n"
        "    try: importlib.import_module(name)\n"
        "    except Exception: missing.append(name)\n"
        "print('MISSING:' + ','.join(missing))\n"
    )
    proc = subprocess.run(python.split() + ["-c", code] + list(REQUIRED_IMPORTS), capture_output=True, text=True)
    for line in (proc.stdout or "").splitlines():
        if line.startswith("MISSING:"):
            return [item for item in line[len("MISSING:") :].split(",") if item]
    return list(REQUIRED_IMPORTS)


def find_on_mount(name):
    for root in MOUNT_ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, files in os.walk(root):
            if name in files and os.path.getsize(os.path.join(dirpath, name)) > 1024 * 1024:
                return os.path.join(dirpath, name)
            dirnames[:] = [d for d in dirnames if d not in (".git", "node_modules")]
            if dirpath.count(os.sep) - root.count(os.sep) > 3:
                dirnames[:] = []
    return None


def fetch(url, dest, minutes=None):
    """Download with curl, retrying for as long as the budget allows.

    A single curl invocation exhausts its retries inside one DNS outage window
    and then gives up, which is how an intermittent fault reads as a permanent
    block. Downloads are resumed (`-C -`) across attempts, so the cost of a
    failed attempt is just wall-clock time.
    """
    if minutes is None:
        minutes = float(os.environ.get("WM_DOWNLOAD_MINUTES", "45"))
    tmp = dest + ".part"
    deadline = time.time() + minutes * 60
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        rc, _ = sh(
            f"curl -sS -L --retry 5 --retry-delay 5 --retry-all-errors -C - "
            f"--connect-timeout 30 --max-time 1800 -o {tmp} {url}",
            timeout=1900,
        )
        if rc == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 1024 * 1024:
            os.replace(tmp, dest)
            return True
        print(
            f"  download attempt {attempt} failed after {(deadline - time.time()) / 60:.1f} "
            "min of budget left; resuming",
            flush=True,
        )
        time.sleep(15)
    return False


def stage_inputs(tree, data_dir):
    os.makedirs(data_dir, exist_ok=True)
    print("--- staging fixture inputs ---", flush=True)
    missing = []
    for local_name, published in NEEDED.items():
        target = os.path.join(data_dir, local_name)
        if os.path.exists(target) and os.path.getsize(target) > 1024 * 1024:
            print(f"already present: {target} ({os.path.getsize(target)} B)", flush=True)
            continue
        mounted = find_on_mount(published) or find_on_mount(local_name)
        if mounted:
            print(f"found on mount: {mounted}", flush=True)
            try:
                os.link(mounted, target)
            except OSError:
                shutil.copyfile(mounted, target)
            print(f"linked -> {target} ({os.path.getsize(target)} B)", flush=True)
            continue
        print(f"not on mount, downloading {published}", flush=True)
        if fetch(f"{CADC_PUB}/{published}", target):
            print(f"downloaded -> {target} ({os.path.getsize(target)} B)", flush=True)
        else:
            missing.append(published)
    return missing


def find_existing_env(tree):
    """Return the interpreter of a pixi env already built in the staged tree.

    The staged tree lives at a fixed absolute path on the shared mount, so an
    env built there once by any job is usable, unchanged, by every later job.
    That matters because building it needs PyPI (`setuptools` for the editable
    `weightmask` dependency) and the cluster's DNS fails about half the time.

    `weightmask` is pure Python, so the env interpreter plus
    PYTHONPATH=<tree> imports it exactly as the editable install would.
    """
    envs = os.path.join(tree, ".pixi", "envs")
    if not os.path.isdir(envs):
        return None
    for name in sorted(os.listdir(envs)):
        python = os.path.join(envs, name, "bin", "python")
        if os.path.exists(python):
            return python
    return None


def prepare_environment(tree, env, skip_probe):
    """Pick an interpreter for the job, building the pixi env only if needed."""
    if skip_probe:
        print("WM_SKIP_ENV=1: using the image interpreter unprobed", flush=True)
        return "python3"

    # With WM_COPY_TREE the working tree is a copy without .pixi, but the
    # original sits at a fixed path on the shared mount, so its interpreter is
    # still the right one; only PYTHONPATH has to point at the working copy.
    existing = find_existing_env(tree) or find_existing_env(os.environ.get("WM_TREE_ORIGIN", ""))
    if existing:
        print(f"found pixi env interpreter already built in the tree: {existing}", flush=True)
        missing = check_imports(existing)
        print(f"  its missing imports: {missing or 'none'}", flush=True)
        if not missing:
            print("reusing the existing env; no install needed", flush=True)
            env["PYTHONPATH"] = tree
            return existing
        print("  the existing env is incomplete; falling through", flush=True)

    system_python = shutil.which("python3") or "python3"
    missing = check_imports(system_python)
    print(f"system python3 {system_python} missing: {missing or 'none'}", flush=True)
    if not missing:
        print("image interpreter satisfies every required import; skipping pixi", flush=True)
        env["PYTHONPATH"] = tree
        return "python3"
    print("--- building the pixi environment ---", flush=True)
    pixi = shutil.which("pixi")
    if not pixi:
        print("FATAL: no pixi and the image interpreter is incomplete", flush=True)
        return None
    sh(f"{pixi} --version")
    tries = int(os.environ.get("WM_PIXI_TRIES", "12"))
    shared_cache = os.environ.get("WM_SHARED_CACHE", "/arc/projects/mlao/cfhtcast/weightmask-perf/pixi-cache")
    if os.path.isdir(shared_cache):
        # The September campaign left every conda-forge package it had fetched
        # here, which is the difference between a solve that needs one flaky
        # PyPI call and one that needs hundreds of flaky conda calls.
        env["PIXI_CACHE_DIR"] = shared_cache
        print(f"using the surviving shared package cache: {shared_cache}", flush=True)
    rc = 1
    for attempt in range(1, tries + 1):
        rc, _ = sh(f"{pixi} install", cwd=tree, timeout=3600, env=env)
        if rc == 0:
            print(f"pixi install succeeded on attempt {attempt}", flush=True)
            break
        # A partially failed install still leaves a usable conda env when only
        # the PyPI step failed, so adopt it rather than starting over.
        if attempt == 1:
            partial = find_existing_env(tree)
            if partial:
                missing = check_imports(partial)
                if not missing:
                    print(
                        f"pixi install failed but {partial} already satisfies every "
                        "import; using it with PYTHONPATH instead",
                        flush=True,
                    )
                    env["PYTHONPATH"] = tree
                    return partial
        # The failure is transient name resolution, so waiting longer than the
        # outage is what actually helps; pixi caches whatever it did fetch.
        delay = min(5 * attempt, 60)
        print(
            f"pixi install failed (attempt {attempt}/{tries}); cluster DNS is flaky, retrying in {delay}s",
            flush=True,
        )
        time.sleep(delay)
    if rc != 0:
        print("FATAL: pixi install failed", flush=True)
        return None
    return f"{pixi} run python"


def main():
    tree = os.environ.get("WM_TREE") or (sys.argv[1] if len(sys.argv) > 1 else "")
    command = os.environ.get("WM_COMMAND", "")
    if not tree or not command:
        print("FATAL: WM_TREE and WM_COMMAND are both required", flush=True)
        return 2

    print("===== POD RUN =====", flush=True)
    print("host", os.environ.get("HOSTNAME"), "tree", tree, flush=True)
    print("command", command, flush=True)
    print("cpus", os.cpu_count(), flush=True)
    if not os.path.isdir(tree):
        print(f"FATAL: tree does not exist: {tree}", flush=True)
        return 2

    # Thread hygiene: the campaign found image-exported OMP/MKL defaults perturb
    # parallel-efficiency measurements, so they are dropped rather than trusted.
    env = dict(os.environ)
    env.pop("OMP_NUM_THREADS", None)
    env.pop("MKL_NUM_THREADS", None)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("PYTHONPATH", None)
    scratch = "/scratch" if os.path.isdir("/scratch") else "/tmp"
    env["PIXI_CACHE_DIR"] = os.path.join(scratch, "wm-pixi-cache")
    env["WM_TREE"] = tree
    env["TMPDIR"] = scratch

    rc, _ = sh(f"git -C {tree} rev-parse HEAD", env=env)
    if rc != 0:
        sh(f"ls -la {tree}", env=env)

    if os.environ.get("WM_COPY_TREE") == "1":
        # The shared mount is common to every job; a job that writes into the
        # staged tree would race any other job pointed at the same revision.
        # The pixi env is excluded: it is large, and because the staged tree
        # sits at a fixed absolute path its interpreter stays valid from
        # scratch.
        scratch_tree = os.path.join(scratch, "tree")
        print(f"--- copying tree to {scratch_tree} (excluding .pixi) ---", flush=True)
        sh(f"rm -rf {scratch_tree}", env=env)
        rc, _ = sh(f"mkdir -p {scratch_tree}", env=env)
        rc, _ = sh(
            f"tar --exclude=./.pixi -C {tree} -cf - . | tar -C {scratch_tree} -xf -",
            timeout=1800,
            env=env,
        )
        if rc != 0 or not os.path.isdir(os.path.join(scratch_tree, "weightmask")):
            print("FATAL: could not copy the staged tree", flush=True)
            return 3
        env["WM_TREE_ORIGIN"] = tree
        tree = scratch_tree
        env["WM_TREE"] = tree

    runner = prepare_environment(tree, env, os.environ.get("WM_SKIP_ENV") == "1")
    if runner is None:
        print("===== POD RUN FAILED (environment) =====", flush=True)
        return 3
    env["WM_PY"] = runner
    print(f"interpreter for this job: {runner}", flush=True)
    if "{py}" in command:
        command = command.replace("{py}", runner)
        print(f"expanded command: {command}", flush=True)

    if os.environ.get("WM_STAGE_INPUTS") == "1":
        missing = stage_inputs(tree, os.path.join(tree, "benchmark_data", "megacam", "perf"))
        if missing:
            print(f"FATAL: could not stage {missing}", flush=True)
            return 4
        sh(f"ls -la {tree}/benchmark_data/megacam/perf", env=env)

    print("--- running WM_COMMAND ---", flush=True)
    rc, _ = sh(command, cwd=tree, timeout=int(os.environ.get("WM_TIMEOUT", "7200")), env=env)

    tag = os.environ.get("WM_RESULTS_TAG")
    if tag:
        destination = os.path.join(RESULTS_ROOT, tag)
        try:
            os.makedirs(destination, exist_ok=True)
            copied = []
            for name in (
                "release-evidence.json",
                "science-gate.md",
                "science-gate.json",
                "trail_truth.score.json",
                "streak_inject.tsv",
                "cr_faint.tsv",
                "perf",
                "synthetic_v2",
                "benchmarks",
            ):
                src = os.path.join(tree, "test_outputs", "harness", name)
                if not os.path.exists(src):
                    src = os.path.join(tree, "test_outputs", name)
                if not os.path.exists(src):
                    continue
                target = os.path.join(destination, name)
                if os.path.isdir(src):
                    shutil.rmtree(target, ignore_errors=True)
                    shutil.copytree(src, target)
                else:
                    shutil.copy(src, target)
                copied.append(name)
            print(f"copied {copied} -> {destination}", flush=True)
        except OSError as exc:
            print(f"could not copy results: {exc}", flush=True)

    print(f"===== POD RUN DONE rc={rc} =====", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
