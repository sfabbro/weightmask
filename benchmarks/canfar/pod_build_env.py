"""In-pod payload: build the pixi environment for a staged tree.

The gate needs fitsio, scikit-image and astroscrappy, which none of the public
astro images ship, so the environment has to be built in the pod. The cluster's
NodeLocal DNS cache fails intermittently (measured 2026-10-12), which makes
`pixi install` a race: each attempt dies on name resolution after ~26 s.

Two facts make that survivable:

* pixi caches every package it did manage to fetch, so repeated attempts make
  cumulative progress.
* The September campaign left a populated pixi cache on the shared project
  mount. If it still holds the pinned packages, a solve can complete with far
  less network traffic than a cold one.

This payload reports what the shared cache holds and then tries to install,
first against the shared cache and then job-locally.

Env:
    WM_TREE         staged tree (required)
    WM_TRIES        attempts per cache mode (default 10)
    WM_SHARED_CACHE shared cache path (default the September one)
"""

from __future__ import annotations

import os
import subprocess
import time

SHARED_CACHE = os.environ.get("WM_SHARED_CACHE", "/arc/projects/mlao/cfhtcast/weightmask-perf/pixi-cache")


def sh(command, cwd=None, timeout=None, env=None):
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


def main():
    tree = os.environ.get("WM_TREE", "")
    if not tree or not os.path.isdir(tree):
        print(f"FATAL: WM_TREE is not a directory: {tree!r}", flush=True)
        return 2
    tries = int(os.environ.get("WM_TRIES", "10"))
    scratch = "/scratch" if os.path.isdir("/scratch") else "/tmp"

    print("===== POD BUILD ENV =====", flush=True)
    print("host", os.environ.get("HOSTNAME"), "tree", tree, flush=True)
    print("--- shared cache survey ---", flush=True)
    sh(f"du -sh {SHARED_CACHE} 2>&1")
    sh(f"ls {SHARED_CACHE} 2>&1 | head -20")
    sh(f"find {SHARED_CACHE} -name '*.conda' 2>/dev/null | wc -l")
    sh(f"find {SHARED_CACHE} -name 'pixi-*.tar.zst' 2>/dev/null | wc -l")
    sh(f"ls {SHARED_CACHE}/*/ 2>&1 | head -20")

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("PYTHONPATH", None)
    sh(f"cat {tree}/pixi.toml", env=env)

    for mode, cache in (("shared-cache", SHARED_CACHE), ("job-local-cache", os.path.join(scratch, "pixi-cache"))):
        print(f"===== pixi install with {mode}: {cache} =====", flush=True)
        if mode == "shared-cache" and not os.path.isdir(cache):
            print("  shared cache does not exist; skipping", flush=True)
            continue
        env["PIXI_CACHE_DIR"] = cache
        rc = 1
        for attempt in range(1, tries + 1):
            rc, _ = sh("pixi install", cwd=tree, timeout=3600, env=env)
            if rc == 0:
                print(f"  SUCCESS on attempt {attempt} ({mode})", flush=True)
                break
            delay = min(5 * attempt, 45)
            print(f"  attempt {attempt}/{tries} failed; retrying in {delay}s", flush=True)
            time.sleep(delay)
        if rc == 0:
            sh(
                "pixi run python -c \"import fitsio, skimage, astroscrappy, sep; print('IMPORTS_OK')\"",
                cwd=tree,
                timeout=900,
                env=env,
            )
            break

    print(f"===== POD BUILD ENV DONE rc={rc} =====", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
