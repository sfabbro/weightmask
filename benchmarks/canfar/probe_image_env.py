"""In-pod payload: report which packages a candidate image already provides.

The image must satisfy the gate's imports or every job pays for a pixi solve
that the cluster's flaky DNS frequently interrupts. Probing images one pod at a
time is cheap because the pods are isolated, so several can run at once.

Set WM_IMAGE_LABEL so the log says which image was tested.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys

PACKAGES = (
    "numpy",
    "scipy",
    "astropy",
    "fitsio",
    "sep",
    "skimage",
    "astroscrappy",
    "yaml",
    "pandas",
    "matplotlib",
)


def curl(url, timeout=10):
    try:
        proc = subprocess.run(
            ["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", str(timeout), url],
            capture_output=True,
            text=True,
            timeout=timeout + 5,
        )
        return proc.stdout.strip() or proc.stderr.strip()[:60]
    except Exception as exc:  # noqa: BLE001
        return f"EXC {exc}"


def main():
    print("IMAGE_PROBE_START", flush=True)
    print("label:", os.environ.get("WM_IMAGE_LABEL", "<unset>"), flush=True)
    print("python:", sys.version.split()[0], sys.executable, flush=True)
    missing = []
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            print(f"  OK    {name:14s} {getattr(module, '__version__', '?')}", flush=True)
        except Exception as exc:  # noqa: BLE001
            missing.append(name)
            print(f"  MISS  {name:14s} {type(exc).__name__}", flush=True)
    print("MISSING_SET:" + ",".join(missing), flush=True)
    for tool in ("pixi", "conda", "mamba", "pip", "pip3", "python3.13", "git", "curl"):
        print(f"  tool {tool:10s} {shutil.which(tool)}", flush=True)
    print("net pypi.org        ", curl("https://pypi.org/simple/"), flush=True)
    print("net files.pythonhosted", curl("https://files.pythonhosted.org/"), flush=True)
    print("net conda.anaconda  ", curl("https://conda.anaconda.org/conda-forge/noarch/repodata.json"), flush=True)
    print("IMAGE_PROBE_END", flush=True)


if __name__ == "__main__":
    main()
