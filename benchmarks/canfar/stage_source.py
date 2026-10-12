#!/usr/bin/env python3
"""Stage this repository at a given ref onto CANFAR shared storage.

The September campaign bootstrapped its jobs with `git clone` inside a pod.
That path is dead: the cluster's NodeLocal DNS cache intermittently fails, so
github.com and codeload.github.com do not resolve (measured 2026-10-12). The
project mount is still shared and writable by pods, so instead of cloning we
push a source tarball *through the session API* and extract it there:

    Mac: git archive REF | gzip -9   (641 KB at 313cb43)
      -> base64 -> in-pod extractor script
      -> pod_ship.py (form-encoded request body; no request-line limit)
      -> pod writes and unpacks it under the shared mount

Once staged, any number of later pods can use the tree directly, so the cost is
paid once per commit rather than per job.

The destination is keyed by commit SHA so several candidates can coexist and a
job can state exactly which tree it measured.

Usage:
    python3 benchmarks/canfar/stage_source.py [--ref REF] [--dest DIR] [--keep]

Env:
    WM_STAGE_DEST   default /arc/projects/mlao/cfhtcast/weightmask-perf/repos
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_DEST = os.environ.get("WM_STAGE_DEST", "/arc/projects/mlao/cfhtcast/weightmask-perf/repos")

PAYLOAD_TEMPLATE = '''\
"""Extract a staged weightmask source tree onto the shared CANFAR mount."""
import base64
import hashlib
import io
import os
import sys
import tarfile

DATA = "{payload}"
DEST = "{dest}"
EXPECT = "{expect}"
print("STAGE_START", os.environ.get("HOSTNAME", ""))
raw = base64.b64decode(DATA)
digest = hashlib.sha256(raw).hexdigest()
print("tarball_bytes", len(raw), "sha256", digest)
if digest != EXPECT:
    print("STAGE_FAILED: tarball sha256 mismatch, expected", EXPECT)
    sys.exit(3)
print("STAGE_SHA256_OK")
manifest = os.path.join(DEST, "SOURCE.tar.gz")
os.makedirs(DEST, exist_ok=True)
with open(manifest, "wb") as handle:
    handle.write(raw)
try:
    archive = tarfile.open(fileobj=io.BytesIO(raw))
    archive.extractall(DEST, filter="data")
except TypeError:
    archive = tarfile.open(fileobj=io.BytesIO(raw))
    archive.extractall(DEST)
archive.close()
names = []
for dirpath, _dirs, files in os.walk(DEST):
    for name in files:
        names.append(os.path.relpath(os.path.join(dirpath, name), DEST))
print("file_count", len(names))
for probe in ("weightmask/streaks.py", "weightmask.yml", "benchmarks/science_gate.py",
              "pyproject.toml", "tests/test_science_gate.py"):
    print("present" if probe in names else "MISSING", probe)
print("total_bytes", sum(os.path.getsize(os.path.join(DEST, n)) for n in names))
print("STAGE_END")
'''


def git_archive(ref: str) -> bytes:
    proc = subprocess.run(["git", "archive", "--format=tar", ref], cwd=ROOT, capture_output=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"git archive {ref} failed: {proc.stderr.decode(errors='replace')}")
    packed = subprocess.run(["gzip", "-9"], input=proc.stdout, capture_output=True, check=False)
    if packed.returncode != 0:
        raise SystemExit("gzip failed")
    return packed.stdout


def resolve_sha(ref: str) -> str:
    proc = subprocess.run(["git", "rev-parse", f"{ref}^{{commit}}"], cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"cannot resolve {ref}: {proc.stderr.strip()}")
    return proc.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--dest", default=DEFAULT_DEST)
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--ram", type=int, default=4)
    parser.add_argument("--keep", action="store_true", help="keep the staging session")
    options = parser.parse_args()

    sha = resolve_sha(options.ref)
    tarball = git_archive(options.ref)
    target = os.path.join(options.dest, f"cand-{sha[:12]}")
    payload_source = PAYLOAD_TEMPLATE.format(
        payload=base64.b64encode(tarball).decode("ascii"),
        dest=target,
        expect=hashlib.sha256(tarball).hexdigest(),
    )
    print(f"[stage] ref {options.ref} -> {sha}")
    print(f"[stage] tarball {len(tarball)} B -> payload script {len(payload_source)} B -> dest {target}")

    with tempfile.TemporaryDirectory(prefix="wm_stage_") as workdir:
        script = Path(workdir) / "stage_payload.py"
        script.write_text(payload_source)
        command = [
            sys.executable,
            str(HERE / "pod_ship.py"),
            str(script),
            "--name",
            f"wm-stage-{sha[:8]}",
            "--cpu",
            str(options.cpu),
            "--ram",
            str(options.ram),
            "--wait",
            "600",
        ]
        if options.keep:
            command.append("--keep")
        return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
