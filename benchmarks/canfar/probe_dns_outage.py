"""In-pod payload: sample name resolution over time.

The cluster's NodeLocal DNS cache failures look like a permanent egress policy
change if you only ever try once. Sampling over several minutes separates the
two: a policy would fail every sample, an intermittent fault will succeed in
some windows. That distinction decides whether a job should retry harder or
give up, so this reports a success rate per host rather than a single verdict.

Env:
    WM_SAMPLES   samples per host (default 30)
    WM_INTERVAL  seconds between samples (default 10)
"""

from __future__ import annotations

import os
import socket
import subprocess
import time

HOSTS = (
    "www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca",
    "cadc-ccda.hia-iha.nrc-cnrc.gc.ca",
    "ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca",
    "cadc.nrc.ca",
    "ws-uv.canfar.net",
    "pypi.org",
    "files.pythonhosted.org",
    "conda.anaconda.org",
    "github.com",
    "google.com",
)


def resolve(host):
    try:
        return socket.gethostbyname(host)
    except Exception as exc:  # noqa: BLE001
        return f"FAIL {type(exc).__name__}"


def main():
    samples = int(os.environ.get("WM_SAMPLES", "30"))
    interval = float(os.environ.get("WM_INTERVAL", "10"))
    print("===== DNS SAMPLER =====", flush=True)
    print(f"host={os.environ.get('HOSTNAME')} samples={samples} interval={interval}s", flush=True)
    print("resolv.conf:", flush=True)
    try:
        print(open("/etc/resolv.conf").read(), flush=True)
    except OSError as exc:
        print(f"  unreadable: {exc}", flush=True)

    results = {host: [] for host in HOSTS}
    for index in range(1, samples + 1):
        line = []
        for host in HOSTS:
            value = resolve(host)
            ok = not value.startswith("FAIL")
            results[host].append(ok)
            line.append("OK" if ok else "--")
        print(f"sample {index:03d}/{samples}  " + " ".join(line), flush=True)
        if index < samples:
            time.sleep(interval)

    print("--- summary ---", flush=True)
    for host in HOSTS:
        good = sum(results[host])
        first_ok = next((i + 1 for i, ok in enumerate(results[host]) if ok), None)
        print(
            f"  {host:48s} {good:3d}/{samples} ok  first_success_sample={first_ok}",
            flush=True,
        )

    print("--- CADC data reachability during a window (if any resolved) ---", flush=True)
    proc = subprocess.run(
        [
            "curl",
            "-sS",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            "--max-time",
            "20",
            "-r",
            "0-1023",
            "https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHT/1013719p.fits.fz",
        ],
        capture_output=True,
        text=True,
    )
    print(f"  range GET -> {proc.stdout.strip() or proc.stderr.strip()[:120]}", flush=True)

    print("--- candidate data locations on the mount ---", flush=True)
    for path in (
        "/arc/projects/mlao/cfhtcast/cfhtcast-data",
        "/arc/projects/mlao/cfhtcast/weightmask-perf",
        "/arc/home/sfabbro",
    ):
        print(f"  ls {path}:", flush=True)
        listing = subprocess.run(["bash", "-lc", f"ls -la {path} 2>&1 | head -25"], capture_output=True, text=True)
        print(listing.stdout, flush=True)
    print(
        subprocess.run(
            [
                "bash",
                "-lc",
                "find /arc/projects/mlao/cfhtcast -maxdepth 4 -name '*.fits.fz' 2>/dev/null | head -20",
            ],
            capture_output=True,
            text=True,
        ).stdout,
        flush=True,
    )
    print("===== DNS SAMPLER DONE =====", flush=True)


if __name__ == "__main__":
    main()
