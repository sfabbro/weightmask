"""In-pod survey: DNS, egress, mounts, tooling, and whether the September
bootstrap clone survived on the shared project mount.

Runs via benchmarks/canfar/pod_exec.sh, which hex-encodes this file into a
single space-free argv token. Kept dependency-free (stdlib only) because a
fresh pod has no guarantee beyond python3.
"""

import os
import socket
import subprocess
import sys

PROBE_START = "PODPROBE_START"
PROBE_END = "PODPROBE_END"


def sh(command, timeout=120):
    try:
        proc = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
        out = (proc.stdout or "").rstrip()
        err = (proc.stderr or "").rstrip()
        if proc.returncode != 0:
            return f"rc={proc.returncode} out={out!r} err={err!r}"[:600]
        return out[:2000]
    except Exception as exc:  # noqa: BLE001 - probe must never die
        return f"EXC {type(exc).__name__}: {exc}"[:300]


def resolve(host):
    try:
        return socket.gethostbyname(host)
    except Exception as exc:  # noqa: BLE001
        return f"FAIL {type(exc).__name__}: {exc}"


def main():
    print(PROBE_START, flush=True)
    print("python", sys.version.split()[0], flush=True)
    print("hostname", socket.gethostname(), flush=True)
    print("cwd", os.getcwd(), flush=True)
    print("uid", os.getuid(), "user", os.environ.get("USER"), flush=True)

    print("--- identity ---", flush=True)
    print(sh("id"), flush=True)
    print(sh("cat /etc/os-release | head -3"), flush=True)

    print("--- mount layout ---", flush=True)
    print("ls /arc ->", sh("ls /arc 2>&1"), flush=True)
    print("ls /arc/home/sfabbro ->", sh("ls /arc/home/sfabbro 2>&1 | head -25"), flush=True)
    print(
        "ls /arc/projects/mlao/cfhtcast ->",
        sh("ls /arc/projects/mlao/cfhtcast 2>&1 | head -25"),
        flush=True,
    )
    print(
        "weightmask-perf ->",
        sh("ls -la /arc/projects/mlao/cfhtcast/weightmask-perf 2>&1 | head -25"),
        flush=True,
    )
    print(
        "repos ->",
        sh("ls -la /arc/projects/mlao/cfhtcast/weightmask-perf/repos 2>&1 | head -25"),
        flush=True,
    )
    print(
        "september bootstrap clone? ->",
        sh("ls -la /arc/projects/mlao/cfhtcast/weightmask-perf/repos/bootstrap2 2>&1 | head -25"),
        flush=True,
    )
    print(
        "  its HEAD ->",
        sh("git -C /arc/projects/mlao/cfhtcast/weightmask-perf/repos/bootstrap2 log --oneline -3 2>&1"),
        flush=True,
    )
    print(
        "  its remotes ->",
        sh("git -C /arc/projects/mlao/cfhtcast/weightmask-perf/repos/bootstrap2 remote -v 2>&1"),
        flush=True,
    )
    print(
        "  its refs ->",
        sh(
            "git -C /arc/projects/mlao/cfhtcast/weightmask-perf/repos/bootstrap2 for-each-ref --format='%(refname) %(objectname:short)' 2>&1 | head -20"
        ),
        flush=True,
    )
    print(
        "  can it see our candidate commit? ->",
        sh(
            "git -C /arc/projects/mlao/cfhtcast/weightmask-perf/repos/bootstrap2 cat-file -t 313cb436e7f9ef7adea3ca9ad75a9bf4e163de6b 2>&1"
        ),
        flush=True,
    )

    print("--- writable mounts ---", flush=True)
    marker = "/arc/home/sfabbro/.wm-pod-probe-write"
    print("write /arc/home/sfabbro ->", sh(f"touch {marker} && ls -la {marker} && echo WROTE_HOME"), flush=True)
    pmk = "/arc/projects/mlao/cfhtcast/weightmask-perf/.wm-pod-probe-write"
    print("write project mount ->", sh(f"touch {pmk} && ls -la {pmk} && echo WROTE_PROJECT"), flush=True)
    print("scratch ->", sh("ls -ld /scratch 2>&1"), flush=True)

    print("--- DNS ---", flush=True)
    for host in (
        "github.com",
        "codeload.github.com",
        "raw.githubusercontent.com",
        "pypi.org",
        "files.pythonhosted.org",
        "conda.anaconda.org",
        "ws-uv.canfar.net",
        "www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca",
        "ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca",
        "cadc.nrc.ca",
        "google.com",
    ):
        print(f"  {host:52s} {resolve(host)}", flush=True)

    print("--- proxy env ---", flush=True)
    print(sh("env | grep -iE 'proxy|no_proxy' | sort"), flush=True)
    print(sh("cat /etc/resolv.conf 2>&1 | head -5"), flush=True)

    print("--- egress (8s timeouts) ---", flush=True)
    urls = (
        "https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHT/1013719p.fits.fz",
        "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/vault/",
        "https://github.com/",
        "https://codeload.github.com/astroai/weightmask/tar.gz/refs/heads/main",
        "https://pypi.org/simple/",
        "https://conda.anaconda.org/conda-forge/noarch/repodata.json",
        "https://ws-uv.canfar.net/skaha/availability",
    )
    for url in urls:
        code = sh(f"curl -sS -o /dev/null -w '%{{http_code}}' --max-time 8 -r 0-1023 '{url}' 2>&1")
        print(f"  {url}\n      -> {code}", flush=True)

    print("--- tooling ---", flush=True)
    for tool in ("git", "curl", "wget", "python3", "pixi", "conda", "java", "tar", "xz", "zstd"):
        print(f"  {tool:8s} {sh('command -v ' + tool + ' 2>&1')}", flush=True)
    print("pixi --version ->", sh("pixi --version 2>&1"), flush=True)
    print(
        "host certs ->",
        sh("ls -la /arc/home/sfabbro/.ssl/ 2>&1"),
        flush=True,
    )
    print(
        "cert subject ->",
        sh("openssl x509 -in /arc/home/sfabbro/.ssl/cadcproxy.pem -noout -subject -enddate 2>&1"),
        flush=True,
    )

    print("--- in-pod certificate fetch attempt ---", flush=True)
    print(
        sh(
            "curl -sS -o /dev/null -w 'vault-with-cert=%{http_code}' --max-time 10 "
            "--cert /arc/home/sfabbro/.ssl/cadcproxy.pem --key /arc/home/sfabbro/.ssl/cadcproxy.pem "
            "'https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/vault/sfabbro' 2>&1"
        ),
        flush=True,
    )

    print(PROBE_END, flush=True)


if __name__ == "__main__":
    main()
