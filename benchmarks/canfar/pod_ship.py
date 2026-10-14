#!/usr/bin/env python3
"""Ship a payload into a CANFAR headless pod without a pre-existing clone.

Why this exists
---------------
The campaign wrappers assume a git checkout already sits on a pod-reachable
mount. That assumption broke when the cluster's NodeLocal DNS cache started
failing (measured 2026-10-12): github.com and codeload.github.com do not
resolve from inside a pod, while the CANFAR API does, so the source has to
arrive through the session API itself.

That is possible, but two separate limits apply, and they are easy to confuse:

1. The `canfar` CLI sends `cmd`/`args` as **query parameters**. Tomcat refuses a
   request line above its maxHttpHeaderSize, so an 11.9 KB payload comes back
   as `HTTP Status 400 - Bad Request` before the app ever sees it (measured:
   ~6 KB of usable args via the query string).

   The same parameters are accepted in an
   `application/x-www-form-urlencoded` **request body**, which is not subject
   to that limit. A 920 KB body is accepted with HTTP 200.

2. The body is accepted, but the container's argv is not unlimited: a 920 KB
   payload reaches the pod as `exec /usr/local/bin/python3: argument list too
   long`, because a single argv string is capped by MAX_ARG_STRLEN (128 KiB).

So the payload is lzma+base64 compressed and, when it exceeds one token, split
across several space-separated argv tokens that a tiny bootstrap expression
rejoins from `sys.argv`:

    cmd  = python3
    args = -c <bootstrap> <chunk1> <chunk2> ...

The bootstrap itself contains no spaces, which matters because Skaha joins
`cmd` and `args` into one string and splits it again server-side, so any space
becomes an argument boundary. lzma over a tarball of source shrinks ~6x, enough
that the whole repository at a commit fits in six tokens.

The Skaha API is also degraded right now: roughly half of all requests fail with
HTTP 500 (`Error calling /ac to augment subject`, `unexpected failure download
certificate`) because its CADC dependency calls fail intermittently. Creates are
retried until accepted.

Logs are fetched with the `canfar` CLI so this script does not depend on the
exact log endpoint shape.

Usage
-----
    python3 benchmarks/canfar/pod_ship.py SCRIPT.py [options]

    --name NAME        session name (default: wm-ship-<random>)
    --cpu N --ram N    resources (default 1 / 4)
    --image IMAGE      container image
    --wait SECONDS     deadline for the pod to finish (default 900)
    --keep             do not delete the session
    --max-body BYTES   refuse to send a body larger than this (default 4 MiB)
    --dry-run          report the assembled body size, create nothing

Exit status 0 when the pod ran and reported Completed.
"""

from __future__ import annotations

import argparse
import base64
import json
import lzma
import os
import random
import ssl
import string
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = os.environ.get("CANFAR_API", "https://ws-uv.canfar.net/skaha/v1/session")
CERT = os.environ.get("CANFAR_CERT", str(Path.home() / ".canfar" / "cadcproxy.pem"))
# Must be an image the platform lists publicly. `astroai/improc-webterm` is not
# in GET /skaha/v1/image, so using it fails permanently with HTTP 400 "No
# authentication provided for unknown or private image" rather than
# intermittently, which makes it indistinguishable from the /ac flakiness by
# retry count alone. `astroai/base` is listed as a public headless image.
DEFAULT_IMAGE = os.environ.get("CANFAR_IMAGE", "images.canfar.net/astroai/base:latest")

# MAX_ARG_STRLEN is 32 pages (128 KiB) on Linux; stay clear of the edge.
MAX_ARG_CHARS = 120_000
# Rejoins the base64 chunks in sys.argv into the payload and executes it. No
# spaces, so it survives the server-side split as a single token.
BOOTSTRAP = (
    "exec(compile(__import__('lzma').decompress(__import__('base64')"
    ".b64decode(''.join(__import__('sys').argv[1:]))),'payload','exec'))"
)

FAILURE_CLASSES = (
    ("Error calling /ac", "500 /ac augment"),
    ("download certificate", "500 download certificate"),
    ("No authentication provided", "400 private image (not public in the catalog)"),
    ("Missing parameter", "400 missing parameter"),
)


def classify(body: str) -> str:
    for needle, label in FAILURE_CLASSES:
        if needle in body:
            return label
    return "other"


def build_args(script: bytes) -> tuple[str, dict]:
    """Encode a script as one or more space-free argv tokens.

    Returns the args string and a small bookkeeping dict for reporting. The
    payload is lzma+base64; a single token carries it when it fits under
    MAX_ARG_STRLEN, otherwise it is split and the bootstrap rejoins it.
    """
    blob = base64.b64encode(lzma.compress(script, preset=9 | lzma.PRESET_EXTREME)).decode("ascii")
    if len(blob) <= MAX_ARG_CHARS:
        expr = "exec(__import__('lzma').decompress(__import__('base64').b64decode('" + blob + "')))"
        return "-c " + expr, {"mode": "single-token", "tokens": 1, "blob": len(blob)}
    chunks = [blob[i : i + MAX_ARG_CHARS] for i in range(0, len(blob), MAX_ARG_CHARS)]
    info = {"mode": "chunked", "tokens": len(chunks), "blob": len(blob)}
    return "-c " + BOOTSTRAP + " " + " ".join(chunks), info


def http_session() -> tuple[ssl.SSLContext, object]:
    ctx = ssl.create_default_context()
    ctx.load_cert_chain(CERT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx, urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))


def post(opener, params: dict, timeout: int) -> tuple[int, str]:
    body = urllib.parse.urlencode(params, doseq=True).encode()
    request = urllib.request.Request(API, data=body, method="POST")
    request.add_header("Content-Type", "application/x-www-form-urlencoded")
    with opener.open(request, timeout=timeout) as response:
        return response.status, response.read().decode(errors="replace")


def image_is_public(opener, image: str, tries: int = 8) -> bool:
    """Check the public catalog before submitting.

    A private or unknown image returns HTTP 400 on every attempt, which looks
    identical to a retryable failure until the retry budget is exhausted. The
    catalog endpoint is itself flaky, so this also retries, and an unreachable
    catalog is reported as unknown rather than treated as a rejection.
    """
    catalog = API.replace("/session", "/image")
    for _ in range(tries):
        try:
            with opener.open(catalog, timeout=60) as response:
                entries = json.loads(response.read().decode())
        except Exception:  # noqa: BLE001 - flaky endpoint, retry then give up
            time.sleep(4)
            continue
        listed = {entry.get("id") for entry in entries if isinstance(entry, dict)}
        return image in listed
    return False


def canfar(*args: str, timeout: int = 180) -> str:
    binary = os.environ.get("CANFAR_BIN") or "canfar"
    try:
        proc = subprocess.run([binary, *args], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"<canfar {' '.join(args)} failed: {exc}>"
    return (proc.stdout or "") + (proc.stderr or "")


def canfar_retry(*args: str, tries: int = 10, sleep: float = 5.0) -> str:
    """`canfar` subcommands fail with the same transient 500s as create.

    Fetching logs in particular returns `Error calling /ac to augment subject`
    about half the time, which reads as an empty log rather than a retryable
    error.
    """
    output = ""
    for _ in range(tries):
        output = canfar(*args)
        if "Error calling /ac" not in output and "HTTP 500 error" not in output:
            return output
        time.sleep(sleep)
    return output + "\n<pod_ship: giving up after retrying the API>"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("script")
    parser.add_argument("--name")
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--ram", type=int, default=4)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--wait", type=int, default=900)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--tries", type=int, default=30)
    parser.add_argument("--retry-sleep", type=float, default=6.0)
    parser.add_argument("--max-body", type=int, default=4 * 1024 * 1024)
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="environment variable for the pod (repeatable); the shipped script reads these",
    )
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args()

    source = Path(options.script).read_bytes()
    args, encoding = build_args(source)
    params = {
        "name": options.name or "wm-ship-" + "".join(random.choices(string.ascii_lowercase, k=6)),
        "image": options.image,
        "cores": str(options.cpu),
        "ram": str(options.ram),
        "type": "headless",
        "cmd": "python3",
        "args": args,
    }
    if options.env:
        params["env"] = list(options.env)
    body = urllib.parse.urlencode(params, doseq=True).encode()
    print(
        f"[pod_ship] script {len(source)} B -> lzma+b64 {encoding['blob']} B "
        f"({encoding['mode']}, {encoding['tokens']} argv token(s)) -> form body {len(body)} B",
        flush=True,
    )
    if len(args) > 6000:
        print(
            "[pod_ship] note: this payload would NOT fit the query-string path "
            "(~6 KB cap); it relies on the form-body path.",
            flush=True,
        )
    if len(body) > options.max_body:
        print(f"[pod_ship] body exceeds --max-body {options.max_body}; refusing to send", file=sys.stderr)
        return 2
    if options.dry_run:
        print("[pod_ship] dry run: nothing created", flush=True)
        return 0

    ctx, opener = http_session()
    if not image_is_public(opener, options.image):
        print(
            f"[pod_ship] WARNING: {options.image} is not in the public image catalog. "
            "Private images need an x-skaha-registry-auth header and will otherwise be "
            "rejected with HTTP 400 on every attempt.",
            file=sys.stderr,
            flush=True,
        )

    session_id = None
    classes: list[str] = []
    for attempt in range(1, options.tries + 1):
        try:
            status, text = post(opener, params, timeout=120)
        except urllib.error.HTTPError as exc:
            text = exc.read().decode(errors="replace")
            status = exc.code
            label = classify(text)
            classes.append(label)
            print(f"[pod_ship] attempt {attempt}: HTTP {status} ({label}) retrying", flush=True)
            time.sleep(options.retry_sleep)
            continue
        except (urllib.error.URLError, TimeoutError, ssl.SSLError) as exc:
            classes.append("transport")
            print(f"[pod_ship] attempt {attempt}: transport error {exc}; retrying", flush=True)
            time.sleep(options.retry_sleep)
            continue
        if status == 200:
            session_id = text.strip().strip('"')
            print(f"[pod_ship] session {session_id} accepted on attempt {attempt}", flush=True)
            break
        classes.append(classify(text))
        print(f"[pod_ship] attempt {attempt}: HTTP {status} ({classify(text)}) retrying", flush=True)
        time.sleep(options.retry_sleep)

    if not session_id:
        print("[pod_ship] FAILED: no session accepted", file=sys.stderr)
        for label in sorted(set(classes)):
            print(f"  {classes.count(label):3d} x {label}", file=sys.stderr)
        return 1

    deadline = time.time() + options.wait
    status_text = ""
    while time.time() < deadline:
        info = canfar_retry("info", session_id, tries=3)
        for line in info.splitlines():
            if "Status" in line:
                status_text = line.strip().strip("│").strip()
                break
        if any(word in status_text for word in ("Completed", "Succeeded", "Failed", "Error")):
            break
        time.sleep(10)

    print(f"[pod_ship] final status: {status_text or 'unknown'}", flush=True)
    print("[pod_ship] ===== pod log =====", flush=True)
    print(canfar_retry("logs", session_id), flush=True)
    print("[pod_ship] ===== end pod log =====", flush=True)

    ok = "Completed" in status_text or "Succeeded" in status_text
    if not options.keep:
        # --force: without it the client prompts for confirmation and an
        # unattended harness would block on stdin until the shell is reclaimed.
        canfar_retry("delete", "--force", session_id, tries=3)
        print(f"[pod_ship] session {session_id} delete requested", flush=True)
    if classes:
        print("[pod_ship] create-failure classes:", flush=True)
        for label in sorted(set(classes)):
            print(f"  {classes.count(label):3d} x {label}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
