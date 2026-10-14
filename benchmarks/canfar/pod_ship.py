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

The platform's own `env` parameter is *lossy* for the values this harness
sends. Carrying `WM_COMMAND` through it did not deliver the variable at all, so
the runner reported `FATAL: WM_TREE and WM_COMMAND are both required` -- which
reads exactly like a flaky control plane, and was recorded as one. That failure
is pod behaviour, so it is certain; bisecting it on 2026-10-13 correlated with
`=` in the value (a value containing one did not arrive, and one ending in `=`
arrived with it stripped), which is how `--tb=short` and `-o flat=x` vanished.
Treat that correlation as the working rule rather than the proven parser:
absence was read from the log, and the log is not a faithful channel (see
below). A `$` is separately rejected with `400 Illegal group reference`.

Environment is therefore carried *inside the payload*: argv token 1 is
`base64(json(env))` and the bootstrap applies it before the script runs, so no
value is subject to the platform's parser. argv is proven lossless -- an 8-token
base64 stage payload sha256-verified inside the pod.

One caution when reading any of this back: **`canfar logs` rewrites the text it
renders.** A pod that deliberately set `WM_SHORTCODE=a:b:c` prints `a:b:c`, and
the log shows `a\U0001f171c` -- an emoji expansion applied to the rendering. That
briefly looked like a transport fault here. To read an exact value back, have
the payload print base64 taken inside the pod: the alphabet has no `:`, so the
log has nothing to rewrite.

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
    --delete-unfinished  delete the session even when the deadline expired while
                       it was still running (default: leave it alone)
    --max-body BYTES   refuse to send a body larger than this (default 4 MiB)
    --dry-run          report the assembled body size, create nothing

A deadline that expires while the pod is still running must not delete the
session. On 2026-10-13 a fan-out of seven MegaCam exposures was launched with a
fixed `--wait`; four finished and three were deleted mid-run, which left each of
them with an unrenamed `.wm-tmp-*` mask and no per-HDU stats. The deletion was
ours, not the platform's, and it looked exactly like a cluster fault. Long jobs
are the expected case here, so the session now survives its harness.

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
# Applies the environment, then rejoins argv[2:] into the payload and executes
# it. It must contain no space, because a space is an argument boundary by the
# time the server has re-split cmd and args -- which is also why the imports go
# through `__import__` (`import a,b` needs one). `tests/test_canfar_scripts.py`
# asserts the no-space property and drives the whole thing through a split.
# argv[1] is the environment as base64 JSON; echoing it first makes the pod log
# the record of what the job actually ran with.
BOOTSTRAP = (
    "_i=__import__;"
    "_sys=_i('sys');_os=_i('os');"
    "_env=_i('json').loads(_i('base64').b64decode(_sys.argv[1]));"
    "_os.environ.update(_env);"
    "print('WM_ENV',_i('json').dumps(_env,sort_keys=True),flush=True);"
    "exec(compile(_i('lzma').decompress(_i('base64').b64decode(''.join(_sys.argv[2:]))),'payload','exec'))"
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


def build_args(script: bytes, env: dict[str, str]) -> tuple[str, dict]:
    """Encode the script and its environment as space-free argv tokens.

    Returns the args string and a small bookkeeping dict for reporting. The
    payload is lzma+base64 and is split whenever it exceeds MAX_ARG_STRLEN,
    which the bootstrap rejoins; the environment rides along as argv[1] so it
    never touches the platform's lossy `env` parser.
    """
    blob = base64.b64encode(lzma.compress(script, preset=9 | lzma.PRESET_EXTREME)).decode("ascii")
    chunks = [blob[i : i + MAX_ARG_CHARS] for i in range(0, len(blob), MAX_ARG_CHARS)]
    env_token = base64.b64encode(json.dumps(env, sort_keys=True).encode()).decode("ascii")
    info = {"tokens": len(chunks) + 1, "blob": len(blob), "env_keys": sorted(env)}
    return "-c " + BOOTSTRAP + " " + env_token + " " + " ".join(chunks), info


def parse_env(pairs: list[str]) -> dict[str, str]:
    """Turn repeated ``KEY=VALUE`` arguments into a map, rejecting bad input.

    A value is free to contain `=` and `$` -- that is the whole reason this
    does not go through the platform's `env` parameter -- but the key still has
    to be a usable environment name, and a key given twice is a mistake rather
    than a silent last-writer-wins.
    """
    env: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"--env needs KEY=VALUE, got {pair!r}")
        if not key.isidentifier():
            raise SystemExit(f"--env key {key!r} is not a valid environment name")
        if key in env:
            raise SystemExit(f"--env key {key!r} given twice")
        env[key] = value
    return env


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


def deletion_plan(status_text: str, keep: bool, delete_unfinished: bool) -> tuple[bool, str]:
    """Decide whether to delete the session, and say why.

    Returns ``(delete, reason)``. A session that is still running when the local
    deadline expires is left in place: deleting it would destroy work the pod is
    in the middle of, and the caller cannot tell that apart from a platform
    fault. Split out from ``main`` so the rule is testable without a pod.
    """
    if keep:
        return False, "--keep"
    if status_text_is_finished(status_text):
        return True, "the pod finished"
    if delete_unfinished:
        return True, "--delete-unfinished was given"
    return False, "the deadline expired while the pod was still running"


def status_text_is_finished(status_text: str) -> bool:
    return any(word in status_text for word in ("Completed", "Succeeded", "Failed", "Error"))


def build_params(options, args: str) -> dict:
    """Assemble the create request.

    Deliberately has no ``env`` key: the platform's ``env`` parameter drops any
    entry whose value contains ``=``, so the environment travels in ``args``
    instead. Split out from ``main`` so a test can assert that absence -- a
    re-added ``params["env"]`` would silently start losing variables again.
    """
    return {
        "name": options.name or "wm-ship-" + "".join(random.choices(string.ascii_lowercase, k=6)),
        "image": options.image,
        "cores": str(options.cpu),
        "ram": str(options.ram),
        "type": "headless",
        "cmd": "python3",
        "args": args,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("script")
    parser.add_argument("--name")
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--ram", type=int, default=4)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--wait", type=int, default=900)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument(
        "--delete-unfinished",
        action="store_true",
        help="delete the session even if the deadline expired while it was still running",
    )
    parser.add_argument("--tries", type=int, default=30)
    parser.add_argument("--retry-sleep", type=float, default=6.0)
    parser.add_argument("--max-body", type=int, default=4 * 1024 * 1024)
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "environment variable for the pod (repeatable); the shipped script reads these. "
            "Carried inside the payload, so values may contain '=' and '$'"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args()

    env = parse_env(options.env)
    source = Path(options.script).read_bytes()
    args, encoding = build_args(source, env)
    params = build_params(options, args)
    body = urllib.parse.urlencode(params, doseq=True).encode()
    print(
        f"[pod_ship] script {len(source)} B -> lzma+b64 {encoding['blob']} B "
        f"({encoding['tokens']} argv tokens incl. env) -> form body {len(body)} B",
        flush=True,
    )
    for key in encoding["env_keys"]:
        print(f"[pod_ship] env {key}={env[key]!r} (in-payload)", flush=True)
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
        if status_text_is_finished(status_text):
            break
        time.sleep(10)

    print(f"[pod_ship] final status: {status_text or 'unknown'}", flush=True)
    print("[pod_ship] ===== pod log =====", flush=True)
    print(canfar_retry("logs", session_id), flush=True)
    print("[pod_ship] ===== end pod log =====", flush=True)

    ok = "Completed" in status_text or "Succeeded" in status_text
    delete, reason = deletion_plan(status_text, options.keep, options.delete_unfinished)
    if delete:
        # --force: without it the client prompts for confirmation and an
        # unattended harness would block on stdin until the shell is reclaimed.
        canfar_retry("delete", "--force", session_id, tries=3)
        print(f"[pod_ship] session {session_id} delete requested ({reason})", flush=True)
    else:
        print(
            f"[pod_ship] session {session_id} kept running ({reason}); "
            f"its results land on the mount as it writes them, so it does not need this process",
            flush=True,
        )
    if classes:
        print("[pod_ship] create-failure classes:", flush=True)
        for label in sorted(set(classes)):
            print(f"  {classes.count(label):3d} x {label}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
