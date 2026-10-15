#!/usr/bin/env python3
"""Process one MegaCam exposure on node-local scratch, leaving the reviewable
artifacts on the shared mount.

A pod is the only place that has CADC egress, node-local scratch and enough
cores to process a 36-CCD exposure, and everything on /scratch dies with the
session. So the exposure itself is downloaded to scratch (fast, throwaway) while
each artifact a human needs afterwards is written to the shared mount as the run
proceeds, not at the end.

Usage (in a pod, from the staged tree):
    {py} benchmarks/canfar/process_exposure.py OBSID --out DIR [options]

    --out DIR       fetchable destination (pod_run exports WM_RESULTS_ROOT)
    --work DIR      scratch directory for the exposure and the large products
    --workers N     per-HDU worker threads (default 8)
    --limit K       process only the first K science HDUs
    --hdus 2,3,4    process exactly these HDU indices instead of all
    --url URL       override the CADC URL (probed by default)
    --flat PATH     flat frame, matched to each science HDU by index
    --keep          keep the downloaded exposure under --work when done

Outputs in --out:
    <OBSID>.mask.fits.fz    combined quality mask, one HDU per science HDU
    <OBSID>.hdu_stats.tsv   per-HDU pixel counts per quality bit
    <OBSID>.run.json        what was processed, with what, and how long it took

The weight/invvar/sky products go to --work: they are large float32 maps, they
are reproducible from the exposure plus the config, and no review of "what did
the mask do" needs them beside the mask.

No persistence prior is applied. That prior is defined across *other* science
frames of the same detectors in the same run, and one observation id has none.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import fitsio
import numpy as np
import yaml

from weightmask import mef
from weightmask.cli import determine_output_paths
from weightmask.contract import QUALITY_BITS
from weightmask.utils import image_hdus


def _pod_run():
    """The sibling runner owns the CADC base URL and the download retry policy.

    Imported by name rather than by package path: in a pod this script runs as
    `python benchmarks/canfar/process_exposure.py`, where its own directory (not
    the tree root) is on sys.path, while the local test suite imports it as
    `benchmarks.canfar.process_exposure`.
    """
    try:
        from benchmarks.canfar import pod_run
    except ImportError:  # pragma: no cover - the pod's layout
        import pod_run  # type: ignore[no-redef]

    return pod_run


def candidate_urls(obsid: str) -> list[str]:
    """CADC addresses published CFHT files by name, with and without the 'p'."""
    base = _pod_run().CADC_PUB
    return [f"{base}/{obsid}p.fits.fz", f"{base}/{obsid}.fits.fz"]


def probe(url: str) -> int | None:
    """HTTP status of a 1 KB ranged GET on ``url``, or None if unreachable.

    A ranged GET rather than a HEAD, following redirects: CADC answers 301 for
    these paths (measured 2026-10-13 — a HEAD probe reported 301 and the driver
    concluded the exposure was missing while the download itself would have
    worked), and the campaign's own probes already use `curl -r 0-1023`.
    """
    proc = subprocess.run(
        [
            "curl",
            "-sS",
            "-L",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            "--max-time",
            "60",
            "-r",
            "0-1023",
            url,
        ],
        capture_output=True,
        text=True,
    )
    code = (proc.stdout or "").strip()
    return int(code) if code.isdigit() else None


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mask_bit_counts(mask_path: Path | str, quality_bits: dict[str, int] | None = None) -> list[dict]:
    """Per-HDU pixel counts for each quality bit in a written mask file.

    Kept free of pod-specific state so a test can point it at a small FITS file.
    """

    bits = quality_bits or QUALITY_BITS
    rows: list[dict] = []
    with fitsio.FITS(str(mask_path)) as handle:
        for index in range(len(handle)):
            try:
                data = handle[index].read()
                header = handle[index].read_header()
            except Exception as exc:  # an interrupted run leaves trailing slots
                rows.append({"hdu": index, "error": f"{type(exc).__name__}: {exc}"})
                continue
            if data is None or getattr(data, "ndim", 0) != 2:
                continue
            counts = {name: int(np.count_nonzero(data & bit)) for name, bit in bits.items()}
            row = {
                "hdu": index,
                "extname": str(header.get("EXTNAME", "")).strip(),
                "npix": int(data.size),
                "flagged": int(np.count_nonzero(data)),
            }
            row.update(counts)
            rows.append(row)
    return rows


def write_stats(mask_path: Path, stats_path: Path) -> list[dict]:
    rows = mask_bit_counts(mask_path)
    columns = ["hdu", "extname", "npix", "flagged", *QUALITY_BITS]
    lines = ["\t".join(columns)]
    for row in rows:
        if "error" in row:
            lines.append("\t".join([str(row["hdu"]), "", "", "", "", *[""] * len(QUALITY_BITS)]))
            continue
        lines.append("\t".join(str(row.get(name, "")) for name in columns))
    stats_path.write_text("\n".join(lines) + "\n")
    print(f"[process] wrote {stats_path} ({len(rows)} HDUs)", flush=True)
    return rows


def monitor(mask_path: Path, stop: threading.Event, interval: float = 60.0) -> None:
    """Print a liveness line while the pipeline runs.

    The pipeline writes HDUs in index order, so the mask file's size and mtime
    are a real progress signal; a run that is stuck on one HDU stops both.
    """

    def cpu_seconds() -> float:
        try:
            with open("/proc/self/stat") as handle:
                fields = handle.read().split()
            ticks = os.sysconf("SC_CLK_TCK")
            return (float(fields[13]) + float(fields[14])) / ticks
        except Exception:
            return -1.0

    started = time.time()
    while not stop.wait(interval):
        elapsed = time.time() - started
        size = mask_path.stat().st_size if mask_path.exists() else 0
        age = (time.time() - mask_path.stat().st_mtime) if mask_path.exists() else -1.0
        print(
            f"[monitor] elapsed {elapsed:.0f}s mask={size} B mask_age={age:.0f}s cpu={cpu_seconds():.0f}s",
            flush=True,
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("obsid", help="CFHT observation id, e.g. 1729091")
    parser.add_argument("--out", required=True, help="fetchable destination directory")
    parser.add_argument("--work", default="/scratch/wm-exposures", help="scratch working directory")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="process only the first K science HDUs")
    parser.add_argument("--hdus", default="", help="comma-separated HDU indices to process")
    parser.add_argument("--url", default="", help="override the CADC URL")
    parser.add_argument("--flat", default="", help="flat frame, matched by HDU index")
    parser.add_argument("--config", default="weightmask/weightmask.yml")
    parser.add_argument("--keep", action="store_true", help="keep the downloaded exposure")
    options = parser.parse_args(argv)

    fetch = _pod_run().fetch

    work = Path(options.work) / options.obsid
    out = Path(options.out)
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    record = {
        "obsid": options.obsid,
        "url": "",
        "started_at": utc_now(),
        "status": "running",
        "hdu_limit": options.limit,
        "hdu_list": options.hdus,
        "workers": options.workers,
        "config": options.config,
        "flat": options.flat,
        "work": str(work),
        "out": str(out),
    }
    record_path = out / f"{options.obsid}.run.json"

    def save_record(**updates) -> None:
        record.update(updates)
        record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(f"[process] {record_path} status={record['status']}", flush=True)

    save_record()

    # --- 1. the exposure, on scratch -------------------------------------------------
    exposure = work / f"{options.obsid}p.fits.fz"
    url = options.url
    if url:
        status = probe(url)
        if status not in (200, 206):
            save_record(status="failed", error=f"{url} returned HTTP {status}")
            return 2
        print(f"[process] {url} -> HTTP {status}", flush=True)
    else:
        for candidate in candidate_urls(options.obsid):
            status = probe(candidate)
            print(f"[process] probe {candidate} -> HTTP {status}", flush=True)
            if status in (200, 206):
                url = candidate
                break
        if not url:
            save_record(status="failed", error="no CADC URL answered 200/206")
            return 2
    record["url"] = url
    if exposure.exists() and exposure.stat().st_size > 1024 * 1024:
        print(f"[process] already on scratch: {exposure} ({exposure.stat().st_size} B)", flush=True)
    else:
        if not fetch(url, str(exposure)):
            save_record(status="failed", error=f"download failed: {url}")
            return 3
    bytes_on_disk = exposure.stat().st_size
    print(f"[process] exposure {exposure} {bytes_on_disk} B", flush=True)
    record["exposure_bytes"] = bytes_on_disk
    record["exposure_sha256"] = sha256_of(exposure)
    save_record()

    # --- 2. which HDUs ---------------------------------------------------------------
    hdul_input = fitsio.FITS(str(exposure), "r")
    try:
        try:
            hdus = [int(i) for i in image_hdus(hdul_input, None)]
        except Exception as exc:
            save_record(status="failed", error=f"image_hdus: {exc}")
            hdul_input.close()
            return 4
        if options.hdus:
            wanted = [item.strip() for item in options.hdus.split(",") if item.strip()]
            hdus = [i for i in hdus if str(i) in wanted]
        if options.limit > 0:
            hdus = hdus[: options.limit]
        if not hdus:
            save_record(status="failed", error="no HDUs selected")
            hdul_input.close()
            return 4
        shapes = []
        for i in hdus:
            try:
                dims = (hdul_input[i].get_info() or {}).get("dims") or []
                if len(dims) >= 2:
                    shapes.append([int(dims[-2]), int(dims[-1])])
            except Exception:
                continue
        record["hdus"] = hdus
        record["shapes"] = shapes
        save_record()
        print(f"[process] {len(hdus)} science HDUs: {hdus}", flush=True)

        # --- 3. process, streaming every HDU's outputs to disk -----------------------
        with open(options.config) as handle:
            config = yaml.safe_load(handle)
        config = _deep_merge(config, {"output_params": {"compress": True}})
        config_digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        record["config_sha256"] = config_digest
        record["compress"] = True
        save_record()

        mask_path = out / f"{options.obsid}.mask.fits.fz"
        weight_path = work / f"{options.obsid}.weight.fits.fz"
        args = argparse.Namespace(
            output_map=str(weight_path),
            output_mask=str(mask_path),
            output_invvar=None,
            output_sky=None,
            output_weight_raw=None,
            individual_masks=False,
        )
        paths = determine_output_paths(args, str(exposure), config)
        for key in ("out_invvar_path", "out_sky_path", "out_weight_raw_path"):
            paths[key] = None
        paths["individual_mask_paths"] = {}
        hdul_flat = fitsio.FITS(str(options.flat), "r") if options.flat else None

        stop = threading.Event()
        watcher = threading.Thread(target=monitor, args=(mask_path, stop), daemon=True)
        watcher.start()
        started = time.time()
        try:
            n_ok = mef.process_all_hdus(
                hdus,
                hdul_input,
                hdul_flat,
                config,
                paths,
                flat_path=str(options.flat) if options.flat else None,
                max_workers=options.workers,
                input_path=str(exposure),
            )
        finally:
            wall = time.time() - started
            stop.set()
            if hdul_flat is not None:
                hdul_flat.close()
    finally:
        try:
            hdul_input.close()
        except Exception:
            pass

    record["seconds"] = round(wall, 1)
    record["hdus_processed"] = int(n_ok)
    print(f"[process] {n_ok}/{len(hdus)} HDUs in {wall:.0f}s", flush=True)

    # --- 4. the reviewable summary ---------------------------------------------------
    if mask_path.exists():
        record["mask_bytes"] = mask_path.stat().st_size
        record["mask_sha256"] = sha256_of(mask_path)
        rows = write_stats(mask_path, out / f"{options.obsid}.hdu_stats.tsv")
        record["hdu_stats"] = rows
    else:
        save_record(status="failed", error=f"no mask written at {mask_path}")
        return 5

    if not options.keep:
        for leftover in (weight_path,):
            try:
                leftover.unlink()
            except OSError:
                pass
    save_record(status="ok" if n_ok == len(hdus) else "partial", finished_at=utc_now())
    print(f"[process] done: {options.obsid} -> {out}", flush=True)
    return 0


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


if __name__ == "__main__":
    raise SystemExit(main())
