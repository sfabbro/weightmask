"""Re-derive the reviewable numbers from what actually landed on the mount.

Runs in a separate CANFAR session from the processing jobs and trusts nothing
the driver recorded: it reopens each mask, recomputes every quality-bit count
from the pixels, and compares those counts both with the shipped stats table and
with the digest and size in `run.json`. A file that is truncated, half-written or
simply not what was claimed shows up here as a mismatch rather than as a number
someone copied from a log.

It also writes one summary table per campaign into the fetchable results root,
so the review does not depend on reading pod logs at all.

Usage (in a pod, from the staged tree):
    {py} benchmarks/canfar/verify_exposures.py [--root DIR] [--out FILE]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

RESULTS_ROOT = "/arc/projects/mlao/cfhtcast/weightmask-perf/results"
SUMMARY_NAME = "streak_summary.tsv"

# Summary column -> the ``QUALITY_BITS`` key it is read from. Written as a
# mapping because the bit names are not the prose names: `QUALITY_BITS` has `CR`
# and `SAT`, so looking up `COSMIC_RAY` or `SATURATED` yields a zero column that
# reads like a detector with no cosmics in it.
SUMMARY_COLUMNS = (
    ("obsid", None),
    ("status", None),
    ("hdus", None),
    ("hdus_processed", None),
    ("mask_bytes", None),
    ("seconds", None),
    ("streak_px", "STREAK"),
    ("streak_px_per_hdu", None),
    ("streak_hdus", None),
    ("cr_px", "CR"),
    ("sat_px", "SAT"),
    ("bad_px", "BAD"),
    ("no_data_px", "NO_DATA"),
    ("invalid_var_px", "INVALID_VARIANCE"),
    ("flagged_px", None),
    ("npix", None),
    ("stats_rows", None),
    ("digest_match", None),
    ("stats_match", None),
    ("error", None),
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_stats_table(path: Path) -> tuple[dict[int, dict[str, int]], list[str]]:
    """Parse the shipped per-HDU stats table: rows by HDU index, plus its header.

    The header comes back too because the cross-check can only compare columns
    the table actually has; comparing a column it never had would report every
    row as a mismatch and drown the real disagreement.
    """
    if not path.exists():
        return {}, []
    rows: dict[int, dict[str, int]] = {}
    with open(path, "r", errors="replace") as handle:
        lines = handle.read().splitlines()
    if not lines:
        return {}, []
    header = lines[0].split("\t")
    for line in lines[1:]:
        values = line.split("\t")
        if len(values) != len(header):
            continue
        row: dict[str, int] = {}
        for name, value in zip(header, values):
            try:
                row[name] = int(value)
            except ValueError:
                row[name] = 0
        rows[row.get("hdu", -1)] = row
    return rows, header


def recompute(mask_path: Path, bits: dict[str, int]) -> list[dict[str, int]]:
    """Recompute every bit count from the mask pixels, HDU by HDU."""
    import fitsio
    import numpy as np

    rows: list[dict[str, int]] = []
    with fitsio.FITS(str(mask_path)) as handle:
        for index in range(len(handle)):
            try:
                data = handle[index].read()
            except Exception as exc:  # noqa: BLE001 - a truncated slot is the finding
                rows.append({"hdu": index, "npix": 0, "flagged": 0, "error": f"{type(exc).__name__}"})
                continue
            if data is None or getattr(data, "ndim", 0) != 2:
                continue
            row = {
                "hdu": index,
                "npix": int(data.size),
                "flagged": int(np.count_nonzero(data)),
            }
            for name, bit in bits.items():
                row[name] = int(np.count_nonzero(data & bit))
            rows.append(row)
    return rows


def summarise(obsid: str, directory: Path, bits: dict[str, int]) -> dict[str, object]:
    record: dict[str, object] = {"obsid": obsid, "dir": str(directory)}
    run_json = directory / f"{obsid}.run.json"
    if run_json.exists():
        try:
            record["run"] = json.loads(run_json.read_text())
        except json.JSONDecodeError as exc:
            record["run_error"] = str(exc)
    mask = directory / f"{obsid}.mask.fits.fz"
    if not mask.exists():
        record["error"] = "no mask"
        return record

    try:
        rows = recompute(mask, bits)
    except Exception as exc:  # noqa: BLE001 - an unreadable mask is the answer
        # A half-written `.fz` can fail at open rather than per-HDU. That is a
        # finding about the artifact, not a reason for the review to stop.
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    good = [row for row in rows if "error" not in row]
    stats, stats_header = read_stats_table(directory / f"{obsid}.hdu_stats.tsv")

    record["mask_bytes"] = mask.stat().st_size
    record["digest_match"] = sha256_of(mask) == (record.get("run") or {}).get("mask_sha256")
    record["recomputed_rows"] = len(good)
    record["holes"] = sum(1 for row in rows if "error" in row)
    record["mses"] = [row.get("error") for row in rows if "error" in row][:5]
    record["npix"] = sum(row["npix"] for row in good)
    record["flagged"] = sum(row["flagged"] for row in good)
    for name in bits:
        record[name] = sum(row.get(name, 0) for row in good)
    record["streak_hdus"] = sum(1 for row in good if row.get("STREAK", 0) > 0)
    record["stats_rows"] = len(stats)
    # Cross-check the shipped table column by column: a stats file that does not
    # match the pixels it claims to describe is worse than no stats file.
    compared = [name for name in ("STREAK", "CR", "SAT", "flagged", "npix") if name in stats_header]
    mismatches = []
    for row in good:
        shipped = stats.get(row["hdu"])
        if shipped is None:
            mismatches.append((row["hdu"], "missing"))
            continue
        for name in compared:
            if shipped.get(name) != row.get(name):
                mismatches.append((row["hdu"], name, shipped.get(name), row.get(name)))
    record["stats_mismatches"] = mismatches[:8]
    record["stats_match"] = not mismatches
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=RESULTS_ROOT)
    parser.add_argument("--out", default="")
    parser.add_argument("--only", default="", help="comma-separated observation ids")
    options = parser.parse_args(argv)

    from weightmask.contract import QUALITY_BITS

    root = Path(options.root)
    wanted = {item.strip() for item in options.only.split(",") if item.strip()}
    obsids = []
    for entry in sorted(root.glob("ns-*")):
        obsid = entry.name[len("ns-") :]
        if wanted and obsid not in wanted:
            continue
        # This tool's own output directory matches `ns-*` too, and any stray
        # directory under the results root would otherwise be reported as an
        # exposure with no artifacts.
        if not ((entry / f"{obsid}.run.json").exists() or (entry / f"{obsid}.mask.fits.fz").exists()):
            print(f"[verify] skipping {entry.name}: no run.json and no mask", flush=True)
            continue
        obsids.append((obsid, entry))
    if not obsids:
        print(f"[verify] no results under {root}", flush=True)
        return 1

    print(f"[verify] {len(obsids)} exposure result(s) under {root}", flush=True)
    summaries = []
    for obsid, directory in obsids:
        record = summarise(obsid, directory, QUALITY_BITS)
        summaries.append(record)
        run = record.get("run") or {}
        print(
            "[verify] {obsid}: status={status} hdus={rows}/{expected} holes={holes} "
            "streak_px={streak} streak_hdus={shdus} digest_ok={digest} stats_ok={stats}".format(
                obsid=obsid,
                status=run.get("status", "?"),
                rows=record.get("recomputed_rows", 0),
                expected=len(run.get("hdus") or []),
                holes=record.get("holes", 0),
                streak=record.get("STREAK", 0),
                shdus=record.get("streak_hdus", 0),
                digest=record.get("digest_match"),
                stats=record.get("stats_match"),
            ),
            flush=True,
        )
        for mismatch in record.get("stats_mismatches") or []:
            print(f"    stats mismatch: {mismatch}", flush=True)
        for missing in record.get("mses") or []:
            print(f"    unreadable HDU slot: {missing}", flush=True)

    out_path = Path(options.out) if options.out else root / "ns-review" / SUMMARY_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(name for name, _ in SUMMARY_COLUMNS)]
    for record in summaries:
        run = record.get("run") or {}
        processed = record.get("recomputed_rows", 0)
        values: dict[str, object] = {
            "obsid": record["obsid"],
            "status": run.get("status", "missing"),
            "hdus": len(run.get("hdus") or []),
            "hdus_processed": run.get("hdus_processed", processed),
            "mask_bytes": record.get("mask_bytes", 0),
            "seconds": run.get("seconds", 0),
            "streak_px_per_hdu": round(record.get("STREAK", 0) / processed, 1) if processed else 0,
            "streak_hdus": record.get("streak_hdus", 0),
            "flagged_px": record.get("flagged", 0),
            "npix": record.get("npix", 0),
            "stats_rows": record.get("stats_rows", 0),
            "digest_match": record.get("digest_match", False),
            "stats_match": record.get("stats_match", False),
            "error": record.get("error", ""),
        }
        for name, bit in SUMMARY_COLUMNS:
            if bit and name not in values:
                values[name] = record.get(bit, 0)
        lines.append("\t".join(str(values.get(name, "")) for name, _ in SUMMARY_COLUMNS))
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[verify] wrote {out_path} ({len(summaries)} rows)", flush=True)
    print("[verify] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
