#!/usr/bin/env python3
"""Opt-in MegaCam science gate.

Runs only when ``benchmark_data/megacam/perf`` contains FITS files. A missing
directory is a skip: the report says so, and a skip is not a pass. This task
is not part of ``pixi run test``.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import json

try:
    from .release_evidence import git_commit, sha256, utc_now, write_manifest
except ImportError:
    from release_evidence import git_commit, sha256, utc_now, write_manifest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join("benchmark_data", "megacam", "perf")
DEFAULT_OUT = os.path.join("test_outputs", "harness", "science-gate.md")
DEFAULT_MANIFEST = os.path.join("test_outputs", "harness", "science-gate.json")
INPUT_MANIFEST = os.path.join("benchmarks", "trail_truth", "megacam_real_labels.json")
METRIC_REVISIONS = ["truth-band-recall-v2", "truth-pixel-recall-v3", "additive-finite-injection-v3"]
WORM_AUDIT = 0.275
WORM_DROP = 0.02
SINGLE_FLOOR = 0.15
CONTINUOUS_RECALL = 0.75
RECALL5 = 0.90


def _fits_paths(directory):
    found = []
    if not os.path.isdir(directory):
        return found
    for dirpath, _dirs, files in os.walk(directory):
        for name in sorted(files):
            lower = name.lower()
            if lower.endswith(".fits") or lower.endswith(".fz"):
                found.append(os.path.join(dirpath, name))
    return found


def _read_tsv(path):
    with open(path) as handle:
        lines = [line.rstrip("\n") for line in handle if line.strip()]
    if not lines:
        return []
    header = lines[0].split("\t")
    if len(set(header)) != len(header) or any(not key for key in header):
        raise ValueError("invalid TSV header")
    rows = []
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) != len(header):
            raise ValueError("TSV row does not match its header")
        rows.append(dict(zip(header, parts)))
    return rows


def _mean(values):
    nums = [float(value) for value in values]
    if not nums or any(not math.isfinite(value) or not 0 <= value <= 1 for value in nums):
        raise ValueError("recall metrics must be nonempty, finite, and between 0 and 1")
    return sum(nums) / len(nums)


def _run(args):
    print("+ " + " ".join(args), flush=True)
    proc = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    text = (proc.stdout or "") + (proc.stderr or "")
    if text:
        print(text, end="" if text.endswith("\n") else "\n")
    return proc.returncode, text


def _write(path, lines):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def _evidence(version, command, status, data_ids):
    config = os.path.join(ROOT, "weightmask.yml")
    input_manifest = os.path.join(ROOT, INPUT_MANIFEST)
    return {
        "schema": "weightmask.release-evidence.v1",
        "status": status,
        "version": version,
        "commit_sha": git_commit(ROOT),
        "config_sha": sha256(config) if os.path.isfile(config) else "",
        "input_manifest_sha": sha256(input_manifest) if os.path.isfile(input_manifest) else "",
        "input_sha": sha256(input_manifest) if os.path.isfile(input_manifest) else "",
        "manifest_sha": sha256(input_manifest) if os.path.isfile(input_manifest) else "",
        "metric_revisions": METRIC_REVISIONS,
        "command": command,
        "timestamp": utc_now(),
        "data_ids": sorted(data_ids),
        "real_trail_recall": "n/a",
        # The shipped fixture has no independently confirmed trail. A numeric
        # recall recorded later clears this exclusion.
        "scope_exclusions": ["real_trail_recall"],
    }


def _repo_path(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def _write_evidence(path, evidence):
    write_manifest(path, evidence)


def _fixture_data_ids():
    path = os.path.join(ROOT, INPUT_MANIFEST)
    try:
        with open(path) as handle:
            fixture = json.load(handle)
        return list(fixture.get("data_requirements", {}).get("exposures", {}))
    except (OSError, ValueError, TypeError):
        return []


def _skip(out_path, data_dir, manifest_path, version, command):
    line = (
        f"{data_dir} has no FITS files. science-gate skipped. "
        "A skip is not a pass: artefact FP, injected recall, cosmic-ray, and timing gates were not run."
    )
    print(line)
    _write(
        out_path,
        [
            "# science-gate",
            "",
            "status: skipped",
            "",
            line,
            "",
            "No MegaCam timing or false-positive rate was measured in this run.",
        ],
    )
    _write_evidence(manifest_path, _evidence(version, command, "skipped", _fixture_data_ids()))
    return 0


def _score(data_dir, out_dir, failures, notes):
    code, text = _run(
        [
            sys.executable,
            "benchmarks/score_trail_truth.py",
            "--detector",
            "streaks",
            "--apply-persistence",
            *(["--data-root", data_dir] if data_dir is not None else []),
            "--report",
            os.path.join(out_dir, "trail_truth.score.json"),
        ]
    )
    notes.append("score_trail_truth --detector streaks --apply-persistence")
    if code != 0:
        failures.append(f"score_trail_truth exited {code}")
    for line in text.splitlines():
        if line.startswith("| streaks |") or line.startswith("Wrote "):
            notes.append(line.strip())


def _inject(exposure, hdu, out_dir, failures, notes):
    tsv = os.path.join(out_dir, "streak_inject.tsv")
    code, _text = _run(
        [
            sys.executable,
            "benchmarks/streak_inject.py",
            exposure,
            "--hdu",
            str(hdu),
            "--quick",
            "--seeds",
            "1",
            "--summary-only",
            "--out",
            tsv,
        ]
    )
    if code != 0 or not os.path.exists(tsv):
        failures.append(f"streak_inject exited {code}")
        return
    try:
        rows = [row for row in _read_tsv(tsv) if str(row.get("dashed", "")).lower() in ("false", "0")]
        recall = _mean(row["recall"] for row in rows)
        recall5 = _mean(row["recall5"] for row in rows)
    except (OSError, KeyError, ValueError) as exc:
        failures.append(f"streak_inject invalid metrics: {exc}")
        return
    notes.append(f"streak_inject continuous recall {recall:.3f}, recall5 {recall5:.3f} on {exposure} HDU {hdu}")
    if recall < CONTINUOUS_RECALL:
        failures.append(f"continuous recall {recall:.3f} < {CONTINUOUS_RECALL:.2f}")
    if recall5 < RECALL5:
        failures.append(f"recall at 5px {recall5:.3f} < {RECALL5:.2f}")


def _cosmics(exposure, hdu, out_dir, failures, notes):
    tsv = os.path.join(out_dir, "cr_faint.tsv")
    code, _text = _run(
        [
            sys.executable,
            "benchmarks/cr_faint_curves.py",
            exposure,
            "--hdu",
            str(hdu),
            "--seeds",
            "1",
            "--out",
            tsv,
        ]
    )
    if code != 0 or not os.path.exists(tsv):
        failures.append(f"cr_faint_curves exited {code}")
        return
    try:
        rows = [row for row in _read_tsv(tsv) if row.get("variant") == "main+faint"]
        worm = _mean(row["worm_recall"] for row in rows)
        single = _mean(row["single_recall"] for row in rows)
    except (OSError, KeyError, ValueError) as exc:
        failures.append(f"cr_faint_curves invalid metrics: {exc}")
        return
    notes.append(f"cr two-pass worm {worm:.3f} (audit {WORM_AUDIT:.3f}), single-pixel {single:.3f}")
    if worm < WORM_AUDIT - WORM_DROP:
        failures.append(f"worm recall {worm:.3f} dropped by more than {WORM_DROP:.2f} from {WORM_AUDIT:.3f}")
    if single < SINGLE_FLOOR:
        failures.append(f"single-pixel recall {single:.3f} < {SINGLE_FLOOR:.2f}")


def _perf(exposure, out_dir, failures, notes):
    perf_dir = os.path.join(out_dir, "perf")
    code, text = _run(
        [
            sys.executable,
            "benchmarks/perf_megacam.py",
            "--exposure-file",
            exposure,
            "--hdu-limit",
            "2",
            "--no-cprofile",
            "--out-dir",
            perf_dir,
        ]
    )
    notes.append(
        f"perf_megacam on 2 HDUs of {os.path.basename(exposure)} without --compare-baseline "
        "(streak prior, sky handoff, and sentinel weights are allowed to differ)"
    )
    if code != 0:
        failures.append(f"perf_megacam exited {code}")
    for line in text.splitlines():
        if line.startswith("Wrote ") or line.startswith("| streaks |") or line.startswith("exposures="):
            notes.append(line.strip())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", help=f"Relocate fixture inputs here (default uses fixture paths; probe {DEFAULT_DATA})."
    )
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--version", default=None)
    parser.add_argument("--hdu", type=int, default=1)
    args = parser.parse_args(argv)
    version = args.version or "0.0.0"
    command = "pixi run science-gate" + (" " + " ".join(argv) if argv else "")
    manifest_path = _repo_path(args.manifest)

    selected_data = args.data_dir if args.data_dir is not None else DEFAULT_DATA
    data_dir = selected_data if os.path.isabs(selected_data) else os.path.join(ROOT, selected_data)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    fits = _fits_paths(data_dir)
    if not fits:
        return _skip(
            out_path,
            os.path.relpath(data_dir, ROOT) if data_dir.startswith(ROOT) else data_dir,
            manifest_path,
            version,
            command,
        )

    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    exposure = next((path for path in fits if "1013719p" in os.path.basename(path)), fits[0])
    failures = []
    notes = [f"data: `{exposure}`"]
    _score(data_dir if args.data_dir is not None else None, out_dir, failures, notes)
    _inject(exposure, args.hdu, out_dir, failures, notes)
    _cosmics(exposure, args.hdu, out_dir, failures, notes)
    _perf(exposure, out_dir, failures, notes)
    status = "failed" if failures else "passed"
    evidence = _evidence(version, command, status, _fixture_data_ids())
    try:
        score_path = os.path.join(out_dir, "trail_truth.score.json")
        with open(score_path) as handle:
            score = json.load(handle)
        recall = score.get("totals", {}).get("streaks", {}).get("trail_recall")
        if recall is None:
            evidence["real_trail_recall"] = "n/a"
            evidence["scope_exclusions"] = ["real_trail_recall"]
        else:
            evidence["real_trail_recall"] = recall
            evidence["scope_exclusions"] = []
    except (OSError, ValueError, TypeError):
        pass
    _write_evidence(manifest_path, evidence)
    lines = ["# science-gate", "", f"status: {status}", ""]
    if failures:
        lines.append("## Failures")
        lines.extend(f"- {item}" for item in failures)
        lines.append("")
    lines.append("## Recorded")
    lines.extend(f"- {item}" for item in notes if item)
    lines.append("")
    lines.append("No speedup is claimed from this file unless a timing line above was produced by perf_megacam.")
    _write(out_path, lines)
    print(f"science-gate status: {status}")
    print(f"Wrote {out_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
