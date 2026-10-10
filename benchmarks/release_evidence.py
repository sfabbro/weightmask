from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "weightmask.release-evidence.v1"
REQUIRED_FIELDS = (
    "schema",
    "status",
    "version",
    "commit_sha",
    "config_sha",
    "input_manifest_sha",
    "metric_revisions",
    "command",
    "timestamp",
    "data_ids",
    "real_trail_recall",
)
SCOPE_EXCLUSIONS = {"real_trail_recall"}


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git_commit(repo: str | Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def validate_manifest(
    manifest,
    *,
    version=None,
    commit_sha=None,
    config_sha=None,
    input_manifest_sha=None,
    metric_revisions=None,
    data_ids=None,
    require_qualified=False,
):
    errors = []
    if not isinstance(manifest, dict):
        return ["release evidence manifest is missing or not an object"]
    for field in REQUIRED_FIELDS:
        if field not in manifest:
            errors.append(f"release evidence is missing {field}")
    if errors:
        return errors
    if manifest["schema"] != SCHEMA:
        errors.append("release evidence has an unsupported schema")
    if manifest["status"] not in {"passed", "failed", "skipped"}:
        errors.append("release evidence has an invalid status")
    elif manifest["status"] != "passed":
        errors.append(f"release evidence status is {manifest['status']}, not passed")
    if version is not None and manifest["version"] != version:
        errors.append("release evidence version does not match the release")
    if commit_sha is not None and manifest["commit_sha"] != commit_sha:
        errors.append("release evidence commit SHA does not match HEAD")
    if config_sha is not None and manifest["config_sha"] != config_sha:
        errors.append("release evidence configuration SHA does not match")
    if input_manifest_sha is not None and manifest["input_manifest_sha"] != input_manifest_sha:
        errors.append("release evidence input/manifest SHA does not match")
    actual_metric_revisions = manifest["metric_revisions"]
    actual_data_ids = manifest["data_ids"]
    if metric_revisions is not None and (
        not isinstance(actual_metric_revisions, list) or actual_metric_revisions != list(metric_revisions)
    ):
        errors.append("release evidence metric revisions do not match")
    if data_ids is not None and (not isinstance(actual_data_ids, list) or actual_data_ids != list(data_ids)):
        errors.append("release evidence data IDs do not match")
    if not isinstance(actual_metric_revisions, list) or not actual_metric_revisions:
        errors.append("release evidence metric revisions are empty")
    if not isinstance(actual_data_ids, list):
        errors.append("release evidence data IDs are not a list")
    if not re.fullmatch(r"[0-9a-f]{40}", str(manifest["commit_sha"])):
        errors.append("release evidence commit SHA is invalid")
    for field in ("config_sha", "input_manifest_sha"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(manifest[field])):
            errors.append(f"release evidence {field} is invalid")
    if not isinstance(manifest["command"], str) or not manifest["command"].strip():
        errors.append("release evidence command is empty")
    if not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", str(manifest["timestamp"])):
        errors.append("release evidence timestamp is invalid")
    recall = manifest["real_trail_recall"]
    exclusions = manifest.get("scope_exclusions", [])
    if not isinstance(exclusions, list) or any(not isinstance(item, str) for item in exclusions):
        errors.append("release evidence scope exclusions are invalid")
        exclusions = []
    unknown = sorted(set(exclusions) - SCOPE_EXCLUSIONS)
    if unknown:
        errors.append("release evidence has unknown scope exclusions: " + ", ".join(unknown))
    if recall == "n/a":
        if "real_trail_recall" not in exclusions:
            errors.append("release evidence real-trail recall is n/a")
    elif (
        isinstance(recall, bool)
        or not isinstance(recall, (int, float))
        or not math.isfinite(recall)
        or not 0 <= recall <= 1
    ):
        errors.append("release evidence real-trail recall is invalid")
    elif "real_trail_recall" in exclusions:
        errors.append("release evidence excludes real-trail recall but reports a number")
    # require_qualified is part of the caller contract. It does not relax these
    # errors: release_check fails on them only when the flag is set.
    del require_qualified
    return errors


def write_manifest(path: str | Path, manifest: dict) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return destination


def read_manifest(path: str | Path):
    return json.loads(Path(path).read_text())
