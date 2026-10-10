from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from weightmask.config import configuration_errors

def manifest_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _set_override(config, dotted, value):
    parts = str(dotted).split(".")
    if any(not part for part in parts):
        raise ValueError(f"invalid configuration override path: {dotted!r}")
    node = config
    for part in parts[:-1]:
        child = node.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"invalid configuration override path: {dotted!r}")
        node = child
    node[parts[-1]] = value


def _override_config(config_set):
    config = {}
    for key, value in config_set.items():
        if isinstance(value, dict) and "." not in str(key):
            nested = _override_config(value)
            target = config.setdefault(str(key), {})
            if not isinstance(target, dict):
                raise ValueError(f"invalid configuration override path: {key!r}")
            target.update(nested)
        else:
            _set_override(config, key, value)
    return config


def _validate_overrides(group):
    errors = configuration_errors(_override_config(group.get("config_set", {})))
    if errors:
        raise ValueError("invalid configuration override: " + " ".join(errors))


def _unresolved(value):
    if value is None or value == "":
        return True
    return isinstance(value, str) and (value.strip().upper() == "TBD" or "TBD" in value.upper())


def _group(manifest, exp_id):
    try:
        return next(group for group in manifest["groups"] if group["exp_id"] == exp_id)
    except (KeyError, StopIteration) as exc:
        raise ValueError(f"unknown CANFAR experiment group: {exp_id}") from exc


def validate_manifest(manifest, exp_id=None):
    groups = manifest.get("groups", [])
    selected = groups if exp_id is None else [_group(manifest, exp_id)]
    for group in selected:
        _validate_overrides(group)
    if exp_id is None:
        return manifest
    group = selected[0]
    for ids_key, publisher_key in (("safe_ids", "publisherIDs"), ("flat_safe_ids", "flat_publisherIDs")):
        if len(group.get(ids_key, [])) != len(group.get(publisher_key, [])):
            raise ValueError(f"{exp_id} {ids_key}/{publisher_key} cardinality mismatch")
    required = ("safe_ids", "publisherIDs", "flat_safe_ids", "flat_publisherIDs", "flat_paths")
    unresolved = [field for field in required if not group.get(field) or any(_unresolved(item) for item in group[field])]
    if unresolved:
        raise ValueError(f"{exp_id} group has unresolved inputs: {', '.join(unresolved)}")
    if exp_id == "E7" and len(group["flat_safe_ids"]) > 1:
        raise ValueError("E7 multiple flats require an explicit per-exposure mapping")
    return manifest


def validate_manifest_path(path: str | Path, exp_id=None):
    path = Path(path)
    manifest = json.loads(path.read_text())
    validate_manifest(manifest, exp_id)
    return manifest


if __name__ == "__main__":
    try:
        validate_manifest_path(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    except (IndexError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
