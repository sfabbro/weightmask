from __future__ import annotations

import hashlib
import json
import math
import random
import re
import shutil
import statistics
from pathlib import Path

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)
MIN_REPEATS = 5
FIXED_THREAD_ENVIRONMENT = {name: "1" for name in THREAD_VARIABLES}


def validate_repeats(repeats: int) -> int:
    repeats = int(repeats)
    if repeats < MIN_REPEATS:
        raise ValueError(f"repetitions must be at least {MIN_REPEATS}")
    return repeats


def interleaved_schedule(
    repeats: int,
    *,
    seed: int,
    arms: tuple[str, str] = ("A", "B"),
    cold_repetitions: int = 0,
) -> list[tuple[int, str]]:
    repeats = validate_repeats(repeats)
    if len(arms) != 2 or arms[0] == arms[1]:
        raise ValueError("protocol requires two distinct arms")
    if cold_repetitions < 0:
        raise ValueError("cold_repetitions must be non-negative")
    rng = random.Random(seed)
    schedule = []
    for repetition in range(repeats + cold_repetitions):
        pair = list(arms)
        rng.shuffle(pair)
        schedule.extend((repetition, arm) for arm in pair)
    return schedule


def summarize(values) -> dict[str, float | int]:
    samples = [float(value) for value in values]
    if not samples or not all(math.isfinite(value) for value in samples):
        raise ValueError("timing samples must be finite and non-empty")
    median = statistics.median(samples)
    quartiles = statistics.quantiles(samples, n=4, method="inclusive") if len(samples) > 1 else [median] * 3
    deviations = [abs(value - median) for value in samples]
    return {
        "n": len(samples),
        "median": float(median),
        "iqr": float(quartiles[2] - quartiles[0]),
        "mad": float(statistics.median(deviations)),
    }


def thread_environment(_environment: dict[str, str] | None = None) -> dict[str, str]:
    return dict(FIXED_THREAD_ENVIRONMENT)


def cache_directory(root: str | Path, cache_state: str, arm: str, repetition: int) -> Path:
    root = Path(root)
    if cache_state == "cold":
        return root / "cold" / f"{arm}-{repetition}"
    if cache_state == "warm":
        return root / "warm" / arm
    raise ValueError(f"unknown cache state: {cache_state}")


def cleanup_cache_directory(path: str | Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logical_config_hash(path: str | Path, ephemeral_paths=()) -> str:
    text = Path(path).read_text()
    for ephemeral in ephemeral_paths:
        text = text.replace(str(ephemeral), "<EPHEMERAL>")
    text = re.sub(r"(?m)^(\s*bad_mask_cache_dir:)\s*.*$", r"\1 <EPHEMERAL>", text)
    text = re.sub(
        r"(?mi)^(\s*(?:cache_dir|temp_dir|tmp_dir|output_dir):)\s*.*$",
        r"\1 <EPHEMERAL>",
        text,
    )
    digest = hashlib.sha256()
    digest.update(text.encode())
    return digest.hexdigest()


def _hashed_inputs(paths) -> list[dict[str, str]]:
    return [{"path": str(Path(path)), "sha256": sha256_path(path)} for path in paths]


def _hashed_configs(paths) -> list[dict[str, str]]:
    return [{"path": str(Path(path)), "sha256": logical_config_hash(path)} for path in paths]


def require_correctness_equivalence(correctness: dict[str, bool]) -> None:
    if len(correctness) != 2 or not all(bool(value) for value in correctness.values()):
        raise ValueError("correctness equivalence is required before timing claims")


def release_eligibility(
    repeats: int,
    correctness_equivalent: bool,
    baseline_equivalent: bool,
    *,
    full_products: bool = False,
    distinct_arms: bool = False,
) -> dict:
    failures = []
    if int(repeats) < MIN_REPEATS:
        failures.append(f"fewer than {MIN_REPEATS} repetitions")
    if not correctness_equivalent:
        failures.append("WM-20 correctness equivalence failed")
    if not baseline_equivalent:
        failures.append("baseline product equivalence failed")
    if not full_products:
        failures.append("full product equivalence requires --all-products")
    if not distinct_arms:
        failures.append("baseline and treatment arms are not distinct")
    return {"eligible": not failures, "failures": failures}


def distinct_arm_definitions(arm_definitions: dict[str, dict] | None) -> bool:
    if not arm_definitions or len(arm_definitions) != 2:
        return False
    values = [json.dumps(value, sort_keys=True) for value in arm_definitions.values()]
    return values[0] != values[1]


def protocols_release_eligibility(
    protocols: list[dict],
    repeats: int,
    baseline_equivalent: bool,
    *,
    full_products: bool,
) -> dict:
    distinct_protocols = [protocol for protocol in protocols if protocol.get("distinct_arms")]
    correctness = bool(protocols) and all(protocol.get("correctness_equivalent", False) for protocol in protocols)
    return release_eligibility(
        repeats,
        correctness,
        baseline_equivalent,
        full_products=full_products,
        distinct_arms=bool(distinct_protocols),
    )


def make_report(
    *,
    instrument: str,
    input_paths,
    config_paths,
    repeats: int,
    seed: int,
    schedule,
    samples: dict[str, dict[str, list[float]]],
    correctness: dict[str, bool],
    thread_environment: dict[str, str],
    cold_repetitions: int = 0,
    arm_definitions: dict[str, dict] | None = None,
) -> dict:
    repeats = validate_repeats(repeats)
    require_correctness_equivalence(correctness)
    if len(schedule) < 2 * repeats or (len(schedule) - 2 * repeats) % 2:
        raise ValueError("schedule must contain warm repetitions plus complete cold pairs")
    if len(schedule) == 2 * repeats:
        raise ValueError("schedule must include separate cold runs")
    for arm, values in samples.get("warm", {}).items():
        if len(values) < repeats:
            raise ValueError(f"warm samples for {arm} require at least {repeats} warm samples")
    if not samples.get("cold") or any(not values for values in samples["cold"].values()):
        raise ValueError("each arm requires a separate cold sample")
    results = {
        cache: {arm: summarize(values) for arm, values in arms.items()}
        for cache, arms in samples.items()
    }
    return {
        "instrument": instrument,
        "protocol": {
            "repeats": repeats,
            "seed": int(seed),
            "schedule": [[int(repetition), arm] for repetition, arm in schedule],
            "cold_repetitions": int(cold_repetitions or (len(schedule) - 2 * repeats) // 2),
            "cache_states": sorted(samples),
            "cache_definition": "cold runs use a private cache directory per sample and a fresh run; warm runs reuse one directory per arm",
            "arms": arm_definitions or {},
            "distinct_arms": distinct_arm_definitions(arm_definitions),
        },
        "inputs": _hashed_inputs(input_paths),
        "configuration": _hashed_configs(config_paths),
        "thread_environment": dict(sorted(thread_environment.items())),
        "correctness": {"arms": dict(correctness), "equivalent": True},
        "results": results,
    }


def write_report(report: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
