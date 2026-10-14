#!/usr/bin/env python3
"""What does one full MegaCam exposure cost, with and without streak detection?

Every per-stage number so far has been the *streak stage* alone, which is the
wrong unit for anyone deciding whether to run it: a survey pays per exposure, and
an exposure is usually 36 chips. This measures the exposure.

Runs the production CLI over all 2-D science HDUs of one MEF, once with
``streak_masking.enable: true`` and once with ``enable: false``, and reports wall
time for each plus the difference. Both arms write their own outputs, so neither
short-circuits a stage.

Three measurement hazards, each of which produced a wrong answer before it was
handled here:

* **Order.** The first run of either arm pays for importing skimage/scipy,
  populating the on-disk flat-bad-pixel cache, and faulting the 350 MB flat into
  the page cache. Measuring the arms in sequence and reporting the first one made
  streaks look 2.6x *slower*. Arm order is randomized and timed runs are split
  into cold-cache and warm-cache observations.
* **Profiling in the wrong process.** A ``cProfile`` in the parent measures
  ``subprocess.run`` waiting on the child, which attributes the entire run to
  ``select.poll``. Nothing is profiled here: the pipeline already prints a
  per-stage timing line per chip (``--- Image processed in N seconds ---
  top=cosmics:Xs streaks:Ys``), and that is parsed.
* **Zero detections.** A streak search can cost time without finding a trail.
  Zero detections measure that search cost, but do not establish recall.

Wall time is wall time. The default comparison runs ``--nproc 1`` so the number is
one core's cost and the stage attribution stays readable; ``--scaling`` sweeps
``--nproc`` for what a parallel survey run actually costs.

Usage:
    pixi run exposure-time
    pixi run exposure-time -- --exposure benchmark_data/megacam/perf/1013719p.fits.fz
    pixi run exposure-time -- --repeats 5                # five repeats per arm
    pixi run exposure-time -- --scaling 1 2 4 8         # how it uses the cores
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import fitsio

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from benchmarks import perf_protocol

FLAT = os.path.join(REPO, "benchmark_data", "megacam", "perf", "flat_08Bm01_r.fits.fz")

CHIP_TIMING = re.compile(r"--- Image processed in ([\d.]+) seconds --- top=(.*)")
STREAK_PIXELS = re.compile(r"Final streak mask includes ([\d,]+) new pixels")


def make_config(outdir, enable_streaks, *, config_id=None, cache_dir=None):
    """Toggle the stage by rewriting the one line.

    The CLI then does exactly what a user would do, rather than taking a code path
    only this benchmark can reach. The assert is not decoration: a silent failure
    to rewrite would leave both arms identical and the comparison meaningless.
    """
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(REPO, "weightmask.yml")) as handle:
        text = handle.read()
    if not enable_streaks:
        text = text.replace("streak_masking:\n  enable: true", "streak_masking:\n  enable: false", 1)
        assert "streak_masking:\n  enable: false" in text, "failed to disable streak_masking"
    if cache_dir is not None:
        text = re.sub(
            r"(?m)^(\s*bad_mask_cache_dir:)\s*.*$",
            rf'\1 "{Path(cache_dir)}"',
            text,
            count=1,
        )
    suffix = f"_{config_id}" if config_id else ""
    path = os.path.join(outdir, f"config_{'on' if enable_streaks else 'off'}{suffix}.yml")
    with open(path, "w") as handle:
        handle.write(text)
    return path


def run_once(exposure, enable_streaks, outdir, nproc=1, run_id=None, environment=None, cache_dir=None):
    output = os.path.join(outdir, run_id or "run", "on" if enable_streaks else "off")
    os.makedirs(output, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "weightmask.cli",
        exposure,
        "--flat_image",
        FLAT,
        "--config",
        make_config(outdir, enable_streaks, config_id=run_id, cache_dir=cache_dir),
        "--output_mask",
        os.path.join(output, "mask.fits"),
        "--output_map",
        os.path.join(output, "weight.fits"),
        "--nproc",
        str(nproc),
    ]
    # --hdu takes a single index and repeated uses overwrite rather than accumulate,
    # so a multi-chip subset is not expressible as flags. Pass nothing: the
    # default is every 2-D science HDU.
    # An earlier version appended one --hdu per chip and silently measured a
    # single-chip run while reporting it as eight.
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=REPO, capture_output=True, text=True, env=environment)
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(f"CLI failed (exit {completed.returncode}):\n{completed.stderr[-3000:]}")
    return elapsed, completed.stdout


def parse(stdout):
    """Per-chip wall time, per-stage totals, streak pixels.

    The pipeline's timing line prints only the top 3 stages *per chip*, so a stage
    that is not in a given chip's top 3 is simply absent from that chip's line. A
    stage total summed from these lines is therefore a lower bound, and a stage
    that only appears in one chip's top 3 looks like it ran on one chip when it
    ran on all of them. Count how many chips reported each stage and say so, rather
    than presenting the sum as if it were complete.
    """
    chip_seconds = []
    stages = {}
    stage_chips = {}
    for match in CHIP_TIMING.finditer(stdout):
        chip_seconds.append(float(match.group(1)))
        for stage in match.group(2).split():
            if ":" not in stage:
                continue
            name, value = stage.rsplit(":", 1)
            stages[name] = stages.get(name, 0.0) + float(value.rstrip("s"))
            stage_chips[name] = stage_chips.get(name, 0) + 1
    pixels = sum(float(m.group(1).replace(",", "")) for m in STREAK_PIXELS.finditer(stdout))
    return chip_seconds, stages, stage_chips, pixels


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--exposure",
        default=os.path.join(REPO, "benchmark_data", "megacam", "long", "996195p.fits.fz"),
    )
    parser.add_argument("--repeats", type=int, default=5, help="timed runs per arm; minimum 5")
    parser.add_argument("--seed", type=int, default=0, help="seed for the deterministic A/B schedule")
    parser.add_argument("--report", help="write the deterministic JSON timing report here")
    parser.add_argument(
        "--scaling",
        type=int,
        nargs="*",
        help="instead of the on/off comparison, sweep --nproc over these values with streaks on. "
        "Reports wall time, speedup and parallel efficiency.",
    )
    parser.add_argument(
        "--hdus",
        type=int,
        nargs="*",
        help=argparse.SUPPRESS,  # rejected in main(); kept only to give a clear message
    )
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args(argv)

    try:
        perf_protocol.validate_repeats(args.repeats)
    except ValueError as error:
        parser.error(str(error))
    if args.scaling is not None and (not args.scaling or args.scaling[0] != 1 or min(args.scaling) < 1):
        parser.error("--scaling must start with 1 and contain only positive worker counts")
    if args.hdus:
        parser.error(
            "--hdus is not supported. --hdu takes a single index and repeated uses overwrite, "
            "and a renumbered subset MEF pairs chip 0 with a flat that has no HDU 0 -- the "
            "pipeline then fails that chip. Time the whole exposure instead."
        )
    with fitsio.FITS(args.exposure) as hdus:
        infos = [hdu.get_info() for hdu in hdus]
    shapes = [info["dims"] for info in infos if info.get("hdutype") == 0 and info.get("ndims") == 2]
    chips = len(shapes)
    if not chips:
        parser.error("exposure contains no 2-D science HDUs")
    pixels = sum(height * width for height, width in shapes)
    outdir = tempfile.mkdtemp(prefix="wm_exposure_time_")
    try:
        print(f"exposure: {os.path.relpath(args.exposure, REPO)}")
        print(f"chips:    {chips} ({pixels / 1e6:.1f} Mpix total)")
        print(f"flat:     {os.path.relpath(FLAT, REPO)}")
        print(f"cores:    {os.cpu_count()} logical\n", flush=True)

        exposure = args.exposure

        if args.scaling:
            return scaling(args, exposure, outdir, chips)
        comparison(args, exposure, outdir, chips, report_path=args.report)
    finally:
        if not args.keep:
            shutil.rmtree(outdir, ignore_errors=True)
    return 0


def scaling(args, exposure, outdir, chips):
    """Wall time against --nproc, with streaks on. Survey runs are not single-core.

    Efficiency is wall_1 / (wall_n * n), not wall_1 / wall_n: the first is how much
    of the machine the run actually used, the second only says it got faster. A
    number that scales sub-linearly can reflect serial work or resource
    contention; this table does not distinguish the causes.
    """
    values = args.scaling
    if not values or values[0] != 1 or min(values) < 1:
        raise ValueError("scaling must start with 1 and contain only positive worker counts")
    perf_protocol.validate_repeats(args.repeats)
    print("warmup (discarded)...", flush=True)
    run_once(exposure, True, outdir, nproc=values[0])
    print(flush=True)

    rows = []
    for nproc in values:
        times = []
        for repeat in range(args.repeats):
            elapsed, _stdout = run_once(exposure, True, outdir, nproc=nproc, run_id=f"scale-{nproc}-{repeat}")
            times.append(elapsed)
        summary = perf_protocol.summarize(times)
        rows.append((nproc, summary))
        print(
            f"  nproc={nproc:<3d} median={summary['median']:7.1f}s "
            f"IQR={summary['iqr']:.1f}s MAD={summary['mad']:.1f}s "
            f"({['%.1f' % t for t in times]})",
            flush=True,
        )

    base = rows[0][1]["median"]
    print("\n=== scaling, streaks ON ===")
    print(f"  {'nproc':>6} {'median':>9} {'speedup':>9} {'efficiency':>11} {'s/chip':>9}")
    for nproc, summary in rows:
        wall = summary["median"]
        speedup = base / wall
        efficiency = base / (wall * nproc)
        print(f"  {nproc:6d} {wall:8.1f}s {speedup:8.2f}x {100 * efficiency:10.0f}% {wall / chips:8.2f}s")
    if len(rows) > 1:
        marginal = (rows[-2][1]["median"] - rows[-1][1]["median"]) / (rows[-1][1]["median"] or 1) * 100
        print(f"  last step bought {marginal:.0f}% -- near-zero means more workers did not help")
    return 0


def _arm_output_equivalent(output_dirs):
    from benchmarks import perf_megacam

    if len(output_dirs) < 2:
        return True
    reference = output_dirs[0]
    return all(perf_megacam._compare_products(path, reference)["ok"] for path in output_dirs[1:])


def comparison(args, exposure, outdir, chips, report_path=None):
    repeats = perf_protocol.validate_repeats(args.repeats)
    seed = int(getattr(args, "seed", 0))
    schedule = perf_protocol.interleaved_schedule(
        repeats,
        seed=seed,
        arms=("baseline", "treatment"),
        cold_repetitions=1,
    )
    environment = perf_protocol.thread_environment(os.environ)
    process_environment = os.environ.copy()
    process_environment.update(environment)
    print(f"timing:   {repeats} run(s) per arm; randomized seed={seed}; median/IQR/MAD reported\n", flush=True)

    samples = {"cold": {"baseline": [], "treatment": []}, "warm": {"baseline": [], "treatment": []}}
    parsed = {"baseline": [], "treatment": []}
    output_dirs = {"baseline": [], "treatment": []}
    config_paths = []
    warm_cache_root = Path(outdir) / "cache"
    for arm in ("baseline", "treatment"):
        warm_cache = perf_protocol.cache_directory(warm_cache_root, "warm", arm, 0)
        prewarm_id = f"prewarm-{arm}"
        run_once(
            exposure,
            arm == "treatment",
            outdir,
            environment=process_environment,
            run_id=prewarm_id,
            cache_dir=warm_cache,
        )
        shutil.rmtree(Path(outdir) / prewarm_id, ignore_errors=True)
    for repetition, arm in schedule:
        enable = arm == "treatment"
        run_id = f"timed-{repetition}-{arm}"
        cache = "cold" if not parsed[arm] else "warm"
        cache_dir = perf_protocol.cache_directory(warm_cache_root, cache, arm, repetition)
        try:
            elapsed, stdout = run_once(
                exposure,
                enable,
                outdir,
                environment=process_environment,
                run_id=run_id,
                cache_dir=cache_dir,
            )
        finally:
            if cache == "cold":
                perf_protocol.cleanup_cache_directory(cache_dir)
        samples[cache][arm].append(elapsed)
        parsed[arm].append((repetition, elapsed, parse(stdout)))
        output_dirs[arm].append(os.path.join(outdir, run_id, "on" if enable else "off"))
        config_paths.append(make_config(outdir, enable, config_id=run_id, cache_dir=cache_dir))

    correctness = {arm: _arm_output_equivalent(paths) for arm, paths in output_dirs.items()}
    perf_protocol.require_correctness_equivalence(correctness)
    report = perf_protocol.make_report(
        instrument="MegaCam exposure-time",
        input_paths=[exposure, FLAT],
        config_paths=sorted(set(config_paths)),
        repeats=repeats,
        seed=seed,
        schedule=schedule,
        samples=samples,
        correctness=correctness,
        thread_environment=environment,
        cold_repetitions=1,
        arm_definitions={
            "baseline": {"streak_masking.enable": False},
            "treatment": {"streak_masking.enable": True},
        },
    )
    selected = {}
    for arm in ("baseline", "treatment"):
        warm_runs = parsed[arm][1:]
        target = report["results"]["warm"][arm]["median"]
        selected[arm] = min(warm_runs, key=lambda item: abs(item[1] - target))
    report["protocol"]["selected"] = {
        arm: {"arm": arm, "cache": "warm", "repetition": run[0]} for arm, run in selected.items()
    }
    report["protocol"]["prewarmed"] = {arm: True for arm in ("baseline", "treatment")}
    report["protocol"]["eligibility"] = perf_protocol.release_eligibility(
        repeats,
        all(correctness.values()),
        False,
        full_products=False,
        distinct_arms=True,
    )
    if report_path:
        perf_protocol.write_report(report, report_path)
    off_t = report["results"]["warm"]["baseline"]["median"]
    on_t = report["results"]["warm"]["treatment"]["median"]
    off_chips, off_stages, off_stage_chips, off_px = selected["baseline"][2]
    on_chips, on_stages, on_stage_chips, on_px = selected["treatment"][2]
    delta = on_t - off_t
    print("\n=== the number ===")
    print(
        f"  streaks OFF   {off_t:8.1f}s   {off_t / chips:6.2f} s/chip"
        f"   (median chip {statistics.median(off_chips):5.2f}s; "
        f"IQR {report['results']['warm']['baseline']['iqr']:.1f}s; MAD {report['results']['warm']['baseline']['mad']:.1f}s)"
    )
    print(
        f"  streaks ON    {on_t:8.1f}s   {on_t / chips:6.2f} s/chip"
        f"   (median chip {statistics.median(on_chips):5.2f}s; "
        f"IQR {report['results']['warm']['treatment']['iqr']:.1f}s; MAD {report['results']['warm']['treatment']['mad']:.1f}s)"
    )
    print(f"  difference    {delta:+8.1f}s   = {100 * delta / off_t:+.1f}% of the streak-free run")
    print(f"  streak stage's share of the full run: {100 * delta / on_t:.0f}%")

    if off_px == 0.0 and on_px == 0.0:
        print("\n  NOTE: both arms reported zero streak pixels. The difference measures")
        print("  the search cost on this exposure, without evidence of trail recovery.")
    elif off_px:
        print(f"\n  NOTE: the streaks-OFF arm still reported {off_px:,.0f} px, so the stage")
        print("  was not fully off. Check the config rewrite.")

    print("\n=== per-stage totals, streaks ON ===")
    print("  the pipeline prints only each chip's top 3 stages, so a stage absent from a")
    print("  chip's line did not run slowly there -- it just was not in the top 3")
    for name, value in sorted(on_stages.items(), key=lambda kv: -kv[1]):
        print(
            f"  {name:12s} {value:7.1f}s  {100 * value / on_t:4.1f}% of run  "
            f"[top-3 on {on_stage_chips[name]}/{len(on_chips)} chips]"
        )
    print("\n=== per-stage totals, streaks OFF ===")
    for name, value in sorted(off_stages.items(), key=lambda kv: -kv[1]):
        print(
            f"  {name:12s} {value:7.1f}s  {100 * value / off_t:4.1f}% of run  "
            f"[top-3 on {off_stage_chips[name]}/{len(on_chips)} chips]"
        )
    print("\n  Quote the end-to-end difference. These sums are a top-3 sketch and")
    print("  undercount any stage that is usually 4th.")

    if args.keep:
        print(f"\noutputs kept in {outdir}")
    return report



if __name__ == "__main__":
    sys.exit(main())
