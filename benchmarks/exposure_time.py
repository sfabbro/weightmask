#!/usr/bin/env python3
"""What does one full MegaCam exposure cost, with and without streak detection?

Every per-stage number so far has been the *streak stage* alone, which is the
wrong unit for anyone deciding whether to run it: a survey pays per exposure, and
an exposure is 36 chips. This measures the exposure.

Runs the production CLI over all 36 science extensions of one MEF, once with
``streak_masking.enable: true`` and once with ``enable: false``, and reports wall
time for each plus the difference. Both arms write their own outputs, so neither
short-circuits a stage.

Three measurement hazards, each of which produced a wrong answer before it was
handled here:

* **Order.** The first run of either arm pays for importing skimage/scipy,
  populating the on-disk flat-bad-pixel cache, and faulting the 350 MB flat into
  the page cache. Measuring the arms in sequence and reporting the first one made
  streaks look 2.6x *slower*. Both arms are now warmed up and discarded first, and
  each arm is timed ``--repeats`` times with the fastest reported.
* **Profiling in the wrong process.** A ``cProfile`` in the parent measures
  ``subprocess.run`` waiting on the child, which attributes the entire run to
  ``select.poll``. Nothing is profiled here: the pipeline already prints a
  per-stage timing line per chip (``--- Image processed in N seconds ---
  top=cosmics:Xs streaks:Ys``), and that is parsed.
* **Two copies of the same run.** If neither arm reported a streak pixel the
  comparison would be meaningless, so the total is checked and called out.

Wall time is wall time: single-process, single-thread (``--nproc 1``), so these
are what one core does.

Usage:
    pixi run exposure-time
    pixi run exposure-time -- --exposure benchmark_data/megacam/perf/1013719p.fits.fz
    pixi run exposure-time -- --hdus 1 2 --repeats 1     # quick smoke run
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

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLAT = os.path.join(REPO, "benchmark_data", "megacam", "perf", "flat_08Bm01_r.fits.fz")

CHIP_TIMING = re.compile(r"--- Image processed in ([\d.]+) seconds --- top=(.*)")
STREAK_PIXELS = re.compile(r"Final streak mask includes ([\d,]+) new pixels")


def make_config(outdir, enable_streaks):
    """Toggle the stage by rewriting the one line.

    The CLI then does exactly what a user would do, rather than taking a code path
    only this benchmark can reach. The assert is not decoration: a silent failure
    to rewrite would leave both arms identical and the comparison meaningless.
    """
    with open(os.path.join(REPO, "weightmask.yml")) as handle:
        text = handle.read()
    if not enable_streaks:
        text = text.replace("streak_masking:\n  enable: true", "streak_masking:\n  enable: false", 1)
        assert "streak_masking:\n  enable: false" in text, "failed to disable streak_masking"
    path = os.path.join(outdir, f"config_{'on' if enable_streaks else 'off'}.yml")
    with open(path, "w") as handle:
        handle.write(text)
    return path


def run_once(exposure, enable_streaks, hdus, outdir):
    output = os.path.join(outdir, "on" if enable_streaks else "off")
    os.makedirs(output, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "weightmask.cli",
        exposure,
        "--flat_image",
        FLAT,
        "--config",
        make_config(outdir, enable_streaks),
        "--output_mask",
        os.path.join(output, "mask.fits"),
        "--output_map",
        os.path.join(output, "weight.fits"),
        "--nproc",
        "1",
    ]
    for hdu in hdus or []:
        command += ["--hdu", str(hdu)]

    started = time.perf_counter()
    completed = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
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
    parser.add_argument("--hdus", type=int, nargs="*", help="process only these HDU indices")
    parser.add_argument("--repeats", type=int, default=2, help="timed runs per arm; fastest reported")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args(argv)

    chips = len(args.hdus) if args.hdus else 36
    outdir = tempfile.mkdtemp(prefix="wm_exposure_time_")
    try:
        print(f"exposure: {os.path.relpath(args.exposure, REPO)}")
        print(f"chips:    {chips} x 9.8 Mpix = {chips * 9.8:.0f} Mpix")
        print(f"flat:     {os.path.relpath(FLAT, REPO)}")
        print(f"timing:   {args.repeats} run(s) per arm after a discarded warmup, fastest reported\n", flush=True)

        print("warmup (discarded)...", flush=True)
        for enable in (True, False):
            run_once(args.exposure, enable, args.hdus, outdir)
        print(flush=True)

        results = {}
        for label, enable in (("streaks OFF", False), ("streaks ON", True)):
            runs = []
            for _ in range(args.repeats):
                elapsed, stdout = run_once(args.exposure, enable, args.hdus, outdir)
                runs.append((elapsed, parse(stdout)))
            best = min(runs, key=lambda r: r[0])
            results[label] = best
            pixels = best[1][3]
            print(
                f"  {label}: {best[0]:7.1f}s  ({best[0] / chips:.2f} s/chip)  "
                f"chips timed {len(best[1][0])}  streak px {pixels:,.0f}  "
                f"all {['%.1f' % r[0] for r in runs]}",
                flush=True,
            )

        off_t, (off_chips, off_stages, off_stage_chips, off_px) = results["streaks OFF"]
        on_t, (on_chips, on_stages, on_stage_chips, on_px) = results["streaks ON"]
        delta = on_t - off_t
        print("\n=== the number ===")
        print(
            f"  streaks OFF   {off_t:8.1f}s   {off_t / chips:6.2f} s/chip"
            f"   (median chip {statistics.median(off_chips):5.2f}s)"
        )
        print(
            f"  streaks ON    {on_t:8.1f}s   {on_t / chips:6.2f} s/chip"
            f"   (median chip {statistics.median(on_chips):5.2f}s)"
        )
        print(f"  difference    {delta:+8.1f}s   = {100 * delta / off_t:+.1f}% of the streak-free run")
        print(f"  the full run is {off_t / on_t:.2f}x faster without streak detection")
        print(f"  streak stage's share of the full run: {100 * delta / on_t:.0f}%")

        if off_px == 0.0 and on_px == 0.0:
            print("\n  WARNING: both arms reported zero streak pixels, so the difference above")
            print("  is two copies of the same run rather than a measurement of the stage.")
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
    finally:
        if not args.keep:
            shutil.rmtree(outdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
