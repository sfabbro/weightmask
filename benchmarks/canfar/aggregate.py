#!/usr/bin/env python3
"""Render the campaign results table from per-job metrics.json files.

Runs on CANFAR against the project mount (stdlib only) or locally on a
mirrored results dir:
  python3 benchmarks/canfar/aggregate.py [results_root] [--manifest PATH]

Writes results/table.md: one row per job with wall, efficiency, maxRSS,
stage deltas vs E0, mask-diff, and floor/gate checklist. Promotion itself
stays manual: winners need floor met + gates green
(`pixi run test`, `benchmark-megacam` suite, `streak_inject.py` recall for
streak-touching variants); then append before/after rows to
test_outputs/perf/megacam_perf_after.md and merge the variant.
"""

from __future__ import annotations

import json
import os
import sys

FLOORS = {"E2": ("hdu_total", 8.0), "E3": ("hdu_total", 8.0), "E4": ("cosmics", None)}


def load_metrics(root):
    out = {}
    for name in sorted(os.listdir(root)):
        mp = os.path.join(root, name, "metrics.json")
        if os.path.isfile(mp):
            try:
                out[name] = json.load(open(mp))
            except Exception as e:
                out[name] = {"_error": str(e)[:160]}
    return out


def fmt(x, digits=1, scale=1.0):
    if x is None:
        return "-"
    try:
        return f"{float(x) * scale:.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def stage_mean(metrics, stage):
    try:
        return float(metrics["per_stage"][stage]["mean_per_hdu_s"])
    except (KeyError, TypeError, ValueError):
        return None


def main(argv=None):
    argv = list(argv or sys.argv[1:])
    root = argv[0] if argv else os.path.join(
        os.environ.get("PROJECT_MOUNT", "/arc/projects/mlao/cfhtcast"),
        "weightmask-perf",
        "results",
    )

    metrics = load_metrics(root)
    e0 = metrics.get("E0-w8", {})
    e0_hdu = stage_mean(e0, "hdu_total")
    e0_cosmics = stage_mean(e0, "cosmics")
    e0_streak = None
    try:
        e0_streak = float(e0["per_stage"]["streaks"]["total_s"])
    except (KeyError, TypeError, ValueError):
        pass

    lines = ["# weightmask CANFAR speed campaign", ""]
    if e0 and not e0.get("_error"):
        lines.append(
            f"Control E0-w8: wall={fmt(e0.get('wall_s'))}s eff={fmt(e0.get('parallel_efficiency'), 3)} "
            f"maxRSS={fmt(e0.get('max_rss_kb'), 0, 1 / 1024)}MB mpix/s={fmt(e0.get('mpix_s'), 2)} "
            f"streak_share={fmt((e0.get('per_stage', {}).get('streaks') or {}).get('share'), 3)}"
        )
        if e0_streak:
            lines.append(f"Streak prize bound on control data: {e0_streak:.0f}s total.")
        lines.append("")

    lines += [
        "| group | wall_s | eff | maxRSS_MB | mpix_s | d_hdu_s | floor | mask_diff | gates |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    verdicts = []
    for name in sorted(metrics):
        m = metrics[name]
        if m.get("_error"):
            lines.append(f"| {name} | ERROR | - | - | - | - | - | {m['_error']} | - |")
            continue
        exp = (m.get("exp_id") or name.split("-")[0])
        wall = fmt(m.get("wall_s"))
        eff = fmt(m.get("parallel_efficiency"), 3)
        rss = fmt(m.get("max_rss_kb"), 0, 1 / 1024)
        mps = fmt(m.get("mpix_s"), 2)
        hdu = stage_mean(m, "hdu_total")
        dhdu = (e0_hdu - hdu) if (e0_hdu is not None and hdu is not None) else None
        floor, check = "-", "-"
        spec = FLOORS.get(exp)
        if spec:
            stage, need = spec
            if need is None:  # E4: halve cosmics mean/HDU vs control
                got = stage_mean(m, stage)
                if got is not None and e0_cosmics:
                    check = "MET" if got <= 0.5 * e0_cosmics else "miss"
                    floor = f"cosmics<={0.5 * e0_cosmics:.1f}s/HDU"
            elif dhdu is not None:
                check = "MET" if dhdu >= need else "miss"
                floor = f"-{need:.0f}s/HDU"
        diff = m.get("mask_diff", {})
        dmode = diff.get("mode", "-")
        dsum = dmode
        if dmode == "pixel":
            dsum = f"kept={diff.get('kept')} lost={diff.get('lost')} gained={diff.get('gained')}"
        elif dmode in ("fraction", "fraction-self"):
            fr = diff.get("exp_fracs") or {"exp": diff.get("exp_frac"), "ctrl": diff.get("control_frac")}
            dsum = "frac:" + ",".join(f"{k}={fmt(v, 3)}" for k, v in fr.items() if v is not None)
        lines.append(f"| {name} | {wall} | {eff} | {rss} | {mps} | {fmt(dhdu, 1)} | {floor}:{check} | {dsum} | TBD |")
        if check == "MET":
            verdicts.append(f"- {name}: floor {floor} MET — run gates, then promote.")

    # E5 twin identity from checksums (masks are dropped for non-kept jobs).
    twins = [n for n in metrics if n.startswith("E5-")]
    if len(twins) >= 2 and all(not metrics[t].get("_error") for t in twins):
        sums = [metrics[t].get("mask_checksums", {}) for t in twins]
        vals = [sorted(s.values()) for s in sums]
        same = len(vals[0]) > 0 and all(v == vals[0] for v in vals[1:])
        verdicts.append(f"- E5 twins mask-identical: {same} ({', '.join(twins)}).")
        if not same:
            verdicts.append("  WARNING: thread setting changed masks — do not promote E5.")

    e1 = metrics.get("E1-w8", {})
    if e1 and not e1.get("_error"):
        try:
            st = float(e1["per_stage"]["streaks"]["total_s"])
            verdicts.append(f"- E1 streak stage total with masking off: {st:.1f}s (expect ~0).")
        except (KeyError, TypeError, ValueError):
            pass

    lines += ["", "## Promotion checklist", ""]
    lines += verdicts or ["- No jobs collected yet."]
    lines += [
        "",
        "Gates per promoted winner: `pixi run test`, `python -m tests.benchmarks.run --suite megacam_real`,",
        "`benchmarks/streak_inject.py` recall >= E0 (streak-touching variants only).",
        "Promote by appending before/after rows to `test_outputs/perf/megacam_perf_after.md`",
        "and merging the variant branch into the main line.",
    ]
    table = os.path.join(root, "table.md")
    with open(table, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Wrote {table} ({len(metrics)} job(s)).")
    print("\n".join(verdicts or ["- No jobs collected yet."]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
