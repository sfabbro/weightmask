"""Measure the star-trail veto's length window, in family-length multiples.

Why this exists
---------------
``streak_masking.star_trail_veto`` drops a component that is parallel to the
frame's own star trails *and* no longer than ``length_factor`` times the family's
median length. That second condition was designed rather than measured: the
config comment said raising it "trades these false positives for that
sensitivity", and nothing in the project put a number on either side of the
trade. This harness measures both sides on the same frame, in the units the veto
itself compares.

Two object classes are swept, both at the family's own angle, because that is
where the length condition is the only thing left deciding:

  satellite  a uniform, PSF-width trail -- a satellite whose apparent motion is
             parallel to the field's. Dropping one of these is the cost.
  chain      the field's own star trails placed collinearly with gaps -- the
             campaign's dominant false positive. Keeping one is the cost.

The family is estimated from the frame through the production input path
(``capture_detector_inputs``), not from a fixture's nominal numbers, because the
statistic biases low: a 55 px rendered trail measures 47 px as a family member,
and the veto compares that number against itself. Every ratio below is therefore
in *measured* family lengths, with the geometric length alongside it.

What it measures, and what it does not
--------------------------------------
Every cell is also far above the size floor: the smallest injected body is 93 px
against the shipped ``min_mask_pixels`` of 24, so that gate never decides a cell
here and the numbers below are the length and angle conditions alone.

Each cell hands the veto a component of known geometry and records the decision,
so this is the veto's boundary and not the stage's recall. Recall on a
non-sidereal frame is the acceptance harness's job
(``benchmarks/streak_nonsidereal_acceptance.py``), whose injected cells are set
at 10 to 90 degrees off the family angle -- deliberately outside the veto's
reach, which is why nothing there moved when the veto was enabled. The one
stage-level cell here (``--stage``) runs the campaign's gapped-chain geometry
through ``detect_streaks`` with the veto on and off, which is what turns the
per-component boundary into a footprint.

Usage
-----
    {py} benchmarks/star_trail_veto_window.py [options]

    --size N                 frame side in pixels (default 1024; the family
                             estimator needs the room -- at 512 the fixture
                             yields no family at all)
    --star-trail-length N    the field's star-trail length (default 55 px,
                             measured on 1851960 HDU 17)
    --trail-angle DEG        the field's star-trail angle (default 40.2 deg)
    --stars N                background trailed stars in the fixture
    --flat PATH              flat frame (primary HDU), as production would have
                             it; without one the no-flat path masks a lot of the
                             frame
    --no-stage               skip the one stage-level cell (the function sweep
                             is the rest of the run)
    --set KEY=VALUE          config override, e.g. --set streak_masking.star_trail_veto.length_factor=2.0
    --out PATH               write the JSON report here
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
try:
    from production_inputs import capture_detector_inputs
    from streak_inject import apply_overrides
    from streak_nonsidereal_acceptance import render_chain, render_trail, synthetic_frame
except ImportError:  # pragma: no cover - the package layout
    from benchmarks.production_inputs import capture_detector_inputs  # type: ignore[no-redef]
    from benchmarks.streak_inject import apply_overrides  # type: ignore[no-redef]
    from benchmarks.streak_nonsidereal_acceptance import (  # type: ignore[no-redef]
        render_chain,
        render_trail,
        synthetic_frame,
    )

from weightmask.streaks import (  # noqa: E402
    _angular_distance_deg,
    _component_geometry,
    _drop_star_trail_family_components,
    _label_components,
    detect_streaks,
    frame_trail_family,
)

PARALLEL_MULTIPLES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 6.0, 8.0)
ANGLE_OFFSETS = (0.0, 5.0, 9.0, 11.0, 15.0, 30.0, 90.0)
ANGLE_OFFSET_MULTIPLE = 2.0
CHAIN_GAP_FACTORS = (1.0, 1.6, 2.5)
CHAIN_SEGMENTS = (2, 3, 4, 6, 8, 12)
CONTAMINATION_MULTIPLES = (2.0, 4.0, 8.0)
# The campaign geometry for the stage cell: several star trails collinear with
# gaps, which is what a line fitted through their heads looks like on the frame.
STAGE_CHAIN_SEGMENTS = 8
STAGE_CHAIN_GAP_FACTOR = 1.6


def _component_of(body, angle_deg):
    """Span, the veto's length statistics and the orientation of a body's components.

    A satellite is one component. A chain is several, and the length that decides
    its fate is then the median across them: the veto judges each component on
    its own, so a per-segment length is what it acts on, while the span reported
    here is the whole chain's extent.
    """
    components = _label_components(body)
    if components is None:
        return None
    labeled, n_components, _sizes = components
    geometry = _component_geometry(labeled, n_components)
    lengths = np.asarray(geometry["length_px"][1:], dtype=np.float64)
    ys, xs = np.nonzero(body)
    theta = np.radians(float(angle_deg))
    along = (xs - xs.mean()) * np.cos(theta) + (ys - ys.mean()) * np.sin(theta)
    return {
        "span_px": float(along.max() - along.min()),
        "length_px": float(np.median(lengths)),
        "largest_length_px": float(lengths.max()),
        "angle_deg": float(geometry["angle_deg"][1]),
        "size": int(geometry["size"][1]),
        "components": int(n_components),
    }


def probe(body, family, veto_cfg, *, angle_deg, kind, label, length=None, gap_factor=None, segments=None):
    """One cell: what the veto does to a component of known geometry."""
    geometry = _component_of(body, angle_deg)
    if geometry is None:
        raise ValueError("an empty body has nothing to probe")
    kept, dropped = _drop_star_trail_family_components(body, family, veto_cfg, min_pixels=24)
    family_length = float(family["length_px"])
    row = {
        "kind": kind,
        "label": label,
        "geometric_length_px": None if length is None else float(length),
        "geometric_multiple": None if length is None else round(float(length) / family_length, 2),
        "gap_factor": gap_factor,
        "segments": segments,
        "span_px": round(geometry["span_px"], 1),
        "span_multiple": round(geometry["span_px"] / family_length, 2),
        "length_px": round(geometry["length_px"], 1),
        "length_multiple": round(geometry["length_px"] / family_length, 2),
        "largest_length_px": round(geometry["largest_length_px"], 1),
        "largest_length_multiple": round(geometry["largest_length_px"] / family_length, 2),
        "size_px": geometry["size"],
        "components": geometry["components"],
        "angle_offset_deg": round(abs(float(_angular_distance_deg(geometry["angle_deg"], family["angle_deg"]))), 2),
        "dropped_components": int(dropped["components"]),
        "dropped_px": int(dropped["px"]),
        "dropped": bool(dropped["components"] > 0),
        "kept_px": int(np.count_nonzero(kept)),
    }
    return row


def parallel_cells(shape, *, angle_deg, trail_length, family, veto_cfg, rms_map):
    """A satellite flying parallel to the field, swept in star-trail lengths."""
    rows = []
    for multiple in PARALLEL_MULTIPLES:
        _flux, body = render_trail(
            shape,
            angle_deg=angle_deg,
            length=multiple * trail_length,
            cx=0.5 * shape[1],
            cy=0.5 * shape[0],
            peak_sig=8.0,
            rms_map=rms_map,
        )
        rows.append(
            probe(
                body,
                family,
                veto_cfg,
                angle_deg=angle_deg,
                kind="satellite-parallel",
                label=f"parallel {multiple:.1f}x",
                length=multiple * trail_length,
            )
        )
    return rows


def angle_cells(shape, *, angle_deg, trail_length, family, veto_cfg, rms_map):
    """The same length at increasing offsets, to show which condition binds where."""
    rows = []
    length = ANGLE_OFFSET_MULTIPLE * trail_length
    for offset in ANGLE_OFFSETS:
        object_angle = (angle_deg + offset) % 180.0
        _flux, body = render_trail(
            shape,
            angle_deg=object_angle,
            length=length,
            cx=0.5 * shape[1],
            cy=0.5 * shape[0],
            peak_sig=8.0,
            rms_map=rms_map,
        )
        row = probe(
            body,
            family,
            veto_cfg,
            angle_deg=object_angle,
            kind="satellite-offset",
            label=f"offset {offset:.1f} deg",
            length=length,
        )
        row["offset_deg"] = offset
        rows.append(row)
    return rows


def chain_cells(shape, *, angle_deg, trail_length, family, veto_cfg, rms_map):
    """The field's own trails, collinear with gaps, swept in span."""
    rows = []
    for gap_factor in CHAIN_GAP_FACTORS:
        for segments in CHAIN_SEGMENTS:
            step = trail_length * gap_factor
            chain = {
                "kind": "chain",
                "angle": angle_deg,
                "length": trail_length,
                "segments": [
                    (
                        0.5 * shape[1] + (index - (segments - 1) / 2.0) * step * np.cos(np.radians(angle_deg)),
                        0.5 * shape[0] + (index - (segments - 1) / 2.0) * step * np.sin(np.radians(angle_deg)),
                    )
                    for index in range(segments)
                ],
                "count": segments,
                "rng": np.random.default_rng(0),
            }
            _flux, body = render_chain(shape, chain, rms_map, np.random.default_rng(1))
            rows.append(
                probe(
                    body,
                    family,
                    veto_cfg,
                    angle_deg=angle_deg,
                    kind="chain",
                    label=f"chain {segments} seg, gap {gap_factor:.1f}l",
                    length=(segments - 1) * step + trail_length,
                    gap_factor=gap_factor,
                    segments=segments,
                )
            )
    return rows


def contamination_cells(shape, *, angle_deg, trail_length, frame, header, config, veto_cfg, rms_map, flat):
    """Does the parallel trail the veto is judging move the family it is judged against?"""
    rows = []
    for multiple in CONTAMINATION_MULTIPLES:
        flux, _body = render_trail(
            shape,
            angle_deg=angle_deg,
            length=multiple * trail_length,
            cx=0.5 * shape[1],
            cy=0.5 * shape[0],
            peak_sig=8.0,
            rms_map=rms_map,
        )
        data_sub, rms, existing = capture_detector_inputs(frame + flux, header, config, flat=flat)
        family = frame_trail_family(data_sub, rms, existing, veto_cfg)
        rows.append(
            {
                "injected_multiple": multiple,
                "family": None
                if family is None
                else {
                    "angle_deg": family["angle_deg"],
                    "length_px": family["length_px"],
                    "count": family["count"],
                    "population": family["population"],
                },
            }
        )
    return rows


def stage_cell(shape, *, angle_deg, trail_length, frame, header, config, streak_cfg, rms_map, flat):
    """The campaign's chain geometry through the stage, with the veto on and off.

    A per-component boundary is only half the answer: what a false positive costs
    is the footprint the stage leaves on the frame, which is what the camera's
    masks are made of.
    """
    step = trail_length * STAGE_CHAIN_GAP_FACTOR
    segments = STAGE_CHAIN_SEGMENTS
    chain = {
        "kind": "chain",
        "angle": angle_deg,
        "length": trail_length,
        "segments": [
            (
                0.5 * shape[1] + (index - (segments - 1) / 2.0) * step * np.cos(np.radians(angle_deg)),
                0.5 * shape[0] + (index - (segments - 1) / 2.0) * step * np.sin(np.radians(angle_deg)),
            )
            for index in range(segments)
        ],
        "count": segments,
        "rng": np.random.default_rng(0),
    }
    flux, body = render_chain(shape, chain, rms_map, np.random.default_rng(1))
    data_sub, rms, existing = capture_detector_inputs(frame + flux, header, config, flat=flat)
    out = {
        "chain_segments": segments,
        "gap_factor": STAGE_CHAIN_GAP_FACTOR,
        "chain_span_px": round((segments - 1) * step + trail_length, 1),
        "chain_px": int(np.count_nonzero(body)),
    }
    for name, enable in (("veto_on", True), ("veto_off", False)):
        cfg = dict(streak_cfg)
        cfg["star_trail_veto"] = dict(cfg.get("star_trail_veto") or {}, enable=enable)
        t0 = time.time()
        mask = detect_streaks(data_sub, rms, existing, cfg)
        out[name] = {
            "mask_px": int(np.count_nonzero(mask)),
            "chain_px_in_mask": int(np.count_nonzero(mask & body)),
            "seconds": round(time.time() - t0, 1),
        }
    return out


def window(rows):
    """The largest multiple still dropped and the smallest one already kept."""
    parallel = [row for row in rows if row["kind"] == "satellite-parallel"]
    dropped = [row["length_multiple"] for row in parallel if row["dropped"]]
    kept = [row["length_multiple"] for row in parallel if not row["dropped"]]
    if not dropped or not kept:
        return {"dropped_through": None, "kept_from": None}
    return {"dropped_through": max(dropped), "kept_from": min(kept)}


def _parallel_dropped(shape, *, angle_deg, length, family, veto_cfg, rms_map):
    """Would the veto drop one parallel trail of this geometric length?"""
    _flux, body = render_trail(
        shape,
        angle_deg=angle_deg,
        length=float(length),
        cx=0.5 * shape[1],
        cy=0.5 * shape[0],
        peak_sig=8.0,
        rms_map=rms_map,
    )
    row = probe(
        body,
        family,
        veto_cfg,
        angle_deg=angle_deg,
        kind="satellite-parallel",
        label=f"boundary {length:.1f} px",
        length=length,
    )
    return row["dropped"]


def refine_boundary(shape, *, angle_deg, family, veto_cfg, rms_map, lower_px, upper_px, iterations=8):
    """Bisect the coarse bracket for the length where the veto's decision flips.

    The sweep below is spaced half a star-trail length apart, which brackets the
    crossing but does not locate it. The crossing is what the config claims to
    set, so it is worth resolving: eight iterations put it inside a hundredth of
    the family's length.
    """
    lower = float(lower_px)
    upper = float(upper_px)
    for _ in range(int(iterations)):
        middle = 0.5 * (lower + upper)
        if _parallel_dropped(
            shape, angle_deg=angle_deg, length=middle, family=family, veto_cfg=veto_cfg, rms_map=rms_map
        ):
            lower = middle
        else:
            upper = middle
    family_length = float(family["length_px"])
    return {
        "crossing_px": round(0.5 * (lower + upper), 2),
        "crossing_multiple": round(0.5 * (lower + upper) / family_length, 3),
        "dropped_through_px": round(lower, 2),
        "kept_from_px": round(upper, 2),
        "iterations": int(iterations),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--star-trail-length", type=float, default=55.0)
    parser.add_argument("--trail-angle", type=float, default=40.2)
    parser.add_argument("--stars", type=int, default=60)
    parser.add_argument("--sky", type=float, default=571.0)
    parser.add_argument("--rms", type=float, default=21.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flat", default="")
    parser.add_argument("--no-stage", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--out", default="")
    options = parser.parse_args(argv)

    from astropy.io import fits

    size = int(options.size)
    shape = (size, size)
    angle_deg = float(options.trail_angle)
    trail_length = float(options.star_trail_length)
    frame, _truth = synthetic_frame(
        shape,
        angle_deg=angle_deg,
        trail_length=trail_length,
        stars=options.stars,
        sky=options.sky,
        rms=options.rms,
        seed=options.seed,
    )
    flat = None
    if options.flat:
        # The primary HDU, which is what a single-chip flat carries.
        import fitsio

        flat = np.asarray(fitsio.read(options.flat, ext=0), dtype=np.float32)
    header = fits.Header({"NAXIS": 2, "SATURATE": 65535.0, "EXPTIME": 60.0, "GAIN": 1.6, "RDNOISE": 5.0})

    with open("weightmask.yml") as handle:
        config = yaml.safe_load(handle)
    config = apply_overrides(config, options.set)
    streak_cfg = dict(config["streak_masking"])
    veto_cfg = dict(streak_cfg["star_trail_veto"])
    if not veto_cfg.get("enable", False):
        raise SystemExit("the veto is disabled in this configuration; there is no window to measure")
    length_factor = float(veto_cfg.get("length_factor", 4.0))

    rms_map = np.full(shape, float(options.rms), dtype=np.float32)
    t0 = time.time()
    data_sub, production_rms, existing = capture_detector_inputs(frame, header, config, flat=flat)
    family = frame_trail_family(data_sub, production_rms, existing, veto_cfg)
    if family is None:
        raise SystemExit(
            "the fixture yielded no star-trail family, so the veto is inert on it; "
            "raise --stars or --size, or check --trail-angle"
        )
    print(f"fixture: {shape[0]}x{shape[1]}, {options.stars} trailed stars, {time.time() - t0:.1f}s", flush=True)
    print(
        f"family: {family['angle_deg']} deg, {family['length_px']} px, "
        f"{family['count']} of {family['population']} elongated sources, "
        f"spread {family['angle_spread_deg']} deg",
        flush=True,
    )
    print(
        f"a {trail_length:.0f} px rendered star trail measures {family['length_px']:.2f} px as a family member "
        f"({family['length_px'] / trail_length:.2f}x), and the veto compares that number against itself",
        flush=True,
    )
    print(f"shipped: angle_tol_deg {veto_cfg.get('angle_tol_deg')}, length_factor {length_factor}", flush=True)

    rows = []
    rows.extend(
        parallel_cells(
            shape, angle_deg=angle_deg, trail_length=trail_length, family=family, veto_cfg=veto_cfg, rms_map=rms_map
        )
    )
    rows.extend(
        angle_cells(
            shape, angle_deg=angle_deg, trail_length=trail_length, family=family, veto_cfg=veto_cfg, rms_map=rms_map
        )
    )
    rows.extend(
        chain_cells(
            shape, angle_deg=angle_deg, trail_length=trail_length, family=family, veto_cfg=veto_cfg, rms_map=rms_map
        )
    )

    for kind, title in (
        ("satellite-parallel", "a satellite parallel to the field's own trails"),
        ("satellite-offset", f"the same trail at 2.0x, rotated away from the family angle"),
    ):
        print(f"\n=== {title} ===", flush=True)
        print(f"{'cell':18s} {'geometric':>18s} {'measured':>18s} {'off':>6s}  decision", flush=True)
        for row in (item for item in rows if item["kind"] == kind):
            geometric = (
                f"{row['geometric_length_px']:.0f} px {row['geometric_multiple']:.2f}x"
                if row["geometric_length_px"] is not None
                else "-"
            )
            measured = f"{row['length_px']:.0f} px {row['length_multiple']:.2f}x"
            print(
                f"{row['label']:18s} {geometric:>18s} {measured:>18s} "
                f"{row['angle_offset_deg']:5.2f}  {'DROPPED' if row['dropped'] else 'kept'}"
                f"  ({row['dropped_px']} px in {row['dropped_components']} component(s))",
                flush=True,
            )

    print("\n=== chains of the field's own trails, by span ===", flush=True)
    print(f"{'cell':24s} {'span':>18s} {'per-component':>18s}  decision", flush=True)
    for row in (item for item in rows if item["kind"] == "chain"):
        print(
            f"{row['label']:24s} {row['span_px']:7.0f} px {row['span_multiple']:5.2f}x "
            f"{row['length_px']:7.0f} px {row['length_multiple']:5.2f}x  "
            f"{'DROPPED' if row['dropped'] else 'kept'}"
            f"  ({row['dropped_px']} px, {row['dropped_components']} of {row['components']} component(s) of "
            f"{row['size_px']} px)",
            flush=True,
        )

    contamination = contamination_cells(
        shape,
        angle_deg=angle_deg,
        trail_length=trail_length,
        frame=frame,
        header=header,
        config=config,
        veto_cfg=veto_cfg,
        rms_map=rms_map,
        flat=flat,
    )
    print("\n=== the parallel trail as an input to the family it is judged against ===", flush=True)
    for row in contamination:
        family_here = row["family"]
        text = (
            "no family at all"
            if family_here is None
            else (
                f"{family_here['length_px']:.2f} px, angle {family_here['angle_deg']}, {family_here['count']} sources"
            )
        )
        print(f"  injected {row['injected_multiple']:4.1f}x: {text}", flush=True)

    stage = None
    if not options.no_stage:
        print("\n=== the campaign chain geometry through the stage (veto on vs off) ===", flush=True)
        stage = stage_cell(
            shape,
            angle_deg=angle_deg,
            trail_length=trail_length,
            frame=frame,
            header=header,
            config=config,
            streak_cfg=streak_cfg,
            rms_map=rms_map,
            flat=flat,
        )
        print(
            f"  {stage['chain_segments']} segments, gap {stage['gap_factor']}l: span {stage['chain_span_px']} px, "
            f"chain body {stage['chain_px']} px",
            flush=True,
        )
        print(
            f"  veto on : mask {stage['veto_on']['mask_px']:6d} px, {stage['veto_on']['chain_px_in_mask']:6d} of the chain"
            f"  ({stage['veto_on']['seconds']}s)",
            flush=True,
        )
        print(
            f"  veto off: mask {stage['veto_off']['mask_px']:6d} px, {stage['veto_off']['chain_px_in_mask']:6d} of the chain"
            f"  ({stage['veto_off']['seconds']}s)",
            flush=True,
        )

    measure = window(rows)
    print("\n=== the window ===", flush=True)
    print(
        f"  the sweep brackets it between {measure['dropped_through']:.2f}x and {measure['kept_from']:.2f}x "
        f"the measured family length",
        flush=True,
    )
    parallel_rows = [row for row in rows if row["kind"] == "satellite-parallel"]
    dropped_px = [row["geometric_length_px"] for row in parallel_rows if row["dropped"]]
    kept_px = [row["geometric_length_px"] for row in parallel_rows if not row["dropped"]]
    boundary = None
    if measure["dropped_through"] is None or not dropped_px or not kept_px:
        # Every cell fell the same way, so the sweep has no bracket to bisect. A
        # length_factor outside the sweep's range does that, and the report then
        # says so rather than bisecting between two points that are not there.
        print(
            "  every swept length fell the same way, so there is no crossing inside the sweep; "
            f"the shipped length_factor is {length_factor}",
            flush=True,
        )
    else:
        bracket_low = max(dropped_px)
        bracket_high = min(kept_px)
        boundary = refine_boundary(
            shape,
            angle_deg=angle_deg,
            family=family,
            veto_cfg=veto_cfg,
            rms_map=rms_map,
            lower_px=bracket_low,
            upper_px=bracket_high,
        )
        print(
            f"  bisected between {bracket_low:.0f} px and {bracket_high:.0f} px: the decision flips at "
            f"{boundary['crossing_px']:.2f} px = {boundary['crossing_multiple']:.3f}x the family's measured length "
            f"({boundary['dropped_through_px']:.2f} px dropped, {boundary['kept_from_px']:.2f} px kept)",
            flush=True,
        )
        print(
            f"  shipped length_factor is {length_factor}: the crossing is "
            f"{boundary['crossing_multiple'] / length_factor:.3f} of the configured factor",
            flush=True,
        )

    report = {
        "size": size,
        "seed": options.seed,
        "stars": options.stars,
        "trail_angle": angle_deg,
        "star_trail_length": trail_length,
        "flat": options.flat or None,
        "family": family,
        "family_measured_fraction_of_trail": round(float(family["length_px"]) / trail_length, 3),
        "length_factor": length_factor,
        "angle_tol_deg": veto_cfg.get("angle_tol_deg"),
        "window": measure,
        "boundary": boundary,
        "rows": rows,
        "family_contamination": contamination,
        "stage": stage,
        "config_overrides": options.set,
    }
    if options.out:
        os.makedirs(os.path.dirname(os.path.abspath(options.out)), exist_ok=True)
        with open(options.out, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"\nwrote {options.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
