"""Acceptance test for the streak stage on a *non-sidereal* frame.

Why this exists
---------------
Every recall number the stage has was measured on sidereal fields, where a star
is a point and the only long straight thing in the frame is a trail. On a field
tracked at the object's own rate every star is a trail, so "long, thin and
straight" stops being evidence: measured on the 2026-10-14 campaign, all seven
exposures' STREAK products were detector defects and star-trail chains, with no
satellite trail among 1.77M flagged pixels.

This harness puts the question in the only form that can be answered: a frame
with a known star-trail family, into which four classes are injected --

  positive  satellite trail: PSF-width, uniform along its length, at an angle
            the star-trail family does not share
  negative  star trail: the family's own angle and length, bright head, fading
            tail -- what the field is full of
  negative  chain: several star trails placed collinearly with gaps, which is
            what the campaign's false positives actually were
  negative  column / row band: axis-aligned, constant width, full span, faint
            -- the detector defects the campaign's masks were mostly made of

-- and scores what the stage accepts. The control run is the one that matters:
a non-sidereal frame with *no* satellite injected must produce an empty STREAK
mask, and today it does not.

Inputs come from the production path (`capture_detector_inputs`), because the
same trap that produced the campaign's false positives lives there: without a
flat, chip-fixed columns and rows survive into `data_sub` and the stage reads
them as bright lines.

Usage
-----
    {py} benchmarks/streak_nonsidereal_acceptance.py [options]

    --frame synthetic|real   synthetic non-sidereal frame (default), or a crop
                             of a real non-sidereal exposure
    --exposure PATH --hdu N  the real frame, with --crop y0,x0
    --size N                 frame side in pixels (default 1024)
    --star-trail-length N    the field's star-trail length (default 55 px,
                             measured on 1851960 HDU 17)
    --trail-angle DEG        the field's star-trail angle (default 40.2 deg,
                             measured on 1851960 HDU 17)
    --stars N                background trailed stars in a synthetic frame
    --flat PATH              flat frame, as production would have it; without
                             one the axis-aligned defects are expected to be
                             accepted, and the run says so
    --quick                  one positive cell per class instead of the matrix
    --set KEY=VALUE          config override, e.g. --set streak_masking.mask_params.min_mask_pixels=64
    --out PATH               write the JSON report here
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import fitsio
import numpy as np
from astropy.io import fits as _fits
from scipy import ndimage
from scipy.ndimage import binary_dilation

from weightmask.config import default_config, resolve_config
from weightmask.process import header_gain_read_noise
from weightmask.streaks import detect_streaks

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.production_inputs import capture_detector_inputs
from benchmarks.streak_inject import apply_overrides, trail_recall

POSITIVE_OFFSETS_QUICK = (30.0, 90.0)
POSITIVE_OFFSETS_FULL = (10.0, 30.0, 60.0, 90.0)
POSITIVE_LENGTHS = (400.0, 800.0)
POSITIVE_SIGMAS = (4.0, 8.0, 12.0)
FOUND_LINE_RECALL = 0.5
FALSE_POSITIVE_FRACTION = 0.1


def _unit(angle_deg):
    theta = np.radians(float(angle_deg))
    return np.cos(theta), np.sin(theta)


def render_trail(shape, *, angle_deg, length, cx, cy, peak_sig, rms_map, head_sig=0.0, decay=None):
    """One trail in sky-sigma units: uniform body, optional bright head.

    A satellite is the ``head_sig=0`` case. A star trail is the other one: the
    head is the star and the flux falls off along the trail, which is why a
    star's trail is not uniformly bright.
    """
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    dx, dy = _unit(angle_deg)
    along = (xx - cx) * dx + (yy - cy) * dy
    across = np.abs((xx - cx) * (-dy) + (yy - cy) * dx)
    on_trail = (along >= -length / 2.0) & (along <= length / 2.0) & (across <= 3.0)
    profile = np.exp(-0.5 * (across / 1.1) ** 2)
    flux = np.where(on_trail, peak_sig * profile, 0.0)
    if head_sig:
        span = float(decay if decay is not None else length / 3.0)
        taper = np.exp(-np.clip(along + length / 2.0, 0.0, None) / max(span, 1.0))
        flux = np.where(on_trail, peak_sig * taper * profile, 0.0)
        head = np.exp(-0.5 * (((along + length / 2.0) / 1.6) ** 2 + (across / 1.6) ** 2))
        flux = flux + head_sig * head
    body = on_trail & (profile > 0.3)
    if rms_map is not None:
        flux = flux * np.asarray(rms_map, dtype=np.float64)
    return flux.astype(np.float32), body


def band(shape, *, axis, start, width, amplitude, rms_map):
    """An axis-aligned band spanning the frame, in sky-sigma units."""
    height, width_ = shape
    flux = np.zeros(shape, dtype=np.float32)
    if axis == "column":
        flux[:, start : start + width] = amplitude
    else:
        flux[start : start + width, :] = amplitude
    if rms_map is not None:
        flux = flux * np.asarray(rms_map, dtype=np.float64)
    body = flux != 0
    del height, width_
    return flux.astype(np.float32), body


def synthetic_frame(shape, *, angle_deg, trail_length, stars, sky, rms, seed, columns=(3, 11), rows=(3, 8)):
    """A non-sidereal frame: sky noise, a family of trailed stars, and defects.

    The star trails are the point of the fixture: every one shares the field's
    angle and length, with a bright head and a fading tail, so a detector that
    accepts star trails is being tested rather than a detector that meets noise.
    """
    rng = np.random.default_rng(seed)
    height, width = shape
    frame = sky + rng.normal(0.0, rms, size=shape).astype(np.float32)
    truth = {"star_trails": [], "columns": [], "rows": []}
    rms_map = np.full(shape, rms, dtype=np.float32)
    for _ in range(int(stars)):
        cx = float(rng.uniform(0.05 * width, 0.95 * width))
        cy = float(rng.uniform(0.05 * height, 0.95 * height))
        head_sig = float(rng.uniform(30.0, 200.0))
        flux, body = render_trail(
            shape,
            angle_deg=angle_deg + float(rng.normal(0.0, 1.5)),
            length=trail_length,
            cx=cx,
            cy=cy,
            peak_sig=head_sig * 0.25,
            rms_map=rms_map,
            head_sig=head_sig,
        )
        frame += flux
        truth["star_trails"].append({"cx": cx, "cy": cy, "angle": angle_deg, "body": body})
    for width_ in columns:
        start = int(rng.uniform(0.1 * width, 0.9 * width))
        flux, body = band(shape, axis="column", start=start, width=int(width_), amplitude=2.0, rms_map=rms_map)
        frame += flux
        truth["columns"].append({"start": start, "width": int(width_), "body": body})
    for height_ in rows:
        start = int(rng.uniform(0.1 * height, 0.9 * height))
        flux, body = band(shape, axis="row", start=start, width=int(height_), amplitude=2.0, rms_map=rms_map)
        frame += flux
        truth["rows"].append({"start": start, "width": int(height_), "body": body})
    return frame, truth


def place_positives(shape, *, angle_deg, offsets, lengths, sigmas, rng, margin=0.12):
    """Place satellite-like trails, one per (offset, length, sigma) cell."""
    height, width = shape
    cells = []
    for offset in offsets:
        for length in lengths:
            for sigma in sigmas:
                cx = float(rng.uniform(width * margin, width * (1 - margin)))
                cy = float(rng.uniform(height * margin, height * (1 - margin)))
                cells.append(
                    {
                        "kind": "satellite",
                        "angle": (angle_deg + offset) % 180.0,
                        "offset_deg": offset,
                        "length": float(length),
                        "sigma": float(sigma),
                        "cx": cx,
                        "cy": cy,
                    }
                )
    return cells


def place_chains(shape, *, angle_deg, trail_length, count, rng, segments=4, gap_factor=1.6):
    """Collinear star trails with gaps: the campaign's false-positive geometry."""
    height, width = shape
    # A chain is placed clear of the frame edges by its own span.
    span = segments * trail_length * gap_factor
    cx = float(rng.uniform(0.2 * width, 0.8 * width))
    cy = float(rng.uniform(0.2 * height, 0.8 * height))
    dx, dy = _unit(angle_deg)
    starts = []
    for index in range(int(segments)):
        offset = (index - (segments - 1) / 2.0) * trail_length * gap_factor
        starts.append((cx + offset * dx, cy + offset * dy))
    del span
    return {
        "kind": "chain",
        "angle": angle_deg,
        "length": trail_length,
        "segments": starts,
        "count": int(count),
        "rng": rng,
    }


def render_chain(shape, chain, rms_map, rng):
    """Add a chain's gapped star trails; return the flux and the union body."""
    flux = np.zeros(shape, dtype=np.float32)
    body = np.zeros(shape, dtype=bool)
    for cx, cy in chain["segments"]:
        piece, piece_body = render_trail(
            shape,
            angle_deg=chain["angle"] + float(rng.normal(0.0, 1.0)),
            length=chain["length"],
            cx=cx,
            cy=cy,
            peak_sig=float(rng.uniform(10.0, 40.0)),
            rms_map=rms_map,
            head_sig=float(rng.uniform(40.0, 120.0)),
        )
        flux += piece
        body |= piece_body
    return flux, body


def coverage(mask, body):
    """Fraction of a truth body the returned mask covers, and the pixel count."""
    n_px = int(np.count_nonzero(body))
    if n_px == 0:
        return 0.0, 0
    hits = int(np.count_nonzero(mask & body))
    return hits / n_px, hits


def score(mask, cells, chains, *, control_mask=None):
    """Verdicts for every injected object, plus the control's own footprint."""
    rows = []
    for cell in cells:
        body = cell.get("body")
        if body is None:
            continue
        _exact, _tolerant, line = trail_recall(mask, body, dilation=5, line_half_width=2.0)
        rows.append(
            {
                "kind": cell["kind"],
                "offset_deg": cell.get("offset_deg"),
                "length": cell.get("length"),
                "sigma": cell.get("sigma"),
                "angle": round(float(cell["angle"]), 1),
                "line_recall": round(float(line), 3),
                "found": bool(line >= FOUND_LINE_RECALL),
                "body_coverage": round(coverage(mask, body)[0], 3),
            }
        )
    for chain in chains:
        cover, hits = coverage(mask, chain["body"])
        # A chain is a false positive when the stage covers its segments *or*
        # draws a line through the gaps between them, which is what the campaign
        # saw: the mask ran 931 px across mostly empty sky.
        rows.append(
            {
                "kind": "chain",
                "offset_deg": 0.0,
                "length": chain["length"],
                "sigma": None,
                "angle": round(float(chain["angle"]), 1),
                "line_recall": None,
                "found": None,
                "body_coverage": round(cover, 3),
                "covered_px": hits,
                "false_positive": bool(cover > FALSE_POSITIVE_FRACTION),
            }
        )
    return rows


def summarize_components(mask, *, top=6):
    """Sizes and axis spans of a mask's connected components.

    A footprint count says the stage fired; the components say on what. A
    full-span component with a two- or three-pixel minor axis is an axis-aligned
    detector band, a component that is long and diagonal is a fitted trail, and
    the difference decides whether the fix belongs in the mask or in the vetoes.
    """

    labels, count = ndimage.label(mask)
    if count == 0:
        return []
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    order = np.argsort(sizes[1:])[::-1][: int(top)] + 1
    height, width = mask.shape
    out = []
    for label in order:
        rows, cols = np.nonzero(labels == label)
        span_y = int(rows.max() - rows.min() + 1)
        span_x = int(cols.max() - cols.min() + 1)
        out.append(
            {
                "px": int(sizes[label]),
                "y_span": span_y,
                "x_span": span_x,
                "y0": int(rows.min()),
                "x0": int(cols.min()),
                "axis_aligned": bool(span_y >= height // 2 and span_x <= 16 or span_x >= width // 2 and span_y <= 16),
            }
        )
    return out


def attribute(mask, truth, *, dilation=2):
    """Where a mask's pixels sit, by truth class.

    The control's footprint is the number that matters, and a count alone does
    not say whether it came from the star-trail field, from a detector band, or
    from noise. Pixel attribution answers that: each class is grown by
    ``dilation`` so a line drawn a pixel off a trail still counts as being on it.
    """

    total = int(np.count_nonzero(mask))
    grown = {}
    for name in ("star_trails", "columns", "rows"):
        union = np.zeros(mask.shape, dtype=bool)
        for item in truth.get(name, []):
            union |= item["body"]
        grown[name] = binary_dilation(union, iterations=dilation) if union.any() else union
    accounted = np.zeros(mask.shape, dtype=bool)
    out = {}
    for name, region in grown.items():
        hit = mask & region & ~accounted
        accounted |= hit
        out[name] = int(np.count_nonzero(hit))
    out["unexplained"] = int(np.count_nonzero(mask & ~accounted))
    out["total"] = total
    return out


def detector_negative_rows(mask, truth):
    rows = []
    for name, axis in (("columns", "column"), ("rows", "row")):
        for item in truth[name]:
            cover, hits = coverage(mask, item["body"])
            rows.append(
                {
                    "kind": f"{axis}-band",
                    "start": item["start"],
                    "width": item["width"],
                    "body_coverage": round(cover, 3),
                    "covered_px": hits,
                    "false_positive": bool(cover > FALSE_POSITIVE_FRACTION),
                }
            )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frame", choices=("synthetic", "real"), default="synthetic")
    parser.add_argument("--exposure", default="")
    parser.add_argument("--hdu", type=int, default=1)
    parser.add_argument("--crop", default="0,0", help="y0,x0 for --frame real")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--star-trail-length", type=float, default=55.0)
    parser.add_argument("--trail-angle", type=float, default=40.2)
    parser.add_argument("--stars", type=int, default=60)
    parser.add_argument("--sky", type=float, default=571.0)
    parser.add_argument("--rms", type=float, default=21.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flat", default="")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--out", default="")
    options = parser.parse_args(argv)

    size = int(options.size)
    flat = None
    if options.frame == "real":
        if not options.exposure:
            raise SystemExit("--frame real needs --exposure")
        y0, x0 = (int(part) for part in options.crop.split(","))
        with fitsio.FITS(options.exposure, "r") as handle:
            frame = handle[options.hdu].read()[y0 : y0 + size, x0 : x0 + size].astype(np.float32)
            header = handle[options.hdu].read_header()
        if options.flat:
            with fitsio.FITS(options.flat, "r") as handle:
                flat = handle[options.hdu].read()[y0 : y0 + size, x0 : x0 + size].astype(np.float32)
        truth = {}
    else:
        frame, truth = synthetic_frame(
            (size, size),
            angle_deg=options.trail_angle,
            trail_length=options.star_trail_length,
            stars=options.stars,
            sky=options.sky,
            rms=options.rms,
            seed=options.seed,
        )

        header = _fits.Header({"NAXIS": 2, "SATURATE": 65535.0, "EXPTIME": 60.0, "GAIN": 1.6, "RDNOISE": 5.0})
        flat = None

    config = apply_overrides(default_config(), options.set)
    streak_cfg = dict(config["streak_masking"])

    gain, _read_noise = header_gain_read_noise(header, resolve_config(config)["variance"])
    print(f"frame {frame.shape}, gain {gain}", flush=True)

    # --- the control: a non-sidereal frame with no satellite injected ---------
    t0 = time.time()
    clean_data_sub, clean_rms, clean_existing = capture_detector_inputs(frame, header, config, flat=flat)
    print(
        f"production inputs: existing mask {int(clean_existing.sum())} px "
        f"({'with flat' if flat is not None else 'NO FLAT'}), {time.time() - t0:.1f}s",
        flush=True,
    )
    existing_px = int(clean_existing.sum())
    t0 = time.time()
    control_mask = detect_streaks(clean_data_sub, clean_rms, clean_existing, dict(streak_cfg)).mask
    control_px = int(np.count_nonzero(control_mask))
    print(f"control (no satellite injected): STREAK {control_px} px in {time.time() - t0:.1f}s", flush=True)

    # --- the injected frame ---------------------------------------------------
    rng = np.random.default_rng(options.seed + 1000)
    offsets = POSITIVE_OFFSETS_QUICK if options.quick else POSITIVE_OFFSETS_FULL
    lengths = POSITIVE_LENGTHS[:1] if options.quick else POSITIVE_LENGTHS
    cells = place_positives(
        frame.shape,
        angle_deg=options.trail_angle,
        offsets=offsets,
        lengths=lengths,
        sigmas=POSITIVE_SIGMAS,
        rng=rng,
    )
    chains = [
        place_chains(
            frame.shape, angle_deg=options.trail_angle, trail_length=options.star_trail_length, count=1, rng=rng
        )
    ]
    truth = truth or {}
    truth.setdefault("star_trails", [])
    truth.setdefault("columns", [])
    truth.setdefault("rows", [])

    rms_map = np.full(frame.shape, float(options.rms), dtype=np.float32)
    injected = frame.copy()
    for cell in cells:
        flux, body = render_trail(
            frame.shape,
            angle_deg=cell["angle"],
            length=cell["length"],
            cx=cell["cx"],
            cy=cell["cy"],
            peak_sig=cell["sigma"],
            rms_map=clean_rms if clean_rms is not None else rms_map,
        )
        injected += flux
        cell["body"] = body
    assert len(cells) > 0, cells
    for chain in chains:
        flux, body = render_chain(frame.shape, chain, clean_rms if clean_rms is not None else rms_map, rng)
        injected += flux
        chain["body"] = body

    t0 = time.time()
    data_sub, rms, existing = capture_detector_inputs(injected, header, config, flat=flat)
    mask = detect_streaks(data_sub, rms, existing, dict(streak_cfg)).mask
    print(f"injected frame: STREAK {int(np.count_nonzero(mask))} px in {time.time() - t0:.1f}s", flush=True)

    rows = score(mask, cells, chains, control_mask=control_mask)
    # The bands and star trails are present in the control too, so the control
    # mask is where they are tested as negatives: with no satellite injected, a
    # long straight band is the only thing the stage could be responding to.
    rows.extend(detector_negative_rows(control_mask, truth))
    for row in detector_negative_rows(mask, truth):
        row["kind"] += " (injected)"
        rows.append(row)
    control_attribution = attribute(control_mask, truth)
    mask_attribution = attribute(mask, truth)
    control_components = summarize_components(control_mask)

    positives = [row for row in rows if row["kind"] == "satellite"]
    found = [row for row in positives if row["found"]]
    false_positives = [row for row in rows if row.get("false_positive")]
    print("\nsatellite recall by cell (offset from the star-trail angle, length, sigma):")
    for row in sorted(positives, key=lambda item: (item["offset_deg"], item["length"], item["sigma"])):
        flag = "FOUND" if row["found"] else "missed"
        print(
            f"  {row['offset_deg']:5.0f} deg  {row['length']:5.0f} px  {row['sigma']:4.0f} sigma"
            f"  line_recall {row['line_recall']:.2f}  {flag}",
            flush=True,
        )
    print(f"\npositives: {len(found)}/{len(positives)} found", flush=True)
    print(f"control footprint (should be 0): {control_px} px", flush=True)
    where = ", ".join(f"{name} {count}" for name, count in control_attribution.items())
    if options.frame == "real":
        # A real crop carries no modelled star-trail family, so every one of its
        # pixels would be reported as unexplained. Say so instead of printing a
        # zero that reads as "no star trails here".
        where = where.replace("star_trails 0", "star_trails unmodelled")
    print(f"  where the control's pixels sit: {where}", flush=True)
    print("negatives accepted as STREAK (should be none):", flush=True)
    for row in rows:
        if row["kind"].split(" ")[0] in ("chain", "column-band", "row-band"):
            print(
                f"  {row['kind']:22s} coverage {row['body_coverage']:.2f}"
                f" ({row.get('covered_px', 0)} px)  {'FALSE POSITIVE' if row.get('false_positive') else 'rejected'}",
                flush=True,
            )

    print("\ncontrol's largest components (px, spans, axis-aligned):", flush=True)
    for item in control_components:
        print(
            f"  {item['px']:6d} px  at y{item['y0']},x{item['x0']}  spans {item['y_span']}x{item['x_span']}"
            f"  {'AXIS-ALIGNED' if item['axis_aligned'] else 'diagonal'}",
            flush=True,
        )

    report = {
        "frame": options.frame,
        "size": size,
        "seed": options.seed,
        "trail_angle": options.trail_angle,
        "star_trail_length": options.star_trail_length,
        "flat": options.flat or None,
        "existing_mask_px": existing_px,
        "control_px": control_px,
        "positives": len(positives),
        "positives_found": len(found),
        "false_positives": len(false_positives),
        "rows": rows,
        "control_attribution": control_attribution,
        "mask_attribution": mask_attribution,
        "control_components": control_components,
        "config_overrides": options.set,
    }
    if options.out:
        with open(options.out, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"\nwrote {options.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
