#!/usr/bin/env python3
"""Curate a *reviewed* real-trail ground-truth set for MegaCam.

Why this exists
---------------
Every detection threshold in ``weightmask.streaks`` is currently tuned against
*injected* trails (``benchmarks/streak_inject.py``).  Injected trails are clean
Gaussians: they do not reproduce the clutter that actually drives the
thresholds -- bad columns, bleed trails, chip edges, ghosts, glints.  This tool
mines real exposures and curates a label set out of evidence the scored
detector cannot produce on its own.

The protocol (four independent evidence layers)
-----------------------------------------------
1. **Propose.**  Either the cheap binned-Hough stage
   (``_detect_streaks_houghpeaks``), the production pipeline
   (``detect_streaks``, candidates taken from connected components of its mask),
   or both.  Each candidate's geometry is *re-measured* by PCA of the pixels
   behind it, so the measurement is one step removed from the proposer's own
   reported angle.  The proposer is recorded per entry (``proposed_by``): a
   scorer should trust recall on entries some *other* proposer contributed.

2. **Corroborate across the focal plane.**  A MegaCam ``fits.fz`` carries a
   per-CCD TAN WCS that shares ``CRVAL`` across every CCD and differs only in
   ``CRPIX`` and (slightly) the ``CD`` matrix.  The mosaic therefore has one
   common tangent plane, and because the gnomonic projection maps great circles
   to *straight lines*, a real trail crossing the mosaic projects onto one line
   on every CCD it crosses.  So: map each measured line into the shared
   ``(xi, eta)`` frame and group lines that coincide.  A group spanning >= 2 CCDs
   is structure spanning the focal plane.

3. **Veto detector-fixed structure.**  Three independent vetoes, because
   collinearity alone is *not* sufficient -- a chip edge is collinear across CCDs
   by construction:

   * ``chip_replica``: two group members sitting at the same CCD-*local* offset
     and angle.  A sky line lands at a different CCD-local offset on every chip
     it crosses, so equal offsets mean the feature is replicated per chip.
     (This is what catches the MegaCam top-edge row at y ~ 4596 that the naive
     cross-CCD test happily groups.)
   * ``static``: the same CCD-local line in another epoch of the *same pointing*
     (identical ``CRVAL`` means identical WCS, so a satellite trail moves and a
     bad column does not).
   * ``axis_veto``: a candidate within 1.5 deg of a CCD axis sitting on a column
     or row whose robust noise is an outlier for that chip.

   A group any of whose members is vetoed is *poisoned*: its clean members drop
   to ``uncertain`` rather than being promoted to ``trail``.

4. **Measure independently.**  The bilinear-sampled profile along the candidate
   line against a background band 25-60 px off-line, in raw counts with a robust
   off-line sigma.  ``support_px`` is the longest run above 4 sigma: a real ridge
   keeps it long.  Normalising by the background-rms *map* instead was tried
   first and produced 24000-sigma peaks, because that map has degenerate values;
   the off-line band is self-calibrating and immune to that.

Output
------
``--out`` writes a JSON fixture; ``--review-sheet`` writes a Markdown table with
every entry and its evidence so the labels can be audited row by row.  Entries
labelled ``trail`` are the scored ground truth; ``artefact`` entries are the
scored negatives (a detector that masks them is wrong); ``uncertain`` entries are
recorded for review but excluded from scoring.

Usage
-----
    pixi run python benchmarks/curate_trail_truth.py \
        --data-root benchmark_data/megacam/perf \
        --out benchmarks/trail_truth/megacam_real_trails.json \
        --review-sheet benchmarks/trail_truth/megacam_real_trails.review.md

It never hard-codes a cluster path: ``--data-root`` defaults to the repo's
``benchmark_data/megacam/perf``, and an exposure it cannot find is reported
rather than silently skipped.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

#: MegaCam's 0.185"/pixel gives 5.194e-05 deg/pixel (the header CD value).  Used
#: only to turn mosaic tolerances into pixels a reviewer can sanity-check.
DEG_PER_MOSAIC_PX = 5.194e-05

DEFAULT_DATA_ROOT = os.path.join("benchmark_data", "megacam", "perf")
DEFAULT_OUT = os.path.join("benchmarks", "trail_truth", "megacam_real_labels.json")
DEFAULT_REVIEW_SHEET = os.path.join("benchmarks", "trail_truth", "megacam_real_labels.review.md")
DEFAULT_CONFIG = "weightmask.yml"
SCHEMA = "weightmask.trail_truth.v1"


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def _normalise_normal(nx: float, ny: float) -> tuple:
    """Return a unit normal with a canonical sign so lines compare directly."""
    norm = math.hypot(nx, ny)
    if norm <= 0:
        return (0.0, 0.0)
    nx, ny = nx / norm, ny / norm
    if nx < 0 or (nx == 0.0 and ny < 0):
        nx, ny = -nx, -ny
    return (float(nx), float(ny))


def _angle_between(normals_a, normals_b) -> float:
    """Angle in degrees between two unit normals (sign-insensitive)."""
    dot = abs(normals_a[0] * normals_b[0] + normals_a[1] * normals_b[1])
    return float(math.degrees(math.acos(min(1.0, max(-1.0, dot)))))


def _clip_line_to_shape(point, direction, shape):
    """Clip an infinite line to the image rectangle. Returns None if outside."""
    h, w = shape
    x0, y0 = float(point[0]), float(point[1])
    dx, dy = float(direction[0]), float(direction[1])
    ts = []
    if abs(dx) > 1e-12:
        for x in (0.0, w - 1.0):
            ts.append((x - x0) / dx)
    if abs(dy) > 1e-12:
        for y in (0.0, h - 1.0):
            ts.append((y - y0) / dy)
    pts = []
    for t in ts:
        x, y = x0 + t * dx, y0 + t * dy
        if -1e-6 <= x <= w - 1 + 1e-6 and -1e-6 <= y <= h - 1 + 1e-6:
            pts.append((x, y))
    if len(pts) < 2:
        return None
    best = None
    best_d = -1.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
            if d > best_d:
                best_d, best = d, (pts[i], pts[j])
    if best_d <= 1.0:
        return None
    return best


def _bilinear(image, xs, ys):
    """Bilinear sample at fractional ``(xs, ys)`` with edge clamping."""
    h, w = image.shape
    xs = np.clip(xs, 0.0, w - 1.000001)
    ys = np.clip(ys, 0.0, h - 1.000001)
    x0 = np.floor(xs).astype(np.intp)
    y0 = np.floor(ys).astype(np.intp)
    fx = xs - x0
    fy = ys - y0
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    top = image[y0, x0] * (1.0 - fx) + image[y0, x1] * fx
    bot = image[y1, x0] * (1.0 - fx) + image[y1, x1] * fx
    return top * (1.0 - fy) + bot * fy


def _mad_std(values) -> float:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def geometry_from_header(header) -> dict | None:
    """Extract the per-CCD tangent-plane map from a MegaCam science header.

    Returns ``None`` when the WCS keywords needed to place the CCD in the shared
    mosaic frame are absent -- such HDUs cannot take part in the cross-CCD layer
    and are reported as WCS-less rather than guessed at.
    """
    required = ("CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CD1_1", "CD1_2", "CD2_1", "CD2_2")
    keys = {name: header.get(name) for name in required}
    if any(value is None for value in keys.values()):
        return None
    return {
        "crval": [float(keys["CRVAL1"]), float(keys["CRVAL2"])],
        # FITS is 1-indexed; array coordinates are 0-indexed.
        "crpix": [float(keys["CRPIX1"]) - 1.0, float(keys["CRPIX2"]) - 1.0],
        "cd": [
            [float(keys["CD1_1"]), float(keys["CD1_2"])],
            [float(keys["CD2_1"]), float(keys["CD2_2"])],
        ],
        "ctypes": [str(header.get("CTYPE1", "")), str(header.get("CTYPE2", ""))],
    }


def ccd_to_mosaic(points, geom):
    """Map 0-indexed CCD pixels to the shared tangent-plane frame (degrees)."""
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    dx = points[:, 0] - geom["crpix"][0]
    dy = points[:, 1] - geom["crpix"][1]
    cd = np.asarray(geom["cd"], dtype=np.float64)
    xi = cd[0, 0] * dx + cd[0, 1] * dy
    eta = cd[1, 0] * dx + cd[1, 1] * dy
    return np.column_stack([xi, eta])


def line_to_mosaic(point, direction, geom):
    """Transform a CCD-frame line into the mosaic frame as ``(normal, offset)``.

    The pair is canonicalised *jointly*: ``(n, d)`` and ``(-n, -d)`` describe the
    same line, and a line must be flipped in both or neither.  Canonicalising the
    normal alone is unsound here, because MegaCam chips are mirrored in ``CD1_1``
    -- which flips the mosaic direction of a chip-vertical line and hence the
    sign of its offset.  Flipping the normal without the offset would then make
    two chip-vertical lines on *opposite sides* of the mosaic compare as equal,
    which is exactly what a first pass of this tool did (it produced sixteen
    bogus "cross-CCD" trail groups from pairs of bad columns at +-xi).
    """
    mosaic_point = ccd_to_mosaic([point], geom)[0]
    cd = np.asarray(geom["cd"], dtype=np.float64)
    mosaic_dir = cd @ np.asarray(direction, dtype=np.float64)
    norm = float(np.hypot(*mosaic_dir))
    if norm <= 0:
        return None
    mosaic_dir = mosaic_dir / norm
    nx, ny = -mosaic_dir[1], mosaic_dir[0]
    length = math.hypot(nx, ny)
    if length <= 0:
        return None
    nx, ny = nx / length, ny / length
    offset = float(nx * mosaic_point[0] + ny * mosaic_point[1])
    if nx < 0 or (nx == 0.0 and ny < 0):
        nx, ny, offset = -nx, -ny, -offset
    return {"normal": [float(nx), float(ny)], "offset_deg": offset}


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
def longest_support_run(profile, threshold=4.0, max_gap_samples=5):
    """Longest near-contiguous run above ``threshold``; returns (span, start, peak).

    A short trail clipped to the image's full chord gives a tiny fraction-above
    even when the trail is unmistakable, so the curation rule uses this run
    length rather than a fraction of the chord.
    """
    profile = np.asarray(profile)
    hits = np.nonzero(profile > threshold)[0]
    if hits.size == 0:
        return 0, 0, 0.0
    breaks = np.nonzero(np.diff(hits) > max_gap_samples + 1)[0] + 1
    runs = np.split(hits, breaks)
    best = max(runs, key=lambda run: int(run[-1] - run[0]))
    span = int(best[-1] - best[0] + 1)
    segment = profile[best[0] : best[-1] + 1]
    return span, int(best[0]), float(np.max(segment))


def _independent_line_stats(data_sub, point, direction, shape, offsets=(25.0, 60.0), threshold=4.0):
    """Measure the ridge along a candidate line from the pixels, not from the mask."""
    clipped = _clip_line_to_shape(point, direction, shape)
    if clipped is None:
        return None
    (x0, y0), (x1, y1) = clipped
    length = math.hypot(x1 - x0, y1 - y0)
    n = max(int(length), 8)
    t = np.linspace(0.0, 1.0, n)
    xs = x0 + (x1 - x0) * t
    ys = y0 + (y1 - y0) * t
    perpendicular = np.array([-direction[1], direction[0]], dtype=np.float64)

    on_line = _bilinear(data_sub, xs, ys)
    background = []
    for offset in offsets:
        background.append(_bilinear(data_sub, xs + offset * perpendicular[0], ys + offset * perpendicular[1]))
        background.append(_bilinear(data_sub, xs - offset * perpendicular[0], ys - offset * perpendicular[1]))
    background = np.concatenate(background)
    bg_median = float(np.median(background))
    bg_sigma = _mad_std(background) or (float(np.std(background)) or 1e-6)
    z = (on_line - bg_median) / bg_sigma
    spacing_px = length / max(n - 1, 1)
    span, start, peak = longest_support_run(z, threshold)
    return {
        "length_px": float(length),
        "n_samples": int(n),
        "z_max": float(np.max(z)),
        "z_median": float(np.median(z)),
        "frac_above_4": float(np.mean(z > 4.0)),
        "support_samples": int(span),
        "support_px": float(span * spacing_px),
        "support_start_px": float(start * spacing_px),
        "support_peak_z": float(peak),
        "bg_median_counts": float(bg_median),
        "bg_sigma_counts": float(bg_sigma),
    }


def _pca_line(xs, ys):
    """PCA a set of pixels; returns the point, unit direction and the two sigmas."""
    px = np.asarray(xs, dtype=np.float64)
    py = np.asarray(ys, dtype=np.float64)
    mx, my = float(px.mean()), float(py.mean())
    dx = px - mx
    dy = py - my
    cov = np.array(
        [
            [float(np.mean(dx * dx)), float(np.mean(dx * dy))],
            [float(np.mean(dx * dy)), float(np.mean(dy * dy))],
        ]
    )
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    direction = eigenvectors[:, int(np.argmax(eigenvalues))]
    if direction[0] < 0 or (direction[0] == 0 and direction[1] < 0):
        direction = -direction
    major = float(math.sqrt(max(eigenvalues.max(), 0.0)))
    minor = float(math.sqrt(max(eigenvalues.min(), 0.0)))
    along = dx * direction[0] + dy * direction[1]
    return {
        "point": [mx, my],
        "direction": [float(direction[0]), float(direction[1])],
        "normal": list(_normalise_normal(-direction[1], direction[0])),
        "n_mask_px": int(px.size),
        "rms_along_px": major,
        "rms_across_px": minor,
        "width_px": float(4.0 * minor),
        "length_px": float(4.0 * major),
        "span_along_px": float(along.max() - along.min()),
        "elongation": float(major / minor) if minor > 0 else float("inf"),
    }


def _measure_candidate(mask, candidate_normal, candidate_offset, cx, cy, corridor_px):
    """Re-measure one candidate's geometry from the mask pixels behind it."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    signed = candidate_normal[0] * (xs - cx) + candidate_normal[1] * (ys - cy) - candidate_offset
    keep = np.abs(signed) <= corridor_px
    if int(keep.sum()) < 16:
        return None
    return _pca_line(xs[keep], ys[keep])


def _components_from_mask(mask, min_px, max_components, min_elongation=1.0):
    """Measure every substantial, sufficiently elongated connected component."""
    from scipy import ndimage as ndi

    labeled, count = ndi.label(mask)
    if count == 0:
        return []
    sizes = np.bincount(labeled.ravel(), minlength=count + 1)
    order = [int(index) for index in np.argsort(sizes[1:])[::-1] + 1 if sizes[index] >= min_px]
    measured = []
    for index in order[:max_components]:
        ys, xs = np.nonzero(labeled == index)
        candidate = _pca_line(xs, ys)
        if candidate["elongation"] >= min_elongation:
            measured.append(candidate)
    return measured


def _bright_components(data_sub, threshold_sig, min_px, min_elongation, max_components):
    """Propose bright elongated components above a fixed sky-relative threshold.

    This is the proposer that actually finds real MegaCam trails.  The Hough
    prescreen and the production pipeline both rank detector artefacts above the
    satellite trail in the 2008/2017 fields measured here, while a bright trail
    is simply the brightest elongated connected component in the chip.  Kept
    deliberately outside ``weightmask`` so the proposer is not the code the
    fixture is used to score, and thresholded high (default 20 sigma) so it stays
    a high-precision miner rather than a second detector.
    """
    median = float(np.median(data_sub))
    sigma = _mad_std(data_sub) or 1.0
    return _components_from_mask(data_sub > median + threshold_sig * sigma, min_px, max_components, min_elongation)


def _column_and_row_noise(data_sub):
    """Robust per-column and per-row noise, for the axis veto."""
    column = data_sub - np.median(data_sub, axis=0, keepdims=True)
    row = data_sub - np.median(data_sub, axis=1, keepdims=True)
    return 1.4826 * np.median(np.abs(column), axis=0), 1.4826 * np.median(np.abs(row), axis=1)


def _dedupe_candidates(candidates, tol_px, tol_deg):
    """Drop candidates that repeat an earlier one's CCD-local line."""
    kept = []
    for candidate in candidates:
        duplicate = False
        for existing in kept:
            if _angle_between(candidate["normal"], existing["normal"]) > tol_deg:
                continue
            if abs(candidate["measured_offset"] - existing["measured_offset"]) > tol_px:
                continue
            duplicate = True
            break
        if not duplicate:
            kept.append(candidate)
    return kept


def mine_hdu(job):
    """Mine one (exposure, HDU): returns a serialisable candidate record."""
    import fitsio
    import yaml

    from weightmask.background import estimate_background
    from weightmask.streaks import (
        _candidate_from_rho_theta,
        _detect_streaks_houghpeaks,
        _detect_streaks_mrt_like,
        detect_streaks,
    )

    path, hdu, config_path, corridor_px, quiet, proposers, component_params = job
    (
        min_component_px,
        max_components,
        bright_threshold_sig,
        bright_min_px,
        bright_min_elongation,
        radon_max_candidates,
    ) = component_params
    sink = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
    with sink:
        config = yaml.safe_load(open(config_path))
        streak_cfg = dict(config["streak_masking"])
        streak_cfg["enable"] = True
        with fitsio.FITS(path) as handle:
            science = np.ascontiguousarray(handle[hdu].read().astype(np.float32))
            header = handle[hdu].read_header()

        record = {
            "hdu": int(hdu),
            "shape": [int(science.shape[0]), int(science.shape[1])],
            "ccdname": str(header.get("CCDNAME", "") or ""),
            "extver": int(header.get("EXTVER", -1)),
            "saturate": float(header.get("SATURATE", 0.0) or 0.0),
            "candidates": [],
            "wcs": None,
        }
        geom = geometry_from_header(header)
        if geom is not None:
            # ZNAXIS* hold the true uncompressed geometry of tiled-compressed HDUs.
            geom["shape"] = [
                int(header.get("ZNAXIS1", science.shape[1])),
                int(header.get("ZNAXIS2", science.shape[0])),
            ]
            record["wcs"] = geom

        existing = np.zeros(science.shape, dtype=bool)
        sky, rms = estimate_background(science, existing, config["sep_background"])
        data_sub = science - sky
        column_sigma, row_sigma = _column_and_row_noise(data_sub)

        candidates = []
        if "houghpeaks" in proposers:
            mask, accepted, _ = _detect_streaks_houghpeaks(data_sub, rms, existing, streak_cfg)
            cx, cy = 0.5 * (science.shape[1] - 1), 0.5 * (science.shape[0] - 1)
            for candidate in accepted:
                theta = math.radians(float(candidate["theta_deg"]))
                rho = float(candidate["rho"])
                measured = _measure_candidate(mask, (math.cos(theta), math.sin(theta)), rho, cx, cy, corridor_px)
                if measured is None:
                    continue
                measured["proposed_by"] = ["houghpeaks"]
                measured["detector"] = {
                    "theta_deg": float(candidate["theta_deg"]),
                    "rho": rho,
                    "confidence": float(candidate.get("confidence", float("nan"))),
                }
                candidates.append(measured)
        if "pipeline" in proposers:
            pipeline_mask = detect_streaks(data_sub, rms, existing, streak_cfg)
            for measured in _components_from_mask(pipeline_mask, min_component_px, max_components):
                measured["proposed_by"] = ["pipeline"]
                measured["detector"] = {}
                candidates.append(measured)
        if "bright" in proposers:
            for measured in _bright_components(
                data_sub, bright_threshold_sig, bright_min_px, bright_min_elongation, max_components
            ):
                measured["proposed_by"] = ["bright"]
                measured["detector"] = {}
                candidates.append(measured)
        if "radon" in proposers:
            # Orientation-agnostic: an exhaustive binned-Radon line search, so the
            # candidate set cannot be biased toward whatever connected components
            # or Hough bins happen to favour.  Uses the production rescue with its
            # accept gates opened, because curation -- not the rescue -- is the
            # filter here.
            rescue_cfg = dict(streak_cfg)
            rescue_cfg["mrt_rescue_params"] = {
                "enable": True,
                "theta_step_deg": 2.0,
                "peak_threshold_sig": 2.0,
                "max_candidates": radon_max_candidates,
                "confidence_threshold": 0.0,
                "bin": 4,
                "sigma_rel_floor": 0.001,
                "sinogram_highpass": 101,
            }
            _, rescue_accepted, _ = _detect_streaks_mrt_like(data_sub, rms, existing, rescue_cfg)
            for candidate in rescue_accepted:
                # Take the line straight from the production builder.  ``radon``
                # parameterises a line *reflected* relative to the Hesse convention
                # ``_candidate_from_rho_theta`` speaks, so rebuilding the line here
                # from (theta, rho) by hand reproduces the mirror bug the rescue
                # itself was fixed for -- and the mirrored line is what the
                # independent profile and the mosaic test would then be scored on.
                built = _candidate_from_rho_theta(
                    -float(candidate["rho"]), 180.0 - float(candidate["theta_deg"]), science.shape
                )
                if built is None:
                    continue
                (x0, y0), (x1, y1) = built["endpoints"]
                dx, dy = float(x1) - float(x0), float(y1) - float(y0)
                span = math.hypot(dx, dy)
                if span <= 0:
                    continue
                direction = (dx / span, dy / span)
                candidates.append(
                    {
                        "point": [0.5 * (float(x0) + float(x1)), 0.5 * (float(y0) + float(y1))],
                        "direction": [direction[0], direction[1]],
                        "normal": list(_normalise_normal(-direction[1], direction[0])),
                        "n_mask_px": 0,
                        "rms_along_px": 0.0,
                        "rms_across_px": 0.0,
                        "width_px": 0.0,
                        "length_px": span,
                        "span_along_px": span,
                        "elongation": float("nan"),
                        "proposed_by": ["radon"],
                        "detector": {
                            "theta_deg": float(candidate["theta_deg"]),
                            "rho": float(candidate["rho"]),
                            "peak_snr": float(candidate.get("peak_snr", float("nan"))),
                            "confidence": float(candidate.get("confidence", float("nan"))),
                        },
                    }
                )

    if not candidates:
        return record

    h, w = science.shape
    cx, cy = 0.5 * (w - 1), 0.5 * (h - 1)
    for candidate in candidates:
        candidate["measured_offset"] = float(
            candidate["normal"][0] * (candidate["point"][0] - cx)
            + candidate["normal"][1] * (candidate["point"][1] - cy)
        )

    candidates = _dedupe_candidates(candidates, tol_px=6.0, tol_deg=1.5)

    median_column = float(np.median(column_sigma)) or 1.0
    median_row = float(np.median(row_sigma)) or 1.0
    for candidate in candidates:
        candidate["independent"] = _independent_line_stats(
            data_sub, candidate["point"], candidate["direction"], science.shape
        )
        vertical_align = _angle_between(candidate["normal"], (1.0, 0.0))
        horizontal_align = _angle_between(candidate["normal"], (0.0, 1.0))
        candidate["axis_alignment_deg"] = float(min(vertical_align, horizontal_align))
        veto = None
        if vertical_align <= 1.5:
            x = int(round(candidate["point"][0]))
            if 0 <= x < w and column_sigma[x] > 3.0 * median_column:
                veto = {
                    "kind": "bad_column",
                    "coord": x,
                    "axis_sigma": float(column_sigma[x]),
                    "median_axis_sigma": median_column,
                }
        elif horizontal_align <= 1.5:
            y = int(round(candidate["point"][1]))
            if 0 <= y < h and row_sigma[y] > 3.0 * median_row:
                veto = {
                    "kind": "bad_row",
                    "coord": y,
                    "axis_sigma": float(row_sigma[y]),
                    "median_axis_sigma": median_row,
                }
        candidate["axis_veto"] = veto
        if geom is not None:
            mosaic = line_to_mosaic(candidate["point"], candidate["direction"], geom)
            if mosaic is not None:
                mosaic["offset_px"] = mosaic["offset_deg"] / DEG_PER_MOSAIC_PX
                clipped = _clip_line_to_shape(candidate["point"], candidate["direction"], science.shape)
                if clipped is not None:
                    normal = np.asarray(mosaic["normal"], dtype=np.float64)
                    along = np.array([normal[1], -normal[0]])
                    projected = ccd_to_mosaic(list(clipped), geom) @ along / DEG_PER_MOSAIC_PX
                    mosaic["t_range_px"] = [float(projected.min()), float(projected.max())]
                candidate["mosaic"] = mosaic
        record["candidates"].append(candidate)
    return record


# --------------------------------------------------------------------------- #
# grouping / curation
# --------------------------------------------------------------------------- #
def _connected_groups(members, max_gap_px):
    """Split members by where they actually sit *along* their shared line.

    Two chips can share one mosaic line while sitting on completely different
    parts of the sky: the line is infinite, so a bad column on chip A and a bad
    column on the mirrored chip B in the same mosaic column land on the same
    ``(normal, offset)`` while being thousands of pixels apart along it.  A real
    trail instead forms a contiguous chain, separated only by the chip gap.
    """
    positioned = [member for member in members if member.get("mosaic", {}).get("t_range_px")]
    without = [member for member in members if not member.get("mosaic", {}).get("t_range_px")]
    if len(positioned) <= 1:
        return [positioned] + [[member] for member in without] if positioned or without else []

    positioned = sorted(positioned, key=lambda member: member["mosaic"]["t_range_px"][0])
    chunks = [[positioned[0]]]
    reach = positioned[0]["mosaic"]["t_range_px"][1]
    for member in positioned[1:]:
        start, end = member["mosaic"]["t_range_px"]
        if start - reach > max_gap_px:
            chunks.append([member])
            reach = end
        else:
            chunks[-1].append(member)
            reach = max(reach, end)
    chunks.extend([member] for member in without)
    return chunks


def group_candidates(candidates, tol_deg, tol_px, max_gap_px):
    """Union-find candidates of one exposure whose mosaic lines coincide."""
    parent = list(range(len(candidates)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            left, right = candidates[i], candidates[j]
            if "mosaic" not in left or "mosaic" not in right:
                continue
            if _angle_between(left["mosaic"]["normal"], right["mosaic"]["normal"]) > tol_deg:
                continue
            if abs(left["mosaic"]["offset_px"] - right["mosaic"]["offset_px"]) > tol_px:
                continue
            root_j, root_i = find(j), find(i)
            if root_i != root_j:
                parent[root_j] = root_i

    groups = {}
    for i in range(len(candidates)):
        groups.setdefault(find(i), []).append(i)
    out = {}
    for root, indices in groups.items():
        for chunk_index, chunk in enumerate(_connected_groups([candidates[index] for index in indices], max_gap_px)):
            out[(root, chunk_index)] = chunk
    return out


def _same_ccd(left, right):
    if left.get("ccdname") and left.get("ccdname") == right.get("ccdname"):
        return True
    return left.get("extver", -1) == right.get("extver", -1) and left.get("extver", -1) >= 0


def find_static_matches(candidate, others, tol_deg, tol_px, min_support_px):
    """Candidates that reappear at the same CCD-local line in another epoch."""
    matches = []
    if (candidate["independent"] or {}).get("support_px", 0.0) < min_support_px:
        return matches
    for other in others:
        if not _same_ccd(candidate, other):
            continue
        if _angle_between(candidate["normal"], other["normal"]) > tol_deg:
            continue
        if abs(candidate["measured_offset"] - other["measured_offset"]) > tol_px:
            continue
        if (other["independent"] or {}).get("support_px", 0.0) < min_support_px:
            continue
        matches.append(other)
    return matches


def find_chip_replicas(members, tol_px, tol_deg):
    """Members of one mosaic group that share a CCD-local line.

    A sky line lands at a different CCD-local offset on each chip it crosses, so
    two members at the *same* CCD-local offset are replicated per chip -- which
    is how a chip edge or serial-register row gets collinear across the mosaic.
    """
    replicas = {}
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            left, right = members[i], members[j]
            if _angle_between(left["normal"], right["normal"]) > tol_deg:
                continue
            if abs(left["measured_offset"] - right["measured_offset"]) > tol_px:
                continue
            replicas.setdefault(i, []).append(j)
            replicas.setdefault(j, []).append(i)
    return replicas


def curate(grouped, args):
    """Apply the curation rule and return the labelled entries."""
    entries = []
    for group_id, members in grouped.items():
        ccds = sorted({member["ccdname"] or f"hdu{member['hdu']}" for member in members})
        replicas = find_chip_replicas(members, args.replica_tol_px, args.replica_tol_deg)
        group_note = {
            "id": group_id,
            "n_candidates": len(members),
            "ccds": ccds,
            "n_ccds": len(ccds),
            "chip_replicated": bool(replicas),
        }
        if replicas:
            group_note["chip_replica_offsets_px"] = sorted(
                round(abs(members[i]["measured_offset"] - members[j]["measured_offset"]), 2)
                for i, js in replicas.items()
                for j in js
            )
        offsets = [member["mosaic"]["offset_px"] for member in members if "mosaic" in member]
        if len(offsets) > 1:
            group_note["max_offset_spread_px"] = round(float(max(offsets) - min(offsets)), 3)

        members = list(enumerate(members))
        poisoned = bool(replicas)
        for _, member in members:
            if member.get("static_matches") or member.get("axis_veto"):
                poisoned = True

        for index, member in members:
            independent = member["independent"] or {}
            reasons = []
            own_veto = None
            if member.get("static_matches"):
                own_veto = "static_across_epochs"
            elif member.get("axis_veto"):
                own_veto = member["axis_veto"]["kind"]
            elif index in replicas:
                own_veto = "chip_replica"

            cross_ccd_ok = len(ccds) >= args.min_group_ccds
            independent_ok = (
                independent.get("support_px", 0.0) >= args.min_support_px
                and independent.get("z_max", 0.0) >= args.min_z_max
            )
            # A line within a couple of degrees of a CCD axis is exactly where
            # bad columns and serial-register rows live, and no single epoch can
            # tell such a trail from such a column.  Left to `uncertain` rather
            # than guessed at in either direction.
            axis_ok = member["axis_alignment_deg"] >= args.min_axis_deg

            if own_veto:
                label = "artefact"
            elif cross_ccd_ok and independent_ok and axis_ok and not poisoned:
                label = "trail"
            else:
                label = "uncertain"

            if own_veto:
                reasons.append(f"{own_veto} veto")
            if member.get("static_matches"):
                reasons.append(
                    "same CCD-local line in "
                    + ", ".join(sorted({match["exposure"] for match in member["static_matches"]}))
                )
            if index in replicas:
                reasons.append("CCD-local line repeated on another chip in this group")
            if replicas and index not in replicas:
                reasons.append("group line is chip-replicated")
            if cross_ccd_ok:
                reasons.append(f"mosaic line shared by {len(ccds)} CCDs")
            else:
                reasons.append("single-CCD group: no focal-plane corroboration")
            if not axis_ok:
                reasons.append(f"within {args.min_axis_deg} deg of a CCD axis")
            if independent_ok:
                reasons.append(
                    f"independent ridge support {independent.get('support_px', 0.0):.0f} px, peak {independent.get('z_max', 0.0):.1f} sigma"
                )
            else:
                reasons.append(
                    f"independent ridge too weak (support {independent.get('support_px', 0.0):.0f} px, peak {independent.get('z_max', 0.0):.1f} sigma)"
                )

            entries.append(
                {
                    "exposure": member["exposure"],
                    "hdu": member["hdu"],
                    "ccdname": member["ccdname"],
                    "label": label,
                    "proposed_by": member.get("proposed_by", []),
                    "geometry": {
                        "point": [round(member["point"][0], 3), round(member["point"][1], 3)],
                        "direction": [round(member["direction"][0], 6), round(member["direction"][1], 6)],
                        "normal": [round(member["normal"][0], 6), round(member["normal"][1], 6)],
                        "offset_px": round(member["measured_offset"], 3),
                        "span_along_px": round(member["span_along_px"], 2),
                        "length_px": round(member["length_px"], 2),
                        "width_px": round(member["width_px"], 2),
                        "elongation": round(member["elongation"], 2),
                        "n_mask_px": member["n_mask_px"],
                    },
                    "mosaic": {
                        "normal": [round(value, 9) for value in member["mosaic"]["normal"]],
                        "offset_deg": member["mosaic"]["offset_deg"],
                        "offset_px": round(member["mosaic"]["offset_px"], 3),
                        "group": group_note,
                    }
                    if "mosaic" in member
                    else None,
                    "detector": member["detector"],
                    "independent": independent,
                    "evidence": {
                        "axis_alignment_deg": round(member["axis_alignment_deg"], 3),
                        "axis_veto": member["axis_veto"],
                        "static_matches": [
                            {
                                "exposure": match["exposure"],
                                "hdu": match["hdu"],
                                "offset_px": round(match["measured_offset"], 3),
                                "support_px": round((match["independent"] or {}).get("support_px", 0.0), 1),
                            }
                            for match in member.get("static_matches", [])
                        ],
                    },
                    "review": {"status": "auto-curated", "reasons": reasons},
                }
            )
    entries.sort(key=lambda item: (item["exposure"], item["hdu"], item["geometry"]["offset_px"]))
    return entries


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def exposure_inventory(data_root, exposures):
    """Read every exposure's identity and HDU list without loading any pixels."""
    import fitsio

    inventory = {}
    missing = []
    for name in exposures:
        found = None
        for suffix in (".fits.fz", ".fits"):
            candidate = os.path.join(data_root, name + suffix)
            if os.path.exists(candidate):
                found = candidate
                break
        if found is None:
            missing.append(name)
            continue
        with fitsio.FITS(found) as handle:
            header = handle[0].read_header()
            hdus = []
            for index in range(1, len(handle)):
                try:
                    hdu_header = handle[index].read_header()
                except (OSError, RuntimeError, ValueError):
                    continue
                if int(hdu_header.get("NAXIS", 0)) != 2:
                    continue
                hdus.append(index)
            inventory[name] = {
                "path": found,
                "object": str(header.get("OBJECT", "") or ""),
                "date_obs": str(header.get("DATE-OBS", "") or ""),
                "exptime": float(header.get("EXPTIME", 0.0) or 0.0),
                "filter": str(header.get("FILTER", "") or ""),
                "crval": [float(header.get("CRVAL1", float("nan"))), float(header.get("CRVAL2", float("nan")))],
                "nhdu": len(hdus),
                "hdus": hdus,
            }
    return inventory, missing


def pointing_groups(inventory):
    """Group exposures by CRVAL: identical pointing means identical WCS."""
    groups = {}
    for name, info in sorted(inventory.items()):
        key = f"{info['object']}[{info['crval'][0]:.4f},{info['crval'][1]:.4f}]"
        groups.setdefault(key, []).append(name)
    return groups


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="directory holding the exposure files")
    parser.add_argument("--exposures", help="comma-separated root names (default: every exposure under --data-root)")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--review-sheet", default=DEFAULT_REVIEW_SHEET)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--propose",
        default="bright,houghpeaks,radon",
        help="comma-separated proposers from {bright,houghpeaks,pipeline,radon}; 'pipeline' runs the production detect_streaks",
    )
    parser.add_argument("--corridor-px", type=float, default=30.0, help="mask-pixel corridor for the PCA re-measure")
    parser.add_argument("--min-component-px", type=int, default=400, help="smallest pipeline mask component to measure")
    parser.add_argument("--max-candidates", type=int, default=6, help="components measured per HDU")
    parser.add_argument(
        "--bright-threshold-sig", type=float, default=20.0, help="bright proposer threshold, in chip sigma"
    )
    parser.add_argument("--bright-min-px", type=int, default=150, help="bright proposer minimum component area")
    parser.add_argument("--bright-min-elongation", type=float, default=6.0, help="bright proposer minimum elongation")
    parser.add_argument(
        "--radon-max-candidates",
        type=int,
        default=16,
        help="Radon proposer lines per HDU; each is measured with a full along-line profile, so this sets the run cost",
    )
    parser.add_argument("--group-tol-deg", type=float, default=2.0, help="mosaic normal tolerance for grouping")
    parser.add_argument("--group-tol-px", type=float, default=15.0, help="mosaic offset tolerance (pixels)")
    parser.add_argument("--min-group-ccds", type=int, default=2, help="CCDs that must share a mosaic line")
    parser.add_argument("--min-support-px", type=float, default=200.0, help="independent ridge extent floor")
    parser.add_argument("--min-z-max", type=float, default=6.0, help="independent ridge peak floor, in off-line sigma")
    parser.add_argument(
        "--min-axis-deg",
        type=float,
        default=5.0,
        help=(
            "a `trail` must be at least this far from both CCD axes. The lines within a few degrees "
            "of a CCD axis are dominated by bad columns and register rows, and the Radon proposer's "
            "2 deg angle grid cannot resolve a near-axis trail from such a column -- measured, not "
            "assumed: candidates at exactly 2.00/4.00 deg were grid-quantised columns. Excluding the "
            "first 5 deg costs at most 5/180 of possible orientations."
        ),
    )
    parser.add_argument(
        "--group-max-gap-px",
        type=float,
        default=800.0,
        help="largest gap along the shared line that still counts as one physical chain",
    )
    parser.add_argument(
        "--replica-tol-px", type=float, default=8.0, help="CCD-local offset tolerance for chip replication"
    )
    parser.add_argument(
        "--replica-tol-deg", type=float, default=2.0, help="CCD-local angle tolerance for chip replication"
    )
    parser.add_argument("--static-tol-deg", type=float, default=1.5)
    parser.add_argument("--static-tol-px", type=float, default=25.0)
    parser.add_argument(
        "--static-scope",
        choices=["dataset", "pointing"],
        default="dataset",
        help="'dataset' compares every exposure sharing a CCD (a bad column is a property of the chip, so it needs no shared WCS); 'pointing' only same-CRVAL epochs",
    )
    parser.add_argument(
        "--static-min-epochs",
        type=int,
        default=2,
        help="distinct other exposures that must show the same CCD-local ridge before it counts as detector-fixed",
    )
    parser.add_argument(
        "--entries",
        choices=["labelled", "all"],
        default="labelled",
        help="which entries the JSON fixture carries; the review sheet always lists every entry",
    )
    parser.add_argument("--max-hdus", type=int, default=0, help="debug: cap HDUs per exposure")
    parser.add_argument("--no-veto-persistence", action="store_true", help="debug: skip the static-structure layer")
    parser.add_argument("--quiet", action="store_true", default=True)
    parser.add_argument("--verbose", dest="quiet", action="store_false")
    args = parser.parse_args(argv)

    if args.exposures:
        wanted = [name.strip() for name in args.exposures.split(",") if name.strip()]
    else:
        wanted = sorted(
            os.path.splitext(os.path.splitext(name)[0])[0]
            for name in os.listdir(args.data_root)
            if name.endswith((".fits.fz", ".fits")) and not name.startswith("flat_")
        )
    if not wanted:
        parser.error(f"no exposures found under {args.data_root}")

    proposers = [item.strip() for item in args.propose.split(",") if item.strip()]
    unknown = sorted(set(proposers) - {"bright", "houghpeaks", "pipeline", "radon"})
    if unknown:
        parser.error(f"unknown proposer(s): {', '.join(unknown)}")
    if not proposers:
        parser.error("--propose must name at least one proposer")

    inventory, missing = exposure_inventory(args.data_root, wanted)
    if missing:
        print(f"ERROR: no file found for exposures: {', '.join(missing)}", file=sys.stderr)
        return 2
    if not inventory:
        print(f"ERROR: nothing to mine under {args.data_root}", file=sys.stderr)
        return 2

    plan = []
    for name, info in sorted(inventory.items()):
        hdus = info["hdus"][: args.max_hdus] if args.max_hdus else info["hdus"]
        plan.extend((name, hdu) for hdu in hdus)
    component_params = (
        args.min_component_px,
        args.max_candidates,
        args.bright_threshold_sig,
        args.bright_min_px,
        args.bright_min_elongation,
        args.radon_max_candidates,
    )
    jobs = [
        (
            inventory[name]["path"],
            hdu,
            args.config,
            args.corridor_px,
            args.quiet,
            proposers,
            component_params,
        )
        for name, hdu in plan
    ]

    print(f"Mining {len(plan)} HDU record(s) from {len(inventory)} exposure(s), proposers={','.join(proposers)}")
    started = time.time()
    records = {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for (name, hdu), record in zip(plan, pool.map(mine_hdu, jobs, chunksize=1)):
            record["exposure"] = name
            records[(name, hdu)] = record
    print(f"Mined in {time.time() - started:.0f} s")

    n_wcs = sum(1 for record in records.values() if record["wcs"])
    n_hit = sum(1 for record in records.values() if record["candidates"])
    n_cand = sum(len(record["candidates"]) for record in records.values())
    print(f"  {n_cand} candidate(s) on {n_hit} HDU(s); {n_wcs}/{len(records)} HDU(s) carry the mosaic WCS")

    flat = []
    for record in records.values():
        for candidate in record["candidates"]:
            candidate["exposure"] = record["exposure"]
            candidate["hdu"] = record["hdu"]
            candidate["ccdname"] = record["ccdname"]
            candidate["extver"] = record["extver"]
            candidate["static_matches"] = []
            flat.append(candidate)

    groups = pointing_groups(inventory)

    # Layer 3: a candidate that reappears at the same CCD-local line in other
    # exposures *of that same chip* is detector-fixed, not sky-borne.  A bad
    # column is a property of the chip, so this needs no shared WCS and is
    # strictly stronger than only comparing same-pointing epochs.  Requiring
    # several corroborating exposures keeps an unrelated trail or star that
    # happens to lie on the line in one other frame from vetoing a real trail.
    if not args.no_veto_persistence:
        if args.static_scope == "dataset":
            by_chip = {}
            for item in flat:
                by_chip.setdefault(item["ccdname"] or f"hdu{item['hdu']}", []).append(item)
            vetoed_entries = 0
            for items in by_chip.values():
                for candidate in items:
                    others = [other for other in items if other["exposure"] != candidate["exposure"]]
                    matches = find_static_matches(
                        candidate, others, args.static_tol_deg, args.static_tol_px, args.min_support_px
                    )
                    exposures = {match["exposure"] for match in matches}
                    if len(exposures) >= args.static_min_epochs:
                        candidate["static_matches"] = matches
                        vetoed_entries += 1
            print(
                f"  static-structure layer (dataset scope, >= {args.static_min_epochs} other exposures): "
                f"{vetoed_entries} candidate(s) shown detector-fixed"
            )
        else:
            matches = 0
            for names in groups.values():
                if len(names) < 2:
                    continue
                for name in names:
                    others = [other for other in names if other != name]
                    for candidate in [item for item in flat if item["exposure"] == name]:
                        same_hdu = [
                            item for item in flat if item["exposure"] in others and item["hdu"] == candidate["hdu"]
                        ]
                        found = find_static_matches(
                            candidate, same_hdu, args.static_tol_deg, args.static_tol_px, args.min_support_px
                        )
                        if len({match["exposure"] for match in found}) >= args.static_min_epochs:
                            candidate["static_matches"] = found
                        matches += len(found)
            print(f"  static-structure layer (pointing scope): {matches} same-line match(es)")

    # Layer 2: group coincident mosaic lines *within* each exposure.
    grouped = {}
    for name in sorted(inventory):
        subset = [item for item in flat if item["exposure"] == name]
        for key, members in group_candidates(
            subset, args.group_tol_deg, args.group_tol_px, args.group_max_gap_px
        ).items():
            grouped[f"{name}#{key}"] = members

    entries = curate(grouped, args)

    summary = {
        "n_entries": len(entries),
        "by_label": {
            label: sum(1 for entry in entries if entry["label"] == label)
            for label in ("trail", "artefact", "uncertain")
        },
        "by_veto": {},
        "by_proposer": {},
        "by_exposure": {},
    }
    for entry in entries:
        bucket = summary["by_exposure"].setdefault(entry["exposure"], {"trail": 0, "artefact": 0, "uncertain": 0})
        bucket[entry["label"]] += 1
        key = "+".join(entry["proposed_by"]) or "unknown"
        summary["by_proposer"][key] = summary["by_proposer"].get(key, 0) + 1
        if entry["label"] == "artefact":
            if entry["evidence"]["static_matches"]:
                veto = "static_across_epochs"
            elif entry["evidence"]["axis_veto"]:
                veto = entry["evidence"]["axis_veto"]["kind"]
            elif entry["mosaic"] and entry["mosaic"]["group"]["chip_replicated"]:
                veto = "chip_replica"
            else:
                veto = "unattributed"
            summary["by_veto"][veto] = summary["by_veto"].get(veto, 0) + 1

    committed = [entry for entry in entries if entry["label"] != "uncertain"] if args.entries == "labelled" else entries
    if args.entries == "labelled":
        print(
            f"  fixture carries the {len(committed)} labelled entries; "
            f"{summary['by_label']['uncertain']} uncertain entries are in the review sheet only"
        )

    fixture = {
        "schema": SCHEMA,
        "instrument": "MegaPrime/MegaCam",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": {
            "propose": f"proposers={'+'.join(proposers)}; geometry re-measured by PCA of the pixels behind each candidate",
            "cross_ccd": "shared-CRVAL per-CCD TAN WCS maps every CCD into one tangent plane; the gnomonic projection maps great circles to straight lines, so a real trail is one collinear line on every CCD it crosses",
            "chip_replica": "a sky line lands at a different CCD-local offset on each chip, so members sharing a CCD-local line are replicated per chip (chip edges, serial-register rows)",
            "static_structure": "identical CRVAL means identical WCS, so detector-fixed structure repeats at the same CCD pixels in another epoch of the same pointing; real trails move",
            "axis_veto": "a candidate sitting on a column/row whose robust noise is an outlier for that chip, and (for a `trail`) the min-axis-deg exclusion",
            "independent_measurement": "bilinear profile along the candidate line against a background band 25-60 px off-line in raw counts; support_px is the longest run above 4 sigma",
        },
        "tolerances": {
            "propose": proposers,
            "corridor_px": args.corridor_px,
            "min_component_px": args.min_component_px,
            "bright_threshold_sig": args.bright_threshold_sig,
            "bright_min_px": args.bright_min_px,
            "bright_min_elongation": args.bright_min_elongation,
            "radon_max_candidates": args.radon_max_candidates,
            "group_tol_deg": args.group_tol_deg,
            "group_tol_px": args.group_tol_px,
            "min_group_ccds": args.min_group_ccds,
            "min_support_px": args.min_support_px,
            "min_z_max": args.min_z_max,
            "min_axis_deg": args.min_axis_deg,
            "group_max_gap_px": args.group_max_gap_px,
            "entries": args.entries,
            "replica_tol_px": args.replica_tol_px,
            "replica_tol_deg": args.replica_tol_deg,
            "static_tol_deg": args.static_tol_deg,
            "static_tol_px": args.static_tol_px,
            "deg_per_mosaic_px": DEG_PER_MOSAIC_PX,
        },
        "data_requirements": {
            "data_root": args.data_root,
            "exposures": {
                name: {
                    "path": info["path"],
                    "object": info["object"],
                    "date_obs": info["date_obs"],
                    "exptime": info["exptime"],
                    "filter": info["filter"],
                    "crval": info["crval"],
                    "nhdu": info["nhdu"],
                    "pointing_group": next(key for key, names in groups.items() if name in names),
                }
                for name, info in sorted(inventory.items())
            },
            "note": "The MegaCam exposures are not committed (benchmark_data/ is gitignored); this fixture stores geometry and provenance only.",
        },
        "summary": summary,
        "entries": committed,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_fixture(fixture, args.out)
    print(f"Wrote {args.out}: {summary['by_label']}")

    if args.review_sheet:
        os.makedirs(os.path.dirname(os.path.abspath(args.review_sheet)), exist_ok=True)
        with open(args.review_sheet, "w") as handle:
            handle.write(review_sheet(fixture))
        print(f"Wrote {args.review_sheet}")

    if summary["by_label"]["trail"] == 0:
        print("WARNING: no entry passed the trail rule; nothing is scored as ground truth.", file=sys.stderr)
    return 0


def write_fixture(fixture, path):
    """Write the fixture with one compact entry per line.

    The entry list is the part that grows, and a reviewer needs to see a changed
    label as a changed line rather than as a field somewhere in a 22k-line pretty
    print.  Everything else stays indented and readable.
    """
    head = {key: value for key, value in fixture.items() if key != "entries"}
    body = json.dumps(head, indent=2)
    assert body.endswith("\n}") or body.endswith("}"), "unexpected json layout"
    if body.endswith("\n}"):
        prefix = body[:-2]
    else:
        prefix = body[:-1]
    prefix = prefix.rstrip()
    if not prefix.endswith("{"):
        prefix += ","
    with open(path, "w") as handle:
        handle.write(prefix + "\n")
        if not prefix.endswith("{"):
            handle.write('  "entries": [\n')
        else:
            handle.write('  "entries": [\n')
        for index, entry in enumerate(fixture["entries"]):
            comma = "," if index + 1 < len(fixture["entries"]) else ""
            handle.write("    " + json.dumps(entry, separators=(",", ":")) + comma + "\n")
        handle.write("  ]\n}\n")


def review_sheet(fixture) -> str:
    """Render the fixture as a table a human can sign off row by row."""
    lines = [
        "# MegaCam real-trail ground truth -- review sheet",
        "",
        f"Generated {fixture['generated_utc']} from `{fixture['data_requirements']['data_root']}`",
        f"(proposers `{'+'.join(fixture['tolerances']['propose'])}`).",
        "",
        "Labels come from the rule in `benchmarks/curate_trail_truth.py`; the evidence columns are what",
        "it decided on. `trail` rows are scored as ground truth, `artefact` rows as scored negatives,",
        f"`uncertain` rows are excluded. Summary: `{json.dumps(fixture['summary']['by_label'])}`",
        "",
        "| label | exposure | hdu | CCD | group | CCDs | repl | spread px | axis deg | support px | peak sig | width px | span px | static | veto | by | reasons |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for entry in fixture["entries"]:
        mosaic = entry["mosaic"]
        group = mosaic["group"] if mosaic else {"id": "-", "n_ccds": 0, "chip_replicated": False}
        independent = entry["independent"] or {}
        veto = entry["evidence"]["axis_veto"]
        lines.append(
            "| {label} | {exposure} | {hdu} | {ccd} | {group} | {n} | {repl} | {spread} | {align:.2f} | {support:.0f} | {peak:.1f} | {width:.1f} | {span:.0f} | {static} | {veto} | {by} | {reasons} |".format(
                label=entry["label"],
                exposure=entry["exposure"],
                hdu=entry["hdu"],
                ccd=entry["ccdname"] or "-",
                group=str(group["id"]),
                n=group["n_ccds"],
                repl="yes" if group.get("chip_replicated") else "-",
                spread=group.get("max_offset_spread_px"),
                align=entry["evidence"]["axis_alignment_deg"],
                support=independent.get("support_px", 0.0),
                peak=independent.get("z_max", 0.0),
                width=entry["geometry"]["width_px"],
                span=entry["geometry"]["span_along_px"],
                static=len(entry["evidence"]["static_matches"]),
                veto=veto["kind"] if veto else "-",
                by="+".join(entry["proposed_by"]) or "-",
                reasons="; ".join(entry["review"]["reasons"]).replace("|", "/"),
            )
        )
    multi = {}
    for entry in fixture["entries"]:
        if entry["mosaic"] and entry["mosaic"]["group"]["n_ccds"] > 1:
            multi[entry["mosaic"]["group"]["id"]] = entry["mosaic"]["group"]
    lines += ["", "## Multi-CCD mosaic groups", ""]
    if not multi:
        lines.append("None: no candidate's mosaic line was shared by two CCDs.")
    for key, group in sorted(multi.items()):
        lines.append(
            f"- `{key}`: {group['n_ccds']} CCDs ({', '.join(group['ccds'])}), "
            f"max mosaic offset spread {group.get('max_offset_spread_px')} px, "
            f"chip-replicated: {group.get('chip_replicated')}"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
