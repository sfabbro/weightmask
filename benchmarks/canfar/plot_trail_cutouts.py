#!/usr/bin/env python3
"""Show what the streak stage did: real-data cutouts beside the mask it wrote.

One image per exposure, up to N cutouts each: a crop of the *real* frame
centred on a detected trail on the left, and the same crop of the quality mask
on the right with one colour per defect class. The left panel is zscale
stretched over its own pixels, so the crop is judged at the contrast a reviewer
would pick by hand rather than at min/max, where a handful of saturated stars
flattens the trail.

Why this runs in a pod: the masks live on the shared mount and the exposures
are a CADC download (a pod is where that egress is), so the PNGs and the index
are written to the mount and the exposure is cached on scratch -- `--work` keeps
a re-run from downloading it twice.

Usage (in a pod, from the staged tree):
    {py} benchmarks/canfar/plot_trail_cutouts.py --root ROOT --out DIR

    --root DIR      results root holding ns-<obsid>/ artifacts (WM_RESULTS_ROOT)
    --out DIR       destination for the PNGs and index.json (<root>/ns-review/cutouts)
    --obsids LIST   comma-separated; default: every ns-<obsid> that has a mask
    --work DIR      scratch cache for the exposures
    --cutouts N     cutouts per exposure (default 3)
    --size N        cutout side in pixels (default 256)
    --contrast F    zscale contrast (default 0.25, astropy's own default)
    --download      fetch an exposure from CADC when it is not already cached
    --margin F      a candidate window is only drawn when the masked line's 90th
                    percentile is at least this multiple of the window's
                    brightest clean pixels (default 1.0). A window that fails
                    is reported under "skipped" instead of being published as
                    a trail.
    --detected      also paint the DETECTED bit. Off by default: DETECTED is a
                    source catalogue that covers ~11% of a frame, not a defect
                    class, and it hides the classes that are.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from weightmask.contract import QUALITY_BITS

# Painted in this order, first class wins a pixel: a pixel can carry several
# bits, and the review question is "what did the streak stage do", so STREAK is
# never overpainted by a coincident CR or saturation flag.
CLASS_COLORS = {
    "STREAK": (0.84, 0.09, 0.11),
    "CR": (0.09, 0.55, 0.95),
    "SAT": (1.0, 0.72, 0.0),
    "BAD": (0.62, 0.13, 0.72),
    "NO_DATA": (0.55, 0.55, 0.58),
    "INVALID_VARIANCE": (0.0, 0.68, 0.42),
    "DETECTED": (0.86, 0.86, 0.86),
}
BACKGROUND = (1.0, 1.0, 1.0)
LEGEND_ORDER = ("STREAK", "CR", "SAT", "BAD", "NO_DATA", "INVALID_VARIANCE", "DETECTED")


def paint_classes(mask, *, detected=False, bits=None):
    """RGB image of one mask cutout, one colour per defect class.

    ``detected=False`` leaves the DETECTED bit unpainted (background): it is a
    source catalogue rather than a defect, and painting it would cover the
    classes a reviewer is looking for.
    """
    import numpy as np

    bits = bits or QUALITY_BITS
    height, width = mask.shape
    image = np.empty((height, width, 3), dtype=float)
    image[...] = BACKGROUND
    order = list(LEGEND_ORDER if detected else [name for name in LEGEND_ORDER if name != "DETECTED"])
    for name in reversed(order):
        bit = int(bits[name])
        image[(mask & bit) != 0] = CLASS_COLORS[name]
    return image


def class_counts(mask, *, bits=None):
    """How many pixels of each class are inside a cutout, painted bits included."""
    import numpy as np

    bits = bits or QUALITY_BITS
    return {name: int(np.count_nonzero(mask & bit)) for name, bit in bits.items()}


def trail_candidates(mask, *, min_pixels=25, bits=None):
    """Component candidates in one HDU's mask, longest first.

    The trail is broken into segments wherever the mask breaks, so segments are
    ranked by their long axis and a reviewer sees the parts of one trail rather
    than a scatter of equally-weighted fragments. ``span`` is the bounding box's
    long side: cheap, monotone in how long the segment is, and it does not need
    a principal-axis fit to be useful for ranking.
    """
    import numpy as np
    from scipy import ndimage

    bits = bits or QUALITY_BITS
    streak = (mask & int(bits["STREAK"])) != 0
    labels, count = ndimage.label(streak, structure=np.ones((3, 3), dtype=int))
    if count == 0:
        return []
    candidates = []
    for index, slices in enumerate(ndimage.find_objects(labels), start=1):
        if slices is None:
            continue
        ys, xs = slices
        selected = labels[slices] == index
        npix = int(np.count_nonzero(selected))
        if npix < min_pixels:
            continue
        local_y, local_x = np.nonzero(selected)
        rows = local_y + ys.start
        cols = local_x + xs.start
        height = ys.stop - ys.start
        width = xs.stop - xs.start
        candidates.append(
            {
                "npix": npix,
                "span": int(max(height, width)),
                "centroid": [float(rows.mean()), float(cols.mean())],
                "bounds": [int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())],
            }
        )
    candidates.sort(key=lambda item: (-item["span"], -item["npix"]))
    return candidates


def select_cutouts(per_hdu, *, count, size, overlap=0.5):
    """Pick up to ``count`` candidate windows, longest segment first.

    ``per_hdu`` maps a mask HDU index to its ranked candidates, and they are
    merged before selection: taking them HDU by HDU would spend every panel on
    the first HDU, while sorting by span interleaves the HDUs and a sheet ends
    up showing the longest trails the mask actually has. A candidate whose
    window overlaps an already-chosen one by more than ``overlap`` is dropped
    rather than spending a panel on the same piece of sky twice.
    """
    merged = []
    for key, candidates in per_hdu.items():
        for candidate in candidates:
            merged.append(dict(candidate, hdu=key))
    merged.sort(key=lambda item: (-item["span"], -item["npix"], item["hdu"]))
    picks = []
    for candidate in merged:
        if len(picks) >= count:
            break
        y, x = candidate["centroid"]
        y0, x0 = y - size / 2.0, x - size / 2.0
        clash = False
        for pick in picks:
            qy, qx = pick["centroid"]
            qy0, qx0 = qy - size / 2.0, qx - size / 2.0
            overlap_y = max(0.0, min(y0 + size, qy0 + size) - max(y0, qy0))
            overlap_x = max(0.0, min(x0 + size, qx0 + size) - max(x0, qx0))
            if overlap_y * overlap_x > overlap * size * size:
                clash = True
                break
        if clash:
            continue
        picks.append(candidate)
    return picks


def signal_prominence(data, mask, *, bits=None):
    """How much brighter the masked trail is than the clean sky in the cutout.

    Medians are the wrong probe here and were measured to be: an intermittent
    trail is dashed, the mask closes the gaps, so half the masked pixels are
    sky and the masked median sits on the sky median (measured 2026-10-14 on
    1851960 HDU 17: 574.3 vs 571.2). What the eye sees in the left panel is the
    ridge, so that is what is reported -- the 90th percentile of the masked
    pixels against the window's brightest clean pixels (its 99.9th percentile),
    plus how much of the masked line clears that bar at all.
    """
    import numpy as np

    bits = bits or QUALITY_BITS
    finite = np.isfinite(data)
    streak = ((mask & int(bits["STREAK"])) != 0) & finite
    clean = (mask == 0) & finite
    if not streak.any() or not clean.any():
        return {"sky_median": None, "sky_p999": None, "trail_p50": None, "trail_p90": None, "ridge": None}
    trail = data[streak]
    sky = data[clean]
    sky_median = float(np.median(sky))
    sky_high = float(np.percentile(sky, 99.9))
    trail_p90 = float(np.percentile(trail, 90))
    return {
        "sky_median": round(sky_median, 2),
        "sky_p999": round(sky_high, 2),
        "trail_p50": round(float(np.median(trail)), 2),
        "trail_p90": round(trail_p90, 2),
        "ridge": round(trail_p90 / sky_high, 3) if sky_high else None,
        "bright_fraction": round(float(np.mean(trail > sky_high)), 4),
    }


def window(centre, size, shape):
    """A ``size`` x ``size`` window around ``centre``, kept inside ``shape``."""
    height, width = shape
    y0 = int(round(centre[0] - size / 2.0))
    x0 = int(round(centre[1] - size / 2.0))
    y0 = max(0, min(y0, max(0, height - size)))
    x0 = max(0, min(x0, max(0, width - size)))
    return y0, x0, min(height, y0 + size), min(width, x0 + size)


def zscale_limits(data, *, contrast=0.25):
    """The zscale interval of a cutout, or (0, 1) when there is nothing finite."""
    import numpy as np
    from astropy.visualization import ZScaleInterval

    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = ZScaleInterval(contrast=contrast).get_limits(finite)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return float(np.min(finite)), float(np.max(finite)) or 1.0
    return float(low), float(high)


def grey_panel(cutout, *, low, high):
    """0..1 greyscale of a real-data cutout, clipped to the zscale interval."""
    import numpy as np

    scaled = (np.asarray(cutout, dtype=float) - low) / (high - low)
    scaled[~np.isfinite(scaled)] = 0.0
    return np.clip(scaled, 0.0, 1.0)


def exposure_hdu_for(extname, hdus, ccdname, position):
    """Which exposure HDU a mask HDU describes, and by which route.

    Two routes, deliberately: the mask names the detector it was built from
    (``MASK_<CCDNAME>``) and the driver recorded the science HDU list it
    processed in order. The name is used when it matches, and the recorded order
    is the fallback; the route travels with the image so a reviewer can see
    which one a panel rests on.
    """
    wanted = str(extname or "").replace("MASK_", "").strip()
    if wanted and wanted in ccdname:
        return int(ccdname[wanted]), "ccdname"
    if position < len(hdus):
        return int(hdus[position]), "order"
    return None, "none"


def mask_hdu_indices(path):
    """(mask HDU index, EXTNAME) for every image HDU that holds a mask."""
    import fitsio

    entries = []
    with fitsio.FITS(str(path)) as handle:
        for index in range(1, len(handle)):
            info = handle[index].get_info()
            if len(info.get("dims") or []) != 2:
                continue
            header = handle[index].read_header()
            entries.append((index, str(header.get("EXTNAME", "")).strip()))
    return entries


def render(
    obsid,
    mask_path,
    exposure_path,
    hdus,
    out_path,
    *,
    count=3,
    size=256,
    contrast=0.25,
    detected=False,
    margin=1.0,
):
    """Write one exposure's cutout sheet, and return the index entry describing it."""
    import fitsio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    entries = mask_hdu_indices(mask_path)
    if not entries:
        raise ValueError(f"no mask HDUs in {mask_path}")

    with fitsio.FITS(str(mask_path)) as mask_handle:
        mask_hdus = {}
        for index, _extname in entries:
            mask_hdus[index] = mask_handle[index].read()
    with fitsio.FITS(str(exposure_path)) as exposure_handle:
        ccdname = {}
        for index in range(1, len(exposure_handle)):
            header = exposure_handle[index].read_header()
            name = str(header.get("CCDNAME", "")).strip()
            if name:
                ccdname.setdefault(name, index)

        per_hdu = {}
        mapping = {}
        for position, (index, extname) in enumerate(entries):
            hdu, route = exposure_hdu_for(extname, hdus, ccdname, position)
            if hdu is None:
                continue
            mapping[str(index)] = {"hdu": int(hdu), "extname": extname, "route": route}
            per_hdu[index] = trail_candidates(mask_hdus[index], bits=QUALITY_BITS)
        # More candidates than panels: a window whose masked line turns out to
        # sit on the sky is skipped rather than published as "a trail".
        picks = select_cutouts(per_hdu, count=max(count * 4, count), size=size)

        panels = []
        skipped = []
        for pick in picks:
            if len(panels) >= count:
                break
            index = pick["hdu"]
            info = mapping[str(index)]
            hdu = info["hdu"]
            mask = mask_hdus[index]
            y0, x0, y1, x1 = window(pick["centroid"], size, mask.shape)
            mask_cut = mask[y0:y1, x0:x1]
            data_cut = exposure_handle[hdu].read()[y0:y1, x0:x1]
            signal = signal_prominence(data_cut, mask_cut)
            ridge = signal.get("ridge")
            if ridge is None or ridge < margin:
                skipped.append(
                    {
                        "hdu": int(hdu),
                        "window": [y0, x0, y1, x1],
                        "npix": pick["npix"],
                        "span": pick["span"],
                        "ridge": ridge,
                    }
                )
                continue
            low, high = zscale_limits(data_cut, contrast=contrast)
            extname = str(exposure_handle[hdu].read_header().get("EXTNAME", "")).strip()
            panels.append(
                {
                    "hdu": int(hdu),
                    "mask_hdu": int(index),
                    "extname": extname,
                    "ccdname": info["extname"].replace("MASK_", ""),
                    "route": info["route"],
                    "window": [y0, x0, y1, x1],
                    "centroid": [round(value, 1) for value in pick["centroid"]],
                    "span": pick["span"],
                    "npix": pick["npix"],
                    "zscale": [round(low, 3), round(high, 3)],
                    "classes": class_counts(mask_cut),
                    "signal": signal,
                    "grey": grey_panel(data_cut, low=low, high=high),
                    "rgb": paint_classes(mask_cut, detected=detected),
                }
            )

    if not panels:
        raise ValueError(f"no trail cutouts for {obsid}: the mask has no STREAK pixels")

    rows = len(panels)
    figure, axes = plt.subplots(rows, 2, figsize=(2 * size / 100.0, rows * size / 100.0), dpi=100, squeeze=False)
    figure.patch.set_facecolor("white")
    for row, panel in enumerate(panels):
        left, right = axes[row][0], axes[row][1]
        left.imshow(panel["grey"], cmap="gray", origin="lower", vmin=0.0, vmax=1.0, interpolation="nearest")
        left.set_title(
            f"{obsid}  data  HDU {panel['hdu']} ({panel['extname']}, {panel['ccdname']})  "
            f"y={panel['window'][0]}, x={panel['window'][1]}  zscale {panel['zscale'][0]:.0f}..{panel['zscale'][1]:.0f}",
            fontsize=6,
        )
        right.imshow(panel["rgb"], origin="lower", vmin=0.0, vmax=1.0, interpolation="nearest")
        right.set_title(
            f"mask  {os.path.basename(str(mask_path))}  streaks {panel['classes']['STREAK']} px "
            f"(segment {panel['npix']} px, span {panel['span']} px)",
            fontsize=6,
        )
        for axis in (left, right):
            axis.set_xticks([])
            axis.set_yticks([])
    handles = [
        Patch(facecolor=CLASS_COLORS[name], edgecolor="0.3", label=name) for name in LEGEND_ORDER if name != "DETECTED"
    ]
    if detected:
        handles.append(Patch(facecolor=CLASS_COLORS["DETECTED"], edgecolor="0.3", label="DETECTED (sources)"))
    figure.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=6, frameon=False)
    figure.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.04, wspace=0.02, hspace=0.12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=100, facecolor="white")
    plt.close(figure)

    return {
        "obsid": obsid,
        "mask": str(mask_path),
        "exposure": str(exposure_path),
        "png": out_path.name,
        "cutouts": [{key: value for key, value in panel.items() if key not in ("grey", "rgb")} for panel in panels],
        "skipped": skipped,
        "mapping": mapping,
    }


# A cache entry smaller than this is a partial download, not an exposure: the
# same floor process_exposure.py applies before it trusts a file on scratch.
MIN_EXPOSURE_BYTES = 1024 * 1024


def exposure_cache(work, obsid):
    return Path(work) / obsid / f"{obsid}p.fits.fz"


def ensure_exposure(work, obsid, record, *, download):
    """Return the cached exposure path, downloading it when allowed.

    ``record['url']`` is the address the processing run actually verified, so it
    is tried first: re-probing CADC for a name that already answered 200/206 is
    another chance to lose the run to a transient.
    """
    path = exposure_cache(work, obsid)
    if path.exists() and path.stat().st_size > MIN_EXPOSURE_BYTES:
        print(f"[cutouts] cached exposure {path} ({path.stat().st_size} B)", flush=True)
        return path
    if not download:
        return None
    try:
        from benchmarks.canfar import pod_run, process_exposure
    except ImportError:  # pragma: no cover - the pod's layout
        import pod_run  # type: ignore[no-redef]
        import process_exposure  # type: ignore[no-redef]

    fetch = pod_run.fetch
    url = record.get("url", "")
    if not url or process_exposure.probe(url) not in (200, 206):
        url = ""
        for candidate in process_exposure.candidate_urls(obsid):
            status = process_exposure.probe(candidate)
            print(f"[cutouts] probe {candidate} -> HTTP {status}", flush=True)
            if status in (200, 206):
                url = candidate
                break
    if not url:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fetch(url, str(path)):
        return None
    print(f"[cutouts] downloaded {url} -> {path} ({path.stat().st_size} B)", flush=True)
    return path


def discover(root):
    """Every ns-<obsid> directory under ``root`` that holds a mask."""
    found = []
    for entry in sorted(Path(root).glob("ns-*")):
        if not entry.is_dir():
            continue
        for mask in sorted(entry.glob("*.mask.fits*")):
            found.append((mask.name.split(".")[0], mask))
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root", default=os.environ.get("WM_RESULTS_ROOT", "/arc/projects/mlao/cfhtcast/weightmask-perf/results")
    )
    parser.add_argument("--out", default="")
    parser.add_argument("--obsids", default="")
    parser.add_argument("--work", default="/scratch/wm-exposures")
    parser.add_argument("--cutouts", type=int, default=3)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--contrast", type=float, default=0.25)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--detected", action="store_true")
    options = parser.parse_args(argv)

    # Named in the log so a pod whose environment lacks it says so immediately
    # rather than at the first exposure.
    import matplotlib

    print(f"[cutouts] matplotlib {matplotlib.__version__}", flush=True)

    root = Path(options.root)
    out = Path(options.out) if options.out else root / "ns-review" / "cutouts"
    wanted = [item.strip() for item in options.obsids.split(",") if item.strip()]
    available = discover(root)
    if wanted:
        available = [(obsid, mask) for obsid, mask in available if obsid in wanted]
    if not available:
        print(f"[cutouts] nothing to render under {root}", flush=True)
        return 2

    index = []
    failures = []
    for obsid, mask_path in available:
        record_path = mask_path.parent / f"{obsid}.run.json"
        record = json.loads(record_path.read_text()) if record_path.exists() else {}
        hdus = [int(item) for item in record.get("hdus", [])]
        exposure = ensure_exposure(options.work, obsid, record, download=options.download)
        if exposure is None:
            print(f"[cutouts] SKIP {obsid}: no exposure at {exposure_cache(options.work, obsid)}", flush=True)
            failures.append({"obsid": obsid, "mask": str(mask_path), "error": "no exposure"})
            continue
        # Named after the directory, not the obsid: a results root can hold the
        # same exposure more than once (a re-run, a probe), and those are
        # different artifacts that must not overwrite each other.
        png = out / f"{mask_path.parent.name}.cutouts.png"
        try:
            entry = render(
                obsid,
                mask_path,
                exposure,
                hdus,
                png,
                count=options.cutouts,
                size=options.size,
                contrast=options.contrast,
                detected=options.detected,
                margin=options.margin,
            )
        except Exception as exc:  # noqa: BLE001 - one bad exposure must not stop the rest
            print(f"[cutouts] FAIL {mask_path.parent.name}: {type(exc).__name__}: {exc}", flush=True)
            failures.append({"obsid": obsid, "mask": str(mask_path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        entry["bytes"] = png.stat().st_size
        index.append(entry)
        print(
            f"[cutouts] {obsid}: {len(entry['cutouts'])} cutouts -> {png} ({entry['bytes']} B)"
            f"  skipped={len(entry['skipped'])}",
            flush=True,
        )

    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "index.json"
    index_path.write_text(json.dumps({"exposures": index, "failures": failures}, indent=2, sort_keys=True) + "\n")
    print(f"[cutouts] wrote {index_path} ({len(index)} exposures, {len(failures)} failures)", flush=True)
    return 0 if index else 1


if __name__ == "__main__":
    sys.exit(main())
