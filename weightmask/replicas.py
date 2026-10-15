"""Chip-replica veto: drop STREAK where one CCD-local line was written on several chips."""

from __future__ import annotations

import fitsio
import numpy as np
from astropy.wcs import WCS
from scipy import ndimage

from .contract import (
    DEFAULT_ZERO_WEIGHT_BITS,
    QualityBit,
    build_weight_product,
)


def line_geometry(ys, xs, shape):
    """CCD-local line of a streak: angle in degrees and offset from the chip centre.

    The normal is sign-canonicalised so two copies of the same column compare
    equal. Common-frame geometry must resolve whether equal local lines are a
    sky-consistent trail or detector-fixed replicas.
    """
    ys = np.asarray(ys)
    xs = np.asarray(xs)
    if xs.size < 24:
        return None
    x = xs.astype(np.float64)
    y = ys.astype(np.float64)
    mx, my = float(x.mean()), float(y.mean())
    xc, yc = x - mx, y - my
    cxx = float(np.dot(xc, xc))
    cyy = float(np.dot(yc, yc))
    cxy = float(np.dot(xc, yc))
    theta = 0.5 * float(np.arctan2(2.0 * cxy, cxx - cyy))
    direction = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    h, w = shape
    cx, cy = 0.5 * (w - 1), 0.5 * (h - 1)
    offset = float(normal[0] * (mx - cx) + normal[1] * (my - cy))
    if normal[0] < 0.0 or (normal[0] == 0.0 and normal[1] < 0.0):
        offset = -offset
    angle = float(np.degrees(np.arctan2(direction[1], direction[0])) % 180.0)
    return {
        "offset_px": offset,
        "angle_deg": angle,
        "point": [mx, my],
        "direction": direction.tolist(),
    }


def _angle_sep_deg(left, right):
    sep = abs(float(left) - float(right)) % 180.0
    return min(sep, 180.0 - sep)


def replica_indices(geometries, tol_px=25.0, tol_deg=1.5):
    """Indices whose CCD-local streak line is shared with another chip."""
    replicas = set()
    for i in range(len(geometries)):
        left = geometries[i]
        if left is None:
            continue
        for j in range(i + 1, len(geometries)):
            right = geometries[j]
            if right is None:
                continue
            if _angle_sep_deg(left["angle_deg"], right["angle_deg"]) > tol_deg:
                continue
            if abs(left["offset_px"] - right["offset_px"]) > tol_px:
                continue
            replicas.add(i)
            replicas.add(j)
    return replicas


def _replica_component_indices(catalogs, tol_px=25.0, tol_deg=1.5):
    peers = [set() for _ in catalogs]
    detector_fixed_edges = set()
    for i in range(len(catalogs)):
        left = catalogs[i]
        left_local = left.get("local")
        if left_local is None:
            continue
        for j in range(i + 1, len(catalogs)):
            right = catalogs[j]
            if left.get("hdu") == right.get("hdu"):
                continue
            right_local = right.get("local")
            if right_local is None:
                continue
            if _angle_sep_deg(left_local["angle_deg"], right_local["angle_deg"]) > tol_deg:
                continue
            if abs(left_local["offset_px"] - right_local["offset_px"]) > tol_px:
                continue
            peers[i].add(j)
            peers[j].add(i)
            left_common = left.get("common")
            right_common = right.get("common")
            if left_common is None or right_common is None:
                continue
            if np.allclose(left_common["crval"], right_common["crval"], rtol=0.0, atol=1.0e-8):
                common_angle = _angle_sep_deg(left_common["angle_deg"], right_common["angle_deg"])
                scale = max(left_common["scale_deg_per_px"], right_common["scale_deg_per_px"])
                common_offset = abs(left_common["offset_deg"] - right_common["offset_deg"])
                if common_angle > tol_deg or common_offset > tol_px * scale:
                    detector_fixed_edges.add((i, j))
    replicas = set()
    for i, j in detector_fixed_edges:
        if peers[i] == {j} and peers[j] == {i}:
            replicas.update((i, j))
    return replicas


def _common_wcs_geometry(header):
    if header is None:
        return None
    try:
        has_cd = all(header.get(name) is not None for name in ("CD1_1", "CD1_2", "CD2_1", "CD2_2"))
        has_cdelt = all(header.get(name) is not None for name in ("CDELT1", "CDELT2"))
        if not has_cd and not has_cdelt:
            return None
        # WCS() warns and substitutes a default for a card it cannot parse
        # (e.g. CDELT1 = "not a number"), which would hand back a plausible but
        # wrong plate scale. Refuse such a header instead of failing open.
        for name in (
            "CRVAL1",
            "CRVAL2",
            "CRPIX1",
            "CRPIX2",
            "CDELT1",
            "CDELT2",
            "CD1_1",
            "CD1_2",
            "CD2_1",
            "CD2_2",
            "PC1_1",
            "PC1_2",
            "PC2_1",
            "PC2_2",
        ):
            value = header.get(name)
            if value is None:
                continue
            try:
                float(value)
            except (TypeError, ValueError):
                return None
        cards = {}
        for name in (
            "CTYPE1",
            "CTYPE2",
            "CUNIT1",
            "CUNIT2",
            "CRVAL1",
            "CRVAL2",
            "CRPIX1",
            "CRPIX2",
            "CDELT1",
            "CDELT2",
            "CD1_1",
            "CD1_2",
            "CD2_1",
            "CD2_2",
            "PC1_1",
            "PC1_2",
            "PC2_1",
            "PC2_2",
        ):
            value = header.get(name)
            if value is not None:
                cards[name] = value
        wcs = WCS(cards, naxis=2)
        if not wcs.has_celestial:
            return None
        celestial = wcs.celestial
        ctypes = [str(value).upper() for value in celestial.wcs.ctype]
        if len(ctypes) != 2 or any(not value.endswith("-TAN") for value in ctypes):
            return None
        celestial.wcs.set()
        crval = np.asarray(celestial.wcs.crval, dtype=np.float64)
        crpix = np.asarray(celestial.wcs.crpix, dtype=np.float64) - 1.0
        cd = np.asarray(celestial.pixel_scale_matrix, dtype=np.float64)
        probe = np.asarray(celestial.all_pix2world([[0.0, 0.0]], 0), dtype=np.float64)
    except Exception:
        return None
    if crval.shape != (2,) or crpix.shape != (2,) or cd.shape != (2, 2):
        return None
    if not np.all(np.isfinite(crval)) or not np.all(np.isfinite(crpix)) or not np.all(np.isfinite(cd)):
        return None
    if probe.shape != (1, 2) or not np.all(np.isfinite(probe)):
        return None
    determinant = float(np.linalg.det(cd))
    if not np.isfinite(determinant) or abs(determinant) <= np.finfo(np.float64).tiny:
        return None
    return {
        "crval": crval,
        "crpix": crpix,
        "cd": cd,
        "scale_deg_per_px": float(np.sqrt(abs(determinant))),
    }


def _line_to_common_geometry(local, wcs):
    if local is None or wcs is None:
        return None
    point = np.asarray(local["point"], dtype=np.float64)
    direction = wcs["cd"] @ np.asarray(local["direction"], dtype=np.float64)
    length = float(np.hypot(*direction))
    if not np.isfinite(length) or length <= 0.0:
        return None
    direction /= length
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    mosaic_point = wcs["cd"] @ (point - wcs["crpix"])
    offset = float(np.dot(normal, mosaic_point))
    if normal[0] < 0.0 or (normal[0] == 0.0 and normal[1] < 0.0):
        normal = -normal
        offset = -offset
    return {
        "angle_deg": float(np.degrees(np.arctan2(normal[1], normal[0])) % 180.0),
        "normal": normal.tolist(),
        "offset_deg": offset,
        "scale_deg_per_px": wcs["scale_deg_per_px"],
        "crval": wcs["crval"],
    }


def streak_catalog(hdu_index, mask_data, header=None):
    """Connected STREAK components and their detector/common-frame geometry."""
    if mask_data is None:
        return []
    data = np.asarray(mask_data)
    streak = (data & np.uint32(QualityBit.STREAK)) != 0
    labels, count = ndimage.label(streak, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return []
    shape = tuple(data.shape)
    wcs = _common_wcs_geometry(header)
    catalogs = []
    for component, region in enumerate(ndimage.find_objects(labels), 1):
        if region is None:
            continue
        ys, xs = np.nonzero(labels[region] == component)
        if ys.size < 24:
            continue
        ys += region[0].start
        xs += region[1].start
        local = line_geometry(ys, xs, shape)
        catalogs.append(
            {
                "hdu": int(hdu_index),
                "component": int(component),
                "ys": np.asarray(ys, dtype=np.int32),
                "xs": np.asarray(xs, dtype=np.int32),
                "shape": shape,
                "local": local,
                "common": _line_to_common_geometry(local, wcs),
            }
        )
    return catalogs


def _rewrite_hdu(fits_obj, pos, data):
    # Existing HDUs retain their compression; ImageHDU.write has no compress option.
    fits_obj[pos].write(np.ascontiguousarray(data))


def clear_chip_replicas(catalogs, writers, config):
    """Drop STREAK where the same CCD-local line was written on another chip.

    Coordinates were recorded at flush time, so this does not hold the MEF's
    arrays. Weight is restored from inverse variance only where STREAK was the
    only zero-weight bit and the map product is a weight.
    """
    if len(catalogs) < 2:
        return True
    cleared = _replica_component_indices(catalogs)
    if not cleared:
        return True
    mask_writer = writers.get("mask")
    if mask_writer is None:
        return True
    output_format = config["output_params"]["output_map_format"]
    conf_cfg = config["confidence_params"]
    per_hdu_confidence = output_format == "confidence" and conf_cfg["normalize_scope"] != "per_exposure"
    streak_bit = np.uint32(QualityBit.STREAK)
    exclusion = DEFAULT_ZERO_WEIGHT_BITS & ~QualityBit.STREAK
    exclude_detected = config["output_params"]["mask_detected_in_weight"]
    if exclude_detected:
        exclusion |= QualityBit.DETECTED
    other_bits = np.uint32(exclusion)
    map_writer = writers.get("map")
    ivar_writer = writers.get("invvar")
    raw_writer = writers.get("weight_raw")
    ind_writer = writers.get("ind_streak")

    def _open(writer):
        if writer is None:
            return None
        return fitsio.FITS(writer.out_path, "rw")

    handles = []
    try:
        fmask = fitsio.FITS(mask_writer.out_path, "rw")
        handles.append(fmask)
        fmap = _open(map_writer)
        fivar = _open(ivar_writer)
        fraw = _open(raw_writer)
        find = _open(ind_writer)
        handles.extend(handle for handle in (fmap, fivar, fraw, find) if handle is not None)
        individual_positions = {}
        if ind_writer is not None:
            for index in sorted(cleared):
                cat = catalogs[index]
                hdu = cat["hdu"]
                if hdu in individual_positions:
                    continue
                spos = ind_writer.positions.get(hdu)
                if spos is None or find is None or spos >= len(find):
                    print(f"  ERROR: chip-replica veto: missing individual STREAK position for HDU {hdu}.")
                    return False
                if np.shape(find[spos].read()) != cat["shape"]:
                    print(f"  ERROR: chip-replica veto: individual STREAK shape mismatch for HDU {hdu}.")
                    return False
                individual_positions[hdu] = spos
        n_cleared = 0
        for index in sorted(cleared):
            cat = catalogs[index]
            pos = mask_writer.positions.get(cat["hdu"])
            if pos is None or pos >= len(fmask):
                print(f"  WARNING: chip-replica veto: unknown mask position for HDU {cat['hdu']}.")
                continue
            data = fmask[pos].read()
            ys, xs = cat["ys"], cat["xs"]
            if ys.size == 0 or data.shape != cat["shape"]:
                print(f"  WARNING: chip-replica veto: shape mismatch for HDU {cat['hdu']}.")
                continue
            pixels = data[ys, xs]
            only = ((pixels & streak_bit) != 0) & ((pixels & other_bits) == 0)
            needs_restore = bool(np.any(only)) and (fmap is not None or fraw is not None)
            ipos = ivar_writer.positions.get(cat["hdu"]) if ivar_writer is not None else None
            ivar = fivar[ipos].read() if fivar is not None and ipos is not None else None
            if needs_restore and np.shape(ivar) != data.shape:
                print(
                    f"  WARNING: chip-replica veto: weight map could not be restored on HDU {cat['hdu']}; retaining STREAK."
                )
                continue
            data[ys, xs] = pixels & ~np.array(streak_bit, dtype=data.dtype)
            for handle, writer in ((fmap, map_writer), (fraw, raw_writer)):
                if handle is None or not needs_restore:
                    continue
                wpos = writer.positions.get(cat["hdu"])
                if wpos is None:
                    raise OSError(f"missing output position for HDU {cat['hdu']}")
                if writer is map_writer and per_hdu_confidence:
                    product = build_weight_product(
                        ivar,
                        data.astype(np.uint32, copy=False),
                        exclude_detected=exclude_detected,
                        confidence_percentile=conf_cfg["normalize_percentile"],
                    )
                    output = product.confidence * (100.0 if conf_cfg["scale_to_100"] else 1.0)
                else:
                    output = handle[wpos].read()
                    output[ys[only], xs[only]] = ivar[ys[only], xs[only]]
                _rewrite_hdu(handle, wpos, output.astype(writer.dtype or np.float32, copy=False))
            _rewrite_hdu(fmask, pos, data)
            if find is not None:
                spos = individual_positions[cat["hdu"]]
                ind = find[spos].read()
                ind[ys, xs] = 0
                _rewrite_hdu(find, spos, ind.astype(np.uint8, copy=False))
            n_cleared += 1
        if n_cleared:
            print(f"  Chip-replica veto cleared {n_cleared} STREAK components.")
        return True
    except OSError as exc:
        print(f"  ERROR: chip-replica veto failed: {exc}")
        return False
    finally:
        for handle in handles:
            try:
                handle.close()
            except Exception:
                pass
