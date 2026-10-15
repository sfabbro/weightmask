from typing import NamedTuple

import numpy as np
import sep
from scipy.ndimage import binary_dilation
from skimage.draw import ellipse

from .config import section_config
from .errors import StageFailure


class ObjectDetection(NamedTuple):
    added: np.ndarray
    """DETECTED pixels outside ``existing_mask``."""
    handoff: np.ndarray
    """Footprints of elongated sources offered to the streak stage; empty unless handoff is enabled."""


def _adaptive_extract_threshold(data_sub, bkg_rms_map, existing_mask, base_extract_thresh):
    extract_thresh = base_extract_thresh
    valid_mask = ~existing_mask if existing_mask is not None else np.ones(data_sub.shape, dtype=bool)
    if bkg_rms_map is not None:
        valid_mask &= np.isfinite(bkg_rms_map) & (bkg_rms_map > 0)
    valid_data = data_sub[valid_mask]

    if len(valid_data) > 1000:
        step = max(1, len(valid_data) // 100000)
        sampled_data = valid_data[::step]
        p50, p90, p99 = np.percentile(sampled_data, [50, 90, 99])
        mad_approx = np.median(np.abs(sampled_data - p50)) * 1.4826
        if mad_approx > 0:
            tail_ratio = (p99 - p90) / mad_approx
            if tail_ratio > 1.5:
                clutter_penalty = min(1.0 + 0.25 * tail_ratio, 2.0)
                extract_thresh = base_extract_thresh * clutter_penalty
                print(
                    f"  [Adaptive SEP] Crowding penalty {clutter_penalty:.2f}x applied "
                    f"(tail ratio {tail_ratio:.1f}). Extraction threshold -> {extract_thresh:.2f} sigma."
                )
    return extract_thresh


def _run_sep_extract(data_sub, bkg_rms_map, existing_mask, thresh, cfg, segmentation_map=False):
    return sep.extract(
        data_sub,
        thresh=thresh,
        err=bkg_rms_map,
        mask=existing_mask,
        minarea=cfg["min_area"],
        deblend_nthresh=cfg["deblend_nthresh"],
        deblend_cont=float(cfg["deblend_cont"]),
        clean=cfg["clean"],
        clean_param=float(cfg["clean_param"]),
        segmentation_map=segmentation_map,
    )


def _elongation(objects):
    with np.errstate(divide="ignore", invalid="ignore"):
        return objects["a"] / np.maximum(objects["b"], 1e-9)


def _labels_mask(segmap, labels):
    lookup = np.zeros(int(np.max(segmap)) + 1, dtype=bool)
    lookup[labels[labels < lookup.size]] = True
    return lookup[segmap]


def _paint_spikes(mask, objects, thresh, length_base, width):
    h, w = mask.shape
    hw = width // 2
    for obj in objects:
        s_len = int(length_base * (1.0 + 0.2 * np.log10(obj["flux"] / thresh)))
        xc, yc = int(obj["x"] + 0.5), int(obj["y"] + 0.5)
        mask[max(0, yc - hw) : min(h, yc + hw + 1), max(0, xc - s_len) : min(w - 1, xc + s_len) + 1] = True
        mask[max(0, yc - s_len) : min(h - 1, yc + s_len) + 1, max(0, xc - hw) : min(w, xc + hw + 1)] = True


def _apply_vectorized_ellipse_mask(object_mask, objects, scaled_a, scaled_b, base_k):
    try:
        sep.mask_ellipse(
            object_mask,
            objects["x"],
            objects["y"],
            scaled_a,
            scaled_b,
            objects["theta"],
            r=base_k,
        )
        return object_mask
    except Exception as e:
        print(f"  [SEP] Vectorized ellipse mask failed: {e}")

    h, w = object_mask.shape
    for i in range(len(objects)):
        if scaled_a[i] <= 0 or scaled_b[i] <= 0:
            continue
        try:
            sep.mask_ellipse(
                object_mask,
                objects["x"][i : i + 1],
                objects["y"][i : i + 1],
                scaled_a[i : i + 1],
                scaled_b[i : i + 1],
                objects["theta"][i : i + 1],
                r=base_k,
            )
        except Exception:
            try:
                cy, cx = objects["y"][i], objects["x"][i]
                ry, rx = scaled_b[i] * base_k, scaled_a[i] * base_k
                rr, cc = ellipse(
                    int(cy + 0.5),
                    int(cx + 0.5),
                    ry,
                    rx,
                    shape=(h, w),
                    rotation=-objects["theta"][i],
                )
                object_mask[rr, cc] = True
            except Exception as exc:
                raise StageFailure("objects.postprocess", exc) from exc
    return object_mask


def detect_objects(data_sub, bkg_rms_map, existing_mask, config):
    """SEP object masks for the background-subtracted image.

    A bright seed pass masks cores, a second pass at the adaptive threshold finds
    the rest; kept objects get flux-scaled halo ellipses and diffraction spikes.
    With ``handoff_elongated_to_streak``, sources with a/b >= ``max_elongation``
    are painted into ``handoff`` instead of the DETECTED mask.
    """
    cfg = section_config("sep_objects", config)
    try:
        return _detect_objects(data_sub, bkg_rms_map, existing_mask, cfg)
    except StageFailure:
        raise
    except Exception as exc:
        raise StageFailure("objects.postprocess", exc) from exc


def _detect_objects(data_sub, bkg_rms_map, existing_mask, cfg):
    shape = data_sub.shape
    d_sub = np.require(data_sub, dtype=np.float32, requirements=["C", "A"])
    b_rms = np.require(bkg_rms_map, dtype=np.float32, requirements=["C", "A"])
    m_in = None if existing_mask is None else np.require(existing_mask, dtype=np.bool_, requirements=["C", "A"])
    handoff_enabled = cfg["handoff_elongated_to_streak"]
    max_elongation = float(cfg["max_elongation"])

    extract_thresh = _adaptive_extract_threshold(d_sub, b_rms, m_in, float(cfg["extract_thresh"]))
    try:
        seeds = _run_sep_extract(d_sub, b_rms, m_in, float(cfg["seed_thresh_factor"]) * extract_thresh, cfg)
    except Exception as exc:
        raise StageFailure("objects.seed", exc) from exc
    seed_handoff = (_elongation(seeds) >= max_elongation) if handoff_enabled else np.zeros(len(seeds), dtype=bool)
    seed_mask = np.zeros(shape, dtype=bool)
    handoff_mask = np.zeros(shape, dtype=bool)
    if len(seeds) > 0:
        # A bright bar is swallowed by its own seed ellipse, so the second
        # extract never sees it; its seed footprint carries the handoff.
        seed_a, seed_b = np.maximum(seeds["a"], 1.0), np.maximum(seeds["b"], 1.0)
        seed_k = max(1.5, float(cfg["ellipse_k"]) * 0.8)
        _apply_vectorized_ellipse_mask(seed_mask, seeds, seed_a, seed_b, seed_k)
        if np.any(seed_handoff):
            _apply_vectorized_ellipse_mask(
                handoff_mask, seeds[seed_handoff], seed_a[seed_handoff], seed_b[seed_handoff], seed_k
            )

    second_pass_mask = seed_mask if m_in is None else seed_mask | m_in
    try:
        objects, segmap = _run_sep_extract(d_sub, b_rms, second_pass_mask, extract_thresh, cfg, segmentation_map=True)
    except Exception as exc:
        raise StageFailure("objects.main", exc) from exc
    main_handoff = (_elongation(objects) >= max_elongation) if handoff_enabled else np.zeros(len(objects), dtype=bool)
    if np.any(main_handoff):
        print(
            f"  Handing off {int(np.count_nonzero(main_handoff))} elongated detections to the streak pipeline "
            f"(elongation >= {max_elongation:.2f})."
        )
    labels = np.arange(1, len(objects) + 1)

    keep = np.concatenate((seeds[~seed_handoff], objects[~main_handoff]))
    handoff = np.concatenate((seeds[seed_handoff], objects[main_handoff]))
    print(
        f"  Detected {len(seeds) + len(objects)} objects "
        f"({len(keep)} kept for masking, thresh={extract_thresh:.1f} sigma)."
    )

    object_mask = np.zeros(shape, dtype=bool)
    if len(keep) > 0 or len(handoff) > 0:
        object_mask |= _labels_mask(segmap, labels[~main_handoff])
        handoff_mask |= _labels_mask(segmap, labels[main_handoff])

        if cfg["dynamic_halo_scaling"]:
            print("  Applying capped brightness-aware halo masking...")
            fluxes = np.clip(np.concatenate((keep, handoff))["flux"], 1e-5, None)
            flux_ref = max(float(np.percentile(fluxes, cfg["halo_flux_reference_percentile"])), 1e-5)

            def halo_scale(objs):
                boost = np.maximum(np.log10(np.clip(objs["flux"], 1e-5, None) / flux_ref), 0.0)
                return np.clip(1.0 + cfg["halo_brightness_factor"] * boost, 1.0, cfg["max_halo_multiplier"])

        else:

            def halo_scale(objs):
                return np.ones(len(objs))

        base_k = float(cfg["ellipse_k"])
        keep_scale, handoff_scale = halo_scale(keep), halo_scale(handoff)
        object_mask = _apply_vectorized_ellipse_mask(
            object_mask, keep, keep["a"] * keep_scale, keep["b"] * keep_scale, base_k
        )
        handoff_mask = _apply_vectorized_ellipse_mask(
            handoff_mask, handoff, handoff["a"] * handoff_scale, handoff["b"] * handoff_scale, base_k
        )
        if cfg["dynamic_halo_scaling"]:
            print(f"    Halo scaling multiplier range: 1.0x to {np.max(keep_scale) if keep_scale.size else 1.0:.2f}x")

        if cfg["spike_enable"]:
            thresh = float(cfg["spike_flux_thresh"])
            keep_bright = keep[(keep["flux"] > thresh) & (_elongation(keep) < 1.8)]
            handoff_bright = handoff[(handoff["flux"] > thresh) & (_elongation(handoff) < 1.8)]
            if len(keep_bright) or len(handoff_bright):
                print(
                    f"    Applying diffraction spike masking to {len(keep_bright) + len(handoff_bright)} "
                    f"bright stars (Flux > {thresh:.2e})..."
                )
            _paint_spikes(object_mask, keep_bright, thresh, cfg["spike_length_base"], cfg["spike_width"])
            _paint_spikes(handoff_mask, handoff_bright, thresh, cfg["spike_length_base"], cfg["spike_width"])

        # Safety margin over segmentation ragged edges and halo fringes
        # (measured: -70% fringe leakage for +3pp mask at radius 2).
        radius = cfg["mask_dilation_radius"]
        if radius > 0:
            object_mask = binary_dilation(object_mask, iterations=radius)
            handoff_mask = binary_dilation(handoff_mask, iterations=radius)

    added = object_mask if existing_mask is None else object_mask & ~existing_mask.astype(bool)
    return ObjectDetection(added, handoff_mask if handoff_enabled else np.zeros(shape, dtype=bool))
