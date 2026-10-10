import numpy as np
from skimage.draw import ellipse

from .errors import StageFailure

try:
    import sep
except Exception:
    sep = None


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


def _run_sep_extract(data_sub, bkg_rms_map, existing_mask, thresh, min_area, config, segmentation_map=False):
    return sep.extract(
        data_sub,
        thresh=thresh,
        err=bkg_rms_map,
        mask=existing_mask,
        minarea=min_area,
        deblend_nthresh=int(config.get("deblend_nthresh", 32)),
        deblend_cont=float(config.get("deblend_cont", 0.005)),
        clean=bool(config.get("clean", True)),
        clean_param=float(config.get("clean_param", 1.0)),
        segmentation_map=segmentation_map,
    )


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
    """
    Detect astronomical objects in the background-subtracted image.

    Args:
        data_sub (ndarray): Background-subtracted image data
        bkg_rms_map (ndarray): Background RMS map
        existing_mask (ndarray): Boolean mask of already masked pixels
        config (dict): Configuration dictionary for object detection

    Returns:
        ndarray: Boolean mask of newly detected object pixels
    """
    # Use bool directly for the mask
    object_mask = np.zeros(data_sub.shape, dtype=bool)

    if sep is None:
        raise StageFailure("objects.backend", "SEP backend is unavailable")

    try:
        clean_config = config if config is not None else {}
        base_extract_thresh = float(clean_config.get("extract_thresh", 3.0))
        min_area = int(clean_config.get("min_area", 10))

        # Force EVERYTHING to be clean, C-contiguous 32-bit floats
        d_sub = np.require(data_sub, dtype=np.float32, requirements=["C", "A"])
        b_rms = np.require(bkg_rms_map, dtype=np.float32, requirements=["C", "A"])

        m_in = None
        if existing_mask is not None:
            m_in = np.require(existing_mask, dtype=np.bool_, requirements=["C", "A"])

        extract_thresh = _adaptive_extract_threshold(d_sub, b_rms, m_in, base_extract_thresh)

        seed_thresh = float(clean_config.get("seed_thresh_factor", 1.25)) * extract_thresh
        try:
            seed_objects = _run_sep_extract(
                d_sub, b_rms, m_in, seed_thresh, min_area, clean_config, segmentation_map=False
            )
        except Exception as exc:
            raise StageFailure("objects.seed", exc) from exc
        seed_mask = np.zeros_like(object_mask, dtype=bool)
        elongated_seed = np.zeros_like(object_mask, dtype=bool)
        keep_seed_objects = seed_objects
        if len(seed_objects) > 0:
            seed_scaled_a = np.maximum(seed_objects["a"], 1.0)
            seed_scaled_b = np.maximum(seed_objects["b"], 1.0)
            seed_k = max(1.5, float(clean_config.get("ellipse_k", 2.0)) * 0.8)
            _apply_vectorized_ellipse_mask(seed_mask, seed_objects, seed_scaled_a, seed_scaled_b, seed_k)
            # A bright bar is swallowed by its own seed ellipse, so the second
            # extract never sees it. Preserve its footprint for the handoff.
            if clean_config.get("handoff_elongated_to_streak", True):
                with np.errstate(divide="ignore", invalid="ignore"):
                    seed_elong = seed_objects["a"] / np.maximum(seed_objects["b"], 1e-9)
                keep_seed = seed_elong >= float(clean_config.get("max_elongation", 3.0))
                keep_seed_objects = seed_objects[~keep_seed]
                if np.any(keep_seed):
                    _apply_vectorized_ellipse_mask(
                        elongated_seed,
                        seed_objects[keep_seed],
                        seed_scaled_a[keep_seed],
                        seed_scaled_b[keep_seed],
                        seed_k,
                    )

        second_pass_mask = seed_mask | (m_in if m_in is not None else np.zeros_like(seed_mask))
        try:
            objects, segmap = _run_sep_extract(
                d_sub,
                b_rms,
                second_pass_mask,
                extract_thresh,
                min_area,
                clean_config,
                segmentation_map=True,
            )
        except Exception as exc:
            raise StageFailure("objects.main", exc) from exc

        # Track highly elongated detections so the streak detector can claim them later.
        elongated_count = 0
        handoff_enabled = bool(clean_config.get("handoff_elongated_to_streak", True))
        keep_objects = objects
        keep_seg_labels = np.arange(1, len(objects) + 1)
        if len(objects) > 0:
            max_elongation = float(clean_config.get("max_elongation", 3.0))
            with np.errstate(divide="ignore", invalid="ignore"):
                elongation = objects["a"] / np.maximum(objects["b"], 1e-9)
            if handoff_enabled:
                valid_obj = elongation < max_elongation
                elongated_count = int(np.count_nonzero(~valid_obj))
                if elongated_count > 0:
                    print(
                        f"  Handing off {elongated_count} elongated detections to the streak pipeline "
                        f"(elongation >= {max_elongation:.2f})."
                    )
                keep_seg_labels = keep_seg_labels[valid_obj]
                keep_objects = objects[valid_obj]

        elongated_mask = np.zeros(data_sub.shape, dtype=bool)
        elongated_mask |= elongated_seed
        handoff_main_objects = np.zeros(0, dtype=objects.dtype)
        handoff_seg_labels = np.zeros(0, dtype=int)
        if segmap is not None and len(objects) > 0:
            dropped = np.arange(1, len(objects) + 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                elong = objects["a"] / np.maximum(objects["b"], 1e-9)
            dropped = dropped[elong >= float(clean_config.get("max_elongation", 3.0))]
            handoff_main_objects = objects[elong >= float(clean_config.get("max_elongation", 3.0))]
            handoff_seg_labels = dropped
            max_label = int(np.max(segmap))
            if max_label > 0 and dropped.size:
                lookup = np.zeros(max_label + 1, dtype=bool)
                ok = dropped[(dropped >= 0) & (dropped <= max_label)]
                lookup[ok] = True
                elongated_mask |= lookup[segmap]
        # Elongation handoff metadata is passed through the stage config.
        clean_config["_elongated_for_sky"] = elongated_mask

        # Seed ellipses exclude bright cores from the second pass. Their
        # non-elongated objects still need the same halo/spike masking below.
        keep_objects = np.concatenate((keep_seed_objects, keep_objects))
        print(
            f"  Detected {len(seed_objects) + len(objects)} objects "
            f"({len(keep_objects)} kept for masking, thresh={extract_thresh:.1f} sigma)."
        )

        handoff_seed_objects = (
            seed_objects[
                seed_objects["a"] / np.maximum(seed_objects["b"], 1e-9)
                >= float(clean_config.get("max_elongation", 3.0))
            ]
            if len(seed_objects) > 0
            else seed_objects[:0]
        )
        handoff_objects = np.concatenate((handoff_seed_objects, handoff_main_objects))
        clean_config["_elongated_candidate_fluxes"] = handoff_objects["flux"].copy()
        if not handoff_enabled:
            handoff_objects = objects[:0]
            handoff_seed_objects = seed_objects[:0]
            handoff_seg_labels = np.zeros(0, dtype=int)
        clean_config["_elongated_fallback_mask"] = np.zeros(data_sub.shape, dtype=bool)
        if len(keep_objects) > 0 or len(handoff_objects) > 0:
            base_k = float(clean_config.get("ellipse_k", 2.0))
            halo_brightness_factor = float(clean_config.get("halo_brightness_factor", 0.15))
            halo_flux_reference_percentile = float(clean_config.get("halo_flux_reference_percentile", 50.0))
            max_halo_multiplier = float(clean_config.get("max_halo_multiplier", 1.8))
            halo_enabled = bool(clean_config.get("dynamic_halo_scaling", True))

            handoff_mask = elongated_seed.copy()
            if segmap is not None and len(keep_seg_labels) > 0:
                max_label = int(np.max(segmap))
                if max_label > 0:
                    lookup = np.zeros(max_label + 1, dtype=bool)
                    valid_labels = keep_seg_labels[(keep_seg_labels >= 0) & (keep_seg_labels <= max_label)]
                    lookup[valid_labels] = True
                    object_mask |= lookup[segmap]
            if segmap is not None and len(handoff_seg_labels) > 0:
                max_label = int(np.max(segmap))
                if max_label > 0:
                    lookup = np.zeros(max_label + 1, dtype=bool)
                    valid_labels = handoff_seg_labels[(handoff_seg_labels >= 0) & (handoff_seg_labels <= max_label)]
                    lookup[valid_labels] = True
                    handoff_mask |= lookup[segmap]

            if halo_enabled:
                print("  Applying capped brightness-aware halo masking...")
                all_objects = np.concatenate((keep_objects, handoff_objects))
                valid_fluxes = np.clip(all_objects["flux"], 1e-5, None)
                flux_ref = np.percentile(valid_fluxes, halo_flux_reference_percentile)
                flux_ref = max(float(flux_ref), 1e-5)
                keep_boost = np.maximum(np.log10(np.clip(keep_objects["flux"], 1e-5, None) / flux_ref), 0.0)
                handoff_boost = np.maximum(np.log10(np.clip(handoff_objects["flux"], 1e-5, None) / flux_ref), 0.0)
                scale_multiplier = np.clip(1.0 + halo_brightness_factor * keep_boost, 1.0, max_halo_multiplier)
                handoff_scale_multiplier = np.clip(
                    1.0 + halo_brightness_factor * handoff_boost, 1.0, max_halo_multiplier
                )
            else:
                scale_multiplier = np.ones(len(keep_objects))
                handoff_scale_multiplier = np.ones(len(handoff_objects))

            scaled_a = keep_objects["a"] * scale_multiplier
            scaled_b = keep_objects["b"] * scale_multiplier

            if not object_mask.flags["C_CONTIGUOUS"]:
                object_mask = np.ascontiguousarray(object_mask)

            object_mask = _apply_vectorized_ellipse_mask(object_mask, keep_objects, scaled_a, scaled_b, base_k)
            handoff_scaled_a = handoff_objects["a"] * handoff_scale_multiplier
            handoff_scaled_b = handoff_objects["b"] * handoff_scale_multiplier
            handoff_mask = _apply_vectorized_ellipse_mask(
                handoff_mask, handoff_objects, handoff_scaled_a, handoff_scaled_b, base_k
            )

            if halo_enabled:
                max_scale = np.max(scale_multiplier) if scale_multiplier.size else 1.0
                print(f"    Halo scaling multiplier range: 1.0x to {max_scale:.2f}x")

            # --- 2. Diffraction Spike Masking ---
            if clean_config.get("spike_enable", True):
                spike_thresh = float(clean_config.get("spike_flux_thresh", 1e5))
                spike_length_base = int(clean_config.get("spike_length_base", 100))
                spike_width = int(clean_config.get("spike_width", 3))
                with np.errstate(divide="ignore", invalid="ignore"):
                    compactness = keep_objects["a"] / np.maximum(keep_objects["b"], 1e-9)
                    handoff_compactness = handoff_objects["a"] / np.maximum(handoff_objects["b"], 1e-9)
                bright_mask = (keep_objects["flux"] > spike_thresh) & (compactness < 1.8)
                handoff_bright_mask = (handoff_objects["flux"] > spike_thresh) & (handoff_compactness < 1.8)
                h, w = object_mask.shape
                if np.any(bright_mask) or np.any(handoff_bright_mask):
                    print(
                        f"    Applying diffraction spike masking to {np.count_nonzero(bright_mask) + np.count_nonzero(handoff_bright_mask)} bright stars (Flux > {spike_thresh:.2e})..."
                    )
                for obj in keep_objects[bright_mask]:
                    s_len = int(spike_length_base * (1.0 + 0.2 * np.log10(obj["flux"] / spike_thresh)))
                    xc, yc = int(obj["x"] + 0.5), int(obj["y"] + 0.5)
                    hw = spike_width // 2
                    xstart, xend = max(0, xc - s_len), min(w - 1, xc + s_len)
                    object_mask[max(0, yc - hw) : min(h, yc + hw + 1), xstart : xend + 1] = True
                    ystart, yend = max(0, yc - s_len), min(h - 1, yc + s_len)
                    object_mask[ystart : yend + 1, max(0, xc - hw) : min(w, xc + hw + 1)] = True
                for obj in handoff_objects[handoff_bright_mask]:
                    s_len = int(spike_length_base * (1.0 + 0.2 * np.log10(obj["flux"] / spike_thresh)))
                    xc, yc = int(obj["x"] + 0.5), int(obj["y"] + 0.5)
                    hw = spike_width // 2
                    xstart, xend = max(0, xc - s_len), min(w - 1, xc + s_len)
                    handoff_mask[max(0, yc - hw) : min(h, yc + hw + 1), xstart : xend + 1] = True
                    ystart, yend = max(0, yc - s_len), min(h - 1, yc + s_len)
                    handoff_mask[ystart : yend + 1, max(0, xc - hw) : min(w, xc + hw + 1)] = True

            candidate_mask = handoff_mask.copy()
            # Safety margin: dilate the combined mask to cover segmentation
            # ragged edges and halo fringes cheaply (measured: -70% fringe
            # leakage for +3pp mask at radius 2).
            try:
                dil_radius = int(clean_config.get("mask_dilation_radius", 0))
            except (TypeError, ValueError):
                dil_radius = 0
            if dil_radius > 0:
                from scipy.ndimage import binary_dilation

                object_mask = binary_dilation(object_mask, iterations=dil_radius)
                handoff_mask = binary_dilation(handoff_mask, iterations=dil_radius)
                from scipy.ndimage import label as connected_components

                candidate_labels, candidate_count = connected_components(candidate_mask)
                if dil_radius > 0:
                    expanded_candidate = np.zeros_like(candidate_mask)
                    for component in range(1, candidate_count + 1):
                        component_mask = candidate_labels == component
                        expanded_candidate |= binary_dilation(component_mask, iterations=dil_radius)
                    candidate_mask = expanded_candidate
            clean_config["_elongated_candidate_mask"] = (
                candidate_mask if handoff_enabled else np.zeros_like(handoff_mask)
            )
            clean_config["_elongated_fallback_mask"] = object_mask.copy() if not handoff_enabled else handoff_mask
        # Only return newly detected pixels (not already in existing_mask).
        # This must sit outside the `keep_objects` branch: when nothing is
        # detected, object_mask is still all-False and this correctly yields an
        # empty mask. Leaving the assignment inside raised UnboundLocalError,
        # which the handler below turned into a spurious ERROR on a normal
        # outcome (the returned mask was correct; only the message was wrong).
        m_orig = existing_mask.astype(bool) if existing_mask is not None else np.zeros_like(object_mask)
        obj_add_mask = object_mask & (~m_orig)
        return obj_add_mask
    except StageFailure:
        raise
    except Exception as exc:
        raise StageFailure("objects.postprocess", exc) from exc
