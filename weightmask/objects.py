import numpy as np
import sep
from skimage.draw import ellipse


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
            except Exception:
                continue
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

    try:
        clean_config = config or {}
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
        seed_objects = _run_sep_extract(d_sub, b_rms, m_in, seed_thresh, min_area, clean_config, segmentation_map=False)
        seed_mask = np.zeros_like(object_mask, dtype=bool)
        if len(seed_objects) > 0:
            seed_scaled_a = np.maximum(seed_objects["a"], 1.0)
            seed_scaled_b = np.maximum(seed_objects["b"], 1.0)
            _apply_vectorized_ellipse_mask(
                seed_mask,
                seed_objects,
                seed_scaled_a,
                seed_scaled_b,
                max(1.5, float(clean_config.get("ellipse_k", 2.0)) * 0.8),
            )

        second_pass_mask = seed_mask | (m_in if m_in is not None else np.zeros_like(seed_mask))
        objects, segmap = _run_sep_extract(
            d_sub,
            b_rms,
            second_pass_mask,
            extract_thresh,
            min_area,
            clean_config,
            segmentation_map=True,
        )

        # Track highly elongated detections so the streak detector can claim them later.
        elongated_count = 0
        keep_objects = objects
        keep_seg_labels = np.arange(1, len(objects) + 1)
        if len(objects) > 0:
            max_elongation = float(clean_config.get("max_elongation", 3.0))
            with np.errstate(divide="ignore", invalid="ignore"):
                elongation = objects["a"] / np.maximum(objects["b"], 1e-9)
            if clean_config.get("handoff_elongated_to_streak", True):
                valid_obj = elongation < max_elongation
                elongated_count = int(np.sum(~valid_obj))
                if elongated_count > 0:
                    print(
                        f"  Handing off {elongated_count} elongated detections to the streak pipeline "
                        f"(elongation >= {max_elongation:.2f})."
                    )
                keep_seg_labels = keep_seg_labels[valid_obj]
                keep_objects = objects[valid_obj]

        print(
            f"  Detected {len(objects)} objects ({len(keep_objects)} kept for masking, thresh={extract_thresh:.1f} sigma)."
        )

        if len(keep_objects) > 0:
            base_k = float(clean_config.get("ellipse_k", 2.0))
            halo_brightness_factor = float(clean_config.get("halo_brightness_factor", 0.15))
            halo_flux_reference_percentile = float(clean_config.get("halo_flux_reference_percentile", 50.0))
            max_halo_multiplier = float(clean_config.get("max_halo_multiplier", 1.8))
            halo_enabled = bool(clean_config.get("dynamic_halo_scaling", True))

            if segmap is not None and len(keep_seg_labels) > 0:
                footprint_mask = np.isin(segmap, keep_seg_labels)
                object_mask |= footprint_mask

            if halo_enabled:
                print("  Applying capped brightness-aware halo masking...")
                valid_fluxes = np.clip(keep_objects["flux"], 1e-5, None)
                flux_ref = np.percentile(valid_fluxes, halo_flux_reference_percentile)
                flux_ref = max(float(flux_ref), 1e-5)
                flux_boost = np.maximum(np.log10(valid_fluxes / flux_ref), 0.0)
                scale_multiplier = 1.0 + halo_brightness_factor * flux_boost
                scale_multiplier = np.clip(scale_multiplier, 1.0, max_halo_multiplier)
            else:
                scale_multiplier = np.ones(len(keep_objects))

            scaled_a = keep_objects["a"] * scale_multiplier
            scaled_b = keep_objects["b"] * scale_multiplier

            if not object_mask.flags["C_CONTIGUOUS"]:
                object_mask = np.ascontiguousarray(object_mask)

            object_mask = _apply_vectorized_ellipse_mask(object_mask, keep_objects, scaled_a, scaled_b, base_k)

            if halo_enabled:
                print(f"    Halo scaling multiplier range: 1.0x to {np.max(scale_multiplier):.2f}x")

            # --- 2. Diffraction Spike Masking ---
            if clean_config.get("spike_enable", True):
                spike_thresh = float(clean_config.get("spike_flux_thresh", 1e5))
                with np.errstate(divide="ignore", invalid="ignore"):
                    compactness = keep_objects["a"] / np.maximum(keep_objects["b"], 1e-9)
                bright_mask = (keep_objects["flux"] > spike_thresh) & (compactness < 1.8)

                if np.any(bright_mask):
                    print(
                        f"    Applying diffraction spike masking to {np.sum(bright_mask)} bright stars (Flux > {spike_thresh:.2e})..."
                    )
                    spike_length_base = int(clean_config.get("spike_length_base", 100))
                    spike_width = int(clean_config.get("spike_width", 3))

                    h, w = object_mask.shape
                    for obj in keep_objects[bright_mask]:
                        # Scale spike length slightly by flux
                        s_len = int(spike_length_base * (1.0 + 0.2 * np.log10(obj["flux"] / spike_thresh)))
                        xc, yc = int(obj["x"] + 0.5), int(obj["y"] + 0.5)

                        hw = spike_width // 2

                        # Horizontal spike
                        xstart, xend = max(0, xc - s_len), min(w - 1, xc + s_len)
                        object_mask[max(0, yc - hw) : min(h, yc + hw + 1), xstart : xend + 1] = True

                        # Vertical spike
                        ystart, yend = max(0, yc - s_len), min(h - 1, yc + s_len)
                        object_mask[ystart : yend + 1, max(0, xc - hw) : min(w, xc + hw + 1)] = True

            # Only return newly detected pixels (not already in existing_mask)
            m_orig = existing_mask.astype(bool) if existing_mask is not None else np.zeros_like(object_mask)
            obj_add_mask = object_mask & (~m_orig)
            return obj_add_mask

    except Exception as e:
        print(f"  Object detection failed: {e}")

    return np.zeros(data_sub.shape, dtype=bool)
