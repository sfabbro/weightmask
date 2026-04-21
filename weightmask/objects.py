import numpy as np
import sep
from skimage.draw import ellipse


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

        # --- Adaptive SEP Threshold (Density/Clutter Scaling) ---
        extract_thresh = base_extract_thresh
        valid_mask = ~m_in if m_in is not None else np.ones(data_sub.shape, dtype=bool)
        if bkg_rms_map is not None:
            valid_mask &= bkg_rms_map > 0
        valid_data = data_sub[valid_mask]

        if len(valid_data) > 1000:
            # ⚡ Bolt: Subsample large arrays before calculating global robust statistics
            step = max(1, len(valid_data) // 100000)
            sampled_data = valid_data[::step]
            p50, p99 = np.percentile(sampled_data, [50, 99])
            mad_approx = np.median(np.abs(sampled_data - p50)) * 1.4826
            if mad_approx > 0:
                tail_ratio = (p99 - p50) / mad_approx
                if tail_ratio > 3.0:
                    clutter_penalty = min(tail_ratio / 3.0, 3.0)
                    extract_thresh = base_extract_thresh * np.sqrt(clutter_penalty)
                    print(
                        f"  [Adaptive SEP] Clutter penalty {clutter_penalty:.2f}x applied (tail ratio {tail_ratio:.1f}). Scaled extraction threshold to {extract_thresh:.2f} sigma."
                    )

        # Extract objects using SEP
        objects = sep.extract(
            d_sub,
            thresh=extract_thresh,
            err=b_rms,
            mask=m_in,
            minarea=min_area,
            segmentation_map=False,
        )

        # Filter highly elongated objects so the streak detector can find them later
        if len(objects) > 0:
            max_elongation = float(clean_config.get("max_elongation", 3.0))
            with np.errstate(divide="ignore", invalid="ignore"):
                elongation = objects["a"] / np.maximum(objects["b"], 1e-9)
            valid_obj = elongation < max_elongation
            objects = objects[valid_obj]

        print(f"  Detected {len(objects)} objects (thresh={extract_thresh:.1f} sigma).")

        if len(objects) > 0:
            base_k = float(clean_config.get("ellipse_k", 2.0))
            alpha = float(clean_config.get("halo_scale_factor", 0.5))

            # 1. Calculate semi-axes with optional halo scaling
            if clean_config.get("dynamic_halo_scaling", True):
                print("  Applying dynamic halo masking based on object flux...")
                valid_fluxes = np.clip(objects["flux"], 1e-5, None)
                min_flux = np.percentile(valid_fluxes, 10)
                flux_ratio = np.clip(valid_fluxes / min_flux, 1.0, None)
                scale_multiplier = 1.0 + alpha * np.log10(flux_ratio)
            else:
                scale_multiplier = np.ones(len(objects))

            scaled_a = objects["a"] * scale_multiplier
            scaled_b = objects["b"] * scale_multiplier

            if not object_mask.flags["C_CONTIGUOUS"]:
                object_mask = np.ascontiguousarray(object_mask)

            # ⚡ Bolt: Use vectorized sep.mask_ellipse directly on the boolean mask.
            # Avoids O(N) Python iteration overhead for large source lists.
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
            except Exception as e:
                print(f"  [Bolt] Error applying vectorized ellipse mask: {e}")
                # Fallback to individual ellipse rendering

                h, w = object_mask.shape
                for i in range(len(objects)):
                    if scaled_a[i] > 0 and scaled_b[i] > 0:
                        # Try individual sep.mask_ellipse call first (faster than skimage)
                        try:
                            sep.mask_ellipse(
                                object_mask,
                                np.array([objects["x"][i]]),
                                np.array([objects["y"][i]]),
                                np.array([scaled_a[i]]),
                                np.array([scaled_b[i]]),
                                np.array([objects["theta"][i]]),
                                r=base_k,
                            )
                        except Exception:
                            # Final fallback to skimage.draw.ellipse if SEP fails for this source
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

            if clean_config.get("dynamic_halo_scaling", True):
                print(f"    Halo scaling multiplier range: 1.0x to {np.max(scale_multiplier):.2f}x")

            # --- 2. Diffraction Spike Masking ---
            if clean_config.get("spike_enable", True):
                spike_thresh = float(clean_config.get("spike_flux_thresh", 1e5))
                bright_mask = objects["flux"] > spike_thresh

                if np.any(bright_mask):
                    print(
                        f"    Applying diffraction spike masking to {np.sum(bright_mask)} bright stars (Flux > {spike_thresh:.2e})..."
                    )
                    spike_length_base = int(clean_config.get("spike_length_base", 100))
                    spike_width = int(clean_config.get("spike_width", 3))

                    h, w = object_mask.shape
                    for obj in objects[bright_mask]:
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

    except Exception:
        pass

    return np.zeros(data_sub.shape, dtype=bool)
