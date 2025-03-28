# weight.py
import numpy as np
import warnings

# Import MASK_BITS from the main package level if possible,
# otherwise define them here. Assuming they might be accessible via __init__.py
try:
    from . import MASK_BITS
except ImportError:
    # Fallback definition if run standalone or imports fail
    MASK_BITS = {
        'BAD':      1 << 0,  # 1
        'SAT':      1 << 1,  # 2
        'CR':       1 << 2,  # 4
        'DETECTED': 1 << 3,  # 8  (NOTE: DETECTED objects usually KEEP their weight)
        'STREAK':   1 << 4,  # 16
    }

def generate_weight_and_confidence(inv_variance_map, final_mask_int, config):
    """
    Calculates the weight map (masked inverse variance) and a
    confidence map (normalized weight map).

    Args:
        inv_variance_map (ndarray): The calculated inverse variance map (unmasked).
        final_mask_int (ndarray): The final combined integer bitmask.
        config (dict): Configuration dictionary, expected to contain sections like
                       'output_params' (for masking behavior) and
                       'confidence_params' (for normalization).

    Returns:
        tuple: (weight_map, confidence_map)
               Returns (None, None) if input inv_variance_map is None.
    """
    if inv_variance_map is None:
        return None, None

    out_cfg = config.get('output_params', {})
    conf_cfg = config.get('confidence_params', {})

    # --- 1. Create Weight Map ---
    print("  Calculating weight map (masked inverse variance)...")

    # Define which mask bits correspond to "bad" pixels that get zero weight
    # By default, DETECTED objects *keep* their weight. Add MASK_BITS['DETECTED']
    # here if detected objects should also be zeroed in the weight map.
    zero_weight_mask_bits = (
        MASK_BITS['BAD'] |
        MASK_BITS['SAT'] |
        MASK_BITS['CR'] |
        MASK_BITS['STREAK']
    )
    # Add DETECTED bit if configured
    if out_cfg.get('mask_detected_in_weight', False):
        print("    NOTE: Detected objects will be masked (zero weight).")
        zero_weight_mask_bits |= MASK_BITS['DETECTED']

    bad_pixel_mask = (final_mask_int & zero_weight_mask_bits) > 0

    # Create weight map from inverse variance but mask out problematic pixels
    weight_map = inv_variance_map.copy()
    # Ensure non-finite inv_var values also result in zero weight
    weight_map[~np.isfinite(weight_map)] = 0.0
    weight_map[bad_pixel_mask] = 0.0
    num_masked = np.sum(bad_pixel_mask)
    print(f"    Masked {num_masked} pixels in weight map.")

    # --- 2. Create Confidence Map (Normalized Weight Map) ---
    print("  Calculating continuous confidence map (normalized weight map)...")
    conf_dtype_str = conf_cfg.get('dtype', 'float32')
    conf_dtype = getattr(np, conf_dtype_str, np.float32) # Default to float32
    normalize_percentile = conf_cfg.get('normalize_percentile', 99.0) # Percentile for normalization
    scale_to_100 = conf_cfg.get('scale_to_100', False) # Option for 0-100 range

    confidence_map = np.zeros_like(weight_map, dtype=conf_dtype)
    valid_weights = weight_map[weight_map > 0]

    if valid_weights.size > 0:
        # Determine normalization factor
        if 0 < normalize_percentile <= 100:
             norm_factor = np.percentile(valid_weights, normalize_percentile)
             print(f"    Normalizing using {normalize_percentile:.1f}th percentile weight: {norm_factor:.4g}")
        else:
             norm_factor = np.max(valid_weights)
             print(f"    Normalizing using maximum weight: {norm_factor:.4g}")

        if norm_factor > 1e-12: # Use a slightly more generous epsilon for normalization factor
            # Normalize the weight map (clipping at 1.0)
            confidence_map = np.clip(weight_map / norm_factor, 0.0, 1.0)

            if scale_to_100:
                confidence_map *= 100.0
                print("    Scaled confidence map to 0-100 range.")
            else:
                print("    Confidence map range: 0-1.")
        else:
             print("    WARNING: Normalization factor is near zero. Confidence map will be zeros.")
    else:
        print("    WARNING: No positive weights found. Confidence map will be zeros.")

    # Ensure final map has the correct dtype
    confidence_map = confidence_map.astype(conf_dtype)

    return weight_map, confidence_map