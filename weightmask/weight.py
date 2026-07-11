# weight.py
import numpy as np

from .contract import ProducerMetadata, build_weight_product

# Import MASK_BITS from the main package level if possible,
# otherwise define them here. Assuming they might be accessible via __init__.py
try:
    from . import MASK_BITS
except ImportError:
    # Fallback definition if run standalone or imports fail
    MASK_BITS = {
        "BAD": 1 << 0,  # 1
        "SAT": 1 << 1,  # 2
        "CR": 1 << 2,  # 4
        "DETECTED": 1 << 3,  # 8  (NOTE: DETECTED objects usually KEEP their weight)
        "STREAK": 1 << 4,  # 16
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

    out_cfg = config.get("output_params", {})
    conf_cfg = config.get("confidence_params", {})
    normalize_percentile = conf_cfg.get("normalize_percentile", 99.0)
    scale_to_100 = conf_cfg.get("scale_to_100", False)

    # --- 1. Create Weight Map ---
    print("  Calculating weight map (masked inverse variance)...")

    mask_detected = out_cfg.get("mask_detected_in_weight", False)
    if mask_detected:
        print("    NOTE: Detected objects will be masked (zero weight).")

    # The legacy API used an out-of-range percentile to request max-based
    # normalization.  Keep that behavior while the contract API stays strict.
    contract_percentile = normalize_percentile if 0 < normalize_percentile <= 100 else 100.0
    product = build_weight_product(
        inv_variance_map,
        final_mask_int,
        exclude_detected=mask_detected,
        confidence_percentile=contract_percentile,
        producer=ProducerMetadata(version="1.0.0"),
    )
    weight_map = product.weight
    num_masked = np.count_nonzero(weight_map == 0.0)
    print(f"    Masked {num_masked} pixels in weight map.")

    # --- 2. Create Confidence Map (Normalized Weight Map) ---
    print("  Calculating continuous confidence map (normalized weight map)...")
    conf_dtype_str = conf_cfg.get("dtype", "float32")
    conf_dtype = getattr(np, conf_dtype_str, np.float32)  # Default to float32

    confidence_map = product.confidence
    if np.any(weight_map > 0):
        print(f"    Normalizing using {normalize_percentile:.1f}th percentile weight.")
        if scale_to_100:
            confidence_map = confidence_map * 100.0
            print("    Scaled confidence map to 0-100 range.")
        else:
            print("    Confidence map range: 0-1.")
    else:
        print("    WARNING: No positive weights found. Confidence map will be zeros.")

    # Ensure final map has the correct dtype
    confidence_map = confidence_map.astype(conf_dtype)

    return weight_map, confidence_map
