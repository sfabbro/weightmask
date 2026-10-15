import numpy as np

from . import __version__
from .config import resolve_config
from .contract import ProducerMetadata, build_weight_product


def generate_weight_and_confidence(inv_variance_map, final_mask_int, config):
    """
    Calculates the weight map (masked inverse variance) and a
    confidence map (normalized weight map).

    Args:
        inv_variance_map (ndarray): The calculated inverse variance map (unmasked).
        final_mask_int (ndarray): The final combined integer bitmask.
        config (dict): Configuration overrides (merged over the packaged defaults);
                       reads ``output_params`` and ``confidence_params``.

    Returns:
        tuple: (weight_map, confidence_map, product), product a WeightMaskProduct.
    """
    config = resolve_config(config)
    conf_cfg = config["confidence_params"]
    normalize_percentile = conf_cfg["normalize_percentile"]

    # --- 1. Create Weight Map ---
    print("  Calculating weight map (masked inverse variance)...")

    mask_detected = config["output_params"]["mask_detected_in_weight"]
    if mask_detected:
        print("    NOTE: Detected objects will be masked (zero weight).")

    product = build_weight_product(
        inv_variance_map,
        final_mask_int,
        exclude_detected=mask_detected,
        confidence_percentile=normalize_percentile,
        producer=ProducerMetadata(version=__version__),
    )
    weight_map = product.weight
    num_masked = np.count_nonzero(weight_map == 0.0)
    print(f"    Masked {num_masked} pixels in weight map.")

    # --- 2. Create Confidence Map (Normalized Weight Map) ---
    print("  Calculating continuous confidence map (normalized weight map)...")
    confidence_map = product.confidence
    if np.any(weight_map > 0):
        print(f"    Normalizing using {normalize_percentile:.1f}th percentile weight.")
        if conf_cfg["scale_to_100"]:
            confidence_map = confidence_map * 100.0
            print("    Scaled confidence map to 0-100 range.")
        else:
            print("    Confidence map range: 0-1.")
    else:
        print("    WARNING: No positive weights found. Confidence map will be zeros.")

    # Ensure final map has the correct dtype
    confidence_map = confidence_map.astype(conf_cfg["dtype"], copy=False)

    return weight_map, confidence_map, product
