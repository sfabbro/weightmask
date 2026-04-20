from typing import Optional

import numpy as np

from .cli import process_image, validate_config


class WeightMapGenerator:
    def __init__(self, config: dict):
        """
        Initialize the WeightMapGenerator with a configuration dictionary.
        """
        self.config = config
        if not validate_config(self.config):
            print("WARNING: Configuration validation failed during initialization.")

    def process(
        self,
        data: np.ndarray,
        header: Optional[dict] = None,
        flat_data: Optional[np.ndarray] = None,
        tile_size: int = 1024,
    ) -> dict:
        """
        Process a science image to generate mask and weight products.

        Args:
            data: Science image data as a numpy array.
            header: Science image header as a dictionary (optional).
            flat_data: Flat field image data (optional).
            tile_size: Size of tiles for processing (default 1024).

        Returns:
            A dictionary containing the generated products:
            - weight_map: The final weight map.
            - flag_map: The bitmask.
            - inv_variance_map: The inverse variance map.
            - confidence_map: The confidence map.
            - sky_map: The background sky map.
            - individual_masks: Component masks (bad, sat, cr, obj, streak).
        """
        if header is None:
            header = {}

        result = process_image(data, header, flat_data, self.config, tile_size)

        if result[0] is None:
            return {}

        mask_data, inv_var, weight, confidence, sky, info = result

        return {
            "weight_map": weight,
            "flag_map": mask_data,
            "inv_variance_map": inv_var,
            "confidence_map": confidence,
            "sky_map": sky,
            "bkg_rms_map": info.get("individual_masks", {}).get(
                "bkg_rms"
            ),  # May not be explicitly returned in this form
            "individual_masks": info.get("individual_masks", {}),
        }
