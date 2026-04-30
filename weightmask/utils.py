import re

import numpy as np


def extract_hdu_spec(filepath):
    """
    Extract HDU specifier from CFITSIO-style filename (e.g., 'file.fits[1]')

    Args:
        filepath (str): Path with potential HDU specifier

    Returns:
        tuple: (clean_path, hdu_index) where hdu_index is None if not specified
    """
    if filepath is None:
        return None, None

    match = re.match(r"^(.*?)(?:\[(\d+)\])?$", filepath)
    if match:
        path, hdu_spec = match.groups()
        if hdu_spec is not None:
            return path, int(hdu_spec)
    return filepath, None


def create_binary_mask(mask_data, bit_flag):
    """
    Create a binary mask (0/1) from a bitmask for a specific flag.

    Args:
        mask_data (ndarray): Bitmask array
        bit_flag (int): Bit flag to extract

    Returns:
        ndarray: Binary mask (0=not set, 1=set)
    """
    # Performance optimization: Use direct boolean casting instead of np.where.
    # This avoids allocating unnecessary intermediate arrays and executes
    # significantly faster.
    return ((mask_data & bit_flag) > 0).astype(np.uint8)


def _parse_config_value(val):
    """
    Parse a single configuration value, converting strings to
    bool/int/float if applicable.
    """
    if not isinstance(val, (str, bytes)):
        return val
    try:
        if val.lower() in ("true", "yes", "on"):
            return True
        elif val.lower() in ("false", "no", "off"):
            return False
        else:
            try:
                if "." in val:
                    return float(val)
                else:
                    return int(val)
            except ValueError:
                return float(val)
    except (ValueError, TypeError):
        return val


def clean_config_dict(config: dict) -> dict:
    """
    Recursively clean a configuration dictionary, converting numeric strings
    and boolean strings into their proper types.
    """
    clean_dict = {}
    if not config:
        return clean_dict

    for k, v in config.items():
        if isinstance(v, dict):
            clean_dict[k] = clean_config_dict(v)
        elif isinstance(v, list):
            clean_list = []
            for item in v:
                if isinstance(item, dict):
                    clean_list.append(clean_config_dict(item))
                else:
                    clean_list.append(_parse_config_value(item))
            clean_dict[k] = clean_list
        else:
            clean_dict[k] = _parse_config_value(v)
    return clean_dict
