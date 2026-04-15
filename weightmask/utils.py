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
        
    match = re.match(r'^(.*?)(?:\[(\d+)\])?$', filepath)
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
    return np.where((mask_data & bit_flag) > 0, 1, 0).astype(np.uint8)

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
                elif isinstance(item, (str, bytes)):
                    try:
                        if item.lower() in ('true', 'yes', 'on'): clean_list.append(True)
                        elif item.lower() in ('false', 'no', 'off'): clean_list.append(False)
                        else:
                            try:
                                if '.' in item: clean_list.append(float(item))
                                else: clean_list.append(int(item))
                            except ValueError:
                                clean_list.append(float(item))
                    except (ValueError, TypeError):
                        clean_list.append(item)
                else:
                    clean_list.append(item)
            clean_dict[k] = clean_list
        elif isinstance(v, (str, bytes)):
            try:
                if v.lower() in ('true', 'yes', 'on'): clean_dict[k] = True
                elif v.lower() in ('false', 'no', 'off'): clean_dict[k] = False
                else:
                    try:
                        if '.' in v: clean_dict[k] = float(v)
                        else: clean_dict[k] = int(v)
                    except ValueError:
                        clean_dict[k] = float(v)
            except (ValueError, TypeError):
                clean_dict[k] = v
        else:
            clean_dict[k] = v
    return clean_dict
