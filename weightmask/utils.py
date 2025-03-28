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
