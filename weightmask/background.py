import numpy as np
import sep
from scipy.ndimage import median_filter
from astropy.stats import mad_std
import warnings

def estimate_background(sci_data, mask, config):
    """
    Estimate background and background RMS using a configured method.

    Args:
        sci_data (ndarray): Science image data array
        mask (ndarray): Boolean mask of pixels to exclude from background estimation
        config (dict): Configuration dictionary for background estimation

    Returns:
        tuple: (background_map, background_rms_map) or (None, None) on failure.
    """
    method = config.get('method', 'sep').lower()
    print(f"  Estimating background using method: '{method}'")

    bkg_map, bkg_rms_map = None, None

    if method == 'sep':
        try:
            bkg = sep.Background(
                sci_data,
                mask=mask,
                bw=config.get('box_size', 128),
                bh=config.get('box_size', 128),
                fw=config.get('filter_size', 3),
                fh=config.get('filter_size', 3),
                maskthresh=0.0  # Don't threshold the mask, use it as-is
            )
            bkg_map = bkg.back()
            bkg_rms_map = bkg.rms()
            print(f"    SEP background global RMS: {bkg.globalrms:.3f}")
            
            # Check for edge artifacts and apply smoothing if needed
            edge_width = 50  # pixels from edge to check
            h, w = sci_data.shape
            edge_regions = [
                bkg_map[:edge_width, :],     # top
                bkg_map[-edge_width:, :],    # bottom  
                bkg_map[:, :edge_width],     # left
                bkg_map[:, -edge_width:]     # right
            ]
            
            # If edge values deviate significantly from center, apply smoothing
            center_median = np.median(bkg_map[edge_width:-edge_width, edge_width:-edge_width])
            for i, edge_region in enumerate(edge_regions):
                edge_median = np.median(edge_region)
                if abs(edge_median - center_median) > 50:  # 50 ADU threshold
                    print(f"    Detected edge artifact (region {i}): {edge_median:.1f} vs {center_median:.1f} ADU")
                    # Apply light smoothing to the entire background
                    from scipy.ndimage import gaussian_filter
                    bkg_map = gaussian_filter(bkg_map, sigma=2.0)
                    print(f"    Applied Gaussian smoothing to reduce edge artifacts")
                    break
        except Exception as e:
            warnings.warn(f"SEP background estimation failed: {e}. Falling back to global.", RuntimeWarning)
            try:
                bkg_val, bkg_rms_global = sep.background(sci_data, mask=mask)
                bkg_map = np.full(sci_data.shape, bkg_val, dtype=np.float32)
                bkg_rms_map = np.full(sci_data.shape, bkg_rms_global, dtype=np.float32)
                print(f"    Using global SEP RMS: {bkg_rms_global:.3f}")
            except Exception as e2:
                warnings.warn(f"Global SEP background also failed: {e2}", RuntimeWarning)
                return None, None

    elif method == 'median_filter':
        try:
            default_size = max(15, int(min(sci_data.shape) / 20) // 2 * 2 + 1) # Default to ~5% of smaller axis, ensure odd
            kernel_size = config.get('median_kernel_size', default_size)
            print(f"    Using median filter with kernel size: {kernel_size}")
            
            bkg_map = median_filter(sci_data, size=kernel_size)
            
            # For RMS, calculate a robust global value from the subtracted image
            data_sub = sci_data - bkg_map
            # Use only non-masked pixels for RMS calculation
            global_rms = mad_std(data_sub[~mask], ignore_nan=True)
            bkg_rms_map = np.full(sci_data.shape, global_rms, dtype=np.float32)
            print(f"    Median filter global RMS (mad_std): {global_rms:.3f}")

        except Exception as e:
            warnings.warn(f"Median filter background estimation failed: {e}", RuntimeWarning)
            return None, None
    else:
        warnings.warn(f"Unknown background estimation method: '{method}'", RuntimeWarning)
        return None, None

    # Ensure valid RMS values before returning
    if bkg_rms_map is not None:
        bkg_rms_map = np.where(np.isfinite(bkg_rms_map) & (bkg_rms_map > 0), bkg_rms_map, np.inf)
    
    return bkg_map, bkg_rms_map
