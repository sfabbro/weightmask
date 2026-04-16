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
        # Proactively check mask coverage. If too high, local background estimation will be unstable.
        mask_fraction = np.mean(mask)
        if mask_fraction > 0.8:
            print(f"    WARNING: High mask coverage ({mask_fraction:.1%}). Proactively falling back to global mode.")
            try:
                # Use a single large box for global estimation
                bkg = sep.Background(sci_data, mask=mask, bw=sci_data.shape[1], bh=sci_data.shape[0])
                if bkg.globalback == 0 and np.any(~mask):
                     # Fall through to robust median below
                     method = 'robust_median_fallback'
                else:
                    bkg_map = np.full(sci_data.shape, bkg.globalback, dtype=np.float32)
                    bkg_rms_map = np.full(sci_data.shape, bkg.globalrms, dtype=np.float32)
                    return bkg_map, bkg_rms_map
            except Exception as e:
                 warnings.warn(f"Global SEP fallback failed: {e}. Trying robust median.", RuntimeWarning)
                 method = 'robust_median_fallback' # Trigger the final fallback logic below
        else:
            # Try SEP background with a tiered internal retry for box size robustness
            box_size = config.get('box_size', 128)
            filter_size = config.get('filter_size', 3)
            
            for attempt, scale in enumerate([1, 2, 4]):
                current_box = box_size * scale
                try:
                    if attempt > 0:
                        print(f"    Retrying SEP with larger box size: {current_box}")
                    
                    bkg = sep.Background(
                        sci_data,
                        mask=mask,
                        bw=current_box,
                        bh=current_box,
                        fw=filter_size,
                        fh=filter_size,
                        maskthresh=0.0
                    )
                    bkg_map = bkg.back()
                    bkg_rms_map = bkg.rms()
                    print(f"    SEP background global RMS: {bkg.globalrms:.3f} (box={current_box})")
                    
                    # Edge artifact check (existing logic)
                    edge_width = 50
                    h, w = sci_data.shape
                    edge_regions = [
                        bkg_map[:edge_width, :], bkg_map[-edge_width:, :],
                        bkg_map[:, :edge_width], bkg_map[:, -edge_width:]
                    ]
                    center_median = np.median(bkg_map[edge_width:-edge_width, edge_width:-edge_width])
                    for i, edge_region in enumerate(edge_regions):
                        edge_median = np.median(edge_region)
                        if abs(edge_median - center_median) > 50:
                            from scipy.ndimage import gaussian_filter
                            bkg_map = gaussian_filter(bkg_map, sigma=2.0)
                            print(f"    Applied Gaussian smoothing to reduce edge artifacts")
                            break
                    break # Success!
                    
                except Exception as e:
                    if attempt == 2: # Final attempt failed
                        warnings.warn(f"All SEP background retries failed: {e}. Falling back to robust global median.", RuntimeWarning)
                        method = 'robust_median_fallback'
                    continue

    if method == 'robust_median_fallback' or method == 'median_filter':
        try:
            if method == 'robust_median_fallback':
                print("    Using robust global median fallback.")
                kernel_size = 0 # Dummy for log
            else:
                default_size = max(15, int(min(sci_data.shape) / 20) // 2 * 2 + 1)
                kernel_size = config.get('median_kernel_size', default_size)
                print(f"    Using median filter with kernel size: {kernel_size}")
            
            if kernel_size > 0:
                bkg_map = median_filter(sci_data, size=kernel_size)
            else:
                # Global robust median
                valid_data = sci_data[~mask] if np.any(~mask) else sci_data
                step = max(1, valid_data.size // 100000)
                bkg_val = np.median(valid_data.ravel()[::step])
                bkg_map = np.full(sci_data.shape, bkg_val, dtype=np.float32)
            
            data_sub = sci_data - bkg_map
            valid_sub = data_sub[~mask] if np.any(~mask) else data_sub
            step_sub = max(1, valid_sub.size // 100000)
            global_rms = mad_std(valid_sub.ravel()[::step_sub], ignore_nan=True)
            bkg_rms_map = np.full(sci_data.shape, global_rms, dtype=np.float32)
            print(f"    Final RMS (mad_std): {global_rms:.3f}")

        except Exception as e:
            warnings.warn(f"Final background fallback failed: {e}", RuntimeWarning)
            return None, None
    elif method not in ['sep', 'robust_median_fallback', 'median_filter']:
        warnings.warn(f"Unknown background estimation method: '{method}'", RuntimeWarning)
        return None, None

    # Ensure valid RMS values before returning (LSST-inspired sanity check)
    if bkg_rms_map is not None:
        # Protect against zeros or non-finite values which break weight maps
        invalid = ~np.isfinite(bkg_rms_map) | (bkg_rms_map <= 0)
        if np.any(invalid):
             bkg_rms_map = np.where(~invalid, bkg_rms_map, np.inf)
    
    return bkg_map, bkg_rms_map
