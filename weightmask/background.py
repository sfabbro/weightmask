import numpy as np
import sep

def estimate_background(sci_data, mask, config):
    """
    Estimate background and background RMS using SEP.
    
    Args:
        sci_data (ndarray): Science image data array
        mask (ndarray): Boolean mask of pixels to exclude from background estimation
        config (dict): Configuration dictionary for background estimation
        
    Returns:
        tuple: (background_map, background_rms_map)
    """
    try:
        # Use SEP for background estimation
        bkg = sep.Background(
            sci_data, 
            mask=mask, 
            bw=config['box_size'], 
            bh=config['box_size'], 
            fw=config['filter_size'], 
            fh=config['filter_size']
        )
        bkg_map = bkg.back()
        bkg_rms_map = bkg.rms()
        print(f"  Background global RMS: {bkg.globalrms:.3f}")
        
    except Exception as e:
        # Fallback to global background estimation if local fails
        print(f"  Background estimation failed: {e}. Using global.")
        bkg_val, bkg_rms_global = sep.background(sci_data, mask=mask)
        bkg_map = np.full(sci_data.shape, bkg_val, dtype=np.float32)
        bkg_rms_map = np.full(sci_data.shape, bkg_rms_global, dtype=np.float32)
        print(f"  Using global RMS: {bkg_rms_global:.3f}")
    
    # Ensure valid RMS values
    bkg_rms_map = np.where(np.isfinite(bkg_rms_map) & (bkg_rms_map > 0), bkg_rms_map, np.inf)
    
    return bkg_map, bkg_rms_map
