import numpy as np
from astroscrappy import detect_cosmics

def detect_cosmic_rays(sci_data, existing_mask, saturation_level, gain, read_noise, config):
    """
    Detect cosmic rays in the science data.
    
    Args:
        sci_data (ndarray): Science image data array
        existing_mask (ndarray): Boolean mask of already masked pixels
        saturation_level (float): Saturation level for the detector
        gain (float): Gain value in e-/ADU
        read_noise (float): Read noise in electrons
        config (dict): Configuration dicionary for cosmic ray detection
        
    Returns:
        ndarray: Boolean mask of newly detected cosmic ray pixels
    """
    try:
        # Use astroscrappy to detect cosmic rays
        crmask_bool, _ = detect_cosmics(
            sci_data, 
            inmask=existing_mask, 
            satlevel=saturation_level, 
            gain=gain, 
            readnoise=read_noise, 
            sigclip=config['sigclip'], 
            objlim=config['objlim'], 
            verbose=False
        )
        
        # Only return newly detected pixels (not already in existing_mask)
        cr_add_mask = crmask_bool & (~existing_mask)
        return cr_add_mask
        
    except Exception as e:
        print(f"  Astroscrappy failed: {e}")
        return np.zeros(sci_data.shape, dtype=bool)
