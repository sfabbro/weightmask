import numpy as np
from astroscrappy import detect_cosmics
import warnings

def detect_cosmic_rays(sci_data, existing_mask, saturation_level, gain, read_noise, config, bkg_rms_map=None):
    """
    Detect cosmic rays in the science data.

    Args:
        sci_data (ndarray): Science image data array
        existing_mask (ndarray): Boolean mask of already masked pixels
        saturation_level (float): Saturation level for the detector
        gain (float): Gain value in e-/ADU
        read_noise (float): Read noise in electrons
        config (dict): Configuration dictionary for cosmic ray detection
        bkg_rms_map (ndarray, optional): Background RMS map for dynamic sigclip.

    Returns:
        ndarray: Boolean mask of newly detected cosmic ray pixels
    """
    sigclip = config.get('sigclip', 4.5)
    objlim = config.get('objlim', 5.0)

    # Dynamically adjust sigclip based on background noise if enabled
    if config.get('dynamic_sigclip', True) and bkg_rms_map is not None:
        try:
            # Use the median of the background RMS as a robust noise indicator
            median_rms = np.median(bkg_rms_map[bkg_rms_map > 0])
            if np.isfinite(median_rms) and median_rms > 0.1:
                # Heuristic: Lower sigclip for noisier images to increase sensitivity,
                # but don't go below a reasonable floor (3.0) or above ceiling (8.0)
                # Scaling factors: base=4.5, noise_scale=10.0, offset=1.0
                dynamic_clip = 4.5 * (10.0 / (median_rms + 1.0))
                sigclip = np.clip(dynamic_clip, 3.0, 8.0)  # Bound between 3.0-8.0 sigma
                print(f"    Dynamically adjusted sigclip to {sigclip:.2f} based on background RMS of {median_rms:.2f}")
        except Exception as e:
            warnings.warn(f"Dynamic sigclip adjustment failed: {e}", RuntimeWarning)

    try:
        # Use astroscrappy to detect cosmic rays
        crmask_bool, _ = detect_cosmics(
            sci_data,
            inmask=existing_mask,
            satlevel=saturation_level,
            gain=gain,
            readnoise=read_noise,
            sigclip=sigclip,
            objlim=objlim,
            verbose=False
        )

        # Only return newly detected pixels (not already in existing_mask)
        cr_add_mask = crmask_bool & (~existing_mask)
        return cr_add_mask

    except Exception as e:
        print(f"  Astroscrappy failed: {e}")
        return np.zeros(sci_data.shape, dtype=bool)