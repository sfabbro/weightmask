import numpy as np
from astroscrappy import detect_cosmics
import warnings
from scipy.ndimage import convolve

def _get_psf_peakiness(fwhm):
    """
    Calculate the expected peakiness (ratio of central pixel to total 3x3 flux)
    of a 2D Gaussian PSF.
    """
    sigma = fwhm / 2.355
    # Create 3x3 Gaussian kernel
    x, y = np.mgrid[-1:2, -1:2]
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    # Peakiness is the central value
    return kernel[1, 1]

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

        # LSST-inspired PSF protection:
        # If a detected CR is actually "fuzzy" enough to be a star according to the PSF,
        # we might want to unmask it to prevent over-flagging star cores.
        if config.get('psf_aware', True):
            psf_fwhm = config.get('psf_fwhm_guess', 3.0)
            print(f"    Applying PSF-aware protection (FWHM guess: {psf_fwhm:.1f} pix)")
            
            # We MUST subtract background to get true PSF peakiness
            # Estimate local background using a simple median or just a global mode
            # For robustness, we'll use a local 5x5 median filter or subtract the global mode
            sky_est = np.median(sci_data) # Global fallback
            sci_sub = np.maximum(sci_data - sky_est, 0.0)
            
            # Calculate peakiness of every pixel: I(x,y) / sum(I_3x3)
            # We use a 3x3 uniform filter for the sum
            uniform_3x3 = np.ones((3, 3), dtype=np.float32)
            local_flux_sum = convolve(sci_sub, uniform_3x3, mode='constant', cval=0.0)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                # Peakiness of the SIGNAL, not the raw counts
                peakiness = sci_sub / local_flux_sum
            
            # Expected peakiness for a star
            psf_peak_thresh = _get_psf_peakiness(psf_fwhm)
            # Add a safety margin (e.g. 10%)
            cr_thresh = psf_peak_thresh * 1.1
            
            # If a pixel in the CR mask is LESS peaky than cr_thresh, 
            # and it is reasonably bright, we unmask it as a potential star core.
            # Use SNR threshold for protection to avoid saving noise blobs.
            # If bkg_rms_map is available, use it for local SNR threshold.
            if bkg_rms_map is not None:
                snr_map = sci_sub / (bkg_rms_map / gain)
                star_protection_mask = (peakiness < cr_thresh) & (snr_map > 5.0)
            else:
                star_protection_mask = (peakiness < cr_thresh) & (sci_sub > 5.0 * read_noise / gain)
            
            protected_count = np.sum(crmask_bool.astype(bool) & star_protection_mask)
            if protected_count > 0:
                print(f"    PSF protection: Saved {protected_count} pixels (likely star cores) from CR flagging.")
                crmask_bool = crmask_bool.astype(bool) & (~star_protection_mask)

        # Apply morphological dilation to catch the wings of the cosmic rays
        if config.get('dilate_cr', True):
            print("    Applying morphological dilation to cosmic ray mask...")
            from scipy.ndimage import binary_dilation as scipy_binary_dilation
            dilation_radius = config.get('dilation_radius', 1)
            
            # Create a disk structural element manually to avoid importing skimage
            y, x = np.ogrid[-dilation_radius:dilation_radius+1, -dilation_radius:dilation_radius+1]
            selem = (x**2 + y**2) <= dilation_radius**2

            if selem.size > 0:
                crmask_bool = scipy_binary_dilation(crmask_bool, structure=selem)

        # Only return newly detected pixels (not already in existing_mask)
        cr_add_mask = crmask_bool & (~existing_mask)
        
        num_new_pixels = np.sum(cr_add_mask)
        if num_new_pixels > 0:
             print(f"  Detected {num_new_pixels} new cosmic ray pixels.")
        
        return cr_add_mask

    except Exception as e:
        print(f"  Astroscrappy failed: {e}")
        return np.zeros(sci_data.shape, dtype=bool)