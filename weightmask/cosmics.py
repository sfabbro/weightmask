import warnings

import numpy as np
from astroscrappy import detect_cosmics
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


def _adjust_dynamic_sigclip(config, bkg_rms_map, default_sigclip):
    """Dynamically adjust sigclip based on background noise."""
    sigclip = default_sigclip
    if not config.get("dynamic_sigclip", True) or bkg_rms_map is None:
        return sigclip

    try:
        valid_rms = bkg_rms_map[bkg_rms_map > 0]
        step = max(1, len(valid_rms) // 100000)
        median_rms = np.median(valid_rms[::step]) if len(valid_rms) > 0 else 0.0
        if np.isfinite(median_rms) and median_rms > 0.1:
            dynamic_clip = 4.5 * (10.0 / (median_rms + 1.0))
            sigclip = np.clip(dynamic_clip, 3.0, 8.0)
            print(f"    Dynamically adjusted sigclip to {sigclip:.2f} based on background RMS of {median_rms:.2f}")
    except Exception as e:
        warnings.warn(f"Dynamic sigclip adjustment failed: {e}", RuntimeWarning)

    return float(sigclip)


def _apply_psf_protection(crmask_bool, sci_data, config, gain, read_noise, bkg_rms_map):
    """Apply PSF-aware protection to prevent over-flagging star cores."""
    if not config.get("psf_aware", True):
        return crmask_bool

    psf_fwhm = config.get("psf_fwhm_guess", 3.0)
    print(f"    Applying PSF-aware protection (FWHM guess: {psf_fwhm:.1f} pix)")

    sky_est = np.median(sci_data[::10, ::10])
    sci_sub = np.maximum(sci_data - sky_est, 0.0)

    uniform_3x3 = np.ones((3, 3), dtype=np.float32)
    local_flux_sum = convolve(sci_sub, uniform_3x3, mode="constant", cval=0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        peakiness = sci_sub / local_flux_sum

    psf_peak_thresh = _get_psf_peakiness(psf_fwhm)
    cr_thresh = psf_peak_thresh * 1.1

    if bkg_rms_map is not None:
        snr_map = sci_sub / (bkg_rms_map / gain)
        star_protection_mask = (peakiness < cr_thresh) & (snr_map > 5.0)
    else:
        star_protection_mask = (peakiness < cr_thresh) & (sci_sub > 5.0 * read_noise / gain)

    protected_count = np.sum(crmask_bool.astype(bool) & star_protection_mask)
    if protected_count > 0:
        print(f"    PSF protection: Saved {protected_count} pixels (likely star cores) from CR flagging.")
        crmask_bool = crmask_bool.astype(bool) & (~star_protection_mask)

    return crmask_bool


def _apply_morphological_dilation(crmask_bool, config):
    """Apply morphological dilation to catch the wings of the cosmic rays."""
    if not config.get("dilate_cr", True):
        return crmask_bool

    print("    Applying morphological dilation to cosmic ray mask...")
    from skimage.morphology import binary_dilation, disk

    dilation_radius = config.get("dilation_radius", 1)
    selem = disk(dilation_radius)
    if selem.size > 0:
        crmask_bool = binary_dilation(crmask_bool, footprint=selem)

    return crmask_bool


def detect_cosmic_rays(
    sci_data,
    existing_mask,
    saturation_level,
    gain,
    read_noise,
    config,
    bkg_rms_map=None,
):
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
    sigclip = _adjust_dynamic_sigclip(config, bkg_rms_map, default_sigclip=config.get("sigclip", 4.5))
    objlim = config.get("objlim", 5.0)

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
            verbose=False,
        )

        crmask_bool = _apply_psf_protection(crmask_bool, sci_data, config, gain, read_noise, bkg_rms_map)

        crmask_bool = _apply_morphological_dilation(crmask_bool, config)

        # Only return newly detected pixels (not already in existing_mask)
        cr_add_mask = crmask_bool & (~existing_mask)

        num_new_pixels = np.sum(cr_add_mask)
        if num_new_pixels > 0:
            print(f"  Detected {num_new_pixels} new cosmic ray pixels.")

        return cr_add_mask

    except Exception as e:
        print(f"  Astroscrappy failed: {e}")
        return np.zeros(sci_data.shape, dtype=bool)
