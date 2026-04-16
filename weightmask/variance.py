import numpy as np
import warnings
from astropy.stats import mad_std
from scipy.stats import linregress

def _calculate_empirical_noise_params(sci_data, obj_mask, patch_size, robust_sigma_clip):
    """
    Internal: Calculate gain and read noise empirically from the image data.

    Args:
        sci_data (ndarray): The full science image data.
        obj_mask (ndarray): Boolean mask where True indicates objects to ignore.
        patch_size (int): The size of the patches to analyze.
        robust_sigma_clip (float): Sigma value for clipping outlier patches before fitting.

    Returns:
        tuple: (empirical_gain, empirical_read_noise_e) or (None, None) on failure.
    """
    print("    Starting empirical noise parameter calculation...")
    img_shape = sci_data.shape
    patch_variances = []
    patch_medians = []

    # Iterate over patches
    for y in range(0, img_shape[0], patch_size):
        for x in range(0, img_shape[1], patch_size):
            patch_slice = (slice(y, y + patch_size), slice(x, x + patch_size))
            patch_data = sci_data[patch_slice]
            patch_mask = obj_mask[patch_slice]

            # Use only non-masked pixels in the patch
            valid_pixels = patch_data[~patch_mask]

            # Ensure enough valid pixels are available
            if valid_pixels.size < 100:
                continue

            # Calculate robust statistics for the patch
            patch_median = np.median(valid_pixels)
            # Use Median Absolute Deviation (MAD) for a robust standard deviation
            patch_std_robust = mad_std(valid_pixels, ignore_nan=True)

            if patch_std_robust > 1e-6: # Avoid patches with zero variance
                patch_variances.append(patch_std_robust**2)
                patch_medians.append(patch_median)

    if len(patch_medians) < 10:
        print("    ERROR: Not enough valid patches found to perform empirical fit.")
        return None, None

    # Convert to numpy arrays for analysis
    patch_medians = np.array(patch_medians)
    patch_variances = np.array(patch_variances)

    # Robustly filter outlier patches (e.g., those with residual CRs or nebulosity)
    med_var = np.median(patch_variances)
    std_var = mad_std(patch_variances)
    good_patches_mask = (patch_variances < med_var + robust_sigma_clip * std_var)

    if np.sum(good_patches_mask) < 10:
        print("    ERROR: Not enough stable patches after outlier clipping.")
        return None, None

    # Perform linear regression: Variance = A + B * Median
    # B = 1 / gain, A = read_noise_adu^2
    fit_medians = patch_medians[good_patches_mask]
    fit_variances = patch_variances[good_patches_mask]

    try:
        slope, intercept, r_value, p_value, std_err = linregress(fit_medians, fit_variances)

        if slope < 1e-9 or np.isnan(slope) or np.isnan(intercept):
            print("    ERROR: Linear regression resulted in invalid slope or intercept.")
            return None, None

        # Derive gain and read noise from fit parameters
        empirical_gain = 1.0 / slope
        # Read noise in ADU is sqrt of intercept (variance at zero signal)
        read_noise_adu = np.sqrt(max(0, intercept))
        # Convert read noise to electrons
        empirical_read_noise_e = read_noise_adu * empirical_gain

        print(f"    Empirical Fit Results: Gain = {empirical_gain:.3f} e-/ADU, Read Noise = {empirical_read_noise_e:.3f} e-")
        return empirical_gain, empirical_read_noise_e

    except Exception as e:
        print(f"    ERROR: Linear regression for noise parameters failed: {e}")
        return None, None


def _calculate_inverse_variance_theoretical(sky_map, flat_map, gain, read_noise_e, epsilon):
    """
    Internal: Calculate inverse variance based on theoretical noise model.
    """
    valid_flat_mask = (flat_map > epsilon)
    safe_flat = np.where(valid_flat_mask, flat_map, epsilon)
    safe_sky = np.maximum(sky_map, 0.0)

    # Variance in electrons = (Signal in electrons from sky) + (Read Noise in electrons)^2
    variance_e = (safe_sky / safe_flat) * gain + read_noise_e**2

    # Inverse variance in ADU^2 = gain^2 / variance_e
    inv_variance = np.zeros_like(variance_e)
    valid_variance = variance_e > epsilon
    inv_variance[valid_variance] = gain**2 / variance_e[valid_variance]

    # Mask out invalid regions
    inv_variance[~valid_flat_mask] = 0.0

    return inv_variance.astype(np.float32)


def _rescale_variance_robust(inv_variance, sci_data, sky_map, obj_mask, epsilon):
    """
    Internal: Robustly rescale variance so that SNR standard deviation is 1.
    Mimics LSST ScaleVarianceTask.
    """
    if inv_variance is None or sci_data is None or obj_mask is None:
        return inv_variance

    # SNR = (data - sky) * sqrt(inv_variance)
    # Background should have SNR stdev of 1.0
    data_sub = sci_data - sky_map
    snr = data_sub[~obj_mask] * np.sqrt(inv_variance[~obj_mask])
    
    # Filter out non-finite SNRs
    snr = snr[np.isfinite(snr)]
    if snr.size < 100:
        return inv_variance

    # ⚡ Bolt: Subsample large arrays before calculating global robust statistics
    step = max(1, snr.size // 100000)

    # Use Interquartile Range for robust stdev
    q1, q3 = np.percentile(snr[::step], (25, 75))
    robust_stdev = 0.7413 * (q3 - q1)
    
    if robust_stdev > 0:
        scale_factor = 1.0 / (robust_stdev**2)
        print(f"    Rescaling variance plane by factor {scale_factor:.3f} (SNR robust stdev was {robust_stdev:.3f})")
        return inv_variance * scale_factor
    
    return inv_variance


def _unbias_variance(inv_variance, sci_data, sky_map, gain, epsilon):
    """
    Internal: Remove signal-dependent Poisson noise to get background-only variance.
    Mimics LSST remove_signal_from_variance.
    """
    if inv_variance is None or gain <= 0:
        return inv_variance
    
    # Var_total = Var_bg + Signal/Gain
    # 1/InvVar_total = 1/InvVar_bg + Signal/Gain
    # 1/InvVar_bg = 1/InvVar_total - Signal/Gain
    
    # Signal is (sci_data - sky_map)
    signal = np.maximum(sci_data - sky_map, 0.0)
    
    # Work in variance space
    with np.errstate(divide='ignore', invalid='ignore'):
        var_total = 1.0 / inv_variance
        var_bg = var_total - (signal / gain)
        
        # Ensure positivity
        var_bg = np.maximum(var_bg, 1e-6) # Floor at tiny noise
        new_inv_variance = 1.0 / var_bg
        
    new_inv_variance[~np.isfinite(new_inv_variance)] = 0.0
    return new_inv_variance.astype(np.float32)


def _calculate_inverse_variance_rms(bkg_rms_map, epsilon):
    """
    Internal: Calculate inverse variance based on SEP's background RMS map.
    """
    if bkg_rms_map is None:
        warnings.warn("RMS map is None, cannot calculate variance from RMS.", RuntimeWarning)
        return None

    variance_adu = bkg_rms_map**2
    inv_variance = np.where(variance_adu > epsilon, 1.0 / variance_adu, 0.0)
    inv_variance[~np.isfinite(inv_variance)] = 0.0

    return inv_variance.astype(np.float32)


def calculate_inverse_variance(variance_cfg, sky_map, flat_map, bkg_rms_map, sci_data=None, obj_mask=None):
    """
    Calculate inverse variance map using the specified method.

    Args:
        variance_cfg (dict): Configuration dictionary for variance calculation.
        sky_map (ndarray): Background sky map in ADU.
        flat_map (ndarray): Flat field response map.
        bkg_rms_map (ndarray): Background RMS map in ADU (from SEP, used by 'rms_map').
        sci_data (ndarray, optional): Full science data array, required for 'empirical_fit'.
        obj_mask (ndarray, optional): Object mask, required for 'empirical_fit'.

    Returns:
        ndarray or None: Inverse variance map, or None if method is invalid or prerequisites missing.
    """
    method = variance_cfg.get('method', 'theoretical').lower()
    epsilon = variance_cfg.get('epsilon', 1e-9)
    print(f"  Calculating Inverse Variance using method: '{method}'")

    # Get header/default gain and read noise for theoretical method
    gain = variance_cfg.get('gain', 1.0)
    read_noise_e = variance_cfg.get('read_noise', 0.0)

    inv_var = None
    if method == 'empirical_fit':
        if sci_data is None or obj_mask is None:
            warnings.warn("Empirical fit method requires science data and object mask.", RuntimeWarning)
            return None
        
        patch_size = variance_cfg.get('empirical_patch_size', 128)
        clip_sigma = variance_cfg.get('empirical_clip_sigma', 3.0)
        
        emp_gain, emp_rn_e = _calculate_empirical_noise_params(sci_data, obj_mask, patch_size, clip_sigma)
        
        if emp_gain is None or emp_rn_e is None:
            print("  WARNING: Empirical fit failed. Falling back to theoretical method with default/header values.")
            inv_var = _calculate_inverse_variance_theoretical(sky_map, flat_map, gain, read_noise_e, epsilon)
        else:
            inv_var = _calculate_inverse_variance_theoretical(sky_map, flat_map, emp_gain, emp_rn_e, epsilon)
            gain = emp_gain # For unbiasing below

    elif method == 'theoretical':
        if flat_map is None or sky_map is None:
            warnings.warn("Theoretical method requires flat and sky maps.", RuntimeWarning)
            return None
        inv_var = _calculate_inverse_variance_theoretical(sky_map, flat_map, gain, read_noise_e, epsilon)

    elif method == 'rms_map':
        inv_var = _calculate_inverse_variance_rms(bkg_rms_map, epsilon)

    else:
        warnings.warn(f"Invalid variance calculation method '{method}'.", RuntimeWarning)
        return None

    # Post-processing: Rescaling and Unbiasing
    if inv_var is not None:
        # 1. Option to rescale variance to match observed noise (SNR=1 check)
        if variance_cfg.get('rescale_variance', False) and sci_data is not None and obj_mask is not None:
            inv_var = _rescale_variance_robust(inv_var, sci_data, sky_map, obj_mask, epsilon)
            
        # 2. Option to unbias variance (remove Poisson contribution of objects)
        if variance_cfg.get('unbias_variance', False) and sci_data is not None:
            inv_var = _unbias_variance(inv_var, sci_data, sky_map, gain, epsilon)

    return inv_var
