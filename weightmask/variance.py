import numpy as np
import warnings

def _calculate_inverse_variance_theoretical(sky_map, flat_map, gain, read_noise_e, epsilon):
    """
    Internal: Calculate inverse variance based on theoretical noise model.
    (Based on the original function)
    """
    # Create mask for valid flat values
    valid_flat_mask = (flat_map > epsilon)
    safe_flat = np.where(valid_flat_mask, flat_map, epsilon)
    safe_sky = np.maximum(sky_map, 0.0) # Ensure non-negative sky

    # Noise components in electrons^2
    read_noise_var_e = read_noise_e**2
    # Poisson noise from sky signal (in ADU) converted to electrons
    # Assumes sky_map is in ADU *before* flat-fielding correction conceptually
    # signal_var_e = safe_sky * gain # Variance(photons) = Signal(photons)
    # Let's recalculate variance assuming sky_map is the *observed* sky in ADU
    # Signal in observed ADU = sky_map
    # Signal in electrons = sky_map * gain
    # Variance in electrons (Poisson) = sky_map * gain
    signal_var_e = safe_sky * gain

    total_var_e = read_noise_var_e + signal_var_e

    # Convert total electron variance to observed ADU variance
    # Variance_obs(ADU) = Variance_total(e-) * (Flat^2 / Gain^2) <- This isn't quite right
    # Let's use the standard formula Var(ADU) = (Signal(e-) + RN(e-)^2) / Gain^2
    # Signal(e-) here should be the signal *before* flat fielding: sky_map * gain / safe_flat
    # This gets complicated if sky_map is derived *after* flat fielding.
    # Assuming sky_map is the *true* sky in ADU:
    # Signal_e = sky_map * safe_flat * gain
    # Variance_e = Signal_e + read_noise_var_e
    # Variance_ADU = Variance_e / gain^2 # Variance in ideal ADU before flat
    # Variance_ADU_obs = Variance_ADU * safe_flat^2 # Variance after flat applied

    # Simpler approach based on original code's apparent logic:
    # Variance_ADU_obs = (sky_map * flat_map * gain + read_noise_e**2) / (gain**2 * flat_map**2) ?? No.

    # Let's stick to the original code's calculation:
    # total_var_e = read_noise_var_e + (safe_sky * safe_flat * gain) # Total variance in e- corresponding to observed ADU counts
    # variance_adu = total_var_e / ((gain**2 * safe_flat**2) + epsilon) # Variance in observed ADU^2, accounting for flat

    # Revised theoretical calculation based on standard propagation:
    # Variance(ADU) = [ sky(ADU) / gain * flat^2 + (read_noise_e/gain)^2 * flat^2 ] ??? No.
    # Correct: Var(ADU_final) = Var( (sky_ADU_orig * Flat + Bias_ADU_orig - Bias_ADU_master) / Flat )
    # Assume bias subtracted. Var(ADU_final) = Var(sky_ADU_orig) ~= [sky_ADU_orig * gain + RN_e^2] / gain^2
    # So InvVar should be ~ gain^2 / (sky_ADU_orig*gain + RN_e^2)
    # If sky_map is the observed sky: sky_map = sky_ADU_orig * Flat
    # sky_ADU_orig = sky_map / Flat
    # InvVar ~ gain^2 / ( (sky_map/safe_flat)*gain + read_noise_var_e ) --- Let's use this one.

    variance_e = (safe_sky / safe_flat) * gain + read_noise_var_e # Variance in electrons related to intrinsic signal + read noise
    # Add epsilon to denominator gain^2 to prevent division by zero if gain is somehow zero
    inv_variance = (gain**2 + epsilon) / (variance_e + epsilon)

    # Mask out invalid regions
    inv_variance[~valid_flat_mask] = 0.0
    # Ensure variance_e is positive before division
    inv_variance[variance_e < epsilon] = 0.0

    return inv_variance.astype(np.float32)


def _calculate_inverse_variance_rms(bkg_rms_map, epsilon):
    """
    Internal: Calculate inverse variance based on SEP's background RMS map.

    Assumes bkg_rms_map is the standard deviation in observed ADU.
    """
    if bkg_rms_map is None:
        warnings.warn("RMS map is None, cannot calculate variance from RMS.", RuntimeWarning)
        return None

    variance_adu = bkg_rms_map**2
    # Calculate inverse variance with protection against division by zero / non-positive variance
    inv_variance = np.where(variance_adu > epsilon, 1.0 / variance_adu, 0.0)

    # Mask out non-finite values just in case
    inv_variance[~np.isfinite(inv_variance)] = 0.0

    return inv_variance.astype(np.float32)


def calculate_inverse_variance(method, sky_map, flat_map, gain, read_noise_e, bkg_rms_map, epsilon):
    """
    Calculate inverse variance map using the specified method.

    Args:
        method (str): Calculation method ('theoretical' or 'rms_map').
        sky_map (ndarray): Background sky map in ADU (used by 'theoretical').
        flat_map (ndarray): Flat field response map (used by 'theoretical').
        gain (float): Gain in e-/ADU (used by 'theoretical').
        read_noise_e (float): Read noise in electrons (used by 'theoretical').
        bkg_rms_map (ndarray): Background RMS map in ADU (from SEP, used by 'rms_map').
        epsilon (float): Small value to avoid division by zero.

    Returns:
        ndarray or None: Inverse variance map, or None if method is invalid or prerequisites missing.
    """
    method = method.lower()
    print(f"  Calculating Inverse Variance using method: '{method}'")

    if method == 'theoretical':
        if flat_map is None:
             warnings.warn("Flat map is None, cannot calculate theoretical variance.", RuntimeWarning)
             return None
        if sky_map is None:
             warnings.warn("Sky map is None, cannot calculate theoretical variance.", RuntimeWarning)
             return None
        return _calculate_inverse_variance_theoretical(sky_map, flat_map, gain, read_noise_e, epsilon)
    elif method == 'rms_map':
        return _calculate_inverse_variance_rms(bkg_rms_map, epsilon)
    else:
        warnings.warn(f"Invalid variance calculation method '{method}'. Choose 'theoretical' or 'rms_map'.", RuntimeWarning)
        return None