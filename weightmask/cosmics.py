import warnings

import numpy as np
from scipy.ndimage import convolve

from .utils import rms_or_robust, rms_valid_mask, robust_rms

try:
    from astroscrappy import detect_cosmics
except Exception:  # pragma: no cover - exercised indirectly in environments without astroscrappy
    detect_cosmics = None


def _get_psf_peakiness(fwhm):
    """
    Calculate the expected peakiness (ratio of central pixel to total 3x3 flux)
    of a 2D Gaussian PSF.
    """
    fwhm = max(float(fwhm), 0.1)
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
        # Only measured pixels: the ``inf`` sentinel is not a noisy measurement,
        # and including it would let a mostly-unmeasured chip pose as noiseless.
        valid_rms = bkg_rms_map[rms_valid_mask(bkg_rms_map)]
        step = max(1, len(valid_rms) // 100000)
        median_rms = np.median(valid_rms[::step]) if len(valid_rms) > 0 else 0.0
        if np.isfinite(median_rms) and median_rms > 0.1:
            dynamic_clip = 4.5 * (10.0 / (median_rms + 1.0))
            sigclip = np.clip(dynamic_clip, 3.0, 8.0)
            print(f"    Dynamically adjusted sigclip to {sigclip:.2f} based on background RMS of {median_rms:.2f}")
    except Exception as e:
        warnings.warn(f"Dynamic sigclip adjustment failed: {e}", RuntimeWarning)

    return float(sigclip)


def _adjust_dynamic_objlim(config, existing_mask, default_objlim):
    """Increase object protection in crowded scenes with many masked pixels."""
    objlim = float(default_objlim)
    if not config.get("dynamic_objlim", True) or existing_mask is None:
        return objlim

    coverage = float(np.mean(existing_mask))
    if coverage <= 0:
        return objlim
    objlim *= 1.0 + min(coverage * 4.0, 1.5)
    print(f"    Dynamically adjusted objlim to {objlim:.2f} based on pre-mask coverage {coverage:.2%}")
    return float(objlim)


def _header_value(header, key):
    if header is None:
        return None
    get = getattr(header, "get", None)
    if not callable(get):
        return None
    try:
        return get(key, None)
    except Exception:
        return None


def _fwhm_from_header(header, default):
    """Pixels from SEEING (arcsec) and a pixel scale, else ``default``."""
    seeing = _header_value(header, "SEEING")
    pix = _header_value(header, "PIXSCAL1")
    if pix is None:
        pix = _header_value(header, "PIXSCALE")
    if pix is None:
        pix = _header_value(header, "PIXSIZE")
    try:
        seeing_f = float(seeing)
        pix_f = float(pix)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(seeing_f) or not np.isfinite(pix_f) or pix_f <= 0 or seeing_f <= 0:
        return float(default)
    return seeing_f / pix_f


def _apply_psf_protection(crmask_bool, sci_data, config, gain, read_noise, bkg_rms_map, sky_map=None, header=None):
    """Apply PSF-aware protection to prevent over-flagging star cores."""
    if not config.get("psf_aware", True):
        return crmask_bool

    psf_fwhm = _fwhm_from_header(header, config.get("psf_fwhm_guess", 3.0))
    print(f"    Applying PSF-aware protection (FWHM: {psf_fwhm:.1f} pix)")

    if sky_map is not None and np.shape(sky_map) == np.shape(sci_data):
        sci_sub = np.maximum(np.asarray(sci_data, dtype=np.float32) - np.asarray(sky_map, dtype=np.float32), 0.0)
    else:
        sampled = sci_data[::10, ::10]
        finite_sampled = sampled[np.isfinite(sampled)]
        sky_est = np.median(finite_sampled) if finite_sampled.size > 0 else 0.0
        sci_sub = np.maximum(sci_data - sky_est, 0.0)

    uniform_3x3 = np.ones((3, 3), dtype=np.float32)
    local_flux_sum = convolve(sci_sub, uniform_3x3, mode="constant", cval=0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        peakiness = sci_sub / local_flux_sum

    psf_peak_thresh = _get_psf_peakiness(psf_fwhm)
    cr_thresh = psf_peak_thresh * 1.1

    if bkg_rms_map is not None:
        # Detection threshold: with no measurable RMS the SNR test cannot be
        # asserted, so no protection is claimed there and those pixels stay
        # eligible for flagging -- the conservative direction for a gate whose
        # failure mode is leaving a cosmic ray unmasked.
        valid_rms = rms_valid_mask(bkg_rms_map)
        usable_rms = np.maximum(np.where(valid_rms, bkg_rms_map, 1.0), 1e-6)
        snr_map = np.where(valid_rms, sci_sub / usable_rms, 0.0)
        star_protection_mask = (peakiness < cr_thresh) & (snr_map > 5.0)
    else:
        gain_val = max(float(gain), 1e-6)
        star_protection_mask = (peakiness < cr_thresh) & (sci_sub > 5.0 * read_noise / gain_val)

    protected_count = np.count_nonzero(crmask_bool.astype(bool) & star_protection_mask)
    if protected_count > 0:
        print(f"    PSF protection: Saved {protected_count} pixels (likely star cores) from CR flagging.")
        crmask_bool = crmask_bool.astype(bool) & (~star_protection_mask)

    return crmask_bool


def _apply_morphological_dilation(crmask_bool, config):
    """Apply morphological dilation to catch the wings of the cosmic rays."""
    if not config.get("dilate_cr", True):
        return crmask_bool

    print("    Applying morphological dilation to cosmic ray mask...")
    from skimage.morphology import dilation, disk

    dilation_radius = config.get("dilation_radius", 1)
    selem = disk(dilation_radius)
    crmask_bool = dilation(crmask_bool, footprint=selem)

    return crmask_bool


def _axis_lengths(region):
    """Major/minor axis lengths of a region (skimage >= 0.26 spelling)."""
    return float(region.axis_major_length), float(region.axis_minor_length)


def _post_filter_components(crmask_bool, sci_data, bkg_rms_map, config):
    """Reject large, diffuse components that are unlikely to be cosmic rays."""
    from skimage.measure import label, regionprops

    labeled = label(crmask_bool.astype(np.uint8), connectivity=2)
    if labeled.max() == 0:
        return crmask_bool

    max_component_area = int(config.get("max_component_area", 12))
    min_contrast_sigma = float(config.get("min_component_contrast_sigma", 4.0))
    filtered = np.zeros_like(crmask_bool, dtype=bool)
    if bkg_rms_map is not None:
        # Rejection gate: it must still be able to judge each component, so
        # unknown-RMS pixels borrow the robust global value rather than being
        # exempted from the test.
        safe_rms = rms_or_robust(bkg_rms_map)
    else:
        safe_rms = np.ones_like(sci_data, dtype=np.float32)

    for region in regionprops(labeled, intensity_image=sci_data):
        coords = region.coords
        if region.area > max_component_area:
            continue
        local_values = sci_data[coords[:, 0], coords[:, 1]]
        local_rms = safe_rms[coords[:, 0], coords[:, 1]]
        snr = np.nanmax(local_values / np.maximum(local_rms, 1e-6))
        if not np.isfinite(snr) or snr < min_contrast_sigma:
            continue
        filtered[coords[:, 0], coords[:, 1]] = True
    return filtered


def _filter_faint_components(crmask_bool, sci_data, bkg_rms_map, faint_cfg):
    """Keep only elongated, high-contrast components from a low-threshold CR pass.

    Single/double-pixel hits at low sigclip are indistinguishable from noise
    and star shot noise, so they are dropped; multi-pixel worms (elongated,
    bright vs local rms) are almost never stars. Returns the gated subset.
    """
    from skimage.measure import label, regionprops

    labeled = label(np.ascontiguousarray(crmask_bool.astype(np.uint8)), connectivity=2)
    if labeled.max() == 0:
        return np.zeros_like(crmask_bool, dtype=bool)

    min_area = int(faint_cfg.get("min_component_area", 3))
    max_area = int(faint_cfg.get("max_component_area", 12))
    min_elongation = float(faint_cfg.get("min_elongation", 2.0))
    min_contrast_sigma = float(faint_cfg.get("min_contrast_sigma", 4.0))
    if bkg_rms_map is not None:
        # Rejection gate (see _post_filter_components): substitute, do not skip.
        med_rms = robust_rms(bkg_rms_map, default=1.0)
        safe_rms = rms_or_robust(bkg_rms_map, fallback=med_rms)
    else:
        safe_rms = np.ones_like(sci_data, dtype=np.float32)
    filtered = np.zeros_like(crmask_bool, dtype=bool)
    for region in regionprops(labeled, intensity_image=sci_data):
        if not (min_area <= region.area <= max_area):
            continue
        major_length, minor_length = _axis_lengths(region)
        elongation = major_length / max(minor_length, 1e-9)
        if elongation < min_elongation:
            continue
        coords = region.coords
        snr = np.nanmax(sci_data[coords[:, 0], coords[:, 1]] / np.maximum(safe_rms[coords[:, 0], coords[:, 1]], 1e-6))
        if not np.isfinite(snr) or snr < min_contrast_sigma:
            continue
        filtered[coords[:, 0], coords[:, 1]] = True
    return filtered


def _detect_residual_faint_components(
    sci_data,
    existing_mask,
    sky_map,
    bkg_rms_map,
    config,
    *,
    gain=1.0,
    read_noise=5.0,
    header=None,
    psf_aware=True,
):
    """Find faint multi-pixel CR candidates from a cheap residual map.

    The full L.A.Cosmic pass remains responsible for compact and single-pixel
    detections. This path is an opt-in enhancement for elongated residuals and
    deliberately applies the existing PSF and component gates before returning
    candidates.
    """
    if sky_map is None:
        return np.zeros(sci_data.shape, dtype=bool)

    residual = np.maximum(np.asarray(sci_data, dtype=np.float32) - np.asarray(sky_map, dtype=np.float32), 0.0)
    if bkg_rms_map is None:
        safe_rms = np.ones(sci_data.shape, dtype=np.float32)
    else:
        med_rms = robust_rms(bkg_rms_map, default=1.0)
        safe_rms = rms_or_robust(bkg_rms_map, fallback=med_rms)

    threshold_sig = float(config.get("threshold_sig", 4.0))
    candidate = residual >= threshold_sig * safe_rms
    candidate &= ~np.asarray(existing_mask, dtype=bool)
    candidate &= np.isfinite(residual)

    if psf_aware:
        candidate = _apply_psf_protection(
            candidate,
            sci_data,
            config,
            gain,
            read_noise,
            bkg_rms_map,
            sky_map=sky_map,
            header=header,
        )

    faint_cfg = {
        "min_component_area": int(config.get("min_component_area", 4)),
        "max_component_area": int(config.get("max_component_area", 12)),
        "min_elongation": float(config.get("min_elongation", 4.0)),
        "min_contrast_sigma": float(config.get("min_contrast_sigma", 4.0)),
    }
    return _filter_faint_components(candidate, sci_data, bkg_rms_map, faint_cfg)


def detect_cosmic_rays(
    sci_data,
    existing_mask,
    saturation_level,
    gain,
    read_noise,
    config,
    bkg_rms_map=None,
    sky_map=None,
    header=None,
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
        sky_map (ndarray, optional): Sky map used by the residual faint-CR path.
        header (dict, optional): FITS header forwarded to PSF protection.

    Returns:
        ndarray: Boolean mask of newly detected cosmic ray pixels
    """
    if detect_cosmics is None:
        print("  Astroscrappy is unavailable; returning empty cosmic-ray mask.")
        return np.zeros(sci_data.shape, dtype=bool)

    sigclip = _adjust_dynamic_sigclip(config, bkg_rms_map, default_sigclip=config.get("sigclip", 4.5))
    objlim = _adjust_dynamic_objlim(config, existing_mask, default_objlim=config.get("objlim", 5.0))

    faint_cfg = config.get("faint_cr", {})
    # One-pass mode: the loose thresholds that used to justify a second full
    # L.A.Cosmic run are applied to the primary pass, and the morphology gate
    # (elongated, high-contrast components only) does the discrimination the
    # second pass used to do. Costs one pass instead of two; whether the
    # completeness/false-positive trade is acceptable is decided by
    # ``benchmarks/cr_faint_curves.py``, not here.
    single_pass = bool(config.get("single_pass", False)) and faint_cfg.get("enable", False)
    if single_pass:
        sigclip = float(faint_cfg.get("sigclip", sigclip))
        objlim = objlim * float(faint_cfg.get("objlim_boost", 1.5))
        print("    Single-pass CR mode: loose thresholds with the morphology gate.")

    enhancement = str(faint_cfg.get("enhancement", "lacosmic")).lower()
    if enhancement not in ("lacosmic", "residual"):
        raise ValueError(f"Unknown faint-CR enhancement mode: {enhancement}")

    try:
        # Use astroscrappy (L.A.Cosmic) to detect cosmic rays. The dominant
        # cost knob is ``niter``: every iteration re-runs the median/Laplacian
        # filters over the whole CCD, and the later iterations only add the
        # faintest marginal detections. The remaining knobs are forwarded so
        # callers can tune the speed/quality tradeoff without code changes.
        crmask_bool, _ = detect_cosmics(
            sci_data,
            inmask=existing_mask,
            satlevel=saturation_level,
            gain=gain,
            readnoise=read_noise,
            sigclip=sigclip,
            objlim=objlim,
            niter=int(config.get("niter", 4)),
            sepmed=bool(config.get("sepmed", True)),
            cleantype=config.get("cleantype", "meanmask"),
            fsmode=config.get("fsmode", "median"),
            psffwhm=float(config.get("psffwhm", 2.5)),
            psfsize=int(config.get("psfsize", 7)),
            verbose=False,
        )

        if single_pass:
            # The morphology gate subsumes both the PSF protection (a star is
            # round, so it fails min_elongation) and the component post-filter,
            # which is exactly why one pass can replace two.
            crmask_bool = _filter_faint_components(
                np.ascontiguousarray(crmask_bool.astype(bool)), sci_data, bkg_rms_map, faint_cfg
            )
        else:
            crmask_bool = _apply_psf_protection(
                crmask_bool, sci_data, config, gain, read_noise, bkg_rms_map, sky_map=sky_map, header=header
            )
            crmask_bool = _post_filter_components(crmask_bool.astype(bool), sci_data, bkg_rms_map, config)

        crmask_bool = _apply_morphological_dilation(crmask_bool, config)

        if faint_cfg.get("enable", False) and not single_pass:
            if enhancement == "residual":
                print("    Running residual faint-CR enhancement...")
                faint_kept = _detect_residual_faint_components(
                    sci_data,
                    existing_mask | crmask_bool,
                    sky_map,
                    bkg_rms_map,
                    faint_cfg.get("residual", {}),
                    gain=gain,
                    read_noise=read_noise,
                    header=header,
                    psf_aware=bool(config.get("psf_aware", True)),
                )
            elif enhancement == "lacosmic":
                print("    Running faint-CR pass (raised sigclip + morphology gate)...")
                faint_raw, _ = detect_cosmics(
                    sci_data,
                    inmask=(existing_mask | crmask_bool),
                    satlevel=saturation_level,
                    gain=gain,
                    readnoise=read_noise,
                    sigclip=float(faint_cfg.get("sigclip", 5.0)),
                    objlim=objlim * float(faint_cfg.get("objlim_boost", 1.5)),
                    niter=int(faint_cfg.get("niter", 1)),
                    sepmed=bool(config.get("sepmed", True)),
                    cleantype=config.get("cleantype", "meanmask"),
                    fsmode=config.get("fsmode", "median"),
                    psffwhm=float(config.get("psffwhm", 2.5)),
                    psfsize=int(config.get("psfsize", 7)),
                    verbose=False,
                )
                faint_kept = _filter_faint_components(
                    np.ascontiguousarray(faint_raw.astype(bool)), sci_data, bkg_rms_map, faint_cfg
                )
            else:
                raise ValueError(f"Unknown faint-CR enhancement mode: {enhancement}")
            n_faint = int(np.count_nonzero(faint_kept))
            if n_faint:
                print(f"    Faint-CR pass kept {n_faint} pixels.")
            crmask_bool = crmask_bool | faint_kept
        # Only return newly detected pixels (not already in existing_mask)
        cr_add_mask = crmask_bool & (~existing_mask)

        num_new_pixels = np.count_nonzero(cr_add_mask)
        if num_new_pixels > 0:
            print(f"  Detected {num_new_pixels} new cosmic ray pixels.")

        return cr_add_mask

    except Exception as e:
        print(f"  Astroscrappy failed: {e}")
        return np.zeros(sci_data.shape, dtype=bool)
