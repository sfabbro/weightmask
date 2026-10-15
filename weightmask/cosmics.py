import numpy as np
from astroscrappy import detect_cosmics
from scipy.ndimage import convolve
from skimage.measure import label, regionprops
from skimage.morphology import dilation, disk

from .config import section_config
from .errors import StageFailure
from .utils import rms_or_robust, rms_valid_mask


def _get_psf_peakiness(fwhm):
    """Central-pixel fraction of the 3x3 flux of a 2D Gaussian PSF."""
    sigma = max(float(fwhm), 0.1) / 2.355
    x, y = np.mgrid[-1:2, -1:2]
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return (kernel / kernel.sum())[1, 1]


def _adjust_dynamic_objlim(enabled, existing_mask, objlim):
    """Increase object protection in crowded scenes with many masked pixels."""
    objlim = float(objlim)
    if not enabled or existing_mask is None:
        return objlim
    coverage = float(np.mean(existing_mask))
    if coverage <= 0:
        return objlim
    objlim *= 1.0 + min(coverage * 4.0, 1.5)
    print(f"    Dynamically adjusted objlim to {objlim:.2f} based on pre-mask coverage {coverage:.2%}")
    return objlim


def _fwhm_from_header(header, default):
    """Pixels from SEEING (arcsec) and a pixel scale, else ``default``."""
    if header is None:
        return float(default)
    seeing = header.get("SEEING")
    pix = next((header.get(k) for k in ("PIXSCAL1", "PIXSCALE", "PIXSIZE") if header.get(k) is not None), None)
    try:
        seeing_f, pix_f = float(seeing), float(pix)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(seeing_f) or not np.isfinite(pix_f) or pix_f <= 0 or seeing_f <= 0:
        return float(default)
    return seeing_f / pix_f


def _apply_psf_protection(crmask, sci_data, fwhm_guess, gain, read_noise, bkg_rms_map, sky_map=None, header=None):
    """Unflag pixels whose 3x3 peakiness looks like a star core rather than a CR."""
    psf_fwhm = _fwhm_from_header(header, fwhm_guess)
    print(f"    Applying PSF-aware protection (FWHM: {psf_fwhm:.1f} pix)")

    if sky_map is not None and np.shape(sky_map) == np.shape(sci_data):
        sci_sub = np.maximum(np.asarray(sci_data, dtype=np.float32) - np.asarray(sky_map, dtype=np.float32), 0.0)
    else:
        sampled = sci_data[::10, ::10]
        finite_sampled = sampled[np.isfinite(sampled)]
        sky_est = np.median(finite_sampled) if finite_sampled.size > 0 else 0.0
        sci_sub = np.maximum(sci_data - sky_est, 0.0)

    local_flux_sum = convolve(sci_sub, np.ones((3, 3), dtype=np.float32), mode="constant", cval=0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        peakiness = sci_sub / local_flux_sum
    cr_thresh = _get_psf_peakiness(psf_fwhm) * 1.1

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
        star_protection_mask = (peakiness < cr_thresh) & (sci_sub > 5.0 * read_noise / max(float(gain), 1e-6))

    crmask = crmask.astype(bool)
    protected_count = np.count_nonzero(crmask & star_protection_mask)
    if protected_count > 0:
        print(f"    PSF protection: Saved {protected_count} pixels (likely star cores) from CR flagging.")
        crmask &= ~star_protection_mask
    return crmask


def _axis_lengths(region):
    """Major/minor axis lengths of a region (skimage >= 0.26 spelling)."""
    return float(region.axis_major_length), float(region.axis_minor_length)


def _filter_components(
    crmask, sci_data, bkg_rms_map, sky_map, *, max_area, min_contrast, min_area=1, min_elongation=0.0
):
    """Keep connected components within the area range, elongation and peak-SNR gates."""
    labeled = label(np.ascontiguousarray(crmask, dtype=np.uint8), connectivity=2)
    filtered = np.zeros(crmask.shape, dtype=bool)
    if labeled.max() == 0:
        return filtered
    # Rejection gate: it must still be able to judge each component, so
    # unknown-RMS pixels borrow the robust global value rather than being
    # exempted from the test.
    safe_rms = rms_or_robust(bkg_rms_map) if bkg_rms_map is not None else np.ones_like(sci_data, dtype=np.float32)
    if sky_map is None:
        finite = sci_data[np.isfinite(sci_data)]
        sky_map = np.median(finite) if finite.size else 0.0
    contrast = sci_data - sky_map
    for region in regionprops(labeled):
        if not (min_area <= region.area <= max_area):
            continue
        if min_elongation > 0:
            major_length, minor_length = _axis_lengths(region)
            if major_length / max(minor_length, 1e-9) < min_elongation:
                continue
        rows, cols = region.coords[:, 0], region.coords[:, 1]
        snr = np.nanmax(contrast[rows, cols] / np.maximum(safe_rms[rows, cols], 1e-6))
        if np.isfinite(snr) and snr >= min_contrast:
            filtered[rows, cols] = True
    return filtered


def _filter_faint_components(crmask, sci_data, bkg_rms_map, faint_cfg, sky_map=None):
    """Keep only elongated, high-contrast components from a low-threshold CR pass.

    Single/double-pixel hits at low sigclip are indistinguishable from noise
    and star shot noise, so they are dropped; multi-pixel worms (elongated,
    bright vs local rms) are almost never stars.
    """
    return _filter_components(
        crmask,
        sci_data,
        bkg_rms_map,
        sky_map,
        min_area=faint_cfg["min_component_area"],
        max_area=faint_cfg["max_component_area"],
        min_elongation=faint_cfg["min_elongation"],
        min_contrast=faint_cfg["min_contrast_sigma"],
    )


def _detect_residual_faint_components(sci_data, existing_mask, sky_map, bkg_rms_map, cfg, *, gain, read_noise, header):
    """Find faint multi-pixel CR candidates from a cheap residual map.

    The full L.A.Cosmic pass remains responsible for compact and single-pixel
    detections. This opt-in path targets elongated residuals and applies the
    PSF and component gates before returning candidates.
    """
    if sky_map is None:
        return np.zeros(sci_data.shape, dtype=bool)
    residual_cfg = cfg["faint_cr"]["residual"]
    residual = np.maximum(np.asarray(sci_data, dtype=np.float32) - np.asarray(sky_map, dtype=np.float32), 0.0)
    if bkg_rms_map is None:
        safe_rms = np.ones(sci_data.shape, dtype=np.float32)
    else:
        # Detection requires a measured RMS; unknown pixels cannot seed a CR.
        safe_rms = rms_or_robust(bkg_rms_map, fallback=np.inf)
    candidate = residual >= residual_cfg["threshold_sig"] * safe_rms
    candidate &= ~np.asarray(existing_mask, dtype=bool)
    candidate &= np.isfinite(residual)
    if cfg["psf_aware"]:
        candidate = _apply_psf_protection(
            candidate, sci_data, cfg["psf_fwhm_guess"], gain, read_noise, bkg_rms_map, sky_map=sky_map, header=header
        )
    return _filter_faint_components(candidate, sci_data, bkg_rms_map, residual_cfg, sky_map=sky_map)


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
        config (dict): ``cosmic_ray`` section overrides (merged over the packaged defaults)
        bkg_rms_map (ndarray, optional): Background RMS map for post-detection gates.
        sky_map (ndarray, optional): Sky map used by the residual faint-CR path.
        header (dict, optional): FITS header forwarded to PSF protection.

    Returns:
        ndarray: Boolean mask of newly detected cosmic ray pixels
    """
    cfg = section_config("cosmic_ray", config)
    faint_cfg = cfg["faint_cr"]
    sigclip = cfg["sigclip"]
    objlim = _adjust_dynamic_objlim(cfg["dynamic_objlim"], existing_mask, cfg["objlim"])
    primary_niter = cfg["niter"]
    faint_active = faint_cfg["enable"] and faint_cfg["niter"] > 0
    # One-pass mode: the loose thresholds that used to justify a second full
    # L.A.Cosmic run are applied to the primary pass, and the morphology gate
    # (elongated, high-contrast components only) does the discrimination the
    # second pass used to do.
    single_pass = primary_niter > 0 and cfg["single_pass"] and faint_active
    if single_pass:
        sigclip = faint_cfg["sigclip"]
        objlim = objlim * faint_cfg["objlim_boost"]
        print("    Single-pass CR mode: loose thresholds with the morphology gate.")

    def lacosmic(inmask, sigclip, objlim, niter):
        crmask, _ = detect_cosmics(
            sci_data,
            inmask=inmask,
            satlevel=saturation_level,
            gain=gain,
            readnoise=read_noise,
            sigclip=sigclip,
            objlim=objlim,
            niter=niter,
            sepmed=cfg["sepmed"],
            cleantype=cfg["cleantype"],
            fsmode=cfg["fsmode"],
            psffwhm=cfg["psffwhm"],
            psfsize=cfg["psfsize"],
            verbose=False,
        )
        return np.ascontiguousarray(crmask, dtype=bool)

    crmask = np.zeros(sci_data.shape, dtype=bool)
    if primary_niter > 0:
        try:
            crmask = lacosmic(existing_mask, sigclip, objlim, primary_niter)
        except Exception as exc:
            raise StageFailure("cosmic_ray.primary", exc) from exc
        try:
            if single_pass:
                crmask = _filter_faint_components(crmask, sci_data, bkg_rms_map, faint_cfg, sky_map=sky_map)
            else:
                if cfg["psf_aware"]:
                    crmask = _apply_psf_protection(
                        crmask, sci_data, cfg["psf_fwhm_guess"], gain, read_noise, bkg_rms_map, sky_map, header
                    )
                crmask = _filter_components(
                    crmask,
                    sci_data,
                    bkg_rms_map,
                    sky_map,
                    max_area=cfg["max_component_area"],
                    min_contrast=cfg["min_component_contrast_sigma"],
                )
            if cfg["dilate_cr"]:
                print("    Applying morphological dilation to cosmic ray mask...")
                crmask = dilation(crmask, footprint=disk(cfg["dilation_radius"]))
        except Exception as exc:
            raise StageFailure("cosmic_ray.postprocess", exc) from exc

    if faint_active and not single_pass:
        try:
            if faint_cfg["enhancement"] == "residual":
                print("    Running residual faint-CR enhancement...")
                faint_kept = _detect_residual_faint_components(
                    sci_data,
                    existing_mask | crmask,
                    sky_map,
                    bkg_rms_map,
                    cfg,
                    gain=gain,
                    read_noise=read_noise,
                    header=header,
                )
            else:
                print("    Running faint-CR pass (configured sigclip + morphology gate)...")
                faint_raw = lacosmic(
                    existing_mask | crmask,
                    faint_cfg["sigclip"],
                    objlim * faint_cfg["objlim_boost"],
                    faint_cfg["niter"],
                )
                faint_kept = _filter_faint_components(faint_raw, sci_data, bkg_rms_map, faint_cfg, sky_map=sky_map)
        except Exception as exc:
            raise StageFailure("cosmic_ray.faint", exc) from exc
        n_faint = int(np.count_nonzero(faint_kept))
        if n_faint:
            print(f"    Faint-CR pass kept {n_faint} pixels.")
        crmask = crmask | faint_kept

    cr_add_mask = crmask & ~existing_mask
    num_new_pixels = np.count_nonzero(cr_add_mask)
    if num_new_pixels > 0:
        print(f"  Detected {num_new_pixels} new cosmic ray pixels.")
    return cr_add_mask
