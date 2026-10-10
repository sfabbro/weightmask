import os
import re

import numpy as np

from . import config as _config

clean_config_dict = _config.clean_config_dict


def paths_alias(first, second):
    """Whether filenames refer to the same destination, including links."""
    if os.path.realpath(first) == os.path.realpath(second):
        return True
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def rms_valid_mask(rms):
    """Pixels where a background RMS measurement exists.

    ``weightmask.background.estimate_background`` marks pixels whose local RMS
    it could not measure (after SEP failed, or a non-positive/``inf`` value came
    back) with ``+inf``. That sentinel is part of the contract, not an accident,
    and it can cover a large fraction of a chip: on a real MegaPrime HDU it is
    7% of all pixels and 150 of 2112 columns entirely.

    Callers must choose, deliberately, between the two legitimate readings:

    * a *detection threshold* cannot be asserted without an RMS, so those
      pixels must be inert -- see ``_refine_trail_mask``;
    * a *rejection / quality gate* still has to judge what it can, so it
      substitutes a robust value -- see ``rms_or_robust`` and
      ``cosmics._post_filter_components``.

    Returns:
        ndarray[bool]: True where the RMS is finite and positive.
    """
    if rms is None:
        return None
    rms = np.asarray(rms)
    return np.isfinite(rms) & (rms > 0)


def robust_rms(rms, default=1.0):
    """Median of the *valid* entries of an RMS map (see ``rms_valid_mask``).

    Uses an explicit validity mask rather than ``np.nanmedian``: the sentinel is
    ``inf``, which ``nanmedian`` does not ignore, so a map that is more than half
    sentinel would silently return ``inf`` and turn every substitution into a
    no-detection. That flip is data-dependent and was a real latent bug in the
    four call sites this helper replaces.
    """
    valid = rms_valid_mask(rms)
    if valid is None or not np.any(valid):
        return float(default)
    median = float(np.median(np.asarray(rms)[valid]))
    return median if np.isfinite(median) and median > 0 else float(default)


def rms_or_robust(rms, fallback=None):
    """Replace sentinel/unknown RMS pixels with a robust value.

    Only for call sites that must produce a number for every pixel (rejection
    gates, contrast tests). Detection thresholds should instead treat unknown
    pixels as inert, so that an unmeasurable region can never manufacture a
    detection.

    Args:
        rms (ndarray): RMS map, possibly containing the ``inf`` sentinel.
        fallback (float, optional): Value to substitute. Defaults to
            ``robust_rms(rms)``.

    Returns:
        ndarray: Same shape as ``rms``, free of sentinel values.
    """
    if rms is None:
        return None
    rms = np.asarray(rms)
    valid = rms_valid_mask(rms)
    if fallback is None:
        fallback = robust_rms(rms)
    return np.where(valid, rms, float(fallback))


def extract_hdu_spec(filepath):
    """
    Extract HDU specifier from CFITSIO-style filename (e.g., 'file.fits[1]')

    Args:
        filepath (str): Path with potential HDU specifier

    Returns:
        tuple: (clean_path, hdu_index) where hdu_index is None if not specified
    """
    if filepath is None:
        return None, None

    match = re.match(r"^(.*?)(?:\[(\d+)\])?$", filepath)
    if match:
        path, hdu_spec = match.groups()
        if hdu_spec is not None:
            return path, int(hdu_spec)
    return filepath, None


def create_binary_mask(mask_data, bit_flag):
    """
    Create a binary mask (0/1) from a bitmask for a specific flag.

    Args:
        mask_data (ndarray): Bitmask array
        bit_flag (int): Bit flag to extract

    Returns:
        ndarray: Binary mask (0=not set, 1=set)
    """
    # Performance optimization: Use direct boolean casting instead of np.where.
    # This avoids allocating unnecessary intermediate arrays and executes
    # significantly faster.
    return ((mask_data & bit_flag) > 0).astype(np.uint8)
