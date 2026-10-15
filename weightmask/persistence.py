"""Detector-fixed rows and columns from other exposures of the same CCD (``--persistence``)."""

import warnings

import numpy as np


def _profile_outliers(profile, sigma):
    """True where a 1-D profile exceeds ``sigma`` robust standard deviations."""
    profile = np.asarray(profile, dtype=np.float64)
    finite = np.isfinite(profile)
    outliers = np.zeros(profile.shape, dtype=bool)
    if not np.any(finite):
        return outliers
    values = profile[finite]
    med = float(np.median(values))
    mad = 1.4826 * float(np.median(np.abs(values - med)))
    if not np.isfinite(mad) or mad <= 1e-12:
        mad = 1e-12
    outliers[finite] = values > med + float(sigma) * mad
    return outliers


def _finite_axis_profiles(frame):
    if np.all(np.isfinite(frame)):
        return np.median(frame, axis=0), np.median(frame, axis=1)
    values = frame.copy()
    values[~np.isfinite(values)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(values, axis=0), np.nanmedian(values, axis=1)


class PersistenceProfileAccumulator:
    def __init__(self, shape, min_other=2, sigma=3.0):
        if len(shape) != 2:
            raise ValueError("persistent_axis_mask frames must be 2-D")
        self.shape = tuple(shape)
        self.min_other = int(min_other)
        self.sigma = float(sigma)
        self.frame_count = 0
        self.col_hits = np.zeros(self.shape[1], dtype=np.intp)
        self.row_hits = np.zeros(self.shape[0], dtype=np.intp)

    def add(self, frame):
        frame = np.asarray(frame, dtype=np.float64)
        if frame.shape != self.shape:
            raise ValueError("persistent_axis_mask frames must share a shape")
        col_profile, row_profile = _finite_axis_profiles(frame)
        self.col_hits += _profile_outliers(col_profile, self.sigma)
        self.row_hits += _profile_outliers(row_profile, self.sigma)
        self.frame_count += 1

    def mask(self):
        mask = np.zeros(self.shape, dtype=bool)
        hot_cols = np.nonzero(self.col_hits >= self.min_other)[0]
        hot_rows = np.nonzero(self.row_hits >= self.min_other)[0]
        if hot_cols.size:
            mask[:, hot_cols] = True
        if hot_rows.size:
            mask[hot_rows, :] = True
        return mask


def persistent_axis_mask(frames, min_other=2, sigma=3.0):
    """Columns and rows that are bright outliers in at least ``min_other`` frames.

    ``frames`` are other exposures of one CCD (not the frame being masked). An
    axis is detector-fixed when its median is a robust outlier in that many
    frames — the same count the trail curator requires of other epochs. A trail
    that exists in only one exposure does not qualify.
    """
    accumulator = None
    for frame in frames:
        array = np.asarray(frame, dtype=np.float64)
        if accumulator is None:
            accumulator = PersistenceProfileAccumulator(array.shape, min_other=min_other, sigma=sigma)
        accumulator.add(array)
        del array, frame
    if accumulator is None:
        raise ValueError("persistent_axis_mask needs at least one frame")
    return accumulator.mask()
