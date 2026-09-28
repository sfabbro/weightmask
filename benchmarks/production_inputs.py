#!/usr/bin/env python3
"""Capture the exact inputs ``process_image`` hands to ``detect_streaks``.

Benchmark harnesses need to score a streak detector, and the obvious way to do
that -- take the science frame, run one background pass with an empty mask, call
``detect_streaks`` -- measures a code path production never runs. Two things
differ, and both change the answer:

``existing_mask``
    ``detect_streaks`` gates its full-resolution sweep, its Radon rescue and its
    sparse RANSAC on the exclusion mask. With no exclusions the cheap binned
    prescreen accepts chip-fixed clutter, which sets ``prescreen_confirmed`` and
    silently skips all three, so a detector scored that way never runs the stages
    that find faint trails. Production always passes
    ``interim_mask_bool | final_obj_mask | detector_prior``.

the background
    Production iterates: a preliminary pass, then cosmic-ray and bleed masking,
    then a final pass over the accumulated mask. A single pass with an empty mask
    leaves the chip-fixed columns and rows in ``data_sub`` as bright linear
    features, so the streak detector finds and masks them. Those are not streak
    false positives; they are the background stage's job, done downstream.

Rather than re-derive either input here -- where it would silently drift from
``process_image`` -- this drives the real pipeline with ``detect_streaks``
replaced by a capture.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np


def capture_detector_inputs(sci, header, config, detector_prior=None, flat=None):
    """Return ``(data_sub, bkg_rms_map, existing_mask)`` as production builds them.

    Parameters mirror ``process_image``. ``flat`` defaults to ``None`` (flat = 1)
    so harnesses that have always scored the un-flat-fielded frame keep doing so;
    only the background iteration and the exclusion mask change.

    Raises ``RuntimeError`` if the pipeline returns before the streak stage,
    which is the loud failure mode: a silently un-captured input would put the
    harness back on the non-production path without saying so.
    """
    import weightmask.process as proc

    captured = {}

    def _capture(data_sub, bkg_rms_map, existing_mask, _streak_cfg):
        shape = sci.shape
        captured["data_sub"] = np.array(data_sub, dtype=np.float32, copy=True)
        captured["rms"] = None if bkg_rms_map is None else np.array(bkg_rms_map, dtype=np.float32, copy=True)
        captured["existing"] = (
            np.zeros(shape, dtype=bool) if existing_mask is None else existing_mask.astype(bool, copy=True)
        )
        # Hand back an empty mask so the pipeline's own bookkeeping stays clean.
        return np.zeros(shape, dtype=bool)

    original = proc.detect_streaks
    proc.detect_streaks = _capture
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            proc.process_image(sci, header, flat, config, detector_prior=detector_prior)
    finally:
        proc.detect_streaks = original
    if "data_sub" not in captured:
        raise RuntimeError("process_image never reached the streak stage (is streak_masking.enable false?)")
    return captured["data_sub"], captured["rms"], captured["existing"]


def streak_config(config, enable=True):
    """Force ``streak_masking.enable`` on ``config`` in place; return that sub-dict.

    ``process_image`` only reaches the streak stage when this is enabled, so the
    capture needs it on. Mutates and returns the *sub-dict* for the caller's own
    detector calls -- ``config`` itself is still what gets passed to
    ``capture_detector_inputs``.
    """
    streak_cfg = dict(config.get("streak_masking", {}))
    streak_cfg["enable"] = bool(enable)
    config["streak_masking"] = dict(streak_cfg)
    return streak_cfg
