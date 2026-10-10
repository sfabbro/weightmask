#!/usr/bin/env python3
"""Capture the exact inputs ``process_image`` hands to ``detect_streaks``.

Benchmark harnesses need to score a streak detector, and the obvious way to do
that -- take the science frame, run one background pass with an empty mask, call
``detect_streaks`` -- measures a code path production never runs. Two things
differ, and both change the answer:

``existing_mask``
    ``detect_streaks`` uses exclusions in its Hough and contour searches, and
    searches residual bright pixels with sparse RANSAC. With no exclusions,
    the binned prescreen can accept chip-fixed clutter as trails.
    Production passes
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
import copy
import io

import numpy as np


def _lookup_header(header, keys, default):
    values = keys if isinstance(keys, (list, tuple)) else [keys]
    for key in values:
        if not isinstance(key, str) or not key:
            continue
        value = None
        getter = getattr(header, "get", None)
        if callable(getter):
            value = getter(key, None)
        if value is None:
            try:
                value = header[key]
            except (KeyError, TypeError, AttributeError):
                value = None
        if value is not None:
            return value
    return default


def production_gain_read_noise(header, config):
    """Resolve gain/read noise with the same precedence as ``process_image``."""
    variance = config.get("variance", {}) if isinstance(config, dict) else {}
    gain_default = variance.get("default_gain", 1.0)
    read_default = variance.get("default_rdnoise", 0.0)
    gain_raw = _lookup_header(header, variance.get("gain_keyword", "GAIN"), gain_default)
    read_raw = _lookup_header(header, variance.get("rdnoise_keyword", "RDNOISE"), read_default)
    def coerce(raw, fallback, minimum, label):
        try:
            value = float(raw)
            if not np.isfinite(value) or value < minimum:
                raise ValueError
        except (TypeError, ValueError):
            value = float(fallback)
        if not np.isfinite(value) or value < minimum:
            raise ValueError(f"invalid {label} in production variance configuration")
        return value

    gain = coerce(gain_raw, gain_default, 1e-12, "gain")
    read_noise = coerce(read_raw, read_default, 0.0, "read noise")
    return gain, read_noise


def _snapshot(value):
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    return copy.deepcopy(value)


class _CaptureStop(Exception):
    pass


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


def capture_detector_calls(sci, header, config, detector_prior=None, flat=None, stop_after=None):
    """Run production and return immutable snapshots of detector call boundaries."""
    import weightmask.process as proc

    calls = {name: [] for name in ("saturation", "cosmics", "objects", "streaks")}
    calls["order"] = []
    originals = {
        "saturation": proc.detect_saturated_pixels,
        "cosmics": proc.detect_cosmic_rays,
        "objects": proc.detect_objects,
        "streaks": proc.detect_streaks,
    }

    def record(name, function):
        def wrapped(*args, **kwargs):
            call = {"args": _snapshot(args), "kwargs": _snapshot(kwargs)}
            calls[name].append(call)
            calls["order"].append(name)
            result = function(*args, **kwargs)
            call["result"] = _snapshot(result)
            if name == stop_after:
                raise _CaptureStop
            return result

        return wrapped

    proc.detect_saturated_pixels = record("saturation", originals["saturation"])
    proc.detect_cosmic_rays = record("cosmics", originals["cosmics"])
    proc.detect_objects = record("objects", originals["objects"])
    proc.detect_streaks = record("streaks", originals["streaks"])
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            proc.process_image(sci, header, flat, config, detector_prior=detector_prior)
    except _CaptureStop:
        pass
    finally:
        proc.detect_saturated_pixels = originals["saturation"]
        proc.detect_cosmic_rays = originals["cosmics"]
        proc.detect_objects = originals["objects"]
        proc.detect_streaks = originals["streaks"]
    if not calls["order"]:
        raise RuntimeError("process_image made no detector calls")
    return calls


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
