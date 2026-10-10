"""Declarative validation and packaged access for the weightmask configuration."""

import hashlib
import os
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import resources
from numbers import Integral, Real
from pathlib import Path

import numpy as np

_CONFIG_NAME = "weightmask.yml"

__all__ = [
    "clean_config_dict",
    "copy_default_config",
    "default_config_bytes",
    "default_config_path",
    "default_config_resource",
    "default_config_sha256",
    "default_config_text",
]


def default_config_resource():
    return resources.files(__package__).joinpath(_CONFIG_NAME)


@contextmanager
def default_config_path():
    with resources.as_file(default_config_resource()) as path:
        yield path


def default_config_bytes():
    return default_config_resource().read_bytes()


def default_config_text():
    return default_config_bytes().decode("utf-8")


def default_config_sha256():
    return hashlib.sha256(default_config_bytes()).hexdigest()


def _parse_config_value(value):
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return value
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    lowered = stripped.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False

    digits = stripped[1:] if stripped[:1] in ("+", "-") else stripped
    if stripped and digits.isdecimal():
        return int(stripped)

    if "e" in lowered or "." in lowered:
        try:
            return float(stripped)
        except ValueError:
            pass
    return value


def clean_config_dict(config):
    if not config:
        return {}

    clean_dict = {}
    for key, value in config.items():
        if isinstance(value, dict):
            clean_dict[key] = clean_config_dict(value)
        elif isinstance(value, list):
            clean_dict[key] = [
                clean_config_dict(item) if isinstance(item, dict) else _parse_config_value(item) for item in value
            ]
        else:
            clean_dict[key] = _parse_config_value(value)
    return clean_dict


def _destination_path(destination):
    destination = Path(destination)
    try:
        mode = destination.lstat().st_mode
    except FileNotFoundError:
        mode = None
    if mode is not None and stat.S_ISLNK(mode):
        raise FileExistsError(destination)
    if mode is not None and stat.S_ISDIR(mode):
        destination /= _CONFIG_NAME
        try:
            child_mode = destination.lstat().st_mode
            if stat.S_ISLNK(child_mode) or not stat.S_ISREG(child_mode):
                raise FileExistsError(destination)
        except FileNotFoundError:
            pass
    elif mode is not None and not stat.S_ISREG(mode):
        raise FileExistsError(destination)
    return destination


def copy_default_config(destination, *, overwrite=False):
    destination = _destination_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = default_config_bytes()
    if not overwrite:
        fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        identity = os.fstat(fd)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            try:
                current = destination.stat(follow_symlinks=False)
                if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
                    os.unlink(destination)
            except OSError:
                pass
            raise
        return destination
    try:
        existing_mode = stat.S_IMODE(destination.lstat().st_mode)
    except FileNotFoundError:
        existing_mode = None
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        if existing_mode is not None:
            os.fchmod(fd, existing_mode)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            current_mode = destination.lstat().st_mode
            if not stat.S_ISREG(current_mode):
                raise FileExistsError(destination)
        except FileNotFoundError:
            pass
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return destination


@dataclass(frozen=True)
class _Rule:
    kind: str
    minimum: float | None = None
    maximum: float | None = None
    exclusive_minimum: bool = False
    choices: frozenset | None = None
    nullable: bool = False
    odd: bool = False
    casefold: bool = False


def _number(minimum=None, maximum=None, *, exclusive_minimum=False, nullable=False):
    return _Rule("number", minimum, maximum, exclusive_minimum, nullable=nullable)


def _integer(minimum=None, maximum=None, *, exclusive_minimum=False, odd=False):
    return _Rule("integer", minimum, maximum, exclusive_minimum, odd=odd)


def _enum(*choices, casefold=False):
    return _Rule("string", choices=frozenset(choices), casefold=casefold)


_BOOL = _Rule("boolean")
_STRING = _Rule("string")
_NULLABLE_STRING = _Rule("string", nullable=True)
_KEYWORDS = _Rule("keywords")
_POSITIVE = _number(0, exclusive_minimum=True)
_NONNEGATIVE = _number(0)
_FRACTION = _number(0, 1)
_POSITIVE_INTEGER = _integer(0, exclusive_minimum=True)
_NONNEGATIVE_INTEGER = _integer(0)
_PERCENTILE = _number(0, 100, exclusive_minimum=True)


CONFIG_SCHEMA = {
    "flat_masking": {
        "local_filter_size": _POSITIVE_INTEGER,
        "local_low_thresh": _POSITIVE,
        "local_high_thresh": _POSITIVE,
        "col_enable": _BOOL,
        "col_deriv_sigma": _POSITIVE,
        "col_dead_thresh": _FRACTION,
        "dead_ccd_enable": _BOOL,
        "dead_ccd_mad_sigma": _POSITIVE,
        "dead_ccd_min_rel_dev": _NONNEGATIVE,
        "dead_ccd_min_hdus": _POSITIVE_INTEGER,
        "dead_ccd_badpix_fraction": _FRACTION,
        "bad_mask_cache": _BOOL,
        "bad_mask_cache_dir": _NULLABLE_STRING,
    },
    "dark_masking": {"hot_sigma": _POSITIVE},
    "saturation": {
        "keyword": _KEYWORDS,
        "effective_full_scale": _number(0, exclusive_minimum=True, nullable=True),
        "histogram_params": {
            "hist_min_adu": _number(0, nullable=True),
            "hist_max_adu": _number(0, exclusive_minimum=True, nullable=True),
            "guard_fraction": _number(0, 1, exclusive_minimum=True),
            "max_upper_factor": _number(1),
            "min_tail_pixels": _POSITIVE_INTEGER,
        },
        "bleed_adaptive_cap": _BOOL,
        "bleed_cap_core_factor": _POSITIVE,
        "bleed_cap_min": _NONNEGATIVE_INTEGER,
        "bleed_cap_max": _NONNEGATIVE_INTEGER,
        "mask_bleed_trails": _BOOL,
        "bleed_thresh_sigma": _POSITIVE,
        "bleed_grow_vertical": _NONNEGATIVE_INTEGER,
        "bleed_grow_horizontal": _NONNEGATIVE_INTEGER,
        "fallback_level": _POSITIVE,
    },
    "sep_background": {
        "method": _enum("sep", "median_filter", "robust_median_fallback"),
        "box_size": _POSITIVE_INTEGER,
        "auto_box_scaling": _BOOL,
        "filter_size": _POSITIVE_INTEGER,
        "iterations": _NONNEGATIVE_INTEGER,
        "mask_threshold": _FRACTION,
        "max_box_size": _POSITIVE_INTEGER,
        "median_kernel_size": _integer(0, exclusive_minimum=True, odd=True),
        "smooth_surface_fallback": _BOOL,
        "dip_repair_enable": _BOOL,
        "dip_repair_sigma": _POSITIVE,
        "dip_repair_max_fraction": _FRACTION,
        "edge_artifact_thresh": _NONNEGATIVE,
    },
    "cosmic_ray": {
        "sigclip": _POSITIVE,
        "objlim": _POSITIVE,
        "niter": _NONNEGATIVE_INTEGER,
        "dynamic_objlim": _BOOL,
        "psf_aware": _BOOL,
        "psf_fwhm_guess": _POSITIVE,
        "dilate_cr": _BOOL,
        "dilation_radius": _NONNEGATIVE_INTEGER,
        "max_component_area": _POSITIVE_INTEGER,
        "min_component_contrast_sigma": _NONNEGATIVE,
        "single_pass": _BOOL,
        "sepmed": _BOOL,
        "cleantype": _enum("median", "medmask", "meanmask", "idw"),
        "fsmode": _enum("median", "convolve"),
        "psffwhm": _POSITIVE,
        "psfsize": _integer(0, exclusive_minimum=True, odd=True),
        "faint_cr": {
            "enable": _BOOL,
            "enhancement": _enum("lacosmic", "residual"),
            "sigclip": _POSITIVE,
            "objlim_boost": _POSITIVE,
            "niter": _NONNEGATIVE_INTEGER,
            "min_component_area": _POSITIVE_INTEGER,
            "max_component_area": _POSITIVE_INTEGER,
            "min_elongation": _POSITIVE,
            "min_contrast_sigma": _NONNEGATIVE,
            "residual": {
                "threshold_sig": _POSITIVE,
                "min_component_area": _POSITIVE_INTEGER,
                "max_component_area": _POSITIVE_INTEGER,
                "min_elongation": _POSITIVE,
                "min_contrast_sigma": _NONNEGATIVE,
            },
        },
    },
    "sep_objects": {
        "extract_thresh": _POSITIVE,
        "min_area": _POSITIVE_INTEGER,
        "deblend_nthresh": _POSITIVE_INTEGER,
        "deblend_cont": _FRACTION,
        "clean": _BOOL,
        "clean_param": _NONNEGATIVE,
        "ellipse_k": _POSITIVE,
        "seed_thresh_factor": _POSITIVE,
        "dynamic_halo_scaling": _BOOL,
        "halo_brightness_factor": _NONNEGATIVE,
        "halo_flux_reference_percentile": _PERCENTILE,
        "max_halo_multiplier": _number(1),
        "mask_dilation_radius": _NONNEGATIVE_INTEGER,
        "max_elongation": _POSITIVE,
        "handoff_elongated_to_streak": _BOOL,
        "spike_enable": _BOOL,
        "spike_flux_thresh": _NONNEGATIVE,
        "spike_length_base": _POSITIVE_INTEGER,
        "spike_width": _POSITIVE_INTEGER,
    },
    "streak_masking": {
        "enable": _BOOL,
        "mode": _enum("auto_ground"),
        "profile_accept": _BOOL,
        "debug": _BOOL,
        "houghpeak_params": {
            "enable": _BOOL,
            "bin": _POSITIVE_INTEGER,
            "thresh_sig": _POSITIVE,
            "min_votes": _POSITIVE_INTEGER,
            "max_candidates": _POSITIVE_INTEGER,
            "confidence_threshold": _FRACTION,
        },
        "contour_params": {
            "enable": _BOOL,
            "thresh_sig": _POSITIVE,
            "min_span": _POSITIVE,
            "shape_cut": _FRACTION,
            "area_cut": _POSITIVE,
            "radius_dev_cut": _NONNEGATIVE,
            "confidence_threshold": _FRACTION,
        },
        "mask_params": {
            "strip_length": _POSITIVE_INTEGER,
            "strip_width": _POSITIVE_INTEGER,
            "profile_sigma_threshold": _POSITIVE,
            "profile_percentile": _PERCENTILE,
            "rotation_interpolation_order": _integer(0, 5),
            "padding": _NONNEGATIVE_INTEGER,
            "min_mask_pixels": _POSITIVE_INTEGER,
            "min_row_hits": _POSITIVE_INTEGER,
            "min_row_hit_fraction": _FRACTION,
            "min_col_hit_fraction": _FRACTION,
            "max_support_width": _POSITIVE_INTEGER,
            "max_premasked_fraction": _Rule("number", 0, 1, nullable=True),
            "max_component_sigma": _number(0, exclusive_minimum=True, nullable=True),
            "max_row_gap": _NONNEGATIVE_INTEGER,
            "min_row_run": _NONNEGATIVE_INTEGER,
            "angle_fan_slack": _NONNEGATIVE,
        },
        "enable_sparse_ransac": _BOOL,
        "sparse_ransac_params": {
            "detect_thresh_sig": _POSITIVE,
            "residual_threshold": _POSITIVE,
            "min_inliers": _POSITIVE_INTEGER,
            "min_length": _POSITIVE,
            "min_line_density": _POSITIVE,
            "dilation_radius": _NONNEGATIVE_INTEGER,
            "max_trials": _POSITIVE_INTEGER,
            "max_trails": _POSITIVE_INTEGER,
        },
    },
    "variance": {
        "method": _enum("theoretical", "rms_map", "empirical_fit"),
        "gain_keyword": _KEYWORDS,
        "rdnoise_keyword": _KEYWORDS,
        "default_gain": _POSITIVE,
        "default_rdnoise": _NONNEGATIVE,
        "epsilon": _POSITIVE,
        "flat_rel_noise": _NONNEGATIVE,
        "rescale_variance": _BOOL,
        "empirical_patch_size": _POSITIVE_INTEGER,
        "empirical_clip_sigma": _POSITIVE,
    },
    "confidence_params": {
        "dtype": _enum("float16", "float32", "float64"),
        "normalize_percentile": _PERCENTILE,
        "normalize_scope": _enum("per_hdu", "per_exposure"),
        "scale_to_100": _BOOL,
    },
    "output_params": {
        "output_map_format": _enum("weight", "confidence"),
        "mask_detected_in_weight": _BOOL,
        "mask_bitpix": _Rule("integer", choices=frozenset({8, 16, 32, 64})),
        "ivar_bitpix": _Rule("integer", choices=frozenset({32, 64, -32, -64})),
        "compress": _BOOL,
        "sky_format": _enum("full", "mesh", casefold=True),
    },
}


# One phrase per rule kind, so a missing value and a wrong-typed value are
# described the same way. Falling back to the raw kind keeps an unknown rule
# name reporting something rather than raising on the error path.
_KIND_DESCRIPTIONS = {
    "boolean": "a boolean",
    "integer": "an integer",
    "number": "a number",
    "string": "a non-empty string",
    "keywords": "a header keyword string or non-empty list of strings",
}


def _type_error(path, expected):
    return f"'{path}' must be {expected}."


def _validate_rule(value, rule, path):
    description = _KIND_DESCRIPTIONS.get(rule.kind, rule.kind)
    if value is None:
        return [] if rule.nullable else [_type_error(path, description)]
    if rule.kind == "boolean":
        if not isinstance(value, (bool, np.bool_)):
            return [_type_error(path, description)]
        return []
    if rule.kind == "integer":
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            return [_type_error(path, description)]
    elif rule.kind == "number":
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            return [_type_error(path, description)]
        if not np.isfinite(value):
            return [f"'{path}' must be finite."]
    elif rule.kind == "string":
        if not isinstance(value, str) or not value:
            return [_type_error(path, description)]
    elif rule.kind == "keywords":
        values = [value] if isinstance(value, str) else value
        if (
            not isinstance(values, (list, tuple))
            or not values
            or not all(isinstance(item, str) and item for item in values)
        ):
            return [_type_error(path, description)]
        return []
    else:
        return [f"Internal validation error for '{path}'."]

    compared = value.casefold() if rule.casefold and isinstance(value, str) else value
    if rule.choices is not None and compared not in rule.choices:
        try:
            ordered = sorted(rule.choices)
        except TypeError:  # heterogeneous choices have no natural order
            ordered = sorted(rule.choices, key=str)
        allowed = ", ".join(map(repr, ordered))
        return [f"'{path}' must be one of {allowed}."]
    if rule.minimum is not None:
        invalid = value <= rule.minimum if rule.exclusive_minimum else value < rule.minimum
        if invalid:
            relation = "greater than" if rule.exclusive_minimum else "at least"
            return [f"'{path}' must be {relation} {rule.minimum}."]
    if rule.maximum is not None and value > rule.maximum:
        return [f"'{path}' must be at most {rule.maximum}."]
    if rule.odd and value % 2 == 0:
        return [f"'{path}' must be odd."]
    return []


def _validate_mapping(value, schema, path=""):
    if not isinstance(value, dict):
        name = path or "Configuration"
        return [_type_error(name, "a dictionary")]
    errors = []
    unknown = sorted(set(value) - set(schema), key=str)
    for key in unknown:
        errors.append(f"Unsupported configuration key '{path + '.' if path else ''}{key}'.")
    for key, child in schema.items():
        if key not in value:
            continue
        child_path = f"{path}.{key}" if path else key
        if isinstance(child, dict):
            errors.extend(_validate_mapping(value[key], child, child_path))
        else:
            errors.extend(_validate_rule(value[key], child, child_path))
    return errors


def _nested(config, path, default=None):
    value = config
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _cross_field_errors(config):
    checks = [
        (("flat_masking", "local_low_thresh"), 0.5, ("flat_masking", "local_high_thresh"), 2.0, "<"),
        (("saturation", "bleed_cap_min"), 20, ("saturation", "bleed_cap_max"), 200, "<="),
        (
            ("cosmic_ray", "faint_cr", "min_component_area"),
            3,
            ("cosmic_ray", "faint_cr", "max_component_area"),
            12,
            "<=",
        ),
        (
            ("cosmic_ray", "faint_cr", "residual", "min_component_area"),
            4,
            ("cosmic_ray", "faint_cr", "residual", "max_component_area"),
            12,
            "<=",
        ),
        (
            ("streak_masking", "mask_params", "min_row_hits"),
            8,
            ("streak_masking", "mask_params", "strip_length"),
            256,
            "<=",
        ),
        (
            ("streak_masking", "mask_params", "max_support_width"),
            16,
            ("streak_masking", "mask_params", "strip_width"),
            96,
            "<=",
        ),
    ]
    errors = []
    for left_path, left_default, right_path, right_default, operator in checks:
        left = _nested(config, left_path, left_default)
        right = _nested(config, right_path, right_default)
        valid = left < right if operator == "<" else left <= right
        if not valid:
            errors.append(f"'{'.'.join(left_path)}' must be {operator} '{'.'.join(right_path)}'.")
    box_size = _nested(config, ("sep_background", "box_size"), 128)
    max_box_size = _nested(config, ("sep_background", "max_box_size"), max(box_size, 1024))
    if box_size > max_box_size:
        errors.append("'sep_background.box_size' must be <= 'sep_background.max_box_size'.")
    hist_min = _nested(config, ("saturation", "histogram_params", "hist_min_adu"))
    hist_max = _nested(config, ("saturation", "histogram_params", "hist_max_adu"))
    if hist_min is not None and hist_max is not None and hist_min >= hist_max:
        errors.append(
            "'saturation.histogram_params.hist_min_adu' must be < 'saturation.histogram_params.hist_max_adu'."
        )
    return errors


def configuration_errors(config):
    errors = _validate_mapping(config, CONFIG_SCHEMA)
    if not errors:
        errors.extend(_cross_field_errors(config))
    return errors
