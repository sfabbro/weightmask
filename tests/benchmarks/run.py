import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

try:
    import fitsio
except Exception:  # pragma: no cover
    fitsio = None

from tests.benchmarks.download_data import process_suite, validate_case_file
from tests.simulate_and_test import _f1, run_masking_test
from weightmask.bad import detect_bad_pixels
from weightmask.contract import (
    INVERSE_VARIANCE_SEMANTICS,
    MASK_POLARITY,
    QualityBit,
    build_weight_product,
    valid_inverse_variance,
)
from weightmask.cosmics import detect_cosmic_rays
from weightmask.streaks import detect_streaks
from weightmask.variance import calculate_inverse_variance

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = Path(__file__).resolve().parent / "manifests"
OUTPUT_ROOT = ROOT / "test_outputs" / "benchmarks"


def _load_repo_config():
    with open(ROOT / "weightmask.yml", "r") as handle:
        return yaml.safe_load(handle)


def load_manifest(suite_name):
    manifest_path = MANIFEST_DIR / f"{suite_name}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest for suite '{suite_name}'")
    with open(manifest_path, "r") as handle:
        return json.load(handle)


def _mask_stats(pred_mask, gt_mask, eval_gt_mask=None):
    if eval_gt_mask is None:
        eval_gt_mask = gt_mask
    tp = int(np.sum(pred_mask & eval_gt_mask))
    fp = int(np.sum(pred_mask & (~eval_gt_mask)))
    fn = int(np.sum((~pred_mask) & gt_mask))
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    recall = min(recall, 1.0)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(_f1(precision, recall)),
        "pred_area": int(np.sum(pred_mask)),
        "gt_area": int(np.sum(gt_mask)),
        "overmask_fraction": float(fp / (np.sum(pred_mask) + 1e-9)) if np.sum(pred_mask) > 0 else 0.0,
        "width_ratio": float(np.sum(pred_mask) / (np.sum(gt_mask) + 1e-9)) if np.sum(gt_mask) > 0 else 0.0,
    }


def _simple_hough_baseline(data_sub, bkg_rms):
    """Satdet-style baseline without MRT rescue (calls private helpers)."""
    from weightmask.streaks import (
        _detect_streaks_contours,
        _detect_streaks_houghpeaks,
        _detect_streaks_satdet,
    )

    config = {
        "enable": True,
        "mode": "auto_ground",
        "debug": True,
        "enable_sparse_ransac": False,
        "satdet_params": {
            "rescale_percentiles": [2.0, 98.5],
            "gaussian_sigmas": [1.5],
            "canny_low_threshold": 0.08,
            "canny_high_threshold": 0.25,
            "small_edge_perimeter": 40,
            "hough_threshold": 8,
            "hough_min_line_length": 90,
            "hough_max_line_gap": 25,
            "cluster_angle_tol_deg": 3.0,
            "cluster_rho_tol_px": 24.0,
            "min_cluster_segments": 2,
            "edge_buffer": 20,
            "min_edge_touches": 0,
            "min_interior_span": 80.0,
            "min_segment_density": 0.01,
            "candidate_corridor_radius": 10,
            "max_existing_mask_fraction": 0.85,
            "confidence_threshold": 0.25,
        },
        "mask_params": {
            "strip_length": 196,
            "strip_width": 64,
            "profile_sigma_threshold": 1.0,
            "profile_percentile": 70.0,
            "rotation_interpolation_order": 1,
            "padding": 2,
            "min_mask_pixels": 16,
            "min_row_hits": 5,
            "min_row_hit_fraction": 0.2,
            "max_support_width": 18,
        },
    }
    empty = np.zeros_like(data_sub, dtype=bool)
    mask = np.zeros_like(data_sub, dtype=bool)
    hp_mask, _, _ = _detect_streaks_houghpeaks(data_sub, bkg_rms, empty, config)
    mask |= hp_mask
    ct_mask, _, _ = _detect_streaks_contours(data_sub, bkg_rms, empty, config)
    mask |= ct_mask
    sat_mask, _, _ = _detect_streaks_satdet(data_sub, bkg_rms, empty, config)
    mask |= sat_mask
    return mask


def _rubin_compatible_baseline(data_sub, bkg_rms):
    config = {
        "enable": True,
        "mode": "auto_ground",
        "debug": True,
        "enable_sparse_ransac": False,
        "satdet_params": {
            "rescale_percentiles": [1.0, 99.2],
            "gaussian_sigmas": [1.0, 2.0],
            "canny_low_threshold": 0.06,
            "canny_high_threshold": 0.18,
            "small_edge_perimeter": 35,
            "hough_threshold": 5,
            "hough_min_line_length": 60,
            "hough_max_line_gap": 18,
            "cluster_angle_tol_deg": 2.0,
            "cluster_rho_tol_px": 30.0,
            "min_cluster_segments": 2,
            "edge_buffer": 12,
            "min_edge_touches": 0,
            "min_interior_span": 70.0,
            "min_segment_density": 0.01,
            "candidate_corridor_radius": 10,
            "max_existing_mask_fraction": 0.9,
            "confidence_threshold": 0.20,
        },
        "mrt_rescue_params": {
            "theta_step_deg": 1.0,
            "peak_threshold_sig": 4.0,
            "max_candidates": 4,
            "confidence_threshold": 0.22,
        },
        "mask_params": {
            "strip_length": 220,
            "strip_width": 72,
            "profile_sigma_threshold": 1.0,
            "profile_percentile": 72.0,
            "rotation_interpolation_order": 1,
            "padding": 2,
            "min_mask_pixels": 16,
            "min_row_hits": 4,
            "min_row_hit_fraction": 0.18,
            "max_support_width": 20,
        },
    }
    return detect_streaks(data_sub, bkg_rms, np.zeros_like(data_sub, dtype=bool), config)


def _benchmark_synthetic_bad_pixels(seed, size):
    rng = np.random.default_rng(seed)
    flat = np.ones((size, size), dtype=np.float32)
    truth = np.zeros((size, size), dtype=bool)
    hot_coords = rng.integers(8, size - 8, size=(20, 2))
    for y, x in hot_coords:
        flat[y, x] = 4.0
        truth[y, x] = True
    dead_coords = rng.integers(8, size - 8, size=(15, 2))
    for y, x in dead_coords:
        flat[y, x] = 0.1
        truth[y, x] = True
    bad_col = int(rng.integers(12, size - 12))
    flat[:, bad_col] = 0.05
    truth[:, bad_col] = True

    pred = detect_bad_pixels(
        flat,
        {
            "local_filter_size": 9,
            "local_low_thresh": 0.5,
            "local_high_thresh": 1.8,
            "col_enable": True,
            "col_deriv_sigma": 5.0,
            "col_dead_thresh": 0.1,
        },
        using_unit_flat=False,
    )
    return _mask_stats(pred, truth)


def _synthetic_v2_cases():
    return [
        {
            "name": "synthetic_sparse",
            "size": 384,
            "noise": 5.0,
            "stars": 20,
            "streak": 50.0,
            "mask_pct": 0.0,
            "regime_type": "normal",
            "seed": 11,
        },
        {
            "name": "synthetic_complex",
            "size": 384,
            "noise": 12.0,
            "stars": 80,
            "streak": 40.0,
            "mask_pct": 0.0,
            "regime_type": "complex",
            "seed": 21,
        },
        {
            "name": "synthetic_gradient_step",
            "size": 384,
            "noise": 12.0,
            "stars": 80,
            "streak": 35.0,
            "mask_pct": 0.0,
            "regime_type": "amplifier_step",
            "seed": 31,
        },
        {
            "name": "synthetic_variable_width",
            "size": 384,
            "noise": 10.0,
            "stars": 60,
            "streak": 45.0,
            "mask_pct": 0.0,
            "regime_type": "variable_width_streak",
            "seed": 41,
        },
        {
            "name": "synthetic_elongated_sources",
            "size": 384,
            "noise": 10.0,
            "stars": 60,
            "streak": 35.0,
            "mask_pct": 0.0,
            "regime_type": "elongated_galaxies",
            "seed": 51,
        },
    ]


def run_synthetic_v2(with_baselines=False, selected_cases=None):
    out_dir = OUTPUT_ROOT / "synthetic_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for case in _synthetic_v2_cases():
        if selected_cases and case["name"] not in selected_cases:
            continue
        args = SimpleNamespace(**case)
        metrics, products = run_masking_test(str(ROOT / "weightmask.yml"), args, save_fits=False, return_products=True)
        case_result = {"weightmask": metrics}
        from scipy.ndimage import binary_dilation

        dilated_streak_gt = binary_dilation(products["ground_truth"]["streak"], iterations=2)
        streak_stats = _mask_stats(
            products["masks"]["streaks"], products["ground_truth"]["streak"], eval_gt_mask=dilated_streak_gt
        )
        case_result["streak_stats"] = streak_stats
        case_result["bad_pixel_stats"] = _benchmark_synthetic_bad_pixels(case["seed"], case["size"])
        if with_baselines:
            data_sub = products["science"] - np.nanmedian(products["science"])
            case_result["baselines"] = {
                "simple_hough": _mask_stats(
                    _simple_hough_baseline(data_sub, products["bkg_rms"]),
                    products["ground_truth"]["streak"],
                    eval_gt_mask=dilated_streak_gt,
                ),
                "rubin_compatible_kht": _mask_stats(
                    _rubin_compatible_baseline(data_sub, products["bkg_rms"]),
                    products["ground_truth"]["streak"],
                    eval_gt_mask=dilated_streak_gt,
                ),
            }
        results[case["name"]] = case_result
        np.savez_compressed(
            out_dir / f"{case['name']}.npz",
            science=products["science"],
            bkg_rms=products["bkg_rms"],
            streak_pred=products["masks"]["streaks"].astype(np.uint8),
            streak_truth=products["ground_truth"]["streak"].astype(np.uint8),
        )
    failures = []
    if not selected_cases and results:
        streak_f1 = [result["streak_stats"]["f1"] for result in results.values()]
        object_recall = [result["weightmask"]["Objects"][1] for result in results.values()]
        bad_pixel_f1 = [result["bad_pixel_stats"]["f1"] for result in results.values()]
        if float(np.mean(streak_f1)) < 0.20:
            failures.append(f"Synthetic-v2 average streak F1 {float(np.mean(streak_f1)):.3f} < 0.200")
        if float(np.mean(object_recall)) < 0.90:
            failures.append(f"Synthetic-v2 average object recall {float(np.mean(object_recall)):.3f} < 0.900")
        if float(np.mean(bad_pixel_f1)) < 0.50:
            failures.append(f"Synthetic-v2 average bad-pixel F1 {float(np.mean(bad_pixel_f1)):.3f} < 0.500")
    return {"suite": "synthetic_v2", "results": results, "gate_failures": failures}


def _run_manifest_suite(manifest, with_baselines=False, selected_cases=None):
    suite_results = {}
    failures = []
    for case in manifest["cases"]:
        if selected_cases and case["case_id"] not in selected_cases:
            continue
        print(f"[{manifest['suite']}] evaluating {case['case_id']} ...")
        case_path = ROOT / case["local_path"]
        result = {
            "case_id": case["case_id"],
            "source_identifier": case["source_identifier"],
            "status": "missing_data",
            "download_url": case.get("download_url"),
        }
        if not case_path.exists() or fitsio is None:
            failures.append(f"{case['case_id']}: required science data is unavailable at {case['local_path']}")
            suite_results[case["case_id"]] = result
            continue

        if case["label_recipe"] == "dark_injection_v1":
            dark_path = ROOT / case.get("dark_local_path", "")
            if not dark_path.exists():
                result["validation_error"] = f"required dark residual is unavailable at {case.get('dark_local_path')}"
                failures.append(f"{case['case_id']}: {result['validation_error']}")
                suite_results[case["case_id"]] = result
                continue

        valid, reason = validate_case_file(case, case_path)
        if not valid:
            result["status"] = "invalid_instrument"
            result["validation_error"] = reason
            failures.append(f"{case['case_id']}: {reason}")
            suite_results[case["case_id"]] = result
            continue

        science_sha256 = None
        if case["label_recipe"] == "manual_streak_mask":
            science_sha256, provenance_error = _validate_science_pin(case, case_path)
            if science_sha256 is not None:
                result["science_sha256"] = science_sha256
            if provenance_error:
                result["status"] = "invalid_science_provenance"
                result["validation_error"] = provenance_error
                failures.append(f"{case['case_id']}: {provenance_error}")
                suite_results[case["case_id"]] = result
                continue

        data, hdr = _load_case_image(case_path)
        if data is None:
            result["status"] = "load_failed"
            failures.append(f"{case['case_id']}: science image could not be loaded")
            suite_results[case["case_id"]] = result
            continue
        science_shape = data.shape
        data, crop = _select_case_cutout(case, data)

        truth_mask = None
        label_evidence = None
        if case["label_recipe"] == "manual_streak_mask":
            truth_mask, label_evidence, label_status, label_error = _load_case_label(case, science_shape, crop)
            if label_error:
                result["status"] = label_status
                result["validation_error"] = label_error
                failures.append(f"{case['case_id']}: {label_error}")
                suite_results[case["case_id"]] = result
                continue

        reference_inverse_variance, native_quality_mask, reference_error = _load_quality_reference(
            case, case_path, hdr, crop
        )
        if reference_error:
            result["status"] = "invalid_quality_reference"
            result["validation_error"] = reference_error
            failures.append(f"{case['case_id']}: {reference_error}")
            suite_results[case["case_id"]] = result
            continue

        result["status"] = "loaded"
        result["shape"] = list(data.shape)
        result["crop"] = crop
        result["instrument"] = hdr.get("INSTRUME") or hdr.get("DETECTOR")
        result["extname"] = hdr.get("EXTNAME")
        result["exptime"] = hdr.get("EXPTIME")
        if science_sha256 is not None:
            result["label_artifact"] = label_evidence
        metrics = _evaluate_real_case(
            case,
            data,
            hdr,
            with_baselines=with_baselines,
            truth_mask=truth_mask,
            reference_inverse_variance=reference_inverse_variance,
            native_quality_mask=native_quality_mask,
        )
        result.update(metrics)
        quality_metrics = metrics.get("quality_validation")
        if quality_metrics is not None:
            failures.extend(_quality_gate_failures(case["case_id"], quality_metrics, manifest.get("quality_gates", {})))
        suite_results[case["case_id"]] = result
    return {"suite": manifest["suite"], "results": suite_results, "gate_failures": failures}


def _load_case_image(case_path):
    """Load the largest likely science image from a FITS file."""
    best = None
    hdr_best = None
    best_idx = None
    with fitsio.FITS(case_path) as hdul:
        for idx, hdu in enumerate(hdul):
            try:
                hdr = hdu.read_header()
                naxis = int(hdr.get("NAXIS", 0))
            except Exception:
                continue
            if naxis != 2:
                continue
            extname = str(hdr.get("EXTNAME", "")).upper()
            if extname in {"ERR", "DQ"}:
                continue
            naxis1 = int(hdr.get("NAXIS1", 0))
            naxis2 = int(hdr.get("NAXIS2", 0))
            if naxis1 <= 0 or naxis2 <= 0:
                continue
            score = int(naxis1 * naxis2)
            if best is None or score > best[0]:
                best = (score, (naxis2, naxis1))
                hdr_best = hdr
                best_idx = idx
    if best is None:
        return None, None
    with fitsio.FITS(case_path) as hdul:
        return hdul[best_idx].read().astype(np.float32), hdr_best


def _iter_windows(shape, window_shape, stride):
    h, w = shape
    wh, ww = window_shape
    ys = list(range(0, max(h - wh, 0) + 1, stride))
    xs = list(range(0, max(w - ww, 0) + 1, stride))
    if not ys or ys[-1] != max(h - wh, 0):
        ys.append(max(h - wh, 0))
    if not xs or xs[-1] != max(w - ww, 0):
        xs.append(max(w - ww, 0))
    for y0 in sorted(set(ys)):
        for x0 in sorted(set(xs)):
            yield y0, x0, y0 + wh, x0 + ww


def _stable_seed(text):
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(text)) % (2**32)


def _center_crop(data, crop_shape):
    h, w = data.shape
    ch, cw = crop_shape
    ch = min(ch, h)
    cw = min(cw, w)
    y0 = max(0, (h - ch) // 2)
    x0 = max(0, (w - cw) // 2)
    return data[y0 : y0 + ch, x0 : x0 + cw], {"y0": y0, "x0": x0, "height": ch, "width": cw}


def _random_crop(data, crop_shape, seed):
    h, w = data.shape
    ch, cw = crop_shape
    ch = min(ch, h)
    cw = min(cw, w)
    rng = np.random.default_rng(seed)
    y0 = 0 if h == ch else int(rng.integers(0, h - ch + 1))
    x0 = 0 if w == cw else int(rng.integers(0, w - cw + 1))
    return data[y0 : y0 + ch, x0 : x0 + cw], {"y0": y0, "x0": x0, "height": ch, "width": cw}


def _select_case_cutout(case, data):
    """Select one deterministic native-resolution cutout for a real-data case."""
    crop_shape = tuple(case.get("crop_shape", [1024, 1024]))
    if data.shape[0] <= crop_shape[0] and data.shape[1] <= crop_shape[1]:
        return data, {"y0": 0, "x0": 0, "height": data.shape[0], "width": data.shape[1]}
    mode = case.get("crop_mode", "random")
    if mode == "center":
        return _center_crop(data, crop_shape)
    seed = int(case.get("cutout_seed", _stable_seed(case["case_id"])))
    return _random_crop(data, crop_shape, seed)


def _crop_reference(array, crop):
    """Apply a science cutout to a same-frame reference plane."""
    array = np.asarray(array)
    expected_shape = (int(crop["height"]), int(crop["width"]))
    if array.ndim != 2:
        raise ValueError(f"reference plane must be 2-D, got shape {array.shape}")
    if array.shape == expected_shape:
        return array
    y0 = int(crop["y0"])
    x0 = int(crop["x0"])
    y1 = y0 + expected_shape[0]
    x1 = x0 + expected_shape[1]
    if y1 > array.shape[0] or x1 > array.shape[1]:
        raise ValueError(f"reference plane shape {array.shape} does not cover science crop {crop}")
    return array[y0:y1, x0:x1]


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _label_artifact(case):
    artifact = case.get("label_artifact")
    if not isinstance(artifact, dict):
        return None, "manual label recipe requires a label_artifact mapping"
    required = {"path", "sha256", "science_sha256", "coordinate_frame", "polarity"}
    missing = sorted(required - artifact.keys())
    if missing:
        return None, f"label_artifact is missing required fields: {', '.join(missing)}"
    if artifact["coordinate_frame"] != "science_full_frame":
        return None, "label_artifact coordinate_frame must be science_full_frame"
    if artifact["polarity"] != "one_means_trail":
        return None, "label_artifact polarity must be one_means_trail"
    return artifact, None


def _valid_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _validate_science_pin(case, case_path):
    """Require an exact source-exposure pin before accepting a manual label."""
    artifact, error = _label_artifact(case)
    if error:
        return None, error
    expected = artifact["science_sha256"]
    if not _valid_sha256(expected):
        return None, "label_artifact science_sha256 is not a lowercase SHA-256 digest"
    actual = _sha256_file(case_path)
    if actual != expected:
        return actual, f"science SHA-256 mismatch: expected {expected}, got {actual}"
    return actual, None


def _load_case_label(case, science_shape, crop):
    """Load a provenance-pinned, full-science-frame binary manual label."""
    artifact, error = _label_artifact(case)
    if error:
        return None, None, "invalid_label_provenance", error
    label_path_value = artifact["path"]
    if not isinstance(label_path_value, str) or not label_path_value:
        return None, None, "invalid_label_provenance", "label_artifact path must be a non-empty string"
    label_path = ROOT / label_path_value
    if not label_path.exists():
        return None, None, "missing_labels", f"required label is unavailable at {label_path_value}"
    expected_hash = artifact["sha256"]
    if not _valid_sha256(expected_hash):
        return (
            None,
            None,
            "invalid_label_provenance",
            "label_artifact sha256 is not pinned to a lowercase SHA-256 digest",
        )
    actual_hash = _sha256_file(label_path)
    if actual_hash != expected_hash:
        return (
            None,
            None,
            "invalid_label_provenance",
            f"label SHA-256 mismatch: expected {expected_hash}, got {actual_hash}",
        )
    label, _ = _load_case_image(label_path)
    if label is None:
        return None, None, "invalid_labels", f"could not load required label at {label_path_value}"
    if label.shape != tuple(science_shape):
        return (
            None,
            None,
            "invalid_labels",
            f"full-frame label shape {label.shape} does not match science shape {tuple(science_shape)}",
        )
    if not np.all(np.isfinite(label)) or not np.all((label == 0) | (label == 1)):
        return None, None, "invalid_labels", "manual label must be a finite binary 0/1 mask"
    try:
        label = _crop_reference(label, crop).astype(bool)
    except ValueError as exc:
        return None, None, "invalid_labels", str(exc)
    if not np.any(label):
        return (
            None,
            None,
            "invalid_labels",
            f"required manual label at {label_path_value} contains no flagged pixels in the selected cutout",
        )
    evidence = {
        "path": label_path_value,
        "sha256": actual_hash,
        "science_sha256": artifact["science_sha256"],
        "coordinate_frame": artifact["coordinate_frame"],
        "polarity": artifact["polarity"],
    }
    return label, evidence, None, None


def _load_named_plane(case_path, extname, extver, crop):
    """Load an ERR/DQ plane matching the selected science extension."""
    with fitsio.FITS(case_path) as hdul:
        for hdu in hdul:
            try:
                header = hdu.read_header()
                if str(header.get("EXTNAME", "")).upper() != extname.upper():
                    continue
                if extver is not None and str(header.get("EXTVER")) != str(extver):
                    continue
                return _crop_reference(hdu.read(), crop)
            except (OSError, RuntimeError, TypeError, ValueError):
                continue
    return None


def _load_quality_reference(case, case_path, science_header, crop):
    """Load instrument-provided inverse variance and quality-mask semantics."""
    reference = case.get("quality_reference")
    if reference is None:
        return None, None, None
    if not isinstance(reference, dict):
        return None, None, "quality_reference must be a mapping"

    inverse_variance_name = reference.get("inverse_variance")
    quality_mask_name = reference.get("quality_mask")
    if not inverse_variance_name or not quality_mask_name:
        return None, None, "quality_reference requires inverse_variance and quality_mask planes"

    extver = science_header.get("EXTVER")
    error = _load_named_plane(case_path, inverse_variance_name, extver, crop)
    dq = _load_named_plane(case_path, quality_mask_name, extver, crop)
    if error is None or dq is None:
        return (
            None,
            None,
            (f"missing matching {inverse_variance_name}/{quality_mask_name} planes for science EXTVER={extver}"),
        )

    error = np.asarray(error, dtype=np.float64)
    inverse_variance = np.zeros(error.shape, dtype=np.float32)
    usable = np.isfinite(error) & (error > 0)
    inverse_variance[usable] = (1.0 / np.square(error[usable])).astype(np.float32)
    return inverse_variance, np.asarray(dq) != 0, None


def _robust_sigma(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    q1, q3 = np.percentile(values, (25.0, 75.0))
    return float(0.7413 * (q3 - q1))


def _image_inverse_variance(data, source_mask, header):
    """Run the classical theoretical-plus-rescaling path for flat-fielded data."""
    finite = np.isfinite(data)
    sky_level = float(np.nanmedian(data))
    noise = _robust_sigma(data[finite] - sky_level)
    sky_map = np.full(data.shape, max(sky_level, 0.0), dtype=np.float32)
    flat_map = np.ones(data.shape, dtype=np.float32)
    bkg_rms = np.full(data.shape, max(noise, 1e-6), dtype=np.float32)
    variance_cfg = dict(_load_repo_config().get("variance", {}))
    variance_cfg["gain"] = float(header.get("GAIN", variance_cfg.get("default_gain", 1.5)))
    variance_cfg["read_noise"] = float(header.get("RDNOISE", variance_cfg.get("default_rdnoise", 5.0)))
    return calculate_inverse_variance(
        variance_cfg,
        sky_map,
        flat_map,
        bkg_rms,
        sci_data=data,
        obj_mask=source_mask,
    )


def _quality_validation_metrics(data, predicted_mask, truth_mask, inverse_variance, native_quality_mask=None):
    """Measure the W1 quality semantics without changing detector algorithms."""
    data = np.asarray(data)
    predicted_mask = np.asarray(predicted_mask, dtype=bool)
    truth_mask = np.asarray(truth_mask, dtype=bool)
    native_quality_mask = (
        np.zeros(data.shape, dtype=bool) if native_quality_mask is None else np.asarray(native_quality_mask, dtype=bool)
    )
    for name, array in {
        "predicted mask": predicted_mask,
        "truth mask": truth_mask,
        "native quality mask": native_quality_mask,
        "inverse variance": inverse_variance,
    }.items():
        if np.asarray(array).shape != data.shape:
            raise ValueError(f"{name} shape {np.asarray(array).shape} does not match science shape {data.shape}")

    quality_mask = np.zeros(data.shape, dtype=np.uint32)
    quality_mask[native_quality_mask] |= np.uint32(QualityBit.BAD_PIXEL)
    quality_mask[predicted_mask] |= np.uint32(QualityBit.STREAK)
    product = build_weight_product(inverse_variance, quality_mask)

    finite = np.isfinite(data)
    center = float(np.nanmedian(data))
    sigma = _robust_sigma(data[finite] - center)
    source_mask = finite & ((data - center) > 8.0 * sigma) if sigma > 0 else np.zeros(data.shape, dtype=bool)
    reference_flagged = truth_mask | native_quality_mask
    usable_ivar = valid_inverse_variance(inverse_variance)
    background = finite & usable_ivar & ~reference_flagged & ~source_mask
    if sigma > 0:
        background &= np.abs(data - center) < 3.0 * sigma

    overmask_fraction = float(np.mean(predicted_mask[background])) if np.any(background) else None
    positive_flux = np.maximum(data - center, 0.0)
    flux_region = source_mask & ~reference_flagged
    source_flux = float(np.sum(positive_flux[flux_region]))
    flux_bias_fraction = (
        float(np.sum(positive_flux[flux_region & predicted_mask]) / source_flux) if source_flux > 0 else None
    )
    standardized_background = (data[background] - np.median(data[background])) * np.sqrt(
        product.inverse_variance[background]
    )
    noise_scale = _robust_sigma(standardized_background) if standardized_background.size else None

    excluded = native_quality_mask | predicted_mask | ~usable_ivar
    clean = usable_ivar & ~native_quality_mask & ~predicted_mask
    invalid = ~usable_ivar
    return {
        "mask_polarity": product.metadata["quality_mask"].mask_polarity,
        "inverse_variance_semantics": product.metadata["inverse_variance"].semantics,
        "flagged_pixels_have_zero_weight": bool(np.all(product.weight[excluded] == 0)),
        "clean_weight_matches_inverse_variance": bool(
            np.allclose(product.weight[clean], product.inverse_variance[clean])
        ),
        "invalid_variance_is_flagged": bool(
            np.all((product.quality_mask[invalid] & np.uint32(QualityBit.INVALID_VARIANCE)) != 0)
        ),
        "overmask_fraction": overmask_fraction,
        "flux_bias_fraction": flux_bias_fraction,
        "noise_scale": noise_scale,
        "background_pixels": int(np.count_nonzero(background)),
        "source_pixels": int(np.count_nonzero(flux_region)),
    }


def _quality_gate_failures(case_id, metrics, gates):
    failures = []
    if metrics.get("mask_polarity") != MASK_POLARITY:
        failures.append(f"{case_id}: mask_polarity is not {MASK_POLARITY}")
    if metrics.get("inverse_variance_semantics") != INVERSE_VARIANCE_SEMANTICS:
        failures.append(f"{case_id}: inverse_variance_semantics is not {INVERSE_VARIANCE_SEMANTICS}")
    for key in (
        "flagged_pixels_have_zero_weight",
        "clean_weight_matches_inverse_variance",
        "invalid_variance_is_flagged",
    ):
        if not metrics.get(key):
            failures.append(f"{case_id}: {key} is false")

    limits = (
        ("overmask_fraction", "max_overmask_fraction", lambda value, limit: value > limit, ">"),
        ("flux_bias_fraction", "max_flux_bias_fraction", lambda value, limit: value > limit, ">"),
        ("noise_scale", "noise_scale_min", lambda value, limit: value < limit, "<"),
        ("noise_scale", "noise_scale_max", lambda value, limit: value > limit, ">"),
    )
    for metric_name, gate_name, failed, relation in limits:
        value = metrics.get(metric_name)
        limit = gates.get(gate_name)
        if value is None:
            failures.append(f"{case_id}: {metric_name} is unavailable")
        elif limit is not None and failed(value, limit):
            failures.append(f"{case_id}: {metric_name} {value:.6f} {relation} {limit:.6f}")
    return failures


def _write_debug_mask(case_id, suite_name, **arrays):
    out_dir = OUTPUT_ROOT / suite_name / case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "debug_masks.npz", **arrays)


def _match_dark_to_science(dark, science_shape, seed):
    """Select one deterministic native-resolution dark cutout matching the science cutout."""
    crop, _ = _random_crop(dark, science_shape, seed)
    return crop


def _evaluate_blank_control(data, bkg_rms, with_baselines):
    streak_cfg = _load_repo_config()["streak_masking"]
    streak_mask = detect_streaks(
        data - np.nanmedian(data),
        bkg_rms,
        np.zeros_like(data, dtype=bool),
        {**streak_cfg, "enable": True, "mode": "auto_ground", "debug": True, "enable_sparse_ransac": True},
    )
    metrics = {
        "weightmask": {
            "streak_area_fraction": float(np.mean(streak_mask)),
            "streak_pixels": int(np.sum(streak_mask)),
        }
    }
    baselines = {}
    if with_baselines:
        baselines["simple_hough"] = {
            "streak_pixels": int(np.sum(_simple_hough_baseline(data - np.nanmedian(data), bkg_rms)))
        }
        baselines["rubin_compatible_kht"] = {
            "streak_pixels": int(np.sum(_rubin_compatible_baseline(data - np.nanmedian(data), bkg_rms)))
        }
    return metrics, streak_mask, baselines


def _evaluate_dark_injection(case, data, hdr, with_baselines):
    dark_path = ROOT / case.get("dark_local_path", "")
    if not dark_path.exists():
        return (
            {"weightmask": {"status": "missing_dark_residual"}},
            {
                "pred_bad": np.zeros_like(data, dtype=bool),
                "pred_cr": np.zeros_like(data, dtype=bool),
                "truth": np.zeros_like(data, dtype=bool),
            },
            {},
        )

    dark, _ = _load_case_image(dark_path)
    if dark is None:
        return (
            {"weightmask": {"status": "load_failed_dark_residual"}},
            {
                "pred_bad": np.zeros_like(data, dtype=bool),
                "pred_cr": np.zeros_like(data, dtype=bool),
                "truth": np.zeros_like(data, dtype=bool),
            },
            {},
        )

    science = data.astype(np.float32)
    dark = _match_dark_to_science(dark.astype(np.float32), science.shape, _stable_seed(case["case_id"] + "_dark"))
    injected = science + dark
    finite_dark = dark[np.isfinite(dark)]
    if finite_dark.size == 0:
        truth = np.zeros_like(dark, dtype=bool)
    else:
        truth = dark > np.percentile(finite_dark, 99.5)

    pred_bad = detect_bad_pixels(
        np.where(np.isfinite(dark), 1.0 + dark / max(np.nanmax(np.abs(dark)), 1.0), 1.0).astype(np.float32),
        {
            "local_filter_size": 9,
            "local_low_thresh": 0.5,
            "local_high_thresh": 1.8,
            "col_enable": True,
            "col_deriv_sigma": 5.0,
            "col_dead_thresh": 0.1,
        },
        using_unit_flat=False,
    )
    existing = np.zeros_like(injected, dtype=bool)
    pred_cr = detect_cosmic_rays(
        injected,
        existing,
        float(hdr.get("SATURATE", 65535.0)),
        float(hdr.get("GAIN", 1.5)),
        float(hdr.get("RDNOISE", 5.0)),
        {"sigclip": 5.0, "objlim": 5.0, "dynamic_objlim": True, "psf_aware": True, "dilate_cr": False},
        bkg_rms_map=np.full_like(injected, np.nanstd(injected - np.nanmedian(injected)) + 1e-3),
    )
    metrics = {
        "weightmask": {
            "bad_pixels": _mask_stats(pred_bad, truth),
            "cosmics": _mask_stats(pred_cr, truth),
        }
    }
    baselines = {}
    if with_baselines:
        dark_thresh = truth
        baselines["dark_threshold_baseline"] = _mask_stats(dark_thresh, truth)
        baselines["astroscrappy_only"] = _mask_stats(pred_cr, truth)
    return metrics, {"pred_bad": pred_bad, "pred_cr": pred_cr, "truth": truth}, baselines


def _evaluate_streak_case(
    case,
    data,
    hdr,
    with_baselines,
    truth_mask,
    reference_inverse_variance,
    native_quality_mask,
):
    bkg_rms = np.full_like(data, np.nanstd(data - np.nanmedian(data)) + 1e-3)
    data_sub = data - np.nanmedian(data)
    streak_cfg = _load_repo_config()["streak_masking"]
    weightmask_mask = detect_streaks(
        data_sub,
        bkg_rms,
        np.zeros_like(data, dtype=bool),
        {**streak_cfg, "enable": True, "mode": "auto_ground", "debug": True, "enable_sparse_ransac": True},
    )
    metrics = {
        "weightmask": {
            "streak_pixels": int(np.sum(weightmask_mask)),
            "streak_area_fraction": float(np.mean(weightmask_mask)),
        }
    }
    if truth_mask is None:
        raise ValueError(f"{case['case_id']} requires a manual streak label")
    if reference_inverse_variance is None:
        sigma = _robust_sigma(data_sub)
        source_mask = np.isfinite(data_sub) & (data_sub > 8.0 * sigma) if sigma > 0 else np.zeros_like(data, bool)
        reference_inverse_variance = _image_inverse_variance(data, source_mask, hdr)
        reference_name = "classical_theoretical_rescaled"
    else:
        reference_name = "instrument_err_dq"
    if reference_inverse_variance is None:
        raise ValueError(f"{case['case_id']} could not produce inverse variance")
    metrics["quality_validation"] = _quality_validation_metrics(
        data,
        weightmask_mask,
        truth_mask,
        reference_inverse_variance,
        native_quality_mask=native_quality_mask,
    )
    metrics["quality_validation"]["reference"] = reference_name
    baselines = {}
    if with_baselines:
        if "simple_hough" in case.get("comparators", []):
            hough_mask = _simple_hough_baseline(data_sub, bkg_rms)
            baselines["simple_hough"] = {
                "streak_pixels": int(np.sum(hough_mask)),
                "overlap_with_weightmask": int(np.sum(hough_mask & weightmask_mask)),
            }
        if "rubin_compatible_kht" in case.get("comparators", []):
            rubin_mask = _rubin_compatible_baseline(data_sub, bkg_rms)
            baselines["rubin_compatible_kht"] = {
                "streak_pixels": int(np.sum(rubin_mask)),
                "overlap_with_weightmask": int(np.sum(rubin_mask & weightmask_mask)),
            }
        if "acstools_detsat" in case.get("comparators", []):
            try:
                baselines["acstools_detsat"] = {"status": "available"}
            except Exception:
                baselines["acstools_detsat"] = {"status": "unavailable"}
        if "acstools_findsat_mrt" in case.get("comparators", []):
            try:
                baselines["acstools_findsat_mrt"] = {"status": "available"}
            except Exception:
                baselines["acstools_findsat_mrt"] = {"status": "unavailable"}
    return metrics, weightmask_mask, baselines


def _evaluate_real_case(
    case,
    data,
    hdr,
    with_baselines=False,
    truth_mask=None,
    reference_inverse_variance=None,
    native_quality_mask=None,
):
    suite_name = "megacam_real" if "megacam" in case["case_id"] else "acs_compare"
    if case["label_recipe"] == "blank_control":
        bkg_rms = np.full_like(data, np.nanstd(data - np.nanmedian(data)) + 1e-3)
        metrics, streak_mask, baselines = _evaluate_blank_control(data, bkg_rms, with_baselines)
        _write_debug_mask(
            case["case_id"], suite_name, science=data.astype(np.float32), streak_pred=streak_mask.astype(np.uint8)
        )
        metrics["baselines"] = baselines
        return metrics
    if case["label_recipe"] == "dark_injection_v1":
        metrics, arrays, baselines = _evaluate_dark_injection(case, data, hdr, with_baselines)
        _write_debug_mask(
            case["case_id"],
            suite_name,
            science=data.astype(np.float32),
            **{k: v.astype(np.uint8) if v.dtype == bool else v for k, v in arrays.items()},
        )
        metrics["baselines"] = baselines
        return metrics

    metrics, streak_mask, baselines = _evaluate_streak_case(
        case,
        data,
        hdr,
        with_baselines,
        truth_mask,
        reference_inverse_variance,
        native_quality_mask,
    )
    _write_debug_mask(
        case["case_id"],
        suite_name,
        science=data.astype(np.float32),
        streak_pred=streak_mask.astype(np.uint8),
        streak_truth=truth_mask.astype(np.uint8),
    )
    metrics["baselines"] = baselines
    return metrics


def _write_suite_outputs(summary):
    out_dir = OUTPUT_ROOT / summary["suite"]
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "metrics.json"
    md_path = out_dir / "summary.md"
    with open(json_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    lines = [f"# Benchmark Summary: {summary['suite']}", ""]
    if summary.get("gate_failures"):
        lines.extend(["## Gate Failures", ""])
        lines.extend([f"- {item}" for item in summary["gate_failures"]])
        lines.append("")
    lines.extend(["## Results", ""])
    for name, result in summary["results"].items():
        status = result.get("status", "ok")
        lines.append(f"- `{name}`: {status}")
    with open(md_path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    return json_path, md_path


def run_suite(suite, with_baselines=False, selected_cases=None, download=False):
    if suite == "synthetic_v2":
        return run_synthetic_v2(with_baselines=with_baselines, selected_cases=selected_cases)
    if suite in {"megacam_real", "acs_compare"}:
        if download:
            process_suite(suite)
        manifest = load_manifest(suite)
        return _run_manifest_suite(manifest, with_baselines=with_baselines, selected_cases=selected_cases)
    if suite == "all":
        combined = {}
        failures = []
        for item in ("synthetic_v2", "megacam_real", "acs_compare"):
            summary = run_suite(item, with_baselines=with_baselines, selected_cases=selected_cases, download=download)
            combined[item] = summary["results"]
            failures.extend(summary.get("gate_failures", []))
            _write_suite_outputs(summary)
        return {"suite": "all", "results": combined, "gate_failures": failures}
    raise ValueError(f"Unknown suite '{suite}'")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run WeightMask benchmark suites.")
    parser.add_argument("--suite", required=True, choices=["synthetic_v2", "megacam_real", "acs_compare", "all"])
    parser.add_argument("--with-baselines", action="store_true")
    parser.add_argument("--download", action="store_true", help="Attempt to download missing real data")
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args(argv)

    summary = run_suite(
        args.suite, with_baselines=args.with_baselines, selected_cases=set(args.case) or None, download=args.download
    )
    json_path, md_path = _write_suite_outputs(summary)
    print(f"Wrote benchmark metrics to {json_path}")
    print(f"Wrote benchmark summary to {md_path}")
    if summary.get("gate_failures"):
        print("Benchmark gate failures:")
        for item in summary["gate_failures"]:
            print(f"  - {item}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
