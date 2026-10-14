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
from tests.simulate_and_test import GENERATOR_REVISION, METRIC_REVISION, _f1, run_masking_test
from weightmask.bad import detect_bad_pixels, detect_dark_hot_pixels
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

_SYNTHETIC_CASE_GATES = {
    "synthetic_sparse": {
        "centerline_coverage_min": 0.35,
        "along_trail_coverage_min": 0.35,
        "false_positive_per_mpix_max": 1000.0,
        "overmask_pixels_per_truth_max": 0.25,
        "object_recall_min": 0.90,
        "bad_pixel_f1_min": 0.50,
    }
}


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
    """Compatibility precision/recall statistics for benchmark diagnostics."""
    if eval_gt_mask is None:
        eval_gt_mask = gt_mask
    tp = int(np.sum(pred_mask & eval_gt_mask))
    fp = int(np.sum(pred_mask & (~eval_gt_mask)))
    fn = int(np.sum((~pred_mask) & gt_mask))
    precision = tp / (tp + fp + 1e-9)
    core_hits = int(np.sum(pred_mask & gt_mask))
    recall = core_hits / (core_hits + fn + 1e-9)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(_f1(precision, recall)),
        "pred_area": int(np.sum(pred_mask)),
        "gt_area": int(np.sum(gt_mask)),
        "overmask_fraction": float(fp / (np.sum(pred_mask) + 1e-9)) if np.sum(pred_mask) > 0 else 0.0,
        "false_positive_pixels": fp,
        "false_positive_per_mpix": float(fp / pred_mask.size * 1e6),
        "overmask_pixels_per_truth": float(fp / (np.sum(gt_mask) + 1e-9)),
        "width_ratio": float(np.sum(pred_mask) / (np.sum(gt_mask) + 1e-9)) if np.sum(gt_mask) > 0 else 0.0,
    }


def _streak_geometry_stats(pred_mask, components, centerlines):
    from scipy.ndimage import binary_dilation

    components = np.asarray(components)
    centerlines = np.asarray(centerlines)
    tolerant = binary_dilation(pred_mask, iterations=2)
    centerline_coverages = []
    along_coverages = []
    for label in np.unique(components):
        if label <= 0:
            continue
        coords = np.argwhere(centerlines == label)
        if coords.size == 0:
            continue
        centerline = centerlines == label
        centerline_coverages.append(float(np.mean(pred_mask[centerline])))
        if len(coords) < 2:
            along_coverages.append(float(np.any(tolerant[centerline])))
            continue
        centered = coords - np.mean(coords, axis=0)
        _, _, vectors = np.linalg.svd(centered, full_matrices=False)
        positions = centered @ vectors[0]
        hits = tolerant[coords[:, 0], coords[:, 1]]
        if not np.any(hits):
            along_coverages.append(0.0)
        else:
            span = float(np.max(positions) - np.min(positions))
            along_coverages.append(float((np.max(positions[hits]) - np.min(positions[hits])) / max(span, 1.0)))
    return {
        "centerline_coverage": float(np.mean(centerline_coverages)) if centerline_coverages else 0.0,
        "along_trail_coverage": float(np.mean(along_coverages)) if along_coverages else 0.0,
    }


def _synthetic_case_gate_failures(case_name, result):
    gates = _SYNTHETIC_CASE_GATES.get(case_name)
    if gates is None:
        return []
    failures = []
    streak = result.get("streak_stats", {})
    object_metrics = result.get("weightmask", {}).get("Objects", (np.nan, np.nan))
    bad_pixels = result.get("bad_pixel_stats", {})
    values = {
        "centerline coverage": streak.get("centerline_coverage"),
        "along-trail coverage": streak.get("along_trail_coverage"),
        "false positives per Mpix": streak.get("false_positive_per_mpix"),
        "overmask pixels per truth": streak.get("overmask_pixels_per_truth"),
        "object recall": object_metrics[1] if len(object_metrics) > 1 else np.nan,
        "bad-pixel F1": bad_pixels.get("f1"),
    }
    def finite(value):
        try:
            return bool(np.isfinite(float(value)))
        except (TypeError, ValueError):
            return False

    failures.extend(f"{case_name}: {name} is nonfinite" for name, value in values.items() if not finite(value))
    limits = (
        ("centerline coverage", "centerline_coverage_min", lambda value, limit: value < limit, "<"),
        ("along-trail coverage", "along_trail_coverage_min", lambda value, limit: value < limit, "<"),
        ("false positives per Mpix", "false_positive_per_mpix_max", lambda value, limit: value > limit, ">"),
        ("overmask pixels per truth", "overmask_pixels_per_truth_max", lambda value, limit: value > limit, ">"),
        ("object recall", "object_recall_min", lambda value, limit: value < limit, "<"),
        ("bad-pixel F1", "bad_pixel_f1_min", lambda value, limit: value < limit, "<"),
    )
    for name, gate_name, failed, relation in limits:
        if gate_name not in gates:
            failures.append(f"{case_name}: missing gate {gate_name}")
            continue
        value = values[name]
        limit = gates[gate_name]
        if finite(value) and failed(float(value), limit):
            failures.append(f"{case_name}: {name} {float(value):.3f} {relation} {limit:.3f}")
    return failures


def _simple_hough_baseline(data_sub, bkg_rms, thresh_sig=5.0, half_width=3):
    """A plain Hough-transform streak detector, independent of weightmask.

    A comparator is only worth its name if it is a *different* algorithm reached
    through no weightmask code. The two that used to sit here both violated that
    and neither ran: ``_simple_hough_baseline`` imported the deleted
    ``_detect_streaks_satdet`` and crashed, and ``_rubin_compatible_kht`` called
    ``detect_streaks`` itself, so a "comparator" was scored against the thing it
    was meant to be compared to. Both are gone.

    What remains is the honest minimal version of the name: threshold, Hough
    transform, keep up to four strongest lines, paint their corridors. No candidate scoring,
    no strip refinement, no vetoes -- the point is to be simple.
    """
    from skimage.transform import hough_line, hough_line_peaks

    binary = np.isfinite(data_sub) & (data_sub > thresh_sig * bkg_rms)
    if not binary.any():
        return np.zeros_like(data_sub, dtype=bool)
    accumulator, angles, distances = hough_line(binary)
    # hough_line_peaks returns three parallel arrays, not a list of tuples.
    vote, theta, rho = hough_line_peaks(accumulator, angles, distances, min_distance=4, min_angle=4, threshold=40)
    mask = np.zeros_like(data_sub, dtype=bool)
    yy, xx = np.mgrid[0 : data_sub.shape[0], 0 : data_sub.shape[1]]
    for angle, offset in zip(theta[:4], rho[:4]):
        normal = (np.cos(angle), np.sin(angle))
        distance = np.abs(xx * normal[0] + yy * normal[1] - offset)
        mask |= distance <= half_width
    return mask


def _radon_baseline(data_sub, bkg_rms, thresh_sig=4.0, theta_step_deg=2.0, min_length=60.0):
    """A standalone Radon line search, independent of weightmask.

    Replaces the ``simple_radon`` comparator the MegaCam manifest asked for and
    the code never provided -- the name sat in ``comparators`` with no
    implementation behind it, so requesting it silently produced no baseline at
    all. This is the matched-filter idea the deleted production rescue used
    (Nir, Zackay & Ofek 2018: a PSF-broadened line template scored along rotated
    axes), written out longhand so the comparison does not route through
    weightmask code.

    For each trial angle the image is projected onto the perpendicular axis. Two
    sinograms are accumulated: the matched-filter sum, and a count of pixels above
    ``thresh_sig``. A line is accepted where the count reaches ``min_length`` and
    the sum peaks along that row. The count is the gate that does the work -- a
    single hot pixel has count 1 and is rejected no matter how bright, which is
    the failure a sum-threshold alone gets wrong.
    """
    finite = np.isfinite(data_sub)
    normalized = np.where(finite, data_sub / np.maximum(bkg_rms, 1e-6), 0.0)
    bright = np.where(finite & (data_sub > thresh_sig * bkg_rms), 1.0, 0.0).astype(np.float32)
    height, width = data_sub.shape
    diagonal = int(np.ceil(np.hypot(height, width))) + 2
    padded_sum = np.zeros((diagonal, diagonal), dtype=np.float32)
    padded_bright = np.zeros((diagonal, diagonal), dtype=np.float32)
    offset_y, offset_x = (diagonal - height) // 2, (diagonal - width) // 2
    padded_sum[offset_y : offset_y + height, offset_x : offset_x + width] = normalized
    padded_bright[offset_y : offset_y + height, offset_x : offset_x + width] = bright
    grid_y, grid_x = np.mgrid[0:diagonal, 0:diagonal]

    mask = np.zeros_like(data_sub, dtype=bool)
    # The image occupies padded rows/cols [offset_y:, offset_x:]. The corridor is
    # measured against those same padded coordinates -- an earlier version used
    # the top-left of the padded grid as if it were the image origin, which shifted
    # every corridor by the pad offset and so found the line at 0/260.
    window_y = grid_y[offset_y : offset_y + height, offset_x : offset_x + width]
    window_x = grid_x[offset_y : offset_y + height, offset_x : offset_x + width]
    for theta_deg in np.arange(0.0, 180.0, theta_step_deg):
        theta = np.deg2rad(theta_deg)
        normal_x, normal_y = np.cos(theta), np.sin(theta)
        # rho is measured on the padded grid, whose x*nx + y*ny range depends on
        # the angle and can go negative. Shift into non-negative bins and keep the
        # offset, so bin i is the geometric offset (i + rho_offset).
        raw = grid_x * normal_x + grid_y * normal_y
        rho_offset = int(np.floor(raw.min())) - 1
        rho = (np.floor(raw) - rho_offset).astype(np.int32)
        n_bins = diagonal * 2 + 4
        projection_sum = np.bincount(rho.ravel(), weights=padded_sum.ravel(), minlength=n_bins)
        support = np.bincount(rho.ravel(), weights=padded_bright.ravel(), minlength=n_bins)

        long_enough = support >= min_length
        if not long_enough.any():
            continue
        candidates = np.flatnonzero(long_enough)
        for index in candidates:
            # Local maximum along rho within its own connected run, so a star
            # cluster at one rho cannot borrow a neighbouring angle's peak.
            if projection_sum[index] < projection_sum[index - 1] or projection_sum[index] < projection_sum[
                index + 1
            ]:
                continue
            # Bin i is the geometric offset (i + rho_offset); the corridor is
            # drawn around that, in the same padded-grid units the window uses.
            distance = np.abs(window_x * normal_x + window_y * normal_y - (index + rho_offset))
            mask |= distance <= 1.5
    return mask


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


def _evaluate_synthetic_case(case, with_baselines=False):
    args = SimpleNamespace(**case)
    metrics, products = run_masking_test(str(ROOT / "weightmask.yml"), args, save_fits=False, return_products=True)
    case_result = {"weightmask": metrics}
    from scipy.ndimage import binary_dilation

    dilated_streak_gt = binary_dilation(products["ground_truth"]["streak"], iterations=2)
    streak_stats = _mask_stats(
        products["masks"]["streaks"], products["ground_truth"]["streak"], eval_gt_mask=dilated_streak_gt
    )
    streak_stats.update(
        _streak_geometry_stats(
            products["masks"]["streaks"],
            products["ground_truth"].get("streak_components", np.zeros_like(products["ground_truth"]["streak"])),
            products["ground_truth"].get("streak_centerlines", np.zeros_like(products["ground_truth"]["streak"])),
        )
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
            "radon": _mask_stats(
                _radon_baseline(data_sub, products["bkg_rms"]),
                products["ground_truth"]["streak"],
                eval_gt_mask=dilated_streak_gt,
            ),
        }
    return case_result, products


def run_synthetic_v2(with_baselines=False, selected_cases=None):
    out_dir = OUTPUT_ROOT / "synthetic_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for case in _synthetic_v2_cases():
        if selected_cases and case["name"] not in selected_cases:
            continue
        case_result, products = _evaluate_synthetic_case(case, with_baselines=with_baselines)
        results[case["name"]] = case_result
        np.savez_compressed(
            out_dir / f"{case['name']}.npz",
            science=products["science"],
            bkg_rms=products["bkg_rms"],
            streak_pred=products["masks"]["streaks"].astype(np.uint8),
            streak_truth=products["ground_truth"]["streak"].astype(np.uint8),
            streak_components=products["ground_truth"].get("streak_components", np.zeros_like(products["ground_truth"]["streak"], dtype=np.int16)),
            streak_centerlines=products["ground_truth"].get("streak_centerlines", np.zeros_like(products["ground_truth"]["streak"], dtype=np.int16)),
        )
    failures = [] if results else ["No benchmark cases were evaluated"]
    for name, result in results.items():
        failures.extend(_synthetic_case_gate_failures(name, result))
        scores = {
            "streak F1": result["streak_stats"]["f1"],
            "object recall": result["weightmask"]["Objects"][1],
            "bad-pixel F1": result["bad_pixel_stats"]["f1"],
        }
        failures.extend(f"{name}: {metric} is nonfinite" for metric, value in scores.items() if not np.isfinite(value))
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
    return {
        "suite": "synthetic_v2",
        "generator_revision": GENERATOR_REVISION,
        "metric_revision": METRIC_REVISION,
        "results": results,
        "gate_failures": failures,
    }


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
    if not suite_results:
        failures.append("No benchmark cases were evaluated")
    return {"suite": manifest["suite"], "metric_revision": METRIC_REVISION, "results": suite_results, "gate_failures": failures}


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
        if value is None or not np.isfinite(value):
            failures.append(f"{case_id}: {metric_name} is unavailable or nonfinite")
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
        baselines["radon"] = {
            "streak_pixels": int(np.sum(_radon_baseline(data - np.nanmedian(data), bkg_rms)))
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

    pred_bad = detect_dark_hot_pixels(dark, _load_repo_config().get("dark_masking", {}))
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
    # ponytail: no independent dark comparator is installed. Truth scored
    # against itself and the postfiltered CR result are not valid baselines.
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
        if "radon" in case.get("comparators", []):
            radon_mask = _radon_baseline(data_sub, bkg_rms)
            baselines["radon"] = {
                "streak_pixels": int(np.sum(radon_mask)),
                "overlap_with_weightmask": int(np.sum(radon_mask & weightmask_mask)),
            }
        # A requested comparator with no implementation must be reported as not
        # run, not dropped. ``simple_radon`` sat in the MegaCam manifest with no
        # code behind it, so asking for it silently produced no baseline and the
        # summary looked complete. Same for the two acstools comparators, whose
        # old handlers were try/except around a dict literal and so always
        # answered "available" without running anything.
        for wanted in case.get("comparators", []):
            if wanted not in baselines:
                baselines[wanted] = {"status": "not_implemented", "reason": "no comparator code in this repository"}
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
    if selected_cases:
        suites = ("synthetic_v2", "megacam_real", "acs_compare") if suite == "all" else (suite,)
        known = {
            name: {case["name"] for case in _synthetic_v2_cases()}
            if name == "synthetic_v2"
            else {case["case_id"] for case in load_manifest(name)["cases"]}
            for name in suites
        }
        unknown = selected_cases - set().union(*known.values())
        if unknown:
            return {"suite": suite, "results": {}, "gate_failures": [f"Unknown benchmark case(s): {', '.join(sorted(unknown))}"]}
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
            if selected_cases and not selected_cases.intersection(known[item]):
                continue
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
