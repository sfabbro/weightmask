import argparse
import os

import fitsio
import numpy as np
import yaml

from weightmask.background import estimate_background
from weightmask.cosmics import detect_cosmic_rays
from weightmask.objects import detect_objects
from weightmask.satur import detect_saturated_pixels, grow_bleed_trails
from weightmask.streaks import detect_streaks


def _f1(precision, recall):
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _generate_background(size, x, y, regime_type):
    bkg_base = 100.0
    if regime_type in ["complex", "extreme_gradient", "amplifier_step"]:
        bkg_map = bkg_base + 20.0 * np.sin(x / 100.0) * np.cos(y / 150.0)
        if regime_type == "extreme_gradient":
            bkg_map += 0.5 * x + 0.2 * y
        elif regime_type == "amplifier_step":
            bkg_map += 0.05 * x
            bkg_map[:, size // 2 :] += 35.0
        else:
            bkg_map += 0.05 * x
    else:
        bkg_map = np.ones((size, size), dtype=np.float32) * bkg_base
    return bkg_map


def _add_stars(data, gt, size, x, y, num_stars, regime_type, rng):
    from skimage.draw import disk

    center_x, center_y = size / 2, size / 2
    for _ in range(num_stars):
        cx, cy = rng.integers(20, size - 20, size=2)
        flux = rng.uniform(500, 5000)

        if regime_type in ["complex", "extreme_gradient", "amplifier_step"]:
            r = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
            sigma = 1.5 + 2.0 * (r / (size / np.sqrt(2)))
            star = flux * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
        elif regime_type == "elliptical_psf":
            sigma_x = rng.uniform(1.5, 2.5)
            sigma_y = sigma_x * rng.uniform(2.0, 3.5)
            theta = np.deg2rad(rng.uniform(0, 180))

            a = np.cos(theta) ** 2 / (2 * sigma_x**2) + np.sin(theta) ** 2 / (2 * sigma_y**2)
            b = -np.sin(2 * theta) / (4 * sigma_x**2) + np.sin(2 * theta) / (4 * sigma_y**2)
            c = np.sin(theta) ** 2 / (2 * sigma_x**2) + np.cos(theta) ** 2 / (2 * sigma_y**2)

            star = flux * np.exp(-(a * (x - cx) ** 2 + 2 * b * (x - cx) * (y - cy) + c * (y - cy) ** 2))
        elif regime_type == "elongated_galaxies":
            sigma_x = rng.uniform(2.0, 4.0)
            sigma_y = sigma_x * rng.uniform(1.5, 3.0)
            theta = np.deg2rad(rng.uniform(0, 180))

            a = np.cos(theta) ** 2 / (2 * sigma_x**2) + np.sin(theta) ** 2 / (2 * sigma_y**2)
            b = -np.sin(2 * theta) / (4 * sigma_x**2) + np.sin(2 * theta) / (4 * sigma_y**2)
            c = np.sin(theta) ** 2 / (2 * sigma_x**2) + np.cos(theta) ** 2 / (2 * sigma_y**2)

            star = flux * np.exp(-(a * (x - cx) ** 2 + 2 * b * (x - cx) * (y - cy) + c * (y - cy) ** 2))
        else:
            sigma = rng.uniform(1.5, 3.0)
            star = flux * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))

        data += star

        rr, cc = disk((int(cy), int(cx)), 2, shape=(size, size))
        gt["stars"][rr, cc] = True


def _add_saturated_stars(data, gt, size, x, y, regime_type):
    from skimage.draw import disk

    cx_sat, cy_sat = int(size * 0.3), int(size * 0.7)
    flux_sat = 500000
    sigma_sat = 4.0 if regime_type == "normal" else 4.5
    star_sat = flux_sat * np.exp(-((x - cx_sat) ** 2 + (y - cy_sat) ** 2) / (2 * sigma_sat**2))

    rr_b, cc_b = disk((cy_sat, cx_sat), 10, shape=(size, size))
    gt["stars"][rr_b, cc_b] = True

    data += star_sat

    bleed_length = 60
    data[cy_sat - bleed_length : cy_sat + bleed_length, cx_sat - 1 : cx_sat + 2] += flux_sat
    gt["sat"][data >= 65000] = True


def _apply_noise(data, bkg_map, size, noise_level, regime_type, rng):
    if regime_type != "normal":
        safe_data = np.maximum(data, 0.0)
        noisy_data = rng.poisson(safe_data).astype(np.float32)

        read_noise = noise_level
        noisy_data += rng.normal(0, read_noise, size=(size, size)).astype(np.float32)

        data = noisy_data
        bkg_rms = np.sqrt(bkg_map + read_noise**2).astype(np.float32)
    else:
        data += rng.normal(0, noise_level, size=(size, size)).astype(np.float32)
        bkg_rms = np.ones_like(data) * noise_level

    data = np.clip(data, 0, 65535.0)
    return data, bkg_rms


def _add_cosmic_rays(data, gt, size, rng):
    for _ in range(20):
        cr_x, cr_y = rng.integers(10, size - 10, size=2)
        cr_length = int(rng.integers(1, 5))
        for i in range(cr_length):
            data[cr_y + i, cr_x] += rng.uniform(2000, 10000)
            gt["cr"][cr_y + i, cr_x] = True


def _add_streaks(data, gt, size, noise_level, regime_type):
    from skimage.draw import line

    s_flux = 10.0 * noise_level
    rr, cc = line(100, 100, size - 100, size - 200)

    if regime_type in {"thick_streak", "variable_width_streak"}:
        for w in range(-3, 4):
            rr_w, cc_w = line(100 + w, 100, size - 100 + w, size - 200)
            for i, j in zip(rr_w, cc_w):
                if 0 <= i < size and 0 <= j < size:
                    flux_scale = 0.8
                    if regime_type == "variable_width_streak":
                        flux_scale *= 0.5 + 0.5 * np.sin((i + j) / 60.0)
                    data[i, j] += s_flux * flux_scale
                    gt["streak"][
                        max(0, i - 2) : min(size, i + 3),
                        max(0, j - 2) : min(size, j + 3),
                    ] = True
    else:
        for i, j in zip(rr, cc):
            if 0 <= i < size and 0 <= j < size:
                data[i, j] += s_flux
                gt["streak"][max(0, i - 1) : min(size, i + 2), max(0, j - 1) : min(size, j + 2)] = True

    dot_flux = 20.0 * noise_level
    dots_y0, dots_x0 = 200, 800
    dots_yf, dots_xf = 800, 200
    rr, cc = line(dots_y0, dots_x0, dots_yf, dots_xf)
    for i, j in zip(rr, cc):
        if 0 <= i < size and 0 <= j < size:
            gt["streak"][max(0, i - 1) : min(size, i + 2), max(0, j - 1) : min(size, j + 2)] = True

    if regime_type != "normal":
        cycle_len = 40
        for idx, (i, j) in enumerate(zip(rr, cc)):
            pos = idx % cycle_len
            if pos < 15:
                flux = dot_flux * (1.0 + 0.5 * np.sin(pos * np.pi / 15))
            elif pos == 25:
                flux = dot_flux * 0.5
            else:
                flux = 0

            if flux > 0 and 0 <= i < size and 0 <= j < size:
                data[i, j] += flux

        dots_y0_2, dots_x0_2 = 100, 900
        dots_yf_2, dots_xf_2 = 900, 100
        rr2, cc2 = line(dots_y0_2, dots_x0_2, dots_yf_2, dots_xf_2)
        faint_flux = 5.0 * noise_level
        for i, j in zip(rr2, cc2):
            if 0 <= i < size and 0 <= j < size:
                gt["streak"][max(0, i - 1) : min(size, i + 2), max(0, j - 1) : min(size, j + 2)] = True

        for i, j in zip(rr2[::15], cc2[::15]):
            if 0 <= i < size and 0 <= j < size:
                data[i, j] += faint_flux
    else:
        for i, j in zip(rr[::10], cc[::10]):
            if 0 <= i < size and 0 <= j < size:
                data[i, j] += dot_flux


def _inject_dark_defects(data, gt, size, noise_level, rng):
    hot_pixels = 25
    cols = rng.choice(np.arange(8, size - 8), size=3, replace=False)
    for _ in range(hot_pixels):
        x = int(rng.integers(8, size - 8))
        y = int(rng.integers(8, size - 8))
        data[y, x] += 30.0 * noise_level
        gt["defects"][y, x] = True
    for col in cols:
        data[:, col] += 8.0 * noise_level
        gt["defects"][:, col] = True


def create_simulated_data(size=1024, noise_level=10.0, num_stars=50, streak_flux=30.0, regime_type="normal", seed=None):
    print(
        f"Generating synthetic data (size={size}, noise={noise_level}, stars={num_stars}, streak_flux={streak_flux}, regime_type={regime_type}, seed={seed})..."
    )
    rng = np.random.default_rng(seed)

    x, y = np.meshgrid(np.arange(size), np.arange(size))

    bkg_map = _generate_background(size, x, y, regime_type)
    data = bkg_map.copy().astype(np.float32)

    gt = {
        "stars": np.zeros((size, size), dtype=bool),
        "sat": np.zeros((size, size), dtype=bool),
        "cr": np.zeros((size, size), dtype=bool),
        "streak": np.zeros((size, size), dtype=bool),
        "defects": np.zeros((size, size), dtype=bool),
    }

    _add_stars(data, gt, size, x, y, num_stars, regime_type, rng)
    _add_saturated_stars(data, gt, size, x, y, regime_type)
    data, bkg_rms = _apply_noise(data, bkg_map, size, noise_level, regime_type, rng)
    _add_cosmic_rays(data, gt, size, rng)
    _add_streaks(data, gt, size, noise_level, regime_type)
    if regime_type in {"complex", "extreme_gradient", "amplifier_step", "variable_width_streak"}:
        _inject_dark_defects(data, gt, size, noise_level, rng)

    return data, bkg_rms, gt


def evaluate_mask(pred_mask, gt_mask, name, pre_mask=None):
    """Calculates precision and recall."""
    # Mask out the pre-mask area if specified (e.g. for background robustness test)
    if pre_mask is not None:
        p_mask = pred_mask & (~pre_mask)
        g_mask = gt_mask & (~pre_mask)
    else:
        p_mask = pred_mask
        g_mask = gt_mask

    if name == "Objects":
        # Objects have dynamic halos based on flux in the weightmask.
        # A tiny ground truth disk (r=2) will classify all the correct halo masking as False Positives.
        # We aggressively dilate the ground truth mask for evaluation to cover expected halos.
        from scipy.ndimage import binary_dilation

        g_mask_eval = binary_dilation(g_mask, iterations=6)  # Allow up to 6px radius of 'true' halo
    else:
        g_mask_eval = g_mask

    true_positives = np.sum(p_mask & g_mask_eval)
    false_positives = np.sum(p_mask & (~g_mask_eval))
    false_negatives = np.sum((~p_mask) & g_mask)  # Keep original GT for FN

    precision = true_positives / (true_positives + false_positives + 1e-9)
    recall = true_positives / (true_positives + false_negatives + 1e-9)

    # Cap recall at 1.0 (dilation might cause TP > GTSum)
    recall = min(recall, 1.0)

    print(
        f"  [{name}] Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {_f1(precision, recall):.3f} "
        f"(TP:{true_positives}, FP:{false_positives}, FN:{false_negatives}, PredSum:{np.sum(p_mask)}, GTSum:{np.sum(g_mask)})"
    )
    return precision, recall


def _benchmark_gate_failures(results):
    streak_f1 = {name: _f1(*metrics.get("Streaks", (0.0, 0.0))) for name, metrics in results.items()}
    cosmic_f1 = [_f1(*metrics.get("Cosmics", (0.0, 0.0))) for metrics in results.values()]
    object_f1 = [_f1(*metrics.get("Objects", (0.0, 0.0))) for metrics in results.values()]

    failures = []
    avg_streak_f1 = float(np.mean(list(streak_f1.values()))) if streak_f1 else 0.0
    if avg_streak_f1 < 0.65:
        failures.append(f"Average streak F1 {avg_streak_f1:.3f} < 0.650")

    if streak_f1.get("Complex Ideal (Var Bkg/PSF)", 0.0) < 0.45:
        failures.append(
            f"Complex Ideal (Var Bkg/PSF) streak F1 {streak_f1.get('Complex Ideal (Var Bkg/PSF)', 0.0):.3f} < 0.450"
        )

    if streak_f1.get("Extreme Poisson Crowded", 0.0) < 0.25:
        failures.append(
            f"Extreme Poisson Crowded streak F1 {streak_f1.get('Extreme Poisson Crowded', 0.0):.3f} < 0.250"
        )

    zero_failures = [name for name, metrics in results.items() if metrics.get("Streaks", (0.0, 0.0)) == (0.0, 0.0)]
    if zero_failures:
        failures.append("Zero streak precision/recall in: " + ", ".join(sorted(zero_failures)))

    avg_cosmic_f1 = float(np.mean(cosmic_f1)) if cosmic_f1 else 0.0
    if avg_cosmic_f1 < 0.67:
        failures.append(f"Average cosmic-ray F1 {avg_cosmic_f1:.3f} < 0.670")

    avg_object_f1 = float(np.mean(object_f1)) if object_f1 else 0.0
    if avg_object_f1 < 0.40:
        failures.append(f"Average object F1 {avg_object_f1:.3f} < 0.400")

    crowded_object_recall = results.get("Galactic Plane (Crowded)", {}).get("Objects", (0.0, 0.0))[1]
    if crowded_object_recall < 0.95:
        failures.append(f"Crowded-field object recall {crowded_object_recall:.3f} < 0.950")

    if any(metrics.get("CR_Star_Overlap", 0) > 0 for metrics in results.values()):
        failures.append("CR-star overlap is non-zero in one or more regimes")

    return failures


def run_masking_test(config_path, args, save_fits=True, return_products=False):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    sci_data, bkg_rms, gt = create_simulated_data(
        size=args.size,
        noise_level=args.noise,
        num_stars=args.stars,
        streak_flux=args.streak,
        regime_type=getattr(args, "regime_type", "normal"),
        seed=getattr(args, "seed", None),
    )
    existing_mask = np.zeros_like(sci_data, dtype=bool)

    import sep

    sep.set_extract_pixstack(500000)  # Increase limit for heavily masked fields

    output_dir = "test_outputs"
    os.makedirs(output_dir, exist_ok=True)

    if save_fits:
        fitsio.write(os.path.join(output_dir, "simulated_raw.fits"), sci_data, clobber=True)

    metrics = {}

    initial_mask = np.zeros_like(sci_data, dtype=bool)
    if hasattr(args, "mask_pct") and args.mask_pct > 0:
        # Create a heavy pre-mask (e.g. 85% of pixels)
        initial_mask = np.random.random(sci_data.shape) < args.mask_pct
        existing_mask |= initial_mask
        print(f"  Pre-masking {args.mask_pct * 100:.1f}% of image for background robustness test...")

    # 1. Test Saturation & Bleed
    sat_level, method, sat_mask = detect_saturated_pixels(sci_data, None, config.get("saturation", {}))

    # Apply new bleed growing
    sat_mask = grow_bleed_trails(sci_data, sat_mask, sci_data * 0, bkg_rms, config.get("saturation", {}))

    metrics["Saturation"] = evaluate_mask(sat_mask, gt["sat"], "Saturation", pre_mask=initial_mask)
    if save_fits:
        fitsio.write(
            os.path.join(output_dir, "mask_sat.fits"),
            sat_mask.astype(np.uint8),
            clobber=True,
        )
    existing_mask |= sat_mask

    # 2. Test Cosmics
    avg_read_noise = float(np.median(bkg_rms))
    cr_mask = detect_cosmic_rays(
        sci_data,
        existing_mask,
        sat_level,
        gain=1.0,
        read_noise=avg_read_noise,
        config=config.get("cosmic_ray", {}),
        bkg_rms_map=bkg_rms,
    )
    metrics["Cosmics"] = evaluate_mask(cr_mask, gt["cr"], "Cosmics", pre_mask=initial_mask)
    if save_fits:
        fitsio.write(
            os.path.join(output_dir, "mask_cr.fits"),
            cr_mask.astype(np.uint8),
            clobber=True,
        )

    # Check if CR mask is eating the stars
    overlap = int(np.sum(cr_mask & gt["stars"]))
    metrics["CR_Star_Overlap"] = overlap
    if overlap > 0:
        print(f"  WARNING: Cosmic Ray mask overlaps with {overlap} Ground Truth Star pixels!")

    existing_mask |= cr_mask

    # 3. Test Objects
    sky_map, _ = estimate_background(sci_data, existing_mask, config.get("sep_background", {}))
    data_sub = sci_data - sky_map
    # Use initial_mask here to avoid losing objects that were wrongly flagged as CRs/Saturation
    # to get cleaner metrics on object detection itself.
    obj_mask = detect_objects(data_sub, bkg_rms, initial_mask, config.get("sep_objects", {}))
    metrics["Objects"] = evaluate_mask(obj_mask, gt["stars"], "Objects", pre_mask=initial_mask)
    if save_fits:
        fitsio.write(
            os.path.join(output_dir, "mask_obj.fits"),
            obj_mask.astype(np.uint8),
            clobber=True,
        )
    existing_mask |= obj_mask

    # 4. Test Streaks (Run after objects for clean background)
    sky_map, _ = estimate_background(sci_data, existing_mask, config.get("sep_background", {}))
    data_sub = sci_data - sky_map
    streak_mask = detect_streaks(data_sub, bkg_rms, existing_mask, config.get("streak_masking", {}))
    metrics["Streaks"] = evaluate_mask(streak_mask, gt["streak"], "Streaks", pre_mask=initial_mask)
    if save_fits:
        fitsio.write(
            os.path.join(output_dir, "mask_streak.fits"),
            streak_mask.astype(np.uint8),
            clobber=True,
        )
    existing_mask |= streak_mask & ~existing_mask

    if return_products:
        return metrics, {
            "science": sci_data,
            "bkg_rms": bkg_rms,
            "ground_truth": gt,
            "masks": {
                "saturation": sat_mask,
                "cosmics": cr_mask,
                "objects": obj_mask,
                "streaks": streak_mask,
            },
        }

    return metrics


def run_auto_sweep(report_file=None):
    print("==================================================")
    print("  RUNNING WEIGHTMASK ALGORITHM BENCHMARK SWEEP")
    print("==================================================")

    regimes = [
        {
            "name": "Ideal (Low Noise, Sparse)",
            "size": 512,
            "noise": 5.0,
            "stars": 20,
            "streak": 50.0,
            "regime": "normal",
        },
        {
            "name": "Noisy Sparse (High Noise)",
            "size": 512,
            "noise": 25.0,
            "stars": 20,
            "streak": 25.0,
            "regime": "normal",
        },
        {
            "name": "Galactic Plane (Crowded)",
            "size": 512,
            "noise": 10.0,
            "stars": 200,
            "streak": 40.0,
            "regime": "normal",
        },
        {
            "name": "Ultra-Faint Artifacts",
            "size": 512,
            "noise": 15.0,
            "stars": 50,
            "streak": 15.0,
            "regime": "normal",
        },
        # Complex Regimes
        {
            "name": "Complex Ideal (Var Bkg/PSF)",
            "size": 512,
            "noise": 5.0,
            "stars": 20,
            "streak": 50.0,
            "regime": "complex",
        },
        {
            "name": "Complex Noisy",
            "size": 512,
            "noise": 25.0,
            "stars": 20,
            "streak": 25.0,
            "regime": "complex",
        },
        {
            "name": "Complex Crowded",
            "size": 512,
            "noise": 10.0,
            "stars": 200,
            "streak": 40.0,
            "regime": "complex",
        },
        # Extreme Edge Cases
        {
            "name": "Extreme Gradient Bkg",
            "size": 512,
            "noise": 10.0,
            "stars": 50,
            "streak": 30.0,
            "regime": "extreme_gradient",
        },
        {
            "name": "Elliptical PSF/Tracking err",
            "size": 512,
            "noise": 10.0,
            "stars": 50,
            "streak": 30.0,
            "regime": "elliptical_psf",
        },
        {
            "name": "Thick Satellite Streak",
            "size": 512,
            "noise": 10.0,
            "stars": 50,
            "streak": 100.0,
            "regime": "thick_streak",
        },
        {
            "name": "Extreme Poisson Crowded",
            "size": 512,
            "noise": 40.0,
            "stars": 300,
            "streak": 20.0,
            "regime": "complex",
        },
    ]

    results = {}
    for r in regimes:
        print(f"\n>>> REGIME: {r['name']}")
        print(
            f"Noise: {r['noise']} | Stars: {r['stars']} | Streak Flux: {r['streak']} | Regime: {r.get('regime', 'normal')}"
        )

        # Build fake args
        class Args:
            pass

        args = Args()
        args.size = r["size"]
        args.noise = r["noise"]
        args.stars = r["stars"]
        args.streak = r["streak"]
        args.mask_pct = r.get("mask_pct", 0.0)
        args.regime_type = r.get("regime", "normal")

        try:
            metrics = run_masking_test("weightmask.yml", args, save_fits=False)
            results[r["name"]] = metrics
        except Exception as e:
            print(f"  Regime {r['name']} failed: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 50)
    print("  FINAL BENCHMARK SUMMARY")
    print("=" * 50)
    for name, metrics in results.items():
        cr_overlap = metrics.get("CR_Star_Overlap", 0)
        streak_f1 = _f1(*metrics["Streaks"])
        print(
            f"{name:30s} | Streaks P/R/F1: {metrics['Streaks'][0]:.3f}/{metrics['Streaks'][1]:.3f}/{streak_f1:.3f} "
            f"| CR-Star Overlap: {cr_overlap}"
        )

    failures = _benchmark_gate_failures(results)
    if failures:
        print("\nBenchmark gate failures:")
        for failure in failures:
            print(f"  - {failure}")
    else:
        print("\nBenchmark gate PASSED.")

    if report_file:
        print(f"\nGenerating Markdown report: {report_file}")
        with open(report_file, "w") as f:
            f.write("# Weightmask Benchmark Report\n\n")
            f.write(
                "| Regime | Saturation P | Saturation R | Saturation F1 | "
                "Cosmics P | Cosmics R | Cosmics F1 | "
                "Objects P | Objects R | Objects F1 | "
                "Streaks P | Streaks R | Streaks F1 | CR-Star Overlap |\n"
            )
            f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
            for name, metrics in results.items():
                sat_p, sat_r = metrics.get("Saturation", (0, 0))
                cr_p, cr_r = metrics.get("Cosmics", (0, 0))
                obj_p, obj_r = metrics.get("Objects", (0, 0))
                streak_p, streak_r = metrics.get("Streaks", (0, 0))
                cr_overlap = metrics.get("CR_Star_Overlap", 0)
                f.write(
                    f"| {name} | "
                    f"{sat_p:.3f} | {sat_r:.3f} | {_f1(sat_p, sat_r):.3f} | "
                    f"{cr_p:.3f} | {cr_r:.3f} | {_f1(cr_p, cr_r):.3f} | "
                    f"{obj_p:.3f} | {obj_r:.3f} | {_f1(obj_p, obj_r):.3f} | "
                    f"{streak_p:.3f} | {streak_r:.3f} | {_f1(streak_p, streak_r):.3f} | "
                    f"{cr_overlap} |\n"
                )

            if failures:
                f.write("\n## Benchmark Gate Failures\n\n")
                for failure in failures:
                    f.write(f"- {failure}\n")
            else:
                f.write("\n## Benchmark Gate\n\nPassed.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-sweep", action="store_true")
    parser.add_argument("--report", type=str, help="Path to generate Markdown report")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--noise", type=float, default=10.0)
    parser.add_argument("--stars", type=int, default=50)
    parser.add_argument("--streak", type=float, default=30.0)
    parser.add_argument("--complex_mode", action="store_true")
    args = parser.parse_args()

    if args.auto_sweep:
        run_auto_sweep(report_file=args.report)
    else:
        run_masking_test("weightmask.yml", args, save_fits=True)
