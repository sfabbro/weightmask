import numpy as np
import os
from tests.simulate_and_test import create_simulated_data


def run_complex_demonstration():
    """
    Generates a complex simulation and runs the full pipeline on it,
    printing metrics and saving outputs.
    """
    print("==================================================")
    print("  WEIGHTMASK COMPLEX SIMULATION DEMONSTRATION")
    print("==================================================")

    # Configuration
    size = 1024
    noise = 10.0
    stars = 50
    streak_flux = 30.0
    output_dir = "example_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Create Complex Data
    data, bkg_rms_true, gt = create_simulated_data(
        size=size,
        noise_level=noise,
        num_stars=stars,
        streak_flux=streak_flux,
        complex_mode=True,
    )

    print(f"  Data range: {np.min(data):.1f} to {np.max(data):.1f}")
    print(f"  Mean Bkg RMS map: {np.mean(bkg_rms_true):.1f}")

    # 2. Prepare mock config file
    config_path = "weightmask.yml"
    if not os.path.exists(config_path):
        print("Error: weightmask.yml not found in root.")
        return

    # 3. Run Pipeline via internal benchmark logic to get metrics
    from tests.simulate_and_test import run_masking_test

    class Args:
        pass

    Args.size = size
    Args.noise = noise
    Args.stars = stars
    Args.streak = streak_flux
    Args.mask_pct = 0.0
    Args.complex_mode = True

    print("\nRunning Weightmask Detection Pipeline...")
    metrics = run_masking_test(config_path, Args, save_fits=True)

    print("\n==================================================")
    print("  COMPLEX REGIME PERFORMANCE")
    print("==================================================")
    for name, (p, r) in metrics.items():
        print(f"{name:12} | Precision: {p:.3f} | Recall: {r:.3f}")

    print(f"\nExample outputs saved in: {output_dir}/")


if __name__ == "__main__":
    run_complex_demonstration()
