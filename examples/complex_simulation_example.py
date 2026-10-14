import os
import sys

import numpy as np

# Run as `python examples/complex_simulation_example.py`: put the repo root on
# the path so `tests` is importable (same shim as examples/test_real_mef.py).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)  # noqa: E402
from tests.simulate_and_test import create_simulated_data, run_masking_test  # noqa: E402


def run_complex_demonstration():
    """
    Generates a complex simulation and runs staged detector diagnostics,
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
    seed = 42
    # 1. Inspect Complex Data summary with the same deterministic seed
    data, bkg_rms_true, _gt = create_simulated_data(
        size=size,
        noise_level=noise,
        num_stars=stars,
        streak_flux=streak_flux,
        regime_type="complex",
        seed=seed,
    )

    print(f"  Data range: {np.min(data):.1f} to {np.max(data):.1f}")
    print(f"  Mean Bkg RMS map: {np.mean(bkg_rms_true):.1f}")

    # 2. Prepare config file from repo root
    config_path = os.path.join(REPO_ROOT, "weightmask.yml")
    if not os.path.exists(config_path):
        print("Error: weightmask.yml not found in root.")
        return 1

    # 3. Run staged detectors via the internal benchmark helper to get metrics
    class Args:
        pass

    Args.size = size
    Args.noise = noise
    Args.stars = stars
    Args.streak = streak_flux
    Args.mask_pct = 0.0
    Args.regime_type = "complex"
    Args.seed = seed

    print("\nRunning Weightmask Detector Diagnostics...")
    metrics = run_masking_test(config_path, Args, save_fits=True)

    print("\n==================================================")
    print("  COMPLEX REGIME PERFORMANCE")
    print("==================================================")
    for name, val in metrics.items():
        if isinstance(val, (tuple, list)):
            p, r = val
            print(f"{name:12} | Precision: {p:.3f} | Recall: {r:.3f}")
        else:
            print(f"{name:12} | {val}")

    print("\nExample outputs saved in: test_outputs/")
    return 0


if __name__ == "__main__":
    sys.exit(run_complex_demonstration())
