import os
import sys

# Run as `python examples/real_world_robustness.py`: put the repo root on the
# path so `tests` is importable (same shim as examples/test_real_mef.py).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)  # noqa: E402
from tests.simulate_and_test import run_masking_test  # noqa: E402


def demonstrate_robustness():
    """
    Runs staged detector diagnostics on a synthetic crowded field with gradients.
    """
    print("==================================================")
    print("  WEIGHTMASK CROWDED-FIELD SIMULATION EXAMPLE")
    print("==================================================")

    # Configuration: Extremely crowded field with background gradients
    class Args:
        size = 1024
        noise = 15.0
        stars = 500  # Very crowded
        streak = 40.0
        mask_pct = 0.0
        regime_type = "complex"

    config_path = os.path.join(REPO_ROOT, "weightmask.yml")
    print(f"Goal: Detect artifacts in a field with {Args.stars} stars and Poisson noise.")

    # Run masking test
    metrics = run_masking_test(config_path, Args, save_fits=True)

    print("\nRobustness Metrics Summary:")
    for name, val in metrics.items():
        if not isinstance(val, (tuple, list)) or len(val) < 2:
            print(f"  {name:12} | {val}")
            continue
        p, r = val
        print(f"  {name:12} | Precision: {p:.3f} | Recall: {r:.3f}")

    print("\nVisual artifacts produced in 'test_outputs/':")
    print("  - mask_streak.fits: Shows the high-SNR satellite trail detection.")
    print("  - mask_obj.fits: Shows the deblended object mask.")
    print("  - mask_sat.fits: Shows saturated cores and grew bleed trails.")


if __name__ == "__main__":
    demonstrate_robustness()
