from tests.simulate_and_test import run_masking_test


def demonstrate_robustness():
    """
    Demonstrates the library's robustness against extreme gradients and crowded fields.
    """
    print("==================================================")
    print("  WEIGHTMASK REAL-WORLD ROBUSTNESS EXAMPLE")
    print("==================================================")

    # Configuration: Extremely crowded field with background gradients
    class Args:
        size = 1024
        noise = 15.0
        stars = 500  # Very crowded
        streak = 40.0
        mask_pct = 0.0
        regime_type = "complex"

    config_path = "weightmask.yml"
    print(f"Goal: Detect artifacts in a field with {Args.stars} stars and Poisson noise.")

    # Run masking test
    metrics = run_masking_test(config_path, Args, save_fits=True)

    print("\nRobustness Metrics Summary:")
    for name, (p, r) in metrics.items():
        # Heuristic assessment
        status = "PASSED" if r > 0.4 else "DIAGNOSTIC"
        if name == "Saturation" or name == "Cosmics":
            status = "PASSED" if r > 0.9 else "WARNING"

        print(f"  {name:12} | Recall: {r:.3f} | Status: {status}")

    print("\nVisual artifacts produced in 'test_outputs/':")
    print("  - mask_streak.fits: Shows the high-SNR satellite trail detection.")
    print("  - mask_obj.fits: Shows the deblended object mask.")
    print("  - mask_sat.fits: Shows saturated cores and grew bleed trails.")


if __name__ == "__main__":
    demonstrate_robustness()
