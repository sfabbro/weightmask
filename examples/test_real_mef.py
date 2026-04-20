import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402
import yaml
from astropy.io import fits

from weightmask.pipeline import WeightMapGenerator  # noqa: E402


def evaluate_real_mef():
    fits_path = os.path.join("benchmark_data", "cfht_megacam_1043132p.fits.fz")
    config_path = "weightmask.yml"

    if not os.path.exists(fits_path):
        print(f"File not found: {fits_path}")
        return

    print(f"Loading {fits_path} ...")

    with yaml.safe_load(open(config_path, "r")) as f:
        config = f

    # Instantiate the pipeline
    wg = WeightMapGenerator(config)

    with fits.open(fits_path) as hdul:
        # CFHT MEFs usually have 36 or 40 science extensions. We'll just test on extension 1 and 2.
        for ext_idx in [1, 2]:
            print(f"\n--- Processing Extension {ext_idx} ---")
            header = hdul[ext_idx].header
            data = hdul[ext_idx].data

            print(f"Data shape: {data.shape}")
            print(f"Gain: {header.get('GAIN', 'N/A')}, Readnoise: {header.get('RDNOISE', 'N/A')}")

            # The pipeline handles background subtraction internally
            result = wg.process(data, header=header)

            wmap = result["weight_map"]
            flag_map = result["flag_map"]

            bkg_rms = result.get("bkg_rms_map")
            if bkg_rms is not None:
                print(f"Global Background RMS approx: {bkg_rms.mean():.2f}")

            print(f"Done. Calculated weight map min/max: {wmap.min():.2e} / {wmap.max():.2e}")
            print(f"Total masked pixels: {(flag_map > 0).sum()} / {flag_map.size}")


if __name__ == "__main__":
    evaluate_real_mef()
