import os
import sys

import yaml
from astropy.io import fits

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)  # noqa: E402
from weightmask.process import process_image  # noqa: E402
from weightmask.utils import clean_config_dict  # noqa: E402


def evaluate_real_mef():
    fits_path = os.path.join(REPO_ROOT, "benchmark_data", "megacam", "megacam_streak_case.fits")
    config_path = os.path.join(REPO_ROOT, "weightmask.yml")

    if not os.path.exists(fits_path):
        print(f"File not found: {fits_path}")
        return 1

    print(f"Loading {fits_path} ...")

    with open(config_path, "r") as f:
        config = clean_config_dict(yaml.safe_load(f))

    failed = False
    with fits.open(fits_path) as hdul:
        # CFHT MEFs usually have 36 or 40 science extensions. We'll just test on extension 1 and 2.
        for ext_idx in [1, 2]:
            print(f"\n--- Processing Extension {ext_idx} ---")
            header = hdul[ext_idx].header
            data = hdul[ext_idx].data

            print(f"Data shape: {data.shape}")
            print(f"Gain: {header.get('GAIN', 'N/A')}, Readnoise: {header.get('RDNOISE', 'N/A')}")

            mask_data, _inv_var, weight_map, _conf, _sky, _info = process_image(data, header, None, config)

            if weight_map is None:
                print("Processing failed for this extension.")
                failed = True
                continue

            print(f"Done. Calculated weight map min/max: {weight_map.min():.2e} / {weight_map.max():.2e}")
            print(f"Total flagged pixels: {(mask_data > 0).sum()} / {mask_data.size}")
    return int(failed)


if __name__ == "__main__":
    sys.exit(evaluate_real_mef())
