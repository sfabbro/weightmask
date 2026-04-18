import time
import numpy as np
import scipy.ndimage
from weightmask.satur import grow_bleed_trails

def create_mock_data():
    h, w = 4000, 4000
    sci_data = np.random.normal(100, 10, (h, w)).astype(np.float32)
    sat_mask = np.zeros((h, w), dtype=bool)
    sky_map = np.full((h, w), 100.0, dtype=np.float32)
    bkg_rms_map = np.full((h, w), 10.0, dtype=np.float32)

    # Add multiple saturated columns
    for col in np.random.choice(w, 100, replace=False):
        start = np.random.randint(100, h - 500)
        length = np.random.randint(50, 400)
        sat_mask[start:start+length, col] = True
        sci_data[start-50:start+length+50, col] = np.linspace(150, 100, length+100)

    config = {
        "mask_bleed_trails": True,
        "bleed_thresh_sigma": 5.0,
        "bleed_grow_vertical": 50,
        "bleed_grow_horizontal": 2,
    }
    return sci_data, sat_mask, sky_map, bkg_rms_map, config

def run_benchmark():
    sci_data, sat_mask, sky_map, bkg_rms_map, config = create_mock_data()

    start_time = time.time()
    grow_bleed_trails(sci_data, sat_mask, sky_map, bkg_rms_map, config)
    end_time = time.time()

    print(f"Original execution time: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    run_benchmark()
