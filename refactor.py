import re

with open("tests/simulate_and_test.py", "r") as f:
    content = f.read()

# Let's manually replace the `create_simulated_data` function with helper functions.

helpers = """
def _generate_background(size, x, y, regime_type):
    bkg_base = 100.0
    if regime_type in ["complex", "extreme_gradient"]:
        bkg_map = bkg_base + 20.0 * np.sin(x / 100.0) * np.cos(y / 150.0)
        if regime_type == "extreme_gradient":
            bkg_map += 0.5 * x + 0.2 * y
        else:
            bkg_map += 0.05 * x
    else:
        bkg_map = np.ones((size, size), dtype=np.float32) * bkg_base
    return bkg_map

def _add_stars(data, gt, size, x, y, num_stars, regime_type):
    from skimage.draw import disk
    center_x, center_y = size / 2, size / 2
    for _ in range(num_stars):
        cx, cy = np.random.randint(20, size - 20, size=2)
        flux = np.random.uniform(500, 5000)

        if regime_type in ["complex", "extreme_gradient"]:
            r = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
            sigma = 1.5 + 2.0 * (r / (size / np.sqrt(2)))
            star = flux * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
        elif regime_type == "elliptical_psf":
            sigma_x = np.random.uniform(1.5, 2.5)
            sigma_y = sigma_x * np.random.uniform(2.0, 3.5)
            theta = np.deg2rad(np.random.uniform(0, 180))

            a = np.cos(theta) ** 2 / (2 * sigma_x**2) + np.sin(theta) ** 2 / (2 * sigma_y**2)
            b = -np.sin(2 * theta) / (4 * sigma_x**2) + np.sin(2 * theta) / (4 * sigma_y**2)
            c = np.sin(theta) ** 2 / (2 * sigma_x**2) + np.cos(theta) ** 2 / (2 * sigma_y**2)

            star = flux * np.exp(-(a * (x - cx) ** 2 + 2 * b * (x - cx) * (y - cy) + c * (y - cy) ** 2))
        else:
            sigma = np.random.uniform(1.5, 3.0)
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

def _apply_noise(data, bkg_map, size, noise_level, regime_type):
    if regime_type != "normal":
        safe_data = np.maximum(data, 0.0)
        noisy_data = np.random.poisson(safe_data).astype(np.float32)

        read_noise = noise_level
        noisy_data += np.random.normal(0, read_noise, size=(size, size)).astype(np.float32)

        data = noisy_data
        bkg_rms = np.sqrt(bkg_map + read_noise**2).astype(np.float32)
    else:
        data += np.random.normal(0, noise_level, size=(size, size)).astype(np.float32)
        bkg_rms = np.ones_like(data) * noise_level

    data = np.clip(data, 0, 65535.0)
    return data, bkg_rms

def _add_cosmic_rays(data, gt, size):
    for _ in range(20):
        cr_x, cr_y = np.random.randint(10, size - 10, size=2)
        cr_length = np.random.randint(1, 5)
        for i in range(cr_length):
            data[cr_y + i, cr_x] += np.random.uniform(2000, 10000)
            gt["cr"][cr_y + i, cr_x] = True

def _add_streaks(data, gt, size, noise_level, regime_type):
    from skimage.draw import line

    s_flux = 10.0 * noise_level
    rr, cc = line(100, 100, size - 100, size - 200)

    if regime_type == "thick_streak":
        for w in range(-3, 4):
            rr_w, cc_w = line(100 + w, 100, size - 100 + w, size - 200)
            for i, j in zip(rr_w, cc_w):
                if 0 <= i < size and 0 <= j < size:
                    data[i, j] += s_flux * 0.8
                    gt["streak"][max(0, i - 2) : min(size, i + 3), max(0, j - 2) : min(size, j + 3)] = True
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


def create_simulated_data(
    size=1024, noise_level=10.0, num_stars=50, streak_flux=30.0, regime_type="normal"
):
    print(
        f"Generating synthetic data (size={size}, noise={noise_level}, stars={num_stars}, streak_flux={streak_flux}, regime_type={regime_type})..."
    )

    x, y = np.meshgrid(np.arange(size), np.arange(size))

    bkg_map = _generate_background(size, x, y, regime_type)
    data = bkg_map.copy().astype(np.float32)

    gt = {
        "stars": np.zeros((size, size), dtype=bool),
        "sat": np.zeros((size, size), dtype=bool),
        "cr": np.zeros((size, size), dtype=bool),
        "streak": np.zeros((size, size), dtype=bool),
    }

    _add_stars(data, gt, size, x, y, num_stars, regime_type)
    _add_saturated_stars(data, gt, size, x, y, regime_type)
    data, bkg_rms = _apply_noise(data, bkg_map, size, noise_level, regime_type)
    _add_cosmic_rays(data, gt, size)
    _add_streaks(data, gt, size, noise_level, regime_type)

    return data, bkg_rms, gt
"""

import re

# find the start and end of create_simulated_data
start_idx = content.find("def create_simulated_data(")
end_idx = content.find("def evaluate_mask(")

new_content = content[:start_idx] + helpers + "\n\n" + content[end_idx:]

with open("tests/simulate_and_test.py", "w") as f:
    f.write(new_content)
