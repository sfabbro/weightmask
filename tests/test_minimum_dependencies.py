from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def test_minimum_runtime_dependencies_support_representative_pipeline_operations(tmp_path):
    import astroscrappy
    import fitsio
    import sep
    import yaml
    from astropy.stats import mad_std
    from scipy.ndimage import gaussian_filter
    from skimage.draw import ellipse

    image = np.zeros((32, 32), dtype=np.float32)
    image[16, 16] = 100.0
    assert np.isfinite(mad_std(image))
    assert gaussian_filter(image, sigma=1.0).shape == image.shape
    row, col = ellipse(16, 16, 2, 3, shape=image.shape)
    assert len(row) > 0 and len(col) > 0
    objects = sep.extract(image, thresh=5.0, minarea=1)
    assert len(objects) == 1
    cosmic_mask, cleaned = astroscrappy.detect_cosmics(image, sigclip=5.0, niter=1)
    assert cosmic_mask.shape == image.shape
    assert cleaned.shape == image.shape
    config = yaml.safe_load((REPO / "weightmask.yml").read_text())
    assert isinstance(config, dict)

    path = tmp_path / "uint64.fits"
    expected = np.array([[0, 2**40]], dtype=np.uint64)
    fitsio.write(path, expected, clobber=True)
    restored = fitsio.read(path)
    np.testing.assert_array_equal(restored.astype(np.uint64), expected)
