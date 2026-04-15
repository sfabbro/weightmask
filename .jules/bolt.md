
## 2024-05-18 - Avoid Python loops for pixel masking
**Learning:** Pure Python iteration to draw thousands of ellipses using `skimage.draw.ellipse` creates a significant bottleneck on dense star fields. Vectorized C-extensions like `sep.mask_ellipse` offer ~50-100x speedups.
**Action:** Always prefer C-level vectorized operations (`sep.mask_ellipse`, `skimage.draw.polygon2mask`, etc.) over manual pixel-level `for` loops when generating masks in Python.
## 2024-05-18 - Mocking Context Managers with Dunder Methods
**Learning:** When mocking `fitsio.FITS` used as a context manager (e.g. `with fitsio.FITS(...) as f`), it is necessary to mock both the `__enter__` return value and any specific dunder methods used within the block (like `__len__` for checking file length).
**Action:** Use `mock_fits.return_value.__enter__.return_value = mock_fits_instance` and `mock_fits_instance.__len__.return_value = ...` to properly simulate different FITS file states (empty vs. non-empty).
