
## 2024-05-18 - Avoid Python loops for pixel masking
**Learning:** Pure Python iteration to draw thousands of ellipses using `skimage.draw.ellipse` creates a significant bottleneck on dense star fields. Vectorized C-extensions like `sep.mask_ellipse` offer ~50-100x speedups.
**Action:** Always prefer C-level vectorized operations (`sep.mask_ellipse`, `skimage.draw.polygon2mask`, etc.) over manual pixel-level `for` loops when generating masks in Python.

## 2026-04-14 - Subsampling for robust statistics
**Learning:** Calculating `np.median` or `np.percentile` on full high-resolution image arrays (e.g., 4000x4000) for global background or MAD estimates can introduce significant delays (>0.3s per call) without yielding more accurate global statistics.
**Action:** Always subsample large arrays (e.g., `data[::10, ::10]` or `data.ravel()[::step]`) when calculating global robust statistics like median or MAD to achieve massive speedups.
