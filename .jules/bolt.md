
## 2024-05-18 - Avoid Python loops for pixel masking
**Learning:** Pure Python iteration to draw thousands of ellipses using `skimage.draw.ellipse` creates a significant bottleneck on dense star fields. Vectorized C-extensions like `sep.mask_ellipse` offer ~50-100x speedups.
**Action:** Always prefer C-level vectorized operations (`sep.mask_ellipse`, `skimage.draw.polygon2mask`, etc.) over manual pixel-level `for` loops when generating masks in Python.

## 2023-10-25 - Parallelizing heavy skimage operations
**Learning:** Nested loops on image blocks mapping sequentially to expensive skimage functions (like `skimage.filters.frangi`) can be a significant bottleneck. Python's `concurrent.futures.ThreadPoolExecutor` handles block parallelization very efficiently here, dropping the filter time significantly (e.g. 4x speedup).
**Action:** Always parallelize block processing on image data using `ThreadPoolExecutor` or `ProcessPoolExecutor` where operations on blocks are independent.
