
## 2024-05-18 - Avoid Python loops for pixel masking
**Learning:** Pure Python iteration to draw thousands of ellipses using `skimage.draw.ellipse` creates a significant bottleneck on dense star fields. Vectorized C-extensions like `sep.mask_ellipse` offer ~50-100x speedups.
**Action:** Always prefer C-level vectorized operations (`sep.mask_ellipse`, `skimage.draw.polygon2mask`, etc.) over manual pixel-level `for` loops when generating masks in Python.

## 2024-05-19 - Replace pure Python loops with NumPy 2D array slicing
**Learning:** Pure Python iteration to assign array elements row by row or column by column within a bounding box introduces noticeable overhead. Native C-level NumPy 2D slicing can directly assign values to subarrays, achieving ~2x speedups.
**Action:** Always prefer 2D NumPy array slicing over manual Python `for` loops when assigning values to regular geometric regions like rectangles or lines within an image/mask.
