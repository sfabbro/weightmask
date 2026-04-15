
## 2024-05-18 - Avoid Python loops for pixel masking
**Learning:** Pure Python iteration to draw thousands of ellipses using `skimage.draw.ellipse` creates a significant bottleneck on dense star fields. Vectorized C-extensions like `sep.mask_ellipse` offer ~50-100x speedups.
**Action:** Always prefer C-level vectorized operations (`sep.mask_ellipse`, `skimage.draw.polygon2mask`, etc.) over manual pixel-level `for` loops when generating masks in Python.
## 2024-05-18 - [Vectorize region growing]
**Learning:** Pixel-by-pixel Python `for` loops in image processing algorithms like bleed trail growth cause massive performance bottlenecks.
**Action:** Replace sequential boundary limit checks with NumPy vectorized operations (slicing and `np.argmin` on boolean arrays) to determine termination points instantly. Reverse slicing can be elegantly handled using a `-1` end index condition to capture the 0th array element cleanly.
